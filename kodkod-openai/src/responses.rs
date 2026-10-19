use std::collections::BTreeMap;
use std::io::{self, Write};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, CompactError, CompactOptions, ContextBudget, ContextCompaction, Conversation,
    Message, Provider, ProviderCompletion, ProviderEvent, ProviderStream, RequestDiagnostic,
    RequestDiagnosticEvent, RequestFailureKind, RequestFingerprint, RequestIdentity,
    RequestInputPrefixFingerprint, RequestMetadata, RequestObserver, RequestOperation,
    RequestOptionClass, RequestProfile, RequestPromptFingerprints, RequestUsage, ToolCall,
    ToolExecutorError, ToolResult, ToolResultOutcome, ToolSpec,
};
use kodkod_http::{SseDecoder, SseEvent};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::usage::parse_request_usage;
use crate::{CredentialSource, OpenAiError, OpenAiModel};

/// Controls Responses request formatting while keeping the OpenAI-compatible
/// request profile as the default for existing callers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesRequestProfile {
    #[default]
    OpenAiCompatible,
    ChatGptPlan,
}

#[derive(Clone, Copy)]
struct ResponsesRoute<'a> {
    profile: ResponsesRequestProfile,
    endpoint: &'a str,
}

impl ResponsesRequestProfile {
    fn is_chatgpt_plan(self) -> bool {
        matches!(self, Self::ChatGptPlan)
    }
}

pub fn responses_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    format!("{base}/responses")
}

/// Opaque Responses output retained for stateless replay.
///
/// Output items are intentionally not exposed. In particular, encrypted
/// reasoning and future item fields must be returned to the service unchanged.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ResponsesContinuation {
    checkpoints: Vec<ResponseCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    native_window: Option<NativeContextWindow>,
}

/// Canonical output returned by `/responses/compact`. Its opaque items must be
/// replayed verbatim and remain bound to the route that created them.
#[derive(Clone, Serialize, Deserialize)]
struct NativeContextWindow {
    model: String,
    profile: ResponsesRequestProfile,
    endpoint: String,
    output: Vec<Value>,
    /// Generated-token usage for `/responses/compact`; this is only a lower
    /// bound for the replay window because canonical output may retain inputs.
    #[serde(
        default,
        rename = "output_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    compaction_output_tokens: Option<u64>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ResponseCheckpoint {
    model: String,
    #[serde(default)]
    profile: ResponsesRequestProfile,
    assistant_message_index: usize,
    assistant: AssistantMessage,
    output: Vec<Value>,
}

impl std::fmt::Debug for ResponsesContinuation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesContinuation")
            .field("checkpoint_count", &self.checkpoints.len())
            .field("has_native_window", &self.native_window.is_some())
            .finish_non_exhaustive()
    }
}

/// Provider for the HTTP/SSE Responses API.
///
/// It uses stateless replay (`store: false`) and does not implement the newer
/// WebSocket-only steering or asynchronous tool-result transport.
#[derive(Clone)]
pub struct OpenAiResponsesProvider<M = ()> {
    responses_url: String,
    credentials: Option<Arc<dyn CredentialSource>>,
    instructions: Option<String>,
    profile: ResponsesRequestProfile,
    native_compaction: bool,
    request_observer: Option<Arc<dyn RequestObserver>>,
    client: reqwest::Client,
    _model: PhantomData<M>,
}

impl<M> OpenAiResponsesProvider<M> {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            responses_url: responses_url(&base_url.into()),
            credentials: None,
            instructions: None,
            profile: ResponsesRequestProfile::default(),
            native_compaction: false,
            request_observer: None,
            client: reqwest::Client::new(),
            _model: PhantomData,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.credentials = Some(Arc::new(crate::StaticCredentials::bearer(api_key.into())));
        self
    }

    pub fn with_credentials(mut self, credentials: Arc<dyn CredentialSource>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Set the optional Responses `instructions` field for every request.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Apply an explicit request profile. The default preserves the generic
    /// OpenAI-compatible wire format.
    pub fn with_profile(mut self, profile: ResponsesRequestProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Enable the standalone `/responses/compact` endpoint for deployments
    /// known to support it. Generic compatible gateways keep this disabled.
    pub fn with_native_compaction(mut self, enabled: bool) -> Self {
        self.native_compaction = enabled;
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Observe metadata-only diagnostics for each physical request attempt.
    pub fn with_request_observer(mut self, observer: Arc<dyn RequestObserver>) -> Self {
        self.request_observer = Some(observer);
        self
    }

    pub fn responses_url(&self) -> &str {
        &self.responses_url
    }

    fn compaction_url(&self) -> String {
        format!("{}/compact", self.responses_url.trim_end_matches('/'))
    }
}

struct RequestAttemptGuard {
    observer: Arc<dyn RequestObserver>,
    identity: RequestIdentity,
    operation: RequestOperation,
    started_at: Instant,
    terminal: bool,
}

impl RequestAttemptGuard {
    fn start(
        observer: Option<Arc<dyn RequestObserver>>,
        operation: RequestOperation,
        metadata: RequestMetadata,
    ) -> Option<Self> {
        let observer = observer?;
        let identity = RequestIdentity::next_process_unique();
        observe_diagnostic(
            observer.as_ref(),
            RequestDiagnostic {
                identity,
                operation,
                metadata: Some(metadata),
                event: RequestDiagnosticEvent::Started,
            },
        );
        Some(Self {
            observer,
            identity,
            operation,
            started_at: Instant::now(),
            terminal: false,
        })
    }

    fn headers(&self, status: u16, provider_request_id: Option<String>) {
        observe_diagnostic(
            self.observer.as_ref(),
            RequestDiagnostic {
                identity: self.identity,
                operation: self.operation,
                metadata: None,
                event: RequestDiagnosticEvent::Headers {
                    status,
                    provider_request_id,
                },
            },
        );
    }

    fn finish(
        &mut self,
        failure: Option<RequestFailureKind>,
        response_id: Option<String>,
        usage: Option<RequestUsage>,
    ) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        observe_diagnostic(
            self.observer.as_ref(),
            RequestDiagnostic {
                identity: self.identity,
                operation: self.operation,
                metadata: None,
                event: RequestDiagnosticEvent::Finished {
                    duration_micros: self.started_at.elapsed().as_micros().min(u64::MAX as u128)
                        as u64,
                    failure,
                    response_id,
                    usage,
                },
            },
        );
    }
}

impl Drop for RequestAttemptGuard {
    fn drop(&mut self) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        observe_diagnostic(
            self.observer.as_ref(),
            RequestDiagnostic {
                identity: self.identity,
                operation: self.operation,
                metadata: None,
                event: RequestDiagnosticEvent::Abandoned {
                    duration_micros: self.started_at.elapsed().as_micros().min(u64::MAX as u128)
                        as u64,
                    failure: RequestFailureKind::Unknown,
                },
            },
        );
    }
}

fn observe_diagnostic(observer: &dyn RequestObserver, diagnostic: RequestDiagnostic) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        observer.observe(diagnostic);
    }));
}

fn request_fingerprint(
    observer: &dyn RequestObserver,
    domain: &str,
    bytes: &[u8],
) -> RequestFingerprint {
    let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        observer.fingerprint(domain, bytes)
    }))
    .ok()
    .flatten()
    .filter(|value| {
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    });
    RequestFingerprint {
        byte_len: bytes.len(),
        value,
    }
}

fn safe_provider_id(value: Option<&str>) -> Option<String> {
    value
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 128
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        })
        .map(str::to_owned)
}

fn response_request_id(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| safe_provider_id(Some(value)))
}

fn failure_kind(error: &OpenAiError) -> RequestFailureKind {
    match error {
        OpenAiError::Http(_) => RequestFailureKind::Transport,
        OpenAiError::Api { status, .. } => RequestFailureKind::HttpStatus(*status),
        OpenAiError::ResponseFailed { .. } => RequestFailureKind::ProviderResponse,
        OpenAiError::Incomplete { .. } => RequestFailureKind::Incomplete,
        OpenAiError::Json(_) | OpenAiError::EmptyResponse | OpenAiError::Protocol(_) => {
            RequestFailureKind::Protocol
        }
        OpenAiError::Credentials(_)
        | OpenAiError::UnsupportedDocument { .. }
        | OpenAiError::DocumentLimitExceeded { .. } => RequestFailureKind::Unknown,
    }
}

fn finish_attempt_failure(
    attempt: &mut Option<RequestAttemptGuard>,
    error: &OpenAiError,
    response_id: Option<String>,
    usage: Option<RequestUsage>,
) {
    if let Some(attempt) = attempt {
        attempt.finish(Some(failure_kind(error)), response_id, usage);
    }
}

fn terminal_event_diagnostics(event: &SseEvent) -> Option<(Option<String>, Option<RequestUsage>)> {
    if event.data == "[DONE]" {
        return None;
    }
    let payload: Value = serde_json::from_str(&event.data).ok()?;
    let kind = payload
        .get("type")
        .and_then(Value::as_str)
        .or(event.event_name.as_deref())
        .unwrap_or_default();
    if !matches!(
        kind,
        "response.completed" | "response.failed" | "response.incomplete" | "error"
    ) {
        return None;
    }
    let response = payload.get("response").unwrap_or(&payload);
    let usage_value = if response.get("usage").is_some() {
        response
    } else {
        &payload
    };
    let usage = (usage_value.get("usage").is_some()).then(|| parse_request_usage(usage_value));
    let response_id = safe_provider_id(
        response
            .get("id")
            .or_else(|| payload.get("id"))
            .and_then(Value::as_str),
    );
    Some((response_id, usage))
}

fn safe_model_id(value: &str) -> Option<String> {
    (!value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
        }))
    .then(|| value.to_owned())
}

fn classify_reasoning_effort(value: Option<&str>) -> RequestOptionClass {
    match value {
        None | Some("none") => RequestOptionClass::None,
        Some("low") => RequestOptionClass::Low,
        Some("medium") => RequestOptionClass::Medium,
        Some("high") => RequestOptionClass::High,
        Some("xhigh") => RequestOptionClass::XHigh,
        Some(_) => RequestOptionClass::Other,
    }
}

fn classify_service_tier(value: Option<&str>) -> RequestOptionClass {
    match value {
        None => RequestOptionClass::None,
        Some("auto") => RequestOptionClass::Auto,
        Some("default") => RequestOptionClass::Default,
        Some("flex") => RequestOptionClass::Flex,
        Some("priority") => RequestOptionClass::Priority,
        Some(_) => RequestOptionClass::Other,
    }
}

fn request_metadata<M: OpenAiModel>(
    observer: &dyn RequestObserver,
    model: &M,
    profile: ResponsesRequestProfile,
    conversation: &Conversation,
    tools: &[ToolSpec],
    continuation: &ResponsesContinuation,
    request: &Value,
) -> RequestMetadata {
    let request_size = json_size(request);
    let continuation_bytes = continuation_output_bytes(continuation);
    let (image_count, image_bytes, document_count, document_bytes) = attachment_counts(request);
    let input = request
        .get("input")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut prompt_fingerprints = RequestPromptFingerprints::default();
    if let Some(instructions) = request.get("instructions") {
        let bytes = serde_json::to_vec(instructions).unwrap_or_default();
        prompt_fingerprints.instructions =
            Some(request_fingerprint(observer, "instructions", &bytes));
    }
    if let Some(tools_value) = request.get("tools") {
        let bytes = serde_json::to_vec(tools_value).unwrap_or_default();
        prompt_fingerprints.tools = Some(request_fingerprint(observer, "tools", &bytes));
    }
    for count in [1usize, 4, 16, 64] {
        if input.len() >= count {
            let bytes = serde_json::to_vec(&input[..count]).unwrap_or_default();
            prompt_fingerprints
                .input_prefixes
                .push(RequestInputPrefixFingerprint {
                    item_count: Some(count),
                    fingerprint: request_fingerprint(
                        observer,
                        &format!("input-prefix-{count}"),
                        &bytes,
                    ),
                });
        }
    }
    let input_bytes = serde_json::to_vec(input).unwrap_or_default();
    prompt_fingerprints
        .input_prefixes
        .push(RequestInputPrefixFingerprint {
            item_count: None,
            fingerprint: request_fingerprint(observer, "input-total", &input_bytes),
        });

    RequestMetadata {
        model: safe_model_id(model.id()),
        profile: match profile {
            ResponsesRequestProfile::OpenAiCompatible => RequestProfile::OpenAiCompatible,
            ResponsesRequestProfile::ChatGptPlan => RequestProfile::ChatGptPlan,
        },
        reasoning_effort: classify_reasoning_effort(
            request.pointer("/reasoning/effort").and_then(Value::as_str),
        ),
        service_tier: classify_service_tier(request.get("service_tier").and_then(Value::as_str)),
        vision_enabled: model.supports_vision(),
        context_estimate_tokens: conversation.estimate_tokens(tools).saturating_add(
            <OpenAiResponsesProvider<M> as Provider>::estimate_continuation_tokens(continuation),
        ),
        context_budget_tokens: model.context_budget().map(|budget| budget.max_input_tokens),
        message_count: conversation.messages().len(),
        tool_definition_count: tools.len(),
        image_count,
        image_bytes,
        document_count,
        document_bytes,
        serialized_request_bytes: request_size,
        continuation_bytes,
        prompt_fingerprints,
    }
}

fn attachment_counts(request: &Value) -> (usize, Option<usize>, usize, Option<usize>) {
    let mut counts = InputAttachmentCounts::new();
    if let Some(input) = request.get("input") {
        count_input_attachments(input, &mut counts);
    }
    (
        counts.image_count,
        counts.image_bytes_known.then_some(counts.image_bytes),
        counts.document_count,
        counts.document_bytes_known.then_some(counts.document_bytes),
    )
}

#[derive(Default)]
struct InputAttachmentCounts {
    image_count: usize,
    image_bytes: usize,
    image_bytes_known: bool,
    document_count: usize,
    document_bytes: usize,
    document_bytes_known: bool,
}

impl InputAttachmentCounts {
    fn new() -> Self {
        Self {
            image_bytes_known: true,
            document_bytes_known: true,
            ..Self::default()
        }
    }
}

fn count_input_attachments(value: &Value, counts: &mut InputAttachmentCounts) {
    match value {
        Value::Array(values) => {
            for value in values {
                count_input_attachments(value, counts);
            }
        }
        Value::Object(object) => {
            match object.get("type").and_then(Value::as_str) {
                Some("input_image") => {
                    counts.image_count = counts.image_count.saturating_add(1);
                    if let Some(bytes) = object
                        .get("image_url")
                        .or_else(|| object.get("image"))
                        .and_then(decoded_data_url_byte_len)
                    {
                        counts.image_bytes = counts.image_bytes.saturating_add(bytes);
                    } else {
                        counts.image_bytes_known = false;
                    }
                }
                Some("input_file") => {
                    counts.document_count = counts.document_count.saturating_add(1);
                    if let Some(bytes) = object
                        .get("file_data")
                        .or_else(|| object.get("file_url"))
                        .and_then(decoded_data_url_byte_len)
                    {
                        counts.document_bytes = counts.document_bytes.saturating_add(bytes);
                    } else {
                        counts.document_bytes_known = false;
                    }
                }
                _ => {}
            }
            for nested in object.values() {
                count_input_attachments(nested, counts);
            }
        }
        _ => {}
    }
}

fn decoded_data_url_byte_len(value: &Value) -> Option<usize> {
    let url = value.as_str()?;
    let encoded = url.strip_prefix("data:")?.split_once(";base64,")?.1;
    let padding = encoded
        .bytes()
        .rev()
        .take_while(|byte| *byte == b'=')
        .count();
    if padding > 2
        || encoded.len() % 4 == 1
        || encoded[..encoded.len().saturating_sub(padding)]
            .bytes()
            .any(|byte| !byte.is_ascii_alphanumeric() && !matches!(byte, b'+' | b'/'))
        || encoded[encoded.len().saturating_sub(padding)..]
            .bytes()
            .any(|byte| byte != b'=')
    {
        return None;
    }
    encoded
        .len()
        .checked_mul(3)?
        .checked_div(4)?
        .checked_sub(padding)
}

fn continuation_output_bytes(continuation: &ResponsesContinuation) -> usize {
    let mut total = 0usize;
    for checkpoint in &continuation.checkpoints {
        total = total.saturating_add(json_size(&checkpoint.output));
    }
    if let Some(window) = &continuation.native_window {
        total = total.saturating_add(json_size(&window.output));
    }
    total
}

fn json_size<T: Serialize + ?Sized>(value: &T) -> usize {
    struct CountingWriter(usize);
    impl Write for CountingWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.0 = self.0.saturating_add(buffer.len());
            Ok(buffer.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut writer = CountingWriter(0);
    if serde_json::to_writer(&mut writer, value).is_ok() {
        writer.0
    } else {
        0
    }
}

// Keep media sizing aligned with kodkod-core's tokenizer-free estimates. These
// fixed item costs avoid mistaking inline base64 payload bytes for prompt tokens.
const NATIVE_IMAGE_TOKEN_ESTIMATE: u64 = 2_000;
const NATIVE_FILE_TOKEN_ESTIMATE: u64 = 8_000;

fn native_window_token_estimate(window: &NativeContextWindow) -> u64 {
    let generated_lower_bound = window.compaction_output_tokens.unwrap_or(0);
    let readable_and_media = window.output.iter().fold(0_u64, |sum, value| {
        sum.saturating_add(native_readable_and_media_tokens(value))
    });

    // `output_tokens` is generated usage, so it may omit retained user inputs.
    // Plain text and media have bounded heuristics. Opaque ciphertext size is
    // recorded as bytes in diagnostics, never converted into estimated tokens.
    generated_lower_bound.max(readable_and_media)
}

fn native_readable_and_media_tokens(value: &Value) -> u64 {
    match value {
        Value::Array(values) => values.iter().fold(0_u64, |sum, value| {
            sum.saturating_add(native_readable_and_media_tokens(value))
        }),
        Value::Object(fields) => {
            let item_type = fields.get("type").and_then(Value::as_str);
            let media_tokens = match item_type {
                Some("input_image") => NATIVE_IMAGE_TOKEN_ESTIMATE,
                Some("input_file") => NATIVE_FILE_TOKEN_ESTIMATE,
                _ => 0,
            };
            fields.iter().fold(media_tokens, |sum, (key, value)| {
                if matches!(
                    key.as_str(),
                    "type" | "encrypted_content" | "image_url" | "file_data" | "data"
                ) {
                    return sum;
                }
                let child_tokens = if matches!(
                    key.as_str(),
                    "text" | "arguments" | "name" | "call_id" | "filename" | "refusal"
                ) {
                    value
                        .as_str()
                        .map_or(0, |text| text.len().div_ceil(3) as u64)
                } else {
                    native_readable_and_media_tokens(value)
                };
                sum.saturating_add(child_tokens)
            })
        }
        _ => 0,
    }
}

impl<M> OpenAiResponsesProvider<M>
where
    M: OpenAiModel,
{
    async fn compact_native(
        &self,
        continuation: &ResponsesContinuation,
        model: &M,
        conversation: &Conversation,
        options: CompactOptions,
    ) -> Result<ContextCompaction<ResponsesContinuation>, CompactError<OpenAiError>> {
        let projected = options.retain_recent_tool_image_observations.map_or_else(
            || conversation.clone(),
            |keep| conversation.retain_recent_tool_image_observations(keep),
        );
        validate_documents(&projected, model.supports_vision()).map_err(CompactError::Provider)?;
        let input = build_input_for_endpoint(
            model.id(),
            &projected,
            continuation,
            self.profile,
            &self.responses_url,
        )
        .map_err(|error| CompactError::Context(error.to_string()))?;
        let input_size = serde_json::to_vec(&input)
            .map_err(OpenAiError::from)
            .map_err(CompactError::Provider)?
            .len();
        if input_size > MAX_COMPACTION_INPUT_BYTES {
            return Err(CompactError::Context(
                "native compaction input exceeded its size limit".into(),
            ));
        }

        let credentials = match &self.credentials {
            Some(source) => Some(
                source
                    .credentials()
                    .await
                    .map_err(OpenAiError::from)
                    .map_err(CompactError::Provider)?,
            ),
            None => None,
        };
        let mut request = json!({ "model": model.id(), "input": input });
        if let Some(tier) = model.request_options().service_tier {
            request["service_tier"] = Value::String(tier);
        }
        let instructions = self
            .instructions
            .as_deref()
            .or(conversation.system_prompt());
        if let Some(instructions) = instructions {
            request["instructions"] = Value::String(instructions.to_owned());
        }
        let mut attempt = self.request_observer.as_deref().and_then(|observer| {
            RequestAttemptGuard::start(
                self.request_observer.clone(),
                RequestOperation::Compaction,
                request_metadata(
                    observer,
                    model,
                    self.profile,
                    &projected,
                    &[],
                    continuation,
                    &request,
                ),
            )
        });
        let mut http_request = self.client.post(self.compaction_url()).json(&request);
        if let Some(credentials) = credentials {
            http_request = http_request.headers(credentials.headers().clone());
        }
        let response = match http_request.send().await {
            Ok(response) => response,
            Err(error) => {
                let error = OpenAiError::from(error);
                if let Some(attempt) = &mut attempt {
                    attempt.finish(Some(failure_kind(&error)), None, None);
                }
                return Err(CompactError::Provider(error));
            }
        };
        let status = response.status();
        if let Some(attempt) = &attempt {
            attempt.headers(status.as_u16(), response_request_id(&response));
        }
        if !status.is_success() {
            let body = match read_limited_body(response, MAX_ERROR_BODY_BYTES).await {
                Ok(body) => body,
                Err(error) => {
                    if let Some(attempt) = &mut attempt {
                        attempt.finish(
                            Some(RequestFailureKind::HttpStatus(status.as_u16())),
                            None,
                            None,
                        );
                    }
                    return Err(CompactError::Provider(error));
                }
            };
            let fallback = if body.is_empty() {
                status.to_string()
            } else {
                body.clone()
            };
            let body: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            let code = body
                .pointer("/error/code")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let message = body
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or(fallback);
            if is_native_compaction_unsupported(self.profile, status.as_u16(), code.as_deref()) {
                if let Some(attempt) = &mut attempt {
                    attempt.finish(
                        Some(RequestFailureKind::HttpStatus(status.as_u16())),
                        None,
                        None,
                    );
                }
                return Err(CompactError::Unsupported(
                    "the Responses endpoint does not support standalone compaction".into(),
                ));
            }
            if let Some(attempt) = &mut attempt {
                attempt.finish(
                    Some(RequestFailureKind::HttpStatus(status.as_u16())),
                    None,
                    None,
                );
            }
            return Err(CompactError::Provider(OpenAiError::Api {
                status: status.as_u16(),
                code,
                message,
            }));
        }

        let body = match read_bounded_body(response, MAX_COMPACTION_RESPONSE_BYTES).await {
            Ok(body) => body,
            Err(error) => {
                if let Some(attempt) = &mut attempt {
                    attempt.finish(Some(failure_kind(&error)), None, None);
                }
                return Err(CompactError::Provider(error));
            }
        };
        let payload: Value = match serde_json::from_slice(&body) {
            Ok(payload) => payload,
            Err(_) => {
                if let Some(attempt) = &mut attempt {
                    attempt.finish(Some(RequestFailureKind::Protocol), None, None);
                }
                return Err(CompactError::Context(
                    "native compaction returned malformed JSON".into(),
                ));
            }
        };
        let request_usage = parse_request_usage(&payload);
        let response_id = safe_provider_id(payload.get("id").and_then(Value::as_str));
        let output = match validate_native_compaction_response(&payload) {
            Ok(output) => output,
            Err(error) => {
                if let Some(attempt) = &mut attempt {
                    attempt.finish(
                        Some(RequestFailureKind::Protocol),
                        response_id,
                        Some(request_usage),
                    );
                }
                return Err(CompactError::Context(error));
            }
        };
        if let Some(attempt) = &mut attempt {
            attempt.finish(None, response_id, Some(request_usage));
        }

        let mut compacted_conversation = Conversation::new();
        if let Some(system_prompt) = conversation.system_prompt() {
            compacted_conversation = compacted_conversation.with_system_prompt(system_prompt);
        }
        Ok(ContextCompaction {
            conversation: compacted_conversation,
            continuation: ResponsesContinuation {
                checkpoints: Vec::new(),
                native_window: Some(NativeContextWindow {
                    model: model.id().to_owned(),
                    profile: self.profile,
                    endpoint: self.responses_url.clone(),
                    output,
                    compaction_output_tokens: request_usage.token_usage.output_tokens,
                }),
            },
            usage: request_usage.token_usage,
        })
    }
}

impl<M> Provider for OpenAiResponsesProvider<M>
where
    M: OpenAiModel,
{
    type Model = M;
    type Error = OpenAiError;
    type Continuation = ResponsesContinuation;

    fn with_request_observer(mut self, observer: Arc<dyn RequestObserver>) -> Self {
        self.request_observer = Some(observer);
        self
    }

    fn supports_vision(&self, model: &M) -> bool {
        model.supports_vision()
    }

    fn supports_document(&self, model: &M, mime: &str) -> bool {
        supports_inline_document(mime) && (mime != "application/pdf" || model.supports_vision())
    }

    fn create_continuation(&self, _model: &M) -> Self::Continuation {
        ResponsesContinuation::default()
    }

    fn estimate_continuation_tokens(continuation: &Self::Continuation) -> u64 {
        // Readable reasoning summaries are lower-bound estimates. Native compact
        // usage counts generated tokens, not necessarily every retained replay item.
        let mut estimate = 0_u64;
        for item in continuation
            .checkpoints
            .iter()
            .flat_map(|checkpoint| &checkpoint.output)
        {
            if item.get("type").and_then(Value::as_str) != Some("reasoning") {
                continue;
            }
            let Some(summary) = item.get("summary").and_then(Value::as_array) else {
                continue;
            };
            for part in summary {
                if part.get("type").and_then(Value::as_str) == Some("summary_text")
                    && let Some(text) = part.get("text").and_then(Value::as_str)
                {
                    estimate = estimate.saturating_add(text.len().div_ceil(4) as u64);
                }
            }
        }
        if let Some(window) = &continuation.native_window {
            estimate = estimate.saturating_add(native_window_token_estimate(window));
        }
        estimate
    }

    fn context_budget(&self, model: &M) -> Option<ContextBudget> {
        model.context_budget()
    }

    fn has_opaque_context(&self, continuation: &Self::Continuation) -> bool {
        continuation.native_window.is_some()
    }

    fn validate_context(&self, model: &M, continuation: &Self::Continuation) -> Result<(), String> {
        validate_continuation_provenance(
            model.id(),
            &self.responses_url,
            continuation,
            self.profile,
        )
        .map_err(|error| error.to_string())
    }

    fn is_context_overflow(&self, error: &Self::Error) -> bool {
        error.is_context_overflow()
    }

    async fn compact_context(
        &self,
        continuation: &Self::Continuation,
        model: &M,
        conversation: &Conversation,
        options: CompactOptions,
    ) -> Result<ContextCompaction<Self::Continuation>, CompactError<Self::Error>> {
        if !self.native_compaction {
            if continuation.native_window.is_some() {
                return Err(CompactError::Context(
                    "cannot compact a native Responses window while native compaction is disabled"
                        .into(),
                ));
            }
            let compacted = self
                .compact_with_usage(model, conversation, options)
                .await?;
            return Ok(ContextCompaction {
                conversation: compacted.conversation,
                continuation: self.create_continuation(model),
                usage: compacted.usage,
            });
        }

        self.compact_native(continuation, model, conversation, options)
            .await
    }

    async fn complete(
        &self,
        continuation: &Self::Continuation,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, Self::Continuation), Self::Error> {
        self.complete_with_usage(continuation, model, conversation, tools)
            .await
            .map(|completion| (completion.message, completion.continuation))
    }

    async fn complete_with_usage(
        &self,
        continuation: &Self::Continuation,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<ProviderCompletion<Self::Continuation>, Self::Error> {
        let mut stream = self.complete_stream(continuation, model, conversation, tools);
        while let Some(event) = stream.next().await {
            if let ProviderEvent::Completed(completion) = event? {
                return Ok(completion);
            }
        }
        Err(OpenAiError::Protocol(
            "stream ended without a terminal event".into(),
        ))
    }

    fn complete_stream<'a>(
        &'a self,
        continuation: &'a Self::Continuation,
        model: &'a M,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, Self::Continuation, Self::Error> {
        Box::pin(async_stream::try_stream! {
            validate_documents(conversation, model.supports_vision())?;
            let credentials = match &self.credentials {
                Some(source) => Some(source.credentials().await?),
                None => None,
            };
            let options = model.request_options();
            let request = build_request_for_endpoint(
                model.id(),
                options.clone(),
                self.instructions.as_deref(),
                conversation,
                tools,
                continuation,
                ResponsesRoute {
                    profile: self.profile,
                    endpoint: &self.responses_url,
                },
            )?;
            let mut attempt = if let Some(observer) = self.request_observer.as_ref() {
                let metadata = request_metadata(
                    observer.as_ref(),
                    model,
                    self.profile,
                    conversation,
                    tools,
                    continuation,
                    &request,
                );
                RequestAttemptGuard::start(
                    Some(observer.clone()),
                    RequestOperation::Completion,
                    metadata,
                )
            } else {
                None
            };
            let mut http_request = self.client.post(&self.responses_url).json(&request);
            if let Some(credentials) = credentials {
                http_request = http_request.headers(credentials.headers().clone());
            }

            let response = match http_request.send().await {
                Ok(response) => response,
                Err(error) => {
                    let error = OpenAiError::from(error);
                    if let Some(attempt) = &mut attempt {
                        attempt.finish(Some(failure_kind(&error)), None, None);
                    }
                    Err(error)?
                }
            };
            let status = response.status();
            if let Some(attempt) = &attempt {
                attempt.headers(status.as_u16(), response_request_id(&response));
            }
            if !status.is_success() {
                let body = match read_limited_body(response, MAX_ERROR_BODY_BYTES).await {
                    Ok(body) => body,
                    Err(error) => {
                        if let Some(attempt) = &mut attempt {
                            attempt.finish(
                                Some(RequestFailureKind::HttpStatus(status.as_u16())),
                                None,
                                None,
                            );
                        }
                        Err(error)?;
                        unreachable!("error propagation exits the stream");
                    }
                };
                let fallback = if body.is_empty() { status.to_string() } else { body.clone() };
                let body: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
                let code = body.pointer("/error/code").and_then(Value::as_str).map(str::to_owned);
                let message = body
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or(fallback);
                if let Some(attempt) = &mut attempt {
                    attempt.finish(
                        Some(RequestFailureKind::HttpStatus(status.as_u16())),
                        None,
                        None,
                    );
                }
                Err(OpenAiError::Api {
                    status: status.as_u16(),
                    code,
                    message,
                })?;
                unreachable!("error propagation exits the stream");
            }

            let mut bytes = response.bytes_stream();
            let mut decoder = SseDecoder::default();
            let mut output = OutputAccumulator::default();
            while let Some(chunk) = bytes.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        let error = OpenAiError::from(error);
                        finish_attempt_failure(&mut attempt, &error, None, None);
                        Err(error)?;
                        unreachable!("error propagation exits the stream");
                    }
                };
                let events = match decoder.push(&chunk) {
                    Ok(events) => events,
                    Err(message) => {
                        let error = OpenAiError::Protocol(message);
                        finish_attempt_failure(&mut attempt, &error, None, None);
                        Err(error)?;
                        unreachable!("error propagation exits the stream");
                    }
                };
                for event in events {
                    let event_diagnostics = terminal_event_diagnostics(&event);
                    let delta = match response_text_delta(&event) {
                        Ok(delta) => delta,
                        Err(error) => {
                            let (response_id, usage) =
                                event_diagnostics.clone().unwrap_or((None, None));
                            finish_attempt_failure(
                                &mut attempt,
                                &error,
                                response_id,
                                usage,
                            );
                            Err(error)?;
                            unreachable!("error propagation exits the stream");
                        }
                    };
                    if let Some(delta) = delta {
                        yield ProviderEvent::TextDelta(delta);
                    }
                    let terminal = match handle_sse_event(event, &mut output) {
                        Ok(terminal) => terminal,
                        Err(error) => {
                            let (response_id, usage) =
                                event_diagnostics.unwrap_or((None, None));
                            finish_attempt_failure(
                                &mut attempt,
                                &error,
                                response_id,
                                usage,
                            );
                            Err(error)?;
                            unreachable!("error propagation exits the stream");
                        }
                    };
                    if let Some(terminal) = terminal {
                        let response_id =
                            safe_provider_id(terminal.get("id").and_then(Value::as_str));
                        let request_usage = parse_request_usage(&terminal);
                        let final_output = match terminal
                            .get("output")
                            .and_then(Value::as_array)
                            .cloned()
                        {
                            Some(output) => output,
                            None => {
                                let error = OpenAiError::Protocol(
                                    "completed response missing output".into(),
                                );
                                finish_attempt_failure(
                                    &mut attempt,
                                    &error,
                                    response_id,
                                    Some(request_usage),
                                );
                                Err(error)?;
                                unreachable!("error propagation exits the stream");
                            }
                        };
                        let assistant = match parse_output_with_profile(&final_output, self.profile)
                        {
                            Ok(assistant) => assistant,
                            Err(error) => {
                                finish_attempt_failure(
                                    &mut attempt,
                                    &error,
                                    response_id,
                                    Some(request_usage),
                                );
                                Err(error)?;
                                unreachable!("error propagation exits the stream");
                            }
                        };
                        let mut next = continuation.clone();
                        next.checkpoints.push(ResponseCheckpoint {
                            model: model.id().to_owned(),
                            profile: self.profile,
                            assistant_message_index: conversation.messages().len(),
                            assistant: assistant.clone(),
                            output: final_output,
                        });
                        if let Some(attempt) = &mut attempt {
                            attempt.finish(None, response_id, Some(request_usage));
                        }
                        yield ProviderEvent::Completed(ProviderCompletion::new(
                            assistant,
                            next,
                            request_usage.token_usage,
                        ));
                        return;
                    }
                }
            }
            if let Err(message) = decoder.finish() {
                let error = OpenAiError::Protocol(message);
                finish_attempt_failure(&mut attempt, &error, None, None);
                Err(error)?;
                unreachable!("error propagation exits the stream");
            }
            let error = OpenAiError::Protocol("stream ended without a terminal event".into());
            finish_attempt_failure(&mut attempt, &error, None, None);
            Err(error)?;
            unreachable!("error propagation exits the stream");
        })
    }
}

#[cfg(test)]
fn build_request(
    model: &str,
    options: crate::OpenAiRequestOptions,
    configured_instructions: Option<&str>,
    conversation: &Conversation,
    tools: &[ToolSpec],
    continuation: &ResponsesContinuation,
) -> Result<Value, OpenAiError> {
    build_request_with_profile(
        model,
        options,
        configured_instructions,
        conversation,
        tools,
        continuation,
        ResponsesRequestProfile::OpenAiCompatible,
    )
}

#[cfg(test)]
fn build_request_with_profile(
    model: &str,
    options: crate::OpenAiRequestOptions,
    configured_instructions: Option<&str>,
    conversation: &Conversation,
    tools: &[ToolSpec],
    continuation: &ResponsesContinuation,
    profile: ResponsesRequestProfile,
) -> Result<Value, OpenAiError> {
    build_request_for_endpoint(
        model,
        options,
        configured_instructions,
        conversation,
        tools,
        continuation,
        ResponsesRoute {
            profile,
            endpoint: "",
        },
    )
}

fn build_request_for_endpoint(
    model: &str,
    options: crate::OpenAiRequestOptions,
    configured_instructions: Option<&str>,
    conversation: &Conversation,
    tools: &[ToolSpec],
    continuation: &ResponsesContinuation,
    route: ResponsesRoute<'_>,
) -> Result<Value, OpenAiError> {
    let input = build_input_for_endpoint(
        model,
        conversation,
        continuation,
        route.profile,
        route.endpoint,
    )?;

    let instructions = configured_instructions.or(conversation.system_prompt());
    let mut request = json!({
        "model": model,
        "input": input,
        "tools": convert_tools(tools, route.profile),
        "stream": true,
        "store": false,
        "include": ["reasoning.encrypted_content"]
    });
    if let Some(effort) = options.reasoning_effort {
        request["reasoning"] = json!({"effort": effort});
    }
    if let Some(tier) = options.service_tier {
        request["service_tier"] = Value::String(tier);
    }
    if let Some(instructions) = instructions {
        request["instructions"] = Value::String(instructions.to_owned());
    }
    Ok(request)
}

fn build_input_for_endpoint(
    model: &str,
    conversation: &Conversation,
    continuation: &ResponsesContinuation,
    profile: ResponsesRequestProfile,
    endpoint: &str,
) -> Result<Vec<Value>, OpenAiError> {
    validate_continuation_for_endpoint(model, conversation, continuation, profile, endpoint)?;
    validate_document_mime_types(conversation)?;
    let mut input = continuation
        .native_window
        .as_ref()
        .map_or_else(Vec::new, |window| window.output.clone());
    let checkpoints: BTreeMap<usize, &ResponseCheckpoint> = continuation
        .checkpoints
        .iter()
        .map(|checkpoint| (checkpoint.assistant_message_index, checkpoint))
        .collect();
    for (index, message) in conversation.messages().iter().enumerate() {
        if let Some(checkpoint) = checkpoints.get(&index) {
            input.extend(checkpoint.output.iter().cloned());
        } else {
            input.extend(convert_message_with_profile(message, profile));
        }
    }
    Ok(input)
}

fn validate_continuation_for_endpoint(
    model: &str,
    conversation: &Conversation,
    continuation: &ResponsesContinuation,
    profile: ResponsesRequestProfile,
    endpoint: &str,
) -> Result<(), OpenAiError> {
    validate_continuation_provenance(model, endpoint, continuation, profile)?;
    let mut previous_index = None;
    for checkpoint in &continuation.checkpoints {
        if checkpoint.model != model {
            return Err(OpenAiError::Protocol(
                "continuation belongs to a different model".into(),
            ));
        }
        if checkpoint.profile != profile {
            return Err(OpenAiError::Protocol(
                "continuation belongs to a different Responses request profile".into(),
            ));
        }
        if previous_index.is_some_and(|index| checkpoint.assistant_message_index <= index) {
            return Err(OpenAiError::Protocol(
                "continuation checkpoints are out of order".into(),
            ));
        }
        let expected = Message::Assistant(checkpoint.assistant.clone());
        if conversation
            .messages()
            .get(checkpoint.assistant_message_index)
            != Some(&expected)
        {
            return Err(OpenAiError::Protocol(
                "continuation does not match conversation history".into(),
            ));
        }
        previous_index = Some(checkpoint.assistant_message_index);
    }
    Ok(())
}

fn validate_continuation_provenance(
    model: &str,
    endpoint: &str,
    continuation: &ResponsesContinuation,
    profile: ResponsesRequestProfile,
) -> Result<(), OpenAiError> {
    let mut previous_index = None;
    for checkpoint in &continuation.checkpoints {
        if checkpoint.model != model {
            return Err(OpenAiError::Protocol(
                "continuation belongs to a different model".into(),
            ));
        }
        if checkpoint.profile != profile {
            return Err(OpenAiError::Protocol(
                "continuation belongs to a different Responses request profile".into(),
            ));
        }
        if previous_index.is_some_and(|index| checkpoint.assistant_message_index <= index) {
            return Err(OpenAiError::Protocol(
                "continuation checkpoints are out of order".into(),
            ));
        }
        validate_output_window(&checkpoint.output).map_err(|message| {
            OpenAiError::Protocol(format!("invalid continuation output: {message}"))
        })?;
        previous_index = Some(checkpoint.assistant_message_index);
    }
    let Some(window) = &continuation.native_window else {
        return Ok(());
    };
    if window.model != model {
        return Err(OpenAiError::Protocol(
            "native context window belongs to a different model".into(),
        ));
    }
    if window.profile != profile {
        return Err(OpenAiError::Protocol(
            "native context window belongs to a different Responses request profile".into(),
        ));
    }
    if !endpoint.is_empty() && window.endpoint != endpoint {
        return Err(OpenAiError::Protocol(
            "native context window belongs to a different Responses endpoint".into(),
        ));
    }
    validate_native_compaction_output(&window.output).map_err(|message| {
        OpenAiError::Protocol(format!("invalid native context window: {message}"))
    })
}

fn convert_message_with_profile(message: &Message, profile: ResponsesRequestProfile) -> Vec<Value> {
    match message {
        Message::System(system) => vec![json!({
            "role": if profile.is_chatgpt_plan() { "developer" } else { "system" },
            "content": [{"type": "input_text", "text": system.content()}]
        })],
        Message::User(user) => {
            let mut content = vec![json!({"type": "input_text", "text": user.content()})];
            content.extend(user.images().iter().map(|image| {
                json!({"type": "input_image", "image_url": image.to_data_url(), "detail": "auto"})
            }));
            content.extend(user.documents().iter().map(|document| {
                json!({
                    "type": "input_file",
                    "filename": document.filename(),
                    "file_data": document.to_data_url(),
                })
            }));
            vec![json!({"role": "user", "content": content})]
        }
        Message::Assistant(assistant) => {
            let mut items = Vec::new();
            if !assistant.content().is_empty() {
                items.push(json!({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": assistant.content()}]
                }));
            }
            items.extend(assistant.tool_calls().iter().map(|call| {
                let mut item = json!({
                    "type": "function_call",
                    "call_id": call.id(),
                    "name": call.name(),
                    "arguments": serde_json::to_string(call.arguments()).unwrap_or_else(|_| "{}".into())
                });
                if profile.is_chatgpt_plan() {
                    item["namespace"] = Value::String(FUNCTION_NAMESPACE.into());
                }
                item
            }));
            items
        }
        Message::ToolResult(result) => convert_tool_result(result),
    }
}

fn supports_inline_document(mime: &str) -> bool {
    matches!(
        mime,
        "application/pdf"
            | "text/plain"
            | "text/markdown"
            | "text/csv"
            | "text/tab-separated-values"
            | "text/html"
            | "text/xml"
            | "application/json"
            | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            | "application/msword"
            | "application/rtf"
            | "text/rtf"
            | "application/vnd.oasis.opendocument.text"
            | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            | "application/vnd.ms-powerpoint"
            | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            | "application/vnd.ms-excel"
            | "application/csv"
    )
}

const MAX_INLINE_DOCUMENT_BYTES: usize = 50 * 1024 * 1024;

fn validate_documents(
    conversation: &Conversation,
    supports_vision: bool,
) -> Result<(), OpenAiError> {
    validate_document_mime_types(conversation)?;
    let mut total_bytes = 0usize;
    for message in conversation.messages() {
        if let Message::User(user) = message {
            if !supports_vision
                && user
                    .documents()
                    .iter()
                    .any(|document| document.mime() == "application/pdf")
            {
                return Err(OpenAiError::UnsupportedDocument {
                    provider: "OpenAI Responses",
                    mime: "application/pdf".to_owned(),
                });
            }
            for document in user.documents() {
                total_bytes = total_bytes.saturating_add(document.data().len());
                if document.data().len() >= MAX_INLINE_DOCUMENT_BYTES
                    || total_bytes > MAX_INLINE_DOCUMENT_BYTES
                {
                    return Err(OpenAiError::DocumentLimitExceeded {
                        provider: "OpenAI Responses",
                        limit_bytes: MAX_INLINE_DOCUMENT_BYTES,
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_document_mime_types(conversation: &Conversation) -> Result<(), OpenAiError> {
    for message in conversation.messages() {
        if let Message::User(user) = message
            && let Some(document) = user
                .documents()
                .iter()
                .find(|document| !supports_inline_document(document.mime()))
        {
            return Err(OpenAiError::UnsupportedDocument {
                provider: "OpenAI Responses",
                mime: document.mime().to_owned(),
            });
        }
    }
    Ok(())
}

fn convert_tool_result(result: &ToolResult) -> Vec<Value> {
    let output = match result.outcome() {
        ToolResultOutcome::Success(output) => {
            serde_json::to_string(output.value()).unwrap_or_default()
        }
        ToolResultOutcome::Error(ToolExecutorError::UnknownTool(name)) => {
            format!("unknown tool: {name}")
        }
        ToolResultOutcome::Error(ToolExecutorError::Tool(error)) => error.message().to_owned(),
    };
    let mut items = vec![json!({
        "type": "function_call_output",
        "call_id": result.tool_call_id(),
        "output": output
    })];
    if let ToolResultOutcome::Success(output) = result.outcome()
        && !output.images().is_empty()
    {
        let mut content = vec![json!({
            "type": "input_text",
            "text": format!("Images returned by tool call '{}'.", result.tool_call_id())
        })];
        content.extend(output.images().iter().map(|image| {
            json!({"type": "input_image", "image_url": image.to_data_url(), "detail": "auto"})
        }));
        items.push(json!({"role": "user", "content": content}));
    }
    items
}

fn convert_tool(spec: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": spec.name(),
        "description": spec.description(),
        "parameters": spec.input_schema(),
        "strict": false
    })
}

const FUNCTION_NAMESPACE: &str = "functions";

fn convert_tools(tools: &[ToolSpec], profile: ResponsesRequestProfile) -> Vec<Value> {
    if profile.is_chatgpt_plan() {
        if tools.is_empty() {
            Vec::new()
        } else {
            vec![json!({
                "type": "namespace",
                "name": FUNCTION_NAMESPACE,
                "description": "Available functions.",
                "tools": tools.iter().map(convert_tool).collect::<Vec<_>>()
            })]
        }
    } else {
        tools.iter().map(convert_tool).collect()
    }
}

const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;
const MAX_ASSEMBLED_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_ITEMS: usize = 512;
const MAX_COMPACTION_INPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_NATIVE_COMPACTION_OUTPUT_BYTES: usize = MAX_COMPACTION_INPUT_BYTES;
const MAX_NATIVE_COMPACTION_OUTPUT_ITEMS: usize = 16_384;
const MAX_COMPACTION_RESPONSE_BYTES: usize = MAX_COMPACTION_INPUT_BYTES;

fn validate_output_window(output: &[Value]) -> Result<(), String> {
    validate_bounded_output_window(output, MAX_ASSEMBLED_OUTPUT_BYTES, MAX_OUTPUT_ITEMS)
}

fn validate_bounded_output_window(
    output: &[Value],
    max_bytes: usize,
    max_items: usize,
) -> Result<(), String> {
    if output.is_empty() {
        return Err("output window is empty".into());
    }
    if output.len() > max_items {
        return Err("output window contains too many items".into());
    }
    let mut serialized_bytes = 0usize;
    for item in output {
        if !item.is_object() {
            return Err("output window contains a non-object item".into());
        }
        let item_bytes = serde_json::to_vec(item)
            .map_err(|_| "output window item could not be serialized".to_owned())?
            .len();
        serialized_bytes = serialized_bytes
            .checked_add(item_bytes)
            .filter(|bytes| *bytes <= max_bytes)
            .ok_or_else(|| "output window exceeds its size limit".to_owned())?;
    }
    Ok(())
}

fn validate_native_compaction_response(payload: &Value) -> Result<Vec<Value>, String> {
    if payload.get("object").and_then(Value::as_str) != Some("response.compaction") {
        return Err("native compaction response has an invalid object type".into());
    }
    let output = payload
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| "native compaction response is missing output".to_owned())?;
    validate_native_compaction_output(output)?;
    Ok(output.clone())
}

fn validate_native_compaction_output(output: &[Value]) -> Result<(), String> {
    validate_bounded_output_window(
        output,
        MAX_NATIVE_COMPACTION_OUTPUT_BYTES,
        MAX_NATIVE_COMPACTION_OUTPUT_ITEMS,
    )?;
    let mut found_compaction = false;
    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("compaction") {
            continue;
        }
        found_compaction = true;
        if item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err("compaction item is missing non-empty encrypted_content".into());
        }
    }
    if !found_compaction {
        return Err("output window is missing a compaction item".into());
    }
    Ok(())
}

fn is_native_compaction_unsupported(
    profile: ResponsesRequestProfile,
    status: u16,
    code: Option<&str>,
) -> bool {
    matches!(status, 404 | 405)
        || (status == 400 && code == Some("unsupported_operation"))
        // ChatPass may authorize inference while denying this optional operation.
        // This exact scoped capability denial is not an invalid credential.
        || (profile.is_chatgpt_plan()
            && status == 401
            && code == Some("hardened_oauth_rule_missing"))
}

async fn read_bounded_body(
    response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, OpenAiError> {
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let new_len = bytes
            .len()
            .checked_add(chunk.len())
            .filter(|length| *length <= limit)
            .ok_or_else(|| {
                OpenAiError::Protocol("native compaction response exceeded its size limit".into())
            })?;
        bytes.reserve(new_len.saturating_sub(bytes.len()));
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

#[cfg(test)]
fn process_sse_chunk(
    decoder: &mut SseDecoder,
    output: &mut OutputAccumulator,
    chunk: &[u8],
) -> Result<Option<Value>, OpenAiError> {
    for event in decoder.push(chunk).map_err(OpenAiError::Protocol)? {
        if let Some(terminal) = handle_sse_event(event, output)? {
            return Ok(Some(terminal));
        }
    }
    Ok(None)
}

async fn read_limited_body(
    response: reqwest::Response,
    limit: usize,
) -> Result<String, OpenAiError> {
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let remaining = limit.saturating_sub(bytes.len());
        bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if bytes.len() == limit {
            break;
        }
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[derive(Default)]
struct OutputAccumulator {
    items: BTreeMap<usize, Value>,
    serialized_bytes: usize,
}

impl OutputAccumulator {
    fn insert(&mut self, payload: &Value) -> Result<(), OpenAiError> {
        let index = payload
            .get("output_index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| {
                OpenAiError::Protocol("output_item.done missing valid output_index".into())
            })?;
        if index >= MAX_OUTPUT_ITEMS {
            return Err(OpenAiError::Protocol(format!(
                "output_index {index} exceeds item limit"
            )));
        }
        let item = payload
            .get("item")
            .cloned()
            .ok_or_else(|| OpenAiError::Protocol("output_item.done missing item".into()))?;
        if self.items.contains_key(&index) {
            return Err(OpenAiError::Protocol(format!(
                "duplicate output_item.done index {index}"
            )));
        }
        if let Some(id) = item.get("id").and_then(Value::as_str)
            && self
                .items
                .values()
                .any(|existing| existing.get("id").and_then(Value::as_str) == Some(id))
        {
            return Err(OpenAiError::Protocol(format!(
                "duplicate output item id at index {index}"
            )));
        }
        let bytes = serde_json::to_vec(&item)?.len();
        self.serialized_bytes = self
            .serialized_bytes
            .checked_add(bytes)
            .filter(|bytes| *bytes <= MAX_ASSEMBLED_OUTPUT_BYTES)
            .ok_or_else(|| {
                OpenAiError::Protocol("assembled response output exceeded size limit".into())
            })?;
        self.items.insert(index, item);
        Ok(())
    }

    fn validate_final(&self, final_output: &[Value]) -> Result<(), OpenAiError> {
        for (&index, streamed) in &self.items {
            let final_item = final_output.get(index).ok_or_else(|| {
                OpenAiError::Protocol(format!(
                    "completed output missing streamed item at index {index}"
                ))
            })?;
            validate_item_identity(index, streamed, final_item)?;
        }
        Ok(())
    }

    fn take_ordered(&mut self) -> Result<Vec<Value>, OpenAiError> {
        for (expected, actual) in self.items.keys().copied().enumerate() {
            if expected != actual {
                return Err(OpenAiError::Protocol(format!(
                    "response output indices are not contiguous: expected {expected}, got {actual}"
                )));
            }
        }
        self.serialized_bytes = 0;
        Ok(std::mem::take(&mut self.items).into_values().collect())
    }
}

fn validate_item_identity(
    index: usize,
    streamed: &Value,
    final_item: &Value,
) -> Result<(), OpenAiError> {
    for field in ["type", "id"] {
        if streamed.get(field) != final_item.get(field) {
            return Err(OpenAiError::Protocol(format!(
                "completed output {field} disagrees with streamed item at index {index}"
            )));
        }
    }
    if streamed.get("type").and_then(Value::as_str) == Some("function_call")
        && streamed.get("call_id") != final_item.get("call_id")
    {
        return Err(OpenAiError::Protocol(format!(
            "completed output call_id disagrees with streamed item at index {index}"
        )));
    }
    Ok(())
}

fn handle_sse_event(
    event: SseEvent,
    output: &mut OutputAccumulator,
) -> Result<Option<Value>, OpenAiError> {
    if event.data == "[DONE]" {
        return Ok(None);
    }
    let payload: Value = serde_json::from_str(&event.data)?;
    let kind = payload
        .get("type")
        .and_then(Value::as_str)
        .or(event.event_name.as_deref())
        .unwrap_or_default();
    match kind {
        "response.output_item.done" => {
            output.insert(&payload)?;
            Ok(None)
        }
        "response.completed" => {
            let mut response = payload
                .get("response")
                .cloned()
                .ok_or_else(|| OpenAiError::Protocol("completed event missing response".into()))?;
            let final_output = response
                .get("output")
                .and_then(Value::as_array)
                .ok_or_else(|| OpenAiError::Protocol("completed response missing output".into()))?;
            if final_output.is_empty() && !output.items.is_empty() {
                response["output"] = Value::Array(output.take_ordered()?);
            } else {
                output.validate_final(final_output)?;
            }
            Ok(Some(response))
        }
        "response.failed" => Err(response_failure(&payload)),
        "response.incomplete" => {
            let reason = payload
                .pointer("/response/incomplete_details/reason")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_owned();
            Err(OpenAiError::Incomplete { reason })
        }
        "error" => Err(response_failure(&payload)),
        _ => Ok(None),
    }
}

fn response_text_delta(event: &SseEvent) -> Result<Option<String>, OpenAiError> {
    if event.data == "[DONE]" {
        return Ok(None);
    }
    let payload: Value = serde_json::from_str(&event.data)?;
    let kind = payload
        .get("type")
        .and_then(Value::as_str)
        .or(event.event_name.as_deref())
        .unwrap_or_default();
    match kind {
        "response.output_text.delta" | "response.refusal.delta" => payload
            .get("delta")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .map(Some)
            .ok_or_else(|| OpenAiError::Protocol(format!("{kind} missing delta"))),
        _ => Ok(None),
    }
}

fn response_failure(event: &Value) -> OpenAiError {
    let error = event
        .pointer("/response/error")
        .or_else(|| event.pointer("/error"));
    let code = error
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .filter(|code| !code.is_empty())
        .or_else(|| {
            event
                .get("code")
                .and_then(Value::as_str)
                .filter(|code| !code.is_empty())
        })
        .map(str::to_owned)
        .or_else(|| {
            error
                .and_then(|error| error.get("type"))
                .and_then(Value::as_str)
                .filter(|error_type| !error_type.is_empty() && *error_type != "error")
                .map(str::to_owned)
        });
    let message = error
        .and_then(|error| error.get("message"))
        .or_else(|| event.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("response failed")
        .to_owned();
    OpenAiError::ResponseFailed { code, message }
}

#[cfg(test)]
fn parse_output(output: &[Value]) -> Result<AssistantMessage, OpenAiError> {
    parse_output_with_profile(output, ResponsesRequestProfile::OpenAiCompatible)
}

fn parse_output_with_profile(
    output: &[Value],
    profile: ResponsesRequestProfile,
) -> Result<AssistantMessage, OpenAiError> {
    if output.is_empty() {
        return Err(OpenAiError::Protocol(
            "completed response returned no output items".into(),
        ));
    }
    let mut text = String::new();
    let mut calls = Vec::new();
    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let content = item
                    .get("content")
                    .and_then(Value::as_array)
                    .ok_or_else(|| OpenAiError::Protocol("message item missing content".into()))?;
                for part in content {
                    match part.get("type").and_then(Value::as_str) {
                        Some("output_text") => {
                            text.push_str(&required_string(part, "text")?);
                        }
                        Some("refusal") => {
                            text.push_str(&required_string(part, "refusal")?);
                        }
                        Some(kind) => {
                            return Err(OpenAiError::Protocol(format!(
                                "unsupported response content type: {kind}"
                            )));
                        }
                        None => {
                            return Err(OpenAiError::Protocol(
                                "response content missing type".into(),
                            ));
                        }
                    }
                }
            }
            Some("function_call") => {
                let call_id = required_string(item, "call_id")?;
                if profile.is_chatgpt_plan()
                    && let Some(namespace) = item.get("namespace").filter(|value| !value.is_null())
                {
                    let namespace = namespace.as_str().ok_or_else(|| {
                        OpenAiError::Protocol(
                            "function call namespace must be a string or null".into(),
                        )
                    })?;
                    if namespace != FUNCTION_NAMESPACE {
                        return Err(OpenAiError::Protocol(format!(
                            "unexpected function namespace: {namespace}"
                        )));
                    }
                }
                let name = required_string(item, "name")?;
                let arguments = required_string(item, "arguments")?;
                let arguments = serde_json::from_str(&arguments)?;
                calls.push(ToolCall::new(call_id, name, arguments));
            }
            Some("reasoning") => {}
            Some(kind) => {
                return Err(OpenAiError::Protocol(format!(
                    "unsupported response output type: {kind}"
                )));
            }
            None => return Err(OpenAiError::Protocol("response output missing type".into())),
        }
    }
    if text.is_empty() && calls.is_empty() {
        return Err(OpenAiError::Protocol(
            "completed response contained no assistant output".into(),
        ));
    }
    Ok(AssistantMessage::new(text).with_tool_calls(calls))
}

fn required_string(item: &Value, field: &str) -> Result<String, OpenAiError> {
    item.get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| OpenAiError::Protocol(format!("function call missing {field}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kodkod_core::{
        Document, Image, Message, ProviderEvent, RetryPolicy, RetryProvider, Retryable,
        SystemMessage, TokenUsage, ToolSpec, UserMessage,
    };
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[derive(Clone)]
    struct TestModel;
    impl OpenAiModel for TestModel {
        fn id(&self) -> &str {
            "gpt-test"
        }
        fn supports_vision(&self) -> bool {
            true
        }
    }

    #[derive(Default)]
    struct RecordingObserver {
        diagnostics: Mutex<Vec<RequestDiagnostic>>,
        fingerprint_domains: Mutex<Vec<String>>,
    }

    impl RecordingObserver {
        fn diagnostics(&self) -> Vec<RequestDiagnostic> {
            self.diagnostics.lock().unwrap().clone()
        }
    }

    impl RequestObserver for RecordingObserver {
        fn observe(&self, diagnostic: RequestDiagnostic) {
            self.diagnostics.lock().unwrap().push(diagnostic);
        }

        fn fingerprint(&self, domain: &str, bytes: &[u8]) -> Option<String> {
            self.fingerprint_domains
                .lock()
                .unwrap()
                .push(domain.to_owned());
            Some(format!("digest_{}", bytes.len()))
        }
    }

    struct RotatingCredentials(AtomicUsize);

    impl CredentialSource for RotatingCredentials {
        fn credentials(&self) -> crate::CredentialFuture<'_> {
            Box::pin(async {
                let token = if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    "first"
                } else {
                    "second"
                };
                crate::RequestCredentials::bearer(token)
            })
        }
    }

    fn completed(output: Value) -> String {
        format!(
            "event: response.completed\ndata: {}\n\n",
            json!({
                "type": "response.completed",
                "response": {"status": "completed", "output": output}
            })
        )
    }

    fn completed_with_usage(output: Value, usage: Value) -> String {
        format!(
            "event: response.completed\ndata: {}\n\n",
            json!({
                "type": "response.completed",
                "response": {"status": "completed", "output": output, "usage": usage}
            })
        )
    }

    fn diagnostic_response(output: Value, usage: Value, id: &str) -> String {
        format!(
            "event: response.completed\ndata: {}\n\n",
            json!({
                "type": "response.completed",
                "response": {
                    "id": id,
                    "status": "completed",
                    "output": output,
                    "usage": usage
                }
            })
        )
    }

    #[tokio::test]
    async fn diagnostics_are_content_free_and_finish_before_completed_is_yielded() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-request-id", "req_safe-123")
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(diagnostic_response(
                        json!([{
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "private-response"}]
                        }]),
                        json!({
                            "input_tokens": 20,
                            "input_tokens_details": {
                                "cached_tokens": 4,
                                "cache_write_tokens": 2
                            },
                            "output_tokens": 3,
                            "total_tokens": 23,
                            "prompt_cache_diagnostics": {
                                "reason": "cache_hit",
                                "type": "prompt_prefix",
                                "private": "private-diagnostic"
                            }
                        }),
                        "resp_safe-456",
                    )),
            )
            .expect(1)
            .mount(&server)
            .await;

        let observer = Arc::new(RecordingObserver::default());
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_profile(ResponsesRequestProfile::ChatGptPlan)
            .with_instructions("private-instructions")
            .with_request_observer(observer.clone());
        let provider = provider.clone();
        let mut conversation = Conversation::new().with_system_prompt("private-system");
        conversation.push_user_message(
            UserMessage::new("private-user")
                .with_images(vec![Image::new("image/png", b"private-image".to_vec())])
                .with_documents(vec![
                    Document::try_new("text/plain", "private.txt", b"private-document").unwrap(),
                ]),
        );
        let tools = [ToolSpec::new(
            "private-tool",
            "private tool description",
            json!({"type":"object"}),
        )];
        let continuation = ResponsesContinuation::default();
        let mut stream = provider.complete_stream(&continuation, &TestModel, &conversation, &tools);

        let completed = stream.next().await.unwrap().unwrap();
        assert!(matches!(completed, ProviderEvent::Completed(_)));
        assert!(stream.next().await.is_none());

        let diagnostics = observer.diagnostics();
        assert_eq!(diagnostics.len(), 3);
        assert!(matches!(
            diagnostics[0].event,
            RequestDiagnosticEvent::Started
        ));
        assert_eq!(diagnostics[0].operation, RequestOperation::Completion);
        let metadata = diagnostics[0].metadata.as_ref().unwrap();
        assert_eq!(metadata.model.as_deref(), Some("gpt-test"));
        assert_eq!(metadata.profile, RequestProfile::ChatGptPlan);
        assert_eq!(metadata.message_count, 1);
        assert_eq!(metadata.tool_definition_count, 1);
        assert_eq!((metadata.image_count, metadata.image_bytes), (1, Some(13)));
        assert_eq!(
            (metadata.document_count, metadata.document_bytes),
            (1, Some(16))
        );
        assert!(metadata.serialized_request_bytes > 0);
        assert_eq!(
            metadata
                .prompt_fingerprints
                .instructions
                .as_ref()
                .unwrap()
                .value
                .as_deref(),
            Some("digest_22")
        );
        assert!(matches!(
            diagnostics[1].event,
            RequestDiagnosticEvent::Headers {
                status: 200,
                provider_request_id: Some(_)
            }
        ));
        assert!(matches!(
            diagnostics[2].event,
            RequestDiagnosticEvent::Finished {
                failure: None,
                response_id: Some(ref id),
                usage: Some(RequestUsage {
                    cache_write_tokens: Some(2),
                    ..
                }),
                ..
            } if id == "resp_safe-456"
        ));
        assert_eq!(
            observer.fingerprint_domains.lock().unwrap().as_slice(),
            ["instructions", "tools", "input-prefix-1", "input-total"]
        );
        let diagnostic_text = format!("{diagnostics:?}");
        for secret in [
            "private-instructions",
            "private-system",
            "private-user",
            "private-image",
            "private-document",
            "private-tool",
            "private tool description",
            "private-response",
            "private-diagnostic",
        ] {
            assert!(!diagnostic_text.contains(secret));
        }
        assert_eq!(
            diagnostics
                .iter()
                .filter(|diagnostic| matches!(
                    diagnostic.event,
                    RequestDiagnosticEvent::Finished { usage: Some(_), .. }
                ))
                .count(),
            1
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn dropping_a_stream_emits_abandoned_without_claiming_remote_cancellation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(format!(
                        "event: response.output_text.delta\ndata: {}\n\n",
                        json!({
                            "type": "response.output_text.delta",
                            "delta": "private-stream-text"
                        })
                    )),
            )
            .expect(1)
            .mount(&server)
            .await;
        let observer = Arc::new(RecordingObserver::default());
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_request_observer(observer.clone());
        let conversation = Conversation::new();
        let continuation = ResponsesContinuation::default();
        let mut stream = provider.complete_stream(&continuation, &TestModel, &conversation, &[]);
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            ProviderEvent::TextDelta(_)
        ));
        drop(stream);

        let diagnostics = observer.diagnostics();
        assert_eq!(diagnostics.len(), 3);
        assert!(matches!(
            diagnostics[2].event,
            RequestDiagnosticEvent::Abandoned {
                failure: RequestFailureKind::Unknown,
                ..
            }
        ));
        assert!(!format!("{diagnostics:?}").contains("private-stream-text"));
        server.verify().await;
    }

    #[tokio::test]
    async fn terminal_provider_errors_preserve_allowlisted_ids_and_usage() {
        let cases = [
            (
                "response.failed",
                json!({
                    "type":"response.failed",
                    "response": {
                        "id":"resp_failed_safe",
                        "error":{"code":"server_error", "message":"private-error"},
                        "usage": {
                            "input_tokens":7,
                            "input_tokens_details":{"cached_tokens":2},
                            "output_tokens":1,
                            "total_tokens":8
                        }
                    }
                }),
                RequestFailureKind::ProviderResponse,
                "resp_failed_safe",
                7,
                2,
            ),
            (
                "response.incomplete",
                json!({
                    "type":"response.incomplete",
                    "response": {
                        "id":"resp_incomplete_safe",
                        "incomplete_details":{"reason":"max_output_tokens"},
                        "usage": {
                            "input_tokens":9,
                            "input_tokens_details":{"cached_tokens":3},
                            "output_tokens":2,
                            "total_tokens":11
                        }
                    }
                }),
                RequestFailureKind::Incomplete,
                "resp_incomplete_safe",
                9,
                3,
            ),
        ];

        for (event_name, payload, expected_failure, expected_id, input_tokens, cached_tokens) in
            cases
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/responses"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .insert_header("content-type", "text/event-stream")
                        .set_body_string(format!("event: {event_name}\ndata: {payload}\n\n")),
                )
                .expect(1)
                .mount(&server)
                .await;
            let observer = Arc::new(RecordingObserver::default());
            let provider =
                OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
                    .with_request_observer(observer.clone());

            assert!(
                provider
                    .complete_with_usage(
                        &ResponsesContinuation::default(),
                        &TestModel,
                        &Conversation::new(),
                        &[],
                    )
                    .await
                    .is_err()
            );

            let diagnostics = observer.diagnostics();
            let finished = diagnostics
                .iter()
                .find_map(|diagnostic| match &diagnostic.event {
                    RequestDiagnosticEvent::Finished {
                        failure,
                        response_id,
                        usage,
                        ..
                    } => Some((*failure, response_id.as_deref(), *usage)),
                    _ => None,
                });
            let (failure, response_id, usage) = finished.expect("terminal diagnostic");
            assert_eq!(failure, Some(expected_failure));
            assert_eq!(response_id, Some(expected_id));
            let usage = usage.expect("terminal usage");
            assert_eq!(usage.token_usage.input_tokens, Some(input_tokens));
            assert_eq!(usage.token_usage.cached_input_tokens, Some(cached_tokens));
            assert!(!format!("{diagnostics:?}").contains("private-error"));
            server.verify().await;
        }
    }

    #[tokio::test]
    async fn malformed_completed_event_finishes_with_usage_instead_of_abandoned() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(format!(
                        "event: response.completed\ndata: {}\n\n",
                        json!({
                            "type":"response.completed",
                            "response": {
                                "id":"resp_malformed_safe",
                                "status":"completed",
                                "usage":{"input_tokens":13, "output_tokens":2, "total_tokens":15}
                            }
                        })
                    )),
            )
            .expect(1)
            .mount(&server)
            .await;
        let observer = Arc::new(RecordingObserver::default());
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_request_observer(observer.clone());

        assert!(
            provider
                .complete_with_usage(
                    &ResponsesContinuation::default(),
                    &TestModel,
                    &Conversation::new(),
                    &[],
                )
                .await
                .is_err()
        );

        let diagnostics = observer.diagnostics();
        assert!(matches!(
            diagnostics.last().map(|diagnostic| &diagnostic.event),
            Some(RequestDiagnosticEvent::Finished {
                failure: Some(RequestFailureKind::Protocol),
                response_id: Some(id),
                usage: Some(RequestUsage { .. }),
                ..
            }) if id == "resp_malformed_safe"
        ));
        assert!(!diagnostics.iter().any(|diagnostic| matches!(
            diagnostic.event,
            RequestDiagnosticEvent::Abandoned { .. }
        )));
        server.verify().await;
    }

    #[tokio::test]
    async fn truncated_sse_finishes_as_protocol_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "event: response.completed\ndata: {\"type\":\"response.completed\"}",
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;
        let observer = Arc::new(RecordingObserver::default());
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_request_observer(observer.clone());

        assert!(
            provider
                .complete_with_usage(
                    &ResponsesContinuation::default(),
                    &TestModel,
                    &Conversation::new(),
                    &[],
                )
                .await
                .is_err()
        );

        let diagnostics = observer.diagnostics();
        assert!(matches!(
            diagnostics.last().map(|diagnostic| &diagnostic.event),
            Some(RequestDiagnosticEvent::Finished {
                failure: Some(RequestFailureKind::Protocol),
                ..
            })
        ));
        assert!(!diagnostics.iter().any(|diagnostic| matches!(
            diagnostic.event,
            RequestDiagnosticEvent::Abandoned { .. }
        )));
        server.verify().await;
    }

    #[tokio::test]
    async fn each_retry_has_a_distinct_started_and_terminal_request_identity() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer first"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "error": {"code":"server_error", "message":"private failure text"}
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer second"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(completed_with_usage(
                        json!([{
                            "type":"message", "role":"assistant",
                            "content":[{"type":"output_text", "text":"done"}]
                        }]),
                        json!({"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}),
                    )),
            )
            .expect(1)
            .mount(&server)
            .await;
        let observer = Arc::new(RecordingObserver::default());
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_credentials(Arc::new(RotatingCredentials(AtomicUsize::new(0))));
        let retry_provider = RetryProvider::with_policy(
            provider,
            RetryPolicy {
                max_attempts: 2,
                initial_backoff: Duration::ZERO,
                max_backoff: Duration::ZERO,
                backoff_multiplier: 1.0,
            },
        )
        .with_request_observer(observer.clone());

        let completion = retry_provider
            .complete_with_usage(
                &ResponsesContinuation::default(),
                &TestModel,
                &Conversation::new(),
                &[],
            )
            .await
            .unwrap();
        assert_eq!(completion.usage.input_tokens, Some(5));

        let diagnostics = observer.diagnostics();
        let started: Vec<_> = diagnostics
            .iter()
            .filter(|diagnostic| matches!(diagnostic.event, RequestDiagnosticEvent::Started))
            .collect();
        let finished: Vec<_> = diagnostics
            .iter()
            .filter(|diagnostic| {
                matches!(diagnostic.event, RequestDiagnosticEvent::Finished { .. })
            })
            .collect();
        assert_eq!(started.len(), 2);
        assert_eq!(finished.len(), 2);
        assert_ne!(
            started[0].identity.request_id,
            started[1].identity.request_id
        );
        assert_eq!(
            finished
                .iter()
                .filter(|diagnostic| matches!(
                    diagnostic.event,
                    RequestDiagnosticEvent::Finished {
                        usage: Some(_),
                        failure: None,
                        ..
                    }
                ))
                .count(),
            1
        );
        assert!(!format!("{diagnostics:?}").contains("private failure text"));
        server.verify().await;
    }

    fn output_item_done(index: usize, item: Value) -> String {
        format!(
            "event: response.output_item.done\ndata: {}\n\n",
            json!({
                "type": "response.output_item.done",
                "output_index": index,
                "item": item
            })
        )
    }

    fn parse_complete_stream(body: &[u8]) -> Result<Value, OpenAiError> {
        let mut decoder = SseDecoder::default();
        let mut output = OutputAccumulator::default();
        if let Some(response) = process_sse_chunk(&mut decoder, &mut output, body)? {
            return Ok(response);
        }
        decoder.finish().map_err(OpenAiError::Protocol)?;
        Err(OpenAiError::Protocol(
            "stream ended without a terminal event".into(),
        ))
    }

    #[test]
    fn ordinary_response_window_rejects_too_many_items() {
        let output = vec![json!({"role":"user"}); MAX_OUTPUT_ITEMS + 1];
        assert!(validate_output_window(&output).is_err());
    }

    #[test]
    fn native_compaction_window_has_a_separate_item_limit() {
        let mut output = vec![json!({"type":"compaction", "encrypted_content":"opaque"})];
        output.extend(vec![
            json!({"type":"retained_context"});
            MAX_NATIVE_COMPACTION_OUTPUT_ITEMS - 1
        ]);
        assert!(validate_native_compaction_output(&output).is_ok());

        output.push(json!({"type":"retained_context"}));
        assert!(validate_native_compaction_output(&output).is_err());
    }

    #[tokio::test]
    async fn native_compaction_accepts_a_9_mib_retained_user_item() {
        let retained_text = "x".repeat(9 * 1024 * 1024);
        let expected_len = retained_text.len();
        let compacted_output = json!([
            {"type":"compaction", "encrypted_content":"opaque-window"},
            {"role":"user", "content":[{"type":"input_text", "text":retained_text.clone()}]}
        ]);
        let compacted_items = compacted_output.as_array().unwrap();
        assert!(validate_native_compaction_output(compacted_items).is_ok());
        assert!(validate_output_window(compacted_items).is_err());

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/compact"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object":"response.compaction",
                "output": compacted_output,
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_native_compaction(true);
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new(retained_text));
        let compacted = provider
            .compact_context(
                &ResponsesContinuation::default(),
                &TestModel,
                &conversation,
                CompactOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(
            compacted
                .continuation
                .native_window
                .as_ref()
                .unwrap()
                .output[1]["content"][0]["text"]
                .as_str()
                .unwrap()
                .len(),
            expected_len
        );
        assert!(Provider::validate_context(&provider, &TestModel, &compacted.continuation).is_ok());
        server.verify().await;
    }

    #[tokio::test]
    async fn native_compaction_forwards_explicit_speed_and_preserves_unset_behavior() {
        struct TierModel(Option<&'static str>);
        impl OpenAiModel for TierModel {
            fn id(&self) -> &str {
                "tier-fixture"
            }
            fn supports_vision(&self) -> bool {
                false
            }
            fn request_options(&self) -> crate::OpenAiRequestOptions {
                crate::OpenAiRequestOptions {
                    service_tier: self.0.map(str::to_owned),
                    ..Default::default()
                }
            }
        }
        for tier in [None, Some("default"), Some("priority")] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/responses/compact"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "object": "response.compaction",
                    "output": [{"type": "compaction", "encrypted_content": "fixture-opaque"}],
                    "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}
                })))
                .expect(1)
                .mount(&server)
                .await;
            let provider =
                OpenAiResponsesProvider::<TierModel>::new(format!("{}/v1", server.uri()))
                    .with_native_compaction(true);
            let mut history = Conversation::new();
            history.push_user_message(UserMessage::new("fixture history"));
            provider
                .compact_context(
                    &ResponsesContinuation::default(),
                    &TierModel(tier),
                    &history,
                    CompactOptions::default(),
                )
                .await
                .unwrap();
            let requests = server.received_requests().await.unwrap();
            let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
            assert_eq!(body.get("service_tier").and_then(Value::as_str), tier);
            assert!(body.get("reasoning").is_none());
        }
    }

    #[tokio::test]
    async fn native_compaction_replays_the_complete_opaque_window_and_appends_new_input() {
        let server = MockServer::start().await;
        let compacted_output = json!([
            {"type":"compaction", "encrypted_content":"opaque-compaction", "future":"kept"},
            {"role":"user", "content":[
                {"type":"input_text", "text":"retained user item"},
                {"type":"input_image", "image_url":"data:image/png;base64,aGVsbG8="},
                {"type":"input_file", "filename":"retained.txt", "file_data":"data:text/plain;base64,ZmlsZQ=="}
            ]}
        ]);
        Mock::given(method("POST"))
            .and(path("/v1/responses/compact"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object":"response.compaction",
                "output": compacted_output,
                "usage": {"input_tokens": 200, "output_tokens": 10, "total_tokens": 210}
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(completed(json!([{
                    "type":"message", "role":"assistant",
                    "content":[{"type":"output_text", "text":"continued"}]
                }]))),
            )
            .expect(1)
            .mount(&server)
            .await;

        let model = OptionsModel;
        let provider = OpenAiResponsesProvider::<OptionsModel>::new(format!("{}/v1", server.uri()))
            .with_native_compaction(true);
        let mut original = Conversation::new().with_system_prompt("system");
        original.push_user_message(UserMessage::new("old history"));
        let compacted = provider
            .compact_context(
                &ResponsesContinuation::default(),
                &model,
                &original,
                CompactOptions {
                    keep_tail_tokens: 0,
                    ..CompactOptions::default()
                },
            )
            .await
            .unwrap();

        assert!(compacted.conversation.is_empty());
        assert_eq!(compacted.conversation.system_prompt(), Some("system"));
        assert_eq!(compacted.usage.input_tokens, Some(200));
        assert_eq!(
            <OpenAiResponsesProvider<OptionsModel> as Provider>::estimate_continuation_tokens(
                &compacted.continuation
            ),
            10_010,
            "retained user text and media contribute to the replay-window heuristic"
        );
        assert!(Provider::has_opaque_context(
            &provider,
            &compacted.continuation
        ));
        assert!(Provider::validate_context(&provider, &model, &compacted.continuation).is_ok());
        assert!(!format!("{:?}", compacted.continuation).contains("opaque-compaction"));

        let serialized = serde_json::to_value(&compacted.continuation).unwrap();
        assert_eq!(
            serialized["native_window"]["output"], compacted_output,
            "the compact endpoint output must be retained without normalizing opaque fields"
        );
        let restored: ResponsesContinuation = serde_json::from_value(serialized).unwrap();
        assert_eq!(
            <OpenAiResponsesProvider<OptionsModel> as Provider>::estimate_continuation_tokens(
                &restored
            ),
            10_010,
            "native checkpoint restore must retain usage and retained-media estimates"
        );
        let mut next = compacted.conversation.clone();
        next.push_user_message(UserMessage::new("new turn"));
        let completion = provider
            .complete(&restored, &model, &next, &[])
            .await
            .unwrap();
        assert_eq!(completion.0.content(), "continued");

        let requests = server.received_requests().await.unwrap();
        let compact_request: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(compact_request["model"], "gpt-options");
        assert_eq!(compact_request["instructions"], "system");
        assert_eq!(compact_request["input"][0]["role"], "user");
        assert_eq!(
            compact_request["input"][0]["content"][0]["text"],
            "old history"
        );
        assert!(compact_request.get("stream").is_none());
        assert!(compact_request.get("store").is_none());
        assert!(compact_request.get("reasoning").is_none());
        assert_eq!(compact_request["service_tier"], "burst");

        let continuation_request: Value = serde_json::from_slice(&requests[1].body).unwrap();
        let request_input = continuation_request["input"].as_array().unwrap();
        assert_eq!(&request_input[..2], compacted_output.as_array().unwrap());
        assert_eq!(
            attachment_counts(&continuation_request),
            (1, Some(5), 1, Some(4))
        );
        assert_eq!(continuation_request["input"][2]["role"], "user");
        assert_eq!(
            continuation_request["input"][2]["content"][0]["text"],
            "new turn"
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn native_compaction_rejects_malformed_windows_and_only_marks_known_unsupported_routes() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/compact"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object":"response.compaction",
                "output": [null]
            })))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_native_compaction(true);
        let malformed = provider
            .compact_context(
                &ResponsesContinuation::default(),
                &TestModel,
                &Conversation::new(),
                CompactOptions::default(),
            )
            .await;
        assert!(matches!(malformed, Err(CompactError::Context(_))));

        let unsupported = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/compact"))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({
                "error": {"code":"not_found", "message":"route unavailable"}
            })))
            .expect(1)
            .mount(&unsupported)
            .await;
        let unsupported_provider =
            OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", unsupported.uri()))
                .with_native_compaction(true);
        let result = unsupported_provider
            .compact_context(
                &ResponsesContinuation::default(),
                &TestModel,
                &Conversation::new(),
                CompactOptions::default(),
            )
            .await;
        assert!(matches!(result, Err(CompactError::Unsupported(_))));

        let denied = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/compact"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {"code":"invalid_api_key", "message":"unauthorized"}
            })))
            .expect(1)
            .mount(&denied)
            .await;
        let denied_provider =
            OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", denied.uri()))
                .with_native_compaction(true);
        let result = denied_provider
            .compact_context(
                &ResponsesContinuation::default(),
                &TestModel,
                &Conversation::new(),
                CompactOptions::default(),
            )
            .await;
        assert!(matches!(
            result,
            Err(CompactError::Provider(OpenAiError::Api { status: 401, .. }))
        ));
    }

    #[tokio::test]
    async fn malformed_native_compaction_preserves_response_usage() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/compact"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id":"resp_compact_safe",
                "object":"response.compaction",
                "output":[null],
                "usage":{
                    "input_tokens":31,
                    "input_tokens_details":{"cached_tokens":7},
                    "output_tokens":4,
                    "total_tokens":35
                }
            })))
            .expect(1)
            .mount(&server)
            .await;
        let observer = Arc::new(RecordingObserver::default());
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_native_compaction(true)
            .with_request_observer(observer.clone());

        let result = provider
            .compact_context(
                &ResponsesContinuation::default(),
                &TestModel,
                &Conversation::new(),
                CompactOptions::default(),
            )
            .await;
        assert!(matches!(result, Err(CompactError::Context(_))));

        let diagnostics = observer.diagnostics();
        assert!(matches!(
            diagnostics.last().map(|diagnostic| &diagnostic.event),
            Some(RequestDiagnosticEvent::Finished {
                failure: Some(RequestFailureKind::Protocol),
                response_id: Some(id),
                usage: Some(RequestUsage { .. }),
                ..
            }) if id == "resp_compact_safe"
        ));
        let usage = diagnostics
            .iter()
            .find_map(|diagnostic| match &diagnostic.event {
                RequestDiagnosticEvent::Finished { usage, .. } => *usage,
                _ => None,
            })
            .expect("response usage");
        assert_eq!(usage.token_usage.input_tokens, Some(31));
        assert_eq!(usage.token_usage.cached_input_tokens, Some(7));
        server.verify().await;
    }

    #[test]
    fn chatpass_compact_capability_denial_is_narrowly_scoped() {
        use ResponsesRequestProfile::{ChatGptPlan, OpenAiCompatible};
        assert!(is_native_compaction_unsupported(
            ChatGptPlan,
            401,
            Some("hardened_oauth_rule_missing")
        ));
        assert!(!is_native_compaction_unsupported(
            OpenAiCompatible,
            401,
            Some("hardened_oauth_rule_missing")
        ));
        assert!(!is_native_compaction_unsupported(
            ChatGptPlan,
            401,
            Some("invalid_api_key")
        ));
        assert!(!is_native_compaction_unsupported(
            ChatGptPlan,
            403,
            Some("chatpass_v2_scope_not_authorized")
        ));
    }

    #[tokio::test]
    async fn native_compaction_rejects_noncanonical_success_before_replacing_context() {
        let valid_item = json!({"type":"compaction", "encrypted_content":"opaque"});
        let malformed_responses = [
            (
                "missing response.compaction envelope",
                json!({"output":[valid_item.clone()]}),
            ),
            (
                "ordinary message output",
                json!({
                    "object":"response.compaction",
                    "output":[{
                        "type":"message",
                        "role":"assistant",
                        "content":[{"type":"output_text", "text":"looks successful"}]
                    }]
                }),
            ),
            (
                "empty encrypted content",
                json!({
                    "object":"response.compaction",
                    "output":[{"type":"compaction", "encrypted_content":""}]
                }),
            ),
            (
                "malformed compaction item after a valid item",
                json!({
                    "object":"response.compaction",
                    "output":[valid_item, {"type":"compaction", "encrypted_content":null}]
                }),
            ),
        ];

        let continuation = ResponsesContinuation::default();
        let continuation_before = serde_json::to_value(&continuation).unwrap();
        let mut conversation = Conversation::new().with_system_prompt("keep this system prompt");
        conversation.push_user_message(UserMessage::new("keep this user message"));

        for (case, payload) in malformed_responses {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/responses/compact"))
                .respond_with(ResponseTemplate::new(200).set_body_json(payload))
                .expect(1)
                .mount(&server)
                .await;
            let provider =
                OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
                    .with_native_compaction(true);
            let result = provider
                .compact_context(
                    &continuation,
                    &TestModel,
                    &conversation,
                    CompactOptions::default(),
                )
                .await;

            assert!(
                matches!(result, Err(CompactError::Context(_))),
                "{case} should be rejected as invalid compaction"
            );
            assert_eq!(
                serde_json::to_value(&continuation).unwrap(),
                continuation_before,
                "{case} must leave the original continuation intact"
            );
            assert_eq!(
                conversation.system_prompt(),
                Some("keep this system prompt")
            );
            assert_eq!(conversation.messages().len(), 1);
            server.verify().await;
        }
    }

    #[test]
    fn restored_native_window_requires_a_nonempty_compaction_item() {
        let provider = OpenAiResponsesProvider::<TestModel>::new("https://example.test/v1")
            .with_native_compaction(true);
        let continuation = ResponsesContinuation {
            checkpoints: Vec::new(),
            native_window: Some(NativeContextWindow {
                model: "gpt-test".into(),
                profile: ResponsesRequestProfile::OpenAiCompatible,
                endpoint: "https://example.test/v1/responses".into(),
                output: vec![json!({"type":"message", "content":[]})],
                compaction_output_tokens: None,
            }),
        };
        assert!(Provider::validate_context(&provider, &TestModel, &continuation).is_err());
    }

    #[test]
    fn native_window_estimate_counts_media_without_counting_base64_payload_bytes() {
        let image_url = format!("data:image/png;base64,{}", "A".repeat(250_000));
        let file_data = format!("data:application/pdf;base64,{}", "B".repeat(500_000));
        let continuation = ResponsesContinuation {
            checkpoints: Vec::new(),
            native_window: Some(NativeContextWindow {
                model: "gpt-test".into(),
                profile: ResponsesRequestProfile::OpenAiCompatible,
                endpoint: "https://example.test/v1/responses".into(),
                output: vec![
                    json!({"type":"compaction", "encrypted_content":"opaque-compaction"}),
                    json!({"role":"user", "content":[
                        {"type":"input_text", "text":"abc"},
                        {"type":"input_image", "image_url":image_url},
                        {"type":"input_file", "filename":"f.pdf", "file_data":file_data}
                    ]}),
                ],
                compaction_output_tokens: Some(10),
            }),
        };

        assert_eq!(
            <OpenAiResponsesProvider<TestModel> as Provider>::estimate_continuation_tokens(
                &continuation
            ),
            10_003
        );
    }

    #[test]
    fn native_window_without_usage_uses_visible_lower_bounds_not_ciphertext_bytes() {
        let opaque_only = ResponsesContinuation {
            checkpoints: Vec::new(),
            native_window: Some(NativeContextWindow {
                model: "gpt-test".into(),
                profile: ResponsesRequestProfile::OpenAiCompatible,
                endpoint: "https://example.test/v1/responses".into(),
                output: vec![json!({
                    "type":"compaction",
                    "encrypted_content":"opaque-compaction"
                })],
                compaction_output_tokens: None,
            }),
        };
        let retained_media = ResponsesContinuation {
            checkpoints: Vec::new(),
            native_window: Some(NativeContextWindow {
                model: "gpt-test".into(),
                profile: ResponsesRequestProfile::OpenAiCompatible,
                endpoint: "https://example.test/v1/responses".into(),
                output: vec![json!({
                    "role":"user",
                    "content":[{"type":"input_image", "image_url":"data:image/png;base64,AAAA"}]
                })],
                compaction_output_tokens: None,
            }),
        };

        assert_eq!(
            <OpenAiResponsesProvider<TestModel> as Provider>::estimate_continuation_tokens(
                &opaque_only
            ),
            0
        );
        assert_eq!(
            <OpenAiResponsesProvider<TestModel> as Provider>::estimate_continuation_tokens(
                &retained_media
            ),
            NATIVE_IMAGE_TOKEN_ESTIMATE
        );
    }

    #[test]
    fn missing_usage_ciphertext_does_not_trigger_a_false_hard_capacity_overflow() {
        let continuation = ResponsesContinuation {
            checkpoints: Vec::new(),
            native_window: Some(NativeContextWindow {
                model: "gpt-test".into(),
                profile: ResponsesRequestProfile::OpenAiCompatible,
                endpoint: "https://example.test/v1/responses".into(),
                output: vec![json!({
                    "type":"compaction",
                    "encrypted_content":"x".repeat(1_048_576)
                })],
                compaction_output_tokens: None,
            }),
        };
        let estimate =
            <OpenAiResponsesProvider<TestModel> as Provider>::estimate_continuation_tokens(
                &continuation,
            );
        let budget = 829_800;

        assert_eq!(estimate, 0);
        assert!(estimate < budget);
    }

    #[tokio::test]
    async fn native_window_provenance_rejects_model_profile_and_endpoint_changes() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/compact"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object":"response.compaction",
                "output": [{"type":"compaction", "encrypted_content":"opaque"}]
            })))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_native_compaction(true);
        let compacted = provider
            .compact_context(
                &ResponsesContinuation::default(),
                &TestModel,
                &Conversation::new(),
                CompactOptions::default(),
            )
            .await
            .unwrap();

        let other_endpoint = OpenAiResponsesProvider::<TestModel>::new("https://example.test/v1")
            .with_native_compaction(true);
        assert!(
            Provider::validate_context(&other_endpoint, &TestModel, &compacted.continuation)
                .is_err()
        );

        let other_profile =
            OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
                .with_profile(ResponsesRequestProfile::ChatGptPlan)
                .with_native_compaction(true);
        assert!(
            Provider::validate_context(&other_profile, &TestModel, &compacted.continuation)
                .is_err()
        );

        struct OtherModel;
        impl OpenAiModel for OtherModel {
            fn id(&self) -> &str {
                "other-model"
            }
            fn supports_vision(&self) -> bool {
                false
            }
        }
        let other_model_provider =
            OpenAiResponsesProvider::<OtherModel>::new(format!("{}/v1", server.uri()));
        assert!(
            Provider::validate_context(&other_model_provider, &OtherModel, &compacted.continuation)
                .is_err()
        );
    }

    #[tokio::test]
    async fn streams_and_replays_opaque_output_with_fresh_credentials() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer first"))
            .respond_with(ResponseTemplate::new(200).set_body_string(completed(json!([
                {"id":"rs_1","type":"reasoning","encrypted_content":"opaque","phase":"analysis"},
                {"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}
            ]))))
            .expect(1)
            .mount(&server).await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer second"))
            .respond_with(ResponseTemplate::new(200).set_body_string(completed(json!([
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}
            ]))))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_credentials(Arc::new(RotatingCredentials(AtomicUsize::new(0))))
            .with_instructions("developer");
        let mut conversation = Conversation::new();
        conversation.push_user_message(
            UserMessage::new("look").with_images(vec![Image::new("image/png", b"png")]),
        );
        let (assistant, continuation) = provider
            .complete(
                &ResponsesContinuation::default(),
                &TestModel,
                &conversation,
                &[],
            )
            .await
            .unwrap();
        assert_eq!(assistant.tool_calls()[0].name(), "lookup");

        let serialized = serde_json::to_value(&continuation).unwrap();
        assert_eq!(
            serialized["checkpoints"][0]["output"][0]["encrypted_content"],
            "opaque"
        );
        let (second, _) = provider
            .complete(
                &ResponsesContinuation::default(),
                &TestModel,
                &conversation,
                &[],
            )
            .await
            .unwrap();
        assert_eq!(second.content(), "done");
    }

    #[tokio::test]
    async fn preserves_responses_usage_for_complete_and_stream() {
        let server = MockServer::start().await;
        let body = completed_with_usage(
            json!([{
                "type":"message", "role":"assistant",
                "content":[{"type":"output_text", "text":"done"}]
            }]),
            json!({
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 25},
                "output_tokens": 40,
                "output_tokens_details": {"reasoning_tokens": 12},
                "total_tokens": 140
            }),
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .expect(2)
            .mount(&server)
            .await;
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()));
        let conversation = Conversation::new();

        let completion = provider
            .complete_once_with_usage(&TestModel, &conversation, &[])
            .await
            .unwrap();
        let expected = TokenUsage {
            input_tokens: Some(100),
            cached_input_tokens: Some(25),
            output_tokens: Some(40),
            reasoning_output_tokens: Some(12),
            total_tokens: Some(140),
        };
        assert_eq!(completion.message.content(), "done");
        assert_eq!(completion.usage, expected);

        let continuation = provider.create_continuation(&TestModel);
        let mut stream = provider.complete_stream(&continuation, &TestModel, &conversation, &[]);
        let ProviderEvent::Completed(completion) = stream.next().await.unwrap().unwrap() else {
            panic!("expected terminal completion");
        };
        assert_eq!(completion.usage, expected);
        server.verify().await;
    }

    #[test]
    fn assembles_done_items_when_codex_completed_output_is_empty() {
        let reasoning = json!({
            "id":"rs_1", "type":"reasoning", "encrypted_content":"opaque"
        });
        let call = json!({
            "id":"fc_1", "type":"function_call", "call_id":"call_1",
            "name":"lookup", "arguments":"{\"q\":\"x\"}"
        });
        let message = json!({
            "id":"msg_1", "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"working"}]
        });
        let body = format!(
            "event: response.output_item.added\ndata: {}\n\n{}{}{}{}",
            json!({
                "type":"response.output_item.added", "output_index":0,
                "item":{"id":"rs_1", "type":"reasoning"}
            }),
            output_item_done(0, reasoning),
            output_item_done(1, call),
            output_item_done(2, message),
            completed(json!([])),
        );
        let response = parse_complete_stream(body.as_bytes()).unwrap();
        let output = response["output"].as_array().unwrap().clone();
        let assistant = parse_output(&output).unwrap();
        let continuation = ResponsesContinuation {
            checkpoints: vec![ResponseCheckpoint {
                model: "gpt-test".into(),
                profile: ResponsesRequestProfile::OpenAiCompatible,
                assistant_message_index: 1,
                assistant: assistant.clone(),
                output,
            }],
            native_window: None,
        };

        assert_eq!(assistant.content(), "working");
        assert_eq!(assistant.tool_calls()[0].name(), "lookup");
        let serialized = serde_json::to_value(continuation).unwrap();
        let output = serialized["checkpoints"][0]["output"].as_array().unwrap();
        assert_eq!(output[0]["encrypted_content"], "opaque");
        assert_eq!(output[1]["call_id"], "call_1");
        assert_eq!(output[2]["id"], "msg_1");
    }

    #[test]
    fn completed_output_is_authoritative_and_may_add_metadata() {
        let streamed = json!({
            "id":"msg_1", "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"done"}]
        });
        let final_item = json!({
            "id":"msg_1", "type":"message", "role":"assistant",
            "status":"completed", "content":[{"type":"output_text", "text":"final"}]
        });
        let response = parse_complete_stream(
            format!(
                "{}{}",
                output_item_done(0, streamed),
                completed(json!([final_item]))
            )
            .as_bytes(),
        )
        .unwrap();
        assert_eq!(response["output"][0]["id"], "msg_1");
        assert_eq!(response["output"][0]["status"], "completed");
        assert_eq!(response["output"][0]["content"][0]["text"], "final");
    }

    #[test]
    fn chatgpt_plan_function_calls_accept_optional_namespace_and_replay_raw_items() {
        for namespace in [None, Some(Value::Null), Some(json!(FUNCTION_NAMESPACE))] {
            let mut call = json!({
                "id":"fc_1", "type":"function_call", "status":"completed",
                "call_id":"call_1", "name":"lookup", "arguments":"{\"q\":\"x\"}"
            });
            if let Some(namespace) = namespace {
                call["namespace"] = namespace;
            }

            let response = parse_complete_stream(
                format!(
                    "{}{}",
                    output_item_done(0, call.clone()),
                    completed(json!([call.clone()]))
                )
                .as_bytes(),
            )
            .unwrap();
            let output = response["output"].as_array().unwrap();
            let assistant =
                parse_output_with_profile(output, ResponsesRequestProfile::ChatGptPlan).unwrap();
            assert_eq!(assistant.tool_calls()[0].name(), "lookup");
            assert_eq!(assistant.tool_calls()[0].id(), "call_1");

            let continuation = ResponsesContinuation {
                checkpoints: vec![ResponseCheckpoint {
                    model: "gpt-test".into(),
                    profile: ResponsesRequestProfile::ChatGptPlan,
                    assistant_message_index: 1,
                    assistant: assistant.clone(),
                    output: output.clone(),
                }],
                native_window: None,
            };
            let serialized = serde_json::to_value(&continuation).unwrap();
            let restored: ResponsesContinuation = serde_json::from_value(serialized).unwrap();
            assert_eq!(restored.checkpoints[0].output[0], call);

            let mut conversation = Conversation::new();
            conversation.push_user_message(UserMessage::new("look"));
            conversation.push_message(Message::Assistant(assistant));
            let request = build_request_with_profile(
                "gpt-test",
                Default::default(),
                None,
                &conversation,
                &[],
                &restored,
                ResponsesRequestProfile::ChatGptPlan,
            )
            .unwrap();
            let replayed = &request["input"][1];
            assert_eq!(replayed, &call);
            assert_eq!(replayed.get("namespace"), call.get("namespace"));
        }
    }

    #[test]
    fn rejects_duplicate_gapped_or_inconsistent_done_items() {
        let first = json!({
            "id":"msg_1", "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"one"}]
        });
        let second = json!({
            "id":"msg_2", "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"two"}]
        });

        let duplicate = format!(
            "{}{}{}",
            output_item_done(0, first.clone()),
            output_item_done(0, second.clone()),
            completed(json!([]))
        );
        assert!(
            matches!(parse_complete_stream(duplicate.as_bytes()), Err(OpenAiError::Protocol(message)) if message.contains("duplicate"))
        );

        let gapped = format!(
            "{}{}",
            output_item_done(1, first.clone()),
            completed(json!([]))
        );
        assert!(
            matches!(parse_complete_stream(gapped.as_bytes()), Err(OpenAiError::Protocol(message)) if message.contains("contiguous"))
        );

        let inconsistent = format!(
            "{}{}",
            output_item_done(0, first),
            completed(json!([second]))
        );
        assert!(
            matches!(parse_complete_stream(inconsistent.as_bytes()), Err(OpenAiError::Protocol(message)) if message.contains("id disagrees"))
        );
    }

    #[test]
    fn rejects_truncation_after_a_done_item() {
        let stream = format!(
            "{}event: response.completed\ndata: {{",
            output_item_done(
                0,
                json!({
                    "id":"msg_1", "type":"message", "role":"assistant",
                    "content":[{"type":"output_text", "text":"partial"}]
                })
            )
        );
        assert!(
            matches!(parse_complete_stream(stream.as_bytes()), Err(OpenAiError::Protocol(message)) if message.contains("truncated"))
        );
    }

    #[test]
    fn bounds_accumulated_output_items_and_bytes() {
        let mut output = OutputAccumulator::default();
        let too_many = json!({
            "output_index": MAX_OUTPUT_ITEMS,
            "item": {"id":"msg_limit", "type":"message"}
        });
        assert!(
            matches!(output.insert(&too_many), Err(OpenAiError::Protocol(message)) if message.contains("item limit"))
        );

        let oversized = json!({
            "output_index": 0,
            "item": {
                "id":"msg_large", "type":"message",
                "content":"x".repeat(MAX_ASSEMBLED_OUTPUT_BYTES)
            }
        });
        assert!(
            matches!(output.insert(&oversized), Err(OpenAiError::Protocol(message)) if message.contains("size limit"))
        );
    }

    #[test]
    fn provider_error_events_preserve_structured_failure_codes() {
        let top_level = json!({
            "type": "error",
            "code": "server_error",
            "message": "Our servers are currently overloaded. Please try again later."
        });
        let error =
            parse_complete_stream(format!("event: error\ndata: {top_level}\n\n").as_bytes())
                .unwrap_err();
        assert!(matches!(
            error,
            OpenAiError::ResponseFailed { code: Some(code), message }
                if code == "server_error"
                    && message == "Our servers are currently overloaded. Please try again later."
        ));

        let nested_type = json!({
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "server overloaded"
            }
        });
        let error =
            parse_complete_stream(format!("event: error\ndata: {nested_type}\n\n").as_bytes())
                .unwrap_err();
        assert!(matches!(
            error,
            OpenAiError::ResponseFailed { code: Some(code), .. }
                if code == "overloaded_error"
        ));

        let generic = json!({
            "type": "error",
            "message": "Our servers are currently overloaded. Please try again later."
        });
        let error = parse_complete_stream(format!("event: error\ndata: {generic}\n\n").as_bytes())
            .unwrap_err();
        assert!(matches!(
            error,
            OpenAiError::ResponseFailed { code: None, .. }
        ));

        let top_level_quota_with_null_nested_code = json!({
            "type": "error",
            "code": "subscription_sharing_usage_limit_exceeded",
            "error": {
                "code": null,
                "type": "overloaded_error",
                "message": "Our servers are currently overloaded. Please try again later."
            }
        });
        let error = parse_complete_stream(
            format!("event: error\ndata: {top_level_quota_with_null_nested_code}\n\n").as_bytes(),
        )
        .unwrap_err();
        assert!(matches!(
            &error,
            OpenAiError::ResponseFailed { code: Some(code), .. }
                if code == "subscription_sharing_usage_limit_exceeded"
        ));
        assert!(!error.is_retryable());

        let nested_quota_precedes_transient_hints = json!({
            "type": "error",
            "code": "server_error",
            "error": {
                "code": "subscription_sharing_usage_limit_exceeded",
                "type": "overloaded_error",
                "message": "Our servers are currently overloaded. Please try again later."
            }
        });
        let error = parse_complete_stream(
            format!("event: error\ndata: {nested_quota_precedes_transient_hints}\n\n").as_bytes(),
        )
        .unwrap_err();
        assert!(matches!(
            &error,
            OpenAiError::ResponseFailed { code: Some(code), .. }
                if code == "subscription_sharing_usage_limit_exceeded"
        ));
        assert!(!error.is_retryable());
    }

    #[test]
    fn incomplete_event_wins_over_accumulated_done_items() {
        let stream = format!(
            "{}event: response.incomplete\ndata: {}\n\n",
            output_item_done(
                0,
                json!({
                    "id":"msg_1", "type":"message", "role":"assistant",
                    "content":[{"type":"output_text", "text":"partial"}]
                })
            ),
            json!({
                "type":"response.incomplete",
                "response":{"incomplete_details":{"reason":"max_output_tokens"}}
            })
        );
        assert!(matches!(
            parse_complete_stream(stream.as_bytes()),
            Err(OpenAiError::Incomplete { reason }) if reason == "max_output_tokens"
        ));
    }

    #[test]
    fn request_replaces_normalized_assistant_with_opaque_items() {
        let opaque =
            json!({"id":"rs_1","type":"reasoning","encrypted_content":"secret","phase":"analysis"});
        let continuation = ResponsesContinuation {
            checkpoints: vec![ResponseCheckpoint {
                model: "gpt-test".into(),
                profile: ResponsesRequestProfile::OpenAiCompatible,
                assistant_message_index: 1,
                assistant: AssistantMessage::new("answer"),
                output: vec![
                    opaque.clone(),
                    json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}),
                ],
            }],
            native_window: None,
        };
        let mut conversation = Conversation::new().with_system_prompt("system");
        conversation.push_user_message(UserMessage::new("question"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("answer")));
        conversation.push_user_message(UserMessage::new("next"));
        let request = build_request(
            "gpt-test",
            crate::OpenAiRequestOptions::default(),
            None,
            &conversation,
            &[],
            &continuation,
        )
        .unwrap();
        assert_eq!(request["stream"], true);
        assert_eq!(request["store"], false);
        assert_eq!(request["include"][0], "reasoning.encrypted_content");
        assert_eq!(request["input"][1], opaque);
        assert_eq!(request["instructions"], "system");
    }

    #[test]
    fn reports_incomplete_terminal_event() {
        let error = parse_complete_stream(
            format!(
                "event: response.incomplete\ndata: {}\n\n",
                json!({
                    "type":"response.incomplete",
                    "response":{"incomplete_details":{"reason":"max_output_tokens"}}
                })
            )
            .as_bytes(),
        )
        .unwrap_err();
        assert!(
            matches!(error, OpenAiError::Incomplete { reason } if reason == "max_output_tokens")
        );
    }

    #[test]
    fn continuation_debug_does_not_reveal_encrypted_reasoning() {
        let continuation = ResponsesContinuation {
            checkpoints: vec![ResponseCheckpoint {
                model: "gpt-test".into(),
                profile: ResponsesRequestProfile::OpenAiCompatible,
                assistant_message_index: 0,
                assistant: AssistantMessage::new(""),
                output: vec![json!({"type":"reasoning", "encrypted_content":"top-secret"})],
            }],
            native_window: None,
        };
        assert!(!format!("{continuation:?}").contains("top-secret"));
    }

    #[test]
    fn continuation_estimate_counts_readable_reasoning_summary_not_encrypted_bytes() {
        let summary = "readable reasoning summary";
        let continuation = ResponsesContinuation {
            checkpoints: vec![ResponseCheckpoint {
                model: "gpt-test".into(),
                profile: ResponsesRequestProfile::OpenAiCompatible,
                assistant_message_index: 0,
                assistant: AssistantMessage::new(""),
                output: vec![json!({
                    "type":"reasoning",
                    "encrypted_content":"opaque".repeat(16_384),
                    "summary":[{"type":"summary_text", "text":summary}]
                })],
            }],
            native_window: None,
        };

        assert_eq!(
            <OpenAiResponsesProvider<TestModel> as Provider>::estimate_continuation_tokens(
                &continuation
            ),
            summary.len().div_ceil(4) as u64
        );
    }

    #[test]
    fn decoder_handles_utf8_chunk_boundaries_and_crlf() {
        let stream = completed(json!([{
            "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"räksmörgås"}]
        }]))
        .replace('\n', "\r\n");
        let bytes = stream.as_bytes();
        let split = stream.find('ä').unwrap() + 1;
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(&bytes[..split]).unwrap().is_empty());
        let events = decoder.push(&bytes[split..]).unwrap();
        let response = events
            .into_iter()
            .find_map(|event| handle_sse_event(event, &mut OutputAccumulator::default()).unwrap())
            .unwrap();
        assert_eq!(
            parse_output(response["output"].as_array().unwrap())
                .unwrap()
                .content(),
            "räksmörgås"
        );
    }

    #[test]
    fn decoder_surfaces_truncated_stream_and_terminal_before_finish() {
        let mut decoder = SseDecoder::default();
        decoder.push(b"event: response.created\ndata: {").unwrap();
        assert!(matches!(decoder.finish(), Err(message) if message.contains("truncated")));

        let terminal = completed(json!([{
            "type":"message", "role":"assistant", "content":[{"type":"output_text", "text":"done"}]
        }]));
        let mut decoder = SseDecoder::default();
        let events = decoder.push(terminal.as_bytes()).unwrap();
        assert!(events.into_iter().any(|event| {
            handle_sse_event(event, &mut OutputAccumulator::default())
                .unwrap()
                .is_some()
        }));
    }

    #[test]
    fn continuation_rejects_edited_history_and_model_changes() {
        let checkpoint = ResponseCheckpoint {
            model: "gpt-test".into(),
            profile: ResponsesRequestProfile::OpenAiCompatible,
            assistant_message_index: 1,
            assistant: AssistantMessage::new("original"),
            output: vec![
                json!({"type":"message", "role":"assistant", "content":[{"type":"output_text", "text":"original"}]}),
            ],
        };
        let continuation = ResponsesContinuation {
            checkpoints: vec![checkpoint],
            native_window: None,
        };
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("question"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("edited")));
        assert!(
            build_request(
                "gpt-test",
                Default::default(),
                None,
                &conversation,
                &[],
                &continuation
            )
            .is_err()
        );
        conversation.replace_messages(vec![
            Message::User(UserMessage::new("question")),
            Message::Assistant(AssistantMessage::new("original")),
        ]);
        assert!(
            build_request(
                "different",
                Default::default(),
                None,
                &conversation,
                &[],
                &continuation
            )
            .is_err()
        );
    }

    #[test]
    fn continuation_profile_provenance_rejects_switches_without_rewriting_raw_output() {
        let raw_call = json!({
            "type":"function_call", "call_id":"call_1", "name":"lookup", "arguments":"{}"
        });
        let assistant = AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
            "call_1",
            "lookup",
            json!({}),
        )]);
        let legacy = json!({
            "checkpoints": [{
                "model":"gpt-test",
                "assistant_message_index":1,
                "assistant":serde_json::to_value(&assistant).unwrap(),
                "output":[raw_call.clone()]
            }]
        });
        let continuation: ResponsesContinuation = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            continuation.checkpoints[0].profile,
            ResponsesRequestProfile::OpenAiCompatible
        );

        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("call lookup"));
        conversation.push_message(Message::Assistant(assistant));
        let generic_request = build_request(
            "gpt-test",
            Default::default(),
            None,
            &conversation,
            &[],
            &continuation,
        )
        .unwrap();
        assert_eq!(generic_request["input"][1], raw_call);

        let error = build_request_with_profile(
            "gpt-test",
            Default::default(),
            None,
            &conversation,
            &[],
            &continuation,
            ResponsesRequestProfile::ChatGptPlan,
        )
        .unwrap_err();
        assert!(
            matches!(error, OpenAiError::Protocol(message) if message.contains("request profile"))
        );
    }

    #[test]
    fn malformed_or_unknown_output_is_rejected() {
        assert!(
            parse_output(&[json!({"type":"function_call", "call_id":"x", "name":"f"})]).is_err()
        );
        assert!(parse_output(&[json!({"type":"future_action", "id":"x"})]).is_err());
    }

    #[test]
    fn chatgpt_profile_rejects_other_or_non_string_namespaces() {
        let call = json!({
            "type":"function_call", "call_id":"x", "name":"lookup",
            "namespace":"other", "arguments":"{}"
        });
        assert!(parse_output_with_profile(&[call], ResponsesRequestProfile::ChatGptPlan).is_err());

        for namespace in [json!(7), json!(false), json!([]), json!({})] {
            let call = json!({
                "type":"function_call", "call_id":"x", "name":"lookup",
                "namespace":namespace, "arguments":"{}"
            });
            assert!(
                parse_output_with_profile(&[call], ResponsesRequestProfile::ChatGptPlan).is_err()
            );
        }
    }

    #[test]
    fn chatgpt_profile_still_requires_function_call_fields() {
        for call in [
            json!({"type":"function_call", "name":"lookup", "arguments":"{}"}),
            json!({"type":"function_call", "call_id":"x", "arguments":"{}"}),
            json!({"type":"function_call", "call_id":"x", "name":"lookup"}),
        ] {
            assert!(
                parse_output_with_profile(&[call], ResponsesRequestProfile::ChatGptPlan).is_err()
            );
        }
    }

    #[test]
    fn serializes_inline_document_as_responses_input_file() {
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("summarize").with_documents(vec![
            Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
        ]));

        let request = build_request(
            "gpt-test",
            Default::default(),
            None,
            &conversation,
            &[],
            &ResponsesContinuation::default(),
        )
        .unwrap();

        assert_eq!(
            request["input"][0]["content"][1],
            json!({
                "type": "input_file",
                "filename": "notes.pdf",
                "file_data": "data:application/pdf;base64,JVBERg==",
            })
        );
    }

    #[test]
    fn request_options_are_independent_of_the_model_id() {
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("question"));
        let request = build_request(
            "gpt-test",
            crate::OpenAiRequestOptions {
                reasoning_effort: Some("xhigh".into()),
                service_tier: Some("fast".into()),
            },
            None,
            &conversation,
            &[],
            &ResponsesContinuation::default(),
        )
        .unwrap();
        assert_eq!(request["model"], "gpt-test");
        assert_eq!(request["reasoning"]["effort"], "xhigh");
        assert_eq!(request["service_tier"], "fast");
    }

    #[test]
    fn generic_profile_keeps_system_items_and_flat_function_tools() {
        let mut conversation = Conversation::new();
        conversation.push_message(Message::System(SystemMessage::new("inline system")));
        let tools = [ToolSpec::new(
            "lookup",
            "Look up a value.",
            json!({"type":"object"}),
        )];
        let request = build_request(
            "gpt-test",
            Default::default(),
            None,
            &conversation,
            &tools,
            &ResponsesContinuation::default(),
        )
        .unwrap();

        assert_eq!(request["input"][0]["role"], "system");
        assert_eq!(request["tools"].as_array().unwrap().len(), 1);
        assert_eq!(request["tools"][0]["type"], "function");
        assert_eq!(request["tools"][0]["name"], "lookup");
        assert!(request["tools"][0].get("namespace").is_none());
    }

    struct OptionsModel;

    impl OpenAiModel for OptionsModel {
        fn id(&self) -> &str {
            "gpt-options"
        }

        fn supports_vision(&self) -> bool {
            false
        }

        fn request_options(&self) -> crate::OpenAiRequestOptions {
            crate::OpenAiRequestOptions {
                reasoning_effort: Some("future-effort".into()),
                service_tier: Some("burst".into()),
            }
        }
    }

    #[tokio::test]
    async fn sends_request_options_on_the_responses_wire() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_string(completed(json!([
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}]
                }
            ]))))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiResponsesProvider::<OptionsModel>::new(format!("{}/v1", server.uri()));
        provider
            .complete_once(&OptionsModel, &Conversation::new(), &[])
            .await
            .unwrap();

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["model"], "gpt-options");
        assert_eq!(body["reasoning"]["effort"], "future-effort");
        assert_eq!(body["service_tier"], "burst");
        server.verify().await;
    }

    #[tokio::test]
    async fn rejects_unsupported_documents_before_a_responses_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()));
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("read").with_documents(vec![
            Document::try_new("application/zip", "notes.zip", b"PK").unwrap(),
        ]));

        let error = provider
            .complete_once(&TestModel, &conversation, &[])
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            OpenAiError::UnsupportedDocument { provider: "OpenAI Responses", mime }
                if mime == "application/zip"
        ));
        server.verify().await;
    }
}
