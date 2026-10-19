//! Google Gemini GenerateContent provider for [`kodkod_core`].

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use async_stream::try_stream;
use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, CompactError, CompactOptions, ContextBudget, ContextCompaction, Conversation,
    Message, Provider, ProviderCompletion, ProviderEvent, ProviderStream, RequestDiagnostic,
    RequestDiagnosticEvent, RequestFailureKind, RequestFingerprint, RequestIdentity,
    RequestInputPrefixFingerprint, RequestMetadata, RequestObserver, RequestOperation,
    RequestOptionClass, RequestProfile, RequestPromptFingerprints, RequestUsage, RetryAfter,
    Retryable, TokenUsage, ToolCall, ToolExecutorError, ToolResult, ToolResultOutcome, ToolSpec,
};
use kodkod_http::SseDecoder;
pub use kodkod_http::{
    CredentialError, CredentialFuture, CredentialSource, RequestCredentials, StaticCredentials,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;
const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// Model properties needed by the Google GenerateContent adapter.
pub trait GoogleModel: Sync {
    fn id(&self) -> &str;

    fn supports_vision(&self) -> bool {
        false
    }

    fn supports_document(&self, _mime: &str) -> bool {
        false
    }

    fn context_budget(&self) -> Option<ContextBudget> {
        None
    }
}

impl GoogleModel for String {
    fn id(&self) -> &str {
        self
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn supports_document(&self, mime: &str) -> bool {
        self.supports_vision() && mime == "application/pdf"
    }
}

impl GoogleModel for str {
    fn id(&self) -> &str {
        self
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn supports_document(&self, mime: &str) -> bool {
        self.supports_vision() && mime == "application/pdf"
    }
}

/// Opaque signed assistant parts which Gemini requires on later turns.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct GoogleContinuation {
    #[serde(default)]
    checkpoints: Vec<GoogleCheckpoint>,
}

#[derive(Clone, Serialize, Deserialize)]
struct GoogleCheckpoint {
    model: String,
    assistant_message_index: usize,
    assistant: AssistantMessage,
    parts: Vec<Value>,
}

impl fmt::Debug for GoogleContinuation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GoogleContinuation")
            .field("checkpoint_count", &self.checkpoints.len())
            .finish_non_exhaustive()
    }
}

/// Provider for the Google Gemini `generateContent` and `streamGenerateContent` APIs.
#[derive(Clone)]
pub struct GoogleGenerateContentProvider<M = ()> {
    endpoint: String,
    credentials: Option<Arc<dyn CredentialSource>>,
    request_observer: Option<Arc<dyn RequestObserver>>,
    client: reqwest::Client,
    _model: PhantomData<fn() -> M>,
}

impl<M> GoogleGenerateContentProvider<M> {
    /// Create a provider from the API version base URL, such as
    /// `https://generativelanguage.googleapis.com/v1beta`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            endpoint: base_url.into().trim_end_matches('/').to_owned(),
            credentials: None,
            request_observer: None,
            client: reqwest::Client::new(),
            _model: PhantomData,
        }
    }

    /// Configure a Google API key using Google's `x-goog-api-key` header.
    pub fn with_api_key(mut self, api_key: impl AsRef<str>) -> Self {
        self.credentials = Some(Arc::new(GoogleApiKeyCredentials(
            api_key.as_ref().to_owned(),
        )));
        self
    }

    /// Supply application-owned per-request credentials.
    pub fn with_credentials(mut self, credentials: Arc<dyn CredentialSource>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Use a caller-configured HTTP client.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Observe content-free metadata and lifecycle diagnostics for each request.
    pub fn with_request_observer(mut self, observer: Arc<dyn RequestObserver>) -> Self {
        self.request_observer = Some(observer);
        self
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn request_url(&self, model: &str, streaming: bool) -> String {
        let operation = if streaming {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        format!(
            "{}/models/{}:{operation}",
            self.endpoint,
            encode_path_segment(model.strip_prefix("models/").unwrap_or(model))
        )
    }

    async fn request_credentials(&self) -> Result<Option<RequestCredentials>, GoogleError> {
        match &self.credentials {
            Some(source) => source
                .credentials()
                .await
                .map(Some)
                .map_err(GoogleError::Credentials),
            None => Ok(None),
        }
    }
}

struct GoogleApiKeyCredentials(String);

impl CredentialSource for GoogleApiKeyCredentials {
    fn credentials(&self) -> CredentialFuture<'_> {
        Box::pin(async {
            let value = HeaderValue::from_str(&self.0)
                .map_err(|error| CredentialError::new(error.to_string()))?;
            let mut headers = HeaderMap::new();
            headers.insert(HeaderName::from_static("x-goog-api-key"), value);
            Ok(RequestCredentials::from_headers(headers))
        })
    }
}

impl<M: GoogleModel + Send> Provider for GoogleGenerateContentProvider<M> {
    type Model = M;
    type Error = GoogleError;
    type Continuation = GoogleContinuation;

    fn with_request_observer(mut self, observer: Arc<dyn RequestObserver>) -> Self {
        self.request_observer = Some(observer);
        self
    }

    fn supports_vision(&self, model: &M) -> bool {
        model.supports_vision()
    }

    fn supports_document(&self, model: &M, mime: &str) -> bool {
        model.supports_document(mime)
    }

    fn context_budget(&self, model: &M) -> Option<ContextBudget> {
        model.context_budget()
    }

    fn create_continuation(&self, _model: &M) -> Self::Continuation {
        GoogleContinuation::default()
    }

    fn estimate_continuation_tokens(continuation: &Self::Continuation) -> u64 {
        continuation_token_estimate(continuation)
    }

    fn validate_context(&self, model: &M, continuation: &Self::Continuation) -> Result<(), String> {
        let mut previous_index = None;
        for checkpoint in &continuation.checkpoints {
            if checkpoint.model != model.id() {
                return Err("continuation belongs to a different model".into());
            }
            if previous_index.is_some_and(|index| checkpoint.assistant_message_index <= index) {
                return Err("continuation checkpoints are out of order".into());
            }
            previous_index = Some(checkpoint.assistant_message_index);
        }
        Ok(())
    }

    fn is_context_overflow(&self, error: &Self::Error) -> bool {
        error.is_context_overflow()
    }

    async fn compact_context(
        &self,
        continuation: &Self::Continuation,
        model: &Self::Model,
        conversation: &Conversation,
        options: CompactOptions,
    ) -> Result<ContextCompaction<Self::Continuation>, CompactError<Self::Error>> {
        validate_continuation(model.id(), conversation.messages(), continuation)
            .map_err(|error| CompactError::Context(error.to_string()))?;
        let compacted = self
            .compact_with_usage(model, conversation, options)
            .await?;
        let retained = compacted
            .retained_message_indices
            .iter()
            .copied()
            .collect::<std::collections::HashMap<_, _>>();
        let mut next = GoogleContinuation::default();
        for checkpoint in &continuation.checkpoints {
            if let Some(&assistant_message_index) =
                retained.get(&checkpoint.assistant_message_index)
            {
                let mut checkpoint = checkpoint.clone();
                checkpoint.assistant_message_index = assistant_message_index;
                next.checkpoints.push(checkpoint);
            }
        }
        Ok(ContextCompaction {
            conversation: compacted.conversation,
            continuation: next,
            usage: compacted.usage,
        })
    }

    async fn complete(
        &self,
        continuation: &GoogleContinuation,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, GoogleContinuation), GoogleError> {
        self.complete_with_usage(continuation, model, conversation, tools)
            .await
            .map(|completion| (completion.message, completion.continuation))
    }

    async fn complete_with_usage(
        &self,
        continuation: &GoogleContinuation,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<ProviderCompletion<GoogleContinuation>, GoogleError> {
        validate_documents(conversation, model)?;
        let request = build_request(model.id(), conversation, tools, continuation)?;
        validate_request_size(&request)?;
        let credentials = self.request_credentials().await?;
        let mut attempt = self.request_observer.as_deref().and_then(|observer| {
            RequestAttemptGuard::start(
                self.request_observer.clone(),
                google_request_metadata(
                    observer,
                    model,
                    conversation,
                    tools,
                    continuation,
                    &request,
                ),
            )
        });
        let mut builder = self
            .client
            .post(self.request_url(model.id(), false))
            .json(&request);
        if let Some(credentials) = credentials {
            builder = builder.headers(credentials.headers().clone());
        }
        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) => {
                let error = GoogleError::from(error);
                finish_attempt_failure(&mut attempt, &error, None, None);
                return Err(error);
            }
        };
        let status = response.status();
        if let Some(attempt) = &attempt {
            attempt.headers(status.as_u16(), google_request_id(&response));
        }
        if !status.is_success() {
            let error = match api_error_from_response(status.as_u16(), response).await {
                Ok(error) => error,
                Err(error) => {
                    finish_attempt_failure(&mut attempt, &error, None, None);
                    return Err(error);
                }
            };
            finish_attempt_failure(&mut attempt, &error, None, None);
            return Err(error);
        }
        let body: Value = match response.json().await {
            Ok(body) => body,
            Err(error) => {
                let error = GoogleError::from(error);
                finish_attempt_failure(&mut attempt, &error, None, None);
                return Err(error);
            }
        };
        let completion = match parse_response(&body, continuation, model.id(), conversation) {
            Ok(completion) => completion,
            Err(error) => {
                finish_attempt_failure(
                    &mut attempt,
                    &error,
                    google_response_id(&body),
                    Some(request_usage(parse_usage(&body))),
                );
                return Err(error);
            }
        };
        if let Some(attempt) = &mut attempt {
            attempt.finish(
                None,
                google_response_id(&body),
                Some(request_usage(completion.usage)),
            );
        }
        Ok(completion)
    }

    fn complete_stream<'a>(
        &'a self,
        continuation: &'a GoogleContinuation,
        model: &'a M,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, GoogleContinuation, GoogleError> {
        Box::pin(try_stream! {
            validate_documents(conversation, model)?;
            let request = build_request(model.id(), conversation, tools, continuation)?;
            validate_request_size(&request)?;
            let credentials = self.request_credentials().await?;
            let mut attempt = self.request_observer.as_deref().and_then(|observer| {
                RequestAttemptGuard::start(
                    self.request_observer.clone(),
                    google_request_metadata(
                        observer,
                        model,
                        conversation,
                        tools,
                        continuation,
                        &request,
                    ),
                )
            });
            let mut builder = self
                .client
                .post(self.request_url(model.id(), true))
                .json(&request);
            if let Some(credentials) = credentials {
                builder = builder.headers(credentials.headers().clone());
            }
            let response = match builder.send().await {
                Ok(response) => response,
                Err(error) => {
                    let error = GoogleError::from(error);
                    finish_attempt_failure(&mut attempt, &error, None, None);
                    Err(error)?;
                    unreachable!("error propagation exits the stream");
                }
            };
            let status = response.status();
            if let Some(attempt) = &attempt {
                attempt.headers(status.as_u16(), google_request_id(&response));
            }
            if !status.is_success() {
                let error = match api_error_from_response(status.as_u16(), response).await {
                    Ok(error) => error,
                    Err(error) => {
                        finish_attempt_failure(&mut attempt, &error, None, None);
                        Err(error)?;
                        unreachable!("error propagation exits the stream");
                    }
                };
                finish_attempt_failure(&mut attempt, &error, None, None);
                Err(error)?;
                unreachable!("error propagation exits the stream");
            }
            if response.headers().get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.split(';').next() == Some("application/json"))
            {
                let body: Value = match response.json().await {
                    Ok(body) => body,
                    Err(error) => {
                        let error = GoogleError::from(error);
                        finish_attempt_failure(&mut attempt, &error, None, None);
                        Err(error)?;
                        unreachable!("error propagation exits the stream");
                    }
                };
                let completion = match parse_response(
                    &body, continuation, model.id(), conversation,
                ) {
                    Ok(completion) => completion,
                    Err(error) => {
                        finish_attempt_failure(
                            &mut attempt,
                            &error,
                            google_response_id(&body),
                            Some(request_usage(parse_usage(&body))),
                        );
                        Err(error)?;
                        unreachable!("error propagation exits the stream");
                    }
                };
                if let Some(attempt) = &mut attempt {
                    attempt.finish(
                        None,
                        google_response_id(&body),
                        Some(request_usage(completion.usage)),
                    );
                }
                yield ProviderEvent::Completed(completion);
                return;
            }

            let mut decoder = SseDecoder::default();
            let mut accumulator = GoogleStreamAccumulator::default();
            let mut bytes = response.bytes_stream();
            while let Some(chunk) = bytes.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        let error = GoogleError::from(error);
                        finish_attempt_failure(&mut attempt, &error, None, None);
                        Err(error)?;
                        unreachable!("error propagation exits the stream");
                    }
                };
                let events = match decoder.push(&chunk) {
                    Ok(events) => events,
                    Err(message) => {
                        let error = GoogleError::Protocol(message);
                        finish_attempt_failure(&mut attempt, &error, None, None);
                        Err(error)?;
                        unreachable!("error propagation exits the stream");
                    }
                };
                for event in events {
                    let delta = match accumulator.push(&event.data) {
                        Ok(delta) => delta,
                        Err(error) => {
                            finish_attempt_failure(&mut attempt, &error, None, Some(request_usage(accumulator.usage)));
                            Err(error)?;
                            unreachable!("error propagation exits the stream");
                        }
                    };
                    if let Some(delta) = delta {
                        yield ProviderEvent::TextDelta(delta);
                    }
                }
            }
            if let Err(message) = decoder.finish() {
                let error = GoogleError::Protocol(message);
                finish_attempt_failure(&mut attempt, &error, None, Some(request_usage(accumulator.usage)));
                Err(error)?;
                unreachable!("error propagation exits the stream");
            }
            let stream_usage = accumulator.usage;
            let completion = match accumulator.finish(continuation, model.id(), conversation) {
                Ok(completion) => completion,
                Err(error) => {
                    finish_attempt_failure(&mut attempt, &error, None, Some(request_usage(stream_usage)));
                    Err(error)?;
                    unreachable!("error propagation exits the stream");
                }
            };
            if let Some(attempt) = &mut attempt {
                attempt.finish(None, None, Some(request_usage(completion.usage)));
            }
            yield ProviderEvent::Completed(completion);
        })
    }
}

/// Build the REST `generateContent` URL for a model ID.
pub fn generate_content_url(base_url: &str, model_id: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    let model_id = model_id.strip_prefix("models/").unwrap_or(model_id);
    format!(
        "{base_url}/models/{}:generateContent",
        encode_path_segment(model_id)
    )
}

/// Build the REST `streamGenerateContent` URL for a model ID.
pub fn stream_generate_content_url(base_url: &str, model_id: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    let model_id = model_id.strip_prefix("models/").unwrap_or(model_id);
    format!(
        "{base_url}/models/{}:streamGenerateContent?alt=sse",
        encode_path_segment(model_id)
    )
}

fn encode_path_segment(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            output.push(byte as char);
        } else {
            use fmt::Write as _;
            let _ = write!(output, "%{byte:02X}");
        }
    }
    output
}

fn build_request(
    model: &str,
    conversation: &Conversation,
    tools: &[ToolSpec],
    continuation: &GoogleContinuation,
) -> Result<Value, GoogleError> {
    if model.trim().is_empty() {
        return Err(GoogleError::Protocol("model ID is empty".into()));
    }
    validate_continuation(model, conversation.messages(), continuation)?;
    let mut system_messages = conversation
        .system_prompt()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut contents = Vec::new();
    let mut index = 0;
    while index < conversation.messages().len() {
        match &conversation.messages()[index] {
            Message::System(system) => {
                system_messages.push(system.content().to_owned());
                index += 1;
            }
            Message::ToolResult(_) => {
                let mut parts = Vec::new();
                while let Some(Message::ToolResult(result)) = conversation.messages().get(index) {
                    parts.extend(convert_tool_result(
                        result,
                        &conversation.messages()[..index],
                        continuation,
                    )?);
                    index += 1;
                }
                contents.push(json!({"role":"user", "parts":parts}));
            }
            Message::Assistant(assistant) => {
                if let Some(checkpoint) = continuation
                    .checkpoints
                    .iter()
                    .find(|checkpoint| checkpoint.assistant_message_index == index)
                {
                    contents.push(json!({"role":"model", "parts":checkpoint.parts}));
                } else {
                    contents.push(convert_assistant(assistant));
                }
                index += 1;
            }
            Message::User(user) => {
                contents.push(convert_user(user)?);
                index += 1;
            }
        }
    }
    if contents.is_empty() {
        return Err(GoogleError::Protocol(
            "GenerateContent request has no user or model content".into(),
        ));
    }
    let mut request = json!({"contents":contents});
    if !system_messages.is_empty() {
        request["systemInstruction"] = json!({"parts":[{"text":system_messages.join("\n\n")}]});
    }
    if !tools.is_empty() {
        request["tools"] = json!([{
            "functionDeclarations": tools.iter().map(|tool| json!({
                "name":tool.name(),
                "description":tool.description(),
                "parametersJsonSchema":tool.input_schema()
            })).collect::<Vec<_>>()
        }]);
    }
    Ok(request)
}

fn convert_user(user: &kodkod_core::UserMessage) -> Result<Value, GoogleError> {
    let mut parts = Vec::new();
    if !user.content().is_empty() || (user.images().is_empty() && user.documents().is_empty()) {
        parts.push(json!({"text":user.content()}));
    }
    for image in user.images() {
        parts.push(inline_data(image.mime(), image.data()));
    }
    for document in user.documents() {
        match document.mime() {
            "application/pdf" => parts.push(inline_data(document.mime(), document.data())),
            "text/plain" => {
                let text = std::str::from_utf8(document.data()).map_err(|_| {
                    GoogleError::InvalidDocumentText {
                        filename: document.filename().to_owned(),
                    }
                })?;
                parts.push(json!({"text":format!("{}\n{}", document.filename(), text)}));
            }
            mime => {
                return Err(GoogleError::UnsupportedDocument {
                    mime: mime.to_owned(),
                });
            }
        }
    }
    Ok(json!({"role":"user", "parts":parts}))
}

fn convert_assistant(assistant: &AssistantMessage) -> Value {
    let mut parts = Vec::new();
    if !assistant.content().is_empty() {
        parts.push(json!({"text":assistant.content()}));
    }
    for call in assistant.tool_calls() {
        let mut function_call = json!({
            "name":call.name(),
            "args":object_or_wrapped(call.arguments().clone())
        });
        if let Some(object) = function_call.as_object_mut() {
            object.insert("id".to_owned(), Value::String(call.id().to_owned()));
        }
        parts.push(json!({"functionCall":function_call}));
    }
    json!({"role":"model", "parts":parts})
}

fn convert_tool_result(
    result: &ToolResult,
    preceding: &[Message],
    continuation: &GoogleContinuation,
) -> Result<Vec<Value>, GoogleError> {
    let (name, remote_id) = find_tool_call(result.tool_call_id(), preceding, continuation)?;
    let response = match result.outcome() {
        ToolResultOutcome::Success(output) => object_or_wrapped(output.value().clone()),
        ToolResultOutcome::Error(ToolExecutorError::UnknownTool(name)) => {
            json!({"error":format!("unknown tool: {name}")})
        }
        ToolResultOutcome::Error(ToolExecutorError::Tool(error)) => {
            json!({"error":error.message()})
        }
    };
    let mut function_response = json!({"name":name, "response":response});
    if let (Some(id), Some(object)) = (remote_id, function_response.as_object_mut()) {
        object.insert("id".to_owned(), Value::String(id));
    }
    let mut parts = vec![json!({"functionResponse":function_response})];
    if let ToolResultOutcome::Success(output) = result.outcome() {
        parts.extend(
            output
                .images()
                .iter()
                .map(|image| inline_data(image.mime(), image.data())),
        );
    }
    Ok(parts)
}

fn find_tool_call(
    call_id: &str,
    preceding: &[Message],
    continuation: &GoogleContinuation,
) -> Result<(String, Option<String>), GoogleError> {
    for (message_index, message) in preceding.iter().enumerate().rev() {
        let Message::Assistant(assistant) = message else {
            continue;
        };
        let Some((call_index, call)) = assistant
            .tool_calls()
            .iter()
            .enumerate()
            .find(|(_, call)| call.id() == call_id)
        else {
            continue;
        };
        let checkpoint = continuation
            .checkpoints
            .iter()
            .find(|checkpoint| checkpoint.assistant_message_index == message_index);
        if let Some(checkpoint) = checkpoint {
            let function_call = checkpoint
                .parts
                .iter()
                .filter_map(|part| part.get("functionCall"))
                .nth(call_index)
                .ok_or_else(|| {
                    GoogleError::Protocol(format!(
                        "tool call {} has no matching signed function call",
                        call_id
                    ))
                })?;
            if function_call.get("name").and_then(Value::as_str) != Some(call.name()) {
                return Err(GoogleError::Protocol(format!(
                    "tool call {} does not match its signed function call",
                    call_id
                )));
            }
            let remote_id = function_call.get("id").and_then(Value::as_str);
            let expected_id = remote_id.unwrap_or_else(|| call.id());
            if call_id != expected_id {
                return Err(GoogleError::Protocol(format!(
                    "tool call {} does not match its signed function call ID",
                    call_id
                )));
            }
            return Ok((call.name().to_owned(), remote_id.map(str::to_owned)));
        }
        return Ok((call.name().to_owned(), Some(call.id().to_owned())));
    }
    Err(GoogleError::Protocol(format!(
        "tool result {} has no preceding function call",
        call_id
    )))
}

fn fallback_tool_call_id(assistant_index: usize, call_index: usize) -> String {
    format!("google-{assistant_index}-{call_index}")
}

fn inline_data(mime: &str, bytes: &[u8]) -> Value {
    let data_url = format!("data:{mime};base64,{}", encode_base64(bytes));
    let data = data_url
        .split_once(',')
        .map(|(_, encoded)| encoded)
        .unwrap_or_default();
    json!({"inlineData":{"mimeType":mime, "data":data}})
}

fn encode_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity((bytes.len() * 4).div_ceil(3));
    for chunk in bytes.chunks(3) {
        let mut block = [0; 3];
        block[..chunk.len()].copy_from_slice(chunk);
        let value = ((block[0] as u32) << 16) | ((block[1] as u32) << 8) | block[2] as u32;
        output.push(ALPHABET[((value >> 18) & 0x3f) as usize] as char);
        output.push(ALPHABET[((value >> 12) & 0x3f) as usize] as char);
        output.push(if chunk.len() > 1 {
            ALPHABET[((value >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            ALPHABET[(value & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    output
}

fn object_or_wrapped(value: Value) -> Value {
    if value.is_object() {
        value
    } else {
        json!({"result":value})
    }
}

fn validate_documents<M: GoogleModel>(
    conversation: &Conversation,
    model: &M,
) -> Result<(), GoogleError> {
    for message in conversation.messages() {
        if let Message::User(user) = message {
            if !user.images().is_empty() && !model.supports_vision() {
                return Err(GoogleError::UnsupportedImage);
            }
            for document in user.documents() {
                if !model.supports_document(document.mime()) {
                    return Err(GoogleError::UnsupportedDocument {
                        mime: document.mime().to_owned(),
                    });
                }
                if !matches!(document.mime(), "application/pdf" | "text/plain") {
                    return Err(GoogleError::UnsupportedDocument {
                        mime: document.mime().to_owned(),
                    });
                }
                if document.mime() == "text/plain" {
                    std::str::from_utf8(document.data()).map_err(|_| {
                        GoogleError::InvalidDocumentText {
                            filename: document.filename().to_owned(),
                        }
                    })?;
                }
            }
        }
    }
    Ok(())
}

fn validate_continuation(
    model: &str,
    messages: &[Message],
    continuation: &GoogleContinuation,
) -> Result<(), GoogleError> {
    let mut previous_index = None;
    for checkpoint in &continuation.checkpoints {
        if checkpoint.model != model {
            return Err(GoogleError::Protocol(
                "continuation belongs to a different model".into(),
            ));
        }
        if previous_index.is_some_and(|index| checkpoint.assistant_message_index <= index) {
            return Err(GoogleError::Protocol(
                "continuation checkpoints are out of order".into(),
            ));
        }
        if messages.get(checkpoint.assistant_message_index)
            != Some(&Message::Assistant(checkpoint.assistant.clone()))
        {
            return Err(GoogleError::Protocol(
                "continuation does not match conversation history".into(),
            ));
        }
        previous_index = Some(checkpoint.assistant_message_index);
    }
    Ok(())
}

fn parse_response(
    body: &Value,
    continuation: &GoogleContinuation,
    model: &str,
    conversation: &Conversation,
) -> Result<ProviderCompletion<GoogleContinuation>, GoogleError> {
    if let Some(error) = body.get("error") {
        return Err(api_error_from_value(200, error));
    }
    if let Some(reason) = prompt_block_reason(body) {
        return Err(GoogleError::Protocol(format!(
            "prompt was blocked: {reason}"
        )));
    }
    let candidates = body
        .get("candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| GoogleError::Protocol("response missing candidates".into()))?;
    let candidate = candidates.first().ok_or(GoogleError::EmptyResponse)?;
    let reason = required_string(candidate, "finishReason")?;
    validate_finish_reason(&reason)?;
    let content = candidate
        .get("content")
        .ok_or_else(|| GoogleError::Protocol("candidate missing content".into()))?;
    let parts = content
        .get("parts")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| GoogleError::Protocol("candidate content missing parts".into()))?;
    let assistant_index = conversation.messages().len();
    let message = parse_parts(&parts, assistant_index)?;
    if message.content().is_empty() && message.tool_calls().is_empty() {
        return Err(GoogleError::EmptyResponse);
    }
    let next = checkpoint_google(continuation, model, assistant_index, message.clone(), parts);
    Ok(ProviderCompletion::new(message, next, parse_usage(body)))
}

fn checkpoint_google(
    continuation: &GoogleContinuation,
    model: &str,
    assistant_message_index: usize,
    assistant: AssistantMessage,
    parts: Vec<Value>,
) -> GoogleContinuation {
    let mut next = continuation.clone();
    next.checkpoints.push(GoogleCheckpoint {
        model: model.to_owned(),
        assistant_message_index,
        assistant,
        parts,
    });
    next
}

fn parse_parts(parts: &[Value], assistant_index: usize) -> Result<AssistantMessage, GoogleError> {
    let mut text = String::new();
    let mut thoughts = String::new();
    let mut tool_calls = Vec::new();
    for part in parts {
        if let Some(value) = part.get("text") {
            let value = value.as_str().ok_or_else(|| {
                GoogleError::Protocol("candidate text part is not a string".into())
            })?;
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                thoughts.push_str(value);
            } else {
                text.push_str(value);
            }
            continue;
        }
        if let Some(function_call) = part.get("functionCall") {
            let name = required_string(function_call, "name")?;
            let arguments = function_call
                .get("args")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let id = function_call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| fallback_tool_call_id(assistant_index, tool_calls.len()));
            tool_calls.push(ToolCall::new(id, name, arguments));
            continue;
        }
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        if part.get("thoughtSignature").is_some() {
            continue;
        }
        return Err(GoogleError::Protocol(
            "candidate included an unsupported non-text part".into(),
        ));
    }
    let message = AssistantMessage::new(text).with_tool_calls(tool_calls);
    Ok(if thoughts.is_empty() {
        message
    } else {
        message.with_reasoning_content(thoughts)
    })
}

fn validate_finish_reason(reason: &str) -> Result<(), GoogleError> {
    match reason {
        "STOP" => Ok(()),
        "MAX_TOKENS" => Err(GoogleError::Incomplete {
            reason: reason.to_owned(),
        }),
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" => {
            Err(GoogleError::Protocol(format!(
                "candidate was blocked: {reason}"
            )))
        }
        "FINISH_REASON_UNSPECIFIED"
        | "OTHER"
        | "MALFORMED_FUNCTION_CALL"
        | "UNEXPECTED_TOOL_CALL"
        | "NO_IMAGE"
        | "IMAGE_RECITATION" => Err(GoogleError::Protocol(format!(
            "candidate did not finish normally: {reason}"
        ))),
        other => Err(GoogleError::Protocol(format!(
            "unknown candidate finish reason: {other}"
        ))),
    }
}

fn prompt_block_reason(body: &Value) -> Option<&str> {
    body.pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str)
        .filter(|reason| *reason != "BLOCK_REASON_UNSPECIFIED")
}

fn parse_usage(body: &Value) -> TokenUsage {
    let usage = body.get("usageMetadata").unwrap_or(&Value::Null);
    let candidate_tokens = usage.get("candidatesTokenCount").and_then(Value::as_u64);
    let reasoning_tokens = usage.get("thoughtsTokenCount").and_then(Value::as_u64);
    TokenUsage {
        input_tokens: usage.get("promptTokenCount").and_then(Value::as_u64),
        cached_input_tokens: usage.get("cachedContentTokenCount").and_then(Value::as_u64),
        output_tokens: match (candidate_tokens, reasoning_tokens) {
            (Some(candidate), Some(reasoning)) => Some(candidate.saturating_add(reasoning)),
            (Some(candidate), None) => Some(candidate),
            (None, Some(reasoning)) => Some(reasoning),
            (None, None) => None,
        },
        reasoning_output_tokens: reasoning_tokens,
        total_tokens: usage.get("totalTokenCount").and_then(Value::as_u64),
    }
}

fn merge_usage(target: &mut TokenUsage, body: &Value) {
    let usage = parse_usage(body);
    if usage.input_tokens.is_some() {
        target.input_tokens = usage.input_tokens;
    }
    if usage.cached_input_tokens.is_some() {
        target.cached_input_tokens = usage.cached_input_tokens;
    }
    if usage.output_tokens.is_some() {
        target.output_tokens = usage.output_tokens;
    }
    if usage.reasoning_output_tokens.is_some() {
        target.reasoning_output_tokens = usage.reasoning_output_tokens;
    }
    if usage.total_tokens.is_some() {
        target.total_tokens = usage.total_tokens;
    }
}

#[derive(Default)]
struct GoogleStreamAccumulator {
    parts: Vec<Value>,
    finish_reason: Option<String>,
    usage: TokenUsage,
}

impl GoogleStreamAccumulator {
    fn push(&mut self, data: &str) -> Result<Option<String>, GoogleError> {
        if data == "[DONE]" {
            return Ok(None);
        }
        let body: Value = serde_json::from_str(data)?;
        if let Some(error) = body.get("error") {
            return Err(api_error_from_value(200, error));
        }
        if let Some(reason) = prompt_block_reason(&body) {
            return Err(GoogleError::Protocol(format!(
                "prompt was blocked: {reason}"
            )));
        }
        merge_usage(&mut self.usage, &body);
        let candidates = body.get("candidates").and_then(Value::as_array);
        if let Some(candidate) = candidates.and_then(|candidates| candidates.first()) {
            let parts = candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if self.finish_reason.is_some() && !parts.is_empty() {
                return Err(GoogleError::Protocol(
                    "stream emitted candidate content after its finish reason".into(),
                ));
            }
            let mut delta = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str)
                    && part.get("thought").and_then(Value::as_bool) != Some(true)
                {
                    delta.push_str(text);
                }
                append_stream_part(&mut self.parts, part)?;
            }
            if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
                if self.finish_reason.is_some() {
                    return Err(GoogleError::Protocol(
                        "stream emitted more than one candidate finish reason".into(),
                    ));
                }
                self.finish_reason = Some(reason.to_owned());
            }
            return Ok((!delta.is_empty()).then_some(delta));
        }
        Ok(None)
    }

    fn finish(
        self,
        continuation: &GoogleContinuation,
        model: &str,
        conversation: &Conversation,
    ) -> Result<ProviderCompletion<GoogleContinuation>, GoogleError> {
        let reason = self.finish_reason.ok_or_else(|| {
            GoogleError::Protocol("stream ended before a candidate finish reason".into())
        })?;
        validate_finish_reason(&reason)?;
        let assistant_index = conversation.messages().len();
        let message = parse_parts(&self.parts, assistant_index)?;
        if message.content().is_empty() && message.tool_calls().is_empty() {
            return Err(GoogleError::EmptyResponse);
        }
        let next = checkpoint_google(
            continuation,
            model,
            assistant_index,
            message.clone(),
            self.parts,
        );
        Ok(ProviderCompletion::new(message, next, self.usage))
    }
}

fn append_stream_part(parts: &mut Vec<Value>, incoming: Value) -> Result<(), GoogleError> {
    if let Some(function_call) = incoming.get("functionCall") {
        let id = function_call.get("id").and_then(Value::as_str);
        if id.is_some() {
            let name = function_call
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| GoogleError::Protocol("stream function call missing name".into()))?;
            if let Some(existing) = parts.iter_mut().find(|part| {
                let Some(existing) = part.get("functionCall") else {
                    return false;
                };
                existing.get("name").and_then(Value::as_str) == Some(name)
                    && existing.get("id").and_then(Value::as_str) == id
            }) {
                let existing_call = existing
                    .get_mut("functionCall")
                    .and_then(Value::as_object_mut)
                    .ok_or_else(|| {
                        GoogleError::Protocol("stream function call is malformed".into())
                    })?;
                let incoming_call = function_call.as_object().ok_or_else(|| {
                    GoogleError::Protocol("stream function call is malformed".into())
                })?;
                for (key, value) in incoming_call {
                    if key == "args" {
                        merge_json_objects(
                            existing_call.entry(key.clone()).or_insert(json!({})),
                            value,
                        );
                    } else {
                        existing_call.insert(key.clone(), value.clone());
                    }
                }
                if let Some(object) = incoming.as_object() {
                    for (key, value) in object {
                        if key != "functionCall"
                            && let Some(target) = existing.as_object_mut()
                        {
                            target.insert(key.clone(), value.clone());
                        }
                    }
                }
                return Ok(());
            }
        }
    }
    parts.push(incoming);
    Ok(())
}

fn merge_json_objects(target: &mut Value, incoming: &Value) {
    match (target.as_object_mut(), incoming.as_object()) {
        (Some(target), Some(incoming)) => {
            for (key, value) in incoming {
                merge_json_objects(target.entry(key.clone()).or_insert(Value::Null), value);
            }
        }
        _ => *target = incoming.clone(),
    }
}

fn required_string(value: &Value, field: &str) -> Result<String, GoogleError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| GoogleError::Protocol(format!("response missing {field}")))
}

fn validate_request_size(request: &Value) -> Result<(), GoogleError> {
    if serde_json::to_vec(request)?.len() > MAX_REQUEST_BYTES {
        return Err(GoogleError::RequestLimitExceeded {
            limit_bytes: MAX_REQUEST_BYTES,
        });
    }
    Ok(())
}

async fn api_error_from_response(
    status: u16,
    response: reqwest::Response,
) -> Result<GoogleError, GoogleError> {
    let retry_after = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| RetryAfter::from_header(status, value, std::time::SystemTime::now()));
    let body = read_limited_body(response, MAX_ERROR_BODY_BYTES).await?;
    let parsed = serde_json::from_slice::<Value>(&body).ok();
    let error = parsed
        .as_ref()
        .and_then(|value| value.get("error"))
        .unwrap_or_else(|| parsed.as_ref().unwrap_or(&Value::Null));
    let mut result = api_error_from_value(status, error);
    if let GoogleError::Api {
        retry_after: api_retry_after,
        ..
    } = &mut result
    {
        *api_retry_after = retry_after;
    }
    if let GoogleError::Api { message, .. } = &mut result
        && message.is_empty()
    {
        *message = if body.is_empty() {
            status.to_string()
        } else {
            String::from_utf8_lossy(&body).into_owned()
        };
    }
    Ok(result)
}

async fn read_limited_body(
    response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, GoogleError> {
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
    Ok(bytes)
}

struct RequestAttemptGuard {
    observer: Arc<dyn RequestObserver>,
    identity: RequestIdentity,
    started_at: std::time::Instant,
    terminal: bool,
}

impl RequestAttemptGuard {
    fn start(
        observer: Option<Arc<dyn RequestObserver>>,
        metadata: RequestMetadata,
    ) -> Option<Self> {
        let observer = observer?;
        let identity = RequestIdentity::next_process_unique();
        observe_diagnostic(
            observer.as_ref(),
            RequestDiagnostic {
                identity,
                operation: RequestOperation::Completion,
                metadata: Some(metadata),
                event: RequestDiagnosticEvent::Started,
            },
        );
        Some(Self {
            observer,
            identity,
            started_at: std::time::Instant::now(),
            terminal: false,
        })
    }

    fn headers(&self, status: u16, provider_request_id: Option<String>) {
        observe_diagnostic(
            self.observer.as_ref(),
            RequestDiagnostic {
                identity: self.identity,
                operation: RequestOperation::Completion,
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
                operation: RequestOperation::Completion,
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
                operation: RequestOperation::Completion,
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

fn request_fingerprint_for_value(
    observer: &dyn RequestObserver,
    domain: &str,
    value: &Value,
) -> RequestFingerprint {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    request_fingerprint(observer, domain, &bytes)
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

fn google_request_id(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get("x-goog-request-id")
        .or_else(|| response.headers().get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| safe_provider_id(Some(value)))
}

fn google_response_id(body: &Value) -> Option<String> {
    safe_provider_id(
        body.get("responseId")
            .or_else(|| body.get("id"))
            .and_then(Value::as_str),
    )
}

fn google_request_metadata<M: GoogleModel + Send>(
    observer: &dyn RequestObserver,
    model: &M,
    conversation: &Conversation,
    tools: &[ToolSpec],
    continuation: &GoogleContinuation,
    request: &Value,
) -> RequestMetadata {
    let contents = request
        .get("contents")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut prompt_fingerprints = RequestPromptFingerprints::default();
    if let Some(instruction) = request.get("systemInstruction") {
        prompt_fingerprints.instructions = Some(request_fingerprint_for_value(
            observer,
            "instructions",
            instruction,
        ));
    }
    if let Some(tool_config) = request.get("tools") {
        prompt_fingerprints.tools = Some(request_fingerprint_for_value(
            observer,
            "tools",
            tool_config,
        ));
    }
    for count in [1usize, 4, 16, 64] {
        if contents.len() >= count {
            let prefix = Value::Array(contents[..count].to_vec());
            prompt_fingerprints
                .input_prefixes
                .push(RequestInputPrefixFingerprint {
                    item_count: Some(count),
                    fingerprint: request_fingerprint_for_value(
                        observer,
                        &format!("input-prefix-{count}"),
                        &prefix,
                    ),
                });
        }
    }
    let all_contents = Value::Array(contents.to_vec());
    prompt_fingerprints
        .input_prefixes
        .push(RequestInputPrefixFingerprint {
            item_count: None,
            fingerprint: request_fingerprint_for_value(observer, "input-total", &all_contents),
        });
    let mut attachments = GoogleAttachmentCounts::default();
    count_inline_data(request, &mut attachments);
    let continuation_bytes = continuation
        .checkpoints
        .iter()
        .map(|checkpoint| serde_json::to_vec(&checkpoint.parts).map_or(0, |bytes| bytes.len()))
        .sum();
    RequestMetadata {
        model: safe_model_id(model.id()),
        profile: RequestProfile::GoogleGenerateContent,
        reasoning_effort: RequestOptionClass::None,
        service_tier: RequestOptionClass::None,
        vision_enabled: model.supports_vision(),
        context_estimate_tokens: conversation
            .estimate_tokens(tools)
            .saturating_add(continuation_token_estimate(continuation)),
        context_budget_tokens: model.context_budget().map(|budget| budget.max_input_tokens),
        message_count: conversation.messages().len(),
        tool_definition_count: tools.len(),
        image_count: attachments.image_count,
        image_bytes: Some(attachments.image_bytes),
        document_count: attachments.document_count,
        document_bytes: Some(attachments.document_bytes),
        serialized_request_bytes: serde_json::to_vec(request).map_or(0, |bytes| bytes.len()),
        continuation_bytes,
        prompt_fingerprints,
    }
}

fn safe_model_id(value: &str) -> Option<String> {
    (!value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
        }))
    .then(|| value.to_owned())
}

fn continuation_token_estimate(continuation: &GoogleContinuation) -> u64 {
    continuation
        .checkpoints
        .iter()
        .flat_map(|checkpoint| &checkpoint.parts)
        .filter_map(|part| part.get("thoughtSignature").and_then(Value::as_str))
        .map(|signature| signature.len().div_ceil(4) as u64)
        .sum()
}

#[derive(Default)]
struct GoogleAttachmentCounts {
    image_count: usize,
    image_bytes: usize,
    document_count: usize,
    document_bytes: usize,
}

fn count_inline_data(value: &Value, counts: &mut GoogleAttachmentCounts) {
    if let Some(inline_data) = value.get("inlineData") {
        let mime = inline_data
            .get("mimeType")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let bytes = inline_data
            .get("data")
            .and_then(Value::as_str)
            .map(base64_decoded_len)
            .unwrap_or_default();
        if mime.starts_with("image/") {
            counts.image_count += 1;
            counts.image_bytes = counts.image_bytes.saturating_add(bytes);
        } else {
            counts.document_count += 1;
            counts.document_bytes = counts.document_bytes.saturating_add(bytes);
        }
        return;
    }
    match value {
        Value::Array(values) => values
            .iter()
            .for_each(|value| count_inline_data(value, counts)),
        Value::Object(values) => values
            .values()
            .for_each(|value| count_inline_data(value, counts)),
        _ => {}
    }
}

fn base64_decoded_len(value: &str) -> usize {
    let padding = value
        .bytes()
        .rev()
        .take_while(|byte| *byte == b'=')
        .count()
        .min(2);
    value
        .len()
        .saturating_mul(3)
        .saturating_div(4)
        .saturating_sub(padding)
}

fn request_usage(token_usage: TokenUsage) -> RequestUsage {
    RequestUsage {
        token_usage,
        cache_write_tokens: None,
        cache_diagnostics: None,
        effective_service_tier: None,
    }
}

fn failure_kind(error: &GoogleError) -> RequestFailureKind {
    match error {
        GoogleError::Http(_) => RequestFailureKind::Transport,
        GoogleError::Api { status, .. } => RequestFailureKind::HttpStatus(*status),
        GoogleError::Incomplete { .. } => RequestFailureKind::Incomplete,
        GoogleError::Json(_) | GoogleError::EmptyResponse | GoogleError::Protocol(_) => {
            RequestFailureKind::Protocol
        }
        GoogleError::Credentials(_)
        | GoogleError::UnsupportedImage
        | GoogleError::UnsupportedDocument { .. }
        | GoogleError::InvalidDocumentText { .. }
        | GoogleError::RequestLimitExceeded { .. } => RequestFailureKind::Unknown,
    }
}

fn finish_attempt_failure(
    attempt: &mut Option<RequestAttemptGuard>,
    error: &GoogleError,
    response_id: Option<String>,
    usage: Option<RequestUsage>,
) {
    if let Some(attempt) = attempt {
        attempt.finish(Some(failure_kind(error)), response_id, usage);
    }
}

fn api_error_from_value(status: u16, error: &Value) -> GoogleError {
    let code = error
        .get("status")
        .and_then(Value::as_str)
        .or_else(|| error.get("code").and_then(Value::as_str))
        .map(str::to_owned);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    GoogleError::Api {
        status,
        code,
        message,
        retry_after: None,
    }
}

#[derive(Debug)]
pub enum GoogleError {
    Http(reqwest::Error),
    Json(serde_json::Error),
    Credentials(CredentialError),
    UnsupportedImage,
    UnsupportedDocument {
        mime: String,
    },
    InvalidDocumentText {
        filename: String,
    },
    RequestLimitExceeded {
        limit_bytes: usize,
    },
    Api {
        status: u16,
        code: Option<String>,
        message: String,
        retry_after: Option<RetryAfter>,
    },
    EmptyResponse,
    Incomplete {
        reason: String,
    },
    Protocol(String),
}

impl GoogleError {
    pub fn is_context_overflow(&self) -> bool {
        match self {
            Self::Api {
                status: 400,
                code,
                message,
                ..
            } => {
                code.as_deref() == Some("context_length_exceeded") || {
                    let message = message.to_ascii_lowercase();
                    message.contains("context length exceeded")
                        || message.contains("maximum context length")
                        || message.contains("context window exceeded")
                        || (message.contains("input token count")
                            && message.contains("maximum")
                            && (message.contains("exceed") || message.contains("limit")))
                }
            }
            _ => false,
        }
    }
}

impl fmt::Display for GoogleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(error) => write!(f, "http request failed: {error}"),
            Self::Json(error) => write!(f, "failed to parse response: {error}"),
            Self::Credentials(error) => write!(f, "credentials unavailable: {error}"),
            Self::UnsupportedImage => {
                f.write_str("Google GenerateContent model does not support images")
            }
            Self::UnsupportedDocument { mime } => write!(
                f,
                "Google GenerateContent does not support inline documents with MIME type {mime}"
            ),
            Self::InvalidDocumentText { filename } => {
                write!(f, "text/plain document {filename:?} is not valid UTF-8")
            }
            Self::RequestLimitExceeded { limit_bytes } => write!(
                f,
                "Google GenerateContent request exceeds the {limit_bytes}-byte limit"
            ),
            Self::Api {
                status,
                code,
                message,
                ..
            } => match code {
                Some(code) => write!(f, "api error ({status}, {code}): {message}"),
                None => write!(f, "api error ({status}): {message}"),
            },
            Self::EmptyResponse => f.write_str("GenerateContent returned no usable content"),
            Self::Incomplete { reason } => write!(f, "candidate incomplete: {reason}"),
            Self::Protocol(message) => {
                write!(f, "Google GenerateContent protocol error: {message}")
            }
        }
    }
}

impl Error for GoogleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Http(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Credentials(error) => Some(error),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for GoogleError {
    fn from(error: reqwest::Error) -> Self {
        Self::Http(error)
    }
}

impl From<serde_json::Error> for GoogleError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl Retryable for GoogleError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Http(error) => error.is_connect() || error.is_timeout() || error.is_request(),
            Self::Api { status, .. } => matches!(status, 408 | 429 | 500 | 502 | 503 | 504 | 529),
            _ => false,
        }
    }

    fn retry_after(&self) -> Option<RetryAfter> {
        match self {
            Self::Api {
                status: 429 | 503,
                retry_after,
                ..
            } => *retry_after,
            _ => None,
        }
    }

    fn http_status(&self) -> Option<u16> {
        match self {
            Self::Api { status, .. } => Some(*status),
            _ => None,
        }
    }

    fn api_code(&self) -> Option<&str> {
        match self {
            Self::Api { code, .. } => code.as_deref(),
            _ => None,
        }
    }
}

impl From<CredentialError> for GoogleError {
    fn from(error: CredentialError) -> Self {
        Self::Credentials(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kodkod_core::{
        Image, RequestDiagnostic, SystemMessage, ToolOutput, ToolResult, UserMessage,
    };
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
    use serde_json::json;
    use std::sync::Mutex;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct TestModel;

    impl GoogleModel for TestModel {
        fn id(&self) -> &str {
            "gemini-test"
        }

        fn supports_vision(&self) -> bool {
            true
        }

        fn supports_document(&self, mime: &str) -> bool {
            mime == "application/pdf"
        }

        fn context_budget(&self) -> Option<ContextBudget> {
            Some(ContextBudget {
                max_input_tokens: 100_000,
            })
        }
    }

    fn provider(base: String) -> GoogleGenerateContentProvider<TestModel> {
        GoogleGenerateContentProvider::new(format!("{}/v1beta", base.trim_end_matches('/')))
            .with_credentials(Arc::new(StaticCredentials::bearer("test-token")))
    }

    #[derive(Default)]
    struct RecordingObserver(Mutex<Vec<RequestDiagnostic>>);

    impl RequestObserver for RecordingObserver {
        fn observe(&self, diagnostic: RequestDiagnostic) {
            self.0.lock().unwrap().push(diagnostic);
        }

        fn fingerprint(&self, _domain: &str, _bytes: &[u8]) -> Option<String> {
            Some("test-keyed-fingerprint".into())
        }
    }

    fn response(parts: Value) -> String {
        json!({
            "candidates":[{"content":{"role":"model", "parts":parts}, "finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":20,"cachedContentTokenCount":4,"candidatesTokenCount":8,"thoughtsTokenCount":2,"totalTokenCount":30}
        }).to_string()
    }

    #[tokio::test]
    async fn posts_schema_images_and_function_results_then_replays_signed_parts() {
        let server = MockServer::start().await;
        let signed_parts = json!([
            {"text":"I will inspect that.", "thought":true, "thoughtSignature":"signed-thought"},
            {"functionCall":{"name":"lookup", "args":{"query":"x"}, "id":"call-1"}, "thoughtSignature":"signed-call"}
        ]);
        let second_response = response(json!([{"text":"done"}]));
        let captured = server.uri();
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-test:generateContent"))
            .and(header(AUTHORIZATION, "Bearer test-token"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(response(signed_parts.clone())),
            )
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        let first_provider = provider(captured.clone());
        let mut conversation = Conversation::new().with_system_prompt("system prompt");
        conversation.push_message(Message::System(SystemMessage::new("second system")));
        conversation.push_user_message(
            UserMessage::new("inspect").with_images(vec![Image::new("image/png", b"png")]),
        );
        let tools = [ToolSpec::new(
            "lookup",
            "lookup by query",
            json!({
                "type":"object", "properties":{"query":{"type":"string"}}, "required":["query"],
                "additionalProperties":false
            }),
        )];
        let completion = first_provider
            .complete_once_with_usage(&TestModel, &conversation, &tools)
            .await
            .unwrap();
        assert_eq!(completion.message.content(), "");
        assert_eq!(completion.message.tool_calls()[0].id(), "call-1");
        assert_eq!(
            completion.message.reasoning_content(),
            Some("I will inspect that.")
        );
        assert_eq!(completion.usage.input_tokens, Some(20));
        assert_eq!(completion.usage.cached_input_tokens, Some(4));
        assert_eq!(completion.usage.output_tokens, Some(10));
        assert_eq!(completion.usage.reasoning_output_tokens, Some(2));
        assert_eq!(completion.usage.total_tokens, Some(30));
        assert_eq!(
            GoogleGenerateContentProvider::<TestModel>::estimate_continuation_tokens(
                &completion.continuation
            ),
            7
        );

        conversation.push_message(Message::Assistant(completion.message));
        conversation.push_message(Message::ToolResult(ToolResult::success(
            "call-1",
            ToolOutput::new(json!({"found":true}))
                .with_images(vec![Image::new("image/png", b"out")]),
        )));
        let replay_provider = provider(captured);
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-test:generateContent"))
            .respond_with(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                assert_eq!(
                    body["systemInstruction"]["parts"][0]["text"],
                    "system prompt\n\nsecond system"
                );
                assert_eq!(body["contents"][1]["parts"], signed_parts);
                assert_eq!(
                    body["contents"][2]["parts"][0]["functionResponse"]["name"],
                    "lookup"
                );
                assert_eq!(
                    body["contents"][2]["parts"][0]["functionResponse"]["id"],
                    "call-1"
                );
                assert_eq!(
                    body["contents"][2]["parts"][1]["inlineData"]["data"],
                    "b3V0"
                );
                ResponseTemplate::new(200).set_body_string(second_response.clone())
            })
            .expect(1)
            .mount(&server)
            .await;
        let next = replay_provider
            .complete_with_usage(&completion.continuation, &TestModel, &conversation, &tools)
            .await
            .unwrap();
        assert_eq!(next.message.content(), "done");
    }

    #[tokio::test]
    async fn streams_deltas_and_final_usage_and_requires_terminal_reason() {
        let server = MockServer::start().await;
        let stream_body = concat!(
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2,\"thoughtsTokenCount\":1,\"totalTokenCount\":7}}\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-test:streamGenerateContent"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header(CONTENT_TYPE, "text/event-stream")
                    .set_body_string(stream_body),
            )
            .expect(1)
            .mount(&server)
            .await;
        let provider = provider(server.uri());
        let mut conversation = Conversation::new().with_system_prompt("system");
        conversation.push_user_message(UserMessage::new("say hello"));
        let continuation = GoogleContinuation::default();
        let mut stream = provider.complete_stream(&continuation, &TestModel, &conversation, &[]);
        let mut deltas = String::new();
        let mut completed = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                ProviderEvent::TextDelta(delta) => deltas.push_str(&delta),
                ProviderEvent::Completed(value) => completed = Some(value),
            }
        }
        assert_eq!(deltas, "hello");
        let completion = completed.unwrap();
        assert_eq!(completion.message.content(), "hello");
        assert_eq!(completion.usage.input_tokens, Some(5));
        assert_eq!(completion.usage.output_tokens, Some(3));
        assert_eq!(completion.usage.reasoning_output_tokens, Some(1));

        let mut incomplete = GoogleStreamAccumulator::default();
        incomplete
            .push(r#"{"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}"#)
            .unwrap();
        assert!(
            incomplete
                .finish(&GoogleContinuation::default(), "gemini-test", &conversation)
                .is_err()
        );
    }

    #[test]
    fn parallel_same_name_calls_without_native_ids_stay_distinct_and_replay_by_ordinal() {
        let mut parts = Vec::new();
        append_stream_part(
            &mut parts,
            json!({
                "functionCall":{"name":"lookup", "args":{"query":"first"}},
                "thoughtSignature":"signed-first"
            }),
        )
        .unwrap();
        append_stream_part(
            &mut parts,
            json!({
                "functionCall":{"name":"lookup", "args":{"query":"second"}},
                "thoughtSignature":"signed-second"
            }),
        )
        .unwrap();
        assert_eq!(parts.len(), 2);

        let assistant = parse_parts(&parts, 0).unwrap();
        assert_eq!(assistant.tool_calls().len(), 2);
        assert_eq!(assistant.tool_calls()[0].arguments()["query"], "first");
        assert_eq!(assistant.tool_calls()[1].arguments()["query"], "second");
        let continuation = GoogleContinuation {
            checkpoints: vec![GoogleCheckpoint {
                model: "gemini-test".into(),
                assistant_message_index: 0,
                assistant: assistant.clone(),
                parts: parts.clone(),
            }],
        };
        let mut history = Conversation::new();
        history.push_message(Message::Assistant(assistant));
        let first = convert_tool_result(
            &ToolResult::success("google-0-0", json!({"value":"first"})),
            history.messages(),
            &continuation,
        )
        .unwrap();
        let second = convert_tool_result(
            &ToolResult::success("google-0-1", json!({"value":"second"})),
            history.messages(),
            &continuation,
        )
        .unwrap();
        assert_eq!(first[0]["functionResponse"]["response"]["value"], "first");
        assert_eq!(second[0]["functionResponse"]["response"]["value"], "second");
        assert!(first[0]["functionResponse"].get("id").is_none());
        assert_eq!(continuation.checkpoints[0].parts, parts);
    }

    #[test]
    fn no_id_call_does_not_inherit_native_id_from_earlier_same_name_call() {
        let parts = vec![
            json!({"functionCall":{"name":"lookup", "args":{"query":"native"}, "id":"native-A"}}),
            json!({"functionCall":{"name":"lookup", "args":{"query":"fallback"}}}),
        ];
        let assistant = parse_parts(&parts, 0).unwrap();
        let continuation = GoogleContinuation {
            checkpoints: vec![GoogleCheckpoint {
                model: "gemini-test".into(),
                assistant_message_index: 0,
                assistant: assistant.clone(),
                parts,
            }],
        };
        let mut history = Conversation::new();
        history.push_message(Message::Assistant(assistant));
        let native = convert_tool_result(
            &ToolResult::success("native-A", json!({"value":"native"})),
            history.messages(),
            &continuation,
        )
        .unwrap();
        let fallback = convert_tool_result(
            &ToolResult::success("google-0-1", json!({"value":"fallback"})),
            history.messages(),
            &continuation,
        )
        .unwrap();
        assert_eq!(native[0]["functionResponse"]["id"], "native-A");
        assert!(fallback[0]["functionResponse"].get("id").is_none());
    }

    #[tokio::test]
    async fn compaction_rebases_signed_tool_checkpoint_and_replays_it_afterward() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-test:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_string(response(json!([
                {"functionCall":{"name":"summarize", "args":{"summary":"old history"}}}
            ]))))
            .expect(1)
            .mount(&server)
            .await;

        let parts = vec![json!({
            "functionCall":{"name":"lookup", "args":{"query":"latest"}},
            "thoughtSignature":"must-survive-compaction"
        })];
        let assistant = parse_parts(&parts, 3).unwrap();
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("old user request"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("x".repeat(600))));
        conversation.push_user_message(UserMessage::new("latest request"));
        conversation.push_message(Message::Assistant(assistant.clone()));
        conversation.push_message(Message::ToolResult(ToolResult::success(
            "google-3-0",
            json!({"found":true}),
        )));
        let continuation = GoogleContinuation {
            checkpoints: vec![GoogleCheckpoint {
                model: "gemini-test".into(),
                assistant_message_index: 3,
                assistant,
                parts: parts.clone(),
            }],
        };
        let tail_tokens = conversation.messages()[3].estimate_tokens()
            + conversation.messages()[4].estimate_tokens();
        let provider = provider(server.uri());
        let compacted = provider
            .compact_context(
                &continuation,
                &TestModel,
                &conversation,
                CompactOptions {
                    keep_tail_tokens: tail_tokens,
                    ..CompactOptions::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(compacted.continuation.checkpoints.len(), 1);
        let checkpoint = &compacted.continuation.checkpoints[0];
        assert_eq!(checkpoint.assistant_message_index, 2);
        assert_eq!(checkpoint.parts, parts);
        let request = build_request(
            "gemini-test",
            &compacted.conversation,
            &[],
            &compacted.continuation,
        )
        .unwrap();
        assert_eq!(request["contents"][2]["parts"], json!(parts));
        assert!(
            request["contents"][3]["parts"][0]["functionResponse"]
                .get("id")
                .is_none()
        );
    }

    #[tokio::test]
    async fn complete_with_usage_emits_content_free_google_diagnostics() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-test:generateContent"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-goog-request-id", "google-request-1")
                    .set_body_string(response(json!([{"text":"private response"}]))),
            )
            .expect(1)
            .mount(&server)
            .await;
        let observer = Arc::new(RecordingObserver::default());
        let provider = provider(server.uri()).with_request_observer(observer.clone());
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("private prompt"));
        provider
            .complete_once_with_usage(&TestModel, &conversation, &[])
            .await
            .unwrap();

        let diagnostics = observer.0.lock().unwrap();
        assert_eq!(diagnostics.len(), 3);
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.metadata.as_ref().is_some_and(|metadata| {
                metadata.profile == RequestProfile::GoogleGenerateContent
                    && metadata.model.as_deref() == Some("gemini-test")
                    && metadata.prompt_fingerprints.input_prefixes.len() == 2
            })
        }));
        assert!(diagnostics.iter().any(|diagnostic| {
            matches!(
                diagnostic.event,
                RequestDiagnosticEvent::Headers { status: 200, .. }
            )
        }));
        assert!(diagnostics.iter().any(|diagnostic| {
            matches!(
                diagnostic.event,
                RequestDiagnosticEvent::Finished {
                    failure: None,
                    usage: Some(RequestUsage {
                        token_usage: TokenUsage {
                            output_tokens: Some(10),
                            ..
                        },
                        ..
                    }),
                    ..
                }
            )
        }));
        assert!(!format!("{diagnostics:?}").contains("private prompt"));
        assert!(!format!("{diagnostics:?}").contains("private response"));
    }

    #[test]
    fn continuation_debug_hides_signed_content_and_urls_encode_model_ids() {
        let continuation = GoogleContinuation {
            checkpoints: vec![GoogleCheckpoint {
                model: "model".into(),
                assistant_message_index: 0,
                assistant: AssistantMessage::new("text"),
                parts: vec![json!({"thoughtSignature":"private-signature"})],
            }],
        };
        assert!(!format!("{continuation:?}").contains("private-signature"));
        assert_eq!(
            generate_content_url("https://example.test/v1beta/", "models/a b"),
            "https://example.test/v1beta/models/a%20b:generateContent"
        );
        assert_eq!(
            stream_generate_content_url("https://example.test/v1beta", "model"),
            "https://example.test/v1beta/models/model:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn retries_only_transient_statuses_and_context_overflow_is_typed() {
        assert!(
            GoogleError::Api {
                status: 429,
                code: None,
                message: "rate limited".into(),
                retry_after: None,
            }
            .is_retryable()
        );
        assert!(
            !GoogleError::Api {
                status: 403,
                code: None,
                message: "forbidden".into(),
                retry_after: None,
            }
            .is_retryable()
        );
        assert!(
            GoogleError::Api {
                status: 400,
                code: Some("INVALID_ARGUMENT".into()),
                message: "input token count exceeds maximum".into(),
                retry_after: None,
            }
            .is_context_overflow()
        );
        assert!(
            !GoogleError::Api {
                status: 400,
                code: Some("INVALID_ARGUMENT".into()),
                message: "context length is too long to describe".into(),
                retry_after: None,
            }
            .is_context_overflow()
        );
    }
}
