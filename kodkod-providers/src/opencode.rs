use std::error::Error;
use std::fmt;
use std::sync::{Arc, OnceLock};

use futures_util::StreamExt;
use kodkod_anthropic::{
    AnthropicContinuation, AnthropicError, AnthropicMessagesProvider, AnthropicModel,
};
use kodkod_core::{
    AssistantMessage, CompactError, CompactOptions, ContextBudget, ContextCompaction, Conversation,
    Message, Provider, ProviderCompletion, ProviderEvent, ProviderStream, RequestObserver,
    Retryable, ToolSpec,
};
use kodkod_google::{GoogleContinuation, GoogleError, GoogleGenerateContentProvider, GoogleModel};
use kodkod_http::{CredentialSource, RequestCredentials, StaticCredentials};
use kodkod_openai::{
    OpenAiCompatibleProvider, OpenAiError, OpenAiModel, OpenAiRequestOptions,
    OpenAiResponsesProvider, ResponsesContinuation,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, USER_AGENT};
use serde::{Deserialize, Serialize};

use crate::{Protocol, ProviderConfigError, fixed_endpoint_client, header_value};
pub const OPEN_CODE_GO_ENDPOINT: &str = "https://opencode.ai/inference/go";
pub const OPEN_CODE_GO_OPENAI_ENDPOINT: &str = "https://opencode.ai/inference/go/openai/v1";
pub const OPEN_CODE_GO_ANTHROPIC_ENDPOINT: &str = "https://opencode.ai/inference/go/anthropic/v1";
pub const OPEN_CODE_GO_GOOGLE_ENDPOINT: &str = "https://opencode.ai/inference/go/google/v1beta";
pub const OPEN_CODE_ZEN_ENDPOINT: &str = "https://opencode.ai/zen/v1";

/// An OpenCode service with a fixed base endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenCodeService {
    Go,
    Zen,
}

impl OpenCodeService {
    pub const fn endpoint(self) -> &'static str {
        match self {
            Self::Go => OPEN_CODE_GO_ENDPOINT,
            Self::Zen => OPEN_CODE_ZEN_ENDPOINT,
        }
    }
}

/// Reviewed metadata for one OpenCode model and its official endpoint protocol.
#[derive(Clone, Debug)]
pub struct OpenCodeModel {
    service: OpenCodeService,
    id: String,
    name: String,
    protocol: Protocol,
    vision: bool,
    pdf: bool,
    context_window_tokens: Option<u64>,
    max_output_tokens: Option<u64>,
    reasoning_efforts: Vec<String>,
    default_reasoning_effort: Option<String>,
    selected_reasoning_effort: Option<String>,
}

impl OpenCodeModel {
    pub const fn service(&self) -> OpenCodeService {
        self.service
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn protocol(&self) -> Protocol {
        self.protocol
    }

    pub const fn supports_vision(&self) -> bool {
        self.vision
    }

    pub const fn supports_pdf(&self) -> bool {
        self.pdf
    }

    pub fn supports_reasoning(&self) -> bool {
        !self.reasoning_efforts.is_empty()
    }

    pub fn reasoning_efforts(&self) -> &[String] {
        &self.reasoning_efforts
    }

    pub fn default_reasoning_effort(&self) -> Option<&str> {
        self.default_reasoning_effort.as_deref()
    }

    pub fn selected_reasoning_effort(&self) -> Option<&str> {
        self.selected_reasoning_effort.as_deref()
    }

    pub const fn context_window_tokens(&self) -> Option<u64> {
        self.context_window_tokens
    }

    pub const fn max_output_tokens(&self) -> Option<u64> {
        self.max_output_tokens
    }

    pub fn context_budget(&self) -> Option<ContextBudget> {
        self.context_window_tokens
            .map(|context_window_tokens| ContextBudget {
                max_input_tokens: context_window_tokens
                    .saturating_sub(self.max_output_tokens.unwrap_or_default()),
            })
    }

    /// Return a per-request model value with a validated reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: &str) -> Result<Self, ProviderConfigError> {
        if !self
            .reasoning_efforts
            .iter()
            .any(|supported| supported == effort)
        {
            return Err(ProviderConfigError::new(format!(
                "OpenCode model {} does not support reasoning effort {effort}",
                self.id
            )));
        }
        self.selected_reasoning_effort = Some(effort.to_owned());
        Ok(self)
    }

    pub fn request_options(&self) -> OpenAiRequestOptions {
        OpenAiRequestOptions {
            reasoning_effort: self.selected_reasoning_effort.clone(),
            service_tier: None,
        }
    }
}

impl OpenAiModel for OpenCodeModel {
    fn id(&self) -> &str {
        self.id()
    }

    fn supports_vision(&self) -> bool {
        self.supports_vision()
    }

    fn request_options(&self) -> OpenAiRequestOptions {
        OpenCodeModel::request_options(self)
    }

    fn context_budget(&self) -> Option<ContextBudget> {
        OpenCodeModel::context_budget(self)
    }
}

impl AnthropicModel for OpenCodeModel {
    fn id(&self) -> &str {
        self.id()
    }

    fn supports_vision(&self) -> bool {
        self.supports_vision()
    }
}

impl GoogleModel for OpenCodeModel {
    fn id(&self) -> &str {
        self.id()
    }

    fn supports_vision(&self) -> bool {
        self.supports_vision()
    }

    fn supports_document(&self, mime: &str) -> bool {
        self.supports_pdf() && self.supports_vision() && mime == "application/pdf"
    }

    fn context_budget(&self) -> Option<ContextBudget> {
        self.context_budget()
    }
}

/// The reviewed OpenCode catalog bundled with this crate.
///
/// PDF capability is a frozen 2026-09-08 snapshot of
/// <https://models.dev/api.json>; it is independent of vision capability.
/// DeepSeek V4.1 Flash metadata matches the upstream catalog at
/// <https://api-docs.deepseek.com/api/list-models/>.
pub fn open_code_catalog() -> &'static [OpenCodeModel] {
    #[derive(Deserialize)]
    struct CatalogEntry {
        service: OpenCodeService,
        id: String,
        name: String,
        protocol: Protocol,
        vision: bool,
        pdf: bool,
        #[serde(default)]
        context_window_tokens: Option<u64>,
        #[serde(default)]
        max_output_tokens: Option<u64>,
        #[serde(default)]
        reasoning_efforts: Vec<String>,
        #[serde(default)]
        default_reasoning_effort: Option<String>,
    }

    static MODELS: OnceLock<Vec<OpenCodeModel>> = OnceLock::new();
    MODELS.get_or_init(|| {
        serde_json::from_str::<Vec<CatalogEntry>>(include_str!("open-code-models.json"))
            .expect("bundled OpenCode model catalog must be valid")
            .into_iter()
            .map(|entry| OpenCodeModel {
                service: entry.service,
                id: entry.id,
                name: entry.name,
                protocol: entry.protocol,
                vision: entry.vision,
                pdf: entry.pdf,
                context_window_tokens: entry.context_window_tokens,
                max_output_tokens: entry.max_output_tokens,
                reasoning_efforts: entry.reasoning_efforts,
                selected_reasoning_effort: entry.default_reasoning_effort.clone(),
                default_reasoning_effort: entry.default_reasoning_effort,
            })
            .collect()
    })
}

pub fn open_code_model(service: OpenCodeService, id: &str) -> Option<&'static OpenCodeModel> {
    open_code_catalog()
        .iter()
        .find(|model| model.service == service && model.id == id)
}

#[derive(Clone)]
enum OpenCodeAuth {
    ApiKey {
        bearer: RequestCredentials,
        messages: RequestCredentials,
    },
    Credentials(Arc<dyn CredentialSource>),
}

/// OpenCode provider that fixes the service endpoint and routes catalog models.
#[derive(Clone)]
pub struct OpenCodeProvider {
    service: OpenCodeService,
    auth: OpenCodeAuth,
    session: HeaderValue,
    user_agent: HeaderValue,
    client: reqwest::Client,
    endpoint: String,
    request_observer: Option<Arc<dyn RequestObserver>>,
}

impl OpenCodeProvider {
    pub fn with_api_key(
        service: OpenCodeService,
        api_key: impl Into<String>,
        session: impl fmt::Display,
        user_agent: impl AsRef<str>,
    ) -> Result<Self, ProviderConfigError> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(ProviderConfigError::new("OpenCode API key is empty"));
        }
        let mut message_headers = HeaderMap::new();
        message_headers.insert(
            HeaderName::from_static("x-api-key"),
            header_value("OpenCode API key", &api_key)?,
        );
        let bearer = RequestCredentials::bearer(&api_key)
            .map_err(|error| ProviderConfigError::new(error.to_string()))?;
        Self::new(
            service,
            OpenCodeAuth::ApiKey {
                bearer,
                messages: RequestCredentials::from_headers(message_headers),
            },
            session,
            user_agent,
        )
    }

    pub fn with_credentials(
        service: OpenCodeService,
        credentials: Arc<dyn CredentialSource>,
        session: impl fmt::Display,
        user_agent: impl AsRef<str>,
    ) -> Result<Self, ProviderConfigError> {
        Self::new(
            service,
            OpenCodeAuth::Credentials(credentials),
            session,
            user_agent,
        )
    }

    fn new(
        service: OpenCodeService,
        auth: OpenCodeAuth,
        session: impl fmt::Display,
        user_agent: impl AsRef<str>,
    ) -> Result<Self, ProviderConfigError> {
        let session = session.to_string();
        if session.trim().is_empty() {
            return Err(ProviderConfigError::new("OpenCode session is empty"));
        }
        if user_agent.as_ref().trim().is_empty() {
            return Err(ProviderConfigError::new("user agent is empty"));
        }
        Ok(Self {
            service,
            auth,
            session: header_value("OpenCode session", &session)?,
            user_agent: header_value("user agent", user_agent.as_ref())?,
            client: fixed_endpoint_client()?,
            endpoint: service.endpoint().to_owned(),
            request_observer: None,
        })
    }

    /// Use a caller-configured HTTP client without changing the fixed endpoint.
    ///
    /// The caller remains responsible for client-wide proxy, redirect, timeout,
    /// and default-header policy. Provider-owned headers are applied per request.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Observe content-free request diagnostics from every routed provider.
    pub fn with_request_observer(mut self, observer: Arc<dyn RequestObserver>) -> Self {
        self.request_observer = Some(observer);
        self
    }

    pub const fn service(&self) -> OpenCodeService {
        self.service
    }

    pub const fn endpoint(&self) -> &'static str {
        self.service.endpoint()
    }

    fn credentials(&self, protocol: Protocol) -> Arc<dyn CredentialSource> {
        let auth: Arc<dyn CredentialSource> = match &self.auth {
            OpenCodeAuth::ApiKey { messages, .. } if protocol == Protocol::Messages => {
                Arc::new(StaticCredentials::new(messages.clone()))
            }
            OpenCodeAuth::ApiKey { bearer, .. } => Arc::new(StaticCredentials::new(bearer.clone())),
            OpenCodeAuth::Credentials(credentials) => credentials.clone(),
        };
        Arc::new(OpenCodeHttpCredentials {
            auth,
            session: self.session.clone(),
            user_agent: self.user_agent.clone(),
        })
    }

    fn validate_model(&self, model: &OpenCodeModel) -> Result<(), ProviderConfigError> {
        if model.service != self.service {
            return Err(ProviderConfigError::new(format!(
                "model {} belongs to OpenCode {:?}, not {:?}",
                model.id, model.service, self.service
            )));
        }
        Ok(())
    }

    fn endpoint_for_protocol(&self, protocol: Protocol) -> String {
        if self.service != OpenCodeService::Go {
            return self.endpoint.clone();
        }
        let endpoint = match protocol {
            Protocol::Responses | Protocol::ChatCompletions => "openai/v1",
            Protocol::Messages => "anthropic/v1",
            Protocol::GenerateContent => "google/v1beta",
        };
        format!("{}/{endpoint}", self.endpoint.trim_end_matches('/'))
    }

    #[cfg(test)]
    pub(crate) fn with_test_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = endpoint;
        self
    }
}

fn ensure_reasoning_content(model: &OpenCodeModel, message: AssistantMessage) -> AssistantMessage {
    if model.supports_reasoning() && message.reasoning_content().is_none() {
        message.with_reasoning_content("")
    } else {
        message
    }
}

fn prepare_reasoning_replay(
    model: &OpenCodeModel,
    conversation: &Conversation,
    tools: &[ToolSpec],
) -> Conversation {
    let mut request = conversation.clone();
    if model.supports_reasoning() && !tools.is_empty() {
        request.replace_messages(
            conversation
                .messages()
                .iter()
                .map(|message| match message {
                    Message::Assistant(assistant) if assistant.reasoning_content().is_none() => {
                        Message::Assistant(assistant.clone().with_reasoning_content(""))
                    }
                    _ => message.clone(),
                })
                .collect(),
        );
    }
    request
}

struct OpenCodeHttpCredentials {
    auth: Arc<dyn CredentialSource>,
    session: HeaderValue,
    user_agent: HeaderValue,
}

impl CredentialSource for OpenCodeHttpCredentials {
    fn credentials(&self) -> kodkod_http::CredentialFuture<'_> {
        Box::pin(async {
            let credentials = self.auth.credentials().await?;
            let mut headers = credentials.headers().clone();
            headers.insert(
                HeaderName::from_static("x-opencode-session"),
                self.session.clone(),
            );
            headers.insert(USER_AGENT, self.user_agent.clone());
            Ok(RequestCredentials::from_headers(headers))
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OpenCodeContinuation {
    Responses(ResponsesContinuation),
    ChatCompletions,
    Messages(AnthropicContinuation),
    GenerateContent(GoogleContinuation),
}

#[derive(Debug)]
pub enum ProviderError {
    Configuration(ProviderConfigError),
    OpenAi(OpenAiError),
    Anthropic(AnthropicError),
    Google(GoogleError),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(error) => error.fmt(f),
            Self::OpenAi(error) => error.fmt(f),
            Self::Anthropic(error) => error.fmt(f),
            Self::Google(error) => error.fmt(f),
        }
    }
}

impl Error for ProviderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            Self::OpenAi(error) => Some(error),
            Self::Anthropic(error) => Some(error),
            Self::Google(error) => Some(error),
        }
    }
}

impl Retryable for ProviderError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Configuration(_) => false,
            Self::OpenAi(error) => error.is_retryable(),
            Self::Anthropic(error) => error.is_retryable(),
            Self::Google(error) => error.is_retryable(),
        }
    }
}

impl From<ProviderConfigError> for ProviderError {
    fn from(error: ProviderConfigError) -> Self {
        Self::Configuration(error)
    }
}

impl From<OpenAiError> for ProviderError {
    fn from(error: OpenAiError) -> Self {
        Self::OpenAi(error)
    }
}

impl From<AnthropicError> for ProviderError {
    fn from(error: AnthropicError) -> Self {
        Self::Anthropic(error)
    }
}

impl From<GoogleError> for ProviderError {
    fn from(error: GoogleError) -> Self {
        Self::Google(error)
    }
}

impl Provider for OpenCodeProvider {
    type Model = OpenCodeModel;
    type Error = ProviderError;
    type Continuation = OpenCodeContinuation;

    fn with_request_observer(mut self, observer: Arc<dyn RequestObserver>) -> Self {
        self.request_observer = Some(observer);
        self
    }

    fn supports_vision(&self, model: &OpenCodeModel) -> bool {
        model.service == self.service && model.supports_vision()
    }

    fn supports_document(&self, model: &OpenCodeModel, mime: &str) -> bool {
        model.service == self.service
            && model.supports_pdf()
            && model.supports_vision()
            && mime == "application/pdf"
    }

    fn context_budget(&self, model: &OpenCodeModel) -> Option<ContextBudget> {
        (model.service == self.service)
            .then(|| model.context_budget())
            .flatten()
    }

    fn is_context_overflow(&self, error: &Self::Error) -> bool {
        match error {
            ProviderError::OpenAi(OpenAiError::Api {
                status: 400,
                code: Some(code),
                ..
            })
            | ProviderError::OpenAi(OpenAiError::ResponseFailed {
                code: Some(code), ..
            }) => code == "context_length_exceeded",
            ProviderError::Google(error) => error.is_context_overflow(),
            _ => false,
        }
    }

    fn create_continuation(&self, model: &OpenCodeModel) -> Self::Continuation {
        match model.protocol {
            Protocol::Responses => OpenCodeContinuation::Responses(Default::default()),
            Protocol::ChatCompletions => OpenCodeContinuation::ChatCompletions,
            Protocol::Messages => OpenCodeContinuation::Messages(Default::default()),
            Protocol::GenerateContent => OpenCodeContinuation::GenerateContent(Default::default()),
        }
    }

    fn estimate_continuation_tokens(continuation: &Self::Continuation) -> u64 {
        match continuation {
            OpenCodeContinuation::Responses(continuation) => {
                <OpenAiResponsesProvider<OpenCodeModel> as Provider>::estimate_continuation_tokens(
                    continuation,
                )
            }
            OpenCodeContinuation::ChatCompletions => 0,
            OpenCodeContinuation::Messages(continuation) => {
                <AnthropicMessagesProvider<OpenCodeModel> as Provider>::estimate_continuation_tokens(
                    continuation,
                )
            }
            OpenCodeContinuation::GenerateContent(continuation) => <GoogleGenerateContentProvider<
                OpenCodeModel,
            > as Provider>::estimate_continuation_tokens(
                continuation
            ),
        }
    }

    fn validate_context(
        &self,
        model: &OpenCodeModel,
        continuation: &Self::Continuation,
    ) -> Result<(), String> {
        self.validate_model(model)
            .map_err(|error| error.to_string())?;
        match (model.protocol, continuation) {
            (Protocol::Responses, OpenCodeContinuation::Responses(continuation)) => {
                OpenAiResponsesProvider::new(self.endpoint_for_protocol(Protocol::Responses))
                    .validate_context(model, continuation)
            }
            (Protocol::ChatCompletions, OpenCodeContinuation::ChatCompletions) => Ok(()),
            (Protocol::Messages, OpenCodeContinuation::Messages(continuation)) => {
                AnthropicMessagesProvider::new(self.endpoint_for_protocol(Protocol::Messages))
                    .validate_context(model, continuation)
            }
            (Protocol::GenerateContent, OpenCodeContinuation::GenerateContent(continuation)) => {
                GoogleGenerateContentProvider::<OpenCodeModel>::new(
                    self.endpoint_for_protocol(Protocol::GenerateContent),
                )
                .validate_context(model, continuation)
            }
            _ => Err("OpenCode continuation protocol does not match the selected model".into()),
        }
    }

    async fn compact_context(
        &self,
        continuation: &Self::Continuation,
        model: &Self::Model,
        conversation: &Conversation,
        options: CompactOptions,
    ) -> Result<ContextCompaction<Self::Continuation>, CompactError<Self::Error>> {
        self.validate_model(model)
            .map_err(|error| CompactError::Context(error.to_string()))?;
        match (model.protocol, continuation) {
            (Protocol::Responses, OpenCodeContinuation::Responses(continuation)) => {
                let provider =
                    OpenAiResponsesProvider::new(self.endpoint_for_protocol(Protocol::Responses))
                        .with_credentials(self.credentials(Protocol::Responses))
                        .with_client(self.client.clone());
                let provider = match &self.request_observer {
                    Some(observer) => provider.with_request_observer(observer.clone()),
                    None => provider,
                };
                let compacted = provider
                    .compact_context(continuation, model, conversation, options)
                    .await
                    .map_err(map_compact_error)?;
                Ok(ContextCompaction {
                    conversation: compacted.conversation,
                    continuation: OpenCodeContinuation::Responses(compacted.continuation),
                    usage: compacted.usage,
                })
            }
            (Protocol::Messages, OpenCodeContinuation::Messages(continuation)) => {
                let provider =
                    AnthropicMessagesProvider::new(self.endpoint_for_protocol(Protocol::Messages))
                        .with_credentials(self.credentials(Protocol::Messages))
                        .with_client(self.client.clone());
                let provider = match &self.request_observer {
                    Some(observer) => provider.with_request_observer(observer.clone()),
                    None => provider,
                };
                let compacted = provider
                    .compact_context(continuation, model, conversation, options)
                    .await
                    .map_err(map_compact_error)?;
                Ok(ContextCompaction {
                    conversation: compacted.conversation,
                    continuation: OpenCodeContinuation::Messages(compacted.continuation),
                    usage: compacted.usage,
                })
            }
            (Protocol::GenerateContent, OpenCodeContinuation::GenerateContent(continuation)) => {
                let provider = GoogleGenerateContentProvider::new(
                    self.endpoint_for_protocol(Protocol::GenerateContent),
                )
                .with_credentials(self.credentials(Protocol::GenerateContent))
                .with_client(self.client.clone());
                let provider = match &self.request_observer {
                    Some(observer) => provider.with_request_observer(observer.clone()),
                    None => provider,
                };
                let compacted = provider
                    .compact_context(continuation, model, conversation, options)
                    .await
                    .map_err(map_compact_error)?;
                Ok(ContextCompaction {
                    conversation: compacted.conversation,
                    continuation: OpenCodeContinuation::GenerateContent(compacted.continuation),
                    usage: compacted.usage,
                })
            }
            (Protocol::ChatCompletions, OpenCodeContinuation::ChatCompletions) => {
                if self.has_opaque_context(continuation) {
                    return Err(CompactError::Context(
                        "Provider must implement contextual compaction for opaque history".into(),
                    ));
                }
                let compacted = self
                    .compact_with_usage(model, conversation, options)
                    .await?;
                Ok(ContextCompaction {
                    conversation: compacted.conversation,
                    continuation: OpenCodeContinuation::ChatCompletions,
                    usage: compacted.usage,
                })
            }
            _ => Err(CompactError::Context(
                "OpenCode continuation protocol does not match the selected model".into(),
            )),
        }
    }

    fn has_opaque_context(&self, continuation: &Self::Continuation) -> bool {
        match continuation {
            OpenCodeContinuation::Responses(continuation) => {
                OpenAiResponsesProvider::<OpenCodeModel>::new(
                    self.endpoint_for_protocol(Protocol::Responses),
                )
                .has_opaque_context(continuation)
            }
            OpenCodeContinuation::ChatCompletions
            | OpenCodeContinuation::Messages(_)
            | OpenCodeContinuation::GenerateContent(_) => false,
        }
    }

    async fn complete(
        &self,
        continuation: &Self::Continuation,
        model: &OpenCodeModel,
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
        model: &OpenCodeModel,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<ProviderCompletion<Self::Continuation>, Self::Error> {
        self.validate_model(model)?;
        self.validate_documents(model, conversation)?;
        match (model.protocol, continuation) {
            (Protocol::Responses, OpenCodeContinuation::Responses(continuation)) => {
                let endpoint = self.endpoint_for_protocol(Protocol::Responses);
                let provider = OpenAiResponsesProvider::new(endpoint)
                    .with_credentials(self.credentials(Protocol::Responses))
                    .with_client(self.client.clone());
                let provider = match &self.request_observer {
                    Some(observer) => provider.with_request_observer(observer.clone()),
                    None => provider,
                };
                let completion = provider
                    .complete_with_usage(continuation, model, conversation, tools)
                    .await?;
                Ok(ProviderCompletion::new(
                    completion.message,
                    OpenCodeContinuation::Responses(completion.continuation),
                    completion.usage,
                ))
            }
            (Protocol::ChatCompletions, OpenCodeContinuation::ChatCompletions) => {
                let endpoint = self.endpoint_for_protocol(Protocol::ChatCompletions);
                let request_conversation = prepare_reasoning_replay(model, conversation, tools);
                let provider = OpenAiCompatibleProvider::new(endpoint)
                    .with_pdf_inputs(model.supports_pdf())
                    .with_credentials(self.credentials(Protocol::ChatCompletions))
                    .with_client(self.client.clone());
                let completion = provider
                    .complete_with_usage(&(), model, &request_conversation, tools)
                    .await?;
                Ok(ProviderCompletion::new(
                    ensure_reasoning_content(model, completion.message),
                    OpenCodeContinuation::ChatCompletions,
                    completion.usage,
                ))
            }
            (Protocol::Messages, OpenCodeContinuation::Messages(continuation)) => {
                let endpoint = self.endpoint_for_protocol(Protocol::Messages);
                let provider = AnthropicMessagesProvider::new(endpoint)
                    .with_credentials(self.credentials(Protocol::Messages))
                    .with_client(self.client.clone());
                let completion = provider
                    .complete_with_usage(continuation, model, conversation, tools)
                    .await?;
                Ok(ProviderCompletion::new(
                    completion.message,
                    OpenCodeContinuation::Messages(completion.continuation),
                    completion.usage,
                ))
            }
            (Protocol::GenerateContent, OpenCodeContinuation::GenerateContent(continuation)) => {
                let endpoint = self.endpoint_for_protocol(Protocol::GenerateContent);
                let provider = GoogleGenerateContentProvider::new(endpoint)
                    .with_credentials(self.credentials(Protocol::GenerateContent))
                    .with_client(self.client.clone());
                let provider = match &self.request_observer {
                    Some(observer) => provider.with_request_observer(observer.clone()),
                    None => provider,
                };
                let completion = provider
                    .complete_with_usage(continuation, model, conversation, tools)
                    .await?;
                Ok(ProviderCompletion::new(
                    completion.message,
                    OpenCodeContinuation::GenerateContent(completion.continuation),
                    completion.usage,
                ))
            }
            _ => Err(ProviderConfigError::new(
                "OpenCode continuation protocol does not match the selected model",
            )
            .into()),
        }
    }

    fn complete_stream<'a>(
        &'a self,
        continuation: &'a Self::Continuation,
        model: &'a OpenCodeModel,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, Self::Continuation, Self::Error> {
        Box::pin(async_stream::try_stream! {
            self.validate_model(model)?;
            self.validate_documents(model, conversation)?;
            match (model.protocol, continuation) {
                (Protocol::Responses, OpenCodeContinuation::Responses(continuation)) => {
                    let endpoint = self.endpoint_for_protocol(Protocol::Responses);
                    let provider = OpenAiResponsesProvider::new(endpoint)
                        .with_credentials(self.credentials(Protocol::Responses))
                        .with_client(self.client.clone());
                    let provider = match &self.request_observer {
                        Some(observer) => provider.with_request_observer(observer.clone()),
                        None => provider,
                    };
                    let mut stream = provider.complete_stream(continuation, model, conversation, tools);
                    while let Some(event) = stream.next().await {
                        match event? {
                            ProviderEvent::TextDelta(delta) => yield ProviderEvent::TextDelta(delta),
                            ProviderEvent::Completed(completion) => {
                                yield ProviderEvent::Completed(ProviderCompletion::new(
                                    completion.message,
                                    OpenCodeContinuation::Responses(completion.continuation),
                                    completion.usage,
                                ));
                                return;
                            }
                        }
                    }
                }
                (Protocol::ChatCompletions, OpenCodeContinuation::ChatCompletions) => {
                    let endpoint = self.endpoint_for_protocol(Protocol::ChatCompletions);
                    let request_conversation = prepare_reasoning_replay(model, conversation, tools);
                    let provider = OpenAiCompatibleProvider::new(endpoint)
                        .with_pdf_inputs(model.supports_pdf())
                        .with_credentials(self.credentials(Protocol::ChatCompletions))
                        .with_client(self.client.clone());
                    let mut stream = provider.complete_stream(
                        &(), model, &request_conversation, tools,
                    );
                    while let Some(event) = stream.next().await {
                        match event? {
                            ProviderEvent::TextDelta(delta) => yield ProviderEvent::TextDelta(delta),
                            ProviderEvent::Completed(completion) => {
                                yield ProviderEvent::Completed(ProviderCompletion::new(
                                    ensure_reasoning_content(model, completion.message),
                                    OpenCodeContinuation::ChatCompletions,
                                    completion.usage,
                                ));
                                return;
                            }
                        }
                    }
                }
                (Protocol::Messages, OpenCodeContinuation::Messages(continuation)) => {
                    let endpoint = self.endpoint_for_protocol(Protocol::Messages);
                    let provider = AnthropicMessagesProvider::new(endpoint)
                        .with_credentials(self.credentials(Protocol::Messages))
                        .with_client(self.client.clone());
                    let mut stream = provider.complete_stream(continuation, model, conversation, tools);
                    while let Some(event) = stream.next().await {
                        match event? {
                            ProviderEvent::TextDelta(delta) => yield ProviderEvent::TextDelta(delta),
                            ProviderEvent::Completed(completion) => {
                                yield ProviderEvent::Completed(ProviderCompletion::new(
                                    completion.message,
                                    OpenCodeContinuation::Messages(completion.continuation),
                                    completion.usage,
                                ));
                                return;
                            }
                        }
                    }
                }
                (Protocol::GenerateContent, OpenCodeContinuation::GenerateContent(continuation)) => {
                    let endpoint = self.endpoint_for_protocol(Protocol::GenerateContent);
                    let provider = GoogleGenerateContentProvider::new(endpoint)
                        .with_credentials(self.credentials(Protocol::GenerateContent))
                        .with_client(self.client.clone());
                    let provider = match &self.request_observer {
                        Some(observer) => provider.with_request_observer(observer.clone()),
                        None => provider,
                    };
                    let mut stream = provider.complete_stream(continuation, model, conversation, tools);
                    while let Some(event) = stream.next().await {
                        match event? {
                            ProviderEvent::TextDelta(delta) => yield ProviderEvent::TextDelta(delta),
                            ProviderEvent::Completed(completion) => {
                                yield ProviderEvent::Completed(ProviderCompletion::new(
                                    completion.message,
                                    OpenCodeContinuation::GenerateContent(completion.continuation),
                                    completion.usage,
                                ));
                                return;
                            }
                        }
                    }
                }
                _ => Err(ProviderConfigError::new(
                    "OpenCode continuation protocol does not match the selected model",
                ))?,
            }
        })
    }
}

fn map_compact_error<E: Into<ProviderError>>(
    error: CompactError<E>,
) -> CompactError<ProviderError> {
    match error {
        CompactError::NothingToCompact => CompactError::NothingToCompact,
        CompactError::Unsupported(message) => CompactError::Unsupported(message),
        CompactError::Context(message) => CompactError::Context(message),
        CompactError::EmptySummary => CompactError::EmptySummary,
        CompactError::MissingSummary => CompactError::MissingSummary,
        CompactError::Provider(error) => CompactError::Provider(error.into()),
    }
}

impl OpenCodeProvider {
    fn validate_documents(
        &self,
        model: &OpenCodeModel,
        conversation: &Conversation,
    ) -> Result<(), ProviderConfigError> {
        for message in conversation.messages() {
            if let Message::User(user) = message {
                for document in user.documents() {
                    if !self.supports_document(model, document.mime()) {
                        return Err(ProviderConfigError::new(format!(
                            "OpenCode does not support inline documents with MIME type {}",
                            document.mime()
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}
