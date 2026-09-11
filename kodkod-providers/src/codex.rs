use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, Conversation, Message, Provider, ProviderCompletion, ProviderEvent,
    ProviderStream, ToolSpec,
};
use kodkod_http::{CredentialError, CredentialSource, RequestCredentials};
use kodkod_openai::{OpenAiError, OpenAiModel, OpenAiResponsesProvider, ResponsesContinuation};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};

use crate::{Protocol, ProviderConfigError, fixed_endpoint_client, header_value};
pub const CODEX_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex";

/// A currently valid Codex subscription access token and its ChatGPT account.
pub struct CodexAccess {
    pub access_token: String,
    pub account_id: String,
}

impl fmt::Debug for CodexAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodexAccess").finish_non_exhaustive()
    }
}

pub type CodexAccessFuture<'a> =
    Pin<Box<dyn Future<Output = Result<CodexAccess, CredentialError>> + Send + 'a>>;

/// Application-owned source of fresh Codex subscription access data.
pub trait CodexCredentialSource: Send + Sync {
    fn access(&self) -> CodexAccessFuture<'_>;
}

/// ChatGPT-backed Codex provider with a fixed service endpoint.
#[derive(Clone)]
pub struct CodexProvider<M = String> {
    source: Arc<dyn CodexCredentialSource>,
    originator: HeaderValue,
    client: reqwest::Client,
    endpoint: String,
    _model: std::marker::PhantomData<M>,
}

impl<M> CodexProvider<M> {
    pub fn new(source: Arc<dyn CodexCredentialSource>) -> Result<Self, ProviderConfigError> {
        Ok(Self {
            source,
            originator: HeaderValue::from_static("kodkod"),
            client: fixed_endpoint_client()?,
            endpoint: CODEX_ENDPOINT.to_owned(),
            _model: std::marker::PhantomData,
        })
    }

    /// Identify the calling application in the Codex `originator` header.
    pub fn with_originator(
        mut self,
        originator: impl AsRef<str>,
    ) -> Result<Self, ProviderConfigError> {
        self.originator = header_value("originator", originator.as_ref())?;
        Ok(self)
    }

    /// Use a caller-configured HTTP client without changing the fixed endpoint.
    ///
    /// The caller remains responsible for client-wide proxy, redirect, timeout,
    /// and default-header policy. Provider-owned headers are applied per request.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub const fn protocol(&self) -> Protocol {
        Protocol::Responses
    }

    pub const fn endpoint(&self) -> &'static str {
        CODEX_ENDPOINT
    }

    fn inner(&self) -> OpenAiResponsesProvider<M> {
        OpenAiResponsesProvider::new(&self.endpoint)
            .with_credentials(Arc::new(CodexHttpCredentials {
                source: self.source.clone(),
                originator: self.originator.clone(),
            }))
            .with_client(self.client.clone())
    }

    #[cfg(test)]
    pub(crate) fn with_test_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = endpoint;
        self
    }
}

impl<M> Provider for CodexProvider<M>
where
    M: OpenAiModel + Send,
{
    type Model = M;
    type Error = OpenAiError;
    type Continuation = ResponsesContinuation;

    fn supports_vision(&self, model: &M) -> bool {
        model.supports_vision()
    }

    fn supports_document(&self, model: &M, mime: &str) -> bool {
        model.supports_vision() && mime == "application/pdf"
    }

    fn create_continuation(&self, _model: &M) -> Self::Continuation {
        ResponsesContinuation::default()
    }

    fn estimate_continuation_tokens(continuation: &Self::Continuation) -> u64 {
        <OpenAiResponsesProvider<M> as Provider>::estimate_continuation_tokens(continuation)
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
        validate_documents(model, conversation)?;
        self.inner()
            .complete_with_usage(continuation, model, conversation, tools)
            .await
    }

    fn complete_stream<'a>(
        &'a self,
        continuation: &'a Self::Continuation,
        model: &'a M,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, Self::Continuation, Self::Error> {
        Box::pin(async_stream::try_stream! {
            validate_documents(model, conversation)?;
            let inner = self.inner();
            let mut stream = inner.complete_stream(continuation, model, conversation, tools);
            while let Some(event) = stream.next().await {
                match event? {
                    ProviderEvent::TextDelta(delta) => yield ProviderEvent::TextDelta(delta),
                    ProviderEvent::Completed(completion) => {
                        yield ProviderEvent::Completed(completion);
                        return;
                    }
                }
            }
        })
    }
}

fn validate_documents<M: OpenAiModel>(
    model: &M,
    conversation: &Conversation,
) -> Result<(), OpenAiError> {
    for message in conversation.messages() {
        if let Message::User(user) = message {
            for document in user.documents() {
                if !(model.supports_vision() && document.mime() == "application/pdf") {
                    return Err(OpenAiError::UnsupportedDocument {
                        provider: "Codex subscription Responses",
                        mime: document.mime().to_owned(),
                    });
                }
            }
        }
    }
    Ok(())
}

struct CodexHttpCredentials {
    source: Arc<dyn CodexCredentialSource>,
    originator: HeaderValue,
}

impl CredentialSource for CodexHttpCredentials {
    fn credentials(&self) -> kodkod_http::CredentialFuture<'_> {
        Box::pin(async {
            let access = self.source.access().await?;
            if access.access_token.trim().is_empty() {
                return Err(CredentialError::new("Codex access token is empty"));
            }
            if access.account_id.trim().is_empty() {
                return Err(CredentialError::new("ChatGPT account ID is empty"));
            }
            let mut authorization = header_value(
                "Codex access token",
                &format!("Bearer {}", access.access_token),
            )
            .map_err(|error| CredentialError::new(error.to_string()))?;
            authorization.set_sensitive(true);
            let mut account = header_value("ChatGPT account ID", &access.account_id)
                .map_err(|error| CredentialError::new(error.to_string()))?;
            account.set_sensitive(true);
            let mut headers = HeaderMap::new();
            headers.insert(AUTHORIZATION, authorization);
            headers.insert(HeaderName::from_static("chatgpt-account-id"), account);
            headers.insert(
                HeaderName::from_static("originator"),
                self.originator.clone(),
            );
            Ok(RequestCredentials::from_headers(headers))
        })
    }
}
