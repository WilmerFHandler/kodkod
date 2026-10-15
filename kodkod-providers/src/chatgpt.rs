use std::sync::Arc;

use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, CompactError, CompactOptions, ContextBudget, ContextCompaction, Conversation,
    Provider, ProviderCompletion, ProviderEvent, ProviderStream, ToolSpec,
};
use kodkod_http::CredentialSource;
use kodkod_openai::{
    OpenAiError, OpenAiModel, OpenAiResponsesProvider, ResponsesContinuation,
    ResponsesRequestProfile,
};

use crate::{Protocol, ProviderConfigError, fixed_endpoint_client};

/// Public OpenAI Responses API base URL for ChatGPT-plan sharing.
pub const CHATGPT_ENDPOINT: &str = "https://api.openai.com/v1";

/// Uses a caller-owned credential source with the public Responses endpoint
/// and the stateless request profile required for ChatGPT-plan sharing.
#[derive(Clone)]
pub struct ChatGptProvider<M = String> {
    source: Arc<dyn CredentialSource>,
    client: reqwest::Client,
    endpoint: String,
    _model: std::marker::PhantomData<M>,
}

impl<M> ChatGptProvider<M> {
    pub fn new(source: Arc<dyn CredentialSource>) -> Result<Self, ProviderConfigError> {
        Ok(Self {
            source,
            client: fixed_endpoint_client()?,
            endpoint: CHATGPT_ENDPOINT.to_owned(),
            _model: std::marker::PhantomData,
        })
    }

    pub const fn protocol(&self) -> Protocol {
        Protocol::Responses
    }

    pub const fn endpoint(&self) -> &'static str {
        CHATGPT_ENDPOINT
    }

    fn inner(&self) -> OpenAiResponsesProvider<M> {
        OpenAiResponsesProvider::new(&self.endpoint)
            .with_credentials(self.source.clone())
            .with_client(self.client.clone())
            .with_profile(ResponsesRequestProfile::ChatGptPlan)
            .with_native_compaction(true)
    }

    #[cfg(test)]
    pub(crate) fn with_test_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = endpoint;
        self
    }
}

impl<M> Provider for ChatGptProvider<M>
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
        self.inner().supports_document(model, mime)
    }

    fn context_budget(&self, model: &M) -> Option<ContextBudget> {
        self.inner().context_budget(model)
    }

    fn has_opaque_context(&self, continuation: &Self::Continuation) -> bool {
        self.inner().has_opaque_context(continuation)
    }

    fn validate_context(&self, model: &M, continuation: &Self::Continuation) -> Result<(), String> {
        self.inner().validate_context(model, continuation)
    }

    fn is_context_overflow(&self, error: &Self::Error) -> bool {
        self.inner().is_context_overflow(error)
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
        self.inner()
            .complete_with_usage(continuation, model, conversation, tools)
            .await
    }

    async fn compact_context(
        &self,
        continuation: &Self::Continuation,
        model: &M,
        conversation: &Conversation,
        options: CompactOptions,
    ) -> Result<ContextCompaction<Self::Continuation>, CompactError<Self::Error>> {
        self.inner()
            .compact_context(continuation, model, conversation, options)
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
