pub mod error;
pub mod event;

use std::borrow::Cow;
use std::sync::Arc;

use async_stream::try_stream;
use futures::stream::BoxStream;

pub use error::AgentError;
pub use event::AgentEvent;

use crate::{Conversation, Model, Provider, Tool, ToolExecutor};

pub struct Agent<P> {
    provider: P,
    tools: ToolExecutor,
    max_tool_rounds: Option<usize>,
}

impl<P> Agent<P>
where
    P: Provider + Sync,
{
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            tools: ToolExecutor::new(),
            max_tool_rounds: None,
        }
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub fn tools(&self) -> &ToolExecutor {
        &self.tools
    }

    pub fn max_tool_rounds(&self) -> Option<usize> {
        self.max_tool_rounds
    }

    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.register_tool(tool);
        self
    }

    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.max_tool_rounds = Some(max_tool_rounds);
        self
    }

    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.register(tool);
    }

    /// Run the agent loop on the current conversation, streaming progress events.
    ///
    /// The conversation must already include the user message (and any images) for
    /// this turn. The provider uses `model` for the request target and vision handling.
    pub fn run<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a Model,
    ) -> BoxStream<'a, Result<AgentEvent, AgentError>> {
        Box::pin(try_stream! {
            let tool_specs = self.tools.specs();
            let mut tool_rounds_executed = 0;

            loop {
                let provider_input: Cow<'_, Conversation> = if model.vision() {
                    Cow::Borrowed(conversation)
                } else {
                    Cow::Owned(conversation.without_images())
                };

                let message = self
                    .provider
                    .complete(model, &provider_input, &tool_specs)
                    .await
                    .map_err(AgentError::Provider)?;
                let tool_calls = message.tool_calls().to_vec();

                yield AgentEvent::AssistantReply(message.clone());
                conversation.push_assistant_message(message.clone());

                if tool_calls.is_empty() {
                    yield AgentEvent::Completed(message);
                    return;
                }

                if let Some(max) = self.max_tool_rounds
                    && tool_rounds_executed >= max
                {
                    Err(AgentError::MaxToolRoundsExceeded { max })?;
                }

                for tool_call in &tool_calls {
                    yield AgentEvent::ToolStarted(tool_call.clone());
                    let result = self.tools.execute(tool_call).await;
                    yield AgentEvent::ToolFinished(result.clone());
                    conversation.push_tool_result(result);
                }

                tool_rounds_executed += 1;
            }
        })
    }
}
