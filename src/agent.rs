pub mod error;
pub mod event;

use std::sync::Arc;

use async_stream::try_stream;
use futures::stream::BoxStream;

pub use error::AgentError;
pub use event::AgentEvent;

use crate::{Conversation, Provider, Tool, ToolExecutor};

const DEFAULT_MAX_TOOL_ROUNDS: usize = 8;

pub struct Agent<P> {
    provider: P,
    tools: ToolExecutor,
    max_tool_rounds: usize,
}

impl<P> Agent<P>
where
    P: Provider + Sync,
{
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            tools: ToolExecutor::new(),
            max_tool_rounds: DEFAULT_MAX_TOOL_ROUNDS,
        }
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub fn tools(&self) -> &ToolExecutor {
        &self.tools
    }

    pub fn max_tool_rounds(&self) -> usize {
        self.max_tool_rounds
    }

    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.register_tool(tool);
        self
    }

    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.max_tool_rounds = max_tool_rounds;
        self
    }

    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.register(tool);
    }

    pub fn run<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        prompt: impl Into<String>,
    ) -> BoxStream<'a, Result<AgentEvent, AgentError>> {
        let prompt = prompt.into();

        Box::pin(try_stream! {
            conversation.push_user_message(prompt);

            let tool_specs = self.tools.specs();
            let mut tool_rounds_executed = 0;

            loop {
                let message = self
                    .provider
                    .complete(conversation, &tool_specs)
                    .await
                    .map_err(AgentError::Provider)?;
                let tool_calls = message.tool_calls().to_vec();

                yield AgentEvent::AssistantReply(message.clone());
                conversation.push_assistant_message(message.clone());

                if tool_calls.is_empty() {
                    yield AgentEvent::Completed(message);
                    return;
                }

                if tool_rounds_executed >= self.max_tool_rounds {
                    Err(AgentError::MaxToolRoundsExceeded {
                        max: self.max_tool_rounds,
                    })?;
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