pub mod error;

use std::sync::Arc;

pub use error::AgentError;

use crate::{AssistantMessage, Conversation, Provider, Tool, ToolExecutor};

const DEFAULT_MAX_TOOL_ROUNDS: usize = 8;

pub struct Agent<P> {
    provider: P,
    tools: ToolExecutor,
    max_tool_rounds: usize,
}

impl<P> Agent<P>
where
    P: Provider,
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

    pub async fn run(
        &self,
        conversation: &mut Conversation,
        prompt: impl Into<String>,
    ) -> Result<AssistantMessage, AgentError> {
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

            conversation.push_assistant_message(message.clone());

            if tool_calls.is_empty() {
                return Ok(message);
            }

            if tool_rounds_executed >= self.max_tool_rounds {
                return Err(AgentError::MaxToolRoundsExceeded {
                    max: self.max_tool_rounds,
                });
            }

            for tool_call in &tool_calls {
                let result = self.tools.execute(tool_call).await;
                conversation.push_tool_result(result);
            }

            tool_rounds_executed += 1;
        }
    }
}
