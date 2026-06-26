pub mod control;
pub mod error;
pub mod event;

use std::borrow::Cow;
use std::sync::Arc;

use async_stream::try_stream;

pub use control::TaskControl;
pub use error::AgentError;
pub use event::AgentEvent;

/// Streaming events from an [`Agent::run`] call.
pub type Task<'a> = futures::stream::BoxStream<'a, Result<AgentEvent, AgentError>>;

use crate::{Conversation, Message, Model, Provider, Tool, ToolExecutor};

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
    ///
    /// Pass a [`TaskControl`] so external callers can cancel the run between
    /// provider rounds (e.g. a GUI "Cancel" button) or steer it by injecting new
    /// user messages at round boundaries.
    pub fn run<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a Model,
        control: &'a TaskControl,
    ) -> Task<'a> {
        Box::pin(try_stream! {
            let tool_specs = self.tools.specs();
            let mut tool_rounds_executed = 0;

            loop {
                if control.is_cancelled() {
                    Err(AgentError::Cancelled)?;
                }

                // Inject any user messages queued via TaskControl::steer. Each is
                // appended to the conversation before the next provider call, so the
                // model sees it this round. The event is yielded first so callers can
                // mirror it into shared state.
                for user in control.drain_pending_steers() {
                    yield AgentEvent::Steered(user.clone());
                    conversation.push_message(Message::User(user));
                }

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
                conversation.push_message(Message::Assistant(message.clone()));

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
                    conversation.push_message(Message::ToolResult(result));
                }

                tool_rounds_executed += 1;
            }
        })
    }
}
