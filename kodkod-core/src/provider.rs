use std::error::Error;
use std::future::Future;

use crate::{AssistantMessage, Conversation, ToolSpec};

pub trait Provider {
    type Model: Sync;
    type Error: Error + Send + Sync + 'static;

    fn supports_vision(&self, model: &Self::Model) -> bool;

    /// Whether the model is explicitly supported for computer-use tools.
    fn supports_computer_use(&self, _model: &Self::Model) -> bool {
        false
    }

    /// Produce the assistant response for one provider round.
    ///
    /// The returned future may be dropped before completion when the agent run
    /// is cancelled or its event stream is dropped. Implementations must be
    /// cancellation-safe and should propagate a drop to any in-flight transport
    /// request where possible. Work spawned independently by an implementation
    /// is not stopped automatically. Dropping stops local polling and transport
    /// ownership; whether remote generation and billing stop is controlled by
    /// the provider and transport.
    fn complete(
        &self,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, Self::Error>> + Send;
}
