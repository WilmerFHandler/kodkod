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

    fn complete(
        &self,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, Self::Error>> + Send;
}
