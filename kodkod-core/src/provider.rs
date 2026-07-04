use std::error::Error;
use std::future::Future;

use crate::{AssistantMessage, Conversation, ToolSpec};

pub trait Provider {
    type Model: Sync;
    type Error: Error + Send + Sync + 'static;

    fn supports_vision(&self, model: &Self::Model) -> bool;

    fn complete(
        &self,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, Self::Error>> + Send;
}
