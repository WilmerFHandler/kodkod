pub mod error;

use std::future::Future;

pub use error::ProviderError;

use crate::{AssistantMessage, Conversation, ToolSpec};

pub trait Provider {
    fn complete(
        &self,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send;
}
