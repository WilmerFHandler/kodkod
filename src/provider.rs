pub mod error;
#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;

use std::future::Future;

pub use error::ProviderError;
#[cfg(feature = "openai-compatible")]
pub use openai_compatible::OpenAiCompatibleProvider;

use crate::{AssistantMessage, Conversation, ToolSpec};

pub trait Provider {
    fn complete(
        &self,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send;
}
