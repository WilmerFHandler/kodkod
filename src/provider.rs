pub mod error;
#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;
pub mod retry;

use std::future::Future;

pub use error::{ProviderError, ProviderErrorKind};
#[cfg(feature = "openai-compatible")]
pub use openai_compatible::complete_openai_compatible;
pub use retry::{RetryPolicy, RetryProvider};

use crate::{AssistantMessage, Conversation, ToolSpec};

pub trait Provider {
    type Model: Sync;

    fn supports_vision(&self, model: &Self::Model) -> bool;

    fn complete(
        &self,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send;
}
