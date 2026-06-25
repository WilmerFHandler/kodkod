pub mod error;
#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;
pub mod retry;

use std::future::Future;

pub use error::{ProviderError, ProviderErrorKind};
#[cfg(feature = "openai-compatible")]
pub use openai_compatible::OpenAiCompatibleProvider;
pub use retry::{RetryPolicy, RetryProvider};

use crate::{AssistantMessage, Conversation, ToolSpec};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    id: String,
    display_name: String,
    supports_vision: bool,
}

impl Model {
    /// Create a model that does not support vision.
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            supports_vision: false,
        }
    }

    /// Create a model that supports vision/image inputs.
    pub fn with_vision(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            supports_vision: true,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// Whether the model accepts image content in user prompts.
    pub fn vision(&self) -> bool {
        self.supports_vision
    }
}

pub trait Provider {
    fn models(&self) -> Vec<Model> {
        Vec::new()
    }

    fn complete(
        &self,
        model: &Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send;
}
