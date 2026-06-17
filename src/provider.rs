pub mod error;
#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;

use std::future::Future;

pub use error::ProviderError;
#[cfg(feature = "openai-compatible")]
pub use openai_compatible::OpenAiCompatibleProvider;

use crate::{AssistantMessage, Conversation, ToolSpec};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    id: String,
    display_name: String,
}

impl Model {
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }
}

pub trait Provider {
    fn models(&self) -> Vec<Model> {
        Vec::new()
    }

    fn complete(
        &self,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send;
}
