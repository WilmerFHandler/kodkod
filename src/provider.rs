pub mod error;

use std::future::Future;

pub use error::ProviderError;

use crate::{AssistantMessage, Conversation};

pub trait Provider {
    fn complete(
        &self,
        conversation: &Conversation,
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send;
}
