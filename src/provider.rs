use std::future::Future;

use crate::{AssistantMessage, Conversation, ProviderError};

pub trait Provider {
    fn complete(
        &self,
        conversation: &Conversation,
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send;
}
