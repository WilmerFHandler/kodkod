use crate::{AssistantMessage, Conversation, Provider, ProviderError};

#[derive(Debug, Clone)]
pub struct Agent<P> {
    provider: P,
}

impl<P> Agent<P>
where
    P: Provider,
{
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub async fn run(
        &self,
        conversation: &mut Conversation,
        prompt: impl Into<String>,
    ) -> Result<AssistantMessage, ProviderError> {
        conversation.push_user_message(prompt);
        let message = self.provider.complete(conversation).await?;
        conversation.push_assistant_message(message.clone());
        Ok(message)
    }
}
