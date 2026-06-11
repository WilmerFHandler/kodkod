use std::fmt;
use std::future::Future;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserMessage {
    content: String,
}

impl UserMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantMessage {
    content: String,
}

impl AssistantMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemMessage {
    content: String,
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Conversation {
    system_prompt: Option<String>,
    messages: Vec<Message>,
}

impl Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn push_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(Message::User(UserMessage::new(content)));
    }

    pub fn push_assistant_message(&mut self, message: AssistantMessage) {
        self.messages.push(Message::Assistant(message));
    }

    pub fn push_system_message(&mut self, content: impl Into<String>) {
        self.messages
            .push(Message::System(SystemMessage::new(content)));
    }
}

pub trait Provider {
    fn complete(
        &self,
        conversation: &Conversation,
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError {
    message: String,
}

impl ProviderError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderError {}

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

#[cfg(test)]
mod tests {
    use std::future::{Future, ready};
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    use super::*;

    struct EchoProvider;

    impl Provider for EchoProvider {
        fn complete(
            &self,
            conversation: &Conversation,
        ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
            let prompt = conversation.messages().iter().rev().find_map(|message| {
                if let Message::User(message) = message {
                    Some(message.content().to_owned())
                } else {
                    None
                }
            });

            ready(Ok(AssistantMessage::new(
                prompt.unwrap_or_else(|| "no prompt".to_owned()),
            )))
        }
    }

    #[test]
    fn conversation_tracks_messages() {
        let mut conversation = Conversation::new().with_system_prompt("Be concise.");

        conversation.push_user_message("hello");
        conversation.push_assistant_message(AssistantMessage::new("hi"));

        assert_eq!(conversation.system_prompt(), Some("Be concise."));
        assert_eq!(conversation.messages().len(), 2);
        assert!(matches!(
            &conversation.messages()[0],
            Message::User(message) if message.content() == "hello"
        ));
        assert!(matches!(
            &conversation.messages()[1],
            Message::Assistant(message) if message.content() == "hi"
        ));
    }

    #[test]
    fn agent_appends_user_and_assistant_messages() {
        let agent = Agent::new(EchoProvider);
        let mut conversation = Conversation::new();

        let response = block_on(agent.run(&mut conversation, "hello")).unwrap();

        assert_eq!(response.content(), "hello");
        assert_eq!(conversation.messages().len(), 2);
        assert!(matches!(
            &conversation.messages()[0],
            Message::User(message) if message.content() == "hello"
        ));
        assert!(matches!(
            &conversation.messages()[1],
            Message::Assistant(message) if message.content() == "hello"
        ));
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        let waker = Waker::from(Arc::new(NoopWaker));
        let mut context = Context::from_waker(&waker);
        let mut future = Box::pin(future);

        loop {
            match Pin::new(&mut future).poll(&mut context) {
                Poll::Ready(output) => return output,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    struct NoopWaker;

    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }
}
