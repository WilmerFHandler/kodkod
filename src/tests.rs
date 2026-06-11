use std::future::{Future, ready};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use crate::{Agent, AssistantMessage, Conversation, Message, Provider, ProviderError};

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
