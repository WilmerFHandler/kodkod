use serde::{Deserialize, Serialize};

use crate::{Message, UserMessage};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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

    /// User turns as views over this conversation's messages.
    pub fn turns(&self) -> crate::turns::Turns<'_> {
        crate::turns::turns(self.messages())
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Append a user message verbatim (preserves `steered`, images, etc.).
    pub fn push_user_message(&mut self, user: UserMessage) {
        self.messages.push(Message::User(user));
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn replace_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    /// Return a copy of this conversation with all image attachments removed.
    pub fn without_images(&self) -> Self {
        Self {
            system_prompt: self.system_prompt.clone(),
            messages: self
                .messages
                .iter()
                .map(|message| match message {
                    Message::User(user) => Message::User(
                        UserMessage::new(user.content())
                            .with_images(Vec::new())
                            .with_steered(user.steered()),
                    ),
                    _ => message.clone(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssistantMessage, Message, UserMessage};

    #[test]
    fn push_message_preserves_steered_flag() {
        let mut conv = Conversation::new();
        conv.push_message(Message::User(UserMessage::new("go")));
        conv.push_message(Message::Assistant(AssistantMessage::new("ok")));
        conv.push_message(Message::User(
            UserMessage::new("steer me").with_steered(true),
        ));

        assert_eq!(conv.turns().count(), 1);
        let steer = match conv.messages().get(2).unwrap() {
            Message::User(u) => u,
            _ => panic!("expected user"),
        };
        assert!(steer.steered(), "steered flag must survive push_message");
    }

    #[test]
    fn push_user_message_preserves_steered_flag() {
        let mut conv = Conversation::new();
        conv.push_user_message(UserMessage::new("go"));
        conv.push_user_message(UserMessage::new("steer me").with_steered(true));
        assert_eq!(conv.turns().count(), 1);
        let steer = match conv.messages().get(1).unwrap() {
            Message::User(u) => u,
            _ => panic!("expected user"),
        };
        assert!(steer.steered());
    }

    #[test]
    fn without_images_preserves_steered_flag() {
        let mut conv = Conversation::new();
        conv.push_user_message(UserMessage::new("x").with_steered(true));
        let stripped = conv.without_images();
        let user = match stripped.messages().first().unwrap() {
            Message::User(u) => u,
            _ => panic!("expected user"),
        };
        assert!(user.steered());
    }
}
