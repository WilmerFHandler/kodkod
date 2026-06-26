use serde::{Deserialize, Serialize};

use crate::{AssistantMessage, Image, Message, SystemMessage, ToolResult, UserMessage};

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

    pub fn push_user_message(&mut self, content: impl Into<String>) {
        self.push_user_message_with_images(content, Vec::new());
    }

    pub fn push_user_message_with_images(
        &mut self,
        content: impl Into<String>,
        images: Vec<Image>,
    ) {
        self.messages
            .push(Message::User(UserMessage::new(content).with_images(images)));
    }

    pub fn push_assistant_message(&mut self, message: AssistantMessage) {
        self.messages.push(Message::Assistant(message));
    }

    pub fn push_system_message(&mut self, content: impl Into<String>) {
        self.messages
            .push(Message::System(SystemMessage::new(content)));
    }

    pub fn push_tool_result(&mut self, result: ToolResult) {
        self.messages.push(Message::ToolResult(result));
    }

    pub fn push_message(&mut self, message: Message) {
        match message {
            // Push user messages verbatim to preserve all fields (e.g. the
            // `steered` flag that keeps injected messages in the same turn).
            Message::User(user) => self.messages.push(Message::User(user)),
            Message::Assistant(assistant) => self.push_assistant_message(assistant),
            Message::System(system) => self.push_system_message(system.content()),
            Message::ToolResult(result) => self.push_tool_result(result),
        }
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
    use crate::{Message, UserMessage};

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
    fn without_images_preserves_steered_flag() {
        let mut conv = Conversation::new();
        conv.push_message(Message::User(UserMessage::new("x").with_steered(true)));
        let stripped = conv.without_images();
        let user = match stripped.messages().first().unwrap() {
            Message::User(u) => u,
            _ => panic!("expected user"),
        };
        assert!(user.steered());
    }
}
