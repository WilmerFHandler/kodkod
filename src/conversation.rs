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
            Message::User(user) => {
                self.push_user_message_with_images(user.content(), user.images().to_vec());
            }
            Message::Assistant(assistant) => self.push_assistant_message(assistant),
            Message::System(system) => self.push_system_message(system.content()),
            Message::ToolResult(result) => self.push_tool_result(result),
        }
    }

    /// Return a copy of this conversation with all image attachments removed.
    pub fn without_images(&self) -> Self {
        Self {
            system_prompt: self.system_prompt.clone(),
            messages: self
                .messages
                .iter()
                .map(|message| match message {
                    Message::User(user) => {
                        Message::User(UserMessage::new(user.content()).with_images(Vec::new()))
                    }
                    _ => message.clone(),
                })
                .collect(),
        }
    }
}
