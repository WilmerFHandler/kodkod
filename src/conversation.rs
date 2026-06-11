use crate::{AssistantMessage, Message, SystemMessage, UserMessage};

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
