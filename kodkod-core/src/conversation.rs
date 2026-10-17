use serde::{Deserialize, Serialize};

use crate::{Message, ToolSpec, UserMessage};

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

    pub fn set_system_prompt(&mut self, prompt: Option<&str>) {
        self.system_prompt = prompt.map(str::to_owned);
    }

    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Tokenizer-free estimate of the tokens this conversation would send.
    ///
    /// Pass `&[]` when tools are not part of the request. Continuation tokens
    /// are not included; add `Provider::estimate_continuation_tokens` separately.
    pub fn estimate_tokens(&self, tools: &[ToolSpec]) -> u64 {
        crate::estimate::estimate_conversation(self, tools)
    }

    /// User turns as views over this conversation's messages.
    pub fn turns(&self) -> crate::turns::Turns<'_> {
        crate::turns::turns(self.messages())
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Append a user message verbatim (preserves `steered`, images, documents, etc.).
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
                            .with_documents(user.documents().to_vec())
                            .with_steered(user.steered()),
                    ),
                    Message::ToolResult(result) => Message::ToolResult(result.without_images()),
                    _ => message.clone(),
                })
                .collect(),
        }
    }

    /// Return a copy that retains image attachments on the newest `keep` image-bearing
    /// tool results. User images, documents, tool values, call IDs, and message order stay intact.
    pub fn retain_recent_tool_image_observations(&self, keep: usize) -> Self {
        let mut remaining = keep;
        let messages = self
            .messages
            .iter()
            .rev()
            .map(|message| match message {
                Message::ToolResult(result)
                    if matches!(result.outcome(), crate::ToolResultOutcome::Success(output) if !output.images().is_empty()) =>
                {
                    if remaining > 0 {
                        remaining -= 1;
                        message.clone()
                    } else {
                        Message::ToolResult(result.without_images())
                    }
                }
                _ => message.clone(),
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        Self {
            system_prompt: self.system_prompt.clone(),
            messages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AssistantMessage, Document, Image, Message, ToolCall, ToolOutput, ToolResult, UserMessage,
    };
    use serde_json::json;

    #[test]
    fn steered_flag_survives_push_and_image_stripping() {
        let mut conv = Conversation::new();
        conv.push_user_message(UserMessage::new("go"));
        conv.push_message(Message::User(
            UserMessage::new("steer me").with_steered(true),
        ));
        assert_eq!(conv.turns().count(), 1);
        assert!(matches!(
            conv.messages().last(),
            Some(Message::User(user)) if user.steered()
        ));

        let stripped = conv.without_images();
        assert!(matches!(
            stripped.messages().last(),
            Some(Message::User(user)) if user.steered()
        ));
    }

    #[test]
    fn without_images_strips_attachments() {
        use crate::{Document, Image, ToolOutput, ToolResult};

        let mut conv = Conversation::new();
        conv.push_user_message(
            UserMessage::new("describe")
                .with_images(vec![Image::new("image/png", vec![0x89, 0x50])])
                .with_documents(vec![
                    Document::try_new("text/plain", "notes.txt", b"notes").unwrap(),
                ]),
        );
        conv.push_message(Message::ToolResult(ToolResult::success(
            "call_1",
            ToolOutput::new(serde_json::json!({"path": "image.png"}))
                .with_images(vec![Image::new("image/png", vec![0x89, 0x50])]),
        )));
        let stripped = conv.without_images();
        assert!(matches!(
            stripped.messages().first(),
            Some(Message::User(user)) if user.content() == "describe"
                && user.images().is_empty()
                && user.documents().len() == 1
        ));
        assert!(matches!(
            stripped.messages().get(1),
            Some(Message::ToolResult(result))
                if matches!(result.outcome(), crate::ToolResultOutcome::Success(output) if output.images().is_empty())
        ));
    }

    #[test]
    fn long_tool_image_history_retains_recent_observations_and_reduces_request_size() {
        let image = Image::new("image/png", vec![7; 4096]);
        let document =
            Document::try_new("text/plain", "notes.txt", b"keep this attachment".to_vec()).unwrap();
        let mut transcript = Conversation::new();
        transcript.push_user_message(
            UserMessage::new("inspect these steps")
                .with_images(vec![image])
                .with_documents(vec![document]),
        );
        let mut active = transcript.clone();
        for step in 0..12 {
            let call_id = format!("call-{step}");
            let assistant =
                Message::Assistant(
                    AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                        call_id.clone(),
                        "screenshot",
                        json!({"step": step}),
                    )]),
                );
            let result = Message::ToolResult(ToolResult::success(
                call_id,
                ToolOutput::new(json!({"step": step, "caption": "screen state"}))
                    .with_images(vec![Image::new("image/png", vec![step as u8; 16_384])]),
            ));
            transcript.push_message(assistant.clone());
            transcript.push_message(result.clone());
            active.push_message(assistant);
            active.push_message(result);
            active = active.retain_recent_tool_image_observations(2);
        }

        let before_tokens = transcript.estimate_tokens(&[]);
        let after_tokens = active.estimate_tokens(&[]);
        let before_bytes = serde_json::to_vec(&transcript).unwrap().len();
        let after_bytes = serde_json::to_vec(&active).unwrap().len();
        let retained_images = active
            .messages()
            .iter()
            .filter(|message| {
                matches!(message, Message::ToolResult(result) if matches!(result.outcome(), crate::ToolResultOutcome::Success(output) if !output.images().is_empty()))
            })
            .count();

        assert_eq!(active.messages().len(), transcript.messages().len());
        assert_eq!(retained_images, 2);
        assert!(matches!(
            active.messages().first(),
            Some(Message::User(user)) if user.images().len() == 1 && user.documents().len() == 1
        ));
        for step in 0..12 {
            let call_id = format!("call-{step}");
            assert!(active.messages().iter().any(
                |message| matches!(message, Message::Assistant(assistant) if assistant.tool_calls().iter().any(|call| call.id() == call_id.as_str()))
            ));
            assert!(active.messages().iter().any(
                |message| matches!(message, Message::ToolResult(result) if result.tool_call_id() == call_id.as_str() && result.value().is_some())
            ));
        }
        assert!(
            after_tokens < before_tokens,
            "heuristic input tokens {before_tokens} -> {after_tokens}; request bytes {before_bytes} -> {after_bytes}"
        );
        assert!(
            after_bytes < before_bytes,
            "request bytes {before_bytes} -> {after_bytes}; heuristic input tokens {before_tokens} -> {after_tokens}"
        );
    }
}
