use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    AssistantMessage, Conversation, Message, Model, Provider, ProviderError, ToolCall,
    ToolResult, ToolResultOutcome, ToolSpec,
};

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl OpenAiCompatibleProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_client(reqwest::Client::new(), base_url, None)
    }

    pub fn with_api_key(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self::with_client(
            reqwest::Client::new(),
            base_url,
            Some(api_key.into()),
        )
    }

    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::with_client(
            reqwest::Client::new(),
            OPENAI_BASE_URL,
            Some(api_key.into()),
        )
    }

    pub fn with_client(
        client: reqwest::Client,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

impl Provider for OpenAiCompatibleProvider {
    async fn complete(
        &self,
        model: &Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<AssistantMessage, ProviderError> {
        let conversation = if model.vision() {
            Cow::Borrowed(conversation)
        } else {
            Cow::Owned(conversation.without_images())
        };
        let request = ChatCompletionRequest::from_agent_input(model.id(), &conversation, tools)?;
        let mut builder = self.client.post(self.completions_url()).json(&request);

        if let Some(api_key) = &self.api_key {
            builder = builder.bearer_auth(api_key);
        }

        let response = builder
            .send()
            .await
            .map_err(|error| ProviderError::new(format!("OpenAI request failed: {error}")))?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::new(format!(
                "OpenAI request failed with status {status}: {body}"
            )));
        }

        let response = response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|error| {
                ProviderError::new(format!("OpenAI response was not valid JSON: {error}"))
            })?;

        response.into_assistant_message()
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatTool>,
}

impl<'a> ChatCompletionRequest<'a> {
    fn from_agent_input(
        model: &'a str,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<Self, ProviderError> {
        let mut messages = Vec::new();

        if let Some(system_prompt) = conversation.system_prompt() {
            messages.push(ChatMessage::system(system_prompt));
        }

        for message in conversation.messages() {
            messages.push(ChatMessage::from_message(message)?);
        }

        Ok(Self {
            model,
            messages,
            tools: tools.iter().map(ChatTool::from_spec).collect(),
        })
    }
}

#[derive(Debug)]
struct ChatMessage {
    role: &'static str,
    content: ChatContent,
    tool_calls: Vec<ChatToolCall>,
    tool_call_id: Option<String>,
}

impl ChatMessage {
    fn system(content: &str) -> Self {
        Self::new("system", ChatContent::text(content))
    }

    fn new(role: &'static str, content: ChatContent) -> Self {
        Self {
            role,
            content,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    fn from_message(message: &Message) -> Result<Self, ProviderError> {
        match message {
            Message::System(message) => {
                Ok(Self::new("system", ChatContent::text(message.content())))
            }
            Message::User(message) => Ok(Self::new("user", ChatContent::from_user(message))),
            Message::Assistant(message) => {
                let mut chat_message = Self::new("assistant", ChatContent::text(message.content()));
                chat_message.tool_calls = message
                    .tool_calls()
                    .iter()
                    .map(ChatToolCall::from_tool_call)
                    .collect::<Result<_, _>>()?;
                Ok(chat_message)
            }
            Message::ToolResult(result) => Ok(Self::from_tool_result(result)?),
        }
    }

    fn from_tool_result(result: &ToolResult) -> Result<Self, ProviderError> {
        Ok(Self {
            role: "tool",
            content: ChatContent::text(tool_result_content(result)?),
            tool_calls: Vec::new(),
            tool_call_id: Some(result.tool_call_id().to_owned()),
        })
    }
}

impl Serialize for ChatMessage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut len = 2;
        if !self.tool_calls.is_empty() {
            len += 1;
        }
        if self.tool_call_id.is_some() {
            len += 1;
        }

        let mut state = serializer.serialize_struct("ChatMessage", len)?;
        state.serialize_field("role", self.role)?;
        state.serialize_field("content", &self.content)?;
        if !self.tool_calls.is_empty() {
            state.serialize_field("tool_calls", &self.tool_calls)?;
        }
        if let Some(tool_call_id) = &self.tool_call_id {
            state.serialize_field("tool_call_id", tool_call_id)?;
        }
        state.end()
    }
}

/// OpenAI chat message content: either a plain string or a list of content parts.
#[derive(Debug)]
enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

impl ChatContent {
    fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    fn from_user(message: &crate::UserMessage) -> Self {
        if message.images().is_empty() {
            return Self::Text(message.content().to_owned());
        }

        let mut parts = Vec::with_capacity(message.images().len() + 1);
        parts.push(ChatContentPart::text(message.content()));
        for image in message.images() {
            parts.push(ChatContentPart::image_url(image.to_data_url()));
        }
        Self::Parts(parts)
    }
}

impl Serialize for ChatContent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text(text) => serializer.serialize_str(text),
            Self::Parts(parts) => parts.serialize(serializer),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ChatContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ChatImageUrl },
}

impl ChatContentPart {
    fn text(content: impl Into<String>) -> Self {
        Self::Text {
            text: content.into(),
        }
    }

    fn image_url(url: String) -> Self {
        Self::ImageUrl {
            image_url: ChatImageUrl { url },
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatImageUrl {
    url: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: ChatToolFunctionCall,
}

impl ChatToolCall {
    fn from_tool_call(call: &ToolCall) -> Result<Self, ProviderError> {
        Ok(Self {
            id: call.id().to_owned(),
            kind: "function".to_owned(),
            function: ChatToolFunctionCall {
                name: call.name().to_owned(),
                arguments: serde_json::to_string(call.arguments()).map_err(|error| {
                    ProviderError::new(format!("tool call arguments were not valid JSON: {error}"))
                })?,
            },
        })
    }

    fn into_tool_call(self) -> Result<ToolCall, ProviderError> {
        if self.kind != "function" {
            return Err(ProviderError::new(format!(
                "unsupported OpenAI tool call type: {}",
                self.kind
            )));
        }

        Ok(ToolCall::new(
            self.id,
            self.function.name,
            parse_tool_arguments(&self.function.arguments)?,
        ))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatToolFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct ChatTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: ChatToolFunction,
}

impl ChatTool {
    fn from_spec(spec: &ToolSpec) -> Self {
        Self {
            kind: "function",
            function: ChatToolFunction {
                name: spec.name().to_owned(),
                description: spec.description().to_owned(),
                parameters: spec.input_schema().clone(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

impl ChatCompletionResponse {
    fn into_assistant_message(self) -> Result<AssistantMessage, ProviderError> {
        let choice = self
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::new("OpenAI response did not include a choice"))?;

        choice.message.into_assistant_message()
    }
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatAssistantMessage,
}

#[derive(Debug, Deserialize)]
struct ChatAssistantMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ChatToolCall>,
}

impl ChatAssistantMessage {
    fn into_assistant_message(self) -> Result<AssistantMessage, ProviderError> {
        let tool_calls = self
            .tool_calls
            .into_iter()
            .map(ChatToolCall::into_tool_call)
            .collect::<Result<_, _>>()?;

        Ok(AssistantMessage::new(self.content.unwrap_or_default()).with_tool_calls(tool_calls))
    }
}

fn parse_tool_arguments(arguments: &str) -> Result<Value, ProviderError> {
    if arguments.trim().is_empty() {
        return Ok(Value::Object(Default::default()));
    }

    serde_json::from_str(arguments).map_err(|error| {
        ProviderError::new(format!(
            "OpenAI tool call arguments were not valid JSON: {error}"
        ))
    })
}

fn tool_result_content(result: &ToolResult) -> Result<String, ProviderError> {
    match result.outcome() {
        ToolResultOutcome::Success(value) => serde_json::to_string(value).map_err(|error| {
            ProviderError::new(format!("tool result value was not valid JSON: {error}"))
        }),
        ToolResultOutcome::Error(error) => serde_json::to_string(error).map_err(|error| {
            ProviderError::new(format!("tool result error was not valid JSON: {error}"))
        }),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{Image, ToolExecutorError, UserMessage};

    #[test]
    fn request_serializes_messages_and_tools_for_chat_completions() {
        let mut conversation = Conversation::new().with_system_prompt("Be concise.");
        conversation.push_user_message("hello");
        conversation.push_assistant_message(AssistantMessage::new("").with_tool_calls(vec![
            ToolCall::new("call_1", "echo", json!({ "value": "hello" })),
        ]));
        conversation.push_tool_result(ToolResult::success("call_1", json!({ "value": "hello" })));

        let request = ChatCompletionRequest::from_agent_input(
            "gpt-test",
            &conversation,
            &[ToolSpec::new(
                "echo",
                "Echoes arguments.",
                json!({
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    },
                    "required": ["value"]
                }),
            )],
        )
        .unwrap();

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "model": "gpt-test",
                "messages": [
                    { "role": "system", "content": "Be concise." },
                    { "role": "user", "content": "hello" },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": "{\"value\":\"hello\"}"
                                }
                            }
                        ]
                    },
                    {
                        "role": "tool",
                        "content": "{\"value\":\"hello\"}",
                        "tool_call_id": "call_1"
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "description": "Echoes arguments.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "value": { "type": "string" }
                                },
                                "required": ["value"]
                            }
                        }
                    }
                ]
            })
        );
    }

    #[test]
    fn request_omits_tools_when_none_are_registered() {
        let conversation = Conversation::new();
        let request =
            ChatCompletionRequest::from_agent_input("gpt-test", &conversation, &[]).unwrap();

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "model": "gpt-test",
                "messages": []
            })
        );
    }

    #[test]
    fn request_serializes_user_message_with_images_as_content_parts() {
        let image = Image::new("image/png", vec![0x89, 0x50]);
        let user = UserMessage::new("describe this").with_images(vec![image]);

        let message = ChatMessage::from_message(&Message::User(user)).unwrap();
        let value = serde_json::to_value(&message).unwrap();

        assert_eq!(value["role"], "user");
        let parts = value["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "describe this");
        assert_eq!(parts[1]["type"], "image_url");
        assert!(
            parts[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }

    #[test]
    fn request_serializes_user_message_without_images_as_plain_string() {
        let user = UserMessage::new("hello");

        let message = ChatMessage::from_message(&Message::User(user)).unwrap();
        let value = serde_json::to_value(&message).unwrap();

        assert_eq!(value["role"], "user");
        assert_eq!(value["content"], "hello");
    }

    #[test]
    fn response_parses_assistant_message_with_tool_calls() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": "{\"value\":\"hello\"}"
                                }
                            }
                        ]
                    }
                }
            ]
        }))
        .unwrap();

        let message = response.into_assistant_message().unwrap();

        assert_eq!(message.content(), "");
        assert_eq!(
            message.tool_calls(),
            &[ToolCall::new("call_1", "echo", json!({ "value": "hello" }))]
        );
    }

    #[test]
    fn response_rejects_invalid_tool_call_arguments() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                "name": "echo",
                                    "arguments": "{"
                                }
                            }
                        ]
                    }
                }
            ]
        }))
        .unwrap();

        let error = response.into_assistant_message().unwrap_err();

        assert!(
            error
                .message()
                .starts_with("OpenAI tool call arguments were not valid JSON")
        );
    }

    #[test]
    fn tool_errors_are_sent_as_structured_json_strings() {
        let message = ChatMessage::from_message(&Message::ToolResult(ToolResult::failure(
            "call_1",
            ToolExecutorError::UnknownTool("missing".to_owned()),
        )))
        .unwrap();

        assert_eq!(
            serde_json::to_value(message).unwrap(),
            json!({
                "role": "tool",
                "content": "{\"type\":\"unknown_tool\",\"value\":\"missing\"}",
                "tool_call_id": "call_1"
            })
        );
    }

    #[test]
    fn core_message_serde_stays_independent_from_openai_adapter() {
        assert_eq!(
            serde_json::to_value(Message::User(UserMessage::new("hello"))).unwrap(),
            json!({
                "role": "user",
                "content": "hello"
            })
        );
    }

    #[test]
    fn request_strips_images_for_non_vision_model() {
        let image = Image::new("image/png", vec![0x89, 0x50]);
        let mut conversation = Conversation::new();
        conversation.push_user_message_with_images("describe this", vec![image]);

        let request = ChatCompletionRequest::from_agent_input(
            "gpt-test",
            &conversation.without_images(),
            &[],
        )
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let content = &value["messages"][0]["content"];
        assert!(content.is_string());
        assert_eq!(content.as_str().unwrap(), "describe this");
    }

    #[test]
    fn request_preserves_images_for_vision_model() {
        let image = Image::new("image/png", vec![0x89, 0x50]);
        let mut conversation = Conversation::new();
        conversation.push_user_message_with_images("describe this", vec![image]);
        let model = Model::with_vision("gpt-vision", "GPT Vision");

        let request = ChatCompletionRequest::from_agent_input(model.id(), &conversation, &[])
            .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let content = &value["messages"][0]["content"];
        assert!(content.is_array());
    }
}
