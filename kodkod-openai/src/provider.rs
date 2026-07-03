use std::marker::PhantomData;

use kodkod::{AssistantMessage, Conversation, Provider, ToolSpec};

use crate::api::{ApiErrorResponse, ChatCompletionResponse};
use crate::convert::{build_request, parse_assistant_message};
use crate::error::OpenAiError;
use crate::model::OpenAiModel;

/// Provider for OpenAI-compatible `/chat/completions` endpoints.
///
/// `M` is a zero-sized type marker tying this provider to your [`OpenAiModel`]
/// implementation (e.g. `OpenAiProvider::<MyModel>::new(url)`).
#[derive(Clone)]
pub struct OpenAiProvider<M = ()> {
    chat_completions_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
    _model: PhantomData<M>,
}

impl<M> OpenAiProvider<M> {
    pub fn new(base_url: impl Into<String>) -> Self {
        let base = base_url.into().trim_end_matches('/').to_owned();
        Self {
            chat_completions_url: format!("{base}/chat/completions"),
            api_key: None,
            client: reqwest::Client::new(),
            _model: PhantomData,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub fn chat_completions_url(&self) -> &str {
        &self.chat_completions_url
    }
}

impl<M> Provider for OpenAiProvider<M>
where
    M: OpenAiModel,
{
    type Model = M;
    type Error = OpenAiError;

    fn supports_vision(&self, model: &M) -> bool {
        model.supports_vision()
    }

    async fn complete(
        &self,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<AssistantMessage, Self::Error> {
        let request = build_request(model.id(), conversation, tools);
        let mut http_request = self
            .client
            .post(&self.chat_completions_url)
            .json(&request);

        if let Some(api_key) = &self.api_key {
            http_request = http_request.bearer_auth(api_key);
        }

        let response = http_request.send().await?;
        let status = response.status();

        if !status.is_success() {
            let message = match response.json::<ApiErrorResponse>().await {
                Ok(body) => body.error.message,
                Err(_) => status.to_string(),
            };
            return Err(OpenAiError::Api {
                status: status.as_u16(),
                message,
            });
        }

        let body = response.json::<ChatCompletionResponse>().await?;
        parse_assistant_message(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kodkod::UserMessage;
    use serde_json::json;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct TestModel {
        id: &'static str,
        vision: bool,
    }

    impl OpenAiModel for TestModel {
        fn id(&self) -> &str {
            self.id
        }

        fn supports_vision(&self) -> bool {
            self.vision
        }
    }

    #[tokio::test]
    async fn posts_chat_completion_and_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_json(json!({
                "model": "llama3",
                "messages": [{ "role": "user", "content": "hello" }],
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "content": "world",
                        "tool_calls": []
                    }
                }]
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::<TestModel>::new(format!("{}/v1", server.uri()));
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("hello"));

        let model = TestModel {
            id: "llama3",
            vision: false,
        };
        let message = provider
            .complete(&model, &conversation, &[])
            .await
            .expect("completion should succeed");

        assert_eq!(message.content(), "world");
        assert!(message.tool_calls().is_empty());
    }
}