# lynx-agent

`lynx-agent` is a small Rust library for running provider-agnostic agent loops
with tool calling support.

The core crate defines conversation state, the [`Provider`] trait, tool traits,
tool execution, and structured message/result types. The optional
`openai-compatible` feature adds [`complete_openai_compatible`] for OpenAI-shaped
HTTP APIs.

## Installation

```toml
[dependencies]
lynx-agent = "0.1"
```

Enable the OpenAI-compatible helper when you need the shared HTTP adapter:

```toml
[dependencies]
lynx-agent = { version = "0.1", features = ["openai-compatible"] }
```

## Example

Providers bring their own model type. The agent only asks for vision support and
delegates the actual request to `complete`.

```rust
use std::future::ready;

use futures::StreamExt;
use lynx_agent::{
    Agent, AgentEvent, AssistantMessage, Conversation, Provider, ProviderError,
    TaskControl, ToolSpec,
};

struct EchoModel;

struct EchoProvider;

impl Provider for EchoProvider {
    type Model = EchoModel;

    fn supports_vision(&self, _model: &EchoModel) -> bool {
        false
    }

    fn complete(
        &self,
        _model: &EchoModel,
        conversation: &Conversation,
        _tools: &[ToolSpec],
    ) -> impl std::future::Future<Output = Result<AssistantMessage, ProviderError>> + Send {
        let content = conversation
            .messages()
            .iter()
            .rev()
            .find_map(|message| match message {
                lynx_agent::Message::User(user) => Some(user.content().to_owned()),
                _ => None,
            })
            .unwrap_or_else(|| "hello".to_owned());

        ready(Ok(AssistantMessage::new(content)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new(EchoProvider);
    let mut conversation = Conversation::new();
    conversation.push_user_message(lynx_agent::UserMessage::new("hello"));

    let model = EchoModel;
    let control = TaskControl::new();
    let mut stream = agent.run(&mut conversation, &model, &control);

    while let Some(event) = stream.next().await {
        if let AgentEvent::Completed(message) = event? {
            assert_eq!(message.content(), "hello");
            break;
        }
    }

    Ok(())
}
```

With the `openai-compatible` feature:

```rust,no_run
use futures::StreamExt;
use lynx_agent::{
    Agent, AgentEvent, Conversation, Provider, ProviderError, TaskControl, ToolSpec,
    complete_openai_compatible,
};

struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
}

struct OpenAiModel(&'static str);

impl Provider for OpenAiProvider {
    type Model = OpenAiModel;

    fn supports_vision(&self, _model: &OpenAiModel) -> bool {
        false
    }

    async fn complete(
        &self,
        model: &OpenAiModel,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<lynx_agent::AssistantMessage, ProviderError> {
        complete_openai_compatible(
            &self.client,
            "https://api.openai.com/v1",
            Some(&self.api_key),
            model.0,
            conversation,
            tools,
        )
        .await
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAiProvider {
        client: reqwest::Client::new(),
        api_key: std::env::var("OPENAI_API_KEY")?,
    };
    let agent = Agent::new(provider);
    let mut conversation = Conversation::new();
    conversation.push_user_message(lynx_agent::UserMessage::new("Write one short sentence."));

    let model = OpenAiModel("gpt-4.1-mini");
    let mut stream = agent.run(&mut conversation, &model, &TaskControl::new());

    while let Some(event) = stream.next().await {
        if let AgentEvent::Completed(message) = event? {
            println!("{}", message.content());
            break;
        }
    }

    Ok(())
}
```

## License

MIT