# lynx-agent

`lynx-agent` is a small Rust library for running provider-agnostic agent loops
with tool calling support.

The core crate defines conversation state, provider traits, tool traits, tool
execution, and structured message/result types. The optional
`openai-compatible` feature adds a `reqwest`-based adapter for OpenAI-compatible
chat completion APIs.

## Installation

```toml
[dependencies]
lynx-agent = "0.1"
```

Enable the OpenAI-compatible provider when you want the built-in HTTP adapter:

```toml
[dependencies]
lynx-agent = { version = "0.1", features = ["openai-compatible"] }
```

## Example

```rust
use std::future::ready;

use lynx_agent::{Agent, AssistantMessage, Conversation, Provider, ProviderError, ToolSpec};

struct EchoProvider;

impl Provider for EchoProvider {
    fn complete(
        &self,
        conversation: &Conversation,
        _tools: &[ToolSpec],
    ) -> impl std::future::Future<Output = Result<AssistantMessage, ProviderError>> + Send {
        let content = conversation
            .messages()
            .last()
            .map(|_| "hello from lynx")
            .unwrap_or("hello");

        ready(Ok(AssistantMessage::new(content)))
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new(EchoProvider);
    let mut conversation = Conversation::new();

    let response = agent.run(&mut conversation, "hello").await?;
    assert_eq!(response.content(), "hello from lynx");
    Ok(())
}
```

With the `openai-compatible` feature:

```rust,no_run
use lynx_agent::{Agent, Conversation, OpenAiCompatibleProvider};

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAiCompatibleProvider::openai(
        std::env::var("OPENAI_API_KEY")?,
        "gpt-4.1-mini",
    );
    let agent = Agent::new(provider);
    let mut conversation = Conversation::new();

    let response = agent.run(&mut conversation, "Write one short sentence.").await?;
    println!("{}", response.content());
    Ok(())
}
```

## License

MIT
