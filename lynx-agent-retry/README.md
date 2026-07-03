# lynx-agent-retry

Optional retry middleware for [`lynx-agent`] providers.

Wrap any [`Provider`] with [`RetryProvider`] to retry transient failures between
`complete` attempts. Whether an error is retryable is determined by the
[`Retryable`] trait on the provider's associated error type.

[`lynx-agent`]: https://github.com/WilmerFHandler/lynx-agent
[`Provider`]: https://docs.rs/lynx-agent/latest/lynx_agent/trait.Provider.html
[`RetryProvider`]: https://docs.rs/lynx-agent-retry/latest/lynx_agent_retry/struct.RetryProvider.html
[`Retryable`]: https://docs.rs/lynx-agent-retry/latest/lynx_agent_retry/trait.Retryable.html

## Installation

```toml
[dependencies]
lynx-agent = "0.1"
lynx-agent-retry = "0.1"
```

## Example

```rust
use lynx_agent::{Agent, Provider, ProviderError, /* ... */};
use lynx_agent_retry::{RetryPolicy, RetryProvider};

let provider = RetryProvider::with_policy(my_provider, RetryPolicy::default());
let agent = Agent::new(provider);
```

[`ProviderError`] from `lynx-agent` already implements [`Retryable`] when using
the OpenAI-compatible transport helper.

[`ProviderError`]: https://docs.rs/lynx-agent/latest/lynx_agent/struct.ProviderError.html

## License

MIT