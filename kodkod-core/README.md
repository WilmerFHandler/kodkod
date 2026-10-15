# kodkod

`kodkod` is a small Rust library for running provider-agnostic agent loops
with tool calling support.

The crate defines conversation state, the [`Provider`] trait, tool traits,
tool execution, retry middleware, and structured message/result types.

## Installation

```toml
[dependencies]
kodkod = "0.1"
```

## Example

Providers bring their own model and error types. The agent only asks for vision
support and delegates the actual request to `complete`.

```rust
use std::error::Error;
use std::fmt;
use std::future::ready;

use futures::StreamExt;
use kodkod::{
    Agent, AgentEvent, AssistantMessage, Conversation, Provider, TaskControl, ToolSpec,
};

struct EchoModel;

#[derive(Debug)]
struct EchoError;

impl fmt::Display for EchoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("echo provider failed")
    }
}

impl Error for EchoError {}

struct EchoProvider;

impl Provider for EchoProvider {
    type Model = EchoModel;
    type Error = EchoError;
    type Continuation = ();

    fn create_continuation(&self, _model: &EchoModel) -> Self::Continuation {}

    fn supports_vision(&self, _model: &EchoModel) -> bool {
        false
    }

    fn complete(
        &self,
        _continuation: &Self::Continuation,
        _model: &EchoModel,
        conversation: &Conversation,
        _tools: &[ToolSpec],
    ) -> impl std::future::Future<
        Output = Result<(AssistantMessage, Self::Continuation), Self::Error>,
    > + Send {
        let content = conversation
            .messages()
            .iter()
            .rev()
            .find_map(|message| match message {
                kodkod::Message::User(user) => Some(user.content().to_owned()),
                _ => None,
            })
            .unwrap_or_else(|| "hello".to_owned());

        ready(Ok((AssistantMessage::new(content), ())))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new(EchoProvider);
    let mut conversation = Conversation::new();
    let model = EchoModel;
    let control = TaskControl::new();
    let mut stream = agent.run_turn(
        &mut conversation,
        &model,
        kodkod::UserMessage::new("hello"),
        &control,
    );

    while let Some(event) = stream.next().await {
        if let AgentEvent::Completed(message, _usage) = event? {
            assert_eq!(message.content(), "hello");
            break;
        }
    }

    Ok(())
}
```

With retry middleware:

```rust
use kodkod::RetryProvider;

let agent = Agent::new(RetryProvider::new(EchoProvider));
```

`Conversation::estimate_tokens` is a tokenizer-free request-size heuristic.
`AgentContext::estimated_current_context` wraps that heuristic, including
provider-private continuation replay, in a `ContextEstimate` tagged with
`ContextEstimateProvenance::Heuristic`. It is separate from authoritative
`TokenUsage`: providers report input, cached input, output, reasoning output,
and total counts when available, while missing fields remain `None`. A terminal
`ProviderCompletion` carries the usage atomically with its assistant message
and continuation; `AgentEvent::Completed` aggregates those values across tool
and compaction rounds. Implement `complete_with_usage` when a custom provider
has reliable response counts, while existing `complete` implementations
continue to work with unknown usage.
Inline documents use `Document::try_new(mime, filename, bytes)` and
`UserMessage::with_documents`. They stay in the application-owned conversation
as validated bytes; a provider must opt in to each MIME type with
`Provider::supports_document` before an agent submits them.
`Provider::compact` is a default method: it sends the dropped prefix (including
any earlier summary) to `complete_once` and returns a new conversation that keeps
the system prompt, the latest user plus later steers, and a recent protocol-safe
tail.

## License

MIT

## Automatic compaction and durable context

For a long-running agent, use `context_from_history` and `run_turn_context` rather
than the fresh-context `run_turn` convenience API. `AgentContext` owns active
messages and their matching provider state; `Conversation` remains portable data.

```rust,ignore
let mut active = agent.context_from_history(&model, "provider/model/account", &history);
let mut updates = agent.run_turn_context(
    &mut active, &model, user, &control, Some(CompactionPolicy::default()),
);
while let Some(update) = updates.next().await {
    let update = update?;
    // Append semantic user/assistant/tool events to your original transcript.
    // Save that transcript and update.checkpoint atomically BEFORE polling again.
    persist(update)?;
}
```

The host supplies a stable, non-secret binding for the selected provider, model,
endpoint and account. A checkpoint carries a versioned hash cursor into the
original transcript. `restore_context` verifies that prefix, accepts an appended
suffix, and validates provider compatibility; edited or incompatible history
requires an explicit rebase from the original transcript. Exporting a checkpoint
requires only that the provider's continuation support Serde. Provider state must
never contain login credentials. The old split `run_turn_in` API remains usable,
but marks its context ineligible for canonical execution or checkpoint export;
rebase with `context_from_history` when migrating it.

`CompactionPolicy` uses the provider's effective input budget unless explicitly
overridden. Unknown limits do not invent a context window. The core checks before
model requests, after completed tool batches, and on explicit compact requests.
It preserves recent input and tool-call/result pairing, picks up steering arriving
during compaction, and permits one recovery after a typed context-overflow error.
A failed/cancelled replacement never replaces accepted context. A reduction that
still cannot fit fails instead of issuing repeated requests or deleting history.
Large portable transcripts are summarized in bounded, protocol-safe chunks;
single oversized exchanges need an explicit reduction by the caller.

Providers implement `compact_context` to return replacement messages, authoritative
provider state, and usage together. The default is a portable summary. Adapters
with opaque history must override it, and must never summarize only the visible
messages while silently discarding their hidden state. Explicitly unsupported
native compaction can fall back to the portable path only while the full source
is available. Authorization, quota and network failures do not trigger that fallback.
The same maintenance operation serves manual and automatic compaction.

A pull-stream checkpoint is a persistence boundary, not an exactly-once tool
executor. Persist tool intent before dispatch. A crash after dispatch and before a
result leaves an uncertain outcome; reconcile it or use explicit tool idempotency,
never replay the tool merely because its receipt is missing.
