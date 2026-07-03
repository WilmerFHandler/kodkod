//! A small provider-agnostic agent loop with tool calling support.
//!
//! `lynx-agent` provides the core types needed to run an agent against any
//! backend that implements [`Provider`]. The crate keeps provider integration,
//! conversation state, and tool execution separate so applications can bring
//! their own model backend and tool implementations.
//!
//! Enable the `openai-compatible` feature for
//! [`complete_openai_compatible`](provider::complete_openai_compatible).
//!
pub mod agent;
pub mod conversation;
pub mod message;
pub mod provider;
pub mod tool;
pub mod turns;

pub use agent::{Agent, AgentError, AgentEvent, Task, TaskControl};
pub use conversation::Conversation;
pub use message::{AssistantMessage, Image, Message, SystemMessage, UserMessage};
#[cfg(feature = "openai-compatible")]
pub use provider::complete_openai_compatible;
pub use provider::{
    Provider, ProviderError, ProviderErrorKind, RetryPolicy, RetryProvider, Retryable,
};
pub use tool::{
    Tool, ToolCall, ToolError, ToolExecutor, ToolExecutorError, ToolFuture, ToolResult,
    ToolResultOutcome, ToolSpec,
};
pub use turns::{Turn, TurnIter, Turns, turns};

#[cfg(test)]
mod tests;
