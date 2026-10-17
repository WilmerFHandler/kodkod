//! A small provider-agnostic agent loop with tool calling support.
//!
//! `kodkod-core` provides the core types needed to run an agent against any
//! backend that implements [`Provider`]. The crate keeps provider integration,
//! conversation state, and tool execution separate so applications can bring
//! their own model backend and tool implementations.

pub mod agent;
pub mod conversation;
pub mod message;
pub mod provider;
pub mod request_diagnostics;
pub mod retry;
pub mod tool;
pub mod turns;

mod compact;
mod context;
mod estimate;
mod usage;

pub use agent::{Agent, AgentError, AgentEvent, SteerError, Task, TaskControl};
pub use compact::{
    CompactError, CompactOptions, CompactionResult, DEFAULT_KEEP_TAIL_TOKENS, TurnCompaction,
};
pub use context::{
    AgentContext, AgentUpdate, CompactionPolicy, ContextBudget, ContextCheckpoint,
    ContextCompaction, ContextCursor, ContextMaintenanceState, ContextMeasurement, ContextTask,
};
pub use conversation::Conversation;
pub use message::{
    AssistantMessage, Document, DocumentError, Image, Message, SystemMessage, UserMessage,
};
pub use provider::{Provider, ProviderCompletion, ProviderEvent, ProviderStream};
pub use request_diagnostics::{
    RequestCacheDiagnostics, RequestCacheReason, RequestCacheType, RequestDiagnostic,
    RequestDiagnosticEvent, RequestFailureKind, RequestFingerprint, RequestIdentity,
    RequestInputPrefixFingerprint, RequestMetadata, RequestObserver, RequestOperation,
    RequestOptionClass, RequestProfile, RequestPromptFingerprints, RequestUsage,
};
pub use retry::{RetryPolicy, RetryProvider, Retryable};
pub use tool::{
    Tool, ToolCall, ToolError, ToolExecutor, ToolExecutorError, ToolFuture, ToolOutput, ToolResult,
    ToolResultOutcome, ToolSpec,
};
pub use turns::{Turn, TurnIter, Turns, turns};
pub use usage::{ContextEstimate, ContextEstimateProvenance, TokenUsage};

#[cfg(test)]
mod tests;
