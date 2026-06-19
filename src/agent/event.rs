use crate::{AssistantMessage, ToolCall, ToolResult};

/// Incremental progress from a running agent turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    /// The provider returned an assistant message for the current round.
    AssistantReply(AssistantMessage),
    /// A tool call is about to execute.
    ToolStarted(ToolCall),
    /// A tool call finished.
    ToolFinished(ToolResult),
    /// The turn completed with a final assistant message (no pending tool calls).
    Completed(AssistantMessage),
}
