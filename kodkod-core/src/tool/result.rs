use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ToolExecutorError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    tool_call_id: String,
    outcome: ToolResultOutcome,
}

impl ToolResult {
    pub fn new(tool_call_id: impl Into<String>, outcome: ToolResultOutcome) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            outcome,
        }
    }

    pub fn success(tool_call_id: impl Into<String>, value: Value) -> Self {
        Self::new(tool_call_id, ToolResultOutcome::Success(value))
    }

    pub fn failure(tool_call_id: impl Into<String>, error: ToolExecutorError) -> Self {
        Self::new(tool_call_id, ToolResultOutcome::Error(error))
    }

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    pub fn outcome(&self) -> &ToolResultOutcome {
        &self.outcome
    }

    pub fn value(&self) -> Option<&Value> {
        match &self.outcome {
            ToolResultOutcome::Success(value) => Some(value),
            ToolResultOutcome::Error(_) => None,
        }
    }

    pub fn error(&self) -> Option<&ToolExecutorError> {
        match &self.outcome {
            ToolResultOutcome::Success(_) => None,
            ToolResultOutcome::Error(error) => Some(error),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ToolResultOutcome {
    #[serde(rename = "success")]
    Success(Value),
    #[serde(rename = "error")]
    Error(ToolExecutorError),
}
