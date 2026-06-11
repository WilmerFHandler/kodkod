use serde_json::Value;

use crate::ToolExecutorError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    tool_call_id: String,
    result: Result<Value, ToolExecutorError>,
}

impl ToolResult {
    pub fn new(tool_call_id: impl Into<String>, result: Result<Value, ToolExecutorError>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            result,
        }
    }

    pub fn success(tool_call_id: impl Into<String>, value: Value) -> Self {
        Self::new(tool_call_id, Ok(value))
    }

    pub fn error(tool_call_id: impl Into<String>, error: ToolExecutorError) -> Self {
        Self::new(tool_call_id, Err(error))
    }

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    pub fn result(&self) -> &Result<Value, ToolExecutorError> {
        &self.result
    }
}
