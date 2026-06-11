use std::collections::HashMap;
use std::sync::Arc;

use crate::{Tool, ToolCall, ToolExecutorError, ToolResult, ToolSpec};

#[derive(Default)]
pub struct ToolExecutor {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.spec().name().to_owned(), tool);
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut specs = self
            .tools
            .values()
            .map(|tool| tool.spec())
            .collect::<Vec<_>>();
        specs.sort_by(|left, right| left.name().cmp(right.name()));
        specs
    }

    pub async fn execute(&self, call: &ToolCall) -> ToolResult {
        let Some(tool) = self.tools.get(call.name()) else {
            return ToolResult::failure(
                call.id(),
                ToolExecutorError::UnknownTool(call.name().to_owned()),
            );
        };

        match tool.execute(call.arguments()).await {
            Ok(value) => ToolResult::success(call.id(), value),
            Err(error) => ToolResult::failure(call.id(), ToolExecutorError::Tool(error)),
        }
    }
}
