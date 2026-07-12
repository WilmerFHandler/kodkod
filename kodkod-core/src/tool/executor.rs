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
        self.specs_for_vision(true)
    }

    pub fn specs_for_vision(&self, vision_enabled: bool) -> Vec<ToolSpec> {
        let mut specs = self
            .tools
            .values()
            .filter(|tool| vision_enabled || !tool.requires_vision())
            .map(|tool| tool.spec())
            .collect::<Vec<_>>();
        specs.sort_by(|left, right| left.name().cmp(right.name()));
        specs
    }

    pub async fn execute(&self, call: &ToolCall) -> ToolResult {
        self.execute_for_vision(call, true).await
    }

    pub async fn execute_for_vision(&self, call: &ToolCall, vision_enabled: bool) -> ToolResult {
        let Some(tool) = self.tools.get(call.name()) else {
            return ToolResult::failure(
                call.id(),
                ToolExecutorError::UnknownTool(call.name().to_owned()),
            );
        };

        if tool.requires_vision() && !vision_enabled {
            return ToolResult::failure(
                call.id(),
                ToolExecutorError::Tool(crate::ToolError::new(format!(
                    "tool '{}' requires a vision-capable model",
                    call.name()
                ))),
            );
        }

        match tool.execute(call.arguments()).await {
            Ok(value) => ToolResult::success(call.id(), value),
            Err(error) => ToolResult::failure(call.id(), ToolExecutorError::Tool(error)),
        }
    }
}
