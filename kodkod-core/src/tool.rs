pub mod call;
pub mod error;
pub mod executor;
pub mod result;
pub mod spec;

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

pub use call::ToolCall;
pub use error::{ToolError, ToolExecutorError};
pub use executor::ToolExecutor;
pub use result::{ToolOutput, ToolResult, ToolResultOutcome};
pub use spec::ToolSpec;

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>>;

pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    /// Whether this tool can only produce a meaningful result for vision models.
    fn requires_vision(&self) -> bool {
        false
    }

    /// Whether this tool may only be used by models with explicit computer-use support.
    fn requires_computer_use(&self) -> bool {
        false
    }

    fn execute<'a>(&'a self, arguments: &'a Value) -> ToolFuture<'a>;
}
