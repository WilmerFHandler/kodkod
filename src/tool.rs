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
pub use result::ToolResult;
pub use spec::ToolSpec;

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + 'a>>;

pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    fn execute<'a>(&'a self, arguments: &'a Value) -> ToolFuture<'a>;
}
