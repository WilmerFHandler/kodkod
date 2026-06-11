use std::future::{Future, ready};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use serde_json::{Value, json};

use crate::{
    Agent, AssistantMessage, Conversation, Message, Provider, ProviderError, Tool, ToolCall,
    ToolError, ToolExecutor, ToolExecutorError, ToolFuture, ToolResult, ToolSpec,
};

struct EchoProvider;

impl Provider for EchoProvider {
    fn complete(
        &self,
        conversation: &Conversation,
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
        let prompt = conversation.messages().iter().rev().find_map(|message| {
            if let Message::User(message) = message {
                Some(message.content().to_owned())
            } else {
                None
            }
        });

        ready(Ok(AssistantMessage::new(
            prompt.unwrap_or_else(|| "no prompt".to_owned()),
        )))
    }
}

struct EchoTool;

impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "echo",
            "Returns the provided arguments unchanged.",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            }),
        )
    }

    fn execute<'a>(&'a self, arguments: &'a Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(arguments.clone()) })
    }
}

struct FailingTool;

impl Tool for FailingTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("fail", "Always fails.", json!({ "type": "object" }))
    }

    fn execute<'a>(&'a self, _arguments: &'a Value) -> ToolFuture<'a> {
        Box::pin(async { Err(ToolError::new("boom")) })
    }
}

#[test]
fn conversation_tracks_messages() {
    let mut conversation = Conversation::new().with_system_prompt("Be concise.");

    conversation.push_user_message("hello");
    conversation.push_assistant_message(AssistantMessage::new("hi"));

    assert_eq!(conversation.system_prompt(), Some("Be concise."));
    assert_eq!(conversation.messages().len(), 2);
    assert!(matches!(
        &conversation.messages()[0],
        Message::User(message) if message.content() == "hello"
    ));
    assert!(matches!(
        &conversation.messages()[1],
        Message::Assistant(message) if message.content() == "hi"
    ));
}

#[test]
fn agent_appends_user_and_assistant_messages() {
    let agent = Agent::new(EchoProvider);
    let mut conversation = Conversation::new();

    let response = block_on(agent.run(&mut conversation, "hello")).unwrap();

    assert_eq!(response.content(), "hello");
    assert_eq!(conversation.messages().len(), 2);
    assert!(matches!(
        &conversation.messages()[0],
        Message::User(message) if message.content() == "hello"
    ));
    assert!(matches!(
        &conversation.messages()[1],
        Message::Assistant(message) if message.content() == "hello"
    ));
}

#[test]
fn assistant_messages_can_include_tool_calls() {
    let call = ToolCall::new("call_1", "echo", json!({ "value": "hello" }));
    let message = AssistantMessage::new("").with_tool_calls(vec![call.clone()]);

    assert_eq!(message.tool_calls(), &[call]);
}

#[test]
fn conversation_tracks_tool_results() {
    let mut conversation = Conversation::new();
    let result = ToolResult::success("call_1", json!({ "value": "hello" }));

    conversation.push_tool_result(result.clone());

    assert!(matches!(
        &conversation.messages()[0],
        Message::ToolResult(message) if message == &result
    ));
}

#[test]
fn tool_executor_registers_and_executes_tools() {
    let mut executor = ToolExecutor::new();
    executor.register(Arc::new(EchoTool));

    let call = ToolCall::new("call_1", "echo", json!({ "value": "hello" }));
    let result = block_on(executor.execute(&call));

    assert_eq!(result.tool_call_id(), "call_1");
    assert_eq!(result.result(), &Ok(json!({ "value": "hello" })));
    assert!(executor.has_tool("echo"));
    assert_eq!(executor.specs()[0].name(), "echo");
}

#[test]
fn tool_executor_reports_unknown_tools() {
    let executor = ToolExecutor::new();
    let call = ToolCall::new("call_1", "missing", json!({}));

    let result = block_on(executor.execute(&call));

    assert_eq!(result.tool_call_id(), "call_1");
    assert_eq!(
        result.result(),
        &Err(ToolExecutorError::UnknownTool("missing".to_owned()))
    );
}

#[test]
fn tool_executor_wraps_tool_failures() {
    let mut executor = ToolExecutor::new();
    executor.register(Arc::new(FailingTool));

    let call = ToolCall::new("call_1", "fail", json!({}));
    let result = block_on(executor.execute(&call));

    assert_eq!(result.tool_call_id(), "call_1");
    assert_eq!(
        result.result(),
        &Err(ToolExecutorError::Tool(ToolError::new("boom")))
    );
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);

    loop {
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}
