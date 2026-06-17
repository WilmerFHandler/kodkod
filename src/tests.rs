use std::future::{Future, ready};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use serde_json::{Value, json};

use crate::{
    Agent, AgentError, AssistantMessage, Conversation, Message, Provider, ProviderError, Tool,
    ToolCall, ToolError, ToolExecutor, ToolExecutorError, ToolFuture, ToolResult,
    ToolResultOutcome, ToolSpec, UserMessage,
};

struct EchoProvider;

impl Provider for EchoProvider {
    fn complete(
        &self,
        conversation: &Conversation,
        _tools: &[ToolSpec],
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

#[test]
fn provider_models_default_to_empty() {
    assert!(EchoProvider.models().is_empty());
}

#[derive(Default)]
struct RecordingProvider {
    seen_tool_names: Arc<Mutex<Vec<String>>>,
}

impl Provider for RecordingProvider {
    fn complete(
        &self,
        _conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
        self.seen_tool_names
            .lock()
            .unwrap()
            .extend(tools.iter().map(|tool| tool.name().to_owned()));

        ready(Ok(AssistantMessage::new("done")))
    }
}

#[derive(Default)]
struct ToolCallingProvider {
    calls: Arc<AtomicUsize>,
}

impl Provider for ToolCallingProvider {
    fn complete(
        &self,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
        let call_count = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(tools.iter().any(|tool| tool.name() == "echo"));

        let has_tool_result = conversation
            .messages()
            .iter()
            .any(|message| matches!(message, Message::ToolResult(_)));

        ready(Ok(if call_count == 0 {
            AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                "call_1",
                "echo",
                json!({ "value": "hello" }),
            )])
        } else {
            assert!(has_tool_result);
            AssistantMessage::new("done")
        }))
    }
}

struct AlwaysToolCallingProvider;

impl Provider for AlwaysToolCallingProvider {
    fn complete(
        &self,
        _conversation: &Conversation,
        _tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
        ready(Ok(AssistantMessage::new("").with_tool_calls(vec![
            ToolCall::new("call_1", "missing", json!({})),
        ])))
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
fn agent_passes_registered_tool_specs_to_provider() {
    let provider = RecordingProvider::default();
    let seen_tool_names = provider.seen_tool_names.clone();
    let agent = Agent::new(provider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    block_on(agent.run(&mut conversation, "hello")).unwrap();

    assert_eq!(*seen_tool_names.lock().unwrap(), vec!["echo"]);
}

#[test]
fn agent_executes_tool_calls_until_final_response() {
    let provider = ToolCallingProvider::default();
    let calls = provider.calls.clone();
    let agent = Agent::new(provider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let response = block_on(agent.run(&mut conversation, "hello")).unwrap();

    assert_eq!(response.content(), "done");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(conversation.messages().len(), 4);
    assert!(matches!(&conversation.messages()[0], Message::User(_)));
    assert!(matches!(
        &conversation.messages()[1],
        Message::Assistant(message) if message.tool_calls().len() == 1
    ));
    assert!(matches!(
        &conversation.messages()[2],
        Message::ToolResult(result) if result.value() == Some(&json!({ "value": "hello" }))
    ));
    assert!(matches!(
        &conversation.messages()[3],
        Message::Assistant(message) if message.content() == "done"
    ));
}

#[test]
fn agent_stops_after_max_tool_rounds() {
    let agent = Agent::new(AlwaysToolCallingProvider).with_max_tool_rounds(0);
    let mut conversation = Conversation::new();

    let result = block_on(agent.run(&mut conversation, "hello"));

    assert!(matches!(
        result,
        Err(AgentError::MaxToolRoundsExceeded { max: 0 })
    ));
    assert_eq!(conversation.messages().len(), 2);
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
    assert_eq!(
        result.outcome(),
        &ToolResultOutcome::Success(json!({ "value": "hello" }))
    );
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
        result.outcome(),
        &ToolResultOutcome::Error(ToolExecutorError::UnknownTool("missing".to_owned()))
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
        result.outcome(),
        &ToolResultOutcome::Error(ToolExecutorError::Tool(ToolError::new("boom")))
    );
}

#[test]
fn conversation_round_trips_through_json() {
    let mut conversation = Conversation::new().with_system_prompt("Be concise.");
    conversation.push_user_message("hello");
    conversation.push_assistant_message(AssistantMessage::new("").with_tool_calls(vec![
        ToolCall::new("call_1", "echo", json!({ "value": "hello" })),
    ]));
    conversation.push_tool_result(ToolResult::success("call_1", json!({ "value": "hello" })));
    conversation.push_assistant_message(AssistantMessage::new("done"));

    let encoded = serde_json::to_string(&conversation).unwrap();
    let decoded: Conversation = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, conversation);
}

#[test]
fn messages_serialize_as_role_tagged_objects() {
    assert_eq!(
        serde_json::to_value(Message::User(UserMessage::new("hello"))).unwrap(),
        json!({
            "role": "user",
            "content": "hello"
        })
    );
    assert_eq!(
        serde_json::to_value(Message::Assistant(AssistantMessage::new("hi"))).unwrap(),
        json!({
            "role": "assistant",
            "content": "hi",
            "tool_calls": []
        })
    );
    assert_eq!(
        serde_json::to_value(Message::ToolResult(ToolResult::success(
            "call_1",
            json!({ "value": "hello" })
        )))
        .unwrap(),
        json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "outcome": {
                "type": "success",
                "value": { "value": "hello" }
            }
        })
    );
}

#[test]
fn tool_spec_round_trips_through_json() {
    let spec = EchoTool.spec();

    let encoded = serde_json::to_string(&spec).unwrap();
    let decoded: ToolSpec = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, spec);
}

#[test]
fn tool_result_outcomes_serialize_with_explicit_shape() {
    let success = ToolResult::success("call_1", json!({ "value": "hello" }));
    let error = ToolResult::failure(
        "call_2",
        ToolExecutorError::UnknownTool("missing".to_owned()),
    );

    assert_eq!(
        serde_json::to_value(&success).unwrap(),
        json!({
            "tool_call_id": "call_1",
            "outcome": {
                "type": "success",
                "value": { "value": "hello" }
            }
        })
    );
    assert_eq!(
        serde_json::to_value(&error).unwrap(),
        json!({
            "tool_call_id": "call_2",
            "outcome": {
                "type": "error",
                "value": {
                    "type": "unknown_tool",
                    "value": "missing"
                }
            }
        })
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
