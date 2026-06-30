use std::future::{Future, ready};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use futures::StreamExt;

use serde_json::{Value, json};

use crate::{
    Agent, AgentError, AgentEvent, AssistantMessage, Conversation, Image, Message, Model, Provider,
    ProviderError, TaskControl, Tool, ToolCall, ToolError, ToolExecutor, ToolExecutorError,
    ToolFuture, ToolResult, ToolResultOutcome, ToolSpec, UserMessage,
};

struct EchoProvider;

impl Provider for EchoProvider {
    fn complete(
        &self,
        _model: &Model,
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
        _model: &Model,
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
        _model: &Model,
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
        _model: &Model,
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

    conversation.push_user_message(UserMessage::new("hello"));
    conversation.push_message(Message::Assistant(AssistantMessage::new("hi")));

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

    let model = Model::new("echo", "Echo");
    let response = block_on(collect_run(&agent, &mut conversation, "hello", &model)).unwrap();

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

    let model = Model::new("echo", "Echo");
    block_on(collect_run(&agent, &mut conversation, "hello", &model)).unwrap();

    assert_eq!(*seen_tool_names.lock().unwrap(), vec!["echo"]);
}

#[test]
fn agent_executes_tool_calls_until_final_response() {
    let provider = ToolCallingProvider::default();
    let calls = provider.calls.clone();
    let agent = Agent::new(provider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = Model::new("echo", "Echo");
    let response = block_on(collect_run(&agent, &mut conversation, "hello", &model)).unwrap();

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

struct SlowEchoTool {
    max_in_flight: Arc<AtomicUsize>,
    in_flight: Arc<AtomicUsize>,
}

impl SlowEchoTool {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let max_in_flight = Arc::new(AtomicUsize::new(0));
        (
            Self {
                max_in_flight: Arc::clone(&max_in_flight),
                in_flight: Arc::new(AtomicUsize::new(0)),
            },
            max_in_flight,
        )
    }
}

impl Tool for SlowEchoTool {
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
        let max_in_flight = Arc::clone(&self.max_in_flight);
        let in_flight = Arc::clone(&self.in_flight);
        Box::pin(async move {
            let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut observed = max_in_flight.load(Ordering::SeqCst);
            while observed < now {
                match max_in_flight.compare_exchange_weak(
                    observed,
                    now,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(current) => observed = current,
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(arguments.clone())
        })
    }
}

struct TwoToolCallsProvider;

impl Provider for TwoToolCallsProvider {
    fn complete(
        &self,
        _model: &Model,
        conversation: &Conversation,
        _tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
        let has_tool_results = conversation
            .messages()
            .iter()
            .filter(|message| matches!(message, Message::ToolResult(_)))
            .count();

        ready(Ok(if has_tool_results == 0 {
            AssistantMessage::new("").with_tool_calls(vec![
                ToolCall::new("call_1", "echo", json!({ "value": "a" })),
                ToolCall::new("call_2", "echo", json!({ "value": "b" })),
            ])
        } else {
            assert_eq!(has_tool_results, 2);
            AssistantMessage::new("done")
        }))
    }
}

#[tokio::test]
async fn agent_executes_multiple_tool_calls_in_parallel() {
    let (slow_tool, max_in_flight) = SlowEchoTool::new();
    let agent = Agent::new(TwoToolCallsProvider).with_tool(Arc::new(slow_tool));
    let mut conversation = Conversation::new();

    let model = Model::new("echo", "Echo");
    let response = collect_run(&agent, &mut conversation, "hello", &model)
        .await
        .unwrap();

    assert_eq!(response.content(), "done");
    assert!(
        max_in_flight.load(Ordering::SeqCst) >= 2,
        "expected at least two tool calls to overlap, got max_in_flight={}",
        max_in_flight.load(Ordering::SeqCst)
    );
}

#[test]
fn run_reports_tool_progress_in_order() {
    let provider = ToolCallingProvider::default();
    let agent = Agent::new(provider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = Model::new("echo", "Echo");
    let events = block_on(collect_events(&agent, &mut conversation, "hello", &model)).unwrap();

    assert!(
        matches!(&events[0], AgentEvent::AssistantReply(message) if message.tool_calls().len() == 1)
    );
    assert!(matches!(&events[1], AgentEvent::ToolStarted(call) if call.name() == "echo"));
    assert!(matches!(
        &events[2],
        AgentEvent::ToolFinished(result) if result.tool_call_id() == "call_1" && result.value().is_some()
    ));
    assert!(matches!(
        &events[3],
        AgentEvent::AssistantReply(message) if message.content() == "done" && message.tool_calls().is_empty()
    ));
    assert!(matches!(
        &events[4],
        AgentEvent::Completed(message) if message.content() == "done"
    ));
}

#[test]
fn agent_stops_after_max_tool_rounds() {
    let agent = Agent::new(AlwaysToolCallingProvider).with_max_tool_rounds(0);
    let mut conversation = Conversation::new();

    let model = Model::new("echo", "Echo");
    let result = block_on(collect_run(&agent, &mut conversation, "hello", &model));

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

    conversation.push_message(Message::ToolResult(result.clone()));

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
    conversation.push_user_message(UserMessage::new("hello"));
    conversation.push_message(Message::Assistant(AssistantMessage::new("").with_tool_calls(vec![
        ToolCall::new("call_1", "echo", json!({ "value": "hello" })),
    ])));
    conversation.push_message(Message::ToolResult(ToolResult::success("call_1", json!({ "value": "hello" }))));
    conversation.push_message(Message::Assistant(AssistantMessage::new("done")));

    let encoded = serde_json::to_string(&conversation).unwrap();
    let decoded: Conversation = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, conversation);
}

#[test]
fn conversation_without_images_strips_image_attachments() {
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("describe this").with_images(vec![Image::new("image/png", vec![0x89, 0x50])]));
    conversation.push_message(Message::Assistant(AssistantMessage::new("ok")));

    let stripped = conversation.without_images();

    assert_eq!(stripped.messages().len(), 2);
    assert!(matches!(
        &stripped.messages()[0],
        Message::User(user) if user.content() == "describe this" && user.images().is_empty()
    ));
    assert!(matches!(
        &stripped.messages()[1],
        Message::Assistant(message) if message.content() == "ok"
    ));
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

async fn collect_events<P: Provider + Sync>(
    agent: &Agent<P>,
    conversation: &mut Conversation,
    prompt: &str,
    model: &Model,
) -> Result<Vec<AgentEvent>, AgentError> {
    conversation.push_user_message(UserMessage::new(prompt));
    let control = TaskControl::new();
    let mut stream = agent.run(conversation, model, &control);
    let mut events = Vec::new();

    while let Some(item) = stream.next().await {
        events.push(item?);
    }

    Ok(events)
}

async fn collect_run<P: Provider + Sync>(
    agent: &Agent<P>,
    conversation: &mut Conversation,
    prompt: &str,
    model: &Model,
) -> Result<AssistantMessage, AgentError> {
    conversation.push_user_message(UserMessage::new(prompt));
    let control = TaskControl::new();
    let mut stream = agent.run(conversation, model, &control);

    while let Some(item) = stream.next().await {
        if let Ok(AgentEvent::Completed(message)) = item {
            return Ok(message);
        }
        item?;
    }

    Err(AgentError::Provider(ProviderError::new(
        "agent stream ended without completion",
    )))
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

#[test]
fn image_in_conversation_json_roundtrips() {
    use crate::{Conversation, Image};
    let mut conversation = Conversation::new();
    conversation
        .push_user_message(UserMessage::new("describe").with_images(vec![Image::new("image/png", vec![0x89, 0x50])]));
    let encoded = serde_json::to_string_pretty(&conversation).unwrap();
    let decoded: Conversation = serde_json::from_str(&encoded).expect(&encoded);
    assert_eq!(decoded, conversation);
}

#[test]
fn steered_message_is_injected_between_rounds() {
    // On round 1 the provider returns a tool call, then the test steers a
    // message before round 2. Round 2 asserts the steer is present in the
    // conversation and replies with it.
    struct SteerAwareProvider {
        calls: Arc<AtomicUsize>,
        saw_steer: Arc<Mutex<bool>>,
    }

    impl Provider for SteerAwareProvider {
        fn complete(
            &self,
            _model: &Model,
            conversation: &Conversation,
            tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
            let round = self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(tools.iter().any(|tool| tool.name() == "echo"));

            let reply = if round == 0 {
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "call_1",
                    "echo",
                    json!({ "value": "hello" }),
                )])
            } else {
                // Round 1: the steered message should now be in the conversation.
                let found = conversation.messages().iter().any(|message| {
                    matches!(message,
                        Message::User(user) if user.content() == "wait, stop")
                });
                *self.saw_steer.lock().unwrap() = found;
                AssistantMessage::new(if found { "ack" } else { "missed" })
            };

            ready(Ok(reply))
        }
    }

    let saw_steer = Arc::new(Mutex::new(false));
    let provider = SteerAwareProvider {
        calls: Arc::new(AtomicUsize::new(0)),
        saw_steer: saw_steer.clone(),
    };
    let agent = Agent::new(provider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = Model::new("echo", "Echo");
    conversation.push_user_message(UserMessage::new("go"));
    let control = TaskControl::new();
    let mut stream = agent.run(&mut conversation, &model, &control);

    // Round 1: assistant replies with a tool call.
    drain_until_assistant_reply(&mut stream);
    drain_until_tool_finished(&mut stream);

    // Now queue a steer before round 2 begins.
    control.steer(UserMessage::new("wait, stop"));

    // Round 2: the loop drains the steer, appends it, then calls the provider.
    let mut final_message = None;
    while let Some(item) = block_on(stream.next()) {
        if let AgentEvent::Completed(message) = item.unwrap() {
            final_message = Some(message);
            break;
        }
    }

    assert_eq!(final_message.unwrap().content(), "ack");
    assert!(*saw_steer.lock().unwrap(), "provider did not see the steer");
}

#[test]
fn steer_events_are_emitted_in_order() {
    // A provider that loops through a tool call each round up to a max,
    // allowing multiple steers to be queued and drained across boundaries.
    struct LoopingProvider {
        calls: Arc<AtomicUsize>,
        rounds: usize,
    }

    impl Provider for LoopingProvider {
        fn complete(
            &self,
            _model: &Model,
            _conversation: &Conversation,
            tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
            let round = self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(tools.iter().any(|tool| tool.name() == "echo"));
            ready(Ok(if round + 1 >= self.rounds {
                AssistantMessage::new("done")
            } else {
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "call_1",
                    "echo",
                    json!({}),
                )])
            }))
        }
    }

    let agent = Agent::new(LoopingProvider {
        calls: Arc::new(AtomicUsize::new(0)),
        rounds: 3,
    })
    .with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = Model::new("echo", "Echo");
    conversation.push_user_message(UserMessage::new("start"));
    let control = TaskControl::new();

    // Collect every event from the run while queuing steers between rounds.
    // The stream borrows `conversation` mutably, so drive it to completion and
    // drop it before inspecting the conversation.
    let events = {
        let mut stream = agent.run(&mut conversation, &model, &control);
        // Round 1: tool call then tool result.
        block_on(stream.next()).unwrap().unwrap(); // AssistantReply
        block_on(stream.next()).unwrap().unwrap(); // ToolStarted
        block_on(stream.next()).unwrap().unwrap(); // ToolFinished
        // Queue two steers before round 2 begins.
        control.steer(UserMessage::new("one"));
        control.steer(UserMessage::new("two"));
        // Drain to completion: round 2 (with the steers) and round 3 (final).
        let mut collected = Vec::new();
        while let Some(item) = block_on(stream.next()) {
            collected.push(item.unwrap());
        }
        collected
    };

    // The two steers appear as Steered events, in queue order.
    let steered: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Steered(user) => Some(user.content()),
            _ => None,
        })
        .collect();
    assert_eq!(steered, vec!["one", "two"]);

    // The conversation holds the steered messages.
    let user_contents: Vec<&str> = conversation
        .messages()
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(user.content()),
            _ => None,
        })
        .collect();
    assert!(user_contents.contains(&"one"));
    assert!(user_contents.contains(&"two"));
}

#[test]
fn drain_pending_steers_empties_the_queue() {
    let control = TaskControl::new();
    control.steer(UserMessage::new("a"));
    control.steer(UserMessage::new("b"));

    let drained = control.drain_pending_steers();
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].content(), "a");
    assert_eq!(drained[1].content(), "b");
    // Second drain is empty: steers are consumed once.
    assert!(control.drain_pending_steers().is_empty());
}

#[test]
fn cancel_takes_precedence_over_steer() {
    // If a steer and a cancel are both pending, the cancel check runs first
    // and the turn ends without consuming the steer.
    struct IdleProvider;
    impl Provider for IdleProvider {
        fn complete(
            &self,
            _model: &Model,
            _conversation: &Conversation,
            _tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
            ready(Ok(AssistantMessage::new("").with_tool_calls(vec![
                ToolCall::new("call_1", "echo", json!({})),
            ])))
        }
    }

    let agent = Agent::new(IdleProvider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = Model::new("echo", "Echo");
    conversation.push_user_message(UserMessage::new("start"));
    let control = TaskControl::new();
    let mut stream = agent.run(&mut conversation, &model, &control);

    // Round 1 returns a tool call.
    drain_until_tool_finished(&mut stream);

    // Queue both a steer and a cancel.
    control.steer(UserMessage::new("ignored"));
    control.cancel();

    // The loop should cancel before processing the steer.
    let mut cancelled = false;
    while let Some(item) = block_on(stream.next()) {
        match item {
            Err(AgentError::Cancelled) => {
                cancelled = true;
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    assert!(cancelled);
}

/// Pull events from the stream until (and including) the first assistant reply.
fn drain_until_assistant_reply(stream: &mut crate::Task<'_>) {
    while let Some(item) = block_on(stream.next()) {
        if let AgentEvent::AssistantReply(_) = item.unwrap() {
            return;
        }
    }
    panic!("stream ended before an assistant reply");
}

/// Pull events until a ToolFinished has been emitted (completing a tool round).
fn drain_until_tool_finished(stream: &mut crate::Task<'_>) {
    while let Some(item) = block_on(stream.next()) {
        if let AgentEvent::ToolFinished(_) = item.unwrap() {
            return;
        }
    }
    panic!("stream ended before a tool finished");
}
