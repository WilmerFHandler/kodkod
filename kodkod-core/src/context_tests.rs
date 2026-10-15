use super::*;
use crate::*;
use futures::StreamExt;
use std::{
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Serialize, Deserialize)]
struct State {
    rounds: usize,
}
#[derive(Default)]
struct Fixture {
    requests: Mutex<Vec<Conversation>>,
    compactions: AtomicUsize,
    effects: AtomicUsize,
    overflow: usize,
    tool: bool,
    hang: bool,
    no_progress: bool,
    steer: Option<TaskControl>,
    reject: bool,
}
#[derive(Clone)]
struct Backend(Arc<Fixture>);
impl Provider for Backend {
    type Model = ();
    type Error = io::Error;
    type Continuation = State;
    fn supports_vision(&self, _: &()) -> bool {
        true
    }
    fn create_continuation(&self, _: &()) -> State {
        State { rounds: 0 }
    }
    fn validate_context(&self, _: &(), _: &State) -> Result<(), String> {
        if self.0.reject {
            Err("incompatible fixture".into())
        } else {
            Ok(())
        }
    }
    fn is_context_overflow(&self, e: &io::Error) -> bool {
        e.kind() == io::ErrorKind::InvalidInput
    }
    async fn complete(
        &self,
        c: &State,
        _: &(),
        history: &Conversation,
        _: &[ToolSpec],
    ) -> Result<(AssistantMessage, State), io::Error> {
        let mut requests = self.0.requests.lock().unwrap();
        requests.push(history.clone());
        if requests.len() <= self.0.overflow {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "context length",
            ));
        }
        let reply = if self.0.tool
            && c.rounds == 0
            && !history
                .messages()
                .iter()
                .any(|m| matches!(m, Message::ToolResult(_)))
        {
            AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                "call",
                "effect",
                serde_json::json!({}),
            )])
        } else {
            AssistantMessage::new("done")
        };
        Ok((
            reply,
            State {
                rounds: c.rounds + 1,
            },
        ))
    }
    async fn compact_context(
        &self,
        _: &State,
        _: &(),
        history: &Conversation,
        _: CompactOptions,
    ) -> Result<ContextCompaction<State>, CompactError<io::Error>> {
        self.0.compactions.fetch_add(1, Ordering::SeqCst);
        if self.0.hang {
            futures::future::pending::<()>().await;
        }
        if let Some(control) = &self.0.steer {
            control.steer(UserMessage::new("new steering")).unwrap();
        }
        let conversation = if self.0.no_progress {
            history.clone()
        } else {
            let mut c = Conversation::new();
            c.set_system_prompt(history.system_prompt());
            c.push_message(Message::Assistant(AssistantMessage::new("checkpoint")));
            if let Some(Message::User(u)) = history
                .messages()
                .iter()
                .rev()
                .find(|m| matches!(m, Message::User(_)))
            {
                c.push_user_message(u.clone());
            }
            c
        };
        Ok(ContextCompaction {
            conversation,
            continuation: State { rounds: 1 },
            usage: TokenUsage::zero(),
        })
    }
}
struct EffectTool(Arc<Fixture>);
impl Tool for EffectTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "effect",
            "fixture effect",
            serde_json::json!({"type":"object"}),
        )
    }
    fn execute<'a>(&'a self, _: &'a serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move {
            self.0.effects.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutput::from(
                serde_json::json!({"text":"x".repeat(10000)}),
            ))
        })
    }
}
fn history() -> Conversation {
    let mut c = Conversation::new();
    c.push_user_message(UserMessage::new("old ".repeat(2000)));
    c.push_message(Message::Assistant(AssistantMessage::new("old answer")));
    c
}
fn policy(limit: u64) -> CompactionPolicy {
    CompactionPolicy {
        budget: Some(ContextBudget {
            max_input_tokens: limit,
        }),
        keep_tail_tokens: 10,
    }
}

#[tokio::test]
async fn automatic_compaction_keeps_original_history_and_restores_nonclone_provider_state() {
    let fixture = Arc::new(Fixture::default());
    let agent = Agent::new(Backend(fixture.clone()));
    let mut original = history();
    let mut context = agent.context_from_history(&(), "account/model", &original);
    let control = TaskControl::new();
    original.push_user_message(UserMessage::new("continue"));
    let mut task = agent.run_turn_context(
        &mut context,
        &(),
        UserMessage::new("continue"),
        &control,
        Some(policy(1000)),
    );
    let mut saved = None;
    while let Some(update) = task.next().await {
        let update = update.unwrap();
        if let AgentEvent::AssistantReply(m, _) = &update.event {
            original.push_message(Message::Assistant(m.clone()));
        }
        if let Some(c) = update.checkpoint {
            assert_eq!(c.cursor, ContextCursor::for_history(&original));
            saved = Some(c);
        }
    }
    drop(task);
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);
    assert_eq!(original.messages().len(), 4);
    assert!(context.conversation().estimate_tokens(&[]) < original.estimate_tokens(&[]));
    let value = serde_json::to_vec(&saved.unwrap()).unwrap();
    let checkpoint: ContextCheckpoint<State> = serde_json::from_slice(&value).unwrap();
    let mut restored = agent
        .restore_context(&(), "account/model", &original, checkpoint.clone())
        .unwrap();
    assert!(
        agent
            .restore_context(&(), "other-account", &original, checkpoint.clone())
            .is_err()
    );
    let mut edited = Conversation::new();
    edited.push_user_message(UserMessage::new("edited"));
    assert!(
        agent
            .restore_context(&(), "account/model", &edited, checkpoint.clone())
            .is_err()
    );
    let mut resumed = agent
        .run_turn_context(
            &mut restored,
            &(),
            UserMessage::new("next"),
            &TaskControl::new(),
            Some(policy(1000)),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(resumed.pop().unwrap().is_ok());
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);
    original.push_user_message(UserMessage::new("accepted while stopping"));
    let restored = agent
        .restore_context(&(), "account/model", &original, checkpoint)
        .unwrap();
    assert_eq!(
        restored.checkpoint().unwrap().cursor,
        ContextCursor::for_history(&original)
    );
}

#[tokio::test]
async fn compaction_between_tool_rounds_does_not_repeat_effects() {
    let fixture = Arc::new(Fixture {
        tool: true,
        ..Default::default()
    });
    let agent =
        Agent::new(Backend(fixture.clone())).with_tool(Arc::new(EffectTool(fixture.clone())));
    let mut context = agent.context_from_history(&(), "fixture", &Conversation::new());
    let events = agent
        .run_turn_context(
            &mut context,
            &(),
            UserMessage::new("go"),
            &TaskControl::new(),
            Some(policy(1500)),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(fixture.effects.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn overflow_recovery_is_bounded_and_does_not_rerun_tools() {
    let fixture = Arc::new(Fixture {
        overflow: usize::MAX,
        ..Default::default()
    });
    let agent = Agent::new(Backend(fixture.clone()));
    let mut context = agent.context_from_history(&(), "fixture", &history());
    let events = agent
        .run_turn_context(
            &mut context,
            &(),
            UserMessage::new("go"),
            &TaskControl::new(),
            Some(policy(100000)),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.last(), Some(Err(AgentError::Provider(_)))));
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn cancellation_during_compaction_keeps_the_previous_active_input() {
    let fixture = Arc::new(Fixture {
        hang: true,
        ..Default::default()
    });
    let agent = Agent::new(Backend(fixture));
    let original = history();
    let mut context = agent.context_from_history(&(), "fixture", &original);
    let control = TaskControl::new();
    let mut events = agent.run_turn_context(
        &mut context,
        &(),
        UserMessage::new("go"),
        &control,
        Some(policy(1000)),
    );
    assert!(matches!(
        events.next().await.unwrap().unwrap().event,
        AgentEvent::CompactionStarted
    ));
    control.cancel();
    assert!(matches!(
        events.next().await,
        Some(Err(AgentError::Cancelled))
    ));
    drop(events);
    assert_eq!(&context.conversation().messages()[..2], original.messages());
    assert_eq!(context.conversation().messages().len(), 3);
}

#[tokio::test]
async fn steering_arriving_during_compaction_is_in_the_next_request() {
    let control = TaskControl::new();
    let fixture = Arc::new(Fixture {
        steer: Some(control.clone()),
        ..Default::default()
    });
    let agent = Agent::new(Backend(fixture.clone()));
    let mut context = agent.context_from_history(&(), "fixture", &history());
    let events = agent
        .run_turn_context(
            &mut context,
            &(),
            UserMessage::new("go"),
            &control,
            Some(policy(1000)),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert!(
        fixture.requests.lock().unwrap()[0]
            .messages()
            .iter()
            .any(|m| matches!(m,Message::User(u) if u.content()=="new steering"))
    );
}

#[tokio::test]
async fn ineffective_compaction_and_oversized_retained_input_stop_before_inference() {
    let fixture = Arc::new(Fixture {
        no_progress: true,
        ..Default::default()
    });
    let agent = Agent::new(Backend(fixture.clone()));
    let mut context = agent.context_from_history(&(), "fixture", &history());
    let events = agent
        .run_turn_context(
            &mut context,
            &(),
            UserMessage::new("go"),
            &TaskControl::new(),
            Some(policy(1000)),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        events.last(),
        Some(Err(AgentError::ContextMaintenance(_)))
    ));
    assert!(fixture.requests.lock().unwrap().is_empty());
    let fixture = Arc::new(Fixture::default());
    let agent = Agent::new(Backend(fixture.clone()));
    let mut context = agent.context_from_history(&(), "fixture", &history());
    let events = agent
        .run_turn_context(
            &mut context,
            &(),
            UserMessage::new("new ".repeat(2000)),
            &TaskControl::new(),
            Some(policy(1000)),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        events.last(),
        Some(Err(AgentError::ContextMaintenance(_)))
    ));
    assert!(fixture.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn legacy_split_context_cannot_be_exported_or_used_as_a_canonical_checkpoint() {
    let agent = Agent::new(Backend(Arc::new(Fixture::default())));
    let mut context = agent.new_context(&());
    let mut history = Conversation::new();
    let events = agent
        .run_turn_in(
            &mut history,
            &(),
            UserMessage::new("go"),
            &mut context,
            &TaskControl::new(),
            None,
        )
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert!(context.checkpoint().is_err());
    let events = agent
        .run_turn_context(
            &mut context,
            &(),
            UserMessage::new("next"),
            &TaskControl::new(),
            Some(policy(1000)),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        events.last(),
        Some(Err(AgentError::ContextMaintenance(_)))
    ));
}

#[tokio::test]
async fn incompatible_live_context_is_rejected_before_input_is_accepted() {
    let agent = Agent::new(Backend(Arc::new(Fixture {
        reject: true,
        ..Default::default()
    })));
    let mut context = agent.context_from_history(&(), "fixture", &history());
    let previous = context.checkpoint().unwrap();
    let events = agent
        .run_turn_context(
            &mut context,
            &(),
            UserMessage::new("go"),
            &TaskControl::new(),
            Some(policy(1000)),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        events.last(),
        Some(Err(AgentError::ContextMaintenance(_)))
    ));
    assert_eq!(context.checkpoint().unwrap().cursor, previous.cursor);
}

#[derive(Clone)]
struct BoundedSummary(Arc<Mutex<Vec<Conversation>>>);
impl Provider for BoundedSummary {
    type Model = ();
    type Error = io::Error;
    type Continuation = ();
    fn create_continuation(&self, _: &()) {}
    fn supports_vision(&self, _: &()) -> bool {
        false
    }
    fn context_budget(&self, _: &()) -> Option<ContextBudget> {
        Some(ContextBudget {
            max_input_tokens: 1800,
        })
    }
    async fn complete(
        &self,
        _: &(),
        _: &(),
        history: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, ()), io::Error> {
        assert!(
            history.estimate_tokens(tools) < 1800,
            "summary request must fit"
        );
        assert!(tools.iter().any(|t| t.name() == "summarize"));
        self.0.lock().unwrap().push(history.clone());
        Ok((
            AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                "summary",
                "summarize",
                serde_json::json!({"summary":"All constraints and completed work retained"}),
            )]),
            (),
        ))
    }
}
#[tokio::test]
async fn a_long_rebased_transcript_is_summarized_in_bounded_chunks_without_omitting_messages() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let provider = BoundedSummary(calls.clone());
    let mut history = Conversation::new();
    for index in 0..20 {
        history.push_user_message(UserMessage::new(format!(
            "unique-source-{index} {}",
            "detail ".repeat(180)
        )));
        history.push_message(Message::Assistant(AssistantMessage::new("acknowledged")));
    }
    let compacted = provider
        .compact_with_usage(
            &(),
            &history,
            CompactOptions {
                keep_tail_tokens: 5,
            },
        )
        .await
        .unwrap();
    assert!(compacted.conversation.estimate_tokens(&[]) < history.estimate_tokens(&[]));
    let calls = calls.lock().unwrap();
    assert!(calls.len() > 1);
    for index in 0..19 {
        assert!(calls.iter().any(|c|c.messages().iter().any(|m|matches!(m,Message::User(u) if u.content().contains(&format!("unique-source-{index} "))))),"old input {index} must be summarized, not dropped");
    }
}
