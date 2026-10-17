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
    #[serde(default)]
    opaque: bool,
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
    opaque_compaction: bool,
    compact_input_tokens: Option<u64>,
    measured_input_tokens: Mutex<Option<u64>>,
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
        State {
            rounds: 0,
            opaque: false,
        }
    }
    fn has_opaque_context(&self, continuation: &State) -> bool {
        continuation.opaque
    }
    fn estimate_continuation_tokens(continuation: &State) -> u64 {
        u64::from(continuation.opaque)
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
                opaque: c.opaque,
            },
        ))
    }
    async fn complete_with_usage(
        &self,
        continuation: &State,
        model: &(),
        history: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<ProviderCompletion<State>, io::Error> {
        let (message, next) = self.complete(continuation, model, history, tools).await?;
        let usage = (*self.0.measured_input_tokens.lock().unwrap()).map_or_else(
            TokenUsage::unknown,
            |input_tokens| TokenUsage {
                input_tokens: Some(input_tokens),
                cached_input_tokens: None,
                output_tokens: Some(1),
                reasoning_output_tokens: None,
                total_tokens: Some(input_tokens.saturating_add(1)),
            },
        );
        Ok(ProviderCompletion::new(message, next, usage))
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
            continuation: State {
                rounds: 1,
                opaque: self.0.opaque_compaction,
            },
            usage: self
                .0
                .compact_input_tokens
                .map_or_else(TokenUsage::zero, |input_tokens| TokenUsage {
                    input_tokens: Some(input_tokens),
                    cached_input_tokens: None,
                    output_tokens: Some(10),
                    reasoning_output_tokens: None,
                    total_tokens: Some(input_tokens.saturating_add(10)),
                }),
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
        ..CompactionPolicy::default()
    }
}

#[test]
fn appended_system_context_is_checkpointed_without_advancing_transcript_cursor() {
    let agent = Agent::new(Backend(Arc::new(Fixture::default())));
    let transcript = history();
    let mut context = agent.context_from_history(&(), "fixture", &transcript);
    let original_cursor = context.checkpoint().unwrap().cursor;
    context.append_system_context(SystemMessage::new("latest host snapshot"));
    context.maintenance_state.soft_compaction_after_tokens = Some(900);

    let checkpoint = context.checkpoint().unwrap();
    assert_eq!(checkpoint.cursor, original_cursor);
    assert!(matches!(
        checkpoint.conversation.messages().last(),
        Some(Message::System(message)) if message.content() == "latest host snapshot"
    ));
    let encoded = serde_json::to_value(&checkpoint).unwrap();
    let mut legacy = encoded.clone();
    legacy.as_object_mut().unwrap().remove("maintenance_state");
    let legacy: ContextCheckpoint<State> = serde_json::from_value(legacy).unwrap();
    assert_eq!(legacy.maintenance_state, ContextMaintenanceState::default());

    let mut legacy = encoded.clone();
    legacy
        .get_mut("maintenance_state")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove("awaiting_native_measurement");
    let legacy: ContextCheckpoint<State> = serde_json::from_value(legacy).unwrap();
    assert_eq!(
        legacy.maintenance_state,
        ContextMaintenanceState {
            soft_compaction_after_tokens: Some(900),
            awaiting_native_measurement: false,
        }
    );

    let checkpoint: ContextCheckpoint<State> = serde_json::from_value(encoded).unwrap();
    let restored = agent
        .restore_context(&(), "fixture", &transcript, checkpoint)
        .unwrap();
    assert_eq!(restored.checkpoint().unwrap().cursor, original_cursor);
    assert_eq!(
        restored.maintenance_state.soft_compaction_after_tokens,
        Some(900)
    );
    assert!(matches!(
        restored.conversation().messages().last(),
        Some(Message::System(message)) if message.content() == "latest host snapshot"
    ));
}

#[tokio::test]
async fn failed_soft_maintenance_waits_for_growth_across_checkpoint_restore() {
    let fixture = Arc::new(Fixture {
        no_progress: true,
        ..Default::default()
    });
    let agent = Agent::new(Backend(fixture.clone()));
    let mut transcript = history();
    let mut context = agent.context_from_history(&(), "fixture", &transcript);
    let policy = CompactionPolicy {
        maintenance_trigger_tokens: Some(100),
        maintenance_growth_tokens: 100_000,
        keep_tail_tokens: 10,
        ..CompactionPolicy::default()
    };
    let first_user = UserMessage::new("first");
    let first_control = TaskControl::new();
    let mut task = agent.run_turn_context(
        &mut context,
        &(),
        first_user.clone(),
        &first_control,
        Some(policy),
    );
    let mut saved = None;
    let mut saw_failed_soft_maintenance = false;
    while let Some(update) = task.next().await {
        let update = update.unwrap();
        if matches!(update.event, AgentEvent::CompactionFailed(_)) {
            saw_failed_soft_maintenance = true;
        }
        if let AgentEvent::AssistantReply(message, _) = &update.event {
            transcript.push_message(Message::User(first_user.clone()));
            transcript.push_message(Message::Assistant(message.clone()));
        }
        if let Some(checkpoint) = update.checkpoint {
            saved = Some(checkpoint);
        }
    }
    drop(task);
    assert!(saw_failed_soft_maintenance);
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);
    let checkpoint = saved.unwrap();
    assert!(
        checkpoint
            .maintenance_state
            .soft_compaction_after_tokens
            .is_some()
    );
    let encoded = serde_json::to_vec(&checkpoint).unwrap();
    let checkpoint = serde_json::from_slice(&encoded).unwrap();
    let mut restored = agent
        .restore_context(&(), "fixture", &transcript, checkpoint)
        .unwrap();

    let second_control = TaskControl::new();
    let updates = agent
        .run_turn_context(
            &mut restored,
            &(),
            UserMessage::new("second"),
            &second_control,
            Some(policy),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(updates.iter().all(Result::is_ok));
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn opaque_maintenance_uses_measured_input_as_growth_baseline_after_restore() {
    let fixture = Arc::new(Fixture {
        opaque_compaction: true,
        compact_input_tokens: Some(500_000),
        measured_input_tokens: Mutex::new(Some(500_000)),
        ..Default::default()
    });
    let agent = Agent::new(Backend(fixture.clone()));
    let mut transcript = history();
    let mut context = agent.context_from_history(&(), "fixture", &transcript);
    let policy = CompactionPolicy {
        maintenance_trigger_tokens: Some(100),
        maintenance_growth_tokens: 32_000,
        keep_tail_tokens: 10,
        ..CompactionPolicy::default()
    };
    let first_user = UserMessage::new("first");
    let first_control = TaskControl::new();
    let mut first_task = agent.run_turn_context(
        &mut context,
        &(),
        first_user.clone(),
        &first_control,
        Some(policy),
    );
    let mut saved = None;
    while let Some(update) = first_task.next().await {
        let update = update.unwrap();
        if let AgentEvent::AssistantReply(message, _) = &update.event {
            transcript.push_message(Message::User(first_user.clone()));
            transcript.push_message(Message::Assistant(message.clone()));
        }
        if let Some(checkpoint) = update.checkpoint {
            saved = Some(checkpoint);
        }
    }
    drop(first_task);
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);

    let checkpoint = saved.unwrap();
    assert_eq!(
        checkpoint.maintenance_state.soft_compaction_after_tokens,
        Some(532_001)
    );
    let checkpoint = serde_json::from_slice(&serde_json::to_vec(&checkpoint).unwrap()).unwrap();
    let mut restored = agent
        .restore_context(&(), "fixture", &transcript, checkpoint)
        .unwrap();
    let second_control = TaskControl::new();
    let second_updates = agent
        .run_turn_context(
            &mut restored,
            &(),
            UserMessage::new("second"),
            &second_control,
            Some(policy),
        )
        .collect::<Vec<_>>()
        .await;

    assert!(second_updates.iter().all(Result::is_ok));
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn opaque_maintenance_rebases_from_first_measurement_then_triggers_at_soft_limit() {
    let fixture = Arc::new(Fixture {
        opaque_compaction: true,
        compact_input_tokens: Some(500_000),
        measured_input_tokens: Mutex::new(None),
        ..Default::default()
    });
    let agent = Agent::new(Backend(fixture.clone()));
    let mut transcript = Conversation::new();
    transcript.push_user_message(UserMessage::new("old ".repeat(140_000)));
    let mut context = agent.context_from_history(&(), "fixture", &transcript);
    let policy = CompactionPolicy {
        maintenance_trigger_tokens: Some(128_000),
        maintenance_growth_tokens: 32_000,
        keep_tail_tokens: 10,
        ..CompactionPolicy::default()
    };
    let first_user = UserMessage::new("first");
    let first_control = TaskControl::new();
    let mut first_task = agent.run_turn_context(
        &mut context,
        &(),
        first_user.clone(),
        &first_control,
        Some(policy),
    );
    let mut saved = None;
    while let Some(update) = first_task.next().await {
        let update = update.unwrap();
        if let AgentEvent::AssistantReply(message, _) = &update.event {
            transcript.push_message(Message::User(first_user.clone()));
            transcript.push_message(Message::Assistant(message.clone()));
        }
        if let Some(checkpoint) = update.checkpoint {
            saved = Some(checkpoint);
        }
    }
    drop(first_task);
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);

    let checkpoint = saved.unwrap();
    assert!(checkpoint.maintenance_state.awaiting_native_measurement);
    assert_eq!(
        checkpoint.maintenance_state.soft_compaction_after_tokens,
        Some(532_000)
    );
    let checkpoint = serde_json::from_slice(&serde_json::to_vec(&checkpoint).unwrap()).unwrap();
    let mut restored = agent
        .restore_context(&(), "fixture", &transcript, checkpoint)
        .unwrap();

    *fixture.measured_input_tokens.lock().unwrap() = Some(50_000);
    let second_control = TaskControl::new();
    let second_updates = agent
        .run_turn_context(
            &mut restored,
            &(),
            UserMessage::new("small next turn"),
            &second_control,
            Some(policy),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(second_updates.iter().all(Result::is_ok));
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);
    let remeasured = restored.checkpoint().unwrap();
    assert!(!remeasured.maintenance_state.awaiting_native_measurement);
    assert_eq!(
        remeasured.maintenance_state.soft_compaction_after_tokens,
        Some(82_001)
    );

    let third_control = TaskControl::new();
    let third_updates = agent
        .run_turn_context(
            &mut restored,
            &(),
            UserMessage::new("x".repeat(300_000)),
            &third_control,
            Some(policy),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(third_updates.iter().all(Result::is_ok));
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn manual_opaque_compaction_waits_for_the_first_measured_response() {
    let fixture = Arc::new(Fixture {
        opaque_compaction: true,
        compact_input_tokens: None,
        measured_input_tokens: Mutex::new(None),
        ..Default::default()
    });
    let agent = Agent::new(Backend(fixture.clone()));
    let transcript = history();
    let mut context = agent.context_from_history(&(), "fixture", &transcript);
    context.measurement = Some(ContextMeasurement {
        tokens: 500_000,
        estimated_tokens: transcript.estimate_tokens(&[]),
    });

    agent
        .compact_context(&mut context, &(), CompactOptions::default())
        .await
        .unwrap();
    let checkpoint = context.checkpoint().unwrap();
    assert!(checkpoint.maintenance_state.awaiting_native_measurement);
    assert_eq!(
        checkpoint.maintenance_state.soft_compaction_after_tokens,
        Some(532_000)
    );

    *fixture.measured_input_tokens.lock().unwrap() = Some(50_000);
    let policy = CompactionPolicy {
        maintenance_trigger_tokens: Some(128_000),
        maintenance_growth_tokens: 32_000,
        keep_tail_tokens: 10,
        ..CompactionPolicy::default()
    };
    let control = TaskControl::new();
    let updates = agent
        .run_turn_context(
            &mut context,
            &(),
            UserMessage::new("small after manual compaction"),
            &control,
            Some(policy),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(updates.iter().all(Result::is_ok));
    assert_eq!(fixture.compactions.load(Ordering::SeqCst), 1);
    let checkpoint = context.checkpoint().unwrap();
    assert!(!checkpoint.maintenance_state.awaiting_native_measurement);
    assert_eq!(
        checkpoint.maintenance_state.soft_compaction_after_tokens,
        Some(82_001)
    );
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
                ..CompactOptions::default()
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

#[derive(Clone, Copy, Default)]
struct RequestTotals {
    requests: usize,
    heuristic_input_tokens: u64,
    serialized_input_bytes: usize,
}

#[derive(Default)]
struct BenchmarkFixture {
    totals: Mutex<RequestTotals>,
    screenshot_id: AtomicUsize,
}

#[derive(Clone)]
struct BenchmarkProvider(Arc<BenchmarkFixture>);

impl Provider for BenchmarkProvider {
    type Model = ();
    type Error = io::Error;
    type Continuation = usize;

    fn supports_vision(&self, _: &()) -> bool {
        true
    }

    fn context_budget(&self, _: &()) -> Option<ContextBudget> {
        Some(ContextBudget {
            max_input_tokens: 1_000_000,
        })
    }

    fn create_continuation(&self, _: &()) -> usize {
        0
    }

    async fn complete(
        &self,
        continuation: &usize,
        _: &(),
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, usize), io::Error> {
        let serialized = serde_json::to_vec(&(conversation, tools)).unwrap();
        let mut totals = self.0.totals.lock().unwrap();
        totals.requests += 1;
        totals.heuristic_input_tokens = totals
            .heuristic_input_tokens
            .saturating_add(conversation.estimate_tokens(tools));
        totals.serialized_input_bytes = totals
            .serialized_input_bytes
            .saturating_add(serialized.len());
        drop(totals);

        let message = if tools.iter().any(|tool| tool.name() == "summarize") {
            AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                "summary",
                "summarize",
                serde_json::json!({
                    "summary": "Deterministic summary of prior screenshot observations."
                }),
            )])
        } else if matches!(conversation.messages().last(), Some(Message::User(_))) {
            let id = self.0.screenshot_id.fetch_add(1, Ordering::SeqCst);
            AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                format!("screenshot-{id}"),
                "screenshot",
                serde_json::json!({"step": id}),
            )])
        } else {
            AssistantMessage::new("Screenshot recorded.")
        };
        Ok((message, continuation.saturating_add(1)))
    }
}

struct BenchmarkScreenshotTool;

impl Tool for BenchmarkScreenshotTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "screenshot",
            "Capture the current synthetic application state.",
            serde_json::json!({"type":"object", "properties":{"step":{"type":"integer"}}}),
        )
    }

    fn execute<'a>(&'a self, arguments: &'a serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let _ = self;
            let step = arguments["step"].as_u64().unwrap_or_default();
            let images = (0..4)
                .map(|index| Image::new("image/png", vec![step.wrapping_add(index) as u8; 2048]))
                .collect();
            Ok(ToolOutput::new(serde_json::json!({
                "step": step,
                "caption": "synthetic app state"
            }))
            .with_images(images))
        })
    }
}

async fn run_screenshot_benchmark(policy: CompactionPolicy) -> RequestTotals {
    let fixture = Arc::new(BenchmarkFixture::default());
    let agent =
        Agent::new(BenchmarkProvider(fixture.clone())).with_tool(Arc::new(BenchmarkScreenshotTool));
    let mut context = agent.context_from_history(&(), "fixture", &Conversation::new());
    for round in 0..32 {
        let control = TaskControl::new();
        let updates = agent
            .run_turn_context(
                &mut context,
                &(),
                UserMessage::new(format!("inspect synthetic screen {round}")),
                &control,
                Some(policy),
            )
            .collect::<Vec<_>>()
            .await;
        assert!(updates.iter().all(Result::is_ok));
    }
    *fixture.totals.lock().unwrap()
}

#[tokio::test]
async fn thirty_two_turn_screenshot_simulation_reduces_provider_input_size() {
    let baseline = run_screenshot_benchmark(CompactionPolicy {
        budget: Some(ContextBudget {
            max_input_tokens: 1_000_000,
        }),
        keep_tail_tokens: 24_000,
        ..CompactionPolicy::default()
    })
    .await;
    let maintained = run_screenshot_benchmark(CompactionPolicy {
        budget: Some(ContextBudget {
            max_input_tokens: 1_000_000,
        }),
        keep_tail_tokens: 2_000,
        maintenance_trigger_tokens: Some(32_000),
        maintenance_growth_tokens: 16_000,
        retain_recent_tool_image_observations: Some(2),
    })
    .await;

    eprintln!(
        "32 synthetic screenshot turns: heuristic input tokens {} -> {}, serialized provider input bytes {} -> {}, provider calls {} -> {}",
        baseline.heuristic_input_tokens,
        maintained.heuristic_input_tokens,
        baseline.serialized_input_bytes,
        maintained.serialized_input_bytes,
        baseline.requests,
        maintained.requests,
    );
    assert!(maintained.heuristic_input_tokens < baseline.heuristic_input_tokens);
    assert!(maintained.serialized_input_bytes < baseline.serialized_input_bytes);
}
