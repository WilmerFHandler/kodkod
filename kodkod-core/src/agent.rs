pub mod control;
pub mod error;
pub mod event;

use std::borrow::Cow;
use std::future::Future;
use std::sync::Arc;
use std::task::Poll;

use async_stream::try_stream;
use futures::StreamExt;

pub use control::{SteerError, TaskControl};
pub use error::AgentError;
pub use event::AgentEvent;

/// Streaming events from an [`Agent::run_turn`] call.
///
/// Dropping the task drops local provider and tool futures. It cannot undo
/// external effects already performed by a provider or tool.
pub type Task<'a, E> = futures::stream::BoxStream<'a, Result<AgentEvent, AgentError<E>>>;

use crate::{
    CompactError, Conversation, Message, Provider, ProviderCompletion, ProviderEvent, TokenUsage,
    Tool, ToolExecutor, TurnCompaction, UserMessage,
};

async fn cancel_or<T>(control: &TaskControl, operation: impl Future<Output = T>) -> Option<T> {
    let cancellation = control.cancelled();
    futures::pin_mut!(operation, cancellation);
    futures::future::poll_fn(|cx| {
        if cancellation.as_mut().poll(cx).is_ready() {
            Poll::Ready(None)
        } else {
            operation.as_mut().poll(cx).map(Some)
        }
    })
    .await
}

pub struct Agent<P> {
    provider: P,
    tools: ToolExecutor,
    max_tool_rounds: Option<usize>,
    immediate_steering: bool,
}

pub use crate::context::AgentContext;
use crate::{
    AgentUpdate, CompactionPolicy, ContextCheckpoint, ContextCompaction, ContextCursor,
    ContextMaintenanceState, ContextMeasurement, ContextTask,
};

enum ContextSlot<'a, P: Provider> {
    Owned(Arc<P::Continuation>),
    Borrowed(&'a mut Arc<P::Continuation>),
}
impl<P: Provider> ContextSlot<'_, P> {
    fn continuation(&self) -> &P::Continuation {
        match self {
            Self::Owned(c) => c,
            Self::Borrowed(c) => c,
        }
    }
    fn shared(&self) -> Arc<P::Continuation> {
        match self {
            Self::Owned(c) => c.clone(),
            Self::Borrowed(c) => (*c).clone(),
        }
    }
    fn checkpoint(&mut self, continuation: P::Continuation) {
        match self {
            Self::Owned(c) => *c = Arc::new(continuation),
            Self::Borrowed(c) => **c = Arc::new(continuation),
        }
    }
}
struct ContextTracker<'a> {
    cursor: &'a mut ContextCursor,
    binding: &'a str,
    measurement: &'a mut Option<ContextMeasurement>,
    maintenance_state: &'a mut ContextMaintenanceState,
}
fn update<P: Provider>(
    event: AgentEvent,
    conversation: &Conversation,
    context: Option<&ContextSlot<'_, P>>,
    tracker: &Option<ContextTracker<'_>>,
) -> AgentUpdate<P::Continuation> {
    let accepted = matches!(
        &event,
        AgentEvent::Steered(_)
            | AgentEvent::AssistantReply(..)
            | AgentEvent::ToolFinished(_)
            | AgentEvent::Compacted(_)
            | AgentEvent::CompactionFailed(_)
    );
    let checkpoint = if accepted {
        tracker
            .as_ref()
            .zip(context)
            .map(|(t, c)| ContextCheckpoint {
                version: 1,
                binding: t.binding.to_owned(),
                cursor: t.cursor.clone(),
                conversation: conversation.clone(),
                continuation: c.shared(),
                measurement: *t.measurement,
                maintenance_state: *t.maintenance_state,
            })
    } else {
        None
    };
    AgentUpdate { event, checkpoint }
}
fn legacy<'a, C: Send + 'a, E: 'a>(task: ContextTask<'a, C, E>) -> Task<'a, E> {
    Box::pin(task.map(|result| result.map(|update| update.event)))
}

async fn maintain<P: Provider>(
    provider: &P,
    c: &P::Continuation,
    m: &P::Model,
    conversation: &Conversation,
    options: crate::CompactOptions,
) -> Result<ContextCompaction<P::Continuation>, CompactError<P::Error>> {
    match provider.compact_context(c, m, conversation, options).await {
        Err(CompactError::Unsupported(_)) if !provider.has_opaque_context(c) => {
            let portable = provider
                .compact_with_usage(m, conversation, options)
                .await?;
            Ok(ContextCompaction {
                conversation: portable.conversation,
                continuation: provider.create_continuation(m),
                usage: portable.usage,
            })
        }
        Err(CompactError::Provider(error))
            if provider.is_context_overflow(&error) && !provider.has_opaque_context(c) =>
        {
            let portable = provider
                .compact_with_usage(m, conversation, options)
                .await?;
            Ok(ContextCompaction {
                conversation: portable.conversation,
                continuation: provider.create_continuation(m),
                usage: portable.usage,
            })
        }
        result => result,
    }
}

struct RunState<'a, P: Provider> {
    context: ContextSlot<'a, P>,
    compaction: Option<TurnCompaction<&'a P::Model>>,
    guard: TurnGuard<'a>,
    tracker: Option<ContextTracker<'a>>,
    policy: Option<CompactionPolicy>,
}

struct TurnGuard<'a> {
    control: &'a TaskControl,
}
impl Drop for TurnGuard<'_> {
    fn drop(&mut self) {
        self.control.close();
    }
}

impl<P> Agent<P>
where
    P: Provider,
{
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            tools: ToolExecutor::new(),
            max_tool_rounds: None,
            immediate_steering: false,
        }
    }
    pub fn provider(&self) -> &P {
        &self.provider
    }
    pub fn tools(&self) -> &ToolExecutor {
        &self.tools
    }
    pub fn max_tool_rounds(&self) -> Option<usize> {
        self.max_tool_rounds
    }
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.register_tool(tool);
        self
    }
    /// Restart in-progress client-only inference from its last accepted checkpoint
    /// when steering arrives. Running local tools still finish normally. Applications
    /// must handle AssistantPreviewReset and should not opt in for hosted tools
    /// with remote side effects. This is not provider-native WebSocket steering.
    pub fn with_immediate_steering(mut self) -> Self {
        self.immediate_steering = true;
        self
    }

    pub fn with_max_tool_rounds(mut self, max: usize) -> Self {
        self.max_tool_rounds = Some(max);
        self
    }
    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.register(tool);
    }

    /// Create a fresh context which can retain provider continuation across turns.
    pub fn new_context(&self, model: &P::Model) -> AgentContext<P> {
        AgentContext {
            continuation: Arc::new(self.provider.create_continuation(model)),
            conversation: Conversation::new(),
            cursor: ContextCursor::default(),
            binding: String::new(),
            measurement: None,
            maintenance_state: ContextMaintenanceState::default(),
            managed: true,
        }
    }

    /// Run one logical user turn with a fresh provider continuation.
    pub fn run_turn<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a P::Model,
        user: UserMessage,
        control: &'a TaskControl,
    ) -> Task<'a, P::Error> {
        if control.open().is_err() {
            return Box::pin(futures::stream::once(async {
                Err(AgentError::ControlAlreadyUsed)
            }));
        }
        conversation.push_message(Message::User(user.with_steered(false)));
        let context = ContextSlot::Owned(Arc::new(self.provider.create_continuation(model)));
        legacy(self.run_opened(
            conversation,
            model,
            control,
            RunState {
                context,
                compaction: None,
                guard: TurnGuard { control },
                tracker: None,
                policy: None,
            },
        ))
    }

    /// Run one logical user turn in a retained provider context.
    pub fn run_turn_in<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a P::Model,
        user: UserMessage,
        context: &'a mut AgentContext<P>,
        control: &'a TaskControl,
        compaction: Option<TurnCompaction<&'a P::Model>>,
    ) -> Task<'a, P::Error> {
        if control.open().is_err() {
            return Box::pin(futures::stream::once(async {
                Err(AgentError::ControlAlreadyUsed)
            }));
        }
        context.managed = false;
        conversation.push_message(Message::User(user.with_steered(false)));
        legacy(self.run_opened(
            conversation,
            model,
            control,
            RunState {
                context: ContextSlot::Borrowed(&mut context.continuation),
                compaction,
                guard: TurnGuard { control },
                tracker: None,
                policy: None,
            },
        ))
    }

    /// Compatibility entry point. Uses a fresh continuation without appending a prompt.
    #[deprecated(note = "use run_turn; it explicitly starts a logical user turn")]
    pub fn run<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a P::Model,
        control: &'a TaskControl,
    ) -> Task<'a, P::Error> {
        if control.open().is_err() {
            return Box::pin(futures::stream::once(async {
                Err(AgentError::ControlAlreadyUsed)
            }));
        }
        let context = ContextSlot::Owned(Arc::new(self.provider.create_continuation(model)));
        legacy(self.run_opened(
            conversation,
            model,
            control,
            RunState {
                context,
                compaction: None,
                guard: TurnGuard { control },
                tracker: None,
                policy: None,
            },
        ))
    }

    /// Seed active context from a complete transcript. `binding` is a stable, non-secret
    /// provider/model/account identity supplied by the host, never an access token.
    pub fn context_from_history(
        &self,
        model: &P::Model,
        binding: impl Into<String>,
        history: &Conversation,
    ) -> AgentContext<P> {
        AgentContext {
            conversation: history.clone(),
            continuation: Arc::new(self.provider.create_continuation(model)),
            cursor: ContextCursor::for_history(history),
            binding: binding.into(),
            measurement: None,
            maintenance_state: ContextMaintenanceState::default(),
            managed: true,
        }
    }

    /// Restore only against the exact source transcript. Edits and incompatible accounts
    /// require an explicit rebase from complete history, never silently dropped native state.
    pub fn restore_context(
        &self,
        model: &P::Model,
        binding: &str,
        history: &Conversation,
        checkpoint: ContextCheckpoint<P::Continuation>,
    ) -> Result<AgentContext<P>, String> {
        if checkpoint.version != 1
            || checkpoint.binding != binding
            || checkpoint.cursor.position > history.messages().len()
        {
            return Err(
                "Context checkpoint does not match its transcript or provider binding".into(),
            );
        }
        let mut prefix = Conversation::new();
        for message in &history.messages()[..checkpoint.cursor.position] {
            prefix.push_message(message.clone());
        }
        if checkpoint.cursor != ContextCursor::for_history(&prefix) {
            return Err("Context checkpoint source history was edited".into());
        }
        self.provider
            .validate_context(model, &checkpoint.continuation)?;
        let mut active = AgentContext {
            conversation: checkpoint.conversation,
            continuation: checkpoint.continuation,
            cursor: checkpoint.cursor,
            binding: checkpoint.binding,
            measurement: checkpoint.measurement,
            maintenance_state: checkpoint.maintenance_state,
            managed: true,
        };
        for message in &history.messages()[active.cursor.position..] {
            active.cursor.append(message);
            active.conversation.push_message(message.clone());
        }
        active.set_system_prompt(history.system_prompt());
        Ok(active)
    }

    /// Canonical persistent loop. Persist each checkpoint together with its semantic
    /// transcript event before polling again; this also fences tool dispatch on save errors.
    pub fn run_turn_context<'a>(
        &'a self,
        context: &'a mut AgentContext<P>,
        model: &'a P::Model,
        user: UserMessage,
        control: &'a TaskControl,
        policy: Option<CompactionPolicy>,
    ) -> ContextTask<'a, P::Continuation, P::Error> {
        if !context.managed {
            return Box::pin(futures::stream::once(async {
                Err(AgentError::ContextMaintenance(
                    "Legacy split context must be explicitly rebased before canonical execution"
                        .into(),
                ))
            }));
        }
        if let Err(message) = self.provider.validate_context(model, &context.continuation) {
            return Box::pin(futures::stream::once(async move {
                Err(AgentError::ContextMaintenance(message))
            }));
        }
        if control.open().is_err() {
            return Box::pin(futures::stream::once(async {
                Err(AgentError::ControlAlreadyUsed)
            }));
        }
        let message = Message::User(user.with_steered(false));
        context.cursor.append(&message);
        context.conversation.push_message(message);
        let tracker = ContextTracker {
            cursor: &mut context.cursor,
            binding: &context.binding,
            measurement: &mut context.measurement,
            maintenance_state: &mut context.maintenance_state,
        };
        self.run_opened(
            &mut context.conversation,
            model,
            control,
            RunState {
                context: ContextSlot::Borrowed(&mut context.continuation),
                compaction: None,
                guard: TurnGuard { control },
                tracker: Some(tracker),
                policy,
            },
        )
    }

    /// Manual maintenance uses the same contextual provider operation as automatic maintenance.
    /// No mutation occurs until the replacement has been returned successfully.
    pub async fn compact_context(
        &self,
        context: &mut AgentContext<P>,
        model: &P::Model,
        options: crate::CompactOptions,
    ) -> Result<TokenUsage, CompactError<P::Error>> {
        if !context.managed {
            return Err(CompactError::Context(
                "Legacy split context must be explicitly rebased before compaction".into(),
            ));
        }
        self.provider
            .validate_context(model, &context.continuation)
            .map_err(CompactError::Context)?;
        let replacement = maintain(
            &self.provider,
            &context.continuation,
            model,
            &context.conversation,
            options,
        )
        .await?;
        let has_opaque_context = self.provider.has_opaque_context(&replacement.continuation);
        let prior_estimate = context
            .conversation
            .estimate_tokens(&[])
            .saturating_add(P::estimate_continuation_tokens(&context.continuation));
        let replacement_estimate = replacement
            .conversation
            .estimate_tokens(&[])
            .saturating_add(P::estimate_continuation_tokens(&replacement.continuation));
        let maintenance_baseline = prior_estimate
            .max(
                context
                    .measurement
                    .map_or(0, |measurement| measurement.tokens),
            )
            .max(replacement_estimate)
            .max(replacement.usage.input_tokens.unwrap_or(0));
        context.conversation = replacement.conversation;
        context.continuation = Arc::new(replacement.continuation);
        context.measurement = None;
        context.maintenance_state.soft_compaction_after_tokens = has_opaque_context.then(|| {
            maintenance_baseline
                .saturating_add(CompactionPolicy::default().maintenance_growth_tokens.max(1))
        });
        context.maintenance_state.awaiting_native_measurement = has_opaque_context;
        Ok(replacement.usage)
    }

    fn run_opened<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a P::Model,
        control: &'a TaskControl,
        state: RunState<'a, P>,
    ) -> ContextTask<'a, P::Continuation, P::Error> {
        let RunState {
            context,
            compaction,
            guard,
            mut tracker,
            policy,
        } = state;
        Box::pin(try_stream! {
            // Keep these options so terminal paths release retained state and
            // close steering before yielding Completed.
            let mut context = Some(context);
            let mut guard = Some(guard);
            let vision_enabled = self.provider.supports_vision(model);
            let computer_use_enabled = self.provider.supports_computer_use(model);
            let tool_specs = self
                .tools
                .specs_for_capabilities(vision_enabled, computer_use_enabled);
            let mut tool_rounds_executed = 0;
            let mut turn_usage = TokenUsage::zero();
            let mut force_compaction = false;
            let mut overflow_recovered = false;

            'rounds: loop {
                let steers = control
                    .drain_pending_steers_unless_cancelled()
                    .ok_or(AgentError::Cancelled)?;
                for user in steers {
                    let message = Message::User(user.clone());
                    if let Some(t) = tracker.as_mut() {
                        t.cursor.append(&message);
                    }
                    conversation.push_message(message);
                    yield update(
                        AgentEvent::Steered(user),
                        conversation,
                        context.as_ref(),
                        &tracker,
                    );
                }

                if let Some(policy) = policy {
                    let raw_estimate = conversation.estimate_tokens(&tool_specs).saturating_add(
                        P::estimate_continuation_tokens(
                            context.as_ref().expect("context exists").continuation(),
                        ),
                    );
                    let measured = tracker
                        .as_ref()
                        .and_then(|t| *t.measurement)
                        .map_or(0, |m| {
                            m.tokens
                                .saturating_add(raw_estimate.saturating_sub(m.estimated_tokens))
                        });
                    let estimate = raw_estimate.max(measured);
                    let budget = policy
                        .budget
                        .or_else(|| self.provider.context_budget(model));
                    let hard_due = budget.is_some_and(|b| estimate >= b.max_input_tokens);
                    let soft_due = policy
                        .maintenance_trigger_tokens
                        .is_some_and(|trigger| estimate >= trigger)
                        && tracker
                            .as_ref()
                            .and_then(|t| t.maintenance_state.soft_compaction_after_tokens)
                            .is_none_or(|after| estimate >= after);
                    let manual = control.take_compact_request();
                    if force_compaction || hard_due || soft_due || manual {
                        yield update(
                            AgentEvent::CompactionStarted,
                            conversation,
                            context.as_ref(),
                            &tracker,
                        );
                        let replacement = cancel_or(
                            control,
                            maintain(
                                &self.provider,
                                context.as_ref().expect("context exists").continuation(),
                                model,
                                conversation,
                                policy.options(),
                            ),
                        )
                        .await
                        .ok_or(AgentError::Cancelled)?;
                        match replacement {
                            Ok(ContextCompaction {
                                conversation: replacement,
                                continuation,
                                usage,
                            }) => {
                                if control.is_cancelled() {
                                    Err(AgentError::Cancelled)?;
                                }
                                let next_estimate = replacement
                                    .estimate_tokens(&tool_specs)
                                    .saturating_add(P::estimate_continuation_tokens(&continuation));
                                let has_opaque_context =
                                    self.provider.has_opaque_context(&continuation);
                                let ineffective = next_estimate >= estimate
                                    && !has_opaque_context;
                                if (hard_due || force_compaction) && ineffective {
                                    turn_usage.add_round(usage);
                                    Err(AgentError::ContextMaintenance("Compaction made no room; reduce the input or use a larger context window".into()))?;
                                }
                                if let Some(t) = tracker.as_mut() {
                                    // Native compaction returns an opaque replay window whose
                                    // measured post-compaction size may be unknown. Use the
                                    // actual pre-compaction input count as a temporary growth
                                    // baseline so the next measured round does not immediately
                                    // retrigger soft maintenance.
                                    let maintenance_baseline = if has_opaque_context {
                                        estimate
                                            .max(next_estimate)
                                            .max(usage.input_tokens.unwrap_or(0))
                                    } else {
                                        next_estimate
                                    };
                                    t.maintenance_state.soft_compaction_after_tokens = Some(
                                        maintenance_baseline.saturating_add(
                                            policy.maintenance_growth_tokens.max(1),
                                        ),
                                    );
                                    t.maintenance_state.awaiting_native_measurement =
                                        has_opaque_context;
                                }
                                turn_usage.add_round(usage);
                                if soft_due
                                    && ineffective
                                    && !hard_due
                                    && !force_compaction
                                    && !manual
                                {
                                    yield update(
                                        AgentEvent::CompactionFailed(
                                            "Soft maintenance made no room; it will retry after context grows".into(),
                                        ),
                                        conversation,
                                        context.as_ref(),
                                        &tracker,
                                    );
                                } else {
                                    *conversation = replacement;
                                    context
                                        .as_mut()
                                        .expect("context exists")
                                        .checkpoint(continuation);
                                    if let Some(t) = tracker.as_mut() {
                                        *t.measurement = None;
                                    }
                                    yield update(
                                        AgentEvent::Compacted(conversation.clone()),
                                        conversation,
                                        context.as_ref(),
                                        &tracker,
                                    );
                                }
                            }
                            Err(CompactError::NothingToCompact)
                                if !hard_due && !force_compaction && !manual =>
                            {
                                if soft_due
                                    && let Some(t) = tracker.as_mut()
                                {
                                    t.maintenance_state.soft_compaction_after_tokens = Some(
                                        estimate.saturating_add(
                                            policy.maintenance_growth_tokens.max(1),
                                        ),
                                    );
                                }
                                if soft_due {
                                    yield update(
                                        AgentEvent::CompactionFailed(
                                            "Soft maintenance had nothing to compact; it will retry after context grows".into(),
                                        ),
                                        conversation,
                                        context.as_ref(),
                                        &tracker,
                                    );
                                }
                            }
                            Err(error) => {
                                if soft_due
                                    && !hard_due
                                    && !force_compaction
                                    && let Some(t) = tracker.as_mut()
                                {
                                    t.maintenance_state.soft_compaction_after_tokens = Some(
                                        estimate.saturating_add(
                                            policy.maintenance_growth_tokens.max(1),
                                        ),
                                    );
                                }
                                yield update(
                                    AgentEvent::CompactionFailed(error.to_string()),
                                    conversation,
                                    context.as_ref(),
                                    &tracker,
                                );
                                if hard_due || force_compaction {
                                    Err(AgentError::ContextMaintenance(error.to_string()))?;
                                }
                            }
                        }
                        force_compaction = false;
                        // Accept any steering that arrived while maintenance was in flight.
                        let steers = control
                            .drain_pending_steers_unless_cancelled()
                            .ok_or(AgentError::Cancelled)?;
                        for user in steers {
                            let message = Message::User(user.clone());
                            if let Some(t) = tracker.as_mut() {
                                t.cursor.append(&message);
                            }
                            conversation.push_message(message);
                            yield update(
                                AgentEvent::Steered(user),
                                conversation,
                                context.as_ref(),
                                &tracker,
                            );
                        }
                        let final_size = conversation.estimate_tokens(&tool_specs).saturating_add(
                            P::estimate_continuation_tokens(
                                context.as_ref().expect("context exists").continuation(),
                            ),
                        );
                        if budget.is_some_and(|b| final_size >= b.max_input_tokens) {
                            Err(AgentError::ContextMaintenance("The retained input, instructions, or tools still exceed this model's input budget".into()))?;
                        }
                    }
                }

                if let Some(document) = conversation.messages().iter().find_map(|message| {
                    let Message::User(user) = message else {
                        return None;
                    };
                    user.documents()
                        .iter()
                        .find(|document| !self.provider.supports_document(model, document.mime()))
                }) {
                    Err(AgentError::UnsupportedDocument {
                        mime: document.mime().to_owned(),
                    })?;
                }

                if let Some(compaction) = compaction.filter(|_| control.take_compact_request()) {
                    yield update(
                        AgentEvent::CompactionStarted,
                        conversation,
                        context.as_ref(),
                        &tracker,
                    );
                    let result = match cancel_or(
                        control,
                        self.provider.compact_with_usage(
                            compaction.model,
                            conversation,
                            compaction.options,
                        ),
                    )
                    .await
                    {
                        Some(result) if !control.is_cancelled() => result,
                        _ => Err(AgentError::Cancelled)?,
                    };
                    match result {
                        Ok(compacted) => {
                            let crate::CompactionResult {
                                conversation: compacted,
                                usage,
                            } = compacted;
                            turn_usage.add_round(usage);
                            // Conversation and continuation are one checkpoint.
                            *conversation = compacted.clone();
                            context
                                .as_mut()
                                .expect("agent context exists")
                                .checkpoint(self.provider.create_continuation(model));
                            yield update(
                                AgentEvent::Compacted(compacted),
                                conversation,
                                context.as_ref(),
                                &tracker,
                            );
                        }
                        Err(CompactError::NothingToCompact) => {}
                        Err(error) => {
                            yield update(
                                AgentEvent::CompactionFailed(error.to_string()),
                                conversation,
                                context.as_ref(),
                                &tracker,
                            );
                        }
                    }
                }

                let provider_input: Cow<'_, Conversation> = if vision_enabled {
                    Cow::Borrowed(conversation)
                } else {
                    Cow::Owned(conversation.without_images())
                };

                let mut provider_stream = self.provider.complete_stream(
                    context
                        .as_ref()
                        .expect("agent context exists")
                        .continuation(),
                    model,
                    &provider_input,
                    &tool_specs,
                );
                let completed = loop {
                    // Cancellation wins when both become ready in one poll.
                    let event = if self.immediate_steering {
                        let steering = control.steering_ready();
                        let next = provider_stream.next();
                        futures::pin_mut!(steering, next);
                        let interrupted = futures::future::poll_fn(|cx| {
                            if control.is_cancelled() {
                                Poll::Ready(Some(None))
                            } else if steering.as_mut().poll(cx).is_ready() {
                                Poll::Ready(None)
                            } else {
                                next.as_mut().poll(cx).map(|event| Some(Some(event)))
                            }
                        });
                        match cancel_or(control, interrupted).await {
                            Some(None) => {
                                drop(provider_stream);
                                drop(provider_input);
                                yield update(
                                    AgentEvent::AssistantPreviewReset,
                                    conversation,
                                    context.as_ref(),
                                    &tracker,
                                );
                                continue 'rounds;
                            }
                            Some(Some(event)) => event,
                            None => None,
                        }
                    } else {
                        cancel_or(control, provider_stream.next()).await
                    };
                    match event {
                        Some(Some(Ok(ProviderEvent::TextDelta(delta)))) if !control.is_cancelled() => {
                            yield update(
                                AgentEvent::AssistantTextDelta(delta),
                                conversation,
                                context.as_ref(),
                                &tracker,
                            );
                        }
                        Some(Some(Ok(ProviderEvent::Completed(ProviderCompletion {
                            message,
                            continuation,
                            usage,
                        }))))
                            if !control.is_cancelled() =>
                        {
                            break Ok((message, continuation, usage));
                        }
                        Some(Some(Ok(_))) | None => Err(AgentError::Cancelled)?,
                        Some(Some(Err(error))) => break Err(error),
                        Some(None) => Err(AgentError::ProviderStreamEnded)?,
                    }
                };
                drop(provider_stream);
                drop(provider_input);
                let (message, next_continuation, round_usage) = match completed {
                    Ok(completed) => completed,
                    Err(error)
                        if policy.is_some()
                            && !overflow_recovered
                            && self.provider.is_context_overflow(&error) =>
                    {
                        overflow_recovered = true;
                        force_compaction = true;
                        yield update(
                            AgentEvent::AssistantPreviewReset,
                            conversation,
                            context.as_ref(),
                            &tracker,
                        );
                        continue 'rounds;
                    }
                    Err(error) => Err(AgentError::Provider(error))?,
                };
                overflow_recovered = false;
                turn_usage.add_round(round_usage);

                // The transcript and continuation are one logical checkpoint.
                // No cancellation check occurs between these operations: once
                // accepted, both survive cancellation racing immediately after.
                let tool_calls = message.tool_calls().to_vec();
                let accepted = Message::Assistant(message.clone());
                if let Some(t) = tracker.as_mut() {
                    t.cursor.append(&accepted);
                }
                conversation.push_message(accepted);
                context
                    .as_mut()
                    .expect("agent context exists")
                    .checkpoint(next_continuation);
                if let Some(t) = tracker.as_mut()
                    && let Some(input) = round_usage.input_tokens
                {
                    let estimated_tokens = conversation.estimate_tokens(&tool_specs).saturating_add(
                        P::estimate_continuation_tokens(
                            context.as_ref().expect("context exists").continuation(),
                        ),
                    );
                    let measured_tokens = input.saturating_add(round_usage.output_tokens.unwrap_or(0));
                    *t.measurement = Some(ContextMeasurement {
                        tokens: measured_tokens,
                        estimated_tokens,
                    });
                    if t.maintenance_state.awaiting_native_measurement
                        && let Some(policy) = policy
                    {
                        t.maintenance_state.soft_compaction_after_tokens = Some(
                            measured_tokens.saturating_add(
                                policy.maintenance_growth_tokens.max(1),
                            ),
                        );
                        t.maintenance_state.awaiting_native_measurement = false;
                    }
                }
                yield update(
                    AgentEvent::AssistantReply(message.clone(), round_usage),
                    conversation,
                    context.as_ref(),
                    &tracker,
                );

                if tool_calls.is_empty() {
                    if !control.close_if_empty() {
                        continue;
                    }
                    drop(context.take());
                    drop(guard.take());
                    yield update(
                        AgentEvent::Completed(message, turn_usage),
                        conversation,
                        context.as_ref(),
                        &tracker,
                    );
                    return;
                }

                if let Some(max) = self.max_tool_rounds
                    && tool_rounds_executed >= max
                {
                    Err(AgentError::MaxToolRoundsExceeded { max })?;
                }

                for tool_call in &tool_calls {
                    yield update(
                        AgentEvent::ToolStarted(tool_call.clone()),
                        conversation,
                        context.as_ref(),
                        &tracker,
                    );
                }
                let mut results = futures::stream::FuturesUnordered::new();
                for tool_call in &tool_calls {
                    results.push(self.tools.execute_for_capabilities(
                        tool_call,
                        vision_enabled,
                        computer_use_enabled,
                    ));
                }
                while !results.is_empty() {
                    let Some(result) = cancel_or(control, results.next()).await else {
                        Err(AgentError::Cancelled)?
                    };
                    let Some(result) = result else { break };
                    let accepted = Message::ToolResult(result.clone());
                    if let Some(t) = tracker.as_mut() {
                        t.cursor.append(&accepted);
                    }
                    conversation.push_message(accepted);
                    yield update(
                        AgentEvent::ToolFinished(result),
                        conversation,
                        context.as_ref(),
                        &tracker,
                    );
                }
                tool_rounds_executed += 1;
            }
        })
    }
}
