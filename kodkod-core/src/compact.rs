use std::collections::HashSet;
use std::error::Error as StdError;
use std::fmt;

use serde_json::{Value, json};

use crate::{AssistantMessage, Conversation, Message, Provider, TokenUsage, ToolSpec, UserMessage};

pub const DEFAULT_KEEP_TAIL_TOKENS: u64 = 20_000;

const COMPACTION_SYSTEM: &str = include_str!("compact/prompt.txt");
const COMPACTION_USER: &str =
    "Call summarize now with the handoff summary. Do not reply to the user.";
const SUMMARIZE_TOOL: &str = "summarize";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactOptions {
    pub keep_tail_tokens: u64,
    /// Preserve only the newest image-bearing tool results while compacting.
    pub retain_recent_tool_image_observations: Option<usize>,
}

/// Per-turn compaction settings. The model may differ from the turn model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnCompaction<M> {
    pub model: M,
    pub options: CompactOptions,
}

impl Default for CompactOptions {
    fn default() -> Self {
        Self {
            keep_tail_tokens: DEFAULT_KEEP_TAIL_TOKENS,
            retain_recent_tool_image_observations: None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CompactError<E> {
    NothingToCompact,
    Unsupported(String),
    Context(String),
    EmptySummary,
    MissingSummary,
    Provider(E),
}

/// A compacted conversation and the usage consumed to produce its summary.
#[derive(Debug)]
pub struct CompactionResult {
    pub conversation: Conversation,
    pub usage: TokenUsage,
}

impl<E: fmt::Display> fmt::Display for CompactError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) | Self::Context(message) => f.write_str(message),
            Self::NothingToCompact => f.write_str("conversation has nothing to compact"),
            Self::EmptySummary => f.write_str("compaction model returned an empty summary"),
            Self::MissingSummary => f.write_str("compaction model did not call summarize"),
            Self::Provider(error) => write!(f, "{error}"),
        }
    }
}

impl<E: StdError + 'static> StdError for CompactError<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::NothingToCompact
            | Self::EmptySummary
            | Self::MissingSummary
            | Self::Unsupported(_)
            | Self::Context(_) => None,
        }
    }
}

pub(crate) async fn run_with_usage<P: Provider + ?Sized>(
    provider: &P,
    model: &P::Model,
    conversation: &Conversation,
    options: CompactOptions,
) -> Result<CompactionResult, CompactError<P::Error>> {
    let projected = options.retain_recent_tool_image_observations.map_or_else(
        || conversation.clone(),
        |keep| conversation.retain_recent_tool_image_observations(keep),
    );
    let plan = match plan_with_tail(projected.messages(), options.keep_tail_tokens) {
        Some(plan) => plan,
        None if projected != *conversation => {
            return Ok(CompactionResult {
                conversation: projected,
                usage: TokenUsage::zero(),
            });
        }
        None => return Err(CompactError::NothingToCompact),
    };
    let summary_input = summarizer_messages(&projected, plan.tail_start, &plan.kept);
    let request = summarizer_conversation(&summary_input);
    let tools = [summarize_tool()];
    let budget = provider.context_budget(model).map(|b| b.max_input_tokens);
    let (summary, usage) = if budget.is_some_and(|limit| request.estimate_tokens(&tools) >= limit) {
        summarize_bounded(provider, model, &summary_input, budget.unwrap(), &tools).await?
    } else {
        let completion = provider
            .complete_once_with_usage(model, &request, &tools)
            .await
            .map_err(CompactError::Provider)?;
        (summary_from_reply(&completion.message)?, completion.usage)
    };
    Ok(CompactionResult {
        conversation: splice(&projected, &plan.kept, AssistantMessage::new(summary)),
        usage,
    })
}

/// Rebase long transcripts without silently dropping old exchanges. Tool batches stay
/// together; a single unsplittable oversized item produces an actionable error.
async fn summarize_bounded<P: Provider + ?Sized>(
    provider: &P,
    model: &P::Model,
    messages: &[Message],
    budget: u64,
    tools: &[ToolSpec],
) -> Result<(String, TokenUsage), CompactError<P::Error>> {
    let mut boundaries = Vec::new();
    let mut pending = HashSet::new();
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::Assistant(a) => {
                for call in a.tool_calls() {
                    pending.insert(call.id().to_owned());
                }
            }
            Message::ToolResult(r) => {
                pending.remove(r.tool_call_id());
            }
            _ => {}
        }
        if pending.is_empty() {
            boundaries.push(index + 1);
        }
    }
    let mut costs = vec![0_u64];
    for message in messages {
        costs.push(
            costs
                .last()
                .unwrap()
                .saturating_add(message.estimate_tokens()),
        );
    }
    let mut start = 0;
    let mut summary = String::new();
    let mut usage = TokenUsage::zero();
    while start < messages.len() {
        let mut request = Conversation::new().with_system_prompt(COMPACTION_SYSTEM.trim());
        if !summary.is_empty() {
            request.push_message(Message::Assistant(AssistantMessage::new(&summary)));
        }
        request.push_user_message(UserMessage::new(COMPACTION_USER));
        let overhead = request.estimate_tokens(tools);
        let end=boundaries.iter().copied().skip_while(|end|*end<=start).take_while(|end|overhead.saturating_add(costs[*end].saturating_sub(costs[start]))<budget).last().ok_or_else(||CompactError::Context("A single retained exchange exceeds the compaction budget; reduce its attachments or tool output before retrying".into()))?;
        let mut request = Conversation::new().with_system_prompt(COMPACTION_SYSTEM.trim());
        if !summary.is_empty() {
            request.push_message(Message::Assistant(AssistantMessage::new(&summary)));
        }
        for message in &messages[start..end] {
            request.push_message(message.clone());
        }
        request.push_user_message(UserMessage::new(COMPACTION_USER));
        let completion = provider
            .complete_once_with_usage(model, &request, tools)
            .await
            .map_err(CompactError::Provider)?;
        summary = summary_from_reply(&completion.message)?;
        usage.add_round(completion.usage);
        start = end;
    }
    if summary.is_empty() {
        return Err(CompactError::EmptySummary);
    }
    Ok((summary, usage))
}

struct CompactionPlan {
    kept: Vec<usize>,
    tail_start: usize,
}

#[cfg(test)]
fn plan(messages: &[Message], keep_tail_tokens: u64) -> Option<Vec<usize>> {
    plan_with_tail(messages, keep_tail_tokens).map(|plan| plan.kept)
}

fn plan_with_tail(messages: &[Message], keep_tail_tokens: u64) -> Option<CompactionPlan> {
    if messages.is_empty() {
        return None;
    }

    let mut tail_starts_at = messages.len();
    let mut tail_tokens = 0_u64;
    while tail_starts_at > 0 {
        let next = messages[tail_starts_at - 1].estimate_tokens();
        if next > keep_tail_tokens.saturating_sub(tail_tokens) {
            break;
        }
        tail_tokens += next;
        tail_starts_at -= 1;
    }
    tail_starts_at = protocol_safe_tail_start(messages, tail_starts_at);

    let mut kept = pinned_indices(messages);
    kept.extend(tail_starts_at..messages.len());
    kept.sort_unstable();
    kept.dedup();

    if kept.len() == messages.len() {
        None
    } else {
        Some(CompactionPlan {
            kept,
            tail_start: tail_starts_at,
        })
    }
}

fn pinned_indices(messages: &[Message]) -> Vec<usize> {
    let latest_user = messages
        .iter()
        .rposition(|message| matches!(message, Message::User(user) if !user.steered()));
    let latest_system = messages
        .iter()
        .enumerate()
        .rposition(|(_, message)| matches!(message, Message::System(_)));
    let mut pinned = Vec::new();
    if let Some(latest_user) = latest_user {
        pinned.extend(messages.iter().enumerate().skip(latest_user).filter_map(
            |(index, message)| match message {
                Message::User(user) if index == latest_user || user.steered() => Some(index),
                _ => None,
            },
        ));
    }
    if let Some(latest_system) = latest_system {
        pinned.push(latest_system);
    }
    pinned.sort_unstable();
    pinned.dedup();
    pinned
}

fn protocol_safe_tail_start(messages: &[Message], mut start: usize) -> usize {
    while let Some(index) = first_protocol_violation(messages, start) {
        start = index + 1;
    }
    start
}

fn first_protocol_violation(messages: &[Message], start: usize) -> Option<usize> {
    let mut index = start;
    while index < messages.len() {
        match &messages[index] {
            Message::ToolResult(_) => return Some(index),
            Message::Assistant(assistant) if !assistant.tool_calls().is_empty() => {
                let mut pending = HashSet::new();
                for call in assistant.tool_calls() {
                    if !pending.insert(call.id()) {
                        return Some(index);
                    }
                }

                let call_index = index;
                index += 1;
                while !pending.is_empty() {
                    let Some(Message::ToolResult(result)) = messages.get(index) else {
                        return Some(call_index);
                    };
                    if !pending.remove(result.tool_call_id()) {
                        return Some(index);
                    }
                    index += 1;
                }
            }
            _ => index += 1,
        }
    }
    None
}

fn splice(conversation: &Conversation, kept: &[usize], summary: AssistantMessage) -> Conversation {
    let mut compacted = match conversation.system_prompt() {
        Some(prompt) => Conversation::new().with_system_prompt(prompt),
        None => Conversation::new(),
    };
    compacted.replace_messages(splice_messages(conversation.messages(), kept, summary));
    compacted
}

fn splice_messages(
    messages: &[Message],
    kept: &[usize],
    summary: AssistantMessage,
) -> Vec<Message> {
    let first_dropped = (0..messages.len()).find(|index| kept.binary_search(index).is_err());
    let mut out = Vec::with_capacity(kept.len() + 1);
    for (index, message) in messages.iter().enumerate() {
        if Some(index) == first_dropped {
            out.push(Message::Assistant(summary.clone()));
        }
        if kept.binary_search(&index).is_ok() {
            out.push(message.clone());
        }
    }
    if first_dropped.is_none() {
        out.insert(0, Message::Assistant(summary));
    }
    out
}

fn summarizer_messages(
    conversation: &Conversation,
    tail_start: usize,
    kept: &[usize],
) -> Vec<Message> {
    let messages = conversation.messages();
    let latest_system = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, message)| matches!(message, Message::System(_)))
        .map(|(index, message)| (index, (*message).clone()));
    let mut summary_messages =
        Vec::with_capacity(tail_start + usize::from(latest_system.is_some()));
    for (index, message) in messages.iter().take(tail_start).enumerate() {
        if kept.binary_search(&index).is_err()
            || latest_system
                .as_ref()
                .is_some_and(|(system_index, _)| *system_index == index)
        {
            summary_messages.push((*message).clone());
        }
    }
    if let Some((index, system)) = latest_system
        && index >= tail_start
    {
        summary_messages.push(system);
    }
    summary_messages
}

fn summarizer_conversation(messages: &[Message]) -> Conversation {
    let mut request = Conversation::new().with_system_prompt(COMPACTION_SYSTEM.trim());
    for message in messages {
        request.push_message(message.clone());
    }
    request.push_user_message(UserMessage::new(COMPACTION_USER));
    request
}

fn summarize_tool() -> ToolSpec {
    ToolSpec::new(
        SUMMARIZE_TOOL,
        "Submit the compaction handoff summary.",
        json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The handoff summary in the required Markdown structure."
                }
            },
            "required": ["summary"]
        }),
    )
}

fn summary_from_reply<E>(message: &AssistantMessage) -> Result<String, CompactError<E>> {
    let call = message
        .tool_calls()
        .iter()
        .find(|call| call.name() == SUMMARIZE_TOOL)
        .ok_or(CompactError::MissingSummary)?;
    let summary = match call.arguments().get("summary") {
        Some(Value::String(summary)) => summary.trim(),
        _ => return Err(CompactError::MissingSummary),
    };
    if summary.is_empty() {
        Err(CompactError::EmptySummary)
    } else {
        Ok(summary.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Provider, SystemMessage, ToolCall, ToolResult, ToolSpec};
    use serde_json::json;
    use std::error::Error;
    use std::future::{Future, ready};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[test]
    fn empty_conversation_has_nothing_to_compact() {
        assert_eq!(plan(&[], 20_000), None);
    }

    #[test]
    fn nothing_dropped_when_budget_covers_a_safe_conversation() {
        let messages = vec![
            Message::User(UserMessage::new("hello")),
            Message::Assistant(AssistantMessage::new("hi")),
        ];
        assert_eq!(plan(&messages, u64::MAX), None);
    }

    #[test]
    fn keeps_a_recent_token_bounded_suffix_and_latest_user() {
        let messages = vec![
            Message::User(UserMessage::new("old")),
            Message::Assistant(AssistantMessage::new("x".repeat(300))),
            Message::User(UserMessage::new("latest instructions")),
            Message::Assistant(AssistantMessage::new("recent answer")),
        ];
        let budget = messages[3].estimate_tokens();
        assert_eq!(plan(&messages, budget), Some(vec![2, 3]));
    }

    #[test]
    fn preserves_latest_ordinary_user_and_steers_when_tail_is_empty() {
        let messages = vec![
            Message::User(UserMessage::new("do this exactly")),
            Message::Assistant(AssistantMessage::new("x".repeat(300))),
            Message::User(UserMessage::new("and this").with_steered(true)),
            Message::Assistant(AssistantMessage::new("y".repeat(300))),
        ];
        assert_eq!(plan(&messages, 1), Some(vec![0, 2]));
    }

    #[test]
    fn does_not_start_a_tail_with_an_orphaned_tool_result() {
        let messages = vec![
            Message::User(UserMessage::new("run it")),
            Message::Assistant(
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "call-1",
                    "read_file",
                    json!({"path": "large"}),
                )]),
            ),
            Message::ToolResult(ToolResult::success("call-1", json!("result"))),
            Message::Assistant(AssistantMessage::new("done")),
        ];
        let budget = messages[2].estimate_tokens() + messages[3].estimate_tokens();
        assert_eq!(plan(&messages, budget), Some(vec![0, 3]));
    }

    #[test]
    fn omits_incomplete_tool_exchanges_from_the_verbatim_tail() {
        let messages = vec![
            Message::User(UserMessage::new("first request")),
            Message::Assistant(
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "unfinished",
                    "read_file",
                    json!({}),
                )]),
            ),
            Message::User(UserMessage::new("latest request")),
            Message::Assistant(AssistantMessage::new("completed answer")),
        ];
        assert_eq!(plan(&messages, u64::MAX), Some(vec![2, 3]));
    }

    #[test]
    fn does_not_duplicate_last_user_already_in_the_tail() {
        let messages = vec![
            Message::User(UserMessage::new("old")),
            Message::Assistant(AssistantMessage::new("x".repeat(300))),
            Message::User(UserMessage::new("new")),
            Message::Assistant(AssistantMessage::new("recent")),
        ];
        let budget = messages[2].estimate_tokens() + messages[3].estimate_tokens();
        let kept = plan(&messages, budget).unwrap();
        assert_eq!(kept, vec![2, 3]);
        let spliced = splice_messages(&messages, &kept, AssistantMessage::new("SUM"));
        assert_eq!(
            spliced
                .iter()
                .filter(|message| matches!(message, Message::User(user) if user.content() == "new"))
                .count(),
            1
        );
    }

    #[test]
    fn inserts_summary_at_first_dropped_index() {
        let messages = vec![
            Message::User(UserMessage::new("old")),
            Message::Assistant(AssistantMessage::new("old answer")),
            Message::User(UserMessage::new("new")),
        ];
        let spliced = splice_messages(&messages, &[2], AssistantMessage::new("SUM"));
        assert!(matches!(&spliced[0], Message::Assistant(a) if a.content() == "SUM"));
        assert!(matches!(&spliced[1], Message::User(u) if u.content() == "new"));
        assert_eq!(spliced.len(), 2);
    }

    #[test]
    fn mid_exchange_steer_inserts_summary_between_user_and_steer() {
        let messages = vec![
            Message::User(UserMessage::new("latest")),
            Message::Assistant(
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "call-1",
                    "read",
                    json!({}),
                )]),
            ),
            Message::User(UserMessage::new("nudge").with_steered(true)),
            Message::ToolResult(ToolResult::success("call-1", json!("ok"))),
            Message::Assistant(AssistantMessage::new("done")),
        ];
        let kept = plan(&messages, u64::MAX).unwrap();
        assert_eq!(kept, vec![0, 2, 4]);
        let spliced = splice_messages(&messages, &kept, AssistantMessage::new("SUM"));
        assert!(matches!(&spliced[0], Message::User(u) if u.content() == "latest" && !u.steered()));
        assert!(matches!(&spliced[1], Message::Assistant(a) if a.content() == "SUM"));
        assert!(matches!(&spliced[2], Message::User(u) if u.steered()));
        assert!(matches!(&spliced[3], Message::Assistant(a) if a.content() == "done"));
    }

    #[test]
    fn interleaved_tool_exchange_is_not_a_valid_tail() {
        let messages = vec![
            Message::User(UserMessage::new("run both")),
            Message::Assistant(AssistantMessage::new("").with_tool_calls(vec![
                ToolCall::new("call-1", "one", json!({})),
                ToolCall::new("call-2", "two", json!({})),
            ])),
            Message::ToolResult(ToolResult::success("call-1", json!("one"))),
            Message::User(UserMessage::new("keep going").with_steered(true)),
            Message::ToolResult(ToolResult::success("call-2", json!("two"))),
            Message::Assistant(AssistantMessage::new("done")),
        ];
        assert_eq!(plan(&messages, u64::MAX), Some(vec![0, 3, 5]));
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestError(&'static str);

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl Error for TestError {}

    fn summary_reply(summary: &str) -> AssistantMessage {
        AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
            "sum-1",
            SUMMARIZE_TOOL,
            json!({ "summary": summary }),
        )])
    }

    #[derive(Clone)]
    struct FakeProvider {
        reply: Result<AssistantMessage, TestError>,
        created: Arc<AtomicUsize>,
        completes: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<(usize, Conversation)>>>,
    }

    impl FakeProvider {
        fn with_summary(summary: &str) -> Self {
            Self::new(Ok(summary_reply(summary)))
        }

        fn new(reply: Result<AssistantMessage, TestError>) -> Self {
            Self {
                reply,
                created: Arc::new(AtomicUsize::new(0)),
                completes: Arc::new(AtomicUsize::new(0)),
                seen: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl Provider for FakeProvider {
        type Model = ();
        type Error = TestError;
        type Continuation = ();

        fn supports_vision(&self, _model: &Self::Model) -> bool {
            false
        }

        fn create_continuation(&self, _model: &Self::Model) -> Self::Continuation {
            self.created.fetch_add(1, Ordering::SeqCst);
        }

        fn complete(
            &self,
            _continuation: &Self::Continuation,
            _model: &Self::Model,
            conversation: &Conversation,
            tools: &[ToolSpec],
        ) -> impl Future<Output = Result<(AssistantMessage, Self::Continuation), Self::Error>> + Send
        {
            self.completes.fetch_add(1, Ordering::SeqCst);
            self.seen
                .lock()
                .unwrap()
                .push((tools.len(), conversation.clone()));
            ready(self.reply.clone().map(|message| (message, ())))
        }
    }

    fn history_needing_compact() -> Conversation {
        let mut conversation = Conversation::new().with_system_prompt("be helpful");
        conversation.push_user_message(UserMessage::new("old task"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("x".repeat(300))));
        conversation.push_user_message(UserMessage::new("latest task"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("y".repeat(300))));
        conversation.push_user_message(UserMessage::new("nudge").with_steered(true));
        conversation.push_message(Message::Assistant(AssistantMessage::new("recent")));
        conversation
    }

    fn tiny_tail() -> CompactOptions {
        CompactOptions {
            keep_tail_tokens: 1,
            ..CompactOptions::default()
        }
    }

    fn last_message_tail(conversation: &Conversation) -> CompactOptions {
        CompactOptions {
            keep_tail_tokens: conversation.messages().last().unwrap().estimate_tokens(),
            ..CompactOptions::default()
        }
    }

    #[tokio::test]
    async fn compact_uses_a_fresh_one_shot_summarizer_request() {
        let provider = FakeProvider::with_summary("SUM");
        let conversation = history_needing_compact();
        let compacted = provider
            .compact(&(), &conversation, last_message_tail(&conversation))
            .await
            .unwrap();

        assert_eq!(provider.created.load(Ordering::SeqCst), 1);
        assert_eq!(provider.completes.load(Ordering::SeqCst), 1);
        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, 1);
        assert_ne!(seen[0].1, conversation);
        assert_eq!(seen[0].1.system_prompt(), Some(COMPACTION_SYSTEM.trim()));
        assert_ne!(seen[0].1.system_prompt(), conversation.system_prompt());
        let request = seen[0].1.messages();
        assert!(
            request
                .iter()
                .any(|message| matches!(message, Message::User(u) if u.content() == "old task"))
        );
        assert!(
            !request
                .iter()
                .any(|message| matches!(message, Message::Assistant(a) if a.content() == "recent"))
        );
        assert!(matches!(
            request.last(),
            Some(Message::User(user)) if user.content() == COMPACTION_USER
        ));

        assert_eq!(compacted.system_prompt(), Some("be helpful"));
        assert!(matches!(
            compacted.messages().first(),
            Some(Message::Assistant(a)) if a.content() == "SUM"
        ));
        assert!(
            compacted
                .messages()
                .iter()
                .any(|message| matches!(message, Message::User(u) if u.content() == "latest task"))
        );
        assert!(
            compacted
                .messages()
                .iter()
                .any(|message| matches!(message, Message::User(u) if u.steered()))
        );
        assert!(
            compacted
                .messages()
                .iter()
                .any(|message| matches!(message, Message::Assistant(a) if a.content() == "recent"))
        );
        assert!(
            !compacted
                .messages()
                .iter()
                .any(|message| matches!(message, Message::User(u) if u.content() == "old task"))
        );
    }

    #[tokio::test]
    async fn portable_compaction_keeps_latest_inline_system_context_verbatim() {
        let provider = FakeProvider::with_summary("SUM");
        let mut conversation = Conversation::new().with_system_prompt("stable instructions");
        conversation.push_user_message(UserMessage::new("old task"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("x".repeat(300))));
        conversation.push_message(Message::System(SystemMessage::new(
            "latest host snapshot: preserve these current notes and references",
        )));
        conversation.push_user_message(UserMessage::new("latest task"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("recent answer")));

        let compacted = provider
            .compact(&(), &conversation, last_message_tail(&conversation))
            .await
            .unwrap();

        let seen = provider.seen.lock().unwrap();
        assert!(seen[0].1.messages().iter().any(
            |message| matches!(message, Message::System(system) if system.content().starts_with("latest host snapshot:"))
        ));
        assert!(compacted.messages().iter().any(
            |message| matches!(message, Message::System(system) if system.content() == "latest host snapshot: preserve these current notes and references")
        ));
    }

    #[tokio::test]
    async fn compact_sends_an_existing_summary_as_a_normal_message() {
        let provider = FakeProvider::with_summary("SUM");
        let mut conversation = Conversation::new();
        conversation.push_message(Message::Assistant(AssistantMessage::new("old summary")));
        conversation.push_message(Message::User(UserMessage::new("more work")));
        conversation.push_message(Message::Assistant(AssistantMessage::new("x".repeat(300))));
        conversation.push_message(Message::User(UserMessage::new("latest")));
        conversation.push_message(Message::Assistant(AssistantMessage::new("recent")));

        let _ = provider
            .compact(&(), &conversation, last_message_tail(&conversation))
            .await
            .unwrap();

        let seen = provider.seen.lock().unwrap();
        assert!(matches!(
            seen[0].1.messages().first(),
            Some(Message::Assistant(a)) if a.content() == "old summary"
        ));
        assert!(
            !seen[0]
                .1
                .messages()
                .iter()
                .any(|message| matches!(message, Message::Assistant(a) if a.content() == "recent"))
        );
    }

    #[tokio::test]
    async fn compact_does_not_mutate_the_original_conversation() {
        let provider = FakeProvider::with_summary("SUM");
        let conversation = history_needing_compact();
        let before = conversation.clone();
        let _ = provider
            .compact(&(), &conversation, tiny_tail())
            .await
            .unwrap();
        assert_eq!(conversation, before);
    }

    #[tokio::test]
    async fn empty_summary_is_an_error() {
        let provider = FakeProvider::with_summary("  \n");
        let error = provider
            .compact(&(), &history_needing_compact(), tiny_tail())
            .await
            .unwrap_err();
        assert_eq!(error, CompactError::EmptySummary);
        assert_eq!(provider.completes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn text_reply_without_summarize_tool_is_an_error() {
        let provider = FakeProvider::new(Ok(AssistantMessage::new("SUM")));
        let error = provider
            .compact(&(), &history_needing_compact(), tiny_tail())
            .await
            .unwrap_err();
        assert_eq!(error, CompactError::MissingSummary);
        assert_eq!(provider.completes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn provider_failure_is_wrapped() {
        let provider = FakeProvider::new(Err(TestError("nope")));
        let error = provider
            .compact(&(), &history_needing_compact(), tiny_tail())
            .await
            .unwrap_err();
        assert_eq!(error, CompactError::Provider(TestError("nope")));
    }

    #[tokio::test]
    async fn nothing_to_compact_does_not_call_the_provider() {
        let provider = FakeProvider::with_summary("SUM");
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("hello"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("hi")));
        let error = provider
            .compact(&(), &conversation, CompactOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error, CompactError::NothingToCompact);
        assert_eq!(provider.created.load(Ordering::SeqCst), 0);
        assert_eq!(provider.completes.load(Ordering::SeqCst), 0);
    }
}
