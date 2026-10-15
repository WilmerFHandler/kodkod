//! Durable active model context, independent of transcript storage and UI.
use crate::{AgentError, AgentEvent, CompactOptions, Conversation, Message, Provider, TokenUsage};
use futures::Stream;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fmt, pin::Pin, sync::Arc};

/// Effective input budget, after reserving room for output and request uncertainty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextBudget {
    pub max_input_tokens: u64,
}

/// Automatic maintenance policy. Unknown model budgets do not invent a window size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionPolicy {
    pub budget: Option<ContextBudget>,
    pub keep_tail_tokens: u64,
}
impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            budget: None,
            keep_tail_tokens: crate::DEFAULT_KEEP_TAIL_TOKENS,
        }
    }
}
impl CompactionPolicy {
    pub fn options(self) -> CompactOptions {
        CompactOptions {
            keep_tail_tokens: self.keep_tail_tokens,
        }
    }
}

/// Hash-chained prefix of the original transcript. Compaction never advances this cursor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCursor {
    pub position: usize,
    pub fingerprint: String,
}
impl ContextCursor {
    pub fn for_history(history: &Conversation) -> Self {
        let mut cursor = Self::default();
        for message in history.messages() {
            cursor.append(message);
        }
        cursor
    }
    pub(crate) fn append(&mut self, message: &Message) {
        let mut hash = Sha256::new();
        hash.update(self.fingerprint.as_bytes());
        hash.update(serde_json::to_vec(message).expect("messages are JSON values"));
        self.fingerprint = format!("{:x}", hash.finalize());
        self.position += 1;
    }
}

/// Accepted replacement input and provider state form one checkpoint.
/// Serialization is available only when the provider state supports it. No credentials belong here.
#[derive(Serialize, Deserialize)]
pub struct ContextCheckpoint<C> {
    pub version: u32,
    pub binding: String,
    pub cursor: ContextCursor,
    pub conversation: Conversation,
    pub continuation: Arc<C>,
    #[serde(default)]
    pub measurement: Option<ContextMeasurement>,
}
impl<C> Clone for ContextCheckpoint<C> {
    fn clone(&self) -> Self {
        Self {
            version: self.version,
            binding: self.binding.clone(),
            cursor: self.cursor.clone(),
            conversation: self.conversation.clone(),
            continuation: self.continuation.clone(),
            measurement: self.measurement,
        }
    }
}
impl<C> fmt::Debug for ContextCheckpoint<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextCheckpoint")
            .field("version", &self.version)
            .field("cursor", &self.cursor)
            .field("active_messages", &self.conversation.messages().len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextMeasurement {
    pub tokens: u64,
    pub estimated_tokens: u64,
}

/// A provider's explicit compaction result, including authoritative native replay state.
pub struct ContextCompaction<C> {
    pub conversation: Conversation,
    pub continuation: C,
    pub usage: TokenUsage,
}

/// One loop update. Persist accepted events and their checkpoint before polling again.
/// Preview/status events have no new checkpoint. Tool intents are still the host's responsibility.
pub struct AgentUpdate<C> {
    pub event: AgentEvent,
    pub checkpoint: Option<ContextCheckpoint<C>>,
}
pub type ContextTask<'a, C, E> =
    Pin<Box<dyn Stream<Item = Result<AgentUpdate<C>, AgentError<E>>> + Send + 'a>>;

/// Active messages and matching provider state, retained across user turns.
pub struct AgentContext<P: Provider> {
    pub(crate) conversation: Conversation,
    pub(crate) continuation: Arc<P::Continuation>,
    pub(crate) cursor: ContextCursor,
    pub(crate) binding: String,
    pub(crate) measurement: Option<ContextMeasurement>,
    pub(crate) managed: bool,
}
impl<P: Provider> AgentContext<P> {
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }
    /// Update current instructions without replacing accepted messages or native history.
    pub fn set_system_prompt(&mut self, prompt: Option<&str>) {
        self.conversation.set_system_prompt(prompt);
    }
    pub fn checkpoint(&self) -> Result<ContextCheckpoint<P::Continuation>, String> {
        if !self.managed {
            return Err(
                "Legacy split context must be explicitly rebased before checkpointing".into(),
            );
        }
        Ok(ContextCheckpoint {
            version: 1,
            binding: self.binding.clone(),
            cursor: self.cursor.clone(),
            conversation: self.conversation.clone(),
            continuation: self.continuation.clone(),
            measurement: self.measurement,
        })
    }
    pub fn estimated_continuation_tokens(&self) -> u64 {
        P::estimate_continuation_tokens(&self.continuation)
    }
    pub fn estimated_current_context(
        &self,
        conversation: &Conversation,
        tools: &[crate::ToolSpec],
    ) -> crate::ContextEstimate {
        crate::ContextEstimate::heuristic(
            conversation
                .estimate_tokens(tools)
                .saturating_add(self.estimated_continuation_tokens()),
        )
    }
}

#[cfg(test)]
#[path = "context_tests.rs"]
mod tests;
