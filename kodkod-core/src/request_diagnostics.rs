//! Opt-in, content-free diagnostics for provider requests.
//!
//! Observers receive counts, safe classifications, keyed fingerprints, request
//! lifecycle events, and provider-reported usage. Prompt and response contents
//! are never part of a [`RequestDiagnostic`].

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::TokenUsage;

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// A process-local identity for one physical provider request attempt.
///
/// Retry wrappers call the provider again for each physical retry, so each
/// attempt has a fresh `request_id` and `attempt_id` is `1` at this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RequestIdentity {
    pub process_id: u32,
    pub request_id: u64,
    pub attempt_id: u64,
}

impl RequestIdentity {
    /// Allocate an identity for a physical request. This is crate-public so
    /// provider implementations can share the same process-wide sequence.
    pub fn next_process_unique() -> Self {
        let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            process_id: std::process::id(),
            request_id,
            attempt_id: 1,
        }
    }
}

/// Whether a request performs normal inference or provider-native compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestOperation {
    Completion,
    Compaction,
}

/// A reviewed request profile classification, without endpoint or account data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestProfile {
    OpenAiCompatible,
    ChatGptPlan,
}

/// A bounded classification for known request option values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestOptionClass {
    None,
    Low,
    Medium,
    High,
    XHigh,
    Auto,
    Default,
    Flex,
    Priority,
    Other,
}

/// A fingerprint result returned by the caller-supplied keyed fingerprinting
/// function. Values are restricted to a bounded URL-safe alphabet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestFingerprint {
    pub byte_len: usize,
    pub value: Option<String>,
}

/// Fingerprints of stable request-prefix components. `input_prefixes` includes
/// up to the first 1, 4, 16, and 64 serialized input items, plus the whole input.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestPromptFingerprints {
    pub instructions: Option<RequestFingerprint>,
    pub tools: Option<RequestFingerprint>,
    pub input_prefixes: Vec<RequestInputPrefixFingerprint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestInputPrefixFingerprint {
    /// `None` identifies the complete input; otherwise this is the item count.
    pub item_count: Option<usize>,
    pub fingerprint: RequestFingerprint,
}

/// Request metadata safe to retain in operational diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// A bounded safe model identifier, or `None` if it contains unsafe data.
    pub model: Option<String>,
    pub profile: RequestProfile,
    pub reasoning_effort: RequestOptionClass,
    pub service_tier: RequestOptionClass,
    pub vision_enabled: bool,
    pub context_estimate_tokens: u64,
    pub context_budget_tokens: Option<u64>,
    pub message_count: usize,
    pub tool_definition_count: usize,
    pub image_count: usize,
    /// Decoded bytes from actual `input_image` items; unknown if a retained
    /// item does not expose a parseable inline data URL.
    pub image_bytes: Option<usize>,
    pub document_count: usize,
    /// Decoded bytes from actual `input_file` items; unknown if bytes are not
    /// present as a parseable inline data URL.
    pub document_bytes: Option<usize>,
    /// Compact JSON byte count for the outbound request body.
    pub serialized_request_bytes: usize,
    /// Compact JSON bytes of opaque continuation output replayed into the input.
    pub continuation_bytes: usize,
    pub prompt_fingerprints: RequestPromptFingerprints,
}

/// Provider failure category that does not include provider error text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "status", rename_all = "snake_case")]
pub enum RequestFailureKind {
    Transport,
    HttpStatus(u16),
    ProviderResponse,
    Protocol,
    Incomplete,
    Unknown,
}

/// Allowlisted classifications from provider prompt-cache diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestCacheReason {
    Hit,
    Miss,
    Write,
    Disabled,
    Ineligible,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestCacheType {
    PromptPrefix,
    Input,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestCacheDiagnostics {
    pub reason: Option<RequestCacheReason>,
    pub cache_type: Option<RequestCacheType>,
}

/// Request-level usage. Cache-write tokens remain separate from `TokenUsage`
/// to preserve source compatibility for existing `TokenUsage` struct literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestUsage {
    pub token_usage: TokenUsage,
    pub cache_write_tokens: Option<u64>,
    pub cache_diagnostics: Option<RequestCacheDiagnostics>,
}

/// A lifecycle event for one physical provider request attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "details", rename_all = "snake_case")]
pub enum RequestDiagnosticEvent {
    Started,
    Headers {
        status: u16,
        provider_request_id: Option<String>,
    },
    Finished {
        duration_micros: u64,
        failure: Option<RequestFailureKind>,
        response_id: Option<String>,
        usage: Option<RequestUsage>,
    },
    /// The observing future or stream was dropped before a terminal event.
    /// This describes local observation only and does not imply remote cancel.
    Abandoned {
        duration_micros: u64,
        failure: RequestFailureKind,
    },
}

/// One metadata-only diagnostic record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestDiagnostic {
    pub identity: RequestIdentity,
    pub operation: RequestOperation,
    /// Present only on `Started`; later events correlate through `identity`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<RequestMetadata>,
    pub event: RequestDiagnosticEvent,
}

/// Optional request lifecycle observer. Implementations should return a keyed
/// digest from `fingerprint` and must not retain its input bytes.
pub trait RequestObserver: Send + Sync {
    fn observe(&self, diagnostic: RequestDiagnostic);

    /// Compute a caller-controlled keyed fingerprint. The default disables
    /// fingerprints, so enabling observation never requires prompt hashing.
    fn fingerprint(&self, _domain: &str, _bytes: &[u8]) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_are_unique_within_a_process() {
        let first = RequestIdentity::next_process_unique();
        let second = RequestIdentity::next_process_unique();
        assert_eq!(first.process_id, second.process_id);
        assert_ne!(first.request_id, second.request_id);
        assert_eq!(first.attempt_id, 1);
    }
}
