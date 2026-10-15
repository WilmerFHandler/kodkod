use serde::{Deserialize, Serialize};

use kodkod_core::ContextBudget;

/// Optional per-request controls carried by an OpenAI model value.
///
/// The provider forwards these strings without validating them, so callers can
/// use values added by a future API version before Kodkod publishes a new
/// release. The owned values also allow settings loaded from persistent data to
/// survive a serialize/deserialize round trip unchanged.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiRequestOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

/// Metadata for a model served by an OpenAI-compatible API.
pub trait OpenAiModel: Sync {
    fn id(&self) -> &str;
    fn supports_vision(&self) -> bool;
    fn request_options(&self) -> OpenAiRequestOptions {
        OpenAiRequestOptions::default()
    }

    /// Effective input-token budget for this model, when the caller has reviewed one.
    fn context_budget(&self) -> Option<ContextBudget> {
        None
    }
}

impl OpenAiModel for String {
    fn id(&self) -> &str {
        self
    }

    fn supports_vision(&self) -> bool {
        false
    }
}

impl OpenAiModel for str {
    fn id(&self) -> &str {
        self
    }

    fn supports_vision(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::OpenAiRequestOptions;
    use serde_json::json;

    #[test]
    fn request_options_round_trip_unknown_values() {
        let options = OpenAiRequestOptions {
            reasoning_effort: Some("future-effort".into()),
            service_tier: Some("burst".into()),
        };

        let serialized = serde_json::to_value(&options).unwrap();
        assert_eq!(
            serialized,
            json!({
                "reasoning_effort": "future-effort",
                "service_tier": "burst"
            })
        );
        let restored: OpenAiRequestOptions = serde_json::from_value(serialized).unwrap();
        assert_eq!(restored, options);
    }

    #[test]
    fn request_options_omit_unset_values() {
        assert_eq!(
            serde_json::to_value(OpenAiRequestOptions::default()).unwrap(),
            json!({})
        );
    }
}
