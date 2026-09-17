use serde::{Deserialize, Serialize};

/// Reasoning effort requested from a model which supports reasoning controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl ReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        }
    }
}

/// Provider service tier requested independently of the selected model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Auto,
    Default,
    Priority,
    Flex,
}

impl ServiceTier {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Default => "default",
            Self::Priority => "priority",
            Self::Flex => "flex",
        }
    }
}

/// Optional per-request controls carried by an OpenAI model value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiRequestOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
}

/// Metadata for a model served by an OpenAI-compatible API.
pub trait OpenAiModel: Sync {
    fn id(&self) -> &str;
    fn supports_vision(&self) -> bool;
    fn request_options(&self) -> OpenAiRequestOptions {
        OpenAiRequestOptions::default()
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
