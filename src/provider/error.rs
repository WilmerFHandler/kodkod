use std::fmt;

use serde::{Deserialize, Serialize};

/// What went wrong when talking to a model backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderErrorKind {
    /// Transport failure (DNS, connection reset, timeout, etc.).
    Request,
    /// HTTP response with a non-success status.
    Http,
    /// Success HTTP status but the body could not be parsed or validated.
    Response,
    /// Other failures (configuration, serialization before the request, etc.).
    #[serde(other)]
    Other,
}

impl Default for ProviderErrorKind {
    fn default() -> Self {
        Self::Other
    }
}

/// Error from a [`Provider::complete`](super::Provider::complete) call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderError {
    message: String,
    #[serde(default)]
    kind: ProviderErrorKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    status_code: Option<u16>,
}

impl ProviderError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::other(message)
    }

    pub fn request(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ProviderErrorKind::Request,
            status_code: None,
        }
    }

    pub fn http(status_code: u16, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ProviderErrorKind::Http,
            status_code: Some(status_code),
        }
    }

    pub fn response(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ProviderErrorKind::Response,
            status_code: None,
        }
    }

    pub fn other(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ProviderErrorKind::Other,
            status_code: None,
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn kind(&self) -> ProviderErrorKind {
        self.kind
    }

    pub fn status_code(&self) -> Option<u16> {
        self.status_code
    }

    /// Whether a transient failure might succeed on a later attempt.
    pub fn is_retryable(&self) -> bool {
        match self.kind {
            ProviderErrorKind::Request => true,
            ProviderErrorKind::Http => self
                .status_code
                .is_some_and(|code| matches!(code, 408 | 429 | 500 | 502 | 503 | 504)),
            ProviderErrorKind::Response | ProviderErrorKind::Other => false,
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderError {}
