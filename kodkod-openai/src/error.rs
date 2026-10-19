use std::error::Error;
use std::fmt;

use kodkod_core::{RetryAfter, Retryable};

use crate::CredentialError;

#[derive(Debug)]
pub enum OpenAiError {
    Http(reqwest::Error),
    Api {
        status: u16,
        code: Option<String>,
        message: String,
        retry_after: Option<RetryAfter>,
    },
    ResponseFailed {
        code: Option<String>,
        message: String,
    },
    Json(serde_json::Error),
    EmptyResponse,
    Credentials(CredentialError),
    UnsupportedDocument {
        provider: &'static str,
        mime: String,
    },
    DocumentLimitExceeded {
        provider: &'static str,
        limit_bytes: usize,
    },
    Incomplete {
        reason: String,
    },
    Protocol(String),
}

impl OpenAiError {
    pub(crate) fn retry_after_from_response(
        status: u16,
        response: &reqwest::Response,
    ) -> Option<RetryAfter> {
        let value = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)?
            .to_str()
            .ok()?;
        RetryAfter::from_header(status, value, std::time::SystemTime::now())
    }

    pub(crate) fn is_context_overflow(&self) -> bool {
        match self {
            Self::Api {
                status: 400,
                code: Some(code),
                ..
            }
            | Self::ResponseFailed {
                code: Some(code), ..
            } => code == "context_length_exceeded",
            _ => false,
        }
    }
}

impl fmt::Display for OpenAiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(error) => write!(f, "http request failed: {error}"),
            Self::Api {
                status,
                code,
                message,
                ..
            } => match code {
                Some(code) => write!(f, "api error ({status}, {code}): {message}"),
                None => write!(f, "api error ({status}): {message}"),
            },
            Self::ResponseFailed { code, message } => match code {
                Some(code) => write!(f, "response failed ({code}): {message}"),
                None => write!(f, "response failed: {message}"),
            },
            Self::Json(error) => write!(f, "failed to parse response: {error}"),
            Self::EmptyResponse => f.write_str("chat completion returned no choices"),
            Self::Credentials(error) => write!(f, "credentials unavailable: {error}"),
            Self::UnsupportedDocument { provider, mime } => {
                write!(
                    f,
                    "{provider} does not support inline documents with MIME type {mime}"
                )
            }
            Self::DocumentLimitExceeded {
                provider,
                limit_bytes,
            } => {
                write!(
                    f,
                    "{provider} inline documents exceed the {limit_bytes}-byte request limit"
                )
            }
            Self::Incomplete { reason } => write!(f, "response incomplete: {reason}"),
            Self::Protocol(message) => write!(f, "response protocol error: {message}"),
        }
    }
}

impl From<reqwest::Error> for OpenAiError {
    fn from(error: reqwest::Error) -> Self {
        Self::Http(error)
    }
}

impl From<serde_json::Error> for OpenAiError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<CredentialError> for OpenAiError {
    fn from(error: CredentialError) -> Self {
        Self::Credentials(error)
    }
}

impl Error for OpenAiError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Http(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Credentials(error) => Some(error),
            _ => None,
        }
    }
}

impl Retryable for OpenAiError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Http(error) => error.is_connect() || error.is_timeout() || error.is_request(),
            Self::Api { status, code, .. } => {
                !is_subscription_sharing_quota_limit(code.as_deref())
                    && matches!(*status, 429 | 500 | 502 | 503 | 504)
            }
            Self::ResponseFailed { code, message } => {
                is_retryable_response_failure(code.as_deref(), message)
            }
            Self::Json(_)
            | Self::EmptyResponse
            | Self::Credentials(_)
            | Self::UnsupportedDocument { .. }
            | Self::DocumentLimitExceeded { .. }
            | Self::Incomplete { .. }
            | Self::Protocol(_) => false,
        }
    }

    fn retry_after(&self) -> Option<RetryAfter> {
        match self {
            Self::Api {
                status: 429 | 503,
                retry_after,
                ..
            } => *retry_after,
            _ => None,
        }
    }

    fn http_status(&self) -> Option<u16> {
        match self {
            Self::Api { status, .. } => Some(*status),
            _ => None,
        }
    }

    fn api_code(&self) -> Option<&str> {
        match self {
            Self::Api { code, .. } | Self::ResponseFailed { code, .. } => code.as_deref(),
            _ => None,
        }
    }
}

const KNOWN_PROVIDER_OVERLOAD_MESSAGE: &str =
    "Our servers are currently overloaded. Please try again later.";

fn is_retryable_response_failure(code: Option<&str>, message: &str) -> bool {
    match code {
        Some("server_error" | "overloaded_error") => true,
        Some(_) => false,
        None => message.trim() == KNOWN_PROVIDER_OVERLOAD_MESSAGE,
    }
}

fn is_subscription_sharing_quota_limit(code: Option<&str>) -> bool {
    code == Some("subscription_sharing_usage_limit_exceeded")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_api_errors() {
        assert!(
            OpenAiError::Api {
                status: 429,
                code: None,
                message: "rate limited".into(),
                retry_after: None,
            }
            .is_retryable()
        );
        assert!(
            OpenAiError::Api {
                status: 503,
                code: None,
                message: "unavailable".into(),
                retry_after: None,
            }
            .is_retryable()
        );
        assert!(
            !OpenAiError::Api {
                status: 401,
                code: None,
                message: "unauthorized".into(),
                retry_after: None,
            }
            .is_retryable()
        );
    }

    #[test]
    fn non_retryable_parse_errors() {
        assert!(!OpenAiError::EmptyResponse.is_retryable());
    }

    #[test]
    fn retries_only_known_transient_stream_failures() {
        for code in ["server_error", "overloaded_error"] {
            assert!(
                OpenAiError::ResponseFailed {
                    code: Some(code.into()),
                    message: "provider unavailable".into(),
                }
                .is_retryable()
            );
        }

        for code in [
            "subscription_sharing_usage_limit_exceeded",
            "subscription_sharing_usage_unavailable",
            "subscription_sharing_user_not_eligible",
            "unknown_error",
        ] {
            assert!(
                !OpenAiError::ResponseFailed {
                    code: Some(code.into()),
                    message: KNOWN_PROVIDER_OVERLOAD_MESSAGE.into(),
                }
                .is_retryable()
            );
        }

        assert!(
            OpenAiError::ResponseFailed {
                code: None,
                message: format!(" {KNOWN_PROVIDER_OVERLOAD_MESSAGE} "),
            }
            .is_retryable()
        );
        assert!(
            !OpenAiError::ResponseFailed {
                code: None,
                message: "The server is temporarily unavailable".into(),
            }
            .is_retryable()
        );
        assert!(!OpenAiError::Protocol(KNOWN_PROVIDER_OVERLOAD_MESSAGE.into()).is_retryable());
    }

    #[test]
    fn context_recovery_requires_the_context_length_error_code() {
        assert!(
            OpenAiError::Api {
                status: 400,
                code: Some("context_length_exceeded".into()),
                message: "input exceeded the model context".into(),
                retry_after: None,
            }
            .is_context_overflow()
        );
        assert!(
            OpenAiError::ResponseFailed {
                code: Some("context_length_exceeded".into()),
                message: "input exceeded the model context".into(),
            }
            .is_context_overflow()
        );
        assert!(
            !OpenAiError::Api {
                status: 400,
                code: Some("invalid_api_key".into()),
                message: "unauthorized".into(),
                retry_after: None,
            }
            .is_context_overflow()
        );
        assert!(
            !OpenAiError::Api {
                status: 413,
                code: Some("context_length_exceeded".into()),
                message: "too large".into(),
                retry_after: None,
            }
            .is_context_overflow()
        );
    }

    #[test]
    fn subscription_quota_limit_overrides_http_retry_statuses() {
        let code = "subscription_sharing_usage_limit_exceeded";
        {
            assert!(
                !OpenAiError::Api {
                    status: 429,
                    code: Some(code.into()),
                    message: "plan limit".into(),
                    retry_after: None,
                }
                .is_retryable()
            );
            assert!(
                !OpenAiError::Api {
                    status: 503,
                    code: Some(code.into()),
                    message: "sharing unavailable".into(),
                    retry_after: None,
                }
                .is_retryable()
            );
            assert!(
                !OpenAiError::ResponseFailed {
                    code: Some(code.into()),
                    message: "response failed".into(),
                }
                .is_retryable()
            );
        }

        assert!(
            OpenAiError::Api {
                status: 503,
                code: Some("subscription_sharing_usage_unavailable".into()),
                message: "usage unavailable".into(),
                retry_after: None,
            }
            .is_retryable()
        );
    }

    #[test]
    fn retry_after_is_exposed_as_typed_error_metadata() {
        let retry_after = RetryAfter::Delay(std::time::Duration::from_secs(90));
        let error = OpenAiError::Api {
            status: 429,
            code: Some("rate_limit_exceeded".into()),
            message: "slow down".into(),
            retry_after: Some(retry_after),
        };
        assert_eq!(error.retry_after(), Some(retry_after));
        assert_eq!(
            OpenAiError::Api {
                status: 401,
                code: None,
                message: "unauthorized".into(),
                retry_after: Some(retry_after),
            }
            .retry_after(),
            None
        );
    }
}
