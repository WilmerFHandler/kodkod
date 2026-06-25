use std::fmt;

use crate::ProviderError;

#[derive(Debug)]
pub enum AgentError {
    Provider(ProviderError),
    MaxToolRoundsExceeded { max: usize },
    /// The caller requested cancellation via [`TaskControl`](super::TaskControl).
    Cancelled,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(error) => write!(f, "provider failed: {error}"),
            Self::MaxToolRoundsExceeded { max } => {
                write!(f, "assistant requested tools for more than {max} rounds")
            }
            Self::Cancelled => write!(f, "agent run was cancelled"),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::MaxToolRoundsExceeded { .. } | Self::Cancelled => None,
        }
    }
}

impl From<ProviderError> for AgentError {
    fn from(error: ProviderError) -> Self {
        Self::Provider(error)
    }
}
