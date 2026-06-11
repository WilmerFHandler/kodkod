use std::fmt;

use crate::ProviderError;

#[derive(Debug)]
pub enum AgentError {
    Provider(ProviderError),
    MaxToolRoundsExceeded { max: usize },
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(error) => write!(f, "provider failed: {error}"),
            Self::MaxToolRoundsExceeded { max } => {
                write!(f, "assistant requested tools for more than {max} rounds")
            }
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::MaxToolRoundsExceeded { .. } => None,
        }
    }
}

impl From<ProviderError> for AgentError {
    fn from(error: ProviderError) -> Self {
        Self::Provider(error)
    }
}
