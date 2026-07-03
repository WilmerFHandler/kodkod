use lynx_agent::{ProviderError, ProviderErrorKind};

use crate::Retryable;

impl Retryable for ProviderError {
    fn is_retryable(&self) -> bool {
        match self.kind() {
            ProviderErrorKind::Request => true,
            ProviderErrorKind::Http => self
                .status_code()
                .is_some_and(|code| matches!(code, 408 | 429 | 500 | 502 | 503 | 504)),
            ProviderErrorKind::Response | ProviderErrorKind::Other => false,
        }
    }
}
