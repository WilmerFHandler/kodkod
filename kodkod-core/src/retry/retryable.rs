use crate::retry::RetryAfter;

/// Whether a provider failure might succeed on a later attempt.
pub trait Retryable {
    fn is_retryable(&self) -> bool;

    /// HTTP status from the provider response, when this error represents one.
    fn http_status(&self) -> Option<u16> {
        None
    }

    /// Provider machine code, when the adapter exposes one.
    fn api_code(&self) -> Option<&str> {
        None
    }

    /// A server-provided minimum delay before retrying, when the error includes one.
    ///
    /// This is retry timing metadata, not a quota reset time.
    fn retry_after(&self) -> Option<RetryAfter> {
        None
    }
}
