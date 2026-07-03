/// Whether a provider failure might succeed on a later attempt.
pub trait Retryable {
    fn is_retryable(&self) -> bool;
}
