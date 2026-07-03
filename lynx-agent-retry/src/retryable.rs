/// Whether a failure from a provider call might succeed on a later attempt.
pub trait Retryable {
    fn is_retryable(&self) -> bool;
}
