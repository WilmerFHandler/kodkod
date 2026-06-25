use std::future::Future;
use std::time::Duration;

use crate::{AssistantMessage, Conversation, Model, Provider, ProviderError, ToolSpec};

/// Policy for retrying transient [`ProviderError`]s with exponential backoff.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts for one `complete` call (including the first).
    pub max_attempts: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    /// Multiplier applied to the backoff after each failed retryable attempt.
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    pub fn disabled() -> Self {
        Self {
            max_attempts: 1,
            ..Self::default()
        }
    }

    fn backoff_after_attempt(&self, failed_attempt_index: u32) -> Duration {
        let exp = failed_attempt_index.saturating_sub(1);
        let mut millis = self.initial_backoff.as_millis() as f64;
        for _ in 0..exp {
            millis *= self.backoff_multiplier;
        }
        let capped = millis.min(self.max_backoff.as_millis() as f64);
        Duration::from_millis(capped as u64)
    }
}

/// Wraps any [`Provider`] and retries retryable failures between attempts.
#[derive(Debug, Clone)]
pub struct RetryProvider<P> {
    inner: P,
    policy: RetryPolicy,
}

impl<P> RetryProvider<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            policy: RetryPolicy::default(),
        }
    }

    pub fn with_policy(inner: P, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }

    pub fn inner(&self) -> &P {
        &self.inner
    }

    pub fn into_inner(self) -> P {
        self.inner
    }

    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }
}

impl<P> Provider for RetryProvider<P>
where
    P: Provider + Sync,
{
    fn models(&self) -> Vec<Model> {
        self.inner.models()
    }

    fn complete(
        &self,
        model: &Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
        async move {
            let max = self.policy.max_attempts.max(1);
            let mut attempt = 0u32;

            loop {
                attempt += 1;
                match self.inner.complete(model, conversation, tools).await {
                    Ok(message) => return Ok(message),
                    Err(error) if error.is_retryable() && attempt < max => {
                        tokio::time::sleep(self.policy.backoff_after_attempt(attempt)).await;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct FlakyProvider {
        calls: Arc<AtomicU32>,
        fail_until: u32,
    }

    impl Provider for FlakyProvider {
        fn complete(
            &self,
            _model: &Model,
            _conversation: &Conversation,
            _tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
            let calls = Arc::clone(&self.calls);
            let fail_until = self.fail_until;
            async move {
                let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                if n <= fail_until {
                    Err(ProviderError::http(503, format!("synthetic failure {n}")))
                } else {
                    Ok(AssistantMessage::new("ok"))
                }
            }
        }
    }

    #[tokio::test]
    async fn retries_retryable_http_until_success() {
        let inner = FlakyProvider {
            calls: Arc::new(AtomicU32::new(0)),
            fail_until: 2,
        };
        let provider = RetryProvider::with_policy(
            inner,
            RetryPolicy {
                max_attempts: 4,
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(5),
                backoff_multiplier: 2.0,
            },
        );

        let model = Model::new("m", "M");
        let conversation = Conversation::new();
        let message = provider
            .complete(&model, &conversation, &[])
            .await
            .expect("should succeed after retries");

        assert_eq!(message.content(), "ok");
    }

    #[tokio::test]
    async fn does_not_retry_non_retryable_errors() {
        struct Once401;
        impl Provider for Once401 {
            fn complete(
                &self,
                _model: &Model,
                _conversation: &Conversation,
                _tools: &[ToolSpec],
            ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
                async { Err(ProviderError::http(401, "unauthorized")) }
            }
        }

        let model = Model::new("m", "M");
        let provider = RetryProvider::new(Once401);
        let err = provider
            .complete(&model, &Conversation::new(), &[])
            .await
            .unwrap_err();

        assert_eq!(err.status_code(), Some(401));
    }
}
