use crate::{AssistantMessage, Conversation, Provider, ToolSpec};

use super::{RetryPolicy, Retryable};

/// Wraps any [`Provider`] whose [`Provider::Error`] implements [`Retryable`].
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
    P::Error: Retryable,
{
    type Model = P::Model;
    type Error = P::Error;

    fn supports_vision(&self, model: &Self::Model) -> bool {
        self.inner.supports_vision(model)
    }

    async fn complete(
        &self,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<AssistantMessage, Self::Error> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    use crate::ProviderError;

    #[derive(Clone, Debug)]
    struct TestModel;

    impl TestModel {
        fn vision(&self) -> bool {
            false
        }
    }

    struct FlakyProvider {
        calls: Arc<AtomicU32>,
        fail_until: u32,
    }

    impl Provider for FlakyProvider {
        type Model = TestModel;
        type Error = ProviderError;

        fn supports_vision(&self, model: &TestModel) -> bool {
            model.vision()
        }

        fn complete(
            &self,
            _model: &TestModel,
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

        let model = TestModel;
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
            type Model = TestModel;
            type Error = ProviderError;

            fn supports_vision(&self, model: &TestModel) -> bool {
                model.vision()
            }

            fn complete(
                &self,
                _model: &TestModel,
                _conversation: &Conversation,
                _tools: &[ToolSpec],
            ) -> impl Future<Output = Result<AssistantMessage, ProviderError>> + Send {
                async { Err(ProviderError::http(401, "unauthorized")) }
            }
        }

        let model = TestModel;
        let provider = RetryProvider::new(Once401);
        let err = provider
            .complete(&model, &Conversation::new(), &[])
            .await
            .unwrap_err();

        assert_eq!(err.status_code(), Some(401));
    }
}
