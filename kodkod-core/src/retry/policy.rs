use std::time::Duration;

/// Policy for retrying transient provider failures with exponential backoff.
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

    pub(crate) fn backoff_after_attempt(&self, failed_attempt_index: u32) -> Duration {
        let exp = failed_attempt_index.saturating_sub(1);
        let mut millis = self.initial_backoff.as_millis() as f64;
        for _ in 0..exp {
            millis *= self.backoff_multiplier;
        }
        let capped = millis.min(self.max_backoff.as_millis() as f64);
        Duration::from_millis(capped as u64)
    }
}
