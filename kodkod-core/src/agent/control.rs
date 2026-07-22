use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use event_listener::Event;

use crate::UserMessage;

#[derive(Debug)]
struct TaskControlInner {
    cancelled: AtomicBool,
    cancellation: Event,
    steers: Mutex<VecDeque<UserMessage>>,
}

/// Handle to control an in-flight [`Agent`](super::Agent) run.
///
/// Cheap to clone — all clones share the same cancellation signal and steering
/// mailbox. Pass a `TaskControl` into [`Agent::run`](super::Agent::run) so
/// external callers (e.g. a GUI session) can request cancellation or inject new
/// user messages between rounds.
#[derive(Debug, Clone)]
pub struct TaskControl {
    inner: Arc<TaskControlInner>,
}

impl TaskControl {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(TaskControlInner {
                cancelled: AtomicBool::new(false),
                cancellation: Event::new(),
                steers: Mutex::new(VecDeque::new()),
            }),
        }
    }

    /// Request cancellation.
    ///
    /// This is idempotent. An agent currently awaiting its provider is woken so
    /// it can drop the provider future and finish with
    /// [`AgentError::Cancelled`](super::AgentError::Cancelled). Active tool
    /// futures are allowed to finish, and cancellation is observed before the
    /// next provider round.
    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::SeqCst) {
            self.inner.cancellation.notify(usize::MAX);
        }
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Wait until cancellation is requested.
    ///
    /// This future is executor-independent, supports any number of concurrent
    /// waiters, and resolves immediately if cancellation was already requested.
    pub async fn cancelled(&self) {
        loop {
            if self.is_cancelled() {
                return;
            }

            let listener = self.inner.cancellation.listen();

            // Cancellation notifications are not retained when no listener is
            // registered, so check again after registering to close that race.
            if self.is_cancelled() {
                return;
            }

            listener.await;
        }
    }

    /// Queue a user message to inject into the running turn at the next
    /// boundary (between rounds, before the next provider call). Multiple
    /// steers are delivered in order. No-op outside a run — the message just
    /// waits until the loop next checks the mailbox.
    pub fn steer(&self, message: UserMessage) {
        // Mark the message as a steering injection so it is treated as part of
        // the current turn rather than starting a new one.
        let message = message.with_steered(true);
        self.inner
            .steers
            .lock()
            .expect("steer queue poisoned")
            .push_back(message);
    }

    /// Take all queued steer messages, leaving the mailbox empty.
    ///
    /// Called by the agent loop at the top of each round.
    pub fn drain_pending_steers(&self) -> Vec<UserMessage> {
        self.inner
            .steers
            .lock()
            .expect("steer queue poisoned")
            .drain(..)
            .collect()
    }
}

impl Default for TaskControl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn cancellation_wakes_all_waiters_and_remains_observable() {
        let control = TaskControl::new();
        let first = control.clone();
        let second = control.clone();

        tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(first.cancelled(), second.cancelled(), async {
                tokio::task::yield_now().await;
                control.cancel();
            });
        })
        .await
        .expect("all cancellation waiters should wake");

        tokio::time::timeout(Duration::from_millis(10), control.cancelled())
            .await
            .expect("cancellation should remain observable");
    }
}
