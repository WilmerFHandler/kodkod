use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::UserMessage;

/// Handle to control an in-flight [`Agent`](super::Agent) run.
///
/// Cheap to clone — all clones share the same cancellation flag and steering
/// mailbox. Pass a `TaskControl` into [`Agent::run`](super::Agent::run) so
/// external callers (e.g. a GUI session) can stop the loop or inject new user
/// messages between rounds.
#[derive(Debug, Clone)]
pub struct TaskControl {
    cancelled: Arc<AtomicBool>,
    steers: Arc<Mutex<VecDeque<UserMessage>>>,
}

impl TaskControl {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            steers: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Request cancellation. The agent loop checks this between rounds.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Queue a user message to inject into the running turn at the next
    /// boundary (between rounds, before the next provider call). Multiple
    /// steers are delivered in order. No-op outside a run — the message just
    /// waits until the loop next checks the mailbox.
    pub fn steer(&self, message: UserMessage) {
        self.steers
            .lock()
            .expect("steer queue poisoned")
            .push_back(message);
    }

    /// Take all queued steer messages, leaving the mailbox empty.
    ///
    /// Called by the agent loop at the top of each round.
    pub fn drain_pending_steers(&self) -> Vec<UserMessage> {
        self.steers
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
