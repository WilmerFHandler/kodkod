use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Handle to control an in-flight [`Agent`](super::Agent) run.
///
/// Cheap to clone — all clones share the same cancellation flag. Pass a
/// `TaskControl` into [`Agent::run`](super::Agent::run) so external callers
/// (e.g. a GUI session) can stop the agent loop between rounds.
#[derive(Debug, Clone)]
pub struct TaskControl {
    cancelled: Arc<AtomicBool>,
}

impl TaskControl {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
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
}

impl Default for TaskControl {
    fn default() -> Self {
        Self::new()
    }
}
