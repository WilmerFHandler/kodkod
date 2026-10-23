use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use event_listener::Event;

use crate::UserMessage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admission {
    Fresh,
    Open,
    Closed,
}

#[derive(Debug)]
struct SteeringMailbox {
    admission: Admission,
    pending: VecDeque<UserMessage>,
}

#[derive(Debug)]
struct TaskControlInner {
    cancelled: AtomicBool,
    compact_requested: AtomicBool,
    cancellation: Event,
    steering_available: Event,
    steering: Mutex<SteeringMailbox>,
}

/// A steering message rejected because this one-turn control is not open.
#[derive(Debug)]
pub struct SteerError {
    message: UserMessage,
}

impl SteerError {
    pub fn into_message(self) -> UserMessage {
        self.message
    }
    pub fn message(&self) -> &UserMessage {
        &self.message
    }
}

impl fmt::Display for SteerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "agent turn is not accepting steering")
    }
}
impl std::error::Error for SteerError {}

/// One-execution handle for cancellation and atomic FIFO steering.
#[derive(Debug, Clone)]
pub struct TaskControl {
    inner: Arc<TaskControlInner>,
}

impl TaskControl {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(TaskControlInner {
                cancelled: AtomicBool::new(false),
                compact_requested: AtomicBool::new(false),
                cancellation: Event::new(),
                steering_available: Event::new(),
                steering: Mutex::new(SteeringMailbox {
                    admission: Admission::Fresh,
                    pending: VecDeque::new(),
                }),
            }),
        }
    }

    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::SeqCst) {
            self.close();
            self.inner.cancellation.notify(usize::MAX);
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Request compaction at the next round boundary.
    ///
    /// The flag is one-shot: the agent consumes it before the next `complete`.
    /// A request after the turn has closed is ignored.
    pub fn request_compact(&self) {
        self.inner.compact_requested.store(true, Ordering::SeqCst);
    }

    pub(crate) fn take_compact_request(&self) -> bool {
        self.inner.compact_requested.swap(false, Ordering::SeqCst)
    }

    pub async fn cancelled(&self) {
        loop {
            if self.is_cancelled() {
                return;
            }
            let listener = self.inner.cancellation.listen();
            if self.is_cancelled() {
                return;
            }
            listener.await;
        }
    }

    /// Queue a steer iff its execution turn is currently open.
    pub fn steer(&self, message: UserMessage) -> Result<(), SteerError> {
        self.enqueue_steer(message, false)
    }

    /// Queue a correction for this one execution, including before its stream starts.
    /// Hosts must bind the control to the intended turn before exposing it. A
    /// cancelled or completed execution never accepts input, and opening the
    /// execution preserves queued corrections in FIFO order.
    pub fn queue_steer(&self, message: UserMessage) -> Result<(), SteerError> {
        self.enqueue_steer(message, true)
    }

    fn enqueue_steer(&self, message: UserMessage, before_start: bool) -> Result<(), SteerError> {
        let mut mailbox = self
            .inner
            .steering
            .lock()
            .expect("steering mailbox poisoned");
        let accepting = mailbox.admission == Admission::Open
            || (before_start && mailbox.admission == Admission::Fresh);
        if !accepting || self.is_cancelled() {
            return Err(SteerError { message });
        }
        mailbox.pending.push_back(message.with_steered(true));
        drop(mailbox);
        self.inner.steering_available.notify(usize::MAX);
        Ok(())
    }

    pub(crate) async fn steering_ready(&self) {
        loop {
            let listener = self.inner.steering_available.listen();
            if !self
                .inner
                .steering
                .lock()
                .expect("steering mailbox poisoned")
                .pending
                .is_empty()
            {
                return;
            }
            listener.await;
        }
    }

    pub(crate) fn open(&self) -> Result<(), ()> {
        let mut mailbox = self
            .inner
            .steering
            .lock()
            .expect("steering mailbox poisoned");
        if mailbox.admission != Admission::Fresh || self.is_cancelled() {
            mailbox.admission = Admission::Closed;
            return Err(());
        }
        mailbox.admission = Admission::Open;
        Ok(())
    }

    /// Drain steering as one operation ordered against cancellation.
    ///
    /// Holding the mailbox lock while observing cancellation means a steer is
    /// either included in this batch or rejected/preceded by cancellation.
    pub(crate) fn drain_pending_steers_unless_cancelled(&self) -> Option<Vec<UserMessage>> {
        let mut mailbox = self
            .inner
            .steering
            .lock()
            .expect("steering mailbox poisoned");
        if self.is_cancelled() {
            return None;
        }
        Some(mailbox.pending.drain(..).collect())
    }

    /// Atomically close if empty. False means a steer linearized first.
    pub(crate) fn close_if_empty(&self) -> bool {
        let mut mailbox = self
            .inner
            .steering
            .lock()
            .expect("steering mailbox poisoned");
        if mailbox.pending.is_empty() {
            mailbox.admission = Admission::Closed;
            true
        } else {
            false
        }
    }

    pub(crate) fn close(&self) {
        self.inner
            .steering
            .lock()
            .expect("steering mailbox poisoned")
            .admission = Admission::Closed;
    }
}

impl Default for TaskControl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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

    #[test]
    fn steer_and_close_are_atomic_and_return_rejected_message() {
        let control = TaskControl::new();
        let rejected = UserMessage::new("early");
        assert_eq!(
            control
                .steer(rejected)
                .unwrap_err()
                .into_message()
                .content(),
            "early"
        );
        control.open().unwrap();
        control.steer(UserMessage::new("one")).unwrap();
        control.steer(UserMessage::new("two")).unwrap();
        assert!(!control.close_if_empty());
        let drained = control.drain_pending_steers_unless_cancelled().unwrap();
        assert_eq!(
            drained.iter().map(UserMessage::content).collect::<Vec<_>>(),
            ["one", "two"]
        );
        assert!(drained.iter().all(UserMessage::steered));
        assert!(control.close_if_empty());
        assert_eq!(
            control
                .steer(UserMessage::new("late"))
                .unwrap_err()
                .into_message()
                .content(),
            "late"
        );
    }

    #[test]
    fn queued_steering_survives_start_and_never_reopens_a_closed_control() {
        let control = TaskControl::new();
        control
            .queue_steer(UserMessage::new("before startup"))
            .unwrap();
        assert!(control.steer(UserMessage::new("open only")).is_err());
        control.open().unwrap();
        control
            .queue_steer(UserMessage::new("after startup"))
            .unwrap();
        let messages = control.drain_pending_steers_unless_cancelled().unwrap();
        assert_eq!(
            messages
                .iter()
                .map(UserMessage::content)
                .collect::<Vec<_>>(),
            ["before startup", "after startup"]
        );
        assert!(messages.iter().all(UserMessage::steered));
        assert!(control.close_if_empty());
        assert!(control.queue_steer(UserMessage::new("too late")).is_err());
        assert!(control.open().is_err());
    }

    #[test]
    fn cancellation_before_start_prevents_queued_steering_delivery() {
        let control = TaskControl::new();
        control.queue_steer(UserMessage::new("pending")).unwrap();
        control.cancel();
        assert!(control.queue_steer(UserMessage::new("cancelled")).is_err());
        assert!(control.drain_pending_steers_unless_cancelled().is_none());
        assert!(control.open().is_err());
    }

    #[test]
    fn compact_request_is_taken_once() {
        let control = TaskControl::new();
        assert!(!control.take_compact_request());
        control.request_compact();
        control.request_compact();
        assert!(control.take_compact_request());
        assert!(!control.take_compact_request());
    }
}
