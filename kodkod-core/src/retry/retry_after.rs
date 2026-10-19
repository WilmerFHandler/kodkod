use std::time::{Duration, SystemTime};

/// A bounded, server-provided minimum delay before another request.
///
/// `BeyondMaximum` means the server supplied a valid delay longer than this
/// library is willing to schedule automatically. It is deliberately distinct
/// from missing or malformed metadata so callers can avoid retrying early.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryAfter {
    Delay(Duration),
    BeyondMaximum,
}

impl RetryAfter {
    /// Maximum delay represented as an automatically schedulable duration.
    pub const MAX_DELAY: Duration = Duration::from_secs(30 * 24 * 60 * 60);

    /// Parse an HTTP `Retry-After` value for a retryable response status.
    ///
    /// Both delta-seconds and HTTP-date forms are supported. Values above
    /// [`MAX_DELAY`](Self::MAX_DELAY) are reported as [`BeyondMaximum`](Self::BeyondMaximum)
    /// rather than shortened, because shortening would retry before the server's
    /// requested minimum delay.
    pub fn from_header(status: u16, value: &str, now: SystemTime) -> Option<Self> {
        if !matches!(status, 429 | 503) {
            return None;
        }

        let value = value.trim();
        if value.is_empty() {
            return None;
        }

        let delay = if value.bytes().all(|byte| byte.is_ascii_digit()) {
            match value.parse::<u64>() {
                Ok(seconds) => Duration::from_secs(seconds),
                Err(_) => return Some(Self::BeyondMaximum),
            }
        } else {
            let date = httpdate::parse_http_date(value).ok()?;
            date.duration_since(now).unwrap_or(Duration::ZERO)
        };

        if delay > Self::MAX_DELAY {
            Some(Self::BeyondMaximum)
        } else {
            Some(Self::Delay(delay))
        }
    }

    /// Return the schedulable delay, or `None` when the server delay exceeds
    /// the automatic scheduling bound.
    pub const fn delay(self) -> Option<Duration> {
        match self {
            Self::Delay(delay) => Some(delay),
            Self::BeyondMaximum => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_delta_seconds_and_bounds_long_delays() {
        let now = SystemTime::UNIX_EPOCH;
        assert_eq!(
            RetryAfter::from_header(429, "120", now),
            Some(RetryAfter::Delay(Duration::from_secs(120)))
        );
        assert_eq!(
            RetryAfter::from_header(503, "2592001", now),
            Some(RetryAfter::BeyondMaximum)
        );
        assert_eq!(
            RetryAfter::from_header(429, "18446744073709551616", now),
            Some(RetryAfter::BeyondMaximum)
        );
    }

    #[test]
    fn parses_http_dates_without_waking_before_the_date() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let future = now + Duration::from_secs(600);
        let value = httpdate::fmt_http_date(future);
        assert_eq!(
            RetryAfter::from_header(429, &value, now),
            Some(RetryAfter::Delay(Duration::from_secs(600)))
        );

        let past = httpdate::fmt_http_date(now - Duration::from_secs(10));
        assert_eq!(
            RetryAfter::from_header(503, &past, now),
            Some(RetryAfter::Delay(Duration::ZERO))
        );
    }

    #[test]
    fn ignores_invalid_headers_and_non_retry_statuses() {
        let now = SystemTime::UNIX_EPOCH;
        assert_eq!(RetryAfter::from_header(401, "10", now), None);
        assert_eq!(RetryAfter::from_header(429, "tomorrow", now), None);
        assert_eq!(RetryAfter::from_header(503, "", now), None);
    }
}
