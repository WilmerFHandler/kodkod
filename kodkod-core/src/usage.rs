use serde::{Deserialize, Serialize};

/// Authoritative token counts reported by a provider for one completed round.
///
/// `None` means that the provider did not expose that field. It is deliberately
/// different from `Some(0)`: aggregating a round with an unknown field keeps the
/// aggregate unknown instead of silently treating the missing value as zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub reasoning_output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

impl TokenUsage {
    /// A usage value for a provider response which did not expose token counts.
    pub const fn unknown() -> Self {
        Self {
            input_tokens: None,
            cached_input_tokens: None,
            output_tokens: None,
            reasoning_output_tokens: None,
            total_tokens: None,
        }
    }

    /// A zero value suitable for starting an aggregate over provider rounds.
    pub const fn zero() -> Self {
        Self {
            input_tokens: Some(0),
            cached_input_tokens: Some(0),
            output_tokens: Some(0),
            reasoning_output_tokens: Some(0),
            total_tokens: Some(0),
        }
    }

    /// Add one terminal provider-round value to this aggregate.
    ///
    /// Each field is summed independently. If either value is unknown, that
    /// field remains unknown in the aggregate; known values never become
    /// falsely precise because another round omitted the field.
    pub fn add_round(&mut self, round: Self) {
        self.input_tokens = add_known(self.input_tokens, round.input_tokens);
        self.cached_input_tokens = add_known(self.cached_input_tokens, round.cached_input_tokens);
        self.output_tokens = add_known(self.output_tokens, round.output_tokens);
        self.reasoning_output_tokens =
            add_known(self.reasoning_output_tokens, round.reasoning_output_tokens);
        self.total_tokens = add_known(self.total_tokens, round.total_tokens);
    }

    /// Return the aggregate of two independently observed usage values.
    pub fn aggregated(mut self, round: Self) -> Self {
        self.add_round(round);
        self
    }
}

fn add_known(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    Some(left?.saturating_add(right?))
}

/// Why a current-context count is an estimate rather than provider usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextEstimateProvenance {
    /// A conservative byte and metadata heuristic used before the provider responds.
    Heuristic,
}

/// A bounded estimate of the context that would be sent on the next request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextEstimate {
    pub tokens: u64,
    pub provenance: ContextEstimateProvenance,
}

impl ContextEstimate {
    pub const fn heuristic(tokens: u64) -> Self {
        Self {
            tokens,
            provenance: ContextEstimateProvenance::Heuristic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_fields_do_not_become_zero_during_aggregation() {
        let mut usage = TokenUsage::zero();
        usage.add_round(TokenUsage {
            input_tokens: Some(12),
            cached_input_tokens: None,
            output_tokens: Some(4),
            reasoning_output_tokens: Some(1),
            total_tokens: Some(16),
        });

        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.cached_input_tokens, None);
        assert_eq!(usage.output_tokens, Some(4));
        assert_eq!(usage.reasoning_output_tokens, Some(1));
        assert_eq!(usage.total_tokens, Some(16));
    }

    #[test]
    fn known_rounds_sum_independently_and_saturate() {
        let first = TokenUsage {
            input_tokens: Some(u64::MAX),
            cached_input_tokens: Some(2),
            output_tokens: Some(3),
            reasoning_output_tokens: Some(4),
            total_tokens: Some(5),
        };
        let second = TokenUsage {
            input_tokens: Some(1),
            cached_input_tokens: Some(3),
            output_tokens: Some(7),
            reasoning_output_tokens: Some(11),
            total_tokens: Some(13),
        };

        assert_eq!(
            first.aggregated(second),
            TokenUsage {
                input_tokens: Some(u64::MAX),
                cached_input_tokens: Some(5),
                output_tokens: Some(10),
                reasoning_output_tokens: Some(15),
                total_tokens: Some(18),
            }
        );
    }
}
