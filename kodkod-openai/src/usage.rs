use kodkod_core::TokenUsage;
use serde_json::Value;

pub(crate) fn parse_usage(value: &Value) -> TokenUsage {
    let usage = value.get("usage").unwrap_or(value);
    TokenUsage {
        input_tokens: first_u64(usage, &["input_tokens", "prompt_tokens"]),
        cached_input_tokens: nested_u64(
            usage,
            &["input_tokens_details", "prompt_tokens_details"],
            "cached_tokens",
        ),
        output_tokens: first_u64(usage, &["output_tokens", "completion_tokens"]),
        reasoning_output_tokens: nested_u64(
            usage,
            &["output_tokens_details", "completion_tokens_details"],
            "reasoning_tokens",
        ),
        total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
    }
}

fn first_u64(value: &Value, fields: &[&str]) -> Option<u64> {
    fields
        .iter()
        .find_map(|field| value.get(*field).and_then(Value::as_u64))
}

fn nested_u64(value: &Value, objects: &[&str], field: &str) -> Option<u64> {
    objects.iter().find_map(|object| {
        value
            .get(*object)
            .and_then(|details| details.get(field))
            .and_then(Value::as_u64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_responses_usage_details() {
        let usage = parse_usage(&json!({
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 25},
                "output_tokens": 40,
                "output_tokens_details": {"reasoning_tokens": 12},
                "total_tokens": 140
            }
        }));
        assert_eq!(
            usage,
            TokenUsage {
                input_tokens: Some(100),
                cached_input_tokens: Some(25),
                output_tokens: Some(40),
                reasoning_output_tokens: Some(12),
                total_tokens: Some(140),
            }
        );
    }

    #[test]
    fn parses_chat_completion_field_names() {
        let usage = parse_usage(&json!({
            "prompt_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens": 6,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 16
        }));
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.cached_input_tokens, Some(3));
        assert_eq!(usage.output_tokens, Some(6));
        assert_eq!(usage.reasoning_output_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(16));
    }
}
