use kodkod_core::{
    RequestCacheDiagnostics, RequestCacheReason, RequestCacheType, RequestServiceTier,
    RequestUsage, TokenUsage,
};
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

pub(crate) fn parse_request_usage(value: &Value) -> RequestUsage {
    let usage = value.get("usage").unwrap_or(value);
    let cache_write_tokens = first_u64(usage, &["cache_write_tokens"]).or_else(|| {
        nested_u64(
            usage,
            &["input_tokens_details", "prompt_tokens_details"],
            "cache_write_tokens",
        )
    });
    let diagnostics = usage
        .get("prompt_cache_diagnostics")
        .or_else(|| value.get("prompt_cache_diagnostics"))
        .and_then(Value::as_object)
        .map(|diagnostics| RequestCacheDiagnostics {
            reason: diagnostics
                .get("reason")
                .and_then(Value::as_str)
                .map(|reason| match reason {
                    "hit" | "cache_hit" => RequestCacheReason::Hit,
                    "miss" | "cache_miss" => RequestCacheReason::Miss,
                    "write" | "cache_write" => RequestCacheReason::Write,
                    "disabled" | "cache_disabled" => RequestCacheReason::Disabled,
                    "ineligible" | "not_eligible" => RequestCacheReason::Ineligible,
                    _ => RequestCacheReason::Other,
                }),
            cache_type: diagnostics
                .get("type")
                .and_then(Value::as_str)
                .map(|cache_type| match cache_type {
                    "prompt_prefix" | "prefix" => RequestCacheType::PromptPrefix,
                    "input" => RequestCacheType::Input,
                    _ => RequestCacheType::Other,
                }),
        });

    RequestUsage {
        token_usage: parse_usage(value),
        cache_write_tokens,
        cache_diagnostics: diagnostics,
        effective_service_tier: value.get("service_tier").and_then(Value::as_str).map(
            |service_tier| match service_tier {
                "default" => RequestServiceTier::Default,
                "fast" => RequestServiceTier::Fast,
                "priority" => RequestServiceTier::Priority,
                "flex" => RequestServiceTier::Flex,
                "scale" => RequestServiceTier::Scale,
                "auto" => RequestServiceTier::Auto,
                "ultrafast" => RequestServiceTier::Ultrafast,
                _ => RequestServiceTier::Other,
            },
        ),
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

    #[test]
    fn parses_cache_write_tokens_and_allowlisted_cache_diagnostics() {
        let usage = parse_request_usage(&json!({
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {
                    "cached_tokens": 25,
                    "cache_write_tokens": 8
                },
                "prompt_cache_diagnostics": {
                    "reason": "cache_hit",
                    "type": "prompt_prefix",
                    "sensitive_future_field": "must not escape"
                }
            }
        }));
        assert_eq!(usage.token_usage.input_tokens, Some(100));
        assert_eq!(usage.token_usage.cached_input_tokens, Some(25));
        assert_eq!(usage.cache_write_tokens, Some(8));
        assert_eq!(
            usage.cache_diagnostics,
            Some(RequestCacheDiagnostics {
                reason: Some(RequestCacheReason::Hit),
                cache_type: Some(RequestCacheType::PromptPrefix),
            })
        );
    }

    #[test]
    fn unknown_cache_diagnostic_values_are_collapsed_to_other() {
        let usage = parse_request_usage(&json!({
            "usage": {
                "prompt_cache_diagnostics": {
                    "reason": "private text",
                    "type": "secret"
                }
            }
        }));
        assert_eq!(
            usage.cache_diagnostics,
            Some(RequestCacheDiagnostics {
                reason: Some(RequestCacheReason::Other),
                cache_type: Some(RequestCacheType::Other),
            })
        );
    }

    #[test]
    fn parses_allowlisted_effective_service_tiers() {
        for (service_tier, expected) in [
            ("default", RequestServiceTier::Default),
            ("fast", RequestServiceTier::Fast),
            ("priority", RequestServiceTier::Priority),
            ("flex", RequestServiceTier::Flex),
            ("scale", RequestServiceTier::Scale),
            ("auto", RequestServiceTier::Auto),
            ("ultrafast", RequestServiceTier::Ultrafast),
        ] {
            let usage = parse_request_usage(&json!({ "service_tier": service_tier }));
            assert_eq!(usage.effective_service_tier, Some(expected));
        }
    }

    #[test]
    fn missing_or_unknown_effective_tiers_are_safe_and_missing_stays_unknown() {
        let missing = parse_request_usage(&json!({ "usage": { "input_tokens": 3 } }));
        assert_eq!(missing.effective_service_tier, None);

        let unknown = parse_request_usage(&json!({
            "service_tier": "private future provider value",
            "usage": { "input_tokens": 3 }
        }));
        assert_eq!(
            unknown.effective_service_tier,
            Some(RequestServiceTier::Other)
        );
        assert!(
            !format!("{unknown:?}").contains("private future provider value"),
            "unknown provider values must never escape diagnostics"
        );

        let non_string = parse_request_usage(&json!({ "service_tier": 7 }));
        assert_eq!(non_string.effective_service_tier, None);
    }

    #[test]
    fn old_request_usage_records_deserialize_without_an_effective_tier() {
        let usage: RequestUsage = serde_json::from_value(json!({
            "token_usage": {},
            "cache_write_tokens": null,
            "cache_diagnostics": null
        }))
        .unwrap();
        assert_eq!(usage.effective_service_tier, None);
    }
}
