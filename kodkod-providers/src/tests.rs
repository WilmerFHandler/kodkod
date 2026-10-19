use super::*;
use futures_util::StreamExt;
use kodkod_anthropic::AnthropicError;
use kodkod_core::{
    AssistantMessage, CompactOptions, ContextBudget, Conversation, Document, Message, Provider,
    ProviderEvent, RetryAfter, RetryPolicy, RetryProvider, Retryable, SystemMessage, TokenUsage,
    ToolOutput, ToolResult, ToolSpec, UserMessage,
};
use kodkod_google::GoogleError;
use kodkod_http::{RequestCredentials, StaticCredentials};
use kodkod_openai::{OpenAiError, OpenAiResponsesProvider};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn opencode_errors_preserve_retry_after_across_go_protocols() {
    let retry_after = RetryAfter::BeyondMaximum;
    let errors = [
        (
            ProviderError::OpenAi(OpenAiError::Api {
                status: 429,
                code: Some("rate_limit_exceeded".into()),
                message: "slow down".into(),
                retry_after: Some(retry_after),
            }),
            Some(429),
            Some("rate_limit_exceeded"),
        ),
        (
            ProviderError::Anthropic(AnthropicError::Api {
                status: 429,
                message: "rate limited".into(),
                retry_after: Some(retry_after),
            }),
            Some(429),
            None,
        ),
        (
            ProviderError::Google(GoogleError::Api {
                status: 503,
                code: Some("UNAVAILABLE".into()),
                message: "try again".into(),
                retry_after: Some(retry_after),
            }),
            Some(503),
            Some("UNAVAILABLE"),
        ),
    ];

    for (error, status, api_code) in errors {
        assert_eq!(error.retry_after(), Some(retry_after));
        assert_eq!(error.http_status(), status);
        assert_eq!(error.api_code(), api_code);
    }
}

struct FreshCodexAccess(AtomicUsize);

struct TestCodexModel;

struct NonVisionCodexModel;

struct OptionsCodexModel;

impl kodkod_openai::OpenAiModel for TestCodexModel {
    fn id(&self) -> &str {
        "gpt-test"
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn context_budget(&self) -> Option<ContextBudget> {
        Some(ContextBudget {
            max_input_tokens: 32_000,
        })
    }
}

impl kodkod_openai::OpenAiModel for NonVisionCodexModel {
    fn id(&self) -> &str {
        "gpt-test"
    }

    fn supports_vision(&self) -> bool {
        false
    }
}

impl kodkod_openai::OpenAiModel for OptionsCodexModel {
    fn id(&self) -> &str {
        "gpt-options"
    }

    fn supports_vision(&self) -> bool {
        false
    }

    fn request_options(&self) -> kodkod_openai::OpenAiRequestOptions {
        kodkod_openai::OpenAiRequestOptions {
            reasoning_effort: Some("future-effort".into()),
            service_tier: Some("burst".into()),
        }
    }
}

impl CodexCredentialSource for FreshCodexAccess {
    fn access(&self) -> CodexAccessFuture<'_> {
        Box::pin(async {
            let request = self.0.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(CodexAccess {
                access_token: format!("token-{request}"),
                account_id: "account-1".into(),
            })
        })
    }
}

fn completed(text: &str) -> String {
    completed_output(json!([{
        "type":"message",
        "role":"assistant",
        "content":[{"type":"output_text", "text":text}]
    }]))
}

fn completed_output(output: serde_json::Value) -> String {
    format!(
        "event: response.completed\ndata: {}\n\n",
        json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": output
            }
        })
    )
}

fn completed_with_usage(text: &str) -> String {
    format!(
        "event: response.completed\ndata: {}\n\n",
        json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 25},
                    "output_tokens": 40,
                    "output_tokens_details": {"reasoning_tokens": 12},
                    "total_tokens": 140
                },
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}]
                }]
            }
        })
    )
}

fn chatgpt_provider<M>() -> ChatGptProvider<M> {
    let credentials = StaticCredentials::new(RequestCredentials::bearer("plan-token").unwrap());
    ChatGptProvider::new(Arc::new(credentials)).unwrap()
}

#[tokio::test]
async fn chatgpt_uses_stateless_namespaced_functions_and_replays_raw_output() {
    let server = MockServer::start().await;
    let raw_reasoning = json!({
        "id":"rs_1", "type":"reasoning", "encrypted_content":"opaque-reasoning"
    });
    let raw_call = json!({
        "id":"fc_1", "type":"function_call", "call_id":"call_1",
        "namespace":"functions", "name":"lookup", "arguments":"{\"q\":\"x\"}"
    });
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(completed_output(json!([
                raw_reasoning.clone(),
                raw_call.clone()
            ]))),
        )
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(completed("found it")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = chatgpt_provider::<TestCodexModel>().with_test_endpoint(server.uri());
    assert_eq!(provider.endpoint(), CHATGPT_ENDPOINT);
    assert_eq!(provider.protocol(), Protocol::Responses);
    let model = TestCodexModel;
    let tools = [ToolSpec::new(
        "lookup",
        "Look up a value.",
        json!({"type":"object", "properties":{"q":{"type":"string"}}}),
    )];
    let mut conversation = Conversation::new().with_system_prompt("Follow the research plan.");
    conversation.push_message(Message::System(SystemMessage::new("Use citations.")));
    conversation.push_user_message(UserMessage::new("find x"));
    let continuation = provider.create_continuation(&model);
    let completion = provider
        .complete_with_usage(&continuation, &model, &conversation, &tools)
        .await
        .unwrap();
    assert_eq!(completion.message.tool_calls()[0].name(), "lookup");

    let call_checkpoint = serde_json::to_value(&completion.continuation).unwrap();
    assert_eq!(
        call_checkpoint["checkpoints"][0]["output"][0],
        raw_reasoning
    );
    assert_eq!(call_checkpoint["checkpoints"][0]["output"][1], raw_call);
    conversation.push_message(Message::Assistant(completion.message));
    conversation.push_message(Message::ToolResult(ToolResult::success(
        "call_1",
        ToolOutput::new(json!({"value":"found"})),
    )));
    let second = provider
        .complete_with_usage(&completion.continuation, &model, &conversation, &tools)
        .await
        .unwrap();
    assert_eq!(second.message.content(), "found it");

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(body["instructions"], "Follow the research plan.");
        assert_eq!(body["tools"][0]["type"], "namespace");
        assert_eq!(body["tools"][0]["name"], "functions");
        assert_eq!(body["tools"][0]["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["tools"][0]["name"], "lookup");
        assert_eq!(body["input"][0]["role"], "developer");
    }
    let second_body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(second_body["input"][2], raw_reasoning);
    assert_eq!(second_body["input"][3], raw_call);
    assert_eq!(second_body["input"][4]["type"], "function_call_output");
    assert_eq!(second_body["input"][4]["call_id"], "call_1");
    server.verify().await;
}

#[tokio::test]
async fn chatgpt_forwards_native_compaction_and_model_budget() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses/compact"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object":"response.compaction",
            "output": [{"type":"compaction", "encrypted_content":"opaque-window"}],
            "usage": {"input_tokens": 99, "output_tokens": 2, "total_tokens": 101}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = chatgpt_provider::<TestCodexModel>().with_test_endpoint(server.uri());
    let model = TestCodexModel;
    assert_eq!(
        provider.context_budget(&model),
        Some(ContextBudget {
            max_input_tokens: 32_000,
        })
    );
    let mut conversation = Conversation::new().with_system_prompt("Follow the plan.");
    conversation.push_user_message(UserMessage::new("remember this"));
    let compacted = provider
        .compact_context(
            &provider.create_continuation(&model),
            &model,
            &conversation,
            CompactOptions::default(),
        )
        .await
        .unwrap();

    assert!(Provider::has_opaque_context(
        &provider,
        &compacted.continuation
    ));
    assert!(Provider::validate_context(&provider, &model, &compacted.continuation).is_ok());
    assert_eq!(compacted.usage.input_tokens, Some(99));
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["instructions"], "Follow the plan.");
    assert_eq!(body["input"][0]["role"], "user");
    server.verify().await;
}

#[tokio::test]
async fn chatgpt_sharing_limit_is_not_retried_but_generic_rate_limit_is() {
    let chatgpt_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {
                "code": "subscription_sharing_usage_limit_exceeded",
                "message": "sharing limit reached"
            }
        })))
        .expect(1)
        .up_to_n_times(1)
        .mount(&chatgpt_server)
        .await;
    let chatgpt = chatgpt_provider::<TestCodexModel>().with_test_endpoint(chatgpt_server.uri());
    let retrying = RetryProvider::with_policy(
        chatgpt,
        RetryPolicy {
            max_attempts: 3,
            initial_backoff: std::time::Duration::ZERO,
            max_backoff: std::time::Duration::ZERO,
            backoff_multiplier: 1.0,
        },
    );
    let model = TestCodexModel;
    let error = retrying
        .complete_once(&model, &Conversation::new(), &[])
        .await
        .unwrap_err();
    assert!(matches!(
        &error,
        OpenAiError::Api { status: 429, code: Some(code), .. }
            if code == "subscription_sharing_usage_limit_exceeded"
    ));
    assert!(!error.is_retryable());
    chatgpt_server.verify().await;

    let generic_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {"message":"ordinary rate limit"}
        })))
        .expect(1)
        .up_to_n_times(1)
        .mount(&generic_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(completed("recovered")))
        .expect(1)
        .mount(&generic_server)
        .await;
    let generic =
        OpenAiResponsesProvider::<TestCodexModel>::new(format!("{}/v1", generic_server.uri()));
    let retrying = RetryProvider::with_policy(
        generic,
        RetryPolicy {
            max_attempts: 2,
            initial_backoff: std::time::Duration::ZERO,
            max_backoff: std::time::Duration::ZERO,
            backoff_multiplier: 1.0,
        },
    );
    assert_eq!(
        retrying
            .complete_once(&model, &Conversation::new(), &[])
            .await
            .unwrap()
            .content(),
        "recovered"
    );
    generic_server.verify().await;
}

#[tokio::test]
async fn chatgpt_preserves_failed_response_code_after_text_delta() {
    let server = MockServer::start().await;
    let body = format!(
        "event: response.output_text.delta\ndata: {}\n\nevent: response.failed\ndata: {}\n\n",
        json!({"type":"response.output_text.delta", "delta":"partial"}),
        json!({
            "type":"response.failed",
            "response":{"error":{
                "code":"subscription_sharing_usage_unavailable",
                "message":"sharing temporarily unavailable"
            }}
        })
    );
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(1)
        .mount(&server)
        .await;
    let provider = chatgpt_provider::<TestCodexModel>().with_test_endpoint(server.uri());
    let model = TestCodexModel;
    let continuation = provider.create_continuation(&model);
    let conversation = Conversation::new();
    let mut stream = provider.complete_stream(&continuation, &model, &conversation, &[]);
    assert!(matches!(
        stream.next().await.unwrap().unwrap(),
        ProviderEvent::TextDelta(delta) if delta == "partial"
    ));
    let error = match stream.next().await.unwrap() {
        Ok(_) => panic!("expected failed response"),
        Err(error) => error,
    };
    assert!(matches!(
        &error,
        OpenAiError::ResponseFailed {
            code: Some(code),
            ..
        }
            if code == "subscription_sharing_usage_unavailable"
    ));
    assert!(!error.is_retryable());
    server.verify().await;
}

#[tokio::test]
async fn chatgpt_does_not_forward_credentials_through_redirects() {
    let redirect_server = MockServer::start().await;
    let target_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer plan-token"))
        .respond_with(
            ResponseTemplate::new(307)
                .insert_header("location", format!("{}/steal", target_server.uri())),
        )
        .expect(1)
        .mount(&redirect_server)
        .await;

    let provider = chatgpt_provider::<TestCodexModel>().with_test_endpoint(redirect_server.uri());
    let error = provider
        .complete_once(&TestCodexModel, &Conversation::new(), &[])
        .await
        .unwrap_err();
    assert!(matches!(error, OpenAiError::Api { status: 307, .. }));
    assert!(target_server.received_requests().await.unwrap().is_empty());
    redirect_server.verify().await;
}

#[tokio::test]
async fn codex_adds_fresh_account_credentials_and_originator() {
    let server = MockServer::start().await;
    for request in 1..=2 {
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(header("authorization", format!("Bearer token-{request}")))
            .and(header("chatgpt-account-id", "account-1"))
            .and(header("originator", "lynx"))
            .respond_with(ResponseTemplate::new(200).set_body_string(completed("done")))
            .expect(1)
            .mount(&server)
            .await;
    }
    let provider =
        CodexProvider::<TestCodexModel>::new(Arc::new(FreshCodexAccess(AtomicUsize::new(0))))
            .unwrap()
            .with_originator("lynx")
            .unwrap()
            .with_test_endpoint(server.uri());
    let conversation = Conversation::new();
    for _ in 0..2 {
        provider
            .complete_once(&TestCodexModel, &conversation, &[])
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn codex_forwards_responses_text_deltas() {
    let server = MockServer::start().await;
    let body = format!(
        "event: response.output_text.delta\ndata: {}\n\n{}",
        json!({"type":"response.output_text.delta","delta":"live"}),
        completed("final")
    );
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;
    let provider =
        CodexProvider::<TestCodexModel>::new(Arc::new(FreshCodexAccess(AtomicUsize::new(0))))
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = TestCodexModel;
    let conversation = Conversation::new();
    let continuation = provider.create_continuation(&model);
    let mut stream = provider.complete_stream(&continuation, &model, &conversation, &[]);
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::TextDelta(text) if text == "live")
    );
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::Completed(completion) if completion.message.content() == "final")
    );
}

#[tokio::test]
async fn codex_forwards_request_options_to_the_responses_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(completed("done")))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        CodexProvider::<OptionsCodexModel>::new(Arc::new(FreshCodexAccess(AtomicUsize::new(0))))
            .unwrap()
            .with_test_endpoint(server.uri());
    provider
        .complete_once(&OptionsCodexModel, &Conversation::new(), &[])
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "gpt-options");
    assert_eq!(body["reasoning"]["effort"], "future-effort");
    assert_eq!(body["service_tier"], "burst");
    server.verify().await;
}

#[tokio::test]
async fn codex_routes_pdf_to_responses_for_completion_and_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(completed_with_usage("done")))
        .expect(2)
        .mount(&server)
        .await;
    let provider =
        CodexProvider::<TestCodexModel>::new(Arc::new(FreshCodexAccess(AtomicUsize::new(0))))
            .unwrap()
            .with_test_endpoint(server.uri());
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("read").with_documents(vec![
        Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
    ]));

    let completion = provider
        .complete_once_with_usage(&TestCodexModel, &conversation, &[])
        .await
        .unwrap();
    assert_eq!(
        completion.usage,
        TokenUsage {
            input_tokens: Some(100),
            cached_input_tokens: Some(25),
            output_tokens: Some(40),
            reasoning_output_tokens: Some(12),
            total_tokens: Some(140),
        }
    );
    let continuation = provider.create_continuation(&TestCodexModel);
    let mut stream = provider.complete_stream(&continuation, &TestCodexModel, &conversation, &[]);
    assert!(matches!(
        stream.next().await.unwrap().unwrap(),
        ProviderEvent::Completed(completion)
            if completion.message.content() == "done"
                && completion.usage.total_tokens == Some(140)
    ));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    for request in requests {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["input"][0]["content"][1]["type"], "input_file");
        assert_eq!(
            body["input"][0]["content"][1]["file_data"],
            "data:application/pdf;base64,JVBERg=="
        );
    }
    server.verify().await;
}

#[tokio::test]
async fn codex_rejects_unverified_documents_before_credentials_or_http() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;
    let accesses = Arc::new(FreshCodexAccess(AtomicUsize::new(0)));
    let provider = CodexProvider::<TestCodexModel>::new(accesses.clone())
        .unwrap()
        .with_test_endpoint(server.uri());
    let nonvision_provider = CodexProvider::<NonVisionCodexModel>::new(accesses.clone())
        .unwrap()
        .with_test_endpoint(server.uri());
    let mut text_document = Conversation::new();
    text_document.push_user_message(UserMessage::new("read").with_documents(vec![
        Document::try_new("text/plain", "notes.txt", b"notes").unwrap(),
    ]));
    let mut pdf = Conversation::new();
    pdf.push_user_message(UserMessage::new("read").with_documents(vec![
        Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
    ]));

    let error = provider
        .complete_once(&TestCodexModel, &text_document, &[])
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        kodkod_openai::OpenAiError::UnsupportedDocument { .. }
    ));
    let continuation = nonvision_provider.create_continuation(&NonVisionCodexModel);
    let mut stream =
        nonvision_provider.complete_stream(&continuation, &NonVisionCodexModel, &pdf, &[]);
    assert!(matches!(
        stream.next().await.unwrap(),
        Err(kodkod_openai::OpenAiError::UnsupportedDocument { .. })
    ));
    assert_eq!(accesses.0.load(Ordering::SeqCst), 0);
    server.verify().await;
}

#[test]
fn catalog_routes_models_by_service_and_protocol() {
    assert_eq!(
        open_code_model(OpenCodeService::Go, "gpt-5.6-luna")
            .unwrap()
            .protocol(),
        Protocol::Responses
    );
    assert_eq!(
        open_code_model(OpenCodeService::Go, "minimax-m2.7")
            .unwrap()
            .protocol(),
        Protocol::Messages
    );
    assert_eq!(
        open_code_model(OpenCodeService::Zen, "minimax-m2.7")
            .unwrap()
            .protocol(),
        Protocol::ChatCompletions
    );
    let deepseek = open_code_model(OpenCodeService::Go, "deepseek-v4.1-flash").unwrap();
    assert_eq!(deepseek.name(), "DeepSeek V4.1 Flash");
    assert_eq!(deepseek.protocol(), Protocol::ChatCompletions);
    assert!(deepseek.supports_vision());
    assert!(deepseek.supports_reasoning());
    assert_eq!(
        deepseek
            .reasoning_efforts()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["low", "high", "max"]
    );
    assert_eq!(deepseek.default_reasoning_effort(), Some("high"));
    assert_eq!(deepseek.context_window_tokens(), Some(1_000_000));
    assert_eq!(deepseek.max_output_tokens(), Some(384_000));
    assert_eq!(
        deepseek.context_budget(),
        Some(ContextBudget {
            max_input_tokens: 616_000,
        })
    );
    assert_eq!(
        deepseek.request_options().reasoning_effort.as_deref(),
        Some("high")
    );
    assert_eq!(deepseek.request_options().service_tier, None);
    let selected = (*deepseek).clone().with_reasoning_effort("low").unwrap();
    assert_eq!(
        selected.request_options().reasoning_effort.as_deref(),
        Some("low")
    );
    assert!((*deepseek).clone().with_reasoning_effort("medium").is_err());
    let mut seen = HashSet::new();
    for model in open_code_catalog() {
        assert!(seen.insert((model.service(), model.id())));
    }

    assert!(
        open_code_model(OpenCodeService::Go, "gpt-5.6-luna")
            .unwrap()
            .supports_pdf()
    );
    assert!(
        !open_code_model(OpenCodeService::Go, "glm-5.3-flash")
            .unwrap()
            .supports_vision()
    );
    assert!(
        !open_code_model(OpenCodeService::Go, "glm-5.3-flash")
            .unwrap()
            .supports_pdf()
    );
    let gemini = open_code_model(OpenCodeService::Go, "gemini-4-argon-contributor").unwrap();
    assert_eq!(gemini.protocol(), Protocol::GenerateContent);
    assert!(gemini.supports_vision());
    assert!(gemini.supports_pdf());
    assert_eq!(gemini.context_window_tokens(), Some(1_048_576));
    assert_eq!(gemini.max_output_tokens(), Some(65_536));
    let luna = open_code_model(OpenCodeService::Go, "gpt-6-luna").unwrap();
    assert_eq!(luna.protocol(), Protocol::Responses);
    assert!(luna.supports_vision());
    assert_eq!(luna.context_window_tokens(), Some(1_050_000));
    assert_eq!(luna.max_output_tokens(), Some(128_000));
    let muse = open_code_model(OpenCodeService::Go, "muse-spark-1.3-contributor").unwrap();
    assert_eq!(muse.protocol(), Protocol::Responses);
    assert!(muse.supports_vision());
    assert!(muse.supports_pdf());
    assert_eq!(muse.context_window_tokens(), None);
    assert_eq!(muse.max_output_tokens(), None);
    let glm = open_code_model(OpenCodeService::Go, "glm-5.3-flash").unwrap();
    assert_eq!(glm.protocol(), Protocol::ChatCompletions);
    assert_eq!(glm.context_window_tokens(), Some(1_000_000));
    assert_eq!(glm.max_output_tokens(), Some(131_072));
    assert!(
        open_code_model(OpenCodeService::Zen, "claude-sonnet-5")
            .unwrap()
            .supports_pdf()
    );
    assert!(
        !open_code_model(OpenCodeService::Go, "grok-4.6")
            .unwrap()
            .supports_pdf()
    );
    assert!(
        !open_code_model(OpenCodeService::Zen, "qwen3.7-max")
            .unwrap()
            .supports_pdf()
    );
}

#[tokio::test]
async fn open_code_go_deepseek_replays_reasoning_and_tool_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .and(header("authorization", "Bearer secret"))
        .and(header("x-opencode-session", "deepseek-session"))
        .and(header("user-agent", "Lynx/1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": "Looking that up.",
                    "reasoning_content": "I should query the tool.",
                    "tool_calls": [{
                        "id": "call_lookup",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"}
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 123,
                "prompt_tokens_details": {"cached_tokens": 17},
                "prompt_cache_hit_tokens": 17,
                "prompt_cache_miss_tokens": 106,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 12},
                "total_tokens": 143
            }
        })))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .and(header("x-opencode-session", "deepseek-session"))
        .and(header("user-agent", "Lynx/1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": "Found it.",
                    "reasoning_content": "The tool returned the answer.",
                    "tool_calls": []
                }
            }],
            "usage": {
                "prompt_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 30},
                "completion_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 158
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "deepseek-session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = (*open_code_model(OpenCodeService::Go, "deepseek-v4.1-flash").unwrap())
        .clone()
        .with_reasoning_effort("low")
        .unwrap();
    let tools = [ToolSpec::new(
        "lookup",
        "Look up a value.",
        json!({"type":"object", "properties":{}}),
    )];
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("Find the value."));
    let continuation = provider.create_continuation(&model);
    let first = provider
        .complete_with_usage(&continuation, &model, &conversation, &tools)
        .await
        .unwrap();
    assert_eq!(
        first.message.reasoning_content(),
        Some("I should query the tool.")
    );
    assert_eq!(first.usage.cached_input_tokens, Some(17));
    assert_eq!(first.usage.reasoning_output_tokens, Some(12));
    conversation.push_message(Message::Assistant(first.message));
    conversation.push_message(Message::ToolResult(ToolResult::success(
        "call_lookup",
        ToolOutput::new(json!({"value": "found"})),
    )));

    let second = provider
        .complete_with_usage(&first.continuation, &model, &conversation, &tools)
        .await
        .unwrap();
    assert_eq!(second.message.content(), "Found it.");
    assert_eq!(second.usage.cached_input_tokens, Some(30));
    assert_eq!(second.usage.reasoning_output_tokens, Some(5));

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert_eq!(request.url.path(), "/openai/v1/chat/completions");
        assert_eq!(
            request.headers.get("x-opencode-session").unwrap(),
            "deepseek-session"
        );
        assert_eq!(request.headers.get("user-agent").unwrap(), "Lynx/1");
    }
    let first_body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(first_body["model"], "deepseek-v4.1-flash");
    assert_eq!(first_body["reasoning_effort"], "low");
    assert!(first_body.get("service_tier").is_none());
    let second_body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        second_body["messages"][1]["reasoning_content"],
        "I should query the tool."
    );
    assert_eq!(
        second_body["messages"][1]["tool_calls"][0]["id"],
        "call_lookup"
    );
    assert_eq!(second_body["messages"][2]["tool_call_id"], "call_lookup");
    server.verify().await;
}

#[tokio::test]
async fn open_code_go_deepseek_replays_empty_reasoning_for_tool_only_turns() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {
                "content": "",
                "tool_calls": [{"id":"call_1", "type":"function", "function":{"name":"lookup", "arguments":"{}"}}]
            }}]
        })))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"content": "done", "tool_calls": []}}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let provider = OpenCodeProvider::with_api_key(
        OpenCodeService::Go,
        "secret",
        "empty-reasoning-session",
        "Lynx/1",
    )
    .unwrap()
    .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Go, "deepseek-v4.1-flash").unwrap();
    let tools = [ToolSpec::new("lookup", "Lookup", json!({"type":"object"}))];
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("lookup"));
    let continuation = provider.create_continuation(model);
    let first = provider
        .complete_with_usage(&continuation, model, &conversation, &tools)
        .await
        .unwrap();
    assert_eq!(first.message.reasoning_content(), Some(""));
    conversation.push_message(Message::Assistant(first.message));
    conversation.push_message(Message::ToolResult(ToolResult::success(
        "call_1",
        ToolOutput::new(json!({"ok": true})),
    )));
    provider
        .complete_with_usage(&first.continuation, model, &conversation, &tools)
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let second_body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(second_body["messages"][1]["reasoning_content"], "");
    server.verify().await;
}

#[tokio::test]
async fn open_code_go_deepseek_uses_generic_summary_compaction() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_summary",
                    "type": "function",
                    "function": {"name": "summarize", "arguments": "{\"summary\":\"Kept facts.\"}"}
                }]
            }}],
            "usage": {
                "prompt_tokens": 45,
                "prompt_tokens_details": {"cached_tokens": 8},
                "completion_tokens": 9,
                "total_tokens": 54
            }
        })))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"content": "Continued", "tool_calls": []}}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "compact-session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Go, "deepseek-v4.1-flash").unwrap();
    assert_eq!(provider.context_budget(model), model.context_budget());
    let continuation = provider.create_continuation(model);
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("Old question"));
    conversation.push_message(Message::Assistant(
        AssistantMessage::new("Old answer").with_reasoning_content("old private reasoning"),
    ));
    conversation.push_user_message(UserMessage::new("Keep this question"));

    let compacted = provider
        .compact_context(
            &continuation,
            model,
            &conversation,
            CompactOptions {
                keep_tail_tokens: 0,
                ..CompactOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(!Provider::has_opaque_context(
        &provider,
        &compacted.continuation
    ));
    assert_eq!(compacted.usage.input_tokens, Some(45));
    assert_eq!(compacted.usage.cached_input_tokens, Some(8));
    assert!(compacted.conversation.messages().iter().any(|message| {
        matches!(message, Message::Assistant(assistant) if assistant.content() == "Kept facts.")
    }));
    let tools = [ToolSpec::new("lookup", "Lookup", json!({"type":"object"}))];
    provider
        .complete_with_usage(
            &compacted.continuation,
            model,
            &compacted.conversation,
            &tools,
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].url.path(), "/openai/v1/chat/completions");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["tools"][0]["function"]["name"], "summarize");
    assert!(body.get("service_tier").is_none());
    let next_body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(next_body["messages"][0]["content"], "Kept facts.");
    assert_eq!(next_body["messages"][0]["reasoning_content"], "");
    server.verify().await;
}

#[tokio::test]
async fn open_code_routes_chat_and_owns_required_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer secret"))
        .and(header("x-opencode-session", "session-1"))
        .and(header("user-agent", "Lynx/1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"content": "world", "tool_calls": []}}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut supplied = HeaderMap::new();
    supplied.insert("authorization", HeaderValue::from_static("Bearer secret"));
    supplied.insert(
        "x-opencode-session",
        HeaderValue::from_static("wrong-session"),
    );
    supplied.insert("user-agent", HeaderValue::from_static("wrong-agent"));
    let provider = OpenCodeProvider::with_credentials(
        OpenCodeService::Zen,
        Arc::new(StaticCredentials::new(RequestCredentials::from_headers(
            supplied,
        ))),
        "session-1",
        "Lynx/1",
    )
    .unwrap()
    .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Zen, "minimax-m2.7").unwrap();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("hello"));
    let message = provider
        .complete_once(model, &conversation, &[])
        .await
        .unwrap();
    assert_eq!(message.content(), "world");
}

#[tokio::test]
async fn open_code_routes_responses_with_bearer_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses"))
        .and(header("authorization", "Bearer secret"))
        .and(header("x-opencode-session", "session-2"))
        .and(header("user-agent", "Lynx/1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(completed("response")))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session-2", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Go, "gpt-5.6-luna").unwrap();
    let message = provider
        .complete_once(model, &Conversation::new(), &[])
        .await
        .unwrap();
    assert_eq!(message.content(), "response");
}

#[tokio::test]
async fn open_code_routes_messages_with_native_api_key_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .and(header("x-api-key", "secret"))
        .and(header("x-opencode-session", "session-3"))
        .and(header("user-agent", "Lynx/1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "message"}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session-3", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Go, "minimax-m2.7").unwrap();
    let message = provider
        .complete_once(model, &Conversation::new(), &[])
        .await
        .unwrap();
    assert_eq!(message.content(), "message");
}

#[tokio::test]
async fn open_code_go_adapters_preserve_retry_after_and_structured_status() {
    let server = MockServer::start().await;
    let retry_after = "120";
    let openai_error = json!({
        "error": {
            "code": "subscription_sharing_usage_limit_exceeded",
            "message": "plan usage limit"
        }
    });
    for route in ["/openai/v1/responses", "/openai/v1/chat/completions"] {
        Mock::given(method("POST"))
            .and(path(route))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", retry_after)
                    .set_body_json(openai_error.clone()),
            )
            .expect(1)
            .mount(&server)
            .await;
    }
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", retry_after)
                .set_body_json(json!({
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "plan usage limit"}
                })),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(
            "/google/v1beta/models/gemini-4-argon-contributor:generateContent",
        ))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", retry_after)
                .set_body_json(json!({
                    "error": {
                        "code": 429,
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "plan usage limit"
                    }
                })),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("hello"));
    let cases = [
        (
            open_code_model(OpenCodeService::Go, "gpt-5.6-luna").unwrap(),
            Some("subscription_sharing_usage_limit_exceeded"),
        ),
        (
            open_code_model(OpenCodeService::Go, "deepseek-v4.1-flash").unwrap(),
            Some("subscription_sharing_usage_limit_exceeded"),
        ),
        (
            open_code_model(OpenCodeService::Go, "minimax-m2.7").unwrap(),
            None,
        ),
        (
            open_code_model(OpenCodeService::Go, "gemini-4-argon-contributor").unwrap(),
            Some("RESOURCE_EXHAUSTED"),
        ),
    ];
    for (model, api_code) in cases {
        let error = provider
            .complete_once(model, &conversation, &[])
            .await
            .unwrap_err();
        assert_eq!(error.http_status(), Some(429));
        assert_eq!(error.api_code(), api_code);
        assert_eq!(
            error.retry_after(),
            Some(RetryAfter::Delay(std::time::Duration::from_secs(120)))
        );
    }
    server.verify().await;
}

#[tokio::test]
async fn open_code_go_truncated_error_bodies_preserve_retry_after_and_status() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let response_server = tokio::spawn(async move {
        for status in [429, 429, 429, 401] {
            let (mut connection, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut header_end = None;
            let mut buffer = [0_u8; 4096];
            while header_end.is_none() {
                let count = connection.read(&mut buffer).await.unwrap();
                assert_ne!(count, 0, "client closed before completing request headers");
                request.extend_from_slice(&buffer[..count]);
                header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4);
                assert!(
                    request.len() <= 64 * 1024,
                    "request headers exceeded the test bound"
                );
            }

            let reason = if status == 401 {
                "Unauthorized"
            } else {
                "Too Many Requests"
            };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\nRetry-After: 86400\r\nContent-Length: 512\r\nConnection: close\r\n\r\n{{\"error\":"
            );
            connection.write_all(response.as_bytes()).await.unwrap();
            connection.shutdown().await.unwrap();
        }
    });

    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(format!("http://{address}"));
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("hello"));
    for model in [
        open_code_model(OpenCodeService::Go, "gpt-5.6-luna").unwrap(),
        open_code_model(OpenCodeService::Go, "minimax-m2.7").unwrap(),
        open_code_model(OpenCodeService::Go, "gemini-4-argon-contributor").unwrap(),
    ] {
        let error = provider
            .complete_once(model, &conversation, &[])
            .await
            .unwrap_err();
        assert_eq!(error.http_status(), Some(429));
        assert_eq!(
            error.retry_after(),
            Some(RetryAfter::Delay(std::time::Duration::from_secs(86_400)))
        );
    }
    let auth_error = provider
        .complete_once(
            open_code_model(OpenCodeService::Go, "gpt-5.6-luna").unwrap(),
            &conversation,
            &[],
        )
        .await
        .unwrap_err();
    assert_eq!(auth_error.http_status(), Some(401));
    assert_eq!(auth_error.retry_after(), None);
    response_server.await.unwrap();
}

#[tokio::test]
async fn open_code_rejects_a_model_from_the_other_service_before_http() {
    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session-1", "Lynx/1")
            .unwrap();
    let model = open_code_model(OpenCodeService::Zen, "minimax-m2.7").unwrap();
    let error = provider
        .complete_once(model, &Conversation::new(), &[])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("belongs to OpenCode"));
}

#[tokio::test]
async fn open_code_rejects_documents_before_http() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;
    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Go, "grok-4.6").unwrap();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("read").with_documents(vec![
        Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
    ]));

    let error = provider
        .complete_once(model, &conversation, &[])
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not support inline documents")
    );
    let continuation = provider.create_continuation(model);
    let mut stream = provider.complete_stream(&continuation, model, &conversation, &[]);
    assert!(matches!(
        stream.next().await.unwrap(),
        Err(ProviderError::Configuration(_))
    ));
    server.verify().await;
}

#[tokio::test]
async fn open_code_routes_pdf_to_the_responses_protocol() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(completed("done")))
        .mount(&server)
        .await;
    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Go, "gpt-5.6-luna").unwrap();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("read").with_documents(vec![
        Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
    ]));

    provider
        .complete_once(model, &conversation, &[])
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["input"][0]["content"][1]["type"], "input_file");
}

#[tokio::test]
async fn open_code_rejects_pdf_for_text_only_chat_completions_model() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;
    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Go, "glm-5.3-flash").unwrap();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("read").with_documents(vec![
        Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
    ]));

    let error = provider
        .complete_once(model, &conversation, &[])
        .await
        .unwrap_err();
    assert!(matches!(&error, ProviderError::Configuration(_)));
    assert!(
        error
            .to_string()
            .contains("does not support inline documents")
    );
    server.verify().await;
}

#[tokio::test]
async fn open_code_routes_pdf_to_the_messages_protocol() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "done"}]
        })))
        .mount(&server)
        .await;
    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Zen, "secret", "session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Zen, "claude-sonnet-5").unwrap();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("read").with_documents(vec![
        Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
    ]));

    provider
        .complete_once(model, &conversation, &[])
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["messages"][0]["content"][0]["type"], "document");
}

#[tokio::test]
async fn open_code_rejects_pdf_for_vision_models_without_a_pdf_capability() {
    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session", "Lynx/1").unwrap();
    let model = open_code_model(OpenCodeService::Go, "grok-4.6").unwrap();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("read").with_documents(vec![
        Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
    ]));

    assert!(matches!(
        provider.complete_once(model, &conversation, &[]).await,
        Err(ProviderError::Configuration(_))
    ));

    let zen_provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Zen, "secret", "session", "Lynx/1")
            .unwrap();
    let qwen = open_code_model(OpenCodeService::Zen, "qwen3.7-max").unwrap();
    assert!(!zen_provider.supports_document(qwen, "application/pdf"));
}

#[tokio::test]
async fn named_provider_does_not_follow_redirects() {
    let source = MockServer::start().await;
    let destination = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(302).insert_header(
            "location",
            format!("{}/chat/completions", destination.uri()),
        ))
        .expect(1)
        .mount(&source)
        .await;
    Mock::given(method("GET"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&destination)
        .await;

    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Zen, "secret", "session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(source.uri());
    let model = open_code_model(OpenCodeService::Zen, "minimax-m2.7").unwrap();
    let error = provider
        .complete_once(model, &Conversation::new(), &[])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("302"));
}

#[test]
fn named_endpoints_and_protocol_serialization_are_stable() {
    assert_eq!(
        CodexProvider::<TestCodexModel>::new(Arc::new(FreshCodexAccess(AtomicUsize::new(0))))
            .unwrap()
            .endpoint(),
        CODEX_ENDPOINT
    );
    assert_eq!(OpenCodeService::Go.endpoint(), OPEN_CODE_GO_ENDPOINT);
    assert_eq!(
        OPEN_CODE_GO_OPENAI_ENDPOINT,
        "https://opencode.ai/inference/go/openai/v1"
    );
    assert_eq!(
        OPEN_CODE_GO_ANTHROPIC_ENDPOINT,
        "https://opencode.ai/inference/go/anthropic/v1"
    );
    assert_eq!(OpenCodeService::Zen.endpoint(), OPEN_CODE_ZEN_ENDPOINT);
    assert_eq!(
        serde_json::to_string(&Protocol::ChatCompletions).unwrap(),
        "\"chat_completions\""
    );
}

#[test]
fn required_header_values_reject_empty_or_invalid_input() {
    assert!(OpenCodeProvider::with_api_key(OpenCodeService::Go, " ", "session", "Lynx/1").is_err());
    assert!(OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", " ", "Lynx/1").is_err());
    assert!(OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session", " ").is_err());
    assert!(
        OpenCodeProvider::with_api_key(
            OpenCodeService::Go,
            "secret\nsecond-header: value",
            "session",
            "Lynx/1",
        )
        .is_err()
    );
}
