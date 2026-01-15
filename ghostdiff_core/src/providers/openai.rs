//! OpenAI-compatible provider integration.
//!
//! This module implements OpenAI-compatible chat completions with
//! automatic recording into `ghostdiff_core::Recorder` via the runtime.
//!
//! ## Minimal Example
//!
//! ```rust,ignore
//! use ghostdiff::track;
//! use ghostdiff_core::providers::openai::{OpenAIClient, ChatMessage, ChatCompletionRequest};
//! use ghostdiff_core::runtime::{init_recorder, take_recorder};
//!
//! #[track(ai = true)]
//! async fn ask(prompt: &str, client: &OpenAIClient, model: &str) -> Result<String, ghostdiff_core::providers::openai::OpenAIError> {
//!     let request = ChatCompletionRequest::new(
//!         model,
//!         vec![ChatMessage::user(prompt)],
//!     );
//!     let response = client.chat_completion(request).await?;
//!     Ok(response.first_content().unwrap_or_default())
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     init_recorder("openai_run");
//!     let client = OpenAIClient::new(std::env::var("OPENAI_API_KEY").unwrap());
//!     let reply = ask("Hello!", &client, "gpt-4.1-mini").await.unwrap();
//!     println!("{}", reply);
//!     let recorder = take_recorder().unwrap();
//!     std::fs::write("run.json", recorder.to_json().unwrap()).unwrap();
//! }
//! ```

use crate::runtime;
use crate::Recorder;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

/// OpenAI-compatible client for chat completions.
#[derive(Clone)]
pub struct OpenAIClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAIClient {
    /// Creates a new client with the default OpenAI-compatible base URL.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, "https://api.openai.com/v1")
    }

    /// Creates a new client with a custom base URL.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
        }
    }

    /// Sends a chat completion request and returns the response.
    ///
    /// This method records:
    /// - Model configuration
    /// - Prompt and messages
    /// - Response message
    /// - Token usage (if provided)
    /// - Token-level logprobs (if available)
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, OpenAIError> {
        record_chat_request(&request);

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let response = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(OpenAIError::Http)?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let message = parse_error_message(&body).unwrap_or(body);
            record_error(&request, status, &message);
            return Err(OpenAIError::Api { status, message });
        }

        let response: ChatCompletionResponse = response.json().await.map_err(OpenAIError::Http)?;

        record_chat_response(&request, &response);

        Ok(response)
    }

    /// Convenience helper: returns the first response content.
    pub async fn chat_completion_text(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<String, OpenAIError> {
        let response = self.chat_completion(request).await?;
        Ok(response.first_content().unwrap_or_default())
    }
}

/// Error type for OpenAI-compatible operations.
#[derive(Debug, Error)]
pub enum OpenAIError {
    /// HTTP or network error.
    #[error("HTTP error: {0}")]
    Http(reqwest::Error),

    /// API returned a non-success status.
    #[error("API error ({status}): {message}")]
    Api { status: StatusCode, message: String },
}

/// Chat completion request (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model ID (e.g., "gpt-4.1-mini")
    pub model: String,

    /// Messages in the conversation.
    pub messages: Vec<ChatMessage>,

    /// Sampling temperature (0.0-2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Whether to return log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Additional, provider-specific parameters.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ChatCompletionRequest {
    /// Creates a new request with minimal required fields.
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            max_tokens: None,
            logprobs: None,
            extra: serde_json::Map::new(),
        }
    }

    /// Sets temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Enables logprobs for token-level tracking.
    pub fn with_logprobs(mut self, enable: bool) -> Self {
        self.logprobs = Some(enable);
        self
    }

    /// Adds a provider-specific parameter.
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

/// OpenAI-compatible chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role ("system", "user", "assistant", "tool")
    pub role: String,

    /// Message content.
    pub content: String,

    /// Optional name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Creates a new message with the specified role and content.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            name: None,
        }
    }

    /// Convenience: system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Convenience: user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Convenience: assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Chat completion response (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Option<TokenUsage>,
}

impl ChatCompletionResponse {
    /// Returns the content of the first choice, if present.
    pub fn first_content(&self) -> Option<String> {
        self.choices
            .first()
            .map(|c| c.message.content.clone())
    }

    /// Returns all assistant message contents.
    pub fn all_contents(&self) -> Vec<String> {
        self.choices
            .iter()
            .map(|c| c.message.content.clone())
            .collect()
    }
}

/// Chat completion choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
}

/// Log probabilities for tokens (if enabled).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatLogProbs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TokenLogProb>>,
}

/// Token-level log probability entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    pub token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TokenAlternative>>,
}

/// Alternative token logprob candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAlternative {
    pub token: String,
    pub logprob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<i64>>,
}

/// Token usage summary (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetails,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetails {
    message: String,
}

fn parse_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<ApiErrorResponse>(body)
        .ok()
        .map(|e| e.error.message)
}

fn record_chat_request(request: &ChatCompletionRequest) {
    runtime::with_recorder(|recorder| {
        record_model_config(recorder, request);
        record_messages(recorder, &request.messages);
        record_prompt(recorder, &request.messages);
        let payload = serde_json::to_string(request).unwrap_or_else(|_| "{}".to_string());
        recorder.track_custom("openai_chat_request", payload);
    });
}

fn record_chat_response(request: &ChatCompletionRequest, response: &ChatCompletionResponse) {
    runtime::with_recorder(|recorder| {
        // Record response metadata
        let meta = json!({
            "id": response.id,
            "model": response.model,
            "choices": response.choices.len(),
        });
        recorder.track_custom("openai_chat_response", meta.to_string());

        // Record each assistant message as AI output
        for choice in &response.choices {
            recorder.track_ai_output(&choice.message.content);

            // Token-level tracking (if logprobs enabled)
            if let Some(logprobs) = &choice.logprobs {
                if let Some(tokens) = &logprobs.content {
                    let payload = serde_json::to_string(tokens).unwrap_or_else(|_| "[]".to_string());
                    recorder.track_custom("ai_tokens", payload);
                }
            }
        }

        // Record usage
        if let Some(usage) = &response.usage {
            let payload = serde_json::to_string(usage).unwrap_or_else(|_| "{}".to_string());
            recorder.track_custom("ai_usage", payload);
        }

        // Record request config to tie request -> response
        let config = json!({
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        });
        recorder.track_custom("ai_model_config", config.to_string());
    });
}

fn record_error(request: &ChatCompletionRequest, status: StatusCode, message: &str) {
    runtime::with_recorder(|recorder| {
        let payload = json!({
            "status": status.as_u16(),
            "message": message,
            "model": request.model,
        });
        recorder.track_custom("ai_error", payload.to_string());
    });
}

fn record_model_config(recorder: &mut Recorder, request: &ChatCompletionRequest) {
    let config = json!({
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    });
    recorder.track_custom("ai_model_config", config.to_string());
}

fn record_messages(recorder: &mut Recorder, messages: &[ChatMessage]) {
    for msg in messages {
        let payload = json!({
            "role": msg.role,
            "content": msg.content,
            "name": msg.name,
        });
        recorder.track_custom("ai_message", payload.to_string());
    }
}

fn record_prompt(recorder: &mut Recorder, messages: &[ChatMessage]) {
    if let Some(last_user) = messages.iter().rev().find(|m| m.role == "user") {
        recorder.track_custom("ai_prompt", last_user.content.clone());
    }
}
