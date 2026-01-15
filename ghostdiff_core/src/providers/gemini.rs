//! Gemini provider integration.
//!
//! Implements direct Gemini API calls (generateContent) with
//! recording into `ghostdiff_core::Recorder` via the runtime.

use crate::runtime;
use crate::Recorder;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

/// Gemini client for generateContent requests.
#[derive(Clone)]
pub struct GeminiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl GeminiClient {
    /// Creates a new client with default Gemini API base URL.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, "https://generativelanguage.googleapis.com/v1beta")
    }

    /// Creates a new client with a custom base URL.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
        }
    }

    /// Sends a generateContent request.
    pub async fn generate_content(
        &self,
        request: GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError> {
        record_request(&request);

        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url.trim_end_matches('/'),
            request.model,
            self.api_key
        );

        let response = self
            .http
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(GeminiError::Http)?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let message = parse_error_message(&body).unwrap_or(body);
            record_error(&request, status, &message);
            return Err(GeminiError::Api { status, message });
        }

        let response: GenerateContentResponse = response.json().await.map_err(GeminiError::Http)?;

        record_response(&request, &response);

        Ok(response)
    }

    /// Convenience helper: returns the first candidate text.
    pub async fn generate_text(
        &self,
        request: GenerateContentRequest,
    ) -> Result<String, GeminiError> {
        let response = self.generate_content(request).await?;
        Ok(response.first_text().unwrap_or_default())
    }
}

/// Errors for Gemini operations.
#[derive(Debug, Error)]
pub enum GeminiError {
    /// HTTP or network error.
    #[error("HTTP error: {0}")]
    Http(reqwest::Error),

    /// API returned a non-success status.
    #[error("API error ({status}): {message}")]
    Api { status: StatusCode, message: String },
}

/// Gemini generateContent request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateContentRequest {
    /// Model ID (e.g., "gemini-1.5-flash")
    #[serde(skip_serializing)]
    pub model: String,

    /// Content messages.
    pub contents: Vec<GeminiContent>,

    /// Generation config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,

    /// Safety settings (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<SafetySetting>>,
}

impl GenerateContentRequest {
    /// Creates a new request.
    pub fn new(model: impl Into<String>, contents: Vec<GeminiContent>) -> Self {
        Self {
            model: model.into(),
            contents,
            generation_config: None,
            safety_settings: None,
        }
    }

    /// Sets generation config.
    pub fn with_generation_config(mut self, cfg: GenerationConfig) -> Self {
        self.generation_config = Some(cfg);
        self
    }
}

/// Gemini content message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

impl GeminiContent {
    /// User message.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            parts: vec![GeminiPart::text(text)],
        }
    }

    /// Model message.
    pub fn model(text: impl Into<String>) -> Self {
        Self {
            role: "model".to_string(),
            parts: vec![GeminiPart::text(text)],
        }
    }
}

/// Gemini content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiPart {
    pub text: String,
}

impl GeminiPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Generation config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

/// Safety setting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: String,
    pub threshold: String,
}

/// Gemini response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateContentResponse {
    pub candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<UsageMetadata>,
}

impl GenerateContentResponse {
    /// Returns first candidate text.
    pub fn first_text(&self) -> Option<String> {
        self.candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .map(|p| p.text.clone())
    }
}

/// Gemini candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub content: CandidateContent,
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
}

/// Candidate content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

/// Usage metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: Option<u64>,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u64>,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: Option<u64>,
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

fn record_request(request: &GenerateContentRequest) {
    runtime::with_recorder(|recorder| {
        record_model_config(recorder, request);
        record_messages(recorder, &request.contents);
        record_prompt(recorder, &request.contents);
        let payload = serde_json::to_string(request).unwrap_or_else(|_| "{}".to_string());
        recorder.track_custom("gemini_request", payload);
    });
}

fn record_response(request: &GenerateContentRequest, response: &GenerateContentResponse) {
    runtime::with_recorder(|recorder| {
        let meta = json!({
            "candidates": response.candidates.len(),
            "model": request.model,
        });
        recorder.track_custom("gemini_response", meta.to_string());

        for candidate in &response.candidates {
            if let Some(text) = candidate.content.parts.first().map(|p| p.text.clone()) {
                recorder.track_ai_output(&text);

                // Token-level tracking (best-effort): whitespace split
                let tokens = text
                    .split_whitespace()
                    .map(|t| json!({"token": t, "logprob": serde_json::Value::Null}))
                    .collect::<Vec<_>>();
                recorder.track_custom(
                    "ai_tokens",
                    serde_json::to_string(&tokens).unwrap_or_else(|_| "[]".to_string()),
                );
            }
        }

        if let Some(usage) = &response.usage_metadata {
            let payload = json!({
                "prompt_tokens": usage.prompt_token_count,
                "completion_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count,
            });
            recorder.track_custom("ai_usage", payload.to_string());
        }
    });
}

fn record_error(request: &GenerateContentRequest, status: StatusCode, message: &str) {
    runtime::with_recorder(|recorder| {
        let payload = json!({
            "status": status.as_u16(),
            "message": message,
            "model": request.model,
        });
        recorder.track_custom("ai_error", payload.to_string());
    });
}

fn record_model_config(recorder: &mut Recorder, request: &GenerateContentRequest) {
    let cfg = request.generation_config.clone();
    let config = json!({
        "model": request.model,
        "temperature": cfg.as_ref().and_then(|c| c.temperature),
        "max_tokens": cfg.as_ref().and_then(|c| c.max_output_tokens),
        "top_p": cfg.as_ref().and_then(|c| c.top_p),
        "top_k": cfg.as_ref().and_then(|c| c.top_k),
    });
    recorder.track_custom("ai_model_config", config.to_string());
}

fn record_messages(recorder: &mut Recorder, contents: &[GeminiContent]) {
    for msg in contents {
        let text = msg
            .parts
            .first()
            .map(|p| p.text.clone())
            .unwrap_or_default();
        let payload = json!({
            "role": msg.role,
            "content": text,
        });
        recorder.track_custom("ai_message", payload.to_string());
    }
}

fn record_prompt(recorder: &mut Recorder, contents: &[GeminiContent]) {
    if let Some(last_user) = contents.iter().rev().find(|m| m.role == "user") {
        let text = last_user
            .parts
            .first()
            .map(|p| p.text.clone())
            .unwrap_or_default();
        recorder.track_custom("ai_prompt", text);
    }
}
