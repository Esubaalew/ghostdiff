//! # AI Semantic Diff Module
//!
//! Provides deterministic, structured semantic diffing for AI-related events,
//! including token divergence, logprob drift, entropy comparison, embeddings,
//! and prompt/config changes.

use crate::recorder::{Event, EventKind, Recorder};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Root-cause classification for AI divergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootCause {
    /// No AI divergence detected.
    None,
    /// Differences likely caused by sampling randomness.
    SamplingInstability,
    /// Prompt changed between runs.
    PromptChange,
    /// Model configuration changed (temperature, max_tokens, etc.).
    ConfigChange,
    /// Model name/ID changed.
    ModelChange,
    /// Upstream function behavior diverged before AI call.
    UpstreamFunctionDivergence,
}

/// Configuration for AI semantic diffing.
#[derive(Debug, Clone)]
pub struct AiDiffConfig {
    /// Threshold for logprob drift (absolute difference in mean logprob).
    pub logprob_drift_threshold: f64,
    /// Threshold for entropy difference (absolute difference in mean -logprob).
    pub entropy_delta_threshold: f64,
    /// Threshold for embedding cosine similarity.
    pub embedding_similarity_threshold: f64,
    /// Maximum tokens to compare when detecting divergence.
    pub max_token_compare: usize,
}

impl Default for AiDiffConfig {
    fn default() -> Self {
        Self {
            logprob_drift_threshold: 0.5,
            entropy_delta_threshold: 0.5,
            embedding_similarity_threshold: 0.85,
            max_token_compare: 4096,
        }
    }
}

/// Structured AI diff result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiDiffResult {
    /// Run A identifier.
    pub run_a_id: String,
    /// Run B identifier.
    pub run_b_id: String,

    /// Detected token divergence.
    pub token_divergence: Option<TokenDivergence>,
    /// Logprob drift summary.
    pub logprob_drift: Option<LogprobDrift>,
    /// Entropy comparison summary.
    pub entropy_comparison: Option<EntropyComparison>,
    /// Embedding similarity comparison.
    pub embedding_similarity: Option<EmbeddingComparison>,

    /// Prompt changed between runs.
    pub prompt_change: bool,
    /// Configuration changed between runs.
    pub config_change: bool,
    /// Model changed between runs.
    pub model_change: bool,
    /// Upstream function behavior diverged.
    pub upstream_divergence: bool,

    /// Root-cause classification.
    pub root_cause: RootCause,

    /// Human-readable explanation.
    pub explanation: String,
}

/// Token divergence information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDivergence {
    /// Index where tokens first diverged.
    pub first_divergence_index: usize,
    /// Token from run A at divergence.
    pub token_a: String,
    /// Token from run B at divergence.
    pub token_b: String,
    /// Total tokens compared.
    pub compared: usize,
}

/// Logprob drift summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobDrift {
    /// Mean logprob for run A.
    pub mean_logprob_a: f64,
    /// Mean logprob for run B.
    pub mean_logprob_b: f64,
    /// Absolute drift between means.
    pub drift: f64,
}

/// Entropy comparison summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyComparison {
    /// Estimated entropy for run A.
    pub entropy_a: f64,
    /// Estimated entropy for run B.
    pub entropy_b: f64,
    /// Absolute difference.
    pub delta: f64,
}

/// Embedding similarity comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingComparison {
    /// Cosine similarity between embeddings.
    pub cosine_similarity: f64,
    /// Whether similarity is above threshold.
    pub above_threshold: bool,
}

/// AI diff engine.
#[derive(Debug, Default)]
pub struct AiDiffEngine {
    config: AiDiffConfig,
}

impl AiDiffEngine {
    /// Creates a new AI diff engine with default configuration.
    pub fn new() -> Self {
        Self {
            config: AiDiffConfig::default(),
        }
    }

    /// Creates a new AI diff engine with a custom configuration.
    pub fn with_config(config: AiDiffConfig) -> Self {
        Self { config }
    }

    /// Compares two recorders and returns AI semantic diff result.
    pub fn compare(&self, run_a: &Recorder, run_b: &Recorder) -> AiDiffResult {
        let a_ctx = AiContext::from_recorder(run_a);
        let b_ctx = AiContext::from_recorder(run_b);

        let token_divergence =
            detect_token_divergence(&a_ctx.tokens, &b_ctx.tokens, self.config.max_token_compare);

        let logprob_drift = compute_logprob_drift(&a_ctx.tokens, &b_ctx.tokens);
        let entropy_comparison = compute_entropy(&a_ctx.tokens, &b_ctx.tokens);

        let embedding_similarity = compare_embeddings(
            &a_ctx.embeddings,
            &b_ctx.embeddings,
            self.config.embedding_similarity_threshold,
        );

        let prompt_change = a_ctx.prompt != b_ctx.prompt;
        let model_change =
            a_ctx.model != b_ctx.model && a_ctx.model.is_some() && b_ctx.model.is_some();
        let config_change =
            a_ctx.config != b_ctx.config && a_ctx.config.is_some() && b_ctx.config.is_some();
        let upstream_divergence = detect_upstream_divergence(
            run_a,
            run_b,
            a_ctx.first_ai_event_index,
            b_ctx.first_ai_event_index,
        );

        let root_cause = classify_root_cause(RootCauseInputs {
            prompt_change,
            config_change,
            model_change,
            upstream_divergence,
            token_divergence: &token_divergence,
            logprob_drift: &logprob_drift,
            entropy: &entropy_comparison,
            config: &self.config,
        });

        let explanation = build_explanation(ExplanationInputs {
            root_cause: &root_cause,
            token_divergence: &token_divergence,
            logprob_drift: &logprob_drift,
            entropy: &entropy_comparison,
            embedding_similarity: &embedding_similarity,
            prompt_change,
            config_change,
            model_change,
            upstream_divergence,
        });

        AiDiffResult {
            run_a_id: run_a.run_id().to_string(),
            run_b_id: run_b.run_id().to_string(),
            token_divergence,
            logprob_drift,
            entropy_comparison,
            embedding_similarity,
            prompt_change,
            config_change,
            model_change,
            upstream_divergence,
            root_cause,
            explanation,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ModelConfigSnapshot {
    model: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
struct TokenRecord {
    token: String,
    logprob: Option<f64>,
}

#[derive(Debug, Clone)]
struct EmbeddingRecord {
    vector: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
struct AiContext {
    prompt: Option<String>,
    model: Option<String>,
    config: Option<ModelConfigSnapshot>,
    tokens: Vec<TokenRecord>,
    embeddings: Vec<EmbeddingRecord>,
    first_ai_event_index: Option<usize>,
}

impl AiContext {
    fn from_recorder(recorder: &Recorder) -> Self {
        let mut ctx = AiContext::default();

        for (idx, event) in recorder.events().iter().enumerate() {
            match &event.kind {
                EventKind::Custom { kind, payload } => match kind.as_str() {
                    "ai_prompt" => {
                        if ctx.first_ai_event_index.is_none() {
                            ctx.first_ai_event_index = Some(idx);
                        }
                        ctx.prompt = Some(payload.clone());
                    }
                    "ai_model_config" => {
                        let cfg = parse_model_config(payload);
                        if let Some(cfg) = cfg {
                            if ctx.model.is_none() {
                                ctx.model = cfg.model.clone();
                            }
                            ctx.config = Some(cfg);
                        }
                    }
                    "ai_tokens" => {
                        if ctx.first_ai_event_index.is_none() {
                            ctx.first_ai_event_index = Some(idx);
                        }
                        let tokens = parse_tokens(payload);
                        ctx.tokens.extend(tokens);
                    }
                    "embedding" => {
                        if ctx.first_ai_event_index.is_none() {
                            ctx.first_ai_event_index = Some(idx);
                        }
                        if let Some(emb) = parse_embedding(payload) {
                            ctx.embeddings.push(emb);
                        }
                    }
                    _ => {}
                },
                EventKind::AIOutput { content } => {
                    if ctx.first_ai_event_index.is_none() {
                        ctx.first_ai_event_index = Some(idx);
                    }
                    if ctx.tokens.is_empty() {
                        ctx.tokens = content
                            .split_whitespace()
                            .map(|t| TokenRecord {
                                token: t.to_string(),
                                logprob: None,
                            })
                            .collect();
                    }
                }
                _ => {}
            }
        }

        ctx
    }
}

fn parse_model_config(payload: &str) -> Option<ModelConfigSnapshot> {
    let value: Value = serde_json::from_str(payload).ok()?;
    let model = value
        .get("model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let temperature = value.get("temperature").and_then(|v| v.as_f64());
    let max_tokens = value.get("max_tokens").and_then(|v| v.as_u64());
    Some(ModelConfigSnapshot {
        model,
        temperature,
        max_tokens,
    })
}

fn parse_tokens(payload: &str) -> Vec<TokenRecord> {
    let value: Value = serde_json::from_str(payload).unwrap_or(Value::Null);
    let mut tokens = Vec::new();

    if let Value::Array(items) = value {
        for item in items {
            if let Some(token) = item.get("token").and_then(|v| v.as_str()) {
                let logprob = item.get("logprob").and_then(|v| v.as_f64());
                tokens.push(TokenRecord {
                    token: token.to_string(),
                    logprob,
                });
            }
        }
    }

    tokens
}

fn parse_embedding(payload: &str) -> Option<EmbeddingRecord> {
    let value: Value = serde_json::from_str(payload).ok()?;
    let vector_value = value
        .get("vector")
        .or_else(|| value.get("embedding"))
        .or_else(|| value.get("data"));

    if let Some(Value::Array(arr)) = vector_value {
        let mut vec = Vec::new();
        for v in arr {
            if let Some(f) = v.as_f64() {
                vec.push(f);
            }
        }
        if !vec.is_empty() {
            return Some(EmbeddingRecord { vector: vec });
        }
    }
    None
}

fn detect_token_divergence(
    tokens_a: &[TokenRecord],
    tokens_b: &[TokenRecord],
    max_compare: usize,
) -> Option<TokenDivergence> {
    let limit = tokens_a.len().min(tokens_b.len()).min(max_compare);

    for i in 0..limit {
        if tokens_a[i].token != tokens_b[i].token {
            return Some(TokenDivergence {
                first_divergence_index: i,
                token_a: tokens_a[i].token.clone(),
                token_b: tokens_b[i].token.clone(),
                compared: limit,
            });
        }
    }

    None
}

fn compute_logprob_drift(
    tokens_a: &[TokenRecord],
    tokens_b: &[TokenRecord],
) -> Option<LogprobDrift> {
    let logprobs_a: Vec<f64> = tokens_a.iter().filter_map(|t| t.logprob).collect();
    let logprobs_b: Vec<f64> = tokens_b.iter().filter_map(|t| t.logprob).collect();

    if logprobs_a.is_empty() || logprobs_b.is_empty() {
        return None;
    }

    let mean_a = logprobs_a.iter().sum::<f64>() / logprobs_a.len() as f64;
    let mean_b = logprobs_b.iter().sum::<f64>() / logprobs_b.len() as f64;
    let drift = (mean_a - mean_b).abs();

    Some(LogprobDrift {
        mean_logprob_a: mean_a,
        mean_logprob_b: mean_b,
        drift,
    })
}

fn compute_entropy(
    tokens_a: &[TokenRecord],
    tokens_b: &[TokenRecord],
) -> Option<EntropyComparison> {
    let ent_a = estimate_entropy(tokens_a)?;
    let ent_b = estimate_entropy(tokens_b)?;
    let delta = (ent_a - ent_b).abs();

    Some(EntropyComparison {
        entropy_a: ent_a,
        entropy_b: ent_b,
        delta,
    })
}

fn estimate_entropy(tokens: &[TokenRecord]) -> Option<f64> {
    let logprobs: Vec<f64> = tokens.iter().filter_map(|t| t.logprob).collect();
    if logprobs.is_empty() {
        return None;
    }

    // Approximate entropy as mean negative logprob
    let mean_neg_logprob = -logprobs.iter().sum::<f64>() / logprobs.len() as f64;
    Some(mean_neg_logprob)
}

fn compare_embeddings(
    emb_a: &[EmbeddingRecord],
    emb_b: &[EmbeddingRecord],
    threshold: f64,
) -> Option<EmbeddingComparison> {
    if emb_a.is_empty() || emb_b.is_empty() {
        return None;
    }

    let sim = cosine_similarity(&emb_a[0].vector, &emb_b[0].vector);
    Some(EmbeddingComparison {
        cosine_similarity: sim,
        above_threshold: sim >= threshold,
    })
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
    let mag_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

fn detect_upstream_divergence(
    run_a: &Recorder,
    run_b: &Recorder,
    ai_idx_a: Option<usize>,
    ai_idx_b: Option<usize>,
) -> bool {
    let limit_a = ai_idx_a.unwrap_or(run_a.events().len());
    let limit_b = ai_idx_b.unwrap_or(run_b.events().len());

    let calls_a = collect_function_signatures(run_a.events(), limit_a);
    let calls_b = collect_function_signatures(run_b.events(), limit_b);

    calls_a != calls_b
}

fn collect_function_signatures(events: &[Event], limit: usize) -> Vec<String> {
    events
        .iter()
        .take(limit)
        .filter_map(|e| match &e.kind {
            EventKind::FunctionCall {
                name,
                args,
                return_value,
            } => Some(format!("{}({:?})->{:?}", name, args, return_value)),
            _ => None,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct RootCauseInputs<'a> {
    prompt_change: bool,
    config_change: bool,
    model_change: bool,
    upstream_divergence: bool,
    token_divergence: &'a Option<TokenDivergence>,
    logprob_drift: &'a Option<LogprobDrift>,
    entropy: &'a Option<EntropyComparison>,
    config: &'a AiDiffConfig,
}

fn classify_root_cause(inputs: RootCauseInputs<'_>) -> RootCause {
    if inputs.upstream_divergence {
        return RootCause::UpstreamFunctionDivergence;
    }
    if inputs.prompt_change {
        return RootCause::PromptChange;
    }
    if inputs.model_change {
        return RootCause::ModelChange;
    }
    if inputs.config_change {
        return RootCause::ConfigChange;
    }

    if inputs.token_divergence.is_some()
        || inputs
            .logprob_drift
            .as_ref()
            .map(|d| d.drift >= inputs.config.logprob_drift_threshold)
            .unwrap_or(false)
        || inputs
            .entropy
            .as_ref()
            .map(|e| e.delta >= inputs.config.entropy_delta_threshold)
            .unwrap_or(false)
    {
        return RootCause::SamplingInstability;
    }

    RootCause::None
}

#[derive(Debug, Clone)]
struct ExplanationInputs<'a> {
    root_cause: &'a RootCause,
    token_divergence: &'a Option<TokenDivergence>,
    logprob_drift: &'a Option<LogprobDrift>,
    entropy: &'a Option<EntropyComparison>,
    embedding_similarity: &'a Option<EmbeddingComparison>,
    prompt_change: bool,
    config_change: bool,
    model_change: bool,
    upstream_divergence: bool,
}

fn build_explanation(inputs: ExplanationInputs<'_>) -> String {
    let mut lines = Vec::new();

    lines.push(format!("Root cause: {:?}", inputs.root_cause));

    if let Some(div) = inputs.token_divergence {
        lines.push(format!(
            "Divergence started at token {} ('{}' vs '{}')",
            div.first_divergence_index, div.token_a, div.token_b
        ));
    } else {
        lines.push("No token-level divergence detected.".to_string());
    }

    if let Some(drift) = inputs.logprob_drift {
        lines.push(format!(
            "Logprob drift: mean_a={:.3}, mean_b={:.3}, drift={:.3}",
            drift.mean_logprob_a, drift.mean_logprob_b, drift.drift
        ));
    }

    if let Some(ent) = inputs.entropy {
        lines.push(format!(
            "Entropy delta: a={:.3}, b={:.3}, delta={:.3}",
            ent.entropy_a, ent.entropy_b, ent.delta
        ));
    }

    if let Some(emb) = inputs.embedding_similarity {
        lines.push(format!(
            "Embedding similarity: {:.3} (above threshold: {})",
            emb.cosine_similarity, emb.above_threshold
        ));
    }

    if inputs.prompt_change {
        lines.push("Prompt changed between runs.".to_string());
    }
    if inputs.config_change {
        lines.push("Model configuration changed between runs.".to_string());
    }
    if inputs.model_change {
        lines.push("Model changed between runs.".to_string());
    }
    if inputs.upstream_divergence {
        lines.push("Upstream function behavior diverged before AI call.".to_string());
    }

    let determinism = match inputs.root_cause {
        RootCause::SamplingInstability => "stochastic",
        RootCause::None => "deterministic",
        _ => "deterministic",
    };

    lines.push(format!("Determinism assessment: {}", determinism));

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Recorder;

    #[test]
    fn test_prompt_change() {
        let mut a = Recorder::new("a");
        let mut b = Recorder::new("b");
        a.track_custom("ai_prompt", "hello");
        b.track_custom("ai_prompt", "hi");

        let diff = AiDiffEngine::new().compare(&a, &b);
        assert!(diff.prompt_change);
        assert_eq!(diff.root_cause, RootCause::PromptChange);
    }

    #[test]
    fn test_token_divergence() {
        let mut a = Recorder::new("a");
        let mut b = Recorder::new("b");
        a.track_custom(
            "ai_tokens",
            r#"[{"token":"Hello","logprob":-0.1},{"token":"world","logprob":-0.2}]"#,
        );
        b.track_custom(
            "ai_tokens",
            r#"[{"token":"Hello","logprob":-0.1},{"token":"mars","logprob":-0.2}]"#,
        );

        let diff = AiDiffEngine::new().compare(&a, &b);
        assert!(diff.token_divergence.is_some());
        assert_eq!(diff.root_cause, RootCause::SamplingInstability);
    }
}
