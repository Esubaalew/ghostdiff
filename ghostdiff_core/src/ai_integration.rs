//! # AI Integration Module
//!
//! Provides tracking capabilities for AI model interactions, including
//! prompts, completions, tokens, and embeddings.
//!
//! This module is designed to integrate with various AI providers and
//! capture detailed information about AI decision-making processes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique identifier for AI interactions.
pub type InteractionId = u64;

/// Timestamp in milliseconds since UNIX epoch.
pub type Timestamp = u64;

/// Represents a single token in an AI interaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    /// The token text or identifier.
    pub value: String,

    /// Log probability of this token (if available).
    pub logprob: Option<f64>,

    /// Token ID from the model's vocabulary (if available).
    pub token_id: Option<u64>,
}

impl Token {
    /// Creates a new token with just the value.
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            logprob: None,
            token_id: None,
        }
    }

    /// Creates a token with log probability information.
    pub fn with_logprob(value: impl Into<String>, logprob: f64) -> Self {
        Self {
            value: value.into(),
            logprob: Some(logprob),
            token_id: None,
        }
    }
}

/// Represents a message in a conversation with an AI model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender (e.g., "system", "user", "assistant").
    pub role: String,

    /// Content of the message.
    pub content: String,

    /// Optional name for the sender.
    pub name: Option<String>,
}

impl Message {
    /// Creates a new message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            name: None,
        }
    }

    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Creates a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Creates an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Configuration used for an AI model call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier (e.g., "gpt-4", "claude-3-opus").
    pub model: String,

    /// Temperature parameter for sampling.
    pub temperature: Option<f64>,

    /// Maximum tokens to generate.
    pub max_tokens: Option<u64>,

    /// Top-p (nucleus) sampling parameter.
    pub top_p: Option<f64>,

    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,

    /// Presence penalty.
    pub presence_penalty: Option<f64>,

    /// Stop sequences.
    pub stop_sequences: Vec<String>,

    /// Additional provider-specific parameters.
    pub extra: HashMap<String, String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model: "unknown".to_string(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: Vec::new(),
            extra: HashMap::new(),
        }
    }
}

impl ModelConfig {
    /// Creates a new model configuration with just the model name.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Sets the temperature parameter.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the max tokens parameter.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

/// Usage statistics for an AI interaction.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u64,

    /// Number of tokens in the completion.
    pub completion_tokens: u64,

    /// Total tokens used.
    pub total_tokens: u64,

    /// Estimated cost in USD (if available).
    pub cost_usd: Option<f64>,
}

impl Usage {
    /// Creates a new usage record.
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cost_usd: None,
        }
    }
}

/// Represents a complete AI interaction (prompt + completion).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    /// Unique identifier for this interaction.
    pub id: InteractionId,

    /// Timestamp when the interaction started.
    pub started_at: Timestamp,

    /// Timestamp when the interaction completed (if finished).
    pub completed_at: Option<Timestamp>,

    /// Messages sent to the model.
    pub messages: Vec<Message>,

    /// Model configuration used.
    pub config: ModelConfig,

    /// The completion generated by the model.
    pub completion: Option<String>,

    /// Individual tokens in the completion (if streaming or detailed logging).
    pub tokens: Vec<Token>,

    /// Usage statistics.
    pub usage: Option<Usage>,

    /// Any error that occurred.
    pub error: Option<String>,

    /// Custom metadata.
    pub metadata: HashMap<String, String>,
}

impl Interaction {
    /// Creates a new interaction.
    fn new(id: InteractionId, messages: Vec<Message>, config: ModelConfig) -> Self {
        Self {
            id,
            started_at: current_timestamp(),
            completed_at: None,
            messages,
            config,
            completion: None,
            tokens: Vec::new(),
            usage: None,
            error: None,
            metadata: HashMap::new(),
        }
    }
}

/// Represents an embedding vector from an AI model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Unique identifier.
    pub id: u64,

    /// The text that was embedded.
    pub text: String,

    /// The embedding vector.
    pub vector: Vec<f64>,

    /// Model used to generate the embedding.
    pub model: String,

    /// Dimensions of the embedding.
    pub dimensions: usize,

    /// Timestamp when created.
    pub created_at: Timestamp,
}

impl Embedding {
    /// Creates a new embedding.
    pub fn new(
        id: u64,
        text: impl Into<String>,
        vector: Vec<f64>,
        model: impl Into<String>,
    ) -> Self {
        let dimensions = vector.len();
        Self {
            id,
            text: text.into(),
            vector,
            model: model.into(),
            dimensions,
            created_at: current_timestamp(),
        }
    }

    /// Computes cosine similarity with another embedding.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::ai_integration::Embedding;
    ///
    /// let emb1 = Embedding::new(1, "hello", vec![1.0, 0.0, 0.0], "test");
    /// let emb2 = Embedding::new(2, "world", vec![1.0, 0.0, 0.0], "test");
    ///
    /// let similarity = emb1.cosine_similarity(&emb2);
    /// assert!((similarity - 1.0).abs() < 0.0001);
    /// ```
    pub fn cosine_similarity(&self, other: &Embedding) -> f64 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot_product: f64 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let magnitude_a: f64 = self.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let magnitude_b: f64 = other.vector.iter().map(|x| x * x).sum::<f64>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a * magnitude_b)
    }
}

/// Tracks AI model interactions for debugging and analysis.
///
/// The `AITracker` provides comprehensive logging of AI interactions,
/// including prompts, completions, token-by-token streaming, and embeddings.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::AITracker;
/// use ghostdiff_core::ai_integration::{Message, ModelConfig};
///
/// let mut tracker = AITracker::new();
///
/// // Start tracking an interaction
/// let interaction_id = tracker.start_interaction(
///     vec![Message::user("What is 2+2?")],
///     ModelConfig::new("gpt-4").with_temperature(0.7),
/// );
///
/// // Log tokens as they stream in
/// tracker.log_token(interaction_id, "The");
/// tracker.log_token(interaction_id, " answer");
/// tracker.log_token(interaction_id, " is");
/// tracker.log_token(interaction_id, " 4");
///
/// // Complete the interaction
/// tracker.complete_interaction(interaction_id, "The answer is 4");
///
/// // Get summary
/// println!("{}", tracker.summary());
/// ```
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AITracker {
    /// All tracked interactions.
    interactions: Vec<Interaction>,

    /// All tracked embeddings.
    embeddings: Vec<Embedding>,

    /// Counter for generating interaction IDs.
    next_interaction_id: InteractionId,

    /// Counter for generating embedding IDs.
    next_embedding_id: u64,

    /// Cumulative usage statistics.
    total_usage: Usage,
}

impl AITracker {
    /// Creates a new AI tracker.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::AITracker;
    ///
    /// let tracker = AITracker::new();
    /// assert_eq!(tracker.interaction_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of tracked interactions.
    pub fn interaction_count(&self) -> usize {
        self.interactions.len()
    }

    /// Returns the number of tracked embeddings.
    pub fn embedding_count(&self) -> usize {
        self.embeddings.len()
    }

    /// Returns all tracked interactions.
    pub fn interactions(&self) -> &[Interaction] {
        &self.interactions
    }

    /// Returns all tracked embeddings.
    pub fn embeddings(&self) -> &[Embedding] {
        &self.embeddings
    }

    /// Returns the total usage across all interactions.
    pub fn total_usage(&self) -> &Usage {
        &self.total_usage
    }

    /// Starts tracking a new AI interaction.
    ///
    /// # Arguments
    ///
    /// * `messages` - The messages sent to the model
    /// * `config` - The model configuration
    ///
    /// # Returns
    ///
    /// The ID of the new interaction for use in subsequent calls.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::AITracker;
    /// use ghostdiff_core::ai_integration::{Message, ModelConfig};
    ///
    /// let mut tracker = AITracker::new();
    /// let id = tracker.start_interaction(
    ///     vec![Message::system("You are helpful."), Message::user("Hi!")],
    ///     ModelConfig::new("claude-3-opus"),
    /// );
    /// ```
    pub fn start_interaction(
        &mut self,
        messages: Vec<Message>,
        config: ModelConfig,
    ) -> InteractionId {
        let id = self.next_interaction_id;
        self.next_interaction_id += 1;

        let interaction = Interaction::new(id, messages, config);
        self.interactions.push(interaction);

        id
    }

    /// Logs a token for a streaming interaction.
    ///
    /// # Arguments
    ///
    /// * `interaction_id` - The ID of the interaction
    /// * `token` - The token text
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::AITracker;
    /// use ghostdiff_core::ai_integration::{Message, ModelConfig};
    ///
    /// let mut tracker = AITracker::new();
    /// let id = tracker.start_interaction(
    ///     vec![Message::user("Hello")],
    ///     ModelConfig::new("gpt-4"),
    /// );
    ///
    /// tracker.log_token(id, "Hello");
    /// tracker.log_token(id, " there");
    /// tracker.log_token(id, "!");
    /// ```
    pub fn log_token(&mut self, interaction_id: InteractionId, token: &str) {
        if let Some(interaction) = self
            .interactions
            .iter_mut()
            .find(|i| i.id == interaction_id)
        {
            interaction.tokens.push(Token::new(token));
        }
    }

    /// Logs a token with log probability information.
    pub fn log_token_with_logprob(
        &mut self,
        interaction_id: InteractionId,
        token: &str,
        logprob: f64,
    ) {
        if let Some(interaction) = self
            .interactions
            .iter_mut()
            .find(|i| i.id == interaction_id)
        {
            interaction.tokens.push(Token::with_logprob(token, logprob));
        }
    }

    /// Completes an interaction with the final output.
    ///
    /// # Arguments
    ///
    /// * `interaction_id` - The ID of the interaction
    /// * `completion` - The complete response from the model
    pub fn complete_interaction(
        &mut self,
        interaction_id: InteractionId,
        completion: impl Into<String>,
    ) {
        if let Some(interaction) = self
            .interactions
            .iter_mut()
            .find(|i| i.id == interaction_id)
        {
            interaction.completion = Some(completion.into());
            interaction.completed_at = Some(current_timestamp());
        }
    }

    /// Records an error for an interaction.
    pub fn record_error(&mut self, interaction_id: InteractionId, error: impl Into<String>) {
        if let Some(interaction) = self
            .interactions
            .iter_mut()
            .find(|i| i.id == interaction_id)
        {
            interaction.error = Some(error.into());
            interaction.completed_at = Some(current_timestamp());
        }
    }

    /// Records usage statistics for an interaction.
    pub fn record_usage(&mut self, interaction_id: InteractionId, usage: Usage) {
        if let Some(interaction) = self
            .interactions
            .iter_mut()
            .find(|i| i.id == interaction_id)
        {
            self.total_usage.prompt_tokens += usage.prompt_tokens;
            self.total_usage.completion_tokens += usage.completion_tokens;
            self.total_usage.total_tokens += usage.total_tokens;
            if let (Some(existing), Some(new)) = (self.total_usage.cost_usd, usage.cost_usd) {
                self.total_usage.cost_usd = Some(existing + new);
            } else if usage.cost_usd.is_some() {
                self.total_usage.cost_usd = usage.cost_usd;
            }

            interaction.usage = Some(usage);
        }
    }

    /// Adds metadata to an interaction.
    pub fn add_metadata(
        &mut self,
        interaction_id: InteractionId,
        key: impl Into<String>,
        value: impl Into<String>,
    ) {
        if let Some(interaction) = self
            .interactions
            .iter_mut()
            .find(|i| i.id == interaction_id)
        {
            interaction.metadata.insert(key.into(), value.into());
        }
    }

    /// Tracks an embedding.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::AITracker;
    ///
    /// let mut tracker = AITracker::new();
    /// tracker.track_embedding(
    ///     "Hello, world!",
    ///     vec![0.1, 0.2, 0.3, 0.4],
    ///     "text-embedding-3-small",
    /// );
    ///
    /// assert_eq!(tracker.embedding_count(), 1);
    /// ```
    pub fn track_embedding(
        &mut self,
        text: impl Into<String>,
        vector: Vec<f64>,
        model: impl Into<String>,
    ) -> u64 {
        let id = self.next_embedding_id;
        self.next_embedding_id += 1;

        let embedding = Embedding::new(id, text, vector, model);
        self.embeddings.push(embedding);

        id
    }

    /// Gets an interaction by ID.
    pub fn get_interaction(&self, id: InteractionId) -> Option<&Interaction> {
        self.interactions.iter().find(|i| i.id == id)
    }

    /// Gets an embedding by ID.
    pub fn get_embedding(&self, id: u64) -> Option<&Embedding> {
        self.embeddings.iter().find(|e| e.id == id)
    }

    /// Generates a summary of all tracked AI activity.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::AITracker;
    /// use ghostdiff_core::ai_integration::{Message, ModelConfig, Usage};
    ///
    /// let mut tracker = AITracker::new();
    /// let id = tracker.start_interaction(
    ///     vec![Message::user("Test")],
    ///     ModelConfig::new("gpt-4"),
    /// );
    /// tracker.complete_interaction(id, "Response");
    /// tracker.record_usage(id, Usage::new(10, 5));
    ///
    /// let summary = tracker.summary();
    /// println!("{}", summary);
    /// ```
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();

        lines.push("AI Tracker Summary".to_string());
        lines.push("==================".to_string());
        lines.push(format!("Interactions: {}", self.interactions.len()));
        lines.push(format!("Embeddings: {}", self.embeddings.len()));
        lines.push(format!(
            "Total tokens: {} (prompt: {}, completion: {})",
            self.total_usage.total_tokens,
            self.total_usage.prompt_tokens,
            self.total_usage.completion_tokens
        ));

        if let Some(cost) = self.total_usage.cost_usd {
            lines.push(format!("Estimated cost: ${:.4}", cost));
        }

        // Model breakdown
        let mut model_counts: HashMap<String, usize> = HashMap::new();
        for interaction in &self.interactions {
            *model_counts
                .entry(interaction.config.model.clone())
                .or_insert(0) += 1;
        }

        if !model_counts.is_empty() {
            lines.push(String::new());
            lines.push("Models used:".to_string());
            for (model, count) in model_counts {
                lines.push(format!("  - {}: {} calls", model, count));
            }
        }

        // Error count
        let error_count = self
            .interactions
            .iter()
            .filter(|i| i.error.is_some())
            .count();
        if error_count > 0 {
            lines.push(format!("\nErrors: {}", error_count));
        }

        lines.join("\n")
    }

    /// Serializes the tracker to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a tracker from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Legacy method for simple token logging (prints to stdout).
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::AITracker;
    ///
    /// AITracker::log_token_simple("hello");
    /// ```
    pub fn log_token_simple(token: &str) {
        println!("[AI] token: {}", token);
    }
}

/// Returns the current timestamp as milliseconds since UNIX epoch.
fn current_timestamp() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as Timestamp)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_basic() {
        let mut tracker = AITracker::new();
        assert_eq!(tracker.interaction_count(), 0);

        let id =
            tracker.start_interaction(vec![Message::user("Hello")], ModelConfig::new("test-model"));

        assert_eq!(tracker.interaction_count(), 1);

        tracker.log_token(id, "Hi");
        tracker.log_token(id, " there");
        tracker.complete_interaction(id, "Hi there");

        let interaction = tracker.get_interaction(id).unwrap();
        assert_eq!(interaction.tokens.len(), 2);
        assert_eq!(interaction.completion, Some("Hi there".to_string()));
    }

    #[test]
    fn test_embedding_similarity() {
        let emb1 = Embedding::new(1, "hello", vec![1.0, 0.0, 0.0], "test");
        let emb2 = Embedding::new(2, "hello", vec![1.0, 0.0, 0.0], "test");
        let emb3 = Embedding::new(3, "world", vec![0.0, 1.0, 0.0], "test");

        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < 0.0001);
        assert!(emb1.cosine_similarity(&emb3).abs() < 0.0001);
    }

    #[test]
    fn test_usage_tracking() {
        let mut tracker = AITracker::new();

        let id1 = tracker.start_interaction(vec![], ModelConfig::new("model"));
        tracker.record_usage(id1, Usage::new(100, 50));

        let id2 = tracker.start_interaction(vec![], ModelConfig::new("model"));
        tracker.record_usage(id2, Usage::new(200, 100));

        assert_eq!(tracker.total_usage().prompt_tokens, 300);
        assert_eq!(tracker.total_usage().completion_tokens, 150);
        assert_eq!(tracker.total_usage().total_tokens, 450);
    }

    #[test]
    fn test_serialization() {
        let mut tracker = AITracker::new();
        let id = tracker.start_interaction(
            vec![Message::system("System"), Message::user("User")],
            ModelConfig::new("gpt-4").with_temperature(0.5),
        );
        tracker.complete_interaction(id, "Response");

        let json = tracker.to_json().unwrap();
        let restored = AITracker::from_json(&json).unwrap();

        assert_eq!(tracker.interaction_count(), restored.interaction_count());
    }
}
