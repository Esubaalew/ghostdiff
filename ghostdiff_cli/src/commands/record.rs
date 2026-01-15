//! Record command implementation.
//!
//! Records program execution events to a JSON file.
//!
//! ## AI Provider Integration
//!
//! Example:
//! ```bash
//! ghostdiff record --output run.json \
//!   --ai-provider openai \
//!   --model gpt-4.1-mini \
//!   --temperature 0.7 \
//!   --prompt "Explain Rust lifetimes" \
//!   --api-key $OPENAI_API_KEY
//! ```

use anyhow::{bail, Context, Result};
use clap::Args;
use colored::Colorize;
use ghostdiff::track;
use ghostdiff_core::providers::openai::{ChatCompletionRequest, ChatMessage, OpenAIClient, OpenAIError};
use ghostdiff_core::providers::gemini::{GeminiClient, GenerateContentRequest, GeminiContent, GenerationConfig, GeminiError};
use ghostdiff_core::runtime::{init_recorder, take_recorder};
use ghostdiff_core::{AITracker, Recorder};
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct RecordArgs {
    /// Output file path for the recorded run (JSON)
    #[arg(short, long)]
    pub output: PathBuf,

    /// Unique identifier for this run
    #[arg(long, default_value = "run")]
    pub run_id: String,

    /// Simulate AI interactions for demo purposes
    #[arg(long)]
    pub simulate_ai: bool,

    /// Number of events to simulate
    #[arg(long, default_value = "5")]
    pub num_events: usize,

    /// AI provider to use (e.g., openai)
    #[arg(long)]
    pub ai_provider: Option<String>,

    /// OpenAI-compatible model (e.g., gpt-4.1-mini)
    #[arg(long)]
    pub model: Option<String>,

    /// Temperature parameter for sampling
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Maximum tokens to generate
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Prompt to send to the AI provider
    #[arg(long)]
    pub prompt: Option<String>,

    /// OpenAI API key (or set OPENAI_API_KEY)
    #[arg(long, env = "OPENAI_API_KEY")]
    pub api_key: Option<String>,

    /// OpenAI-compatible base URL
    #[arg(long, default_value = "https://api.openai.com/v1")]
    pub api_base: String,

    /// Enable token-level logprobs (if supported)
    #[arg(long, default_value_t = false)]
    pub logprobs: bool,
}

/// Execute the record command.
pub async fn execute(args: RecordArgs, verbose: bool) -> Result<()> {
    println!("{}", "📹 Recording execution...".cyan().bold());
    println!();

    if let Some(provider) = args.ai_provider.as_deref() {
        record_with_ai_provider(&args, provider, verbose).await
    } else if args.simulate_ai {
        record_simulated_ai(&args, verbose)
    } else {
        record_demo(&args, verbose)
    }
}

async fn record_with_ai_provider(args: &RecordArgs, provider: &str, verbose: bool) -> Result<()> {
    match provider {
        "openai" => record_openai(args, verbose).await,
        "gemini" => record_gemini(args, verbose).await,
        _ => bail!("Unsupported AI provider: {}", provider),
    }
}

async fn record_gemini(args: &RecordArgs, verbose: bool) -> Result<()> {
    let api_key = match args.api_key.clone() {
        Some(k) => k,
        None => std::env::var("GOOGLE_API_KEY")
            .context("Missing API key (use --api-key or set GOOGLE_API_KEY)")?,
    };

    let model = args
        .model
        .clone()
        .unwrap_or_else(|| "gemini-1.5-flash".to_string());

    let prompt = args
        .prompt
        .clone()
        .context("Missing prompt (use --prompt)")?;

    init_recorder(&args.run_id);

    let base_url = if args.api_base == "https://api.openai.com/v1" {
        "https://generativelanguage.googleapis.com/v1beta".to_string()
    } else {
        args.api_base.clone()
    };

    let client = GeminiClient::with_base_url(api_key, base_url);

    if verbose {
        println!("  Provider: {}", "gemini".yellow());
        println!("  Model: {}", model.cyan());
        println!("  Temperature: {}", args.temperature.unwrap_or(1.0));
        println!("  Prompt: {}", prompt.dimmed());
        println!();
    }

    let response = ask_gemini(
        &prompt,
        &client,
        &model,
        args.temperature,
        args.max_tokens,
    )
    .await
    .map_err(|e| anyhow::anyhow!(e))?;

    if verbose {
        println!("  Response: {}", response.green());
    }

    let mut recorder = take_recorder().context("Recorder not initialized")?;
    recorder.finalize();

    let json = recorder.to_json().context("Failed to serialize recording")?;

    fs::write(&args.output, &json).with_context(|| {
        format!("Failed to write recording to {}", args.output.display())
    })?;

    println!();
    println!(
        "{} Recorded {} events to {}",
        "✓".green().bold(),
        recorder.event_count(),
        args.output.display().to_string().cyan()
    );

    Ok(())
}

async fn record_openai(args: &RecordArgs, verbose: bool) -> Result<()> {
    let api_key = args
        .api_key
        .clone()
        .context("Missing API key (use --api-key or set OPENAI_API_KEY)")?;

    let model = args
        .model
        .clone()
        .unwrap_or_else(|| "gpt-4.1-mini".to_string());

    let prompt = args
        .prompt
        .clone()
        .context("Missing prompt (use --prompt)")?;

    // Initialize runtime recorder for macro-based instrumentation
    init_recorder(&args.run_id);

    // Create client
    let client = OpenAIClient::with_base_url(api_key, &args.api_base);

    if verbose {
        println!("  Provider: {}", "openai".yellow());
        println!("  Model: {}", model.cyan());
        println!("  Temperature: {}", args.temperature.unwrap_or(1.0));
        println!("  Prompt: {}", prompt.dimmed());
        println!();
    }

    // Call the tracked function
    let response = ask(
        &prompt,
        &client,
        &model,
        args.temperature,
        args.max_tokens,
        args.logprobs,
    )
    .await
    .map_err(|e| anyhow::anyhow!(e))?;

    if verbose {
        println!("  Response: {}", response.green());
    }

    // Take recorder and finalize
    let mut recorder = take_recorder().context("Recorder not initialized")?;
    recorder.finalize();

    let json = recorder.to_json().context("Failed to serialize recording")?;

    fs::write(&args.output, &json).with_context(|| {
        format!("Failed to write recording to {}", args.output.display())
    })?;

    println!();
    println!(
        "{} Recorded {} events to {}",
        "✓".green().bold(),
        recorder.event_count(),
        args.output.display().to_string().cyan()
    );

    Ok(())
}

#[track(ai = true, name = "ask_gemini", skip_args = true)]
async fn ask_gemini(
    prompt: &str,
    client: &GeminiClient,
    model: &str,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
) -> Result<String, GeminiError> {
    let mut request = GenerateContentRequest::new(model, vec![GeminiContent::user(prompt)]);

    let config = GenerationConfig {
        temperature,
        max_output_tokens: max_tokens,
        top_p: None,
        top_k: None,
    };
    request = request.with_generation_config(config);

    client.generate_text(request).await
}

#[track(ai = true, name = "ask", skip_args = true)]
async fn ask(
    prompt: &str,
    client: &OpenAIClient,
    model: &str,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    logprobs: bool,
) -> Result<String, OpenAIError> {
    let mut request = ChatCompletionRequest::new(model, vec![ChatMessage::user(prompt)]);

    if let Some(t) = temperature {
        request = request.with_temperature(t);
    }

    if let Some(m) = max_tokens {
        request = request.with_max_tokens(m);
    }

    if logprobs {
        request = request.with_logprobs(true);
    }

    client.chat_completion_text(request).await
}

fn record_simulated_ai(args: &RecordArgs, verbose: bool) -> Result<()> {
    let mut recorder = Recorder::new(&args.run_id);
    record_ai_simulation(&mut recorder, verbose)?;
    recorder.finalize();
    write_recorder(&args.output, &recorder)
}

fn record_demo(args: &RecordArgs, verbose: bool) -> Result<()> {
    let mut recorder = Recorder::new(&args.run_id);
    record_demo_events(&mut recorder, args.num_events, verbose)?;
    recorder.finalize();
    write_recorder(&args.output, &recorder)
}

fn write_recorder(output: &PathBuf, recorder: &Recorder) -> Result<()> {
    let json = recorder
        .to_json()
        .context("Failed to serialize recording")?;

    fs::write(output, &json)
        .with_context(|| format!("Failed to write recording to {}", output.display()))?;

    println!();
    println!(
        "{} Recorded {} events to {}",
        "✓".green().bold(),
        recorder.event_count(),
        output.display().to_string().cyan()
    );

    Ok(())
}

/// Record demo events for testing.
fn record_demo_events(recorder: &mut Recorder, num_events: usize, verbose: bool) -> Result<()> {
    // Simulate a typical program execution
    let events = vec![
        ("initialize", vec!["config.json"], None),
        ("load_data", vec!["input.csv"], Some("1000 rows")),
        ("preprocess", vec![], Some("cleaned")),
        ("train_model", vec!["epochs=10"], Some("accuracy=0.95")),
        ("save_results", vec!["output.json"], None),
        ("cleanup", vec![], None),
        ("finalize", vec![], Some("success")),
    ];

    for (i, (name, args, ret)) in events.iter().take(num_events).enumerate() {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let ret = ret.map(|s| s.to_string());

        let event_id = recorder.track_function_call(*name, args.clone(), ret.clone());

        if verbose {
            println!(
                "  {} Event {}: {}({}) -> {:?}",
                "→".dimmed(),
                event_id,
                name.yellow(),
                args.join(", ").dimmed(),
                ret.as_deref().unwrap_or("void")
            );
        }

        // Add some state snapshots periodically
        if i % 2 == 1 {
            let state = format!(r#"{{"step": {}, "status": "in_progress"}}"#, i);
            recorder.track_state_snapshot(format!("checkpoint_{}", i), state);
        }
    }

    Ok(())
}

/// Record AI simulation events.
fn record_ai_simulation(recorder: &mut Recorder, verbose: bool) -> Result<()> {
    // Create an AI tracker embedded in the recording
    let mut ai_tracker = AITracker::new();

    // Track the main function call
    let main_id = recorder.track_function_call("main", vec![], None);
    recorder.enter_scope(main_id);

    // Simulate an AI interaction
    let prompt = "What is the capital of France?";
    let response = "The capital of France is Paris.";

    if verbose {
        println!("  {} Simulating AI call...", "→".dimmed());
        println!("    Prompt: {}", prompt.yellow());
    }

    // Track in recorder
    recorder.track_custom("ai_prompt", prompt);

    // Track in AI tracker
    let interaction_id = ai_tracker.start_interaction(
        vec![
            ghostdiff_core::ai_integration::Message::system("You are a helpful assistant."),
            ghostdiff_core::ai_integration::Message::user(prompt),
        ],
        ghostdiff_core::ai_integration::ModelConfig::new("gpt-4-simulated").with_temperature(0.7),
    );

    // Simulate streaming tokens
    let tokens: Vec<&str> = response.split_whitespace().collect();
    for token in &tokens {
        ai_tracker.log_token(interaction_id, &format!("{} ", token));
    }

    ai_tracker.complete_interaction(interaction_id, response);
    ai_tracker.record_usage(interaction_id, ghostdiff_core::ai_integration::Usage::new(15, 8));

    // Track AI output in recorder
    recorder.track_ai_output(response);

    if verbose {
        println!("    Response: {}", response.green());
        println!("    Tokens: {}", tokens.len());
    }

    // Track embedding
    recorder.track_custom(
        "embedding",
        r#"{"text": "What is the capital of France?", "dims": 1536}"#,
    );

    // Track state after AI call
    recorder.track_state_snapshot(
        "after_ai_call",
        r#"{"ai_calls": 1, "total_tokens": 23}"#,
    );

    recorder.exit_scope();

    // Store AI tracker data as custom event
    let ai_summary = ai_tracker.summary();
    recorder.track_custom("ai_tracker_summary", &ai_summary);

    Ok(())
}
