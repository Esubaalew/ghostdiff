//! Demo command implementation.
//!
//! Runs an interactive demonstration of ghostdiff capabilities.

use anyhow::Result;
use colored::Colorize;
use ghostdiff_core::ai_integration::{Message, ModelConfig, Usage};
use ghostdiff_core::{AITracker, DiffEngine, Recorder};

/// Execute the demo command.
pub fn execute(verbose: bool) -> Result<()> {
    println!("{}", "🔮 ghostdiff Demo".cyan().bold());
    println!("{}", "=================".cyan());
    println!();
    println!(
        "{}",
        "This demo shows the core capabilities of ghostdiff:".dimmed()
    );
    println!("  1. Recording program events");
    println!("  2. Comparing two runs (diffing)");
    println!("  3. Tracking AI interactions");
    println!();

    demo_recording(verbose)?;
    println!();
    println!("{}", "─".repeat(60).dimmed());
    println!();

    demo_diffing(verbose)?;
    println!();
    println!("{}", "─".repeat(60).dimmed());
    println!();

    demo_ai_tracking(verbose)?;

    println!();
    println!("{}", "─".repeat(60).dimmed());
    println!();
    println!("{}", "🎉 Demo complete!".green().bold());
    println!();
    println!("{}", "Next steps:".bold());
    println!("  • Record a run:    ghostdiff record --output my_run.json");
    println!("  • Compare runs:    ghostdiff diff --run1 a.json --run2 b.json");
    println!("  • Inspect a run:   ghostdiff inspect my_run.json --verbose");
    println!();

    Ok(())
}

/// Demonstrate the recording functionality.
fn demo_recording(verbose: bool) -> Result<()> {
    println!("{}", "📹 Part 1: Recording Events".bold());
    println!();

    let mut recorder = Recorder::new("demo_run");

    // Simulate a program execution
    println!("  Recording a simulated program execution...");
    println!();

    let events = [
        ("main", vec![], None, "Entry point"),
        (
            "load_config",
            vec!["config.json"],
            Some("Config { ... }"),
            "Load configuration",
        ),
        (
            "connect_db",
            vec!["postgres://..."],
            Some("Connection"),
            "Database connection",
        ),
        (
            "fetch_users",
            vec![],
            Some("[User; 100]"),
            "Fetch user data",
        ),
        (
            "process",
            vec!["batch_size=10"],
            Some("ProcessResult"),
            "Process data",
        ),
    ];

    for (name, args, ret, description) in events {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let ret = ret.map(|s| s.to_string());
        let id = recorder.track_function_call(name, args, ret);

        if verbose {
            println!(
                "    {} Event {}: {} - {}",
                "→".dimmed(),
                id,
                name.yellow(),
                description.dimmed()
            );
        }
    }

    // Add a state snapshot
    recorder.track_state_snapshot(
        "after_processing",
        r#"{"users_processed": 100, "errors": 0}"#,
    );

    println!(
        "  {} Recorded {} events",
        "✓".green().bold(),
        recorder.event_count()
    );

    // Show serialization
    if verbose {
        println!();
        println!("  Recording can be serialized to JSON:");
        let json = recorder.to_json()?;
        let preview = &json[..200.min(json.len())];
        println!("    {}", preview.dimmed());
        println!("    {}", "...".dimmed());
    }

    Ok(())
}

/// Demonstrate the diffing functionality.
fn demo_diffing(verbose: bool) -> Result<()> {
    println!("{}", "🔍 Part 2: Comparing Runs".bold());
    println!();

    // Create two slightly different runs
    let mut run_a = Recorder::new("baseline");
    run_a.track_function_call("init", vec![], None);
    run_a.track_ai_output("The answer is 42");
    run_a.track_function_call("compute", vec!["x=10".to_string()], Some("100".to_string()));
    run_a.track_state_snapshot("final", r#"{"result": 100}"#);

    let mut run_b = Recorder::new("experiment");
    run_b.track_function_call("init", vec![], None);
    run_b.track_ai_output("The answer is 43"); // Different!
    run_b.track_function_call("compute", vec!["x=10".to_string()], Some("110".to_string())); // Different!
    run_b.track_state_snapshot("final", r#"{"result": 110}"#); // Different!

    println!("  Comparing 'baseline' vs 'experiment'...");
    println!();

    let result = DiffEngine::compare(&run_a, &run_b);

    // Print summary
    let similarity = result.stats.similarity * 100.0;
    let sim_color = if similarity >= 80.0 {
        "green"
    } else if similarity >= 50.0 {
        "yellow"
    } else {
        "red"
    };

    println!(
        "  Similarity: {}",
        format!("{:.1}%", similarity).color(sim_color).bold()
    );
    println!(
        "  Differences found: {}",
        result.differences.len().to_string().red()
    );
    println!();

    if verbose {
        for diff in &result.differences {
            println!(
                "    {} [severity {}] {}",
                "→".dimmed(),
                diff.severity,
                diff.summary.yellow()
            );
        }
    } else {
        println!("  {}", "Use --verbose to see difference details".dimmed());
    }

    Ok(())
}

/// Demonstrate AI tracking.
fn demo_ai_tracking(verbose: bool) -> Result<()> {
    println!("{}", "🤖 Part 3: AI Interaction Tracking".bold());
    println!();

    let mut tracker = AITracker::new();

    // Simulate an AI conversation
    println!("  Simulating an AI chat interaction...");
    println!();

    let interaction_id = tracker.start_interaction(
        vec![
            Message::system("You are a helpful Rust programming assistant."),
            Message::user("What's the difference between &str and String?"),
        ],
        ModelConfig::new("gpt-4")
            .with_temperature(0.7)
            .with_max_tokens(200),
    );

    // Simulate streaming response
    let response =
        "`&str` is a string slice (borrowed), while `String` is an owned, heap-allocated string.";
    let tokens: Vec<&str> = response.split_whitespace().collect();

    if verbose {
        println!("    Streaming tokens:");
        print!("    ");
    }

    for token in &tokens {
        tracker.log_token(interaction_id, &format!("{} ", token));
        if verbose {
            print!("{} ", token.green());
        }
    }
    if verbose {
        println!();
        println!();
    }

    tracker.complete_interaction(interaction_id, response);
    tracker.record_usage(interaction_id, Usage::new(42, tokens.len() as u64));

    // Track an embedding
    tracker.track_embedding(
        "What's the difference between &str and String?",
        vec![0.1, 0.2, -0.3, 0.4, 0.5], // Simulated
        "text-embedding-ada-002",
    );

    // Print summary
    println!(
        "  {}",
        tracker.summary().lines().collect::<Vec<_>>().join("\n  ")
    );

    Ok(())
}
