//! Inspect command implementation.
//!
//! Displays information about a recorded run.
//!
//! ## Future Integration Points
//!
//! - `--replay`: Interactive time-travel replay
//! - `--timeline`: Visual timeline of events
//! - WASM: Export for browser visualization

use crate::commands::OutputFormat;
use anyhow::{Context, Result};
use clap::Args;
use colored::Colorize;
use ghostdiff_core::recorder::EventKind;
use ghostdiff_core::Recorder;
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Run file to inspect
    pub file: PathBuf,

    /// Output format
    #[arg(long, short, value_enum, default_value = "text")]
    pub format: OutputFormat,

    /// Show all event details
    #[arg(long, short)]
    pub verbose: bool,

    /// Filter events by type (function, ai, async, state, custom)
    #[arg(long)]
    pub filter: Option<String>,

    /// Limit number of events to show
    #[arg(long)]
    pub limit: Option<usize>,

    // === Future flags ===
    // TODO: #[arg(long)]
    // pub replay: bool,  // Interactive replay mode
    //
    // TODO: #[arg(long)]
    // pub timeline: bool,  // ASCII timeline visualization
}

/// Execute the inspect command.
pub fn execute(args: InspectArgs, verbose: bool) -> Result<()> {
    // Load run file
    let json = fs::read_to_string(&args.file)
        .with_context(|| format!("Failed to read {}", args.file.display()))?;

    let recorder = Recorder::from_json(&json)
        .with_context(|| format!("Failed to parse {}", args.file.display()))?;

    match args.format {
        OutputFormat::Text => print_text_inspection(&recorder, &args, verbose)?,
        OutputFormat::Json => {
            // Just echo the JSON (possibly pretty-printed)
            let parsed: serde_json::Value = serde_json::from_str(&json)?;
            println!("{}", serde_json::to_string_pretty(&parsed)?);
        }
        OutputFormat::Compact => print_compact_inspection(&recorder)?,
    }

    Ok(())
}

/// Print human-readable inspection.
fn print_text_inspection(recorder: &Recorder, args: &InspectArgs, verbose: bool) -> Result<()> {
    // Header
    println!(
        "{} {}",
        "🔮 Run:".bold(),
        recorder.run_id().cyan().bold()
    );
    println!();

    // Metadata
    println!("{}", "Metadata".bold().underline());
    let meta = recorder.metadata();
    if let Some(started) = meta.started_at {
        println!("  Started at:  {}", format_timestamp(started));
    }
    if let Some(ended) = meta.ended_at {
        println!("  Ended at:    {}", format_timestamp(ended));
        if let Some(started) = meta.started_at {
            let duration_ms = ended.saturating_sub(started);
            println!("  Duration:    {}ms", duration_ms);
        }
    }
    if !meta.custom.is_empty() {
        println!("  Custom metadata:");
        for (key, value) in &meta.custom {
            println!("    {}: {}", key.dimmed(), value);
        }
    }
    println!();

    // Statistics
    println!("{}", "Statistics".bold().underline());
    let events = recorder.events();
    println!("  Total events: {}", events.len().to_string().cyan());

    // Count by type
    let mut function_calls = 0;
    let mut ai_outputs = 0;
    let mut async_events = 0;
    let mut state_snapshots = 0;
    let mut custom_events = 0;

    for event in events {
        match &event.kind {
            EventKind::FunctionCall { .. } => function_calls += 1,
            EventKind::AIOutput { .. } => ai_outputs += 1,
            EventKind::AsyncTaskSpawned { .. } | EventKind::AsyncTaskCompleted { .. } => {
                async_events += 1
            }
            EventKind::StateSnapshot { .. } => state_snapshots += 1,
            EventKind::Custom { .. } => custom_events += 1,
        }
    }

    println!("  Function calls:   {}", function_calls);
    println!("  AI outputs:       {}", ai_outputs);
    println!("  Async events:     {}", async_events);
    println!("  State snapshots:  {}", state_snapshots);
    println!("  Custom events:    {}", custom_events);
    println!();

    // Events (filtered and limited)
    if args.verbose || verbose {
        println!("{}", "Events".bold().underline());
        println!();

        let filter = args.filter.as_deref();
        let limit = args.limit.unwrap_or(usize::MAX);

        let mut shown = 0;
        for event in events {
            if shown >= limit {
                println!("  {} (showing {} of {})", "...".dimmed(), limit, events.len());
                break;
            }

            // Apply filter
            let type_str = event_type_str(&event.kind);
            if let Some(f) = filter {
                if !type_str.to_lowercase().contains(&f.to_lowercase()) {
                    continue;
                }
            }

            print_event(event, args.verbose)?;
            shown += 1;
        }
    } else {
        println!(
            "{}",
            "Use --verbose to see event details".dimmed()
        );
    }

    Ok(())
}

/// Print a single event.
fn print_event(event: &ghostdiff_core::recorder::Event, verbose: bool) -> Result<()> {
    let id_str = format!("[{:>3}]", event.id).dimmed();
    let type_badge = format_event_type(&event.kind);

    match &event.kind {
        EventKind::FunctionCall {
            name,
            args,
            return_value,
        } => {
            let args_str = if args.is_empty() {
                String::new()
            } else {
                format!("({})", args.join(", "))
            };
            let ret_str = return_value
                .as_ref()
                .map(|r| format!(" -> {}", r.green()))
                .unwrap_or_default();
            println!(
                "  {} {} {}{}{}",
                id_str,
                type_badge,
                name.yellow(),
                args_str.dimmed(),
                ret_str
            );
        }
        EventKind::AIOutput { content } => {
            let preview = truncate(content, 60);
            println!(
                "  {} {} \"{}\"",
                id_str,
                type_badge,
                preview.green()
            );
        }
        EventKind::AsyncTaskSpawned { task_id } => {
            println!(
                "  {} {} spawned: {}",
                id_str,
                type_badge,
                task_id.cyan()
            );
        }
        EventKind::AsyncTaskCompleted { task_id, success } => {
            let status = if *success {
                "✓".green()
            } else {
                "✗".red()
            };
            println!(
                "  {} {} completed: {} {}",
                id_str,
                type_badge,
                task_id.cyan(),
                status
            );
        }
        EventKind::StateSnapshot { label, data } => {
            let preview = truncate(data, 40);
            println!(
                "  {} {} {}: {}",
                id_str,
                type_badge,
                label.magenta(),
                preview.dimmed()
            );
        }
        EventKind::Custom { kind, payload } => {
            let preview = truncate(payload, 40);
            println!(
                "  {} {} [{}] {}",
                id_str,
                type_badge,
                kind.blue(),
                preview.dimmed()
            );
        }
    }

    if verbose {
        if let Some(parent_id) = event.parent_id {
            println!("      parent: {}", parent_id);
        }
        if !event.tags.is_empty() {
            println!("      tags: {}", event.tags.join(", "));
        }
    }

    Ok(())
}

/// Print compact single-line summary.
fn print_compact_inspection(recorder: &Recorder) -> Result<()> {
    let meta = recorder.metadata();
    let duration = match (meta.started_at, meta.ended_at) {
        (Some(start), Some(end)) => format!("{}ms", end.saturating_sub(start)),
        _ => "unknown".to_string(),
    };

    println!(
        "{} events={} duration={}",
        recorder.run_id(),
        recorder.event_count(),
        duration
    );

    Ok(())
}

/// Get event type as string.
fn event_type_str(kind: &EventKind) -> &'static str {
    match kind {
        EventKind::FunctionCall { .. } => "function",
        EventKind::AIOutput { .. } => "ai",
        EventKind::AsyncTaskSpawned { .. } | EventKind::AsyncTaskCompleted { .. } => "async",
        EventKind::StateSnapshot { .. } => "state",
        EventKind::Custom { .. } => "custom",
    }
}

/// Format event type as colored badge.
fn format_event_type(kind: &EventKind) -> String {
    match kind {
        EventKind::FunctionCall { .. } => "FN".cyan().bold().to_string(),
        EventKind::AIOutput { .. } => "AI".green().bold().to_string(),
        EventKind::AsyncTaskSpawned { .. } | EventKind::AsyncTaskCompleted { .. } => {
            "ASYNC".magenta().bold().to_string()
        }
        EventKind::StateSnapshot { .. } => "STATE".blue().bold().to_string(),
        EventKind::Custom { .. } => "CUSTOM".yellow().bold().to_string(),
    }
}

/// Format timestamp as human-readable string.
fn format_timestamp(ts: u64) -> String {
    // Simple formatting - in production, use chrono
    format!("{}", ts)
}

/// Truncate string with ellipsis.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// === Future Integration Points ===
//
// pub fn interactive_replay(recorder: &Recorder) -> Result<()> {
//     // Step through events interactively
//     // Show state at each point
//     // Allow jumping to specific events
//     todo!("Interactive replay")
// }
//
// pub fn ascii_timeline(recorder: &Recorder) -> Result<()> {
//     // Render ASCII timeline visualization
//     // Show event density
//     // Highlight AI outputs
//     todo!("ASCII timeline")
// }
//
// pub fn export_for_wasm(recorder: &Recorder) -> Vec<u8> {
//     // Prepare data for browser visualization
//     // Include rendering hints
//     todo!("WASM export")
// }
