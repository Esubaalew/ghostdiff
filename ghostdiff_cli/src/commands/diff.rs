//! Diff command implementation.
//!
//! Compares two recorded runs and outputs semantic differences.
//!
//! ## Future Integration Points
//!
//! - `--watch`: Live diff as new events are recorded
//! - `--interactive`: Step through differences interactively
//! - WASM: Stream diff results to browser for visualization
//! - AI analysis: Use LLM to explain differences

use crate::commands::OutputFormat;
use anyhow::{Context, Result};
use clap::Args;
use colored::Colorize;
use ghostdiff_core::diff::DifferenceKind;
use ghostdiff_core::{DiffEngine, Recorder};
use ghostdiff_core::ai_diff::AiDiffEngine;
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct DiffArgs {
    /// First run file (baseline)
    #[arg(long)]
    pub run1: PathBuf,

    /// Second run file (comparison)
    #[arg(long)]
    pub run2: PathBuf,

    /// Output format
    #[arg(long, short, value_enum, default_value = "text")]
    pub format: OutputFormat,

    /// Minimum severity level to show (0-10)
    #[arg(long, default_value = "0")]
    pub min_severity: u8,

    /// Show detailed event information
    #[arg(long)]
    pub detailed: bool,

    /// Ignore timing differences
    #[arg(long, default_value = "true")]
    pub ignore_timing: bool,

    /// Enable AI semantic diff
    #[arg(long)]
    pub ai: bool,

    /// Explain AI divergence (requires --ai)
    #[arg(long)]
    pub explain: bool,

    // === Future flags ===
    // TODO: #[arg(long)]
    // pub watch: bool,  // Live diff mode
    //
    // TODO: #[arg(long)]
    // pub explain: bool,  // Use AI to explain differences
}

/// Execute the diff command.
pub fn execute(args: DiffArgs, verbose: bool) -> Result<()> {
    // Load run files
    let run1_json = fs::read_to_string(&args.run1)
        .with_context(|| format!("Failed to read {}", args.run1.display()))?;
    let run2_json = fs::read_to_string(&args.run2)
        .with_context(|| format!("Failed to read {}", args.run2.display()))?;

    let run1 = Recorder::from_json(&run1_json)
        .with_context(|| format!("Failed to parse {}", args.run1.display()))?;
    let run2 = Recorder::from_json(&run2_json)
        .with_context(|| format!("Failed to parse {}", args.run2.display()))?;

    if verbose {
        println!("{}", "🔍 Comparing runs...".cyan().bold());
        println!("  Run 1: {} ({} events)", run1.run_id(), run1.event_count());
        println!("  Run 2: {} ({} events)", run2.run_id(), run2.event_count());
        println!();
    }

    // Compute diff
    let result = DiffEngine::compare(&run1, &run2);

    // Compute AI semantic diff (optional)
    let ai_result = if args.ai {
        Some(AiDiffEngine::new().compare(&run1, &run2))
    } else {
        None
    };

    // Filter by severity
    let filtered_diffs: Vec<_> = result
        .differences
        .iter()
        .filter(|d| d.severity >= args.min_severity)
        .collect();

    // Output based on format
    match args.format {
        OutputFormat::Text => {
            print_text_diff(&result, &filtered_diffs, args.detailed)?;
            if let Some(ai) = &ai_result {
                print_ai_diff(ai, args.explain)?;
            }
        }
        OutputFormat::Json => {
            if let Some(ai) = &ai_result {
                let json = serde_json::json!({"diff": result, "ai": ai});
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                let json = result.to_json().context("Failed to serialize diff")?;
                println!("{}", json);
            }
        }
        OutputFormat::Compact => {
            print_compact_diff(&result, &filtered_diffs)?;
        }
    }

    Ok(())
}

/// Print human-readable text diff.
fn print_text_diff(
    result: &ghostdiff_core::diff::DiffResult,
    diffs: &[&ghostdiff_core::diff::Difference],
    detailed: bool,
) -> Result<()> {
    // Header
    println!(
        "{} {} {} {}",
        "Comparing".bold(),
        result.run_a_id.cyan(),
        "vs".dimmed(),
        result.run_b_id.cyan()
    );
    println!();

    // Statistics
    let similarity_pct = result.stats.similarity * 100.0;
    let similarity_color = if similarity_pct >= 90.0 {
        "green"
    } else if similarity_pct >= 50.0 {
        "yellow"
    } else {
        "red"
    };

    println!("{}", "Statistics".bold().underline());
    println!(
        "  Events:     {} vs {}",
        result.stats.events_a.to_string().cyan(),
        result.stats.events_b.to_string().cyan()
    );
    println!(
        "  Matching:   {}",
        result.stats.matching.to_string().green()
    );
    println!(
        "  Similarity: {}",
        format!("{:.1}%", similarity_pct).color(similarity_color)
    );
    println!(
        "  Differences: {}",
        if diffs.is_empty() {
            "0".green().to_string()
        } else {
            diffs.len().to_string().red().bold().to_string()
        }
    );
    println!();

    if diffs.is_empty() {
        println!("{}", "✓ Runs are identical!".green().bold());
        return Ok(());
    }

    // Differences
    println!("{}", "Differences".bold().underline());
    println!();

    for diff in diffs {
        let severity_badge = format_severity(diff.severity);

        match &diff.kind {
            DifferenceKind::Missing { event, missing_from } => {
                let side = match missing_from {
                    ghostdiff_core::diff::RunSide::RunA => "Run A",
                    ghostdiff_core::diff::RunSide::RunB => "Run B",
                };
                println!(
                    "  {} {} Missing from {}",
                    severity_badge,
                    "MISSING".red().bold(),
                    side.yellow()
                );
                if detailed {
                    println!("    Event: {:?}", event.kind);
                }
            }
            DifferenceKind::ContentMismatch {
                event_a,
                event_b,
                description,
            } => {
                println!(
                    "  {} {} {}",
                    severity_badge,
                    "MISMATCH".yellow().bold(),
                    description.dimmed()
                );
                if detailed {
                    println!("    A: {:?}", event_a.kind);
                    println!("    B: {:?}", event_b.kind);
                }
            }
            DifferenceKind::OrderMismatch {
                expected_position,
                actual_position,
                event,
            } => {
                println!(
                    "  {} {} Event at position {} expected at {}",
                    severity_badge,
                    "ORDER".magenta().bold(),
                    actual_position,
                    expected_position
                );
                if detailed {
                    println!("    Event: {:?}", event.kind);
                }
            }
            DifferenceKind::CountMismatch { count_a, count_b } => {
                println!(
                    "  {} {} Event count differs: {} vs {}",
                    severity_badge,
                    "COUNT".blue().bold(),
                    count_a,
                    count_b
                );
            }
            DifferenceKind::TimingDifference {
                delta_ms, ..
            } => {
                println!(
                    "  {} {} Timing differs by {}ms",
                    severity_badge,
                    "TIMING".dimmed(),
                    delta_ms
                );
            }
        }
        println!();
    }

    Ok(())
}

/// Print compact single-line diff summary.
fn print_compact_diff(
    result: &ghostdiff_core::diff::DiffResult,
    diffs: &[&ghostdiff_core::diff::Difference],
) -> Result<()> {
    let status = if diffs.is_empty() { "MATCH" } else { "DIFF" };
    let similarity = format!("{:.1}%", result.stats.similarity * 100.0);

    println!(
        "{} {} vs {} | similarity={} differences={}",
        status,
        result.run_a_id,
        result.run_b_id,
        similarity,
        diffs.len()
    );

    Ok(())
}

/// Format severity as a colored badge.
fn format_severity(severity: u8) -> String {
    let badge = format!("[{}]", severity);
    match severity {
        0..=2 => badge.dimmed().to_string(),
        3..=5 => badge.yellow().to_string(),
        6..=8 => badge.red().to_string(),
        _ => badge.red().bold().to_string(),
    }
}

// === Future Integration Points ===
//
// pub fn watch_diff(run1: &Path, run2: &Path) -> Result<()> {
//     // Watch files for changes
//     // Re-compute diff on change
//     // Stream updates to terminal or WASM UI
//     todo!("Live diff watching")
// }
//
// pub fn explain_diff_with_ai(diff: &DiffResult) -> Result<String> {
//     // Use LLM to explain why runs diverged
//     // Suggest potential causes
//     // Highlight suspicious patterns
//     todo!("AI-powered diff explanation")
// }
//
// pub fn stream_to_wasm(diff: &DiffResult) -> Vec<u8> {
//     // Serialize diff for WASM UI
//     // Include visualization hints
//     // Support incremental updates
//     todo!("WASM streaming")
// }


fn print_ai_diff(ai: &ghostdiff_core::ai_diff::AiDiffResult, explain: bool) -> Result<()> {
    println!("{}", "AI Semantic Diff".bold().underline());
    println!("  Root cause: {:?}", ai.root_cause);
    if let Some(div) = &ai.token_divergence {
        println!(
            "  Token divergence at {}: '{}' vs '{}'",
            div.first_divergence_index, div.token_a, div.token_b
        );
    }
    if let Some(drift) = &ai.logprob_drift {
        println!("  Logprob drift: {:.3}", drift.drift);
    }
    if let Some(ent) = &ai.entropy_comparison {
        println!("  Entropy delta: {:.3}", ent.delta);
    }
    if let Some(emb) = &ai.embedding_similarity {
        println!("  Embedding similarity: {:.3}", emb.cosine_similarity);
    }
    if explain {
        println!();
        println!("{}", "Explanation".bold().underline());
        println!("{}", ai.explanation);
    }
    println!();
    Ok(())
}
