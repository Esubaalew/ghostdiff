//! # ghostdiff CLI
//!
//! Command-line interface for the ghostdiff time-travel debugger and AI execution differ.
//!
//! ## Usage Examples
//!
//! ### Record a program run
//! ```bash
//! # Record a demo run to JSON
//! ghostdiff record --output run1.json
//!
//! # Record with a custom run ID
//! ghostdiff record --output run2.json --run-id experiment_v2
//!
//! # Record with OpenAI-compatible provider
//! ghostdiff record --output ai_run.json \
//!   --ai-provider openai \
//!   --model gpt-4.1-mini \
//!   --temperature 0.7 \
//!   --prompt "Explain Rust lifetimes" \
//!   --api-key $OPENAI_API_KEY
//! ```
//!
//! ### Compare two runs
//! ```bash
//! # Compare two recorded runs
//! ghostdiff diff --run1 run1.json --run2 run2.json
//! ```
//!
//! ### Inspect a run
//! ```bash
//! # View run summary
//! ghostdiff inspect run1.json
//! ```
//!
//! ### Demo mode
//! ```bash
//! # Run interactive demo
//! ghostdiff demo
//! ```

mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand};

/// 🔮 ghostdiff - Time-Travel Debugger & AI Execution Differ
///
/// Record program executions, compare runs, and track AI model outputs
/// to debug non-deterministic behavior and understand AI decisions.
#[derive(Parser, Debug)]
#[command(name = "ghostdiff")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Disable colored output
    #[arg(long, global = true)]
    no_color: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Record a program execution to a JSON file
    Record(commands::RecordArgs),

    /// Compare two recorded runs and show differences
    Diff(commands::DiffArgs),

    /// Inspect a recorded run file
    Inspect(commands::InspectArgs),

    /// Run interactive demo
    Demo,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Configure colored output
    if cli.no_color {
        colored::control::set_override(false);
    }

    match cli.command {
        Commands::Record(args) => commands::record::execute(args, cli.verbose).await,
        Commands::Diff(args) => commands::diff::execute(args, cli.verbose),
        Commands::Inspect(args) => commands::inspect::execute(args, cli.verbose),
        Commands::Demo => commands::demo::execute(cli.verbose),
    }
}
