//! CLI command implementations.
//!
//! Each subcommand is implemented in its own module for modularity.
//! This structure makes it easy to extend with new commands and
//! provides clear hooks for future WASM UI or live AI tracking integration.

pub mod demo;
pub mod diff;
pub mod inspect;
pub mod record;

// Re-export argument structs
pub use diff::DiffArgs;
pub use inspect::InspectArgs;
pub use record::RecordArgs;

/// Output format options shared across commands.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output for programmatic consumption
    Json,
    /// Compact single-line output
    Compact,
}
