//! # ghostdiff
//!
//! Time-travel debugger and AI execution diffing tool for Rust.
//!
//! This crate provides a unified interface to the ghostdiff ecosystem:
//!
//! - **Automatic instrumentation** via the `#[track]` macro
//! - **Event recording** for function calls, AI outputs, and async tasks
//! - **Semantic diffing** to compare program runs
//! - **AI tracking** for tokens, prompts, and embeddings
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ghostdiff::prelude::*;
//!
//! // Initialize recording for this thread
//! init_recorder("my_session");
//!
//! #[track]
//! fn process_data(input: &str) -> String {
//!     format!("Processed: {}", input)
//! }
//!
//! #[track(ai = true)]
//! async fn run_inference(prompt: &str) -> String {
//!     // Your AI call here
//!     "AI response".to_string()
//! }
//!
//! fn main() {
//!     let result = process_data("hello");
//!     println!("{}", result);
//!
//!     // Save the recording
//!     let recorder = take_recorder().unwrap();
//!     std::fs::write("session.json", recorder.to_json().unwrap()).unwrap();
//! }
//! ```
//!
//! ## The `#[track]` Macro
//!
//! The `#[track]` attribute macro automatically instruments functions:
//!
//! ```rust,ignore
//! use ghostdiff::track;
//!
//! // Basic tracking
//! #[track]
//! fn add(a: i32, b: i32) -> i32 {
//!     a + b
//! }
//!
//! // Custom name and AI flag
//! #[track(name = "model_inference", ai = true)]
//! async fn infer(input: &str) -> String {
//!     todo!()
//! }
//!
//! // With tags
//! #[track(tags = ["critical", "hot-path"])]
//! fn critical_operation() {
//!     // ...
//! }
//!
//! // Skip argument capture (for sensitive data)
//! #[track(skip_args = true)]
//! fn authenticate(password: &str) -> bool {
//!     true
//! }
//! ```
//!
//! ## Comparing Runs
//!
//! ```rust
//! use ghostdiff::{Recorder, DiffEngine};
//!
//! let mut run_a = Recorder::new("baseline");
//! run_a.track_ai_output("The answer is 42");
//!
//! let mut run_b = Recorder::new("experiment");
//! run_b.track_ai_output("The answer is 43");
//!
//! let result = DiffEngine::compare(&run_a, &run_b);
//! println!("{}", result.summary());
//! ```
//!
//! ## Recording Guard (RAII)
//!
//! ```rust,ignore
//! use ghostdiff::runtime::RecordingGuard;
//!
//! fn main() {
//!     // Recording starts, auto-saves to file when guard drops
//!     let _guard = RecordingGuard::new("session", "recording.json");
//!
//!     // All #[track] functions called here will be recorded
//!     my_tracked_function();
//!
//! } // Recording saved automatically
//! ```

// Re-export the track macro
pub use ghostdiff_macros::track;

// Re-export core types
pub use ghostdiff_core::ai_diff;
pub use ghostdiff_core::ai_integration;
pub use ghostdiff_core::diff;
pub use ghostdiff_core::providers;
pub use ghostdiff_core::recorder;
pub use ghostdiff_core::runtime;

pub use ghostdiff_core::AITracker;
pub use ghostdiff_core::DiffEngine;
pub use ghostdiff_core::Recorder;

/// Prelude module for convenient imports.
///
/// ```rust,ignore
/// use ghostdiff::prelude::*;
/// ```
pub mod prelude {
    pub use crate::track;
    pub use crate::AITracker;
    pub use crate::DiffEngine;
    pub use crate::Recorder;

    // Runtime helpers
    pub use ghostdiff_core::runtime::{
        finalize_and_serialize, has_recorder, init_recorder, record_ai_output, record_call,
        record_state, set_recorder, take_recorder, with_recorder, with_recorder_ref,
        RecordingGuard,
    };

    // Common types
    pub use ghostdiff_core::ai_diff::{AiDiffEngine, AiDiffResult, RootCause};
    pub use ghostdiff_core::ai_integration::{Message, ModelConfig, Usage};
    pub use ghostdiff_core::diff::{DiffConfig, DiffResult, Difference, DifferenceKind};
    pub use ghostdiff_core::recorder::{Event, EventKind};
}
