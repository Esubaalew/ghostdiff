//! # ghostdiff_core
//!
//! A time-travel debugger and AI execution diffing library for Rust.
//!
//! This crate provides four main modules:
//!
//! - [`recorder`] - Records program events like function calls, AI outputs, and async tasks
//! - [`diff`] - Compares two program runs and outputs semantic differences
//! - [`ai_integration`] - Tracks AI decisions including tokens, prompts, and embeddings
//! - [`runtime`] - Thread-local runtime support for automatic instrumentation
//! - [`providers`] - OpenAI-compatible provider integrations
//! - [`ai_diff`] - AI semantic diff intelligence
//!
//! ## Example
//!
//! ```rust
//! use ghostdiff_core::{Recorder, DiffEngine, AITracker};
//! use ghostdiff_core::recorder::Event;
//!
//! // Create a recorder for run A
//! let mut recorder_a = Recorder::new("run_a");
//! recorder_a.track_function_call("main", vec![], None);
//! recorder_a.track_ai_output("Hello, world!");
//!
//! // Create a recorder for run B
//! let mut recorder_b = Recorder::new("run_b");
//! recorder_b.track_function_call("main", vec![], None);
//! recorder_b.track_ai_output("Hello, universe!");
//!
//! // Compare the two runs
//! let result = DiffEngine::compare(&recorder_a, &recorder_b);
//! println!("Found {} differences", result.differences.len());
//! ```
//!
//! ## Automatic Instrumentation
//!
//! For automatic function tracking, use the `#[track]` macro from `ghostdiff_macros`:
//!
//! ```rust,ignore
//! use ghostdiff::track;
//! use ghostdiff_core::runtime::{init_recorder, take_recorder};
//!
//! #[track]
//! fn my_function(x: i32) -> i32 {
//!     x * 2
//! }
//!
//! fn main() {
//!     // Initialize recorder for this thread
//!     init_recorder("my_session");
//!
//!     // Call instrumented functions
//!     let result = my_function(21);
//!
//!     // Get the recording
//!     let recorder = take_recorder().unwrap();
//!     println!("{}", recorder.to_json().unwrap());
//! }
//! ```

pub mod ai_integration;
pub mod diff;
pub mod recorder;
pub mod runtime;
pub mod providers;
pub mod ai_diff;

pub use ai_integration::AITracker;
pub use diff::DiffEngine;
pub use recorder::Recorder;
