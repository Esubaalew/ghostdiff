//! # ghostdiff UI
//!
//! WASM-compatible UI for the ghostdiff time-travel debugger.
//!
//! This crate provides a web interface for visualizing program recordings,
//! diffs, and AI interactions. It is designed to be compiled to WebAssembly
//! and run in the browser.
//!
//! ## Status
//!
//! This crate is currently a placeholder. Future versions will include:
//!
//! - Timeline visualization of recorded events
//! - Side-by-side diff view for comparing runs
//! - AI interaction explorer with token-level detail
//! - State snapshot inspector
//!
//! ## Example (Future API)
//!
//! ```ignore
//! use ghostdiff_ui::App;
//!
//! // Initialize the UI with a recording
//! let app = App::new();
//! app.load_recording(recording_json);
//! app.render();
//! ```

// Re-export core types for convenience
pub use ghostdiff_core::{AITracker, DiffEngine, Recorder};

/// Placeholder for the future UI application.
pub struct App {
    _private: (),
}

impl App {
    /// Creates a new UI application instance.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
