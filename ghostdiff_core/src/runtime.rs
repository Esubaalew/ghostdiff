//! # Runtime Module
//!
//! Provides runtime support for automatic function instrumentation.
//!
//! This module manages thread-local storage for the active [`Recorder`],
//! enabling the `#[track]` macro to record events without explicitly
//! passing a recorder to each function.
//!
//! ## Usage
//!
//! ```rust
//! use ghostdiff_core::runtime::{init_recorder, with_recorder, take_recorder};
//! use ghostdiff_core::Recorder;
//!
//! // Initialize a recorder for the current thread
//! init_recorder("my_run");
//!
//! // Use the recorder via callback
//! with_recorder(|recorder| {
//!     recorder.track_function_call("example", vec![], None);
//! });
//!
//! // Get the recorder when done
//! let recorder = take_recorder().expect("recorder was initialized");
//! println!("Recorded {} events", recorder.event_count());
//! ```
//!
//! ## Thread Safety
//!
//! Each thread has its own recorder instance. For multi-threaded applications,
//! you may need to merge recorders or use a shared recorder with synchronization.

use crate::Recorder;
use std::cell::RefCell;

thread_local! {
    /// Thread-local storage for the active recorder.
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

/// Initializes a new recorder for the current thread.
///
/// If a recorder already exists, it will be replaced and the old recorder
/// will be lost. Use [`take_recorder`] first if you need to preserve it.
///
/// # Arguments
///
/// * `run_id` - Unique identifier for this recording session
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::init_recorder;
///
/// init_recorder("session_001");
/// ```
pub fn init_recorder(run_id: impl Into<String>) {
    RECORDER.with(|r| {
        *r.borrow_mut() = Some(Recorder::new(run_id));
    });
}

/// Initializes the thread-local storage with an existing recorder.
///
/// This is useful when you want to continue recording to a previously
/// created or deserialized recorder.
///
/// # Arguments
///
/// * `recorder` - The recorder to install
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::{Recorder, runtime::set_recorder};
///
/// let mut recorder = Recorder::new("custom");
/// recorder.track_function_call("pre_init", vec![], None);
/// set_recorder(recorder);
/// ```
pub fn set_recorder(recorder: Recorder) {
    RECORDER.with(|r| {
        *r.borrow_mut() = Some(recorder);
    });
}

/// Takes ownership of the thread-local recorder, leaving `None` in its place.
///
/// Returns `Some(Recorder)` if a recorder was active, `None` otherwise.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::{init_recorder, take_recorder};
///
/// init_recorder("test");
///
/// let recorder = take_recorder().expect("recorder was initialized");
/// let json = recorder.to_json().unwrap();
/// println!("{}", json);
/// ```
pub fn take_recorder() -> Option<Recorder> {
    RECORDER.with(|r| r.borrow_mut().take())
}

/// Checks if a recorder is currently active for this thread.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::{init_recorder, has_recorder};
///
/// assert!(!has_recorder());
/// init_recorder("test");
/// assert!(has_recorder());
/// ```
pub fn has_recorder() -> bool {
    RECORDER.with(|r| r.borrow().is_some())
}

/// Executes a closure with a mutable reference to the thread-local recorder.
///
/// If no recorder is active, this function does nothing and returns the
/// default value for type `R`.
///
/// This is the primary way the `#[track]` macro interacts with the recorder.
///
/// # Arguments
///
/// * `f` - Closure that receives a mutable reference to the recorder
///
/// # Returns
///
/// The return value of the closure, or `Default::default()` if no recorder.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::{init_recorder, with_recorder};
///
/// init_recorder("test");
///
/// let event_id = with_recorder(|recorder| {
///     recorder.track_function_call("my_func", vec!["arg1".to_string()], None)
/// });
///
/// println!("Created event {}", event_id);
/// ```
pub fn with_recorder<F, R>(f: F) -> R
where
    F: FnOnce(&mut Recorder) -> R,
    R: Default,
{
    RECORDER.with(|r| {
        if let Some(ref mut recorder) = *r.borrow_mut() {
            f(recorder)
        } else {
            R::default()
        }
    })
}

/// Executes a closure with a reference to the thread-local recorder (immutable).
///
/// If no recorder is active, returns `None`.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::{init_recorder, with_recorder_ref};
///
/// init_recorder("test");
///
/// let count = with_recorder_ref(|recorder| {
///     recorder.event_count()
/// });
///
/// println!("Event count: {:?}", count);
/// ```
pub fn with_recorder_ref<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&Recorder) -> R,
{
    RECORDER.with(|r| r.borrow().as_ref().map(f))
}

/// Records a function call using the thread-local recorder.
///
/// This is a convenience function for the most common recording operation.
/// If no recorder is active, the call is silently ignored.
///
/// # Arguments
///
/// * `name` - Function name
/// * `args` - Serialized arguments
/// * `return_value` - Optional return value
///
/// # Returns
///
/// The event ID if recorded, 0 otherwise.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::{init_recorder, record_call};
///
/// init_recorder("test");
/// let id = record_call("my_function", vec!["x=10".to_string()], Some("42".to_string()));
/// ```
pub fn record_call(
    name: impl Into<String>,
    args: Vec<String>,
    return_value: Option<String>,
) -> u64 {
    with_recorder(|recorder| recorder.track_function_call(name, args, return_value))
}

/// Records an AI output using the thread-local recorder.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::{init_recorder, record_ai_output};
///
/// init_recorder("ai_session");
/// record_ai_output("The model generated this response");
/// ```
pub fn record_ai_output(content: impl Into<String>) -> u64 {
    with_recorder(|recorder| recorder.track_ai_output(content))
}

/// Records a state snapshot using the thread-local recorder.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::{init_recorder, record_state};
///
/// init_recorder("test");
/// record_state("checkpoint", r#"{"step": 1, "value": 42}"#);
/// ```
pub fn record_state(label: impl Into<String>, data: impl Into<String>) -> u64 {
    with_recorder(|recorder| recorder.track_state_snapshot(label, data))
}

/// Finalizes the thread-local recorder and returns the JSON representation.
///
/// This is a convenience function for the common pattern of finishing
/// recording and saving to a file.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::{init_recorder, finalize_and_serialize};
///
/// init_recorder("test");
/// // ... recording happens ...
///
/// if let Some(json) = finalize_and_serialize() {
///     std::fs::write("recording.json", json).unwrap();
/// }
/// ```
pub fn finalize_and_serialize() -> Option<String> {
    RECORDER.with(|r| {
        if let Some(ref mut recorder) = *r.borrow_mut() {
            recorder.finalize();
            recorder.to_json().ok()
        } else {
            None
        }
    })
}

/// RAII guard that initializes a recorder and saves it when dropped.
///
/// This provides a convenient way to scope recording to a block of code.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::runtime::RecordingGuard;
///
/// {
///     let _guard = RecordingGuard::new("my_session", "output.json");
///     
///     // All #[track] calls in this scope will be recorded
///     // When _guard drops, recording is saved to output.json
/// }
/// ```
pub struct RecordingGuard {
    output_path: Option<String>,
}

impl RecordingGuard {
    /// Creates a new recording guard.
    ///
    /// # Arguments
    ///
    /// * `run_id` - Identifier for this recording session
    /// * `output_path` - Path to save the JSON when dropped
    pub fn new(run_id: impl Into<String>, output_path: impl Into<String>) -> Self {
        init_recorder(run_id);
        Self {
            output_path: Some(output_path.into()),
        }
    }

    /// Creates a guard that doesn't auto-save on drop.
    ///
    /// Use [`take_recorder`] to manually retrieve the recorder.
    pub fn new_without_save(run_id: impl Into<String>) -> Self {
        init_recorder(run_id);
        Self { output_path: None }
    }

    /// Manually save the recording and disable auto-save on drop.
    pub fn save(&mut self) -> std::io::Result<()> {
        if let Some(ref path) = self.output_path {
            if let Some(json) = finalize_and_serialize() {
                std::fs::write(path, json)?;
            }
            self.output_path = None; // Don't save again on drop
        }
        Ok(())
    }
}

impl Drop for RecordingGuard {
    fn drop(&mut self) {
        if let Some(ref path) = self.output_path {
            if let Some(json) = finalize_and_serialize() {
                let _ = std::fs::write(path, json);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_and_take() {
        init_recorder("test_run");
        assert!(has_recorder());

        let recorder = take_recorder().unwrap();
        assert_eq!(recorder.run_id(), "test_run");
        assert!(!has_recorder());
    }

    #[test]
    fn test_with_recorder() {
        init_recorder("with_test");

        let event_id = with_recorder(|r| r.track_function_call("test", vec![], None));

        assert_eq!(event_id, 0);

        let count = with_recorder_ref(|r| r.event_count());
        assert_eq!(count, Some(1));

        take_recorder(); // cleanup
    }

    #[test]
    fn test_no_recorder() {
        // Ensure no recorder from previous tests
        take_recorder();

        // Should return default (0) without panicking
        let result = with_recorder(|r| r.track_function_call("test", vec![], None));
        assert_eq!(result, 0);
    }

    #[test]
    fn test_convenience_functions() {
        init_recorder("convenience_test");

        record_call("func1", vec![], None);
        record_ai_output("AI response");
        record_state("checkpoint", "{}");

        let count = with_recorder_ref(|r| r.event_count());
        assert_eq!(count, Some(3));

        take_recorder(); // cleanup
    }
}
