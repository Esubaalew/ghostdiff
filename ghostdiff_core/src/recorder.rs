//! # Recorder Module
//!
//! Provides time-travel recording capabilities for program execution.
//!
//! The [`Recorder`] struct captures events during program execution, enabling
//! replay and comparison of different program runs.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique identifier for events, represented as a monotonically increasing u64.
pub type EventId = u64;

/// Represents a point in time as milliseconds since UNIX epoch.
pub type Timestamp = u64;

/// The type of event that was recorded.
///
/// This enum categorizes events for filtering and semantic comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventKind {
    /// A function was called with the given name and arguments.
    FunctionCall {
        /// Name of the function being called.
        name: String,
        /// Serialized arguments passed to the function.
        args: Vec<String>,
        /// Optional return value if the function has completed.
        return_value: Option<String>,
    },

    /// AI model produced an output.
    AIOutput {
        /// The output text or tokens from the AI.
        content: String,
    },

    /// An async task was spawned.
    AsyncTaskSpawned {
        /// Identifier for the spawned task.
        task_id: String,
    },

    /// An async task completed.
    AsyncTaskCompleted {
        /// Identifier for the completed task.
        task_id: String,
        /// Whether the task completed successfully.
        success: bool,
    },

    /// A state snapshot was captured.
    StateSnapshot {
        /// Name or identifier for this snapshot.
        label: String,
        /// Serialized state data.
        data: String,
    },

    /// Custom user-defined event.
    Custom {
        /// Type identifier for the custom event.
        kind: String,
        /// Payload data for the custom event.
        payload: String,
    },
}

/// A single recorded event in the program execution timeline.
///
/// Events are immutable once created and contain all information needed
/// to reconstruct or compare program behavior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Unique identifier for this event.
    pub id: EventId,

    /// Timestamp when this event occurred.
    pub timestamp: Timestamp,

    /// The kind of event and its associated data.
    pub kind: EventKind,

    /// Optional parent event ID for hierarchical event tracking.
    pub parent_id: Option<EventId>,

    /// Optional tags for filtering and categorization.
    pub tags: Vec<String>,
}

impl Event {
    /// Creates a new event with the given kind.
    ///
    /// The event is assigned a unique ID and the current timestamp.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this event
    /// * `kind` - The type and data of the event
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::recorder::{Event, EventKind};
    ///
    /// let event = Event::new(1, EventKind::AIOutput {
    ///     content: "Hello, world!".to_string(),
    /// });
    ///
    /// assert_eq!(event.id, 1);
    /// ```
    pub fn new(id: EventId, kind: EventKind) -> Self {
        Self {
            id,
            timestamp: current_timestamp(),
            kind,
            parent_id: None,
            tags: Vec::new(),
        }
    }

    /// Creates a new event with a parent reference.
    ///
    /// Useful for tracking nested function calls or causally related events.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this event
    /// * `kind` - The type and data of the event
    /// * `parent_id` - The ID of the parent event
    pub fn with_parent(id: EventId, kind: EventKind, parent_id: EventId) -> Self {
        Self {
            id,
            timestamp: current_timestamp(),
            kind,
            parent_id: Some(parent_id),
            tags: Vec::new(),
        }
    }

    /// Adds a tag to this event for filtering purposes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::recorder::{Event, EventKind};
    ///
    /// let mut event = Event::new(1, EventKind::Custom {
    ///     kind: "user_action".to_string(),
    ///     payload: "{}".to_string(),
    /// });
    /// event.add_tag("important");
    ///
    /// assert!(event.tags.contains(&"important".to_string()));
    /// ```
    pub fn add_tag(&mut self, tag: &str) {
        self.tags.push(tag.to_string());
    }

    /// Checks if this event has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Records program events for time-travel debugging.
///
/// The `Recorder` captures a sequence of events during program execution,
/// maintaining a timeline that can be replayed, serialized, or compared
/// with other recordings.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::Recorder;
///
/// let mut recorder = Recorder::new("my_run");
///
/// // Track various events
/// recorder.track_function_call("process_data", vec!["input".to_string()], None);
/// recorder.track_ai_output("Generated response");
///
/// // Replay recorded events
/// for event in recorder.events() {
///     println!("{:?}", event);
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recorder {
    /// Unique identifier for this recording session.
    run_id: String,

    /// The sequence of recorded events.
    events: Vec<Event>,

    /// Counter for generating unique event IDs.
    next_id: EventId,

    /// Stack of parent event IDs for nested tracking.
    parent_stack: Vec<EventId>,

    /// Metadata about this recording session.
    metadata: RecorderMetadata,
}

/// Metadata associated with a recording session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecorderMetadata {
    /// When the recording started.
    pub started_at: Option<Timestamp>,

    /// When the recording ended.
    pub ended_at: Option<Timestamp>,

    /// Custom key-value metadata.
    pub custom: std::collections::HashMap<String, String>,
}

impl Recorder {
    /// Creates a new recorder with the given run identifier.
    ///
    /// # Arguments
    ///
    /// * `run_id` - A unique identifier for this recording session
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::Recorder;
    ///
    /// let recorder = Recorder::new("experiment_001");
    /// assert_eq!(recorder.run_id(), "experiment_001");
    /// ```
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            events: Vec::new(),
            next_id: 0,
            parent_stack: Vec::new(),
            metadata: RecorderMetadata {
                started_at: Some(current_timestamp()),
                ..Default::default()
            },
        }
    }

    /// Returns the run identifier for this recorder.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Returns a reference to all recorded events.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Returns the number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns a reference to the recorder's metadata.
    pub fn metadata(&self) -> &RecorderMetadata {
        &self.metadata
    }

    /// Returns a mutable reference to the recorder's metadata.
    pub fn metadata_mut(&mut self) -> &mut RecorderMetadata {
        &mut self.metadata
    }

    /// Tracks a generic event and returns its ID.
    ///
    /// This is the low-level method for recording events. Prefer using
    /// the specialized methods like [`track_function_call`] or [`track_ai_output`]
    /// when applicable.
    ///
    /// [`track_function_call`]: Recorder::track_function_call
    /// [`track_ai_output`]: Recorder::track_ai_output
    pub fn track(&mut self, kind: EventKind) -> EventId {
        let id = self.next_id;
        self.next_id += 1;

        let mut event = if let Some(&parent_id) = self.parent_stack.last() {
            Event::with_parent(id, kind, parent_id)
        } else {
            Event::new(id, kind)
        };

        // Inherit tags from parent if present
        if let Some(parent_id) = event.parent_id {
            if let Some(parent) = self.events.iter().find(|e| e.id == parent_id) {
                for tag in &parent.tags {
                    if !event.has_tag(tag) {
                        event.add_tag(tag);
                    }
                }
            }
        }

        self.events.push(event);
        id
    }

    /// Tracks a function call event.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the function being called
    /// * `args` - Serialized arguments passed to the function
    /// * `return_value` - Optional return value if already known
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::Recorder;
    ///
    /// let mut recorder = Recorder::new("test");
    /// let event_id = recorder.track_function_call(
    ///     "calculate_sum",
    ///     vec!["1".to_string(), "2".to_string()],
    ///     Some("3".to_string()),
    /// );
    /// ```
    pub fn track_function_call(
        &mut self,
        name: impl Into<String>,
        args: Vec<String>,
        return_value: Option<String>,
    ) -> EventId {
        self.track(EventKind::FunctionCall {
            name: name.into(),
            args,
            return_value,
        })
    }

    /// Tracks an AI output event.
    ///
    /// # Arguments
    ///
    /// * `content` - The output produced by the AI model
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::Recorder;
    ///
    /// let mut recorder = Recorder::new("ai_session");
    /// recorder.track_ai_output("The answer is 42.");
    /// ```
    pub fn track_ai_output(&mut self, content: impl Into<String>) -> EventId {
        self.track(EventKind::AIOutput {
            content: content.into(),
        })
    }

    /// Tracks an async task being spawned.
    ///
    /// # Arguments
    ///
    /// * `task_id` - Unique identifier for the spawned task
    pub fn track_async_spawn(&mut self, task_id: impl Into<String>) -> EventId {
        self.track(EventKind::AsyncTaskSpawned {
            task_id: task_id.into(),
        })
    }

    /// Tracks an async task completing.
    ///
    /// # Arguments
    ///
    /// * `task_id` - Identifier for the completed task
    /// * `success` - Whether the task completed successfully
    pub fn track_async_complete(&mut self, task_id: impl Into<String>, success: bool) -> EventId {
        self.track(EventKind::AsyncTaskCompleted {
            task_id: task_id.into(),
            success,
        })
    }

    /// Captures a state snapshot.
    ///
    /// # Arguments
    ///
    /// * `label` - A name for this snapshot
    /// * `data` - Serialized state data
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::Recorder;
    ///
    /// let mut recorder = Recorder::new("state_tracking");
    /// recorder.track_state_snapshot("before_transform", r#"{"count": 0}"#);
    /// // ... perform transformation ...
    /// recorder.track_state_snapshot("after_transform", r#"{"count": 10}"#);
    /// ```
    pub fn track_state_snapshot(
        &mut self,
        label: impl Into<String>,
        data: impl Into<String>,
    ) -> EventId {
        self.track(EventKind::StateSnapshot {
            label: label.into(),
            data: data.into(),
        })
    }

    /// Tracks a custom user-defined event.
    ///
    /// # Arguments
    ///
    /// * `kind` - Type identifier for the custom event
    /// * `payload` - JSON or other serialized payload
    pub fn track_custom(&mut self, kind: impl Into<String>, payload: impl Into<String>) -> EventId {
        self.track(EventKind::Custom {
            kind: kind.into(),
            payload: payload.into(),
        })
    }

    /// Enters a new scope, making subsequent events children of the current event.
    ///
    /// Use this with [`exit_scope`] to create hierarchical event trees.
    ///
    /// [`exit_scope`]: Recorder::exit_scope
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::Recorder;
    ///
    /// let mut recorder = Recorder::new("nested");
    /// let parent_id = recorder.track_function_call("outer", vec![], None);
    /// recorder.enter_scope(parent_id);
    ///
    /// // This event will have parent_id as its parent
    /// recorder.track_function_call("inner", vec![], None);
    ///
    /// recorder.exit_scope();
    /// ```
    pub fn enter_scope(&mut self, parent_id: EventId) {
        self.parent_stack.push(parent_id);
    }

    /// Exits the current scope, returning to the previous parent.
    pub fn exit_scope(&mut self) {
        self.parent_stack.pop();
    }

    /// Marks the recording as complete and sets the end timestamp.
    pub fn finalize(&mut self) {
        self.metadata.ended_at = Some(current_timestamp());
    }

    /// Serializes the recording to JSON.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::Recorder;
    ///
    /// let mut recorder = Recorder::new("serialization_test");
    /// recorder.track_ai_output("Test output");
    ///
    /// let json = recorder.to_json().expect("serialization failed");
    /// assert!(json.contains("serialization_test"));
    /// ```
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a recording from JSON.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::Recorder;
    ///
    /// let mut original = Recorder::new("roundtrip");
    /// original.track_ai_output("Hello");
    ///
    /// let json = original.to_json().unwrap();
    /// let restored = Recorder::from_json(&json).unwrap();
    ///
    /// assert_eq!(original.run_id(), restored.run_id());
    /// assert_eq!(original.event_count(), restored.event_count());
    /// ```
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Filters events by a predicate, returning a new recorder with only matching events.
    ///
    /// Note: Event IDs and parent relationships are preserved from the original.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::Recorder;
    /// use ghostdiff_core::recorder::EventKind;
    ///
    /// let mut recorder = Recorder::new("filter_test");
    /// recorder.track_function_call("func1", vec![], None);
    /// recorder.track_ai_output("output1");
    /// recorder.track_function_call("func2", vec![], None);
    ///
    /// let ai_only = recorder.filter(|e| matches!(e.kind, EventKind::AIOutput { .. }));
    /// assert_eq!(ai_only.event_count(), 1);
    /// ```
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Event) -> bool,
    {
        let filtered_events: Vec<Event> = self
            .events
            .iter()
            .filter(|e| predicate(e))
            .cloned()
            .collect();

        let max_id = filtered_events.iter().map(|e| e.id).max().unwrap_or(0);

        Self {
            run_id: format!("{}_filtered", self.run_id),
            events: filtered_events,
            next_id: max_id + 1,
            parent_stack: Vec::new(),
            metadata: self.metadata.clone(),
        }
    }

    /// Replays all events, calling the provided callback for each.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::Recorder;
    ///
    /// let mut recorder = Recorder::new("replay_test");
    /// recorder.track_ai_output("Event 1");
    /// recorder.track_ai_output("Event 2");
    ///
    /// let mut count = 0;
    /// recorder.replay(|event| {
    ///     count += 1;
    ///     println!("Replaying event {}: {:?}", event.id, event.kind);
    /// });
    /// assert_eq!(count, 2);
    /// ```
    pub fn replay<F>(&self, mut callback: F)
    where
        F: FnMut(&Event),
    {
        for event in &self.events {
            callback(event);
        }
    }
}

/// Returns the current timestamp as milliseconds since UNIX epoch.
fn current_timestamp() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as Timestamp)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recorder_basic() {
        let mut recorder = Recorder::new("test_run");
        assert_eq!(recorder.run_id(), "test_run");
        assert_eq!(recorder.event_count(), 0);

        recorder.track_ai_output("Hello");
        assert_eq!(recorder.event_count(), 1);
    }

    #[test]
    fn test_nested_events() {
        let mut recorder = Recorder::new("nested_test");

        let parent_id = recorder.track_function_call("parent", vec![], None);
        recorder.enter_scope(parent_id);

        let child_id = recorder.track_function_call("child", vec![], None);
        recorder.exit_scope();

        let child_event = recorder.events().iter().find(|e| e.id == child_id).unwrap();
        assert_eq!(child_event.parent_id, Some(parent_id));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut recorder = Recorder::new("serial_test");
        recorder.track_ai_output("Test content");
        recorder.track_function_call("test_fn", vec!["arg1".to_string()], None);

        let json = recorder.to_json().unwrap();
        let restored = Recorder::from_json(&json).unwrap();

        assert_eq!(recorder.run_id(), restored.run_id());
        assert_eq!(recorder.event_count(), restored.event_count());
    }
}
