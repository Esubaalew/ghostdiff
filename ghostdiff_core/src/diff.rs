//! # Diff Module
//!
//! Provides semantic comparison between two program execution recordings.
//!
//! The [`DiffEngine`] compares recordings and produces a list of [`Difference`]
//! objects that describe how two runs diverged.

use crate::recorder::{Event, EventKind, Recorder};
use serde::{Deserialize, Serialize};

/// Describes the type of difference between two events or runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DifferenceKind {
    /// An event exists in run A but not in run B.
    Missing {
        /// The event that was missing.
        event: Event,
        /// Which run the event was missing from.
        missing_from: RunSide,
    },

    /// An event exists in both runs but has different content.
    ContentMismatch {
        /// The event from run A.
        event_a: Event,
        /// The event from run B.
        event_b: Event,
        /// Description of what differed.
        description: String,
    },

    /// Events occurred in a different order.
    OrderMismatch {
        /// Expected position (from run A).
        expected_position: usize,
        /// Actual position (from run B).
        actual_position: usize,
        /// The event that was out of order.
        event: Event,
    },

    /// The number of events differs between runs.
    CountMismatch {
        /// Number of events in run A.
        count_a: usize,
        /// Number of events in run B.
        count_b: usize,
    },

    /// A timing difference was detected.
    TimingDifference {
        /// Event from run A.
        event_a: Event,
        /// Event from run B.
        event_b: Event,
        /// Time difference in milliseconds.
        delta_ms: i64,
    },
}

/// Indicates which side of the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunSide {
    /// The first run (run A / baseline).
    RunA,
    /// The second run (run B / comparison).
    RunB,
}

/// A single difference found between two program runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Difference {
    /// Unique identifier for this difference.
    pub id: u64,

    /// The kind of difference detected.
    pub kind: DifferenceKind,

    /// Severity of the difference (0-10, higher is more severe).
    pub severity: u8,

    /// Human-readable summary of the difference.
    pub summary: String,
}

impl Difference {
    /// Creates a new difference with the given kind.
    pub fn new(id: u64, kind: DifferenceKind, severity: u8, summary: impl Into<String>) -> Self {
        Self {
            id,
            kind,
            severity,
            summary: summary.into(),
        }
    }
}

/// Configuration for the diff engine.
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Whether to ignore timing differences.
    pub ignore_timing: bool,

    /// Threshold in milliseconds for timing differences to be reported.
    pub timing_threshold_ms: u64,

    /// Whether to compare events by order or just by presence.
    pub strict_ordering: bool,

    /// Whether to include low-severity differences in results.
    pub include_minor: bool,

    /// Minimum severity level to include (0-10).
    pub min_severity: u8,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            ignore_timing: true,
            timing_threshold_ms: 100,
            strict_ordering: false,
            include_minor: true,
            min_severity: 0,
        }
    }
}

/// Result of comparing two program runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    /// Identifier for run A.
    pub run_a_id: String,

    /// Identifier for run B.
    pub run_b_id: String,

    /// All differences found.
    pub differences: Vec<Difference>,

    /// Summary statistics.
    pub stats: DiffStats,
}

/// Statistics about the diff operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffStats {
    /// Total events in run A.
    pub events_a: usize,

    /// Total events in run B.
    pub events_b: usize,

    /// Number of matching events.
    pub matching: usize,

    /// Number of differences found.
    pub differences: usize,

    /// Percentage similarity (0.0 to 1.0).
    pub similarity: f64,
}

/// Engine for comparing two program execution recordings.
///
/// The `DiffEngine` provides semantic comparison capabilities, detecting
/// differences in events, their content, ordering, and timing.
///
/// # Example
///
/// ```rust
/// use ghostdiff_core::{Recorder, DiffEngine};
/// use ghostdiff_core::diff::DiffConfig;
///
/// // Create two recordings
/// let mut run_a = Recorder::new("baseline");
/// run_a.track_ai_output("Hello, world!");
///
/// let mut run_b = Recorder::new("comparison");
/// run_b.track_ai_output("Hello, universe!");
///
/// // Compare with default config
/// let result = DiffEngine::compare(&run_a, &run_b);
///
/// println!("Found {} differences", result.differences.len());
/// println!("Similarity: {:.1}%", result.stats.similarity * 100.0);
/// ```
#[derive(Debug)]
pub struct DiffEngine {
    config: DiffConfig,
}

impl Default for DiffEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffEngine {
    /// Creates a new diff engine with default configuration.
    pub fn new() -> Self {
        Self {
            config: DiffConfig::default(),
        }
    }

    /// Creates a new diff engine with the specified configuration.
    pub fn with_config(config: DiffConfig) -> Self {
        Self { config }
    }

    /// Compares two recordings and returns the differences.
    ///
    /// This is a convenience method that uses default configuration.
    ///
    /// # Arguments
    ///
    /// * `run_a` - The baseline recording
    /// * `run_b` - The comparison recording
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::{Recorder, DiffEngine};
    ///
    /// let mut run_a = Recorder::new("a");
    /// run_a.track_function_call("test", vec![], None);
    ///
    /// let mut run_b = Recorder::new("b");
    /// run_b.track_function_call("test", vec![], Some("result".to_string()));
    ///
    /// let result = DiffEngine::compare(&run_a, &run_b);
    /// assert!(result.differences.len() > 0);
    /// ```
    pub fn compare(run_a: &Recorder, run_b: &Recorder) -> DiffResult {
        let engine = Self::new();
        engine.diff(run_a, run_b)
    }

    /// Performs a detailed comparison of two recordings.
    ///
    /// Uses the engine's configuration to control comparison behavior.
    pub fn diff(&self, run_a: &Recorder, run_b: &Recorder) -> DiffResult {
        let events_a = run_a.events();
        let events_b = run_b.events();
        let mut differences = Vec::new();
        let mut diff_id = 0;
        let mut matching = 0;

        // Check for count mismatch
        if events_a.len() != events_b.len() {
            differences.push(Difference::new(
                diff_id,
                DifferenceKind::CountMismatch {
                    count_a: events_a.len(),
                    count_b: events_b.len(),
                },
                3,
                format!(
                    "Event count differs: {} vs {}",
                    events_a.len(),
                    events_b.len()
                ),
            ));
            diff_id += 1;
        }

        // Compare events pairwise
        let max_len = events_a.len().max(events_b.len());
        for i in 0..max_len {
            match (events_a.get(i), events_b.get(i)) {
                (Some(a), Some(b)) => {
                    if let Some(mut diff) = self.compare_events(a, b) {
                        diff.id = diff_id;
                        differences.push(diff);
                        diff_id += 1;
                    } else {
                        matching += 1;
                    }
                }
                (Some(a), None) => {
                    differences.push(Difference::new(
                        diff_id,
                        DifferenceKind::Missing {
                            event: a.clone(),
                            missing_from: RunSide::RunB,
                        },
                        5,
                        format!("Event {:?} missing from run B", a.kind),
                    ));
                    diff_id += 1;
                }
                (None, Some(b)) => {
                    differences.push(Difference::new(
                        diff_id,
                        DifferenceKind::Missing {
                            event: b.clone(),
                            missing_from: RunSide::RunA,
                        },
                        5,
                        format!("Event {:?} missing from run A", b.kind),
                    ));
                    diff_id += 1;
                }
                (None, None) => unreachable!(),
            }
        }

        // Filter by severity
        let differences: Vec<_> = differences
            .into_iter()
            .filter(|d| d.severity >= self.config.min_severity)
            .collect();

        // Calculate statistics
        let total = events_a.len().max(events_b.len());
        let similarity = if total == 0 {
            1.0
        } else {
            matching as f64 / total as f64
        };

        DiffResult {
            run_a_id: run_a.run_id().to_string(),
            run_b_id: run_b.run_id().to_string(),
            stats: DiffStats {
                events_a: events_a.len(),
                events_b: events_b.len(),
                matching,
                differences: differences.len(),
                similarity,
            },
            differences,
        }
    }

    /// Compares two events and returns a difference if they don't match.
    fn compare_events(&self, a: &Event, b: &Event) -> Option<Difference> {
        // Compare event kinds
        match (&a.kind, &b.kind) {
            (
                EventKind::FunctionCall {
                    name: name_a,
                    args: args_a,
                    return_value: ret_a,
                },
                EventKind::FunctionCall {
                    name: name_b,
                    args: args_b,
                    return_value: ret_b,
                },
            ) => {
                if name_a != name_b || args_a != args_b || ret_a != ret_b {
                    return Some(Difference::new(
                        0,
                        DifferenceKind::ContentMismatch {
                            event_a: a.clone(),
                            event_b: b.clone(),
                            description: format!(
                                "Function call differs: {}({:?}) -> {:?} vs {}({:?}) -> {:?}",
                                name_a, args_a, ret_a, name_b, args_b, ret_b
                            ),
                        },
                        7,
                        format!("Function call '{}' differs", name_a),
                    ));
                }
            }
            (
                EventKind::AIOutput { content: content_a },
                EventKind::AIOutput { content: content_b },
            ) => {
                if content_a != content_b {
                    return Some(Difference::new(
                        0,
                        DifferenceKind::ContentMismatch {
                            event_a: a.clone(),
                            event_b: b.clone(),
                            description: format!(
                                "AI output differs: '{}' vs '{}'",
                                truncate_str(content_a, 50),
                                truncate_str(content_b, 50)
                            ),
                        },
                        8,
                        "AI output content differs".to_string(),
                    ));
                }
            }
            (
                EventKind::StateSnapshot {
                    label: label_a,
                    data: data_a,
                },
                EventKind::StateSnapshot {
                    label: label_b,
                    data: data_b,
                },
            ) => {
                if label_a != label_b || data_a != data_b {
                    return Some(Difference::new(
                        0,
                        DifferenceKind::ContentMismatch {
                            event_a: a.clone(),
                            event_b: b.clone(),
                            description: format!("State snapshot '{}' differs from '{}'", label_a, label_b),
                        },
                        6,
                        format!("State snapshot '{}' differs", label_a),
                    ));
                }
            }
            (kind_a, kind_b) if std::mem::discriminant(kind_a) != std::mem::discriminant(kind_b) => {
                return Some(Difference::new(
                    0,
                    DifferenceKind::ContentMismatch {
                        event_a: a.clone(),
                        event_b: b.clone(),
                        description: format!("Event types differ: {:?} vs {:?}", kind_a, kind_b),
                    },
                    9,
                    "Event types do not match".to_string(),
                ));
            }
            _ => {}
        }

        // Check timing if not ignored
        if !self.config.ignore_timing {
            let delta = (a.timestamp as i64) - (b.timestamp as i64);
            if delta.unsigned_abs() > self.config.timing_threshold_ms {
                return Some(Difference::new(
                    0,
                    DifferenceKind::TimingDifference {
                        event_a: a.clone(),
                        event_b: b.clone(),
                        delta_ms: delta,
                    },
                    2,
                    format!("Timing differs by {}ms", delta),
                ));
            }
        }

        None
    }

    /// Compares two slices of events (legacy API for backward compatibility).
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::DiffEngine;
    ///
    /// let run_a = vec!["start".to_string(), "process".to_string(), "end".to_string()];
    /// let run_b = vec!["start".to_string(), "compute".to_string(), "end".to_string()];
    ///
    /// let diffs = DiffEngine::compare_strings(&run_a, &run_b);
    /// assert_eq!(diffs.len(), 1);
    /// assert_eq!(diffs[0], "process != compute");
    /// ```
    pub fn compare_strings(run_a: &[String], run_b: &[String]) -> Vec<String> {
        let mut diffs = Vec::new();

        for (a, b) in run_a.iter().zip(run_b.iter()) {
            if a != b {
                diffs.push(format!("{} != {}", a, b));
            }
        }

        // Handle length differences
        if run_a.len() > run_b.len() {
            for item in run_a.iter().skip(run_b.len()) {
                diffs.push(format!("{} (missing in B)", item));
            }
        } else if run_b.len() > run_a.len() {
            for item in run_b.iter().skip(run_a.len()) {
                diffs.push(format!("{} (missing in A)", item));
            }
        }

        diffs
    }
}

impl DiffResult {
    /// Returns true if the two runs are identical.
    pub fn is_identical(&self) -> bool {
        self.differences.is_empty()
    }

    /// Returns differences filtered by minimum severity.
    pub fn with_min_severity(&self, min: u8) -> Vec<&Difference> {
        self.differences.iter().filter(|d| d.severity >= min).collect()
    }

    /// Serializes the diff result to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Generates a human-readable summary of the differences.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ghostdiff_core::{Recorder, DiffEngine};
    ///
    /// let mut run_a = Recorder::new("a");
    /// run_a.track_ai_output("Hello");
    ///
    /// let mut run_b = Recorder::new("b");
    /// run_b.track_ai_output("World");
    ///
    /// let result = DiffEngine::compare(&run_a, &run_b);
    /// let summary = result.summary();
    /// println!("{}", summary);
    /// ```
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!(
            "Comparing {} vs {}",
            self.run_a_id, self.run_b_id
        ));
        lines.push(format!(
            "Events: {} vs {} ({} matching)",
            self.stats.events_a, self.stats.events_b, self.stats.matching
        ));
        lines.push(format!(
            "Similarity: {:.1}%",
            self.stats.similarity * 100.0
        ));
        lines.push(format!("Differences: {}", self.stats.differences));

        if !self.differences.is_empty() {
            lines.push(String::new());
            lines.push("Details:".to_string());
            for diff in &self.differences {
                lines.push(format!("  [{}] {}", diff.severity, diff.summary));
            }
        }

        lines.join("\n")
    }
}

/// Truncates a string to the specified length, adding "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_runs() {
        let mut run_a = Recorder::new("a");
        run_a.track_ai_output("Hello");
        run_a.track_function_call("test", vec![], None);

        let mut run_b = Recorder::new("b");
        run_b.track_ai_output("Hello");
        run_b.track_function_call("test", vec![], None);

        let result = DiffEngine::compare(&run_a, &run_b);

        // Only timing differences should be ignored
        assert!(result.stats.similarity > 0.9);
    }

    #[test]
    fn test_different_ai_output() {
        let mut run_a = Recorder::new("a");
        run_a.track_ai_output("Hello, world!");

        let mut run_b = Recorder::new("b");
        run_b.track_ai_output("Hello, universe!");

        let result = DiffEngine::compare(&run_a, &run_b);
        assert!(!result.is_identical());
        assert!(result.differences.iter().any(|d| matches!(
            &d.kind,
            DifferenceKind::ContentMismatch { .. }
        )));
    }

    #[test]
    fn test_missing_events() {
        let mut run_a = Recorder::new("a");
        run_a.track_ai_output("One");
        run_a.track_ai_output("Two");

        let mut run_b = Recorder::new("b");
        run_b.track_ai_output("One");

        let result = DiffEngine::compare(&run_a, &run_b);
        assert!(result.differences.iter().any(|d| matches!(
            &d.kind,
            DifferenceKind::Missing { .. }
        )));
    }

    #[test]
    fn test_compare_strings_legacy() {
        let run_a = vec!["a".to_string(), "b".to_string()];
        let run_b = vec!["a".to_string(), "c".to_string()];

        let diffs = DiffEngine::compare_strings(&run_a, &run_b);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], "b != c");
    }
}
