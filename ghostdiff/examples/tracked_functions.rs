//! Example: Using the `#[track]` macro for automatic function instrumentation
//!
//! This example demonstrates how to use ghostdiff's automatic instrumentation
//! to record function calls, arguments, and return values.
//!
//! Run with:
//! ```bash
//! cargo run --example tracked_functions -p ghostdiff
//! ```

use ghostdiff::prelude::*;
use ghostdiff::track;

// ============================================================================
// Basic Tracked Functions
// ============================================================================

/// A simple tracked function that adds two numbers.
///
/// The macro will automatically record:
/// - Function name: "add"
/// - Arguments: [a, b] as debug strings
/// - Return value: the sum
#[track]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Tracked function with a custom name.
///
/// Recorded as "multiply_numbers" instead of "multiply".
#[track(name = "multiply_numbers")]
fn multiply(x: i32, y: i32) -> i32 {
    x * y
}

/// Tracked function that calls other tracked functions.
///
/// This demonstrates nested event recording with parent-child relationships.
#[track]
fn calculate(a: i32, b: i32) -> i32 {
    let sum = add(a, b);
    let product = multiply(a, b);
    sum + product
}

// ============================================================================
// AI-Related Functions
// ============================================================================

/// Simulated AI inference function.
///
/// The `ai = true` flag marks this as an AI-related operation,
/// which adds additional tracking for AI outputs.
#[track(name = "ai_inference", ai = true)]
fn simulate_inference(prompt: &str) -> String {
    // In a real scenario, this would call an AI model
    format!("AI response to: {}", prompt)
}

/// Another AI function with tags.
#[track(ai = true, tags = ["ml", "embedding"])]
fn compute_embedding(text: &str) -> Vec<f32> {
    // Simulated embedding computation
    vec![0.1, 0.2, 0.3, text.len() as f32 / 100.0]
}

// ============================================================================
// Functions with Sensitive Data
// ============================================================================

/// Function that handles sensitive data.
///
/// Using `skip_args = true` prevents the password from being logged.
#[track(skip_args = true)]
fn authenticate(username: &str, password: &str) -> bool {
    // Password won't be captured in the recording
    let _ = (username, password);
    true
}

/// Function that skips return value capture.
#[track(skip_return = true)]
fn get_secret_token() -> String {
    "super-secret-token-12345".to_string()
}

// ============================================================================
// Complex Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct User {
    id: u64,
    name: String,
}

#[derive(Debug)]
struct ProcessResult {
    success: bool,
    message: String,
    count: usize,
}

/// Function with complex types.
///
/// The Debug trait is used to serialize arguments and return values.
#[track]
fn process_user(user: User) -> ProcessResult {
    ProcessResult {
        success: true,
        message: format!("Processed user: {}", user.name),
        count: 1,
    }
}

/// Function returning Option.
#[track]
fn find_user(id: u64) -> Option<User> {
    if id > 0 {
        Some(User {
            id,
            name: format!("User {}", id),
        })
    } else {
        None
    }
}

/// Function returning Result.
#[track]
fn parse_number(s: &str) -> Result<i32, String> {
    s.parse().map_err(|_| format!("Failed to parse: {}", s))
}

// ============================================================================
// Main Example
// ============================================================================

fn main() {
    println!("🔮 ghostdiff Macro Example\n");

    // Initialize the recorder for this thread
    init_recorder("macro_demo");

    // Record some metadata
    with_recorder(|recorder| {
        recorder
            .metadata_mut()
            .custom
            .insert("example".to_string(), "tracked_functions".to_string());
    });

    println!("📍 Calling tracked functions...\n");

    // Basic math operations
    println!("add(2, 3) = {}", add(2, 3));
    println!("multiply(4, 5) = {}", multiply(4, 5));
    println!("calculate(3, 4) = {} (nested calls)", calculate(3, 4));

    println!();

    // AI operations
    let response = simulate_inference("What is Rust?");
    println!("AI inference: {}", response);

    let embedding = compute_embedding("Hello, world!");
    println!("Embedding: {:?}", embedding);

    println!();

    // Sensitive data handling
    let auth_result = authenticate("user@example.com", "secret123");
    println!("Authentication: {}", if auth_result { "success" } else { "failed" });

    let token = get_secret_token();
    println!("Got token: {}...", &token[..10]);

    println!();

    // Complex types
    let user = User {
        id: 42,
        name: "Alice".to_string(),
    };
    let result = process_user(user);
    println!("Process result: {:?}", result);

    let found = find_user(123);
    println!("Found user: {:?}", found);

    let parsed = parse_number("42");
    println!("Parsed: {:?}", parsed);

    let failed = parse_number("not a number");
    println!("Failed parse: {:?}", failed);

    println!();

    // Finalize and get the recording
    let recorder = take_recorder().expect("recorder was initialized");
    let json = recorder.to_json().expect("serialization works");

    println!("📊 Recording Summary");
    println!("====================");
    println!("Run ID: {}", recorder.run_id());
    println!("Total events: {}", recorder.event_count());
    println!();

    // Print event summary
    println!("Events recorded:");
    for event in recorder.events() {
        match &event.kind {
            ghostdiff::recorder::EventKind::FunctionCall { name, args, return_value } => {
                let args_str = if args.is_empty() {
                    "()".to_string()
                } else {
                    format!("({})", args.join(", "))
                };
                let ret_str = return_value
                    .as_ref()
                    .map(|r| format!(" -> {}", truncate(r, 30)))
                    .unwrap_or_default();
                println!("  [{:>2}] FN {}{}{}", event.id, name, args_str, ret_str);
            }
            ghostdiff::recorder::EventKind::AIOutput { content } => {
                println!("  [{:>2}] AI \"{}\"", event.id, truncate(content, 40));
            }
            ghostdiff::recorder::EventKind::Custom { kind, payload } => {
                println!("  [{:>2}] {} {}", event.id, kind, truncate(payload, 30));
            }
            _ => {
                println!("  [{:>2}] {:?}", event.id, event.kind);
            }
        }
    }

    println!();
    println!("💾 Full JSON recording ({} bytes):", json.len());
    println!("{}", &json[..500.min(json.len())]);
    if json.len() > 500 {
        println!("... (truncated)");
    }

    // Optionally save to file
    // std::fs::write("macro_demo.json", &json).unwrap();
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}
