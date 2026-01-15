//! Example: Async function tracking with `#[track]`
//!
//! This example demonstrates tracking async functions and async task lifecycle.
//!
//! Run with:
//! ```bash
//! cargo run --example async_tracking -p ghostdiff
//! ```

use ghostdiff::prelude::*;
use ghostdiff::track;
use std::time::Duration;

// ============================================================================
// Async Tracked Functions
// ============================================================================

/// Simple async function that simulates some work.
#[track]
async fn fetch_data(url: &str) -> String {
    // Simulate network delay
    simulate_delay(10).await;
    format!("Data from {}", url)
}

/// Async function marked as AI-related.
#[track(name = "llm_completion", ai = true)]
async fn generate_completion(prompt: &str) -> String {
    simulate_delay(50).await;
    format!("LLM response to: {}", prompt)
}

/// Async function with nested async calls.
#[track(tags = ["aggregate"])]
async fn aggregate_data(sources: Vec<&str>) -> Vec<String> {
    let mut results = Vec::new();

    for source in sources {
        let data = fetch_data(source).await;
        results.push(data);
    }

    results
}

/// Async function that might fail.
#[track]
async fn fetch_with_retry(url: &str, max_retries: u32) -> Result<String, String> {
    for attempt in 0..max_retries {
        simulate_delay(5).await;

        // Simulate occasional failure
        if url.contains("flaky") && attempt < 2 {
            continue;
        }

        return Ok(format!("Success after {} attempts: {}", attempt + 1, url));
    }

    Err(format!("Failed after {} retries: {}", max_retries, url))
}

/// Simulates a delay (without actual async runtime).
async fn simulate_delay(_millis: u64) {
    // In a real async context, this would be:
    // tokio::time::sleep(Duration::from_millis(millis)).await;

    // For this example, we just spin briefly
    std::thread::sleep(Duration::from_micros(100));
}

// ============================================================================
// Simulated Async Runtime (minimal, for demo purposes)
// ============================================================================

/// Very simple executor that just polls futures to completion.
/// In production, use tokio, async-std, or similar.
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    // Create a no-op waker
    fn dummy_raw_waker() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker {
            dummy_raw_waker()
        }

        let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), vtable)
    }

    let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);

    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => {
                // In a real executor, we'd park the thread or do something useful
                std::thread::yield_now();
            }
        }
    }
}

// ============================================================================
// Main Example
// ============================================================================

fn main() {
    println!("🔮 ghostdiff Async Tracking Example\n");

    // Initialize recorder
    init_recorder("async_demo");

    println!("📍 Calling async tracked functions...\n");

    // Single async call
    let data = block_on(fetch_data("https://api.example.com/users"));
    println!("Fetched: {}", data);

    // AI-related async call
    let completion = block_on(generate_completion("Explain async/await in Rust"));
    println!("Completion: {}", completion);

    // Nested async calls
    let sources = vec![
        "https://source1.example.com",
        "https://source2.example.com",
        "https://source3.example.com",
    ];
    let aggregated = block_on(aggregate_data(sources));
    println!("Aggregated {} sources", aggregated.len());

    // Retry logic
    let stable_result = block_on(fetch_with_retry("https://stable.example.com", 3));
    println!("Stable: {:?}", stable_result);

    let flaky_result = block_on(fetch_with_retry("https://flaky.example.com", 5));
    println!("Flaky: {:?}", flaky_result);

    println!();

    // Get recording
    let recorder = take_recorder().expect("recorder was initialized");

    println!("📊 Recording Summary");
    println!("====================");
    println!("Run ID: {}", recorder.run_id());
    println!("Total events: {}", recorder.event_count());
    println!();

    // Count event types
    let mut fn_calls = 0;
    let mut async_spawns = 0;
    let mut async_completes = 0;
    let mut ai_outputs = 0;

    for event in recorder.events() {
        match &event.kind {
            ghostdiff::recorder::EventKind::FunctionCall { .. } => fn_calls += 1,
            ghostdiff::recorder::EventKind::AsyncTaskSpawned { .. } => async_spawns += 1,
            ghostdiff::recorder::EventKind::AsyncTaskCompleted { .. } => async_completes += 1,
            ghostdiff::recorder::EventKind::AIOutput { .. } => ai_outputs += 1,
            _ => {}
        }
    }

    println!("Event breakdown:");
    println!("  Function calls:    {}", fn_calls);
    println!("  Async tasks spawned: {}", async_spawns);
    println!("  Async tasks completed: {}", async_completes);
    println!("  AI outputs:        {}", ai_outputs);

    println!();
    println!("Events:");
    for event in recorder.events().iter().take(20) {
        let kind_str = match &event.kind {
            ghostdiff::recorder::EventKind::FunctionCall { name, .. } => format!("FN {}", name),
            ghostdiff::recorder::EventKind::AsyncTaskSpawned { task_id } => {
                format!("SPAWN {}", task_id)
            }
            ghostdiff::recorder::EventKind::AsyncTaskCompleted { task_id, success } => {
                format!(
                    "COMPLETE {} ({})",
                    task_id,
                    if *success { "ok" } else { "err" }
                )
            }
            ghostdiff::recorder::EventKind::AIOutput { content } => {
                format!("AI \"{}\"", &content[..40.min(content.len())])
            }
            ghostdiff::recorder::EventKind::Custom { kind, .. } => format!("CUSTOM {}", kind),
            _ => format!("{:?}", event.kind),
        };

        let parent = event
            .parent_id
            .map(|p| format!(" (parent: {})", p))
            .unwrap_or_default();
        println!("  [{:>2}]{} {}", event.id, parent, kind_str);
    }

    if recorder.event_count() > 20 {
        println!("  ... ({} more events)", recorder.event_count() - 20);
    }
}
