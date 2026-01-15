# GhostDiff

GhostDiff is an open-source Rust tool for time-travel execution recording and semantic diffing of AI behavior. It records program events and AI interactions and produces deterministic explanations for why two runs diverge.

## Why this exists
AI outputs can change between runs, but it’s hard to tell if that change is due to prompt/config/model differences or stochastic sampling. GhostDiff makes the cause explicit using recorded traces and semantic AI diffing.

## Demo (trimmed)
Commands:
```bash
cargo run -p ghostdiff_cli -- record --output run1.json \
  --ai-provider gemini \
  --model gemini-2.5-flash \
  --temperature 0.7 \
  --prompt "Summarize Rust lifetimes in one sentence."

cargo run -p ghostdiff_cli -- record --output run2.json \
  --ai-provider gemini \
  --model gemini-2.5-flash \
  --temperature 0.7 \
  --prompt "Summarize Rust lifetimes in one sentence."

cargo run -p ghostdiff_cli -- diff --run1 run1.json --run2 run2.json --ai --explain
```

Output (trimmed):
```
Comparing run vs run

Statistics
  Events:     13 vs 13
  Matching:   11
  Similarity: 84.6%
  Differences: 2

AI Semantic Diff
  Root cause: SamplingInstability
  Token divergence at 5: 'mechanism' vs 'system'

Explanation
Root cause: SamplingInstability
Divergence started at token 5 ('mechanism' vs 'system')
Determinism assessment: stochastic
```

## Features
- Time-travel recording of function calls, async tasks, and state snapshots
- AI interaction tracking (prompts, messages, outputs)
- Token-level tracing and semantic AI diffing
- Deterministic root-cause classification (e.g., sampling vs. config/prompt changes)
- CLI tooling for record and diff workflows

## Why this is novel: Semantic AI Diffing
**Semantic AI Diffing** goes beyond string comparison by analyzing token divergence and classifying the root cause using deterministic logic, grounded in recorded execution traces.

## Quick start
```bash
# Record two runs
cargo run -p ghostdiff_cli -- record --output run1.json \
  --ai-provider gemini \
  --model gemini-2.5-flash \
  --temperature 0.7 \
  --prompt "Summarize Rust lifetimes in one sentence."

cargo run -p ghostdiff_cli -- record --output run2.json \
  --ai-provider gemini \
  --model gemini-2.5-flash \
  --temperature 0.7 \
  --prompt "Summarize Rust lifetimes in one sentence."

# Diff with AI semantics
cargo run -p ghostdiff_cli -- diff --run1 run1.json --run2 run2.json --ai --explain
```
