# GhostDiff Demo (Real Run)

## Problem
AI behavior changes across runs are hard to debug because small output differences can come from prompt changes, config changes, model changes, or stochastic sampling. The run below shows how GhostDiff records two runs and explains the divergence.

## Steps

### 1) Record run 1
```bash
cargo run -p ghostdiff_cli -- record --output run1.json \
  --ai-provider gemini \
  --model gemini-2.5-flash \
  --temperature 0.7 \
  --prompt "Summarize Rust lifetimes in one sentence."
```

Output:
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.14s
     Running `target/debug/ghostdiff record --output run1.json --ai-provider gemini --model gemini-2.5-flash --temperature 0.7 --prompt 'Summarize Rust lifetimes in one sentence.'`
📹 Recording execution...


✓ Recorded 13 events to run1.json
```

### 2) Record run 2 (same prompt)
```bash
cargo run -p ghostdiff_cli -- record --output run2.json \
  --ai-provider gemini \
  --model gemini-2.5-flash \
  --temperature 0.7 \
  --prompt "Summarize Rust lifetimes in one sentence."
```

Output:
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.04s
     Running `target/debug/ghostdiff record --output run2.json --ai-provider gemini --model gemini-2.5-flash --temperature 0.7 --prompt 'Summarize Rust lifetimes in one sentence.'`
📹 Recording execution...


✓ Recorded 13 events to run2.json
```

### 3) Diff with AI semantics
```bash
cargo run -p ghostdiff_cli -- diff --run1 run1.json --run2 run2.json --ai --explain
```

Output:
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.13s
     Running `target/debug/ghostdiff diff --run1 run1.json --run2 run2.json --ai --explain`
Comparing run vs run

Statistics
  Events:     13 vs 13
  Matching:   11
  Similarity: 84.6%
  Differences: 2

Differences

  [8] MISMATCH AI output differs: 'Rust lifetimes are a compile-time mechanism tha...' vs 'Rust lifetimes are a compile-time system that e...'

  [8] MISMATCH AI output differs: 'Ok("Rust lifetimes are a compile-time mechanism...' vs 'Ok("Rust lifetimes are a compile-time system th...'

AI Semantic Diff
  Root cause: SamplingInstability
  Token divergence at 5: 'mechanism' vs 'system'

Explanation
Root cause: SamplingInstability
Divergence started at token 5 ('mechanism' vs 'system')
Determinism assessment: stochastic
```

## Explanation
- **What changed:** the AI output sentence diverged in wording ("mechanism" vs "system").
- **Where divergence started:** token 5, as reported in the AI Semantic Diff section.
- **Why SamplingInstability:** GhostDiff classified the change as stochastic because the prompt and configuration were identical across runs, and the divergence appears at a specific token with no upstream divergence indicated in the output.

## What this demonstrates
- GhostDiff records full execution traces and AI outputs for both runs.
- The semantic diff pinpoints the first token divergence and classifies the cause.
- The explanation is deterministic and traceable to recorded evidence.
