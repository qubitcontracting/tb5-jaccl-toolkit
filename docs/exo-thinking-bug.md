# Exo Degradation Bug: Thinking Model KV Prefix Cache Accumulation

## Summary

Exo's MLX inference engine degrades after sustained inference with thinking/reasoning models. Subsequent requests time out (30s+) after as few as one completed task. Non-thinking models and Ollama running the same thinking model are unaffected.

## Root Cause

**File:** `src/exo/worker/engines/mlx/generator/generate.py`, lines 647-674

After each inference request, Exo re-encodes ALL generated tokens (including thinking/reasoning tokens) and stores them in the `kv_prefix_cache`:

```python
generated_tokens_array = mx.array(
    tokenizer.encode(
        "".join(generated_text_parts), add_special_tokens=False
    )
)
full_prompt_tokens = mx.concatenate(
    [all_prompt_tokens, generated_tokens_array]
)
kv_prefix_cache.add_kv_cache(full_prompt_tokens, caches, cache_snapshots)
```

With thinking models generating 5,000-8,000 tokens per request (mostly `<think>` reasoning), the cached KV state grows unbounded across requests.

## Secondary Bug: Missing `is_thinking` Flag

Both `generate.py` (line 679) and `batch_generate.py` (line 326) track thinking state via `state.in_thinking` but never pass it to `GenerationResponse`. This prevents the API layer from properly splitting `reasoning_content` from `content`.

## Evidence

### Test Matrix

| Test | Model | Runtime | KV Cache | Post-task Health |
|------|-------|---------|----------|-----------------|
| A22B (thinking) | Qwen3-235B-A22B 8-bit | Exo JACCL | Default | **30s+ TIMEOUT** |
| A22B (thinking) | Qwen3-235B-A22B 4-bit | Ollama | q8_0 | 3.8s HEALTHY |
| A22B (thinking) | Qwen3-235B-A22B 4-bit | Ollama | Default (bf16) | 3.8s HEALTHY |
| Coder-Next (non-thinking) | Qwen3-Coder-Next bf16 | Exo JACCL | Default | 1.2s HEALTHY |

### Benchmark Scores (5-task sustained run, no restarts)

| Runtime | Avg Score | Degradation |
|---------|-----------|-------------|
| Ollama (thinking model) | **73.8/100** | None — all health checks < 7s |
| Exo JACCL (thinking model) | **16.8/100** | Severe — timeouts after task 1 |
| Exo JACCL (non-thinking model) | **39.0/100** | None |

### Task-by-task (Ollama, all HEALTHY)

| Task | Score | Health After |
|------|-------|-------------|
| Static Site Generator | 89/100 | HEALTHY (6.9s) |
| REST API with SQLite | 90/100 | HEALTHY (3.8s) |
| Data Pipeline | 100/100 | HEALTHY (3.8s) |
| CLI Tool | 10/100 | HEALTHY (3.8s) |
| A* Pathfinding | 80/100 | HEALTHY (3.9s) |

## Why Non-Thinking Models Don't Degrade

Coder-Next generates ~200-500 tokens per response (all content, no thinking). The prefix cache stays small. Thinking models generate 5,000-8,000 tokens, most of which are reasoning tokens that bloat the cache.

## Why Ollama Doesn't Degrade

Ollama (llama.cpp backend) resets the KV cache between independent chat completion requests. There is no cross-request prefix caching.

## Suggested Fix

Either:
1. **Exclude thinking tokens** from the prefix cache — only cache content tokens
2. **Reset prefix cache** between independent requests (detect new conversation vs continuation)
3. **Add a cache size limit** that evicts old entries when exceeded
4. **Pass `is_thinking` flag** through to `GenerationResponse` so the API can properly separate reasoning from content

## Hardware

- Mac Studio M3 Ultra (256 GB) + 2x Mac mini M4 Pro (64 GB, 48 GB)
- 3-node Thunderbolt 5 RDMA mesh (JACCL)
- macOS 26.4, MLX 0.31.1-dev, Exo latest

## Reproduction

1. Start 3-node Exo cluster with JACCL
2. Place `mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit` with `MlxJaccl` instance meta
3. Run any multi-turn agentic task (e.g., our CLI tool benchmark)
4. After task completes, send a simple "Hi" request
5. Observe: request times out or takes 30s+ (should be <2s)
