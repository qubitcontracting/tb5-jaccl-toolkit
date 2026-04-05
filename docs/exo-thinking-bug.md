## Bug: Inference degradation with thinking/reasoning models after sustained use

### Description

When running thinking/reasoning models (e.g., Qwen3-235B-A22B-Instruct with thinking mode), Exo degrades after a single completed inference task. Subsequent requests time out (30s+) even for trivial prompts like "Hi". Non-thinking models (e.g., Qwen3-Coder-Next) do not exhibit this behavior.

### Root Cause

The `kv_prefix_cache` in `src/exo/worker/engines/mlx/generator/generate.py` (lines 647-674) accumulates thinking tokens across requests:

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

All generated tokens (including thinking/reasoning tokens) are re-encoded and stored in the prefix cache, causing it to grow unbounded. Each subsequent request inherits the bloated cache state, leading to progressively slower inference until timeouts. Thinking models trigger this fastest (~5,000-8,000 tokens/request), but non-thinking models also degrade after heavy tasks with high total token output.

### Secondary Issue

`GenerationResponse` in both `generate.py` and `batch_generate.py` tracks thinking state (`state.in_thinking`) but never passes `is_thinking` to the response object. This prevents the API from properly splitting `reasoning_content` from `content`.

### Reproduction

1. Start Exo cluster (tested with 3-node JACCL, macOS 26.4, MLX 0.31.1-dev)
2. Place a thinking model: `mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit`
3. Run an agentic coding task requiring multiple tool-use iterations (~5-10 inference calls)
4. After task completes, send a simple request: `{"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}`
5. **Expected:** Response in <2s
6. **Actual:** Request times out (30s+)

### Test Matrix

| Model | Task Weight | Runtime | Post-task Health |
|-------|------------|---------|-----------------|
| A22B (thinking) | 1 light task | **Exo JACCL** | **30s+ TIMEOUT** |
| A22B (thinking) | 5 tasks sustained | Ollama (q8_0 KV) | 3.8s HEALTHY |
| A22B (thinking) | 2 tasks sustained | Ollama (default KV) | 3.8s HEALTHY |
| Coder-Next (non-thinking) | 1 light task (CLI) | Exo JACCL | 1.2s HEALTHY |
| Coder-Next (non-thinking) | 1 heavy task (Data Pipeline) | **Exo JACCL** | **33.9s TIMEOUT** |

**Key finding:** Non-thinking models ALSO degrade on Exo after heavy tasks with high token output. The issue is proportional to total generated tokens cached in `kv_prefix_cache`, not specific to thinking mode. Thinking models trigger it faster (~5,000-8,000 tokens/request vs ~500 tokens/request for non-thinking models).

Ollama (llama.cpp backend) does not exhibit this issue because it resets the KV cache between independent requests rather than using a persistent prefix cache.

### Suggested Fix

1. Reset the prefix cache between independent (non-continuation) requests — most impactful fix
2. Add a cache size limit that evicts old entries when total cached tokens exceed a threshold
3. Exclude thinking tokens from the prefix cache
4. Pass `is_thinking` flag through to `GenerationResponse`

### Environment

- macOS 26.4
- MLX 0.31.1-dev (ce45c52)
- Exo latest (as of 2026-04-05)
- Hardware: Mac Studio M3 Ultra (256GB) + 2x Mac mini M4 Pro (64GB, 48GB)
- JACCL RDMA over Thunderbolt 5
