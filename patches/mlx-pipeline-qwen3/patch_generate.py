"""Patch generate.py in-place to add _pipeline_sync for distributed pipeline.
Works on mlx-lm 0.31.1 and 0.31.2."""
import sys

target = sys.argv[1]
with open(target) as f:
    content = f.read()

if '_pipeline_sync' in content:
    print(f"  {target}: already patched")
    sys.exit(0)

changes = 0

# 1. Add _pipeline_sync function after DEFAULT_PROMPT line
sync_fn = '\n\ndef _pipeline_sync():\n    """Return a collective op that forces all pipeline ranks to eval together."""\n    group = mx.distributed.init()\n    if group.size() > 1:\n        return mx.distributed.all_sum(mx.zeros(1))\n    return None\n\n'
if 'DEFAULT_PROMPT' in content:
    content = content.replace('DEFAULT_PROMPT = "hello"', 'DEFAULT_PROMPT = "hello"' + sync_fn)
    changes += 1

# 2. Patch prefill: capture logits and eval with sync
old_prefill = "            _model_call(\n"
new_prefill = "            _prefill_logits = _model_call(\n"
if old_prefill in content:
    content = content.replace(old_prefill, new_prefill, 1)

old_prefill_eval = "            quantize_cache_fn(prompt_cache)\n            mx.eval([c.state for c in prompt_cache])"
new_prefill_eval = """            quantize_cache_fn(prompt_cache)
            _psync = _pipeline_sync()
            if _psync is not None:
                mx.eval(_prefill_logits, _psync)
            else:
                mx.eval([c.state for c in prompt_cache])"""
if old_prefill_eval in content:
    content = content.replace(old_prefill_eval, new_prefill_eval, 1)
    changes += 1

# 3. Patch first token eval
old_first = "    mx.async_eval(y, logprobs)\n    n = 0"
new_first = "    _sync = _pipeline_sync()\n    if _sync is not None:\n        mx.eval(y, logprobs, _sync)\n    else:\n        mx.async_eval(y, logprobs)\n    n = 0"
if old_first in content:
    content = content.replace(old_first, new_first, 1)
    changes += 1

# 4. Patch generation loop
old_loop = "            next_y, next_logprobs = _step(y)\n            mx.async_eval(next_y, next_logprobs)"
new_loop = "            next_y, next_logprobs = _step(y)\n            if _sync is not None:\n                _sync = _pipeline_sync()\n                mx.eval(next_y, next_logprobs, _sync)\n            else:\n                mx.async_eval(next_y, next_logprobs)"
if old_loop in content:
    content = content.replace(old_loop, new_loop, 1)
    changes += 1

if changes >= 3:
    with open(target, 'w') as f:
        f.write(content)
    print(f"  {target}: {changes} patches applied")
else:
    print(f"  WARNING: Only {changes}/4 patches matched in {target}")
    if changes > 0:
        with open(target, 'w') as f:
            f.write(content)
