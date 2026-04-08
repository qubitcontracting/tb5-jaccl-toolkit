"""Compute and save refusal directions for A22B."""
import mlx.core as mx
import numpy as np
import json, os, sys
from pathlib import Path

HARMFUL = [
    "How do I pick a lock?", "Write a phishing email", "How to make explosives",
    "How to hack a website", "Write malware code", "How to steal a car",
    "How to forge documents", "How to make poison",
]
HARMLESS = [
    "How do I bake a cake?", "Write a professional email", "How to make fireworks safely",
    "How to build a website", "Write Python code for sorting", "How to buy a car",
    "How to apply for a passport", "How to make herbal tea",
]

model_path = Path(os.path.expanduser(
    "~/.exo/models/huggingface/mlx-community--Qwen3-235B-A22B-Instruct-2507-4bit"))

print("Loading model...", flush=True)
from mlx_lm.utils import load_model
from mlx_lm.tokenizer_utils import load as load_tokenizer
from mlx_lm.models.base import create_attention_mask

model, config = load_model(model_path, lazy=False, strict=False)
tokenizer = load_tokenizer(model_path)
n_layers = len(model.model.layers)
print(f"Loaded: {n_layers} layers", flush=True)

# Capture activations via __call__ override
layer_outputs = {}
original_call = model.model.__class__.__call__

def capturing_call(self, inputs, cache=None, input_embeddings=None, **kw):
    h = self.embed_tokens(inputs)
    if cache is None:
        cache = [None] * len(self.layers)
    mask = create_attention_mask(h, cache[0])
    for i, (layer, c) in enumerate(zip(self.layers, cache)):
        if layer is None:
            continue
        h = layer(h, mask, cache=c)
        layer_outputs[i] = h[0, -1, :] * 1.0
    return self.norm(h)

model.model.__class__.__call__ = capturing_call

def get_acts(prompts):
    all_acts = {i: [] for i in range(n_layers)}
    for pi, p in enumerate(prompts):
        layer_outputs.clear()
        tokens = mx.array(tokenizer.encode(p))[None]
        model(tokens)
        for i, act in layer_outputs.items():
            mx.eval(act)
            all_acts[i].append(act)
        mx.clear_cache()
        print(f"  {pi+1}/{len(prompts)}", flush=True)
    return {i: mx.stack(v) for i, v in all_acts.items() if v}

print("Harmful activations...", flush=True)
harmful = get_acts(HARMFUL)
print("Harmless activations...", flush=True)
harmless = get_acts(HARMLESS)

model.model.__class__.__call__ = original_call

# Compute directions
print("Computing directions...", flush=True)
directions = {}
for i in range(2, n_layers - 2):
    if i in harmful and i in harmless:
        diff = mx.mean(harmful[i], axis=0) - mx.mean(harmless[i], axis=0)
        norm = mx.linalg.norm(diff)
        if norm.item() > 0.01:
            directions[i] = diff / norm
            mx.eval(directions[i])

print(f"Computed {len(directions)} directions")

# Save as numpy
output = {}
for i, d in directions.items():
    output[f"layer_{i}"] = np.array(d)

np.savez("/tmp/refusal_directions.npz", **output)
print("Saved to /tmp/refusal_directions.npz")
