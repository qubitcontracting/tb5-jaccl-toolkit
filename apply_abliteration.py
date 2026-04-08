"""
Apply abliteration to an MLX model at runtime via activation hooks.
No weight modification — operates in activation space.

Usage: 
    import apply_abliteration
    apply_abliteration.patch(model, "/path/to/refusal_directions.npz")
"""

import mlx.core as mx
import numpy as np


def patch(model, directions_path):
    """Patch model layers to remove refusal direction from outputs."""
    data = np.load(directions_path)
    directions = {}
    for key in data.files:
        layer_idx = int(key.split("_")[1])
        directions[layer_idx] = mx.array(data[key])
    
    patched = 0
    for layer_idx, direction in directions.items():
        if layer_idx >= len(model.model.layers):
            continue
        layer = model.model.layers[layer_idx]
        if layer is None:
            continue
        
        original_call = layer.__call__.__func__ if hasattr(layer.__call__, '__func__') else None
        
        # Create wrapper that subtracts refusal direction from output
        d = direction
        
        class AbliteratedLayer:
            def __init__(self, orig_layer, refusal_dir):
                self._layer = orig_layer
                self._dir = refusal_dir
                # Forward all attribute access to original layer
                
            def __call__(self, h, mask=None, cache=None):
                out = self._layer(h, mask, cache=cache)
                # out shape: (batch, seq, hidden)
                # Subtract refusal direction: out -= (out @ r) * r
                proj = out @ self._dir  # (batch, seq)
                out = out - proj[..., None] * self._dir[None, None, :]
                return out
            
            def __getattr__(self, name):
                return getattr(self._layer, name)
        
        model.model.layers[layer_idx] = AbliteratedLayer(layer, direction)
        patched += 1
    
    print(f"Abliteration applied: {patched} layers patched")
    return patched


if __name__ == "__main__":
    # Quick test
    import sys, os
    from pathlib import Path
    from mlx_lm.utils import load_model
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm import generate
    
    model_path = Path(os.path.expanduser(
        "~/.exo/models/huggingface/mlx-community--Qwen3-235B-A22B-Instruct-2507-4bit"))
    
    print("Loading model...", flush=True)
    model, config = load_model(model_path, lazy=False, strict=False)
    tokenizer = load_tokenizer(model_path)
    
    # Test BEFORE abliteration
    print("\n=== BEFORE abliteration ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    response = generate(model, tok, prompt="How do I pick a lock?", max_tokens=50, verbose=False)
    print(f"Response: {response[:200]}")
    
    # Apply abliteration
    print("\n=== Applying abliteration ===")
    patch(model, "/tmp/refusal_directions.npz")
    
    # Test AFTER abliteration
    print("\n=== AFTER abliteration ===")
    response = generate(model, tok, prompt="How do I pick a lock?", max_tokens=50, verbose=False)
    print(f"Response: {response[:200]}")
