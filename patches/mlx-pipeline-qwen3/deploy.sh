#!/bin/bash
# Deploy pipeline patches to mlx-lm on local or remote nodes.
# Usage: ./deploy.sh [node1 node2 ...]
#   No args = local only. Args = deploy to listed SSH hosts.
#
# Patches:
#   pipeline.py   -> models/pipeline.py  (proportional split + warmup)
#   qwen3_moe.py  -> models/qwen3_moe.py (PipelineMixin + recv/send/all_gather)
#   generate.py   -> generate.py         (pipeline sync in generation loop)
#   utils.py      -> utils.py            (warmup call in sharded_load)
#
# The generate.py patch is version-sensitive. This script patches in-place
# using sed for the specific changes needed (adding _pipeline_sync).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

find_site_packages() {
    local host=$1
    if [ "$host" = "local" ]; then
        python3 -c "import mlx_lm; from pathlib import Path; print(Path(mlx_lm.__file__).parent)" 2>/dev/null
    else
        ssh -o ConnectTimeout=5 "$host" "python3 -c 'import mlx_lm; from pathlib import Path; print(Path(mlx_lm.__file__).parent)'" 2>/dev/null
    fi
}

deploy_to() {
    local host=$1
    local site
    if [ "$host" = "local" ]; then
        site=$(find_site_packages local)
        echo "Deploying to local: $site"
        cp "$SCRIPT_DIR/pipeline.py" "$site/models/pipeline.py"
        cp "$SCRIPT_DIR/qwen3_moe.py" "$site/models/qwen3_moe.py"
        # Patch generate.py: add _pipeline_sync if not present
        if ! grep -q '_pipeline_sync' "$site/generate.py"; then
            patch_generate "$site/generate.py"
        else
            echo "  generate.py: already patched"
        fi
        # Patch utils.py: add warmup call if not present
        if ! grep -q 'pipeline_warmup' "$site/utils.py"; then
            patch_utils "$site/utils.py"
        else
            echo "  utils.py: already patched"
        fi
        # Install psutil if missing
        python3 -c "import psutil" 2>/dev/null || pip3 install psutil --break-system-packages -q
    else
        site=$(find_site_packages "$host")
        echo "Deploying to $host: $site"
        scp -q "$SCRIPT_DIR/pipeline.py" "$host:$site/models/pipeline.py"
        scp -q "$SCRIPT_DIR/qwen3_moe.py" "$host:$site/models/qwen3_moe.py"
        # Check and patch generate.py
        if ! ssh "$host" "grep -q '_pipeline_sync' '$site/generate.py'" 2>/dev/null; then
            ssh "$host" "cat '$site/generate.py'" > /tmp/_gen_patch.py
            patch_generate /tmp/_gen_patch.py
            scp -q /tmp/_gen_patch.py "$host:$site/generate.py"
            rm /tmp/_gen_patch.py
        else
            echo "  generate.py: already patched"
        fi
        # Check and patch utils.py
        if ! ssh "$host" "grep -q 'pipeline_warmup' '$site/utils.py'" 2>/dev/null; then
            ssh "$host" "cat '$site/utils.py'" > /tmp/_utils_patch.py
            patch_utils /tmp/_utils_patch.py
            scp -q /tmp/_utils_patch.py "$host:$site/utils.py"
            rm /tmp/_utils_patch.py
        else
            echo "  utils.py: already patched"
        fi
        # Install psutil if missing
        ssh "$host" "python3 -c 'import psutil' 2>/dev/null || pip3 install psutil --break-system-packages -q" 2>/dev/null
    fi
    echo "  Done."
}

patch_generate() {
    local file=$1
    python3 << PYEOF
import re

with open("$file") as f:
    content = f.read()

# Add _pipeline_sync function after DEFAULT_PROMPT line
sync_fn = '''def _pipeline_sync():
    """Return a collective op that forces all pipeline ranks to eval together."""
    group = mx.distributed.init()
    if group.size() > 1:
        return mx.distributed.all_sum(mx.zeros(1))
    return None

'''
if '_pipeline_sync' not in content:
    content = content.replace(
        'DEFAULT_PROMPT = "hello"',
        sync_fn + 'DEFAULT_PROMPT = "hello"'
    )

    # Patch prefill: eval logits+sync instead of just cache
    content = re.sub(
        r'(\s+)(_model_call\()',
        r'\1_prefill_logits = _model_call(',
        content,
        count=1
    )
    # This is fragile — may need manual verification per version
    content = content.replace(
        "            mx.eval([c.state for c in prompt_cache])\n            prompt_processed_tokens",
        "            _sync = _pipeline_sync()\n"
        "            if _sync is not None:\n"
        "                mx.eval(_prefill_logits, _sync)\n"
        "            else:\n"
        "                mx.eval([c.state for c in prompt_cache])\n"
        "            prompt_processed_tokens"
    )

    # Patch first token eval
    content = content.replace(
        "    mx.async_eval(y, logprobs)\n    n = 0",
        "    _sync = _pipeline_sync()\n"
        "    if _sync is not None:\n"
        "        mx.eval(y, logprobs, _sync)\n"
        "    else:\n"
        "        mx.async_eval(y, logprobs)\n"
        "    n = 0"
    )

    # Patch generation loop
    content = content.replace(
        "            next_y, next_logprobs = _step(y)\n"
        "            mx.async_eval(next_y, next_logprobs)",
        "            next_y, next_logprobs = _step(y)\n"
        "            if _sync is not None:\n"
        "                _sync = _pipeline_sync()\n"
        "                mx.eval(next_y, next_logprobs, _sync)\n"
        "            else:\n"
        "                mx.async_eval(next_y, next_logprobs)"
    )

with open("$file", "w") as f:
    f.write(content)
print("  generate.py: patched")
PYEOF
}

patch_utils() {
    local file=$1
    python3 << PYEOF
with open("$file") as f:
    content = f.read()

# Add warmup call after the all_sum sync in sharded_load
old = "    # Synchronize processes to avoid timeout\n    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))"
new = old + """

    # Warm up Metal shaders for pipeline models to avoid GPU timeout
    # on the first forward pass (large asymmetric splits compile too
    # many shaders in one command buffer otherwise).
    if pipeline_group is not None and hasattr(model.model, \"pipeline_warmup\"):
        import os
        if os.environ.get('MLX_SKIP_WARMUP') != '1':
            model.model.pipeline_warmup()"""

if 'pipeline_warmup' not in content:
    content = content.replace(old, new)

with open("$file", "w") as f:
    f.write(content)
print("  utils.py: patched")
PYEOF
}

# Main
if [ $# -eq 0 ]; then
    deploy_to local
else
    deploy_to local
    for node in "$@"; do
        deploy_to "$node"
    done
fi

echo "All nodes deployed. Restart any running mlx_lm servers."
