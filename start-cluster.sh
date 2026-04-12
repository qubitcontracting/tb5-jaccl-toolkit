#!/bin/bash
# start-cluster.sh — Start mlx_lm distributed inference on TB5 JACCL cluster
# Run from vader. Usage:
#   start-cluster.sh                          # A22B 8-bit, auto-detect 2 or 3 nodes
#   start-cluster.sh /path/to/model           # Custom model
#   start-cluster.sh --nodes 2                # Force 2-node
#   start-cluster.sh --stop                   # Kill running server
set -e

MODEL="$HOME/.exo/models/huggingface/mlx-community--Qwen3-235B-A22B-Instruct-2507-8bit"
NODES=""
PORT=8888
PYTHON=/opt/homebrew/bin/python3
MLX_LAUNCH=/opt/homebrew/bin/mlx.launch
CONFIG_DIR="$HOME/.config/mlx"
TOOLKIT="$HOME/tb5-jaccl-toolkit"
LOG=/tmp/mlx-server.log

# TB interface map (from README):
#   vader:en3  ↔  voldemort:en4   10.0.1.1 ↔ 10.0.1.2
#   vader:en4  ↔  gargamel:en4    10.0.2.1 ↔ 10.0.2.2
#   voldemort:en3 ↔ gargamel:en3  10.0.3.1 ↔ 10.0.3.2

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes) NODES="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --stop)
            echo "Stopping server..."
            pkill -f "mlx_lm.server\|server-wrapper" 2>/dev/null
            ssh voldemort "pkill -f 'mlx_lm.server\|server-wrapper'" 2>/dev/null
            ssh gargamel "pkill -f 'mlx_lm.server\|server-wrapper'" 2>/dev/null
            echo "Stopped."
            exit 0 ;;
        --*) echo "Unknown: $1"; exit 1 ;;
        *)  MODEL="$1"; shift ;;
    esac
done

# Guard: skip if already running
if curl -s --max-time 3 http://localhost:$PORT/v1/models >/dev/null 2>&1; then
    echo "Server already running on port $PORT"
    exit 0
fi

echo "[$(date)] Starting cluster..." | tee -a "$LOG"
echo "Model: $(basename "$MODEL")" | tee -a "$LOG"

# ── Detect nodes ──
VOLDEMORT_UP=false; GARGAMEL_UP=false
ssh -o ConnectTimeout=3 voldemort "true" 2>/dev/null && VOLDEMORT_UP=true
ssh -o ConnectTimeout=3 gargamel "true" 2>/dev/null && GARGAMEL_UP=true

if [ -z "$NODES" ]; then
    if $VOLDEMORT_UP && $GARGAMEL_UP; then NODES=3
    elif $VOLDEMORT_UP; then NODES=2
    else NODES=1
    fi
fi
echo "Nodes: $NODES" | tee -a "$LOG"

if [ "$NODES" = "1" ]; then
    echo "Single-node mode." | tee -a "$LOG"
    exec $PYTHON -m mlx_lm.server --model "$MODEL" --host 0.0.0.0 --port $PORT
fi

# ── Deploy patches ──
PATCHES="$TOOLKIT/patches/mlx-pipeline-qwen3"
SITE=$($PYTHON -c "import mlx_lm; from pathlib import Path; print(Path(mlx_lm.__file__).parent)")
if [ -z "$SITE" ]; then
    echo "ERROR: Cannot find mlx_lm site-packages" | tee -a "$LOG"
    exit 1
fi
echo "Patching mlx_lm at $SITE..." | tee -a "$LOG"

# Copy ALL patched files to all nodes
NODES_LIST="local"
[ "$NODES" -ge 2 ] && NODES_LIST="$NODES_LIST voldemort"
[ "$NODES" -ge 3 ] && NODES_LIST="$NODES_LIST gargamel"

for node in $NODES_LIST; do
    if [ "$node" = "local" ]; then
        # Full-replace patches (version-independent)
        for f in pipeline.py qwen3_moe.py minimax.py; do
            [ -f "$PATCHES/$f" ] && cp "$PATCHES/$f" "$SITE/models/$f"
        done
        for f in generate.py tokenizer_utils.py; do
            [ -f "$PATCHES/$f" ] && cp "$PATCHES/$f" "$SITE/$f"
        done
        # In-place patches (version-independent)
        [ -f "$PATCHES/patch_server.py" ] && $PYTHON "$PATCHES/patch_server.py" "$SITE/server.py"
        [ -f "$PATCHES/patch_utils.py" ] && $PYTHON "$PATCHES/patch_utils.py" "$SITE/utils.py"
        $PYTHON -c 'import psutil' 2>/dev/null || $PYTHON -m pip install psutil -q 2>/dev/null || true
        echo "  local: patched" | tee -a "$LOG"
    else
        # Full-replace patches
        for f in pipeline.py qwen3_moe.py minimax.py; do
            [ -f "$PATCHES/$f" ] && scp -q "$PATCHES/$f" "$node:$SITE/models/$f"
        done
        for f in generate.py tokenizer_utils.py; do
            [ -f "$PATCHES/$f" ] && scp -q "$PATCHES/$f" "$node:$SITE/$f"
        done
        # In-place patches
        for patcher in patch_server.py patch_utils.py; do
            if [ -f "$PATCHES/$patcher" ]; then
                scp -q "$PATCHES/$patcher" "$node:/tmp/$patcher"
                ssh "$node" "$PYTHON /tmp/$patcher '$SITE/$(echo $patcher | sed s/patch_//)'" 2>/dev/null
            fi
        done
        ssh "$node" "$PYTHON -c 'import psutil' 2>/dev/null || $PYTHON -m pip install psutil -q" 2>/dev/null || true
        echo "  $node: patched" | tee -a "$LOG"
    fi
done

# ── Kill stale processes ──
echo "Cleaning up stale processes..." | tee -a "$LOG"
pkill -f "mlx_lm.server\|server-wrapper\|test_.*node" 2>/dev/null || true
ssh voldemort "pkill -f 'mlx_lm.server\|server-wrapper\|test_.*node'" 2>/dev/null || true
if [ "$NODES" = "3" ]; then
    ssh gargamel "pkill -f 'mlx_lm.server\|server-wrapper\|test_.*node'" 2>/dev/null || true
fi
sleep 2

# ── Bounce RDMA ──
echo "Bouncing RDMA interfaces..." | tee -a "$LOG"
# Down
sudo ifconfig en3 down 2>/dev/null; sudo ifconfig en4 down 2>/dev/null
ssh voldemort "sudo ifconfig en3 down 2>/dev/null; sudo ifconfig en4 down 2>/dev/null" 2>/dev/null
if [ "$NODES" = "3" ]; then
    ssh gargamel "sudo ifconfig en3 down 2>/dev/null; sudo ifconfig en4 down 2>/dev/null" 2>/dev/null
fi
sleep 15

# Up with correct IPs (only bring up interfaces needed for this config)
sudo ifconfig en3 10.0.1.1/24 up  # vader↔voldemort (always)
ssh voldemort "sudo ifconfig en4 10.0.1.2/24 up"  # voldemort↔vader (always)
if [ "$NODES" = "3" ]; then
    sudo ifconfig en4 10.0.2.1/24 up  # vader↔gargamel
    ssh voldemort "sudo ifconfig en3 10.0.3.1/24 up"  # voldemort↔gargamel
    ssh gargamel "sudo ifconfig en4 10.0.2.2/24 up; sudo ifconfig en3 10.0.3.2/24 up"
fi
sleep 5

# Verify RDMA (only check interfaces needed for this config)
echo "Verifying RDMA..." | tee -a "$LOG"
RDMA_OK=true
# vader:en3 ↔ voldemort:en4 (always needed for 2+ nodes)
for pair in "vader rdma_en3" "voldemort rdma_en4"; do
    node=$(echo $pair | cut -d' ' -f1)
    dev=$(echo $pair | cut -d' ' -f2)
    if [ "$node" = "vader" ]; then
        state=$(ibv_devinfo -d $dev 2>/dev/null | grep -oE 'PORT_ACTIVE|PORT_DOWN')
    else
        state=$(ssh $node "ibv_devinfo -d $dev 2>/dev/null | grep -oE 'PORT_ACTIVE|PORT_DOWN'" 2>/dev/null)
    fi
    echo "  $node:$dev = $state" | tee -a "$LOG"
    [ "$state" != "PORT_ACTIVE" ] && RDMA_OK=false
done
# Additional links for 3-node
if [ "$NODES" = "3" ]; then
for pair in "vader rdma_en4" "voldemort rdma_en3" "gargamel rdma_en3" "gargamel rdma_en4"; do
    node=$(echo $pair | cut -d' ' -f1)
    dev=$(echo $pair | cut -d' ' -f2)
    state=$(ssh $node "ibv_devinfo -d $dev 2>/dev/null | grep -oE 'PORT_ACTIVE|PORT_DOWN'" 2>/dev/null)
    echo "  $node:$dev = $state" | tee -a "$LOG"
    [ "$state" != "PORT_ACTIVE" ] && RDMA_OK=false
done
fi
if ! $RDMA_OK; then
    echo "ERROR: RDMA not fully active. Aborting." | tee -a "$LOG"
    exit 1
fi

# ── Deploy device matrix for 3-node ──
if [ "$NODES" = "3" ]; then
    HOSTFILE="$CONFIG_DIR/jaccl-hosts.json"
    # Device matrix: [vader→X, voldemort→X, gargamel→X]
    echo '[[null,"rdma_en3","rdma_en4"],["rdma_en4",null,"rdma_en3"],["rdma_en4","rdma_en3",null]]' > /tmp/jaccl_3node.json
    scp -q /tmp/jaccl_3node.json voldemort:/tmp/jaccl_3node.json
    scp -q /tmp/jaccl_3node.json gargamel:/tmp/jaccl_3node.json
    echo "3-node device matrix deployed" | tee -a "$LOG"
else
    HOSTFILE="$CONFIG_DIR/jaccl-hosts-2node.json"
fi

# ── Stop Ollama on minis to free GPU memory ──
ssh voldemort "launchctl bootout gui/501/homebrew.mxcl.ollama 2>/dev/null; pkill -f 'ollama' 2>/dev/null" 2>/dev/null || true
if [ "$NODES" = "3" ]; then
    ssh gargamel "launchctl bootout gui/501/homebrew.mxcl.ollama 2>/dev/null; pkill -f 'ollama' 2>/dev/null" 2>/dev/null || true
fi
echo "Ollama stopped on minis" | tee -a "$LOG"

# ── Launch server ──
echo "Launching mlx_lm.server (${NODES}-node, port $PORT)..." | tee -a "$LOG"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $MLX_LAUNCH \
    --hostfile "$HOSTFILE" \
    --backend jaccl \
    --python "$PYTHON" \
    -- ~/.config/mlx/server-wrapper.py \
    --model "$MODEL" \
    --host 0.0.0.0 --port $PORT \
    --trust-remote-code --pipeline \
    2>&1 | tee -a "$LOG" &
SERVER_PID=$!

# ── Wait for ready + health check ──
echo "Waiting for server to be ready..." | tee -a "$LOG"
READY=false
for i in $(seq 1 90); do
    if curl -s --max-time 5 http://localhost:$PORT/v1/models >/dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 2
done

if $READY; then
    echo "Server responding. Running health check..." | tee -a "$LOG"
    RESPONSE=$(curl -s --max-time 120 http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"default_model","messages":[{"role":"user","content":"What is 2+2? One word answer."}],"max_tokens":10}' 2>/dev/null)
    ANSWER=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
    echo "Health check response: $ANSWER" | tee -a "$LOG"
    if echo "$ANSWER" | grep -qi "4\|four"; then
        echo "✓ Server healthy and generating correct answers" | tee -a "$LOG"
    else
        echo "⚠ Server responded but answer unexpected: $ANSWER" | tee -a "$LOG"
    fi
else
    echo "ERROR: Server did not become ready in 180s" | tee -a "$LOG"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Keep running
wait $SERVER_PID
