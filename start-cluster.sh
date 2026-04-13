#!/bin/bash
# start-cluster.sh — Start mlx_lm distributed inference on TB5 JACCL cluster
# Run from node 0 (largest node). Usage:
#   start-cluster.sh                          # Default model, auto-detect nodes
#   start-cluster.sh /path/to/model           # Custom model
#   start-cluster.sh --nodes 2                # Force 2-node
#   start-cluster.sh --nodes 1                # Single-node (small models)
#   start-cluster.sh --stop                   # Kill running server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="${CLUSTER_CONF:-$SCRIPT_DIR/cluster.conf}"

if [ ! -f "$CONF" ]; then
    echo "ERROR: Config not found: $CONF"
    echo "Copy cluster.conf.example to cluster.conf and edit for your hardware."
    exit 1
fi
source "$CONF"

MODEL="${DEFAULT_MODEL}"
NODES=""
CONFIG_DIR="$HOME/.config/mlx"
LOG=/tmp/mlx-server.log

# Derived values
NODES_SSH=("$NODE_0" "$NODE_1" "$NODE_2")

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes) NODES="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --stop)
            echo "Stopping server..."
            pkill -f "mlx_lm.server\|server-wrapper" 2>/dev/null
            ssh "${NODES_SSH[1]}" "pkill -f 'mlx_lm.server\|server-wrapper'" 2>/dev/null
            ssh "${NODES_SSH[2]}" "pkill -f 'mlx_lm.server\|server-wrapper'" 2>/dev/null
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
NODE1_UP=false; NODE2_UP=false
ssh -o ConnectTimeout=3 "${NODES_SSH[1]}" "true" 2>/dev/null && NODE1_UP=true
ssh -o ConnectTimeout=3 "${NODES_SSH[2]}" "true" 2>/dev/null && NODE2_UP=true

if [ -z "$NODES" ]; then
    if $NODE1_UP && $NODE2_UP; then NODES=3
    elif $NODE1_UP; then NODES=2
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

NODES_LIST="local"
[ "$NODES" -ge 2 ] && NODES_LIST="$NODES_LIST ${NODES_SSH[1]}"
[ "$NODES" -ge 3 ] && NODES_LIST="$NODES_LIST ${NODES_SSH[2]}"

for node in $NODES_LIST; do
    if [ "$node" = "local" ]; then
        for f in pipeline.py qwen3_moe.py minimax.py; do
            [ -f "$PATCHES/$f" ] && cp "$PATCHES/$f" "$SITE/models/$f"
        done
        for patcher in patch_generate.py patch_server.py patch_utils.py; do
            [ -f "$PATCHES/$patcher" ] && $PYTHON "$PATCHES/$patcher" "$SITE/$(echo $patcher | sed 's/patch_//')"
        done
        $PYTHON -c 'import psutil' 2>/dev/null || $PYTHON -m pip install psutil -q 2>/dev/null || true
        echo "  local: patched" | tee -a "$LOG"
    else
        for f in pipeline.py qwen3_moe.py minimax.py; do
            [ -f "$PATCHES/$f" ] && scp -q "$PATCHES/$f" "$node:$SITE/models/$f"
        done
        for patcher in patch_generate.py patch_server.py patch_utils.py; do
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
ssh "${NODES_SSH[1]}" "pkill -f 'mlx_lm.server\|server-wrapper\|test_.*node'" 2>/dev/null || true
if [ "$NODES" = "3" ]; then
    ssh "${NODES_SSH[2]}" "pkill -f 'mlx_lm.server\|server-wrapper\|test_.*node'" 2>/dev/null || true
fi
sleep 2

# ── Bounce RDMA ──
echo "Bouncing RDMA interfaces..." | tee -a "$LOG"

# Down all TB interfaces on participating nodes
sudo ifconfig "$NODE_0_IF_TO_1" down 2>/dev/null; sudo ifconfig "$NODE_0_IF_TO_2" down 2>/dev/null
ssh "${NODES_SSH[1]}" "sudo ifconfig $NODE_1_IF_TO_0 down 2>/dev/null; sudo ifconfig $NODE_1_IF_TO_2 down 2>/dev/null" 2>/dev/null
if [ "$NODES" = "3" ]; then
    ssh "${NODES_SSH[2]}" "sudo ifconfig $NODE_2_IF_TO_0 down 2>/dev/null; sudo ifconfig $NODE_2_IF_TO_1 down 2>/dev/null" 2>/dev/null
fi
sleep 15

# Up with correct IPs (only interfaces needed for this config)
sudo ifconfig "$NODE_0_IF_TO_1" "$NODE_0_IP_TO_1/24" up
ssh "${NODES_SSH[1]}" "sudo ifconfig $NODE_1_IF_TO_0 $NODE_1_IP_TO_0/24 up"
if [ "$NODES" = "3" ]; then
    sudo ifconfig "$NODE_0_IF_TO_2" "$NODE_0_IP_TO_2/24" up
    ssh "${NODES_SSH[1]}" "sudo ifconfig $NODE_1_IF_TO_2 $NODE_1_IP_TO_2/24 up"
    ssh "${NODES_SSH[2]}" "sudo ifconfig $NODE_2_IF_TO_0 $NODE_2_IP_TO_0/24 up; sudo ifconfig $NODE_2_IF_TO_1 $NODE_2_IP_TO_1/24 up"
fi
sleep 5

# Verify RDMA
echo "Verifying RDMA..." | tee -a "$LOG"
RDMA_OK=true

check_rdma() {
    local node=$1 dev=$2
    local state
    if [ "$node" = "$NODE_0" ]; then
        state=$(timeout 10 ibv_devinfo -d "$dev" 2>/dev/null | grep -oE 'PORT_ACTIVE|PORT_DOWN')
    else
        state=$(ssh -o ConnectTimeout=5 "$node" "timeout 10 ibv_devinfo -d $dev 2>/dev/null | grep -oE 'PORT_ACTIVE|PORT_DOWN'" 2>/dev/null)
    fi
    echo "  $node:$dev = ${state:-TIMEOUT}" | tee -a "$LOG"
    [ "$state" != "PORT_ACTIVE" ] && RDMA_OK=false
}

# node0 ↔ node1 (always needed)
check_rdma "$NODE_0" "rdma_${NODE_0_IF_TO_1}"
check_rdma "$NODE_1" "rdma_${NODE_1_IF_TO_0}"
# Additional links for 3-node
if [ "$NODES" = "3" ]; then
    check_rdma "$NODE_0" "rdma_${NODE_0_IF_TO_2}"
    check_rdma "$NODE_1" "rdma_${NODE_1_IF_TO_2}"
    check_rdma "$NODE_2" "rdma_${NODE_2_IF_TO_0}"
    check_rdma "$NODE_2" "rdma_${NODE_2_IF_TO_1}"
fi

if ! $RDMA_OK; then
    echo "ERROR: RDMA not fully active. Aborting." | tee -a "$LOG"
    exit 1
fi

# ── Deploy device matrix for 3-node ──
if [ "$NODES" = "3" ]; then
    HOSTFILE="$HOSTFILE_3NODE"
    # Build device matrix from config: matrix[i][j] = rdma device on node i for connection to node j
    MATRIX="[[null,\"rdma_${NODE_0_IF_TO_1}\",\"rdma_${NODE_0_IF_TO_2}\"],[\"rdma_${NODE_1_IF_TO_0}\",null,\"rdma_${NODE_1_IF_TO_2}\"],[\"rdma_${NODE_2_IF_TO_0}\",\"rdma_${NODE_2_IF_TO_1}\",null]]"
    echo "$MATRIX" > /tmp/jaccl_3node.json
    scp -q /tmp/jaccl_3node.json "${NODES_SSH[1]}:/tmp/jaccl_3node.json"
    scp -q /tmp/jaccl_3node.json "${NODES_SSH[2]}:/tmp/jaccl_3node.json"
    echo "3-node device matrix deployed" | tee -a "$LOG"
else
    HOSTFILE="$HOSTFILE_2NODE"
fi

# ── Stop Ollama on remote nodes to free GPU memory ──
ssh "${NODES_SSH[1]}" "launchctl bootout gui/501/homebrew.mxcl.ollama 2>/dev/null; pkill -f 'ollama' 2>/dev/null" 2>/dev/null || true
if [ "$NODES" = "3" ]; then
    ssh "${NODES_SSH[2]}" "launchctl bootout gui/501/homebrew.mxcl.ollama 2>/dev/null; pkill -f 'ollama' 2>/dev/null" 2>/dev/null || true
fi
echo "Ollama stopped on remote nodes" | tee -a "$LOG"

# ── Launch server ──
echo "Launching mlx_lm.server (${NODES}-node, port $PORT)..." | tee -a "$LOG"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $MLX_LAUNCH \
    --hostfile "$HOSTFILE" \
    --backend jaccl \
    --python "$PYTHON" \
    -- "$SERVER_WRAPPER" \
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
