#!/bin/bash
# start-cluster.sh — Start mlx_lm distributed inference on TB5 JACCL cluster
# Usage: start-cluster.sh [model-path] [--nodes 2|3]
#
# Examples:
#   start-cluster.sh                                    # A22B 8-bit, auto-detect nodes
#   start-cluster.sh ~/.exo/models/huggingface/mlx-community--Qwen3-Coder-Next-bf16
#   start-cluster.sh --nodes 2                          # Force 2-node (tensor parallel)

set -e

# Defaults
MODEL="/Users/thomas/.exo/models/huggingface/mlx-community--Qwen3-235B-A22B-Instruct-2507-8bit"
NODES=""
PORT=8888
PYTHON=/Users/thomas/exo-src/.venv/bin/python3
MLX_LAUNCH=/opt/homebrew/bin/mlx.launch
CONFIG_DIR=/Users/thomas/.config/mlx

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes) NODES="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --*) echo "Unknown option: $1"; exit 1 ;;
        *) MODEL="$1"; shift ;;
    esac
done

# Guard: skip if already running
if curl -s --max-time 3 http://localhost:$PORT/v1/models >/dev/null 2>&1; then
    echo "Server already running on port $PORT"
    exit 0
fi

# Detect available nodes
echo "Checking nodes..."
VADER_UP=false; VOLDEMORT_UP=false; GARGAMEL_UP=false
ssh -o ConnectTimeout=3 vader "uptime" >/dev/null 2>&1 && VADER_UP=true
ssh -o ConnectTimeout=3 voldemort "uptime" >/dev/null 2>&1 && VOLDEMORT_UP=true
ssh -o ConnectTimeout=3 gargamel "uptime" >/dev/null 2>&1 && GARGAMEL_UP=true

if [ -z "$NODES" ]; then
    if $VADER_UP && $VOLDEMORT_UP && $GARGAMEL_UP; then
        NODES=3
    elif $VADER_UP && $VOLDEMORT_UP; then
        NODES=2
    elif $VADER_UP; then
        NODES=1
    else
        echo "ERROR: No nodes available"
        exit 1
    fi
fi

echo "Nodes: $NODES"

# Assign TB IPs
echo "Setting up TB networking..."
if $VADER_UP; then
    ssh vader "sudo ifconfig en3 10.0.1.1/24 up 2>/dev/null; sudo ifconfig en4 10.0.2.1/24 up 2>/dev/null" &
fi
if $VOLDEMORT_UP; then
    ssh voldemort "sudo ifconfig en4 10.0.1.2/24 up 2>/dev/null; sudo ifconfig en3 10.0.3.1/24 up 2>/dev/null" &
fi
if $GARGAMEL_UP; then
    ssh gargamel "sudo ifconfig en4 10.0.2.2/24 up 2>/dev/null; sudo ifconfig en3 10.0.3.2/24 up 2>/dev/null" &
fi
wait
echo "TB IPs assigned"

# Select hostfile and parallel mode
if [ "$NODES" = "3" ]; then
    HOSTFILE="$CONFIG_DIR/jaccl-hosts.json"
    PARALLEL="--pipeline"
    echo "Mode: 3-node pipeline parallel"
elif [ "$NODES" = "2" ]; then
    HOSTFILE="$CONFIG_DIR/jaccl-hosts-2node.json"
    PARALLEL=""  # tensor parallel (default) for 2-node
    echo "Mode: 2-node tensor parallel"
else
    # Single node - just run mlx_lm.server directly
    echo "Mode: single node"
    exec $PYTHON -m mlx_lm server \
        --model "$MODEL" \
        --host 0.0.0.0 --port $PORT \
        --trust-remote-code --decode-concurrency 1
fi

# Launch distributed
echo "Starting server on port $PORT..."
echo "Model: $(basename $MODEL)"

# Start server in background, then warmup
$MLX_LAUNCH \
    --hostfile "$HOSTFILE" \
    --backend jaccl \
    --python "$PYTHON" \
    -- "$CONFIG_DIR/server-wrapper.py" \
    --model "$MODEL" \
    --host 0.0.0.0 --port $PORT \
    --trust-remote-code --decode-concurrency 1 \
    $PARALLEL &
SERVER_PID=$!

# Wait for server to be ready, then warmup
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s --max-time 5 http://localhost:$PORT/v1/models >/dev/null 2>&1; then
        echo "Server ready, running warmup..."
        curl -s --max-time 60 http://localhost:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"default_model","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}' >/dev/null 2>&1
        echo "Warmup complete — server ready for requests"
        break
    fi
    sleep 5
done

# Keep running
wait $SERVER_PID
