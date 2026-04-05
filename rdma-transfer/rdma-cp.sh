#!/bin/bash
# rdma-cp: Copy files between cluster nodes over JACCL RDMA (~3.5 GB/s)
#
# Usage:
#   rdma-cp <source_path> <dest_node>:<dest_path>
#   rdma-cp vader:~/.exo/models/foo voldemort:~/.exo/models/
#
# Examples:
#   rdma-cp ~/.exo/models/mlx-community--Qwen3.5-397B-A17B-4bit voldemort:~/.exo/models/
#   rdma-cp /tmp/bigfile.bin gargamel:/tmp/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRANSFER_PY="$SCRIPT_DIR/transfer.py"

# Node rank mapping
declare -A NODE_RANK=(
    [vader]=0
    [voldemort]=1
    [gargamel]=2
)

declare -A NODE_IP=(
    [vader]="100.105.116.62"
    [voldemort]="100.121.90.87"
    [gargamel]="100.70.57.45"
)

COORDINATOR_IP="192.168.0.126"  # Vader LAN IP

usage() {
    echo "Usage: rdma-cp <source> <dest_node>:<dest_path>"
    echo ""
    echo "Run from the source node. Source is a local path, dest is node:path."
    echo ""
    echo "Examples:"
    echo "  rdma-cp ~/.exo/models/some-model voldemort:~/.exo/models/"
    echo "  rdma-cp /tmp/bigfile.bin gargamel:/tmp/"
    exit 1
}

if [ $# -ne 2 ]; then
    usage
fi

SOURCE="$1"
DEST="$2"

# Parse dest_node:dest_path
if [[ "$DEST" != *:* ]]; then
    echo "Error: Destination must be in node:path format"
    usage
fi

DEST_NODE="${DEST%%:*}"
DEST_PATH="${DEST#*:}"

# Resolve ~ in dest path
DEST_PATH="${DEST_PATH/#\~//Users/thomas}"

# Get local hostname
LOCAL_HOST=$(hostname -s | tr '[:upper:]' '[:lower:]')

if [ -z "${NODE_RANK[$LOCAL_HOST]}" ]; then
    echo "Error: Unknown local host '$LOCAL_HOST'. Must be vader, voldemort, or gargamel."
    exit 1
fi

if [ -z "${NODE_RANK[$DEST_NODE]}" ]; then
    echo "Error: Unknown dest node '$DEST_NODE'. Must be vader, voldemort, or gargamel."
    exit 1
fi

LOCAL_RANK=${NODE_RANK[$LOCAL_HOST]}
PEER_RANK=${NODE_RANK[$DEST_NODE]}
PEER_IP=${NODE_IP[$DEST_NODE]}

if [ "$LOCAL_RANK" -eq "$PEER_RANK" ]; then
    echo "Error: Source and destination are the same node"
    exit 1
fi

# Check source exists
if [ ! -e "$SOURCE" ]; then
    echo "Error: Source '$SOURCE' does not exist"
    exit 1
fi

# Check disk space on remote
SOURCE_SIZE=$(du -sk "$SOURCE" 2>/dev/null | awk '{print $1}')
REMOTE_AVAIL=$(ssh "$DEST_NODE" "df -k '$DEST_PATH' 2>/dev/null || df -k / 2>/dev/null" | tail -1 | awk '{print $4}')

if [ -n "$SOURCE_SIZE" ] && [ -n "$REMOTE_AVAIL" ]; then
    if [ "$SOURCE_SIZE" -gt "$REMOTE_AVAIL" ]; then
        echo "Error: Not enough space on $DEST_NODE"
        echo "  Need: $(echo "$SOURCE_SIZE" | awk '{printf "%.1f GB", $1/1048576}')"
        echo "  Available: $(echo "$REMOTE_AVAIL" | awk '{printf "%.1f GB", $1/1048576}')"
        exit 1
    fi
    echo "Space check OK: $(echo "$SOURCE_SIZE" | awk '{printf "%.1f GB", $1/1048576}') needed, $(echo "$REMOTE_AVAIL" | awk '{printf "%.1f GB", $1/1048576}') available"
fi

echo "Transferring: $SOURCE → $DEST_NODE:$DEST_PATH"
echo "  Local rank: $LOCAL_RANK ($LOCAL_HOST)"
echo "  Peer rank:  $PEER_RANK ($DEST_NODE)"
echo ""

# Start receiver on remote node
echo "Starting receiver on $DEST_NODE..."
ssh "$DEST_NODE" "cd '$SCRIPT_DIR' && python3 transfer.py --rank $PEER_RANK --peer-rank $LOCAL_RANK --coordinator $COORDINATOR_IP recv '$DEST_PATH'" &
RECV_PID=$!

# Brief pause to let receiver init
sleep 2

# Start sender locally
echo "Starting sender..."
python3 "$TRANSFER_PY" --rank "$LOCAL_RANK" --peer-rank "$PEER_RANK" --coordinator "$COORDINATOR_IP" send "$SOURCE"

# Wait for receiver
wait $RECV_PID
echo "Transfer complete!"
