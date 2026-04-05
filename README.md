# TB5 JACCL Toolkit

Tools and findings for running distributed LLM inference on Apple Silicon clusters using Thunderbolt 5 RDMA (JACCL).

## Hardware

| Node | Machine | Chip | RAM | TB5 Ports |
|------|---------|------|-----|-----------|
| Vader | Mac Studio | M3 Ultra | 256 GB | 6x TB5 |
| Voldemort | Mac mini | M4 Pro | 64 GB | 3x TB5 |
| Gargamel | Mac mini | M4 Pro | 48 GB | 3x TB5 |

Full mesh topology (3 Thunderbolt 5 cables), macOS 26.4, MLX 0.31.1, SIP disabled.

## Key Finding: bridge0 Coexistence

**JACCL works WITHOUT destroying bridge0.** Simply assign IP addresses to individual Thunderbolt interfaces — JACCL and bridge0 can coexist. This eliminates the need for `ifconfig bridge0 destroy` on every boot and avoids killing USB Ethernet adapters on Mac minis.

See [docs/bridge0-coexistence.md](docs/bridge0-coexistence.md) for full details. Also posted to [ml-explore/mlx#3207](https://github.com/ml-explore/mlx/issues/3207).

## Tools

### RDMA File Transfer (`rdma-transfer/`)

Fast file transfer between cluster nodes using JACCL RDMA. **3.3 GB/s** sustained — 13x faster than rsync over SSH.

```bash
# Transfer a model from vader to voldemort
./rdma-cp.sh ~/.exo/models/huggingface/mlx-community--SomeModel voldemort:~/.exo/models/huggingface/

# Or use transfer.py directly for more control
# Receiver:
python3 transfer.py --rank 1 --peer-rank 0 --coordinator 10.0.1.1 \
  --device-matrix /tmp/jaccl_2node.json recv /dest/path/

# Sender:
python3 transfer.py --rank 0 --peer-rank 1 --coordinator 0.0.0.0 \
  --device-matrix /tmp/jaccl_2node.json send /source/path/
```

Tested results:

| Route | Size | Time | Speed |
|-------|------|------|-------|
| Vader → Voldemort | 250 GB | 88s | 2.84 GB/s |
| Vader → Gargamel | 250 GB | 119s | 2.09 GB/s |
| 1 GB test file | 1.1 GB | 0.3s | 3.3 GB/s |

Requires MLX 0.31.1+ with JACCL support (use Exo's venv). Uses CPU stream to avoid Metal GPU timeouts.

### Agent Harness (`agent-harness/`)

Benchmark framework for testing LLMs on agentic coding tasks with tool use. Tests model ability to autonomously write code, run tests, and fix errors.

5 benchmark tasks:
- **CLI Tool with Subcommands** — argparse, JSON storage, tests
- **Static Site Generator** — Markdown parsing, HTML rendering, file watching
- **REST API with SQLite** — Flask/FastAPI, CRUD, database
- **Data Pipeline** — CSV processing, transforms, validation
- **A* Pathfinding** — algorithm implementation (no tools)

```bash
# Run single task
python3 run_all_tasks_model.py "mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit" "A22B" 10

# Run all tasks with fresh cluster restarts between each (recommended for thinking models)
./run_tasks_fresh.sh "mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit" "A22B"
```

### Benchmark Results

| Model | CLI Tool | SSG | REST | Data | A* | Avg | Backend |
|-------|---------|-----|------|------|----|-----|---------|
| Qwen3-235B-A22B 8-bit | **100** | 39 | 45 | — | — | — | Exo JACCL |
| Qwen3-Coder-Next bf16 | 75 | 0 | 55 | 55 | 10 | 39.0 | Exo JACCL |
| MiniMax-M2.5 8-bit | 80.6 | — | — | — | — | — | Exo JACCL |
| qwen2.5:32b | 55 | — | — | — | — | — | Ollama |

Note: Thinking models (A22B, Qwen3.5-397B) degrade after sustained multi-task sessions due to KV cache pressure. Restart the cluster between tasks for accurate benchmarks.

## Exo Patches (`patches/`)

Patches applied to [exo-explore/exo](https://github.com/exo-explore/exo) for 3-node JACCL support:

- **topology.py** — RDMA cycle detection fallback: assumes full mesh when all nodes have TB interfaces. Handles KeyError when nodes aren't in `_graph` yet.
- **system_info.py** — Classifies bridge0 as "thunderbolt" type instead of "unknown"
- **info_gatherer.py** — bridge0 fallback detection without network service
- **discovery.rs** — `EXO_PEERS` retry loop (every 5s), increased ping timeout to 30s

## Setup

### TB Interface IP Map

```
Vader en3=10.0.1.1  ↔  Voldemort en4=10.0.1.2
Vader en4=10.0.2.1  ↔  Gargamel en4=10.0.2.2
Voldemort en3=10.0.3.1  ↔  Gargamel en3=10.0.3.2
```

### JACCL Device Matrix (3-node)

```json
[[null, "rdma_en4", "rdma_en3"], ["rdma_en4", null, "rdma_en3"], ["rdma_en3", "rdma_en4", null]]
```

### Quick Start

1. Assign TB IPs on each node (or use `start-cluster.sh`)
2. Verify RDMA: `ibv_devinfo | grep -E 'hca_id|state:'` — should show `PORT_ACTIVE`
3. Start Exo on all nodes: `bash ~/exo-src/start-cluster.sh`
4. Place model: `curl -X POST http://vader:52415/place_instance -d '{"model_id":"...","min_nodes":3,"instance_meta":"MlxJaccl"}'`

## RDMA Throughput Benchmark

With bridge0 present and active on all nodes:

```
   Size    |   Time    | Throughput
     3.8 MB |    0.7 ms |  5,492 MB/s
    38.1 MB |    5.2 ms |  7,299 MB/s
   381.5 MB |   51.3 ms |  7,431 MB/s
```

7.4 GB/s sustained across 3 nodes with bridge0 intact.

## Known Issues

- **Thinking models degrade** after sustained inference sessions on Exo — KV cache pressure causes timeouts. Restart between heavy sessions.
- **Exo doesn't support** `gemma4`, `mimo_v2_flash`, or `afmoe` (Trinity) model types yet
- **mlx_lm server** has KV cache bugs with some MoE models (BatchRotatingKVCache shape mismatch)
- **Wired memory** from killed MLX processes may not free — SIGTERM usually works, SIGKILL often doesn't. Reboot always frees.
- **Mac Studio TB5 port warning**: Avoid the TB5 port adjacent to the Ethernet port on Mac Studio — reported issues with RDMA on that port

## License

MIT

## Attribution

Built with [Claude Code](https://claude.ai/claude-code) (Anthropic).
