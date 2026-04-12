# MLX Pipeline Parallel Setup Guide

Distributed pipeline inference for large MoE models on Apple Silicon over Thunderbolt 5 RDMA (JACCL).

## Tested Configuration (2026-04-11)

| Component | Version |
|-----------|---------|
| mlx-lm | 0.31.1 / 0.31.2 (both tested) |
| MLX | 0.31.1.dev20260328+ce45c52 (custom GID fix) |
| Python | 3.14.3 (homebrew) |
| macOS | 26.4 |
| Model | Qwen3-235B-A22B-Instruct-2507-8bit |

### Nodes

| Node | Chip | RAM | Layers (3-node) | Layers (2-node) |
|------|------|-----|-----------------|-----------------|
| Vader (Mac Studio) | M3 Ultra | 256 GB | 70 | 80 |
| Voldemort (Mac mini) | M4 Pro | 64 GB | 14 | 14 |
| Gargamel (Mac mini) | M4 Pro | 48 GB | 10 | — |

### Performance

| Config | Split | Startup | Prefill | Generation |
|--------|-------|---------|---------|------------|
| 3-node | 70/14/10 | ~60s | 0.4s | 11.3 tok/s |
| 2-node | 80/14 | ~60s | 0.35s | 17.8 tok/s |
| 1-node | — | ~5s | — | model-dependent |

## Quick Start

From vader:

```bash
# Start 3-node server (auto-detects available nodes)
bash ~/.config/mlx/start-cluster.sh

# Force 2-node
bash ~/.config/mlx/start-cluster.sh --nodes 2

# Single-node (for models that fit on vader alone)
bash ~/.config/mlx/start-cluster.sh --nodes 1

# Custom model
bash ~/.config/mlx/start-cluster.sh /path/to/model

# Custom model + node count
bash ~/.config/mlx/start-cluster.sh --nodes 1 /path/to/small-model

# Stop server
bash ~/.config/mlx/start-cluster.sh --stop
```

The script handles everything: patch deployment, RDMA bounce, device matrix, server launch, warmup, and health check.

## What start-cluster.sh Does

1. **Detects nodes** — SSHs to voldemort and gargamel, picks 1, 2, or 3 node config
   - **Single-node**: Restores stock mlx_lm (pipeline patches break BatchGenerator), runs server directly
   - **Multi-node**: Continues with steps 2-9 below
2. **Deploys patches** — Copies ALL files from `~/tb5-jaccl-toolkit/patches/mlx-pipeline-qwen3/` to `mlx_lm` site-packages on every node. Patches `utils.py` in-place for warmup call.
3. **Kills stale processes** — Any leftover mlx_lm/server-wrapper processes
4. **Bounces RDMA** — Down all TB interfaces, 15s cooldown, up with correct IPs, 5s settle
5. **Verifies RDMA** — Checks PORT_ACTIVE on all links, aborts if any are down
6. **Deploys device matrix** — `/tmp/jaccl_3node.json` on all nodes (for 3-node)
7. **Stops Ollama** — Frees GPU memory on minis
8. **Launches server** — `mlx.launch` with JACCL backend, `--pipeline` flag, server-wrapper.py
9. **Health check** — Waits for server, sends "What is 2+2?", verifies answer contains "4" or "four"

## Patch Files

All in `patches/mlx-pipeline-qwen3/`:

| File | Target | What it does |
|------|--------|-------------|
| `pipeline.py` | `models/pipeline.py` | Proportional memory split + `pipeline_warmup()` for Metal shader pre-compilation |
| `qwen3_moe.py` | `models/qwen3_moe.py` | Adds PipelineMixin (recv/send/all_gather in forward pass) + `make_cache()` |
| `minimax.py` | `models/minimax.py` | Same pipeline support for MiniMax models |
| `generate.py` | `generate.py` | `_pipeline_sync()` in prefill and generation loops — forces all ranks to eval together |
| `server.py` | `server.py` | **Critical:** `is_batchable = False` for pipeline models. Without this, server uses BatchGenerator which has no pipeline sync → GPU timeout |
| `tokenizer_utils.py` | `tokenizer_utils.py` | Tokenizer fixes |
| `patch_utils.py` | (helper) | Patches `utils.py` in-place to call `pipeline_warmup()` in `sharded_load()` |
| `deploy.sh` | (helper) | Standalone deployment script for manual use |

### Why server.py is Critical

The stock `mlx_lm.server` checks `is_batchable` and routes to `BatchGenerator` for inference. `BatchGenerator` has its own eval loop that doesn't include `_pipeline_sync()`. With pipeline models, this causes the fast rank to timeout waiting for the slow rank during collective ops.

The patch adds:
```python
if self.pipeline_group is not None:
    self.is_batchable = False
```

This forces the server to use the single-request `stream_generate` → `generate_step` path which has our pipeline sync patches.

### Version Compatibility Note

The toolkit's `server.py` is based on 0.31.1. Stock 0.31.2's `server.py` has a `rfind_think_start` bug with Qwen3 tokenizers and is NOT used. The toolkit's `generate.py` is stock 0.31.2 with `_pipeline_sync` patches applied in-place — this preserves `BatchGenerator` compatibility for single-node mode.

Single-node mode restores stock mlx_lm via `pip install --force-reinstall` before launching, because the toolkit's 0.31.1 `server.py` calls `BatchGenerator` with arguments that don't exist in 0.31.2's `generate.py`.

### Why pipeline_warmup() is Needed

The first cold `model()` call with 70+ MoE layers builds a massive Metal compute graph. Shader compilation for all layers in one command buffer exceeds Metal's timeout. `pipeline_warmup()` compiles shaders incrementally:

1. **Phase 1:** Run each rank's layers locally in chunks of 5 (no distributed ops, no cross-rank dependencies)
2. **Phase 2:** Run full pipeline forward pass (recv → layers → send → all_gather). Succeeds because all shaders are pre-compiled.

## Thunderbolt Interface Map

```
Vader en3=10.0.1.1  ↔  Voldemort en4=10.0.1.2
Vader en4=10.0.2.1  ↔  Gargamel en4=10.0.2.2
Voldemort en3=10.0.3.1  ↔  Gargamel en3=10.0.3.2
```

## Hostfiles (on vader ~/.config/mlx/)

- `jaccl-hosts.json` — 3-node with coordinator and IBV_DEVICES env
- `jaccl-hosts-2node.json` — 2-node vader+voldemort
- `jaccl-hosts-vader-vol.json` — 2-node (simpler format, also works)

## RDMA Device Matrix (3-node)

```json
[[null,"rdma_en3","rdma_en4"],["rdma_en4",null,"rdma_en3"],["rdma_en4","rdma_en3",null]]
```

Matrix[i][j] = RDMA device on node i for connection to node j.

## RDMA Bounce

Required after any JACCL crash — stale queue pairs prevent reconnection.

```bash
# Down
sudo ifconfig en3 down; sudo ifconfig en4 down  # on each node

# Wait 15 seconds

# Up with correct IPs
sudo ifconfig en3 10.0.1.1/24 up  # vader example
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| GPU Timeout on first inference | Missing `server.py` patch (BatchGenerator path) | Deploy toolkit's `server.py` — `is_batchable = False` |
| GPU Timeout on cold start | Missing warmup | Deploy `pipeline.py` with `pipeline_warmup()` + `patch_utils.py` |
| JACCL connection error 60 | RDMA stale or IPs wrong | Bounce interfaces with 15s cooldown |
| `tensor_group was provided` error | Missing `--pipeline` flag | Add `--pipeline` to server args |
| Python binary as script error | Wrong `--python` path or double python in command | Use `-- script.py` not `-- python script.py` |
| OOM on mini | Equal split instead of proportional | Install `psutil` on all nodes |
| 3-node won't connect | Missing device matrix | Deploy `/tmp/jaccl_3node.json` to all nodes |

## GitHub PRs

- [ml-explore/mlx-lm#1137](https://github.com/ml-explore/mlx-lm/pull/1137) — Pipeline infrastructure (pipeline.py, generate.py)
- [ml-explore/mlx-lm#1138](https://github.com/ml-explore/mlx-lm/pull/1138) — Model support (qwen3_moe.py, minimax.py)
- [ml-explore/mlx#3389](https://github.com/ml-explore/mlx/pull/3389) — JACCL GID fix for 3-node
