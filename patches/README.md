# Exo Patches for 3-Node JACCL

These patches were applied to [exo-explore/exo](https://github.com/exo-explore/exo) to enable reliable 3-node JACCL RDMA inference. Some may have been merged upstream since.

## topology.py — RDMA Cycle Detection Fix

**Problem:** Exo's topology graph has 0 RDMA edges at placement time (timing issue). The `is_rdma_cycle` function raises `KeyError` when nodes aren't in `_graph` yet, causing placement to fail.

**Fix:**
1. Added `KeyError` handler in `is_rdma_cycle` that returns `True` (assumes full mesh) when all nodes have TB Bridge + TB IPs
2. Added RDMA edge logging in `replace_all_out_rdma_connections` for debugging
3. RDMA edges populate ~8 seconds after Exo starts — the fallback covers the gap

## system_info.py — bridge0 Classification

**Problem:** bridge0 was classified as "unknown" interface type, causing Exo to ignore it when detecting Thunderbolt connectivity.

**Fix:** Classify bridge0 as "thunderbolt" type so Exo correctly identifies nodes with Thunderbolt connections.

## info_gatherer.py — bridge0 Fallback Detection

**Problem:** bridge0 detection failed when no macOS Network Service was associated with it.

**Fix:** Added fallback detection that identifies bridge0 without requiring a network service association.

## apply.py — RDMA Event Logging

**Fix:** Added logging for `MacThunderboltConnections` event handler to track when RDMA edges are being populated.

## discovery.rs — Peer Discovery Improvements

**Problem:** With `EXO_PEERS` environment variable, peer discovery would fail silently if a node wasn't reachable on first attempt. Default ping timeout (2.5s) too aggressive for JACCL initialization.

**Fix:**
1. Added retry loop for `EXO_PEERS` — retries every 5 seconds until all peers are reachable
2. Increased ping timeout from 2.5s to 30s
3. Increased ping interval from 2.5s to 10s
4. Requires Rust recompilation after patching

## DSB SY Fence Fix

**Location:** MLX's `fence.cpp` (not Exo)

ARM64 data synchronization barrier (`dsb sy`) added for GPU-CPU memory coherence during RDMA transfers. Without this, stale data can be read from unified memory after RDMA write completion.
