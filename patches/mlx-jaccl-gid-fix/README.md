# MLX JACCL GID Index Fix

## Problem

MLX's JACCL backend hardcodes `sgid_index = 1` when setting up RDMA queue pairs. On some Thunderbolt interfaces (particularly the voldemort↔gargamel link using rdma_en3), the IPv4-mapped GID is at index 2, not index 1. GID[1] is empty, causing `errno 22 (EINVAL)` on `ibv_modify_qp` during the INIT→RTR transition.

This breaks 3+ node JACCL clusters. 2-node clusters work if both sides happen to have GID[1] populated.

## Evidence

```
# GID tables showing the mismatch:
vader rdma_en3:    GID[0]=fe80::...  GID[1]=::ffff:10.0.1.1    ← index 1 OK
voldemort rdma_en3: GID[0]=fe80::...  GID[2]=::ffff:10.0.3.1   ← index 1 MISSING
gargamel rdma_en3:  GID[0]=fe80::...  GID[2]=::ffff:10.0.3.2   ← index 1 MISSING
```

## Fix

In `mlx/distributed/jaccl/utils.cpp`, replace hardcoded `query_gid(ctx, 1, 1, &gid)` with a dynamic scan that tries GID indices 1-3 and falls back to index 0:

### In `Connection::info()`:
```cpp
// Before (broken):
ibv().query_gid(ctx, 1, 1, &gid);

// After (fixed):
memset(&gid, 0, sizeof(gid));
for (int gid_idx = 1; gid_idx <= 3; gid_idx++) {
    ibv_gid tmp;
    if (ibv().query_gid(ctx, 1, gid_idx, &tmp) == 0 && tmp.global.interface_id != 0) {
        gid = tmp;
        break;
    }
}
if (gid.global.interface_id == 0) {
    ibv().query_gid(ctx, 1, 0, &gid);
}
```

### In `Connection::queue_pair_rtr()`:
```cpp
// Before (broken):
attr.ah_attr.grh.sgid_index = 1;

// After (fixed):
attr.ah_attr.grh.sgid_index = 0;
for (int idx = 1; idx <= 3; idx++) {
    ibv_gid tmp;
    if (ibv().query_gid(ctx, 1, idx, &tmp) == 0 && tmp.global.interface_id != 0) {
        attr.ah_attr.grh.sgid_index = idx;
        break;
    }
}
```

## Testing

- **Before fix:** 3-node mlx.launch fails with `[jaccl] Changing queue pair to RTR failed with errno 22`
- **After fix:** 3-node mlx.launch works, 0.37s response time, sustained inference with zero degradation
- 2-node still works (no regression)
- Tested with Qwen3-235B-A22B-Instruct-2507-8bit across Mac Studio M3 Ultra + 2x Mac mini M4 Pro
