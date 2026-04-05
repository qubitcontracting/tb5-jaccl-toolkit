# Finding: JACCL works WITHOUT destroying bridge0

After extensive testing on a 3-node Thunderbolt 5 cluster, we discovered that **bridge0 does NOT need to be destroyed** for JACCL to work. Simply assigning IP addresses to the individual Thunderbolt interfaces is sufficient — JACCL and bridge0 can coexist.

This eliminates the need for the bridge-fix script on every boot, and avoids a critical issue where destroying bridge0 can kill USB Ethernet adapters on Mac minis.

## The Problem with Destroying bridge0

The JACCL guide recommends `ifconfig bridge0 destroy` before every JACCL session. On Mac minis using USB 2.5GbE Ethernet adapters (common when the built-in Ethernet port is in use or insufficient), destroying bridge0 **causes the USB adapter to lose connectivity**. The interface disappears entirely from `ifconfig` and cannot be recovered without a physical replug or reboot. This makes the "destroy on every boot" approach unreliable for headless setups.

## The Fix: Just Assign IPs

Instead of destroying bridge0, simply assign IP addresses to the individual Thunderbolt interfaces:

```bash
# Node 0 (Mac mini - Gargamel)
sudo ifconfig en4 10.0.2.2/24 up   # TB cable to Node 1

# Node 1 (Mac Studio - Vader)
sudo ifconfig en3 10.0.1.1/24 up   # TB cable to Node 2
sudo ifconfig en4 10.0.2.1/24 up   # TB cable to Node 0

# Node 2 (Mac mini - Voldemort)
sudo ifconfig en4 10.0.1.2/24 up   # TB cable to Node 1
```

bridge0 remains intact. USB adapters stay connected. RDMA devices show `PORT_ACTIVE`:

```
$ ibv_devinfo | grep -E 'hca_id|state:'
hca_id: rdma_en3
        state: PORT_ACTIVE (4)
hca_id: rdma_en4
        state: PORT_ACTIVE (4)
```

## Why It Works

The original guidance states bridge0 "absorbs all TB ports into a single interface" which "completely breaks RDMA." This is partially correct — bridge0 does aggregate the interfaces, but the individual TB interfaces can **simultaneously** have their own IP addresses and serve RDMA traffic. The RDMA layer (`libthunderboltrdma.dylib`) accesses the interfaces directly via `ibv_*` calls, bypassing the bridge's network stack.

## Throughput (bridge0 present)

```
   Size    |   Time    | Throughput
     3.8 MB |    0.7 ms |  5,492 MB/s
    38.1 MB |    5.2 ms |  7,299 MB/s
   381.5 MB |   51.3 ms |  7,431 MB/s
```

**7.4 GB/s sustained** across 3 nodes with bridge0 intact — matches throughput with bridge0 destroyed.

## Comparison

| Approach | bridge0 | JACCL | USB Adapter | Boot Script Needed |
|----------|---------|-------|-------------|-------------------|
| Guide (destroy) | Destroyed | ✓ | ✗ May lose connectivity | Yes, every boot |
| **This finding (coexist)** | **Present** | **✓** | **✓ Stays connected** | **Just assign IPs** |
