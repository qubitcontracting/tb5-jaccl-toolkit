# Copyright © 2025 Apple Inc.

import mlx.core as mx


class PipelineMixin:
    def __init__(self):
        super().__init__()
        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.start_idx = 0
        self.end_idx = None

    @property
    def pipeline_layers(self):
        return self.layers[self.start_idx : self.end_idx]

    def pipeline(self, group):
        # Split layers proportionally based on available memory
        import os, psutil
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        n_layers = len(self.layers)

        # Gather available memory from all ranks
        # Use Metal working set (~94% of RAM) minus minimum headroom (25GB)
        import mlx.core as mx_info
        if mx_info.metal.is_available():
            local_ws = mx_info.device_info().get('max_recommended_working_set_size', psutil.virtual_memory().total * 0.94)
        else:
            local_ws = psutil.virtual_memory().total * 0.94
        min_headroom = 25 * (1024**3)  # 25GB for KV cache + compute
        local_avail = max(local_ws - min_headroom, 1024**3)  # At least 1GB
        all_mem = mx.distributed.all_gather(
            mx.array([local_avail], dtype=mx.float32), group=group
        )
        mx.eval(all_mem)
        mem_list = [float(all_mem[i].item()) for i in range(self.pipeline_size)]
        total_mem = sum(mem_list)

        # Calculate proportional layer assignment (reverse: rank 0 = last layers)
        layer_counts = []
        assigned = 0
        for i in range(self.pipeline_size):
            if i == self.pipeline_size - 1:
                count = n_layers - assigned  # Last rank gets remainder
            else:
                count = max(1, int(n_layers * mem_list[i] / total_mem))
            layer_counts.append(count)
            assigned += count
        # Verify total
        assert sum(layer_counts) == n_layers, f"Layer split {layer_counts} sums to {sum(layer_counts)}, expected {n_layers}" 

        # Pipeline assigns in reverse: rank 0 = last layers, rank N = first layers
        # layer_counts[0] = rank 0's count (largest RAM)
        # We need rank 0 to get the most layers, assigned at the END
        # So DON'T reverse - rank 0's count stays at the end
        start = sum(layer_counts[i] for i in range(self.pipeline_size) if i > self.pipeline_rank)
        count = layer_counts[self.pipeline_rank]
        self.start_idx = start
        self.end_idx = start + count
        self.layers = self.layers[:self.end_idx]
        self.layers[:self.start_idx] = [None] * self.start_idx
