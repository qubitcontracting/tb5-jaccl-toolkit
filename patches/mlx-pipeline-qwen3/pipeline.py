# Copyright © 2025 Apple Inc.

import mlx.core as mx
from mlx.utils import tree_flatten


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
        # Calculate max layers per node based on model size + compute overhead
        import mlx.core as mx_info
        if mx_info.metal.is_available():
            local_ws = mx_info.device_info().get('max_recommended_working_set_size', psutil.virtual_memory().total * 0.94)
        else:
            local_ws = psutil.virtual_memory().total * 0.94

        # Reserve per-layer compute overhead (~3GB empirically measured)
        # This covers KV cache, attention scores, MoE activations, and MLX graph buffers
        total_model_bytes = sum(p.nbytes for _, p in tree_flatten(self.parameters()) if isinstance(p, mx.array))
        bytes_per_layer = total_model_bytes / n_layers if n_layers > 0 else 1
        compute_overhead_per_layer = 1.5 * (1024**3)  # 1.5GB per layer (empirically tested)
        embed_overhead = 2.5 * (1024**3)  # embedding + lm_head

        # Max layers this node can handle: (working_set - embed) / (weight_per_layer + compute_per_layer)
        max_local = max(1, int((local_ws - embed_overhead) / (bytes_per_layer + compute_overhead_per_layer)))
        # Express as available bytes for proportional splitting
        local_avail = max(float(max_local) * bytes_per_layer, bytes_per_layer)
        all_mem = mx.distributed.all_gather(
            mx.array([local_avail], dtype=mx.float32), group=group
        )
        mx.eval(all_mem)
        mem_list = [float(all_mem[i].item()) for i in range(self.pipeline_size)]
        total_mem = sum(mem_list)

        # Calculate layer assignment: cap each node at max_local, biggest node absorbs excess
        # First calculate max layers each node can handle
        max_per_node = []
        for i in range(self.pipeline_size):
            ws_i = mem_list[i] + (embed_overhead + max(1, int(mem_list[i] / bytes_per_layer)) * compute_overhead_per_layer)
            max_i = max(1, int((ws_i - embed_overhead) / (bytes_per_layer + compute_overhead_per_layer)))
            max_per_node.append(max_i)

        # Proportional split, capped at max per node
        layer_counts = []
        assigned = 0
        for i in range(self.pipeline_size):
            if i == self.pipeline_size - 1:
                count = n_layers - assigned
            else:
                count = max(1, int(n_layers * mem_list[i] / total_mem))
                count = min(count, max_per_node[i])  # Cap at node's max
            layer_counts.append(count)
            assigned += count

        # If last node got too many (above its max), redistribute to largest node
        largest_idx = mem_list.index(max(mem_list))
        while layer_counts[-1] > max_per_node[-1]:
            excess = layer_counts[-1] - max_per_node[-1]
            layer_counts[-1] = max_per_node[-1]
            layer_counts[largest_idx] += excess

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
