import os
import torch

try:
    import torch.distributed as dist  # type: ignore
except Exception:
    dist = None  # Distributed not available


def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed() -> torch.device:
    """Initialize device and distributed (if available) for inference.

    Falls back to single-process CPU/MPS when torch.distributed or CUDA are unavailable.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    use_ddp = dist is not None and world_size > 1

    if use_ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend, init_method="env://", world_size=world_size, rank=rank
        )

    # Select device
    if torch.cuda.is_available():
        # If DDP, map rank to GPU; otherwise use GPU:0
        gpu_index = rank if use_ddp else 0
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Optional warmup for multi-GPU to avoid first-time latency
    if use_ddp and device.type == "cuda":
        x = torch.ones(1, device=device)
        dist.all_reduce(x)
        torch.cuda.synchronize(device)

    suppress_output(rank if use_ddp else 0)
    return device
from __future__ import annotations
