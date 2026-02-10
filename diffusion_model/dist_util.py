"""Helpers for distributed training (single-node only)."""

import io
import os
import socket
import blobfile as bf
import torch as th
import torch.distributed as dist
from mpi4py import MPI

# 默认每节点 GPU 数量（用于 dev() 函数，可选）
GPUS_PER_NODE = 8


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(sock.getsockname()[1])


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    os.environ["MASTER_ADDR"] =  "127.0.0.1"
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")

def dev():
    """
    Get the device to use for torch.distributed.
    
    Returns:
        torch.device: cuda:{LOCAL_RANK} if CUDA available, else cpu
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file safely in distributed setting.
    
    Only rank 0 loads the file from disk/network, then broadcasts to others.
    """
    if dist.is_available() and dist.is_initialized():
        # Only rank 0 loads the data
        if dist.get_rank() == 0:
            with bf.BlobFile(path, "rb") as f:
                data = f.read()
        else:
            data = None

        # Broadcast from rank 0 to all others
        data_tensor = th.tensor(list(data), dtype=th.uint8) if data is not None else None
        if data is None:
            # Create placeholder tensor on other ranks
            data_len = th.tensor([0], dtype=th.long)
            dist.broadcast(data_len, src=0)
            data_tensor = th.empty(int(data_len.item()), dtype=th.uint8)
        else:
            data_len = th.tensor([len(data)], dtype=th.long)
            dist.broadcast(data_len, src=0)
            dist.broadcast(data_tensor, src=0)

        # Reconstruct bytes
        data_bytes = data_tensor.numpy().tobytes()
        return th.load(io.BytesIO(data_bytes), **kwargs)
    else:
        # Non-distributed case
        with bf.BlobFile(path, "rb") as f:
            return th.load(f, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    
    Args:
        params: Iterable of torch.Tensor
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in params:
        dist.broadcast(p, src=0)