import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.dpp.naive_dpp import DPP


def setup(rank, world_size, backend, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def run_one_iter(model: DPP,
                 optimizer: torch.optim.Optimizer,
                 x_local: torch.Tensor,
                 y_local: torch.Tensor,
                 device: torch.device,
                 use_flatten: bool = False):
    optimizer.zero_grad(set_to_none=True)

    sync_if_cuda(device)
    start_time = time.time()

    logits = model(x_local)
    loss = cross_entropy(logits, y_local)
    loss.backward()

    sync_if_cuda(device)
    comm_start = time.time()
    if use_flatten:
        model.finish_gradient_synchronization_flatten()
    else:
        model.finish_gradient_synchronization()
    sync_if_cuda(device)
    comm_time = time.time() - comm_start

    optimizer.step()

    sync_if_cuda(device)
    total_time = time.time() - start_time

    return total_time, comm_time


def worker(rank: int,
           world_size: int,
           backend: str,
           global_batch_size: int,
           warmup: int,
           num_iter: int,
           master_addr: str,
           master_port: int,
           jsonl_path: str,
           use_flatten: bool = False):
    try:
        setup(rank, world_size, backend=backend, master_addr=master_addr, master_port=master_port)
        use_cuda = backend == "nccl"
        if use_cuda:
            assert torch.cuda.is_available()
            assert torch.cuda.device_count() == 2
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        model = build_xl_model(device=device, dtype=torch.float32)
        model = DPP(model)
        torch.manual_seed(1234)
        x, y = make_random_batch(batch_size=global_batch_size, context_length=512, vocab_size=10000, device=device)

        assert global_batch_size % world_size == 0
        local_batch_size = global_batch_size // world_size
        x_local = x[rank * local_batch_size:(rank + 1) * local_batch_size]
        y_local = y[rank * local_batch_size:(rank + 1) * local_batch_size]

        optimizer = AdamW(model.parameters())

        for _ in range(warmup):
            run_one_iter(model, optimizer, x_local, y_local, device, use_flatten=use_flatten)

        dist.barrier()
        total_times: List[float] = []
        comm_times: List[float] = []

        for _ in range(num_iter):
            total_time, comm_time = run_one_iter(model, optimizer, x_local, y_local, device, use_flatten=use_flatten)
            total_times.append(total_time)
            comm_times.append(comm_time)

        gather_total_times: List[List[float]] = [[] for _ in range(world_size)]
        gather_comm_times: List[List[float]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gather_total_times, total_times)
        dist.all_gather_object(gather_comm_times, comm_times)

        if rank == 0:
            all_total = [t for rank_times in gather_total_times for t in rank_times]
            all_comm = [t for rank_times in gather_comm_times for t in rank_times]
            mean_total_ms = statistics.mean(all_total) * 1000
            mean_comm_ms = statistics.mean(all_comm) * 1000
            comm_fraction = mean_comm_ms / mean_total_ms
            method = "flatten" if use_flatten else "naive"
            row = {
                "backend": backend,
                "method": method,
                "world_size": world_size,
                "global_batch_size": global_batch_size,
                "warmup_steps": warmup,
                "measure_steps": num_iter,
                "mean_total_ms": mean_total_ms,
                "mean_comm_ms": mean_comm_ms,
                "comm_fraction": comm_fraction,
            }
            print(
                f"[{method}] world_size={world_size} "
                f"total={mean_total_ms:.1f}ms comm={mean_comm_ms:.1f}ms "
                f"comm_frac={comm_fraction:.2%}",
                flush=True,
            )
            out = Path(jsonl_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    finally:
        cleanup()


def build_xl_model(device: torch.device, dtype: torch.dtype):
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=512,
        d_model=2560,
        num_layers=32,
        num_heads=32,
        d_ff=10240
    )

    return model.to(device=device, dtype=dtype)


def make_random_batch(batch_size: int, context_length: int, vocab_size: int, device: torch.device):
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive DDP training step")
    parser.add_argument("--backend", type=str, default="nccl", choices=["gloo", "nccl"])
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-iter", type=int, default=20)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--jsonl-path", type=str, default="./results/naive_dpp.jsonl")
    parser.add_argument("--use-flatten", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    mp.spawn(
        fn=worker,
        args=(
            args.world_size,
            args.backend,
            args.global_batch_size,
            args.warmup,
            args.num_iter,
            args.master_addr,
            args.master_port,
            args.jsonl_path,
            args.use_flatten,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == '__main__':
    main()
