import argparse
import os
import statistics
import time
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.dpp.utils import DDPCommBenchmarkReporter, DDPCommRow


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


def worker(rank: int,
           world_size: int,
           backend: str,
           size_bytes: List[int],
           warmup: int,
           num_iter: int,
           master_addr: str,
           master_port: int,
           jsonl_path: str,
           md_path: str):
    try:
        setup(rank, world_size, backend, master_addr, master_port)
        use_cuda = backend == "nccl"
        if use_cuda:
            assert torch.cuda.is_available()
            assert torch.cuda.device_count() >= world_size
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        dtype = torch.float32
        dist.barrier()
        torch.manual_seed(rank + 1234)

        for size in size_bytes:
            numel = size // dtype.itemsize
            data = torch.rand((numel,), dtype=dtype, device=device)

            for _ in range(warmup):
                dist.all_reduce(data, async_op=False)
                sync_if_cuda(device)

            times_ms = []
            for _ in range(num_iter):
                sync_if_cuda(device)
                start = time.time()
                dist.all_reduce(data, async_op=False)
                sync_if_cuda(device)
                end = time.time()
                times_ms.append((end - start) * 1000)

            gather_times: List[List[float]] = [[] for _ in range(world_size)]
            dist.all_gather_object(gather_times, times_ms)

            if rank == 0:
                all_times = [t for rank_times in gather_times for t in rank_times]
                mean_ms = statistics.mean(all_times)
                mean_s = mean_ms / 1000
                algbw_GBps = (size / mean_s) / 1024 ** 3
                busbw_GBps = algbw_GBps * 2 * (world_size - 1) / world_size
                row = DDPCommRow(
                    backend=backend,
                    device="cuda" if use_cuda else "cpu",
                    world_size=world_size,
                    op="all_reduce",
                    size_bytes=size,
                    dtype="float32",
                    warmup_steps=warmup,
                    measure_steps=num_iter,
                    mean_ms=mean_ms,
                    std_ms=statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
                    min_ms=min(all_times),
                    max_ms=max(all_times),
                    algbw_GBps=algbw_GBps,
                    busbw_GBps=busbw_GBps,
                )
                reporter = DDPCommBenchmarkReporter(jsonl_path=jsonl_path, md_path=md_path)
                reporter.append(row)
                reporter.write_markdown()
                print(
                    f"[all_reduce] world_size={world_size} size={size // 1024 ** 2}MB algbw={algbw_GBps:.2f} GB/s busbw={busbw_GBps:.2f} GB/s",
                    flush=True)

    finally:
        cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all-reduce communication")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-iter", type=int, default=20)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--jsonl-path", type=str, default="./results/ddp_comm.jsonl")
    parser.add_argument("--md-path", type=str, default="./results/ddp_comm.md")
    return parser.parse_args()


def main():
    args = parse_args()

    size_bytes_list = [
        1 * 1024 * 1024,
        10 * 1024 * 1024,
        100 * 1024 * 1024,
        1024 * 1024 * 1024,
    ]

    mp.spawn(
        fn=worker,
        args=(
            args.world_size,
            args.backend,
            size_bytes_list,
            args.warmup,
            args.num_iter,
            args.master_addr,
            args.master_port,
            args.jsonl_path,
            args.md_path,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
    # reporter = DDPCommBenchmarkReporter(jsonl_path="./results/ddp_comm.jsonl", md_path="./results/ddp_comm.md")
    # reporter.plot_results("./results")
