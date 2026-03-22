import argparse
import statistics
import timeit
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.utils import MODEL_SIZES, add_shared_benchmark_args, make_random_batch, resolve_device


def init_model(args, device: str):
    vocab_size = args.vocab_size
    context_length = args.context_length
    d_model = args.d_model
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_ff = args.d_ff
    rope_theta = 10000.0

    model = BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
    )

    return model.to(device)


def synchronize(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run_benchmark(
    model: BasicsTransformerLM,
    device: str,
    mode: str,
    warmup_steps: int,
    measure_steps: int,
    x: torch.Tensor,
    y: torch.Tensor,
    mixed_precision: bool = False,
    memory_profile: bool = False,
    model_name: str = "",
    context_length: int = 0,
    batch_size: int = 0,
):
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    autocast_ctx = torch.autocast(device, dtype=torch.bfloat16) if mixed_precision else nullcontext()

    forward_samples = []
    backward_samples = []
    optimizer_samples = []

    with autocast_ctx:
        for i in range(warmup_steps + measure_steps):
            if memory_profile and i == warmup_steps:
                torch.cuda.memory._record_memory_history(max_entries=1000000)

            synchronize(device)
            t0 = timeit.default_timer()
            logits = model(x)
            synchronize(device)
            dt_forward = timeit.default_timer() - t0

            dt_backward = 0.0
            if mode in ("forward_backward", "train_step"):
                loss = cross_entropy(logits, y)
                t1 = timeit.default_timer()
                loss.backward()
                synchronize(device)
                dt_backward = timeit.default_timer() - t1

            dt_optimizer = 0.0
            if mode == "train_step":
                t2 = timeit.default_timer()
                optimizer.step()
                synchronize(device)
                dt_optimizer = timeit.default_timer() - t2
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)

            if i >= warmup_steps:
                forward_samples.append(dt_forward)
                backward_samples.append(dt_backward)
                optimizer_samples.append(dt_optimizer)

        if memory_profile:
            mixed_str = "_bf16" if mixed_precision else ""
            snapshot_filename = (
                f"memory_snapshot_{model_name}_{mode}_ctx{context_length}_bs{batch_size}{mixed_str}.pickle"
            )
            torch.cuda.memory._dump_snapshot(snapshot_filename)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"Memory snapshot saved: {snapshot_filename}")

    def stats(samples):
        if not samples or all(s == 0.0 for s in samples):
            return 0.0, 0.0
        mean = statistics.mean(samples)
        std = statistics.stdev(samples) if len(samples) > 1 else 0.0
        return mean, std

    forward_mean, forward_std = stats(forward_samples)
    backward_mean, backward_std = stats(backward_samples)
    optimizer_mean, optimizer_std = stats(optimizer_samples)

    total_samples = [f + b + o for f, b, o in zip(forward_samples, backward_samples, optimizer_samples)]
    total_mean, total_std = stats(total_samples)

    print(f"mode={mode}")
    print(f"forward:   mean={forward_mean*1000:.3f}ms std={forward_std*1000:.3f}ms")
    print(f"backward:  mean={backward_mean*1000:.3f}ms std={backward_std*1000:.3f}ms")
    print(f"optimizer: mean={optimizer_mean*1000:.3f}ms std={optimizer_std*1000:.3f}ms")
    print(f"total:     mean={total_mean*1000:.3f}ms std={total_std*1000:.3f}ms")

    return {
        "forward_mean": forward_mean, "forward_std": forward_std,
        "backward_mean": backward_mean, "backward_std": backward_std,
        "optimizer_mean": optimizer_mean, "optimizer_std": optimizer_std,
        "total_mean": total_mean, "total_std": total_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark TransformerLM")
    add_shared_benchmark_args(
        parser,
        modes=("forward", "forward_backward", "train_step"),
        default_mode="forward",
        include_output_args=True,
    )
    parser.add_argument("--mixed_precision", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--memory_profile", action="store_true", help="Enable CUDA memory profiling")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"using device: {device}")

    run_all(args, device)

    # model = init_model(args, device)
    #
    # x, y = make_random_batch(args.batch_size, args.context_length, args.vocab_size, device)
    #
    # run_benchmark(model, device, args.mode, args.warmup_steps, args.measure_steps, x, y)


def run_all(args, device: str):
    results = []

    if args.model_size == "all":
        selected_sizes = MODEL_SIZES.items()
    else:
        selected_sizes = [(args.model_size, MODEL_SIZES[args.model_size])]

    for model_name, config in selected_sizes:
        print(f"\n===== Benchmarking {model_name} =====")

        args.d_model = config["d_model"]
        args.d_ff = config["d_ff"]
        args.num_layers = config["num_layers"]
        args.num_heads = config["num_heads"]

        model = init_model(args, device)
        x, y = make_random_batch(args.batch_size, args.context_length, args.vocab_size, device)

        timing = run_benchmark(
            model,
            device,
            args.mode,
            args.warmup_steps,
            args.measure_steps,
            x,
            y,
            mixed_precision=args.mixed_precision,
            memory_profile=args.memory_profile,
            model_name=model_name,
            context_length=args.context_length,
            batch_size=args.batch_size,
        )

        results.append(
            {
                "model_name": model_name,
                "mode": args.mode,
                "mixed_precision": args.mixed_precision,
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "d_model": config["d_model"],
                "d_ff": config["d_ff"],
                "num_layers": config["num_layers"],
                "num_heads": config["num_heads"],
                "forward_mean_ms": timing["forward_mean"] * 1000.0,
                "forward_std_ms": timing["forward_std"] * 1000.0,
                "backward_mean_ms": timing["backward_mean"] * 1000.0,
                "backward_std_ms": timing["backward_std"] * 1000.0,
                "optimizer_mean_ms": timing["optimizer_mean"] * 1000.0,
                "optimizer_std_ms": timing["optimizer_std"] * 1000.0,
                "total_mean_ms": timing["total_mean"] * 1000.0,
                "total_std_ms": timing["total_std"] * 1000.0,
            }
        )

        del model
        del x
        del y
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.skip_write_out:
        return

    export_results_tables(
        results,
        args.output_dir,
        args.mode,
        args.context_length,
        args.warmup_steps,
        args.measure_steps,
    )


def export_results_tables(
    results: list[dict],
    output_dir: str,
    mode: str,
    context_length: int,
    warmup_steps: int,
    measure_steps: int,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    table_columns = [
        "model_name",
        "forward_mean_ms",
        "backward_mean_ms",
        "optimizer_mean_ms",
        "total_mean_ms",
        "d_model",
        "num_layers",
        "context_length",
        "batch_size",
    ]
    table_df = df[table_columns].copy()
    for col in ["forward_mean_ms", "backward_mean_ms", "optimizer_mean_ms", "total_mean_ms"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.3f}")

    mixed_str = "_bf16" if results[0].get("mixed_precision") else ""
    suffix = f"{mode}_ctx{context_length}_wu{warmup_steps}_ms{measure_steps}{mixed_str}"
    csv_path = output_path / f"{suffix}.csv"
    markdown_path = output_path / f"{suffix}.md"

    df.to_csv(csv_path, index=False)
    markdown_path.write_text(table_df.to_markdown(index=False) + "\n", encoding="utf-8")

    print(f"saved: {csv_path}")
    print(f"saved: {markdown_path}")


if __name__ == "__main__":
    main()
    print("Done")
