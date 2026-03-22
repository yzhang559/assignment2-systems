import argparse
import statistics
import timeit
from pathlib import Path

import pandas as pd
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

MODEL_SIZES = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}


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


def make_random_batch(batch_size: int, context_length: int, vocab_size: int, device: str):
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y


def run_benchmark(
    model: BasicsTransformerLM,
    device: str,
    mode: str,
    warmup_steps: int,
    measure_steps: int,
    x: torch.Tensor,
    y: torch.Tensor,
):
    model.train()
    step_times = []

    for _ in range(warmup_steps):
        if mode == "forward":
            with torch.no_grad():
                logits = model.forward(x)
                loss = cross_entropy(logits, y)
        elif mode == "forward_backward":
            model.zero_grad(set_to_none=True)
            logits = model.forward(x)
            loss = cross_entropy(logits, y)
            loss.backward()

    for _ in range(measure_steps):
        if mode == "forward":
            synchronize(device)
            start = timeit.default_timer()
            with torch.no_grad():
                logits = model.forward(x)
                loss = cross_entropy(logits, y)
                synchronize(device)
                end = timeit.default_timer()

        elif mode == "forward_backward":
            model.zero_grad(set_to_none=True)
            synchronize(device)
            start = timeit.default_timer()
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            synchronize(device)
            end = timeit.default_timer()

        step_times.append(end - start)

    mean_time = statistics.mean(step_times)
    std_time = statistics.stdev(step_times) if len(step_times) > 1 else 0.0

    print(f"mode={mode}")
    print(f"step_times={[round(t, 6) for t in step_times]}")
    print(f"mean={mean_time:.6f}s std={std_time:.6f}s")

    return step_times, mean_time, std_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark TransformerLM")

    parser.add_argument("--mode", choices=["forward", "forward_backward"], default="forward")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/mps/cpu). Auto-detected if not set."
    )
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"using device: {device}")

    run_all(args, device)

    # model = init_model(args, device)
    #
    # x, y = make_random_batch(args.batch_size, args.context_length, args.vocab_size, device)
    #
    # run_benchmark(model, device, args.mode, args.warmup_steps, args.measure_steps, x, y)


def run_all(args, device: str):
    results = []

    for model_name, config in MODEL_SIZES.items():
        print(f"\n===== Benchmarking {model_name} =====")

        args.d_model = config["d_model"]
        args.d_ff = config["d_ff"]
        args.num_layers = config["num_layers"]
        args.num_heads = config["num_heads"]

        model = init_model(args, device)
        x, y = make_random_batch(args.batch_size, args.context_length, args.vocab_size, device)

        step_times, mean_time, std_time = run_benchmark(
            model,
            device,
            args.mode,
            args.warmup_steps,
            args.measure_steps,
            x,
            y,
        )

        results.append(
            {
                "model_name": model_name,
                "mode": args.mode,
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "d_model": config["d_model"],
                "d_ff": config["d_ff"],
                "num_layers": config["num_layers"],
                "num_heads": config["num_heads"],
                "mean_time": mean_time,
                "std_time": std_time,
                "mean_time_ms": mean_time * 1000.0,
                "std_time_ms": std_time * 1000.0,
            }
        )

        del model
        del x
        del y
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    export_results_tables(results, args.output_dir, args.mode, args.context_length)


def export_results_tables(results: list[dict], output_dir: str, mode: str, context_length: int):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    table_columns = [
        "model_name",
        "mean_time_ms",
        "std_time_ms",
        "d_model",
        "d_ff",
        "num_layers",
        "num_heads",
        "context_length",
        "batch_size",
    ]
    table_df = df[table_columns].copy()
    table_df["mean_time_ms"] = table_df["mean_time_ms"].map(lambda x: f"{x:.3f}")
    table_df["std_time_ms"] = table_df["std_time_ms"].map(lambda x: f"{x:.3f}")

    suffix = f"{mode}_ctx{context_length}"
    csv_path = output_path / f"{suffix}.csv"
    markdown_path = output_path / f"{suffix}.md"

    df.to_csv(csv_path, index=False)
    markdown_path.write_text(table_df.to_markdown(index=False) + "\n", encoding="utf-8")

    print(f"saved: {csv_path}")
    print(f"saved: {markdown_path}")


if __name__ == "__main__":
    main()
    print("Done")
