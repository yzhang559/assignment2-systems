import argparse

import torch

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


def add_shared_benchmark_args(
    parser: argparse.ArgumentParser,
    *,
    modes: tuple[str, ...],
    default_mode: str,
    include_output_args: bool,
) -> None:
    parser.add_argument("--mode", choices=modes, default=default_mode)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu). Auto-detected if not set.",
    )
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument(
        "--model_size",
        type=str,
        default="all",
        choices=["all", *MODEL_SIZES.keys()],
        help="Run one preset model size, or all sizes from Table 1.",
    )

    if include_output_args:
        parser.add_argument("--output_dir", type=str, default="benchmark_results")
        parser.add_argument("--skip_write_out", action="store_true")


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_random_batch(batch_size: int, context_length: int, vocab_size: int, device: str):
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y
