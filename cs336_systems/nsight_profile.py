"""
Nsight Systems profiling with NVTX annotations.
"""
import argparse
import math
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torch.cuda.nvtx as nvtx
from cs336_systems.utils import MODEL_SIZES, make_random_batch, resolve_device


def parse_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_model_sizes(model_sizes: str) -> list[str]:
    if model_sizes == "all":
        return list(MODEL_SIZES.keys())
    names = parse_csv_arg(model_sizes)
    unknown = [name for name in names if name not in MODEL_SIZES]
    if unknown:
        raise ValueError(f"Unknown model sizes: {unknown}. Valid: {list(MODEL_SIZES)}")
    return names


# ---------------------------------------------------------------------------
# NVTX-annotated attention (inlined from nvtx_annotations.py)
# ---------------------------------------------------------------------------
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with nvtx.range("computing attention scores"):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -torch.inf)

    with nvtx.range("computing softmax"):
        attention_weights = torch.softmax(scores, dim=-1)

    with nvtx.range("final matmul"):
        output = attention_weights @ V

    return output


# ---------------------------------------------------------------------------
# Profiling execution
# ---------------------------------------------------------------------------
def run_profiling(
        model,
        mode: str,
        warmup_steps: int,
        profile_steps: int,
        x: torch.Tensor,
        y: torch.Tensor,
        cross_entropy,
        AdamW,
):
    model.train(mode in ("forward_backward", "train_step"))
    optimizer = AdamW(model.parameters(), lr=1e-3)

    for i in range(warmup_steps + profile_steps):
        phase = "warmup." if i < warmup_steps else "profile."

        with nvtx.range(phase + "forward"):
            logits = model(x)

        with nvtx.range(phase + "loss"):
            loss = cross_entropy(logits, y)

        if mode in ("forward_backward", "train_step"):
            with nvtx.range(phase + "backward"):
                loss.backward()

        if mode == "train_step":
            with nvtx.range(phase + "optimizer_step"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# CLI: direct run
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nsight Systems profiling")
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step"),
        default="forward",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--profile_steps", type=int, default=3)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=list(MODEL_SIZES.keys()),
    )
    # Sweep-mode arguments
    parser.add_argument("--sweep", action="store_true", help="Run nsys sweep over configs")
    parser.add_argument("--modes", type=str, default="forward,forward_backward,train_step")
    parser.add_argument("--contexts", type=str, default="128,256,512,1024")
    parser.add_argument("--model_sizes", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="nsys_results")
    parser.add_argument("--nsys_bin", type=str, default="nsys")
    parser.add_argument("--use_pytorch_annotations", action="store_true")
    parser.add_argument("--python_backtrace_cuda", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def run_single(args):
    import cs336_basics.model as basics_model
    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy
    from cs336_basics.optimizer import AdamW

    device = resolve_device(args.device)
    print(f"Profiling: device={device}")

    config = MODEL_SIZES[args.model_size]
    print(f"Model: {args.model_size}, Mode: {args.mode}, Context: {args.context_length}")

    model = BasicsTransformerLM(
        args.vocab_size,
        args.context_length,
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    x, y = make_random_batch(args.batch_size, args.context_length, args.vocab_size, device)

    run_profiling(model, args.mode, args.warmup_steps, args.profile_steps, x, y, cross_entropy, AdamW)
    print("Profiling run complete.")


def run_sweep(args):
    """Launch nsys for each (model_size, context, mode) combination."""
    modes = parse_csv_arg(args.modes)
    contexts = [int(x) for x in parse_csv_arg(args.contexts)]
    model_sizes = resolve_model_sizes(args.model_sizes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_size in model_sizes:
        for context_length in contexts:
            for mode in modes:
                output_prefix = output_dir / f"{mode}_{model_size}_ctx{context_length}"
                cmd = [
                    "uv",
                    "run",
                    args.nsys_bin,
                    "profile",
                    "--trace=cuda,nvtx,osrt",
                    "--sample=none",
                    "--force-overwrite=true",
                    "-o",
                    str(output_prefix),
                ]
                if args.use_pytorch_annotations:
                    cmd.append("--pytorch=autograd-shapes-nvtx")
                if args.python_backtrace_cuda:
                    cmd.append("--python-backtrace=cuda")
                cmd.extend(
                    [
                        "python",
                        "-m",
                        "cs336_systems.profile",
                        "--mode",
                        mode,
                        "--model_size",
                        model_size,
                        "--context_length",
                        str(context_length),
                        "--warmup_steps",
                        str(args.warmup_steps),
                        "--profile_steps",
                        str(args.profile_steps),
                        "--batch_size",
                        str(args.batch_size),
                        "--vocab_size",
                        str(args.vocab_size),
                        "--device",
                        args.device if args.device is not None else "cuda",
                    ]
                )

                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
