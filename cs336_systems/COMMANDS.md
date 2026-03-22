# Benchmark & Profiling Commands

## Benchmarking (`benchmarking.py`)

Run timing benchmarks for TransformerLM models.

```bash
# Basic usage - benchmark all model sizes with forward pass
uv run python -m cs336_systems.benchmarking

# Benchmark a specific model size
uv run python -m cs336_systems.benchmarking --model_size small

# Benchmark with full training step
uv run python -m cs336_systems.benchmarking --mode train_step --model_size medium

# Custom configuration
uv run python -m cs336_systems.benchmarking \
    --mode forward_backward \
    --model_size large \
    --batch_size 8 \
    --context_length 512 \
    --warmup_steps 5 \
    --measure_steps 20 \
    --output_dir benchmark_results

# Skip writing output files
uv run python -m cs336_systems.benchmarking --mode forward --skip_write_out
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `forward` | `forward`, `forward_backward`, or `train_step` |
| `--model_size` | `all` | `small`, `medium`, `large`, `xl`, `2.7B`, or `all` |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--batch_size` | 4 | Batch size |
| `--context_length` | 256 | Sequence length |
| `--warmup_steps` | 5 | Warmup iterations |
| `--measure_steps` | 10 | Measurement iterations |
| `--output_dir` | `benchmark_results` | Output directory |
| `--skip_write_out` | false | Skip saving results |

---

## Profiling (`profile.py`)

Run Nsight Systems profiling with NVTX annotations.

```bash
# Single profiling run
uv run python -m cs336_systems.profile --mode forward --model_size small

# Full training step profiling
uv run python -m cs336_systems.profile \
    --mode train_step \
    --model_size medium \
    --context_length 256 \
    --warmup_steps 5 \
    --profile_steps 10

# Sweep mode - run nsys for multiple configurations
uv run python -m cs336_systems.profile --sweep \
    --modes forward,forward_backward,train_step \
    --model_sizes small,medium \
    --contexts 128,256,512 \
    --output_dir nsys_results

# Sweep with PyTorch annotations
uv run python -m cs336_systems.profile --sweep \
    --use_pytorch_annotations \
    --python_backtrace_cuda \
    --output_dir nsys_results

# Dry run (print commands without executing)
uv run python -m cs336_systems.profile --sweep --dry_run
```

### Arguments (Single Run)

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `forward` | `forward`, `forward_backward`, or `train_step` |
| `--model_size` | `small` | `small`, `medium`, `large`, `xl`, `2.7B` |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--batch_size` | 4 | Batch size |
| `--context_length` | 256 | Sequence length |
| `--warmup_steps` | 2 | Warmup iterations |
| `--profile_steps` | 3 | Profiling iterations |

### Arguments (Sweep Mode)

| Argument | Default | Description |
|----------|---------|-------------|
| `--sweep` | false | Enable sweep mode |
| `--modes` | `forward,forward_backward,train_step` | Comma-separated modes |
| `--model_sizes` | `all` | Comma-separated sizes or `all` |
| `--contexts` | `128,256,512,1024` | Comma-separated context lengths |
| `--output_dir` | `nsys_results` | Output directory for `.nsys-rep` files |
| `--nsys_bin` | `nsys` | Path to nsys binary |
| `--use_pytorch_annotations` | false | Enable PyTorch autograd NVTX |
| `--python_backtrace_cuda` | false | Enable Python backtrace for CUDA |
| `--dry_run` | false | Print commands without running |
