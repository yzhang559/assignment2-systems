# 1.1.3 Benchmark results

Forward, 5 warmup steps, 10 measure steps

| model_name   |   mean_time_ms |   std_time_ms |   d_model |   d_ff |   num_layers |   num_heads |   context_length |   batch_size |
|:-------------|---------------:|--------------:|----------:|-------:|-------------:|------------:|-----------------:|-------------:|
| small        |         16.998 |         2.995 |       768 |   3072 |           12 |          12 |              256 |            4 |
| medium       |         31.131 |         1.327 |      1024 |   4096 |           24 |          16 |              256 |            4 |
| large        |         64.948 |         0.466 |      1280 |   5120 |           36 |          20 |              256 |            4 |
| xl           |        126.615 |         0.041 |      1600 |   6400 |           48 |          25 |              256 |            4 |
| 2.7B         |        174.129 |         0.219 |      2560 |  10240 |           32 |          32 |              256 |            4 |

Forward + Backward, 5 warmup steps, 10 measure steps

| model_name   |   mean_time_ms |   std_time_ms |   d_model |   d_ff |   num_layers |   num_heads |   context_length |   batch_size |
|:-------------|---------------:|--------------:|----------:|-------:|-------------:|------------:|-----------------:|-------------:|
| small        |         42.912 |         1.809 |       768 |   3072 |           12 |          12 |              256 |            4 |
| medium       |         92.976 |         1.899 |      1024 |   4096 |           24 |          16 |              256 |            4 |
| large        |        201.755 |         1.469 |      1280 |   5120 |           36 |          20 |              256 |            4 |
| xl           |        388.879 |         0.181 |      1600 |   6400 |           48 |          25 |              256 |            4 |
| 2.7B         |        553.312 |         0.549 |      2560 |  10240 |           32 |          32 |              256 |            4 |

Forward, 0 warmup steps, 10 measure steps

| model_name   |   mean_time_ms |   std_time_ms |   d_model |   d_ff |   num_layers |   num_heads |   context_length |   batch_size |   warmup_steps |   measure_steps |
|:-------------|---------------:|--------------:|----------:|-------:|-------------:|------------:|-----------------:|-------------:|---------------:|----------------:|
| small        |         70.593 |       171.054 |       768 |   3072 |           12 |          12 |              256 |            4 |              0 |              10 |
| medium       |         33.323 |        11.7   |      1024 |   4096 |           24 |          16 |              256 |            4 |              0 |              10 |
| large        |         68.008 |         9.473 |      1280 |   5120 |           36 |          20 |              256 |            4 |              0 |              10 |
| xl           |        130.584 |        11.378 |      1600 |   6400 |           48 |          25 |              256 |            4 |              0 |              10 |
| 2.7B         |        179.325 |        16.378 |      2560 |  10240 |           32 |          32 |              256 |            4 |              0 |              10 |

Forward + Backward, 0 warmup steps, 10 measure steps

| model_name   |   mean_time_ms |   std_time_ms |   d_model |   d_ff |   num_layers |   num_heads |   context_length |   batch_size |   warmup_steps |   measure_steps |
|:-------------|---------------:|--------------:|----------:|-------:|-------------:|------------:|-----------------:|-------------:|---------------:|----------------:|
| small        |        100.16  |       189.698 |       768 |   3072 |           12 |          12 |              256 |            4 |              0 |              10 |
| medium       |         97.548 |        21.452 |      1024 |   4096 |           24 |          16 |              256 |            4 |              0 |              10 |
| large        |        211.883 |        25.878 |      1280 |   5120 |           36 |          20 |              256 |            4 |              0 |              10 |
| xl           |        395.944 |        20.659 |      1600 |   6400 |           48 |          25 |              256 |            4 |              0 |              10 |
| 2.7B         |        558.272 |        16.489 |      2560 |  10240 |           32 |          32 |              256 |            4 |              0 |              10 |

b.
See above table, the variability across measurements is low. 

c.
Without 5 warmup steps, the std are much larger. 
The variability is very high for small/medium (e.g., small forward std 171.054 ms > mean 70.593 ms)



# 1.1.4 nsys profile
Use python benchmark to benchmark

| model_name   |   forward_mean_ms |   backward_mean_ms |   optimizer_mean_ms |   total_mean_ms |   d_model |   num_layers |   context_length |   batch_size |
|:-------------|------------------:|-------------------:|--------------------:|----------------:|----------:|-------------:|-----------------:|-------------:|
| small        |            18.794 |             25.497 |              10.429 |          54.72  |       768 |           12 |              256 |            4 |
| medium       |            33.654 |             59.036 |              22.959 |         115.648 |      1024 |           24 |              256 |            4 |
| large        |            76.361 |            137.211 |              51.64  |         265.212 |      1280 |           36 |              256 |            4 |
| xl           |           137.159 |            266.691 |             111.167 |         515.018 |      1600 |           48 |              256 |            4 |
| 2.7B         |           174.307 |            381.704 |             165.247 |         721.258 |      2560 |           32 |              256 |            4 |

Use nvtx profile

nvtx range summary for xl

| Time | Total Time | Instances | Avg | Med | Min | Max | StdDev | Style | Range |
|:-----|:-----------|----------:|:----|:----|:----|:----|:-------|:------|:------|
| 28.1% | 2.418 s | 10 | 241.844 ms | 246.011 ms | 218.188 ms | 248.345 ms | 9.080 ms | PushPop | :profile.backward |
| 15.9% | 1.369 s | 10 | 136.887 ms | 136.969 ms | 135.982 ms | 137.412 ms | 461.698 us | PushPop | :profile.optimizer_step |
| 15.2% | 1.306 s | 5 | 261.181 ms | 246.703 ms | 241.571 ms | 323.325 ms | 34.821 ms | PushPop | :warmup.backward |
| 12.7% | 1.097 s | 5 | 219.322 ms | 93.172 ms | 86.670 ms | 730.251 ms | 285.632 ms | PushPop | :warmup.forward |
| 12.4% | 1.069 s | 10 | 106.932 ms | 94.769 ms | 93.706 ms | 204.925 ms | 34.546 ms | PushPop | :profile.forward |
| 7.8% | 675.206 ms | 5 | 135.041 ms | 137.808 ms | 124.321 ms | 138.266 ms | 6.032 ms | PushPop | :warmup.optimizer_step |
| 3.5% | 299.904 ms | 720 | 416.533 us | 238.474 us | 210.570 us | 105.129 ms | 3.910 ms | PushPop | :scaled dot product attention |
| 2.6% | 221.912 ms | 720 | 308.211 us | 151.625 us | 134.263 us | 91.731 ms | 3.414 ms | PushPop | :computing attention scores |
| 0.7% | 56.506 ms | 5 | 11.301 ms | 1.896 ms | 1.716 ms | 48.859 ms | 20.996 ms | PushPop | :warmup.loss |
| 0.5% | 41.305 ms | 720 | 57.368 us | 54.245 us | 46.752 us | 283.957 us | 11.527 us | PushPop | :final matmul |
| 0.3% | 27.003 ms | 720 | 37.503 us | 18.297 us | 16.761 us | 13.068 ms | 486.303 us | PushPop | :computing softmax |
| 0.3% | 25.171 ms | 10 | 2.517 ms | 2.546 ms | 1.657 ms | 3.588 ms | 687.160 us | PushPop | :profile.loss |

a. picking the XL model, the forward step is around 106.932 ms, the same order of magnitude as the python benchmark 137.159

b. 
sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize256x128x8_stage3_warpsize4x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas
TODO: how to check the instance count of this kernel in single step

c. 
vectorized_elementwise_kernel 
elementwise_kernal

d. 
for XL model, the train step has around 38% matrix multiplication, the forward has around 84% matrix multiplication
the full train step is expensive than forward only with more elementwise operations around 20% of kernal. The ratio is 
around 6.8/1.78s =3.8x

e.
FLOPs ratio of matmul/softmax = O(d) = 1600
Wall time ratio = (7.967 ms+3.514 ms) / 866 us = 13

matrix multiplications in self-attention have ~1600× more FLOPs than softmax (for d=1600), they only take ~13× more 
runtime in practice. This is because GEMMs are highly optimized and compute-efficient, while softmax is 
memory-bound and much less efficient despite having fewer FLOPs.

# 1.1.5 Mixed Precision

a.

    fc1 weight dtype: torch.float32
    fc1 weight dtype within autocase: torch.float32
    fc1 output: torch.float16
    ln output: torch.float32
    fc2 output: torch.float16
    loss dtype: torch.float32
    fc1 weight grad dtype: torch.float32

b.

y = (x - E[x]) / sqrt(var[x]+eps) * gamma + beta

the mean and variance computations are the precision sensitive parts, 
involving sum which accumulates rounding errors

c.

with BF16, every model is faster than FP32, especially for larger models. BF16 has larger dynamic range as FP32, FP16 
has better precision. BF16 also makes the gradient more stable, won't flush to zero or overflow. 

| model_name   |   forward_mean_ms |   backward_mean_ms |   optimizer_mean_ms |   total_mean_ms |   d_model |   num_layers |   context_length |   batch_size |
|:-------------|------------------:|-------------------:|--------------------:|----------------:|----------:|-------------:|-----------------:|-------------:|
| small        |            18.43  |             26.545 |               9.92  |          54.895 |       768 |           12 |              256 |            4 |
| medium       |            34.468 |             47.806 |              22.944 |         105.219 |      1024 |           24 |              256 |            4 |
| large        |            51.945 |             70.629 |              51.162 |         173.735 |      1280 |           36 |              256 |            4 |
| xl           |            71.93  |            101.51  |             104.035 |         277.475 |      1600 |           48 |              256 |            4 |
| 2.7B         |            47.719 |            102.562 |             164.967 |         315.247 |      2560 |           32 |              256 |            4 |

# 1.1.6 Profiling Memory

b.

peak memory for 2.7B model with mixed precision 

256 32.6GB 56GB

c.

without mixed precision

256 32.6GB 51.2GB

memory drops several GB for full train loop, and almost same for forward pass only.

d. 

seq_length = 512, batch_size = 4
seq_length * batch_size * d_model = 512 * 4 * 2560 * 4 bytes / 1024^2 = 20MB

e.

for a seq=512, 
The largest allocations is around 128MB, stack trace 
```shell
```

  So score-tensor size:

  - elements = B * Heads * seq_length * seq_length = 4 * 32 * 512 * 512 = 33554432
  - bytes (FP32) = 33554432 * 4 = 134217728
  - MiB = 134217728 / 1024^2 = 128 MB

the biggest allocations at longer context come from attention softmax/score-sized tensors

# 3 Activation Checkpointing
a. 

Like divide and conquer, recursively nest checkpoints, split N into 2 halves and checkpoint each half, recursive down 
the process until each checkpoint has 1 block. This gives O(logN) levels of nesting.

peak memory O(1), during backward pass, only 1 block's intermediates are alive at a time. 
compute cost, each block is recomputed once per nesting layer, so O(NlogN) total. 

```python
from torch.utils.checkpoint import checkpoint

def checkpoint_block(blocks, x):
    if len(blocks) == 1:
        return blocks[0](x)
    mid = len(blocks) // 2
    left = lambda x: checkpoint_block(blocks[:mid], x)
    right = lambda x: checkpoint_block(blocks[mid:], x)
    x = checkpoint(left, x, use_reentrant=False)
    x = checkpoint(right, x, use_reentrant=False)
    return x
```
although in production, deep nested checkpoints leads to excessive recomputation, this is a theoretical analysis.

b. 

suppose k flat checkpoints, so the peak memory is k inputs for forward pass and N/k blocks for backward pass.
O(k + N/k), the math minimum is when k = N/k, where k = sqrt(N), and the peak memory is O(sqrt(N))

split N into sqrt(N) = 6 segments, each segment has 5 blocks. and wrapped in one checkpoint.

# 4. GPU Kernels 

## FlashAttention-2 Benchmarking

Latencies (ms), batch size=1, causal masking, H100. "—" = OOM, SKIP = intentionally skipped.

### Forward pass — Triton wins clearly

| seq_len | Triton fwd (bf16) | Torch fwd (bf16) | Speedup |
|--------:|------------------:|-----------------:|--------:|
| 1024    | 0.019             | 0.078            |      4× |
| 4096    | 0.054             | 0.311            |    5.8× |
| 8192    | 0.103             | 1.150            |     11× |
| 16384   | 0.267             | 4.327            |     16× |
| 32768   | 0.933             | 17.015           |     18× |
| 65536   | 3.642             | OOM              |       - |

Speedup grows with seq_len because Torch materializes the full O(N²) attention matrix in HBM, while the Triton kernel tiles over it and never writes it out.

### E2E (fwd+bwd) — mixed picture

Both implementations use the same PyTorch `_flash_backward` which materializes the full `(N, N)` `S` and `P` matrices — so backward time dominates and largely cancels the forward advantage. Triton e2e is still ~3× faster at seq_len≥8192 because the forward savings compound.

### OOM pattern

- **Torch backward/e2e OOM** at seq_len≥65536 (bfloat16) and ≥32768 (float32): materializes `N×N` attention matrix (~4–16 GB at those sizes).
- **Triton e2e OOM** at seq_len≥65536: the forward is O(N) memory, but the backward reuses the same PyTorch `_flash_backward` which still materializes `N×N`. A tiled Triton backward would fix this.
- **Triton forward succeeds at all sizes** because it is fully tiled and never writes the full attention matrix.

### Full results

| impl   | seq_len | dim | dtype    | fwd_ms  | bwd_ms  | e2e_ms  |
|--------|--------:|----:|----------|--------:|--------:|--------:|
| triton |     128 |  16 | bfloat16 |   0.007 |   0.401 |   0.408 |
| torch  |     128 |  16 | bfloat16 |   0.213 |   0.576 |   0.789 |
| triton |     128 |  16 | float32  |   0.008 |   0.348 |   0.356 |
| torch  |     128 |  16 | float32  |   0.112 |   0.502 |   0.615 |
| triton |     128 |  32 | bfloat16 |   0.007 |   0.364 |   0.371 |
| torch  |     128 |  32 | bfloat16 |   0.167 |   0.909 |   1.076 |
| triton |     128 |  32 | float32  |   0.008 |   0.426 |   0.434 |
| torch  |     128 |  32 | float32  |   0.101 |   2.649 |   2.750 |
| triton |     128 |  64 | bfloat16 |   0.008 |   0.683 |   0.691 |
| torch  |     128 |  64 | bfloat16 |   0.071 |   0.576 |   0.648 |
| triton |     128 |  64 | float32  |   0.010 |   0.315 |   0.325 |
| torch  |     128 |  64 | float32  |   0.060 |   0.530 |   0.591 |
| triton |     128 | 128 | bfloat16 |   0.009 |   0.547 |   0.556 |
| torch  |     128 | 128 | bfloat16 |   0.068 |   1.760 |   1.828 |
| triton |     128 | 128 | float32  |   0.012 |   0.324 |   0.335 |
| torch  |     128 | 128 | float32  |   0.064 |   0.730 |   0.795 |
| triton |     256 |  16 | bfloat16 |   0.008 |   0.439 |   0.447 |
| torch  |     256 |  16 | bfloat16 |   0.165 |   1.671 |   1.835 |
| triton |     256 |  16 | float32  |   0.009 |   0.370 |   0.378 |
| torch  |     256 |  16 | float32  |   0.127 |   0.962 |   1.089 |
| triton |     256 |  32 | bfloat16 |   0.008 |   0.946 |   0.954 |
| torch  |     256 |  32 | bfloat16 |   0.085 |   1.617 |   1.702 |
| triton |     256 |  32 | float32  |   0.010 |   0.711 |   0.722 |
| torch  |     256 |  32 | float32  |   0.070 |   1.902 |   1.972 |
| triton |     256 |  64 | bfloat16 |   0.009 |   0.738 |   0.747 |
| torch  |     256 |  64 | bfloat16 |   0.089 |   1.785 |   1.873 |
| triton |     256 |  64 | float32  |   0.014 |   0.719 |   0.733 |
| torch  |     256 |  64 | float32  |   0.092 |   1.585 |   1.677 |
| triton |     256 | 128 | bfloat16 |   0.010 |   0.689 |   0.699 |
| torch  |     256 | 128 | bfloat16 |   0.076 |   1.313 |   1.389 |
| triton |     256 | 128 | float32  |   0.016 |   0.705 |   0.721 |
| torch  |     256 | 128 | float32  |   0.068 |   1.461 |   1.529 |
| triton |     512 |  16 | bfloat16 |   0.010 |   1.919 |   1.928 |
| torch  |     512 |  16 | bfloat16 |   0.148 |   2.886 |   3.034 |
| triton |     512 |  16 | float32  |   0.011 |   0.303 |   0.315 |
| torch  |     512 |  16 | float32  |   0.067 |   0.741 |   0.808 |
| triton |     512 |  32 | bfloat16 |   0.010 |   0.688 |   0.699 |
| torch  |     512 |  32 | bfloat16 |   0.077 |   1.164 |   1.241 |
| triton |     512 |  32 | float32  |   0.014 |   0.345 |   0.359 |
| torch  |     512 |  32 | float32  |   0.074 |   0.627 |   0.701 |
| triton |     512 |  64 | bfloat16 |   0.012 |   0.730 |   0.742 |
| torch  |     512 |  64 | bfloat16 |   0.080 |   1.652 |   1.732 |
| triton |     512 |  64 | float32  |   0.021 |   0.710 |   0.732 |
| torch  |     512 |  64 | float32  |   0.069 |   1.358 |   1.427 |
| triton |     512 | 128 | bfloat16 |   0.014 |   0.733 |   0.747 |
| torch  |     512 | 128 | bfloat16 |   0.085 |   1.484 |   1.569 |
| triton |     512 | 128 | float32  |   0.025 |   0.711 |   0.736 |
| torch  |     512 | 128 | float32  |   0.068 |   1.496 |   1.564 |
| triton |    1024 |  16 | bfloat16 |   0.014 |   0.704 |   0.718 |
| torch  |    1024 |  16 | bfloat16 |   0.080 |   1.286 |   1.366 |
| triton |    1024 |  16 | float32  |   0.016 |   0.708 |   0.725 |
| torch  |    1024 |  16 | float32  |   0.097 |   2.643 |   2.740 |
| triton |    1024 |  32 | bfloat16 |   0.015 |   0.686 |   0.701 |
| torch  |    1024 |  32 | bfloat16 |   0.078 |   1.416 |   1.493 |
| triton |    1024 |  32 | float32  |   0.023 |   0.703 |   0.727 |
| torch  |    1024 |  32 | float32  |   0.072 |   2.194 |   2.265 |
| triton |    1024 |  64 | bfloat16 |   0.019 |   0.693 |   0.711 |
| torch  |    1024 |  64 | bfloat16 |   0.078 |   1.169 |   1.247 |
| triton |    1024 |  64 | float32  |   0.038 |   0.683 |   0.721 |
| torch  |    1024 |  64 | float32  |   0.073 |   1.723 |   1.796 |
| triton |    1024 | 128 | bfloat16 |   0.022 |   0.693 |   0.715 |
| torch  |    1024 | 128 | bfloat16 |   0.075 |   1.540 |   1.615 |
| triton |    1024 | 128 | float32  |   0.044 |   0.685 |   0.728 |
| torch  |    1024 | 128 | float32  |   0.081 |   1.439 |   1.520 |
| triton |    2048 |  16 | bfloat16 |   0.022 |   0.708 |   0.730 |
| torch  |    2048 |  16 | bfloat16 |   0.103 |   1.613 |   1.716 |
| triton |    2048 |  16 | float32  |   0.027 |   0.705 |   0.732 |
| torch  |    2048 |  16 | float32  |   0.140 |   1.311 |   1.451 |
| triton |    2048 |  32 | bfloat16 |   0.024 |   0.696 |   0.720 |
| torch  |    2048 |  32 | bfloat16 |   0.101 |   2.002 |   2.104 |
| triton |    2048 |  32 | float32  |   0.040 |   0.702 |   0.742 |
| torch  |    2048 |  32 | float32  |   0.142 |   1.214 |   1.356 |
| triton |    2048 |  64 | bfloat16 |   0.031 |   0.764 |   0.795 |
| torch  |    2048 |  64 | bfloat16 |   0.104 |   1.562 |   1.665 |
| triton |    2048 |  64 | float32  |   0.068 |   0.665 |   0.733 |
| torch  |    2048 |  64 | float32  |   0.153 |   1.549 |   1.702 |
| triton |    2048 | 128 | bfloat16 |   0.035 |   0.699 |   0.735 |
| torch  |    2048 | 128 | bfloat16 |   0.103 |   1.768 |   1.871 |
| triton |    2048 | 128 | float32  |   0.080 |   0.643 |   0.723 |
| torch  |    2048 | 128 | float32  |   0.171 |   1.415 |   1.586 |
| triton |    4096 |  16 | bfloat16 |   0.037 |   0.675 |   0.712 |
| torch  |    4096 |  16 | bfloat16 |   0.315 |   1.277 |   1.592 |
| triton |    4096 |  16 | float32  |   0.047 |   0.691 |   0.739 |
| torch  |    4096 |  16 | float32  |   0.479 |   0.942 |   1.421 |
| triton |    4096 |  32 | bfloat16 |   0.041 |   0.675 |   0.715 |
| torch  |    4096 |  32 | bfloat16 |   0.310 |   0.952 |   1.262 |
| triton |    4096 |  32 | float32  |   0.075 |   0.658 |   0.732 |
| torch  |    4096 |  32 | float32  |   0.491 |   1.256 |   1.747 |
| triton |    4096 |  64 | bfloat16 |   0.054 |   0.676 |   0.730 |
| torch  |    4096 |  64 | bfloat16 |   0.311 |   1.390 |   1.701 |
| triton |    4096 |  64 | float32  |   0.129 |   1.111 |   1.240 |
| torch  |    4096 |  64 | float32  |   0.531 |   2.186 |   2.717 |
| triton |    4096 | 128 | bfloat16 |   0.064 |   0.680 |   0.744 |
| torch  |    4096 | 128 | bfloat16 |   0.312 |   1.206 |   1.518 |
| triton |    4096 | 128 | float32  |   0.154 |   0.699 |   0.853 |
| torch  |    4096 | 128 | float32  |   0.616 |   1.192 |   1.808 |
| triton |    8192 |  16 | bfloat16 |   0.069 |   0.885 |   0.954 |
| torch  |    8192 |  16 | bfloat16 |   1.151 |   2.044 |   3.195 |
| triton |    8192 |  16 | float32  |   0.090 |   1.297 |   1.387 |
| torch  |    8192 |  16 | float32  |   1.716 |   3.380 |   5.096 |
| triton |    8192 |  32 | bfloat16 |   0.077 |   0.887 |   0.964 |
| torch  |    8192 |  32 | bfloat16 |   1.147 |   2.045 |   3.193 |
| triton |    8192 |  32 | float32  |   0.200 |   1.425 |   1.625 |
| torch  |    8192 |  32 | float32  |   1.765 |   3.413 |   5.178 |
| triton |    8192 |  64 | bfloat16 |   0.103 |   0.895 |   0.998 |
| torch  |    8192 |  64 | bfloat16 |   1.150 |   2.040 |   3.190 |
| triton |    8192 |  64 | float32  |   0.251 |   1.737 |   1.989 |
| torch  |    8192 |  64 | float32  |   1.914 |   3.677 |   5.592 |
| triton |    8192 | 128 | bfloat16 |   0.122 |   0.916 |   1.038 |
| torch  |    8192 | 128 | bfloat16 |   1.174 |   2.050 |   3.224 |
| triton |    8192 | 128 | float32  |   0.436 |   2.519 |   2.955 |
| torch  |    8192 | 128 | float32  |   2.256 |   4.316 |   6.572 |
| triton |   16384 |  16 | bfloat16 |   0.194 |   3.412 |   3.606 |
| torch  |   16384 |  16 | bfloat16 |   4.283 |   7.782 |  12.065 |
| triton |   16384 |  16 | float32  |   0.257 |   4.866 |   5.123 |
| torch  |   16384 |  16 | float32  |   6.547 |  12.904 |  19.451 |
| triton |   16384 |  32 | bfloat16 |   0.221 |   3.422 |   3.643 |
| torch  |   16384 |  32 | bfloat16 |   4.278 |   7.794 |  12.071 |
| triton |   16384 |  32 | float32  |   0.450 |   5.343 |   5.793 |
| torch  |   16384 |  32 | float32  |   6.694 |  13.091 |  19.785 |
| triton |   16384 |  64 | bfloat16 |   0.267 |   3.549 |   3.817 |
| torch  |   16384 |  64 | bfloat16 |   4.327 |   7.774 |  12.102 |
| triton |   16384 |  64 | float32  |   0.665 |   6.885 |   7.551 |
| torch  |   16384 |  64 | float32  |   7.389 |  14.311 |  21.700 |
| triton |   16384 | 128 | bfloat16 |   0.376 |   3.657 |   4.033 |
| torch  |   16384 | 128 | bfloat16 |   4.357 |   7.820 |  12.177 |
| triton |   16384 | 128 | float32  |   1.717 |   9.824 |  11.541 |
| torch  |   16384 | 128 | float32  |   8.634 |  16.769 |  25.403 |
| triton |   32768 |  16 | bfloat16 |   0.693 |  13.877 |  14.571 |
| torch  |   32768 |  16 | bfloat16 |  17.002 |  30.734 |  47.736 |
| triton |   32768 |  16 | float32  |   0.891 |  19.261 |  20.152 |
| torch  |   32768 |  16 | float32  |  25.612 |       — |       — |
| triton |   32768 |  32 | bfloat16 |   0.768 |  13.774 |  14.542 |
| torch  |   32768 |  32 | bfloat16 |  16.924 |  30.799 |  47.723 |
| triton |   32768 |  32 | float32  |   1.209 |  20.580 |  21.789 |
| torch  |   32768 |  32 | float32  |  26.265 |       — |       — |
| triton |   32768 |  64 | bfloat16 |   0.933 |  14.249 |  15.183 |
| torch  |   32768 |  64 | bfloat16 |  17.015 |  30.766 |  47.782 |
| triton |   32768 |  64 | float32  |   2.613 |  26.967 |  29.579 |
| torch  |   32768 |  64 | float32  |  28.913 |       — |       — |
| triton |   32768 | 128 | bfloat16 |   1.444 |  14.327 |  15.771 |
| torch  |   32768 | 128 | bfloat16 |  17.055 |  30.835 |  47.890 |
| triton |   32768 | 128 | float32  |   6.804 |  40.875 |  47.679 |
| torch  |   32768 | 128 | float32  |  34.701 |       — |       — |
| triton |   65536 |  16 | bfloat16 |    SKIP |    SKIP |    SKIP |
| torch  |   65536 |  16 | bfloat16 |       — |       — |       — |
| triton |   65536 |  16 | float32  |    SKIP |    SKIP |    SKIP |
| torch  |   65536 |  16 | float32  |       — |       — |       — |
| triton |   65536 |  32 | bfloat16 |   2.987 |       — |       — |
| torch  |   65536 |  32 | bfloat16 |       — |       — |       — |
| triton |   65536 |  32 | float32  |   4.797 |       — |       — |
| torch  |   65536 |  32 | float32  |       — |       — |       — |
| triton |   65536 |  64 | bfloat16 |   3.642 |       — |       — |
| torch  |   65536 |  64 | bfloat16 |       — |       — |       — |
| triton |   65536 |  64 | float32  |  10.398 |       — |       — |
| torch  |   65536 |  64 | float32  |       — |       — |       — |
| triton |   65536 | 128 | bfloat16 |   5.919 |       — |       — |
| torch  |   65536 | 128 | bfloat16 |       — |       — |       — |
| triton |   65536 | 128 | float32  |  27.101 |       — |       — |
| torch  |   65536 | 128 | float32  |       — |       — |       — |
