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
/root/assignment2-systems/cs336-basics/cs336_basics/nn_utils.py:7:softmax
/root/assignment2-systems/cs336-basics/cs336_basics/model.py:430:scaled_dot_product_attention
/root/assignment2-systems/cs336-basics/cs336_basics/model.py:516:forward
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
