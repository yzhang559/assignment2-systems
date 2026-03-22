# Benchmark results

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