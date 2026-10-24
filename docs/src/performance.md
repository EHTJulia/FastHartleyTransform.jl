# Performance

Reuse compatible plans and output buffers for repeated transforms; see [Plans and buffers](@ref). Measure compilation and planning separately from repeated execution.

## CPU and CUDA GPU

Representative forward-transform measurements (FastHartleyTransform 0.1.5, Julia 1.10.12, FFTW 1.10.0, CUDA 5.11.3; AMD EPYC 7543 with four FFTW threads and NVIDIA A40):

Times are median [25th–75th percentile] in milliseconds. Brackets show the central 50% of measured calls, not a confidence interval.

| Type    | Shape           | CPU (ms)              | GPU (ms)                 | Speedup | Speedup with transfers |
|:------- |:--------------- | ---------------------:| ------------------------:| -------:| ----------------------:|
| Float32 | 4,096           | 0.0145 [0.0137, 0.0170] | 0.0595 [0.0587, 0.0606] | 0.24×   | 0.18×                  |
| Float32 | 256 × 256       | 0.472 [0.461, 0.889]    | 0.0657 [0.0648, 0.0669] | 7.19×   | 3.78×                  |
| Float32 | 128 × 128 × 128 | 8.71 [7.64, 9.03]       | 0.446 [0.443, 0.475]    | 19.5×   | 5.70×                  |
| Float64 | 128 × 128 × 128 | 18.0 [17.7, 18.4]       | 1.28 [1.14, 1.33]       | 14.0×   | 5.28×                  |

Measurements use warmed, reusable plans (FFTW `ESTIMATE`; `evals=1`; up to 100 samples or one second per trial). GPU calls are synchronized; compilation and planning are excluded. The GPU column uses resident arrays; the last column includes transfers to and from the CPU. Speedup is the ratio of medians. Results depend on hardware and array size; small arrays can favor the CPU.

The measurement script is `test/gpu/benchmarks.jl`.

## Reactant

Compiled execution with Reactant 0.2.292 and Enzyme 0.13.216, using Float32 arrays of size 64 × 64 × 64:

| Operation | CPU (ms) | A40 GPU (ms) | Speedup |
|:--------- | --------:| ------------:| -------:|
| Transform | 0.765 [0.750, 0.784] | 0.0870 [0.0859, 0.0917] | 8.79× |
| Gradient  | 1.24 [1.21, 1.28] | 0.116 [0.115, 0.120] | 10.7× |

Median [25th–75th percentile] of 100 warmed, synchronized calls; compilation and transfers are excluded. The CPU backend uses its default settings with four CPU cores allocated. The gradient includes the transform and a scalar loss. See `test/reactant/benchmarks.jl` for the workload.
