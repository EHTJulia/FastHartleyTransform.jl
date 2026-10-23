# Performance

Reuse compatible plans and output buffers for repeated transforms; see [Plans and buffers](@ref). Measure compilation and planning separately from repeated execution.

## CPU and CUDA GPU

Representative forward-transform measurements (Julia 1.10.12, FFTW 1.10.0, CUDA 5.11.3; AMD EPYC 7543 with four FFTW threads and NVIDIA A40):

| Type    | Shape           | CPU (ms) | GPU (ms) | Speedup | Speedup with transfers |
|:------- |:--------------- | --------:| --------:| -------:| ----------------------:|
| Float32 | 4,096           | 0.0407   | 0.0608   | 0.67×   | 0.50×                  |
| Float32 | 256 × 256       | 1.476    | 0.0635   | 23.2×   | 12.0×                  |
| Float32 | 128 × 128 × 128 | 9.347    | 0.363    | 25.8×   | 6.20×                  |
| Float64 | 128 × 128 × 128 | 19.960   | 1.137    | 17.6×   | 5.95×                  |

Times are warmed medians with reusable plans (FFTW `ESTIMATE`; up to 100 samples or one second per trial). GPU calls are synchronized; compilation and planning are excluded. The GPU column uses resident arrays; the last column includes transfers to and from the CPU. Speedup is CPU time divided by GPU time. Results depend on hardware and array size; small arrays can favor the CPU.

The measurement script is `test/gpu/benchmarks.jl`.

## Reactant

Compiled execution with Reactant 0.2.292 and Enzyme 0.13.216, using Float32 arrays of size 64 × 64 × 64:

| Operation | CPU (ms) | A40 GPU (ms) | Speedup |
|:--------- | --------:| ------------:| -------:|
| Transform | 0.750    | 0.0859       | 8.73×   |
| Gradient  | 1.205    | 0.1159       | 10.4×   |

Medians of 100 warmed, synchronized calls; compilation and transfers are excluded. The CPU backend uses its default settings with four CPU cores allocated. The gradient includes the transform and a scalar loss. See `test/reactant/benchmarks.jl` for the workload.
