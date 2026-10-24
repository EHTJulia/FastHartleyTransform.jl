# Backends and support

| Backend             | Transforms                                                               | Differentiation             |
|:------------------- |:------------------------------------------------------------------------ |:--------------------------- |
| CPU / FFTW          | Float32/Float64; forward/inverse, selected axes, plans, mutation         | Enzyme forward/reverse      |
| NVIDIA GPU / CUDA 5 | Verified on A40; Float32/Float64, selected axes, plans, mutation, `mul!` | Not validated for `CuArray` |
| Reactant CPU        | Float32/Float64; up to four transform axes, plans, mutation              | Forward/reverse             |
| Reactant NVIDIA GPU | Verified on A40; same transform coverage as Reactant CPU                 | Forward/reverse             |
| TPU, AMD, Metal     | Not validated                                                            | Not validated               |

Distributed/sharded FFTs are not validated. See [Performance](@ref) for CPU/GPU measurements.

## CUDA

Install and load CUDA separately to activate the extension. With a working CUDA installation, the interface is:

```julia
using CUDA, FastHartleyTransform
# Keep the input and transformed data on the GPU during repeated execution.
x = CUDA.randn(Float32, 64, 64)
p = plan_fht(x, (1, 2))
y = p * x
restored = inv(p) * y # The inverse plan includes normalization.
```

The extension uses cuFFT backward FFT plans. `CuArray` execution is separate from Reactant compilation.

## Reactant

Convert arrays with `Reactant.to_rarray`, then compile the calculation with `Reactant.@compile`. Shape or transform-axis changes can trigger recompilation. See [Differentiation and compiled execution](@ref) for setup and constraints.

## Choosing a path

Use CPU FFTW for a direct Julia API with reusable plans. Use Reactant when the larger downstream computation can be compiled together. Select a device based on actual execution and measurements on that device.
