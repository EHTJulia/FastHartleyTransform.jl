# FastHartleyTransform.jl

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Fast real-to-real Hartley transforms for Julia, backed by FFTW on CPU, with optional CUDA and Reactant extensions.

## Start here

  - [Getting started](@ref): installation, forward/inverse transforms, and spatial axes.
  - [Transform conventions](@ref): the kernel, normalization, and multidimensional behavior.
  - [Plans and buffers](@ref): reuse plans and control input mutation.
  - [Differentiation and compiled execution](@ref): Enzyme and Reactant usage.
  - [Backends and support](@ref): verified execution paths and device limitations.
  - [Performance](@ref): plan reuse, warmup, allocations, and timing.
  - [API reference](@ref): exported transform and planning functions.

## A first transform

```@example home
using FastHartleyTransform
x = [1.0, 2.0, 3.0, 4.0]
y = fht(x)
@assert ifht(y) ≈ x
y
```

`fht` is unnormalized; `ifht` includes the inverse scale. The default multidimensional transform uses one cosine-plus-sine kernel, rather than a product of one-dimensional Hartley kernels. See [Transform conventions](@ref) before substituting another DHT implementation.

The common interface and optional CPU Enzyme rules are supplied by [AbstractFastHartleyTransforms.jl](https://EHTJulia.github.io/AbstractFastHartleyTransforms.jl/dev/). Install this package to obtain a concrete transform implementation.
