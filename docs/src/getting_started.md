# Getting started

## Installation

In a Julia environment:

```julia
using Pkg
Pkg.add("FastHartleyTransform")
```

Julia 1.10 or later is required. The maintenance baseline is Julia 1.10.12 LTS. The Enzyme and Reactant APIs described here target AbstractFastHartleyTransforms 0.1.3 and FastHartleyTransform 0.1.2. Check the installed versions before using these optional extensions.

## Forward and inverse transforms

CPU examples use dense `Float32` or `Float64` arrays. Nonmutating calls preserve their input:

```@example basics
using FastHartleyTransform
x = reshape(collect(1.0:12.0), 3, 4)
original = copy(x)
y = fht(x)
@assert x == original
@assert ifht(y) ≈ x
size(y)
```

A name ending in `!` overwrites its input:

```@example basics
scratch = copy(x)
@assert fht!(scratch) === scratch
@assert scratch ≈ y
ifht!(scratch)
@assert scratch ≈ x
nothing # hide
```

## Spatial axes and batches

Without `dims`, all axes are transformed. Select only spatial axes when other axes represent channels or independent samples:

```@example axes
using FastHartleyTransform
x = reshape(sin.(1:120), 4, 5, 2, 3)
dims = (1, 2)
y = fht(x, dims)
@assert size(y) == (4, 5, 2, 3)
@assert ifht(y, dims) ≈ x
@assert y[:, :, 1, 2] ≈ fht(x[:, :, 1, 2])
size(y)
```

Here each of the six spatial slices is independent. The inverse scale is `1/(4*5)`, with no contribution from the channel and batch axes.

For repeated transforms, use [Plans and buffers](@ref). For automatic differentiation, continue with [Differentiation and compiled execution](@ref).
