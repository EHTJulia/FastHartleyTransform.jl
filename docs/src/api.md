```@meta
CurrentModule = AbstractFastHartleyTransforms
```

# API reference

The transform functions below are reexported from AbstractFastHartleyTransforms. CPU-specific planner keywords include `use_r2r`; see [Plans and buffers](@ref) and [Transform conventions](@ref). Backend availability is described in [Backends and support](@ref).

```@autodocs
Modules = [AbstractFastHartleyTransforms]
```

Plans also implement `size`, `ndims`, `length`, `eltype`, `fftdims`, multiplication, inversion, and three-argument `LinearAlgebra.mul!`. Not every FFT plan operation is part of this interface: for example, do not assume five-argument `mul!` or a generic `adjoint(plan)` implementation. The mathematical adjoint is explained in [Transform conventions](@ref).
