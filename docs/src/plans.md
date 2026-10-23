# Plans and buffers

A plan fixes the input shape, element type, axes, and transform convention. Create it once and reuse it for compatible arrays.

```@example plans
using FastHartleyTransform, LinearAlgebra
x = reshape(sin.(1:20), 4, 5)
p = plan_fht(x, (1, 2))
y = p * x
@assert inv(p) * y ≈ x
@assert p \ y ≈ x
@assert size(p) == size(x)
@assert Tuple(fftdims(p)) == (1, 2)
out = similar(x)
mul!(out, p, x)
@assert out ≈ y
nothing # hide
```

| Operation                                    | Behavior                                               |
|:-------------------------------------------- |:------------------------------------------------------ |
| `fht(x, dims)` / `ifht(x, dims)`             | Plan and return a new transformed array                |
| `fht!(x, dims)` / `ifht!(x, dims)`           | Plan and overwrite `x`                                 |
| `plan_fht(x, dims)` / `plan_ifht(x, dims)`   | Construct a reusable nonmutating plan                  |
| `plan_fht!(x, dims)` / `plan_ifht!(x, dims)` | Construct a reusable mutating plan                     |
| `p * x`                                      | Execute the plan; input mutation follows the plan kind |
| `mul!(out, p, x)`                            | Write to the supplied output buffer                    |
| `inv(p)` / `p \ x`                           | Invert the plan / apply its inverse                    |

For three-argument `mul!`, separate input and output arrays and exact same-buffer use are tested. Partially overlapping views are outside the verified contract. An output buffer avoids allocating a new real result, but FFT workspaces can still allocate. A mutating API does not imply an allocation-free implementation.

```@example plans
scratch = copy(x)
p! = plan_fht!(scratch, (1, 2))
@assert p! * scratch === scratch
@assert scratch ≈ y
nothing # hide
```

## CPU planner options

Use a plan when setting FFTW options; the convenience transform calls do not forward planner keywords.

```@example planning
using FastHartleyTransform, FFTW
x = reshape(sin.(1:64), 8, 8)
p = plan_fht(copy(x), (1, 2); flags = FFTW.ESTIMATE, use_r2r = false)
@assert p * x ≈ fht(x)
nothing # hide
```

The default is `flags=FFTW.ESTIMATE` and `timelimit=Inf`. More expensive FFTW planning modes can overwrite their planning input: use a scratch array and keep the original data elsewhere. Treat array layout and strides as part of plan compatibility. Do not concurrently share mutable work buffers without establishing backend-specific safety.

Reactant plans store static metadata, rather than FFTW handles, and accept only the default planning options. CUDA's planner does not expose the CPU keyword options through this interface. See [Backends and support](@ref).
