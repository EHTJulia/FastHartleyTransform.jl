# Differentiation and compiled execution

The transform is real and unnormalized. For axes `dims`, let `M` be the product of their lengths. The default multidimensional Hartley operator `H` has one cosine-plus-sine kernel whose phase sums over those axes. It satisfies `H' = H`, `H * H = M * I`, and `inv(H) = H / M`. Untransformed batch and channel axes do not enter `M`. The `use_r2r=true` option preserves FFTW's separable product of one-dimensional DHTs, which is a different multidimensional operator.

## Enzyme on CPU

Load `Enzyme` and `FastHartleyTransform` to activate the optional EnzymeCore extension supplied by AbstractFastHartleyTransforms 0.1.3. The supported baseline is Julia 1.10.12 and Enzyme 0.13.216 (EnzymeCore 0.8.26). Enzyme is a test dependency, not a mandatory runtime dependency.

The extension supports single-lane forward and reverse differentiation of Float32/Float64 host arrays through `fht`, `ifht`, their mutating variants, fixed plan multiplication, and three-argument `LinearAlgebra.mul!`. Reverse rules apply the same real symmetric operator, including the inverse's normalization. Nonmutating operations accumulate cotangents; mutating operations replace the overwritten input cotangent. Returned aliases and ignored return values are covered by regression tests.

Pass fixed plans as `Enzyme.Const`. The planner is inactive with respect to numerical samples: dimensions, element types, flags, and plan selection are metadata. Prefer planning outside differentiated functions, and use a separate scratch input for destructive FFTW planning modes. Gradients of plan metadata or scalar coefficients stored in a plan are outside this extension's contract; keep differentiable numerical parameters in separate active arguments.

Mixed active and constant broadcasts can require `Enzyme.set_runtime_activity(Enzyme.Reverse)` or the corresponding forward mode. The examples and regression tests enable runtime activity. Provide separate, correctly shaped derivative buffers via `Duplicated`; preserve an original direction vector when a mutating transform overwrites its shadow. Transform inputs that dynamically switch between active arrays and constant arrays are explicitly rejected when runtime activity aliases their primal and shadow buffers; hoist those constant transforms outside differentiation. Batched AD lanes, higher-order differentiation, array views, and CUDA differentiation are not part of the verified baseline. A batch dimension in an ordinary dense array is supported by selecting only the spatial transform axes; it is distinct from batched AD lanes.

## Minimal gradient example

Install Enzyme in the same environment as the concrete package. This example differentiates a scalar loss with respect to the transform input using a fixed plan.

```julia
using Enzyme, FastHartleyTransform

function transform_loss(x, target, plan)
    return sum(abs2, plan * x .- target) / length(x)
end

function loss_and_gradient(x, target, plan)
    dx = zero(x)
    value = transform_loss(x, target, plan)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        transform_loss,
        Enzyme.Active,
        Enzyme.Duplicated(x, dx),
        Enzyme.Const(target),
        Enzyme.Const(plan),
    )
    return value, dx
end

x = collect(1.0:8.0)
target = fill(0.2, 8)
plan = plan_fht(x)
value, dx = loss_and_gradient(x, target, plan)
expected = (2 / length(x)) .* (plan * (plan * x .- target))
@assert dx ≈ expected
```

Reset derivative buffers before independent reverse evaluations; the helper creates a fresh buffer for each call.

## Reactant

Load `Reactant` and `FastHartleyTransform` to activate the separate Reactant extension. Its supported baseline is Reactant 0.2.292 and Enzyme 0.13.216 on Julia 1.10.12.

The extension accepts traced and concrete Reactant Float32/Float64 arrays. It implements transforms and fixed plans using FFT operations that Reactant lowers to its compiler, with no FFTW handle in the compiled program. `real(fft(x)) - imag(fft(x))` gives the same sign convention as the CPU backward-FFT implementation. For more than three transform axes, the complex FFT is composed in groups and the Hartley result is formed only at the end. Applying real Hartley transforms to each group would change the numerical operator.

Use `Reactant.to_rarray` outside the compiled workload and `Reactant.@compile` for the complete transform/loss/gradient computation. Plans store shape, axes, and transform convention as static metadata and may be created before or during tracing. Shape or axis changes require recompilation. A Reactant plan rejects ordinary host arrays; convert inputs with `to_rarray` before compiled execution. Only `FFTW.ESTIMATE` and the default infinite planning time limit are accepted; those FFTW options have no analogous tuning role in Reactant. Mutating APIs preserve their public input/output semantics, but the compiler chooses physical buffer reuse and they do not promise zero allocation.

Reactant CPU and NVIDIA A40 GPU execution are verified for Float32/Float64 transforms through four axes, selected axes, normal and mutating calls, fixed plans, `mul!`, and forward/reverse gradients. The separate CUDA extension supports `CuArray` transforms; its differentiation is not validated. See [Backends and support](@ref) for other devices.

The helper above can also be compiled as a whole. After loading Reactant, create a Reactant plan from Reactant arrays rather than passing the native FFTW plan into the compiler:

```julia
using Reactant
Reactant.set_default_backend("cpu") # Use "gpu" for an NVIDIA GPU.
rx, rt = Reactant.to_rarray.((x, target))
rplan = plan_fht(rx)
executable = Reactant.@compile loss_and_gradient(rx, rt, rplan)
rvalue, rdx = executable(rx, rt, rplan)
@assert Array(rdx) ≈ dx
```

Enzyme rules for Julia/FFTW execution and differentiation of Reactant-compiled FFT operations are separate mechanisms. Do not assume that an EnzymeCore rule is automatically used inside a Reactant compilation.
