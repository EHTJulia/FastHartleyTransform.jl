# Differentiation and compiled execution

Use Enzyme to compute gradients of functions that call Hartley transforms. Use Reactant to compile the same calculation for your chosen backend. See [Backends and support](@ref) for device support.

## Enzyme on CPU

Install Enzyme in the same Julia environment as FastHartleyTransform, then load both packages. The example below computes a scalar loss and its gradient with respect to `x`. Reverse mode computes one derivative for each element of `x` in a single call.

### Minimal gradient example

Create a reusable plan, transform the input, and compare the result with a fixed target:

```julia
using Enzyme, FastHartleyTransform

function transform_loss(x, target, plan)
    # Compare transformed values with the target using mean squared error.
    return sum(abs2, plan * x .- target) / length(x)
end

function loss_and_gradient(x, target, plan)
    # Enzyme adds gradients into dx, so start with a fresh zero buffer.
    dx = zero(x)
    value = transform_loss(x, target, plan)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        transform_loss,
        Enzyme.Active,
        Enzyme.Duplicated(x, dx), # Differentiate x and write its gradient to dx.
        Enzyme.Const(target), # Keep the target and plan fixed.
        Enzyme.Const(plan),
    )
    return value, dx
end

x = collect(1.0:8.0)
target = fill(0.2, 8)
# Build the plan once, outside the function being differentiated.
plan = plan_fht(x)
value, dx = loss_and_gradient(x, target, plan)
```

`value` is the loss; `dx[i]` describes how that loss changes when `x[i]` changes. The Enzyme argument annotations specify what to differentiate:

| Annotation | Meaning in this example |
|:---------- |:----------------------- |
| `Active` | Differentiate the scalar return value. |
| `Duplicated(x, dx)` | Differentiate `x` and accumulate its gradient into `dx`. |
| `Const(target)`, `Const(plan)` | Treat these arguments as fixed. |

`set_runtime_activity` lets Enzyme handle the mix of differentiable and fixed arrays in this calculation. The helper creates a fresh gradient buffer for each evaluation.

### Using other transform calls

The same setup supports `fht`, `ifht`, fixed plans, their mutating variants, and three-argument `mul!` on dense `Float32`/`Float64` CPU arrays. The inverse's normalization is included automatically.

Start with nonmutating calls. When using a mutating call repeatedly, restore any overwritten input and reset its gradient buffer before each independent evaluation. Keep plans fixed; pass any numerical parameters you want to differentiate as separate arguments with their own gradient buffers.

For advanced AD restrictions, see the [differentiation contract](https://EHTJulia.github.io/AbstractFastHartleyTransforms.jl/dev/differentiation/). For transform definitions and normalization, see [Transform conventions](@ref).

## Reactant

Install Reactant in the same environment. Reuse `loss_and_gradient` from above, convert the inputs to Reactant arrays, and build a new plan from those arrays. Then compile the whole loss-and-gradient calculation:

```julia
using Reactant

Reactant.set_default_backend("cpu") # Use "gpu" for an NVIDIA GPU.
# Convert inputs and create a plan for the selected backend.
rx, rt = Reactant.to_rarray.((x, target))
rplan = plan_fht(rx)
# Compile once for these shapes, element types, and transform axes.
executable = Reactant.@compile loss_and_gradient(rx, rt, rplan)
# Reuse the executable for subsequent evaluations.
rvalue, rdx = executable(rx, rt, rplan)
# Copy the gradient back to the CPU to compare it with the Enzyme result.
@assert Array(rdx) ≈ dx
```

Keep arrays on the selected backend for repeated execution. Changes to shapes, element types, or transform axes can require recompilation. Use a plan built from Reactant arrays; an existing CPU FFTW plan cannot be passed into the compiled calculation.
