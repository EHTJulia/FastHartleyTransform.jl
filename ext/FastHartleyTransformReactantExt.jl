module FastHartleyTransformReactantExt

using Reactant: AnyTracedRArray, AnyConcreteRArray
using FFTW: fft, ESTIMATE
using LinearAlgebra: LinearAlgebra
import AbstractFastHartleyTransforms:
    DHTPlan,
    ScaledDHTPlan,
    plan_fht,
    plan_fht!,
    plan_ifht,
    plan_ifht!,
    plan_inv,
    is_inplace,
    fftdims
import Base: *, size

const ReactantArray{T,N} = Union{AnyTracedRArray{T,N},AnyConcreteRArray{T,N}}

# Plans contain only static metadata, so no FFTW handle enters a compiled program.
struct ReactantHartleyPlan{T,N,D,Inplace,Separable} <: DHTPlan{T}
    shape::NTuple{N,Int}
    region::D
end
size(p::ReactantHartleyPlan) = p.shape
fftdims(p::ReactantHartleyPlan) = p.region
is_inplace(::ReactantHartleyPlan{T,N,D,I}) where {T,N,D,I} = I

function make_plan(
    x::ReactantArray{T,N},
    dims,
    inplace;
    flags=ESTIMATE,
    timelimit=Inf,
    use_r2r::Bool=N == 1,
) where {T,N}
    T <: Union{Float32,Float64} ||
        throw(ArgumentError("Reactant FHT requires Float32 or Float64."))
    flags == ESTIMATE || throw(ArgumentError("Reactant plans support only ESTIMATE flags."))
    timelimit == Inf ||
        throw(ArgumentError("Reactant plans do not support FFTW time limits."))
    region = dims isa Integer ? (Int(dims),) : Tuple(Int(d) for d in dims)
    all(d -> 1 <= d <= N, region) || throw(ArgumentError("Invalid transform axes."))
    allunique(region) || throw(ArgumentError("Transform axes must be unique."))
    return ReactantHartleyPlan{T,N,typeof(region),inplace,use_r2r}(size(x), region)
end

plan_fht(x::ReactantArray, dims; kwargs...) = make_plan(x, dims, false; kwargs...)
plan_fht!(x::ReactantArray, dims; kwargs...) = make_plan(x, dims, true; kwargs...)

function plan_inv(p::ReactantHartleyPlan{T}) where {T}
    m = prod(d -> size(p, d), fftdims(p); init=1)
    return ScaledDHTPlan(p, inv(T(m)))
end
plan_ifht(x::ReactantArray, dims; kwargs...) = inv(plan_fht(x, dims; kwargs...))
plan_ifht!(x::ReactantArray, dims; kwargs...) = inv(plan_fht!(x, dims; kwargs...))

function hartley(p::ReactantHartleyPlan{T,N,D,I,S}, x) where {T,N,D,I,S}
    size(x) == size(p) ||
        throw(DimensionMismatch("Input shape does not match the Hartley plan."))
    isempty(p.region) && return copy(x)
    if S
        y = x
        for d in p.region
            fx = fft(y, (d,))
            y = real.(fx) .- imag.(fx)
        end
        return y
    end
    # Keep intermediate results complex: successive real Hartley transforms would
    # produce the separable convention instead of the multidimensional transform.
    fx = fft(x, p.region[1:min(3, length(p.region))])
    for first_axis in 4:3:length(p.region)
        group = p.region[first_axis:min(first_axis + 2, length(p.region))]
        fx = fft(fx, group)
    end
    return real.(fx) .- imag.(fx)
end

function *(p::ReactantHartleyPlan, x::AbstractArray)
    throw(
        ArgumentError(
            "Reactant Hartley plans require Reactant arrays; use to_rarray and compiled execution.",
        ),
    )
end

function *(p::ReactantHartleyPlan, x::ReactantArray)
    y = hartley(p, x)
    if is_inplace(p)
        x .= y
        return x
    end
    return y
end

function LinearAlgebra.mul!(y::ReactantArray, p::ReactantHartleyPlan, x::ReactantArray)
    y .= hartley(p, x)
    return y
end

function *(p::ScaledDHTPlan{T,P}, x::ReactantArray) where {T,P<:ReactantHartleyPlan}
    y = p.p * x
    y .*= p.scale
    return y
end

function LinearAlgebra.mul!(
    y::ReactantArray, p::ScaledDHTPlan{T,P}, x::ReactantArray
) where {T,P<:ReactantHartleyPlan}
    y .= p.scale .* hartley(p.p, x)
    return y
end

end
