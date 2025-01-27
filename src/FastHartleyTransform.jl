module FastHartleyTransform
using PrecompileTools: @setup_workload, @compile_workload
using FFTW: ESTIMATE, plan_r2r, plan_r2r!, plan_bfft, DHT

import AbstractFastHartleyTransforms:
    BFFTPlan,
    BFFTPlanInplace,
    R2RPlan,
    R2RPlanInplace,
    plan_fht,
    plan_fht!,
    plan_ifht,
    plan_ifht!,
    fht,
    fht!,
    ifht,
    ifht!,
    *,
    inv,
    \,
    eltype,
    size,
    length,
    ndims,
    fftdims

export plan_fht,
    plan_fht!,
    plan_ifht,
    plan_ifht!,
    fht,
    fht!,
    ifht,
    ifht!,
    *,
    inv,
    \,
    eltype,
    size,
    length,
    ndims,
    fftdims

function plan_fht(
    A::Array,
    dims;
    flags=ESTIMATE,
    timelimit=Inf,
    use_r2r::Bool=ndims(A) == 1 ? true : false,
)
    if use_r2r
        if ndims(A) > 1
            @warn "Multidimensional r2r transform does not provide the exact transform and likely be slower."
        end
        r2rplan = plan_r2r(A, DHT, dims; flags=flags, timelimit=timelimit)
        return R2RPlan(r2rplan)
    else
        bfftplan = plan_bfft(A, dims; flags=flags, timelimit=timelimit)
        return BFFTPlan(bfftplan)
    end
end

function plan_fht!(
    A::Array,
    dims;
    flags=ESTIMATE,
    timelimit=Inf,
    use_r2r::Bool=ndims(A) == 1 ? true : false,
)
    if use_r2r
        if ndims(A) > 1
            @warn "Multidimensional r2r transform does not provide the exact transform and likely be slower."
        end
        r2rplan = plan_r2r!(A, DHT, dims; flags=flags, timelimit=timelimit)
        return R2RPlanInplace(r2rplan)
    else
        bfftplan = plan_bfft(A, dims; flags=flags, timelimit=timelimit)
        return BFFTPlanInplace(bfftplan)
    end
end

@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    x_list = (rand(Float64, 128), rand(Float64, 128, 128))
    @compile_workload begin
        for x in x_list
            ifht(fht(x))
            fht!(x)
            ifht!(x)
        end
    end
end

end
