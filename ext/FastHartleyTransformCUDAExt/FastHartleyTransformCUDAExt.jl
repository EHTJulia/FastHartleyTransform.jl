module FastHartleyTransformCUDAExt
using PrecompileTools: @setup_workload, @compile_workload
using CUDA: CUDA

import AbstractFastHartleyTransforms:
    BFFTPlan,
    BFFTPlanInplace,
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

plan_fht(A::CUDA.CuArray, dims) = BFFTPlan(CUDA.CUFFT.plan_bfft(A, dims))
plan_fht!(A::CUDA.CuArray, dims) = BFFTPlanInplace(CUDA.CUFFT.plan_bfft(A, dims))

@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    x_list = (CUDA.rand(Float32, 32), CUDA.rand(Float32, 32, 32))
    @compile_workload begin
        for x in x_list
            ifht(fht(x))
            fht!(x)
            ifht!(x)
        end
    end
end

end