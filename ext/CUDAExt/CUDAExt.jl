module CUDAExt
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

import CUDA: CuArray
import CUDA.CUFFT: plan_bfft

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

plan_fht(A::CuArray, dims) = BFFTPlan(plan_bfft(A, dims))
plan_fht!(A::CuArray, dims) = BFFTPlanInplace(plan_bfft(A, dims))
end