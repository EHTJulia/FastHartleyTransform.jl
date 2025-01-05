module FastHartleyTransform
import FFTW: plan_bfft, ESTIMATE
import AbstractFastHartleyTransforms:
    FHTPlan,
    FHTPlanInplace,
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

function plan_fht(A, dims; flags=ESTIMATE, timelimit=Inf)
    bfftplan = plan_bfft(A, dims; flags=flags, timelimit=timelimit)
    return FHTPlan(bfftplan)
end

function plan_fht!(A, dims; flags=ESTIMATE, timelimit=Inf)
    bfftplan = plan_bfft(A, dims; flags=flags, timelimit=timelimit)
    return FHTPlanInplace(bfftplan)
end

end
