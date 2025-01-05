module FHT
import FFTW
import AbstractFHTs: FHTPlan, 
    plan_fht, plan_fht!, plan_ifht, plan_ifht!,
    fht, fht!, ifht, ifht!,
    *, inv, \,
    eltype, size, length, ndims, 
    fftdims, fftfreqs
export plan_fht, plan_fht!, plan_ifht, plan_ifht!,
    fht, fht!, ifht, ifht!,
    *, inv, \,
    eltype, size, length, ndims, 
    fftdims, fftfreqs

function plan_fht(A, dims; flags=FFTW.ESTIMATE, timelimit=Inf)
    bfftplan = FFTW.plan_bfft(A, dims; flags=flags, timelimit=timelimit)
    return FHTPlan(bfftplan)
end

function plan_fht!(A, dims; flags=FFTW.ESTIMATE, timelimit=Inf)
    bfftplan = FFTW.plan_bfft(A, dims; flags=flags, timelimit=timelimit)
    return FHTPlanInplace(bfftplan)
end

end