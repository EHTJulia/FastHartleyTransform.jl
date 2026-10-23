using FFTW
using LinearAlgebra
import AbstractFastHartleyTransforms as AFHT

function direct_hartley(x, dims)
    region = dims isa Integer ? (dims,) : Tuple(dims)
    y = similar(x)
    for k in CartesianIndices(x)
        total = zero(promote_type(eltype(x), Float64))
        for n in CartesianIndices(x)
            all(n[d] == k[d] for d in 1:ndims(x) if d ∉ region) || continue
            phase = sum((k[d] - 1) * (n[d] - 1) / size(x, d) for d in region)
            total += x[n] * (cospi(2phase) + sinpi(2phase))
        end
        y[k] = total
    end
    return y
end
