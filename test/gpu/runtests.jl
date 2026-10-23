using Test, CUDA, FFTW, LinearAlgebra, FastHartleyTransform

include("../reference.jl")

function check_cuda(T, shape, dims)
    x = reshape(T.(sin.(1:prod(shape))), shape)
    reference = direct_hartley(x, dims)
    m = prod(size(x, d) for d in dims)
    tolerance = 100eps(T)
    for transform in (fht, fht!, ifht, ifht!)
        gx = CuArray(x)
        result = transform(gx, dims)
        scale = transform in (ifht, ifht!) ? inv(T(m)) : one(T)
        @test isapprox(Array(result), scale .* reference; rtol=tolerance, atol=tolerance)
        @test Array(gx) ≈ (transform in (fht!, ifht!) ? Array(result) : x)
    end
    for planner in (plan_fht, plan_fht!, plan_ifht, plan_ifht!)
        gx = CuArray(x)
        p = planner(gx, dims)
        scale = planner in (plan_ifht, plan_ifht!) ? inv(T(m)) : one(T)
        expected = scale .* reference
        @test isapprox(Array(p * copy(gx)), expected; rtol=tolerance, atol=tolerance)
        y = similar(gx)
        @test isapprox(
            Array(mul!(y, p, copy(gx))), expected; rtol=tolerance, atol=tolerance
        )
        @test Array(inv(p) * (p * copy(gx))) ≈ x
    end
    @test Array(ifht(fht(CuArray(x), dims), dims)) ≈ x
    return nothing
end

function main()
    CUDA.functional(true) || error("A working CUDA device is required.")
    CUDA.allowscalar(false)
    CUDA.versioninfo()
    @testset "CUDA Hartley transforms" begin
        for T in (Float32, Float64),
            (shape, dims) in (
                ((5,), (1,)),
                ((8,), (1,)),
                ((3, 4), (1, 2)),
                ((3, 4), (2,)),
                ((2, 3, 4), (1, 3)),
            )

            check_cuda(T, shape, dims)
        end
    end
    return nothing
end

main()
