using FastHartleyTransform
using StableRNGs
using Test

function check_inference(p, x)
    @test (@inferred p * copy(x)) isa typeof(x)
end

function check_inv(xinv, xorg, xinvref)
    @test xorg ≈ xinv
    @test xinvref == xinv
end

@testset "FastHartleyTransform.jl" begin
    rng = StableRNG(0)
    x1d = rand(rng, Float64, 128)
    x2d = rand(rng, Float64, 128, 128)
    x3d = rand(rng, Float64, 16, 16, 16)
    for x in (x1d, x2d, x3d)
        # Create forward FHT plans.
        p1 = plan_fht(x)
        p2 = plan_fht!(x)

        # Create inverse FHT plans.
        ip1 = plan_ifht(x)
        ip2 = plan_ifht!(x)
        ip3 = inv(p1)
        ip4 = inv(p2)

        # Check that the forward FHT plans agree.
        y1 = p1 * x
        @test y1 == p2 * deepcopy(x)
        @test y1 == fht(x)
        @test y1 == fht!(deepcopy(x))

        # Check that inverse transforms recover the input.
        x1 = ip1 * y1
        check_inv(x1, x, x1)
        check_inv(ip2 * deepcopy(y1), x, x1)
        check_inv(ip3 * y1, x, x1)
        check_inv(ip4 * deepcopy(y1), x, x1)
        check_inv(ifht(y1), x, x1)
        check_inv(ifht!(deepcopy(y1)), x, x1)
        check_inv(p1 \ y1, x, x1)
        check_inv(p2 \ deepcopy(y1), x, x1)

        for p in (p1, p2, ip1, ip2, ip3, ip4)
            check_inference(p, x)
        end

        # Check plan metadata.
        @test eltype(p1) == eltype(p2)
        @test ndims(p1) == ndims(p2)
        @test length(p1) == length(p2)
        @test fftdims(p1) == fftdims(p2)

        for p in (ip1, ip2, ip3, ip4)
            @test eltype(p) == eltype(p1)
            @test ndims(p) == ndims(p1)
            @test length(p) == length(p1)
            @test fftdims(p) == fftdims(p1)
        end
    end
end

include("numerics.jl")

include("enzyme.jl")
