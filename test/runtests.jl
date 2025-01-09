using FastHartleyTransform
using StableRNGs
using Test

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
        # make FastHartleyTransform plans
        p1 = plan_fht(x)
        p2 = plan_fht!(x)

        # make inverse FastHartleyTransform plans
        ip1 = plan_ifht(x)
        ip2 = plan_ifht!(x)
        ip3 = inv(p1)
        ip4 = inv(p2)

        # check if the forward FPT plans are consistent
        y1 = p1 * x
        @test y1 == p2 * deepcopy(x)
        @test y1 == fht(x)
        @test y1 == fht!(deepcopy(x))

        # check inverse of inverse is the same
        x1 = ip1 * y1
        check_inv(x1, x, x1)
        check_inv(ip2 * deepcopy(y1), x, x1)
        check_inv(ip3 * y1, x, x1)
        check_inv(ip4 * deepcopy(y1), x, x1)
        check_inv(ifht(y1), x, x1)
        check_inv(ifht!(deepcopy(y1)), x, x1)
        check_inv(p1 \ y1, x, x1)
        check_inv(p2 \ deepcopy(y1), x, x1)

        # check methods
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
