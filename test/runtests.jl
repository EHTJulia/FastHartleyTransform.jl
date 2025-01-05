using FHT
using StableRNGs
using Test

@testset "FHT.jl" begin
    rng = StableRNG(0)
    x1d = rand(rng, Float64, 128)
    x2d = rand(rng, Float64, 128, 128)

    for x in (x1d, x2d)
        # make FHT plans
        p1 = plan_fht(x)
        p2 = plan_fht!(x)

        # make inverse FHT plans
        ip1 = plan_ifht(x)
        ip2 = plan_ifht!(x)
        ip3 = inv(p1)
        ip4 = inv(p2)

        # check if the forward FPT plans are consistent
        y1 = p1 * x
        y2 = p2 * deepcopy(x)
        @test y1 == y2

        # check inverse of inverse is the same
        x1 = ip1 * y1
        x2 = ip2 * deepcopy(y1)
        x3 = ip3 * y1 
        x4 = ip4 * deepcopy(y1)
        @test x ≈ x1
        @test x ≈ x2
        @test x ≈ x3
        @test x ≈ x4
    end
end
