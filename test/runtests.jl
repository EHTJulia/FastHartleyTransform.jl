using FHT
using StableRNGs
using Test

@testset "FHT.jl" begin
    rng = StableRNG(0)
    x1d = rand(rng, Float64, 128)
    x2d = rand(rng, Float64, 128, 128)

    for x in (x1d, x2d)
        # make FHT plan
        p1 = plan_fht(x)
        p2 = plan_fht!(x)

        # check if the output is the same
        y = p1 * x
        @test y == p2 * deepcopy(x)

        # make inverse FHT plan
        ip1 = plan_ifht(x)
        ip2 = plan_ifht!(x)
        ip3 = inv(p1)
        ip4 = inv(p2)

        # make inverse FHT plan inplace
        x2 = ip1 * y
        @test x == x2
        
        # check if the plans are the same
        @test x == ip3 * y 
        @test x == ip2 * deepcopy(y)
        @test x == ip4 * deepcopy(y)
    end
end
