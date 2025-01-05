using FHT
using StableRNGs
using Test

@testset "FHT.jl" begin
    rng = StableRNG(0)
    x1d = rand(rng, Float64, 128)
    x2d = rand(rng, Float64, 128, 128)
    x3d = rand(rng, Float64, 16, 16, 16)
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
        y3 = fht(x)
        y4 = fht!(deepcopy(x))
        @test y1 == y2
        @test y1 == y3
        @test y1 == y4

        # check inverse of inverse is the same
        x1 = ip1 * y1
        x2 = ip2 * deepcopy(y1)
        x3 = ip3 * y1 
        x4 = ip4 * deepcopy(y1)
        x5 = ifht(y1) 
        x6 = ifht!(deepcopy(y1))
        @test x ≈ x1
        @test x ≈ x2
        @test x ≈ x3
        @test x ≈ x4
        @test x ≈ x5
        @test x ≈ x6
        @test x1 == x2
        @test x1 == x3
        @test x1 == x4
        @test x1 == x5
        @test x1 == x6

        # check methods
        for p in (p1, p2)
            @test eltype(p) == eltype(p.bfftplan)
            @test ndims(p) == ndims(p.bfftplan)
            @test length(p) == length(p.bfftplan)
            @test fftdims(p) == fftdims(p.bfftplan)
        end

        for p in (ip1,ip2,ip3,ip4)
            @test eltype(p) == eltype(p1)
            @test ndims(p) == ndims(p1)
            @test length(p) == length(p1)
            @test fftdims(p) == fftdims(p1)
        end
    end
end
