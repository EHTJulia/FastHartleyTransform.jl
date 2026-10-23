include("reference.jl")

function check_numerics(T, shape, dims)
    x = reshape(T.(sin.(1:prod(shape))), shape)
    v = reshape(T.(cos.(1:prod(shape))), shape)
    original = copy(x)
    reference = direct_hartley(x, dims)
    region = dims isa Integer ? (dims,) : Tuple(dims)
    m = prod(size(x, d) for d in region)
    for planner in (plan_fht, plan_fht!)
        p = planner(copy(x), dims)
        ip = inv(p)
        @test p * copy(x) ≈ reference
        @test ip * copy(reference) ≈ x
        @test p * (p * copy(x)) ≈ m .* x
        @test dot(p * copy(x), v) ≈ dot(x, p * copy(v))
        @test inv(ip) * copy(x) ≈ reference
        @test fftdims(p) == fftdims(ip)
        @test size(p) == shape
        input = copy(x)
        result = p * input
        @test (result === input) == AFHT.is_inplace(p)
        @test x == original
        y = similar(x)
        @test mul!(y, p, x) ≈ reference
        @test x == original
        @test mul!(y, ip, reference) ≈ x
        sp = T(0.75) * p
        @test sp * copy(x) ≈ T(0.75) .* reference
        @test inv(sp) * (sp * copy(x)) ≈ x
    end
    @test ifht(reference, dims) ≈ x
    @test ifht!(copy(reference), dims) ≈ x
    @test fht(x, dims) ≈ reference
    @test fht!(copy(x), dims) ≈ reference
end

@testset "Hartley definition, axes, normalization, and adjoints" begin
    cases = (
        ((5,), 1),
        ((6,), (1,)),
        ((3, 4), (1, 2)),
        ((3, 4), 2),
        ((3, 2, 4), (1, 3)),
        ((3, 2, 4), (3, 1)),
        ((2, 2, 2, 2), 1:4),
        ((1, 3), (1, 2)),
    )
    for T in (Float32, Float64), (shape, dims) in cases
        check_numerics(T, shape, dims)
    end
    x = reshape(sin.(1:12), 3, 4)
    p = AFHT.R2RPlan(FFTW.plan_r2r(x, FFTW.DHT, (1, 2)))
    q = AFHT.R2RPlanInplace(FFTW.plan_r2r!(copy(x), FFTW.DHT, (1, 2)))
    reference = direct_hartley(direct_hartley(x, (1,)), (2,))
    @test p * x ≈ reference
    @test q * copy(x) ≈ reference
    @test !(reference ≈ direct_hartley(x, (1, 2)))
    @test inv(p) * reference ≈ x
end
