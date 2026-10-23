using Test, Reactant, Enzyme, FFTW, LinearAlgebra, FastHartleyTransform
include("../reference.jl")

planned_transform(p, x) = p * x
function buffered_transform(p, x)
    y = similar(x)
    mul!(y, p, x)
    return y
end

function residual_loss(x, target, dims, inplace)
    m = prod(size(x, d) for d in dims)
    y = inplace ? fht!(copy(x), dims) : fht(x, dims)
    return sum(abs2, y ./ sqrt(eltype(x)(m)) .- target)
end

function residual_gradient(x, target, dims, inplace)
    dx, dt = zero(x), zero(target)
    Enzyme.autodiff(
        Reverse,
        residual_loss,
        Active,
        Duplicated(x, dx),
        Duplicated(target, dt),
        Const(dims),
        Const(inplace),
    )
    return dx, dt
end

function residual_direction(x, target, dims, inplace)
    return Enzyme.autodiff(
        Forward,
        residual_loss,
        Duplicated,
        Duplicated(x, one.(x)),
        Duplicated(target, one.(target)),
        Const(dims),
        Const(inplace),
    )[1]
end

function planned_residual_loss(p, x, target)
    return sum(abs2, p * copy(x) .- target)
end

function planned_residual_gradient(p, x, target)
    dx, dt = zero(x), zero(target)
    Enzyme.autodiff(
        Reverse,
        planned_residual_loss,
        Active,
        Const(p),
        Duplicated(x, dx),
        Duplicated(target, dt),
    )
    return dx, dt
end

function planned_residual_direction(p, x, target)
    return Enzyme.autodiff(
        Forward,
        planned_residual_loss,
        Duplicated,
        Const(p),
        Duplicated(x, one.(x)),
        Duplicated(target, one.(target)),
    )[1]
end

function check_transform(T, shape, dims)
    x = reshape(T.(sin.(1:prod(shape))), shape)
    reference = direct_hartley(x, dims)
    m = prod(size(x, d) for d in dims)
    for transform in (fht, fht!, ifht, ifht!)
        rx = Reactant.to_rarray(x)
        executable = Reactant.@compile transform(rx, dims)
        result = Array(executable(rx, dims))
        scale = transform in (ifht, ifht!) ? inv(T(m)) : one(T)
        @test isapprox(result, scale .* reference; rtol=40eps(T), atol=40eps(T))
        @test Array(rx) ≈ (transform in (fht!, ifht!) ? result : x)
    end
    for planner in (plan_fht, plan_fht!, plan_ifht, plan_ifht!)
        rx = Reactant.to_rarray(x)
        p = planner(rx, dims)
        executable = Reactant.@compile planned_transform(p, rx)
        result = Array(executable(p, rx))
        scale = planner in (plan_ifht, plan_ifht!) ? inv(T(m)) : one(T)
        @test isapprox(result, scale .* reference; rtol=40eps(T), atol=40eps(T))
        rx = Reactant.to_rarray(x)
        executable = Reactant.@compile buffered_transform(p, rx)
        @test Array(executable(p, rx)) ≈ scale .* reference
        @test Array(rx) ≈ x
    end
end

function check_gradient(T, shape, dims)
    x = reshape(T.(sin.(1:prod(shape))), shape)
    target = reshape(T.(0.2cos.(1:prod(shape))), shape)
    m = prod(size(x, d) for d in dims)
    residual = direct_hartley(x, dims) ./ sqrt(T(m)) .- target
    expected_x = direct_hartley(2 .* residual, dims) ./ sqrt(T(m))
    expected_target = -2 .* residual
    for inplace in (false, true)
        rx, rt = Reactant.to_rarray.((x, target))
        gradient = Reactant.@compile residual_gradient(rx, rt, dims, inplace)
        dx, dt = gradient(rx, rt, dims, inplace)
        @test isapprox(Array(dx), expected_x; rtol=100eps(T), atol=100eps(T))
        @test isapprox(Array(dt), expected_target; rtol=100eps(T), atol=100eps(T))
        @test Array(rx) ≈ x
        @test Array(rt) ≈ target
        rx, rt = Reactant.to_rarray.((x, target))
        direction = Reactant.@compile residual_direction(rx, rt, dims, inplace)
        actual = direction(rx, rt, dims, inplace)
        @test isapprox(
            T(actual), sum(expected_x)+sum(expected_target); rtol=100eps(T), atol=100eps(T)
        )
    end
    for planner in (plan_fht, plan_fht!, plan_ifht, plan_ifht!)
        rx, rt = Reactant.to_rarray.((x, target))
        scale = planner in (plan_ifht, plan_ifht!) ? sqrt(T(m)) : inv(sqrt(T(m)))
        p = scale * planner(rx, dims)
        gradient = Reactant.@compile planned_residual_gradient(p, rx, rt)
        dx, dt = gradient(p, rx, rt)
        @test isapprox(Array(dx), expected_x; rtol=100eps(T), atol=100eps(T))
        @test isapprox(Array(dt), expected_target; rtol=100eps(T), atol=100eps(T))
        rx, rt = Reactant.to_rarray.((x, target))
        direction = Reactant.@compile planned_residual_direction(p, rx, rt)
        @test isapprox(
            T(direction(p, rx, rt)),
            sum(expected_x)+sum(expected_target);
            rtol=100eps(T),
            atol=100eps(T),
        )
    end
end

function main()
    Reactant.set_default_backend(get(ENV, "FHT_REACTANT_BACKEND", "cpu"))
    println(
        "Julia=",
        VERSION,
        " Reactant=",
        pkgversion(Reactant),
        " Enzyme=",
        pkgversion(Enzyme),
    )
    println("backend=", get(ENV, "FHT_REACTANT_BACKEND", "cpu"))
    println("devices=", string.(Reactant.devices()))
    @testset "Reactant Hartley transforms and plans" begin
        for T in (Float32, Float64),
            (shape, dims) in (
                ((5,), (1,)),
                ((3, 4), (1, 2)),
                ((2, 3, 2), (1, 3)),
                ((2, 2, 2, 2), (1, 2, 3, 4)),
            )

            check_transform(T, shape, dims)
            check_gradient(T, shape, dims)
        end
        x = reshape(Float32.(sin.(1:12)), 3, 4)
        rx = Reactant.to_rarray(x)
        p = plan_fht(rx, (1, 2); use_r2r=true)
        executable = Reactant.@compile planned_transform(p, rx)
        @test Array(executable(p, rx)) ≈ direct_hartley(direct_hartley(x, (1,)), (2,))
        @test_throws ArgumentError p * x
        @test_throws ArgumentError plan_fht(rx, (1, 1))
        @test_throws ArgumentError plan_fht(rx, (0,))
        @test_throws ArgumentError plan_fht(rx, (1, 2); flags=FFTW.MEASURE)
    end
end

main()
