using Enzyme

function transform_loss(transform, x, target)
    return sum(abs2, transform(x) .- target)
end

function planned_loss(p, x, target)
    return sum(abs2, p * x .- target)
end

function buffer_loss(p, x, target)
    y = similar(x)
    mul!(y, p, x)
    return sum(abs2, y .- target)
end

function aliased_buffer_loss(p, x, target)
    mul!(x, p, x)
    return sum(abs2, x .- target)
end

function ignored_return_loss(p, x, target)
    p * x
    return sum(abs2, x .- target)
end

function repeated_loss(p, x, target)
    a = sum(abs2, x)
    y = p * x
    return a + sum(abs2, y .- target)
end

function two_input_loss(x, target, dims)
    m = prod(size(x, d) for d in dims)
    return sum(abs2, fht(x, dims) ./ sqrt(eltype(x)(m)) .- target)
end

function finite_gradient(f, x)
    h = cbrt(eps(eltype(x)))
    gradient = similar(x)
    for i in eachindex(x)
        plus, minus = copy(x), copy(x)
        plus[i] += h
        minus[i] -= h
        gradient[i] = (f(plus) - f(minus)) / (2h)
    end
    return gradient
end

function check_enzyme_loss(loss, x, expected)
    v = reshape(eltype(x).(cos.(1:length(x))), size(x))
    dx = zero(x)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Reverse), Const(loss), Active, Duplicated(copy(x), dx)
    )
    tolerance = eltype(x) == Float32 ? 2.0f-5 : 2.0e-11
    @test isapprox(dx, expected; rtol = tolerance, atol = tolerance)
    directional = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Forward),
        Const(loss),
        Duplicated,
        Duplicated(copy(x), copy(v)),
    )[1]
    @test isapprox(directional, dot(expected, v); rtol = tolerance, atol = tolerance)
    finite = finite_gradient(a -> loss(copy(a)), x)
    return @test isapprox(
        dx,
        finite;
        rtol = eltype(x) == Float32 ? 5.0f-3 : 3.0e-7,
        atol = eltype(x) == Float32 ? 5.0f-3 : 3.0e-7,
    )
end

@testset "Enzyme CPU rules" begin
    for T in (Float32, Float64),
            (shape, dims) in (((5,), (1,)), ((3, 4), (1, 2)), ((2, 3, 2), (1, 3)))

        x = reshape(T.(sin.(1:prod(shape))), shape)
        target = reshape(T.(0.2cos.(1:prod(shape))), shape)
        h = a -> direct_hartley(a, dims)
        m = prod(size(x, d) for d in dims)
        for planner in (plan_fht, plan_fht!, plan_ifht, plan_ifht!)
            p = planner(copy(x), dims)
            scale = planner in (plan_ifht, plan_ifht!) ? inv(T(m)) : one(T)
            expected = scale .* h(2 .* (scale .* h(x) .- target))
            check_enzyme_loss(a -> planned_loss(p, a, target), x, expected)
            check_enzyme_loss(a -> buffer_loss(p, a, target), x, expected)
            if AFHT.is_inplace(p)
                check_enzyme_loss(a -> aliased_buffer_loss(p, a, target), x, expected)
                check_enzyme_loss(a -> ignored_return_loss(p, a, target), x, expected)
            end
            check_enzyme_loss(a -> repeated_loss(p, a, target), x, expected .+ 2 .* x)
        end
        for transform in (fht, fht!, ifht, ifht!)
            scale = transform in (ifht, ifht!) ? inv(T(m)) : one(T)
            expected = scale .* h(2 .* (scale .* h(x) .- target))
            f = a -> transform(a, dims)
            check_enzyme_loss(a -> transform_loss(f, a, target), x, expected)
        end
        loss = (a, b) -> two_input_loss(a, b, dims)
        dx, dt = zero(x), zero(target)
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Reverse),
            two_input_loss,
            Active,
            Duplicated(copy(x), dx),
            Duplicated(copy(target), dt),
            Const(dims),
        )
        residual = h(x) ./ sqrt(T(m)) .- target
        expected_x = h(2 .* residual) ./ sqrt(T(m))
        expected_target = -2 .* residual
        tolerance = T == Float32 ? 2.0f-5 : 2.0e-11
        @test isapprox(dx, expected_x; rtol = tolerance, atol = tolerance)
        @test isapprox(dt, expected_target; rtol = tolerance, atol = tolerance)
        @test isapprox(
            dx,
            finite_gradient(a -> loss(a, target), x);
            rtol = T == Float32 ? 5.0f-3 : 3.0e-7,
            atol = T == Float32 ? 5.0f-3 : 3.0e-7,
        )
        @test isapprox(
            dt,
            finite_gradient(b -> loss(x, b), target);
            rtol = T == Float32 ? 5.0f-3 : 3.0e-7,
            atol = T == Float32 ? 5.0f-3 : 3.0e-7,
        )
        directional = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Forward),
            two_input_loss,
            Duplicated,
            Duplicated(copy(x), one.(x)),
            Duplicated(copy(target), one.(target)),
            Const(dims),
        )[1]
        @test isapprox(directional, sum(dx) + sum(dt); rtol = tolerance, atol = tolerance)
    end
end

function conditional_loss(p, x, constant, use_active)
    y = use_active ? x : constant
    return sum(abs2, p * y)
end

@testset "Reject dynamically constant transform buffers" begin
    x, constant = sin.(1:5), cos.(1:5)
    original = copy(constant)
    p = plan_fht(x)
    for mode in (Forward, Reverse)
        error_text = try
            return_activity = mode == Forward ? Duplicated : Active
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(mode),
                conditional_loss,
                return_activity,
                Const(p),
                Duplicated(copy(x), zero(x)),
                Const(constant),
                Const(false),
            )
            ""
        catch exception
            sprint(showerror, exception)
        end
        @test occursin("independent derivative buffer", error_text)
        @test constant == original
    end
end
