using Reactant, Enzyme, FFTW, FastHartleyTransform, Test

function residual_loss(x, target)
    return sum(abs2, fht(x) ./ sqrt(eltype(x)(length(x))) .- target)
end

function residual_gradient(x, target)
    dx, dt = zero(x), zero(target)
    Enzyme.autodiff(
        Reverse, residual_loss, Active, Duplicated(x, dx), Duplicated(target, dt)
    )
    return dx, dt
end

function synchronized_call(f, args...)
    result = f(args...)
    Reactant.synchronize(result)
    return result
end

synchronized_call(f, x::Array, args...) = f(x, args...)

function measure(label, f, args...)
    synchronized_call(f, args...)
    samples = parse(Int, get(ENV, "FHT_BENCHMARK_SAMPLES", "100"))
    @assert samples > 0
    times = Vector{Float64}(undef, samples)
    bytes = Vector{Int}(undef, samples)
    for i in eachindex(times)
        trial = @timed synchronized_call(f, args...)
        times[i], bytes[i] = trial.time, trial.bytes
    end
    sort!(times)
    middle = if isodd(samples)
        times[(samples + 1) ÷ 2]
    else
        (times[samples ÷ 2] + times[samples ÷ 2 + 1]) / 2
    end
    println(
        label,
        ",median_s=",
        middle,
        ",minimum_s=",
        first(times),
        ",minimum_host_bytes=",
        minimum(bytes),
        ",samples=",
        samples,
    )
    return nothing
end

function check_case(T, shape)
    x = reshape(T.(sin.(1:prod(shape))), shape)
    target = reshape(T.(cos.(1:prod(shape))), shape)
    expected = fht(x)
    residual = expected ./ sqrt(T(length(x))) .- target
    expected_dx = fht(2 .* residual) ./ sqrt(T(length(x)))
    expected_dt = -2 .* residual
    if get(ENV, "FHT_REACTANT_BACKEND", "cpu") == "cpu"
        dx, dt = residual_gradient(x, target)
        @test dx ≈ expected_dx
        @test dt ≈ expected_dt
        measure("native_cpu_gradient_threads_1", residual_gradient, x, target)
    end
    rx, rt = Reactant.to_rarray.((x, target))
    println("case,type=", T, ",shape=", shape)
    @timev "compile transform (tracing and XLA)" transform = Reactant.@compile fht(rx)
    @timev "compile gradient (tracing and XLA)" gradient = Reactant.@compile residual_gradient(
        rx, rt
    )
    actual = synchronized_call(transform, rx)
    tolerance = 500eps(T)
    @test isapprox(
        Array(actual), expected; rtol = tolerance, atol = tolerance * maximum(abs, expected)
    )
    dx, dt = synchronized_call(gradient, rx, rt)
    @test isapprox(
        Array(dx), expected_dx; rtol = tolerance, atol = tolerance * maximum(abs, expected_dx)
    )
    @test isapprox(
        Array(dt), expected_dt; rtol = tolerance, atol = tolerance * maximum(abs, expected_dt)
    )
    measure("reactant_transform", transform, rx)
    measure("reactant_gradient", gradient, rx, rt)
    return nothing
end

function main()
    backend = get(ENV, "FHT_REACTANT_BACKEND", "cpu")
    Reactant.set_default_backend(backend)
    FFTW.set_num_threads(1)
    println(
        "backend=",
        backend,
        ",devices=",
        string.(Reactant.devices()),
        ",Julia=",
        VERSION,
        ",CPU=",
        Sys.CPU_NAME,
        ",Reactant=",
        pkgversion(Reactant),
        ",Enzyme=",
        pkgversion(Enzyme),
    )
    println("Resident arrays; synchronized execution; compilation and transfers excluded.")
    for T in (Float32, Float64), shape in ((128,), (262144,), (256, 256), (64, 64, 64))
        check_case(T, shape)
    end
    return nothing
end

main()
