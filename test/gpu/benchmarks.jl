using BenchmarkTools, CUDA, FFTW, LinearAlgebra, Statistics, Test, FastHartleyTransform

execute(p, x) = p * x
execute_cuda(p, x) = CUDA.@sync execute(p, x)
function transfer_transform(p, x)
    return Array(p * CuArray(x))
end

function report(label, trial)
    estimate = median(trial)
    println(
        label,
        ",median_ns=",
        estimate.time,
        ",minimum_ns=",
        minimum(trial).time,
        ",p25_ns=",
        quantile(trial.times, 0.25),
        ",p75_ns=",
        quantile(trial.times, 0.75),
        ",host_bytes=",
        estimate.memory,
        ",host_allocations=",
        estimate.allocs,
        ",samples=",
        length(trial),
    )
    return estimate.time
end

function check_case(T, shape, threads, samples)
    FFTW.set_num_threads(threads)
    x = reshape(T.(sin.(1:prod(shape))), shape)
    gx = CuArray(x)
    dims = Tuple(1:ndims(x))
    println("case,type=", T, ",shape=", shape, ",dims=", dims, ",fftw_threads=", threads)
    @timev "first CPU plan (may include compilation)" cpu = plan_fht(
        x, dims; flags = FFTW.ESTIMATE
    )
    @timev "first GPU plan (may include compilation)" gpu = CUDA.@sync plan_fht(gx, dims)
    @timev "first CPU execution (may include compilation)" expected = execute(cpu, x)
    @timev "first GPU execution (may include compilation)" actual = execute_cuda(gpu, gx)
    @test isapprox(
        Array(actual), expected; rtol = 200eps(T), atol = 200eps(T) * maximum(abs, expected)
    )
    @test (@inferred execute(cpu, x)) isa typeof(x)
    @test (@inferred execute(gpu, gx)) isa typeof(gx)
    CUDA.synchronize()
    execute(cpu, x)
    execute_cuda(gpu, gx)
    transfer_transform(gpu, x)
    report(
        "cpu_plan",
        @benchmark plan_fht($x, $dims; flags = FFTW.ESTIMATE) evals = 1 samples = samples seconds = 1
    )
    report(
        "gpu_plan",
        @benchmark CUDA.@sync(plan_fht($gx, $dims)) evals = 1 samples = samples seconds = 1
    )
    c = report(
        "cpu_transform", @benchmark execute($cpu, $x) evals = 1 samples = samples seconds = 1
    )
    g = report(
        "gpu_transform",
        @benchmark execute_cuda($gpu, $gx) evals = 1 samples = samples seconds = 1
    )
    t = report(
        "gpu_with_transfers",
        @benchmark transfer_transform($gpu, $x) evals = 1 samples = samples seconds = 1
    )
    println("speedup,resident=", c / g, ",including_transfers=", c / t)
    # CUDA reports device allocations separately from Julia host allocations.
    CUDA.@time execute_cuda(gpu, gx)
    return nothing
end

function main()
    CUDA.functional(true) || error("A working CUDA device is required.")
    CUDA.allowscalar(false)
    CUDA.versioninfo()
    println(
        "Julia=",
        VERSION,
        ",CPU=",
        Sys.CPU_NAME,
        ",Julia_threads=",
        Threads.nthreads(),
        ",CUDA=",
        pkgversion(CUDA),
        ",FFTW=",
        pkgversion(FFTW),
        ",BenchmarkTools=",
        pkgversion(BenchmarkTools),
        ",flags=ESTIMATE,evals=1",
    )
    samples = parse(Int, get(ENV, "FHT_BENCHMARK_SAMPLES", "100"))
    @assert samples > 0
    for threads in unique((1, Threads.nthreads())),
            T in (Float32, Float64),
            shape in (
                (128,),
                (4096,),
                (262144,),
                (32, 32),
                (256, 256),
                (1024, 1024),
                (16, 16, 16),
                (64, 64, 64),
                (128, 128, 128),
            )

        check_case(T, shape, threads, samples)
    end
    return nothing
end

main()
