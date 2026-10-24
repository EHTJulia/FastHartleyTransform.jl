function _precompile_transform(x, dims)
    y = similar(x)
    fht(x, dims)
    ifht(x, dims)
    fht!(x, dims)
    ifht!(x, dims)
    for planner in (plan_fht, plan_fht!)
        p = planner(x, dims)
        p * x
        LinearAlgebra.mul!(y, p, x)
        ip = inv(p)
        ip * x
        LinearAlgebra.mul!(y, ip, x)
        LinearAlgebra.ldiv!(y, p, x)
    end
    return nothing
end

@setup_workload begin
    for T in (Float32, Float64), shape in ((8,), (4, 4), (4, 4, 4))
        x = zeros(T, shape)
        regions = (1:length(shape), Tuple(1:length(shape)), (length(shape),))
        @compile_workload begin
            for dims in regions
                _precompile_transform(x, dims)
            end
            fht(x)
            ifht(x)
            fht!(x)
            ifht!(x)
        end
    end
end
