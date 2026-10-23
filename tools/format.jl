using Runic

function main(args)
    all(arg -> arg == "--write", args) || error("Usage: format.jl [--write]")
    root = dirname(@__DIR__)
    files = cd(root) do
        split(read(`git ls-files --cached --others --exclude-standard -z -- '*.jl' '*.md'`, String), '\0'; keepempty = false)
    end
    paths = sort!(unique!(joinpath.(root, files)))
    filter!(isfile, paths)
    options = [
        "--write" in args ? "--inplace" : "--check",
        "--docstrings",
        "--languages=julia,julia-repl,jldoctest,@example,@repl,@setup,@eval",
    ]
    return Runic.main(vcat(options, paths))
end

exit(main(ARGS))
