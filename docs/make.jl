using FastHartleyTransform
using AbstractFastHartleyTransforms
using Documenter

DocMeta.setdocmeta!(
    FastHartleyTransform, :DocTestSetup, :(using FastHartleyTransform); recursive=true
)

local_build = get(ENV, "DOCS_LOCAL", "false") == "true"

makedocs(;
    remotes=local_build ? nothing : Dict(),
    modules=[FastHartleyTransform, AbstractFastHartleyTransforms],
    authors="Kazunori Akiyama",
    sitename="FastHartleyTransform.jl",
    format=Documenter.HTML(;
        canonical="https://EHTJulia.github.io/FastHartleyTransform.jl",
        edit_link=local_build ? nothing : "main",
        repolink="https://github.com/EHTJulia/FastHartleyTransform.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Transform conventions" => "mathematics.md",
        "Plans and buffers" => "plans.md",
        "Differentiation" => "differentiation.md",
        "Backends" => "backends.md",
        "Performance" => "performance.md",
        "API reference" => "api.md",
    ],
)

if !local_build
    deploydocs(; repo="github.com/EHTJulia/FastHartleyTransform.jl", devbranch="main")
end
