using FastHartleyTransform
using Documenter

DocMeta.setdocmeta!(
    FastHartleyTransform, :DocTestSetup, :(using FastHartleyTransform); recursive=true
)

makedocs(;
    modules=[FastHartleyTransform],
    authors="Kazunori Akiyama",
    sitename="FastHartleyTransform.jl",
    format=Documenter.HTML(;
        canonical="https://EHTJulia.github.io/FastHartleyTransform.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/EHTJulia/FastHartleyTransform.jl", devbranch="main")
