using FHT
using Documenter

DocMeta.setdocmeta!(FHT, :DocTestSetup, :(using FHT); recursive=true)

makedocs(;
    modules=[FHT],
    authors="Kazunori Akiyama",
    sitename="FHT.jl",
    format=Documenter.HTML(;
        canonical="https://EHTJulia.github.io/FHT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/EHTJulia/FHT.jl",
    devbranch="main",
)
