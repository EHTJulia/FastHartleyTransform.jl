# Contributing

Follow [ColPrac](https://github.com/SciML/ColPrac) and the repository-specific requirements in [AGENTS.md](AGENTS.md).

With compatible AbstractFastHartleyTransforms and FastHartleyTransform checkouts side by side, run:

```sh
julia --project -e 'using Pkg; Pkg.develop(path="../AbstractFastHartleyTransforms"); Pkg.instantiate(); Pkg.test()'
julia --project=tools -e 'using Pkg; Pkg.instantiate()'
julia --project=tools tools/format.jl --write
julia --project=tools tools/format.jl
```

Runic 1.13.0 is pinned in the tooling environment. The format check covers Julia files and documentation examples without rewriting them.

The `test/reactant` and `test/gpu` environments provide optional backend tests and benchmarks. For a local documentation build, configure the `docs` environment with both packages and run `DOCS_LOCAL=true julia --project=docs docs/make.jl`.
