# Coding standards

These requirements apply to implementation, tests, examples, benchmarks, and configuration files in this repository.

## Collaboration

- Follow [ColPrac](https://github.com/SciML/ColPrac) for contributions, reviews, merging, and releases, together with the repository-specific requirements below.
- Submit changes through focused pull requests from topic branches; do not push code changes directly to the default branch. Merge only after CI passes and at least one reviewer other than the author approves. Authors must not approve their own pull requests.

## Julia performance

- At the start of each task, check the latest Julia LTS release at https://julialang.org/downloads/ and report the version and the versioned official Performance Tips URL. The baseline verified on 2026-10-23 is Julia 1.10.12: https://docs.julialang.org/en/v1.10/manual/performance-tips/ (the official manual is versioned by minor release).
- Follow that manual, especially type stability, type inference, avoiding unnecessary allocations, memory-order array access, and appropriate preallocation. Keep performance-critical work inside functions and pass inputs as arguments.
- Preserve numerical correctness, Enzyme differentiation, and Reactant compatibility. Explain recommendations that cannot be applied. Do not assume that host mutation, scalar indexing, SIMD promises, or relaxed floating-point semantics are compatible with every backend.
- Check inference and allocations on the major computational paths, and measure steady-state performance after warmup. Distinguish compilation, plan construction, and transform execution. Account for input mutation and synchronize asynchronous backends when timing them.
- Report the hardware, Julia/package versions, element types, dimensions, planner flags, thread counts, sample settings, and results. Claim improvements only with comparable before/after measurements.

## Formatting

- Use the root `.JuliaFormatter.toml`, with `style = "blue"`, for all Julia files, including tests, documentation examples, extensions, and benchmarks.
- JuliaFormatter is pinned to exactly 2.1.2 in `tools/Project.toml`; keep the tooling manifest committed. Run `julia --project=tools -e 'using Pkg; Pkg.instantiate()'` before use.
- Format with `julia --project=tools tools/format.jl --write` and check with `julia --project=tools tools/format.jl`. A check must not rewrite files and must fail when formatting differs.
- Run the format check at the end of every implementation stage. Keep broad formatting changes in a separate change unit from functional changes.

## Language and reporting

- Write all line comments, block comments, docstrings, and configuration comments in American English. Review existing text as well as additions; keep language cleanup separate from functional changes.
- Explain work and progress to the user in Japanese.
- At the end of every implementation stage, report tests, inference/allocation and performance results, BlueStyle checks, and comment/docstring language review. Explicitly list checks that were not performed, including unavailable backend or differentiation checks.
