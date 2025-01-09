# FastHartleyTransform
| Documentation | Code Status | Guidelines |
| :------- | :------ | :------ |
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://EHTJulia.github.io/FastHartleyTransform.jl/stable/)[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://EHTJulia.github.io/FastHartleyTransform.jl/dev/) | [![Build Status](https://github.com/EHTJulia/FastHartleyTransform.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/EHTJulia/FastHartleyTransform.jl/actions/workflows/CI.yml?query=branch%3Amain)[![Coverage](https://codecov.io/gh/EHTJulia/FastHartleyTransform.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/EHTJulia/FastHartleyTransform.jl) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) |

This package provides a Julia implementation of the Fast Hartley Transform (FHT). It supports both CPUs and NVIDIA CUDA GPUs by using following FFT (fast Fourier transform) kernels:

- CPU implementation uses [`FFTW.jl`](https://github.com/JuliaMath/FFTW.jl).
- NVIDIA GPU implementation uses [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl)'s `CUDA.CUFFT` module. 


The implementation is built up on [`AbstractFastHartleyTransforms.jl`](https://github.com/EHTJulia/AbstractFastHartleyTransforms.jl), providing generic interfaces including `fht(x)`, `fht!(x)`, `ifht(x)`, `ifht!(x)`, `plan_fht(x)`, and `plan_ifht(x)`.
