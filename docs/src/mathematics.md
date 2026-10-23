# Transform conventions

## Definition and sign

Using zero-based indices for the mathematical formula, the forward transform on selected axes is

```math
(Hx)_{\mathbf{k}} = \sum_{\mathbf{n}} x_{\mathbf{n}}
\operatorname{cas}\!\left(2\pi\sum_{d\in\mathrm{dims}}\frac{n_d k_d}{N_d}\right),
\qquad \operatorname{cas}(t)=\cos(t)+\sin(t).
```

The sum ranges only over transformed axes. Indices on other axes remain fixed. Julia array indices start at one, so the corresponding formula uses `n_d-1` and `k_d-1`.

With the usual negative-exponent forward FFT, `H(x) = real(fft(x)) - imag(fft(x))`. Equivalently, it is the sum of real and imaginary parts of the unnormalized backward FFT. CPU/CUDA and Reactant may use these different FFT expressions while implementing the same Hartley convention.

## Normalization and the adjoint

For `M = prod(size(x, d) for d in dims)`, the operator satisfies

```math
H^\mathsf{T}=H,\qquad H^2=M I,\qquad H^{-1}=H/M.
```

Thus `ifht(fht(x, dims), dims) ≈ x`. An orthonormal transform is `fht(x, dims) ./ sqrt(M)`. The reverse derivative of the forward transform is another forward transform; the reverse derivative of the inverse transform includes `1/M`. It is not correct to replace the forward adjoint with the normalized inverse without restoring the factor `M`.

```@example normalization
using FastHartleyTransform, LinearAlgebra
x = reshape(sin.(1:12), 3, 4)
g = reshape(cos.(1:12), 3, 4)
dims = (1, 2)
m = prod(size(x, d) for d in dims)
@assert fht(fht(x, dims), dims) ≈ m .* x
@assert dot(fht(x, dims), g) ≈ dot(x, fht(g, dims))
@assert norm(fht(x, dims) ./ sqrt(m)) ≈ norm(x)
nothing # hide
```

## Multidimensional and separable DHTs

For more than one axis, a single kernel of the summed phase differs from a product of per-axis kernels:

```math
\operatorname{cas}(a+b)\ne\operatorname{cas}(a)\operatorname{cas}(b).
```

The default uses the single-kernel definition. On CPU, `plan_fht(x, dims; use_r2r=true)` explicitly selects FFTW's separable DHT and emits a warning for multidimensional inputs. The separable operator still has its own symmetric inverse with the same normalization, but generally produces different forward values. Use the same convention throughout a model, its inverse, and its derivatives. For one-dimensional arrays, the default selects the equivalent FFTW real-to-real DHT.
