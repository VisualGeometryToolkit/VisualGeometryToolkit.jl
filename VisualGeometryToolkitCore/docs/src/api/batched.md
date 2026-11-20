# Batch API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Per-point maps over arrays, the B-spline kernels and the thin-plate spline. See the
[Batches](../guide/batched.md) guide; the lens surfaces (`project!`, …) are
documented with the [Lenses API](lenses.md).

## Pointwise maps

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["transforms/pointwise.jl"]
```

## B-spline Kernels

The B-spline algebra on the CPU is BSplineKit's (knot vectors, evaluation and
derivatives, Galerkin matrices, recombined bases). Core keeps only what a GPU
kernel needs and BSplineKit cannot give it: the scalar Cox–de Boor basis
[`bspline_basis`](@ref)`(knots, Val(p), x)` — an `SVector` of every basis
function at `x` on a static knot vector, isbits and generic in the element
type, so it runs in a kernel and under ForwardDiff (hoisted from
RadiometricCamera, its arithmetic unchanged; equal to BSplineKit's
`evaluate_all` on the same knots) — its broadcast over a device array, and the
tensor-product surface/4D spline kernels built on it; the `ChainRulesCore`
rrules attach via the package extension.

```julia
using BSplineKit: BSplineBasis, BSplineOrder, knots
k = SVector{11}(knots(BSplineBasis(BSplineOrder(4), range(-0.5, 0.5; length = 5))))   # 7 cubic control points on [-½, ½]
N = bspline_basis(k, Val(3), 0.12)               # SVector{7}, sum(N) == 1
```

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["numerics/bspline.jl"]
```

## The bending energy of a B-spline

[`bending_gram`](@ref) is `∫∫ f_xx² + 2 f_xy² + f_yy²` of a tensor-product
B-spline as a quadratic form in its coefficients, from BSplineKit's Galerkin
matrices; [`bending_rows`](@ref) factors it into a regularizer's rows and
[`bending_stencils`](@ref) gives it cell by cell on a large uniform lattice
([`cubic_lattice_basis`](@ref), Core's generic-model lattice). The natural
boundary condition `f'' = 0` of BSplineKit's recombined basis is
[`natural_extrapolation`](@ref), the rule [`BoundaryRule`](@ref) applies to
a lattice's nodes outside an imaged region by default (`order = 2`); its
`order = 3` is the curvature-preserving Δ³c = 0.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/bspline_energy.jl"]
```

## Penalized least squares

Every penalized solve of the stack is `min ‖A x − b‖² + λ‖D (x − c)‖²`, one
owner per piece. The operator `D`: [`difference_rows`](@ref) (orders 1–3 on a
masked lattice or a texel page, along chosen axes; [`gradient_pairs`](@ref) its
isotropic order-1 pairs) or [`bending_rows`](@ref) (the bending energy), and
[`row_stencils`](@ref) its rows as a nonlinear solve's residual blocks. The
weight's unit: [`penalty_scale`](@ref) with [`mean_positive_eigenvalue`](@ref)
(m̄). The linear solve on sparse operators, on any KernelAbstractions device:
[`cgls`](@ref) on a [`PenalizedOperator`](@ref) of [`CSRMatrix`](@ref)es
(Krylov.jl's CGLS on the stacked operator `[A; √W D]`; on CUDA the products are
cuSPARSE's, under its reproducible algorithm), or [`normal_cg`](@ref) on a normal
operator `AᵀA + DᵀWD` that is only applied (its rows never stored); each
Jacobi-preconditioned on request, each with a stopping rule relative to the
normal equations' right-hand side `‖Aᵀb + DᵀWh‖`. The
weight's choice by generalized cross-validation is
VisualGeometryToolkitEstimation's (`gcv_curve`, `gcv_select`). No penalty fixes a
gauge: a direction neither the data nor the penalty holds is held exactly by
its caller.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/penalized_least_squares.jl"]
```

## Vignetting and lighting

A camera's vignetting fields, the measured radial profile and the scene's
light moved to VisualRadiometryToolkitCore in 0.12.0 (radiometry lifts
geometry: that package depends on this one), under the same names. The
B-spline basis they evaluate ([`bspline_basis`](@ref), [`clamped_knots`](@ref))
and the support protocol of their control points stay here.

## Blur

One blur of an imaging model — a camera's in-focus blur (diffraction, the
sensor's anti-aliasing filter, demosaicing), a print's own blur — as a
[`GaussianBlur`](@ref). A point spread function is a probability density
(where a photon from a point source lands), and a Gaussian blur is the density
`N(0, Σ)`, so it is held by its covariance, a [`CovMat`](@ref)`{2}`
([`cov`](@ref); `var` and `std` per axis): round, axis-aligned (stored as an
`SDiagonal`) or general, px² on a sensor, m² (quantities) on a print. Several
blurs compose into a point spread: two Gaussians in sequence add their
covariances ([`convolve`](@ref)), and a linear map of the plane pushes the
covariance forward, `Σ ↦ J Σ Jᵀ` ([`propagate`](@ref)).

```julia
camera = GaussianBlur(0.6)                     # round, σ = 0.6 px
print  = GaussianBlur(0.02mm, 0.03mm)          # axis-aligned, on the print: m² quantities
total  = convolve(propagate(J, ustrip(mm, print)), camera)   # in mm², through the pixel footprint J (px/mm), then the camera's
std(camera)                                    # (0.6, 0.6) px
```

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/cameras/blur.jl"]
```

## Thin-plate spline

`f(𝐩) = a₁ + a_x 𝐩_x + a_y 𝐩_y + Σᵢ wᵢ U(𝐩, 𝐩ᵢ)` with `U = ½ r² ln r²`, through
nodal values on scattered nodes (hoisted from RadiometricCamera, its arithmetic
unchanged). [`ThinPlateSplineSystem`](@ref) is the host-side solver; the evaluator
[`tps_evaluate`](@ref) and its gradient [`tps_gradient`](@ref) take isbits
`(nodes, w, a)`, allocate nothing and run in a KernelAbstractions kernel.
[`tps_coefficients`](@ref) is a loop (no BLAS), and with the pure-bending
projection [`tps_pure_bending`](@ref) it runs in a kernel on an adapted system
(`Adapt.adapt(backend, tps)`, a `Float32` one by `convert_eltype`).

```julia
tps = ThinPlateSplineSystem(grid_nodes(5))      # 5 × 5 nodes over [-½, ½]²
w, a = tps_coefficients(tps, values)            # linear in the 25 nodal values
w, a = tps_coefficients(tps, tps_pure_bending(tps, values))   # the plane dropped first: a HeightField's (w, a)
f = tps_evaluate(tps.nodes, w, a, SVector(0.1, -0.2))
∇f = tps_gradient(tps.nodes, w, a, SVector(0.1, -0.2))
```

Its analogue on the unit sphere, [`SphericalSplineSystem`](@ref) (Wahba's
`m = 2` kernel [`spherical_spline_kernel`](@ref), a constant in place of the
affine part; also hoisted from RadiometricCamera), has the same shape:
[`spherical_spline_coefficients`](@ref), the isbits evaluator
[`spherical_spline_evaluate`](@ref), and the shared `nodal_weights`,
`bending_energy` and `bending_factor`.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/thin_plate_spline.jl"]
```

## Special functions

Device-safe scalar functions — pure arithmetic, generic over `Real`, so they
compile in a KernelAbstractions kernel at any float width and ForwardDiff
`Dual`s flow through: log Γ and the digamma/trigamma series of the Student-t
adaptation kernels, the normal CDF [`normcdf_as`](@ref) (Abramowitz & Stegun
7.1.26) and the dilogarithm [`dilog`](@ref).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/special.jl"]
```

## Quadrature

The probabilists' Gauss–Hermite rule [`gauss_hermite_prob`](@ref) (an
expectation over `N(0, 1)`, exact to degree `2n − 1`) and the Hammersley
points [`hammersley`](@ref) (a kernel's own low-discrepancy sample from its
index).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/quadrature.jl"]
```

## One-dimensional minimization

[`VisualGeometryToolkitCore.grid_brent_minimum`](@ref) scans an evenly spaced
grid and refines its least value by Brent's method: the stack's one 1-D
minimizer (public, not exported).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/minimize.jl"]
```

## Kernel building blocks

An in-register LU solve for fixed-size systems and a KernelAbstractions
workgroup reduction, for code that writes its own kernels (public, not
exported).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/small_lu.jl", "numerics/ka_reduce.jl"]
```

## Index

```@index
Pages = ["batched.md"]
```
