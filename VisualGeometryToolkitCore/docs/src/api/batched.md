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

## Vignetting

A camera's relative illumination V on the sensor as a B-spline in V on
clamped knots ([`clamped_knots`](@ref), [`bspline_basis`](@ref)), linear in
its control points: a tensor surface over a sensor box
([`TensorVignetting`](@ref)) or a profile in the radius about a centre
([`RadialVignetting`](@ref)). A model is callable, `v(u)`, and isbits, so a
batch of sensor points goes through [`pointwise!`](@ref) on a device;
[`coefficient_jacobian`](@ref) is its row of a least squares in the control
points, [`active_coefficients`](@ref) that row's nonzeros, and
[`pole_regular_basis`](@ref) the radial profiles smooth at their centre. The
estimators that fit them are their owners': a radiometric calibration's,
a panorama's self-calibration.

```julia
v = RadialVignetting(12, SVector(2128.0, 1424.0), 1700.0)   # control points, centre and radius (camera pixels); V ≡ 1
v = with_control_points(v, pole_regular_basis(v) * y)        # a profile with V′(0) = 0
v(SVector(2500.0, 1300.0))                                   # V at a sensor point
```

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/cameras/vignetting.jl"]
```

## Lighting

The scene's light as a radiometric camera model sees it: an
[`EnvironmentLight`](@ref), radiance from every direction given by a
directional map on a sphere of finite radius about a centre, looked up where a
ray from the shaded point leaves the sphere ([`exit_direction`](@ref); the
distant light is its limit as the radius grows). The map is a
[`SphericalHarmonicMap`](@ref): real spherical harmonics per channel
([`sh_basis`](@ref), orthonormal, no Condon–Shortley phase, index
[`sh_index`](@ref)`(l, m) = l² + l + m + 1`), so the radiance is linear in the
coefficients and [`coefficient_jacobian`](@ref) is its row. Everything is
isbits and allocation-free, for a kernel; the centre and radius are lengths
(`ustrip(m, light)` for the kernel's plain form). Hoisted from
RadiometricCamera, bit for bit.

```julia
light = EnvironmentLight(SphericalHarmonicMap(a, 2), SVector(0.0, 0.0, 0.0), 1.5)   # 3 × 9 coefficients, centre, radius
radiance(light, x, ω)                                    # per channel, arriving at x from direction ω
coefficients(light) * coefficient_jacobian(light, x, ω)  # the same, as a row of a least squares
```

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/cameras/lighting.jl"]
```

## Blur

One blur of an imaging model — a camera's in-focus blur (diffraction, the
sensor's anti-aliasing filter, demosaicing), a print's own blur — as a
[`GaussianBlur`](@ref) by its second-moment matrix ([`moment_matrix`](@ref)):
round, axis-aligned (stored as an `SDiagonal`) or general. Several blurs
compose into a point spread: two Gaussians in sequence add their moments
([`convolve`](@ref)), and a linear map of the plane takes `M` to `J M Jᵀ`
([`transform_blur`](@ref)). A blur is a kernel's shape, not a measurement's
uncertainty, so it is not a [`CovMat`](@ref).

```julia
camera = GaussianBlur(0.6)                     # round, 0.6 px
print  = GaussianBlur(0.02, 0.03)              # axis-aligned (mm, on the print)
total  = convolve(transform_blur(J, print), camera)   # the print's blur seen through the pixel footprint J, then the camera's
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
`(nodes, w, a)`, allocate nothing and run in a KernelAbstractions kernel on the
CPU backend.

```julia
tps = ThinPlateSplineSystem(grid_nodes(5))      # 5 × 5 nodes over [-½, ½]²
w, a = tps_coefficients(tps, values)            # linear in the 25 nodal values
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
