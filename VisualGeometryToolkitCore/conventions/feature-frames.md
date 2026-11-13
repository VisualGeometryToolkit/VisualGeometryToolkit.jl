## Feature frames: one interface, every verb once

A detection is a [`FeatureFrame`](@ref), plain or [`Uncertain`](@ref), in an
`Attributed` with its metadata. The family is ordered by degrees of freedom
([`frame_dof`](@ref VisualGeometryToolkitCore.frame_dof)): [`PointFrame`](@ref) 2 (a centre), [`Disc`](@ref) 3 (+ σ),
[`OrientedDisc`](@ref) 4 (+ θ), [`EllipticFrame`](@ref) 5 (a symmetric shape
factor), [`AffineFrame`](@ref) 6 (a general one). The DOF level is a trait,
[`frame_scale`](@ref VisualGeometryToolkitCore.frame_scale): `PointShape()`,
`IsotropicScale()`, `ShapeFactorScale()`.

**Every verb is implemented ONCE**, on `FeatureFrame` or on its trait, from
the per-type primitives: the centre ([`center`](@ref VisualGeometryToolkitCore.center)), the local shape where a
frame has one ([`shape_factor`](@ref); σ for the isotropic frames), and two
one-line constructors per type (its frame from mapped lengths, its similarity
image). A `PointFrame` has only the centre, so it gets the whole interface
through the hoisted code — no verb has a copy per type. It is `isbits` with
the size and layout of `Point2{T}`, and its verbs cost what the `Point2` verbs
cost (tested: no allocation, no `Union` or boxing in the typed code, a
KernelAbstractions kernel on the CPU and the GPU over a `StructArray` of
detections, and a benchmark against `Uncertain{Point2}`).

### The verbs every frame has

| Verb | Every frame (PointFrame included) |
|---|---|
| centre | `center(f)` (also `centroid`, `GeometryBasics.origin`) |
| the point observation of a detection | `center(u::Uncertain{<:FeatureFrame})`, `center(d::Attributed{<:Uncertain{<:FeatureFrame}})` → `Uncertain{Point2}`: the centre and the first 2×2 block of the frame's covariance. THE one conversion: no code takes a centre and a covariance block apart by hand |
| similarity action | `S(f)` for a `SimilarityMap{2}`, an `EuclideanMap{2}`, a rotation or uniform-scaling `LinearMap`: exact, the class kept |
| transfer through any point map | [`transfer`](@ref)`(m, f)`: the centre by `m`; a scaled frame's shape by `m`'s Jacobian at the centre (an `AffineFrame`, first order), exact through a similarity |
| propagate | `propagate(m, u::Uncertain{PointFrame})` through every framed map an uncertain point propagates through (`EuclideanMap`, `SimilarityMap`, `HomographyMap`, `FramedAffine`), the frame kept; for a scaled frame, `propagate(m, center(u))` |
| error ellipse | `Ellipse(u::Uncertain{<:FeatureFrame}; sigma)`: its centre's |
| bounding box | `extrema(f)`, `Rect2(f)`, `Rect2{Int}(f)`, `bbox(f)` (a point's is the point) |
| units | `ustrip(u, f)`, `f * s`, `f / s` (a change of unit or length scale; angles kept) |
| translation | `f + v`, `f - v` |
| GPU | `isbits` values: `adapt` is the identity, a `StructArray` of (uncertain, attributed) frames is columnar |
| codec | VisualGeometryToolkitIO's one detection codec: `center` and the shape, `covariance` over the frame's parameters, centre first ([`frame_dof`](@ref VisualGeometryToolkitCore.frame_dof); persisted-geometry) |

### The verbs only the scaled frames have

| Verb | Disc, OrientedDisc | EllipticFrame, AffineFrame | PointFrame |
|---|---|---|---|
| `scale(f)` | σ | — (read `characteristic_scale`) | absent |
| `shape_factor(f)`, `shape_matrix(f)` | `σI`, `σR(θ)`; `σ²I` exactly | `S`, `A`; `F Fᵀ` | absent |
| `characteristic_scale`, `aspect_ratio` | σ, 1 | `√|det F|`, from its singular values | absent |
| `AffineMap(f)`, `boundary_point(f, θ)` | the canonical → image map | the same | absent |
| `region(f, k)`, `Circle(f, r)`, `Ellipse(f, r)` | the k·σ contour | the same | absent |
| `MvNormal(f)`, `intersects`, `inv(f)` | as documented per verb | as documented per verb | absent |

An absent verb is absent BY DESIGN, not a stub: calling it on a `PointFrame`
is a `MethodError` whose hint names the interface and the point's own verbs.
