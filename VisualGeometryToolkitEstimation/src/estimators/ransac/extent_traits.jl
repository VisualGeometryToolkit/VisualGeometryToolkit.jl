# =============================================================================
# Extent traits — orthogonal "this problem has a known validity region" marker
# =============================================================================
#
# Some RANSAC problems carry domain knowledge that the fitted model is
# supposed to be valid over a specific *region*, not just at the inlier
# correspondences. Calibration boards are the canonical example: when the
# homography maps the board's image features into the world plane, the
# board has a known rectangular extent and the H should be a well-behaved
# (non-bow-tie) projection over the entire rect — not just at the inliers,
# which can cluster in one corner and leave H free to fold elsewhere.
#
# This is *orthogonal* to the problem family (homography / pose /
# fundamental / ...) — any model type can in principle carry an extent.
# So instead of subtyping per-family (`BoundedHomographyProblem`,
# `BoundedAbsolutePoseProblem`, ...) we use the Holy trait pattern:
#
#   extent_trait(p) -> NoExtent() | WithExtent(ext)
#
# Default is `NoExtent()`; the RANSAC consensus hook short-circuits to
# `true` when no extent is declared.
#
# Composition: `BoundedProblem{P,R}` is a generic wrapper around any
# RANSAC problem that adds an extent. `extent_trait(::BoundedProblem)`
# returns the carried extent. All other RANSAC hooks forward to the
# inner problem via `Base.getproperty` + explicit interface methods.
# =============================================================================

"""
    ExtentTrait

Whether a RANSAC problem declares a validity region: [`NoExtent`](@ref) (the
default) or [`WithExtent`](@ref). [`extent_trait`](@ref) returns a problem's
trait, and [`BoundedProblem`](@ref) adds a region to any problem.
"""
abstract type ExtentTrait end

"""
    NoExtent() <: ExtentTrait

Marker that the problem has no associated validity region. The default
for all problem types; consensus-extent checks short-circuit to `true`.
"""
struct NoExtent <: ExtentTrait end

"""
    WithExtent(extent) <: ExtentTrait

Marker that the problem's fitted model is supposed to be a well-behaved
mapping over `extent`. The post-IRLS consensus check then dispatches on
`(model, extent)` to verify (e.g. for `HomographyMat + HyperRectangle{2}`,
the projected rectangle must be a simple convex quadrilateral —
`VisualGeometryToolkitCore.is_orientation_preserving`).
"""
struct WithExtent{R} <: ExtentTrait
    extent::R
end

"""
    extent_trait(p) -> ExtentTrait

Declare the extent (or absence thereof) for a RANSAC problem. Default
is `NoExtent()`. Problems that hold an extent (e.g. `BoundedProblem`)
override this to return `WithExtent(...)`.
"""
extent_trait(::AbstractRansacProblem) = NoExtent()

# =============================================================================
# Extent-aware consensus check — composed into `test_consensus(::BoundedProblem)`
# =============================================================================
#
# The RANSAC loop calls `test_consensus(problem, model, mask)` after Phase 3's
# re-score (see `loop.jl:_test_one_model`). For a `BoundedProblem`, the
# override below composes:
#   (a) the inner problem's own `test_consensus` (e.g. HomographyProblem's
#       cheirality check on the inlier mask), AND
#   (b) the extent check — dispatched on `(extent_trait, model_type)`:
#         - NoExtent       → true (short-circuit; this branch never fires
#                            for BoundedProblem but keeps the trait surface
#                            uniform for any future opt-in types)
#         - WithExtent +
#           HomographyMat  → VGC.is_orientation_preserving(H, extent)
#         - WithExtent +
#           Extrinsics3          → extent visibility + convex-quadrilateral check
#                            (added when needed)

@inline _check_extent(::NoExtent, _p, _model, _mask) = true

# HomographyMat + rectangular extent: simple convex quad in image space.
@inline _check_extent(
    et::WithExtent{<:HyperRectangle{2}}, _p,
    H::HomographyMat, _mask
) =
    is_orientation_preserving(H, et.extent)

# Extrinsics3 + rectangular extent: project the 4 board-rectangle corners (lifted to
# z=0 world points) through the pose + the problem's camera, then require
#   (a) cheirality — every corner in front of the camera (camera-frame Z>0).
#       The per-correspondence cheirality in `residuals!` only guarantees the
#       matched inliers are in front; the board *extent* can still straddle the
#       camera (a grazing pose) and explode the quad. This catches that.
#   (b) a simple, convex, consistently-wound quad in image space — the calibrated
#       analog of `is_orientation_preserving(H, rect)` on the homography path.
# Projection is `VisualGeometryToolkitCore.project(model, ·)` on the RAY domain — the
# same call `residuals!` makes, so the quad test and the residual agree about
# which directions the camera images.
@inline function _check_extent(
        et::WithExtent{<:HyperRectangle{2}}, p,
        pose::EuclideanMap{3, Float64}, _mask
    )
    C = rect_corners(et.extent)                     # VGC's CCW board corners, lifted to z = 0
    R = rotation(pose); t = translation(pose)
    model = getfield(p, :inner).context.model
    # (a) project to image; bail on any corner the lens cannot image. NOT
    # `Xc[3] > 0` — that is the gnomonic chart's domain, and it rejected board
    # corners a fisheye images perfectly well, straddling 90°.
    u = ntuple(4) do i
        q = VgtkCore.project(model, R * SVector(C[i][1], C[i][2], 0.0) + t)
        (isfinite(q[1]) && isfinite(q[2])) ? SVector{2, Float64}(q[1], q[2]) : nothing
    end
    any(isnothing, u) && return false
    # (b) simple convex quad: VGC's winding test, the one `is_orientation_preserving` runs.
    return is_convex_quad(u)
end

# =============================================================================
# BoundedProblem — generic wrapper that adds an extent to any RANSAC problem
# =============================================================================
#
# `BoundedProblem(inner, extent)` wraps any concrete RANSAC problem and
# declares its `extent_trait`. All RANSAC interface methods forward to
# `inner` via field-access (`Base.getproperty`) + explicit interface
# methods. This is the user-facing way to attach extent info to a
# `HomographyProblem`, `AbsolutePoseProblem`, etc., without subtyping
# per family.
#
# Type parameter `K` is forwarded from `inner` so this struct sits in
# the right `AbstractCspondProblem{K}` / `AbstractRansacProblem{K}`
# bucket for trait dispatch (sample size, etc.).

"""
    BoundedProblem(inner::AbstractRansacProblem, extent) -> BoundedProblem

Wrap an existing RANSAC problem with a known validity extent. The wrap
forwards all RANSAC interface methods (`solve`, `residuals!`,
`test_sample`, `test_model`, `test_consensus`, ...) to `inner`, but
declares `extent_trait(::BoundedProblem) = WithExtent(extent)` so the
post-IRLS extent-consensus check fires.

`extent` is typically a `HyperRectangle{2}` (board world-coords);
support for other extent shapes can be added by defining more
`_check_extent(::WithExtent, p, model, mask)` methods.
"""
struct BoundedProblem{P <: AbstractRansacProblem, R, K} <: AbstractRansacProblem{K}
    inner::P
    extent::R
end

function BoundedProblem(inner::AbstractRansacProblem{K}, extent) where {K}
    return BoundedProblem{typeof(inner), typeof(extent), K}(inner, extent)
end

extent_trait(p::BoundedProblem) = WithExtent(p.extent)

# Forward field access for fields the wrapper doesn't own.
Base.getproperty(p::BoundedProblem, sym::Symbol) =
    sym === :inner || sym === :extent ? getfield(p, sym) :
    getproperty(getfield(p, :inner), sym)

# RANSAC interface forwarders. Idx-typed methods match the inner problem
# signatures (HomographyProblem etc. use `idx::AbstractVector{Int}`) to
# avoid dispatch ambiguity vs the generic `AbstractRansacProblem` fallbacks.
data_size(p::BoundedProblem) = data_size(p.inner)
sampler(p::BoundedProblem) = sampler(p.inner)
codimension(::Type{<:BoundedProblem{P}}) where {P} = codimension(P)
model_type(::Type{<:BoundedProblem{P}}) where {P} = model_type(P)
solve(p::BoundedProblem, idx::AbstractVector{Int}) = solve(p.inner, idx)
fit_hypothesis(p::BoundedProblem, θ, idx::AbstractVector{Int}) = fit_hypothesis(p.inner, θ, idx)
residuals!(r::Vector, p::BoundedProblem, m) = residuals!(r, p.inner, m)
test_sample(p::BoundedProblem, idx::AbstractVector{Int}) = test_sample(p.inner, idx)
test_model(p::BoundedProblem, m, idx::AbstractVector{Int}) = test_model(p.inner, m, idx)
# BoundedProblem composes the inner problem's consensus check with the
# extent check. The inner result short-circuits — if the model already
# fails the inner predicate (e.g. cheirality straddle), we don't run the
# more expensive extent geometry. Both must succeed for the model to survive.
test_consensus(p::BoundedProblem, m, mask::BitVector) =
    test_consensus(p.inner, m, mask) &&
    _check_extent(extent_trait(p), p, m, mask)
# LO machinery shared with the inner problem: the one local-optimization step.
refine!(ws, p::BoundedProblem, θ, weights::AbstractVector) =
    refine!(ws, p.inner, θ, weights)
chart_dof(::Type{<:BoundedProblem{P}}) where {P} = chart_dof(P)   # sizes the refit workspace (`_lm_size`)
# Spelled at RSDR's arity, not `args...`: a Varargs forwarder was AMBIGUOUS with
# the constraint-problem protocol's `(problem, _model, ::AbstractVector{Int})` fallback,
# so every call on a bounded problem threw a MethodError (predictive scoring).
provides_solver_covariance(::Type{<:BoundedProblem{P}}) where {P} = provides_solver_covariance(P)
solver_covariance(p::BoundedProblem, model, idx::AbstractVector{Int}) =
    solver_covariance(p.inner, model, idx)
constraint_with_jacobians(p::BoundedProblem, model, i::Int) =
    constraint_with_jacobians(p.inner, model, i)
