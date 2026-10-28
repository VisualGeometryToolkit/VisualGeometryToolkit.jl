# =============================================================================
# MinimalSolveProblem — the consolidated correspondence-based RANSAC problem
# =============================================================================
#
# `AbstractCspondProblem{K} <: AbstractRansacProblem{K}` is the abstract base
# for any RANSAC problem whose data is a collection of correspondences. `K`
# is the minimal sample size (compile-time integer).
#
# ONE concrete type, `MinimalSolveProblem{M, K, ...}`, covers the whole
# family (#21 consolidation — it replaces the former `RelativeProblem` AND
# `AbsolutePoseProblem` structs). The per-family behavior is carried by
# VALUES, not by struct identity:
#
#   solver     — a VGC `MinimalSolver` (Homography4pt/Fundamental7pt/
#                Essential5pt/P3P; carries the NativeSolver/PoseLibSolver
#                backend). `solve(problem, idx)` delegates to the VGC verb,
#                which ALWAYS returns `FixedModels` (count=0 on failure) —
#                the per-problem result-wrapping and the unrolled positional
#                solver calls are gone.
#   normalization — a VGC `CspondNormalization` (H/F) or `nothing` (E, pose).
#                A model changes frame by composition with its two maps:
#                `inv(destination_map(N)) ∘ m ∘ source_map(N)` to the data frame,
#                `destination_map(N) ∘ m ∘ inv(source_map(N))` into the normalized one.
#   cs_hartley — the Hartley-normalized gate view (2D-2D) or `nothing`.
#   cs_solve   — the view the minimal solver consumes: the data itself
#                (F, E's unit bearings, unmapped H), the measurement-mapped
#                point view (mapped H), or world→ray columns (pose).
#   context    — `nothing`, or `PoseContext` (backprojected rays + the
#                imaging transform) for the absolute-pose family.
#
# Public names are the type ALIASES below (`HomographyProblem`,
# `FundMatProblem`, `EssentialProblem`, `AbsolutePoseProblem`) — callers
# never see the type parameters. `RelativeProblem` remains as the internal
# union alias for the 2D-2D families.
#
# Correspondence access goes through the VGC `sources`/`targets` protocol;
# storage is a concrete `Csponds`. `p.cs[i]` still yields a `Cspond` (Pair).
# =============================================================================

# Dependencies: StructArrays; VGC Csponds/sources/targets/CspondNormalization/
#               MinimalSolver verbs; all estimation types from parent module.

# =============================================================================
# Abstract Types
# =============================================================================

"""
    AbstractCspondProblem{K} <: AbstractRansacProblem{K}

Base type for RANSAC problems defined by correspondences.

Subtypes must have fields `cs` (a correspondence container satisfying the
`sources`/`targets` protocol) and `_sampler` (an `AbstractSampler`). Provides
default implementations of `data_size(p) = length(p.cs)` and
`sampler(p) = p._sampler`.
"""
abstract type AbstractCspondProblem{K} <: AbstractRansacProblem{K} end

# =============================================================================
# PoseContext — camera-projection context for the absolute-pose family
# =============================================================================

"""
    PoseContext{F}

Absolute-pose problem context: the backprojected unit ray directions (one
per correspondence, precomputed at construction) and the imaging transform
used by the pixel-reprojection residuals.
"""
struct PoseContext{F, M}
    rays::Vector{Vec3{Float64}}
    proj::F
    model::M                       # the full CameraModel: the residual currency is ITS raw pixels
end

# =============================================================================
# MinimalSolveProblem
# =============================================================================

"""
    MinimalSolveProblem{M, K, T, ...} <: AbstractCspondProblem{K}

The one parametric correspondence problem behind [`HomographyProblem`](@ref),
[`FundMatProblem`](@ref), [`EssentialProblem`](@ref) and
[`AbsolutePoseProblem`](@ref), which are partial applications of it: `M` is
the model type, `K` the minimal sample size and `T` the scalar type. The model
type selects the per-model behaviour (codimension, minimal solver, residual
whitening). The problem holds the correspondences in the caller's frame, its
private normalization, the minimal solver, the sampler and, for absolute pose,
a [`PoseContext`](@ref). Construct one through its aliases.
"""
struct MinimalSolveProblem{
        M,
        K,
        T <: AbstractFloat,
        CS,
        CG,
        CV,
        F,
        MS <: VgtkCore.MinimalSolver,
        CX,
        S <: AbstractSampler,
        H1 <: HomographyMap{3, T},
        H2 <: HomographyMap{3, T},
    } <: AbstractCspondProblem{K}
    cs::CS          # data-frame correspondence container (concrete Csponds)
    cs_hartley::CG  # Hartley-normalized gate view (2D-2D) | nothing (pose)
    cs_solve::CV    # view the minimal solver consumes (see header)
    normalization::F  # CspondNormalization{T} | nothing
    solver::MS      # VGC MinimalSolver (carries the backend)
    context::CX     # nothing | PoseContext
    # Per-image MEASUREMENT-FRAME maps: frame-labeled `HomographyMap`s taking a
    # point from the data frame to the model's frame. Identity (the default)
    # ⇒ the model is scored in the data frame. Only a homography takes
    # non-identity maps — the mapped-homography path (`K⁻¹` as measurement
    # maps), one of the declared pinhole islands; E and F refuse them at
    # construction (`_solve_view`). The labels are declared once at
    # construction: `_check_measurement_frames` runs VGC's `∘` junction law
    # there, so a swapped or mislabeled (src, dst) pair is a MethodError at the
    # constructor. Unlabeled constructions carry `UnlabeledFrame` wildcards.
    src_frame_map::H1
    dst_frame_map::H2
    _sampler::S
    # The sample gate's coincidence tolerance in the units of the cameras' image
    # frame (E only; H and F read the scale-free `_COINCIDENT_NORMALIZED`).
    # Pixels by default; a caller whose cameras are in another frame states it
    # there (a pool record's `EssentialProblem(pair)`: 0.1 px times its pixel → pool scale).
    coincident_tol::T
end

# Public type aliases — the user-facing names: PARTIAL applications of the
# struct (model, sample size[, scalar]), so every trailing parameter stays free
# WITH the struct's own bounds — every alias, and every partial application of
# one (`HomographyProblem{Float64}`), is a subtype of `MinimalSolveProblem`, and
# a method on a problem-type pattern (`codimension`, `chart_dof`) matches the
# bare alias too. (Restating the parameters in a `where` clause loses the
# bounds unless each is copied by hand, once per alias.)
const HomographyProblem = MinimalSolveProblem{HomographyMat{T}, 4, T} where {T <: AbstractFloat}
const FundMatProblem = MinimalSolveProblem{FundMat{T}, 7, T} where {T <: AbstractFloat}
const EssentialProblem = MinimalSolveProblem{EssentialMat{T}, 5, T} where {T <: AbstractFloat}
const AbsolutePoseProblem = MinimalSolveProblem{Extrinsics3, 3, Float64}

"""
    RotationProblem

The RANSAC problem of a pure rotation between two calibrated views: the
`MinimalSolveProblem` whose model is a `RotMatrix{3, Float64}` `R` with
`f₂ ∥ R·f₁` for unit bearings `f₁ => f₂` (the infinite homography). Minimal
sample 2 (VisualGeometryToolkitCore's `Rotation2pt`), codimension 2, a 3-dof
chart. Build it with `RotationProblem(camera1, camera2, pixel_cs)` or from a
bearing `Csponds` carrying `M₁`/`M₂` columns; the public estimator is
[`find_rotation`](@ref).
"""
const RotationProblem = MinimalSolveProblem{RotMatrix{3, Float64, 9}, 2, Float64}

# The problems on unit BEARINGS with their back-projection Jacobians (`M₁`/`M₂`
# columns): what reads those columns — the pixel unit of a threshold, the
# coincidence gate — is written once for both.
const _BearingProblem = Union{EssentialProblem, RotationProblem}

# Internal union alias for the 2D-2D (two-view) families — the former
# `RelativeProblem` struct. Methods written against it apply to H/F/E but
# not to the pose family.
const RelativeProblem = MinimalSolveProblem{<:Union{HomographyMat, AbstractFundMat}}

"""
    _has_measurement_maps(p::RelativeProblem) -> Bool

Non-identity `src_map`/`dst_map` ⇒ the DATA frame is not the model's metric
(solver) frame. Minimal solvers must then map the points before solving —
residuals/scoring/LO all conjugate through the maps, and a solve that ignores
them enforces the model class's frame-specific constraints in the wrong frame
(the 2026-07-19 canonical-scored-E bug). Cheap 18-compare, allocation-free.
"""
@inline _has_measurement_maps(p::MinimalSolveProblem{M, K, T}) where {M, K, T} =
    !(to_homogeneous(p.src_frame_map) == SMatrix{3, 3, T, 9}(I) && to_homogeneous(p.dst_frame_map) == SMatrix{3, 3, T, 9}(I))

# =============================================================================
# Trait methods
# =============================================================================

model_type(::Type{<:MinimalSolveProblem{M}}) where {M} = M

# Codimension d_g — stated ONCE per model family, on the problem type pattern,
# which the concrete problem types AND the bare public aliases (UnionAlls over
# the eltype) both match.
codimension(::Type{<:MinimalSolveProblem{<:HomographyMat}}) = 2    # two constraint equations from v̄ × Hū
codimension(::Type{<:MinimalSolveProblem{<:AbstractFundMat}}) = 1  # scalar epipolar constraint (F and E)
codimension(::Type{<:MinimalSolveProblem{Extrinsics3}}) = 2        # two reprojection equations per point

# Per-model DLT solver dispatch (only two methods, no consolidation needed).
_dlt_fn(::MinimalSolveProblem{<:HomographyMat}) = homography_dlt!
_dlt_fn(::MinimalSolveProblem{<:FundMat}) = fundamental_dlt!

# =============================================================================
# Constructors — defined on the public aliases
# =============================================================================
#
# Shared body for the three 2D-2D constructors. Differences across H/F/E are
# entirely encoded in the (M, solver) pair. Any container satisfying the
# `sources`/`targets` protocol is accepted (Csponds, StructArray of Pairs,
# plain Pair vectors); a `Csponds` with a `:score` attribute column supplies
# the PROSAC prior automatically unless `scores` is supplied explicitly.

# Concrete canonical storage: bare Point2 columns, no attribute columns
# (the score column is consumed by the sampler at construction).
const _PointCsponds{T} = VgtkCore.Csponds{
    Point2{T}, Point2{T}, Vector{Point2{T}}, Vector{Point2{T}}, @NamedTuple{},
}

@inline _canonicalize_pairs(cs::_PointCsponds) = cs
# LENGTH-PRESERVING, and it has to be. This used to read `Point2{T}(p[1], p[2])`
# unconditionally, which on a 3-vector bearing takes components 1 and 2 and
# DISCARDS the third — the same index-and-append pattern that makes a unit ray
# read as a gnomonic point, and nearly right whenever z ≈ 1, so it degrades with
# field of view instead of failing. It also returned a bare `Csponds`, dropping
# every attribute column, so `M₁`/`M₂` validated at the constructor did not
# survive to the gate: the check succeeded and the next line destroyed the property
# it had checked.
@inline _canon_pt(::Type{T}, p::StaticArrays.StaticVector{2}) where {T} = Point2{T}(p[1], p[2])
@inline _canon_pt(::Type{T}, p::StaticArrays.StaticVector{3}) where {T} = Point3{T}(p[1], p[2], p[3])

# The problem scalar is promoted from BOTH sides, so neither endpoint type is
# silently narrowed (a Float64 pixel set against Float32 board points).
_canonicalize_pairs(pairs) = _canonicalize_pairs(
    float(promote_type(_point_scalar(eltype(sources(pairs))), _point_scalar(eltype(targets(pairs))))), pairs
)
# A plane-point endpoint as the public `find_*` signatures accept it: a 2-vector,
# plain or with its measurement noise (`Uncertain`, `exact`).
const _PlanePoint = Union{StaticArrays.StaticVector{2}, VgtkCore.Uncertain{<:StaticArrays.StaticVector{2}}}
# A world-point endpoint of the absolute-pose problem: plain numbers, with or
# without measurement noise (physical points are stripped by the public entry).
const _PlainPoint3 = Union{StaticArrays.StaticVector{3, <:Real}, VgtkCore.Uncertain{<:StaticArrays.StaticVector{3, <:Real}}}

# The scalar of an endpoint type, through an `Uncertain` (or `exact`) wrapper.
_point_scalar(::Type{P}) where {P} = eltype(P)
_point_scalar(::Type{<:VgtkCore.Uncertain{V}}) where {V} = eltype(V)
@inline _canon_pt(::Type{T}, u::VgtkCore.Uncertain) where {T} = _canon_pt(T, geom(u))
function _canonicalize_pairs(::Type{T}, pairs) where {T}
    s = sources(pairs); t = targets(pairs)
    src = [_canon_pt(T, p) for p in s]
    dst = [_canon_pt(T, p) for p in t]
    cs = VgtkCore.Csponds(src, dst)
    # Attribute columns travel with the endpoints they index. Anything derived
    # per correspondence — `M₁`/`M₂`, weights, priors — subsets in lockstep
    # from here on, which is the point of their being columns at all. The
    # endpoints' measurement noise is two more (`Σ₁`/`Σ₂`, VGC `noise_columns`:
    # absent for plain points, so a plain set is the set it always was).
    attrs = merge(VgtkCore.attrs_of(pairs), VgtkCore.noise_columns(s, t))
    return isempty(attrs) ? cs : VgtkCore.withattrs(cs; attrs...)
end

# The measurement-noise rule (VGC conventions/measurement-noise.md) is honored
# by every route a problem computes its residual through — scoring, the chart
# rows of the refit and the lift, the solver covariance. A problem class whose
# routes do not all read the `Σ₁`/`Σ₂` columns refuses them rather than
# scoring one function and descending another.
_check_noise_support(::Type{<:HomographyMat}, cs) = cs
function _check_noise_support(::Type{M}, cs) where {M}
    _has_noise_columns(cs) && throw(ArgumentError(
        "$(nameof(M)) problems take plain points (unit isotropic noise): their Sampson " *
            "LM rows do not read measurement covariances. `Uncertain`/`exact` endpoints are " *
            "supported by HomographyProblem."
    ))
    return cs
end
_has_noise_columns(cs) = VgtkCore.hasattr(cs, :Σ₁) || VgtkCore.hasattr(cs, :Σ₂)
_has_noise_columns(::_PointCsponds) = false

# The prologue every correspondence-problem constructor shares: enough
# correspondences for a `K`-point minimal sample, and the sampler — the
# caller's, or PROSAC on the `:score` attribute column (per-correspondence data
# travels as a column, never a keyword); uniform without one.
function _problem_sampler(pairs, K::Int, sampler)
    n = length(pairs)
    _holds_minimal_sample(K, n) || throw(ArgumentError("Need at least $K correspondences, got $n"))
    return sampler === nothing ? _build_sampler(n, K, VgtkCore.attribute_column(pairs, :score)) : sampler
end

# Minimal solver for a model family at a given backend.
_solver_for(::Type{<:HomographyMat}, b::AbstractSolverBackend) = VgtkCore.Homography4pt(b)
_solver_for(::Type{<:FundMat}, b::AbstractSolverBackend) = VgtkCore.Fundamental7pt(b)
_solver_for(::Type{<:EssentialMat}, b::AbstractSolverBackend) = VgtkCore.Essential5pt(b)

# -----------------------------------------------------------------------------
# Measurement-frame maps — the labeled lift point (VGC EpipolarForm adoption)
# -----------------------------------------------------------------------------
#
# The model frames of the 2D-2D families, as VGC frame tags. `:src`/`:dst`
# name the two views by their ROLE in the correspondence (`u₁ => u₂`): the
# essential model is honest on the per-view calibrated frames
# (`GnomonicFrame`), F and H on the per-view pixel frames (`ImageFrame`).
# Callers whose maps carry view-instance ids relabel at this boundary.
#
# E's entry is reached from ONE place: the construction-time junction check
# (`_check_measurement_frames` in `_make_minimal_problem`), which runs on every
# `EssentialProblem` before `_solve_view` refuses non-identity maps. With the
# identity (unlabeled) maps E always ends up holding, the check is vacuous; a
# labeled map pair is judged against these tags before the refusal fires. No
# scoring, LO or covariance route reaches it for E: E's
# `constraint_with_jacobians` is its own bearing method, and
# `_has_measurement_maps` is always false on a constructed E problem.
_model_frames(::Type{<:EssentialMat}) = (GnomonicFrame{:src}, GnomonicFrame{:dst})
_model_frames(::Type{<:FundMat}) = (ImageFrame{:src}, ImageFrame{:dst})
_model_frames(::Type{<:HomographyMat}) = (ImageFrame{:src}, ImageFrame{:dst})

# A measurement map as a framed `HomographyMap` (data frame → model frame) in
# the problem scalar `T`: VGC's lift of a raw matrix (`UnlabeledFrame` tags —
# the unchecked wildcard) or its eltype conversion of a framed map (labels and
# stored inverse kept). The raw hot-loop currency `src_map` is its `.forward`.
@inline _measurement_map(::Type{T}, m::HomographyMap{3}) where {T} = HomographyMap{3, T}(m)
@inline _measurement_map(::Type{T}, m::StaticArrays.StaticMatrix{3, 3}) where {T} =
    HomographyMap(SMatrix{3, 3, T, 9}(m))
# The identity map a problem without a measurement frame carries.
@inline _identity_map(::Type{T}) where {T} = _measurement_map(T, SMatrix{3, 3, T, 9}(I))

# Model → measurement-frame boundary for the epipolar family, as VGC's CHECKED
# frame law, composition: the data-frame form is `inv(C₂) ∘ M ∘ C₁` (C the
# data→model lifts), and each junction verifies that the map met the model
# frame it was declared for — a labeled mismatch (e.g. a swapped pair) is a
# MethodError, an unlabeled map is the standing wildcard.
@inline function _measurement_epipolar(
        ::Type{M}, mat::AbstractFundMat, src_hm::HomographyMap{3}, dst_hm::HomographyMap{3}
    ) where {M}
    S1, S2 = _model_frames(M)
    return EpipolarForm(inv(dst_hm) ∘ EpipolarMap{S1, S2}(mat) ∘ src_hm)
end

# Construction-time frame check: run VGC's own junction law once on an
# identity placeholder so a mislabeled or swapped map pair fails at the
# constructor, not deep in a residual kernel. The epipolar family (E/F) checks
# through the epipolar map's `∘` junctions; H — a collineation — checks
# through the Map-layer `∘` law on `H_meas = C₂⁻¹ ∘ H ∘ C₁` (same wildcard
# truth table). Unlabeled lifts go unchecked, as everywhere.
# E and F have canonical model frames — E is calibrated by definition
# (`GnomonicFrame`), F is a pixel-plane model by definition (`ImageFrame`) — so
# there is nothing for a caller to declare and the argument is refused rather
# than silently ignored.
function _check_measurement_frames(::Type{M}, src_hm, dst_hm, frames) where {M <: AbstractFundMat}
    frames === nothing || throw(ArgumentError(
        "model_frames is only meaningful for a homography: $(nameof(M))'s model " *
            "frame is fixed by the model class (E calibrated, F pixel-plane)."
    ))
    _measurement_epipolar(
        M, FundMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(I))), src_hm, dst_hm
    )
    return nothing
end
# The model frame is declared INDEPENDENTLY of the maps, and it has to be.
#
# It is true that a mapped homography's model frame is not a property of
# `HomographyMat` — `ImageFrame` is right only when the maps are identity and
# the data frame IS the model frame. But the fix is NOT to read the frame off
# the maps: this check exists to verify that the maps land where the model
# lives, and a reference derived from the maps themselves verifies nothing.
#
# Measured, because it was tried: deriving `S1`, `S2` from the maps' own
# targets makes a SWAPPED pair type-check — `src_map = V2, dst_map = V1`
# stops throwing, because whatever the maps say becomes the definition. The
# junction test at ransac_essential_tests.jl:182 caught it immediately.
#
# So the CALLER declares the model frame and the maps are checked against that
# declaration. `model_frames` is the missing fact, not a second way to say one
# the maps already carry: nothing else in the problem records where the model
# lives, and the declaration's whole value is having a provenance different
# from the thing it checks. A calibrated H declares
# `(GnomonicFrame{:src}, GnomonicFrame{:dst})`, since K⁻¹u is the z = 1 chart. The default preserves every
# existing caller, and an unlabeled map stays the wildcard it has always been.
function _check_measurement_frames(::Type{<:HomographyMat}, src_hm, dst_hm, frames)
    S1, S2 = frames === nothing ? _model_frames(HomographyMat) : frames
    ident = relabel(S1, S2, HomographyMap(SMatrix{3, 3, Float64, 9}(I)))
    inv(dst_hm) ∘ (ident ∘ src_hm)
    return nothing
end

# The solver-frame view for the 2D-2D families: the data itself when the data
# frame IS the solver frame; the measurement-mapped point view otherwise
# (lazy — columns map on access). Only H has a mapped-solve path; E and F
# refuse maps at construction.
#
# The epipolar family TAKES NO MEASUREMENT MAPS.
#
# F: no caller needs a mapped F, and an unused second frame route is how a
# transform comes to be expressed twice. An F problem's data are pixels; that
# is the whole contract (until 2026-09-30 a mapped F was accepted here and
# refused at every `solve`, with a scoring/LO path for it nothing could reach).
#
# E: A map is a MATRIX, so the best it can produce is
# a point on the z = 1 plane — the deleted `_mapped_bearing` returned exactly
# that, and its name said "bearing" while its docstring said "homogeneous z=1 point".
#
# That is not a naming quibble. Measured on a narrow field at f = 800, its
# output has z ≡ 1 and norms 1.0012–1.0185; a genuine back-projection returns
# 1.0 exactly. BOTH satisfy a `StaticVector{3}` check, so the type refusal that
# closes the 2-vector hole does not close this one — the chart survives the
# lift, having only stopped being visible in the type. It is worse further off
# axis (norm 1.41 at 45°) and undefined at 90°, which is the whole reason this
# wave exists.
#
# The fix is structural, not a guard: bearings come from the camera's own
# `backproject`, which is defined for every ray the camera can see. Use
# `EssentialProblem(camera1, camera2, pixel_cs)`.
function _solve_view(::Type{M}, cs, sm::SMatrix{3, 3, T, 9}, dm::SMatrix{3, 3, T, 9}) where {M <: AbstractFundMat, T}
    ident = SMatrix{3, 3, T, 9}(I)
    (sm == ident && dm == ident) || throw(ArgumentError(_no_maps_message(M)))
    return cs
end
_no_maps_message(::Type{<:EssentialMat}) =
    "EssentialProblem takes no measurement maps. A map is a matrix, so it " *
    "can only reach the z = 1 chart — which is undefined for the rays this " *
    "model class exists to handle. Back-project with the camera instead: " *
    "`EssentialProblem(camera1, camera2, pixel_cs)`."
_no_maps_message(::Type{<:FundMat}) =
    "FundMatProblem takes no measurement maps: build it from pixel " *
    "correspondences. (Only a homography carries maps; F has no caller " *
    "that needs one, so no mapped path exists.)"
# A homography acts on 2D projective points, so its mapped view carries points
# through the measurement map WITH the perspective divide (VGC's
# `PerspectiveMap() ∘ HomographyMap`) — unlike E, whose solver consumes
# bearings. Identity maps return `cs` itself (not a copy), which is what makes
# the unmapped path bit-identical by construction rather than by argument. The
# mapped points stay in the problem scalar `T` (the maps' eltype), so the
# normalization built on them — and every model `solve` composes back through
# it — is a `HomographyMat{T}`, the problem's `model_type`, exactly as on the
# unmapped path.
function _solve_view(::Type{<:HomographyMat}, cs, sm::SMatrix{3, 3, T, 9}, dm::SMatrix{3, 3, T, 9}) where {T}
    ident = SMatrix{3, 3, T, 9}(I)
    (sm == ident && dm == ident) && return cs
    view = VgtkCore.lift(
        cs;
        source = PerspectiveMap() ∘ HomographyMap(sm),
        target = PerspectiveMap() ∘ HomographyMap(dm)
    )
    _has_noise_columns(cs) || return view
    # The measurement noise moves with its points: Σ′ = J·Σ·Jᵀ, J the map's
    # derivative at the point (projective, so per point). Only the solve view
    # holds these — scoring and the local optimizations read `cs`'s own.
    return VgtkCore.withattrs(view; _mapped_noise(cs, :Σ₁, sources(cs), sm)..., _mapped_noise(cs, :Σ₂, targets(cs), dm)...)
end
function _mapped_noise(cs, k::Symbol, pts, m::SMatrix{3, 3})
    VgtkCore.hasattr(cs, k) || return (;)
    Σ = getproperty(cs, k)
    col = map(eachindex(pts)) do i
        J = _projective_jacobian(m, pts[i])
        SMatrix{2, 2, Float64, 4}(J * Σ[i] * J')
    end
    return NamedTuple{(k,)}((col,))
end
# ∂π(m·x̃)/∂x for a 2-vector x: Jπ(h)·m₁₂, h = m·x̃.
@inline function _projective_jacobian(m::SMatrix{3, 3}, x)
    h = m * SVector{3, Float64}(x[1], x[2], 1)
    w = inv(h[3])
    Jπ = SMatrix{2, 3, Float64, 6}(w, 0, 0, w, -h[1] * w^2, -h[2] * w^2)
    return Jπ * SMatrix{3, 2, Float64, 6}(m[1, 1], m[2, 1], m[3, 1], m[1, 2], m[2, 2], m[3, 2])
end

# E builds NO normalization. Bearings are unit-norm and scale-free, so there is
# nothing for a conditioning step to condition; the Hartley view it used to
# carry only supplied a frame for a tolerance that now lives in pixels, and
# that frame was one of the three the single coincidence constant was being
# read in. `nothing` here also removes E's round trip through a normalized frame,
# which was overhead on a quantity its model gate discarded.
_needs_normalization(::Type{<:EssentialMat}) = false
_needs_normalization(::Type) = true

# `CspondNormalization(M, cs)` — the per-model correspondence normalization — is
# VisualGeometryToolkitCore's (solver layer): the H/F/E and RadialHomography strategies
# are registered there. RVG consumes it below; it defines no strategies of its own.

# THE construction path of every `MinimalSolveProblem` (H/F/E, the absolute
# pose, the radial homography and the radial camera): each family's constructor
# canonicalizes its data (`_canonicalize_pairs`, attribute columns kept), builds
# its views, normalization, solver and sampler, and lands here; the type
# parameters are read off the values, the problem scalar `T` off the
# canonical data.
function MinimalSolveProblem{M, K}(
        cs, cs_hartley, cs_solve, normalization, solver, context, sampler,
        src_frame_map = _identity_map(eltype(eltype(sources(cs)))),
        dst_frame_map = src_frame_map;
        coincident_tol::Real = VgtkCore.COINCIDENT_PX
    ) where {M, K}
    T = eltype(eltype(sources(cs)))
    coincident_tol > 0 || throw(ArgumentError("coincident_tol must be positive, got $(coincident_tol)"))
    return MinimalSolveProblem{
        M, K, T, typeof(cs), typeof(cs_hartley), typeof(cs_solve),
        typeof(normalization), typeof(solver), typeof(context), typeof(sampler),
        typeof(src_frame_map), typeof(dst_frame_map),
    }(
        cs, cs_hartley, cs_solve, normalization, solver, context,
        src_frame_map, dst_frame_map, sampler, T(coincident_tol),
    )
end

function _make_minimal_problem(
        ::Type{M}, ::Val{K}, pairs;
        sampler::Union{Nothing, AbstractSampler} = nothing,
        backend::AbstractSolverBackend = NativeSolver(),
        src_map::Union{StaticArrays.StaticMatrix{3, 3}, HomographyMap{3}} = SMatrix{3, 3, Float64, 9}(I),
        dst_map::Union{StaticArrays.StaticMatrix{3, 3}, HomographyMap{3}} = SMatrix{3, 3, Float64, 9}(I),
        model_frames = nothing,
        coincident_tol::Union{Nothing, Real} = nothing
    ) where {M, K}
    # E's sample gate reads `coincident_tol` in the cameras' units; H and F
    # read the scale-free `_COINCIDENT_NORMALIZED` on their normalized view, so
    # a tolerance stated for them would be read by nothing: refused.
    coincident_tol === nothing || M <: EssentialMat || throw(ArgumentError(
        "coincident_tol is the essential problem's sample-gate tolerance; a $(nameof(M)) " *
            "problem's gate reads the scale-free tolerance on its normalized view"))
    smplr = _problem_sampler(pairs, K, sampler)
    cs = _check_noise_support(M, _canonicalize_pairs(pairs))
    T = eltype(eltype(sources(cs)))
    slv = _solver_for(M, backend)
    @assert K == VgtkCore.sample_size(slv)
    # Labeled lift of the measurement maps + VGC junction check (a swapped or
    # mislabeled framed pair throws here); raw matrices lift unlabeled.
    #
    # These come BEFORE the normalization, because the normalization conditions
    # the view the solver consumes: for a mapped homography that is the MAPPED
    # view (conditioning the raw pixels and then solving mapped data against it
    # would conflate two frames); for F and unmapped H `cs_solve === cs`. The
    # DLT refit reads the same view.
    sm_h = _measurement_map(T, src_map); dm_h = _measurement_map(T, dst_map)
    _check_measurement_frames(M, sm_h, dm_h, model_frames)
    cs_solve = _solve_view(M, cs, to_homogeneous(sm_h), to_homogeneous(dm_h))
    normalization = _needs_normalization(M) ? CspondNormalization(M, cs_solve) : nothing
    cs_hartley = normalization === nothing ? nothing : LinearAlgebra.normalize(normalization, cs_solve)
    return MinimalSolveProblem{M{T}, K}(cs, cs_hartley, cs_solve, normalization, slv, nothing, smplr, sm_h, dm_h;
        coincident_tol = something(coincident_tol, VgtkCore.COINCIDENT_PX))
end

# THE construction of a centred-model problem — the radial homography and the
# radial camera, whose model is defined about a distortion centre that is
# problem DATA: the canonical data, the model class's fixed-centre
# normalization (`CspondNormalization(M, cs; center)`), and the normalized view
# as the solve view (the kernels are origin-centred). `context` is the problem's
# datum (the radial camera's centre, `_with_datum`), or `nothing`.
function _make_centered_problem(::Type{M}, K::Int, pairs, solver, sampler, center, context) where {M}
    smplr = _problem_sampler(pairs, K, sampler)
    cs = _exact_source_only(M, _canonicalize_pairs(pairs))
    T = eltype(eltype(sources(cs)))
    normalization = CspondNormalization(M, cs; center = SVector{2, Float64}(center[1], center[2]))
    cs_norm = LinearAlgebra.normalize(normalization, cs)
    return MinimalSolveProblem{_model_class(M, T), K}(cs, cs_norm, cs_norm, normalization, solver, context, smplr)
end
# A centred model's residual is the reprojection distance in the TARGET image
# (pixels): its source — a board or plane point — is exact by construction, the
# measurement-noise rule's `exact` source with a unit isotropic target. An
# `exact` source column therefore says what the residual already does and is
# dropped; any other covariance is refused, as is a noisy source the residual
# could not see.
_exact_source_only(::Type, cs::_PointCsponds) = cs
function _exact_source_only(::Type{M}, cs) where {M}
    _has_noise_columns(cs) || return cs
    exact_src = VgtkCore.hasattr(cs, :Σ₁) && all(iszero, cs.Σ₁)
    (exact_src && !VgtkCore.hasattr(cs, :Σ₂)) || throw(ArgumentError(
        "$(nameof(M)) problems score the reprojection distance in the target image: the " *
            "source is exact and the target unit isotropic noise. Supply the sources " *
            "`exact` (or plain) and the targets plain; other covariances are not supported."
    ))
    a = VgtkCore.attrs_of(cs)
    rest = NamedTuple{filter(!=(:Σ₁), keys(a))}(a)
    return VgtkCore.Csponds(sources(cs), targets(cs), rest)
end
_model_class(::Type{VgtkCore.RadialHomography}, ::Type{T}) where {T} = VgtkCore.RadialHomography{T}
_model_class(::Type{VgtkCore.RadialCamera}, ::Type) = VgtkCore.RadialCamera{Float64}

"""
    HomographyProblem(pairs; sampler=nothing, backend=NativeSolver(),
                      src_map=I, dst_map=I, model_frames=nothing)

Construct a `HomographyProblem` from 2D-2D point correspondences.

- `pairs`: any correspondence container satisfying the `sources`/`targets`
  protocol (`Csponds`, `StructArray{<:Pair}`, plain Pair vector). The PROSAC
  prior is its `:score` attribute column when `pairs` is a `Csponds` that has
  one (VGC `withattrs(cs; score = s)`, the descriptor matcher's output):
  sampling ordered by descending score; uniform otherwise. Endpoints carry
  their measurement noise (VisualGeometryToolkitCore
  `conventions/measurement-noise.md`): plain = unit isotropic in the point's
  own frame, `Uncertain(p, CovMat(Σ))` = its covariance, `exact(p)` = none
  (not both endpoints). The constraint is whitened by
  `G₁Σ₁G₁ᵀ + G₂Σ₂G₂ᵀ` in scoring, the refit and the lifts alike; a board or
  world plane is `exact`, so its residual is the transfer error in pixels.
- `sampler`: override the auto-built sampler (advanced).
- `src_map`/`dst_map`: per-view measurement maps, data frame → model frame.
  Identity (the default) means the data frame IS the model frame: pixels in,
  a pixel-frame `H` out. Non-identity maps declare a calibration instead of
  baking it into the data — supply PIXELS and `K⁻¹` maps, and the returned `H`
  is in the model frame the maps point at, verified to 3.7e-13 against a known
  calibrated `H`. Scoring, the local optimizations and the consensus gate all
  treat that `H` consistently: residuals are measured in the data frame
  (pixels), the returned model stays in the model frame.
- `model_frames`: the `(source, destination)` frame tags of the model frame,
  checked against LABELED maps by VGC's junction law at construction. Default
  `nothing` means `(ImageFrame{:src}, ImageFrame{:dst})`.

# Which frame is which: declared, then checked against the maps

The model frame of a mapped homography is not a property of `HomographyMat` —
and it is NOT read off the maps: the maps are what gets checked, so the
reference they are checked against is declared separately. Raw-matrix maps
(and unlabeled `HomographyMap`s) are the unchecked wildcard and need no
declaration. Labeled maps must land in the declared frames, or construction is
a `MethodError`; a calibrated `H` whose `K⁻¹` maps are labeled into the z = 1
chart therefore declares it:

```julia
HomographyProblem(pixel_cs; src_map = C₁, dst_map = C₂,
                  model_frames = (GnomonicFrame{:src}, GnomonicFrame{:dst}))
```

# Limit: mapped homographies are PINHOLE by construction

A measurement map is a MATRIX. A general camera's back-projection is not — it
is nonlinear, and cannot be expressed as a map at all. So anything routed
through `src_map`/`dst_map` is restricted to pinhole cameras, and this path is
a pinhole island for the same structural reason as the relative-pose bundle's
`K` matrices.

Worth being exact about where the restriction comes from: a calibrated
homography is perfectly well defined on bearings — `f₂ ∼ H f₁` needs no affine
plane — so the limit is imposed by the MAP MECHANISM, not by the geometry.
Camera-general calibrated `H` would take back-projection in place of `K⁻¹`,
which is a separate decision alongside the relative-pose bundle carrying camera
models. Nothing is lost for a calibrated pinhole benchmark.
"""
HomographyProblem(pairs; kw...) =
    _make_minimal_problem(HomographyMat, Val(4), pairs; kw...)

"""
    FundMatProblem(pairs; sampler=nothing, backend=NativeSolver())

Construct a `FundMatProblem` from 2D-2D point correspondences. See
[`HomographyProblem`](@ref) for the input container, its `:score` prior and
`sampler`.
"""
FundMatProblem(pairs; kw...) =
    _make_minimal_problem(FundMat, Val(7), pairs; kw...)

# A PAIR VECTOR cannot carry attribute columns, so it cannot carry `M₁`/`M₂` —
# and accepting one would reopen the hole this contract exists to close: the
# problem would build, the gate would find no columns, and it would fall back
# to treating bearings as if a tolerance in their own units meant pixels. The
# refusal is explicit rather than a `MethodError` so the message can say what
# to build instead.
function EssentialProblem(
        ::AbstractVector{<:Pair{<:StaticArrays.StaticVector{3}, <:StaticArrays.StaticVector{3}}};
        kw...
    )
    throw(ArgumentError(
        "EssentialProblem needs the back-projection Jacobians, which a vector of " *
            "Pairs cannot carry. Build a `Csponds` and attach them as columns: " *
            "`withattrs(Csponds(f₁, f₂); M₁, M₂)`, from " *
            "`backproject_with_jacobian(camera, u)`."
    ))
end

function _assert_unit_bearings(fs, which::AbstractString, who::AbstractString = "EssentialProblem")
    worst = 0.0; at = 0
    @inbounds for i in eachindex(fs)
        f = fs[i]
        # FINITE first: `NaN > worst` is false, so a NaN bearing (a pixel the
        # camera cannot back-project, e.g. past a fisheye's fold) sailed through
        # the norm test below and poisoned every LO it entered.
        all(isfinite, f) || throw(ArgumentError(
            "$who needs finite bearings; the $which at index $i is " *
                "$(Tuple(f)). A camera returns NaN for a pixel it cannot " *
                "back-project (e.g. past a fisheye's fold) — drop that " *
                "correspondence before building the problem."
        ))
        d = abs(norm(f) - 1)
        d > worst && (worst = d; at = i)
    end
    worst <= 1.0e-6 || throw(ArgumentError(
        "$who needs UNIT bearings; the $which at index $at has norm " *
            "$(round(norm(fs[at]), digits = 6)). A z = 1 chart point satisfies the " *
            "3-vector signature but is not a bearing — its norm is 1 only on the " *
            "optical axis, ~1.41 at 45° off it, and it does not exist at all at " *
            "90°. Back-project with the camera: " *
            "`$who(camera1, camera2, pixel_cs)`."
    ))
    return nothing
end

# The back-projection Jacobians carry every tolerance into pixels, so a
# non-finite one is as poisonous as a non-finite bearing.
function _assert_finite_jacobians(Ms, name::AbstractString, who::AbstractString = "EssentialProblem")
    @inbounds for i in eachindex(Ms)
        all(isfinite, Ms[i]) || throw(ArgumentError(
            "$who needs finite back-projection Jacobians; `$name` at " *
                "index $i is not finite. Drop that correspondence before building " *
                "the problem."
        ))
    end
    return nothing
end

"""
    EssentialProblem(camera1, camera2, pixel_cs; kw...)

Build an `EssentialProblem` from PIXEL correspondences and the two cameras that
produced them — the entry most callers want, and the only one that cannot get
the frames wrong.

Each view is back-projected with its OWN camera through
`backproject_with_jacobian`, so `M₁` belongs with the source bearings and `M₂`
with the targets by construction rather than by the caller remembering. That
matters more than it looks: swapping the two is norm-preserving and invisible
to Monte-Carlo calibration, and shows only in the tangency contract.

Two cameras, not one, because they are genuinely independent — and because a
single-camera convenience form would make the swap undetectable in exactly the
tests that ought to catch it.
"""
function EssentialProblem(camera1, camera2, pixel_cs; kw...)
    u₁ = sources(pixel_cs); u₂ = targets(pixel_cs)
    fm₁ = map(u -> VgtkCore.backproject_with_jacobian(camera1, u), u₁)
    fm₂ = map(u -> VgtkCore.backproject_with_jacobian(camera2, u), u₂)
    # The caller's attribute columns travel with the correspondences they index
    # — above all `:score`, the PROSAC prior: dropping it silently turned a
    # scored E into uniform sampling while H and F honoured the same column.
    cols = merge(VgtkCore.attrs_of(pixel_cs), (M₁ = last.(fm₁), M₂ = last.(fm₂)))
    return EssentialProblem(
        VgtkCore.Csponds(first.(fm₁), first.(fm₂), cols); kw...
    )
end

"""
    EssentialProblem(bearings; sampler=nothing, backend=NativeSolver())

Construct an `EssentialProblem` from **unit-bearing** correspondences: 3-vector
rays on S², carrying their back-projection Jacobians as the attribute columns
`M₁`/`M₂` (`withattrs(cs; M₁, M₂)`, from
`backproject_with_jacobian`).

Differs from `FundMatProblem` in coordinate space — bearings, not pixels — and
in minimal sample size (5 vs 7).

# Why 3-vectors, and why the length is enforced

The old contract took calibrated 2-vectors and implied `z = 1`. That is the
gnomonic chart, and it does not exist for a ray at or past the horizon — which
is exactly the rays a fisheye or omnidirectional camera produces. Bearings have
no such gap.

The length is a **type constraint, not a convention**, because the failure it
prevents is silent. A consumer that indexes components 1 and 2 of a bearing and
appends 1.0 gets a *nearly right* answer whenever `z ≈ 1`, so it degrades with
field of view rather than failing — the same signature as the defect this
contract removes. A wrong-length input is therefore a `MethodError` here, and
there is deliberately no method accepting either length.

# Why `M₁`/`M₂` are required rather than defaulted

`M = ∂f/∂u` is what carries the gate's and the scoring's tolerance into PIXELS:
`test_sample` converts an angular separation as `angle / σ_max(M)`. **`M` picks
the unit and the call cannot** — an orthonormal tangent basis would put the
same threshold in radians. With no columns there is no defensible unit, so
their absence throws rather than falling back; a fallback is the path by which
a pixel tolerance silently becomes an angular one.

Both are per-view: `M₁` belongs with the source bearings and `M₂` with the
targets. Swapping them is norm-preserving and invisible to Monte-Carlo
calibration — measured median ratio 1.018, best cell inside a 4 SE band — and
shows only in the tangency contract, `fᵀM = 0`, which separates a correct
wiring (~1e-19) from a swapped one (~1e-3) by sixteen orders of magnitude.

E takes no measurement maps: a non-identity `src_map`/`dst_map` is an
`ArgumentError`. Build the bearings with the cameras instead,
`EssentialProblem(camera1, camera2, pixel_cs)`.
"""
function EssentialProblem(bearings::VgtkCore.Csponds; kw...)
    # A `Csponds` carrying the columns arrives here rather than as a Pair vector,
    # so the length check is on its element type and the column check is explicit.
    (eltype(sources(bearings)) <: StaticArrays.StaticVector{3} &&
        eltype(targets(bearings)) <: StaticArrays.StaticVector{3}) || throw(ArgumentError(
        "EssentialProblem takes unit BEARINGS (3-vectors). Got " *
            "$(eltype(sources(bearings))) => $(eltype(targets(bearings))). Calibrated " *
            "2-vectors imply z = 1 — the gnomonic chart — which does not exist for a " *
            "ray at the horizon. Use `backproject_with_jacobian(camera, u)`."
    ))
    # UNIT-NORM, checked once here and deliberately not in the solver: there it
    # would be a per-sample tolerance in a hot loop, validating data the caller
    # built, and redundant once the source is a real back-projection.
    #
    # This is not a delicate tolerance. A z = 1 chart point has norm ≥ 1 — about
    # 1.0185 even on a narrow field, 1.41 at 45° off axis — and coincides with a
    # unit bearing only exactly on the optical axis, where the two vectors are
    # identical anyway. So the margin is enormous everywhere it matters, and the
    # check catches precisely the case the `StaticVector{3}` signature cannot:
    # the chart wearing 3-vector clothing.
    _assert_unit_bearings(sources(bearings), "source")
    _assert_unit_bearings(targets(bearings), "target")
    (VgtkCore.hasattr(bearings, :M₁) && VgtkCore.hasattr(bearings, :M₂)) ||
        throw(ArgumentError(
        "EssentialProblem requires the back-projection Jacobians as attribute " *
            "columns: `withattrs(cs; M₁ = ..., M₂ = ...)`, from " *
            "`backproject_with_jacobian(camera, u)`. They are what put the gate's " *
            "and the scoring's tolerance in PIXELS — M picks the unit and the call " *
            "cannot — so there is deliberately no default."
    ))
    _assert_finite_jacobians(bearings.M₁, "M₁")
    _assert_finite_jacobians(bearings.M₂, "M₂")
    return _make_minimal_problem(EssentialMat, Val(5), bearings; kw...)
end

# =============================================================================
# AbsolutePoseProblem constructor
# =============================================================================

"""
    AbsolutePoseProblem(model::CameraModel, pairs; backend=NativeSolver())

Construct an absolute-pose (P3P) problem from 3D-2D correspondences and a
camera model. Rays are precomputed via `backproject(model, u)` at
construction time; `cs_solve` pairs the world points with those rays for the
P3P solver. Accepts any `sources`/`targets`-protocol container of
3D-source / 2D-target correspondences; a `Csponds` `:score` column is the
PROSAC prior.

The residual currency is the RAW pixel of `model` for every lens: scoring
projects through the intrinsics AND the lens (`residuals!`), and both local
optimizations refine the same reprojection — the Student-t M-step and the
Welsch lift run on VGC's full-camera `constraint_with_jacobians(model, pose,
X, u)`. A `Pinhole` model keeps the classic calibrated kernels
(bit-identical to before). Nothing is undistorted: supply the detections as
measured and the camera as calibrated.

The world points are plain numbers (`StaticVector{3, <:Real}`, or an
`Uncertain` one): a problem is the inner loop, and physical points (lengths)
are a `MethodError` here — [`find_absolute_pose`](@ref) takes them and strips
them in metres once.
"""
function AbsolutePoseProblem(
        model::CameraModel, pairs::AbstractVector{<:Pair{<:_PlainPoint3}};
        backend::AbstractSolverBackend = NativeSolver()
    )
    smplr = _problem_sampler(pairs, 3, nothing)
    cs = _canonicalize_pairs(Float64, pairs)          # Point3 / Point2 storage, Float64
    src = sources(cs)
    rays = backproject.(Ref(model), targets(cs))
    proj = VgtkCore.imaging_transform(intrinsics(model), lens(model))
    ctx = PoseContext(rays, proj, model)
    slv = VgtkCore.P3P(backend)
    cs_solve = VgtkCore.Csponds(src, rays)
    return MinimalSolveProblem{Extrinsics3, 3}(cs, nothing, cs_solve, nothing, slv, ctx, smplr)
end

# =============================================================================
# Shared Interface Methods
# =============================================================================

data_size(p::AbstractCspondProblem) = length(p.cs)
sampler(p::AbstractCspondProblem) = p._sampler

# =============================================================================
# solve — ONE method: the VGC MinimalSolver verb on the solver's view, every
# root made the problem's model (`_candidate`)
# =============================================================================
#
# The VGC verbs ALWAYS return `FixedModels` (count=0 on failure), so there is
# no per-problem result wrapping. A problem with a normalization solves on the
# NORMALIZED view it owns (`cs_hartley`; for a mapped homography the
# normalized MAPPED view) and every root is composed back into the data frame —
# the minimal kernels never normalize, the problem's one normalization is theirs.
# A problem without one (E on its unit bearings, the pose on its world→ray
# view) solves on `cs_solve`.
solve(p::MinimalSolveProblem, idx::AbstractVector{Int}) =
    map(m -> _candidate(p, m), solve(p.solver, _solver_view(p), idx))

@inline _solver_view(p::MinimalSolveProblem) = p.normalization === nothing ? p.cs_solve : p.cs_hartley

# A solver root as the problem's model. Without a normalization (the pose) the
# root IS the model. With one, the frame change is COMPOSITION with the
# normalization's two maps (VGC `∘` on the model class) — then the model class's
# own policy: its class (`_model_class`: a plain-homography root of the radial
# homography's 4-point path gains its zero distortion term) and its datum
# (`_with_datum`: the radial camera's exact centre).
_candidate(::MinimalSolveProblem, m) = m
function _candidate(
        p::MinimalSolveProblem{<:Union{HomographyMat, FundMat, VgtkCore.RadialHomography, VgtkCore.RadialCamera}}, m
    )
    N = p.normalization
    return _with_datum(p, inv(destination_map(N)) ∘ _model_class(p, m) ∘ source_map(N))
end
@inline _model_class(::MinimalSolveProblem, m) = m
@inline _with_datum(::MinimalSolveProblem, m) = m
# E's 5-pt kernel computes in Float64 on the bearings (no normalization): a
# Float32 problem's roots convert to its eltype.
_candidate(::EssentialProblem{T}, E::EssentialMat) where {T <: AbstractFloat} = convert(EssentialMat{T}, E)

# =============================================================================
# test_model — the model-class gate, on the normalized model and view
# =============================================================================
#
# Every feasibility predicate that runs on a candidate model must see the
# model in its normalized frame. Pixel-coordinate scale alone (e.g. a
# homography with h_31 ~ 1e-4 mapping a 1000×1000 image) can spuriously trip a
# scalar check like `σ₁/σ₃ ≤ τ`, even when the model is bit-exact. So the gate
# composes the model into the stored normalization
# (`destination_map(N) ∘ m ∘ inv(source_map(N))`) and hands
# VGC's model-class gate the normalized view: H (conditioning + cheirality), F
# (the oriented epipolar check), the radial homography (conditioning +
# ray-space cheirality). E has no post-solve gate (and no normalization: its
# bearings are already conditioned); the pose and the radial camera answer
# their own (their files / the default).
function test_model(p::MinimalSolveProblem{<:Union{HomographyMat, FundMat, VgtkCore.RadialHomography}}, model, idx::AbstractVector{Int})
    N = p.normalization
    return test_model(destination_map(N) ∘ model ∘ inv(source_map(N)), p.cs_hartley, idx)
end

# Sample-degeneracy gate — dispatched BY FAMILY.
#
# H and F read the coincidence check SCALE-FREE: on the problem's normalized
# view (`cs_hartley`, the view the minimal solver consumes) at
# `_COINCIDENT_NORMALIZED`, not on raw `p.cs` at VGC's absolute `COINCIDENT_PX`.
# An absolute tolerance is a claim about the caller's units that nothing here can
# check, and it was wrong in both directions: a board → image homography in
# METRES (a 5 cm board) admitted no sample at all (0.1 is two board squares), and
# a calibrated view read 0.1 "px" as 120 px at f = 1200. The normalization's
# scale is the problem's own (the whole data set's spread, fixed at construction,
# per view), so the tolerance is ~1/1400 of the data's mean distance from its
# centroid in every unit system. The homography gate's other two parts are frame-free
# (`_orientation_preserving` a sign, `_not_collinear` a ratio).
#
# For a mapped homography the normalized view is of the MAPPED points, the frame
# the solver works in, so the same statement holds there; the radial homography
# reads its target distinctness at the same constant on its fixed-centre
# normalized view (radial_homography.jl).
const _COINCIDENT_NORMALIZED = 1.0e-3

test_sample(p::MinimalSolveProblem{<:Union{HomographyMat, FundMat}}, idx::AbstractVector{Int}) =
    test_sample(model_type(p), p.cs_hartley, idx; tol_px = _COINCIDENT_NORMALIZED)

# E supplies its BEARINGS. VGC's E gate reads the `M₁`/`M₂` columns and converts
# each pair's angular separation back to pixels as `angle / σ_max(M)`, so the
# shared `COINCIDENT_PX` means pixels here —
# which is the whole point of requiring the columns at construction.
#
# There is no normalized view to supply: the constructor builds none for E.
# Bearings are already conditioned — unit-norm and scale-free — so a Hartley
# view conditioned nothing, and the pre-port arrangement (a tolerance in a
# spread-normalized frame) was the third of the three frames that one constant
# was being read in.
#
# `σ_max` is the conservative choice and it is anisotropy-dependent: the
# estimate equals the true separation along M's major axis and under-states it
# by 1/aniso along the minor, so the gate rejects true separations up to
# `tol_px × aniso` — a WIDER net with anisotropy, never a narrower one. It
# begins rejecting genuinely distinct features only at `aniso > d_min/tol_px`,
# i.e. above 10 for a 1 px minimum separation, against 1.48 measured at field
# radius 386 px on a KannalaBrandt fisheye.
#
# `COINCIDENT_PX` means pixels only when the CAMERAS are in pixels: `M` carries
# the separation into the cameras' image frame, whatever that is. A problem
# built from cameras in another frame (a pool record's `EssentialProblem(pair)`, its cameras in
# `NormalizedImageFrame` units) states the tolerance there, as
# `coincident_tol` — read below. Until 2026-10-04 this read the 0.1 px
# default in POOL units (≈ 200 px at a 2000 px half-diagonal): a live RANSAC
# on the pool-frame E rejected 94–100% of its samples as "coincident", and every
# sample on pairs whose SNN mass sits on a tight cluster (7% of PhotoTourism
# test pairs returned no model).
test_sample(p::MinimalSolveProblem{<:EssentialMat}, idx::AbstractVector{Int}) =
    test_sample(model_type(p), p.cs, idx; tol_px = p.coincident_tol)

# constraint_with_jacobians — VGC's, in the DATA frame where `residuals!` scores.
#
# E's predictive-scoring constraint, on BEARINGS with their per-view Jacobians.
#
# THE THIRD ROUTE TO ONE QUANTITY, and the third place this port had to change
# together: `residuals!`, the LM rows, and this. Changing two of the three is
# what produced a 446x LM/scoring mismatch earlier in this port — an optimiser
# descending a function unrelated to the one being scored while reporting
# `:converged`. The rule that falls out: when several functions compute one
# quantity by different routes, ENUMERATE THEM FIRST and change them together;
# changing a subset is not a partial fix, it is a new defect.
#
# All three now go through VGC's tangent form, so agreement is structural: this
# is VGC's per-correspondence constraint of a bearing set (`cd_pair`, which
# reads the `M₁`/`M₂` columns), at a Float64 E — the arithmetic is in E's
# eltype, so the two scoring routes agree at every problem eltype.
#
# Every H/F/E constraint carries its model's rounding scale
# (`constraint_rounding_scale`) as the 5th value, with the isotropic noise
# shape `I` as the 4th: `whiten(cd...)` then applies the ONE Sampson degeneracy
# rule at the scale `sampson_distance` and the LM rows use, so a
# rounding-level pair is judged the same by every scorer.
@inline _with_rounding(cd::Tuple, ρ) = (cd[1], cd[2], cd[3], I, ρ)

@inline function constraint_with_jacobians(p::MinimalSolveProblem{<:EssentialMat}, model::EssentialMat, i::Int)
    E = EssentialMat{Float64}(Tuple(model))
    ρ = @inbounds VgtkCore.constraint_rounding_scale(E, sources(p.cs)[i], targets(p.cs)[i]; M₁ = p.cs.M₁[i], M₂ = p.cs.M₂[i])
    return _with_rounding(@inbounds(VgtkCore.cd_pair(E, p.cs, i)), ρ)
end

# H, F and E carry constraint rows, so predictive scoring admits them
# (estimators/ransac/covariance.jl).
provides_solver_covariance(::Type{<:RelativeProblem}) = true

# The mapped homography's scoring constraint, in the DATA frame where
# `residuals!` scores, at the composed `H_data = inv(C₂) ∘ H ∘ C₁`.
#
# `G_θ` is then carried into the MODEL's own coordinates (chain rule) so that a
# solver-frame covariance contracts with a `G_θ` in its own frame.
#
# THE CHAIN-RULE FACTOR WAS MEASURED, NOT DERIVED, and the measurement is the reason to
# distrust the obvious answer. `vec(H_data) = kron(C₁ᵀ, C₂⁻¹)·vec(H_model)`, so
# the factor is a RIGHT multiplication by that Kronecker — the homography's
# `g` is a 2-vector and its `G_θ` a 2×9, where the epipolar `g` is scalar and
# its `G_θ` a 9-vector contracted on the LEFT by `kron(C1, C2)`. Transcribing
# the epipolar pattern here is not slightly wrong but catastrophically so:
# finite-differenced against the constraint with `C₁ ≠ C₂` and `fx ≠ fy`, the
# form below agrees to 7.4e-10 while the epipolar analogue is off by 2.0e5.
#
# That is the same class of error as the one recorded above: until 2026-09-21 a
# mapped problem scored `g`, `G_x` and `G_θ` in mismatched frames, and it
# survived review because it looked right on paper.
#
# H and F share the unmapped body (`_data_frame_constraint`): the data frame IS
# the model frame. F takes no measurement maps (refused at construction).
@inline _data_frame_constraint(model, c) =
    _with_rounding(VgtkCore.cd_pair(model, c), VgtkCore.constraint_rounding_scale(model, c.first, c.second))
@inline constraint_with_jacobians(p::MinimalSolveProblem{<:FundMat}, model::FundMat, i::Int) =
    _data_frame_constraint(model, @inbounds p.cs[i])
#
# The 4th value is the correspondence's measurement covariance (VGC
# `pair_covariance`, the measurement-noise rule): `I` for plain points — the
# isotropic Sampson error as always — and the endpoints' own from the problem's
# `Σ₁`/`Σ₂` columns, in the data frame where this constraint lives. An `exact`
# source makes the residual the transfer error in the target's units.
@inline function constraint_with_jacobians(p::RelativeProblem, model::HomographyMat, i::Int)
    c = @inbounds p.cs[i]
    _has_measurement_maps(p) || return _with_noise(_data_frame_constraint(model, c), p.cs, i)
    Hm = inv(p.dst_frame_map) ∘ model ∘ p.src_frame_map         # the DATA frame (VGC `∘` on the matrix)
    g, G_x, G_θ = VgtkCore.constraint_with_jacobians(Hm, c)
    # The chain-rule factor: the composition's own derivative (VGC `compose_jacobian`,
    # ∂vec(H_data)/∂vec(H) = kron(C₁ᵀ, C₂⁻¹)), a RIGHT factor of the 2×9 G_θ.
    J = VgtkCore.compose_jacobian(inv(p.dst_frame_map), HomographyMap(model), p.src_frame_map)
    return (g, G_x, G_θ * J, _noise(p.cs, i), VgtkCore.constraint_rounding_scale(Hm, c.first, c.second))
end
@inline _with_noise(cd::Tuple, cs, i::Int) = (cd[1], cd[2], cd[3], _noise(cs, i), cd[5])
@inline _noise(::_PointCsponds, ::Int) = I
@inline _noise(cs, i::Int) = VgtkCore.pair_covariance(cs, i)

# =============================================================================
# The algebraic DLT step — `refine!` with a `DltWorkspace`
# =============================================================================
#
# A linear problem's step is its one-step algorithm under the weights; θ is not
# read. Excluded points are encoded as `weights[i] = 0`. The LO proper runs the
# per-model NonlinLsq chart LM from θ (its workspace is a ManifoldLmWorkspace);
# handing `refine!` a `DltWorkspace` selects the plain weighted DLT instead —
# for direct callers (tests, offline tooling). The DLT runs in the PROBLEM's
# normalization (`p.normalization`, built once at construction) — an inner-loop step
# never builds its own.
function refine!(ws::DltWorkspace, p::RelativeProblem, _θ, weights::AbstractVector)
    _underdetermined(p, weights) && return nothing
    # The weights travel as the correspondences' column (VGC's DLT reads it).
    model = _dlt_fn(p)(ws.buf, p.normalization, withattrs(p.cs_solve; weights); svd_ws = ws.svd)
    isnothing(model) && return nothing
    # VGC's design-matrix gate at its own threshold (the one source), on the
    # normalized solve that just filled `ws.svd`.
    VgtkCore.test_dlt_conditioning(ws.svd) || return nothing
    return model
end
