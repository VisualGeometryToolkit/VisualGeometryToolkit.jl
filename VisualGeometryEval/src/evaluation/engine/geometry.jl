# =============================================================================
# evaluation/engine/geometry.jl — per-pair pool-frame geometry, built ONCE
# per pair and shared by every selector / LO refit / refit family (moved from
# LocalOptimizationDoneRight's engine, K+ 2026-10-09).
#
# The pool stores everything in the pool frame (`NormalizedImageFrame`: image centre, half diagonal):
# csponds, per-model fundamentals, Sampson distances (so validated loss widths
# are pool-frame quantities), plus the per-image pool-frame maps T. The LO
# therefore SCORES in pool-frame space: E on the pool-frame csponds back-projected
# to unit bearings by the pool-frame camera Kc = T·K (RobustVisualGeometry's
# `EssentialProblem(pair)`). The refined E is metric → pose error is read off
# directly. A pool record is VisualGeometryCore's light `PoolPair`: the LO
# engine never reads a pool's residual matrix.
# =============================================================================

"""
    PairGeom

Per-pair calibration and ground truth, the context that scores a pool model's pose error.
It scores any stored pool-frame fundamental's coherent imc pose error and holds
the pool-frame csponds `cs` (`x1 => x2`) as a concretely typed `Csponds`.
"""
struct PairGeom
    Kc1::SMatrix{3,3,Float64,9}
    Kc2::SMatrix{3,3,Float64,9}
    R_gt::SMatrix{3,3,Float64,9}
    t_gt::SVector{3,Float64}
    cs::Csponds{Point2{Float64}, Point2{Float64}, Vector{Point2{Float64}},
                Vector{Point2{Float64}}, @NamedTuple{}}   # pool-frame csponds
end

"""
    problem_type(p::Symbol) -> Type

The RANSAC problem type of a two-view pool track: `:E` → `EssentialProblem`,
`:F` → `FundMatProblem`, `:H` → `HomographyProblem`. The one map from a
track's letter (a pool's `problem`) to its problem class — the value a
[`PairCtx`](@ref) carries as `problem_type`; any other symbol is an
`ArgumentError`. Inside a function, name the result `P` (a local
`problem_type` would shadow this function).
"""
problem_type(p::Symbol) = problem_type(Val(p))
problem_type(::Val{:E}) = EssentialProblem
problem_type(::Val{:F}) = FundMatProblem
problem_type(::Val{:H}) = HomographyProblem
problem_type(::Val{p}) where {p} =
    throw(ArgumentError("no two-view RANSAC problem type for :$p (expected :E, :F or :H)"))

# The pool-frame problem of each family on a pool record: RobustVisualGeometry's
# constructors on VisualGeometryCore's `PoolPair` (E: the csponds back-projected
# to unit bearings by the pool-frame cameras `Kc = T·K`; F and H: the csponds
# verbatim), with the pool-frame calibrations `(Kc₁, Kc₂)` that serve the pose
# metrics (the F pose upgrade, the calibrated-H decomposition of `model_imc`).
_pair_problems(P::Type{<:Union{EssentialProblem,FundMatProblem,HomographyProblem}}, pair::PoolPair) =
    (P(pair), pool_frame_calibrations(pair)...)

# The stored pool-frame csponds as a concretely typed `Point2{Float64}` Csponds
# (the form PoseLib's refine_* and, through its columns, `cheiral_pose_fast`
# consume).
_pair_csponds(pair) = Csponds(Point2{Float64}.(sources(pair.csponds)),
                              Point2{Float64}.(targets(pair.csponds)))

# -----------------------------------------------------------------------------
# THE imc of a model against the pair's GT, dispatched on the MODEL's type — one
# function for pool models (selector checkpoints, the oracle), refit seeds and
# refit results alike:
#   EssentialMat — a metric essential: max(Δr, Δt) [deg] of its best
#                  decomposition (VGC `imc_pose_error`);
#   FundMat      — a pool-frame fundamental (the E and F pools' storage): the
#                  same metric after the pool-frame calibration upgrade
#                  E = Kc₂ᵀ·F·Kc₁ (`imc_pose_error(F, Kc1, Kc2, R, t)`);
#   HomographyMat — a pool-frame homography: the calibrated H = Kc₂⁻¹·H·Kc₁
#                  decomposed (Malis–Vargas), ROTATION error Δr° minimized over
#                  the candidates — the pools' `errors_r`, same metric.
#
# HOMOGRAPHY IS ROTATION ONLY (2026-07-26): HEB's small-baseline pairs make the H
# decomposition's translation DIRECTION ill-conditioned (the scoring paper's
# "HEB plane-assignment caveat"), so max(Δr, Δt°) saturates ~30°+ on a metric a
# perfect H cannot improve. The house HEB protocol scores rotation only. Until
# 2026-09-21 the HEB pools stored the inverse motion and this metric decomposed
# Hᵀ to match; the pools were migrated (`gt_convention` stamp).
#
# The stored per-model errors are VGC's `pool_imc(pair, j)` — for H exactly this
# rotation error, for E/F the max of the per-axis minima, a lower bound on the
# imc here (the oracle prunes on it).
# -----------------------------------------------------------------------------
"""
    model_imc(model, geom::PairGeom) -> Float64
    model_imc(model, ctx::PairCtx)   -> Float64

The pose error [deg] of `model` against the pair's ground truth.
By the model's type: a metric `EssentialMat` → max(Δr, Δt) of its best decomposition; a
pool-frame `FundMat` → the same after the pool-frame calibration upgrade; a
pool-frame `HomographyMat` → the optimistic decomposition ROTATION error.
"""
model_imc(E::EssentialMat, g::PairGeom) = imc_pose_error(E, g.R_gt, g.t_gt)
model_imc(F::FundMat, g::PairGeom) = imc_pose_error(F, g.Kc1, g.Kc2, g.R_gt, g.t_gt)
model_imc(H::HomographyMat, g::PairGeom) =
    first(optimistic_pose_error(H, g.Kc1, g.Kc2, g.R_gt, g.t_gt))

"""
    refit_seed(ctx::PairCtx, model) -> seed

How a stored pool model becomes the seed of a refit in the pair's problem: an
E pool's pool-frame fundamental is upgraded to the metric essential
`Kc₂ᵀ·F·Kc₁`; F and H pool models seed as stored.
"""
refit_seed(ctx, model) = _refit_seed(ctx.problem_type, model, ctx.geom)
_refit_seed(::Type{<:EssentialProblem}, F, g::PairGeom) = EssentialMat(F, g.Kc1, g.Kc2)
_refit_seed(::Type{<:Union{FundMatProblem,HomographyProblem}}, model, ::PairGeom) = model

"""
    PairCtx

Everything the refit layer needs about one pair.
Built ONCE per pair by [`build_pair_ctx`](@ref) and reused across every (LO
refit spec × grid column), it holds the pair itself (a light `PoolPair`), its pool-frame problem (`ep`), the optional
`RelposeBundleProblem` (`bp`, `nothing` unless `bundle=true`), the geometry,
the problem class (`problem_type`) and the refit scoring TEMPLATES: `scoring`,
the scoring every native refit is built from at its own outlier half-width
(`with_width(scoring, OutlierHalfwidth(nat′))`), and `solo_scoring`, the one an
absolute-width (solo) refit sweeps its seed with (by default `scoring`).
"""
struct PairCtx{P,EP,BP,S,S2}
    pair::P
    ep::EP
    bp::BP
    geom::PairGeom
    problem_type::Type
    scoring::S
    solo_scoring::S2
    # Reusable workspace cache (RansacWorkspaces of the native refits, the
    # external refiners' workspaces), keyed by (problem variant, scoring kind, LO type).
    # A fresh workspace is ~1.4 MB of buffers; the refit layer runs thousands
    # of `locally_optimize` calls per pair (refit spec × column), so reuse is the
    # difference between ~20 KB/call and ~1.4 MB/call of GC pressure. Safe:
    # the workspace holds SIZE-typed buffers only (no scoring values baked);
    # every refit re-scores its seed, refreshing all state. One ctx is used by
    # exactly one thread (apply_lo_refits threads over pairs).
    lo_ws::Dict{Symbol,Any}
end

"""
    build_pair_ctx(problem_type, pair::PoolPair; scoring = nothing, solo_scoring = scoring, bundle = false) -> PairCtx

Build the [`PairCtx`](@ref) of one pool pair, once. It holds the family's pool-frame problem
(RobustVisualGeometry's `EssentialProblem(pair)`, `FundMatProblem(pair)` or
`HomographyProblem(pair)`), `Kc`, GT pose and pool-frame points; plus the Schur
bundle problem when `bundle = true`. `problem_type` is the pool's track
([`problem_type`](@ref)`(pool.problem)`). `scoring` is the refit scoring
template (any scoring with an outlier half-width: the caller's — a paper's
Student-t + uniform marginal at its prior); `nothing` is a context that only
reads selections (no native refit).
"""
function build_pair_ctx(problem_type::Type, pair::PoolPair;
                        scoring = nothing, solo_scoring = scoring,
                        bundle::Bool = false)
    ep, Kc1, Kc2 = _pair_problems(problem_type, pair)
    R_gt, t_gt = gt_pose(pair)
    geom = PairGeom(Kc1, Kc2, SMatrix{3,3,Float64,9}(R_gt), SVector{3,Float64}(t_gt),
                    _pair_csponds(pair))
    bp = bundle ? RelposeBundleProblem(Kc1, Kc2, pair.csponds) : nothing
    return PairCtx(pair, ep, bp, geom, problem_type, scoring, solo_scoring, Dict{Symbol,Any}())
end

model_imc(model, ctx::PairCtx) = model_imc(model, ctx.geom)

"""
    essential_bearings(ctx::PairCtx) -> (f₁, f₂)

The unit bearings of an E context's problem (what a cheirality vote reads). Read from `ctx.ep`, which already
back-projected every cspond through the pool camera, rather than re-derived.
Only an E context has bearings: on an F or H context `ctx.ep` holds pool
pixels, so this refuses rather than hand 2-vectors to a ray test.
"""
essential_bearings(ctx::PairCtx) = _essential_bearings(ctx.problem_type, ctx)
_essential_bearings(::Type{<:EssentialProblem}, ctx) = (sources(ctx.ep.cs), targets(ctx.ep.cs))
_essential_bearings(P::Type, _) =
    throw(ArgumentError("the cheirality vote needs an EssentialProblem context; got $P"))

# -----------------------------------------------------------------------------
# Perturbed-seed helpers — THE basin-experiment perturbation (one copy, shared
# by run_basin_ladder.jl and the solo-CV driver so both protocols perturb
# identically; moved here from run_basin_ladder.jl 2026-07-18).
# -----------------------------------------------------------------------------

"""
    perturbed_pose(rng, R_gt, t_gt, δ) -> (RotMatrix, Vec3)

The protocol's δr = δt = δ perturbation: rotation by `δ` [rad] about a seeded
random axis plus a magnitude-`δ` translation jitter (t re-normalized). The
draw order is fixed (axis first, then translation jitter), so a caller-seeded
rng reproduces the seed bit-for-bit (ladder convention: one
`StableRNG(stable_seed(seed, scene, pair, δ))` per (pair, δ), so every arm
sees the identical perturbed seed, on every Julia version). Owned here since VisualGeometryCore 0.18 dropped
`Synthetic.perturb_pose` (this was its only caller); the body is that
function's, verbatim, with δr = δt = δ.
"""
function perturbed_pose(rng::AbstractRNG, R_gt, t_gt, δ::Real)
    ax = normalize(SVector{3, Float64}(randn(rng), randn(rng), randn(rng)))
    Rp = SMatrix{3, 3, Float64}(RotMatrix(AngleAxis(δ, ax...))) * SMatrix{3, 3, Float64}(R_gt)
    tp = normalize(SVector{3, Float64}(t_gt) + δ * SVector{3, Float64}(randn(rng), randn(rng), randn(rng)))
    return RotMatrix(Rp), VisualGeometryCore.Vec3{Float64}(tp)
end

"""
    perturbed_seed_E(rng, R_gt, t_gt, δ) -> EssentialMat

The δ-perturbed metric essential seed `[t]ₓR` of [`perturbed_pose`](@ref) —
the seed model every basin-style E refit starts from.
"""
perturbed_seed_E(rng::AbstractRNG, R_gt, t_gt, δ::Real) = EssentialMat(perturbed_pose(rng, R_gt, t_gt, δ)...)
