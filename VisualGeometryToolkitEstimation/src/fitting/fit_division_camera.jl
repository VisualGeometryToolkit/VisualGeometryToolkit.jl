# =============================================================================
# fit_division_camera — the iterated, gradient-whitened, robustly re-weighted
# division-camera fit (BabelCalib's back-end, made robust)
# =============================================================================
#
# `division_camera_step` is one iteration of the BabelCalib linear
# focal solve: rows from `constraint_with_jacobians(RadialLift, …)`, whitened by
# their frozen Sampson factor (`whitening_factor`), weighted per correspondence. Its gradients need a solution
# to be evaluated at, and its robust weights need the previous residuals — so
# the estimator is an IRLS on the stack's SHARED posterior-weight iteration — a
# `PosteriorIrlsMethod` of `block_coordinate_descent!` configured by a
# `MixtureEmConfig`, its E-step the posterior of the shared Student-t + uniform
# scorer — over the stack as a `DivisionCameraProblem`:
#
#   update_model!     `WeightedRefine`: the problem's `refine!`, one weighted,
#                     whitened linear solve (closed form) at the previous
#                     iteration's solution — the per-board radial cameras are
#                     INPUTS and never re-fitted (see `fit_division_camera`)
#   update_residuals! the iteration's whitened residual norms (pixels) into the
#                     likelihood's buffers (`mahalanobis_distances!`)
#   update_nuisance!/    the likelihood (`PosteriorLikelihood`) — the Student-t
#   update_weights!   inlier + uniform outlier posterior of the scorer: weight =
#                     inlier responsibility × Student-t weight, ŝ² and ν
#                     re-estimated each iteration (the EM's nuisance policies),
#                     seeded at the median residual
#   stopping_criterion `ParameterChangeStop`: (f, k) has stopped moving; the
#                     LAST iterate is the result
#
# The two weightings ride the same channel: whitening (currency) from the
# previous iterate's gradients, robust weights from the previous iterate's
# residuals — both carried into the next solve. The first iteration has no iterate
# to evaluate at: it is whitened at a SEED camera (focal `focal_seed`, k₀ = 0 or
# the given k — weights only) and robustly unweighted; its solution seeds everything.
#
# The stack is conditioned by ONE normalization, built once from the views —
# `CspondNormalization(DivisionCameraState, views; center)`: Hartley at the
# plane points' centroid, Hartley ABOUT the principal point every radial camera
# shares on the image side — which every iteration applies inside, composing
# its solution back into pixels. The IRLS, its posterior, the seed and the scorer's
# `outlier_halfwidth` all live in pixels.
# =============================================================================

"""
    DivisionCameraProblem

The stacked division-camera solve as a problem of the shared posterior-weight
iteration ([`PosteriorIrlsMethod`](@ref) under [`WeightedRefine`](@ref)): the
views' radial cameras and board points, their prior row weights, the
coefficient mode and the stack's one normalization, built by
[`fit_division_camera`](@ref). Its distances (`mahalanobis_distances!`) are the
whitened residual norms of the correspondences (pixels, codimension 2) and its
`refine!` is one weighted, whitened `division_camera_step`. As a TYPE it is what the
fit's `weighting` (its Student-t + uniform E-step) is built for: `scoring_from_toml(table; problem =
DivisionCameraProblem)` or `StudentTUniformScoring(DivisionCameraProblem, h;
scale_prior, label_prior)` (codimension 2).
"""
struct DivisionCameraProblem <: AbstractConstraintProblem
    Hs::Vector
    boards::Vector
    prior::Union{Nothing, Vector{Vector{Float64}}}   # the views' `weights` columns (row priors), in `ids` order
    ncoef::Int
    k::Union{Nothing, Float64}                        # pixels
    normalization::CspondNormalization                # the stack's one normalization (every iteration applies it)
    n::Int                                            # correspondences over every view
    verbose::Bool
end

# A whitened 2-vector per correspondence. The scorer's model DOF `m` (ν = (k − m)·d_g)
# is read only by a partition sweep, which this fit does not run (its posterior is
# median-seeded, see `fit_division_camera`): the type states the smallest stack's, f,
# one coefficient and one view's t₃ over the codimension, rounded up.
codimension(::Type{<:DivisionCameraProblem}) = 2
minimal_sample_size(::Type{<:DivisionCameraProblem}) = 2
minimal_sample_size(p::DivisionCameraProblem) = minimal_sample_size(typeof(p))
data_size(p::DivisionCameraProblem) = p.n
# What the parameter-change stop compares: the focal length and the division
# coefficients (pixels); the poses follow them.
change_parameters(::DivisionCameraProblem, θ::_PositionalDivisionState) = vcat(θ.f, θ.k)

# The whitened residual norms of a state's views, flattened (pixels). A view
# without a pose contributes no rows to the solve; its entries take a sentinel
# (1e6 px, squared 1e12 exactly) so the posterior down-weights them (they never
# enter the solve).
function mahalanobis_distances!(r::AbstractVector, ::DivisionCameraProblem, state::_PositionalDivisionState)
    i = 0
    for v in state.sq_distances, s in v
        i += 1
        r[i] = isfinite(s) ? sqrt(s) : 1.0e6
    end
    return r
end

# Per-board split of the posterior weights, normalized to max 1: the posterior's
# `w` carries the precision 1/ŝ² (a uniform factor the weighted LS is
# invariant to), which would make the reported weights unreadable.
function _board_weights(w::AbstractVector, boards)
    wmax = maximum(w)
    inv_wmax = wmax > 0 ? 1 / wmax : 1.0       # a NORMALIZER, not a scale
    out = Vector{Vector{Float64}}(undef, length(boards))
    i = 0
    for (b, pts) in enumerate(boards)
        out[b] = [inv_wmax * w[i += 1] for _ in pts]
    end
    return out
end

# A view's row weight on an iteration: its prior (the `weights` column) times the posterior.
_combined_weights(prior::Nothing, post) = post
_combined_weights(prior, post) = [p .* q for (p, q) in zip(prior, post)]

# The model block of the posterior-weight iteration (`WeightedRefine`): the next
# solution — the stacked solve whitened at `θ` and weighted by the prior times the
# posterior `w` (`audit`: with the branch audit). `nothing` when it has no system
# to solve (a `DivisionCameraError` from division_camera_step: no admissible pose,
# no depth information, a singular or non-positive solve) — the loop's
# `:solve_failed`, which ends the fit on its incumbent; an argument error is a bug
# and propagates.
function refine!(::Nothing, p::DivisionCameraProblem, θ::_PositionalDivisionState, w::AbstractVector; audit::Bool = false)
    pw = _combined_weights(p.prior, _board_weights(w, p.boards))
    return try
        _division_camera_step(
            p.normalization, p.Hs, p.boards; ncoef = p.ncoef, k = p.k,
            point_weights = pw, whiten_at = θ, audit
        )
    catch e
        e isa DivisionCameraError || rethrow()
        p.verbose && println(audit ? "  audit aborted: " : "  iteration aborted: ", e.msg)
        nothing
    end
end

# -----------------------------------------------------------------------------
# The branch audit (see `_audit_branch` in division_camera_step.jl)
# -----------------------------------------------------------------------------

# One iteration at the solution `θ` whose views' branches are re-judged there
# under the posterior weights `w`. `nothing` when it moves no view — the audit
# iteration is then discarded and the fit is unchanged, bit for bit — or when it
# has no system to solve.
function _branch_audit(p::DivisionCameraProblem, θ::_PositionalDivisionState, w::AbstractVector)
    a = refine!(nothing, p, θ, w; audit = true)
    return a !== nothing && _moved_branches(a, θ) > 0 ? a : nothing
end

# How many views hold a pose in both iterations, in different branches.
function _moved_branches(a::_PositionalDivisionState, b::_PositionalDivisionState)
    n = 0
    for (pa, pb) in zip(a.poses, b.poses)
        (pa === nothing || pb === nothing) && continue
        rotation(pa) != rotation(pb) && (n += 1)
    end
    return n
end

# The verbose trace of one run of the loop, from its `Recorded` state: each
# iterate's f and k, the posterior's ŝ, and the iteration's cost Σ wᵢrᵢ².
function _print_iterations(rec::VgtkCore.Recorded, first_iter::Int)
    for (j, p) in enumerate(rec.iterates)
        println("  iteration $(first_iter + j - 1): f=$(round(p.f; digits = 3)) px  k=$(round.(p.k; sigdigits = 5))  " *
                "ŝ=$(round(sqrt(rec.s2[j]); digits = 4)) px  cost=$(round(rec.cost[j]; sigdigits = 8))")
    end
    return nothing
end
_print_iterations(_state, _first_iter::Int) = nothing

# The posterior-weight iteration's seed of the stack: the posterior of the solution
# `θ` (its distances into the state's buffers, the nuisance and the E-step
# refreshed there) at the cost Σ wᵢrᵢ² of its weights — the incumbent a resumed run
# starts from.
function _division_irls_state(rws::PosteriorWeightState, p::DivisionCameraProblem, θ::_PositionalDivisionState,
                              r::Vector{Float64}, lh; refresh::Bool)
    n = data_size(p)
    mahalanobis_distances!(r, p, θ)
    square_distances!(sq_distances(rws), r, n)
    refresh && (update_nuisance!(lh); update_weights!(lh))
    return PosteriorIrlsState(rws, r, copy(r), θ, weighted_rss(robust_weight(rws), sq_distances(rws), n))
end

# The fit's two configured parts, checked: the E-step's weighting (a Student-t +
# uniform scorer at codimension 2 with a supplied outlier half-width) and the
# loop's EM, whose model update must refit the stack.
function _check_division_config(weighting::StudentTUniformScoring, mixture_em::MixtureEmConfig)
    codimension(weighting) == 2 || throw(ArgumentError(
        "fit_division_camera: the residuals are whitened 2-vectors, so the weighting's codimension must be 2, got " *
            "$(codimension(weighting)) — build it for the stack: `scoring_from_toml(table; problem = DivisionCameraProblem)`"))
    required_outlier_halfwidth(weighting)
    mixture_em.model_update isa WeightedRefine || throw(ArgumentError(
        "fit_division_camera: the loop refits the stack each iteration, so `mixture_em.model_update` must be " *
            "WeightedRefine (\"weighted_refine\"), got $(mixture_em.model_update)"))
    mixture_em.stop isa ParameterChangeStop || throw(ArgumentError(
        "fit_division_camera: each iteration re-whitens the stack at the previous solution, so no cost is comparable " *
            "between iterations and the loop must stop on the change of its parameters: `mixture_em.stop` must be " *
            "ParameterChangeStop (`{ type = \"parameter_change\", rtol = … }`), got $(mixture_em.stop)"))
    return nothing
end

# The defaults of record: configs/default.toml [fit_division_camera] `weighting` and
# `mixture_em`, absent keys inheriting [scoring.student_t_uniform] and [mixture_em].
function _division_weighting_default()
    sc = scoring_from_toml(_dflt("fit_division_camera", "weighting"); problem = DivisionCameraProblem,
                           section = "fit_division_camera.weighting")
    sc isa StudentTUniformScoring || throw(ArgumentError(
        "[fit_division_camera.weighting] must be a Student-t + uniform scorer (type = \"student_t_uniform\"), got $(typeof(sc))"))
    return sc
end
_division_mixture_em_default() = mixture_em_from_toml(_dflt("fit_division_camera", "mixture_em");
                                                      section = "fit_division_camera.mixture_em")

# The removed keywords, refused with the way to say them.
_refuse_division_keyword(::Nothing, ::Symbol) = nothing
function _refuse_division_keyword(_v, key::Symbol)
    key === :outlier_halfwidth && throw(ArgumentError(
        "fit_division_camera: `outlier_halfwidth` is no keyword since 0.9 — the uniform outlier half-width is the " *
            "weighting's: give `weighting = default_student_t_uniform_scoring(DivisionCameraProblem, h)` (configs/default.toml " *
            "[fit_division_camera] `weighting`)"))
    throw(ArgumentError(
        "fit_division_camera: `max_iter` is no keyword since 0.9 — the iteration cap is the EM's: give " *
            "`mixture_em = mixture_em_from_toml(Dict(\"model_update\" => \"weighted_refine\", \"max_iter\" => n))` " *
            "(configs/default.toml [fit_division_camera] `mixture_em`)"))
end

"""
    DivisionCameraMetadata

`FitResult` metadata of [`fit_division_camera`](@ref), keyed by the views'
identity: the converged solve `state` (a `DivisionCameraState{K}` — `f`, the
BabelCalib coefficients, the per-view poses), the `radial_cameras` the fit
consumed, the `views` carrying each correspondence's final IRLS row weight as
their `weights` column (the posterior's weight normalized to max 1, times the
input column when the view had one — the weight the solve ran with), and the
robust-estimation contract's record of the final E-step:
- `weights`, every correspondence's posterior inlier probability, flat in the
  views' key order (`sort(collect(keys(views)))`, each view's in its order) —
  [`PosteriorWeights`](@extref VisualGeometryToolkitCore.PosteriorWeights), so
  `inlier_mask(result)` is the MAP rule (posterior > ½); a row the fit did not
  take (prior weight zero, or a view the final iteration left without a pose),
  and a posterior below `eps()`, is an exact 0; read by `weights(result)`;
- `scale` = ŝ (pixels, a LINEAR scale) with `scale_source` `ScaleEstimated`,
  read by `noise_scale(result)`; and the Student-t `ν`.

The core fields carry the camera (`estimate`), `retcode`, `iterations`, the
support `n` (the rows of positive weight) and the whitened-residual `rms`
weighted by those weights, `√(Σwᵢrᵢ² / Σwᵢ)` (pixels).
"""
struct DivisionCameraMetadata{S, C, V}
    state::S
    radial_cameras::C
    views::V
    weights::Vector{Float64}
    scale::Float64
    scale_source::ScaleSource
    ν::Float64
end
weight_kind(::Type{<:DivisionCameraMetadata}) = PosteriorWeights()

# A view with its weights as the `weights` column (its other columns kept).
_with_weights(cs::VgtkCore.Csponds, w) = withattrs(cs; weights = w)
_with_weights(cs, w) = VgtkCore.Csponds(sources(cs), targets(cs); weights = w)

"""
    fit_division_camera(radial_cameras::AbstractDict{K}, views::AbstractDict{K};
                        image_size = nothing, focal_seed = nothing, ncoef, k = nothing,
                        weighting, mixture_em,   # defaults: configs/default.toml [fit_division_camera]
                        verbose = false)
        -> FitResult{CameraModel, DivisionCameraMetadata}

The BabelCalib linear focal solve as an iterated, gradient-whitened, robustly
re-weighted estimator (see the file header). Keyed by the view's identity `K`
(the key sets must be equal — `ArgumentError`): `radial_cameras[id]` is view
`id`'s radial camera (a `RadialCamera`, or a homography whose rows 1–2 are read)
and `views[id]` its `plane_point => raw_pixel` correspondences — per-board
VALIDATED inliers (the cameras carry the pencil's shared centre) — the inputs of
`division_camera_step`. A view's `weights` column, if it has one, is a
prior row weight: iteration 0 is weighted by it and each later iteration by it times the
posterior. `k` fixes the division coefficient (centred pixels); otherwise
`ncoef` are estimated. Each iteration whitens every row at the previous solution
(the row's Sampson factor held fixed) and weights it by the Student-t/uniform posterior of the
previous iteration's whitened residuals.

THE ROBUST MACHINERY IS THE STACK'S SHARED ONE (configs/default.toml
[fit_division_camera], one schema):

- `weighting` — the E-step: the posterior of a `StudentTUniformScoring` at
  codimension 2 with a supplied outlier half-width, its scale prior the
  posterior's (default the table's `weighting`, read by
  [`scoring_from_toml`](@ref) with `problem = DivisionCameraProblem`; absent keys
  inherit [scoring.student_t_uniform]). Another codimension, or one without
  a half-width, is an `ArgumentError`. A fit takes no `scoring` (the
  robust-estimation contract): the former `scoring` keyword is an
  `ArgumentError` naming `weighting`.
- `mixture_em` — the loop: a [`MixtureEmConfig`](@ref) driving the
  posterior-weight iteration ([`PosteriorIrlsMethod`](@ref)) — its model update
  must be [`WeightedRefine`](@ref) (the stack is refitted each iteration), its
  stop [`ParameterChangeStop`](@ref) (each iteration re-whitens at the previous
  solution, so no cost is comparable between iterations: the fit stops when
  (f, k) moves by less than `rtol` relative and returns the LAST iterate; the
  incumbent rule stopped the contaminated stacks at their second iteration, f 2 %
  off), its nuisance policies the posterior's and its `max_iter` the budget
  (default the table's `mixture_em`, read by [`mixture_em_from_toml`](@ref);
  absent keys inherit [mixture_em]). Anything else is an `ArgumentError`.

`outlier_halfwidth` and `max_iter` are no keywords since 0.9 (an
`ArgumentError` naming the form above): the half-width is the weighting's, the cap
the EM's.

INPUT CONTRACT: each view is that board's VALIDATED correspondences — the
inliers its radial camera was found with — not raw detections. The
Student-t/uniform posterior absorbs what validation leaves (a few per cent of
ray-aligned mismatches, which a radial camera's 1-D line test cannot reject);
it is not a substitute for it.

THE BRANCH AUDIT. Each view's pose branch — the tilt ambiguity of its radial
camera — is chosen on the iteration that first places the view, from that view's own
solve, and kept. That solve cannot separate the tilt pair by residual, so one
mismatch pushed along its ray through the centre can tip it (on the MC scene at
3 % ray-aligned mismatches, 16 % of the views carrying one took a branch 48–85°
off; none of the clean views), and the view's inliers then carry residuals far
above `sigma` — after the posterior has already rejected the mismatch. So once
the IRLS converges — or, if it is still moving at half the `mixture_em.max_iter` budget,
there (a wrong branch is what keeps a stack from converging) — one iteration
re-judges every view's branch at the shared (f, k), robustly; if it moves any,
the IRLS resumes from there within what is left of `max_iter` (the audit counts
as an iteration, so `iterations` never exceeds `max_iter`); if it moves none the
IRLS continues unchanged, bit for bit (clean stacks). Never after
`:solve_failed`, nor at `max_iter = 1`. Measured: 104 → 4
wrong views at 3 %, 66 → 1 at 5 %, no correct view moved, and the held-out
residuals calibrated against ŝ²·I₂ in the tail as well as the bulk
(`test/ransac_mc_calibration_tests.jl` (e)).

Inputs, `k`, the scorer's `outlier_halfwidth`, `focal_seed` and every output are in pixels;
the stack is conditioned inside by one normalization
(`CspondNormalization(DivisionCameraState, views; center)` — Hartley about the
radial cameras' shared centre), built once. `image_size = (width, height)`
(pixels) of the image the views were measured in is REQUIRED for the default
seed only (`ArgumentError` when both it and `focal_seed` are missing): its
half-diagonal `s = hypot(width, height)/2`, a 90° diagonal field of view. With
`focal_seed` given, `image_size` may be omitted and does not enter the result.

Iteration 0 has no previous solution: it is whitened at a SEED camera — focal
`focal_seed` (pixels; by default `s`), k₀ = 0 (the given `k` when `k` fixes it —
the known-k rows are exact only there), each board at the depth that camera implies —
which sets the row WEIGHTS only; `f` and `k` are solved by the iteration. The plain
algebraic rows it replaces weigh each board by its depth (a far board's rows ~10×
a near one's), so a few mismatches on a heavy board dragged the joint solve to
f ≈ 45–300 px against 1226 and half those stacks never recovered; the seed
fixes that and is robust to a seed 0.5×–3× off (measured 2026-09-22; see
`test/ransac_mc_calibration_tests.jl` (e)). Supply a better focal as data — a
previous calibration, EXIF — through `focal_seed`.

The scorer's `outlier_halfwidth` (pixels) is the half-width of the uniform outlier component
— the same quantity the RANSAC problems take; it must NOT be the residual scale
(a half-width at the median residual makes the outlier component as dense as the
inliers and zeroes every weight — measured). The posterior is seeded from the
median residual ([`median_seeded_posterior`](@extref VisualGeometryToolkitCore.median_seeded_posterior)): every point starts an
inlier and the M-step shrinks ŝ as the model improves.

That seed and schedule are load-bearing, not incidental. Five alternatives were
measured against RVG's synthetic (known f₀, 0.4 px noise, 8 % gross outliers) on
2026-09-10 and every one was worse: a frozen MAD anchor 6.5 % off f₀, a global
Jeffreys sweep 3.5 %, a per-board sweep 5.1 %, frozen Welsch 28 %, Welsch with an
adaptive scale 85 % (no uniform component ⇒ the scale update has no fixed point
and ran to ŝ = 29). This estimator lands at 1.25 %. A sweep seed needs residuals
that are already bimodal; iteration 0 here is one linear solve of an uncalibrated
stack (whitened at a seed, not at a fit), so they are not. (Those measurements
predate the seeded iteration 0.)

Iteration 0 alone — whitened at the seed and robustly unweighted — is the step

    N = CspondNormalization(DivisionCameraState, views; center)   # center: the cameras' shared centre
    CameraModel(division_camera_step(N, radial_cameras, views; ncoef, k, whiten_at = focal_seed))

(with `focal_seed` the half-diagonal by default); this function always runs at
least one robust iteration (`mixture_em.max_iter ≥ 1`), and `max_iter = 1` is
that step re-weighted once, with no branch audit. A stack whose
iteration 0 cannot solve at all (no board admits a pose, none carries depth, the solve is
singular) raises the `DivisionCameraError` — there is no iterate to return;
a later iteration that cannot solve ends the fit on the last good one
(`:solve_failed`).

The radial cameras are FIXED. Each view's radial camera is an input, estimated once by the
caller — BlobBoards fits every one linearly about the centre, iterated to its
consensus and polished by a local optimization on the exact point–radial-line
distance — and this stage only turns them into an
initial focal length, coefficients and depths for a bundle. It used to re-fit
them per iteration under the point weights, which could not improve a nonlinear fit
and did wreck one: the perturbation tipped the metric upgrade's cheirality
test onto the mirrored branch for a board the shared solve had placed behind
the camera, turning its pose 179.95° and its whitened residual from 1.7 px to
1102 px, and through the Student-t width that single row collapsed every
weight in the solve — 18 of 106 boards lost their poses, boards whose own
poses under the converged bundle camera were the BEST in the set at 0.14 px
(measured 2026-09-09, fisheye hallway). Hand this function cameras you are
willing to stand behind; it will not launder contaminated ones.

Returns the camera `CameraModel(::DivisionCameraState)` builds — `K = diag(f, f, 1)` at the
cameras' shared centre, `Division{ncoef}` lens in K's chart — plus the
final keyed `DivisionCameraState` (pixels), the `views` with each correspondence's
IRLS row weight as their `weights` column, and the loop outcome — and the
robust-estimation contract's record of the E-step ([`DivisionCameraMetadata`](@ref)):
`weights(result)`, each correspondence's posterior inlier probability
(`PosteriorWeights`: `inlier_mask(result)` is posterior > ½), and
`noise_scale(result)`, the E-step's Student-t scale ŝ (pixels, a LINEAR scale;
`scale_source(result)` is `ScaleEstimated`), with its `ν`. `n` is the support
(the rows of positive weight) and `rms` the posterior-weighted whitened-residual
RMS, `√(Σwᵢrᵢ² / Σwᵢ)` (pixels): a row with prior weight zero, or of a view the
final iteration left without a pose, weighs 0 and is not a residual.

Units: the board points are plain numbers. The radial cameras and the board
poses carry the board plane's unit, which neither a matrix nor this result
records, so physical (unitful) board points are an `ArgumentError`: strip
them in the unit the radial cameras were fit in (the stack's own
boundaries strip in metres, `ustrip.(m, X)`).
"""
function fit_division_camera(
        radial_cameras::AbstractDict{K, <:Union{HomographyMat, RadialCamera}},
        views::AbstractDict{K, <:AbstractVector{<:Cspond}};
        image_size = nothing,
        focal_seed::Union{Nothing, Real} = nothing,
        ncoef::Int = _dflt("fit_division_camera", "ncoef"), k::Union{Nothing, Real} = nothing,
        weighting::StudentTUniformScoring = _division_weighting_default(),
        mixture_em::MixtureEmConfig = _division_mixture_em_default(),
        verbose::Bool = false,
        scoring = RetiredKeyword(),   # refused, with the way to say it (the fit's `weighting`)
        outlier_halfwidth = nothing,   # refused, with the way to say it (the weighting's)
        max_iter = nothing,            # refused, with the way to say it (the EM's)
    ) where {K}
    scoring isa RetiredKeyword || throw(_scoring_is_weighting("fit_division_camera"))
    _refuse_division_keyword(outlier_halfwidth, :outlier_halfwidth)
    _refuse_division_keyword(max_iter, :max_iter)
    _check_division_config(weighting, mixture_em)
    focal_seed === nothing || (focal_seed > 0 && isfinite(focal_seed)) ||
        throw(ArgumentError("fit_division_camera: focal_seed must be positive and finite (pixels), got $focal_seed"))
    focal_seed === nothing && image_size === nothing &&
        throw(ArgumentError("fit_division_camera: give `image_size` (the default seed is its half-diagonal) or `focal_seed`"))
    ids, Hs, boards = _keyed_views(radial_cameras, views)
    c = _shared_center(Hs)                                    # checked, exact
    prior = _point_weights(boards)
    # THE NORMALIZATION, once — `CspondNormalization(DivisionCameraState, views;
    # center)`; every iteration applies it inside and returns pixels.
    N = _division_normalization(boards, c, prior)
    kk = k === nothing ? nothing : Float64(k)
    fseed = Float64(focal_seed === nothing ? hypot(image_size[1], image_size[2]) / 2 : focal_seed)   # pixels; default: a 90° diagonal FOV
    problem = DivisionCameraProblem(Hs, boards, prior, ncoef, kk, N, sum(length, boards), verbose)
    st, lh, method, state0 = _division_irls(problem, weighting, mixture_em, fseed)
    max_iter = mixture_em.max_iter
    verbose && println("  iteration 0: f=$(round(state0.f; digits = 3)) px  k=$(round.(state0.k; sigdigits = 5))  " *
                       "ŝ=$(round(sqrt(s_hat_sq(st.rws)); digits = 4)) px")
    # THE BRANCH AUDIT (`_audit_branch`), once, inside the one `max_iter` budget
    # (the audit is an iteration): at convergence, or — for an IRLS still moving
    # at the CHECKPOINT, half the budget — there. A wrong branch is what keeps a
    # stack from converging: measured on the MC scene at 3 % ray-aligned, 86 of
    # 200 stacks ran out of iterations and carried 70 of the 75 wrong views left
    # when the audit was held to converged stacks (whose tail then failed its
    # calibration gate). When the audit moves a view the IRLS resumes from the
    # audited iteration, its posterior refreshed there, for what remains of the
    # budget; when it moves none, the IRLS simply continues from its incumbent.
    # Never after `:solve_failed`.
    #
    # The termination record (§14): each leg's, the legs combined by `worst_retcode`
    # (the later on a tie). A leg that stopped at the CHECKPOINT did not terminate —
    # the budget was not spent — so only a leg that ended on its own enters with the
    # resumed one.
    checkpoint = max_iter - max_iter ÷ 2
    res = _division_run!(st, lh, method, problem, state0, checkpoint, 1)
    niter, rc = iterations(res), retcode(res)
    record = (rc, stop_rule(res))
    θ = estimate(res)
    # a failed leg (a failed solve, a non-finite start: what `retcode_symbol` names `:solve_failed`) is not audited
    audited = retcode_symbol(record...) !== :solve_failed && niter < max_iter ? _branch_audit(problem, θ, robust_weight(st.rws)) : nothing
    if audited !== nothing
        verbose && println("  branch audit: $(_moved_branches(audited, θ)) view(s) moved; resuming")
        st = _division_irls_state(st.rws, problem, audited, st.residuals, lh; refresh = true)
        res = _division_run!(st, lh, method, problem, audited, max_iter - niter - 1, niter + 2)
        resumed = (retcode(res), stop_rule(res))
        record = rc == ReturnCode.MaxIters ? resumed : worst_retcode(record, resumed)
        niter = niter + 1 + iterations(res)
    elseif rc == ReturnCode.MaxIters && niter < max_iter
        res = _division_run!(st, lh, method, problem, θ, max_iter - niter, niter + 1)
        niter, record = niter + iterations(res), (retcode(res), stop_rule(res))
    end
    best = estimate(res)
    state = _keyed(best, ids)
    # ONE construction: `CameraModel(::DivisionCameraState)`. The state carries
    # the pencil's centre — the principal point every radial camera shares,
    # checked above; views are raw pixels and nothing here re-centres them.
    camera = VgtkCore.CameraModel(state)
    wpost = _combined_weights(prior, _board_weights(robust_weight(st.rws), boards))
    # The E-step's posterior inlier probabilities, flat in the views' order, over
    # the rows that entered the fit: a view without a pose has NaN residuals (the
    # posterior's buffer holds a 1e12 sentinel for them, which is not a residual),
    # and a row of prior weight zero was never fitted — both weigh exactly 0.
    post = inlier_weight(st.rws)
    w = Vector{Float64}(undef, problem.n)
    sq = Vector{Float64}(undef, problem.n)
    l = 0
    for (i, sqi) in enumerate(best.sq_distances), (j, r²) in enumerate(sqi)
        l += 1
        took = isfinite(r²) && (prior === nothing || prior[i][j] > 0)
        w[l] = took ? Float64(post[l]) : 0.0
        sq[l] = took ? r² : 0.0
    end
    support, wrms = _posterior_support_rms!(w, sq)        # a posterior below eps an exact 0; its support and weighted RMS
    return FitResult(
        camera, record..., niter,
        wrms, support,               # the posterior-weighted whitened-residual RMS (px), its support
        DivisionCameraMetadata(
            state, radial_cameras, Dict(ids[i] => _with_weights(views[ids[i]], wpost[i]) for i in eachindex(ids)),
            w, sqrt(s_hat_sq(st.rws)), ScaleEstimated, Float64(student_t_dof(st.rws))
        ),
    )
end

# The median-seeded posterior's start ν: configs/default.toml [fit_division_camera] seed_nu.
const _DIVISION_SEED_NU = Float64(_dflt("fit_division_camera", "seed_nu"))

# The stack's posterior-weight iteration, set up: iteration 0 — the step whitened
# at the seed focal `fseed` (weights only), robustly unweighted — its posterior
# seeded at the median residual under the scorer's outlier half-width and scale
# prior, the nuisance updated under the EM's policies, and the shared method and
# state (`PosteriorIrlsMethod` under `WeightedRefine`, `PosteriorIrlsState`).
# Returns `(state, likelihood, method, iteration 0)`.
function _division_irls(problem::DivisionCameraProblem, weighting::StudentTUniformScoring, mixture_em::MixtureEmConfig,
                        fseed::Float64)
    state0 = _division_camera_step(problem.normalization, problem.Hs, problem.boards; ncoef = problem.ncoef,
                                   k = problem.k, point_weights = problem.prior, whiten_at = fseed)
    n = data_size(problem)
    r = mahalanobis_distances!(Vector{Float64}(undef, n), problem, state0)
    rws = median_seeded_posterior(square_distances!(Vector{Float64}(undef, n), r, n), n;
                                  outlier_halfwidth = required_outlier_halfwidth(weighting), ν = _DIVISION_SEED_NU,
                                  codimension = codimension(weighting), prior = scale_prior(weighting))
    nu = mixture_em.nuisance
    lh = PosteriorLikelihood(rws, n; scale_update = nu.scale, prior_update = nu.inlier_prior, nu_update = nu.nu)
    st = PosteriorIrlsState(rws, r, copy(r), state0, weighted_rss(robust_weight(rws), sq_distances(rws), n))
    return st, lh, PosteriorIrlsMethod(nothing, mixture_em), state0
end

# One run of the loop from `θ` for at most `max_iter` iterations (none at 0), its
# trace printed when the problem is verbose (`Recorded`: the run is the bare run,
# iterate for iterate); the iterations are numbered from `first_iter`.

function _division_run!(st, lh, method, problem::DivisionCameraProblem, θ, max_iter::Int, first_iter::Int)
    run = problem.verbose ? VgtkCore.Recorded(st) : st
    res = block_coordinate_descent!(run, lh, method, problem, θ; max_iter)
    _print_iterations(run, first_iter)
    problem.verbose && println("  -> $(retcode_symbol(res)) after $(iterations(res)) iteration(s)")
    return res
end
