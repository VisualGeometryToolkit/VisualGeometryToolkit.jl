# =============================================================================
# fit_division_camera — the iterated, gradient-whitened, robustly re-weighted
# division-camera fit (BabelCalib's back-end, made robust)
# =============================================================================
#
# `division_camera_step` is one iteration of the BabelCalib linear
# focal solve: rows from `constraint_with_jacobians(RadialLift, …)` under a
# `WhiteningPolicy`, weighted per correspondence. Its gradients need a solution
# to be evaluated at, and its robust weights need the previous residuals — so
# the estimator is an IRLS instance of `block_coordinate_descent!`, exactly like every
# local optimization here:
#
#   update_model!     one weighted, whitened linear solve (closed form) at the
#                     previous iteration's solution — the per-board radial cameras
#                     are INPUTS and never re-fitted (see `fit_division_camera`)
#   update_residuals! the iteration's whitened residual norms² (pixel²) into the
#                     likelihood's buffers
#   update_nuisance!/    the likelihood — the Student-t inlier + uniform outlier
#   update_weights!   posterior (`PosteriorWeightState`): weight = inlier
#                     responsibility × Student-t weight, ŝ² and ν re-estimated
#                     each iteration, partition seeded by the scoring sweep
#   stopping_criterion the shared solution has stopped moving
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
# its solution back into pixels. The IRLS, its posterior, the seed and `outlier_halfwidth` all live in
# pixels.
# =============================================================================

struct DivisionCameraIrls{W <: VgtkCore.WhiteningPolicy} <: VgtkCore.AbstractBlockCoordinateMethod
    Hs::Vector
    boards::Vector
    prior::Union{Nothing, Vector{Vector{Float64}}}   # the views' `weights` columns (row priors), in `ids` order
    ncoef::Int
    k::Union{Nothing, Float64}                        # pixels
    whitening::W
    normalization::CspondNormalization                # the stack's one normalization (every iteration applies it)
    verbose::Bool
end

mutable struct DivisionCameraIrlsState
    state::_PositionalDivisionState
    rws::PosteriorWeightState
    n::Int
    Δ::Float64
    iterations::Int
end

# Flatten the per-board whitened residuals into the likelihood's buffer.
# Boards without a pose contribute no rows to the solve; their entries take a
# sentinel so the posterior down-weights them (they never enter the solve).
function _focal_sq!(sq, state::_PositionalDivisionState)
    i = 0
    for v in state.sq_residuals, s in v
        i += 1
        sq[i] = isfinite(s) ? s : 1.0e12
    end
    return sq
end

# Per-board split of the posterior weights, normalized to max 1: RSDR's
# `w` carries the precision 1/ŝ² (a uniform factor the weighted LS is
# invariant to), which would make the reported weights unreadable.
function _focal_point_weights(st::DivisionCameraIrlsState, boards)
    wmax = maximum(robust_weight(st.rws))
    inv_wmax = wmax > 0 ? 1 / wmax : 1.0       # a NORMALIZER, not a scale
    out = Vector{Vector{Float64}}(undef, length(boards))
    i = 0
    for (b, pts) in enumerate(boards)
        out[b] = [inv_wmax * robust_weight(st.rws)[i += 1] for _ in pts]
    end
    return out
end

# A view's row weight on an iteration: its prior (the `weights` column) times the posterior.
_combined_weights(prior::Nothing, post) = post
_combined_weights(prior, post) = [p .* q for (p, q) in zip(prior, post)]

# The next iteration's solution — the stacked solve whitened at `st.state` and
# weighted by the prior times the current posterior (`audit`: with the branch
# audit). `nothing` when it has no system to solve (a `DivisionCameraError` from
# division_camera_step: no admissible pose, no depth information, a singular or
# non-positive solve); an argument error is a bug and propagates.
function _next_model(st::DivisionCameraIrlsState, m::DivisionCameraIrls; audit::Bool = false)
    pw = _combined_weights(m.prior, _focal_point_weights(st, m.boards))
    return try
        _division_camera_step(
            m.normalization, m.Hs, m.boards; ncoef = m.ncoef, k = m.k,
            point_weights = pw, whiten_at = st.state, whitening = m.whitening, audit
        )
    catch e
        e isa DivisionCameraError || rethrow()
        m.verbose && println(audit ? "  audit aborted: " : "  iteration aborted: ", e.msg)
        nothing
    end
end

function VgtkCore.update_model!(st::DivisionCameraIrlsState, m::DivisionCameraIrls, prob, θ)
    # No system to solve ends the IRLS on the LAST GOOD iteration — `nothing` is
    # the framework's :solve_failed, and `block_coordinate_result` returns `st.state`.
    state = _next_model(st, m)
    state === nothing && return nothing
    st.Δ = abs(state.f - st.state.f) / max(st.state.f, eps())
    st.state = state
    return state
end

function VgtkCore.update_residuals!(st::DivisionCameraIrlsState, m::DivisionCameraIrls, prob, θ)
    _focal_sq!(sq_residuals(st.rws), st.state)
    return nothing
end

function VgtkCore.post_iteration!(st::DivisionCameraIrlsState, m::DivisionCameraIrls, prob, θ, iter)
    st.iterations = iter
    m.verbose && _print_iteration(iter, st.state, st.rws, st.Δ)
    return nothing
end

# The shared solution has stopped moving: the focal's relative change `Δ` of
# the last iteration (set by `update_model!`) is below `tol`.
struct StopWhenFocalChangeLess <: VgtkCore.StoppingCriterion
    tol::Float64
end
VgtkCore.stop_reason(c::StopWhenFocalChangeLess, st::DivisionCameraIrlsState, m, prob, θ, θ_old, iter) =
    st.Δ < c.tol ? :converged : :continue
VgtkCore.stopping_criterion(::DivisionCameraIrls) = StopWhenFocalChangeLess(1.0e-9)

VgtkCore.block_coordinate_result(st::DivisionCameraIrlsState, m::DivisionCameraIrls, prob, θ, stop_reason, final_iter) =
    lo_estimate(st.state, stop_reason; iterations = final_iter)

# -----------------------------------------------------------------------------
# The branch audit (see `_audit_branch` in division_camera_step.jl)
# -----------------------------------------------------------------------------

# One iteration at the converged solution whose views' branches are re-judged there
# under the final weights. `nothing` when it moves no view — the audit iteration is
# then discarded and the fit is unchanged, bit for bit — or when it has no
# system to solve.
function _branch_audit(st::DivisionCameraIrlsState, m::DivisionCameraIrls)
    a = _next_model(st, m; audit = true)
    return a !== nothing && _moved_branches(a, st.state) > 0 ? a : nothing
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

_print_iteration(iter, p::_PositionalDivisionState, rws, Δ) = println(
    "  iteration $iter: f=$(round(p.f; digits = 3)) px  k=$(round.(p.k; sigdigits = 5))  " *
        "ŝ=$(round(sqrt(s_hat_sq(rws)); digits = 4)) px  ν=$(round(student_t_dof(rws); digits = 1))" *
        (isfinite(Δ) ? "  Δf/f=$(round(Δ; sigdigits = 3))" : "")
)

"""
    DivisionCameraMetadata

`FitResult` metadata of [`fit_division_camera`](@ref), keyed by the views'
identity: the converged solve `state` (a `DivisionCameraState{K}` — `f`, the
BabelCalib coefficients, the per-view poses), the `radial_cameras` the fit
consumed, the `views` carrying each correspondence's final posterior weight as
their `weights` column (normalized to max 1; times the input column when the
view had one), and the Student-t `sigma`/`ν` of the final iteration. The core fields
carry the camera (`estimate`), `retcode`, `iterations`, and the
whitened-residual `rms` over `n` rows.
"""
struct DivisionCameraMetadata{S, C, V}
    state::S
    radial_cameras::C
    views::V
    sigma::Float64
    ν::Float64
end

# A view with its weights as the `weights` column (its other columns kept).
_with_weights(cs::VgtkCore.Csponds, w) = withattrs(cs; weights = w)
_with_weights(cs, w) = VgtkCore.Csponds(sources(cs), targets(cs); weights = w)

"""
    fit_division_camera(radial_cameras::AbstractDict{K}, views::AbstractDict{K};
                        image_size = nothing, focal_seed = nothing, ncoef, k = nothing, max_iter, outlier_halfwidth,   # defaults: configs/default.toml [fit_division_camera]
                        whitening = SampsonWhitening(), verbose = false)
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
under `whitening` and weights it by the Student-t/uniform posterior of the
previous iteration's whitened residuals.

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
the IRLS converges — or, if it is still moving at half the `max_iter` budget,
there (a wrong branch is what keeps a stack from converging) — one iteration
re-judges every view's branch at the shared (f, k), robustly; if it moves any,
the IRLS resumes from there within what is left of `max_iter` (the audit counts
as an iteration, so `iterations` never exceeds `max_iter`); if it moves none the
IRLS continues unchanged, bit for bit (clean stacks). Never after
`:solve_failed`, nor at `max_iter = 0`. Measured: 104 → 4
wrong views at 3 %, 66 → 1 at 5 %, no correct view moved, and the held-out
residuals calibrated against ŝ²·I₂ in the tail as well as the bulk
(`test/ransac_mc_calibration_tests.jl` (e)).

Inputs, `k`, `outlier_halfwidth`, `focal_seed` and every output are in pixels;
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

`outlier_halfwidth` (pixels) is the half-width of the uniform outlier component
— the same quantity the RANSAC problems take; it must NOT be the residual scale
(a half-width at the median residual makes the outlier component as dense as the
inliers and zeroes every weight — measured). The posterior is seeded from the
median residual ([`median_seeded_posterior`](@ref)): every point starts an
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

`max_iter = 0` returns iteration 0 alone — whitened at the seed and robustly
unweighted, `retcode` `:max_iterations`, no branch audit — which is

    N = CspondNormalization(DivisionCameraState, views; center)   # center: the cameras' shared centre
    CameraModel(division_camera_step(N, radial_cameras, views; ncoef, k, whiten_at = focal_seed))

bit for bit (with `focal_seed` the half-diagonal by default). The robust
re-weighting this function exists for begins at iteration 1. A stack whose
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
posterior weight as their `weights` column, the Student-t scale `sigma` = ŝ
(pixels — a LINEAR sigma), ν, and the loop outcome. `rms` (pixels) and `n` count
the rows that entered the fit: a row with prior weight zero, or of a view the
final iteration left without a pose, is not a residual.
"""
function fit_division_camera(
        radial_cameras::AbstractDict{K, <:Union{HomographyMat, RadialCamera}},
        views::AbstractDict{K, <:AbstractVector{<:Cspond}};
        image_size = nothing,
        focal_seed::Union{Nothing, Real} = nothing,
        ncoef::Int = _dflt("fit_division_camera", "ncoef"), k::Union{Nothing, Real} = nothing,
        max_iter::Int = _dflt("fit_division_camera", "max_iter"),
        outlier_halfwidth::Real = _dflt("fit_division_camera", "outlier_halfwidth"),
        whitening::VgtkCore.WhiteningPolicy = VgtkCore.SampsonWhitening(),
        verbose::Bool = false
    ) where {K}
    focal_seed === nothing || (focal_seed > 0 && isfinite(focal_seed)) ||
        throw(ArgumentError("fit_division_camera: focal_seed must be positive and finite (pixels), got $focal_seed"))
    focal_seed === nothing && image_size === nothing &&
        throw(ArgumentError("fit_division_camera: give `image_size` (the default seed is its half-diagonal) or `focal_seed`"))
    max_iter >= 0 || throw(ArgumentError("fit_division_camera: max_iter must be ≥ 0, got $max_iter"))
    ids, Hs, boards = _keyed_views(radial_cameras, views)
    c = _shared_center(Hs)                                    # checked, exact
    prior = _point_weights(boards)
    # THE NORMALIZATION, once — `CspondNormalization(DivisionCameraState, views;
    # center)`; every iteration applies it inside and returns pixels.
    N = _division_normalization(boards, c, prior)
    kk = k === nothing ? nothing : Float64(k)
    fseed = Float64(focal_seed === nothing ? hypot(image_size[1], image_size[2]) / 2 : focal_seed)   # pixels; default: a 90° diagonal FOV
    # iteration 0: whitened at the seed (weights only), robustly unweighted
    state0 = _division_camera_step(N, Hs, boards; ncoef, k = kk, point_weights = prior, whiten_at = fseed, whitening)
    n = sum(length, boards)
    sq0 = _focal_sq!(Vector{Float64}(undef, n), state0)
    rws = median_seeded_posterior(sq0, n; outlier_halfwidth, codimension = 2)
    method = DivisionCameraIrls(Hs, boards, prior, ncoef, kk, whitening, N, verbose)
    st = DivisionCameraIrlsState(state0, rws, n, Inf, 0)
    verbose && _print_iteration(0, state0, rws, Inf)
    lh = PosteriorLikelihood(rws, n)
    # THE BRANCH AUDIT (`_audit_branch`), once, inside the one `max_iter` budget
    # (the audit is an iteration): at convergence, or — for an IRLS still moving
    # at the CHECKPOINT, half the budget — there. A wrong branch is what keeps a
    # stack from converging: measured on the MC scene at 3 % ray-aligned, 86 of
    # 200 stacks ran out of iterations and carried 70 of the 75 wrong views left
    # when the audit was held to converged stacks (whose tail then failed its
    # calibration gate). When the audit moves a view the IRLS resumes from the
    # audited iteration, its posterior refreshed there, for what remains of the
    # budget; when it moves none, the IRLS simply continues — bit for bit the
    # uninterrupted run. Never after `:solve_failed`, and never at `max_iter = 0`.
    checkpoint = max_iter - max_iter ÷ 2
    res = block_coordinate_descent!(st, lh, method, nothing, state0; max_iter = checkpoint)
    niter, rc = iterations(res), retcode(res)
    audited = rc !== :solve_failed && niter < max_iter ? _branch_audit(st, method) : nothing
    if audited !== nothing
        verbose && println("  branch audit: $(_moved_branches(audited, st.state)) view(s) moved; resuming")
        st.state = audited; st.Δ = Inf
        _focal_sq!(sq_residuals(rws), audited)
        update_nuisance!(lh); update_weights!(lh)
        res = block_coordinate_descent!(st, lh, method, nothing, audited; max_iter = max_iter - niter - 1)
        niter, rc = niter + 1 + iterations(res), retcode(res)
    elseif rc === :max_iterations && niter < max_iter
        res = block_coordinate_descent!(st, lh, method, nothing, st.state; max_iter = max_iter - niter)
        niter, rc = niter + iterations(res), retcode(res)
    end
    state = _keyed(estimate(res), ids)
    # ONE construction: `CameraModel(::DivisionCameraState)`. The state carries
    # the pencil's centre — the principal point every radial camera shares,
    # checked above; views are raw pixels and nothing here re-centres them.
    camera = VgtkCore.CameraModel(state)
    wpost = _combined_weights(prior, _focal_point_weights(st, boards))
    # The whitened-residual RMS (px) over the rows that entered the fit: a view
    # without a pose has NaN residuals (the posterior's buffer holds a 1e12
    # sentinel for them, which is not a residual), and a row of prior weight
    # zero was never fitted.
    ss, nres = 0.0, 0
    for (i, sqi) in enumerate(st.state.sq_residuals), (j, r²) in enumerate(sqi)
        (isfinite(r²) && (prior === nothing || prior[i][j] > 0)) || continue
        ss += r²; nres += 1
    end
    return FitResult(
        camera, rc, niter,
        nres > 0 ? sqrt(ss / nres) : NaN, nres,               # whitened-residual RMS over the fitted rows (px)
        DivisionCameraMetadata(
            state, radial_cameras, Dict(ids[i] => _with_weights(views[ids[i]], wpost[i]) for i in eachindex(ids)),
            sqrt(s_hat_sq(rws)), Float64(student_t_dof(rws))
        ),
    )
end
