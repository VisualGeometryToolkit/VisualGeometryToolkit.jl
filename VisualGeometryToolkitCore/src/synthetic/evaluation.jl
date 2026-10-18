# =============================================================================
# Evaluation Metrics
# =============================================================================

# Also accept SVector (for conic 6-vecs)
# =============================================================================
# Pose Error (IMC metric)
# =============================================================================
#
# Uses Rotations' rotation_angle, and translation_angle from geometry/transforms/euclidean_map.jl
# Uses motion(::EssentialMat) from geometry/solvers/fundmat.jl
# Uses calibration_matrix from geometry/cameras/

"""
    pose_error(R_est, t_est, R_gt, t_gt) -> (rot_err, trans_err)

Rotation and translation angular errors in radians.
"""
function pose_error(
        R_est::AbstractMatrix, t_est::AbstractVector,
        R_gt::AbstractMatrix, t_gt::AbstractVector
    )
    return (rotation_angle(RotMatrix{3}(SMatrix{3, 3}(R_est)' * SMatrix{3, 3}(R_gt))), translation_angle(t_est, t_gt))
end

"""
    imc_pose_error(F_est, K, R_gt, t_gt) -> Float64

Image Matching Challenge pose error metric (DEGREES).

Decomposes `E = KᵀFK` into 4 possible (R, t) solutions, computes
`max(rotation_error, translation_error)` for each, and returns the minimum.
This is the coherent (single-decomposition) counterpart to
`optimistic_pose_error`, which takes per-axis minima from possibly different
decompositions; both report in degrees.

This is the standard metric from the IMC benchmark: it measures how well
an estimated fundamental matrix recovers the true relative pose.

# Arguments
- `F_est`: Estimated 3×3 fundamental matrix
- `K`: Shared 3×3 calibration matrix (or K1, K2 for different intrinsics)
- `R_gt`: Ground-truth 3×3 rotation matrix
- `t_gt`: Ground-truth 3-vector translation

# Returns
- `Float64`: Pose error in degrees (max of rotation/translation angular errors,
  minimized over the 4 essential matrix decompositions)
"""
function imc_pose_error(
        F_est::AbstractMatrix, K::AbstractMatrix,
        R_gt::AbstractMatrix, t_gt::AbstractVector
    )
    E = EssentialMat(F_est, K)
    _imc_pose_error_from_E(E, R_gt, t_gt)
end

function imc_pose_error(
        F_est::AbstractMatrix,
        K1::AbstractMatrix, K2::AbstractMatrix,
        R_gt::AbstractMatrix, t_gt::AbstractVector
    )
    E = EssentialMat(F_est, K1, K2)
    _imc_pose_error_from_E(E, R_gt, t_gt)
end

"""
    imc_pose_error(F_est, rig::StereoRig) -> Float64

IMC pose error using a `StereoRig` for ground-truth pose and calibration.

Extracts K from the source camera and (R, t) from the rig's extrinsics.
Assumes shared intrinsics.
"""
function imc_pose_error(F_est::AbstractMatrix, rig::StereoRig)
    K = calibration_matrix(source(rig))
    R_gt = SMatrix{3, 3, Float64}(RotMatrix(extrinsics(rig).R))
    t_gt = Vec3{Float64}(extrinsics(rig).t)
    imc_pose_error(F_est, K, R_gt, t_gt)
end

"""
    imc_pose_error(E, R_gt, t_gt) -> Float64

IMC pose error (degrees) of an **already-metric** essential matrix `E` against
the ground-truth pose — decomposes `E` directly via `motion(E)`, no calibration.
Use the `(F, K…)` overloads when starting from a fundamental matrix.
"""
imc_pose_error(E::AbstractMatrix, R_gt::AbstractMatrix, t_gt::AbstractVector) =
    _imc_pose_error_from_E(E, R_gt, t_gt)

function _imc_pose_error_from_E(
        E::AbstractMatrix,
        R_gt::AbstractMatrix,
        t_gt::AbstractVector
    )
    return minimum(e -> max(e[1], e[2]), _candidate_pose_errors(motion(E), R_gt, t_gt))   # DEGREES
end

# The ONE loop over a decomposition's candidates `(R, t, …)`: each one's
# `(Δr, Δt)` in DEGREES against the ground truth. The IMC error takes the best
# single candidate's worse axis, the optimistic error each axis's own minimum.
_candidate_pose_errors(decs, R_gt, t_gt) =
    (rad2deg.(pose_error(d[1], d[2], R_gt, t_gt)) for d in decs)

# Per-axis minima of candidate errors, `(180, 180)` when there are none.
_per_axis_min(errs) =
    foldl((a, b) -> (b[1] < a[1] ? b[1] : a[1], b[2] < a[2] ? b[2] : a[2]), errs; init = (180.0, 180.0))

"""
    optimistic_pose_error(E::AbstractMatrix, R_gt, t_gt) -> (Δr_deg, Δt_deg)

Lower-bound (per-axis-optimistic) pose error of an essential matrix
against a ground-truth pose.  Decomposes `E` into its 4 sign-ambiguous
candidates and returns the per-axis minimum errors:

    Δr_min = min over decomps k of  rad2deg(rotation_angle(RotMatrix{3}(R_kᵀ R_gt)))
    Δt_min = min over decomps k of  rad2deg(translation_angle(t_k, t_gt))

The two minimums may come from **different** decompositions — the
returned tuple does not correspond to any single `(R, t)` pair, and is
therefore unrealizable as a coherent single-decomposition output.  It
bounds from below what any one decomposition could achieve per axis.

Used by the Shekhovtsov RANSAC-Scoring protocol (per-axis independent
min, see `RANSAC-Scoring/model_H.py:438-452`) and by the pool record
`Pools.PoolPair`, which stores `errors_r` / `errors_t` as separate columns.

For a coherent single-decomposition error: select the desired
decomposition externally (e.g. via cheirality on real correspondences,
or `argmin_k max(Δr_k, Δt_k)` against the oracle) and call
`pose_error(R, t, R_gt, t_gt)` directly.

See also: `pose_error`, `imc_pose_error`.
"""
function optimistic_pose_error(
        E::AbstractMatrix,
        R_gt::AbstractMatrix,
        t_gt::AbstractVector
    )
    return _per_axis_min(_candidate_pose_errors(motion(E), R_gt, t_gt))
end

"""
    optimistic_pose_error(H::HomographyMat, R_gt, t_gt) -> (Δr_deg, Δt_deg)

Per-axis-optimistic pose error (DEGREES) of a **calibrated** homography against a
ground-truth pose: decomposes `H` via Malis–Vargas `motion(H)` and returns the
per-axis minima over the (up to 4) `(R, t, n)` candidates, exactly as the
essential-matrix overload does (the two minima may come from different
candidates — a per-axis lower bound).  Returns `(180.0, 180.0)` when the
decomposition is empty (singular `H`).

`H` must already be calibrated; for a pixel-space `H_px` use the intrinsics
form below.  Translation error uses `translation_angle`, which is `0` for a
zero-translation (pure-rotation) candidate.
"""
function optimistic_pose_error(
        H::HomographyMat,
        R_gt::AbstractMatrix,
        t_gt::AbstractVector
    )
    return _per_axis_min(_candidate_pose_errors(motion(H), R_gt, t_gt))
end

"""
    optimistic_pose_error(H::HomographyMat, K1, K2, R_gt, t_gt) -> (Δr_deg, Δt_deg)

The pose error of a PIXEL-space homography with intrinsics `K1` (source view)
and `K2` (target view): calibrated as `inv(K2)·H·K1`, then the calibrated form
above.
"""
optimistic_pose_error(
    H::HomographyMat, K1::AbstractMatrix, K2::AbstractMatrix,
    R_gt::AbstractMatrix, t_gt::AbstractVector
) = optimistic_pose_error(
    HomographyMat{Float64}(Tuple(inv(SMatrix{3, 3, Float64, 9}(K2)) * SMatrix{3, 3, Float64, 9}(H) * SMatrix{3, 3, Float64, 9}(K1))),
    R_gt, t_gt
)

# =============================================================================
# Projective model distance — a model known only up to scale and sign
# =============================================================================

"""
    projective_distance(A, B) -> Float64

Distance between two models known only up to a nonzero scale and a sign (E, F,
H, a radial camera's rows, a homogeneous vector): the Euclidean distance between
the unit-normalized entries, with `B` taken in `A`'s sign,

    ‖â − sign(â·b̂)·b̂‖ = min(‖â − b̂‖, ‖â + b̂‖),   â = vec(A)/‖A‖.

It lies in `[0, √2]`, is `0` exactly when `A ∝ B`, and is symmetric. For small
values it is the angle between the two representatives: `1 − |â·b̂| = d²/2`.
"""
function projective_distance(A, B)
    a = vec(collect(Float64, A)); b = vec(collect(Float64, B))
    length(a) == length(b) || throw(DimensionMismatch("projective_distance: $(length(a)) vs $(length(b)) entries"))
    a ./= norm(a); b ./= norm(b)
    s = dot(a, b) < 0 ? -1.0 : 1.0
    return norm(a .- s .* b)
end

# =============================================================================
# mAA — mean Average Accuracy (the IMC benchmark metric)
# =============================================================================

"""
    maa(errors, t) -> Float64

Accuracy at threshold `t`: the fraction of `errors` strictly below `t`.
Non-finite entries (failed estimates) never succeed and therefore count as
failures in the denominator.
"""
maa(errors, t::Real) = count(e -> e < t, errors) / max(length(errors), 1)

"""
    mean_average_accuracy(errors; T) -> Float64

THE canonical mAA@`T` (≡ AUC@`T`): the normalized area under the empirical CDF
of `errors` from 0 to `T` — `(1/T)∫₀ᵀ F(t)dt`, equivalently
`mean_p max(0, 1 - min(e_p, T)/T)`. Non-finite entries (failed estimates)
contribute 0 but stay in the denominator. This is the EXACT AUC; the former
discrete threshold-average variant is retired — it is strictly coarser (it
cannot resolve errors below `T`/#thresholds). All consumers use this
continuous statistic.

`T` is required, in the errors' own units: the threshold is a protocol
parameter of the caller's benchmark (the IMC pose-error mAA takes `T = 10`
degrees), never a hidden default.
"""
function mean_average_accuracy(errors; T::Real)
    n = length(errors); n == 0 && return NaN
    s = 0.0
    @inbounds for e in errors      # sequential sum, as before: `sum` would pair-sum
        s += maa_score(e; T)
    end
    return s / n
end

"""
    maa_score(e; T) -> Float64

Per-item contribution to the continuous [`mean_average_accuracy`](@ref):
`max(0, 1 - min(e, T)/T)`, and `0` for non-finite `e`. Use this per-sample form
for bootstrap resampling; `mean(e -> maa_score(e; T), errors) ==
mean_average_accuracy(errors; T)`. `T` is required, as for the aggregate.
"""
maa_score(e; T::Real) =
    isfinite(e) ? max(0.0, 1.0 - min(Float64(e), Float64(T)) / Float64(T)) : 0.0
