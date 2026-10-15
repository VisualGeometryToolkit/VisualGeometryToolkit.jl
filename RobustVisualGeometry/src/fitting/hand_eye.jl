# =============================================================================
# Hand-Eye / Robot-World Calibration  —  Fitting Pipeline
# =============================================================================
#
# Recover the rigid transform X = T_RP (pattern in rig) from synchronized
# pose observations A_i = T_WR (mocap) and B_i = T_CP (PnP). The nuisance
# Y = T_WC drops out via the AX = XB relative-motion formulation.
#
# find_hand_eye — thin wrapper around ransac() with least-squares-on-the-
#                inliers LO and MSAC scoring.
#
# See `HandEyeProblem` in src/estimators/ransac/problems/hand_eye.jl for
# the underlying RANSAC problem definition.
# =============================================================================

"""
    find_hand_eye(cs; threshold=0.01·L,
                       max_trials=10000, min_trials=100,
                       local_optimization=Irls{TLSShape}(),
                       motion_pairs=:consecutive,
                       char_length=nothing, min_rel_angle,
                       scoring=nothing, config=nothing) -> RansacEstimate

Robust hand-eye / robot-world calibration via RANSAC + IRLS. Every
default shown is `configs/default.toml`'s (`[ransac]`, `[pipelines.hand_eye]`);
a keyword overrides.

# Arguments
- `cs` — one `A_i => B_i` correspondence per frame, in time order: the
  rig-in-world pose (from motion capture) and the pattern-in-camera pose (from
  PnP; labeled `Extrinsics3` accepted — unlabeled at the problem boundary).

# Keywords
- `threshold::Real` — the MSAC inlier threshold of the default scoring, in
  the TRANSLATION's length unit (the scene's, as the poses carry it): the
  residual stacks the translation error with the rotation error scaled by the
  characteristic length, `L·θ` (see [`HandEyeProblem`](@ref) for the unit
  balance). A value you supply is that length. When omitted it is unit-free:
  `configs/default.toml` `[pipelines.hand_eye]` `threshold` (0.01) is a
  FRACTION of the problem's characteristic length `L`, so the default
  threshold is `0.01·L` — the same whether the poses are in metres or
  millimetres (a rotation error of 0.01 rad ≈ 0.57° reaches it).
  Builds the default scoring only; supplying it with `scoring` is an `ArgumentError`.
- `motion_pairs::Symbol = :consecutive` — `:consecutive` uses `N-1`
  consecutive-frame relative motions; `:all` uses every pair `(i, j)`
  with `i < j`.
- `char_length` — characteristic length forwarded to `HandEyeProblem`
  (default: median rig translation per motion).
- `min_rel_angle` — the smallest rotation (rad) a relative motion may have;
  motions below it are dropped at construction (default `[robot_world_hand_eye]`
  `min_rel_angle_deg` — see [`HandEyeProblem`](@ref)).
- `max_trials`, `min_trials` — RANSAC trial bounds.
- `local_optimization` — default `Irls{TLSShape}()` — HandEyeProblem has no chart lift; its
  weighted refit (Levenberg–Marquardt on the so(3)⊕R³ chart of `Rigid3`) is
  selected by `refine!(lo_ws, ::HandEyeProblem, θ::Rigid3, weights)`.
- `scoring`, `config` — override scoring / `RansacConfig`; the other keywords
  as [`find_homography`](@ref).

Returns an `FitResult{Rigid3, RansacMetadata}` whose `estimate` is
`X = T_RP`. The nuisance camera-in-world transform `Y` is **not**
computed by this pipeline.
`iterations(result)` is the number of trials (hypotheses drawn); test
`converged(result)` before reading the model (see [`RansacEstimate`](@ref)).

# Examples
```julia
A = [Rigid3(...) for _ in 1:N]   # mocap
B = [Rigid3(...) for _ in 1:N]   # PnP
result = find_hand_eye(A .=> B)
X = estimate(result)                # T_RP, the calibration
```
"""
function find_hand_eye(
        cs::AbstractVector{<:Cspond{<:EuclideanMap{3}, <:EuclideanMap{3}}};
        motion_pairs::Symbol = Symbol(_dflt("pipelines.hand_eye", "motion_pairs")),
        char_length::Union{Nothing, Real} = nothing,
        min_rel_angle::Real = deg2rad(_dflt("robot_world_hand_eye", "min_rel_angle_deg")),
        kwargs...
    )
    problem = HandEyeProblem(cs; motion_pairs, char_length, min_rel_angle)
    return _ransac_pipeline(problem, :hand_eye; kwargs...)
end
