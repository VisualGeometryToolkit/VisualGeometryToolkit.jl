# The fits' shared test helpers: the reweighted fits (a rotation, a rigid alignment, a linear model, the robot-world
# calibration), the robot-world reprojection refine, the rotation averaging and its filtered form, and the error of
# an averaged estimate. Include-once safe (functions only).
using StaticArrays: SMatrix
using Rotations: RotMatrix, rotation_angle
using VisualGeometryToolkitCore: ClassDefault, EstimationProblem, RefineProblem, LevenbergMarquardt, solve, AbsoluteRotations, EuclideanMap
using VisualGeometryToolkitEstimation: ReweightedLeastSquares, LinearModel, RobotWorldHandEyeCalibration, RobotWorldHandEyeProblem,
    RotationAveraging, FilteredRotationAveraging

# `ReweightedLeastSquares` on the pairs `cs` for the model `M`, its strategy given; `kw` the run settings
reweighted_fit(M, cs; optimization_strategy = ClassDefault(), kw...) =
    solve(EstimationProblem(M, cs), ReweightedLeastSquares(; optimization_strategy); kw...)
rotation_fit(cs; kw...) = reweighted_fit(RotMatrix{3}, cs; kw...)
rigid_fit(cs; kw...) = reweighted_fit(EuclideanMap{3}, cs; kw...)

# `ReweightedLeastSquares` on the linear model `A x = y` (prior weights optional)
linear_fit(A, y, prior...; optimization_strategy = ClassDefault(), tol = ClassDefault(), kw...) =
    solve(EstimationProblem(LinearModel, (A, y, prior...)), ReweightedLeastSquares(; optimization_strategy, tol); kw...)

# `ReweightedLeastSquares` on a robot-world calibration of `markers`; the noise levels and the angle gate, when given,
# are the problem's context
rwhe_fit(markers; σ_rot = nothing, σ_trans = nothing, min_rel_angle = nothing, optimization_strategy = ClassDefault(),
         tol = ClassDefault(), kw...) =
    solve(EstimationProblem(RobotWorldHandEyeCalibration, markers,
                            NamedTuple(k => v for (k, v) in pairs((; σ_rot, σ_trans, min_rel_angle)) if v !== nothing)),
          ReweightedLeastSquares(; optimization_strategy, tol); kw...)

# the LevenbergMarquardt refine of the robot-world calibration `guess` against the markers' features
rwhe_refine(guess, camera, markers; optimization_strategy = ClassDefault(), kw...) =
    solve(RefineProblem(RobotWorldHandEyeProblem(camera, markers), guess), LevenbergMarquardt(; optimization_strategy); kw...)

# `RotationAveraging` of the relative rotations `d` (a Dict of edges or a ViewGraph)
rotation_averaging(d; optimization_strategy = ClassDefault(), tol = ClassDefault(), seed = :spectral, kw...) =
    solve(EstimationProblem(AbsoluteRotations, d), RotationAveraging(; optimization_strategy, tol, seed); kw...)

# `FilteredRotationAveraging` of the view graph `vg`, its filters given
filtered_averaging(vg; cycle_tol = nothing, orientation_tol = nothing, kw...) =
    solve(EstimationProblem(AbsoluteRotations, vg), FilteredRotationAveraging(; cycle_tol, orientation_tol); kw...)

# the largest angle between the estimated rotation `Rh[k]` and the truth `R[k]`, over the views k in 1:n
max_rotation_error(Rh, R, n) = maximum(k -> rotation_angle(RotMatrix{3}(SMatrix{3, 3}(Rh[k])' * R[k])), 1:n)
