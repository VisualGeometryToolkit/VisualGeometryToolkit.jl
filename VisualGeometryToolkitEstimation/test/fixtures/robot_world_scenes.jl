# The robot-world hand-eye reprojection scene of refine_robot_world_hand_eye_tests.jl (and of the golden records of
# ci/golden/fixtures/est_other_fits.jl): markers of feature points on a rig, a camera looking at it, the rig's poses,
# and each marker's pixels per frame. Data constructors only — no estimator is called; include-once safe (functions only).
using LinearAlgebra: normalize
using StaticArrays: SVector
using Rotations: AngleAxis
using VisualGeometryToolkitCore: CameraModel, project
import VisualGeometryToolkitEstimation as VgtkEstimation

# A marker: a 5×5 grid of points (spacing 3 cm) in its own z = 0 plane.
rw_grid(s = 0.03) = [SVector(s * (i - 3), s * (j - 3), 0.0) for i in 1:5 for j in 1:5]

# The camera Y = T_WC, rig poses A_i = Y ∘ C_i placing the rig ~depth in front
# of the camera (C_i = rig in camera, rotations up to `ang`), M markers X_m on
# the rig — in a row 8 cm apart (`layout = :row`), or, for many markers, on a
# ⌈√M⌉-column grid 15 cm apart (`layout = :grid`) that keeps every marker in
# view. f and the image set the FOV; the rig's lateral jitter scales with it
# so every point stays in the image. Pixel noise σ, isotropic.
function rw_marker_origin(rng, m, M, layout)
    layout === :row && return SVector(0.08 * (m - (M + 1) / 2), 0.03 * randn(rng), 0.0)
    nc = ceil(Int, sqrt(M)); nr = cld(M, nc); r, c = divrem(m - 1, nc)
    return SVector(0.15 * (c - (nc - 1) / 2), 0.15 * (r - (nr - 1) / 2), 0.0)
end
function rw_scene(rng; N = 12, M = 3, f = 600.0, W = 800, H = 600, depth = 1.2, σ = 0.5, ang = 0.35, layout = :row)
    (; rigid, marker_pose) = VgtkEstimation.synthetic_scenes().PoseScenes
    camera = CameraModel(; focal = f, pp = (W / 2, H / 2))
    Y = rigid(AngleAxis(π * rand(rng), normalize(randn(rng, 3))...), 0.5 .* randn(rng, SVector{3, Float64}))
    X = [rigid(AngleAxis(0.3 * rand(rng), normalize(randn(rng, 3))...), rw_marker_origin(rng, m, M, layout)) for m in 1:M]
    lat = 0.35 * depth * (W / 2) / f
    A = map(1:N) do _
        C = rigid(AngleAxis(ang * rand(rng), normalize(randn(rng, 3))...), SVector(lat * (2rand(rng) - 1), lat * (H / W) * (2rand(rng) - 1), depth * (0.85 + 0.3rand(rng))))
        Y ∘ C
    end
    obs = Tuple{Int, Int, Vector{Pair{SVector{3, Float64}, SVector{2, Float64}}}}[]
    for i in 1:N, m in 1:M
        B = marker_pose(Y, A[i], X[m])
        cs = Pair{SVector{3, Float64}, SVector{2, Float64}}[]
        for P in rw_grid()
            Pc = B(P)
            Pc[3] > 0.1 || continue
            u0 = SVector{2, Float64}(project(camera, Pc)...)
            (0 < u0[1] < W && 0 < u0[2] < H) || continue
            push!(cs, P => u0 + σ * randn(rng, SVector{2, Float64}))
        end
        isempty(cs) || push!(obs, (i, m, cs))
    end
    return (; camera, Y, X, A, obs)
end
# The refinement's input: per marker, one `A_i => [X => u, …]` element per frame
# it is seen in (frame order), over the frames in `frames`.
rw_markers(sc; frames = 1:length(sc.A)) = Dict(
    m => [sc.A[i] => cs for (i, mm, cs) in sc.obs if mm == m && i in frames]
        for m in unique(o[2] for o in sc.obs if o[1] in frames)
)
# A seed: the truth perturbed by `rot` rad and `tr` (length) per block.
function rw_seed(rng, sc; rot = 0.01, tr = 0.01)
    (; perturb_pose) = VgtkEstimation.synthetic_scenes().PoseScenes
    pert(T) = perturb_pose(rng, T; σ_R = rot, σ_t = tr)
    return VgtkEstimation.RobotWorldHandEyeCalibration(pert(sc.Y), Dict(m => pert(sc.X[m]) for m in eachindex(sc.X)))
end
