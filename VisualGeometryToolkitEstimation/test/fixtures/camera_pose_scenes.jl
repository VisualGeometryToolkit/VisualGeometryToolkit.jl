# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The scenes of refine_camera_and_poses_tests.jl (its CameraPoseScenes, SplineCameraPoseScenes, BoardScenes and
# RotationCenterScenes setups, each of which includes this file and exports its own names) and of the golden records of
# ci/golden/fixtures/est_camera_fits.jl: cameras, poses and their exact or noisy pixels. Data constructors only — no
# estimator is called; include-once safe (functions only).
using StaticArrays: SVector, SMatrix
using LinearAlgebra: normalize, norm
using Random: Xoshiro, randn, rand
using StableRNGs: StableRNG
using Rotations: RotMatrix, RotXYZ
using VisualGeometryToolkitCore: EuclideanMap, exp_so3, project, rotation, translation, CameraModel, Pinhole, KannalaBrandt,
    SplineLens, ChartSpline, Size2

# --- CameraPoseScenes: a pinhole and a Kannala–Brandt fisheye, V poses, their spiralling scene points ------------------
pinhole_camera() = CameraModel(; focal = 800.0, pp = (640.0, 480.0), lens = Pinhole())
fisheye_camera() = CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
truth_poses(V) = [EuclideanMap(RotMatrix(RotXYZ(0.3 - 0.1v, -0.2 + 0.07v, 0.1v)), SVector(0.1 - 0.05v, -0.05, 2.0 + 0.1v)) for v in 1:V]
# the truth moved by ≈ 0.05 rad and ≈ 0.08 in translation, differently per view
perturbed_pose(E, v) = EuclideanMap(
    RotMatrix{3, Float64}(exp_so3(SVector(0.03, -0.02v / 4, 0.03)) * E.R), E.t + SVector(0.05, -0.04, 0.06 - 0.01v)
)
# per view, scene points whose camera-frame directions spiral out to `θmax` off the axis (`spread`: the polar angle's
# profile in k / n — linear, or `sqrt` for the spline scenes), and their exact pixels through `camera`
function views_of(camera, poses, θmax; n = 60, spread = identity)
    map(enumerate(poses)) do (v, E)
        map(1:n) do k
            θ = θmax * spread(k / n); φ = 2.399963 * k + v
            d = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
            Xc = (2.0 + 1.5 * ((7k + v) % 10) / 10) * d
            X = E.R' * (Xc - E.t)
            X => SVector{2, Float64}(project(camera, Xc))
        end
    end
end

# --- SplineCameraPoseScenes: the fisheye's SplineLens camera (unit controls, 12 × 10 nodes over 1280 × 960) ----------
spline_truth() = CameraModel(SplineLens, fisheye_camera(), Size2(1280, 960); grid_size = (12, 10))
# every present control turned by ≈ `ϵ` rad off its own direction, differently per node (its norm kept)
function perturbed_controls(cam, ϵ)
    b = cam.lens.backward
    C = map(CartesianIndices(b.controls)) do i
        c = b.controls[i]; k = i[1] + 13 * i[2]
        isfinite(c[1]) ? norm(c) * normalize(c / norm(c) + ϵ * SVector(sin(1.3k), cos(0.7k), sin(2.1k + 0.5))) : c
    end
    return CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, C); present = cam.lens.present))
end
# per view, 120 scene points spiralling out to `θmax` (the polar angle ∝ √(k / n)), and their exact pixels
spline_views(camera, poses, θmax; n = 120) = views_of(camera, poses, θmax; n, spread = sqrt)

# --- BoardScenes: three boards, six views, which boards each view sees ---------------------------------------------
# a 9 × 7 grid of points on the board's plane (z = 0), 4 cm pitch
board_points() = [SVector(0.04 * (i - 5), 0.04 * (j - 4), 0.0) for i in 1:9 for j in 1:7]
# three boards around the world origin, tilted differently (board → world)
truth_boards() = Dict(:A => EuclideanMap(RotMatrix(RotXYZ(0.0, 0.0, 0.0)), SVector(0.0, 0.0, 0.0)),
                      :B => EuclideanMap(RotMatrix(RotXYZ(0.25, -0.3, 0.1)), SVector(0.42, 0.05, 0.08)),
                      :C => EuclideanMap(RotMatrix(RotXYZ(-0.2, 0.35, -0.15)), SVector(-0.38, -0.06, 0.12)))
# six cameras looking at the boards from 1.2–1.6 m (world → camera)
view_poses() = Dict(v => EuclideanMap(RotMatrix(RotXYZ(0.15 * sin(v), 0.2 * cos(v), 0.05v)), SVector(0.08 * (v - 3.5), 0.03 * (-1)^v, 1.2 + 0.08v))
                    for v in 1:6)
# which boards each view sees: every board twice or more, the graph connected
board_sees() = Dict(1 => (:A, :B), 2 => (:A, :C), 3 => (:B, :C), 4 => (:A, :B, :C), 5 => (:B,), 6 => (:C, :A))
# the board views, keyed (view, board); the noise drawn view by view in `order` (the views' own key order by default;
# the golden records draw from a StableRNG in sorted order, the same on every Julia)
function board_views(camera, views, boards; σ = 0.0, rng = Xoshiro(1), order = keys(views))
    sees = board_sees()
    out = Dict{Tuple{Int, Symbol}, Vector{Pair{SVector{3, Float64}, SVector{2, Float64}}}}()
    for v in order, b in sees[v]
        out[(v, b)] = [X => SVector{2, Float64}(project(camera, views[v](boards[b](X)))) + σ * SVector(randn(rng), randn(rng))
                       for X in board_points()]
    end
    return out
end
# a pose moved by ≈ 0.04 rad and ≈ 3 cm, differently per key
nudged(E, k) = EuclideanMap(RotMatrix{3, Float64}(exp_so3(SVector(0.02, -0.015, 0.03) .* (1 + 0.1k)) * rotation(E)),
                            translation(E) + SVector(0.03, -0.02, 0.025) .* (1 - 0.07k))

# --- RotationCenterScenes: a 9 × 7 board turned about one fixed point -------------------------------------------------
# the camera ≈ 0.6 m in front of the board, its projection centre 5 cm off the rotation centre; one rotation per axis
function center_truth(axes)
    R0 = SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, -1, 0, 0, 0, -1)
    return (c = R0' * (SVector(0.004, -0.06, 0.05) - SVector(0.0, 0.0, 0.6)), d = SVector(0.004, -0.06, 0.05),
            R = Dict(k => SMatrix{3, 3, Float64, 9}(exp_so3(ω)) * R0 for (k, ω) in enumerate(axes)))
end
# The noise is drawn from a StableRNG, view by view in key order: the same draw on every Julia version (`Xoshiro`'s
# `randn` stream and a `Dict`'s iteration order both may change with it, and Julia 1.12 drew differently from 1.13).
function center_views(camera, t; σ = 0.0, outliers = 0.0, rng = StableRNG(2))
    X = board_points()
    return Dict(k => [x => SVector{2, Float64}(project(camera, t.R[k] * (x - t.c) + t.d)) + σ * SVector(randn(rng), randn(rng)) +
                          (rand(rng) < outliers ? SVector(25.0, -30.0) : zero(SVector{2, Float64})) for x in X]
                for k in sort!(collect(keys(t.R))))
end
