# =============================================================================
# VisualGeometryToolkitEstimationTestExt — the stack's shared synthetic test scenes
# =============================================================================
#
# ONE definition of every scene more than one test draws from, here and in every
# stack package downstream (LocalOptimizationDoneRight's paper tests drew copies of
# them): `synthetic_scenes()` returns the module `TestScenes`, whose submodules are the
# scene families. VisualGeometryToolkitEstimation's own items reach them through
# thin `@testsetup` modules (test/scenes_testsetups.jl).
#
# Every scene is built through the OWNERS' forward models — VisualGeometryToolkitCore's
# `project`/`unproject` on a camera, `EssentialMat(R, t)`, `FundMat(E, K₁, K₂)`,
# `RadialCamera(camera)`, the rigid charts' `retract` — so no test restates the
# geometry it then checks. A scene fixes only what it is: the rotation and
# translation it is seen from, where its points are drawn, the noise, and the
# order the draws come out of the RNG (pinned fixtures downstream depend on that
# order; do not reorder a draw).
# =============================================================================
module VisualGeometryToolkitEstimationTestExt

using Test: Test
using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation

module TestScenes

    # -----------------------------------------------------------------------------
    # Contaminated correspondence sets: inliers with target noise, then outliers,
    # optionally shuffled (first: the other families build on it).
    # -----------------------------------------------------------------------------
    module Contamination
    using Random: AbstractRNG, randperm
    using StaticArrays: SA, SVector
    using VisualGeometryToolkitCore: Line2D, Point2
    using VisualGeometryToolkitEstimation: LineFittingProblem

    export contaminated_pairs, contaminated_line

    _inlier(rng, inliers::AbstractVector, i) = inliers[i]
    _inlier(rng, inlier, i) = inlier(rng, i)

    """
        contaminated_pairs(rng, inliers[, n_in]; σ = 0, n_out = 0, outlier, shuffle = false) -> (pairs, is_inlier)

    `n_in` inlier pairs `a => b + σ·(randn, randn)` — `inliers` is a vector of
    clean pairs, or a function `(rng, i) -> a => b` drawn in turn with each one's
    noise right after it — then `n_out` pairs `outlier(rng, k)`, then, with
    `shuffle`, one `randperm` of both. The noise is drawn at every σ, 0 included,
    so a scene's draws do not depend on σ.
    """
    function contaminated_pairs(
            rng::AbstractRNG, inliers, n_in = length(inliers);
            σ = 0.0, n_out = 0, outlier = nothing, shuffle = false
        )
        pairs = map(1:n_in) do i
            a, b = _inlier(rng, inliers, i)
            a => b + σ * SA[randn(rng), randn(rng)]
        end
        outs = [outlier(rng, k) for k in 1:n_out]
        all_pairs = n_out == 0 ? pairs : [pairs; outs]
        is_inlier = [trues(n_in); falses(n_out)]
        shuffle || return all_pairs, is_inlier
        perm = randperm(rng, length(all_pairs))
        return all_pairs[perm], is_inlier[perm]
    end
    
    """
        contaminated_line(rng) -> (problem, seed)

    57 points on `y = 0` at `x = 0, 1, …, 56` (σ = 0.01 in y), then 3 outliers at
    `y = 0.5` (x = 10, 30, 50), as a `LineFittingProblem`; `seed` is the
    contaminated mean line `y = 0.025` a local optimization starts from.
    """
    function contaminated_line(rng::AbstractRNG)
        pts = [Point2{Float64}(Float64(i), 0.01 * randn(rng)) for i in 0:56]
        append!(pts, [Point2{Float64}(Float64(x), 0.5) for x in (10, 30, 50)])
        return LineFittingProblem(pts), Line2D{Float64}(SVector(0.0, 1.0, -0.025))
    end
    end

    # -----------------------------------------------------------------------------
    # Two-view epipolar scenes: calibrated (normalized-plane) pairs, the pixel
    # cameras the E problems are stated in, and the stereo pixel pairs of F.
    # -----------------------------------------------------------------------------
    module EpipolarScenes
    using Random: AbstractRNG
    using LinearAlgebra: normalize
    using StaticArrays: SA, SMatrix, SVector
    using Rotations: RotXYZ, RotY, RotZ
    using VisualGeometryToolkitCore: CameraModel, EssentialMat, FundMat, Pinhole, Point2, TLSShape, calibration_matrix, cost,
        correspondences, project, px, sampson_residual
    using VisualGeometryToolkitEstimation: EssentialProblem

    export box_point, box_point_far, calibrated_pairs, perturbed_essential, stereo_pixel_pairs, fundmat_scene, truncated_sampson_cost,
        FPX, CX, CY, ECAM, topx, essential_problem, ecs, PIXEL_PAIRS_K, contaminated_pixel_pairs

    # The pinhole of `contaminated_pixel_pairs` (fx = 800, fy = 820, a 640 × 480 frame).
    const PIXEL_PAIRS_K = SA[800.0 0 320; 0 820.0 240; 0 0 1]

    """
        contaminated_pixel_pairs(rng, n; planar) -> Correspondences

    `n` pixel pairs of two `PIXEL_PAIRS_K` views, view 2 at `x₂ = R·x₁ + t` (a fixed
    small rotation, `t = (1, 0.1, 0.05)`): points uniform in `[-1, 1] × [-0.75, 0.75]`
    at depth 4 (`planar`, a homography scene) or uniform in `[3, 5]`; each view-2
    pixel, with probability 0.4, replaced by a uniform one in the frame (a gross
    outlier), else perturbed by 0.5 px. Every draw is made in this order per pair.
    """
    function contaminated_pixel_pairs(rng::AbstractRNG, n; planar)
        K = PIXEL_PAIRS_K
        Rm = SMatrix{3, 3, Float64, 9}(RotXYZ(0.05, -0.1, 0.03)); t = SA[1.0, 0.1, 0.05]
        u1 = Point2{Float64}[]; u2 = Point2{Float64}[]
        for _ in 1:n
            X = SA[2 * (rand(rng) - 0.5), 1.5 * (rand(rng) - 0.5), planar ? 4.0 : 3.0 + 2 * rand(rng)]
            p1 = K * X; p2 = K * (Rm * X + t)
            push!(u1, Point2{Float64}(p1[1] / p1[3], p1[2] / p1[3]))
            push!(u2, rand(rng) < 0.4 ? Point2{Float64}(640 * rand(rng), 480 * rand(rng)) :
                      Point2{Float64}(p2[1] / p2[3] + 0.5 * randn(rng), p2[2] / p2[3] + 0.5 * randn(rng)))
        end
        return correspondences(u1, u2)
    end

    """
        box_point(rng; width = 2, height = 1.5, near = 3, far = 5) -> SVector{3}

    A point uniform in the box `[-width/2, width/2] × [-height/2, height/2] ×
    [near, far]` in front of view 1 (x drawn first, then y, then the depth).
    """
    box_point(rng::AbstractRNG; width = 2.0, height = 1.5, near = 3.0, far = 5.0) =
        SA[width * (rand(rng) - 0.5), height * (rand(rng) - 0.5), near + (far - near) * rand(rng)]

    # A point in the wider, farther box `[-2, 2] × [-1.5, 1.5] × [5, 8]` of the stereo
    # pixel scenes (the F problems').
    box_point_far(rng::AbstractRNG) = box_point(rng; width = 4.0, height = 3.0, near = 5.0, far = 8.0)

    """
        calibrated_pairs(rng, R, t; n, σ, outlier_every = 0) -> (u₁, u₂)

    Calibrated two-view pairs on the z = 1 plane: `n` draws of `box_point`, seen
    by view 2 at `x₂ = R·x₁ + t` (draws behind view 2 are skipped), view 2
    perturbed by `σ` per coordinate.
    Every `outlier_every`-th draw's view-2 point is replaced by a uniform one in
    `[-0.4, 0.4]²` (a gross outlier).
    """
    function calibrated_pairs(rng::AbstractRNG, R, t; n, σ, outlier_every = 0)
        u₁ = Point2{Float64}[]; u₂ = Point2{Float64}[]
        for k in 1:n
            X = box_point(rng)
            Y = R * X + t
            Y[3] <= 0 && continue
            push!(u₁, Point2{Float64}(project(Pinhole(), X)))
            if outlier_every > 0 && k % outlier_every == 0
                push!(u₂, Point2{Float64}(0.8 * (rand(rng) - 0.5), 0.8 * (rand(rng) - 0.5)))
            else
                push!(u₂, Point2{Float64}(project(Pinhole(), Y) + σ * SA[randn(rng), randn(rng)]))
            end
        end
        return u₁, u₂
    end

    """
        perturbed_essential(R, t, δ, Δt) -> EssentialMat

    The seed a local optimization starts from: the pose `(R, t)` rolled by `δ` about
    view 2's optical axis and its direction moved by `Δt`.
    """
    perturbed_essential(R, t, δ, Δt) = EssentialMat(R * RotZ(δ), normalize(t + Δt))

    """
        stereo_pixel_pairs(rng, cam₁, cam₂, R, t; n, point, σ₁ = 0, σ₂ = 0, n_out = 0, outlier)

    Pixel pairs of two cameras, view 2 at `x₂ = R·x₁ + t`: `n` draws of `point(rng)`
    (draws behind view 2 are skipped) projected through each camera, each view
    perturbed by its own `σ` (no draw at σ = 0), then `n_out` pairs `outlier(rng)`.
    """
    function stereo_pixel_pairs(
            rng::AbstractRNG, cam₁, cam₂, R, t;
            n, point, σ₁ = 0.0, σ₂ = 0.0, n_out = 0, outlier = nothing
        )
        u₁ = SVector{2, Float64}[]; u₂ = SVector{2, Float64}[]
        for _ in 1:n
            X = point(rng)
            Y = R * X + t
            Y[3] <= 0 && continue
            p₁ = SVector{2, Float64}(project(cam₁, X))
            σ₁ > 0 && (p₁ += σ₁ * SA[randn(rng), randn(rng)])
            p₂ = SVector{2, Float64}(project(cam₂, Y))
            σ₂ > 0 && (p₂ += σ₂ * SA[randn(rng), randn(rng)])
            push!(u₁, p₁); push!(u₂, p₂)
        end
        for _ in 1:n_out
            o₁, o₂ = outlier(rng)
            push!(u₁, o₁); push!(u₂, o₂)
        end
        return u₁, u₂
    end

    # The two cameras of `fundmat_scene` (different intrinsics) and view 2's pose, `x₂ = R·x₁ + t`.
    const _FUNDMAT_CAMERAS = (CameraModel(; focal = 500.0, pp = (320.0, 240.0)), CameraModel(; focal = 600.0, pp = (300.0, 250.0)))
    const _FUNDMAT_POSE = (RotY(0.15), SA[1.0, 0.2, 0.1])

    """
        fundmat_scene(rng; n_inliers = 100, n_outliers = 30, σ = 0.5) -> (; source, target, F, n_inliers)

    THE two-camera fundamental-matrix scene: two pinhole cameras of different
    intrinsics (f = 500 at (320, 240), f = 600 at (300, 250)), view 2 at
    `x₂ = R·x₁ + t` with `R = RotY(0.15)`, `t = (1, 0.2, 0.1)`; `stereo_pixel_pairs` of
    `n_inliers` draws of `box_point_far`, both views perturbed by `σ` px (no draw at
    σ = 0), then `n_outliers` pairs uniform in `[100, 540] × [50, 430]` and
    `[80, 520] × [60, 440]`. `source`/`target` are the pixels (inliers first; draws behind view
    2 are skipped, so `n_inliers` is the number kept), `F` the true fundamental
    matrix `K₂⁻ᵀ [t]× R K₁⁻¹`. `n_outliers = 0, σ = 0` is the exact scene (the
    minimal-solver tests). Every draw is the caller's `rng`'s, in this order.
    """
    function fundmat_scene(rng::AbstractRNG; n_inliers = 100, n_outliers = 30, σ = 0.5)
        cam₁, cam₂ = _FUNDMAT_CAMERAS; R, t = _FUNDMAT_POSE
        source, target = stereo_pixel_pairs(
            rng, cam₁, cam₂, R, t; n = n_inliers, σ₁ = σ, σ₂ = σ, point = box_point_far, n_out = n_outliers,
            outlier = rng -> (SA[100.0 + 440.0 * rand(rng), 50.0 + 380.0 * rand(rng)], SA[80.0 + 440.0 * rand(rng), 60.0 + 380.0 * rand(rng)])
        )
        F = FundMat(EssentialMat(R, t), calibration_matrix(cam₁), calibration_matrix(cam₂))
        return (; source, target, F, n_inliers = length(source) - n_outliers)
    end

    # The truncated least-squares cost (MSAC at the squared width `c2`) of F's
    # Sampson distances over the pairs `u₁[i] => u₂[i]`.
    truncated_sampson_cost(F, u₁, u₂; c2 = 1.0e-4) =
        cost(TLSShape(), [only(sampson_residual(F, u₁[i], u₂[i])) for i in eachindex(u₁)], c2, length(u₁))

    # The pixel camera the calibrated scenes are moved into. A realistic focal
    # length keeps the PIXEL tolerances (`COINCIDENT_PX` in the sample gate, an
    # outlier half-width) meaningful; an identity camera (f = 1) makes one "pixel"
    # one calibrated unit, and sample admission collapses from 100% to 3.92%.
    # (The identity camera is `IDCAM`, Cameras.)
    const FPX = 800.0
    const CX, CY = 320.0, 240.0
    const ECAM = CameraModel(FPX * px, SA[CX * px, CY * px])

    # A calibrated point imaged by ECAM.
    topx(u) = Point2{Float64}(project(ECAM, SA[u[1], u[2], 1.0]))

    essential_problem(camera, a, b) = EssentialProblem(camera, camera, correspondences(a, b))
    # Calibrated pairs as ECAM's pixels.
    ecs(a, b) = essential_problem(ECAM, topx.(a), topx.(b))
    end

    # -----------------------------------------------------------------------------
    # The plane-induced homography scene
    # -----------------------------------------------------------------------------
    module HomographyScenes
    using Random: AbstractRNG
    using LinearAlgebra: norm
    using StaticArrays: SA, SMatrix, SVector
    using VisualGeometryToolkitCore: HomographyMap, HomographyMat, PerspectiveMap

    export H_TRUE, H_PX, gaussian_source, uniform_source, homography_warp, homography_pairs, max_transfer_error

    # The ground-truth homography, unit Frobenius norm, h₃₃ > 0.
    const H_TRUE = let H = SMatrix{3, 3, Float64, 9}(0.95, 0.12, 1.0e-4, -0.1, 0.93, 2.0e-4, 15.0, -8.0, 1.0)
        H = H / norm(H)
        HomographyMat{Float64}(Tuple(H[3, 3] < 0 ? -H : H))
    end

    # A pixel-scale homography (h₃₃ = 1), and the Gaussian point cloud about
    # (300, 300) its scenes draw from.
    const H_PX = HomographyMat{Float64}(Tuple(SA[1.2 -0.1 30.0; 0.05 1.1 -20.0; 1.0e-4 2.0e-4 1.0]))
    gaussian_source(rng::AbstractRNG) = SVector{2, Float64}(randn(rng, 2) .* 200 .+ 300)

    # The point map of a homography: x ↦ π(H·x̃).
    homography_warp(H) = PerspectiveMap() ∘ HomographyMap(H)

    # How far apart two homographies put the same points: the largest distance
    # between their transfers of `points` (a region's corners, a few probes).
    max_transfer_error(H₁, H₂, points) =
        maximum(norm(homography_warp(H₁)(p) - homography_warp(H₂)(p)) for p in points)

    # A source point uniform in [100, 900] × [100, 700] (x drawn first, then y).
    uniform_source(rng) = SA[100.0 + 800.0 * rand(rng), 100.0 + 600.0 * rand(rng)]

    """
        homography_pairs(rng, H; n_in, n_out = 0, σ = 0, source, target = source) -> (src, dst)

    `n_in` pairs `s => H(s)` with `s = source(rng)` (uniform in `[100, 900] ×
    [100, 700]` by default), the target perturbed by `σ` per coordinate (no draw at
    σ = 0); then `n_out` unrelated pairs `source(rng) => target(rng)`.
    """
    function homography_pairs(rng::AbstractRNG, H; n_in, n_out = 0, σ = 0.0, source = uniform_source, target = source)
        warp = homography_warp(H)
        src = SVector{2, Float64}[]; dst = SVector{2, Float64}[]
        for _ in 1:n_in
            s = SVector{2, Float64}(source(rng))
            d = SVector{2, Float64}(warp(s))
            σ > 0 && (d += σ * SA[randn(rng), randn(rng)])
            push!(src, s); push!(dst, d)
        end
        for _ in 1:n_out
            push!(src, SVector{2, Float64}(source(rng))); push!(dst, SVector{2, Float64}(target(rng)))
        end
        return src, dst
    end
    end

    # -----------------------------------------------------------------------------
    # Rigid poses: random draws, perturbations in the decoupled chart, and the
    # PnP scene's frustum points.
    # -----------------------------------------------------------------------------
    module PoseScenes
    using Random: AbstractRNG
    using LinearAlgebra: norm, normalize
    using StaticArrays: SVector
    using Rotations: AngleAxis, RotMatrix
    using VisualGeometryToolkitCore: EuclideanMap, Point3, RigidExpManifold6, Vec3, retract, rotation, log_so3,
        translation, unproject, project
    using ..Contamination: contaminated_pairs

    export rigid, rand_pose, retract_pose, perturb_pose, frustum_point, rotation_error_deg, translation_error,
        marker_pose, robot_world_hand_eye_scene, lens_board_scene

    """
        lens_board_scene(rng, camera, R, t; σ = 0.3, n_out = 40) -> (src, dst, is_inlier)

    A 320 × 240 board (a 17 × 13 grid of 20-unit pitch on its `z = 0` plane, centred)
    imaged by `camera` at the pose `x_cam = R·X + t` in a 4240 × 2832 frame: every
    board point with a finite pixel, its pixel perturbed by `σ`, then `n_out` gross
    outliers (a uniform board point with a uniform pixel of the frame), shuffled
    (Contamination's `contaminated_pairs`). The absolute-pose scene through a
    distorting lens.
    """
    function lens_board_scene(rng::AbstractRNG, camera, R, t; σ = 0.3, n_out = 40)
        board = [Point3{Float64}(X, Y, 0.0) for X in -160.0:20.0:160.0 for Y in -120.0:20.0:120.0]
        inliers = [Xw => SVector{2, Float64}(project(camera, R * SVector(Xw) + t)) for Xw in board]
        pairs, is_inlier = contaminated_pairs(
            rng, filter(c -> all(isfinite, c.second), inliers); σ, n_out, shuffle = true,
            outlier = (rng, _) -> Point3{Float64}(320 * rand(rng) - 160, 240 * rand(rng) - 120, 0.0) => SVector(4240 * rand(rng), 2832 * rand(rng))
        )
        return first.(pairs), last.(pairs), is_inlier
    end

    # A rigid map from a rotation and a translation, taken as given (no
    # orthonormality check: a caller's drifted rotation stays drifted).
    rigid(R, t) = EuclideanMap(RotMatrix{3, Float64}(R), Vec3{Float64}(t[1], t[2], t[3]); check = false)

    # A random pose: a rotation by a uniform angle in [0, ang_max) about a
    # uniformly random axis, and a Gaussian translation of scale `t_scale`.
    rand_pose(rng::AbstractRNG; ang_max = π, t_scale = 1.0) =
        rigid(AngleAxis(ang_max * rand(rng), normalize(randn(rng, 3))...), t_scale .* randn(rng, 3))

    # `p` moved by `v = (δω, δt)` in the decoupled chart (rotation on the left
    # through the exponential, translation free), as a `rigid` map.
    retract_pose(p, v) = (q = retract(RigidExpManifold6(p), SVector{6, Float64}(Tuple(v))); rigid(q.R, q.t))

    # `p` perturbed in that chart: δω ~ N(0, σ_R²I₃), δt ~ N(0, σ_t²I₃).
    function perturb_pose(rng::AbstractRNG, p; σ_R, σ_t)
        δω = σ_R .* randn(rng, 3); δt = σ_t .* randn(rng, 3)
        return retract_pose(p, vcat(δω, δt))
    end

    # A pose's errors against the truth: the rotation angle (degrees) and the
    # translation distance.
    rotation_error_deg(p, R) = rad2deg(norm(log_so3(rotation(p)' * R)))
    translation_error(p, t) = norm(translation(p) - t)

    # The robot-world constraint A·X = Y·B solved for the observation: the marker
    # pose B the camera sees in the frame whose rig pose is A. (Hand-eye is Y = X.)
    marker_pose(Y, A, X) = inv(Y) ∘ A ∘ X

    """
        robot_world_hand_eye_scene(rng; N, M = 1, Y = (;), X = (;), A = (;)) -> (; Y, X, A)

    The unknowns and rig poses of a robot-world scene `A_i·X_m = Y·B_i^m`: `Y`,
    then the `M` markers `X`, then the `N` rig poses `A`, each a
    `rand_pose(rng; kw...)` with that argument's keywords. Frame i's view of marker
    m is `marker_pose(Y, A[i], X[m])`; noise, dropouts and outliers are the item's.
    """
    function robot_world_hand_eye_scene(rng::AbstractRNG; N, M = 1, Y = (;), X = (;), A = (;))
        Yp = rand_pose(rng; Y...)
        Xp = [rand_pose(rng; X...) for _ in 1:M]
        Ap = [rand_pose(rng; A...) for _ in 1:N]
        return (; Y = Yp, X = Xp, A = Ap)
    end

    """
        frustum_point(rng, camera, R, t; x, y, depth) -> (X, u)

    A world point `X` imaged at pixel `u` by the camera at pose `x_cam = R·X + t`:
    `u` uniform in `x × y`, the depth along its viewing ray uniform in `depth`.
    """
    function frustum_point(rng::AbstractRNG, camera, R, t; x = (40.0, 920.0), y = (40.0, 680.0), depth = (4.0, 10.0))
        u = SVector(x[1] + (x[2] - x[1]) * rand(rng), y[1] + (y[2] - y[1]) * rand(rng))
        d = depth[1] + (depth[2] - depth[1]) * rand(rng)
        Xc = SVector{3, Float64}(unproject(camera, u, d))
        return Point3{Float64}(R' * (Xc - t)), u
    end
    end

    # -----------------------------------------------------------------------------
    # Boards under a division lens (the radial-camera and focal-stack scenes)
    # -----------------------------------------------------------------------------
    module DivisionBoards
    using Random: AbstractRNG
    using LinearAlgebra: norm
    using StaticArrays: SVector
    using Rotations: RotMatrix
    using Rotations: RotXYZ
    using VisualGeometryToolkitCore: PosedCamera, CameraModel, Division, EuclideanMap, RadialCamera, project
    using ..Contamination: contaminated_pairs

    export F0, K0, BOARD_ANGLES, BOARD_TS, BOARD_GRID, board_camera, board_pixels, noisy_board_views,
        RADIAL_BOARD_FRAME, RADIAL_BOARD_CENTER, radial_board

    # The focal length (px) and the division coefficient on the CENTRED PIXEL radius
    # (ρᵤ = ρ_d / (1 + k·ρ_d²)); the lens stores it on the normalized radius, k·f².
    const F0, K0 = 1226.3, -6.4e-7
    # The boards' orientations (RotXYZ angles).
    const BOARD_ANGLES = [(0.4, -0.3, 0.2), (-0.5, 0.2, -0.1), (0.2, 0.5, 0.3), (-0.3, -0.4, 0.1), (0.6, 0.1, -0.2), (0.1, -0.6, 0.4)]
    # Their translations: boards at very different depths (350 to 3000), so the
    # algebraic rows of the far ones dominate an unwhitened focal stack.
    const BOARD_TS = [
        SVector(30.0, -20.0, 400.0), SVector(-40.0, 10.0, 2500.0), SVector(20.0, 30.0, 350.0),
        SVector(-10.0, -30.0, 3000.0), SVector(50.0, 20.0, 600.0), SVector(0.0, 40.0, 2200.0),
    ]
    # A 5 × 5 board grid in its own z = 0 plane.
    const BOARD_GRID = -80.0:40.0:80.0

    """
        board_camera(R, t; f = F0, ks = (K0,), center = (0, 0)) -> PosedCamera

    The camera viewing a board's `z = 0` plane at `x_cam = R·X + t`: focal `f`,
    principal point `center`, and a division lens whose pixel-radius coefficients
    are `ks` (`k_p` on `ρ^{2p}`).
    """
    board_camera(R, t; f = F0, ks = (K0,), center = (0.0, 0.0)) = PosedCamera(
        CameraModel(; focal = f, pp = (Float64(center[1]), Float64(center[2])),
            lens = Division(ntuple(p -> ks[p] * f^(2p), length(ks))...)),   # a Tuple of the lens's fixed order
        EuclideanMap(RotMatrix{3, Float64}(R), SVector{3, Float64}(t))
    )

    # The board's plane points and their pixels, noise free (X outer, Y inner).
    board_pixels(camera; grid = BOARD_GRID) = Pair{SVector{2, Float64}, SVector{2, Float64}}[
        SVector(X, Y) => SVector{2, Float64}(project(camera, SVector(X, Y, 0.0))) for X in grid for Y in grid
    ]

    """
        noisy_board_views(rng, cameras; σ, ε = 0, ν0 = Inf, ρ = 0) -> (boards, inlier)

    Each camera's `board_pixels`, every pixel perturbed by `σ·ξ` — ξ standard
    Gaussian, or a multivariate Student-t of `ν0` dof — then, with probability `ε`,
    a 30 px Gaussian gross outlier, and with probability `ρ` a RAY-ALIGNED mismatch
    (moved ±U[5, 100] px along its own radial line: what a radial camera's 1-D
    line test cannot reject). `inlier` is the true mask, in board order.
    """
    function noisy_board_views(rng::AbstractRNG, cameras; σ, ε = 0.0, ν0 = Inf, ρ = 0.0)
        inlier = Bool[]
        boards = map(cameras) do cam
            pts = Pair{SVector{2, Float64}, SVector{2, Float64}}[]
            for (X, d0) in board_pixels(cam)
                ξ = randn(rng, SVector{2, Float64})
                isfinite(ν0) && (ξ *= sqrt(ν0 / sum(abs2, randn(rng, Int(ν0)))))
                d = d0 + σ * ξ
                out = rand(rng) < ε
                out && (d += 30.0 * randn(rng, SVector{2, Float64}))
                if ρ > 0 && rand(rng) < ρ
                    out = true
                    d += (rand(rng, Bool) ? 1 : -1) * (5 + 95 * rand(rng)) * d0 / max(norm(d0), 1.0e-9)
                end
                push!(pts, X => d); push!(inlier, !out)
            end
            pts
        end
        return boards, inlier
    end

    # The radial camera's frame (4000 × 3000 px) and its centre, about which the
    # lens distorts.
    const RADIAL_BOARD_FRAME = (4000.0, 3000.0)
    const RADIAL_BOARD_CENTER = SVector(RADIAL_BOARD_FRAME[1] / 2, RADIAL_BOARD_FRAME[2] / 2)

    """
        radial_board(rng; f = 900, k = -2e-7, n_out = 20, σ = 0) -> (radial_camera, pairs, is_inlier)

    One board (a 9 × 9 grid of 25-unit pitch, `board_pixels`) seen through a
    division lens of pixel-radius coefficient `k` about `RADIAL_BOARD_CENTER` at a
    fixed pose (focal `f`), its pixels RAW: perturbed by `σ`, then `n_out` gross
    outliers (a uniform board point with a uniform pixel of the frame), shuffled.
    `radial_camera` is the truth, `RadialCamera(camera)`: invariant to the focal
    length and the distortion about the centre.
    """
    function radial_board(rng::AbstractRNG; f = 900.0, k = -2.0e-7, n_out = 20, σ = 0.0)
        cam = board_camera(RotXYZ(0.35, -0.25, 0.15), SVector(40.0, -30.0, 600.0); f, ks = (k,), center = RADIAL_BOARD_CENTER)
        W, H = RADIAL_BOARD_FRAME
        pairs, is_inlier = contaminated_pairs(
            rng, board_pixels(cam; grid = -100.0:25.0:100.0); σ, n_out, shuffle = true,
            outlier = (rng, _) -> SVector(200 * rand(rng) - 100, 200 * rand(rng) - 100) => SVector(W * rand(rng), H * rand(rng))
        )
        return RadialCamera(cam), pairs, is_inlier
    end
    end

    # -----------------------------------------------------------------------------
    # The radially distorted homography: a one-sided division model about the
    # middle of a 1200 × 900 frame (VisualGeometryToolkitCore's `RadialHomography`,
    # ground truth through its own `radial_transfer`).
    # -----------------------------------------------------------------------------
    module RadialScenes
    using Random: AbstractRNG
    using StaticArrays: SMatrix, SVector
    using VisualGeometryToolkitCore: HomographyMat, RadialHomography, radial_transfer
    using ..Contamination: contaminated_pairs

    export RADIAL_HOMOGRAPHY_FRAME, radial_homography_truth, radial_homography_pairs

    # The target frame (px); the distortion centre is its middle.
    const RADIAL_HOMOGRAPHY_FRAME = (1200.0, 900.0)

    """
        radial_homography_truth(k_norm) -> RadialHomography{Float64}

    The ground-truth model: a fixed pixel-scale homography and the division
    coefficient `k_norm` on the radius normalized by the frame's diagonal, about
    the frame's middle.
    """
    function radial_homography_truth(k_norm)
        W, H = RADIAL_HOMOGRAPHY_FRAME
        H0 = SMatrix{3, 3, Float64}(4.0, -0.2, 5.0e-5, 0.3, 4.2, -4.0e-5, 260.0, 200.0, 1.0)
        return RadialHomography{Float64}(HomographyMat{Float64}(Tuple(H0)), k_norm / hypot(W, H)^2, SVector(W / 2, H / 2))
    end

    """
        radial_homography_pairs(rng, model; n_in = 70, n_out = 30, σ = 0) -> (pairs, is_inlier)

    `n_in` sources uniform in `[0, 200] × [0, 150]`, each redrawn until its
    `radial_transfer` lands inside the frame, the target perturbed by `σ`; then
    `n_out` unrelated pairs (a uniform source with a uniform target of the frame),
    shuffled.
    """
    function radial_homography_pairs(rng::AbstractRNG, model; n_in = 70, n_out = 30, σ = 0.0)
        W, H = RADIAL_HOMOGRAPHY_FRAME
        function inlier(rng, _)
            while true
                s = SVector(200 * rand(rng), 150 * rand(rng))
                d = radial_transfer(model, s)
                d === nothing && continue
                (0 < d[1] < W && 0 < d[2] < H) && return s => SVector(d[1], d[2])
            end
        end
        return contaminated_pairs(
            rng, inlier, n_in; σ, n_out, shuffle = true,
            outlier = (rng, _) -> SVector(200 * rand(rng), 150 * rand(rng)) => SVector(W * rand(rng), H * rand(rng))
        )
    end
    end

    # -----------------------------------------------------------------------------
    # The calibrated P3P scene: points in front of a fixed pose, observed by a
    # camera (the identity camera `IDCAM` makes them calibrated coordinates), with
    # per-row √weights.
    # -----------------------------------------------------------------------------
    module P3PScenes
    using Random: AbstractRNG
    using StaticArrays: SA
    using Rotations: RotMatrix, MRP
    using VisualGeometryToolkitCore: Point2, Point3, Vec3, project
    using VisualGeometryToolkitEstimation: AbsolutePoseProblem

    export p3p_scene

    """
        p3p_scene(rng, camera, n; σ) -> (R₀, t₀, problem, sqrtw)

    `n` points seen from the pose `(R₀, t₀)` by `camera`, perturbed by `σ` (in its
    pixels); `sqrtw` in the family's per-ROW convention (length 2n, a point's
    weight on both its rows).
    """
    function p3p_scene(rng::AbstractRNG, camera, n; σ)
        R₀ = RotMatrix(MRP(0.1, -0.05, 0.2))
        t₀ = Vec3{Float64}(0.3, -0.2, 0.5)
        Xs = [Point3{Float64}(randn(rng), randn(rng), 3 + rand(rng)) for _ in 1:n]
        obs = [Point2{Float64}(project(camera, R₀ * X + t₀) + σ * SA[randn(rng), randn(rng)]) for X in Xs]
        sqrtw = Vector{Float64}(undef, 2n)
        for i in 1:n
            w = sqrt(rand(rng))
            sqrtw[2i - 1] = w; sqrtw[2i] = w
        end
        return R₀, t₀, AbsolutePoseProblem(camera, [Xs[i] => obs[i] for i in 1:n]), sqrtw
    end
    end

    # -----------------------------------------------------------------------------
    # Cameras more than one item states its problems on.
    # -----------------------------------------------------------------------------
    module Cameras
    using StaticArrays: SA
    using VisualGeometryToolkitCore: CameraModel, Pinhole, Point2, backproject, project, px

    export IDCAM, calibrated_point

    # The identity camera: its "pixels" ARE calibrated units (f = 1, principal
    # point at the origin).
    const IDCAM = CameraModel(1.0px, SA[0.0px, 0.0px])

    # A camera's pixel as a calibrated point: its bearing on the z = 1 plane.
    calibrated_point(camera, u) = Point2{Float64}(project(Pinhole(), backproject(camera, u)))
    end

end # module TestScenes

VisualGeometryToolkitEstimation.synthetic_scenes() = TestScenes

end # module VisualGeometryToolkitEstimationTestExt
