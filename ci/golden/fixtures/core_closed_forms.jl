# Fixtures of Core's closed-form estimators: the data of VisualGeometryToolkitCore's own tests (input_guards_tests'
# and solver_layer_tests' epipolar scenes, rotation_fits_tests, spectral_rotation_averaging_tests,
# zhang_intrinsics_tests, tls_line_tests, direct_ellipse_tests, nearest_point_tests, bearing_triangulation_tests,
# sample_p3p_tests), PoseLib's test_refine scene (the essential DLT) and the shared synthetic scenes' fundmat_scene.
# A test's noise draw is redrawn from StableRNG(seed), seeds 1–3, in the test's own draw order.
using Test, StableRNGs, StaticArrays, StructArrays, Random, LinearAlgebra
using Rotations: Rotations, RotMatrix, RotXYZ, RotationVec, QuatRotation, AngleAxis
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitCore: Correspondences, withattrs, HomographyMat, Size2, Attributed, μm, px, mm, EuclideanMap,
    sensor, focal_length, centroid, Rect, exp_so3, CameraModel, LogicalIntrinsics, CameraCalibrationMat, backproject

const _EPS = VgtkEstimation.synthetic_scenes().EpipolarScenes

# --- the fundamental matrix ---------------------------------------------------------------------------------------

# input_guards_tests: nine points of a pinhole pair, K = (500, 500, 320, 240), view 2 turned 0.1 rad about y.
function fundmat_guard_pairs(rng)
    K = SMatrix{3, 3}(500.0, 0, 0, 0, 500.0, 0, 320, 240, 1)
    c, s = cos(0.1), sin(0.1)
    R = SMatrix{3, 3}(c, 0, -s, 0, 1, 0, s, 0, c)
    t = SVector(1.0, 0.2, 0.1)
    Xs = [SVector(randn(rng), randn(rng), 5 + rand(rng)) for _ in 1:9]
    return [PerspectiveMap()(K * X) => PerspectiveMap()(K * (R * X + t)) for X in Xs]
end

# solver_layer_tests' two-view scene: a Haar rotation, t = (0.4, −0.2, 0.1), 20 points; the normalized pixels, the
# bearings, and view 2's pixels with 1e-3 noise (its weights testset).
function epipolar_layer_scene(rng)
    R = SMatrix{3, 3, Float64}(rand(rng, RotMatrix{3, Float64}))
    t = SVector(0.4, -0.2, 0.1)
    X3 = [SVector(2randn(rng), 2randn(rng), 6.0 + 2rand(rng)) for _ in 1:20]
    x1b = [normalize(Xi) for Xi in X3]
    x2b = [normalize(R * Xi + t) for Xi in X3]
    u1 = [Point2{Float64}(PerspectiveMap()(Xi)) for Xi in X3]
    u2 = [Point2{Float64}((R * Xi + t)[1] / (R * Xi + t)[3], (R * Xi + t)[2] / (R * Xi + t)[3]) for Xi in X3]
    u2n = [p + Point2{Float64}(1.0e-3randn(rng), 1.0e-3randn(rng)) for p in u2]
    return (; u1, u2, u2n, x1b, x2b)
end

# The shared two-camera F scene (100 inliers at σ = 0.5 px, 30 outliers).
fundmat_scene_pairs(rng) = let s = _EPS.fundmat_scene(rng); Correspondences(s.source, s.target) end

# --- the essential matrix (PoseLib's test_refine scene) -----------------------------------------------------------

function essential_scene(rng)
    R = SMatrix{3, 3, Float64, 9}(RotMatrix(AngleAxis(0.2, normalize([0.3, 1.0, 0.1])...)))
    t = normalize(SVector(1.0, 0.1, 0.05))
    X = [SVector(4 * randn(rng), 3 * randn(rng), 8.0 + 4 * rand(rng)) for _ in 1:200]
    b1 = [normalize(Y) for Y in X]
    b2 = [normalize(R * Y + t) for Y in X]
    b2n = [normalize(b + 1.0e-3 * randn(rng, SVector{3, Float64})) for b in b2]
    c1 = [SVector(Y[1] / Y[3], Y[2] / Y[3]) for Y in X]                       # the z = 1 chart: refused
    c2 = [(Z = R * Y + t; SVector(Z[1] / Z[3], Z[2] / Z[3])) for Y in X]
    return (; b1, b2, b2n, c1, c2)
end

# --- rotations ---------------------------------------------------------------------------------------------------

# rotation_fits_tests' kabsch recovery: 40 unit directions, their images under a Haar rotation, noisy images.
function kabsch_scene(rng)
    Rgt = rand(rng, RotMatrix{3, Float64})
    as = [normalize(randn(rng, SVector{3, Float64})) for _ in 1:40]
    bs = [Rgt * a for a in as]
    bsn = [normalize(b + 1.0e-3 * randn(rng, SVector{3, Float64})) for b in bs]
    return (; as, bs, bsn)
end

# rotation_fits_tests' weighted kabsch: 30 directions, noisy images, weights in [0.1, 1.1).
function kabsch_weighted_scene(rng)
    Rt = exp_so3(SVector(0.2, 0.5, -0.3))
    as = [normalize(SVector(randn(rng), randn(rng), randn(rng))) for _ in 1:30]
    bs = [normalize(Rt * a + 0.01 * SVector(randn(rng), randn(rng), randn(rng))) for a in as]
    w = rand(rng, 30) .+ 0.1
    return (; as, bs, w)
end

# rotation_fits_tests' chordal mean: four symmetric perturbations of a Haar rotation, and five noisy ones.
function chordal_scene(rng)
    Rc = rand(rng, RotMatrix{3, Float64})
    Rs = [RotMatrix(RotationVec((s * 0.05 .* SVector(1.0, -0.5, 0.3))...)) * Rc for s in (-2, -1, 1, 2)]
    R1 = rand(rng, RotMatrix{3, Float64})
    noisy = [RotMatrix(RotationVec((0.1 .* randn(rng, SVector{3, Float64}))...)) * R1 for _ in 1:5]
    return (; Rc, Rs, noisy)
end

# spectral_rotation_averaging_tests: six frames (the first the identity), their complete relative set, a chain, a
# noisy set, a corrupted pair weighted 0.
function rotation_averaging_scene(rng)
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:6)
    rel(i, j) = R[j]' * R[i]
    pairs = Dict((i, j) => rel(i, j) for i in 1:6 for j in (i + 1):6)
    chain = Dict((k, k + 1) => rel(k, k + 1) for k in 1:5)
    noisy = Dict(p => RotMatrix(RotationVec((0.01 .* randn(rng, 3))...)) * Rp for (p, Rp) in pairs)
    bad = copy(pairs); bad[(1, 2)] = RotMatrix(RotationVec(1.0, 0.0, 0.0)) * pairs[(1, 2)]
    weighted = Dict(p => Attributed(Rp, (; weight = (p == (1, 2) ? 0 : 1))) for (p, Rp) in bad)
    mixed = Dict{Tuple{Int, Int}, Any}(p => (p == (1, 2) ? Attributed(Rp, (; weight = 0.0)) : Rp) for (p, Rp) in bad)
    chain3 = Dict((k, k + 1) => rel(k, k + 1) for k in 1:3)
    negative = Dict(p => Attributed(Rp, (; weight = -1.0)) for (p, Rp) in chain3)
    return (; pairs, chain, noisy, bad, weighted, mixed, negative)
end

# --- the calibration from plane homographies (zhang_intrinsics_tests) ---------------------------------------------

const ZHANG_CENTER = SVector(640.0, 480.0)
_zhang_hom(K, R, t) = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(K * hcat(R[:, 1], R[:, 2], t))))
function zhang_views(fx, fy)
    c = ZHANG_CENTER
    K = @SMatrix [fx 0.0 c[1]; 0.0 fy c[2]; 0.0 0.0 1.0]
    angles = [(0.4, -0.3, 0.2), (-0.5, 0.2, -0.1), (0.2, 0.5, 0.3), (-0.3, -0.4, 0.1), (0.6, 0.1, -0.2), (0.1, -0.6, 0.4)]
    ts = [SVector(0.3, -0.2, 2.5), SVector(-0.4, 0.1, 3.0), SVector(0.2, 0.3, 2.2), SVector(-0.1, -0.3, 2.8), SVector(0.5, 0.2, 3.5),
          SVector(0.0, 0.4, 2.0)]
    return [_zhang_hom(K, RotXYZ(a...), t) for (a, t) in zip(angles, ts)]
end
function zhang_fixtures()
    fx, fy = 1226.3, 1219.7
    Hs = zhang_views(fx, fy)
    junk = _zhang_hom((@SMatrix [500.0 0 0; 0 900.0 0; 0 0 1.0]), RotXYZ(0.3, 0.2, 0.1), SVector(0.1, 0.1, 2.0))
    voted = [[Attributed(H, (; weight = 10)) for H in Hs]; Attributed(junk, (; weight = 0))]
    mixed = [Hs[1:3]; [Attributed(H, (; weight = 1.0)) for H in Hs[4:end]]]
    K = @SMatrix [fx 0.0 ZHANG_CENTER[1]; 0.0 fy ZHANG_CENTER[2]; 0.0 0.0 1.0]
    flat = [_zhang_hom(K, RotXYZ(0.0, 0.0, ψ), SVector(0.1ψ, -0.2ψ, 2.0 + ψ)) for ψ in (0.0, 0.3, -0.4, 0.7)]
    τ = 3.9 / 4.0
    return (; Hs, voted, mixed, flat, Hs_τ = zhang_views(1300.0, τ * 1300.0),
            pitch1 = Size2(width = 1.0μm / px, height = 1.0μm / px), pitch = Size2(width = 3.9μm / px, height = 4.0μm / px),
            negative = [Attributed(Hs[1], (; weight = -1.0)); Hs[2:end]], infinite = [Attributed(Hs[1], (; weight = Inf)); Hs[2:end]])
end

# --- the total-least-squares line (tls_line_tests) ----------------------------------------------------------------

tls_exact_points() = [Point2{Float64}(Float64(i), 2.0i + 1.0) for i in 1:12]
tls_noisy_points(rng, n) = [Point2{Float64}(Float64(i), 2.0i + 1.0 + 0.05 * randn(rng)) for i in 1:n]

# --- the direct ellipse fit (direct_ellipse_tests) ----------------------------------------------------------------

const ELLIPSE_E0 = Ellipse(Point2(120.0, 80.0), 30.0, 12.0, 0.6)
ellipse_points(e, n; σ = 0.0, arc = 2π, rng = nothing) =
    [boundary_point(e, φ) + (σ == 0 ? zero(SVector{2, Float64}) : σ * randn(rng, SVector{2})) for φ in range(0, arc, n + 1)[1:(end - 1)]]

# --- the point nearest to lines (nearest_point_tests: one rng through the file's testsets, in their order) --------

function nearest_point_scene(rng)
    X = SVector(0.3, -1.2, 4.5)
    dirs = [normalize(@SVector(randn(rng, 3))) for _ in 1:5]
    ts = randn(rng, 5)
    meet = [(X + ts[i] * dirs[i]) => 3.7 * dirs[i] for i in 1:5]
    noisy = map(1:40) do _
        d = normalize(@SVector(randn(rng, 3)))
        (X + 2randn(rng) * d + 0.01 * @SVector(randn(rng, 3))) => d
    end
    shifted = [(X + randn(rng) * normalize(d) + 0.05 * @SVector(randn(rng, 3))) => d for d in (@SVector(randn(rng, 3)) for _ in 1:12)]
    udirs = [normalize(@SVector(randn(rng, 3))) for _ in 1:4]
    unitful = [((X + randn(rng) * udirs[i]) .* mm) => udirs[i] for i in 1:4]
    return (; meet, noisy, shifted, unitful)
end
nearest_point_two() = correspondences([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)], [SVector(0.0, 0.0, 1.0), SVector(-1.0, 0.0, 1.0)])
nearest_point_2d() = [SVector(0.0, 1.0) => SVector(1.0, 0.0), SVector(2.0, 0.0) => SVector(0.0, 1.0), SVector(0.0, -1.0) => SVector(1.0, 1.0)]
nearest_point_weighted() = [SVector(0.0, 0.0, 0.0) => SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0) => SVector(0.0, 0.0, 1.0),
    SVector(0.0, 0.0, 0.0) => SVector(0.0, 1.0, 0.0), SVector(5.0, 5.0, 5.0) => SVector(1.0, 1.0, 0.0)]
const NP_D = SVector(0.0, 0.6, 0.8)
nearest_point_degenerate() = (
    parallel = [SVector(0.0, 0.0, 0.0) => NP_D, SVector(1.0, 0.0, 0.0) => NP_D, SVector(0.0, 2.0, 0.0) => -2NP_D],
    zero_direction = [SVector(0.0, 0.0, 0.0) => NP_D, SVector(1.0, 0.0, 0.0) => zero(NP_D)],
    one_line = [SVector(0.0, 0.0, 0.0) => NP_D],
)

# --- ray triangulation (bearing_triangulation_tests) --------------------------------------------------------------

const TRI_RM = SMatrix{3, 3}(RotMatrix(RotXYZ(0.03, -0.05, 0.02)))
const TRI_T = SVector(0.8, 0.05, -0.02)
triangulation_pose() = EuclideanMap(RotMatrix{3, Float64}(TRI_RM), TRI_T)
# 200 draws of a point, kept when in front of view 2; the bearing pairs, exact, and with 1e-3 bearing noise (the
# rays then miss each other: the midpoint of their common perpendicular).
function triangulation_rays(rng)
    exact = Tuple{SVector{3, Float64}, SVector{3, Float64}}[]
    for _ in 1:200
        X = SVector(2.0 * randn(rng), 2.0 * randn(rng), 4 + 2rand(rng))
        X2 = TRI_RM * X + TRI_T
        X2[3] > 0.2 || continue
        push!(exact, (normalize(X), normalize(X2)))
    end
    noisy = [(normalize(a + 1.0e-3 * randn(rng, SVector{3, Float64})), normalize(b + 1.0e-3 * randn(rng, SVector{3, Float64}))) for (a, b) in exact]
    return (; exact, noisy)
end
function triangulation_special()
    X = SVector(0.3, 0.1, -3.0)                                       # behind view 1
    behind = (normalize(X), normalize(TRI_RM * X + TRI_T))
    f = normalize(SVector(0.1, -0.05, 1.0))                          # zero parallax: a pure translation along the ray
    parallel = (EuclideanMap(RotMatrix{3, Float64}(SMatrix{3, 3, Float64}(I)), 0.5 * f), f, f)
    fl, c = 800.0, 320.0                                              # an equidistant fisheye's ray at 100°
    fish = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(fl, 0, 0, 0, fl, 0, c, c, 1))), Lens(Equidistant()))
    b = backproject(fish, SVector(c + 1.75 * fl, c))
    f₁ = SVector(b[1], b[2], b[3])
    wide = (f₁, normalize(TRI_RM * (5.0 * f₁) + TRI_T))
    return (; behind, parallel, wide)
end

# --- P3P through a camera model (sample_p3p_tests' camera and world points) ---------------------------------------

function p3p_camera()
    s = sensor("Sony", "IMX174")
    bounds = Rect(s)
    return CameraModel(focal_length(s, deg2rad(40.0); dimension = :horizontal), s.pitch, centroid(bounds) .* px), bounds
end
# The test's 40 plane points, then `n` draws of sample_p3p's configuration: 3 distinct world points of them and 3
# image points uniform in the sensor bounds, as `correspondences(X3, u3)`.
function p3p_draws(rng, bounds, n)
    X = [Point3(200 * randn(rng), 200 * randn(rng), 0.0) for _ in 1:40]
    o, w = VgtkCore.GeometryBasics.origin(bounds), VgtkCore.GeometryBasics.widths(bounds)
    upt() = Point2{Float64}(o .+ rand(rng, SVector{2, Float64}) .* w)
    return map(1:n) do _
        k = randperm(rng, length(X))[1:3]
        X3 = SVector{3}(X[k[1]], X[k[2]], X[k[3]])
        u3 = SVector{3}(upt(), upt(), upt())
        correspondences(X3, u3)
    end
end
