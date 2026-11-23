# Fixtures of Core's minimal solvers, `solve(solver, cs, idx)`: each solver's own test data (circle_tests,
# ellipse_tests, rotation2pt_tests, solver_layer_tests, essential_5pt_native_tests, p3p_native_tests,
# homography_5pt_radial_tests, lens_contract_tests) and a contaminated scene per solver from the shared synthetic
# scenes and the Estimation suite's testsetups (ConicScenes, RotationScene, EpipolarScenes, FindScenes,
# RadialCameraScene), each with a few hundred random index samples drawn as data. Scene noise from StableRNG(seed),
# seeds 1–3; the samples from StableRNG(100 + seed).
using Test, StableRNGs, StaticArrays, StructArrays, Random, LinearAlgebra
using Rotations: Rotations, RotMatrix, RotXYZ, QuatRotation
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitCore: Correspondences, HomographyMat, RadialHomography, radial_transfer, CameraModel, KannalaBrandt,
    backproject, backproject_with_jacobian, project, px

const _MS = VgtkEstimation.synthetic_scenes()

# `n` index samples of `k` distinct indices out of `1:N`.
index_samples(rng, N, k, n) = [randperm(rng, N)[1:k] for _ in 1:n]

# --- circles and ellipses (ConicScenes; circle_tests / ellipse_tests) ---------------------------------------------

Base.include(@__MODULE__, joinpath(@__DIR__, "common.jl"))   # ConicScenes' `circle_scene`, `ellipse_scene`; FindScenes' scenes

# circle_tests: 200 exact triples on random circles.
function circle_triples(rng)
    return map(1:200) do _
        c = SVector(4000rand(rng) - 2000, 4000rand(rng) - 2000); r = 10.0^(4rand(rng) - 1)
        [Point2(c + r * SVector(cos(t), sin(t))) for t in 2π * rand(rng) .+ [0, 2π / 3, 4π / 3] .+ 0.5 .* rand(rng, 3)]
    end
end
circle_degenerate() = (
    collinear = [Point2(0.0, 0.0), Point2(1.0, 2.0), Point2(3.0, 6.0)],
    repeated = [Point2(1.0, 1.0), Point2(1.0, 1.0), Point2(3.0, 0.0)],
    nonfinite = [Point2(0.0, 0.0), Point2(NaN, 1.0), Point2(3.0, 0.0)],
    svectors = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(1.0, 1.0)],
)
# ellipse_tests: 200 exact quintuples on random ellipses.
function ellipse_quintuples(rng)
    return map(1:200) do _
        c = SVector(4000rand(rng) - 2000, 4000rand(rng) - 2000); a = 10.0^(3rand(rng)); b = a * (0.2 + 0.8rand(rng))
        e₀ = Ellipse(Point2(c...), a, b, π * rand(rng))
        [boundary_point(e₀, t) for t in 2π * rand(rng) .+ (0:4) .* (2π / 5) .+ 0.3 .* rand(rng, 5)]
    end
end
ellipse_degenerate() = (
    hyperbola = [Point2(t, 1 / t) for t in (0.5, 1.0, 2.0, -1.0, -3.0)],
    parabola = [Point2(t, t^2) for t in (-2.0, -1.0, 0.0, 1.0, 3.0)],
    line_pair = [Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(2.0, 0.0), Point2(0.0, 1.0), Point2(1.0, 2.0)],
    nonfinite = [Point2(0.0, 0.0), Point2(NaN, 1.0), Point2(3.0, 0.0), Point2(1.0, 1.0), Point2(2.0, 3.0)],
    unit_circle = [SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(0.0, -1.0), SVector(√0.5, √0.5)],
    thin_triple = [Point2(0.0, 0.0), Point2(100.0, 0.0), Point2(50.0, 0.05), Point2(10.0, 60.0), Point2(80.0, 50.0)],
)

# --- rotations (rotation2pt_tests; ransac_rotation_tests' RotationScene) ------------------------------------------

function rotation_pairs(rng)
    return map(1:200) do _
        R = RotMatrix(QuatRotation(normalize(randn(rng, 4))))
        u = [normalize(randn(rng, SVector{3, Float64})) for _ in 1:2]
        Correspondences(3.0 .* u, 0.5 .* [R * x for x in u])
    end
end
rotation_degenerate() = let a = SVector(0.0, 0.0, 1.0), u = [SVector(1.0, 0.0, 0.0), SVector(-2.0, 0.0, 0.0)]
    (antiparallel_source = Correspondences(u, u),
     parallel_target = Correspondences([a, SVector(0.0, 1.0, 0.0)], [a, a]),
     nonfinite = Correspondences([a, SVector(NaN, 0.0, 1.0)], [a, a]))
end
const FISHEYE = CameraModel(300.0px, SA[600.0px, 600.0px]; lens = KannalaBrandt(-0.02))
# RotationScene's fisheye pixel pairs of a pure rotation (rays within 100° of both axes, σ = 0.5 px, 30 % outliers),
# lifted to the bearings the solver takes.
function rotation_scene(rng, R, n; σ = 0.5, outl = 0.3, θmax = deg2rad(100))
    u₁ = SVector{2, Float64}[]; u₂ = SVector{2, Float64}[]
    while length(u₁) < n
        d = normalize(randn(rng, SVector{3, Float64}))
        (acos(clamp(d[3], -1, 1)) < θmax && acos(clamp((R * d)[3], -1, 1)) < θmax) || continue
        a = project(FISHEYE, d); b = project(FISHEYE, R * d)
        (all(isfinite, a) && all(isfinite, b)) || continue
        push!(u₁, a + σ * randn(rng, SVector{2, Float64})); push!(u₂, b + σ * randn(rng, SVector{2, Float64}))
    end
    for i in randperm(rng, n)[1:round(Int, outl * n)]
        while true
            c = SVector(600 + 500 * (2rand(rng) - 1), 600 + 500 * (2rand(rng) - 1))
            all(isfinite, backproject_with_jacobian(FISHEYE, c)[1]) && (u₂[i] = c; break)
        end
    end
    return Correspondences([SVector{3, Float64}(backproject(FISHEYE, u)) for u in u₁], [SVector{3, Float64}(backproject(FISHEYE, u)) for u in u₂])
end

# --- two views (solver_layer_tests' scene; EpipolarScenes' fundmat_scene; essential_5pt_native_tests) -------------

function minimal_epipolar_scene(rng)
    src = [Point2{Float64}(100rand(rng), 100rand(rng)) for _ in 1:20]    # the homography scene's draws come first
    R = SMatrix{3, 3, Float64}(rand(rng, RotMatrix{3, Float64}))
    t = SVector(0.4, -0.2, 0.1)
    X3 = [SVector(2randn(rng), 2randn(rng), 6.0 + 2rand(rng)) for _ in 1:20]
    x1b = [normalize(Xi) for Xi in X3]
    x2b = [normalize(R * Xi + t) for Xi in X3]
    u1 = [Point2{Float64}(PerspectiveMap()(Xi)) for Xi in X3]
    u2 = [Point2{Float64}((R * Xi + t)[1] / (R * Xi + t)[3], (R * Xi + t)[2] / (R * Xi + t)[3]) for Xi in X3]
    return (; u1, u2, x1b, x2b)
end
minimal_fundmat_pairs(rng) = let s = _MS.EpipolarScenes.fundmat_scene(rng); Correspondences(s.source, s.target) end
# essential_5pt_native_tests' ensemble: 400 attempts of a random pose and 5 points, kept when in front of view 2,
# as unit bearings (the test's z = 1 lift normalized: the solver takes bearings).
function essential_quintuples(rng)
    out = Correspondences{SVector{3, Float64}, SVector{3, Float64}, Vector{SVector{3, Float64}}, Vector{SVector{3, Float64}}, @NamedTuple{}}[]
    for _ in 1:400
        R = SMatrix{3, 3, Float64, 9}(rand(rng, RotMatrix{3, Float64}))
        t = normalize(SVector{3, Float64}(randn(rng, 3)...))
        X = [SVector{3, Float64}(2randn(rng), 2randn(rng), 6.0 + 2rand(rng)) for _ in 1:5]
        x2p = [R * Xi + t for Xi in X]
        any(p[3] <= 0.2 for p in x2p) && continue
        push!(out, Correspondences(normalize.(X), normalize.(x2p)))
    end
    return out
end
# PoseLib's test_refine scene: 200 bearings pairs, 40 of view 2 replaced by uniform directions (outliers), 1e-3 noise.
function minimal_essential_scene(rng)
    R = SMatrix{3, 3, Float64, 9}(rand(rng, RotMatrix{3, Float64}))
    t = normalize(SVector(1.0, 0.1, 0.05))
    X = [SVector(4 * randn(rng), 3 * randn(rng), 8.0 + 4 * rand(rng)) for _ in 1:200]
    b1 = [normalize(Y) for Y in X]
    b2 = [normalize(normalize(R * Y + t) + 1.0e-3 * randn(rng, SVector{3, Float64})) for Y in X]
    for i in randperm(rng, 200)[1:40]
        b2[i] = normalize(randn(rng, SVector{3, Float64}))
    end
    return Correspondences(b1, b2)
end

# --- absolute pose (p3p_native_tests; FindScenes' pnp_scene) -----------------------------------------------------

# p3p_native_tests: 300 random poses and 3 points, kept when in front, as `X => bearing` correspondences.
function p3p_triples(rng)
    out = Correspondences{SVector{3, Float64}, SVector{3, Float64}, Vector{SVector{3, Float64}}, Vector{SVector{3, Float64}}, @NamedTuple{}}[]
    for _ in 1:300
        R = SMatrix{3, 3, Float64}(rand(rng, RotMatrix{3, Float64}))
        t = SVector(randn(rng), randn(rng), 3.0 + 2rand(rng))
        X = [SVector{3, Float64}(2randn(rng, 3)...) for _ in 1:3]
        any((R * Xi + t)[3] <= 0.1 for Xi in X) && continue
        push!(out, Correspondences(X, [SVector{3, Float64}(normalize(R * Xi + t)) for Xi in X]))
    end
    return out
end
p3p_degenerate() = let p0 = SVector(0.0, 0.0, 0.0), d = SVector(1.0, 0.0, 0.0), xs = [normalize(SVector(0.1i, 0.05, 1.0)) for i in 1:3]
    (collinear = Correspondences([p0, p0 + d, p0 + 2d], xs), repeated = Correspondences([p0, p0, p0], xs),
     fixed = Correspondences([SVector(1.0, 0.0, 5.0), SVector(-1.0, 1.0, 6.0), SVector(0.5, -1.0, 4.0)],
         [normalize(SVector(0.1, 0.05, 1.0)), normalize(SVector(-0.2, 0.1, 1.0)), normalize(SVector(0.0, -0.15, 1.0))]))
end
# FindScenes' pnp_scene (its frustum points, 120 inliers at σ = 0.5 px, 60 outliers) with the scene's StableRNG(seed),
# the pixels lifted to bearings through the camera.
pnp_bearing_scene(rng) = let cs = pnp_scene(rng)
    Correspondences([SVector{3, Float64}(first(p)) for p in cs], [SVector{3, Float64}(backproject(PCAM, last(p))) for p in cs])
end

# --- the radial homography (homography_5pt_radial_tests; FindScenes' radial_homography_scene) ---------------------

function radial_golden()
    s = [SVector(0.242104535827652, -0.296297354299645), SVector(0.319776775526576, -0.114580331777873),
         SVector(-0.122936754370192, -0.532655969828257), SVector(0.303141152380151, -0.295669384794041),
         SVector(0.446905125282507, -0.528127859697035)]
    d = [SVector(-0.562787932824692, -0.0769625591786919), SVector(-0.612813481767545, -0.0311462459576149),
         SVector(-0.57724705213935, -0.252034297773712), SVector(-0.544031097770073, -0.0440402280844913),
         SVector(-0.424602042497788, 0.0211895908990797)]
    s_bad = [SVector(0.0, 0.0), SVector(0.1, 0.1), SVector(0.2, 0.2), SVector(0.3, 0.3), SVector(0.4, -0.1)]
    return (; golden = Correspondences(s, d), collinear = Correspondences(s_bad, d))
end
_transfer(H, s) = (PerspectiveMap() ∘ HomographyMap(SMatrix{3, 3, Float64}(H)))(s)
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitCore", "test", "fixtures", "division_distortion.jl"))
# The test's synthetic ground-truth ensemble: 200 attempts of a similarity-plus-projective H and a division k.
function radial_quintuples(rng)
    out = Correspondences{SVector{2, Float64}, SVector{2, Float64}, Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}}, @NamedTuple{}}[]
    for _ in 1:200
        θ = 2π * rand(rng); sc = 0.7 + 0.6 * rand(rng)
        v1, v2 = 0.2 * randn(rng), 0.2 * randn(rng)
        t1, t2 = 0.3 * randn(rng), 0.3 * randn(rng)
        H0 = SMatrix{3, 3, Float64}(sc * cos(θ), sc * sin(θ), v1, -sc * sin(θ), sc * cos(θ), v2, t1, t2, 1.0)
        k_gt = -0.4 + 0.55 * rand(rng)
        src = [SVector(rand(rng) - 0.5, rand(rng) - 0.5) for _ in 1:5]
        dst_u = map(p -> _transfer(H0, p), src)
        any(p -> norm(p) > 1.2, dst_u) && continue
        dst = map(p -> distort_division(p, k_gt), dst_u)
        any(isnothing, dst) && continue
        push!(out, Correspondences(src, [p::SVector{2, Float64} for p in dst]))
    end
    return out
end
# The test's Hartley-about-the-distortion-centre scene (a 7 × 7 board through a known RadialHomography), raw and
# in the normalization's frame (the frame the kernel runs in).
function radial_board()
    f0 = 1400.0; c = SVector(1010.0, 520.0)
    R = SMatrix{3, 3}(1.0, 0.0, 0.0, 0.0, cos(0.3), sin(0.3), 0.0, -sin(0.3), cos(0.3))
    t = SVector(40.0, -30.0, 900.0)
    Hm = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(f0, 0, 0, 0, f0, 0, c[1], c[2], 1) * hcat(R[:, 1], R[:, 2], t)))
    m = RadialHomography{Float64}(Hm, -3.0e-7, Point2{Float64}(c[1], c[2]))
    Xs = [SVector(x, y) for x in -150.0:50.0:150.0 for y in -150.0:50.0:150.0]
    cs = Correspondences(Xs, [SVector{2, Float64}(radial_transfer(m, X)) for X in Xs])
    csn = LinearAlgebra.normalize(VgtkCore.CorrespondenceNormalization(RadialHomography, cs; center = c), cs)
    return (; raw = cs, normalized = csn)
end
# FindScenes' radial_homography_scene (70 inliers at σ = 0.3 px, 30 outliers, shuffled) with StableRNG(seed).
function radial_homography_normalized_scene(rng)
    pairs, _ = radial_homography_scene(rng)
    cs = Correspondences(first.(pairs), last.(pairs))
    return (; raw = cs, normalized = LinearAlgebra.normalize(VgtkCore.CorrespondenceNormalization(RadialHomography, cs; center = SVector(RW / 2, RH / 2)), cs))
end

# --- the radial camera (lens_contract_tests; RadialCameraScene) ---------------------------------------------------

# lens_contract_tests' board: a 5 × 5 grid through K = diag(f0, f0, 1), a division lens k0, raw about c.
function radial_camera_board()
    f0, k0 = 1226.3, -6.4e-7
    K = @SMatrix [f0 0.0 0.0; 0.0 f0 0.0; 0.0 0.0 1.0]
    R = RotXYZ(0.4, -0.3, 0.2); t = SVector(30.0, -20.0, 400.0)
    pts = Pair{SVector{2, Float64}, SVector{2, Float64}}[]
    for X in -80.0:40.0:80.0, Y in -80.0:40.0:80.0
        P = R * SVector(X, Y, 0.0) + t
        u = f0 * PerspectiveMap()(P)
        ρ = hypot(u[1], u[2]); r = ρ < 1.0e-12 ? 0.0 : (1 - sqrt(1 - 4k0 * ρ^2)) / (2k0 * ρ)
        push!(pts, SVector(X, Y) => (ρ < 1.0e-12 ? u : u * (r / ρ)))
    end
    c = SVector(931.0, -472.0)
    col = [SVector(t, 2t) => pts[1][2] .* (1 + 0.1t) for t in 1.0:5.0]
    return (; centred = Correspondences(first.(pts), last.(pts)), raw = Correspondences(first.(pts), [x + c for x in last.(pts)]),
            center = c, collinear = Correspondences(first.(col), last.(col)))
end
# RadialCameraScene's contaminated board (StableRNG(seed) in place of its StableRNG(19)).
radial_camera_scene(rng) = let pairs = first(radial_camera_pairs(rng)); Correspondences(first.(pairs), last.(pairs)) end

# --- the homography (PoseLib backend only: the native kernel is the pilot's) --------------------------------------

homography_layer_scene(rng) = let H0 = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, -2.0e-4, 4.0, -3.0, 1.0))),
        src = [Point2{Float64}(100rand(rng), 100rand(rng)) for _ in 1:20]
    Correspondences(src, [Point2{Float64}((PerspectiveMap() ∘ HomographyMap(H0))(p)) for p in src])
end
homography_scene(rng) = let (s, d) = _MS.HomographyScenes.homography_pairs(rng, _MS.HomographyScenes.H_TRUE; n_in = 100, n_out = 30, σ = 0.5)
    Correspondences(s, d)
end
