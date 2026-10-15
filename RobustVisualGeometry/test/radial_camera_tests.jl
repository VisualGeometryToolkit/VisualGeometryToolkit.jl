@testitem "radial_camera" setup = [DivisionBoards, Contamination] begin
# RadialCameraProblem + find_radial_camera — the 1-D radial camera of one board
# (VGC RadialCamera: rows 1–2 of the plane homography, invariant to focal
# length and to radial distortion about the centre). Ground truth: a board
# under a pinhole K with the principal point at the image middle, then the
# division model applied about that middle — the radial camera is untouched
# by it, which is the whole point.
using Test
using StaticArrays, LinearAlgebra, Random
using FiniteDiff: FiniteDiff
using Statistics: median
using Rotations: RotXYZ
using VisualGeometryCore: center, estimate, radial_rows   # VGC accessors (public)
using RobustVisualGeometry
using RobustVisualGeometry: RadialCameraProblem, find_radial_camera,
    solve, refine!, residuals!, data_size, codimension
using VisualGeometryCore: VisualGeometryCore, RadialCamera, radial_camera_5pt,
    constraint_with_jacobians
using VisualGeometryCore.Synthetic: projective_distance

const _CW, _CH = 4000.0, 3000.0
const _CC = SVector(_CW / 2, _CH / 2)

# The true radial camera and a projector through a division lens (k on the
# centred pixel radius) — the pixels are what a detector would measure.
function _board(rng; f = 900.0, k = -2.0e-7, n_out = 20, σ = 0.0)
    cam = board_camera(RotXYZ(0.35, -0.25, 0.15), SVector(40.0, -30.0, 600.0); f, ks = (k,), center = _CC)
    # The pixels are RAW (about `_CC`), and so is the problem: the camera
    # carries its centre, so the ground truth carries the same one.
    pairs, gt = contaminated_pairs(
        rng, board_pixels(cam; grid = -100.0:25.0:100.0); σ, n_out, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng) - 100, 200 * rand(rng) - 100) => SVector(_CW * rand(rng), _CH * rand(rng))
    )
    return RadialCamera(cam), pairs, gt
end
# Radial cameras agree up to scale and sign: their rows' projective distance
# (1 − |cos∠| = d²/2, so d < √(2ε) is |cos∠| within ε of 1).
_pd(a::RadialCamera, b::RadialCamera) = projective_distance(radial_rows(a), radial_rows(b))

@testset "radial camera (RANSAC)" begin
    rng = MersenneTwister(7)
    P_gt, pairs, gt = _board(rng)
    p = RadialCameraProblem(pairs; center = _CC)
    @test data_size(p) == length(pairs) && codimension(typeof(p)) == 1
    # residuals of the truth: zero on inliers, large on outliers
    r = zeros(length(pairs))
    residuals!(r, p, P_gt)
    @test maximum(abs.(r[gt])) < 1.0e-6
    @test minimum(abs.(r[.!gt])) > 1.0
    # the minimal solve on five inliers reproduces the truth up to scale
    idx = findall(gt)[[1, 9, 25, 41, 73]]
    sols = solve(p, idx)
    @test length(sols) == 1
    @test _pd(sols[1], P_gt) < sqrt(2 * 1.0e-8)
    # a collinear sample is refused, not returned
    col = [i for i in eachindex(pairs) if gt[i] && pairs[i].first[2] == -100.0][1:5]
    @test length(solve(p, col)) == 0
    # the chart refit: from a perturbed seed, the inlier weights bring it back to the truth
    w = Float64.(gt)
    seed = RadialCamera{Float64}(radial_rows(P_gt) + 0.02 * norm(radial_rows(P_gt)) * SMatrix{2, 3, Float64, 6}(0.3, -0.5, 0.8, 0.1, -0.4, 0.6), center(P_gt))
    Pf = refine!(nothing, p, seed, w)
    @test Pf !== nothing && _pd(Pf, P_gt) < sqrt(2 * 1.0e-8)
    @test abs(norm(vec(radial_rows(Pf))) - 1) < 1.0e-12                          # homogeneous: unit norm, no entry pinned
    # the chart Jacobian against ForwardDiff on the residual rows
    ch = VisualGeometryCore.minimal_manifold(seed)      # VGC's SphereManifold on vec(P)
    lm = RobustVisualGeometry.ChartLm(ch, p, sqrt.(w))
    @test VisualGeometryCore.check_rows_jacobian(lm, SVector(0.01, -0.02, 0.005, 0.0, 0.015), data_size(p); atol = 1.0e-8)
    # RANSAC recovers the inlier set and the camera under distortion it never models
    est = find_radial_camera(pairs; center = _CC, threshold = 5.0, rng = MersenneTwister(1))
    @test est !== nothing
    @test _pd(estimate(est), P_gt) < sqrt(2 * 1.0e-6)
    m = est.inlier_mask
    @test count(m .& gt) >= count(gt) - 2 && count(m .& .!gt) == 0
    # with noise: still within the noise floor
    P_gt2, pairs2, gt2 = _board(MersenneTwister(3); σ = 0.3)
    est2 = find_radial_camera(pairs2; center = _CC, threshold = 5.0, rng = MersenneTwister(2))
    @test est2 !== nothing
    r2 = zeros(length(pairs2)); residuals!(r2, RadialCameraProblem(pairs2; center = _CC), estimate(est2))
    @test median(abs.(r2[gt2])) < 0.5
    @test find_radial_camera(pairs[1:4]; center = _CC, rng = MersenneTwister(446)) === nothing
end

# The chart rows on the 5-dof complement chart (problems/manifold_providers.jl,
# VisualGeometryCore's rows contract): the chart Jacobian against central
# differences of the provider's own refresh along its retraction, and the
# deployed pipeline (RANSAC + the class default) recovering the truth under
# distortion. (The partition-anchored Welsch-HQ lift on this chart, the old
# default, is LocalOptimizationDoneRight's test.)
@testset "radial camera (chart rows)" begin
    using RobustVisualGeometry: Irls, TLSShape, MinimalSolveProblem, ManifoldLiftProvider
    using VisualGeometryCore: rows_provider, lift_init, lift_refresh!, lift_jacobian!, lift_retract,
        lift_model, lift_residuals
    rng = MersenneTwister(11)
    P_gt, pairs, gt = _board(rng; σ = 0.2)
    p = RadialCameraProblem(pairs; center = _CC)
    @test p isa MinimalSolveProblem
    seed = RadialCamera{Float64}(radial_rows(P_gt) + 0.02 * norm(radial_rows(P_gt)) * SMatrix{2, 3, Float64, 6}(0.3, -0.5, 0.8, 0.1, -0.4, 0.6), center(P_gt))
    prov = rows_provider(p)
    @test prov isa ManifoldLiftProvider{<:Any, 5, 1}                # 5-dof chart, codim 1
    θ = lift_init(prov, seed)                                    # the chart (VGC SphereManifold on vec(P))
    @test abs(norm(vec(radial_rows(lift_model(prov, θ)))) - 1) < 1.0e-12
    @test _pd(lift_model(prov, θ), seed) < sqrt(2 * 1.0e-12)
    # residuals at the chart centre are the problem's own
    lift_refresh!(prov, θ)
    r = zeros(data_size(p)); residuals!(r, p, seed)
    @test maximum(abs.(lift_residuals(prov)[1:data_size(p)] - r)) < 1.0e-9
    lift_jacobian!(prov, θ)
    J = copy(VisualGeometryCore.lift_rows(prov).J[1:data_size(p), :])
    lift_rows(v) = (lift_refresh!(prov, lift_retract(prov, θ, v)); lift_residuals(prov)[1:data_size(p)])
    Jc = FiniteDiff.finite_difference_jacobian(lift_rows, zero(SVector{5, Float64}), Val(:central))
    @test maximum(abs.(J - Jc)) < 1.0e-5 * max(1.0, maximum(abs.(Jc)))
    # the deployed pipeline: RANSAC + the class default
    est = find_radial_camera(pairs; center = _CC, threshold = 5.0, rng = MersenneTwister(4))
    @test est !== nothing
    @test est.lo_retcode != :not_run
    rr = zeros(length(pairs)); residuals!(rr, p, estimate(est))
    @test median(abs.(rr[gt])) < 0.4
    @test count(est.inlier_mask .& .!gt) == 0
    # the shipped default: the core's TLS IRLS, following the scorer
    @test RobustVisualGeometry._dflt_lo(:radial_camera) == Irls{TLSShape}()
end
end
