@testitem "radial_homography" setup = [Contamination] begin
# RadialHomographyProblem + find_radial_homography — one-sided division-model
# radially distorted homography (VGC homography_5pt_radial / Wadenbäck 3DV'26).
#
# Ground truth is generated through the model's own projection
# (radial_transfer), in pixel coordinates with the distortion center at the
# image middle — the `center` find_radial_homography requires.

using Test
using StaticArrays, LinearAlgebra, Random
using VisualGeometryCore: center, converged, division_coeff, estimate, homography   # VGC accessors (public)
using VisualGeometryCore: is_map_inlier   # the posterior MAP inlier rule
using RobustVisualGeometry
using RobustVisualGeometry: RadialHomographyProblem, find_radial_homography,
    solve, refine!, residuals!, data_size
using VisualGeometryCore: VisualGeometryCore, RadialHomography, radial_transfer, backproject, WelschShape,
    HomographyMat, unit_normalize
using VisualGeometryCore.Synthetic: projective_distance

const _RW, _RH = 1200.0, 900.0
_rdiag() = hypot(_RW, _RH)

function _gt_model(k_norm)
    H0 = SMatrix{3, 3, Float64}(
        4.0, -0.2, 5.0e-5, 0.3, 4.2, -4.0e-5, 260.0, 200.0, 1.0
    )
    return RadialHomography{Float64}(
        HomographyMat{Float64}(Tuple(H0)), k_norm / _rdiag()^2,
        SVector(_RW / 2, _RH / 2)
    )
end

# `n_in` sources drawn until their transfer lands in the image, then `n_out`
# unrelated pairs, shuffled.
function _make_scene(rng, m_gt; n_in = 70, n_out = 30, σ = 0.0)
    function inlier(rng, _)
        while true
            s = SVector(200 * rand(rng), 150 * rand(rng))
            d = radial_transfer(m_gt, s)
            d === nothing && continue
            (0 < d[1] < _RW && 0 < d[2] < _RH) && return s => SVector(d[1], d[2])
        end
    end
    return contaminated_pairs(
        rng, inlier, n_in; σ, n_out, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng), 150 * rand(rng)) => SVector(_RW * rand(rng), _RH * rand(rng))
    )
end

@testset "radial homography (RANSAC)" begin
    rng = MersenneTwister(1234)
    m_gt = _gt_model(-0.25)

    @testset "problem-level: minimal solve composes exact pixel-frame models" begin
        pairs, _ = _make_scene(rng, m_gt; n_in = 20, n_out = 0)
        p = RadialHomographyProblem(
            pairs; center = (_RW / 2, _RH / 2), extra_check = true
        )
        idx = collect(1:5)
        sols = solve(p, idx)
        @test length(sols) >= 1
        # extra_check keeps the consistent root: on clean data its composed
        # (pixel-frame) model reproduces the observations at the sample
        ok = false
        for m in sols
            worst = maximum(idx) do i
                d̂ = radial_transfer(m, pairs[i].first)
                d̂ === nothing ? Inf : norm(SVector(d̂[1], d̂[2]) - pairs[i].second)
            end
            worst < 1.0e-6 && (ok = true)
        end
        @test ok
        # residuals! against ground truth: exact on clean inliers
        r = zeros(length(pairs))
        residuals!(r, p, m_gt)
        @test maximum(r) < 1.0e-8
    end

    @testset "find_radial_homography: exact data + outliers ($loname)" for
        (loname, lokw) in (
            ("default (Irls{TLS})", (;)),
            ("Irls{Welsch}", (; local_optimization = Irls{WelschShape}(scale_multiple = 1.5))),
        )   # the old default (the Welsch HQ lift) and the Student-t + uniform IRLS: LocalOptimizationDoneRight's suite

        pairs, gt_mask = _make_scene(rng, m_gt; n_in = 70, n_out = 30)
        res = find_radial_homography(
            pairs; center = (_RW / 2, _RH / 2), threshold = 2.0,
            max_trials = 4000, min_trials = 200,
            rng = MersenneTwister(99), lokw...
        )
        @test res !== nothing
        @test converged(res)
        m = estimate(res)
        @test m isa RadialHomography{Float64}
        @test center(m) ≈ SVector(_RW / 2, _RH / 2)
        @test abs(division_coeff(m) - division_coeff(m_gt)) / abs(division_coeff(m_gt)) < 1.0e-3
        @test projective_distance(homography(m), homography(m_gt)) < 1.0e-4
        w = res.weights
        @test count(i -> gt_mask[i] && is_map_inlier(w[i]), eachindex(w)) >= 65
        @test count(i -> !gt_mask[i] && is_map_inlier(w[i]), eachindex(w)) <= 2
    end

    @testset "noise: recovers sign and magnitude of k" begin
        pairs, _ = _make_scene(rng, m_gt; n_in = 120, n_out = 20, σ = 0.4)
        res = find_radial_homography(
            pairs; center = (_RW / 2, _RH / 2), threshold = 2.5,
            max_trials = 4000, rng = MersenneTwister(7)
        )
        @test res !== nothing
        @test converged(res)
        @test sign(division_coeff(estimate(res))) == sign(division_coeff(m_gt))
        @test abs(division_coeff(estimate(res)) - division_coeff(m_gt)) / abs(division_coeff(m_gt)) < 0.3
    end

    @testset "fit M-step reduces cost from a perturbed seed" begin
        pairs, _ = _make_scene(rng, m_gt; n_in = 40, n_out = 0, σ = 0.3)
        p = RadialHomographyProblem(pairs; center = (_RW / 2, _RH / 2))
        n = data_size(p)
        θ0 = RadialHomography{Float64}(homography(m_gt), division_coeff(m_gt) * 1.5, center(m_gt))
        w = ones(n)
        r0 = zeros(n); residuals!(r0, p, θ0)
        θ1 = refine!(nothing, p, θ0, w)
        @test θ1 !== nothing
        r1 = zeros(n); residuals!(r1, p, θ1)
        @test sum(abs2, r1) < sum(abs2, r0)
    end

    @testset "minimal = :homography — the global search for H with the (H, k) optimizer" begin
        using RobustVisualGeometry: minimal_sample_size, _radial_k_seed
        rng2 = MersenneTwister(77)
        pairs, gt = _make_scene(rng2, m_gt; n_in = 80, n_out = 25, σ = 0.3)
        p4 = RadialHomographyProblem(pairs; center = (_RW / 2, _RH / 2), minimal = :homography)
        @test minimal_sample_size(p4) == 4
        # a four-point hypothesis is a plain homography with no distortion term
        idx = findall(gt)[[1, 12, 30, 55]]
        sols = solve(p4, idx)
        @test length(sols) == 1 && iszero(division_coeff(sols[1]))
        @test all(isfinite, homography(sols[1]))
        # the linear k on the truth's H over the inliers lands on the truth
        H_gt = unit_normalize(SMatrix{3, 3, Float64, 9}(homography(m_gt)))
        k_lin = _radial_k_seed(
            [pr.first for pr in pairs], [pr.second for pr in pairs], H_gt, SVector{2, Float64}(center(m_gt)[1], center(m_gt)[2]), Float64.(gt)
        )
        @test abs(k_lin - division_coeff(m_gt)) < 0.15 * abs(division_coeff(m_gt))
        # RANSAC: the four-point search finds the board, the (H, k) local
        # optimization brings the distorted-away inliers in and recovers k
        est = find_radial_homography(
            pairs; center = (_RW / 2, _RH / 2), minimal = :homography, threshold = 3.0,
            rng = MersenneTwister(5)
        )
        @test est !== nothing
        m = est.inlier_mask
        @test count(m .& gt) >= count(gt) - 3 && count(m .& .!gt) == 0
        @test abs(division_coeff(estimate(est)) - division_coeff(m_gt)) < 0.1 * abs(division_coeff(m_gt))
        @test sign(division_coeff(estimate(est))) == sign(division_coeff(m_gt))
        @test_throws ArgumentError RadialHomographyProblem(pairs; center = (_RW / 2, _RH / 2), minimal = :other)
        # `extra_check` filters the 5-point kernel's roots: on the four-point path
        # it used to be accepted and ignored; now it is refused, and the wrapper
        # (which forwards it) still runs the four-point path when it is unset.
        @test_throws ArgumentError RadialHomographyProblem(
            pairs; center = (_RW / 2, _RH / 2), minimal = :homography, extra_check = true
        )
        @test_throws ArgumentError find_radial_homography(
            pairs; center = (_RW / 2, _RH / 2), minimal = :homography, extra_check = false
        )
    end

    @testset "extent gate: corner cheirality (< 180° across the board)" begin
        rect = VisualGeometryCore.Rect(0.0, 0.0, 200.0, 150.0)
        et = RobustVisualGeometry.WithExtent(rect)
        # ground-truth model: every board corner on the front hemisphere
        @test RobustVisualGeometry._check_extent(et, nothing, m_gt, nothing)
        # a hypothesis that sends a corner ray behind (z < 0 at x = 200)
        Hb = SMatrix{3, 3, Float64, 9}(homography(m_gt))
        Hbad = SMatrix{3, 3, Float64}(
            Hb[1, 1], Hb[2, 1], -0.02,
            Hb[1, 2], Hb[2, 2], 0.0,
            Hb[1, 3], Hb[2, 3], 1.0
        )
        mbad = RadialHomography{Float64}(
            RobustVisualGeometry.VisualGeometryCore.HomographyMat{Float64}(Tuple(Hbad)),
            division_coeff(m_gt), center(m_gt)
        )
        @test !RobustVisualGeometry._check_extent(et, nothing, mbad, nothing)

        # end-to-end: the extent-bounded fit still converges and recovers k
        pairs, _ = _make_scene(rng, m_gt; n_in = 60, n_out = 15)
        res = find_radial_homography(
            pairs; center = (_RW / 2, _RH / 2), threshold = 2.0,
            extent = rect, max_trials = 3000, rng = MersenneTwister(21)
        )
        @test res !== nothing
        @test converged(res)
        @test abs(division_coeff(estimate(res)) - division_coeff(m_gt)) / abs(division_coeff(m_gt)) < 1.0e-3
    end
end
end
