@testitem "fitting_pipelines" setup = [EpipolarScenes, HomographyScenes, PoseScenes, Contamination] begin
    import VisualGeometryToolkitEstimation as VgtkEstimation
# =============================================================================
# Fitting pipelines — the public `find_*` entry points
# =============================================================================
#
# WHY THIS FILE EXISTS: until 2026-09-03 the suite never called a single one of
# these. `src/fitting/{homography,fundmat,essential,pose,line}.jl`
# had literally zero coverage across 15,585 succeeding tests — the problem layer
# underneath was exercised directly, so the wrappers, their defaults, their
# threshold choices and the `_ransac_pipeline` glue shipped unrun.
#
# These are deliberately SMOKE-level: enough outliers that RANSAC has to do its
# job, and a ground-truth check on the returned model. They are not a substitute
# for the per-problem batteries.
# =============================================================================

using Test
using LinearAlgebra
using StaticArrays
using StructArrays
using Random
using StableRNGs
using Rotations: RotMatrix, RotXYZ, RotY
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: converged, estimate, rotation, translation   # VGC accessors (public)
using VisualGeometryToolkitEstimation: RansacWorkspace   # public, not exported
using VisualGeometryToolkitEstimation

@testset "fitting pipelines" begin

    @testset "find_homography" begin
        H = SMatrix{3, 3, Float64, 9}(H_PX)
        src, dst = homography_pairs(                  # inliers: exact transfer + 0.3 px; outliers: unrelated pairs
            StableRNG(11), H_PX; n_in = 120, n_out = 60, σ = 0.3,
            source = gaussian_source, target = rng -> randn(rng, 2) .* 400 .+ 300
        )

        r = find_homography(StructArray(src .=> dst); rng = StableRNG(134))
        @test r !== nothing
        @test converged(r)
        # Every inlier sits in the first 120 slots; demand most of them back
        # and almost no outlier admitted.
        @test count(r.inlier_mask[1:120]) > 100
        @test count(r.inlier_mask[121:end]) < 20
        Ĥ = estimate(r) / estimate(r)[3, 3]
        @test norm(Ĥ - H / H[3, 3]) / norm(H / H[3, 3]) < 1.0e-2

        # A pre-allocated workspace must give the same answer as an internal one:
        # the same draws (same seed as `r`), the same model and consensus — and
        # again when the workspace is reused.
        ws = RansacWorkspace(HomographyProblem{Float64}, length(src))
        for _ in 1:2
            r2 = find_homography(StructArray(src .=> dst); workspace = ws, rng = StableRNG(134))
            @test r2 !== nothing && converged(r2)
            @test estimate(r2) == estimate(r)
            @test r2.inlier_mask == r.inlier_mask
            @test r2.quality == r.quality
        end

        # Below the minimal sample the wrapper returns nothing, it does not throw.
        @test find_homography(StructArray(src[1:3] .=> dst[1:3]); rng = StableRNG(136)) === nothing
    end

    @testset "find_fundmat" begin
        cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
        u₁, u₂ = stereo_pixel_pairs(                  # inliers: 0.4 px in view 2; outliers: uniform pairs
            StableRNG(13), cam, cam, RotMatrix(RotXYZ(0.05, 0.12, -0.03)), SVector{3, Float64}(0.9, 0.05, 0.1);
            n = 140, σ₂ = 0.4, n_out = 60,
            point = rng -> SA[2.0 * (rand(rng) - 0.5), 1.5 * (rand(rng) - 0.5), 4.0 + 3.0 * rand(rng)],
            outlier = rng -> (SA[960 * rand(rng), 720 * rand(rng)], SA[960 * rand(rng), 720 * rand(rng)])
        )
        n_in = length(u₁) - 60

        r = find_fundmat(StructArray(u₁ .=> u₂); rng = StableRNG(137))
        @test r !== nothing
        @test converged(r)
        @test count(r.inlier_mask[1:n_in]) > 0.75 * n_in
        # A fundamental matrix is rank 2.
        F = SMatrix{3, 3, Float64}(estimate(r))
        @test abs(det(F)) / norm(F)^3 < 1.0e-6
        # Recovered inliers satisfy their own epipolar constraint tightly.
        for i in 1:n_in
            r.inlier_mask[i] || continue
            @test sampson_distance(estimate(r), u₁[i], u₂[i]) < 5.0
        end

        ws = RansacWorkspace(FundMatProblem{Float64}, length(u₁))
        @test find_fundmat(StructArray(u₁ .=> u₂); workspace = ws, rng = StableRNG(138)) !== nothing
    end

    @testset "find_essential" begin
        rng = StableRNG(17)                     # calibrated points + 0.002 noise, then 50 outliers
        u₁, u₂ = calibrated_pairs(rng, RotY(0.2), normalize(SA[0.8, 0.1, 0.3]); n = 120, σ = 0.002)
        n_in = length(u₁)
        for _ in 1:50
            push!(u₁, Point2{Float64}(randn(rng), randn(rng)))
            push!(u₂, Point2{Float64}(randn(rng), randn(rng)))
        end

        # Calibrated points moved into the PIXEL domain of ECAM: `find_essential`
        # takes pixels and the cameras that produced them, and its threshold is
        # in pixels too.
        pxcs = csponds(topx.(u₁), topx.(u₂))
        # The noise is 0.002 calibrated = 1.6 px in ECAM, above the class's
        # 1 px convention: a threshold of ~3σ (the default itself is
        # core_defaults' test, on a 0.4 px scene).
        r = find_essential(ECAM, ECAM, pxcs; threshold = 5.0, rng = StableRNG(139))
        @test r !== nothing
        @test converged(r)
        @test count(r.inlier_mask[1:n_in]) > 0.75 * n_in
        # An essential matrix has two equal singular values and one zero.
        s = svd(SMatrix{3, 3, Float64}(estimate(r))).S
        @test abs(s[1] - s[2]) / s[1] < 1.0e-6
        @test s[3] / s[1] < 1.0e-6

        @test find_essential(
            ECAM, ECAM, csponds(topx.(u₁[1:4]), topx.(u₂[1:4])); rng = StableRNG(140)
        ) === nothing
        # The floor refusal: a threshold carried over in the OLD calibrated
        # units is rejected, not silently honoured 800x too tight.
        @test_throws ArgumentError find_essential(ECAM, ECAM, pxcs; threshold = 0.02)
    end

    @testset "find_absolute_pose" begin
        cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
        R_gt = RotMatrix(RotXYZ(0.15, -0.25, 0.1))
        t_gt = Vec3{Float64}(0.3, -0.2, 0.6)
        pairs, _ = contaminated_pairs(                   # inliers: frustum points + 0.5 px; outliers: uniform 2D-3D matches
            StableRNG(19), (rng, _) -> Pair(frustum_point(rng, cam, R_gt, t_gt)...), 120; σ = 0.5, n_out = 60,
            outlier = (rng, _) -> first(frustum_point(rng, cam, R_gt, t_gt)) => SVector(40.0 + 880.0 * rand(rng), 40.0 + 640.0 * rand(rng))
        )
        src, dst = first.(pairs), last.(pairs)

        r = find_absolute_pose(cam, StructArray(src .=> dst); rng = StableRNG(141))
        @test r !== nothing
        @test converged(r)
        @test count(r.inlier_mask[1:120]) > 100
        @test rotation_error_deg(estimate(r), R_gt) < 1.0
        @test translation_error(estimate(r), t_gt) < 0.05

        ws = RansacWorkspace(AbsolutePoseProblem, length(src))
        @test find_absolute_pose(cam, StructArray(src .=> dst); workspace = ws, rng = StableRNG(142)) !== nothing
        # fewer than the P3P minimal sample: no hypothesis; a minimal sample is
        # searched, and holds no consensus beyond itself
        @test find_absolute_pose(cam, StructArray(src[1:2] .=> dst[1:2]); rng = StableRNG(143)) === nothing
        @test !converged(find_absolute_pose(cam, StructArray(src[1:3] .=> dst[1:3]); rng = StableRNG(143)))

        # (PREDICTIVE scoring through the pose family's local optimization —
        # the Student-t + uniform marginal's `Predictive` mode — is
        # LocalOptimizationDoneRight's test.)
    end

    @testset "the default local optimization is the core's TLS IRLS" begin
        # Every pipeline's shipped default follows the scorer's inlier threshold.
        # (The paper's per-class Welsch HQ recipes, the defaults before
        # 2026-10, and their tuning constants are LocalOptimizationDoneRight's.)
        for class in (:homography, :fundmat, :essential, :rotation, :absolute_pose, :radial_homography, :radial_camera, :hand_eye)
            @test VgtkEstimation._dflt_lo(class) == Irls{TLSShape}()
        end
        # the circle's and the ellipse's are the robustified LM (configs/default.toml [pipelines.circle], [pipelines.ellipse])
        @test VgtkEstimation._dflt_lo(:circle) == Triggs{CauchyShape}(scale_multiple = 2.5)
        @test VgtkEstimation._dflt_lo(:ellipse) == Triggs{CauchyShape}(scale_multiple = 2.5)
    end

end
end
