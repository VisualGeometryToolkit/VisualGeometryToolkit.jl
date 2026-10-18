# The camera-aware E calibration trials.
#
# WHY THEY EXIST. The MC calibration harness generated CALIBRATED 2-VECTOR
# correspondences and handed them to the caller's `estimate`/`covariance`
# hooks. Once an essential-matrix problem stopped accepting plane data, those
# hooks could no longer be satisfied — and the consumer could not fix it
# locally, which is the part that matters: the harness's σ is what builds the
# point covariances the calibration test is ABOUT. Converting the data to
# pixels without converting σ would leave the noise model and the data in
# different frames INSIDE the instrument that measures whether covariances are
# calibrated. That is this wave's defect class, reintroduced in the detector.
#
# So the camera-aware path moves BOTH: pixel data, pixel σ, pixel shapes, and
# the geometry carried to bearings through `backproject_with_jacobian`, whose
# `M₁`/`M₂` put a consumer's residual back in pixels — the same whitening the
# B8 Monte-Carlo cells validated (NEES 0.999–1.016, predicted SE 0.010).
#
# This was the FOURTH place the calibrated-2-vector assumption was embedded,
# after `_mapped_bearing`, the deleted `_bearing` lift and `cheiral_pose_fast`.
# The first three were data-preparation paths; this one is a TEST-HARNESS
# interface, which is why no test caught it — the harness is what the tests
# are written in.
#
# TWO DIFFERENT FOCALS (1200 and 600) throughout, and that is load bearing:
# a per-view M₁/M₂ swap moves a residual 1.8% at two focals and only 0.19% at
# equal ones, where it would hide inside anything one would call noise.
@testitem "camera_aware_e_trial" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, LogicalIntrinsics, CameraCalibrationMat,
        Pinhole, backproject, project, tangent_size, model,
        constraint_jacobians, constraint_with_jacobians
    using VisualGeometryToolkitCore.Synthetic: essential_trial, epipolar_band_trial

    cam(f, cx, cy) = CameraModel(
        LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(f, 0, 0, 0, f, 0, cx, cy, 1))),
        Pinhole(),
    )
    CAM1 = cam(1200.0, 640.0, 480.0)
    CAM2 = cam(600.0, 320.0, 240.0)          # DIFFERENT focal — see the header

    # Hooks that record what the harness handed them and return a known
    # covariance, so the σ scaling downstream is readable.
    mk_hooks() = begin
        seen = Ref{Any}(nothing)
        covh = (ch, cs, shapes) -> (seen[] = (ch, cs, shapes);
            Matrix(1.0I, tangent_size(ch), tangent_size(ch)))
        esth = (ch, cs) -> model(ch)
        (seen, esth, covh)
    end

    @testset "the default (calibrated) paths still run — F, E, band" begin
        # Textually untouched; this is the guard that the added branch did not
        # disturb them. The whole suite's F/H/E trials are the real coverage.
        @test essential_trial(σ_px = 0.5, npts = 8)(MersenneTwister(7)) !== nothing
        @test epipolar_band_trial(:E; σ_px = 0.5, npts = 8)(MersenneTwister(9)) !== nothing
        @test epipolar_band_trial(:F; σ_px = 0.5, npts = 9)(MersenneTwister(11)) !== nothing
    end

    @testset "σ MOVES WITH THE DATA — the whole point, made a number" begin
        # With a hook returning I, the trial's returned covariance IS the σ²
        # it scaled by. Calibrated mode must read (σ_px/f)², camera-aware mode
        # σ_px². If the data moved to pixels and σ did not, this is where it
        # would show, and it is the one assertion that cannot succeed by accident.
        _, esth, covh = mk_hooks()
        o_cam = essential_trial(
            σ_px = 0.5, npts = 8, cameras = (CAM1, CAM2), estimate = esth, covariance = covh
        )(MersenneTwister(5))
        o_def = essential_trial(
            σ_px = 0.5, npts = 8, f = 600.0, estimate = esth, covariance = covh
        )(MersenneTwister(5))
        @test o_cam[2][1, 1] ≈ 0.5^2 rtol = 1.0e-12            # pixels
        @test o_def[2][1, 1] ≈ (0.5 / 600)^2 rtol = 1.0e-12    # calibrated
    end

    @testset "the hooks get what an E problem now takes" begin
        seen, esth, covh = mk_hooks()
        out = essential_trial(
            σ_px = 0.5, npts = 8, cameras = (CAM1, CAM2), estimate = esth, covariance = covh
        )(MersenneTwister(21))
        @test out !== nothing
        ch, cs, shapes = seen[]
        @test eltype(cs) <: Pair{<:StaticVector{3}, <:StaticVector{3}}
        @test hasproperty(cs, :M₁) && hasproperty(cs, :M₂)
        @test size(first(shapes)) == (4, 4)

        # THE correctness check: the scene projected through TWO DIFFERENT
        # cameras and backprojected still satisfies the E the trial anchors on.
        # If the two-camera projection were inconsistent with the returned
        # model, everything downstream would be measuring the wrong geometry.
        Em = SMatrix{3, 3, Float64}(model(ch))
        @test maximum(abs(dot(c.second, Em * c.first)) for c in cs) < 1.0e-13   # measured 1.7e-16

        # bearings are unit, and M is tangent — the contract consumers rely on
        @test maximum(abs(norm(c.first) - 1) for c in cs) < 1.0e-12
        @test maximum(abs(norm(c.second) - 1) for c in cs) < 1.0e-12
        for i in eachindex(cs)
            @test maximum(abs.(cs.M₁[i]' * SVector{3}(cs[i].first))) < 1.0e-15
            @test maximum(abs.(cs.M₂[i]' * SVector{3}(cs[i].second))) < 1.0e-15
        end
        # The bearings ARE the cameras' own backprojections: project each back
        # to pixels through its own camera and backproject again — a round trip
        # that only closes if the right camera produced it. Doing this with two
        # DIFFERENT focals is what makes it a test rather than an identity.
        for i in eachindex(cs)
            u1 = project(CAM1, SVector{3}(cs[i].first))
            u2 = project(CAM2, SVector{3}(cs[i].second))
            @test backproject(CAM1, SVector{2, Float64}(u1[1], u1[2])) ≈ cs[i].first atol = 1.0e-12
            @test backproject(CAM2, SVector{2, Float64}(u2[1], u2[2])) ≈ cs[i].second atol = 1.0e-12
        end
    end

    @testset "the band trial hands its hooks bearings too" begin
        seen, esth, covh = mk_hooks()
        out = epipolar_band_trial(
            :E; σ_px = 0.5, npts = 8, cameras = (CAM1, CAM2), estimate = esth, covariance = covh
        )(MersenneTwister(33))
        @test out !== nothing
        _, cs, _ = seen[]
        @test eltype(cs) <: Pair{<:StaticVector{3}, <:StaticVector{3}}
        @test hasproperty(cs, :M₁) && hasproperty(cs, :M₂)
        @test maximum(abs(norm(c.first) - 1) for c in cs) < 1.0e-12
    end

    @testset "camera-aware mode runs on the library's own estimate and covariance" begin
        # The chart machinery reads a bearing set's M₁/M₂ columns — the tangent
        # form of `constraint_with_jacobians` — so camera-aware mode needs no hook.
        out = essential_trial(σ_px = 0.5, npts = 8, cameras = (CAM1, CAM2))(MersenneTwister(41))
        @test out !== nothing
        @test all(isfinite, out[1]) && all(isfinite, out[2])
        @test epipolar_band_trial(:E; σ_px = 0.5, npts = 8, cameras = (CAM1, CAM2))(MersenneTwister(43)) !== nothing

        # The per-pair factors ARE the tangent form, read from the set's columns.
        seen, esth, covh = mk_hooks()
        essential_trial(σ_px = 0.5, npts = 8, cameras = (CAM1, CAM2), estimate = esth, covariance = covh)(MersenneTwister(21))
        ch, cs, _ = seen[]
        A, B = constraint_jacobians(ch, zero(SVector{tangent_size(ch), Float64}), cs)
        for i in eachindex(cs)
            _, Gx, _ = constraint_with_jacobians(model(ch), cs[i].first, cs[i].second; M₁ = cs.M₁[i], M₂ = cs.M₂[i])
            @test B[i, (4i - 3):(4i)] ≈ Gx rtol = 1.0e-12
        end
    end

    @testset "equal pinhole focals: the bearing path IS the calibrated path" begin
        # With both cameras at the scene's own f, pixel noise σ_px on bearings and
        # calibrated noise σ_px/f on z = 1 points are the same perturbation, and
        # the implicit covariance is invariant to each pair's row scale (the
        # bearing constraint is the plane one over ‖x̃₁‖‖x̃₂‖). So the σ²-scaled
        # covariances agree to rounding — minimal and overdetermined — and the
        # tracked estimates to the O(σ/f) weight difference of the two whitenings.
        EQ = (cam(600.0, 250.0, 250.0), cam(600.0, 250.0, 250.0))
        _, esth, _ = mk_hooks()
        for n in (5, 8)
            o_cam = essential_trial(σ_px = 0.5, npts = n, cameras = EQ, estimate = esth)(MersenneTwister(51))
            o_def = essential_trial(σ_px = 0.5, npts = n, estimate = esth)(MersenneTwister(51))
            @test o_cam[2] ≈ o_def[2] rtol = 1.0e-8
        end
        o_cam = essential_trial(σ_px = 0.5, npts = 8, cameras = EQ)(MersenneTwister(53))
        o_def = essential_trial(σ_px = 0.5, npts = 8)(MersenneTwister(53))
        @test o_cam !== nothing && o_def !== nothing
        @test o_cam[1] ≈ o_def[1] rtol = 2.0e-2
    end

    @testset "`cameras` is an E-only argument" begin
        # F is a pixel-plane model: `cameras` is meaningless for it, and saying
        # so beats accepting and ignoring the argument.
        _, esth, covh = mk_hooks()
        @test_throws ArgumentError epipolar_band_trial(
            :F; σ_px = 0.5, cameras = (CAM1, CAM2), estimate = esth, covariance = covh
        )
    end
end
