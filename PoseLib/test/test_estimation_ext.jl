# PoseLibEstimationExt: PoseLib's backend refines of VisualGeometryToolkitEstimation's two-view family problems, so a
# RANSAC's local optimization can be Core's `BackendRefit(LevenbergMarquardt(PoseLibSolver(); optimization_strategy),
# stops)`. The family refines are PoseLib's raw-data refines on the problem's own correspondences, bit for bit; the
# essential refit is the seed's cheirality-resolved pose refined camera-aware on the frame's calibrations and pixels.

import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
using Test
using Random: MersenneTwister
using StaticArrays: SMatrix
using PoseLib
using PoseLib: LeZach
using VisualGeometryToolkitCore: BackendRefit, CameraCalibrationMat, CameraModel, DLT, EssentialMat, EuclideanMap,
    FundMat, HomographyMat, Irls, Lens, LevenbergMarquardt, LmStops, LogicalIntrinsics, LossWidth, NullParameters,
    Perspective, PoseLibSolver, RefineProblem, TLSShape, Vec3, cheiral_pose_fast, solve, sources, targets
using VisualGeometryToolkitEstimation: EssentialProblem, FundMatProblem, HomographyProblem

# `contract_scene` (fixtures/refine_scenes.jl) is test_contract.jl's, which runtests.jl includes first.

@testset "PoseLibEstimationExt: the family problems' backend refines and the BackendRefit routes" begin
    @test Base.get_extension(PoseLib, :PoseLibEstimationExt) isa Module
    d = contract_scene(MersenneTwister(99))
    w = 2.0
    stops = LmStops(; max_iter = 25)
    for strategy in (Irls{TLSShape}(), LeZach())
        alg = LevenbergMarquardt(PoseLibSolver(); optimization_strategy = strategy)
        lo = BackendRefit(alg, stops)
        # H and F: the problem's refine is the raw-data refine on its correspondences, bit for bit
        for (P, G) in ((HomographyProblem(d.hom), HomographyMat(d.H0)), (FundMatProblem(d.rel), FundMat(d.F0)))
            data = P isa HomographyProblem ? d.hom : d.rel
            raw = solve(RefineProblem(data, G), alg; stops, width = w)
            fam = solve(RefineProblem(P, G), alg; stops, width = w)
            @test fam.u === raw.u && (fam.retcode, fam.stop_rule) === (raw.retcode, raw.stop_rule) &&
                fam.stats.nsteps == raw.stats.nsteps
            # BackendRefit's refit is the refined model (a FitResult's `u`), whatever the record
            @test VgtkCore.external_refit!(nothing, lo, P, G, LossWidth(w), nothing) === raw.u
        end
        # the budget spent is a normal stop: the model still comes back
        one = BackendRefit(alg, LmStops(; max_iter = 1))
        @test solve(RefineProblem(HomographyProblem(d.hom), HomographyMat(d.H0)), alg; stops = one.stops, width = w).retcode ===
            VgtkCore.ReturnCode.MaxIters
        @test VgtkCore.external_refit!(nothing, one, HomographyProblem(d.hom), HomographyMat(d.H0), LossWidth(w), nothing) isa
            HomographyMat
        # E: the frame's calibrations and pixels, the seed decomposed on a strided subsample of the problem's bearings
        cam1 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(d.K1)), Lens(Perspective()))
        cam2 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(d.K2)), Lens(Perspective()))
        ep = EssentialProblem(cam1, cam2, d.rel)
        frame = (; Kc1 = d.K1, Kc2 = d.K2, cs = d.rel)
        seed_E = EssentialMat(d.seed.R, d.seed.t)
        for cap in (256, 50)
            le = BackendRefit(alg, stops; cheirality_vote_points = cap)
            ws = VgtkCore.external_workspace(le, ep, frame, nothing)
            f₁, f₂ = sources(ep.cs), targets(ep.cs)
            idx = 1:max(1, length(f₁) ÷ cap):length(f₁)
            R0, t0 = cheiral_pose_fast(seed_E, view(f₁, idx), view(f₂, idx))
            u = solve(RefineProblem(d.rel, EuclideanMap(R0, Vec3{Float64}(t0)),
                                    (; K1 = SMatrix{3, 3, Float64, 9}(d.K1), K2 = SMatrix{3, 3, Float64, 9}(d.K2))),
                      alg; stops, width = w).u
            @test VgtkCore.external_refit!(ws, le, ep, seed_E, LossWidth(w), nothing) === EssentialMat(u.R, u.t)
        end
    end
    # PoseLib honours max_iter alone: Core refuses any other stop setting before the refine runs
    @test VgtkCore.honoured_stops(PoseLibSolver()) === (:max_iter,)
    lsq = LevenbergMarquardt(PoseLibSolver(); optimization_strategy = nothing)
    @test_throws r"honours only max_iter" solve(RefineProblem(HomographyProblem(d.hom), HomographyMat(d.H0)), lsq;
                                                stops = LmStops(; ftol = 1.0e-3))
    # BackendRefit runs a refine algorithm only, and its subsample is at least one point
    @test_throws r"refine algorithm" BackendRefit(DLT(), stops)
    @test_throws ArgumentError BackendRefit(lsq, stops; cheirality_vote_points = 0)
    @test BackendRefit(lsq, stops).cheirality_vote_points == 256
    # its identity is its algorithm's strategy's (the evaluation's labels and markers); its scale is the given width
    tls = BackendRefit(LevenbergMarquardt(PoseLibSolver(); optimization_strategy = Irls{TLSShape}()), stops)
    @test VgtkCore.optimization_path(tls) === :irls && VgtkCore.loss_family(tls) === :tls &&
        VgtkCore.lo_label(tls) == "TLS-IRLS" && VgtkCore.scale_interface(tls) === VgtkCore.FixedScale() &&
        VgtkCore.lo_marker(tls) === VgtkCore.lo_marker(Irls{TLSShape}())
    lz = BackendRefit(LevenbergMarquardt(PoseLibSolver(); optimization_strategy = LeZach()), stops)
    @test VgtkCore.optimization_path(lz) === :gnc && VgtkCore.loss_family(lz) === :tls && VgtkCore.lo_label(lz) == "TLS-LZ" &&
        VgtkCore.scale_interface(lz) === VgtkCore.FixedScale()
end
