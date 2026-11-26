@testitem "DivisionCameraFit" setup = [DivisionBoards] begin
    import VisualGeometryToolkitCore as VgtkCore
    
# DivisionCameraFit — iterated Sampson-whitened, Student-t re-weighted focal stack
using Test, Random, LinearAlgebra
using StableRNGs
using StaticArrays: SVector, SMatrix, @SMatrix
using Rotations: RotXYZ
using VisualGeometryToolkitCore: HomographyMat, RadialCamera, CameraModel, focal_length, intrinsics, lens
using VisualGeometryToolkitCore: parameter_vector, noise_scale
using VisualGeometryToolkitCore: Correspondences, calibration_matrix, radial_rows, CorrespondenceNormalization
using VisualGeometryToolkitCore: MsacScoring
using VisualGeometryToolkitEstimation: DivisionCameraState, DivisionCameraError, DivisionCameraMetadata
using VisualGeometryToolkitEstimation: DivisionCameraFit, Ransac, mixture_em_from_toml, default_config_path
using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
using TOML: TOML
# the fit's EM table at another iteration cap (`em_at`): the suite's fixture file, shared with the golden records
import VisualGeometryToolkitEstimation as VgtkEstimation
include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "division_camera_em.jl"))


@testset "DivisionCameraFit (IRLS on the linear focal stack)" begin
    rng = StableRNG(3)
    f0, k0 = F0, K0
    # boards at very different depths (BOARD_TS): the algebraic rows of the far
    # ones dominate an unwhitened stack — this is what the whitening corrects
    cams = [board_camera(RotXYZ(a...), t) for (a, t) in zip(BOARD_ANGLES, BOARD_TS)]
    Hs = RadialCamera.(cams)
    boards, _ = noisy_board_views(rng, cams; σ = 0.4, ε = 0.08)                     # 0.4 px pixel noise, 8 % gross outliers
    # keyed by the view's identity (here the board index); the pixels are
    # centred (K's principal point at the origin) in a 4000 × 3000 image:
    # half-diagonal 2500, so the default seed f = 2500 px is 2× the truth
    rc, vw = Dict(enumerate(Hs)), Dict(enumerate(boards))
    sz = (4000, 3000)
    # one-shot, algebraic, unweighted: the plain stack
    N = CorrespondenceNormalization(DivisionCameraState, vw; center = SVector(0.0, 0.0))   # the fit's own
    plain = CameraModel(VgtkEstimation._keyed_focal_solve(N, rc, vw))
    res = solve(EstimationProblem(CameraModel, (rc, vw), (; image_size = sz)), DivisionCameraFit())
    f_plain = focal_length(intrinsics(plain))[1]; f_irls = focal_length(intrinsics(res.u))[1]
    @test abs(f_irls - f0) < abs(f_plain - f0)                        # whitening + robust weights beat the plain stack
    @test abs(f_irls - f0) / f0 < 1.0e-2                                  # information limit under 8 % contamination
    @test abs(parameter_vector(lens(res.u))[1] / f_irls^2 - k0) / abs(k0) < 0.05
    @test (res.retcode, res.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.ParameterChange)
    @test 0.2 < noise_scale(res) < 1.0                                        # ŝ recovers the pixel noise scale
    @test @inferred(noise_scale(res)) isa Float64
    md = res.metadata
    @test @inferred(DivisionCameraMetadata(md.state, md.radial_cameras, md.views, md.weights, md.scale, md.scale_source,
                                           md.ν)) isa DivisionCameraMetadata
    @test sort!(collect(keys(res.metadata.views))) == 1:6
    w = reduce(vcat, [res.metadata.views[i].weights for i in 1:6])               # posterior weights, as the views' column
    @test count(<(0.2), w) >= 0.05 * length(w)                          # the outliers are down-weighted
    @test all(i -> res.metadata.views[i].weights isa Vector{Float64} && length(res.metadata.views[i]) == length(boards[i]), 1:6)
    # a unit `weights` column is a no-op prior
    ones_vw = Dict(i => Correspondences(first.(b), last.(b); weights = ones(length(b))) for (i, b) in enumerate(boards))
    @test focal_length(intrinsics(solve(EstimationProblem(CameraModel, (rc, ones_vw), (; image_size = sz)), DivisionCameraFit()).u))[1] ≈ f_irls rtol = 1.0e-10
    # the key sets must agree; the image size is required unless a seed is
    # given; a seed is a positive focal
    @test_throws ArgumentError solve(EstimationProblem(CameraModel, (rc, Dict(i => boards[i] for i in 1:5)), (; image_size = sz)), DivisionCameraFit())
    @test_throws ArgumentError solve(EstimationProblem(CameraModel, (rc, vw)), DivisionCameraFit())
    @test_throws ArgumentError solve(EstimationProblem(CameraModel, (rc, vw), (; image_size = sz, focal_seed = -1.0)), DivisionCameraFit())
    # the budget is the EM's max_iter, the branch audit and its resumed IRLS included
    # (this stack's 8 % gross outliers do move branches)
    for mi in (1, 2, 5)
        @test solve(EstimationProblem(CameraModel, (rc, vw), (; image_size = sz)), DivisionCameraFit(; mixture_em = em_at(mi))).stats.nsteps <= mi
    end
    @test (let r = solve(EstimationProblem(CameraModel, (rc, vw), (; image_size = sz)), DivisionCameraFit(; mixture_em = em_at(1))); (r.retcode, r.stop_rule) end) === (VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap)
    # one iteration moves the solution off the seeded step (the step whitened at the
    # seed, the half-diagonal by default): the robust re-weighting begins there
    p1 = solve(EstimationProblem(CameraModel, (rc, vw), (; image_size = sz)), DivisionCameraFit(; mixture_em = em_at(1))).u
    s0 = CameraModel(VgtkEstimation._keyed_focal_solve(N, rc, vw; whiten_at = 2500.0))
    @test calibration_matrix(p1) != calibration_matrix(s0)
    # with the seed given in pixels the image size does not enter the result, and
    # the normalization is conditioning: nor does where the pixels' origin is
    fA = focal_length(intrinsics(solve(EstimationProblem(CameraModel, (rc, vw), (; image_size = sz, focal_seed = 1800.0)), DivisionCameraFit()).u))[1]
    fB = focal_length(intrinsics(solve(EstimationProblem(CameraModel, (rc, vw), (; image_size = (8000, 6000), focal_seed = 1800.0)), DivisionCameraFit()).u))[1]
    @test fA == fB
    fS = focal_length(intrinsics(solve(EstimationProblem(CameraModel, (rc, vw), (; focal_seed = 1800.0)), DivisionCameraFit()).u))[1]
    @test fS == fA                                                          # no image size needed with a seed
    c = SVector(1990.0, 1510.0)
    rcc = Dict(i => RadialCamera{Float64}(radial_rows(P), c) for (i, P) in rc)
    vwc = Dict(i => [X => x + c for (X, x) in b] for (i, b) in vw)
    resc = solve(EstimationProblem(CameraModel, (rcc, vwc), (; image_size = sz, focal_seed = 1800.0)), DivisionCameraFit())
    @test focal_length(intrinsics(resc.u))[1] ≈ fA rtol = 1.0e-8
    @test calibration_matrix(resc.u)[1:2, 3] == c                # the principal point is the inputs' centre, exact
    # radial cameras straight from the correspondences (the BabelCalib route)
    # work too — ROBUSTLY estimated, as the callers estimate them. `DivisionCameraFit`
    # takes its cameras as given and no longer re-fits them per iteration (that
    # could not improve a nonlinear fit and could destroy one — see the
    # docstring), so a plain DLT over these 8 % gross outliers is garbage in.
    # centre (0, 0): these pixels are already centred (K has its principal
    # point at the origin)
    Ps = [
        solve(EstimationProblem(RadialCameraProblem(b; center = (0.0, 0.0))),
              Ransac(; scoring = MsacScoring(RadialCameraProblem; threshold = 5.0)); rng = StableRNG(9)).u for b in boards
    ]
    @test all(P -> P isa RadialCamera, Ps)
    res2 = solve(EstimationProblem(CameraModel, (Dict(enumerate(Ps)), vw), (; image_size = sz)), DivisionCameraFit())
    @test abs(focal_length(intrinsics(res2.u))[1] - f0) / f0 < 1.0e-2
    # k fixed: the known-coefficient path
    res3 = solve(EstimationProblem(CameraModel, (rc, vw), (; image_size = sz, k = k0)), DivisionCameraFit())
    @test abs(focal_length(intrinsics(res3.u))[1] - f0) / f0 < 5.0e-3
    # a view weighted to zero: it leaves the fit, and its rows leave the rms
    # (they used to enter it at the posterior's 1e12 sentinel: rms ~4e5 px)
    zw = Dict(i => Correspondences(first.(b), last.(b); weights = (i == 6 ? zeros : ones)(length(b))) for (i, b) in enumerate(boards))
    resz = solve(EstimationProblem(CameraModel, (rc, zw), (; image_size = sz)), DivisionCameraFit())
    @test isfinite(resz.rms) && resz.rms < 50.0
    @test resz.n <= sum(length, boards[1:5])
    # weights: negative is an argument error; all zero is the data condition,
    # about a centre off the origin too (it used to blame the normalization)
    neg = Dict(i => Correspondences(first.(b), last.(b); weights = (i == 2 ? fill(-1.0, length(b)) : ones(length(b)))) for (i, b) in enumerate(boards))
    @test_throws ArgumentError solve(EstimationProblem(CameraModel, (rc, neg), (; image_size = sz)), DivisionCameraFit())
    allz = Dict(i => Correspondences(first.(b), last.(b); weights = zeros(length(b))) for (i, b) in vwc)
    @test_throws DivisionCameraError solve(EstimationProblem(CameraModel, (rcc, allz), (; focal_seed = 1800.0)), DivisionCameraFit())
end
end
