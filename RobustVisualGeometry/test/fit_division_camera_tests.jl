@testitem "fit_division_camera" setup = [DivisionBoards] begin
# fit_division_camera — iterated Sampson-whitened, Student-t re-weighted focal stack
using Test, Random, LinearAlgebra
using StaticArrays: SVector, SMatrix, @SMatrix
using Rotations: RotXYZ
using VisualGeometryCore: HomographyMat, RadialCamera, CameraModel, focal_length, intrinsics, lens,
    NoWhitening, SampsonWhitening
using VisualGeometryCore: estimate, lens_parameters, retcode, iterations, rms   # VGC accessors (public)
using VisualGeometryCore: Csponds, calibration_matrix, radial_rows, CspondNormalization
using RobustVisualGeometry: DivisionCameraState, DivisionCameraError
using RobustVisualGeometry: fit_division_camera, find_radial_camera, division_camera_step


@testset "fit_division_camera (IRLS on the linear focal stack)" begin
    rng = MersenneTwister(3)
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
    N = CspondNormalization(DivisionCameraState, vw; center = SVector(0.0, 0.0))   # the fit's own
    plain = CameraModel(division_camera_step(N, rc, vw))
    res = fit_division_camera(rc, vw; image_size = sz)
    f_plain = focal_length(intrinsics(plain))[1]; f_irls = focal_length(intrinsics(estimate(res)))[1]
    @test abs(f_irls - f0) < abs(f_plain - f0)                        # whitening + robust weights beat the plain stack
    @test abs(f_irls - f0) / f0 < 1.0e-2                                  # information limit under 8 % contamination
    @test abs(lens_parameters(lens(estimate(res)))[1] / f_irls^2 - k0) / abs(k0) < 0.05
    @test retcode(res) in (:converged, :max_iterations)
    @test 0.2 < res.sigma < 1.0                                        # ŝ recovers the pixel noise scale
    @test sort!(collect(keys(res.views))) == 1:6
    w = reduce(vcat, [res.views[i].weights for i in 1:6])               # posterior weights, as the views' column
    @test count(<(0.2), w) >= 0.05 * length(w)                          # the outliers are down-weighted
    @test all(i -> res.views[i].weights isa Vector{Float64} && length(res.views[i]) == length(boards[i]), 1:6)
    # a unit `weights` column is a no-op prior
    ones_vw = Dict(i => Csponds(first.(b), last.(b); weights = ones(length(b))) for (i, b) in enumerate(boards))
    @test focal_length(intrinsics(estimate(fit_division_camera(rc, ones_vw; image_size = sz))))[1] ≈ f_irls rtol = 1.0e-10
    # the key sets must agree; the image size is required unless a seed is
    # given; a seed is a positive focal
    @test_throws ArgumentError fit_division_camera(rc, Dict(i => boards[i] for i in 1:5); image_size = sz)
    @test_throws ArgumentError fit_division_camera(rc, vw)
    @test_throws ArgumentError fit_division_camera(rc, vw; image_size = sz, focal_seed = -1.0)
    # the budget is max_iter, the branch audit and its resumed IRLS included
    # (this stack's 8 % gross outliers do move branches)
    for mi in (0, 1, 2, 5)
        @test iterations(fit_division_camera(rc, vw; image_size = sz, max_iter = mi)) <= mi
    end
    @test retcode(fit_division_camera(rc, vw; image_size = sz, max_iter = 0)) === :max_iterations
    # iteration 0 is the step whitened at the seed (the half-diagonal by default), bit for bit
    p0 = estimate(fit_division_camera(rc, vw; image_size = sz, max_iter = 0))
    s0 = CameraModel(division_camera_step(N, rc, vw; whiten_at = 2500.0))
    @test calibration_matrix(p0) == calibration_matrix(s0)
    @test lens_parameters(lens(p0)) == lens_parameters(lens(s0))
    # with the seed given in pixels the image size does not enter the result, and
    # the normalization is conditioning: nor does where the pixels' origin is
    fA = focal_length(intrinsics(estimate(fit_division_camera(rc, vw; image_size = sz, focal_seed = 1800.0))))[1]
    fB = focal_length(intrinsics(estimate(fit_division_camera(rc, vw; image_size = (8000, 6000), focal_seed = 1800.0))))[1]
    @test fA == fB
    fS = focal_length(intrinsics(estimate(fit_division_camera(rc, vw; focal_seed = 1800.0))))[1]
    @test fS == fA                                                          # no image size needed with a seed
    c = SVector(1990.0, 1510.0)
    rcc = Dict(i => RadialCamera{Float64}(radial_rows(P), c) for (i, P) in rc)
    vwc = Dict(i => [X => x + c for (X, x) in b] for (i, b) in vw)
    resc = fit_division_camera(rcc, vwc; image_size = sz, focal_seed = 1800.0)
    @test focal_length(intrinsics(estimate(resc)))[1] ≈ fA rtol = 1.0e-8
    @test calibration_matrix(estimate(resc))[1:2, 3] == c                # the principal point is the inputs' centre, exact
    # radial cameras straight from the correspondences (the BabelCalib route)
    # work too — ROBUSTLY estimated, as the callers estimate them. `fit_division_camera`
    # takes its cameras as given and no longer re-fits them per iteration (that
    # could not improve a nonlinear fit and could destroy one — see the
    # docstring), so a plain DLT over these 8 % gross outliers is garbage in.
    # centre (0, 0): these pixels are already centred (K has its principal
    # point at the origin)
    Ps = [
        find_radial_camera(
                b; center = (0.0, 0.0),
                threshold = 5.0, rng = MersenneTwister(9)
            ).estimate for b in boards
    ]
    @test all(P -> P isa RadialCamera, Ps)
    res2 = fit_division_camera(Dict(enumerate(Ps)), vw; image_size = sz)
    @test abs(focal_length(intrinsics(estimate(res2)))[1] - f0) / f0 < 1.0e-2
    # k fixed: the known-coefficient path
    res3 = fit_division_camera(rc, vw; image_size = sz, k = k0)
    @test abs(focal_length(intrinsics(estimate(res3)))[1] - f0) / f0 < 5.0e-3
    # a view weighted to zero: it leaves the fit, and its rows leave the rms
    # (they used to enter it at the posterior's 1e12 sentinel: rms ~4e5 px)
    zw = Dict(i => Csponds(first.(b), last.(b); weights = (i == 6 ? zeros : ones)(length(b))) for (i, b) in enumerate(boards))
    resz = fit_division_camera(rc, zw; image_size = sz)
    @test isfinite(rms(resz)) && rms(resz) < 50.0
    @test resz.n <= sum(length, boards[1:5])
    # weights: negative is an argument error; all zero is the data condition,
    # about a centre off the origin too (it used to blame the normalization)
    neg = Dict(i => Csponds(first.(b), last.(b); weights = (i == 2 ? fill(-1.0, length(b)) : ones(length(b)))) for (i, b) in enumerate(boards))
    @test_throws ArgumentError fit_division_camera(rc, neg; image_size = sz)
    allz = Dict(i => Csponds(first.(b), last.(b); weights = zeros(length(b))) for (i, b) in vwc)
    @test_throws DivisionCameraError fit_division_camera(rcc, allz; focal_seed = 1800.0)
end
end
