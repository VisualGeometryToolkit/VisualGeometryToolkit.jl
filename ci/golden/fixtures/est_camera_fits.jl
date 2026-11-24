# Fixtures of the camera fits of VisualGeometryToolkitEstimation's suite: the lens-family cases and the synthetic
# fisheyes of camera_regression_testsetups.jl (FitCameraLensCases, FitCameraFisheye, FitCameraNoncentral), the icra2026
# fisheye of camera_verbs_tests.jl, optical_axis_tests.jl's OffAxisSpline and FoldingSources, the RC siemens start
# camera (test/fixtures/rc_siemens_start_camera.jl), the DivisionBoards scenes (synthetic_scenes()), and
# camera_poses_tests.jl's CameraPoseScenes, SplineCameraPoseScenes, BoardScenes and RotationCenterScenes.
# Every function here builds data only; no estimator of the group is called.
using Test, StableRNGs, StaticArrays, Random
using LinearAlgebra: norm, normalize
using Rotations: Rotations
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitCore: CameraModel, Division, Lens, LiftLens, Radial, Tangential, Additive, ThinPrism, Perspective,
    Equidistant, Stereographic, Orthographic, Unified, ExtendedUnified, FieldOfView, Tilt, Even, KannalaBrandt, DoubleSphere,
    Scaramuzza, Pinhole, Ellipse, Rect, projected_radius, radial_part, view_angle_limit, monotone_radius, calibration_matrix,
    lens, SplineLens, NoncentralSplineModel, ChartSpline, BoundedCamera, Size2, px, backproject, project, LogicalIntrinsics,
    CameraCalibrationMat, EuclideanMap, exp_so3, rotation, translation, with_parameters, parameter_vector, with_directions,
    with_present, with_lens_coverage, imaged_region, valid_domain, RadialCamera, Correspondences, radial_rows, Vec3

const _ECF_DB = VgtkEstimation.synthetic_scenes().DivisionBoards

# ---------------------------------------------------------------------------------------------------------------
# FitCameraLensCases: name => (target family, source, region)
# ---------------------------------------------------------------------------------------------------------------
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "lens_regions.jl"))   # `disc`, `disc_angle`
const ECF_KB4 = KannalaBrandt{4}
const ECF_SC3 = Scaramuzza{3}
const ECF_DSL = typeof(Lens(DoubleSphere(0.5, 0.5)))
const ECF_OPENCV3 = Lens{Perspective, Radial{Even, 3, 0, Float64}}
const ECF_EQ = typeof(Lens(Equidistant()))
const ECF_DECENTERED = typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0))))
const ECF_PP = (2120.0, 1416.0)
ecf_mdiv() = CameraModel(; pp = ECF_PP, focal = 927.0, lens = Division(-0.3302, -0.0765))
const ECF_RCORNER = hypot(2120.0, 1416.0)

function ecf_lens_cases()
    pp, mdiv, rcorner = ECF_PP, ecf_mdiv(), ECF_RCORNER
    mkb = CameraModel(; pp, focal = 600.0, lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
    msc = CameraModel(; pp, focal = 927.0, lens = Scaramuzza(-0.3302, 0.02, -0.0765))
    rlit = 927.0 * monotone_radius(mdiv.lens) * 0.97
    cam900 = CameraModel(; pp, focal = 900.0, lens = Division(-0.2, -0.08))
    return [
        "KB4_mkb" => (ECF_KB4, mkb, disc_angle(mkb, view_angle_limit(radial_part(mkb.lens)))),
        "KB4_mdiv" => (ECF_KB4, mdiv, disc(mdiv, rcorner)),
        "SC3_mdiv" => (ECF_SC3, mdiv, disc(mdiv, rcorner)),
        "KB4_msc" => (ECF_KB4, msc, disc(msc, rcorner)),
        "DSL_rlit" => (ECF_DSL, mdiv, disc(mdiv, rlit)),
        "DSL_rcorner" => (ECF_DSL, mdiv, disc(mdiv, rcorner)),
        "OPENCV3" => (ECF_OPENCV3, mdiv, disc(mdiv, rcorner)),
        "OPENCV_decentered" => (ECF_DECENTERED, mdiv, disc(mdiv, rcorner)),
        "Pinhole_30deg" => (Pinhole, mdiv, disc_angle(mdiv, deg2rad(30))),
        "EQ" => (ECF_EQ, mdiv, disc(mdiv, rcorner)),
        "Division2" => (Division{2}, mdiv, disc(mdiv, rcorner)),
        "LiftLens_1_2" => (LiftLens{Radial{Even, 1, 2, Float64}}, mdiv, disc(mdiv, rcorner)),
        "Division3" => (Division{3}, mdiv, disc(mdiv, rcorner)),
        "SC3_div-0.2" => (ECF_SC3, CameraModel(; pp, focal = 927.0, lens = Division(-0.2)), disc(mdiv, rcorner)),
        "SC3_div0.05" => (ECF_SC3, CameraModel(; pp, focal = 927.0, lens = Division(0.05)), disc(mdiv, rcorner)),
        "Division" => (Division, mdiv, disc(mdiv, rcorner)),
        "Division3F64_cam900" => (Division{3, Float64}, cam900, disc(cam900, 2500.0)),
    ]
end

# camera_verbs_tests' icra2026 fisheye (Kannala–Brandt calibration) and its regions to 84° and past 90°
function ecf_icra()
    c = SVector(2138.8425928470033, 1436.604235274649)
    kb = CameraModel(; pp = (c[1], c[2]), focal = 927.4663595958699,
        lens = KannalaBrandt(-0.03371588391401308, 0.0013137235024557887, -0.0019529571546981683, 0.0001827493332399132))
    return (; kb, region = Rect(0, 0, 4256, 2848) ∩ Ellipse(c, 1250.0, 1250.0, 0.0), wide = Rect(0, 0, 4256, 2848) ∩ Ellipse(c, 1400.0, 1400.0, 0.0))
end
const ECF_RAT_TP = typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0), ThinPrism((0.0, 0.0), (0.0, 0.0)))))
const ECF_RAT = Lens{Perspective, Radial{Even, 3, 3, Float64}}
const ECF_POLY = Lens{Perspective, Radial{Even, 3, 0, Float64}}

# ---------------------------------------------------------------------------------------------------------------
# FitCameraFisheye / OffAxisSpline: a KB fisheye (f = 300 px), its 40 × 40 SplineLens camera, the lit disc
# ---------------------------------------------------------------------------------------------------------------
ecf_fisheye_model() = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
function ecf_fisheye()
    FISHEYE = ecf_fisheye_model()
    GRID = CameraModel(SplineLens, FISHEYE, Size2(1000, 1000); grid_size = (40, 40))
    DISC = Ellipse(SVector(500.0, 500.0), 480.0, 480.0, 0.0)
    return (; FISHEYE, GRID, GRID_BOUNDED = BoundedCamera(GRID, Size2(1000, 1000)), DISC, REGION = Rect(0, 0, 1000, 1000) ∩ DISC,
            BC = BoundedCamera(FISHEYE, Size2(1000, 1000)))
end
# a named function source (a closure's name would not be stable across files)
ecf_fisheye_ray(u) = backproject(ecf_fisheye_model(), u)

function ecf_off_axis()
    (; FISHEYE, GRID, DISC, REGION) = ecf_fisheye()
    Q = Rotations.RotMatrix(Rotations.AngleAxis(deg2rad(5.0), 0.6, 0.8, 0.0))
    OFF_AXIS = CameraModel(SplineLens, map(c -> Q * c, GRID.lens.backward.controls), Size2(1000, 1000))
    return (; FISHEYE, ON_AXIS = GRID, OFF_AXIS, OFF_BOUNDED = BoundedCamera(OFF_AXIS, Size2(1000, 1000)), DISC, REGION)
end

# a decentred fisheye's spline (optical_axis_tests)
function ecf_decentred_spline()
    dec = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = Lens(Equidistant(), Additive(Radial((-0.02,)), Tangential(2.0e-3, -1.5e-3))))
    return BoundedCamera(CameraModel(SplineLens, dec, Size2(1000, 1000); grid_size = (40, 40)), Size2(1000, 1000))
end
# a non-central model of the on-axis spline with zero offsets (refused: not supported yet)
function ecf_zero_noncentral()
    G = ecf_fisheye().GRID
    return NoncentralSplineModel(G.intrinsics, G.lens.backward, ChartSpline(G.lens.backward.area, fill(SVector(0.0, 0.0, 0.0), size(G.lens.backward.controls))))
end

# the circular fisheye turned 3° (an equidistant f = 250 px lens to 100°, on 1400 × 1000)
const _ECF_FISH250 = CameraModel(250.0px, SA[700.0px, 500.0px]; lens = KannalaBrandt(0.0))
const _ECF_Q3 = Rotations.RotMatrix(Rotations.AngleAxis(deg2rad(3.0), 1.0, -0.4, 0.0))
ecf_circular_ray(u) = (d = backproject(_ECF_FISH250, u); acos(clamp(d[3], -1, 1)) <= deg2rad(100.0) ? _ECF_Q3 * d : nothing)
function ecf_circular_fisheye()
    spl = CameraModel(SplineLens, ecf_circular_ray, Size2(1400, 1000); grid_size = (30, 22), init = :quasi_interpolate)
    circle = Ellipse(SVector(700.0, 500.0), 0.95 * 250 * deg2rad(100.0), 0.95 * 250 * deg2rad(100.0), 0.0)
    return (; spl, circle)
end

# FitCameraNoncentral: the non-central fisheye (`noncentral_central`, `noncentral_fisheye`): the suite's fixture file
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "noncentral_fisheye.jl"))

# the RC siemens start camera (a re-gridded Schöps calibration) and its support
function ecf_siemens()
    c = include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "rc_siemens_start_camera.jl"))
    C = [SVector{3, Float64}(c.controls[(i - 1) * c.gw + j]) for i in 1:c.gh, j in 1:c.gw]
    K = SMatrix{3, 3, Float64}(c.fx, 0.0, 0.0, c.skew, c.fy, 0.0, c.cx, c.cy, 1.0)
    cam = BoundedCamera(CameraModel(LogicalIntrinsics(CameraCalibrationMat(K; check = false)), SplineLens(ChartSpline(Rect(c.area...), C))),
                        Size2(c.width, c.height))
    return (; cam, support = Rect(1616.0, 593.0, 2159.0, 2321.0))
end

# FoldingSources: a SplineLens ray field θof(r) about (512, 494), K re-gauged to (500, 500), bounded by a 478 px disc
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "folding_sources.jl"))   # `folding_source`
ecf_folding_sources() = [
    "fold25" => folding_source(r -> theta_fold(r; turn = deg2rad(25.0))),
    "fold19.5_eps0.02" => folding_source(r -> theta_fold(r; turn = deg2rad(19.5)); ε = 0.02),
    "slope_collapse" => folding_source(theta_collapse),
    "plain_pinhole" => BoundedCamera(CameraModel(SplineLens, CameraModel(; pp = (512.0, 494.0), focal = 600.0), Size2(1000, 1000); grid_size = (30, 30)),
                                     Size2(1000, 1000)),
]

# the Schöps central-generic source on 640 × 480 (7 × 6 nodes, calibrated over the whole sensor)
function ecf_schoeps()
    controls = [normalize(SVector(0.12 * (gx - 4), 0.12 * (gy - 3.5), 1.0)) for gy in 1:6, gx in 1:7]
    return BoundedCamera(CameraModel(SplineLens, controls, Size2(640, 480); calib_area = Rect(0, 0, 640, 480), unit_controls = true), Size2(640, 480))
end

# GRID with its three left control columns set to the fourth (a degenerate back-projection Jacobian at the left edge)
function ecf_degenerate_grid()
    GRID = ecf_fisheye().GRID
    C = copy(lens(GRID).backward.controls)
    for c in 1:3
        C[:, c] .= C[:, 4]
    end
    return BoundedCamera(with_directions(GRID, C), Size2(1000, 1000))
end

# ---------------------------------------------------------------------------------------------------------------
# DivisionBoards: the six boards, their radial cameras, and their views
# ---------------------------------------------------------------------------------------------------------------
ecf_board_cameras() = [_ECF_DB.board_camera(Rotations.RotXYZ(a...), t) for (a, t) in zip(_ECF_DB.BOARD_ANGLES, _ECF_DB.BOARD_TS)]
ecf_radial_cameras() = Dict(enumerate(RadialCamera.(ecf_board_cameras())))
ecf_board_views(rng; σ = 0.4, ε = 0.0) = Dict(enumerate(first(_ECF_DB.noisy_board_views(rng, ecf_board_cameras(); σ, ε))))
ecf_first_board_pixels() = _ECF_DB.board_pixels(_ECF_DB.board_camera(Rotations.RotXYZ(_ECF_DB.BOARD_ANGLES[1]...), _ECF_DB.BOARD_TS[1]))

# ---------------------------------------------------------------------------------------------------------------
# CameraPoseScenes, SplineCameraPoseScenes, BoardScenes, RotationCenterScenes (camera_poses_tests.jl)
# ---------------------------------------------------------------------------------------------------------------
# the scenes (`pinhole_camera`, `fisheye_camera`, `truth_poses`, `perturbed_pose`, `views_of`, `spline_truth`,
# `perturbed_controls`, `spline_views`, `board_points`, `truth_boards`, `view_poses`, `board_views`, `nudged`,
# `center_truth`, `center_views`): the suite's fixture file
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "camera_pose_scenes.jl"))
ecf_near_pose(E, v) = EuclideanMap(Rotations.RotMatrix{3, Float64}(exp_so3(SVector(0.002, -0.0015v / 4, 0.002)) * E.R), E.t + SVector(0.004, -0.003, 0.005 - 0.001v))
# σ = 0.2 px detections, 8 % of them moved 20–60 px (the robust-refine item's draw, from a StableRNG)
ecf_contaminate(rng, views) = [[X => u + 0.2 * SVector(randn(rng), randn(rng)) +
                                    (rand(rng) < 0.08 ? (20 + 40rand(rng)) * normalize(SVector(randn(rng), randn(rng))) : zero(SVector{2}))
                                for (X, u) in cs] for cs in views]
# the held-SplineLens robust item's draw: outliers only among the inner 80 per view
ecf_contaminate_inner(rng, views) = [[(k <= 80 && rand(rng) < 0.08) ? (X => u + 0.2 * SVector(randn(rng), randn(rng)) + (20 + 40rand(rng)) * normalize(SVector(randn(rng), randn(rng)))) :
                                          (X => u + 0.2 * SVector(randn(rng), randn(rng)))
                                      for (k, (X, u)) in enumerate(cs)] for cs in views]

ecf_bounded(c) = BoundedCamera(c, Size2(1280, 960))
ecf_kb_spline_quasi() = CameraModel(SplineLens, ecf_kb420_ray, Size2(1280, 960); init = :quasi_interpolate)
ecf_kb420_ray(u) = backproject(fisheye_camera(), u)
function ecf_scaled_spline()
    t = spline_truth(); bt = lens(t).backward
    return CameraModel(t.intrinsics, SplineLens(ChartSpline(bt.area, map(c -> 2.0 * c, bt.controls))))
end
function ecf_masked_spline()
    present = trues(10, 12); present[1:3, 9:12] .= false
    return with_present(perturbed_controls(spline_truth(), 0.01), present)
end
function ecf_flat_spline()
    truth = spline_truth(); b = lens(truth).backward
    return CameraModel(truth.intrinsics, SplineLens(ChartSpline(b.area, fill(SVector(0.0, 0.0, 1.0), size(b.controls)))))
end

# the board views, the noise drawn in sorted (view, board) order from a StableRNG (the suite draws from Xoshiro over a Dict)
ecf_board_obs(camera, views, boards; σ = 0.0, rng = StableRNG(1)) = board_views(camera, views, boards; σ, rng, order = sort!(collect(keys(views))))
const ECF_CENTER_AXES = [SVector(0.0, 0.0, 0.0), SVector(0.08, 0.0, 0.0), SVector(0.0, 0.08, 0.0), SVector(-0.06, 0.05, 0.03),
                         SVector(0.05, -0.07, -0.04), SVector(0.0, 0.0, 0.1)]
ecf_center_start(t) = VgtkEstimation.RotationCenterViews(Dict(k => SMatrix{3, 3, Float64, 9}(exp_so3(SVector(0.002, -0.001, 0.0015))) * R for (k, R) in t.R),
                                                         t.c + SVector(0.01, -0.008, 0.012), t.d + SVector(-0.005, 0.004, 0.006))
