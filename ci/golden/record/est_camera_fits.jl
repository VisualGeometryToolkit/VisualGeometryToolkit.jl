# The old camera fits at 3d8a06da: fit_camera, pspline_grid, fit_forward_spline, optical_axis, refine_camera,
# approximate_camera, fit_division_camera, fit_radial_camera and refine_camera_and_poses (the views form, the board form
# and the rotation-centre form), on the CPU default device. Fixtures, variants and keywords are the suite's
# (fit_camera_tests, camera_verbs_tests, optical_axis_tests, spline_fit_sources_tests, termination_record_tests,
# fit_division_camera*_tests, division_solve_golden, fit_radial_camera_tests, units_boundary_tests,
# refine_camera_and_poses_tests); a fit's output that seeds another solve is computed here and stored in the fixture.
using VisualGeometryToolkitCore: estimate, metadata, parameter_names, FitResult, ReturnCode, ClosedForm, ScaleEstimated, Triggs, Irls,
    HuberShape, WelschShape, TLSShape, StudentTUniformScoring, JeffreysPrior, FlatLabelPrior, MsacScoring, CorrespondenceNormalization,
    with_unit, HomographyMat
using VisualGeometryToolkitEstimation: fit_camera, refine_camera, approximate_camera, optical_axis, pspline_grid, fit_forward_spline,
    fit_division_camera, fit_radial_camera, refine_camera_and_poses, find_radial_camera, RadialCameraProblem, mixture_em_from_toml,
    default_config_path, scoring_from_toml, DivisionCameraProblem, MixtureEmConfig, KeepModel, NoImprovementStop, TerminalLO,
    BearingResidual, ReprojectionResidual, held_parameters, RotationCenterViews
using TOML: TOML

# the configured fit_division_camera EM table at another iteration cap
ecf_em_at(max_iter) = mixture_em_from_toml(merge(TOML.parsefile(default_config_path())["fit_division_camera"]["mixture_em"],
                                                 Dict("max_iter" => max_iter)); section = "fit_division_camera.mixture_em")
ecf_em() = mixture_em_from_toml(TOML.parsefile(default_config_path())["fit_division_camera"]["mixture_em"]; section = "fit_division_camera.mixture_em")

const ECF_RETIRED = (:free, :fix, :fixed_params, :hold)

function record_est_camera_fits(cases)
    _record_ecf_lens!(cases)
    _record_ecf_spline!(cases)
    _record_ecf_axis!(cases)
    _record_ecf_division!(cases)
    _record_ecf_radial!(cases)
    _record_ecf_bundle!(cases)
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# parametric lens families: the linear fit, its refine, the two in one
# ---------------------------------------------------------------------------------------------------------------
function _record_ecf_lens!(cases)
    mdiv, rc = ecf_mdiv(), ECF_RCORNER
    for (name, (L, m, region)) in ecf_lens_cases()
        case!(cases, "lensfit/$name/linear", (m, region), ((m, region),) -> fit_camera(L, m, region))
        case!(cases, "lensapprox/$name/default", (m, region), ((m, region),) -> approximate_camera(L, m, region))
    end
    md = (mdiv, ecf_disc(mdiv, rc))
    case!(cases, "lensfit/KB4_mdiv/weight_image", md, ((m, r),) -> fit_camera(ECF_KB4, m, r; weight = :image))
    case!(cases, "lensfit/KB4_mdiv/weight_solid_angle", md, ((m, r),) -> fit_camera(ECF_KB4, m, r; weight = :solid_angle))
    case!(cases, "lensfit/KB4_mdiv/refused_weight_pixels", md, ((m, r),) -> fit_camera(ECF_KB4, m, r; weight = :pixels))
    case!(cases, "lensfit/orthographic_mdiv/refused_view_angle", md, ((m, r),) -> fit_camera(typeof(Lens(Orthographic())), m, r))
    case!(cases, "lensfit/Division1_mdiv/linear", md, ((m, r),) -> fit_camera(Division{1}, m, r))
    case!(cases, "lensfit/LiftLens_mdiv/refused_no_arity", md, ((m, r),) -> fit_camera(LiftLens, m, r))
    case!(cases, "lensfit/KannalaBrandt_mdiv/refused_no_arity", md, ((m, r),) -> fit_camera(KannalaBrandt, m, r))
    case!(cases, "lensfit/DECENTERED_mdiv1000/linear", (mdiv, ecf_disc(mdiv, 1000.0)), ((m, r),) -> fit_camera(ECF_DECENTERED, m, r))
    synth = CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = Lens(Perspective(), Additive(Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)),
        Tangential(1.0e-3, -5.0e-4), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3)))))
    case!(cases, "lensfit/OPENCV3_decentred_source/stripped", (synth, ecf_disc(synth, 500.0)), ((m, r),) -> fit_camera(ECF_OPENCV3, m, r))
    frame = Rect(0, 0, 4240, 2832)
    case!(cases, "lensfit/KB4_mdiv_frame/linear", (mdiv, frame), ((m, r),) -> fit_camera(ECF_KB4, m, r))
    case!(cases, "lensfit/KB4_mdiv/refused_off_pp_region", (mdiv, Rect(0, 0, 100, 100)), ((m, r),) -> fit_camera(ECF_KB4, m, r))
    case!(cases, "lensfit/KB4_mdiv/refused_whole_plane", (mdiv, VgtkCore.WholePlane()), ((m, r),) -> fit_camera(ECF_KB4, m, r))
    kb200 = CameraModel(200.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
    case!(cases, "lensfit/KB4_kb200/refused_beyond_support", (kb200, Rect(0, 0, 1000, 1000)), ((m, r),) -> fit_camera(ECF_KB4, m, r))
    kb135 = CameraModel(SMatrix{3, 3, Float64}(300, 0, 0, 0, 300, 0, 320, 240, 1); lens = KannalaBrandt(-0.135))
    case!(cases, "lensfit/KB4_kb135/refused_beyond_injectivity", (kb135, Rect(-1.0e5, -1.0e5, 2.0e5, 2.0e5)), ((m, r),) -> fit_camera(ECF_KB4, m, r))

    # camera_verbs: the icra2026 fisheye
    ic = ecf_icra()
    case!(cases, "lensfit/RATTP_icra84/linear", (ic.kb, ic.region), ((m, r),) -> fit_camera(ECF_RAT_TP, m, r))
    case!(cases, "lensfit/RATTP_icra_wide/refused_view_angle", (ic.kb, ic.wide), ((m, r),) -> fit_camera(ECF_RAT_TP, m, r))
    case!(cases, "lensapprox/RATTP_icra_wide/refused_view_angle", (ic.kb, ic.wide), ((m, r),) -> approximate_camera(ECF_RAT_TP, m, r))
    case!(cases, "lensfit/KB4_icra_wide/linear", (ic.kb, ic.wide), ((m, r),) -> fit_camera(ECF_KB4, m, r))
    case!(cases, "lensapprox/RATTP_icra84/default", (ic.kb, ic.region), ((m, r),) -> approximate_camera(ECF_RAT_TP, m, r))

    # the linear residual and its refine (camera_verbs: KB4, SC3, Division{2} over the 2000 px disc)
    for (lbl, L) in (("KB4", ECF_KB4), ("SC3", ECF_SC3), ("Division2", Division{2}))
        region = ecf_disc(mdiv, 2000.0)
        g = estimate(fit_camera(L, mdiv, region))
        case!(cases, "lensrefine/$(lbl)_mdiv2000/from_linear", (g, mdiv, region), ((g, m, r),) -> refine_camera(g, m, r))
    end
    # the families with no linear form: refused by fit_camera, started from their anchor by approximate_camera
    nolin = ["DSL" => ECF_DSL, "Unified" => typeof(Lens(Unified(0.0))), "ExtendedUnified" => typeof(Lens(ExtendedUnified(0.0, 0.0))),
             "FieldOfView" => typeof(Lens(Perspective(), FieldOfView(0.0))), "TiltRadial" => typeof(Lens(Perspective(), Tilt(0.0, 0.0) ∘ Radial((0.0, 0.0))))]
    d1500 = (mdiv, ecf_disc(mdiv, 1500.0))
    for (lbl, L) in nolin
        case!(cases, "lensfit/$(lbl)_mdiv1500/refused_no_linear_form", d1500, ((m, r),) -> fit_camera(L, m, r))
        case!(cases, "lensapprox/$(lbl)_mdiv1500/default", d1500, ((m, r),) -> approximate_camera(L, m, r))
    end
    # refine_camera's keywords: L2 only, no scale input; the retired keyword
    g = estimate(fit_camera(ECF_KB4, mdiv, ecf_disc(mdiv, 1500.0)))
    fx = (g, d1500...)
    case!(cases, "lensrefine/KB4_mdiv1500/from_linear", fx, ((g, m, r),) -> refine_camera(g, m, r))
    case!(cases, "lensrefine/KB4_mdiv1500/refused_scale", fx, ((g, m, r),) -> refine_camera(g, m, r; scale = 1.0))
    case!(cases, "lensrefine/KB4_mdiv1500/refused_width", fx, ((g, m, r),) -> refine_camera(g, m, r; width = 1.0))
    case!(cases, "lensrefine/KB4_mdiv1500/refused_strategy_symbol", fx, ((g, m, r),) -> refine_camera(g, m, r; optimization_strategy = :huber))
    case!(cases, "lensrefine/KB4_mdiv1500/refused_local_optimization", fx, ((g, m, r),) -> refine_camera(g, m, r; local_optimization = nothing))
    # from a perturbed guess (focal × 1.05, coefficients × 0.8 of approximate_camera's optimum)
    for (lbl, L, R) in (("KB4", ECF_KB4, rc), ("SC3", ECF_SC3, rc), ("Division2", Division{2}, 2500.0))
        region = ecf_disc(mdiv, R)
        best = estimate(approximate_camera(L, mdiv, region))
        p = collect(parameter_vector(lens(best)))
        K = SMatrix{3, 3, Float64}(calibration_matrix(best))
        guess = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1] * 1.05, 0, 0, 0, K[2, 2] * 1.05, 0, K[1, 3], K[2, 3], 1); check = false)),
            with_parameters(lens(best), p .* 0.8))
        case!(cases, "lensrefine/$(lbl)_mdiv$(round(Int, R))/perturbed_guess", (guess, mdiv, region), ((g, m, r),) -> refine_camera(g, m, r))
    end
    ga = estimate(approximate_camera(ECF_KB4, d1500...))
    K = SMatrix{3, 3, Float64}(calibration_matrix(ga))
    moved = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0, 0, 0, K[2, 2], 0, K[1, 3] + 3, K[2, 3], 1); check = false)), lens(ga))
    case!(cases, "lensrefine/KB4_mdiv1500/refused_moved_pp", (moved, d1500...), ((g, m, r),) -> refine_camera(g, m, r))

    # `fixed` on the rational target (refine) and on the linear fit
    gr = estimate(fit_camera(ECF_RAT, ic.kb, ic.region))
    fr = (gr, ic.kb, ic.region)
    case!(cases, "lensrefine/RAT_icra84/fixed_d", fr, ((g, m, r),) -> refine_camera(g, m, r; fixed = (:d,)))
    case!(cases, "lensrefine/RAT_icra84/fixed_d123", fr, ((g, m, r),) -> refine_camera(g, m, r; fixed = (:d1, :d2, :d3)))
    case!(cases, "lensrefine/RAT_icra84/default", fr, ((g, m, r),) -> refine_camera(g, m, r))
    case!(cases, "lensrefine/RAT_icra84/fixed_all", fr, ((g, m, r),) -> refine_camera(g, m, r; fixed = (:f, :k, :d)))
    case!(cases, "lensrefine/RAT_icra84/refused_unknown_fixed", fr, ((g, m, r),) -> refine_camera(g, m, r; fixed = (:xi,)))
    for kw in ECF_RETIRED
        case!(cases, "lensrefine/RAT_icra84/refused_retired_$kw", fr, ((g, m, r),) -> refine_camera(g, m, r; (kw => (:d,),)...))
        case!(cases, "lensfit/RAT_icra84/refused_retired_$kw", (ic.kb, ic.region), ((m, r),) -> fit_camera(ECF_RAT, m, r; (kw => (:d,),)...))
    end
    case!(cases, "lensapprox/RAT_icra84/default", (ic.kb, ic.region), ((m, r),) -> approximate_camera(ECF_RAT, m, r))
    case!(cases, "lensapprox/RAT_icra84/fixed_d", (ic.kb, ic.region), ((m, r),) -> approximate_camera(ECF_RAT, m, r; fixed = (:d,)))
    case!(cases, "lensfit/RAT_icra84/fixed_d", (ic.kb, ic.region), ((m, r),) -> fit_camera(ECF_RAT, m, r; fixed = (:d,)))
    case!(cases, "lensfit/RAT_icra84/linear", (ic.kb, ic.region), ((m, r),) -> fit_camera(ECF_RAT, m, r))
    case!(cases, "lensfit/POLY_icra84/linear", (ic.kb, ic.region), ((m, r),) -> fit_camera(ECF_POLY, m, r))
    grh = estimate(fit_camera(ECF_RAT, ic.kb, ic.region; fixed = (:d,)))
    case!(cases, "lensrefine/RAT_icra84/fixed_d_from_held_linear", (grh, ic.kb, ic.region), ((g, m, r),) -> refine_camera(g, m, r; fixed = (:d,)))
    # CaliBlobs' pattern: a target instance, the tangential terms and the denominator held at the target's values
    src = CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = Lens(Perspective(), Radial((-0.12, 0.03, 0.004))))
    target = Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(1.0e-4, -2.0e-4)))
    target0 = Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0)))
    dsc = Ellipse(SVector(960.0, 540.0), 700.0, 700.0, 0.0)
    case!(cases, "lensfit/target_instance_tangential/fixed_p_d", (target, src, dsc), ((t, m, r),) -> fit_camera(t, m, r; fixed = (:p, :d)))
    case!(cases, "lensfit/target_instance_zero/fixed_p_d", (target0, src, dsc), ((t, m, r),) -> fit_camera(t, m, r; fixed = (:p, :d)))
    allnames = parameter_names(CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = target))
    case!(cases, "lensfit/target_instance_tangential/fixed_all", (target, src, dsc), ((t, m, r),) -> fit_camera(t, m, r; fixed = allnames))
    ddisc = Ellipse(SVector(2120.0, 1416.0), 2000.0, 2000.0, 0.0)
    case!(cases, "lensfit/division_instance/refused_hold_without_focal", (Division(0.0, -0.0765), mdiv, ddisc),
          ((t, m, r),) -> fit_camera(t, m, r; fixed = (:d2,)))
    case!(cases, "lensfit/division_instance/fixed_f_d2", (Division(0.0, -0.0765), mdiv, ddisc), ((t, m, r),) -> fit_camera(t, m, r; fixed = (:f, :d2)))

    # a lens refitted onto its own family (zero residuals)
    zregion = Ellipse(SVector(320.0, 240.0), 200.0, 200.0, 0.0)
    for (lbl, l) in (("perspective", Lens(Perspective())), ("equidistant", Lens(Equidistant())), ("stereographic", Lens(Stereographic())),
                     ("division2_zero", zero(Division{2})), ("scaramuzza3_zero", zero(Scaramuzza{3})))
        s = CameraModel(; pp = (320.0, 240.0), focal = 300.0, lens = l)
        case!(cases, "lensfit/own_family_$lbl/linear", (s, zregion), ((m, r),) -> fit_camera(typeof(lens(m)), m, r))
        case!(cases, "lensapprox/own_family_$lbl/default", (s, zregion), ((m, r),) -> approximate_camera(typeof(lens(m)), m, r))
    end

    # termination record: the fisheye model over the lit disc; its refine, capped, everything held
    fe = ecf_fisheye()
    case!(cases, "lensfit/KB4_fisheye_disc/linear", (fe.FISHEYE, fe.REGION), ((m, r),) -> fit_camera(ECF_KB4, m, r))
    lin = estimate(fit_camera(ECF_KB4, fe.FISHEYE, fe.REGION))
    ff = (lin, fe.FISHEYE, fe.REGION)
    case!(cases, "lensrefine/KB4_fisheye_disc/default", ff, ((g, m, r),) -> refine_camera(g, m, r))
    case!(cases, "lensrefine/KB4_fisheye_disc/max_iter1", ff, ((g, m, r),) -> refine_camera(g, m, r; max_iter = 1))
    case!(cases, "lensrefine/KB4_fisheye_disc/fixed_all", ff, ((g, m, r),) -> refine_camera(g, m, r; fixed = Tuple(parameter_names(g))))
    case!(cases, "lensapprox/Division2_fisheye_disc/default", (fe.FISHEYE, fe.REGION), ((m, r),) -> approximate_camera(Division{2}, m, r))
    # a parametric lens from a SplineLens camera (bounded and bare), its linear fit, the image density only
    case!(cases, "lensapprox/KB4_grid_bounded/default", (fe.GRID_BOUNDED, fe.REGION), ((m, r),) -> approximate_camera(ECF_KB4, m, r))
    case!(cases, "lensapprox/KB4_grid/default", (fe.GRID, fe.REGION), ((m, r),) -> approximate_camera(ECF_KB4, m, r))
    case!(cases, "lensfit/KB4_grid/linear", (fe.GRID, fe.REGION), ((m, r),) -> fit_camera(ECF_KB4, m, r))
    case!(cases, "lensfit/KB4_grid/refused_solid_angle", (fe.GRID, fe.REGION), ((m, r),) -> fit_camera(ECF_KB4, m, r; weight = :solid_angle))
    case!(cases, "lensapprox/KB4_grid/refused_solid_angle", (fe.GRID, fe.REGION), ((m, r),) -> approximate_camera(ECF_KB4, m, r; weight = :solid_angle))
    # a non-central source: not supported yet
    nc = ecf_zero_noncentral()
    case!(cases, "lensfit/KB4_noncentral/refused", (nc, fe.REGION), ((m, r),) -> fit_camera(ECF_KB4, m, r))
    case!(cases, "lensfit/KB4_noncentral_bounded/refused", (BoundedCamera(nc, Size2(1000, 1000)), fe.REGION), ((m, r),) -> fit_camera(ECF_KB4, m, r))
    case!(cases, "lensapprox/KB4_noncentral/refused", (nc, fe.REGION), ((m, r),) -> approximate_camera(ECF_KB4, m, r))
    case!(cases, "lensrefine/KB4_noncentral/refused", (lin, nc, fe.REGION), ((g, m, r),) -> refine_camera(g, m, r))
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# spline targets: fit_camera(SplineLens / NoncentralSplineModel), pspline_grid, fit_forward_spline
# ---------------------------------------------------------------------------------------------------------------
function _record_ecf_spline!(cases)
    fe = ecf_fisheye()
    gb, R = fe.GRID_BOUNDED, fe.REGION
    case!(cases, "splinefit/grid_bounded_disc/tol1e-3", (gb, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3))
    case!(cases, "splinefit/grid_bounded_disc/tol1e-3_backward_only", (gb, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, forward = false))
    cam = estimate(fit_camera(SplineLens, gb, R; tolerance = 1.0e-3))
    case!(cases, "splinefit/refit_of_fitted/tol1e-3", (BoundedCamera(cam, Size2(1000, 1000)), R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3))
    case!(cases, "splinefit/grid_bounded_disc/refused_grid_size_on_spline", (gb, R),
          ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, grid_size = (40, 40), forward = false))
    case!(cases, "splinefit/grid_bounded_disc/spacing9", (gb, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 9.0, forward = false))
    sensor = Size2(4256, 2848)
    lens14 = CameraModel(1727.0px, SA[2140.0px, 1415.0px]; lens = KannalaBrandt(0.02))
    for g in ((16, 18), (26, 20))
        s = BoundedCamera(CameraModel(SplineLens, lens14, sensor; grid_size = g), sensor)
        case!(cases, "splinefit/rc_lattice_$(g[1])x$(g[2])/spacing60", (s, Rect(0, 0, 4256, 2848)),
              ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 60.0, forward = false))
    end
    case!(cases, "splinefit/kb14_parametric/refused_lattice_spacing", (BoundedCamera(lens14, sensor), Rect(0, 0, 4256, 2848)),
          ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 60.0))
    part = BoundedCamera(CameraModel(SplineLens, fe.GRID, Size2(1000, 1000); grid_size = (30, 30), calib_area = Rect(100, 100, 800, 800)), Size2(1000, 1000))
    case!(cases, "splinefit/partial_lattice_disc/spacing20_extended", (part, R),
          ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 20.0, forward = false))
    case!(cases, "splinefit/partial_lattice_inner/spacing20", (part, Rect(200, 200, 600, 600)),
          ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 20.0, forward = false))
    case!(cases, "splinefit/partial_lattice_disc/refused_support", (part, R),
          ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 20.0, support = Rect(100, 100, 800, 800), forward = false))
    # the Schöps source on its lattice edge
    sch = ecf_schoeps()
    sensor640 = Rect(0.0, 0.0, 640.0, 480.0)
    for (lbl, region, spacing) in (("sensor", sensor640, nothing), ("sensor_spacing80", sensor640, 80.0),
                                   ("sensor_disc380", sensor640 ∩ Ellipse(SVector(320.0, 240.0), 380.0, 380.0, 0.0), nothing),
                                   ("sensor_ellipse", sensor640 ∩ Ellipse(SVector(320.05, 240.0), 319.95, 400.0, 0.0), nothing),
                                   ("inner", Rect(10.0, 10.0, 600.0, 440.0), nothing))
        case!(cases, "splinefit/schoeps_$lbl/tolInf", (sch, region),
              ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = Inf, forward = false, lattice_spacing = spacing))
    end
    # region shapes; out-of-reach tolerances; no weight on a spline target
    case!(cases, "splinefit/grid_bounded_cut/tol1e-3", (gb, Rect(0, 150, 1000, 700) ∩ fe.DISC), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3))
    wide = BoundedCamera(CameraModel(500.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02)), Size2(1000, 1000))
    case!(cases, "splinefit/kb500_whole_sensor/tol1e-3", (wide, Rect(0, 0, 1000, 1000)), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3))
    case!(cases, "splinefit/grid_bounded_whole_sensor/tol1e-3", (gb, Rect(0, 0, 1000, 1000)), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3))
    case!(cases, "splinefit/grid_bounded_disc/tol1e-7_out_of_reach", (gb, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-7))
    case!(cases, "splinefit/grid_bounded_disc/refused_weight", (gb, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, weight = :image))
    # a backward lattice from a parametric model
    bc = fe.BC
    case!(cases, "splinefit/fisheye_bounded_disc/tol1e-4_backward", (bc, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-4, forward = false))
    case!(cases, "splinefit/fisheye_bounded_disc/tol1e-12_backward_out_of_reach", (bc, R),
          ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-12, forward = false))
    small = BoundedCamera(CameraModel(18.0px, SA[30.0px, 30.0px]; lens = KannalaBrandt(-0.02)), Size2(60, 60))
    sreg = Rect(0, 0, 60, 60) ∩ Ellipse(SVector(30.0, 30.0), 29.0, 29.0, 0.0)
    case!(cases, "splinefit/small60/tol1e-6_backward_out_of_reach", (small, sreg), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-6, forward = false))
    case!(cases, "splinefit/fisheye_bounded_disc/grid40x40_backward", (bc, R),
          ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-4, grid_size = (40, 40), forward = false))
    case!(cases, "splinefit/fisheye_bounded_disc/grid40x30_backward", (bc, R),
          ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-4, grid_size = (40, 30), forward = false))
    case!(cases, "splinefit/fisheye_bare/refused_no_image", (fe.FISHEYE, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 0.01))
    case!(cases, "splinefit/fisheye_bounded_disc/refused_tol0", (bc, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 0.0))
    case!(cases, "splinefit/fisheye_bounded_disc/tol1e-2_backward", (bc, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-2, forward = false))
    # a function source: refused; its SplineLens camera (Core's source constructor) fitted
    case!(cases, "splinefit/function_source/refused_no_image", (ecf_fisheye_ray, R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-2))
    fcam = CameraModel(SplineLens, ecf_fisheye_ray, Size2(1000, 1000); grid_size = (40, 40), init = :quasi_interpolate)
    case!(cases, "splinefit/function_camera_bounded/tol1e-2", (BoundedCamera(fcam, Size2(1000, 1000)), R), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-2))
    # a degenerate source Jacobian at the left edge
    case!(cases, "splinefit/degenerate_jacobian_grid/spacing13.5", (ecf_degenerate_grid(), R),
          ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 13.5, forward = false))
    # spline_fit_sources: a bare model refused; a bounded KB past its injectivity disc refused; inside it solved
    m = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    case!(cases, "splinefit/kb250_bare/refused_no_image", (m, Rect(0, 0, 1280, 960)), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-2))
    case!(cases, "splinefit/kb250_bare/refused_no_image_noncentral", (m, Rect(0, 0, 1280, 960)),
          ((s, r),) -> fit_camera(NoncentralSplineModel, s, r; tolerance = 1.0e-2))

    # non-central targets
    S = ecf_noncentral_source()
    case!(cases, "noncentralfit/source_disc/tol3e-4", (S, R), ((s, r),) -> fit_camera(NoncentralSplineModel, s, r; tolerance = 3.0e-4))
    case!(cases, "noncentralfit/source_disc/tol1e-3", (S, R), ((s, r),) -> fit_camera(NoncentralSplineModel, s, r; tolerance = 1.0e-3))
    case!(cases, "noncentralfit/source_translated_disc/tol3e-4", (ecf_noncentral_source(SVector(0.5, -0.25, 2.0)), R),
          ((s, r),) -> fit_camera(NoncentralSplineModel, s, r; tolerance = 3.0e-4))
    case!(cases, "noncentralfit/source_disc/tol1e-9_out_of_reach", (S, R), ((s, r),) -> fit_camera(NoncentralSplineModel, s, r; tolerance = 1.0e-9))
    case!(cases, "noncentralfit/model_bare/refused_no_image", (VgtkCore.model(S), R), ((s, r),) -> fit_camera(NoncentralSplineModel, s, r; tolerance = 1.0e-3))
    case!(cases, "noncentralfit/central_fisheye_bounded/tol1e-3", (bc, R), ((s, r),) -> fit_camera(NoncentralSplineModel, s, r; tolerance = 1.0e-3))
    case!(cases, "noncentralfit/small60/tol1e-6_out_of_reach", (small, sreg), ((s, r),) -> fit_camera(NoncentralSplineModel, s, r; tolerance = 1.0e-6))
    case!(cases, "noncentralfit/fisheye_bare/refused", (fe.FISHEYE, R), ((s, r),) -> fit_camera(NoncentralSplineModel, s, r; tolerance = 0.01))

    # pspline_grid: at the lattice fit_camera chose, and at two given ones
    gw, gh = metadata(fit_camera(SplineLens, bc, R; tolerance = 1.0e-3)).family.grid_size
    for (lbl, gs) in (("fit_choice", (gw, gh)), ("35x25", (35, 25)), ("105x75", (105, 75)))
        case!(cases, "pspline/fisheye_bounded_disc/$lbl", (bc, R, gs), ((s, r, gs),) -> pspline_grid(SplineLens, s, r, gs))
    end
    case!(cases, "pspline/fisheye_bare/refused_no_image", (fe.FISHEYE, R, (35, 25)), ((s, r, gs),) -> pspline_grid(SplineLens, s, r, gs))
    case!(cases, "pspline/fisheye_bounded_disc/refused_3x9", (bc, R, (3, 9)), ((s, r, gs),) -> pspline_grid(SplineLens, s, r, gs))
    gwn, ghn = metadata(fit_camera(NoncentralSplineModel, S, R; tolerance = 1.0e-3)).family.grid_size
    for (lbl, gs) in (("fit_choice", (gwn, ghn)), ("35x25", (35, 25)), ("105x75", (105, 75)))
        case!(cases, "pspline/noncentral_source_disc/$lbl", (S, R, gs), ((s, r, gs),) -> pspline_grid(NoncentralSplineModel, s, r, gs))
    end
    kbm = BoundedCamera(m, Size2(1280, 960))
    case!(cases, "pspline/kb250_bounded_frame/refused_beyond_support", (kbm, Rect(0, 0, 1280, 960), (8, 8)), ((s, r, gs),) -> pspline_grid(SplineLens, s, r, gs))
    case!(cases, "pspline/kb250_bounded_valid_domain/8x8", (kbm, valid_domain(m), (8, 8)), ((s, r, gs),) -> pspline_grid(SplineLens, s, r, gs))
    case!(cases, "pspline/kb250_bare/refused_no_image", (m, Rect(0, 0, 1280, 960), (8, 8)), ((s, r, gs),) -> pspline_grid(SplineLens, s, r, gs))

    # fit_forward_spline
    kb = CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
    fcam12 = BoundedCamera(CameraModel(SplineLens, kb, Size2(1280, 960); grid_size = (12, 10)), Size2(1280, 960))
    case!(cases, "forwardspline/kb_spline_12x10/tol1e-3", fcam12, c -> fit_forward_spline(c; tolerance = 1.0e-3))
    case!(cases, "forwardspline/kb_parametric/refused", BoundedCamera(kb, Size2(1280, 960)), c -> fit_forward_spline(c; tolerance = 1.0e-3))
    case!(cases, "forwardspline/noncentral_bounded/refused", S, c -> fit_forward_spline(c; tolerance = 1.0e-3))
    case!(cases, "forwardspline/noncentral_model/refused", VgtkCore.model(S), c -> fit_forward_spline(c; tolerance = 1.0e-3))
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# optical_axis, and the gauge it gives fit_camera(SplineLens)
# ---------------------------------------------------------------------------------------------------------------
function _record_ecf_axis!(cases)
    oa = ecf_off_axis()
    cov = with_lens_coverage(oa.OFF_BOUNDED, oa.DISC)
    case!(cases, "axis/off_axis_covered/default", cov, c -> optical_axis(c))
    case!(cases, "axis/off_axis_bounded/default", oa.OFF_BOUNDED, c -> optical_axis(c))
    case!(cases, "axis/off_axis_bare/refused", oa.OFF_AXIS, c -> optical_axis(c))
    L = typeof(estimate(optical_axis(cov)).lens)
    case!(cases, "lensapprox/axis_family_off_axis_covered/fixed_p_sx_sy", (L, oa.OFF_BOUNDED, imaged_region(cov)),
          ((L, s, r),) -> approximate_camera(L, s, r; fixed = (:p, :sx, :sy)))
    case!(cases, "splinefit/off_axis_bounded_disc/tol1e-3", (oa.OFF_BOUNDED, oa.REGION), ((s, r),) -> fit_camera(SplineLens, s, r; tolerance = 1.0e-3))
    case!(cases, "lensfit/KB4_off_axis_bounded/linear", (oa.OFF_BOUNDED, oa.REGION), ((s, r),) -> fit_camera(ECF_KB4, s, r))
    case!(cases, "lensapprox/KB4_off_axis_bounded/default", (oa.OFF_BOUNDED, oa.REGION), ((s, r),) -> approximate_camera(ECF_KB4, s, r))
    r = optical_axis(oa.OFF_BOUNDED)
    case!(cases, "lensrefine/axis_off_axis_bare/from_own_optimum", (estimate(r), oa.OFF_AXIS, oa.REGION), ((g, m, r),) -> refine_camera(g, m, r))
    case!(cases, "axis/decentred_covered/default", with_lens_coverage(ecf_decentred_spline(), oa.DISC), c -> optical_axis(c))
    case!(cases, "axis/fisheye_physical_covered/default", BoundedCamera(oa.FISHEYE, Size2(1000, 1000); lens_coverage = oa.DISC), c -> optical_axis(c))
    fe = ecf_fisheye()
    case!(cases, "axis/grid_covered/default", with_lens_coverage(fe.GRID_BOUNDED, fe.DISC), c -> optical_axis(c))
    # the circular fisheye: its image circle, the whole sensor, a given region, a region past the image
    cf = ecf_circular_fisheye()
    sz = Size2(1400, 1000)
    bounded = BoundedCamera(cf.spl, sz; lens_coverage = cf.circle)
    case!(cases, "axis/circular_fisheye_covered/default", bounded, c -> optical_axis(c))
    case!(cases, "axis/circular_fisheye_whole/default", BoundedCamera(cf.spl, sz), c -> optical_axis(c))
    case!(cases, "axis/circular_fisheye_whole/region_circle", (BoundedCamera(cf.spl, sz), cf.circle), ((c, r),) -> optical_axis(c, r))
    case!(cases, "axis/circular_fisheye_covered/region_sensor", (bounded, Rect(0, 0, 1400, 1000)), ((c, r),) -> optical_axis(c, r))
    case!(cases, "axis/circular_fisheye_covered/region_imaged", (bounded, imaged_region(bounded)), ((c, r),) -> optical_axis(c, r))
    # region support: the RC siemens start camera
    sm = ecf_siemens()
    case!(cases, "axis/siemens/refused_beyond_support", sm, ((c, s),) -> optical_axis(c; support = s))
    case!(cases, "lensapprox/KB4_siemens_imaged/refused_beyond_support", sm, ((c, s),) -> approximate_camera(ECF_KB4, c, imaged_region(c); support = s))
    case!(cases, "lensfit/KB4_siemens_imaged/refused_beyond_support", sm, ((c, s),) -> fit_camera(ECF_KB4, c, imaged_region(c); support = s))
    case!(cases, "axis/siemens/region_support", sm, ((c, s),) -> optical_axis(c, s; support = s))
    ga = estimate(optical_axis(sm.cam, sm.support; support = sm.support))
    case!(cases, "lensrefine/axis_siemens_imaged/refused_beyond_support", (ga, sm...), ((g, c, s),) -> refine_camera(g, c, imaged_region(c); support = s))
    case!(cases, "lensrefine/axis_siemens_support/default", (ga, sm...), ((g, c, s),) -> refine_camera(g, c, s; support = s))
    # the folding sources: the order continuation
    for (lbl, src) in ecf_folding_sources()
        case!(cases, "axis/folding_$lbl/default", src, c -> optical_axis(c))
    end
    # the stepwise KannalaBrandt continuation by the public verbs (order_continuation_kb), each stage seeded by the last
    src = last(ecf_folding_sources()[2])
    part = imaged_region(src)
    g = estimate(fit_camera(KannalaBrandt{1}, src, part))
    case!(cases, "lensfit/KB1_folding_fold19.5_eps0.02/linear", (src, part), ((s, r),) -> fit_camera(KannalaBrandt{1}, s, r))
    for N in 1:4
        k = Tuple(Float64.(parameter_vector(lens(g))))
        guess = CameraModel(SMatrix{3, 3, Float64}(calibration_matrix(g)); lens = KannalaBrandt(ntuple(i -> i <= length(k) ? k[i] : 0.0, N)...))
        case!(cases, "lensrefine/KB$(N)_folding_fold19.5_eps0.02/continuation", (guess, src, part), ((g, s, r),) -> refine_camera(g, s, r))
        g = estimate(refine_camera(guess, src, part))
    end
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# fit_division_camera (division_solve_golden's scenes, the item's keywords and refusals, seeds)
# ---------------------------------------------------------------------------------------------------------------
function _record_ecf_division!(cases)
    rc = ecf_radial_cameras()
    sz = (4000, 3000)
    clean = ecf_board_views(StableRNG(11))
    dirty = ecf_board_views(StableRNG(3); ε = 0.08)
    case!(cases, "divcam/golden_clean/default", (rc, clean), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz))
    case!(cases, "divcam/golden_contaminated/default", (rc, dirty), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz))
    case!(cases, "divcam/golden_contaminated/ncoef2", (rc, dirty), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, ncoef = 2))
    case!(cases, "divcam/golden_contaminated/known_k", (rc, dirty), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, k = _ECF_DB.K0))
    for seed in SEEDS
        vs = ecf_board_views(StableRNG(seed); ε = 0.08)
        case!(cases, "divcam/contaminated/default/seed$seed", (rc, vs), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz))
        case!(cases, "divcam/contaminated/ncoef2/seed$seed", (rc, vs), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, ncoef = 2))
        case!(cases, "divcam/contaminated/known_k/seed$seed", (rc, vs), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, k = _ECF_DB.K0))
        case!(cases, "divcam/clean/default/seed$seed", (rc, ecf_board_views(StableRNG(seed))), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz))
    end
    d = (rc, dirty)
    boards = [dirty[i] for i in 1:6]
    ones_vw = Dict(i => Correspondences(first.(b), last.(b); weights = ones(length(b))) for (i, b) in enumerate(boards))
    case!(cases, "divcam/golden_contaminated/unit_weights", (rc, ones_vw), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz))
    case!(cases, "divcam/golden_contaminated/refused_key_mismatch", (rc, Dict(i => boards[i] for i in 1:5)), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz))
    case!(cases, "divcam/golden_contaminated/refused_no_image_size", d, ((rc, v),) -> fit_division_camera(rc, v))
    case!(cases, "divcam/golden_contaminated/refused_negative_seed", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, focal_seed = -1.0))
    for mi in (1, 2, 5)
        case!(cases, "divcam/golden_contaminated/em_max_iter$mi", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, mixture_em = ecf_em_at(mi)))
    end
    case!(cases, "divcam/golden_contaminated/seed1800", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, focal_seed = 1800.0))
    case!(cases, "divcam/golden_contaminated/seed1800_image8000", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = (8000, 6000), focal_seed = 1800.0))
    case!(cases, "divcam/golden_contaminated/seed1800_no_image_size", d, ((rc, v),) -> fit_division_camera(rc, v; focal_seed = 1800.0))
    c = SVector(1990.0, 1510.0)
    rcc = Dict(i => RadialCamera{Float64}(radial_rows(P), c) for (i, P) in rc)
    vwc = Dict(i => [X => x + c for (X, x) in b] for (i, b) in dirty)
    case!(cases, "divcam/golden_contaminated_offcentre/seed1800", (rcc, vwc), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, focal_seed = 1800.0))
    # radial cameras robustly fitted from each board (the BabelCalib route): find_radial_camera's output, stored
    Ps = Dict(i => find_radial_camera(b; center = (0.0, 0.0), scoring = MsacScoring(RadialCameraProblem; threshold = 5.0), rng = StableRNG(9)).estimate
              for (i, b) in enumerate(boards))
    case!(cases, "divcam/golden_contaminated_found_cameras/default", (Ps, dirty), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz))
    zw = Dict(i => Correspondences(first.(b), last.(b); weights = (i == 6 ? zeros : ones)(length(b))) for (i, b) in enumerate(boards))
    case!(cases, "divcam/golden_contaminated/view6_zero_weight", (rc, zw), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz))
    neg = Dict(i => Correspondences(first.(b), last.(b); weights = (i == 2 ? fill(-1.0, length(b)) : ones(length(b)))) for (i, b) in enumerate(boards))
    case!(cases, "divcam/golden_contaminated/refused_negative_weight", (rc, neg), ((rc, v),) -> fit_division_camera(rc, v; image_size = sz))
    allz = Dict(i => Correspondences(first.(b), last.(b); weights = zeros(length(b))) for (i, b) in vwc)
    case!(cases, "divcam/golden_contaminated_offcentre/refused_all_zero", (rcc, allz), ((rc, v),) -> fit_division_camera(rc, v; focal_seed = 1800.0))
    # the shared scorer and EM tables
    sc = scoring_from_toml(TOML.parsefile(default_config_path())["fit_division_camera"]["weighting"]; problem = DivisionCameraProblem)
    em = ecf_em()
    case!(cases, "divcam/golden_contaminated/explicit_tables", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, weighting = sc, mixture_em = em))
    case!(cases, "divcam/golden_contaminated/em_cap3", d,
          ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, mixture_em = MixtureEmConfig(em.model_update, em.nuisance, em.stop, 3)))
    case!(cases, "divcam/golden_contaminated/halfwidth60", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = sz,
          weighting = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 60.0); problem = DivisionCameraProblem)))
    for (lbl, kw) in (("outlier_halfwidth", (; outlier_halfwidth = 20.0)), ("max_iter", (; max_iter = 5)), ("scoring", (; scoring = sc)))
        case!(cases, "divcam/golden_contaminated/refused_old_key_$lbl", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = sz, kw...))
    end
    case!(cases, "divcam/golden_contaminated/refused_codimension1_scorer", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = sz,
          weighting = StudentTUniformScoring(2, 20.0; codimension = 1, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())))
    case!(cases, "divcam/golden_contaminated/refused_keep_model", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = sz,
          mixture_em = MixtureEmConfig(KeepModel(1.0e-9), em.nuisance, em.stop, 30)))
    case!(cases, "divcam/golden_contaminated/refused_no_improvement_stop", d, ((rc, v),) -> fit_division_camera(rc, v; image_size = sz,
          mixture_em = MixtureEmConfig(em.model_update, em.nuisance, NoImprovementStop(), 30)))
    # units: a physical plane is refused
    src = [SVector(0.1 * i, 0.07 * j) for i in 1:4 for j in 1:4]
    phy = [p .* VgtkCore.m for p in src] .=> [SVector(100.0 + 900 * p[1], 50.0 + 880 * p[2]) for p in src]
    case!(cases, "divcam/physical_plane/refused", (Dict(1 => HomographyMat(SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1))), Dict(1 => phy)),
          ((rc, v),) -> fit_division_camera(rc, v; focal_seed = 500.0))
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# fit_radial_camera
# ---------------------------------------------------------------------------------------------------------------
function _record_ecf_radial!(cases)
    pts = ecf_first_board_pixels()
    o = SVector(0.0, 0.0)
    case!(cases, "radialcam/board1/default", pts, p -> fit_radial_camera(p; center = SVector(0.0, 0.0)))
    case!(cases, "radialcam/board1/iterations1", pts, p -> fit_radial_camera(p; center = SVector(0.0, 0.0), iterations = 1))
    case!(cases, "radialcam/board1_4pts/refused_too_few", pts[1:4], p -> fit_radial_camera(p; center = SVector(0.0, 0.0)))
    case!(cases, "radialcam/board1/refused_no_center", pts, p -> fit_radial_camera(p))
    case!(cases, "radialcam/board1_5pts/iterations1", pts[[1, 5, 8, 21, 25]], p -> fit_radial_camera(p; center = SVector(0.0, 0.0), iterations = 1))
    c = SVector(931.0, -472.0)
    case!(cases, "radialcam/board1_raw/centre", ([X => (x + c) for (X, x) in pts], c), ((p, c),) -> fit_radial_camera(p; center = c))
    col = [SVector(t, 2t) => pts[1][2] .* (1 + 0.1t) for t in 1.0:5.0]
    case!(cases, "radialcam/collinear5/iterations1_degenerate", col, p -> fit_radial_camera(p; center = SVector(0.0, 0.0), iterations = 1))
    col8 = [SVector(Float64(i), 0.0) => SVector(10.0i, 0.0) for i in 1:8]
    case!(cases, "radialcam/collinear8/degenerate", col8, p -> fit_radial_camera(p; center = SVector(0.0, 0.0)))
    bad = copy(pts); bad[3] = bad[3][1] => bad[3][2] + SVector(40.0, -25.0)
    w = ones(length(bad)); w[3] = 0.0
    case!(cases, "radialcam/board1_outlier/zero_weight", Correspondences(first.(bad), last.(bad); weights = w), p -> fit_radial_camera(p; center = SVector(0.0, 0.0)))
    case!(cases, "radialcam/board1_outlier/unweighted", bad, p -> fit_radial_camera(p; center = SVector(0.0, 0.0)))
    case!(cases, "radialcam/board1/unit_weights", Correspondences(first.(pts), last.(pts); weights = ones(length(pts))),
          p -> fit_radial_camera(p; center = SVector(0.0, 0.0)))
    src = [SVector(0.1 * i, 0.07 * j) for i in 1:4 for j in 1:4]
    phy = [p .* VgtkCore.m for p in src] .=> [SVector(100.0 + 900 * p[1], 50.0 + 880 * p[2]) for p in src]
    case!(cases, "radialcam/physical_plane/refused", phy, p -> fit_radial_camera(p; center = SVector(400.0, 300.0)))
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# refine_camera_and_poses: the views form, the board form, the rotation-centre form
# ---------------------------------------------------------------------------------------------------------------
function _record_ecf_bundle!(cases)
    ls = (; optimization_strategy = nothing)
    # --- a pinhole camera and five views from exact pixels -----------------------------------------------------------
    cam = ecf_pinhole_camera(); Es = ecf_truth_poses(5)
    views = ecf_views_of(cam, Es, 0.6)
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.04, 1.01, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 6.0, -5.0))
    poses0 = [ecf_perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    fx = (cam0, poses0, views)
    case!(cases, "bundle/pinhole5_exact/ls", fx, ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
    case!(cases, "bundle/pinhole5_exact/ls_max_iter1", fx, ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls..., max_iter = 1))
    # an exact Division fit from exact correspondences (CaliBlobs' README refine), four starts
    truth = CameraModel(LogicalIntrinsics(CameraCalibrationMat(900.0px, [2120.0px, 1416.0px])), Division(-0.2, -0.08))
    board = [Vec3(x, y, 0.0) for x in -60.0:30.0:60.0 for y in -60.0:30.0:60.0]
    dposes = Dict(t => EuclideanMap(Rotations.RotationVec(0.2sin(t), 0.2cos(t), 0.1t), Vec3(40.0t - 140, 20.0 - 10t, 300.0)) for t in 1:6)
    dviews = Dict(t => [X => project(truth, P(X)) for X in board] for (t, P) in dposes)
    for (n, f0) in ((6, 945.0), (6, 930.0), (4, 945.0), (3, 945.0))
        start = CameraModel(LogicalIntrinsics(CameraCalibrationMat(f0 * px, [2120.0px, 1416.0px])), Division(-0.18, -0.072))
        case!(cases, "bundle/division_exact_$(n)views/f$(round(Int, f0))_fixed_aspect",
              (start, Dict(t => dposes[t] for t in 1:n), Dict(t => dviews[t] for t in 1:n)),
              ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:aspect,), ls...))
    end
    # a fisheye past 90°
    fcam = ecf_fisheye_camera(); fEs = ecf_truth_poses(4)
    fviews = ecf_views_of(fcam, fEs, deg2rad(100))
    fθ = parameter_vector(fcam)
    fcam0 = with_parameters(fcam, fθ .* SVector(1.03, 1.0, 1.0, 1.0, 1.0, 0.8, 1.2, 0.9, 1.1) .+ SVector(0.0, 0.0, 0.0, -4.0, 3.0, 0.0, 0.0, 0.0, 0.0))
    ff = (fcam0, [ecf_perturbed_pose(E, v) for (v, E) in enumerate(fEs)], fviews)
    case!(cases, "bundle/fisheye4_exact/ls", ff, ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
    case!(cases, "bundle/fisheye4_exact/ls_fixed_k_skew", ff, ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:k, :skew), ls...))
    # a held camera; the forms; refusals
    Es3 = ecf_truth_poses(3); views3 = ecf_views_of(cam, Es3, 0.5)
    p3 = [ecf_perturbed_pose(E, v) for (v, E) in enumerate(Es3)]
    case!(cases, "bundle/pinhole3_exact/ls_camera_held", (cam, p3, views3), ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = parameter_names(c), ls...))
    case!(cases, "bundle/pinhole3_exact/ls_fixed_aspect", (cam, p3, views3), ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:aspect,), ls...))
    case!(cases, "bundle/pinhole3_exact/ls_fixed_empty", (cam, p3, views3), ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (), ls...))
    for kw in ECF_RETIRED
        case!(cases, "bundle/pinhole3_exact/refused_retired_$kw", (cam, p3, views3), ((c, p, v),) -> refine_camera_and_poses(c, p, v; (; kw => ())...))
    end
    c3 = with_parameters(cam, parameter_vector(cam) .* SVector(1.02, 1.0, 1.0, 1.0, 1.0))
    f3 = (c3, p3, views3)
    case!(cases, "bundle/pinhole3_exact/ls_vector_form", f3, ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
    case!(cases, "bundle/pinhole3_exact/ls_dict_form",
          (c3, Dict(:c => p3[3], :a => p3[1], :b => p3[2]), Dict(:a => views3[1], :b => views3[2], :c => Correspondences(first.(views3[3]), last.(views3[3])))),
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
    case!(cases, "bundle/pinhole3_exact/ls_metres", (c3, [with_unit(P, VgtkCore.m) for P in p3], [[X .* VgtkCore.m => u for (X, u) in cs] for cs in views3]),
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
    case!(cases, "bundle/pinhole3_exact/refused_key_mismatch", (c3, Dict(:a => p3[1]), Dict(:b => views3[1])), ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
    case!(cases, "bundle/pinhole3_exact/refused_length_mismatch", (c3, p3[1:2], views3), ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
    case!(cases, "bundle/pinhole3_exact/refused_unknown_fixed", f3, ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:k9,), ls...))
    case!(cases, "bundle/pinhole3_exact/refused_behind_camera", (c3, [EuclideanMap(Es3[1].R, SVector(0.0, 0.0, -50.0)); p3[2:3]], views3),
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
    case!(cases, "bundle/pinhole3_exact/refused_underdetermined", (c3, p3[1:1], [views3[1][1:2]]), ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
    case!(cases, "bundle/pinhole3_exact/refused_box_nothing_free", f3,
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = parameter_names(c), box = (; f = (1.0, 2.0)), ls...))
    case!(cases, "bundle/pinhole3_exact/ls_max_time_inf", f3, ((c, p, v),) -> refine_camera_and_poses(c, p, v; max_time = Inf, ls...))

    # --- robust strategies on contaminated detections (seeds 1–3) ---------------------------------------------------
    cam1 = with_parameters(cam, θ .* SVector(1.005, 1.0, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 1.0, -1.0))
    poses1 = [ecf_near_pose(E, v) for (v, E) in enumerate(Es)]
    H, W = Triggs{HuberShape}(efficiency_constant = 1.345), Triggs{WelschShape}(efficiency_constant = 2.98)
    for seed in SEEDS
        noisy = ecf_contaminate(StableRNG(seed), views)
        far, near = (cam0, poses0, noisy), (cam1, poses1, noisy)
        case!(cases, "bundle/pinhole5_contaminated_far/ls/seed$seed", far, ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls...))
        case!(cases, "bundle/pinhole5_contaminated_far/default_scale0.2/seed$seed", far, ((c, p, v),) -> refine_camera_and_poses(c, p, v; scale = 0.2))
        case!(cases, "bundle/pinhole5_contaminated_far/triggs_huber1.345_scale0.2/seed$seed", far,
              ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = H, scale = 0.2))
        case!(cases, "bundle/pinhole5_contaminated_far/triggs_welsch2.98_scale0.2/seed$seed", far,
              ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = W, scale = 0.2))
        case!(cases, "bundle/pinhole5_contaminated_near/triggs_huber1.345_scale0.2/seed$seed", near,
              ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = H, scale = 0.2))
        case!(cases, "bundle/pinhole5_contaminated_near/triggs_welsch2.98_scale0.2/seed$seed", near,
              ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = W, scale = 0.2))
        case!(cases, "bundle/pinhole5_contaminated_near/triggs_huber_max_iter1/seed$seed", near,
              ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = Triggs{HuberShape}(), scale = 0.2, max_iter = 1))
        case!(cases, "bundle/pinhole5_contaminated_near/triggs_huber_width/seed$seed", near,
              ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = Triggs{HuberShape}(), width = 1.345 * 0.2))
        rd = refine_camera_and_poses(cam0, poses0, noisy; scale = 0.2)
        case!(cases, "bundle/pinhole5_contaminated_near/default_upstream_scale/seed$seed", (near..., rd),
              ((c, p, v, rd),) -> refine_camera_and_poses(c, p, v; scale = rd))
    end
    noisy = ecf_contaminate(StableRNG(1), views)
    far, near = (cam0, poses0, noisy), (cam1, poses1, noisy)
    case!(cases, "bundle/pinhole5_contaminated_far/refused_no_scale", far, ((c, p, v),) -> refine_camera_and_poses(c, p, v))
    case!(cases, "bundle/pinhole5_contaminated_near/refused_irls", near,
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = Irls{HuberShape}(), scale = 0.2))
    case!(cases, "bundle/pinhole5_contaminated_far/refused_terminal_lo", far,
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = TerminalLO(Irls{HuberShape}())))
    case!(cases, "bundle/pinhole5_contaminated_far/refused_scale_and_width", far, ((c, p, v),) -> refine_camera_and_poses(c, p, v; scale = 0.2, width = 0.5))
    case!(cases, "bundle/pinhole5_contaminated_far/refused_ls_scale", far, ((c, p, v),) -> refine_camera_and_poses(c, p, v; ls..., scale = 0.2))
    case!(cases, "bundle/pinhole5_contaminated_near/refused_scoring", near,
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = W, scale = 0.2, scoring = nothing))
    case!(cases, "bundle/pinhole5_contaminated_far/refused_negative_scale", far, ((c, p, v),) -> refine_camera_and_poses(c, p, v; optimization_strategy = W, scale = -1.0))
    case!(cases, "bundle/pinhole5_contaminated_near/refused_robust", near, ((c, p, v),) -> refine_camera_and_poses(c, p, v; robust = nothing))
    case!(cases, "bundle/pinhole5_contaminated_near/refused_local_optimization", near,
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; local_optimization = W, scale = 0.2))

    # --- the board form -----------------------------------------------------------------------------------------------
    boards = ecf_truth_boards(); vposes = ecf_view_poses()
    bcam0 = with_parameters(cam, θ .* SVector(1.03, 1.0, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 5.0, -4.0))
    views0 = Dict(v => ecf_nudged(E, v) for (v, E) in vposes)
    keyn = Dict(:A => 1, :B => 2, :C => 3)
    boards0 = Dict(b => b === :A ? E : ecf_nudged(E, 7 + keyn[b]) for (b, E) in boards)
    exact = ecf_board_obs(cam, vposes, boards)
    case!(cases, "boardbundle/three_boards_exact/ls", (bcam0, views0, boards0, exact), ((c, v, b, o),) -> refine_camera_and_poses(c, v, b, o; ls...))
    for seed in SEEDS
        nb = ecf_board_obs(cam, vposes, boards; σ = 0.2, rng = StableRNG(seed))
        case!(cases, "boardbundle/three_boards_noisy/ls/seed$seed", (bcam0, views0, boards0, nb), ((c, v, b, o),) -> refine_camera_and_poses(c, v, b, o; ls...))
        case!(cases, "boardbundle/three_boards_noisy/ls_truth_start/seed$seed", (cam, vposes, boards, nb), ((c, v, b, o),) -> refine_camera_and_poses(c, v, b, o; ls...))
        case!(cases, "boardbundle/three_boards_noisy/ls_view1_held/seed$seed", (bcam0, views0, boards0, nb),
              ((c, v, b, o),) -> refine_camera_and_poses(c, v, b, o; held_boards = (), held_views = (1,), ls...))
    end
    vd = merge(vposes, Dict(7 => vposes[2])); bd = merge(boards, Dict(:D => ecf_nudged(boards[:B], 1)))
    obs = merge(exact, Dict((7, :D) => [X => SVector{2, Float64}(project(cam, vd[7](bd[:D](X)))) for X in ecf_board_points()]))
    case!(cases, "boardbundle/two_components_exact/ls", (bcam0, merge(views0, Dict(7 => ecf_nudged(vposes[2], 3))), merge(boards0, Dict(:D => bd[:D])), obs),
          ((c, v, b, o),) -> refine_camera_and_poses(c, v, b, o; ls...))
    nb = ecf_board_obs(cam, vposes, boards; σ = 0.2, rng = StableRNG(1))
    case!(cases, "boardbundle/three_boards_noisy/refused_no_held_pose", (bcam0, views0, boards0, nb),
          ((c, v, b, o),) -> refine_camera_and_poses(c, v, b, o; held_boards = (), ls...))
    case!(cases, "boardbundle/three_boards_noisy/refused_board_without_pose", (bcam0, views0, Dict(:A => boards[:A], :B => boards[:B]), nb),
          ((c, v, b, o),) -> refine_camera_and_poses(c, v, b, o; ls...))
    case!(cases, "boardbundle/three_boards_noisy/refused_view_without_data", (bcam0, merge(views0, Dict(9 => vposes[1])), boards0, nb),
          ((c, v, b, o),) -> refine_camera_and_poses(c, v, b, o; ls...))

    # --- a SplineLens camera: refused free; held -------------------------------------------------------------------
    struth = ecf_spline_truth(); sEs = ecf_truth_poses(3)
    sviews = ecf_spline_views(struth, sEs, 1.1)
    sp0 = [ecf_perturbed_pose(E, v) for (v, E) in enumerate(sEs)]
    pc = ecf_perturbed_controls(struth, 0.01)
    sfx = (ecf_bounded(pc), sp0, sviews)
    case!(cases, "bundle/spline3_exact/refused_free_controls", sfx, ((c, p, v),) -> refine_camera_and_poses(c, p, v))
    case!(cases, "bundle/spline3_exact/refused_free_controls_fixed_empty", sfx, ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = ()))
    for (lbl, penalty) in (("gcv", :gcv), ("1e-8", 1.0e-8), ("0", 0))
        case!(cases, "bundle/spline3_exact/refused_penalty_$(lbl)_held", sfx, ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:controls,), penalty))
        case!(cases, "bundle/spline3_exact/refused_penalty_$lbl", sfx, ((c, p, v),) -> refine_camera_and_poses(c, p, v; penalty))
    end
    case!(cases, "bundle/spline3_exact_bare/refused_free_controls", (pc, sp0, sviews), ((c, p, v),) -> refine_camera_and_poses(c, p, v))
    case!(cases, "bundle/spline3_exact_bare_keyed/refused_free_controls", (pc, Dict(v => sp0[v] for v in 1:3), Dict(v => sviews[v] for v in 1:3)),
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = ()))
    held = (; fixed = (:controls,), optimization_strategy = nothing)
    kb = ecf_fisheye_camera(); bk = BoundedCamera(kb, Size2(1280, 960))
    fitted = estimate(fit_camera(SplineLens, bk, imaged_region(bk); tolerance = 1.0e-2, forward = false))
    for (lbl, tc) in (("fitted", fitted), ("quasi", ecf_kb_spline_quasi()), ("scaled", ecf_scaled_spline()), ("truth", struth))
        case!(cases, "bundle/spline3_$(lbl)_exact/held_ls", (ecf_bounded(tc), sp0, ecf_spline_views(tc, sEs, 1.1)), ((c, p, v),) -> refine_camera_and_poses(c, p, v; held...))
    end
    case!(cases, "bundle/spline3_truth_bare_exact/held_ls", (struth, sp0, sviews), ((c, p, v),) -> refine_camera_and_poses(c, p, v; held...))
    fwd = BoundedCamera(estimate(fit_camera(SplineLens, BoundedCamera(pc, Size2(1280, 960)), imaged_region(BoundedCamera(pc, Size2(1280, 960))); tolerance = 1.0e-3)),
                        Size2(1280, 960))
    case!(cases, "bundle/spline3_with_forward_exact/held_ls", (fwd, sp0, sviews), ((c, p, v),) -> refine_camera_and_poses(c, p, v; held...))
    case!(cases, "bundle/spline3_with_forward_exact/refused_refit_forward", (fwd, sp0, sviews),
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:controls,), refit_forward = false))
    case!(cases, "bundle/spline3_with_forward_exact/refused_forward_tolerance", (fwd, sp0, sviews),
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:controls,), forward_tolerance = 1.0e-3))
    bt = (ecf_bounded(struth), sp0, sviews)
    case!(cases, "bundle/spline3_truth_exact/refused_held_no_scale", bt, ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:controls,)))
    case!(cases, "bundle/spline3_masked_exact/held_ls", (ecf_bounded(ecf_masked_spline()), sp0, sviews), ((c, p, v),) -> refine_camera_and_poses(c, p, v; held...))
    case!(cases, "bundle/spline3_truth_exact/refused_box", bt, ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:controls,), box = (; c_1_1_x = (0.0, 1.0))))
    case!(cases, "bundle/spline3_truth_exact/refused_fixed_one_control", bt, ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:c_1_1_x,)))
    case!(cases, "bundle/spline3_truth_exact/refused_retired_free", bt, ((c, p, v),) -> refine_camera_and_poses(c, p, v; free = ()))
    P = sp0[1]
    case!(cases, "bundle/spline_flat_one_point/refused_fold", (ecf_bounded(ecf_flat_spline()), [P], [[P.R' * (SVector(0.0, 0.0, 2.0) - P.t) => SVector(640.0, 480.0)]]),
          ((c, p, v),) -> refine_camera_and_poses(c, p, v; held...))
    for seed in SEEDS
        sn = ecf_contaminate_inner(StableRNG(seed), sviews)
        f = (ecf_bounded(struth), sp0, sn)
        case!(cases, "bundle/spline3_truth_contaminated/held_ls/seed$seed", f, ((c, p, v),) -> refine_camera_and_poses(c, p, v; held...))
        case!(cases, "bundle/spline3_truth_contaminated/held_triggs_huber1.345_scale0.2/seed$seed", f,
              ((c, p, v),) -> refine_camera_and_poses(c, p, v; fixed = (:controls,), optimization_strategy = H, scale = 0.2))
    end

    # --- rotation-centre views of a held pinhole camera ------------------------------------------------------------
    t = ecf_center_truth(ECF_CENTER_AXES)
    start = ecf_center_start(t)
    hp = held_parameters(cam)
    case!(cases, "centerbundle/pinhole6_exact/ls", (cam, start, ecf_center_views(cam, t)), ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), ls...))
    upstream = FitResult(0.3, ReturnCode.Success, ClosedForm, 0, NaN, 6, (; scale = 0.3, scale_source = ScaleEstimated))
    for seed in SEEDS   # StableRNG(2) is the suite's own draw
        vs = ecf_center_views(cam, t; σ = 0.3, outliers = 0.05, rng = StableRNG(seed))
        id = "/seed$seed"
        f = (cam, start, vs)
        case!(cases, "centerbundle/pinhole6_contaminated/triggs_huber3_upstream$id", (f..., upstream),
              ((c, s, v, u),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), optimization_strategy = Triggs{HuberShape}(efficiency_constant = 3.0), scale = u))
        case!(cases, "centerbundle/pinhole6_contaminated/ls$id", f, ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), ls...))
        case!(cases, "centerbundle/pinhole6_contaminated/triggs_huber1.345_scale0.3$id", f,
              ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), optimization_strategy = H, scale = 0.3))
        case!(cases, "centerbundle/pinhole6_contaminated/triggs_huber_width0.9$id", f,
              ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), optimization_strategy = Triggs{HuberShape}(), width = 0.9))
    end
    case!(cases, "centerbundle/pinhole6_clean_sigma0.1/ls", (cam, start, ecf_center_views(cam, t; σ = 0.1, rng = StableRNG(7))),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), ls...))
    one_axis = ecf_center_truth([SVector(0.0, 0.0, a) for a in -0.1:0.05:0.1])
    case!(cases, "centerbundle/pinhole_one_axis/refused", (cam, RotationCenterViews(one_axis.R, one_axis.c, one_axis.d), ecf_center_views(cam, one_axis)),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), ls...))
    two = ecf_center_truth(ECF_CENTER_AXES[1:2])
    case!(cases, "centerbundle/pinhole_two_views/refused", (cam, RotationCenterViews(two.R, two.c, two.d), ecf_center_views(cam, two)),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), ls...))
    case!(cases, "centerbundle/pinhole6_exact/refused_free_camera", (cam, start, ecf_center_views(cam, t)), ((c, s, v),) -> refine_camera_and_poses(c, s, v; ls...))
    case!(cases, "centerbundle/pinhole6_exact/refused_missing_view", (cam, start, Dict(k => v for (k, v) in ecf_center_views(cam, t) if k != 3)),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), ls...))

    # --- rotation-centre views of a held SplineLens camera ----------------------------------------------------------
    st = ecf_center_truth(4 .* ECF_CENTER_AXES[1:5])
    sstart = ecf_center_start(st)
    sexact = ecf_center_views(struth, st)
    hc = (; fixed = (:controls,))
    case!(cases, "centerbundle/spline5_exact/ls", (struth, sstart, sexact), ((c, s, v),) -> refine_camera_and_poses(c, s, v; hc..., ls...))
    case!(cases, "centerbundle/spline5_exact/ls_held_parameters", (struth, sstart, sexact),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = held_parameters(c), ls...))
    Q = SMatrix{3, 3, Float64, 9}(exp_so3(SVector(0.03, -0.035, 0.015)))
    pxs = [SVector(x, y) for x in range(8.0, 1272.0; length = 12) for y in range(8.0, 952.0; length = 9)]
    rays = [normalize(backproject(struth, u)) for u in pxs]
    rim = Dict(k => [(R' * (2.5 * f - (st.d - R * st.c))) => u for (f, u) in zip(rays, pxs)] for (k, R) in st.R)
    turned = RotationCenterViews(Dict(k => Q * R for (k, R) in st.R), st.c, Q * st.d)
    case!(cases, "centerbundle/spline5_rim_turned_start/ls", (struth, turned, rim), ((c, s, v),) -> refine_camera_and_poses(c, s, v; hc..., ls...))
    snoisy = ecf_center_views(struth, st; σ = 0.3, rng = StableRNG(5))
    for (lbl, strategy, input) in (("triggs_huber3_scale0.3", Triggs{HuberShape}(efficiency_constant = 3.0), (; scale = 0.3)),
                                   ("triggs_huber1.345_scale0.3", H, (; scale = 0.3)),
                                   ("triggs_huber_width0.9", Triggs{HuberShape}(), (; width = 0.9)))
        case!(cases, "centerbundle/spline5_noisy/$lbl", (struth, sstart, snoisy),
              ((c, s, v),) -> refine_camera_and_poses(c, s, v; hc..., optimization_strategy = strategy, input...))
    end
    sbad = ecf_center_views(struth, st; σ = 0.3, outliers = 0.05, rng = StableRNG(3))
    case!(cases, "centerbundle/spline5_contaminated/triggs_huber3_scale0.3", (struth, sstart, sbad),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v; hc..., optimization_strategy = Triggs{HuberShape}(efficiency_constant = 3.0), scale = 0.3))
    case!(cases, "centerbundle/spline5_contaminated/ls", (struth, sstart, sbad), ((c, s, v),) -> refine_camera_and_poses(c, s, v; hc..., ls...))
    present = trues(10, 12); present[8, 10] = false
    case!(cases, "centerbundle/spline5_masked_exact/ls", (with_present(struth, present), sstart, sexact), ((c, s, v),) -> refine_camera_and_poses(c, s, v; hc..., ls...))
    case!(cases, "centerbundle/spline5_exact/refused_free_camera", (struth, sstart, sexact), ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = (), ls...))
    case!(cases, "centerbundle/spline5_exact/refused_box", (struth, sstart, sexact),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v; hc..., box = (; f = (100.0, 900.0)), ls...))
    case!(cases, "centerbundle/spline5_exact/refused_fixed_controls_f", (struth, sstart, sexact),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v; fixed = (:controls, :f), ls...))
    sv5 = ecf_center_views(struth, st; σ = 0.5)
    case!(cases, "centerbundle/spline5_noisy0.5/ls_bearing_residual", (struth, sstart, sv5),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v, BearingResidual(); hc..., ls...))
    case!(cases, "centerbundle/spline5_noisy0.5/ls_reprojection_residual", (struth, sstart, sv5),
          ((c, s, v),) -> refine_camera_and_poses(c, s, v, ReprojectionResidual(); hc..., ls...))
    case!(cases, "centerbundle/spline5_noisy0.5/ls", (struth, sstart, sv5), ((c, s, v),) -> refine_camera_and_poses(c, s, v; hc..., ls...))
    return cases
end
