# The camera fits on Core 0.14 (record/est_camera_fits.jl, case for case): CameraRegression, PSpline, ForwardSpline and
# OpticalAxis on a CameraRegressionProblem; its LevenbergMarquardt refine; the old two-in-one (the linear fit, then its
# refine: `CameraRegression(; refine = LevenbergMarquardt())`); DivisionCameraFit, RadialCameraFit; and the
# LevenbergMarquardt refine of a CameraPosesProblem (the views form, the board form, the rotation-centre form).
#
# Each old call is translated below, keyword by keyword, into the new solve: the problem's settings (support, fixed,
# weight, device, the bundle's held sets and residual; the refused spellings with them, as the old entries refused them),
# the algorithm's fields, the solve's run settings (scale, width, stops: the old LM keywords `max_iter`, `ftol`, `xtol`,
# `max_time` as `stops`). Fixtures that hold an upstream result are rebuilt with the new API (`fixture_map`).
using VisualGeometryToolkitCore: parameter_names, FitResult, ReturnCode, ClosedForm, ScaleEstimated, Triggs, Irls, SolveStats,
    HuberShape, WelschShape, TLSShape, StudentTUniformScoring, JeffreysPrior, FlatLabelPrior, MsacScoring, CorrespondenceNormalization,
    with_unit, HomographyMat, ClassDefault, LmStops, RadialCamera
using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, PSpline, ForwardSpline, OpticalAxis, CameraPosesProblem,
    ImageMeasure, SolidAngleMeasure,
    DivisionCameraFit, RadialCameraFit, RadialCameraProblem, mixture_em_from_toml,
    default_config_path, scoring_from_toml, DivisionCameraProblem, MixtureEmConfig, KeepModel, NoImprovementStop, TerminalLO,
    BearingResidual, ReprojectionResidual, held_parameters, RotationCenterViews
using TOML: TOML
Base.include(@__MODULE__, joinpath(@__DIR__, "_common.jl"))

const ECF_RETIRED = (:free, :fix, :fixed_params, :hold)

# --- the translations of the old calls -----------------------------------------------------------------------------------
_pick(kw, keys) = NamedTuple(k => v for (k, v) in pairs(kw) if k in keys)
_drop(kw, keys) = NamedTuple(k => v for (k, v) in pairs(kw) if !(k in keys))
const _REG_ALG = (:tolerance, :grid_size, :lattice_spacing, :forward)
# the regression from scratch: the problem's settings, the algorithm's fields
fitcam(T, s, r; kw...) = solve(EstimationProblem(CameraRegressionProblem(T, s, r; _drop(kw, _REG_ALG)...)), CameraRegression(; _pick(kw, _REG_ALG)...))
# the refine: the LM's strategy, the solve's scale input and stops (`max_iter` the old cap: the regression's stops of record otherwise)
function refcam(g, s, r; optimization_strategy = ClassDefault(), max_iter = nothing, kw...)
    run = _pick(kw, (:scale, :width, :local_optimization))
    max_iter === nothing || (run = merge(run, (; stops = LmStops(; max_iter))))
    return solve(RefineProblem(CameraRegressionProblem(s, r; _drop(kw, (:scale, :width, :local_optimization))...), g),
                 LevenbergMarquardt(; optimization_strategy); run...)
end
approx(L, s, r; max_iter = nothing, kw...) = max_iter === nothing ?
    solve(EstimationProblem(CameraRegressionProblem(L, s, r; kw...)), CameraRegression(; refine = LevenbergMarquardt())) :
    solve(EstimationProblem(CameraRegressionProblem(L, s, r; kw...)), CameraRegression(; refine = LevenbergMarquardt()); stops = LmStops(; max_iter))
axis(c; kw...) = solve(EstimationProblem(CameraRegressionProblem(c; kw...)), OpticalAxis())
axis(c, r; kw...) = solve(EstimationProblem(CameraRegressionProblem(c, r; kw...)), OpticalAxis())
pspl(T, s, r, gs; kw...) = solve(EstimationProblem(CameraRegressionProblem(T, s, r; kw...)), PSpline(; grid_size = gs))
fwdspl(c; tolerance, kw...) = solve(EstimationProblem(CameraRegressionProblem(c; kw...)), ForwardSpline(; tolerance))
const _DIV_CTX = (:image_size, :focal_seed, :k)
divfit(rc, v; kw...) = solve(EstimationProblem(CameraModel, (rc, v), _pick(kw, _DIV_CTX)), DivisionCameraFit(; _drop(kw, _DIV_CTX)...))
radfit(cs; center = nothing, kw...) =
    solve(center === nothing ? EstimationProblem(RadialCamera, cs) : EstimationProblem(RadialCamera, cs, (; center)), RadialCameraFit(; kw...))
# the bundle: the guess in the result's form; the LM's strategy; the scale input and the old LM keywords as the solve's
const _LM_KEYS = (:max_iter, :ftol, :xtol, :max_time)
function _bundle(f, u0; optimization_strategy = ClassDefault(), kw...)
    run = _pick(kw, (:scale, :width))
    stops = _pick(kw, _LM_KEYS)
    isempty(stops) || (run = merge(run, (; stops = poses_stops(; stops...))))
    return solve(RefineProblem(f(_drop(kw, (:scale, :width, _LM_KEYS...))), u0), LevenbergMarquardt(; optimization_strategy); run...)
end
bundle(c, p, v; kw...) = p isa RotationCenterViews ? _bundle(k -> CameraPosesProblem(v; k...), (; camera = c, views = p); kw...) :
    _bundle(k -> CameraPosesProblem(v; k...), (; camera = c, poses = p); kw...)
bundle(c, p, b, v; kw...) = b isa VgtkEstimation.AbstractViewResidual ? _bundle(k -> CameraPosesProblem(v; residual = b, k...), (; camera = c, views = p); kw...) :
    _bundle(k -> CameraPosesProblem(v; k...), (; camera = c, poses = p, board_poses = b); kw...)
u_of(r) = r.u
md_of(r) = r.metadata

# every case: the result's mapping (a refusal's message names the old entries: its type is compared, its text not); a
# fixture slot that holds an upstream result (the bundle's `rd`, the centre bundle's `upstream`) mapped as a result
# (the bundles' metadata spell the stack's one `s2` and `dof`, read by `s2(r)`/`dof(r)`: the old `s²` and `ν`)
# (the sample density `metadata.weight` was a Symbol, `:image`; it is the measure singleton of that tag, `ImageMeasure()`)
const ECF_MAP = FieldMap(; renames = [RESULT_MAP.renames; "metadata.s2" => "metadata.s²"; "metadata.dof" => "metadata.ν"],
                         new_only = RESULT_MAP.new_only, retyped = RESULT_MAP.retyped, retagged = ["metadata.weight" => "Measure"])
const ECF_REFUSAL = FieldMap(; renames = ECF_MAP.renames, new_only = ECF_MAP.new_only, retyped = ECF_MAP.retyped, ignored = ["exception.msg"],
                             retagged = ECF_MAP.retagged)
# the recorded refusals whose message named an old entry though their id does not say "refused" (a family's linear fit)
const ECF_OLD_MESSAGE = ("lensfit/DSL_rlit/linear", "lensfit/DSL_rcorner/linear")
function rp!(cases, id, fx, run)
    fmap = fx isa Tuple && length(fx) == 4 && fx[4] isa FitResult ? prefixed(ECF_MAP, "4") : IDENTITY_MAP
    return replay!(cases, id, fx, run; map = occursin("refused", id) || id in ECF_OLD_MESSAGE ? ECF_REFUSAL : ECF_MAP, fixture_map = fmap)
end

# the configured DivisionCameraFit EM table at another iteration cap (`em_at`): the suite's fixture file; the two-solve
# recipes (`poses_stops`): the suite's too
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "division_camera_em.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "camera_recipes.jl"))
ecf_em() = mixture_em_from_toml(TOML.parsefile(default_config_path())["division_camera_fit"]["mixture_em"]; section = "division_camera_fit.mixture_em")

const ECF_RETIRED = (:free, :fix, :fixed_params, :hold)

function replay_est_camera_fits(cases)
    _replay_ecf_lens!(cases)
    _replay_ecf_spline!(cases)
    _replay_ecf_axis!(cases)
    _replay_ecf_division!(cases)
    _replay_ecf_radial!(cases)
    _replay_ecf_bundle!(cases)
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# parametric lens families: the linear fit, its refine, the two in one
# ---------------------------------------------------------------------------------------------------------------
function _replay_ecf_lens!(cases)
    mdiv, rc = ecf_mdiv(), ECF_RCORNER
    for (name, (L, m, region)) in ecf_lens_cases()
        rp!(cases, "lensfit/$name/linear", (m, region), ((m, region),) -> fitcam(L, m, region))
        rp!(cases, "lensapprox/$name/default", (m, region), ((m, region),) -> approx(L, m, region))
    end
    md = (mdiv, disc(mdiv, rc))
    rp!(cases, "lensfit/KB4_mdiv/weight_image", md, ((m, r),) -> fitcam(ECF_KB4, m, r; weight = ImageMeasure()))
    rp!(cases, "lensfit/KB4_mdiv/weight_solid_angle", md, ((m, r),) -> fitcam(ECF_KB4, m, r; weight = SolidAngleMeasure()))
    rp!(cases, "lensfit/KB4_mdiv/refused_weight_pixels", md, ((m, r),) -> fitcam(ECF_KB4, m, r; weight = :pixels))
    rp!(cases, "lensfit/orthographic_mdiv/refused_view_angle", md, ((m, r),) -> fitcam(typeof(Lens(Orthographic())), m, r))
    rp!(cases, "lensfit/Division1_mdiv/linear", md, ((m, r),) -> fitcam(Division{1}, m, r))
    rp!(cases, "lensfit/LiftLens_mdiv/refused_no_arity", md, ((m, r),) -> fitcam(LiftLens, m, r))
    rp!(cases, "lensfit/KannalaBrandt_mdiv/refused_no_arity", md, ((m, r),) -> fitcam(KannalaBrandt, m, r))
    rp!(cases, "lensfit/DECENTERED_mdiv1000/linear", (mdiv, disc(mdiv, 1000.0)), ((m, r),) -> fitcam(typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0)))), m, r))
    synth = CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = Lens(Perspective(), Additive(Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)),
        Tangential(1.0e-3, -5.0e-4), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3)))))
    rp!(cases, "lensfit/OPENCV3_decentred_source/stripped", (synth, disc(synth, 500.0)), ((m, r),) -> fitcam(Lens{Perspective, Radial{Even, 3, 0, Float64}}, m, r))
    frame = Rect(0, 0, 4240, 2832)
    rp!(cases, "lensfit/KB4_mdiv_frame/linear", (mdiv, frame), ((m, r),) -> fitcam(ECF_KB4, m, r))
    rp!(cases, "lensfit/KB4_mdiv/refused_off_pp_region", (mdiv, Rect(0, 0, 100, 100)), ((m, r),) -> fitcam(ECF_KB4, m, r))
    rp!(cases, "lensfit/KB4_mdiv/refused_whole_plane", (mdiv, VgtkCore.WholePlane()), ((m, r),) -> fitcam(ECF_KB4, m, r))
    kb200 = CameraModel(200.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
    rp!(cases, "lensfit/KB4_kb200/refused_beyond_support", (kb200, Rect(0, 0, 1000, 1000)), ((m, r),) -> fitcam(ECF_KB4, m, r))
    kb135 = CameraModel(SMatrix{3, 3, Float64}(300, 0, 0, 0, 300, 0, 320, 240, 1); lens = KannalaBrandt(-0.135))
    rp!(cases, "lensfit/KB4_kb135/refused_beyond_injectivity", (kb135, Rect(-1.0e5, -1.0e5, 2.0e5, 2.0e5)), ((m, r),) -> fitcam(ECF_KB4, m, r))

    # camera_verbs: the icra2026 fisheye
    ic = ecf_icra()
    rp!(cases, "lensfit/RATTP_icra84/linear", (ic.kb, ic.region), ((m, r),) -> fitcam(typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0), ThinPrism((0.0, 0.0), (0.0, 0.0))))), m, r))
    rp!(cases, "lensfit/RATTP_icra_wide/refused_view_angle", (ic.kb, ic.wide), ((m, r),) -> fitcam(typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0), ThinPrism((0.0, 0.0), (0.0, 0.0))))), m, r))
    rp!(cases, "lensapprox/RATTP_icra_wide/refused_view_angle", (ic.kb, ic.wide), ((m, r),) -> approx(typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0), ThinPrism((0.0, 0.0), (0.0, 0.0))))), m, r))
    rp!(cases, "lensfit/KB4_icra_wide/linear", (ic.kb, ic.wide), ((m, r),) -> fitcam(ECF_KB4, m, r))
    rp!(cases, "lensapprox/RATTP_icra84/default", (ic.kb, ic.region), ((m, r),) -> approx(typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0), ThinPrism((0.0, 0.0), (0.0, 0.0))))), m, r))

    # the linear residual and its refine (camera_verbs: KB4, SC3, Division{2} over the 2000 px disc)
    for (lbl, L) in (("KB4", ECF_KB4), ("SC3", ECF_SC3), ("Division2", Division{2}))
        region = disc(mdiv, 2000.0)
        g = u_of(fitcam(L, mdiv, region))
        rp!(cases, "lensrefine/$(lbl)_mdiv2000/from_linear", (g, mdiv, region), ((g, m, r),) -> refcam(g, m, r))
    end
    # the families with no linear form: refused by fit_camera, started from their anchor by approximate_camera
    nolin = ["DSL" => typeof(Lens(DoubleSphere(0.5, 0.5))), "Unified" => typeof(Lens(Unified(0.0))), "ExtendedUnified" => typeof(Lens(ExtendedUnified(0.0, 0.0))),
             "FieldOfView" => typeof(Lens(Perspective(), FieldOfView(0.0))), "TiltRadial" => typeof(Lens(Perspective(), Tilt(0.0, 0.0) ∘ Radial((0.0, 0.0))))]
    d1500 = (mdiv, disc(mdiv, 1500.0))
    for (lbl, L) in nolin
        rp!(cases, "lensfit/$(lbl)_mdiv1500/refused_no_linear_form", d1500, ((m, r),) -> fitcam(L, m, r))
        rp!(cases, "lensapprox/$(lbl)_mdiv1500/default", d1500, ((m, r),) -> approx(L, m, r))
    end
    # refine_camera's keywords: L2 only, no scale input; the retired keyword
    g = u_of(fitcam(ECF_KB4, mdiv, disc(mdiv, 1500.0)))
    fx = (g, d1500...)
    rp!(cases, "lensrefine/KB4_mdiv1500/from_linear", fx, ((g, m, r),) -> refcam(g, m, r))
    rp!(cases, "lensrefine/KB4_mdiv1500/refused_scale", fx, ((g, m, r),) -> refcam(g, m, r; scale = 1.0))
    rp!(cases, "lensrefine/KB4_mdiv1500/refused_width", fx, ((g, m, r),) -> refcam(g, m, r; width = 1.0))
    rp!(cases, "lensrefine/KB4_mdiv1500/refused_strategy_symbol", fx, ((g, m, r),) -> refcam(g, m, r; optimization_strategy = :huber))
    rp!(cases, "lensrefine/KB4_mdiv1500/refused_local_optimization", fx, ((g, m, r),) -> refcam(g, m, r; local_optimization = nothing))
    # from a perturbed guess (focal × 1.05, coefficients × 0.8 of approximate_camera's optimum)
    for (lbl, L, R) in (("KB4", ECF_KB4, rc), ("SC3", ECF_SC3, rc), ("Division2", Division{2}, 2500.0))
        region = disc(mdiv, R)
        best = u_of(approx(L, mdiv, region))
        p = collect(parameter_vector(lens(best)))
        K = SMatrix{3, 3, Float64}(calibration_matrix(best))
        guess = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1] * 1.05, 0, 0, 0, K[2, 2] * 1.05, 0, K[1, 3], K[2, 3], 1); check = false)),
            with_parameters(lens(best), p .* 0.8))
        rp!(cases, "lensrefine/$(lbl)_mdiv$(round(Int, R))/perturbed_guess", (guess, mdiv, region), ((g, m, r),) -> refcam(g, m, r))
    end
    ga = u_of(approx(ECF_KB4, d1500...))
    K = SMatrix{3, 3, Float64}(calibration_matrix(ga))
    moved = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0, 0, 0, K[2, 2], 0, K[1, 3] + 3, K[2, 3], 1); check = false)), lens(ga))
    rp!(cases, "lensrefine/KB4_mdiv1500/refused_moved_pp", (moved, d1500...), ((g, m, r),) -> refcam(g, m, r))

    # `fixed` on the rational target (refine) and on the linear fit
    gr = u_of(fitcam(ECF_RAT, ic.kb, ic.region))
    fr = (gr, ic.kb, ic.region)
    rp!(cases, "lensrefine/RAT_icra84/fixed_d", fr, ((g, m, r),) -> refcam(g, m, r; fixed = (:d,)))
    rp!(cases, "lensrefine/RAT_icra84/fixed_d123", fr, ((g, m, r),) -> refcam(g, m, r; fixed = (:d1, :d2, :d3)))
    rp!(cases, "lensrefine/RAT_icra84/default", fr, ((g, m, r),) -> refcam(g, m, r))
    rp!(cases, "lensrefine/RAT_icra84/fixed_all", fr, ((g, m, r),) -> refcam(g, m, r; fixed = (:f, :k, :d)))
    rp!(cases, "lensrefine/RAT_icra84/refused_unknown_fixed", fr, ((g, m, r),) -> refcam(g, m, r; fixed = (:xi,)))
    for kw in ECF_RETIRED
        rp!(cases, "lensrefine/RAT_icra84/refused_retired_$kw", fr, ((g, m, r),) -> refcam(g, m, r; (kw => (:d,),)...))
        rp!(cases, "lensfit/RAT_icra84/refused_retired_$kw", (ic.kb, ic.region), ((m, r),) -> fitcam(ECF_RAT, m, r; (kw => (:d,),)...))
    end
    rp!(cases, "lensapprox/RAT_icra84/default", (ic.kb, ic.region), ((m, r),) -> approx(ECF_RAT, m, r))
    rp!(cases, "lensapprox/RAT_icra84/fixed_d", (ic.kb, ic.region), ((m, r),) -> approx(ECF_RAT, m, r; fixed = (:d,)))
    rp!(cases, "lensfit/RAT_icra84/fixed_d", (ic.kb, ic.region), ((m, r),) -> fitcam(ECF_RAT, m, r; fixed = (:d,)))
    rp!(cases, "lensfit/RAT_icra84/linear", (ic.kb, ic.region), ((m, r),) -> fitcam(ECF_RAT, m, r))
    rp!(cases, "lensfit/POLY_icra84/linear", (ic.kb, ic.region), ((m, r),) -> fitcam(ECF_POLY, m, r))
    grh = u_of(fitcam(ECF_RAT, ic.kb, ic.region; fixed = (:d,)))
    rp!(cases, "lensrefine/RAT_icra84/fixed_d_from_held_linear", (grh, ic.kb, ic.region), ((g, m, r),) -> refcam(g, m, r; fixed = (:d,)))
    # CaliBlobs' pattern: a target instance, the tangential terms and the denominator held at the target's values
    src = CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = Lens(Perspective(), Radial((-0.12, 0.03, 0.004))))
    target = Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(1.0e-4, -2.0e-4)))
    target0 = Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0)))
    dsc = Ellipse(SVector(960.0, 540.0), 700.0, 700.0, 0.0)
    rp!(cases, "lensfit/target_instance_tangential/fixed_p_d", (target, src, dsc), ((t, m, r),) -> fitcam(t, m, r; fixed = (:p, :d)))
    rp!(cases, "lensfit/target_instance_zero/fixed_p_d", (target0, src, dsc), ((t, m, r),) -> fitcam(t, m, r; fixed = (:p, :d)))
    allnames = parameter_names(CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = target))
    rp!(cases, "lensfit/target_instance_tangential/fixed_all", (target, src, dsc), ((t, m, r),) -> fitcam(t, m, r; fixed = allnames))
    ddisc = Ellipse(SVector(2120.0, 1416.0), 2000.0, 2000.0, 0.0)
    rp!(cases, "lensfit/division_instance/refused_hold_without_focal", (Division(0.0, -0.0765), mdiv, ddisc),
          ((t, m, r),) -> fitcam(t, m, r; fixed = (:d2,)))
    rp!(cases, "lensfit/division_instance/fixed_f_d2", (Division(0.0, -0.0765), mdiv, ddisc), ((t, m, r),) -> fitcam(t, m, r; fixed = (:f, :d2)))

    # a lens refitted onto its own family (zero residuals)
    zregion = Ellipse(SVector(320.0, 240.0), 200.0, 200.0, 0.0)
    for (lbl, l) in (("perspective", Lens(Perspective())), ("equidistant", Lens(Equidistant())), ("stereographic", Lens(Stereographic())),
                     ("division2_zero", zero(Division{2})), ("scaramuzza3_zero", zero(Scaramuzza{3})))
        s = CameraModel(; pp = (320.0, 240.0), focal = 300.0, lens = l)
        rp!(cases, "lensfit/own_family_$lbl/linear", (s, zregion), ((m, r),) -> fitcam(typeof(lens(m)), m, r))
        rp!(cases, "lensapprox/own_family_$lbl/default", (s, zregion), ((m, r),) -> approx(typeof(lens(m)), m, r))
    end

    # termination record: the fisheye model over the lit disc; its refine, capped, everything held
    fe = ecf_fisheye()
    rp!(cases, "lensfit/KB4_fisheye_disc/linear", (fe.FISHEYE, fe.REGION), ((m, r),) -> fitcam(ECF_KB4, m, r))
    lin = u_of(fitcam(ECF_KB4, fe.FISHEYE, fe.REGION))
    ff = (lin, fe.FISHEYE, fe.REGION)
    rp!(cases, "lensrefine/KB4_fisheye_disc/default", ff, ((g, m, r),) -> refcam(g, m, r))
    rp!(cases, "lensrefine/KB4_fisheye_disc/max_iter1", ff, ((g, m, r),) -> refcam(g, m, r; max_iter = 1))
    rp!(cases, "lensrefine/KB4_fisheye_disc/fixed_all", ff, ((g, m, r),) -> refcam(g, m, r; fixed = Tuple(parameter_names(g))))
    rp!(cases, "lensapprox/Division2_fisheye_disc/default", (fe.FISHEYE, fe.REGION), ((m, r),) -> approx(Division{2}, m, r))
    # a parametric lens from a SplineLens camera (bounded and bare), its linear fit, the image density only
    rp!(cases, "lensapprox/KB4_grid_bounded/default", (fe.GRID_BOUNDED, fe.REGION), ((m, r),) -> approx(ECF_KB4, m, r))
    rp!(cases, "lensapprox/KB4_grid/default", (fe.GRID, fe.REGION), ((m, r),) -> approx(ECF_KB4, m, r))
    rp!(cases, "lensfit/KB4_grid/linear", (fe.GRID, fe.REGION), ((m, r),) -> fitcam(ECF_KB4, m, r))
    rp!(cases, "lensfit/KB4_grid/refused_solid_angle", (fe.GRID, fe.REGION), ((m, r),) -> fitcam(ECF_KB4, m, r; weight = SolidAngleMeasure()))
    rp!(cases, "lensapprox/KB4_grid/refused_solid_angle", (fe.GRID, fe.REGION), ((m, r),) -> approx(ECF_KB4, m, r; weight = SolidAngleMeasure()))
    # a non-central source: not supported yet
    nc = ecf_zero_noncentral()
    rp!(cases, "lensfit/KB4_noncentral/refused", (nc, fe.REGION), ((m, r),) -> fitcam(ECF_KB4, m, r))
    rp!(cases, "lensfit/KB4_noncentral_bounded/refused", (BoundedCamera(nc, Size2(1000, 1000)), fe.REGION), ((m, r),) -> fitcam(ECF_KB4, m, r))
    rp!(cases, "lensapprox/KB4_noncentral/refused", (nc, fe.REGION), ((m, r),) -> approx(ECF_KB4, m, r))
    rp!(cases, "lensrefine/KB4_noncentral/refused", (lin, nc, fe.REGION), ((g, m, r),) -> refcam(g, m, r))
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# spline targets: fitcam(SplineLens / NoncentralSplineModel), pspline_grid, fit_forward_spline
# ---------------------------------------------------------------------------------------------------------------
function _replay_ecf_spline!(cases)
    fe = ecf_fisheye()
    gb, R = fe.GRID_BOUNDED, fe.REGION
    rp!(cases, "splinefit/grid_bounded_disc/tol1e-3", (gb, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3))
    rp!(cases, "splinefit/grid_bounded_disc/tol1e-3_backward_only", (gb, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, forward = false))
    cam = u_of(fitcam(SplineLens, gb, R; tolerance = 1.0e-3))
    rp!(cases, "splinefit/refit_of_fitted/tol1e-3", (BoundedCamera(cam, Size2(1000, 1000)), R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3))
    rp!(cases, "splinefit/grid_bounded_disc/refused_grid_size_on_spline", (gb, R),
          ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, grid_size = (40, 40), forward = false))
    rp!(cases, "splinefit/grid_bounded_disc/spacing9", (gb, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 9.0, forward = false))
    sensor = Size2(4256, 2848)
    lens14 = CameraModel(1727.0px, SA[2140.0px, 1415.0px]; lens = KannalaBrandt(0.02))
    for g in ((16, 18), (26, 20))
        s = BoundedCamera(CameraModel(SplineLens, lens14, sensor; grid_size = g), sensor)
        rp!(cases, "splinefit/rc_lattice_$(g[1])x$(g[2])/spacing60", (s, Rect(0, 0, 4256, 2848)),
              ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 60.0, forward = false))
    end
    rp!(cases, "splinefit/kb14_parametric/refused_lattice_spacing", (BoundedCamera(lens14, sensor), Rect(0, 0, 4256, 2848)),
          ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 60.0))
    part = BoundedCamera(CameraModel(SplineLens, fe.GRID, Size2(1000, 1000); grid_size = (30, 30), calib_area = Rect(100, 100, 800, 800)), Size2(1000, 1000))
    rp!(cases, "splinefit/partial_lattice_disc/spacing20_extended", (part, R),
          ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 20.0, forward = false))
    rp!(cases, "splinefit/partial_lattice_inner/spacing20", (part, Rect(200, 200, 600, 600)),
          ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 20.0, forward = false))
    rp!(cases, "splinefit/partial_lattice_disc/refused_support", (part, R),
          ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 20.0, support = Rect(100, 100, 800, 800), forward = false))
    # the Schöps source on its lattice edge
    sch = ecf_schoeps()
    sensor640 = Rect(0.0, 0.0, 640.0, 480.0)
    for (lbl, region, spacing) in (("sensor", sensor640, nothing), ("sensor_spacing80", sensor640, 80.0),
                                   ("sensor_disc380", sensor640 ∩ Ellipse(SVector(320.0, 240.0), 380.0, 380.0, 0.0), nothing),
                                   ("sensor_ellipse", sensor640 ∩ Ellipse(SVector(320.05, 240.0), 319.95, 400.0, 0.0), nothing),
                                   ("inner", Rect(10.0, 10.0, 600.0, 440.0), nothing))
        rp!(cases, "splinefit/schoeps_$lbl/tolInf", (sch, region),
              ((s, r),) -> fitcam(SplineLens, s, r; tolerance = Inf, forward = false, lattice_spacing = spacing))
    end
    # region shapes; out-of-reach tolerances; no weight on a spline target
    rp!(cases, "splinefit/grid_bounded_cut/tol1e-3", (gb, Rect(0, 150, 1000, 700) ∩ fe.DISC), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3))
    wide = BoundedCamera(CameraModel(500.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02)), Size2(1000, 1000))
    rp!(cases, "splinefit/kb500_whole_sensor/tol1e-3", (wide, Rect(0, 0, 1000, 1000)), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3))
    rp!(cases, "splinefit/grid_bounded_whole_sensor/tol1e-3", (gb, Rect(0, 0, 1000, 1000)), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3))
    rp!(cases, "splinefit/grid_bounded_disc/tol1e-7_out_of_reach", (gb, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-7))
    rp!(cases, "splinefit/grid_bounded_disc/refused_weight", (gb, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, weight = ImageMeasure()))
    # a backward lattice from a parametric model
    bc = fe.BC
    rp!(cases, "splinefit/fisheye_bounded_disc/tol1e-4_backward", (bc, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-4, forward = false))
    rp!(cases, "splinefit/fisheye_bounded_disc/tol1e-12_backward_out_of_reach", (bc, R),
          ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-12, forward = false))
    small = BoundedCamera(CameraModel(18.0px, SA[30.0px, 30.0px]; lens = KannalaBrandt(-0.02)), Size2(60, 60))
    sreg = Rect(0, 0, 60, 60) ∩ Ellipse(SVector(30.0, 30.0), 29.0, 29.0, 0.0)
    rp!(cases, "splinefit/small60/tol1e-6_backward_out_of_reach", (small, sreg), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-6, forward = false))
    rp!(cases, "splinefit/fisheye_bounded_disc/grid40x40_backward", (bc, R),
          ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-4, grid_size = (40, 40), forward = false))
    rp!(cases, "splinefit/fisheye_bounded_disc/grid40x30_backward", (bc, R),
          ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-4, grid_size = (40, 30), forward = false))
    rp!(cases, "splinefit/fisheye_bare/refused_no_image", (fe.FISHEYE, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 0.01))
    rp!(cases, "splinefit/fisheye_bounded_disc/refused_tol0", (bc, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 0.0))
    rp!(cases, "splinefit/fisheye_bounded_disc/tol1e-2_backward", (bc, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-2, forward = false))
    # a function source: refused; its SplineLens camera (Core's source constructor) fitted
    rp!(cases, "splinefit/function_source/refused_no_image", (ecf_fisheye_ray, R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-2))
    fcam = CameraModel(SplineLens, ecf_fisheye_ray, Size2(1000, 1000); grid_size = (40, 40), init = :quasi_interpolate)
    rp!(cases, "splinefit/function_camera_bounded/tol1e-2", (BoundedCamera(fcam, Size2(1000, 1000)), R), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-2))
    # a degenerate source Jacobian at the left edge
    rp!(cases, "splinefit/degenerate_jacobian_grid/spacing13.5", (ecf_degenerate_grid(), R),
          ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3, lattice_spacing = 13.5, forward = false))
    # spline_fit_sources: a bare model refused; a bounded KB past its injectivity disc refused; inside it solved
    m = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    rp!(cases, "splinefit/kb250_bare/refused_no_image", (m, Rect(0, 0, 1280, 960)), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-2))
    rp!(cases, "splinefit/kb250_bare/refused_no_image_noncentral", (m, Rect(0, 0, 1280, 960)),
          ((s, r),) -> fitcam(NoncentralSplineModel, s, r; tolerance = 1.0e-2))

    # non-central targets
    S = noncentral_fisheye(noncentral_central())
    rp!(cases, "noncentralfit/source_disc/tol3e-4", (S, R), ((s, r),) -> fitcam(NoncentralSplineModel, s, r; tolerance = 3.0e-4))
    rp!(cases, "noncentralfit/source_disc/tol1e-3", (S, R), ((s, r),) -> fitcam(NoncentralSplineModel, s, r; tolerance = 1.0e-3))
    rp!(cases, "noncentralfit/source_translated_disc/tol3e-4", (noncentral_fisheye(noncentral_central(), SVector(0.5, -0.25, 2.0)), R),
          ((s, r),) -> fitcam(NoncentralSplineModel, s, r; tolerance = 3.0e-4))
    rp!(cases, "noncentralfit/source_disc/tol1e-9_out_of_reach", (S, R), ((s, r),) -> fitcam(NoncentralSplineModel, s, r; tolerance = 1.0e-9))
    rp!(cases, "noncentralfit/model_bare/refused_no_image", (VgtkCore.model(S), R), ((s, r),) -> fitcam(NoncentralSplineModel, s, r; tolerance = 1.0e-3))
    rp!(cases, "noncentralfit/central_fisheye_bounded/tol1e-3", (bc, R), ((s, r),) -> fitcam(NoncentralSplineModel, s, r; tolerance = 1.0e-3))
    rp!(cases, "noncentralfit/small60/tol1e-6_out_of_reach", (small, sreg), ((s, r),) -> fitcam(NoncentralSplineModel, s, r; tolerance = 1.0e-6))
    rp!(cases, "noncentralfit/fisheye_bare/refused", (fe.FISHEYE, R), ((s, r),) -> fitcam(NoncentralSplineModel, s, r; tolerance = 0.01))

    # pspline_grid: at the lattice fit_camera chose, and at two given ones
    gw, gh = md_of(fitcam(SplineLens, bc, R; tolerance = 1.0e-3)).family.grid_size
    for (lbl, gs) in (("fit_choice", (gw, gh)), ("35x25", (35, 25)), ("105x75", (105, 75)))
        rp!(cases, "pspline/fisheye_bounded_disc/$lbl", (bc, R, gs), ((s, r, gs),) -> pspl(SplineLens, s, r, gs))
    end
    rp!(cases, "pspline/fisheye_bare/refused_no_image", (fe.FISHEYE, R, (35, 25)), ((s, r, gs),) -> pspl(SplineLens, s, r, gs))
    rp!(cases, "pspline/fisheye_bounded_disc/refused_3x9", (bc, R, (3, 9)), ((s, r, gs),) -> pspl(SplineLens, s, r, gs))
    gwn, ghn = md_of(fitcam(NoncentralSplineModel, S, R; tolerance = 1.0e-3)).family.grid_size
    for (lbl, gs) in (("fit_choice", (gwn, ghn)), ("35x25", (35, 25)), ("105x75", (105, 75)))
        rp!(cases, "pspline/noncentral_source_disc/$lbl", (S, R, gs), ((s, r, gs),) -> pspl(NoncentralSplineModel, s, r, gs))
    end
    kbm = BoundedCamera(m, Size2(1280, 960))
    rp!(cases, "pspline/kb250_bounded_frame/refused_beyond_support", (kbm, Rect(0, 0, 1280, 960), (8, 8)), ((s, r, gs),) -> pspl(SplineLens, s, r, gs))
    rp!(cases, "pspline/kb250_bounded_valid_domain/8x8", (kbm, valid_domain(m), (8, 8)), ((s, r, gs),) -> pspl(SplineLens, s, r, gs))
    rp!(cases, "pspline/kb250_bare/refused_no_image", (m, Rect(0, 0, 1280, 960), (8, 8)), ((s, r, gs),) -> pspl(SplineLens, s, r, gs))

    # fit_forward_spline
    kb = CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
    fcam12 = BoundedCamera(CameraModel(SplineLens, kb, Size2(1280, 960); grid_size = (12, 10)), Size2(1280, 960))
    rp!(cases, "forwardspline/kb_spline_12x10/tol1e-3", fcam12, c -> fwdspl(c; tolerance = 1.0e-3))
    rp!(cases, "forwardspline/kb_parametric/refused", BoundedCamera(kb, Size2(1280, 960)), c -> fwdspl(c; tolerance = 1.0e-3))
    rp!(cases, "forwardspline/noncentral_bounded/refused", S, c -> fwdspl(c; tolerance = 1.0e-3))
    rp!(cases, "forwardspline/noncentral_model/refused", VgtkCore.model(S), c -> fwdspl(c; tolerance = 1.0e-3))
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# optical_axis, and the gauge it gives fitcam(SplineLens)
# ---------------------------------------------------------------------------------------------------------------
function _replay_ecf_axis!(cases)
    oa = ecf_off_axis()
    cov = with_lens_coverage(oa.OFF_BOUNDED, oa.DISC)
    rp!(cases, "axis/off_axis_covered/default", cov, c -> axis(c))
    rp!(cases, "axis/off_axis_bounded/default", oa.OFF_BOUNDED, c -> axis(c))
    rp!(cases, "axis/off_axis_bare/refused", oa.OFF_AXIS, c -> axis(c))
    L = typeof(u_of(axis(cov)).lens)
    rp!(cases, "lensapprox/axis_family_off_axis_covered/fixed_p_sx_sy", (L, oa.OFF_BOUNDED, imaged_region(cov)),
          ((L, s, r),) -> approx(L, s, r; fixed = (:p, :sx, :sy)))
    rp!(cases, "splinefit/off_axis_bounded_disc/tol1e-3", (oa.OFF_BOUNDED, oa.REGION), ((s, r),) -> fitcam(SplineLens, s, r; tolerance = 1.0e-3))
    rp!(cases, "lensfit/KB4_off_axis_bounded/linear", (oa.OFF_BOUNDED, oa.REGION), ((s, r),) -> fitcam(ECF_KB4, s, r))
    rp!(cases, "lensapprox/KB4_off_axis_bounded/default", (oa.OFF_BOUNDED, oa.REGION), ((s, r),) -> approx(ECF_KB4, s, r))
    r = axis(oa.OFF_BOUNDED)
    rp!(cases, "lensrefine/axis_off_axis_bare/from_own_optimum", (u_of(r), oa.OFF_AXIS, oa.REGION), ((g, m, r),) -> refcam(g, m, r))
    rp!(cases, "axis/decentred_covered/default", with_lens_coverage(ecf_decentred_spline(), oa.DISC), c -> axis(c))
    rp!(cases, "axis/fisheye_physical_covered/default", BoundedCamera(oa.FISHEYE, Size2(1000, 1000); lens_coverage = oa.DISC), c -> axis(c))
    fe = ecf_fisheye()
    rp!(cases, "axis/grid_covered/default", with_lens_coverage(fe.GRID_BOUNDED, fe.DISC), c -> axis(c))
    # the circular fisheye: its image circle, the whole sensor, a given region, a region past the image
    cf = ecf_circular_fisheye()
    sz = Size2(1400, 1000)
    bounded = BoundedCamera(cf.spl, sz; lens_coverage = cf.circle)
    rp!(cases, "axis/circular_fisheye_covered/default", bounded, c -> axis(c))
    rp!(cases, "axis/circular_fisheye_whole/default", BoundedCamera(cf.spl, sz), c -> axis(c))
    rp!(cases, "axis/circular_fisheye_whole/region_circle", (BoundedCamera(cf.spl, sz), cf.circle), ((c, r),) -> axis(c, r))
    rp!(cases, "axis/circular_fisheye_covered/region_sensor", (bounded, Rect(0, 0, 1400, 1000)), ((c, r),) -> axis(c, r))
    rp!(cases, "axis/circular_fisheye_covered/region_imaged", (bounded, imaged_region(bounded)), ((c, r),) -> axis(c, r))
    # region support: the RC siemens start camera
    sm = ecf_siemens()
    rp!(cases, "axis/siemens/refused_beyond_support", sm, ((c, s),) -> axis(c; support = s))
    rp!(cases, "lensapprox/KB4_siemens_imaged/refused_beyond_support", sm, ((c, s),) -> approx(ECF_KB4, c, imaged_region(c); support = s))
    rp!(cases, "lensfit/KB4_siemens_imaged/refused_beyond_support", sm, ((c, s),) -> fitcam(ECF_KB4, c, imaged_region(c); support = s))
    rp!(cases, "axis/siemens/region_support", sm, ((c, s),) -> axis(c, s; support = s))
    ga = u_of(axis(sm.cam, sm.support; support = sm.support))
    rp!(cases, "lensrefine/axis_siemens_imaged/refused_beyond_support", (ga, sm...), ((g, c, s),) -> refcam(g, c, imaged_region(c); support = s))
    rp!(cases, "lensrefine/axis_siemens_support/default", (ga, sm...), ((g, c, s),) -> refcam(g, c, s; support = s))
    # the folding sources: the order continuation
    for (lbl, src) in ecf_folding_sources()
        rp!(cases, "axis/folding_$lbl/default", src, c -> axis(c))
    end
    # the stepwise KannalaBrandt continuation by the public verbs (order_continuation_kb), each stage seeded by the last
    src = last(ecf_folding_sources()[2])
    part = imaged_region(src)
    g = u_of(fitcam(KannalaBrandt{1}, src, part))
    rp!(cases, "lensfit/KB1_folding_fold19.5_eps0.02/linear", (src, part), ((s, r),) -> fitcam(KannalaBrandt{1}, s, r))
    for N in 1:4
        k = Tuple(Float64.(parameter_vector(lens(g))))
        guess = CameraModel(SMatrix{3, 3, Float64}(calibration_matrix(g)); lens = KannalaBrandt(ntuple(i -> i <= length(k) ? k[i] : 0.0, N)...))
        rp!(cases, "lensrefine/KB$(N)_folding_fold19.5_eps0.02/continuation", (guess, src, part), ((g, s, r),) -> refcam(g, s, r))
        g = u_of(refcam(guess, src, part))
    end
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# fit_division_camera (division_solve_golden's scenes, the item's keywords and refusals, seeds)
# ---------------------------------------------------------------------------------------------------------------
function _replay_ecf_division!(cases)
    rc = ecf_radial_cameras()
    sz = (4000, 3000)
    clean = ecf_board_views(StableRNG(11))
    dirty = ecf_board_views(StableRNG(3); ε = 0.08)
    rp!(cases, "divcam/golden_clean/default", (rc, clean), ((rc, v),) -> divfit(rc, v; image_size = sz))
    rp!(cases, "divcam/golden_contaminated/default", (rc, dirty), ((rc, v),) -> divfit(rc, v; image_size = sz))
    rp!(cases, "divcam/golden_contaminated/ncoef2", (rc, dirty), ((rc, v),) -> divfit(rc, v; image_size = sz, ncoef = 2))
    rp!(cases, "divcam/golden_contaminated/known_k", (rc, dirty), ((rc, v),) -> divfit(rc, v; image_size = sz, k = _ECF_DB.K0))
    for seed in SEEDS
        vs = ecf_board_views(StableRNG(seed); ε = 0.08)
        rp!(cases, "divcam/contaminated/default/seed$seed", (rc, vs), ((rc, v),) -> divfit(rc, v; image_size = sz))
        rp!(cases, "divcam/contaminated/ncoef2/seed$seed", (rc, vs), ((rc, v),) -> divfit(rc, v; image_size = sz, ncoef = 2))
        rp!(cases, "divcam/contaminated/known_k/seed$seed", (rc, vs), ((rc, v),) -> divfit(rc, v; image_size = sz, k = _ECF_DB.K0))
        rp!(cases, "divcam/clean/default/seed$seed", (rc, ecf_board_views(StableRNG(seed))), ((rc, v),) -> divfit(rc, v; image_size = sz))
    end
    d = (rc, dirty)
    boards = [dirty[i] for i in 1:6]
    ones_vw = Dict(i => Correspondences(first.(b), last.(b); weights = ones(length(b))) for (i, b) in enumerate(boards))
    rp!(cases, "divcam/golden_contaminated/unit_weights", (rc, ones_vw), ((rc, v),) -> divfit(rc, v; image_size = sz))
    rp!(cases, "divcam/golden_contaminated/refused_key_mismatch", (rc, Dict(i => boards[i] for i in 1:5)), ((rc, v),) -> divfit(rc, v; image_size = sz))
    rp!(cases, "divcam/golden_contaminated/refused_no_image_size", d, ((rc, v),) -> divfit(rc, v))
    rp!(cases, "divcam/golden_contaminated/refused_negative_seed", d, ((rc, v),) -> divfit(rc, v; image_size = sz, focal_seed = -1.0))
    for mi in (1, 2, 5)
        rp!(cases, "divcam/golden_contaminated/em_max_iter$mi", d, ((rc, v),) -> divfit(rc, v; image_size = sz, mixture_em = em_at(mi)))
    end
    rp!(cases, "divcam/golden_contaminated/seed1800", d, ((rc, v),) -> divfit(rc, v; image_size = sz, focal_seed = 1800.0))
    rp!(cases, "divcam/golden_contaminated/seed1800_image8000", d, ((rc, v),) -> divfit(rc, v; image_size = (8000, 6000), focal_seed = 1800.0))
    rp!(cases, "divcam/golden_contaminated/seed1800_no_image_size", d, ((rc, v),) -> divfit(rc, v; focal_seed = 1800.0))
    c = SVector(1990.0, 1510.0)
    rcc = Dict(i => RadialCamera{Float64}(radial_rows(P), c) for (i, P) in rc)
    vwc = Dict(i => [X => x + c for (X, x) in b] for (i, b) in dirty)
    rp!(cases, "divcam/golden_contaminated_offcentre/seed1800", (rcc, vwc), ((rc, v),) -> divfit(rc, v; image_size = sz, focal_seed = 1800.0))
    # radial cameras robustly fitted from each board (the BabelCalib route): the radial camera's find, stored
    Ps = Dict(i => ransac_solve(RadialCameraProblem(b; center = (0.0, 0.0)); scoring = MsacScoring(RadialCameraProblem; threshold = 5.0), rng = StableRNG(9)).u
              for (i, b) in enumerate(boards))
    rp!(cases, "divcam/golden_contaminated_found_cameras/default", (Ps, dirty), ((rc, v),) -> divfit(rc, v; image_size = sz))
    zw = Dict(i => Correspondences(first.(b), last.(b); weights = (i == 6 ? zeros : ones)(length(b))) for (i, b) in enumerate(boards))
    rp!(cases, "divcam/golden_contaminated/view6_zero_weight", (rc, zw), ((rc, v),) -> divfit(rc, v; image_size = sz))
    neg = Dict(i => Correspondences(first.(b), last.(b); weights = (i == 2 ? fill(-1.0, length(b)) : ones(length(b)))) for (i, b) in enumerate(boards))
    rp!(cases, "divcam/golden_contaminated/refused_negative_weight", (rc, neg), ((rc, v),) -> divfit(rc, v; image_size = sz))
    allz = Dict(i => Correspondences(first.(b), last.(b); weights = zeros(length(b))) for (i, b) in vwc)
    rp!(cases, "divcam/golden_contaminated_offcentre/refused_all_zero", (rcc, allz), ((rc, v),) -> divfit(rc, v; focal_seed = 1800.0))
    # the shared scorer and EM tables
    sc = scoring_from_toml(TOML.parsefile(default_config_path())["division_camera_fit"]["weighting"]; problem = DivisionCameraProblem)
    em = ecf_em()
    rp!(cases, "divcam/golden_contaminated/explicit_tables", d, ((rc, v),) -> divfit(rc, v; image_size = sz, weighting = sc, mixture_em = em))
    rp!(cases, "divcam/golden_contaminated/em_cap3", d,
          ((rc, v),) -> divfit(rc, v; image_size = sz, mixture_em = MixtureEmConfig(em.model_update, em.nuisance, em.stop, 3)))
    rp!(cases, "divcam/golden_contaminated/halfwidth60", d, ((rc, v),) -> divfit(rc, v; image_size = sz,
          weighting = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 60.0); problem = DivisionCameraProblem)))
    for (lbl, kw) in (("outlier_halfwidth", (; outlier_halfwidth = 20.0)), ("max_iter", (; max_iter = 5)), ("scoring", (; scoring = sc)))
        rp!(cases, "divcam/golden_contaminated/refused_old_key_$lbl", d, ((rc, v),) -> divfit(rc, v; image_size = sz, kw...))
    end
    rp!(cases, "divcam/golden_contaminated/refused_codimension1_scorer", d, ((rc, v),) -> divfit(rc, v; image_size = sz,
          weighting = StudentTUniformScoring(2, 20.0; codimension = 1, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())))
    rp!(cases, "divcam/golden_contaminated/refused_keep_model", d, ((rc, v),) -> divfit(rc, v; image_size = sz,
          mixture_em = MixtureEmConfig(KeepModel(1.0e-9), em.nuisance, em.stop, 30)))
    rp!(cases, "divcam/golden_contaminated/refused_no_improvement_stop", d, ((rc, v),) -> divfit(rc, v; image_size = sz,
          mixture_em = MixtureEmConfig(em.model_update, em.nuisance, NoImprovementStop(), 30)))
    # units: a physical plane is refused
    src = [SVector(0.1 * i, 0.07 * j) for i in 1:4 for j in 1:4]
    phy = [p .* VgtkCore.m for p in src] .=> [SVector(100.0 + 900 * p[1], 50.0 + 880 * p[2]) for p in src]
    rp!(cases, "divcam/physical_plane/refused", (Dict(1 => HomographyMat(SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1))), Dict(1 => phy)),
          ((rc, v),) -> divfit(rc, v; focal_seed = 500.0))
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# fit_radial_camera
# ---------------------------------------------------------------------------------------------------------------
function _replay_ecf_radial!(cases)
    pts = ecf_first_board_pixels()
    o = SVector(0.0, 0.0)
    rp!(cases, "radialcam/board1/default", pts, p -> radfit(p; center = SVector(0.0, 0.0)))
    rp!(cases, "radialcam/board1/iterations1", pts, p -> radfit(p; center = SVector(0.0, 0.0), iterations = 1))
    rp!(cases, "radialcam/board1_4pts/refused_too_few", pts[1:4], p -> radfit(p; center = SVector(0.0, 0.0)))
    rp!(cases, "radialcam/board1/refused_no_center", pts, p -> radfit(p))
    rp!(cases, "radialcam/board1_5pts/iterations1", pts[[1, 5, 8, 21, 25]], p -> radfit(p; center = SVector(0.0, 0.0), iterations = 1))
    c = SVector(931.0, -472.0)
    rp!(cases, "radialcam/board1_raw/centre", ([X => (x + c) for (X, x) in pts], c), ((p, c),) -> radfit(p; center = c))
    col = [SVector(t, 2t) => pts[1][2] .* (1 + 0.1t) for t in 1.0:5.0]
    rp!(cases, "radialcam/collinear5/iterations1_degenerate", col, p -> radfit(p; center = SVector(0.0, 0.0), iterations = 1))
    col8 = [SVector(Float64(i), 0.0) => SVector(10.0i, 0.0) for i in 1:8]
    rp!(cases, "radialcam/collinear8/degenerate", col8, p -> radfit(p; center = SVector(0.0, 0.0)))
    bad = copy(pts); bad[3] = bad[3][1] => bad[3][2] + SVector(40.0, -25.0)
    w = ones(length(bad)); w[3] = 0.0
    rp!(cases, "radialcam/board1_outlier/zero_weight", Correspondences(first.(bad), last.(bad); weights = w), p -> radfit(p; center = SVector(0.0, 0.0)))
    rp!(cases, "radialcam/board1_outlier/unweighted", bad, p -> radfit(p; center = SVector(0.0, 0.0)))
    rp!(cases, "radialcam/board1/unit_weights", Correspondences(first.(pts), last.(pts); weights = ones(length(pts))),
          p -> radfit(p; center = SVector(0.0, 0.0)))
    src = [SVector(0.1 * i, 0.07 * j) for i in 1:4 for j in 1:4]
    phy = [p .* VgtkCore.m for p in src] .=> [SVector(100.0 + 900 * p[1], 50.0 + 880 * p[2]) for p in src]
    rp!(cases, "radialcam/physical_plane/refused", phy, p -> radfit(p; center = SVector(400.0, 300.0)))
    return cases
end

# ---------------------------------------------------------------------------------------------------------------
# refine_camera_and_poses: the views form, the board form, the rotation-centre form
# ---------------------------------------------------------------------------------------------------------------
function _replay_ecf_bundle!(cases)
    ls = (; optimization_strategy = nothing)
    # --- a pinhole camera and five views from exact pixels -----------------------------------------------------------
    cam = pinhole_camera(); Es = truth_poses(5)
    views = views_of(cam, Es, 0.6)
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.04, 1.01, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 6.0, -5.0))
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    fx = (cam0, poses0, views)
    rp!(cases, "bundle/pinhole5_exact/ls", fx, ((c, p, v),) -> bundle(c, p, v; ls...))
    rp!(cases, "bundle/pinhole5_exact/ls_max_iter1", fx, ((c, p, v),) -> bundle(c, p, v; ls..., max_iter = 1))
    # an exact Division fit from exact correspondences (CaliBlobs' README refine), four starts
    truth = CameraModel(LogicalIntrinsics(CameraCalibrationMat(900.0px, [2120.0px, 1416.0px])), Division(-0.2, -0.08))
    board = [Vec3(x, y, 0.0) for x in -60.0:30.0:60.0 for y in -60.0:30.0:60.0]
    dposes = Dict(t => EuclideanMap(Rotations.RotationVec(0.2sin(t), 0.2cos(t), 0.1t), Vec3(40.0t - 140, 20.0 - 10t, 300.0)) for t in 1:6)
    dviews = Dict(t => [X => project(truth, P(X)) for X in board] for (t, P) in dposes)
    for (n, f0) in ((6, 945.0), (6, 930.0), (4, 945.0), (3, 945.0))
        start = CameraModel(LogicalIntrinsics(CameraCalibrationMat(f0 * px, [2120.0px, 1416.0px])), Division(-0.18, -0.072))
        rp!(cases, "bundle/division_exact_$(n)views/f$(round(Int, f0))_fixed_aspect",
              (start, Dict(t => dposes[t] for t in 1:n), Dict(t => dviews[t] for t in 1:n)),
              ((c, p, v),) -> bundle(c, p, v; fixed = (:aspect,), ls...))
    end
    # a fisheye past 90°
    fcam = fisheye_camera(); fEs = truth_poses(4)
    fviews = views_of(fcam, fEs, deg2rad(100))
    fθ = parameter_vector(fcam)
    fcam0 = with_parameters(fcam, fθ .* SVector(1.03, 1.0, 1.0, 1.0, 1.0, 0.8, 1.2, 0.9, 1.1) .+ SVector(0.0, 0.0, 0.0, -4.0, 3.0, 0.0, 0.0, 0.0, 0.0))
    ff = (fcam0, [perturbed_pose(E, v) for (v, E) in enumerate(fEs)], fviews)
    rp!(cases, "bundle/fisheye4_exact/ls", ff, ((c, p, v),) -> bundle(c, p, v; ls...))
    rp!(cases, "bundle/fisheye4_exact/ls_fixed_k_skew", ff, ((c, p, v),) -> bundle(c, p, v; fixed = (:k, :skew), ls...))
    # a held camera; the forms; refusals
    Es3 = truth_poses(3); views3 = views_of(cam, Es3, 0.5)
    p3 = [perturbed_pose(E, v) for (v, E) in enumerate(Es3)]
    rp!(cases, "bundle/pinhole3_exact/ls_camera_held", (cam, p3, views3), ((c, p, v),) -> bundle(c, p, v; fixed = parameter_names(c), ls...))
    rp!(cases, "bundle/pinhole3_exact/ls_fixed_aspect", (cam, p3, views3), ((c, p, v),) -> bundle(c, p, v; fixed = (:aspect,), ls...))
    rp!(cases, "bundle/pinhole3_exact/ls_fixed_empty", (cam, p3, views3), ((c, p, v),) -> bundle(c, p, v; fixed = (), ls...))
    for kw in ECF_RETIRED
        rp!(cases, "bundle/pinhole3_exact/refused_retired_$kw", (cam, p3, views3), ((c, p, v),) -> bundle(c, p, v; (; kw => ())...))
    end
    c3 = with_parameters(cam, parameter_vector(cam) .* SVector(1.02, 1.0, 1.0, 1.0, 1.0))
    f3 = (c3, p3, views3)
    rp!(cases, "bundle/pinhole3_exact/ls_vector_form", f3, ((c, p, v),) -> bundle(c, p, v; ls...))
    rp!(cases, "bundle/pinhole3_exact/ls_dict_form",
          (c3, Dict(:c => p3[3], :a => p3[1], :b => p3[2]), Dict(:a => views3[1], :b => views3[2], :c => Correspondences(first.(views3[3]), last.(views3[3])))),
          ((c, p, v),) -> bundle(c, p, v; ls...))
    rp!(cases, "bundle/pinhole3_exact/ls_metres", (c3, [with_unit(P, VgtkCore.m) for P in p3], [[X .* VgtkCore.m => u for (X, u) in cs] for cs in views3]),
          ((c, p, v),) -> bundle(c, p, v; ls...))
    rp!(cases, "bundle/pinhole3_exact/refused_key_mismatch", (c3, Dict(:a => p3[1]), Dict(:b => views3[1])), ((c, p, v),) -> bundle(c, p, v; ls...))
    rp!(cases, "bundle/pinhole3_exact/refused_length_mismatch", (c3, p3[1:2], views3), ((c, p, v),) -> bundle(c, p, v; ls...))
    rp!(cases, "bundle/pinhole3_exact/refused_unknown_fixed", f3, ((c, p, v),) -> bundle(c, p, v; fixed = (:k9,), ls...))
    rp!(cases, "bundle/pinhole3_exact/refused_behind_camera", (c3, [EuclideanMap(Es3[1].R, SVector(0.0, 0.0, -50.0)); p3[2:3]], views3),
          ((c, p, v),) -> bundle(c, p, v; ls...))
    rp!(cases, "bundle/pinhole3_exact/refused_underdetermined", (c3, p3[1:1], [views3[1][1:2]]), ((c, p, v),) -> bundle(c, p, v; ls...))
    rp!(cases, "bundle/pinhole3_exact/refused_box_nothing_free", f3,
          ((c, p, v),) -> bundle(c, p, v; fixed = parameter_names(c), box = (; f = (1.0, 2.0)), ls...))
    rp!(cases, "bundle/pinhole3_exact/ls_max_time_inf", f3, ((c, p, v),) -> bundle(c, p, v; max_time = Inf, ls...))

    # --- robust strategies on contaminated detections (seeds 1–3) ---------------------------------------------------
    cam1 = with_parameters(cam, θ .* SVector(1.005, 1.0, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 1.0, -1.0))
    poses1 = [ecf_near_pose(E, v) for (v, E) in enumerate(Es)]
    H, W = Triggs{HuberShape}(efficiency_constant = 1.345), Triggs{WelschShape}(efficiency_constant = 2.98)
    for seed in SEEDS
        noisy = ecf_contaminate(StableRNG(seed), views)
        far, near = (cam0, poses0, noisy), (cam1, poses1, noisy)
        rp!(cases, "bundle/pinhole5_contaminated_far/ls/seed$seed", far, ((c, p, v),) -> bundle(c, p, v; ls...))
        rp!(cases, "bundle/pinhole5_contaminated_far/default_scale0.2/seed$seed", far, ((c, p, v),) -> bundle(c, p, v; scale = 0.2))
        rp!(cases, "bundle/pinhole5_contaminated_far/triggs_huber1.345_scale0.2/seed$seed", far,
              ((c, p, v),) -> bundle(c, p, v; optimization_strategy = H, scale = 0.2))
        rp!(cases, "bundle/pinhole5_contaminated_far/triggs_welsch2.98_scale0.2/seed$seed", far,
              ((c, p, v),) -> bundle(c, p, v; optimization_strategy = W, scale = 0.2))
        rp!(cases, "bundle/pinhole5_contaminated_near/triggs_huber1.345_scale0.2/seed$seed", near,
              ((c, p, v),) -> bundle(c, p, v; optimization_strategy = H, scale = 0.2))
        rp!(cases, "bundle/pinhole5_contaminated_near/triggs_welsch2.98_scale0.2/seed$seed", near,
              ((c, p, v),) -> bundle(c, p, v; optimization_strategy = W, scale = 0.2))
        rp!(cases, "bundle/pinhole5_contaminated_near/triggs_huber_max_iter1/seed$seed", near,
              ((c, p, v),) -> bundle(c, p, v; optimization_strategy = Triggs{HuberShape}(), scale = 0.2, max_iter = 1))
        rp!(cases, "bundle/pinhole5_contaminated_near/triggs_huber_width/seed$seed", near,
              ((c, p, v),) -> bundle(c, p, v; optimization_strategy = Triggs{HuberShape}(), width = 1.345 * 0.2))
        rd = bundle(cam0, poses0, noisy; scale = 0.2)
        rp!(cases, "bundle/pinhole5_contaminated_near/default_upstream_scale/seed$seed", (near..., rd),
              ((c, p, v, rd),) -> bundle(c, p, v; scale = rd))
    end
    noisy = ecf_contaminate(StableRNG(1), views)
    far, near = (cam0, poses0, noisy), (cam1, poses1, noisy)
    rp!(cases, "bundle/pinhole5_contaminated_far/refused_no_scale", far, ((c, p, v),) -> bundle(c, p, v))
    rp!(cases, "bundle/pinhole5_contaminated_near/refused_irls", near,
          ((c, p, v),) -> bundle(c, p, v; optimization_strategy = Irls{HuberShape}(), scale = 0.2))
    rp!(cases, "bundle/pinhole5_contaminated_far/refused_terminal_lo", far,
          ((c, p, v),) -> bundle(c, p, v; optimization_strategy = TerminalLO(Irls{HuberShape}())))
    rp!(cases, "bundle/pinhole5_contaminated_far/refused_scale_and_width", far, ((c, p, v),) -> bundle(c, p, v; scale = 0.2, width = 0.5))
    rp!(cases, "bundle/pinhole5_contaminated_far/refused_ls_scale", far, ((c, p, v),) -> bundle(c, p, v; ls..., scale = 0.2))
    rp!(cases, "bundle/pinhole5_contaminated_near/refused_scoring", near,
          ((c, p, v),) -> bundle(c, p, v; optimization_strategy = W, scale = 0.2, scoring = nothing))
    rp!(cases, "bundle/pinhole5_contaminated_far/refused_negative_scale", far, ((c, p, v),) -> bundle(c, p, v; optimization_strategy = W, scale = -1.0))
    rp!(cases, "bundle/pinhole5_contaminated_near/refused_robust", near, ((c, p, v),) -> bundle(c, p, v; robust = nothing))
    rp!(cases, "bundle/pinhole5_contaminated_near/refused_local_optimization", near,
          ((c, p, v),) -> bundle(c, p, v; local_optimization = W, scale = 0.2))

    # --- the board form -----------------------------------------------------------------------------------------------
    boards = truth_boards(); vposes = view_poses()
    bcam0 = with_parameters(cam, θ .* SVector(1.03, 1.0, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 5.0, -4.0))
    views0 = Dict(v => nudged(E, v) for (v, E) in vposes)
    keyn = Dict(:A => 1, :B => 2, :C => 3)
    boards0 = Dict(b => b === :A ? E : nudged(E, 7 + keyn[b]) for (b, E) in boards)
    exact = ecf_board_obs(cam, vposes, boards)
    rp!(cases, "boardbundle/three_boards_exact/ls", (bcam0, views0, boards0, exact), ((c, v, b, o),) -> bundle(c, v, b, o; ls...))
    for seed in SEEDS
        nb = ecf_board_obs(cam, vposes, boards; σ = 0.2, rng = StableRNG(seed))
        rp!(cases, "boardbundle/three_boards_noisy/ls/seed$seed", (bcam0, views0, boards0, nb), ((c, v, b, o),) -> bundle(c, v, b, o; ls...))
        rp!(cases, "boardbundle/three_boards_noisy/ls_truth_start/seed$seed", (cam, vposes, boards, nb), ((c, v, b, o),) -> bundle(c, v, b, o; ls...))
        rp!(cases, "boardbundle/three_boards_noisy/ls_view1_held/seed$seed", (bcam0, views0, boards0, nb),
              ((c, v, b, o),) -> bundle(c, v, b, o; held_boards = (), held_views = (1,), ls...))
    end
    vd = merge(vposes, Dict(7 => vposes[2])); bd = merge(boards, Dict(:D => nudged(boards[:B], 1)))
    obs = merge(exact, Dict((7, :D) => [X => SVector{2, Float64}(project(cam, vd[7](bd[:D](X)))) for X in board_points()]))
    rp!(cases, "boardbundle/two_components_exact/ls", (bcam0, merge(views0, Dict(7 => nudged(vposes[2], 3))), merge(boards0, Dict(:D => bd[:D])), obs),
          ((c, v, b, o),) -> bundle(c, v, b, o; ls...))
    nb = ecf_board_obs(cam, vposes, boards; σ = 0.2, rng = StableRNG(1))
    rp!(cases, "boardbundle/three_boards_noisy/refused_no_held_pose", (bcam0, views0, boards0, nb),
          ((c, v, b, o),) -> bundle(c, v, b, o; held_boards = (), ls...))
    rp!(cases, "boardbundle/three_boards_noisy/refused_board_without_pose", (bcam0, views0, Dict(:A => boards[:A], :B => boards[:B]), nb),
          ((c, v, b, o),) -> bundle(c, v, b, o; ls...))
    rp!(cases, "boardbundle/three_boards_noisy/refused_view_without_data", (bcam0, merge(views0, Dict(9 => vposes[1])), boards0, nb),
          ((c, v, b, o),) -> bundle(c, v, b, o; ls...))

    # --- a SplineLens camera: refused free; held -------------------------------------------------------------------
    struth = spline_truth(); sEs = truth_poses(3)
    sviews = spline_views(struth, sEs, 1.1)
    sp0 = [perturbed_pose(E, v) for (v, E) in enumerate(sEs)]
    pc = perturbed_controls(struth, 0.01)
    sfx = (ecf_bounded(pc), sp0, sviews)
    rp!(cases, "bundle/spline3_exact/refused_free_controls", sfx, ((c, p, v),) -> bundle(c, p, v))
    rp!(cases, "bundle/spline3_exact/refused_free_controls_fixed_empty", sfx, ((c, p, v),) -> bundle(c, p, v; fixed = ()))
    for (lbl, penalty) in (("gcv", :gcv), ("1e-8", 1.0e-8), ("0", 0))
        rp!(cases, "bundle/spline3_exact/refused_penalty_$(lbl)_held", sfx, ((c, p, v),) -> bundle(c, p, v; fixed = (:controls,), penalty))
        rp!(cases, "bundle/spline3_exact/refused_penalty_$lbl", sfx, ((c, p, v),) -> bundle(c, p, v; penalty))
    end
    rp!(cases, "bundle/spline3_exact_bare/refused_free_controls", (pc, sp0, sviews), ((c, p, v),) -> bundle(c, p, v))
    rp!(cases, "bundle/spline3_exact_bare_keyed/refused_free_controls", (pc, Dict(v => sp0[v] for v in 1:3), Dict(v => sviews[v] for v in 1:3)),
          ((c, p, v),) -> bundle(c, p, v; fixed = ()))
    held = (; fixed = (:controls,), optimization_strategy = nothing)
    kb = fisheye_camera(); bk = BoundedCamera(kb, Size2(1280, 960))
    fitted = u_of(fitcam(SplineLens, bk, imaged_region(bk); tolerance = 1.0e-2, forward = false))
    for (lbl, tc) in (("fitted", fitted), ("quasi", ecf_kb_spline_quasi()), ("scaled", ecf_scaled_spline()), ("truth", struth))
        rp!(cases, "bundle/spline3_$(lbl)_exact/held_ls", (ecf_bounded(tc), sp0, spline_views(tc, sEs, 1.1)), ((c, p, v),) -> bundle(c, p, v; held...))
    end
    rp!(cases, "bundle/spline3_truth_bare_exact/held_ls", (struth, sp0, sviews), ((c, p, v),) -> bundle(c, p, v; held...))
    fwd = BoundedCamera(u_of(fitcam(SplineLens, BoundedCamera(pc, Size2(1280, 960)), imaged_region(BoundedCamera(pc, Size2(1280, 960))); tolerance = 1.0e-3)),
                        Size2(1280, 960))
    rp!(cases, "bundle/spline3_with_forward_exact/held_ls", (fwd, sp0, sviews), ((c, p, v),) -> bundle(c, p, v; held...))
    rp!(cases, "bundle/spline3_with_forward_exact/refused_refit_forward", (fwd, sp0, sviews),
          ((c, p, v),) -> bundle(c, p, v; fixed = (:controls,), refit_forward = false))
    rp!(cases, "bundle/spline3_with_forward_exact/refused_forward_tolerance", (fwd, sp0, sviews),
          ((c, p, v),) -> bundle(c, p, v; fixed = (:controls,), forward_tolerance = 1.0e-3))
    bt = (ecf_bounded(struth), sp0, sviews)
    rp!(cases, "bundle/spline3_truth_exact/refused_held_no_scale", bt, ((c, p, v),) -> bundle(c, p, v; fixed = (:controls,)))
    rp!(cases, "bundle/spline3_masked_exact/held_ls", (ecf_bounded(ecf_masked_spline()), sp0, sviews), ((c, p, v),) -> bundle(c, p, v; held...))
    rp!(cases, "bundle/spline3_truth_exact/refused_box", bt, ((c, p, v),) -> bundle(c, p, v; fixed = (:controls,), box = (; c_1_1_x = (0.0, 1.0))))
    rp!(cases, "bundle/spline3_truth_exact/refused_fixed_one_control", bt, ((c, p, v),) -> bundle(c, p, v; fixed = (:c_1_1_x,)))
    rp!(cases, "bundle/spline3_truth_exact/refused_retired_free", bt, ((c, p, v),) -> bundle(c, p, v; free = ()))
    P = sp0[1]
    rp!(cases, "bundle/spline_flat_one_point/refused_fold", (ecf_bounded(ecf_flat_spline()), [P], [[P.R' * (SVector(0.0, 0.0, 2.0) - P.t) => SVector(640.0, 480.0)]]),
          ((c, p, v),) -> bundle(c, p, v; held...))
    for seed in SEEDS
        sn = ecf_contaminate_inner(StableRNG(seed), sviews)
        f = (ecf_bounded(struth), sp0, sn)
        rp!(cases, "bundle/spline3_truth_contaminated/held_ls/seed$seed", f, ((c, p, v),) -> bundle(c, p, v; held...))
        rp!(cases, "bundle/spline3_truth_contaminated/held_triggs_huber1.345_scale0.2/seed$seed", f,
              ((c, p, v),) -> bundle(c, p, v; fixed = (:controls,), optimization_strategy = H, scale = 0.2))
    end

    # --- rotation-centre views of a held pinhole camera ------------------------------------------------------------
    t = center_truth(ECF_CENTER_AXES)
    start = ecf_center_start(t)
    hp = held_parameters(cam)
    rp!(cases, "centerbundle/pinhole6_exact/ls", (cam, start, center_views(cam, t)), ((c, s, v),) -> bundle(c, s, v; fixed = held_parameters(c), ls...))
    upstream = FitResult(0.3, ReturnCode.Success, ClosedForm, SolveStats(), NaN, 6, (; scale = 0.3, scale_source = ScaleEstimated))
    for seed in SEEDS   # StableRNG(2) is the suite's own draw
        vs = center_views(cam, t; σ = 0.3, outliers = 0.05, rng = StableRNG(seed))
        id = "/seed$seed"
        f = (cam, start, vs)
        rp!(cases, "centerbundle/pinhole6_contaminated/triggs_huber3_upstream$id", (f..., upstream),
              ((c, s, v, u),) -> bundle(c, s, v; fixed = held_parameters(c), optimization_strategy = Triggs{HuberShape}(efficiency_constant = 3.0), scale = u))
        rp!(cases, "centerbundle/pinhole6_contaminated/ls$id", f, ((c, s, v),) -> bundle(c, s, v; fixed = held_parameters(c), ls...))
        rp!(cases, "centerbundle/pinhole6_contaminated/triggs_huber1.345_scale0.3$id", f,
              ((c, s, v),) -> bundle(c, s, v; fixed = held_parameters(c), optimization_strategy = H, scale = 0.3))
        rp!(cases, "centerbundle/pinhole6_contaminated/triggs_huber_width0.9$id", f,
              ((c, s, v),) -> bundle(c, s, v; fixed = held_parameters(c), optimization_strategy = Triggs{HuberShape}(), width = 0.9))
    end
    rp!(cases, "centerbundle/pinhole6_clean_sigma0.1/ls", (cam, start, center_views(cam, t; σ = 0.1, rng = StableRNG(7))),
          ((c, s, v),) -> bundle(c, s, v; fixed = held_parameters(c), ls...))
    one_axis = center_truth([SVector(0.0, 0.0, a) for a in -0.1:0.05:0.1])
    rp!(cases, "centerbundle/pinhole_one_axis/refused", (cam, RotationCenterViews(one_axis.R, one_axis.c, one_axis.d), center_views(cam, one_axis)),
          ((c, s, v),) -> bundle(c, s, v; fixed = held_parameters(c), ls...))
    two = center_truth(ECF_CENTER_AXES[1:2])
    rp!(cases, "centerbundle/pinhole_two_views/refused", (cam, RotationCenterViews(two.R, two.c, two.d), center_views(cam, two)),
          ((c, s, v),) -> bundle(c, s, v; fixed = held_parameters(c), ls...))
    rp!(cases, "centerbundle/pinhole6_exact/refused_free_camera", (cam, start, center_views(cam, t)), ((c, s, v),) -> bundle(c, s, v; ls...))
    rp!(cases, "centerbundle/pinhole6_exact/refused_missing_view", (cam, start, Dict(k => v for (k, v) in center_views(cam, t) if k != 3)),
          ((c, s, v),) -> bundle(c, s, v; fixed = held_parameters(c), ls...))

    # --- rotation-centre views of a held SplineLens camera ----------------------------------------------------------
    st = center_truth(4 .* ECF_CENTER_AXES[1:5])
    sstart = ecf_center_start(st)
    sexact = center_views(struth, st)
    hc = (; fixed = (:controls,))
    rp!(cases, "centerbundle/spline5_exact/ls", (struth, sstart, sexact), ((c, s, v),) -> bundle(c, s, v; hc..., ls...))
    rp!(cases, "centerbundle/spline5_exact/ls_held_parameters", (struth, sstart, sexact),
          ((c, s, v),) -> bundle(c, s, v; fixed = held_parameters(c), ls...))
    Q = SMatrix{3, 3, Float64, 9}(exp_so3(SVector(0.03, -0.035, 0.015)))
    pxs = [SVector(x, y) for x in range(8.0, 1272.0; length = 12) for y in range(8.0, 952.0; length = 9)]
    rays = [normalize(backproject(struth, u)) for u in pxs]
    rim = Dict(k => [(R' * (2.5 * f - (st.d - R * st.c))) => u for (f, u) in zip(rays, pxs)] for (k, R) in st.R)
    turned = RotationCenterViews(Dict(k => Q * R for (k, R) in st.R), st.c, Q * st.d)
    rp!(cases, "centerbundle/spline5_rim_turned_start/ls", (struth, turned, rim), ((c, s, v),) -> bundle(c, s, v; hc..., ls...))
    snoisy = center_views(struth, st; σ = 0.3, rng = StableRNG(5))
    for (lbl, strategy, input) in (("triggs_huber3_scale0.3", Triggs{HuberShape}(efficiency_constant = 3.0), (; scale = 0.3)),
                                   ("triggs_huber1.345_scale0.3", H, (; scale = 0.3)),
                                   ("triggs_huber_width0.9", Triggs{HuberShape}(), (; width = 0.9)))
        rp!(cases, "centerbundle/spline5_noisy/$lbl", (struth, sstart, snoisy),
              ((c, s, v),) -> bundle(c, s, v; hc..., optimization_strategy = strategy, input...))
    end
    sbad = center_views(struth, st; σ = 0.3, outliers = 0.05, rng = StableRNG(3))
    rp!(cases, "centerbundle/spline5_contaminated/triggs_huber3_scale0.3", (struth, sstart, sbad),
          ((c, s, v),) -> bundle(c, s, v; hc..., optimization_strategy = Triggs{HuberShape}(efficiency_constant = 3.0), scale = 0.3))
    rp!(cases, "centerbundle/spline5_contaminated/ls", (struth, sstart, sbad), ((c, s, v),) -> bundle(c, s, v; hc..., ls...))
    present = trues(10, 12); present[8, 10] = false
    rp!(cases, "centerbundle/spline5_masked_exact/ls", (with_present(struth, present), sstart, sexact), ((c, s, v),) -> bundle(c, s, v; hc..., ls...))
    rp!(cases, "centerbundle/spline5_exact/refused_free_camera", (struth, sstart, sexact), ((c, s, v),) -> bundle(c, s, v; fixed = (), ls...))
    rp!(cases, "centerbundle/spline5_exact/refused_box", (struth, sstart, sexact),
          ((c, s, v),) -> bundle(c, s, v; hc..., box = (; f = (100.0, 900.0)), ls...))
    rp!(cases, "centerbundle/spline5_exact/refused_fixed_controls_f", (struth, sstart, sexact),
          ((c, s, v),) -> bundle(c, s, v; fixed = (:controls, :f), ls...))
    sv5 = center_views(struth, st; σ = 0.5)
    rp!(cases, "centerbundle/spline5_noisy0.5/ls_bearing_residual", (struth, sstart, sv5),
          ((c, s, v),) -> bundle(c, s, v, BearingResidual(); hc..., ls...))
    rp!(cases, "centerbundle/spline5_noisy0.5/ls_reprojection_residual", (struth, sstart, sv5),
          ((c, s, v),) -> bundle(c, s, v, ReprojectionResidual(); hc..., ls...))
    rp!(cases, "centerbundle/spline5_noisy0.5/ls", (struth, sstart, sv5), ((c, s, v),) -> bundle(c, s, v; hc..., ls...))
    return cases
end
