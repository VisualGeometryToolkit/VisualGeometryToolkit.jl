# The old image-circle fit at 3d8a06da: VisualGeometryToolkitFeatures' `detect_image_circle` (the task verb stays; its
# fit, VisualGeometryToolkitFeaturesEstimationExt over Estimation's find_circle / find_ellipse, is rewritten). The fixtures
# and keyword variants of VisualGeometryToolkitFeatures/test/image_circle_tests.jl, each noise draw and each call's RNG
# from seeds 1–3 (the fixture's seed and the call's paired), and its refusals. Every call runs on the CPU backend
# (`device = CPU()`, explicitly: the default would follow whatever GPU package a session has loaded).
using VisualGeometryToolkitFeatures: detect_image_circle
using VisualGeometryToolkitCore: Circle, Ellipse, LinearRaw
using KernelAbstractions: CPU

const FIC_DEV = CPU()

function record_features_image_circle(cases)
    a = FIC_A
    for seed in SEEDS
        s = "seed$seed"
        rng() = StableRNG(seed)
        # --- partial rim: six frames, each dark over one part of the rim -----------------------------------------------
        pr = fic_partial_rim(seed)
        fr = pr.frames
        case!(cases, "partial_rim/r100/ellipse/$s", fr, f -> detect_image_circle(rng(), f; outlier_halfwidth = a, device = FIC_DEV))
        case!(cases, "partial_rim/r100/circle/$s", fr, f -> detect_image_circle(rng(), Circle, f; outlier_halfwidth = a, device = FIC_DEV))
        case!(cases, "partial_rim/r100/ellipse_type/$s", fr, f -> detect_image_circle(rng(), Ellipse, f; outlier_halfwidth = a, device = FIC_DEV))
        case!(cases, "partial_rim/r100/circle_default_rng/$s", fr,
            f -> (Random.seed!(seed); detect_image_circle(Circle, f; outlier_halfwidth = a, device = FIC_DEV)))
        case!(cases, "partial_rim/r100/ellipse_default_rng/$s", fr,
            f -> (Random.seed!(seed); detect_image_circle(f; outlier_halfwidth = a, device = FIC_DEV)))
        case!(cases, "partial_rim/r100/generator/$s", fr, f -> detect_image_circle(rng(), (x for x in f); outlier_halfwidth = a, device = FIC_DEV))
        # RAW: the green cells' planes by hand, the cells, the mosaics streamed, a circle on the cells
        ms = pr.mosaics
        case!(cases, "partial_rim/r100/green_planes_half/$s", ms,
            m -> detect_image_circle(rng(), [LinearRaw(x; mode = :green_cells).image[:, :, 1] for x in m]; outlier_halfwidth = a / 2, device = FIC_DEV))
        case!(cases, "partial_rim/r100/raw_cells_half/$s", ms,
            m -> detect_image_circle(rng(), [LinearRaw(x; mode = :green_cells) for x in m]; outlier_halfwidth = a / 2, device = FIC_DEV))
        case!(cases, "partial_rim/r100/raw_mosaics_generator_half/$s", ms,
            m -> detect_image_circle(rng(), (x for x in m); outlier_halfwidth = a / 2, device = FIC_DEV))
        case!(cases, "partial_rim/r100/raw_cells_circle/$s", ms,
            m -> detect_image_circle(rng(), Circle, [LinearRaw(x; mode = :green_cells) for x in m]; outlier_halfwidth = a, device = FIC_DEV))
        # --- a disc cut by the frame (two arcs) --------------------------------------------------------------------
        cut = fic_disc_frames(FIC_W, FIC_H, FIC_C, 140.0, 3, seed)
        case!(cases, "cut_disc/r140/ellipse/$s", cut, f -> detect_image_circle(rng(), f; outlier_halfwidth = a, device = FIC_DEV))
        case!(cases, "cut_disc/r140/circle/$s", cut, f -> detect_image_circle(rng(), Circle, f; outlier_halfwidth = a, device = FIC_DEV))
        # --- one arc of rim (a single dark corner) -----------------------------------------------------------------
        one = [fic_one_arc_frame(FIC_W, FIC_H, (120.0, 90.0), 260.0, seed)]
        case!(cases, "one_arc/r260/circle_band0.8_0.25/$s", one,
            f -> detect_image_circle(rng(), Circle, f; outlier_halfwidth = a, rim_band = (0.8, 0.25), device = FIC_DEV))
        case!(cases, "one_arc/r260/circle_default_band/$s", one, f -> detect_image_circle(rng(), Circle, f; outlier_halfwidth = a, device = FIC_DEV))
        # --- a hot pixel in the dark surround ----------------------------------------------------------------------
        clean = fic_disc_frames(FIC_W, FIC_H, FIC_C, 80.0, 3, seed)
        hot = deepcopy(clean); hot[2][30, 40] = 0.9f0
        case!(cases, "disc/r80x3/ellipse/$s", clean, f -> detect_image_circle(rng(), f; outlier_halfwidth = a, device = FIC_DEV))
        case!(cases, "disc/r80x3_hot_pixel/ellipse/$s", hot, f -> detect_image_circle(rng(), f; outlier_halfwidth = a, device = FIC_DEV))
        # --- the deprecated hull keywords (accepted and ignored; a depwarn only under --depwarn=yes/error) ---------
        two = fic_disc_frames(FIC_W, FIC_H, FIC_C, 80.0, 2, seed)
        case!(cases, "disc/r80x2/ellipse/$s", two, f -> detect_image_circle(rng(), f; outlier_halfwidth = a, device = FIC_DEV))
        for (lbl, kw) in (("min_hole_area8", (; min_hole_area = 8)), ("smoothing_weight5e-4", (; smoothing_weight = 5e-4)),
                          ("border_margin5", (; border_margin = 5)))
            case!(cases, "disc/r80x2/deprecated_$lbl/$s", two, f -> detect_image_circle(rng(), f; outlier_halfwidth = a, device = FIC_DEV, kw...))
        end
        # --- a disc centred on the image centre (the log-polar origin) --------------------------------------------
        cen = fic_disc_frames(FIC_W, FIC_H, (FIC_W / 2, FIC_H / 2), 100.0, 3, seed)
        case!(cases, "disc/r100x3_centred/ellipse/$s", cen,
            f -> detect_image_circle(rng(), f; outlier_halfwidth = 0.1 * hypot(FIC_W, FIC_H), device = FIC_DEV))
        # --- rims of known edge and knee, 640 × 480 ----------------------------------------------------------------
        W, H, c, re, rk = 640, 480, (321.3, 240.7), 150.0, 120.0
        ramp = rim_synth(StableRNG(seed), W, H, c, re, rk; sat_at = 95.0, sat_radius = 20)
        case!(cases, "rim/ramp150_120_sat/ellipse/$s", ramp, f -> detect_image_circle(rng(), f; outlier_halfwidth = 50, device = FIC_DEV))
        sharp = rim_synth(StableRNG(seed), W, H, c, re, re; sat_at = 95.0, sat_radius = 20)
        case!(cases, "rim/sharp150_sat/ellipse/$s", sharp, f -> detect_image_circle(rng(), f; outlier_halfwidth = 50, device = FIC_DEV))
        plain = rim_synth(StableRNG(seed), W, H, c, re, re)
        case!(cases, "rim/sharp150/rim_angles72/$s", plain,
            f -> detect_image_circle(rng(), f; outlier_halfwidth = 50, rim_angles = 72, device = FIC_DEV))
        init = VgtkCore.Ellipse(VgtkCore.Point2(318.0, 243.0), 200.0, 200.0, 0.0)
        case!(cases, "rim/sharp150/initial_band0.5_0.2/$s", (plain, init),
            ((f, init),) -> detect_image_circle(rng(), f; outlier_halfwidth = 50, initial = init, rim_band = (0.5, 0.2), device = FIC_DEV))
        # --- the rim under a sensor model (Uncertain cells) and without one ---------------------------------------
        sm = fic_sensor_scene(seed)
        case!(cases, "rim/sensor_ramp150_120/model/$s", sm,
            x -> detect_image_circle(rng(), [LinearRaw(m; mode = :green_cells, sensor_model = x.model) for m in x.mosaics];
                                     outlier_halfwidth = 25, device = FIC_DEV))
        case!(cases, "rim/sensor_ramp150_120/empirical/$s", sm,
            x -> detect_image_circle(rng(), [LinearRaw(m; mode = :green_cells) for m in x.mosaics]; outlier_halfwidth = 25, device = FIC_DEV))
        # --- ramps centred exactly on the image centre, 800 × 600 --------------------------------------------------
        for d in (0.0, 4.0, 8.0)
            fr = fic_soft_ramp(d, seed)
            case!(cases, "centred_rim/soft_ramp_d$(Int(d))/default/$s", fr, f -> detect_image_circle(rng(), f; outlier_halfwidth = 50, device = FIC_DEV))
            for off in ((0.3, -0.4), (25.0, -15.0), (-25.0, 15.0))
                case!(cases, "centred_rim/soft_ramp_d$(Int(d))/initial_$(off[1])_$(off[2])/$s", (fr, fic_centred_initial(off)),
                    ((f, init),) -> detect_image_circle(rng(), f; outlier_halfwidth = 50, initial = init, device = FIC_DEV))
            end
        end
        for (lbl, re) in (("2_3", 2FIC_RMAX / 3), ("1_2", FIC_RMAX / 2))
            case!(cases, "centred_rim/linear_ramp_$lbl/rim_angles180/$s", fic_linear_ramp(re, seed),
                f -> detect_image_circle(rng(), f; outlier_halfwidth = 50, rim_angles = 180, device = FIC_DEV))
        end
    end
    # --- no rim, and the refusals (no noise in the outcome: seed 1's frames) ---------------------------------------
    rng1() = StableRNG(1)
    case!(cases, "refused/lit_frame/nothing", [fill(0.8f0, FIC_H, FIC_W)],
        f -> detect_image_circle(rng1(), f; outlier_halfwidth = a, device = FIC_DEV))
    pr = fic_partial_rim(1)
    fr = pr.frames
    case!(cases, "refused/partial_rim/size_mismatch", fr,
        f -> detect_image_circle(rng1(), [f[1], f[1][1:end-1, :]]; outlier_halfwidth = a, device = FIC_DEV))
    case!(cases, "refused/no_frames", Matrix{Float32}[], f -> detect_image_circle(rng1(), f; outlier_halfwidth = a, device = FIC_DEV))
    two = fic_disc_frames(FIC_W, FIC_H, FIC_C, 80.0, 2, 1)
    case!(cases, "refused/disc/no_outlier_halfwidth", two, f -> detect_image_circle(rng1(), f; device = FIC_DEV))
    case!(cases, "refused/disc/outlier_halfwidth0", two, f -> detect_image_circle(rng1(), f; outlier_halfwidth = 0, device = FIC_DEV))
    case!(cases, "refused/disc/outlier_halfwidth_inf", two, f -> detect_image_circle(rng1(), f; outlier_halfwidth = Inf, device = FIC_DEV))
    case!(cases, "refused/disc/rim_angles4", two, f -> detect_image_circle(rng1(), f; outlier_halfwidth = a, rim_angles = 4, device = FIC_DEV))
    case!(cases, "refused/disc/rim_angles_symbol", two, f -> detect_image_circle(rng1(), f; outlier_halfwidth = a, rim_angles = :fine, device = FIC_DEV))
    case!(cases, "refused/disc/rim_band1_0", two, f -> detect_image_circle(rng1(), f; outlier_halfwidth = a, rim_band = (1.0, 0.0), device = FIC_DEV))
    case!(cases, "refused/disc/rim_band_negative_outer", two,
        f -> detect_image_circle(rng1(), f; outlier_halfwidth = a, rim_band = (0.8, -0.1), device = FIC_DEV))
    case!(cases, "refused/disc/significance1", two, f -> detect_image_circle(rng1(), f; outlier_halfwidth = a, significance = 1, device = FIC_DEV))
    case!(cases, "refused/unread_generator/outlier_halfwidth_negative", nothing,
        _ -> detect_image_circle((error("a frame was read") for _ in 1:1); outlier_halfwidth = -1, device = FIC_DEV))
    ms = pr.mosaics
    case!(cases, "refused/partial_rim/raw_bin2", ms,
        m -> detect_image_circle(rng1(), [LinearRaw(x; mode = :bin2) for x in m]; outlier_halfwidth = a / 2, device = FIC_DEV))
    case!(cases, "refused/partial_rim/cells_then_plane", ms,
        m -> detect_image_circle(rng1(), [LinearRaw(m[1]; mode = :green_cells), LinearRaw(m[2]; mode = :green_cells).image[:, :, 1]];
                                 outlier_halfwidth = a, device = FIC_DEV))
    case!(cases, "refused/partial_rim/plane_then_cells", ms,
        m -> detect_image_circle(rng1(), [LinearRaw(m[1]; mode = :green_cells).image[:, :, 1], LinearRaw(m[2]; mode = :green_cells)];
                                 outlier_halfwidth = a, device = FIC_DEV))
    case!(cases, "refused/partial_rim/shifted_window", (ms[1], pr.shifted),
        ((m1, sh),) -> detect_image_circle(rng1(), [LinearRaw(m1; mode = :green_cells), LinearRaw(sh; mode = :green_cells)];
                                           outlier_halfwidth = a, device = FIC_DEV))
    case!(cases, "refused/not_an_image", [:not_an_image], f -> detect_image_circle(rng1(), f; outlier_halfwidth = a, device = FIC_DEV))
    case!(cases, "refused/partial_rim/model_type_int", fr, f -> detect_image_circle(rng1(), Int, f; outlier_halfwidth = a, device = FIC_DEV))
    return cases
end
