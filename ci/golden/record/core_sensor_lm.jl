# The old noise-model estimators and device LM of VisualGeometryToolkitCore at 3d8a06da: photon_transfer (one frame, the
# series against a ReadNoise Dict, the series against a Darks), master_dark, fit_scale (its one method) and
# static_lm, called on the CPU directly on its tests' problems and on the problems of its two Core callers (the
# parametric lens inverse and the SplineLens projection), their closures rebuilt here from their source.
using VisualGeometryToolkitCore: photon_transfer, master_dark, fit_scale, static_lm, StaticLmStops

# The lens inverse's call (lens_rays.jl, `_inverse_lm`): the distortion's residual and analytic Jacobian, unbounded,
# the step stop at 1e-10 in ulps of S read at the core tolerance, the residual bound, COLMAP's 0.1 step radius.
_core_tol(::Type{Float64}) = 1.0e-10
_core_tol(::Type{Float32}) = 1.0f-5
function inverse_lm_call(d, xd::SVector{2, S}) where {S}
    fj(z) = (yJ = VgtkCore.Cameras._distort_and_jacobian(d, z); y = yJ[1]; (SVector{2, S}(y[1] - xd[1], y[2] - xd[2]), SMatrix{2, 2, S}(yJ[2])))
    inf = S(Inf)
    s = S(1.0e-10 * (eps(S) / eps(Float64)) / _core_tol(S))
    stops = StaticLmStops(SVector(s, s); acceptance = _core_tol(S)^2 + (8 * eps(S))^2 * (xd[1]^2 + xd[2]^2),
        residual_scale = max(one(S), sqrt(xd[1]^2 + xd[2]^2)), max_iter = S === Float64 ? 20 : 12)
    return static_lm(fj, xd, SVector(-inf, -inf), SVector(inf, inf), stops, S(1) / 10)
end

# The SplineLens projection's call (spline_lens.jl, `_spline_project` from the cold start): the direction residual
# d̂(x) − t̂ and its Jacobian, bounded to the backward area, the reference's cost floor, the data's resolution.
function spline_project_call(b, t)
    o, w = VgtkCore.GeometryBasics.origin(b.area), VgtkCore.GeometryBasics.widths(b.area)
    lb = SVector{2, Float64}(o[1], o[2]); ub = lb + SVector{2, Float64}(w[1], w[2])
    s = 1000 ./ SVector{2, Float64}(w[1], w[2])
    tmax = max(abs(t[1]), abs(t[2]), abs(t[3]))
    tn = normalize(SVector{3, Float64}(t[1] / tmax, t[2] / tmax, t[3] / tmax))
    fj(p) = (r = VgtkCore.Cameras._spline_direction_differential(b.controls, b.area, p); (SVector{3, Float64}(r[2]) - tn, r[3]))
    stops = StaticLmStops(s; acceptance = 1.0e-12, max_iter = 100, cost_floor = eps(Float64),
        residual_resolution = 16 * Float64(max(eps(eltype(eltype(b.controls))), eps(Float64))))
    return static_lm(fj, lb + SVector{2, Float64}(w[1], w[2]) / 2, lb, ub, stops)
end

function record_core_sensor_lm(cases)
    # --- photon_transfer: one frame -------------------------------------------------------------------------------
    for seed in (SEEDS..., 101, 102, 103)
        case!(cases, "photon_transfer/one_frame/ptc_frame/seed$seed", ptc_frame(StableRNG(seed)), m -> photon_transfer(m))
    end
    for seed in SEEDS
        m, model = dsnu_frame(StableRNG(10 + seed))
        case!(cases, "photon_transfer/one_frame/dsnu_absent_model/seed$seed", m, m -> photon_transfer(m))
        case!(cases, "photon_transfer/one_frame/dsnu_dark_model/seed$seed", (m, model), ((m, M),) -> photon_transfer(m; sensor_model = M))
        f = ptc_frame(StableRNG(seed))
        case!(cases, "photon_transfer/one_frame/ptc_frame_block4/seed$seed", f, m -> photon_transfer(m; block = 4))
        case!(cases, "photon_transfer/one_frame/ptc_frame_significance0.01/seed$seed", f, m -> photon_transfer(m; significance = 0.01))
        case!(cases, "photon_transfer/one_frame/bright_mismatch_refused/seed$seed", scene_frame(StableRNG(40 + seed); response = gain_response(StableRNG(1000seed + 41))),
            m -> photon_transfer(m))
        case!(cases, "photon_transfer/one_frame/dim_textured/seed$seed", scene_frame(StableRNG(43 + 10seed); levels = DIM_LEVELS, texture = TEXTURE), m -> photon_transfer(m))
        case!(cases, "photon_transfer/one_frame/dim/seed$seed", scene_frame(StableRNG(43 + 10seed); levels = DIM_LEVELS), m -> photon_transfer(m))
        case!(cases, "photon_transfer/one_frame/three_points_refused/seed$seed", three_point_frame(StableRNG(4 + seed)), m -> photon_transfer(m))
    end
    case!(cases, "photon_transfer/one_frame/saturated_refused", synth_mosaic(fill(UInt16(WHITE), 64, 64)), m -> photon_transfer(m))
    case!(cases, "photon_transfer/one_frame/one_level_refused", synth_mosaic(fill(UInt16(1000), 256, 256)), m -> photon_transfer(m))
    case!(cases, "photon_transfer/one_frame/block1_refused", ptc_frame(StableRNG(1)), m -> photon_transfer(m; block = 1))
    case!(cases, "photon_transfer/one_frame/significance0_refused", scene_frame(StableRNG(1)), m -> photon_transfer(m; significance = 0))
    # --- photon_transfer: the series against ReadNoise ------------------------------------------------------------
    for seed in SEEDS
        r = readnoise_series(seed)
        g, t1, t2 = r.groups, r.t1, r.t2
        case!(cases, "photon_transfer/readnoise_series/plain/seed$seed", (g, r.darks), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/readnoise_series/no_normalize/seed$seed", (g, r.darks), ((g, d),) -> photon_transfer(g, d; normalize = false))
        case!(cases, "photon_transfer/readnoise_series/block4/seed$seed", (g, r.darks), ((g, d),) -> photon_transfer(g, d; block = 4))
        case!(cases, "photon_transfer/readnoise_series/std_errors/seed$seed", (g, r.darks_se), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/readnoise_series/wrong_intercept/seed$seed", (g, r.wrong), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/readnoise_series/missing_exposure/seed$seed", (g, Dict(t1 => r.darks[t1])), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/readnoise_series/pairs/seed$seed", (Dict(t => v[1:2] for (t, v) in g), r.darks), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/readnoise_series/four_and_three/seed$seed", (Dict(t1 => g[t1], t2 => g[t2][1:3]), r.darks), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/readnoise_series/one_frame_group/seed$seed", (Dict(t1 => g[t1][1:1], t2 => g[t2]), r.darks), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/readnoise_series/significance1/seed$seed", (g, r.darks), ((g, d),) -> photon_transfer(g, d; significance = 1))
        case!(cases, "photon_transfer/readnoise_series/other_size/seed$seed",
            (Dict(t1 => g[t1], t2 => [synth_mosaic(zeros(UInt16, 64, 64)) for _ in 1:2]), r.darks), ((g, d),) -> photon_transfer(g, d))
    end
    case!(cases, "photon_transfer/readnoise_series/no_group", (Dict{typeof(0.01SEC), Vector{typeof(synth_mosaic(zeros(UInt16, 2, 2)))}}(), Dict(0.01SEC => read_noise_model(SIGMA_EFF))),
        ((g, d),) -> photon_transfer(g, d))
    # --- photon_transfer: the series against Darks ----------------------------------------------------------------
    for seed in SEEDS
        r = darks_series(seed)
        g, t1, t2 = r.groups, r.t1, r.t2
        case!(cases, "photon_transfer/darks_series/plain/seed$seed", (g, r.darks), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/darks_series/no_normalize/seed$seed", (g, r.darks), ((g, d),) -> photon_transfer(g, d; normalize = false))
        case!(cases, "photon_transfer/darks_series/late_defects/seed$seed", (g, r.darks, r.late), ((g, d, D),) -> photon_transfer(g, d; defects = D))
        case!(cases, "photon_transfer/darks_series/half_defects/seed$seed", (g, r.darks, r.half), ((g, d, D),) -> photon_transfer(g, d; defects = D))
        case!(cases, "photon_transfer/darks_series/readnoise_exact_keys/seed$seed", (g, r.float_keyed), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/darks_series/readnoise_one_black/seed$seed",
            (Dict(0.01SEC => g[t1], 0.02SEC => g[t2]), r.float_keyed, r.one_black), ((g, d, M),) -> photon_transfer(g, d; sensor_model = M))
        case!(cases, "photon_transfer/darks_series/no_band/seed$seed", (Dict(t1 => g[t1], 0.03SEC => g[t2]), r.darks), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/darks_series/small_maps/seed$seed", (Dict(t1 => g[t1]), r.small), ((g, d),) -> photon_transfer(g, d))
        case!(cases, "photon_transfer/darks_series/significance0/seed$seed", (g, r.darks), ((g, d),) -> photon_transfer(g, d; significance = 0))
        case!(cases, "photon_transfer/darks_series/one_frame_group/seed$seed", (Dict(t1 => g[t1][1:1]), r.darks), ((g, d),) -> photon_transfer(g, d))
    end
    # --- master_dark --------------------------------------------------------------------------------------------
    for seed in SEEDS
        frames = dark_frames(StableRNG(6 + seed))
        band = (0.5SEC, 1.0SEC)
        case!(cases, "master_dark/planted_16/plain/seed$seed", (frames, band), ((f, b),) -> master_dark(f; exposure_band = b))
        case!(cases, "master_dark/planted_16/false_defects0.2/seed$seed", (frames, band), ((f, b),) -> master_dark(f; exposure_band = b, false_defects = 0.2))
        case!(cases, "master_dark/planted_2/plain/seed$seed", (frames[1:2], band), ((f, b),) -> master_dark(f; exposure_band = b))
        case!(cases, "master_dark/planted_1/refused/seed$seed", (frames[1:1], band), ((f, b),) -> master_dark(f; exposure_band = b))
        case!(cases, "master_dark/other_size/refused/seed$seed", ([frames[1], synth_mosaic(zeros(UInt16, 4, 4))], band), ((f, b),) -> master_dark(f; exposure_band = b))
        case!(cases, "master_dark/planted_16/reversed_band/seed$seed", (frames, (1.0SEC, 0.5SEC)), ((f, b),) -> master_dark(f; exposure_band = b))
        case!(cases, "master_dark/planted_16/false_defects1/seed$seed", (frames, band), ((f, b),) -> master_dark(f; exposure_band = b, false_defects = 1))
        case!(cases, "master_dark/planted_16/no_band/seed$seed", frames, f -> master_dark(f))
    end
    # --- fit_scale ----------------------------------------------------------------------------------------------
    fs = ((p, rss, dof),) -> fit_scale(p, rss, dof)
    case!(cases, "fit_scale/counted_dof_grid/map", fit_scale_counted_grid(), g -> map(fs, g))
    case!(cases, "fit_scale/jeffreys_grid/map", fit_scale_jeffreys_grid(), g -> map(fs, g))
    case!(cases, "fit_scale/floor_grid/map", fit_scale_floor_grid(), g -> map(fs, g))
    for (k, x) in enumerate(fit_scale_integer_args())
        case!(cases, "fit_scale/mixed_number_types/case$k", x, fs)
    end
    for seed in SEEDS
        case!(cases, "fit_scale/random_grid_200/map/seed$seed", fit_scale_random_grid(StableRNG(seed)), g -> map(fs, g))
    end
    # --- static_lm: its tests' problems --------------------------------------------------------------------------
    P = static_lm_cubic()
    cubic(P) = x -> (d = x - P.xs; (P.A * d + 0.1 .* d .^ 3, P.A + Diagonal(0.3 .* d .^ 2), :extra))
    case!(cases, "static_lm/cubic3/root_inside/acceptance1e-20", P,
        P -> static_lm(cubic(P), zero(SVector{3, Float64}), P.lb, P.ub, StaticLmStops(P.s; acceptance = 1.0e-20, max_iter = 50)))
    case!(cases, "static_lm/cubic3/root_outside_box/acceptance1e-20", P,
        P -> static_lm(cubic(P), zero(SVector{3, Float64}), P.lb, SVector(1.0, 1.0, 0.5), StaticLmStops(P.s; acceptance = 1.0e-20, max_iter = 50)))
    case!(cases, "static_lm/cubic3/default_stops", P, P -> static_lm(cubic(P), zero(SVector{3, Float64}), P.lb, P.ub, StaticLmStops(P.s)))
    case!(cases, "static_lm/cubic3/bound0.1", P,
        P -> static_lm(cubic(P), zero(SVector{3, Float64}), P.lb, P.ub, StaticLmStops(P.s; acceptance = 1.0e-20, max_iter = 50), 0.1))
    case!(cases, "static_lm/cubic3/lax_one_iteration", P,
        P -> static_lm(cubic(P), SVector(-0.9, 0.9, -0.9), P.lb, P.ub, StaticLmStops(P.s; acceptance = 1.0e3, max_iter = 1)))
    case!(cases, "static_lm/linear3_float32/acceptance1e-10", P, P -> begin
        A32 = SMatrix{3, 3, Float32}(P.A); xs32 = SVector{3, Float32}(P.xs)
        static_lm(x -> (d = x - xs32; (A32 * d, A32)), zero(SVector{3, Float32}), SVector{3, Float32}(P.lb), SVector{3, Float32}(P.ub),
            StaticLmStops(SVector{3, Float32}(P.s); acceptance = 1.0f-10, max_iter = 50))
    end)
    for acc in (0.5, 2.0)
        case!(cases, "static_lm/offset_residual1/acceptance$acc", (SVector(0.0), SVector(-1.0), SVector(1.0), SVector(1.0)),
            ((x0, lb, ub, s),) -> static_lm(x -> (SVector(x[1] - 0.5, 1.0), SMatrix{2, 1}(1.0, 0.0)), x0, lb, ub, StaticLmStops(s; acceptance = acc)))
    end
    for seed in SEEDS
        case!(cases, "static_lm/cubic3/random_starts_50/seed$seed", (P, static_lm_starts(StableRNG(seed))),
            ((P, x0s),) -> [static_lm(cubic(P), x0, P.lb, P.ub, StaticLmStops(P.s; acceptance = 1.0e-20, max_iter = 50)) for x0 in x0s])
    end
    # --- static_lm: the recorded pixel-LM reference problems -----------------------------------------------------
    ps = pixel_lm_problems()
    for tol2 in (1.0e-2, 1.0e-6, 1.0e-12)
        case!(cases, "static_lm/pixel_reference_1000/tol$tol2", ps, ps -> map(ps) do q
            res, jac = pixel_res(q.coef...), pixel_jac(q.coef...)
            stops = StaticLmStops(SVector(1.0, 1.0); acceptance = tol2, max_iter = q.maxit, cost_floor = eps(Float64))
            static_lm(p -> (res(p, nothing), jac(p, nothing)), SVector(Float64(q.x0), Float64(q.y0)),
                SVector(q.minx, q.miny), SVector(q.maxx + 0.999, q.maxy + 0.999), stops)
        end)
    end
    # --- static_lm: the lens inverse's problems ------------------------------------------------------------------
    for (k, lens) in enumerate(inverse_lenses()), T in (Float64, Float32)
        l = convert_eltype(T, lens)
        case!(cases, "static_lm/lens_inverse/lens$k/$(T)/test_points", (l.distortion, [SVector{2, T}(x) for x in INVERSE_POINTS]),
            ((d, xs),) -> [inverse_lm_call(d, x) for x in xs])
        for seed in SEEDS
            case!(cases, "static_lm/lens_inverse/lens$k/$(T)/random_40/seed$seed", (l.distortion, [SVector{2, T}(x) for x in inverse_points(StableRNG(seed))]),
                ((d, xs),) -> [inverse_lm_call(d, x) for x in xs])
        end
    end
    # --- static_lm: the SplineLens projection's problems ---------------------------------------------------------
    b = spline_camera().lens.backward
    for seed in SEEDS
        case!(cases, "static_lm/spline_projection/kb_fisheye_16x16/rays_60/seed$seed", (b, spline_rays(StableRNG(seed))),
            ((b, ts),) -> [spline_project_call(b, t) for t in ts])
    end
    return cases
end
