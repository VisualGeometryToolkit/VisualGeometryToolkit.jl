# Core's noise-model estimators and device LM on Core 0.14 (spec §3, §9):
#   photon_transfer(m; sensor_model, block, significance) → solve(EstimationProblem(PoissonGaussian, m[, (; sensor_model)]),
#       PhotonTransfer(; block, significance)) — one frame: a FitResult whose `u` is the old bare PoissonGaussian;
#   photon_transfer(groups, darks; defects | sensor_model, …) → the same on `p = (; darks[, defects | sensor_model])`;
#   master_dark(frames; exposure_band, false_defects) → solve(EstimationProblem(Dark, frames, (; exposure_band)),
#       MasterDark(; false_defects)) — `u` the old `dark`, `metadata` the old `read_noise` and `defects`;
#   fit_scale(prior, rss, dof) → solve(EstimationProblem(SquaredScale, (; rss, dof)), ScaleMAP(prior)), `u` the old value;
#   static_lm(fj, x₀, lb, ub, stops[, bound]) → solve(RefineProblem(fj, x₀; lb, ub), StaticLevenbergMarquardt(); stops,
#       bound): the old tuple's slots are the new result's fields — x is `u`, `converged` is `retcode == Success` (the one
#       rule, `static_lm_termination`), J and e are `metadata.jacobian` / `metadata.evaluation`, the rule `stop_rule`;
#       the run reads them back into the old tuple so a recorded VECTOR of solves (one digest) is compared leaf for leaf.
using VisualGeometryToolkitCore: EstimationProblem, RefineProblem, solve, PhotonTransfer, MasterDark, PoissonGaussian, Dark,
    ScaleMAP, SquaredScale, StaticLevenbergMarquardt, StaticLmStops, ReturnCode, NullParameters

# A closed form / one-frame fit that returned a bare value: `u` is it, every other field new.
const _SL_NEW = ["", "resid", "retcode", "stats", "prob", "alg", "original", "stop_rule", "rms", "n", "metadata"]
const _SL_BARE = FieldMap(; renames = ["u" => ""], new_only = _SL_NEW)
const _SL_REFUSAL = FieldMap(; ignored = ["exception.msg"])   # the message named `photon_transfer` / `master_dark`
# A series fit: a FitResult before and after (SciML's names; the result type's parameters changed).
const _SL_SERIES = FieldMap(; renames = ["u" => "estimate", "stats.nsteps" => "iterations"],
    new_only = ["prob", "alg", "original", "stats", "resid"], retyped = [""])
# MasterDark: the old named tuple's `dark` is `u`, its `read_noise` and `defects` the metadata.
const _SL_DARK = FieldMap(; renames = ["u" => "dark", "metadata.read_noise" => "read_noise", "metadata.defects" => "defects"],
    new_only = ["resid", "retcode", "stats", "prob", "alg", "original", "stop_rule", "rms", "n", "metadata"], retyped = [""])

_sl_ptc(m, p = NullParameters(); kw...) = solve(EstimationProblem(PoissonGaussian, m, p), PhotonTransfer(; kw...))
_sl_dark(f, p; kw...) = solve(EstimationProblem(Dark, f, p), MasterDark(; kw...))
_sl_scale((p, rss, dof)) = solve(EstimationProblem(SquaredScale, (; rss, dof)), ScaleMAP(p))
function _sl_lm(fj, x0, lb, ub, stops, bound = nothing)
    r = solve(RefineProblem(fj, x0; lb, ub), StaticLevenbergMarquardt(); stops, bound)
    return (r.u, r.retcode == ReturnCode.Success, r.metadata.jacobian, r.metadata.evaluation, r.stop_rule)
end

# The callers' closures (record/core_sensor_lm.jl), on the device LM's solve.
function _sl_inverse_lm_call(d, xd::SVector{2, S}) where {S}
    fj(z) = (yJ = VgtkCore.Cameras._distort_and_jacobian(d, z); y = yJ[1]; (SVector{2, S}(y[1] - xd[1], y[2] - xd[2]), SMatrix{2, 2, S}(yJ[2])))
    inf = S(Inf)
    tol = S === Float64 ? 1.0e-10 : 1.0f-5
    s = S(1.0e-10 * (eps(S) / eps(Float64)) / tol)
    stops = StaticLmStops(SVector(s, s); acceptance = tol^2 + (8 * eps(S))^2 * (xd[1]^2 + xd[2]^2),
        residual_scale = max(one(S), sqrt(xd[1]^2 + xd[2]^2)), max_iter = S === Float64 ? 20 : 12)
    return _sl_lm(fj, xd, SVector(-inf, -inf), SVector(inf, inf), stops, S(1) / 10)
end
function _sl_spline_project_call(b, t)
    o, w = VgtkCore.GeometryBasics.origin(b.area), VgtkCore.GeometryBasics.widths(b.area)
    lb = SVector{2, Float64}(o[1], o[2]); ub = lb + SVector{2, Float64}(w[1], w[2])
    s = 1000 ./ SVector{2, Float64}(w[1], w[2])
    tmax = max(abs(t[1]), abs(t[2]), abs(t[3]))
    tn = normalize(SVector{3, Float64}(t[1] / tmax, t[2] / tmax, t[3] / tmax))
    fj(p) = (r = VgtkCore.Cameras._spline_direction_differential(b.controls, b.area, p); (SVector{3, Float64}(r[2]) - tn, r[3]))
    stops = StaticLmStops(s; acceptance = 1.0e-12, max_iter = 100, cost_floor = eps(Float64),
        residual_resolution = 16 * Float64(max(eps(eltype(eltype(b.controls))), eps(Float64))))
    return _sl_lm(fj, lb + SVector{2, Float64}(w[1], w[2]) / 2, lb, ub, stops)
end

function replay_core_sensor_lm(cases)
    B, R, S = _SL_BARE, _SL_REFUSAL, _SL_SERIES
    # --- photon_transfer: one frame -----------------------------------------------------------------------------
    for seed in (SEEDS..., 101, 102, 103)
        replay!(cases, "photon_transfer/one_frame/ptc_frame/seed$seed", ptc_frame(StableRNG(seed)), m -> _sl_ptc(m); map = B)
    end
    for seed in SEEDS
        m, model = dsnu_frame(StableRNG(10 + seed))
        replay!(cases, "photon_transfer/one_frame/dsnu_absent_model/seed$seed", m, m -> _sl_ptc(m); map = B)
        replay!(cases, "photon_transfer/one_frame/dsnu_dark_model/seed$seed", (m, model), ((m, M),) -> _sl_ptc(m, (; sensor_model = M)); map = B)
        f = ptc_frame(StableRNG(seed))
        replay!(cases, "photon_transfer/one_frame/ptc_frame_block4/seed$seed", f, m -> _sl_ptc(m; block = 4); map = B)
        replay!(cases, "photon_transfer/one_frame/ptc_frame_significance0.01/seed$seed", f, m -> _sl_ptc(m; significance = 0.01); map = B)
        replay!(cases, "photon_transfer/one_frame/bright_mismatch_refused/seed$seed",
            scene_frame(StableRNG(40 + seed); response = gain_response(StableRNG(1000seed + 41))), m -> _sl_ptc(m); map = R)   # the message named photon_transfer(groups, darks)
        replay!(cases, "photon_transfer/one_frame/dim_textured/seed$seed", scene_frame(StableRNG(43 + 10seed); levels = DIM_LEVELS, texture = TEXTURE),
            m -> _sl_ptc(m); map = B)
        replay!(cases, "photon_transfer/one_frame/dim/seed$seed", scene_frame(StableRNG(43 + 10seed); levels = DIM_LEVELS), m -> _sl_ptc(m); map = B)
        replay!(cases, "photon_transfer/one_frame/three_points_refused/seed$seed", three_point_frame(StableRNG(4 + seed)), m -> _sl_ptc(m); map = R)   # the message named photon_transfer(groups, darks)
    end
    replay!(cases, "photon_transfer/one_frame/saturated_refused", synth_mosaic(fill(UInt16(WHITE), 64, 64)), m -> _sl_ptc(m); map = B)
    replay!(cases, "photon_transfer/one_frame/one_level_refused", synth_mosaic(fill(UInt16(1000), 256, 256)), m -> _sl_ptc(m); map = B)
    replay!(cases, "photon_transfer/one_frame/block1_refused", ptc_frame(StableRNG(1)), m -> _sl_ptc(m; block = 1); map = R)
    replay!(cases, "photon_transfer/one_frame/significance0_refused", scene_frame(StableRNG(1)), m -> _sl_ptc(m; significance = 0); map = R)
    # --- photon_transfer: the series against ReadNoise ----------------------------------------------------------
    for seed in SEEDS
        r = readnoise_series(seed)
        g, t1, t2 = r.groups, r.t1, r.t2
        rn = ((g, d),) -> _sl_ptc(g, (; darks = d))
        replay!(cases, "photon_transfer/readnoise_series/plain/seed$seed", (g, r.darks), rn; map = S)
        replay!(cases, "photon_transfer/readnoise_series/no_normalize/seed$seed", (g, r.darks), ((g, d),) -> _sl_ptc(g, (; darks = d); normalize = false); map = S)
        replay!(cases, "photon_transfer/readnoise_series/block4/seed$seed", (g, r.darks), ((g, d),) -> _sl_ptc(g, (; darks = d); block = 4); map = S)
        replay!(cases, "photon_transfer/readnoise_series/std_errors/seed$seed", (g, r.darks_se), rn; map = S)
        replay!(cases, "photon_transfer/readnoise_series/wrong_intercept/seed$seed", (g, r.wrong), rn; map = S)
        replay!(cases, "photon_transfer/readnoise_series/missing_exposure/seed$seed", (g, Dict(t1 => r.darks[t1])), rn; map = R)
        replay!(cases, "photon_transfer/readnoise_series/pairs/seed$seed", (Dict(t => v[1:2] for (t, v) in g), r.darks), rn; map = S)
        replay!(cases, "photon_transfer/readnoise_series/four_and_three/seed$seed", (Dict(t1 => g[t1], t2 => g[t2][1:3]), r.darks), rn; map = S)
        replay!(cases, "photon_transfer/readnoise_series/one_frame_group/seed$seed", (Dict(t1 => g[t1][1:1], t2 => g[t2]), r.darks), rn; map = R)
        replay!(cases, "photon_transfer/readnoise_series/significance1/seed$seed", (g, r.darks),
            ((g, d),) -> _sl_ptc(g, (; darks = d); significance = 1); map = R)
        replay!(cases, "photon_transfer/readnoise_series/other_size/seed$seed",
            (Dict(t1 => g[t1], t2 => [synth_mosaic(zeros(UInt16, 64, 64)) for _ in 1:2]), r.darks), rn; map = R)
    end
    replay!(cases, "photon_transfer/readnoise_series/no_group",
        (Dict{typeof(0.01SEC), Vector{typeof(synth_mosaic(zeros(UInt16, 2, 2)))}}(), Dict(0.01SEC => read_noise_model(SIGMA_EFF))),
        ((g, d),) -> _sl_ptc(g, (; darks = d)); map = R)
    # --- photon_transfer: the series against Darks --------------------------------------------------------------
    for seed in SEEDS
        r = darks_series(seed)
        g, t1, t2 = r.groups, r.t1, r.t2
        dk = ((g, d),) -> _sl_ptc(g, (; darks = d))
        replay!(cases, "photon_transfer/darks_series/plain/seed$seed", (g, r.darks), dk; map = S)
        replay!(cases, "photon_transfer/darks_series/no_normalize/seed$seed", (g, r.darks), ((g, d),) -> _sl_ptc(g, (; darks = d); normalize = false); map = S)
        replay!(cases, "photon_transfer/darks_series/late_defects/seed$seed", (g, r.darks, r.late), ((g, d, D),) -> _sl_ptc(g, (; darks = d, defects = D)); map = S)
        replay!(cases, "photon_transfer/darks_series/half_defects/seed$seed", (g, r.darks, r.half), ((g, d, D),) -> _sl_ptc(g, (; darks = d, defects = D)); map = S)
        replay!(cases, "photon_transfer/darks_series/readnoise_exact_keys/seed$seed", (g, r.float_keyed), dk; map = R)
        replay!(cases, "photon_transfer/darks_series/readnoise_one_black/seed$seed",
            (Dict(0.01SEC => g[t1], 0.02SEC => g[t2]), r.float_keyed, r.one_black), ((g, d, M),) -> _sl_ptc(g, (; darks = d, sensor_model = M)); map = S)
        replay!(cases, "photon_transfer/darks_series/no_band/seed$seed", (Dict(t1 => g[t1], 0.03SEC => g[t2]), r.darks), dk; map = R)
        replay!(cases, "photon_transfer/darks_series/small_maps/seed$seed", (Dict(t1 => g[t1]), r.small), dk; map = R)
        replay!(cases, "photon_transfer/darks_series/significance0/seed$seed", (g, r.darks), ((g, d),) -> _sl_ptc(g, (; darks = d); significance = 0); map = R)
        replay!(cases, "photon_transfer/darks_series/one_frame_group/seed$seed", (Dict(t1 => g[t1][1:1]), r.darks), dk; map = R)
    end
    # --- master_dark → MasterDark(), the band the context -------------------------------------------------------
    for seed in SEEDS
        frames = dark_frames(StableRNG(6 + seed))
        band = (0.5SEC, 1.0SEC)
        md = ((f, b),) -> _sl_dark(f, (; exposure_band = b))
        replay!(cases, "master_dark/planted_16/plain/seed$seed", (frames, band), md; map = _SL_DARK)
        replay!(cases, "master_dark/planted_16/false_defects0.2/seed$seed", (frames, band),
            ((f, b),) -> _sl_dark(f, (; exposure_band = b); false_defects = 0.2); map = _SL_DARK)
        replay!(cases, "master_dark/planted_2/plain/seed$seed", (frames[1:2], band), md; map = _SL_DARK)
        replay!(cases, "master_dark/planted_1/refused/seed$seed", (frames[1:1], band), md; map = R)
        replay!(cases, "master_dark/other_size/refused/seed$seed", ([frames[1], synth_mosaic(zeros(UInt16, 4, 4))], band), md; map = R)
        replay!(cases, "master_dark/planted_16/reversed_band/seed$seed", (frames, (1.0SEC, 0.5SEC)), md; map = R)
        replay!(cases, "master_dark/planted_16/false_defects1/seed$seed", (frames, band),
            ((f, b),) -> _sl_dark(f, (; exposure_band = b); false_defects = 1); map = R)
        replay!(cases, "master_dark/planted_16/no_band/seed$seed", frames, f -> _sl_dark(f, NullParameters()); map = R)
    end
    # --- fit_scale → ScaleMAP(prior) on SquaredScale ------------------------------------------------------------
    us = g -> map(x -> _sl_scale(x).u, g)   # a recorded vector of values: each solve's `u`
    replay!(cases, "fit_scale/counted_dof_grid/map", fit_scale_counted_grid(), us)
    replay!(cases, "fit_scale/jeffreys_grid/map", fit_scale_jeffreys_grid(), us)
    replay!(cases, "fit_scale/floor_grid/map", fit_scale_floor_grid(), us)
    for (k, x) in enumerate(fit_scale_integer_args())
        replay!(cases, "fit_scale/mixed_number_types/case$k", x, _sl_scale; map = B)
    end
    for seed in SEEDS
        replay!(cases, "fit_scale/random_grid_200/map/seed$seed", fit_scale_random_grid(StableRNG(seed)), us)
    end
    # --- static_lm → StaticLevenbergMarquardt(): its tests' problems --------------------------------------------
    P = static_lm_cubic()
    cubic(P) = x -> (d = x - P.xs; (P.A * d + 0.1 .* d .^ 3, P.A + Diagonal(0.3 .* d .^ 2), :extra))
    replay!(cases, "static_lm/cubic3/root_inside/acceptance1e-20", P,
        P -> _sl_lm(cubic(P), zero(SVector{3, Float64}), P.lb, P.ub, StaticLmStops(P.s; acceptance = 1.0e-20, max_iter = 50)))
    replay!(cases, "static_lm/cubic3/root_outside_box/acceptance1e-20", P,
        P -> _sl_lm(cubic(P), zero(SVector{3, Float64}), P.lb, SVector(1.0, 1.0, 0.5), StaticLmStops(P.s; acceptance = 1.0e-20, max_iter = 50)))
    replay!(cases, "static_lm/cubic3/default_stops", P, P -> _sl_lm(cubic(P), zero(SVector{3, Float64}), P.lb, P.ub, StaticLmStops(P.s)))
    replay!(cases, "static_lm/cubic3/bound0.1", P,
        P -> _sl_lm(cubic(P), zero(SVector{3, Float64}), P.lb, P.ub, StaticLmStops(P.s; acceptance = 1.0e-20, max_iter = 50), 0.1))
    replay!(cases, "static_lm/cubic3/lax_one_iteration", P,
        P -> _sl_lm(cubic(P), SVector(-0.9, 0.9, -0.9), P.lb, P.ub, StaticLmStops(P.s; acceptance = 1.0e3, max_iter = 1)))
    replay!(cases, "static_lm/linear3_float32/acceptance1e-10", P, P -> begin
        A32 = SMatrix{3, 3, Float32}(P.A); xs32 = SVector{3, Float32}(P.xs)
        _sl_lm(x -> (d = x - xs32; (A32 * d, A32)), zero(SVector{3, Float32}), SVector{3, Float32}(P.lb), SVector{3, Float32}(P.ub),
            StaticLmStops(SVector{3, Float32}(P.s); acceptance = 1.0f-10, max_iter = 50))
    end)
    for acc in (0.5, 2.0)
        replay!(cases, "static_lm/offset_residual1/acceptance$acc", (SVector(0.0), SVector(-1.0), SVector(1.0), SVector(1.0)),
            ((x0, lb, ub, s),) -> _sl_lm(x -> (SVector(x[1] - 0.5, 1.0), SMatrix{2, 1}(1.0, 0.0)), x0, lb, ub, StaticLmStops(s; acceptance = acc)))
    end
    for seed in SEEDS
        replay!(cases, "static_lm/cubic3/random_starts_50/seed$seed", (P, static_lm_starts(StableRNG(seed))),
            ((P, x0s),) -> [_sl_lm(cubic(P), x0, P.lb, P.ub, StaticLmStops(P.s; acceptance = 1.0e-20, max_iter = 50)) for x0 in x0s])
    end
    # --- the recorded pixel-LM reference problems ---------------------------------------------------------------
    ps = pixel_lm_problems()
    for tol2 in (1.0e-2, 1.0e-6, 1.0e-12)
        replay!(cases, "static_lm/pixel_reference_1000/tol$tol2", ps, ps -> map(ps) do q
            res, jac = pixel_res(q.coef...), pixel_jac(q.coef...)
            stops = StaticLmStops(SVector(1.0, 1.0); acceptance = tol2, max_iter = q.maxit, cost_floor = eps(Float64))
            _sl_lm(p -> (res(p, nothing), jac(p, nothing)), SVector(Float64(q.x0), Float64(q.y0)),
                SVector(q.minx, q.miny), SVector(q.maxx + 0.999, q.maxy + 0.999), stops)
        end)
    end
    # --- the lens inverse's problems ----------------------------------------------------------------------------
    for (k, lens) in enumerate(inverse_lenses()), T in (Float64, Float32)
        l = convert_eltype(T, lens)
        replay!(cases, "static_lm/lens_inverse/lens$k/$(T)/test_points", (l.distortion, [SVector{2, T}(x) for x in INVERSE_POINTS]),
            ((d, xs),) -> [_sl_inverse_lm_call(d, x) for x in xs])
        for seed in SEEDS
            replay!(cases, "static_lm/lens_inverse/lens$k/$(T)/random_40/seed$seed", (l.distortion, [SVector{2, T}(x) for x in inverse_points(StableRNG(seed))]),
                ((d, xs),) -> [_sl_inverse_lm_call(d, x) for x in xs])
        end
    end
    # --- the SplineLens projection's problems -------------------------------------------------------------------
    b = spline_camera().lens.backward
    for seed in SEEDS
        replay!(cases, "static_lm/spline_projection/kb_fisheye_16x16/rays_60/seed$seed", (b, spline_rays(StableRNG(seed))),
            ((b, ts),) -> [_sl_spline_project_call(b, t) for t in ts])
    end
    return cases
end
