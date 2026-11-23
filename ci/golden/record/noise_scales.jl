# The old noise-scale estimators at 3d8a06da: Core's mad_scale and Estimation's pooled_noise_scale (both forms).
using VisualGeometryToolkitCore: mad_scale
using VisualGeometryToolkitEstimation: find_homography, pooled_noise_scale

function record_noise_scales(cases)
    for seed in SEEDS, (name, rs) in ns_residuals(seed)
        case!(cases, "median_norm_scale/$name/seed$seed", rs, rs -> mad_scale(rs))
    end
    for (name, rs) in ns_exact_residuals()
        case!(cases, "median_norm_scale/$name", rs, rs -> mad_scale(rs))
    end
    case!(cases, "median_norm_scale/refused/plain_vector", [1.0, 2.0, 3.0], rs -> mad_scale(rs))
    for seed in SEEDS
        rs = [find_homography(cs; rng = StableRNG(seed)) for cs in ns_views(seed)]
        failed = find_homography(homography_too_few(); rng = StableRNG(seed))
        case!(cases, "pooled_scale/results4/seed$seed", rs, rs -> pooled_noise_scale(rs))
        case!(cases, "pooled_scale/results1/seed$seed", rs[1:1], rs -> pooled_noise_scale(rs))
        case!(cases, "pooled_scale/results_with_failed/seed$seed", [rs; failed], rs -> pooled_noise_scale(rs))
        case!(cases, "pooled_scale/only_failed/seed$seed", [failed], rs -> pooled_noise_scale(rs))
        s = [VgtkCore.noise_scale(r) for r in rs]
        ν = [r.dof for r in rs]
        case!(cases, "pooled_scale/numbers/seed$seed", (s, ν), ((s, ν),) -> pooled_noise_scale(s, ν))
        case!(cases, "pooled_scale/numbers_left_out/seed$seed", ([s; NaN; 0.7], [ν; 12; 0]), ((s, ν),) -> pooled_noise_scale(s, ν))
        case!(cases, "pooled_scale/refused/length_mismatch/seed$seed", (s, ν[1:2]), ((s, ν),) -> pooled_noise_scale(s, ν))
    end
    case!(cases, "pooled_scale/numbers_none", ([NaN], [5]), ((s, ν),) -> pooled_noise_scale(s, ν))
    return cases
end
