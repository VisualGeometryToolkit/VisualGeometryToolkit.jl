# The old closed-form (one-step) estimators of VisualGeometryToolkitCore at 3d8a06da: fundamental_dlt, essential_dlt
# (its only backend, PoseLib's, which the bare form routes to once PoseLib is loaded), kabsch, chordal_mean,
# spectral_rotation_averaging, zhang_intrinsics, tls_line, direct_ellipse, nearest_point, triangulate and
# p3p(model, cs).
using PoseLib: PoseLib
using VisualGeometryToolkitCore: fundamental_dlt, essential_dlt, kabsch, chordal_mean, spectral_rotation_averaging,
    zhang_intrinsics, tls_line, direct_ellipse, nearest_point, triangulate, p3p, SvdWorkspace, PoseLibSolver,
    fundamental_dlt!, tls_line_statistics, CorrespondenceNormalization, FundMat

function record_core_closed_forms(cases)
    # --- fundamental_dlt ------------------------------------------------------------------------------------------
    for seed in SEEDS
        fcs = fundmat_guard_pairs(StableRNG(seed))
        case!(cases, "fundmat_dlt/guard_8/plain/seed$seed", fcs[1:8], cs -> fundamental_dlt(cs))
        case!(cases, "fundmat_dlt/guard_8/svd_ws/seed$seed", fcs[1:8], cs -> fundamental_dlt(cs; svd_ws = SvdWorkspace{Float64}(100, 9)))
        case!(cases, "fundmat_dlt/guard_9/plain/seed$seed", fcs, cs -> fundamental_dlt(cs))
        case!(cases, "fundmat_dlt/guard_8/refused_long_index_mask/seed$seed", fcs[1:8], cs -> fundamental_dlt(cs; mask = [1:8; 1:3]))
        case!(cases, "fundmat_dlt/guard_7/too_few/seed$seed", fcs[1:7], cs -> fundamental_dlt(cs))
        sc = epipolar_layer_scene(StableRNG(seed))
        exact = Correspondences(sc.u1, sc.u2)
        noisy = Correspondences(sc.u1, sc.u2n)
        w = [i == 1 ? 3.0 : 1.0 for i in 1:20]
        keep = BitVector(i > 3 for i in 1:20)
        case!(cases, "fundmat_dlt/layer_20/exact/seed$seed", exact, cs -> fundamental_dlt(cs))
        case!(cases, "fundmat_dlt/layer_20/noisy/seed$seed", noisy, cs -> fundamental_dlt(cs))
        case!(cases, "fundmat_dlt/layer_20/noisy_pair_vector/seed$seed", collect(noisy), cs -> fundamental_dlt(cs))
        case!(cases, "fundmat_dlt/layer_20/noisy_weights_3_1/seed$seed", withattrs(noisy; weights = w), cs -> fundamental_dlt(cs))
        case!(cases, "fundmat_dlt/layer_20/noisy_repeated_first/seed$seed",
            Correspondences([fill(sc.u1[1], 2); sc.u1], [fill(sc.u2n[1], 2); sc.u2n]), cs -> fundamental_dlt(cs))
        case!(cases, "fundmat_dlt/layer_20/noisy_zero_weights/seed$seed", withattrs(noisy; weights = Float64.(keep)), cs -> fundamental_dlt(cs))
        case!(cases, "fundmat_dlt/layer_20/noisy_bool_mask/seed$seed", (noisy, keep), ((cs, m),) -> fundamental_dlt(cs; mask = m))
        case!(cases, "fundmat_dlt/layer_20/noisy_index_mask/seed$seed", (noisy, findall(keep)), ((cs, m),) -> fundamental_dlt(cs; mask = m))
        case!(cases, "fundmat_dlt/layer_20/refused_weights_keyword/seed$seed", (noisy, w), ((cs, w),) -> fundamental_dlt(cs; weights = w))
        case!(cases, "fundmat_dlt/layer_20/refused_parallel_arrays/seed$seed", (sc.u1, sc.u2n), ((a, b),) -> fundamental_dlt(a, b))
        nonfinite = Correspondences([[Point2{Float64}(NaN, NaN)]; sc.u1[2:9]], sc.u2[1:9])
        case!(cases, "fundmat_dlt/layer_9/nonfinite_row/seed$seed", nonfinite, cs -> fundamental_dlt(cs))
        F = fundmat_scene_pairs(StableRNG(seed))
        case!(cases, "fundmat_dlt/fundmat_scene_130/all/seed$seed", F, cs -> fundamental_dlt(cs))
        case!(cases, "fundmat_dlt/fundmat_scene_130/inliers_mask/seed$seed", (F, collect(1:100)), ((cs, m),) -> fundamental_dlt(cs; mask = m))
        case!(cases, "fundmat_dlt/fundmat_scene_130/svd_ws/seed$seed", F, cs -> fundamental_dlt(cs; svd_ws = SvdWorkspace{Float64}(2 * length(cs), 9)))
        # the inner-loop form: the caller's buffer and the estimator's normalization (built from the data), as an LO refit uses it
        case!(cases, "fundmat_dlt_inplace/fundmat_scene_130/all/seed$seed", F,
            cs -> fundamental_dlt!(Matrix{Float64}(undef, length(cs), 9), CorrespondenceNormalization(FundMat, cs), cs))
        case!(cases, "fundmat_dlt_inplace/fundmat_scene_130/inliers_mask_svd_ws/seed$seed", (F, BitVector(i <= 100 for i in 1:length(F))),
            ((cs, m),) -> fundamental_dlt!(Matrix{Float64}(undef, length(cs), 9), CorrespondenceNormalization(FundMat, cs; mask = m), cs;
                mask = m, svd_ws = SvdWorkspace{Float64}(length(cs), 9)))
        case!(cases, "fundmat_dlt_inplace/layer_20/noisy_weights_3_1/seed$seed", withattrs(noisy; weights = w),
            cs -> fundamental_dlt!(Matrix{Float64}(undef, length(cs), 9), CorrespondenceNormalization(FundMat, cs), cs))
    end
    # --- essential_dlt (PoseLib) --------------------------------------------------------------------------------
    for seed in SEEDS
        e = essential_scene(StableRNG(seed))
        case!(cases, "essential_dlt/refine_scene_20/bare/seed$seed", Correspondences(e.b1[1:20], e.b2[1:20]), cs -> essential_dlt(cs))
        case!(cases, "essential_dlt/refine_scene_20/poselib_pairs/seed$seed", e.b1[1:20] .=> e.b2[1:20], cs -> essential_dlt(PoseLibSolver(), cs))
        case!(cases, "essential_dlt/refine_scene_200/exact/seed$seed", Correspondences(e.b1, e.b2), cs -> essential_dlt(cs))
        case!(cases, "essential_dlt/refine_scene_200/noisy/seed$seed", Correspondences(e.b1, e.b2n), cs -> essential_dlt(cs))
        case!(cases, "essential_dlt/refine_scene_8/minimum/seed$seed", Correspondences(e.b1[1:8], e.b2n[1:8]), cs -> essential_dlt(cs))
        case!(cases, "essential_dlt/refine_scene_7/too_few/seed$seed", Correspondences(e.b1[1:7], e.b2[1:7]), cs -> essential_dlt(cs))
        case!(cases, "essential_dlt/refine_scene_20/refused_chart/seed$seed", Correspondences(e.c1[1:20], e.c2[1:20]),
            cs -> essential_dlt(PoseLibSolver(), cs))
    end
    # --- kabsch -------------------------------------------------------------------------------------------------
    for seed in SEEDS
        k = kabsch_scene(StableRNG(seed))
        o = SVector(3.0, -1.0, 2.0)
        case!(cases, "kabsch/fits_40/exact_pairs/seed$seed", k.as .=> k.bs, cs -> kabsch(cs))
        case!(cases, "kabsch/fits_40/exact_correspondences/seed$seed", Correspondences(k.as, k.bs), cs -> kabsch(cs))
        case!(cases, "kabsch/fits_40/noisy/seed$seed", k.as .=> k.bsn, cs -> kabsch(cs))
        case!(cases, "kabsch/fits_40/offset/seed$seed", (k.as .+ Ref(o)) .=> (k.bs .+ Ref(o)), cs -> kabsch(cs))
        case!(cases, "kabsch/fits_1/too_few/seed$seed", k.as[1:1] .=> k.bs[1:1], cs -> kabsch(cs))
        w1 = zeros(40); w1[7] = 1.0
        w2 = zeros(40); w2[7] = 1.0; w2[11] = 2.0
        case!(cases, "kabsch/fits_40/one_positive_weight/seed$seed", withattrs(Correspondences(k.as, k.bs); weights = w1), cs -> kabsch(cs))
        case!(cases, "kabsch/fits_40/two_positive_weights/seed$seed", withattrs(Correspondences(k.as, k.bs); weights = w2), cs -> kabsch(cs))
        case!(cases, "kabsch/fits_40/refused_parallel_arrays/seed$seed", (k.as, k.bs), ((a, b),) -> kabsch(a, b))
        kw = kabsch_weighted_scene(StableRNG(seed))
        cs = Correspondences(kw.as, kw.bs)
        case!(cases, "kabsch/weighted_30/weights/seed$seed", withattrs(cs; weights = kw.w), cs -> kabsch(cs))
        case!(cases, "kabsch/weighted_30/unit_weights/seed$seed", withattrs(cs; weights = ones(30)), cs -> kabsch(cs))
        case!(cases, "kabsch/weighted_30/unweighted/seed$seed", cs, cs -> kabsch(cs))
        bad = withattrs(Correspondences(vcat(kw.as, [SVector(1.0, 0.0, 0.0)]), vcat(kw.bs, [SVector(0.0, 0.0, -1.0)])); weights = vcat(ones(30), 0.0))
        case!(cases, "kabsch/weighted_31/zero_weight_outlier/seed$seed", bad, cs -> kabsch(cs))
        case!(cases, "kabsch/weighted_31/unit_weight_outlier/seed$seed", withattrs(bad; weights = ones(31)), cs -> kabsch(cs))
        case!(cases, "kabsch/weighted_30/negative_weight/seed$seed", withattrs(cs; weights = vcat(-1.0, ones(29))), cs -> kabsch(cs))
        case!(cases, "kabsch/weighted_30/nan_weight/seed$seed", withattrs(cs; weights = vcat(NaN, ones(29))), cs -> kabsch(cs))
        csm = Correspondences([a .* mm for a in kw.as], [b .* mm for b in kw.bs])
        case!(cases, "kabsch/weighted_30/millimetres_weights/seed$seed", withattrs(csm; weights = kw.w), cs -> kabsch(cs))
        case!(cases, "kabsch/weighted_30/millimetres/seed$seed", csm, cs -> kabsch(cs))
    end
    # --- chordal_mean -------------------------------------------------------------------------------------------
    for seed in SEEDS
        c = chordal_scene(StableRNG(seed))
        case!(cases, "chordal_mean/symmetric_4/rotations/seed$seed", c.Rs, Rs -> chordal_mean(Rs))
        case!(cases, "chordal_mean/symmetric_4/matrices/seed$seed", SMatrix{3, 3, Float64}.(c.Rs), Rs -> chordal_mean(Rs))
        case!(cases, "chordal_mean/single/rotation/seed$seed", [c.Rc], Rs -> chordal_mean(Rs))
        case!(cases, "chordal_mean/noisy_5/rotations/seed$seed", c.noisy, Rs -> chordal_mean(Rs))
    end
    case!(cases, "chordal_mean/empty/refused", RotMatrix{3, Float64}[], Rs -> chordal_mean(Rs))
    # --- spectral_rotation_averaging ----------------------------------------------------------------------------
    for seed in SEEDS
        r = rotation_averaging_scene(StableRNG(seed))
        for key in (:pairs, :chain, :noisy, :bad, :weighted, :mixed, :negative)
            case!(cases, "rotation_averaging/six_frames/$key/seed$seed", getfield(r, key), d -> spectral_rotation_averaging(d))
        end
        case!(cases, "rotation_averaging/six_frames/refused_weights_dict/seed$seed", r.bad,
            d -> spectral_rotation_averaging(d, Dict(p => 1.0 for p in keys(d))))
    end
    case!(cases, "rotation_averaging/empty/refused", Dict{Tuple{Int, Int}, RotMatrix{3, Float64, 9}}(), d -> spectral_rotation_averaging(d))
    # --- zhang_intrinsics ---------------------------------------------------------------------------------------
    z = zhang_fixtures()
    c = ZHANG_CENTER
    case!(cases, "zhang/six_views/free/plain", (z.Hs, c), ((Hs, c),) -> zhang_intrinsics(Hs; center = c))
    case!(cases, "zhang/six_views/free/no_center", z.Hs, Hs -> zhang_intrinsics(Hs))
    case!(cases, "zhang/six_views/free/voted_junk", (z.voted, c), ((Hs, c),) -> zhang_intrinsics(Hs; center = c))
    case!(cases, "zhang/six_views/free/mixed_bare_attributed", (z.mixed, c), ((Hs, c),) -> zhang_intrinsics(Hs; center = c))
    case!(cases, "zhang/six_views/pitch_1um/voted", (z.voted[1:(end - 1)], z.pitch1, c), ((Hs, p, c),) -> zhang_intrinsics(Hs, p; center = c))
    case!(cases, "zhang/six_views_aspect/pitch_3.9_4.0/plain", (z.Hs_τ, z.pitch, c), ((Hs, p, c),) -> zhang_intrinsics(Hs, p; center = c))
    case!(cases, "zhang/fronto_parallel/free/refused", (z.flat, c), ((Hs, c),) -> zhang_intrinsics(Hs; center = c))
    case!(cases, "zhang/fronto_parallel/pitch_3.9_4.0/plain", (z.flat, z.pitch, c), ((Hs, p, c),) -> zhang_intrinsics(Hs, p; center = c))
    case!(cases, "zhang/empty/free/refused", (HomographyMat{Float64}[], c), ((Hs, c),) -> zhang_intrinsics(Hs; center = c))
    case!(cases, "zhang/six_views/free/refused_weights_keyword", (z.Hs, c), ((Hs, c),) -> zhang_intrinsics(Hs; center = c, weights = ones(length(Hs))))
    case!(cases, "zhang/six_views/free/negative_weight", (z.negative, c), ((Hs, c),) -> zhang_intrinsics(Hs; center = c))
    case!(cases, "zhang/six_views/free/infinite_weight", (z.infinite, c), ((Hs, c),) -> zhang_intrinsics(Hs; center = c))
    case!(cases, "zhang/six_views/pitch_length_refused", (z.Hs, Size2(width = 1.0px, height = 1.0px), c),
        ((Hs, p, c),) -> zhang_intrinsics(Hs, p; center = c))
    # --- tls_line -----------------------------------------------------------------------------------------------
    pts = tls_exact_points()
    case!(cases, "tls_line/exact_12/plain", pts, p -> tls_line(p))
    case!(cases, "tls_line/exact_2/too_few", pts[1:2], p -> tls_line(p))
    case!(cases, "tls_line/exact_12/unit_weights", (pts, ones(12)), ((p, w),) -> tls_line(p, w))
    for seed in SEEDS
        p200 = tls_noisy_points(StableRNG(seed), 200)
        case!(cases, "tls_line/noisy_200/plain/seed$seed", p200, p -> tls_line(p))
        p40 = tls_noisy_points(StableRNG(seed), 40)
        bad = copy(p40); bad[7] = Point2(100.0, -300.0)
        w = ones(40); w[7] = 0.0
        w2 = [isodd(i) ? 2.0 : 1.0 for i in 1:40]
        case!(cases, "tls_line/noisy_40/plain/seed$seed", p40, p -> tls_line(p))
        case!(cases, "tls_line/noisy_40/unit_weights/seed$seed", (p40, ones(40)), ((p, w),) -> tls_line(p, w))
        case!(cases, "tls_line/noisy_40_outlier/zero_weight/seed$seed", (bad, w), ((p, w),) -> tls_line(p, w))
        case!(cases, "tls_line/noisy_39/outlier_removed/seed$seed", deleteat!(copy(bad), 7), p -> tls_line(p))
        case!(cases, "tls_line/noisy_40/weights_2_1/seed$seed", (p40, w2), ((p, w),) -> tls_line(p, w))
        case!(cases, "tls_line/noisy_60/repeats/seed$seed", vcat(p40, p40[1:2:end]), p -> tls_line(p))
        case!(cases, "tls_line/noisy_40/two_positive_weights/seed$seed", (p40, [1.0, 1.0, zeros(38)...]), ((p, w),) -> tls_line(p, w))
        case!(cases, "tls_line/noisy_40/negative_weights/seed$seed", (p40, -ones(40)), ((p, w),) -> tls_line(p, w))
        case!(cases, "tls_line/noisy_40/short_weights/seed$seed", (p40, ones(39)), ((p, w),) -> tls_line(p, w))
        case!(cases, "tls_line/noisy_40/infinite_weight/seed$seed", (p40, [Inf, ones(39)...]), ((p, w),) -> tls_line(p, w))
        case!(cases, "tls_line/noisy_40/zero_weights/seed$seed", (p40, zeros(40)), ((p, w),) -> tls_line(p, w))
        # the statistics form: the line, S and JᵀWJ; with a given normal only the offset is estimated
        n0 = SVector(-2.0, 1.0) / sqrt(5)
        case!(cases, "tls_line_statistics/noisy_40/weights_2_1/seed$seed", (p40, w2), ((p, w),) -> tls_line_statistics(p, w))
        case!(cases, "tls_line_statistics/noisy_40/weights_2_1_given_normal/seed$seed", (p40, w2, n0), ((p, w, n),) -> tls_line_statistics(p, w, n))
        case!(cases, "tls_line_statistics/noisy_40/nan_weight/seed$seed", (p40, [NaN, ones(39)...]), ((p, w),) -> tls_line_statistics(p, w))
        case!(cases, "tls_line_statistics/noisy_40/zero_weights/seed$seed", (p40, zeros(40)), ((p, w),) -> tls_line_statistics(p, w))
        case!(cases, "tls_line_statistics/noisy_40/zero_weights_given_normal/seed$seed", (p40, zeros(40), n0), ((p, w, n),) -> tls_line_statistics(p, w, n))
    end
    # --- direct_ellipse -----------------------------------------------------------------------------------------
    e0 = ELLIPSE_E0
    case!(cases, "direct_ellipse/exact_40/plain", ellipse_boundary_points(StableRNG(1), e0, 40), p -> direct_ellipse(p))
    case!(cases, "direct_ellipse/exact_5/minimum", ellipse_boundary_points(StableRNG(1), e0, 5), p -> direct_ellipse(p))
    case!(cases, "direct_ellipse/quarter_arc_30/plain", ellipse_boundary_points(StableRNG(1), e0, 30; arc = π / 2), p -> direct_ellipse(p))
    case!(cases, "direct_ellipse/circle_20/plain", ellipse_boundary_points(StableRNG(1), Ellipse(Point2(-3.0, 7.0), 4.0, 4.0, 0.0), 20), p -> direct_ellipse(p))
    case!(cases, "direct_ellipse/four_points/too_few", [SVector(Float64(i), 0.0) for i in 1:4], p -> direct_ellipse(p))
    case!(cases, "direct_ellipse/collinear_10/none", [SVector(Float64(i), 2.0i) for i in 1:10], p -> direct_ellipse(p))
    for seed in SEEDS
        case!(cases, "direct_ellipse/noisy_60/sigma0.5/seed$seed", ellipse_boundary_points(StableRNG(seed), e0, 60; σ = 0.5), p -> direct_ellipse(p))
    end
    # --- nearest_point ------------------------------------------------------------------------------------------
    case!(cases, "nearest_point/two_lines/correspondences", nearest_point_two(), l -> nearest_point(l))
    case!(cases, "nearest_point/three_2d_lines/pairs", nearest_point_2d(), l -> nearest_point(l))
    lw = nearest_point_weighted()
    lwc = correspondences(first.(lw), last.(lw))
    w = [1.0, 2.0, 0.5, 3.0]
    case!(cases, "nearest_point/four_lines/weights", withattrs(lwc; weights = w), l -> nearest_point(l))
    case!(cases, "nearest_point/four_lines/weights_doubled", withattrs(lwc; weights = 2w), l -> nearest_point(l))
    case!(cases, "nearest_point/four_lines/zero_weight", withattrs(lwc; weights = [1.0, 1.0, 1.0, 0.0]), l -> nearest_point(l))
    case!(cases, "nearest_point/three_lines/unweighted", lw[1:3], l -> nearest_point(l))
    case!(cases, "nearest_point/four_lines/one_positive_weight", withattrs(lwc; weights = [1.0, 0.0, 0.0, 0.0]), l -> nearest_point(l))
    case!(cases, "nearest_point/four_lines/negative_weight", withattrs(lwc; weights = [1.0, -1.0, 1.0, 1.0]), l -> nearest_point(l))
    for (key, l) in pairs(nearest_point_degenerate())
        case!(cases, "nearest_point/degenerate/$key", l, l -> nearest_point(l))
    end
    for seed in SEEDS
        s = nearest_point_scene(StableRNG(seed))
        case!(cases, "nearest_point/meet_5/pairs/seed$seed", s.meet, l -> nearest_point(l))
        case!(cases, "nearest_point/meet_5/correspondences/seed$seed", correspondences(first.(s.meet), last.(s.meet)), l -> nearest_point(l))
        case!(cases, "nearest_point/noisy_40/pairs/seed$seed", s.noisy, l -> nearest_point(l))
        case!(cases, "nearest_point/shifted_12/pairs/seed$seed", s.shifted, l -> nearest_point(l))
        for (k, t) in enumerate((SVector(1.0e3, -2.0e3, 5.0e2), SVector(-0.25, 0.5, 7.0)))
            case!(cases, "nearest_point/shifted_12/translated$k/seed$seed", [(first(l) + t) => last(l) for l in s.shifted], l -> nearest_point(l))
        end
        case!(cases, "nearest_point/millimetres_4/pairs/seed$seed", s.unitful, l -> nearest_point(l))
        case!(cases, "nearest_point/millimetres_4/stripped_mm/seed$seed", [ustrip.(mm, first(l)) => last(l) for l in s.unitful], l -> nearest_point(l))
        uw = [1.0, 0.5, 2.0, 0.0]
        case!(cases, "nearest_point/millimetres_4/weights/seed$seed",
            withattrs(correspondences(first.(s.unitful), last.(s.unitful)); weights = uw), l -> nearest_point(l))
    end
    # --- triangulate --------------------------------------------------------------------------------------------
    pose = triangulation_pose()
    for seed in SEEDS
        r = triangulation_rays(StableRNG(seed))
        case!(cases, "triangulate/front_rays/exact/seed$seed", (pose, r.exact), ((P, fs),) -> [triangulate(P, a, b) for (a, b) in fs])
        case!(cases, "triangulate/front_rays/noisy/seed$seed", (pose, r.noisy), ((P, fs),) -> [triangulate(P, a, b) for (a, b) in fs])
    end
    sp = triangulation_special()
    case!(cases, "triangulate/behind_view1/exact", (pose, sp.behind), ((P, (a, b)),) -> triangulate(P, a, b))
    case!(cases, "triangulate/behind_view1/opposite_ray", (pose, sp.behind), ((P, (a, b)),) -> triangulate(P, -a, b))
    case!(cases, "triangulate/zero_parallax/nan", sp.parallel, ((P, a, b),) -> triangulate(P, a, b))
    case!(cases, "triangulate/fisheye_100deg/exact", (pose, sp.wide), ((P, (a, b)),) -> triangulate(P, a, b))
    # --- p3p(model, cs) -----------------------------------------------------------------------------------------
    model, bounds = p3p_camera()
    for seed in SEEDS
        draws = p3p_draws(StableRNG(seed), bounds, 60)
        case!(cases, "p3p_camera/imx174_draws_60/structarray/seed$seed", (model, draws), ((m, ds),) -> [p3p(m, cs) for cs in ds])
        d1 = first(draws)
        case!(cases, "p3p_camera/imx174_draw1/correspondences/seed$seed", (model, Correspondences(collect(d1.first), collect(d1.second))),
            ((m, cs),) -> p3p(m, cs))
        case!(cases, "p3p_camera/imx174_draw1/pairs/seed$seed", (model, collect(d1)), ((m, cs),) -> p3p(m, cs))
        case!(cases, "p3p_camera/imx174_two/too_few/seed$seed", (model, d1[1:2]), ((m, cs),) -> p3p(m, cs))
        case!(cases, "p3p_camera/imx174_four/too_many/seed$seed", (model, vcat(collect(d1), collect(draws[2])[1:1])), ((m, cs),) -> p3p(m, cs))
    end
    return cases
end
