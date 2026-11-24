# Core's closed forms on Core 0.14: every old estimator function is `solve(EstimationProblem(M, data[, p]), Algorithm())`
# (spec §3): fundamental_dlt → DLT(), essential_dlt → DLT(PoseLibSolver()), kabsch → Kabsch(), chordal_mean →
# ChordalMean(), spectral_rotation_averaging → SpectralAveraging(), zhang_intrinsics → Zhang() with `p = (; center[,
# pitch])`, tls_line → TotalLeastSquares() (weights the points' `weights` column), direct_ellipse → DirectEllipse(),
# nearest_point → NearestPoint(), triangulate → Triangulation() (`p` the pose), p3p(model, cs) →
# MinimalProblem(Extrinsics3, cs, idx, model) → P3P(). A mask is data selection (`cs[mask]`, spec §9); `svd_ws` left the
# public form. The inner-loop forms (`fundamental_dlt!`, `tls_line_statistics`) are unchanged.
using VisualGeometryToolkitCore: EstimationProblem, MinimalProblem, solve, DLT, Kabsch, ChordalMean, SpectralAveraging,
    AbsoluteRotations, Zhang, TotalLeastSquares, DirectEllipse, NearestPoint, Triangulation, P3P, PoseLibSolver,
    fundamental_dlt!, tls_line_statistics, CorrespondenceNormalization, FundMat, EssentialMat, Line2D, Point, Point3,
    Extrinsics3
using Rotations: RotMatrix

# PoseLib implements Core's `essential_8pt` (DLT(PoseLibSolver())'s kernel); loaded when it loads.
const _CF_POSELIB = try
    Base.require(Base.PkgId(Base.UUID("fe2c2836-a0c1-4fd4-bc39-a8c8a2bbd5e8"), "PoseLib")); true
catch err
    err isa InterruptException && rethrow()
    false
end

# A closed form that returned a bare model (or `nothing`): `u` is it, every other field new.
const _CF_NEW = ["", "resid", "retcode", "stats", "prob", "alg", "original", "stop_rule", "rms", "n", "metadata"]
const _CF_BARE = FieldMap(; renames = ["u" => ""], new_only = _CF_NEW)
# A refusal whose message named the old function or its keywords: the type compared, the message not (spec §9: the
# errors name the algorithm now).
const _CF_REFUSAL = FieldMap(; ignored = ["exception.msg"])
# Triangulation returned `(X, λ₁, λ₂)`: the point is `u`, the depths `metadata.depths`.
const _CF_TRI = FieldMap(; renames = ["u" => "1", "metadata.depths.1" => "2", "metadata.depths.2" => "3"],
    new_only = ["resid", "retcode", "stats", "prob", "alg", "original", "stop_rule", "rms", "n", "metadata"], retyped = [""])
# A vector of triangulations, as the old vector of tuples (an array's digest: its old leaves, rebuilt in their order).
_cf_tri(P, (a, b)) = (r = solve(EstimationProblem(Point3, a => b, P), Triangulation()); (r.u, r.metadata.depths...))

_cf_dlt(cs) = solve(EstimationProblem(FundMat, cs), DLT())
_cf_edlt(cs) = solve(EstimationProblem(EssentialMat, cs), DLT(PoseLibSolver()))
_cf_kabsch(cs) = solve(EstimationProblem(RotMatrix{3}, cs), Kabsch())
_cf_chordal(Rs) = solve(EstimationProblem(RotMatrix{3}, Rs), ChordalMean())
_cf_spectral(d) = solve(EstimationProblem(AbsoluteRotations, d), SpectralAveraging())
_cf_zhang(Hs, p) = solve(EstimationProblem(VgtkCore.LogicalIntrinsics, Hs, p), Zhang())
_cf_tls(p) = solve(EstimationProblem(Line2D, p), TotalLeastSquares())
_cf_tls(p, w) = _cf_tls(withattrs(p; weights = w))
_cf_ellipse(p) = solve(EstimationProblem(VgtkCore.Ellipse, p), DirectEllipse())
_cf_nearest(l) = solve(EstimationProblem(Point, l), NearestPoint())
_cf_p3p(m, cs) = solve(MinimalProblem(Extrinsics3, cs, collect(1:length(cs)), m), P3P())

function replay_core_closed_forms(cases)
    B, R = _CF_BARE, _CF_REFUSAL
    # --- fundamental_dlt → DLT() --------------------------------------------------------------------------------
    for seed in SEEDS
        fcs = fundmat_guard_pairs(StableRNG(seed))
        replay!(cases, "fundmat_dlt/guard_8/plain/seed$seed", fcs[1:8], _cf_dlt; map = B)
        replay!(cases, "fundmat_dlt/guard_8/svd_ws/seed$seed", fcs[1:8], _cf_dlt; map = B)
        replay!(cases, "fundmat_dlt/guard_9/plain/seed$seed", fcs, _cf_dlt; map = B)
        # an index list belongs to the inner kernel only (owner, 2026-11-24): one longer than the data is refused there
        replay!(cases, "fundmat_dlt/guard_8/refused_long_index_mask/seed$seed", fcs[1:8],
            cs -> (m = [1:8; 1:3]; fundamental_dlt!(Matrix{Float64}(undef, length(cs), 9), CorrespondenceNormalization(FundMat, cs[m]), cs; mask = m)); map = B)
        replay!(cases, "fundmat_dlt/guard_7/too_few/seed$seed", fcs[1:7], _cf_dlt; map = B)
        sc = epipolar_layer_scene(StableRNG(seed))
        exact = Correspondences(sc.u1, sc.u2)
        noisy = Correspondences(sc.u1, sc.u2n)
        w = [i == 1 ? 3.0 : 1.0 for i in 1:20]
        keep = BitVector(i > 3 for i in 1:20)
        replay!(cases, "fundmat_dlt/layer_20/exact/seed$seed", exact, _cf_dlt; map = B)
        replay!(cases, "fundmat_dlt/layer_20/noisy/seed$seed", noisy, _cf_dlt; map = B)
        replay!(cases, "fundmat_dlt/layer_20/noisy_pair_vector/seed$seed", collect(noisy), _cf_dlt; map = B)
        replay!(cases, "fundmat_dlt/layer_20/noisy_weights_3_1/seed$seed", withattrs(noisy; weights = w), _cf_dlt; map = B)
        replay!(cases, "fundmat_dlt/layer_20/noisy_repeated_first/seed$seed",
            Correspondences([fill(sc.u1[1], 2); sc.u1], [fill(sc.u2n[1], 2); sc.u2n]), _cf_dlt; map = B)
        replay!(cases, "fundmat_dlt/layer_20/noisy_zero_weights/seed$seed", withattrs(noisy; weights = Float64.(keep)), _cf_dlt; map = B)
        replay!(cases, "fundmat_dlt/layer_20/noisy_bool_mask/seed$seed", (noisy, keep), ((cs, m),) -> _cf_dlt(cs[m]); map = B)
        replay!(cases, "fundmat_dlt/layer_20/noisy_index_mask/seed$seed", (noisy, findall(keep)), ((cs, m),) -> _cf_dlt(cs[m]); map = B)
        replay!(cases, "fundmat_dlt/layer_20/refused_weights_keyword/seed$seed", (noisy, w),
            ((cs, w),) -> solve(EstimationProblem(FundMat, cs), DLT(); weights = w); map = R)
        replay!(cases, "fundmat_dlt/layer_20/refused_parallel_arrays/seed$seed", (sc.u1, sc.u2n),
            ((a, b),) -> solve(EstimationProblem(FundMat, a, b), DLT()); map = R)
        nonfinite = Correspondences([[Point2{Float64}(NaN, NaN)]; sc.u1[2:9]], sc.u2[1:9])
        replay!(cases, "fundmat_dlt/layer_9/nonfinite_row/seed$seed", nonfinite, _cf_dlt; map = B)
        F = fundmat_scene_pairs(StableRNG(seed))
        replay!(cases, "fundmat_dlt/fundmat_scene_130/all/seed$seed", F, _cf_dlt; map = B)
        replay!(cases, "fundmat_dlt/fundmat_scene_130/inliers_mask/seed$seed", (F, collect(1:100)), ((cs, m),) -> _cf_dlt(cs[m]); map = B)
        replay!(cases, "fundmat_dlt/fundmat_scene_130/svd_ws/seed$seed", F, _cf_dlt; map = B)
        # the inner-loop form, unchanged
        replay!(cases, "fundmat_dlt_inplace/fundmat_scene_130/all/seed$seed", F,
            cs -> fundamental_dlt!(Matrix{Float64}(undef, length(cs), 9), CorrespondenceNormalization(FundMat, cs), cs))
        replay!(cases, "fundmat_dlt_inplace/fundmat_scene_130/inliers_mask_svd_ws/seed$seed", (F, BitVector(i <= 100 for i in 1:length(F))),
            ((cs, m),) -> fundamental_dlt!(Matrix{Float64}(undef, length(cs), 9), CorrespondenceNormalization(FundMat, cs[m]), cs;
                mask = m, svd_ws = VgtkCore.SvdWorkspace{Float64}(length(cs), 9)))
        replay!(cases, "fundmat_dlt_inplace/layer_20/noisy_weights_3_1/seed$seed", withattrs(noisy; weights = w),
            cs -> fundamental_dlt!(Matrix{Float64}(undef, length(cs), 9), CorrespondenceNormalization(FundMat, cs), cs))
    end
    # --- essential_dlt → DLT(PoseLibSolver()) (its only backend; the bare form routed to it) ------------------------
    for seed in SEEDS
        e = essential_scene(StableRNG(seed))
        replay!(cases, "essential_dlt/refine_scene_20/bare/seed$seed", Correspondences(e.b1[1:20], e.b2[1:20]), _cf_edlt; map = B)
        replay!(cases, "essential_dlt/refine_scene_20/poselib_pairs/seed$seed", e.b1[1:20] .=> e.b2[1:20], _cf_edlt; map = B)
        replay!(cases, "essential_dlt/refine_scene_200/exact/seed$seed", Correspondences(e.b1, e.b2), _cf_edlt; map = B)
        replay!(cases, "essential_dlt/refine_scene_200/noisy/seed$seed", Correspondences(e.b1, e.b2n), _cf_edlt; map = B)
        replay!(cases, "essential_dlt/refine_scene_8/minimum/seed$seed", Correspondences(e.b1[1:8], e.b2n[1:8]), _cf_edlt; map = B)
        replay!(cases, "essential_dlt/refine_scene_7/too_few/seed$seed", Correspondences(e.b1[1:7], e.b2[1:7]), _cf_edlt; map = R)
        replay!(cases, "essential_dlt/refine_scene_20/refused_chart/seed$seed", Correspondences(e.c1[1:20], e.c2[1:20]), _cf_edlt; map = R)
    end
    # --- kabsch → Kabsch() --------------------------------------------------------------------------------------
    for seed in SEEDS
        k = kabsch_scene(StableRNG(seed))
        o = SVector(3.0, -1.0, 2.0)
        replay!(cases, "kabsch/fits_40/exact_pairs/seed$seed", k.as .=> k.bs, _cf_kabsch; map = B)
        replay!(cases, "kabsch/fits_40/exact_correspondences/seed$seed", Correspondences(k.as, k.bs), _cf_kabsch; map = B)
        replay!(cases, "kabsch/fits_40/noisy/seed$seed", k.as .=> k.bsn, _cf_kabsch; map = B)
        replay!(cases, "kabsch/fits_40/offset/seed$seed", (k.as .+ Ref(o)) .=> (k.bs .+ Ref(o)), _cf_kabsch; map = B)
        replay!(cases, "kabsch/fits_1/too_few/seed$seed", k.as[1:1] .=> k.bs[1:1], _cf_kabsch; map = B)
        w1 = zeros(40); w1[7] = 1.0
        w2 = zeros(40); w2[7] = 1.0; w2[11] = 2.0
        replay!(cases, "kabsch/fits_40/one_positive_weight/seed$seed", withattrs(Correspondences(k.as, k.bs); weights = w1), _cf_kabsch; map = B)
        replay!(cases, "kabsch/fits_40/two_positive_weights/seed$seed", withattrs(Correspondences(k.as, k.bs); weights = w2), _cf_kabsch; map = B)
        replay!(cases, "kabsch/fits_40/refused_parallel_arrays/seed$seed", (k.as, k.bs),
            ((a, b),) -> solve(EstimationProblem(RotMatrix{3}, a, b), Kabsch()); map = R)
        kw = kabsch_weighted_scene(StableRNG(seed))
        cs = Correspondences(kw.as, kw.bs)
        replay!(cases, "kabsch/weighted_30/weights/seed$seed", withattrs(cs; weights = kw.w), _cf_kabsch; map = B)
        replay!(cases, "kabsch/weighted_30/unit_weights/seed$seed", withattrs(cs; weights = ones(30)), _cf_kabsch; map = B)
        replay!(cases, "kabsch/weighted_30/unweighted/seed$seed", cs, _cf_kabsch; map = B)
        bad = withattrs(Correspondences(vcat(kw.as, [SVector(1.0, 0.0, 0.0)]), vcat(kw.bs, [SVector(0.0, 0.0, -1.0)])); weights = vcat(ones(30), 0.0))
        replay!(cases, "kabsch/weighted_31/zero_weight_outlier/seed$seed", bad, _cf_kabsch; map = B)
        replay!(cases, "kabsch/weighted_31/unit_weight_outlier/seed$seed", withattrs(bad; weights = ones(31)), _cf_kabsch; map = B)
        replay!(cases, "kabsch/weighted_30/negative_weight/seed$seed", withattrs(cs; weights = vcat(-1.0, ones(29))), _cf_kabsch; map = B)
        replay!(cases, "kabsch/weighted_30/nan_weight/seed$seed", withattrs(cs; weights = vcat(NaN, ones(29))), _cf_kabsch; map = B)
        csm = Correspondences([a .* mm for a in kw.as], [b .* mm for b in kw.bs])
        replay!(cases, "kabsch/weighted_30/millimetres_weights/seed$seed", withattrs(csm; weights = kw.w), _cf_kabsch; map = B)
        replay!(cases, "kabsch/weighted_30/millimetres/seed$seed", csm, _cf_kabsch; map = B)
    end
    # --- chordal_mean → ChordalMean() ---------------------------------------------------------------------------
    for seed in SEEDS
        c = chordal_scene(StableRNG(seed))
        replay!(cases, "chordal_mean/symmetric_4/rotations/seed$seed", c.Rs, _cf_chordal; map = B)
        replay!(cases, "chordal_mean/symmetric_4/matrices/seed$seed", SMatrix{3, 3, Float64}.(c.Rs), _cf_chordal; map = B)
        replay!(cases, "chordal_mean/single/rotation/seed$seed", [c.Rc], _cf_chordal; map = B)
        replay!(cases, "chordal_mean/noisy_5/rotations/seed$seed", c.noisy, _cf_chordal; map = B)
    end
    replay!(cases, "chordal_mean/empty/refused", RotMatrix{3, Float64}[], _cf_chordal; map = B)
    # --- spectral_rotation_averaging → SpectralAveraging() ------------------------------------------------------
    for seed in SEEDS
        r = rotation_averaging_scene(StableRNG(seed))
        for key in (:pairs, :chain, :noisy, :bad, :weighted, :mixed, :negative)
            replay!(cases, "rotation_averaging/six_frames/$key/seed$seed", getfield(r, key), _cf_spectral; map = key === :negative ? R : B)
        end
        replay!(cases, "rotation_averaging/six_frames/refused_weights_dict/seed$seed", r.bad,
            d -> solve(EstimationProblem(AbsoluteRotations, d, Dict(p => 1.0 for p in keys(d))), SpectralAveraging()); map = R)
    end
    replay!(cases, "rotation_averaging/empty/refused", Dict{Tuple{Int, Int}, RotMatrix{3, Float64, 9}}(), _cf_spectral; map = R)
    # --- zhang_intrinsics → Zhang(), the centre (and the pitch) the context ---------------------------------------
    z = zhang_fixtures()
    c = ZHANG_CENTER
    replay!(cases, "zhang/six_views/free/plain", (z.Hs, c), ((Hs, c),) -> _cf_zhang(Hs, (; center = c)); map = B)
    replay!(cases, "zhang/six_views/free/no_center", z.Hs, Hs -> _cf_zhang(Hs, VgtkCore.NullParameters()); map = R)
    replay!(cases, "zhang/six_views/free/voted_junk", (z.voted, c), ((Hs, c),) -> _cf_zhang(Hs, (; center = c)); map = B)
    replay!(cases, "zhang/six_views/free/mixed_bare_attributed", (z.mixed, c), ((Hs, c),) -> _cf_zhang(Hs, (; center = c)); map = B)
    replay!(cases, "zhang/six_views/pitch_1um/voted", (z.voted[1:(end - 1)], z.pitch1, c), ((Hs, p, c),) -> _cf_zhang(Hs, (; center = c, pitch = p)); map = B)
    replay!(cases, "zhang/six_views_aspect/pitch_3.9_4.0/plain", (z.Hs_τ, z.pitch, c), ((Hs, p, c),) -> _cf_zhang(Hs, (; center = c, pitch = p)); map = B)
    replay!(cases, "zhang/fronto_parallel/free/refused", (z.flat, c), ((Hs, c),) -> _cf_zhang(Hs, (; center = c)); map = B)
    replay!(cases, "zhang/fronto_parallel/pitch_3.9_4.0/plain", (z.flat, z.pitch, c), ((Hs, p, c),) -> _cf_zhang(Hs, (; center = c, pitch = p)); map = B)
    replay!(cases, "zhang/empty/free/refused", (HomographyMat{Float64}[], c), ((Hs, c),) -> _cf_zhang(Hs, (; center = c)); map = R)
    replay!(cases, "zhang/six_views/free/refused_weights_keyword", (z.Hs, c),
        ((Hs, c),) -> solve(EstimationProblem(VgtkCore.LogicalIntrinsics, Hs, (; center = c)), Zhang(); weights = ones(length(Hs))); map = R)
    replay!(cases, "zhang/six_views/free/negative_weight", (z.negative, c), ((Hs, c),) -> _cf_zhang(Hs, (; center = c)); map = B)
    replay!(cases, "zhang/six_views/free/infinite_weight", (z.infinite, c), ((Hs, c),) -> _cf_zhang(Hs, (; center = c)); map = B)
    replay!(cases, "zhang/six_views/pitch_length_refused", (z.Hs, Size2(width = 1.0px, height = 1.0px), c),
        ((Hs, p, c),) -> _cf_zhang(Hs, (; center = c, pitch = p)); map = B)
    # --- tls_line → TotalLeastSquares() (weights: the points' column) ---------------------------------------------
    pts = tls_exact_points()
    replay!(cases, "tls_line/exact_12/plain", pts, _cf_tls; map = B)
    replay!(cases, "tls_line/exact_2/too_few", pts[1:2], _cf_tls; map = R)
    replay!(cases, "tls_line/exact_12/unit_weights", (pts, ones(12)), ((p, w),) -> _cf_tls(p, w); map = B)
    for seed in SEEDS
        p200 = tls_noisy_points(StableRNG(seed), 200)
        replay!(cases, "tls_line/noisy_200/plain/seed$seed", p200, _cf_tls; map = B)
        p40 = tls_noisy_points(StableRNG(seed), 40)
        bad = copy(p40); bad[7] = Point2(100.0, -300.0)
        w = ones(40); w[7] = 0.0
        w2 = [isodd(i) ? 2.0 : 1.0 for i in 1:40]
        tw = ((p, w),) -> _cf_tls(p, w)
        replay!(cases, "tls_line/noisy_40/plain/seed$seed", p40, _cf_tls; map = B)
        replay!(cases, "tls_line/noisy_40/unit_weights/seed$seed", (p40, ones(40)), tw; map = B)
        replay!(cases, "tls_line/noisy_40_outlier/zero_weight/seed$seed", (bad, w), tw; map = B)
        replay!(cases, "tls_line/noisy_39/outlier_removed/seed$seed", deleteat!(copy(bad), 7), _cf_tls; map = B)
        replay!(cases, "tls_line/noisy_40/weights_2_1/seed$seed", (p40, w2), tw; map = B)
        replay!(cases, "tls_line/noisy_60/repeats/seed$seed", vcat(p40, p40[1:2:end]), _cf_tls; map = B)
        replay!(cases, "tls_line/noisy_40/two_positive_weights/seed$seed", (p40, [1.0, 1.0, zeros(38)...]), tw; map = R)
        replay!(cases, "tls_line/noisy_40/negative_weights/seed$seed", (p40, -ones(40)), tw; map = B)
        replay!(cases, "tls_line/noisy_40/short_weights/seed$seed", (p40, ones(39)), tw; map = R)   # the message named the old weights argument; the weights are a column now
        replay!(cases, "tls_line/noisy_40/infinite_weight/seed$seed", (p40, [Inf, ones(39)...]), tw; map = B)
        replay!(cases, "tls_line/noisy_40/zero_weights/seed$seed", (p40, zeros(40)), tw; map = R)
        n0 = SVector(-2.0, 1.0) / sqrt(5)
        replay!(cases, "tls_line_statistics/noisy_40/weights_2_1/seed$seed", (p40, w2), ((p, w),) -> tls_line_statistics(p, w))
        replay!(cases, "tls_line_statistics/noisy_40/weights_2_1_given_normal/seed$seed", (p40, w2, n0), ((p, w, n),) -> tls_line_statistics(p, w, n))
        replay!(cases, "tls_line_statistics/noisy_40/nan_weight/seed$seed", (p40, [NaN, ones(39)...]), ((p, w),) -> tls_line_statistics(p, w))
        replay!(cases, "tls_line_statistics/noisy_40/zero_weights/seed$seed", (p40, zeros(40)), ((p, w),) -> tls_line_statistics(p, w))
        replay!(cases, "tls_line_statistics/noisy_40/zero_weights_given_normal/seed$seed", (p40, zeros(40), n0), ((p, w, n),) -> tls_line_statistics(p, w, n))
    end
    # --- direct_ellipse → DirectEllipse() -----------------------------------------------------------------------
    e0 = ELLIPSE_E0
    replay!(cases, "direct_ellipse/exact_40/plain", ellipse_boundary_points(StableRNG(1), e0, 40), _cf_ellipse; map = B)
    replay!(cases, "direct_ellipse/exact_5/minimum", ellipse_boundary_points(StableRNG(1), e0, 5), _cf_ellipse; map = B)
    replay!(cases, "direct_ellipse/quarter_arc_30/plain", ellipse_boundary_points(StableRNG(1), e0, 30; arc = π / 2), _cf_ellipse; map = B)
    replay!(cases, "direct_ellipse/circle_20/plain", ellipse_boundary_points(StableRNG(1), Ellipse(Point2(-3.0, 7.0), 4.0, 4.0, 0.0), 20), _cf_ellipse; map = B)
    replay!(cases, "direct_ellipse/four_points/too_few", [SVector(Float64(i), 0.0) for i in 1:4], _cf_ellipse; map = R)
    replay!(cases, "direct_ellipse/collinear_10/none", [SVector(Float64(i), 2.0i) for i in 1:10], _cf_ellipse; map = B)
    for seed in SEEDS
        replay!(cases, "direct_ellipse/noisy_60/sigma0.5/seed$seed", ellipse_boundary_points(StableRNG(seed), e0, 60; σ = 0.5), _cf_ellipse; map = B)
    end
    # --- nearest_point → NearestPoint() -------------------------------------------------------------------------
    replay!(cases, "nearest_point/two_lines/correspondences", nearest_point_two(), _cf_nearest; map = B)
    replay!(cases, "nearest_point/three_2d_lines/pairs", nearest_point_2d(), _cf_nearest; map = B)
    lw = nearest_point_weighted()
    lwc = correspondences(first.(lw), last.(lw))
    w = [1.0, 2.0, 0.5, 3.0]
    replay!(cases, "nearest_point/four_lines/weights", withattrs(lwc; weights = w), _cf_nearest; map = B)
    replay!(cases, "nearest_point/four_lines/weights_doubled", withattrs(lwc; weights = 2w), _cf_nearest; map = B)
    replay!(cases, "nearest_point/four_lines/zero_weight", withattrs(lwc; weights = [1.0, 1.0, 1.0, 0.0]), _cf_nearest; map = B)
    replay!(cases, "nearest_point/three_lines/unweighted", lw[1:3], _cf_nearest; map = B)
    replay!(cases, "nearest_point/four_lines/one_positive_weight", withattrs(lwc; weights = [1.0, 0.0, 0.0, 0.0]), _cf_nearest; map = R)
    replay!(cases, "nearest_point/four_lines/negative_weight", withattrs(lwc; weights = [1.0, -1.0, 1.0, 1.0]), _cf_nearest; map = B)
    for (key, l) in pairs(nearest_point_degenerate())
        replay!(cases, "nearest_point/degenerate/$key", l, _cf_nearest; map = R)
    end
    for seed in SEEDS
        s = nearest_point_scene(StableRNG(seed))
        replay!(cases, "nearest_point/meet_5/pairs/seed$seed", s.meet, _cf_nearest; map = B)
        replay!(cases, "nearest_point/meet_5/correspondences/seed$seed", correspondences(first.(s.meet), last.(s.meet)), _cf_nearest; map = B)
        replay!(cases, "nearest_point/noisy_40/pairs/seed$seed", s.noisy, _cf_nearest; map = B)
        replay!(cases, "nearest_point/shifted_12/pairs/seed$seed", s.shifted, _cf_nearest; map = B)
        for (k, t) in enumerate((SVector(1.0e3, -2.0e3, 5.0e2), SVector(-0.25, 0.5, 7.0)))
            replay!(cases, "nearest_point/shifted_12/translated$k/seed$seed", [(first(l) + t) => last(l) for l in s.shifted], _cf_nearest; map = B)
        end
        replay!(cases, "nearest_point/millimetres_4/pairs/seed$seed", s.unitful, _cf_nearest; map = B)
        replay!(cases, "nearest_point/millimetres_4/stripped_mm/seed$seed", [ustrip.(mm, first(l)) => last(l) for l in s.unitful], _cf_nearest; map = B)
        uw = [1.0, 0.5, 2.0, 0.0]
        replay!(cases, "nearest_point/millimetres_4/weights/seed$seed",
            withattrs(correspondences(first.(s.unitful), last.(s.unitful)); weights = uw), _cf_nearest; map = B)
    end
    # --- triangulate → Triangulation(), the pose the context ----------------------------------------------------
    pose = triangulation_pose()
    for seed in SEEDS
        r = triangulation_rays(StableRNG(seed))
        replay!(cases, "triangulate/front_rays/exact/seed$seed", (pose, r.exact), ((P, fs),) -> [_cf_tri(P, f) for f in fs])
        replay!(cases, "triangulate/front_rays/noisy/seed$seed", (pose, r.noisy), ((P, fs),) -> [_cf_tri(P, f) for f in fs])
    end
    sp = triangulation_special()
    tri(P, a, b) = solve(EstimationProblem(Point3, a => b, P), Triangulation())
    replay!(cases, "triangulate/behind_view1/exact", (pose, sp.behind), ((P, (a, b)),) -> tri(P, a, b); map = _CF_TRI)
    replay!(cases, "triangulate/behind_view1/opposite_ray", (pose, sp.behind), ((P, (a, b)),) -> tri(P, -a, b); map = _CF_TRI)
    replay!(cases, "triangulate/zero_parallax/nan", sp.parallel, ((P, a, b),) -> tri(P, a, b); map = _CF_TRI)
    replay!(cases, "triangulate/fisheye_100deg/exact", (pose, sp.wide), ((P, (a, b)),) -> tri(P, a, b); map = _CF_TRI)
    # --- p3p(model, cs) → MinimalProblem(Extrinsics3, cs, idx, model) → P3P() (every root: FixedModels, as before) --
    model, bounds = p3p_camera()
    for seed in SEEDS
        draws = p3p_draws(StableRNG(seed), bounds, 60)
        replay!(cases, "p3p_camera/imx174_draws_60/structarray/seed$seed", (model, draws), ((m, ds),) -> [_cf_p3p(m, cs) for cs in ds])
        d1 = first(draws)
        replay!(cases, "p3p_camera/imx174_draw1/correspondences/seed$seed", (model, Correspondences(collect(d1.first), collect(d1.second))),
            ((m, cs),) -> _cf_p3p(m, cs))
        replay!(cases, "p3p_camera/imx174_draw1/pairs/seed$seed", (model, collect(d1)), ((m, cs),) -> _cf_p3p(m, cs))
        replay!(cases, "p3p_camera/imx174_two/too_few/seed$seed", (model, d1[1:2]), ((m, cs),) -> _cf_p3p(m, cs))
        replay!(cases, "p3p_camera/imx174_four/too_many/seed$seed", (model, vcat(collect(d1), collect(draws[2])[1:1])), ((m, cs),) -> _cf_p3p(m, cs))
    end
    return cases
end
