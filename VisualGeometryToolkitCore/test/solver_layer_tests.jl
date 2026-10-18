# Solver/correspondence layer (#21): Csponds container + sources/targets
# protocol, lazy lifts, CspondNormalization push/pull, the MinimalSolver
# protocol (solve → FixedModels), the DLT kernels on correspondences,
# induced-model registry.

@testitem "solver_layer" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore: max_solutions, induced_models, minimal_solver
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CspondNormalization, planar_lift, source_map, target_map, sample_size   # public, not exported
    using VisualGeometryToolkitCore: MappedColumn, lift
    using StaticArrays, LinearAlgebra, Random
    using Rotations: RotMatrix
    using StructArrays: StructArray

    # Apply a 3×3 homography matrix to a 2D point.
    _apply_h(H, p) = Point2{Float64}((PerspectiveMap() ∘ HomographyMap(H))(p))

    _rand_rot(rng) = SMatrix{3, 3, Float64}(rand(rng, RotMatrix{3, Float64}))   # Haar-uniform

    @testset "solver layer (#21)" begin
        rng = MersenneTwister(2026)

        # Shared synthetic homography scene
        H0m = SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, -2.0e-4, 4.0, -3.0, 1.0)
        H0 = HomographyMat{Float64}(Tuple(H0m))
        src = [Point2{Float64}(100rand(rng), 100rand(rng)) for _ in 1:20]
        dst = [_apply_h(H0, p) for p in src]

        @testset "Csponds container" begin
            scores = collect(1.0:20.0)
            ratios = fill(0.8f0, 20)
            cs = Csponds(src, dst; score = scores, ratio = ratios)

            @test cs isa AbstractVector{<:Cspond}
            @test length(cs) == 20
            @test cs[3] === (src[3] => dst[3])
            @test sources(cs) === src
            @test targets(cs) === dst
            @test cs.score === scores
            @test hasattr(cs, :score) && hasattr(cs, :ratio) && !hasattr(cs, :weights)
            @test :score in propertynames(cs)

            # subsetting co-filters every column
            keep = falses(20); keep[1:2:end] .= true
            sub = cs[keep]
            @test length(sub) == 10
            @test sources(sub) == src[keep]
            @test sub.score == scores[keep]
            @test sub.ratio == ratios[keep]

            v = view(cs, 4:8)
            @test length(v) == 5
            @test v.score == scores[4:8]
            @test v[1] === (src[4] => dst[4])

            # withattrs merges columns, shares endpoints, subsets in lockstep
            w = rand(20)
            csw = withattrs(cs; weights = w)
            @test sources(csw) === src
            @test csw.score === scores
            @test csw.weights === w
            @test csw[keep].weights == w[keep]

            # invariants enforced at construction
            @test_throws ArgumentError Csponds(src, dst[1:19])
            @test_throws ArgumentError Csponds(src, dst; score = scores[1:5])
            @test_throws ArgumentError Csponds(src, dst; score = 1.0)
        end

        @testset "sources/targets protocol across containers" begin
            # legacy columnar StructArray (csponds constructor)
            sa = csponds(src, dst)
            @test sources(sa) === sa.first
            @test targets(sa) === sa.second

            # plain Pair vector: lazy view
            pv = [s => d for (s, d) in zip(src, dst)]
            @test sources(pv) isa MappedColumn
            @test collect(sources(pv)) == src
            @test collect(targets(pv)) == dst

            # scored legacy layout (VGF matcher output shape)
            scored = StructArray{ScoredCspond{Point2{Float64}, Point2{Float64}}}(
                (
                    StructArray{Cspond{Point2{Float64}, Point2{Float64}}}((src, dst)),
                    collect(1.0:20.0),
                )
            )
            @test collect(sources(scored)) == src
            @test collect(targets(scored)) == dst

            # scored row-major fallback
            rv = [ScoredCspond(src[i], dst[i], Float64(i)) for i in 1:20]
            @test collect(sources(rv)) == src
            @test collect(targets(rv)) == dst
        end

        @testset "lift / points / planar_lift" begin
            circles = [Circle(p, 2.0) for p in src]
            cs = Csponds(circles, dst; score = collect(1.0:20.0))

            pcs = points(cs)
            @test sources(pcs) isa MappedColumn
            @test sources(pcs)[1] == src[1]              # centroid of the circle
            @test targets(pcs)[1] == dst[1]              # centroid of a point = itself
            @test pcs.score == cs.score                  # attrs survive the lift

            wcs = planar_lift(cs)
            @test sources(wcs)[2] == Point3{Float64}(src[2][1], src[2][2], 0.0)
            @test targets(wcs)[2] == dst[2]

            # centroid closure over wrappers
            @test centroid(Attributed(Circle(src[1], 1.0), (; id = 7))) == src[1]
            @test centroid(Uncertain(src[1], CovMat{2}(σ = 0.5))) == src[1]

            # custom lift with explicit conversions
            lcs = lift(cs; source = c -> 2 .* centroid(c))
            @test sources(lcs)[1] == 2 .* src[1]
            @test targets(lcs) === targets(cs)           # identity side untouched

            # abstract-eltype parents narrow from the first element (the collect
            # convention) — a UnionAll-aliased feature vector must still produce
            # concretely-typed point columns for typed solver signatures
            any_feats = Any[Circle(p, 1.0) for p in src]
            acs = points(Csponds(any_feats, dst))
            @test eltype(sources(acs)) == Point2{Float64}
            @test acs isa AbstractVector{<:Cspond{Point2{Float64}, Point2{Float64}}}
        end

        @testset "CspondNormalization" begin
            cs = Csponds(src, dst)
            f = CspondNormalization(HomographyMat, cs)
            @test f isa CspondNormalization{Float64}

            ncs = normalize(f, cs)
            ns = sources(ncs)
            c = sum(ns) / length(ns)
            @test norm(c) < 1.0e-9
            @test abs(sum(norm, ns) / length(ns) - sqrt(2)) < 1.0e-9
            @test length(targets(ncs)) == 20

            # the sides as affine maps: the accessors agree with normalize
            @test VgtkCore.source_map(f)(src[3]) ≈ sources(ncs)[3] rtol = 1.0e-15
            @test VgtkCore.target_map(f)(dst[3]) ≈ targets(ncs)[3] rtol = 1.0e-15

            # Hartley about a KNOWN centre, weighted: the centre lands on the
            # origin, the weighted mean distance from it is √2; similarity_scale
            # reads a side's scale
            ctr = SVector(40.0, 60.0)
            wts = [isodd(i) ? 2.0 : 0.0 for i in 1:20]
            Tw = AffineMap(HartleyNormalization(; center = ctr), src, wts)
            @test Tw(ctr) == SVector(0.0, 0.0)
            @test sum(wts[i] * norm(Tw(src[i])) for i in 1:20) / sum(wts) ≈ sqrt(2) rtol = 1.0e-14
            @test VgtkCore.similarity_scale(Tw) ≈ Tw.linear[1, 1] rtol = 1.0e-15

            # frame matrices are exact inverses (computed once, analytically)
            @test norm(f.src * f.src_inv - I) < 1.0e-12
            @test norm(f.dst * f.dst_inv - I) < 1.0e-12

            # composition with the two sides round-trips (H and F laws)
            S, D = HomographyMap(source_map(f)), HomographyMap(target_map(f))
            Hn = HomographyMat(to_homogeneous(D ∘ HomographyMap(H0) ∘ inv(S)))
            @test Hn isa HomographyMat
            @test norm(HomographyMat(to_homogeneous(inv(D) ∘ HomographyMap(Hn) ∘ S)) - H0) < 1.0e-9
            # the composed model transfers the normalized points: the map after the
            # model, the source side's inverse before it
            transfer(H, x) = (PerspectiveMap() ∘ HomographyMap(H))(x)
            for x in (SVector(3.0, -2.0), SVector(40.0, 60.0), SVector(-7.5, 11.0))
                @test transfer(Hn, source_map(f)(x)) ≈ target_map(f)(transfer(H0, x)) rtol = 1.0e-12
            end

            F0 = FundMat{Float64}(Tuple(SMatrix{3, 3, Float64}(0.0, 1.0, -0.4, -1.0, 0.1, 3.0, 0.2, -2.5, 1.0)))
            @test norm(parent(D ∘ (inv(D) ∘ EpipolarMap(F0) ∘ S) ∘ inv(S)) - F0) < 1.0e-9
            # the composed correlation holds on the mapped points exactly when the
            # model does on the data: u₂ᵀ F u₁ = D(u₂)ᵀ F̂ S(u₁) up to the homogeneous scale
            Fn = parent(D ∘ EpipolarMap(F0) ∘ inv(S))
            for (u1, u2) in ((SVector(3.0, -2.0), SVector(1.0, 4.0)), (SVector(40.0, 60.0), SVector(-5.0, 2.5)))
                @test EpipolarForm(Fn)(source_map(f)(u1), target_map(f)(u2)) ≈ EpipolarForm(F0)(u1, u2) rtol = 1.0e-10
            end
            E0 = EssentialMat{Float64}(Tuple(SMatrix{3, 3, Float64}(F0)))
            @test parent(inv(D) ∘ EpipolarMap(E0) ∘ S) isa FundMat    # a collineation leaves the essential manifold
        end

        @testset "normalization strategies: AffineMap(strategy, points)" begin
            using CoordinateTransformations: AffineMap
            # Hartley: centroid (default) or a known centre, mean distance √2
            Th = AffineMap(HartleyNormalization(), src)
            c̄ = sum(src) / length(src)
            @test norm(Th(c̄)) < 1.0e-12
            @test sum(norm(Th(p)) for p in src) / length(src) ≈ sqrt(2) rtol = 1.0e-14
            ctr = SVector(40.0, 60.0)
            Tk = AffineMap(HartleyNormalization(; center = ctr), src)
            @test Tk(ctr) == SVector(0.0, 0.0)
            @test sum(norm(Tk(p)) for p in src) / length(src) ≈ sqrt(2) rtol = 1.0e-14
            wts = [isodd(i) ? 2.0 : 0.0 for i in 1:20]
            @test AffineMap(HartleyNormalization(), src, wts) == AffineMap(HartleyNormalization(), src, nothing, wts)
            @test AffineMap(HartleyNormalization(), src, BitVector(isodd(i) for i in 1:20)) ==
                AffineMap(HartleyNormalization(), src[1:2:end])
            # Diagonal: image centre to the origin, corners at unit radius
            Td = AffineMap(DiagonalNormalization(Size2(1200.0, 900.0)))
            @test Td(SVector(600.0, 450.0)) == SVector(0.0, 0.0)
            @test norm(Td(SVector(0.0, 0.0))) ≈ 1 rtol = 1.0e-15
            @test AffineMap(DiagonalNormalization((1200, 900))) == Td
            @test AffineMap(DiagonalNormalization(Size2(width = 1200px, height = 900px))) == Td
            @test AffineMap(DiagonalNormalization((1200, 900); center = (700.0, 400.0)))(SVector(700.0, 400.0)) == SVector(0.0, 0.0)
        end

        @testset "solve — Homography4pt" begin
            cs = Csponds(src, dst)
            s = Homography4pt()
            @test model_type(s) == HomographyMat
            @test sample_size(s) == 4
            @test max_solutions(s) == 1

            sols = solve(s, cs, 1:4)
            @test sols isa FixedModels{1, HomographyMat{Float64}}
            @test length(sols) == 1
            Hs = first(iterate(sols))
            @test all(norm(_apply_h(Hs, src[i]) - dst[i]) < 1.0e-6 for i in 1:20)

            # protocol interop: same solve on the legacy StructArray container
            sols_legacy = solve(s, csponds(src, dst), 1:4)
            @test first(iterate(sols_legacy)) ≈ Hs

            # degenerate (repeated point) → count-0 FixedModels, NOT nothing
            src_d = [src[1], src[1], src[3], src[4]]
            dst_d = [dst[1], dst[1], dst[3], dst[4]]
            bad = solve(s, Csponds(src_d, dst_d), 1:4)
            @test bad isa FixedModels{1, HomographyMat{Float64}}
            @test length(bad) == 0
            @test isempty(collect(bad))
        end

        # Shared two-view epipolar scene (also used by fit — FundamentalDlt)
        R = _rand_rot(rng)
        t = SVector(0.4, -0.2, 0.1)
        E_true = skew(t) * R
        X3 = [SVector(2randn(rng), 2randn(rng), 6.0 + 2rand(rng)) for _ in 1:20]
        x1b = [normalize(Xi) for Xi in X3]                    # bearings, cam1 at origin
        x2b = [normalize(R * Xi + t) for Xi in X3]            # bearings, cam2
        u1 = [Point2{Float64}(PerspectiveMap()(Xi)) for Xi in X3]
        u2 = [Point2{Float64}((R * Xi + t)[1] / (R * Xi + t)[3], (R * Xi + t)[2] / (R * Xi + t)[3]) for Xi in X3]
        F_true = E_true  # normalized cameras: F == E

        _epi(F, a, b) = abs(dot(SVector(b[1], b[2], 1.0), SMatrix{3, 3, Float64}(F) * SVector(a[1], a[2], 1.0)))

        @testset "solve — Fundamental7pt" begin
            cs = Csponds(u1, u2)
            s = Fundamental7pt()
            @test sample_size(s) == 7 && max_solutions(s) == 3

            sols = solve(s, cs, 1:7)
            @test sols isa FixedModels{3, FundMat{Float64}}
            @test 1 <= length(sols) <= 3
            # every returned model satisfies the sample's epipolar constraints
            for F in sols
                @test all(_epi(F, u1[i], u2[i]) < 1.0e-8 for i in 1:7)
            end
            # at least one candidate is the true F (up to scale/sign)
            Ft = FundMat{Float64}(Tuple(SMatrix{3, 3, Float64}(F_true) / norm(F_true)))
            best = minimum(
                min(norm(SMatrix{3, 3, Float64}(F) / norm(F) - Ft), norm(SMatrix{3, 3, Float64}(F) / norm(F) + Ft))
                    for F in sols
            )
            @test best < 1.0e-6

            # degenerate sample (all identical) → count-0, not nothing
            bad = solve(s, Csponds(fill(u1[1], 7), fill(u2[1], 7)), 1:7)
            @test bad isa FixedModels{3, FundMat{Float64}}
            @test length(bad) == 0
        end

        @testset "solve — Essential5pt" begin
            s = Essential5pt()
            @test sample_size(s) == 5 && max_solutions(s) == 10

            # BEARINGS ONLY. A calibrated 2-vector used to be accepted here and
            # lifted to (x, y, 1); that lift is deleted, because the z = 1 plane
            # cannot represent a ray at or past 90° and so silently excluded
            # every fisheye and omnidirectional camera while still returning a
            # plausible EssentialMat. The refusal is by TYPE, so it cannot be
            # re-introduced by a caller's convention.
            @test_throws MethodError solve(s, Csponds(u1, u2), 1:5)
            for cs in (Csponds(x1b, x2b),)
                sols = solve(s, cs, 1:5)
                @test sols isa FixedModels{10, EssentialMat{Float64}}
                @test length(sols) >= 1
                Et = SMatrix{3, 3, Float64}(E_true) / norm(E_true)
                best = minimum(
                    min(norm(SMatrix{3, 3, Float64}(E) / norm(E) - Et), norm(SMatrix{3, 3, Float64}(E) / norm(E) + Et))
                        for E in sols
                )
                @test best < 1.0e-6
            end
        end

        @testset "solve — P3P" begin
            s = P3P()
            @test sample_size(s) == 3 && max_solutions(s) == 4
            @test model_type(s) == Extrinsics3

            Rp = _rand_rot(rng)
            tp = SVector(0.3, -0.5, 4.0)
            Xw = [SVector{3, Float64}(2randn(rng, 3)...) for _ in 1:3]
            xs = [SVector{3, Float64}(normalize(Rp * Xi + tp)) for Xi in Xw]
            cs = Csponds(Xw, xs)

            sols = solve(s, cs, 1:3)
            @test sols isa FixedModels{4, Extrinsics3}
            @test length(sols) >= 1
            err = minimum(norm(SMatrix{3, 3, Float64}(p.R) - Rp) + norm(p.t - tp) for p in sols)
            @test err < 1.0e-6

            # map: the per-model transform, count kept, padding never touched
            m = map(identity, sols)
            @test m isa FixedModels{4, Extrinsics3}
            @test length(m) == length(sols)
            @test all(m[i] == sols[i] for i in 1:length(sols))
            @test all(m.data[i] == zero(Extrinsics3) for i in (length(sols) + 1):4)
            fm = FixedModels{3, SVector{2, Float64}}(2, (SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(NaN, NaN)))
            mf = map(v -> SVector{2, Float32}(v), fm)
            @test mf isa FixedModels{3, SVector{2, Float32}}
            @test mf.data == (SVector{2, Float32}(1, 2), SVector{2, Float32}(3, 4), zero(SVector{2, Float32}))

            # wrong-arity kernel call now throws instead of returning nothing
            @test_throws ArgumentError p3p(NativeSolver(), xs[1:2], Xw[1:2])
        end

        @testset "DLT kernels — homography_dlt / fundamental_dlt on correspondences" begin
            cs = Csponds(src, dst)
            Hf = homography_dlt(cs)
            @test Hf isa HomographyMat{Float64}
            @test all(norm(_apply_h(Hf, src[i]) - dst[i]) < 1.0e-6 for i in 1:20)
            # a mask (Bool, or an index list with a frame) selects the correspondences
            mask = BitVector(isodd(i) for i in 1:20)
            @test all(norm(_apply_h(homography_dlt(cs; mask), src[i]) - dst[i]) < 1.0e-6 for i in 1:20)
            @test all(norm(_apply_h(homography_dlt(cs; mask = findall(mask)), src[i]) - dst[i]) < 1.0e-6 for i in 1:20)   # an index mask
            @test homography_dlt(cs; mask = findall(mask)) == homography_dlt(cs; mask)
            @test_throws MethodError homography_dlt(cs; normalization = CspondNormalization(HomographyMat, cs))   # public forms take none
            # degenerate-but-finite data still yields a (garbage) model — validity
            # gating is the consumer's job (test_model); only non-finite rows are
            # refused, with `nothing`
            @test homography_dlt(Csponds(fill(src[1], 8), fill(dst[1], 8))) isa HomographyMat{Float64}
            src_nan = [[Point2{Float64}(NaN, NaN)]; src[2:8]]
            @test homography_dlt(Csponds(src_nan, dst[1:8])) === nothing

            Ff = fundamental_dlt(Csponds(u1, u2))
            @test Ff isa FundMat{Float64}
            @test all(_epi(Ff, u1[i], u2[i]) < 1.0e-8 for i in 1:20)
        end

        @testset "DLT kernels — weights are the container's column" begin
            # Noisy data, so a weight moves the estimate. An integer weight is the
            # same least-squares problem as that many copies of the correspondence
            # (its rows scale by √w, the Hartley centroid and spread count it w
            # times); a zero weight is the mask that drops it.
            nrng = MersenneTwister(7)
            dn = [p + Point2{Float64}(0.3randn(nrng), 0.3randn(nrng)) for p in dst]
            u2n = [p + Point2{Float64}(1.0e-3randn(nrng), 1.0e-3randn(nrng)) for p in u2]
            w = [i == 1 ? 3.0 : 1.0 for i in 1:20]
            _same(A, B) = norm(SMatrix{3, 3, Float64}(A) - SMatrix{3, 3, Float64}(B)) < 1.0e-9
            csH = Csponds(src, dn); csF = Csponds(u1, u2n)
            Hw = homography_dlt(withattrs(csH; weights = w))
            @test _same(Hw, homography_dlt(Csponds([fill(src[1], 2); src], [fill(dn[1], 2); dn])))
            @test !_same(Hw, homography_dlt(csH))                        # the weight moved the estimate
            Fw = fundamental_dlt(withattrs(csF; weights = w))
            @test _same(Fw, fundamental_dlt(Csponds([fill(u1[1], 2); u1], [fill(u2n[1], 2); u2n])))
            @test !_same(Fw, fundamental_dlt(csF))
            keep = BitVector(i > 3 for i in 1:20); z = Float64.(keep)
            @test _same(homography_dlt(withattrs(csH; weights = z)), homography_dlt(csH; mask = keep))
            @test _same(fundamental_dlt(withattrs(csF; weights = z)), fundamental_dlt(csF; mask = keep))
            # a plain Pair vector carries no column: unweighted, the same kernel
            @test homography_dlt(collect(csH)) == homography_dlt(csH)
            @test fundamental_dlt(collect(csF)) == fundamental_dlt(csF)
            # weights are never a keyword, and there is no parallel-array form
            @test_throws MethodError homography_dlt(csH; weights = w)
            @test_throws MethodError fundamental_dlt(csF; weights = w)
            @test_throws MethodError homography_dlt(src, dn)
            @test_throws MethodError fundamental_dlt(u1, u2n)
            # the LO/IRLS hot path: re-weighting per iteration through the column
            # allocates nothing in the homography kernel beyond the caller's buffers
            A = Matrix{Float64}(undef, 40, 9)
            ws = VgtkCore.SvdWorkspace{Float64}(40, 9)
            # (the estimator's normalization is built once, at its entry; the
            # inner-loop form takes it and never builds one)
            N = CspondNormalization(HomographyMat, withattrs(csH; weights = w))
            _lo_refit(A, N, cs, w, ws) = homography_dlt!(A, N, withattrs(cs; weights = w); svd_ws = ws)
            _lo_refit(A, N, csH, w, ws)
            @test (@allocated _lo_refit(A, N, csH, w, ws)) == 0
            @test _same(_lo_refit(A, N, csH, w, ws), Hw)             # the public form (its SVD path differs only in rounding)
            @test_throws MethodError homography_dlt!(A, csH)           # the inner loop never normalizes itself
        end

        @testset "type stability (hot paths infer concretely)" begin
            # `@inferred` THROWS when a call does not infer concretely, but it
            # records no test — so a testset built only from it reports "no
            # tests" inside a green suite and no reader can tell it ran. The
            # `isa Any` is the recorded form of exactly the claim `@inferred`
            # proves: this call inferred, and returned.
            cs = Csponds(src, dst; score = collect(1.0:20.0))
            @test (@inferred sources(cs)) isa Any
            @test (@inferred targets(cs)) isa Any
            @test (@inferred cs[3]) isa Any
            @test (@inferred (c -> c.score)(cs)) isa Any

            @test (@inferred solve(Homography4pt(), cs, collect(1:4))) isa Any
            @test (@inferred solve(Fundamental7pt(), cs, collect(1:7))) isa Any
            @test (@inferred solve(Essential5pt(), Csponds(x1b, x2b), collect(1:5))) isa Any
            Xw3 = [SVector(1.0, 0.0, 5.0), SVector(-1.0, 1.0, 6.0), SVector(0.5, -1.0, 4.0)]
            xs3 = [normalize(SVector(0.1, 0.05, 1.0)), normalize(SVector(-0.2, 0.1, 1.0)), normalize(SVector(0.0, -0.15, 1.0))]
            @test (@inferred solve(P3P(), Csponds(Xw3, xs3), collect(1:3))) isa Any
            @test (@inferred Union{Nothing, FundMat{Float64}} fundamental_dlt(Csponds(u1, u2))) isa Any
            @test (@inferred Union{Nothing, HomographyMat{Float64}} homography_dlt(withattrs(cs; weights = ones(20)))) isa Any   # the kernel returns `nothing` on non-finite rows

            f = CspondNormalization(HomographyMat, cs)
            @test (@inferred CspondNormalization(HomographyMat, cs)) isa Any
            Sf, Df = HomographyMap(source_map(f)), HomographyMap(target_map(f))
            @test (@inferred Df ∘ HomographyMap(H0) ∘ inv(Sf)) isa Any
            @test (@inferred inv(Df) ∘ HomographyMap(H0) ∘ Sf) isa Any
            F0 = FundMat{Float64}(Tuple(SMatrix{3, 3, Float64}(I)))
            @test (@inferred Df ∘ EpipolarMap(F0) ∘ inv(Sf)) isa Any
            @test (@inferred normalize(f, cs)) isa Any
            @test (@inferred points(cs)[1]) isa Any
        end

        @testset "induced-model registry" begin
            P2 = Point2{Float64}
            V3 = SVector{3, Float64}
            # E is NO LONGER offered for plane data. It sat here until 0.29 with
            # a note that it "presumes calibrated coordinates — it is the
            # caller's frame convention", which made `induced_models` advertise
            # an essential matrix for any pair of 2-vectors.
            @test induced_models(P2, P2) == (HomographyMat, FundMat)
            @test !(EssentialMat in induced_models(P2, P2))
            @test induced_models(Point2{Float64}, V3) == ()

            cs = Csponds(src, dst)
            @test induced_models(cs) == (HomographyMat, FundMat)

            # (3D, 3D) is genuinely ambiguous — P3P's source is a world POINT,
            # E's endpoints are BEARINGS, and a bare SVector{3} is both. The
            # registry reports both rather than choosing one silently.
            @test induced_models(V3, V3) == (Extrinsics3, EssentialMat)

            # …and the by-model lookup is exact despite that ambiguity, which is
            # what makes reporting it safe.
            @test minimal_solver(HomographyMat, P2, P2) isa Homography4pt
            @test minimal_solver(FundMat, P2, P2) isa Fundamental7pt
            @test minimal_solver(Extrinsics3, V3, V3) isa P3P
            @test minimal_solver(EssentialMat, V3, V3) isa Essential5pt
            @test_throws ArgumentError minimal_solver(Extrinsics3, P2, P2)
            @test_throws ArgumentError minimal_solver(EssentialMat, P2, P2)
        end
    end
end
