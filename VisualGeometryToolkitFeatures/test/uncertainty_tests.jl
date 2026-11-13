# Detection uncertainty (`asift_detect(img; uncertainty = FeatureCovariance())`):
# the table's exactness against the real kernels, the kernels' covariance against the refinement's Jacobian by
# finite differences, the API and its round trips, and
# a Monte Carlo calibration on synthetic blobs. The full study, many configs
# and trials, is benchmark/mc/detection_uncertainty_mc.jl (not in the suite).

@testitem "detection covariance: the C table is the pyramid's K·Kᵀ" setup = [UncFixtures] begin
    using VisualGeometryToolkitFeatures: AsiftPlan, _ensure_cov!, _unpack_M
    import KernelAbstractions as KA
    # Identity warp, 256²: octaves −1 (all four pixel parities), 0 and 1, every
    # refinable slice. M from impulses through the real Float32 kernels vs the
    # Float64 adjoint table; the Float32 pyramid limits the agreement.
    plan = AsiftPlan(256, 256; device = KA.CPU(), tilts = [(1.0, 0.0)], upsample = true)
    _ensure_cov!(plan)
    for oi in 1:3, sl in 2:(plan.nb_scales + 1), (px, py) in ((0, 0), (1, 0), (0, 1), (1, 1))
        oi > 1 && (px, py) != (0, 0) && continue
        Mi = impulse_M(plan, oi, sl, px, py)
        Mt = _unpack_M(Array(plan.ctab[oi]), sl, 1 + px + 2py, 1)
        # relative to the functionals' own scale (a correlation-scale error)
        err = maximum(abs.(Mi - Mt) ./ sqrt.(diag(Mi) * diag(Mi)'))
        @test err < 1.0e-4
    end
end

@testitem "detection covariance: the kernels' covariance is J·M·Jᵀ of the refinement, J by finite differences" setup = [UncFixtures] begin
    using VisualGeometryToolkitFeatures: AsiftPlan, asift_detect!, _unpack_M, _backproject_cov!
    using CUDA: CUDA
    import KernelAbstractions as KA
    # The one implementation is the kernels'; the independent check differentiates the map they linearize. A converged
    # extremum's refinement is the map f ↦ (x, y, log σ) of its nine DoG functionals f (gradient, then the Hessian's
    # ss, xx, yy, sx, sy, xy): the offset −Ĥ(f)⁻¹ĝ(f) from the integer extremum, placed on the octave's grid. Its
    # Jacobian, by central differences, carries the functionals' covariance M (the table's): Σ = J·M·Jᵀ.
    function fd_jacobian(F, f)
        y0 = F(f)
        J = zeros(length(y0), length(f))
        sc = maximum(abs, f)
        for j in eachindex(f)
            h = 1.0e-5 * max(abs(f[j]), 1.0e-3 * sc)
            fp = collect(f); fm = collect(f); fp[j] += h; fm[j] -= h
            J[:, j] = (F(fp) - F(fm)) / 2h
        end
        return J
    end
    hessian(f) = [f[4] f[7] f[8]; f[7] f[5] f[9]; f[8] f[9] f[6]]
    offset(f) = -(hessian(f) \ [f[1], f[2], f[3]])
    # Blobs at sub-pixel positions; the identity warp's pool (its back-projection the identity, so `plan.featcov` is
    # the canvas covariance), each entry matched to its integer extremum on the host copy of the DoG.
    img = blob_image(128, 160, [(40.3, 50.6, 3.1, 0.3), (100.8, 70.2, 4.4, -0.3), (120.5, 30.5, 2.2, 0.25),
        (30.7, 100.2, 2.7, -0.2), (80.2, 25.9, 5.5, 0.3), (130.4, 95.3, 3.6, 0.25)])
    devices = CUDA.functional() ? (KA.CPU(), CUDA.CUDABackend()) : (KA.CPU(),)
    for dev in devices
        plan = AsiftPlan(160, 128; device = dev, tilts = [(1.0, 0.0)], upsample = true)
        asift_detect!(plan, permutedims(Float32.(img)); uncertainty = FeatureCovariance())
        n = Int(Array(plan.counth)[1])
        raw = Array(plan.raw)[:, 1:n]; rc = Array(plan.featcov)[:, 1:n]
        S = plan.nb_scales; σ0 = plan.seed_sigma
        checked = 0
        for oi in 1:plan.nb_octaves
            o = oi - 1 + plan.o_first; sf = 2.0^o
            d = Float64.(Array(plan.dogs[oi])[:, :, :, 1])
            Wo, Ho, NS = size(d)
            tab = Array(plan.ctab[oi])
            for X in 2:(Wo - 1), Y in 2:(Ho - 1), sl in 2:(NS - 1)
                f = collect(functionals_at(d, X, Y, sl))
                abs(det(hessian(f))) > 0 || continue
                off = offset(f)
                all(abs.(off) .< 0.6) || continue
                xo = ((X - 1) + off[2]) * sf + (sf - 1); yo = ((Y - 1) + off[3]) * sf + (sf - 1)
                σo = σ0 * 2^(((sl - 1) + off[1] + 0.5) / S) * sf
                k = findfirst(i -> abs(raw[1, i] - xo) < 1.0e-3 && abs(raw[2, i] - yo) < 1.0e-3 &&
                    abs(raw[3, i] - σo) < 1.0e-3 * σo, 1:n)
                k === nothing && continue
                # the refinement, f ↦ (x, y, log σ)
                function refined(g)
                    v = offset(g)
                    return [((X - 1) + v[2]) * sf + (sf - 1), ((Y - 1) + v[3]) * sf + (sf - 1),
                            log(σ0) + ((sl - 1) + v[1] + 0.5) / S * log(2) + log(sf)]
                end
                J = fd_jacobian(refined, f)
                M = _unpack_M(tab, sl, 1 + (X & 1) + 2 * (Y & 1), 1)
                Σfd = J * M * J'
                c = rc[:, k]
                Σk = [c[1] c[2] c[3]; c[2] c[4] c[5]; c[3] c[5] c[6]]
                # entrywise, on the correlation scale (a matrix norm would hide the small log σ entries)
                err = maximum(abs.(Σk - Σfd) ./ sqrt.(diag(Σfd) * diag(Σfd)'))
                @test (dev, err < 1.0e-3) == (dev, true)
                checked += 1
            end
        end
        @test checked >= 5
    end
    # back-projection: the back-warp's linear part A on (x, y), log σ kept; its Jacobian by central differences
    A = [0.8 -1.5; 0.6 2.0]; t = [3.0, -7.0]
    backwarp(v) = [A * v[1:2] + t; v[3]]
    Jb = fd_jacobian(backwarp, [12.0, 30.0, 1.1])
    Σ = [1.0 0.2 0.1; 0.2 2.0 0.3; 0.1 0.3 0.5]
    packed = reshape([Σ[1, 1], Σ[1, 2], Σ[1, 3], Σ[2, 2], Σ[2, 3], Σ[3, 3]], 6, 1)
    out = zeros(6, 1)
    _backproject_cov!(out, packed, 1, A[1, 1], A[1, 2], A[2, 1], A[2, 2])
    Σb = Jb * Σ * Jb'
    @test out[:, 1] ≈ [Σb[1, 1], Σb[1, 2], Σb[1, 3], Σb[2, 2], Σb[2, 3], Σb[3, 3]]
end

@testitem "detection covariance: API, off path unchanged, round trips" setup = [Fixtures, UncFixtures] begin
    using VisualGeometryToolkitCore: feature_frame
    using VisualGeometryToolkitFeatures: ImasFeatureKA
    using VisualGeometryToolkitCore: Uncertain, CovMat, UnlabeledFrame, cov, geom
    using StaticArrays: SMatrix
    dev = KA.CPU()
    off = asift_detect(IMG; device = dev)
    on = asift_detect(IMG; device = dev, uncertainty = FeatureCovariance())
    # the frames, metadata and order are the off path's; only the covariance is added
    @test length(on) == length(off)
    @test all(feature_frame(a) == feature_frame(b) && a.metadata == b.metadata for (a, b) in zip(on, off))
    @test all(geom(a) == b for (a, b) in zip(on, off))
    d = on[1]
    @test getfield(d, :value) isa Uncertain
    @test cov(d) isa CovMat{3, Float64}
    @test d.center == off[1].center && d.σ == off[1].σ && d.response == off[1].response
    @test is_dark(d) == is_dark(off[1]) && is_light(d) == is_light(off[1])
    @test all(cov(x) ≈ cov(x)' && all(>(0), diag(Matrix(cov(x)))) for x in on if all(isfinite, cov(x)))
    # raw: the features are the off path's, bit for bit
    rawoff = asift_detect(IMG; device = dev, raw = true)
    rawon = asift_detect(IMG; device = dev, raw = true, uncertainty = FeatureCovariance())
    @test rawon.features == rawoff
    @test length(rawon.cov) == length(rawoff)
    # type-stable per option (up to the `raw` switch, a runtime Bool)
    @inferred Vector{ImasFeatureKA} asift_detect(IMG; device = dev)
    @inferred NamedTuple{(:features, :cov), Tuple{Vector{ImasFeatureKA}, Vector{CovMat{3, Float64, SMatrix{3, 3, Float64, 9}, UnlabeledFrame{3}}}}} asift_detect(IMG; device = dev, uncertainty = FeatureCovariance())
    nanc = Detection(Uncertain(feature_frame(d), CovMat{3, Float64}(Tuple(fill(NaN, 9)))), d.metadata)
    # FileCache (JLD2) round trip, NaN covariances included
    using JLD2
    path = joinpath(mktempdir(), "unc.jld2")
    c = open_cache(path)
    get!(() -> [d, nanc], c, "dets")                # computed and stored
    close_cache(c)
    c = open_cache(path)
    back = get!(() -> error("recomputed"), c, "dets")
    close_cache(c)
    @test typeof(back) == typeof([d, nanc]) && isequal(Matrix(cov(back[2])), Matrix(cov(nanc))) &&
        cov(back[1]) == cov(d) && feature_frame(back[1]) == feature_frame(d)
    # DescribedFeature carries an uncertain detection
    sa = described_features(on, rand(Float32, length(on), 4))
    @test sa[1].center == d.center && cov(sa[1].detection) == cov(d)
end

@testitem "detection covariance: Monte Carlo calibration (CPU)" setup = [UncFixtures] begin
    # Isotropic blobs at sub-pixel positions over octaves −1…1, light and dark,
    # contrast 0.15–0.3, on the identity warp; iid Gaussian noise of σn = 0.01 on
    # the float input path (no quantization). The scales stay off the octave
    # seams (σ ≈ 4.03·2^o), where noise flips a detection between two octaves
    # whose refined estimates differ by ~0.1–0.2 px — a jump no linearization
    # describes (the full study, benchmark/mc, measures it). Per feature the mean
    # normalized squared error has expectation 3 (2 for the position block) and,
    # over NTRIALS = 120, a standard error √(6/120) ≈ 0.22; pooled over the ten
    # features √(6/1200) ≈ 0.07. Tolerances: pooled mean within ±15 % (≈ 6 s.e.),
    # pooled 95 % coverage within ±4 points (s.e. 0.6), and per feature a mean in
    # (0.5, 1.6)·3: wider than its noise because a blob near a half level pixel
    # of its octave (here (36.2, 92.55, σ 5): y at 45.52 octave-1 pixels) flips
    # between two start pixels whose quadratic refinements differ — an estimator
    # jump the linearization does not see, measured at ≈ 1.4× in the full study.
    # Single-threaded CPU: ≈ 20 s.
    # A Hessian-only model with one fitted scale misses by a factor 0.1–10
    # across these scales (benchmark/mc).
    blobs = [
        (40.3, 30.6, 2.5, 0.25), (90.7, 34.2, 2.9, -0.25), (140.45, 36.9, 3.3, 0.2),
        (36.2, 92.55, 5.0, -0.3), (98.9, 96.1, 5.7, 0.3), (160.6, 100.4, 6.5, -0.3),
        (52.35, 160.8, 2.7, -0.15), (120.15, 158.3, 6.0, 0.15), (178.5, 170.5, 3.1, 0.3),
        (190.1, 40.2, 2.4, 0.3),
    ]
    img = blob_image(210, 230, blobs)
    truth = [(b[1], b[2], b[3]) for b in blobs]
    NTRIALS = 120
    res = mc_run(img, truth; σn = 0.01, ntrials = NTRIALS, seed = 7)
    e3 = Float64[]; e2 = Float64[]; c95 = Float64[]
    for (k, r) in enumerate(res)
        @test length(r.est) >= 0.9 * NTRIALS        # detected, refined, associated
        s = mc_summary(r.est, r.cov)
        @test 0.5 < s.mean_e3 / 3 < 1.6
        push!(e3, s.mean_e3); push!(e2, s.mean_e2); push!(c95, s.cov95_3)
    end
    @test 0.85 < mean(e3) / 3 < 1.15
    @test 0.85 < mean(e2) / 2 < 1.15
    @test abs(mean(c95) - 0.95) < 0.04
end

@testitem "detection covariance: position variance scales as contrast⁻² (Zeisl's ∓H⁻¹ as contrast⁻¹)" setup = [UncFixtures] begin
    using VisualGeometryToolkitFeatures: AsiftPlan
    using VisualGeometryToolkitCore: cov
    import KernelAbstractions as KA
    # Noiseless: the pipeline is linear in the image and a constant background
    # has zero DoG, so scaling a blob's amplitude scales ĝ and Ĥ alike — the same
    # refinement offsets, Ĥ⁻¹·C·Ĥ⁻¹ ∝ a⁻², ∓H⁻¹ ∝ a⁻¹. The empirical variance is
    # ∝ a⁻² (benchmark/mc/zeisl_comparison.jl: slopes −2.0 ± 0.1); Float32 rounding
    # alone separates the measured slopes from −2 and −1, hence 0.01.
    amps = (0.2, 0.4, 0.8)        # above the default detection threshold
    v = Float64[]; z = Float64[]
    for a in amps
        img = Float32.(blob_image(96, 96, [(48.3, 47.6, 4.0, a)]))
        plan = AsiftPlan(96, 96; device = KA.CPU(), tilts = [(1.0, 0.0)])   # its own plan, not the cache's
        d = only(detect_on(plan, img))
        push!(v, tr(Matrix(cov(d))[1:2, 1:2]))
        e = extremum_of(plan, d)
        push!(z, tr(zeisl_cov(e, d.response)))
    end
    slope(y) = log(y[end] / y[1]) / log(amps[end] / amps[1])
    @test abs(slope(v) + 2) < 0.01
    @test abs(slope(z) + 1) < 0.01
end

# A detection's uncertain centre through a map (detection_types.jl, moved from PanoramaStitching): its centre with the
# (x, y) block of its frame's covariance, carried by Core's propagate.
@testitem "detection covariance: propagate(map, detection) — the centre and its position block through the map" setup = [Fixtures, UncFixtures] begin
    using VisualGeometryToolkitCore: Uncertain, CovMat, cov, geom, propagate, center, SimilarityMap, LinearMap, Translation
    using StaticArrays: SMatrix, SVector
    using LinearAlgebra: UniformScaling
    on = asift_detect(IMG; device = KA.CPU(), uncertainty = FeatureCovariance())
    d = first(on)
    S = SimilarityMap(Translation(SVector(1.0, -3.0))) ∘ SimilarityMap(LinearMap(UniformScaling(2.0)))   # x ↦ 2x + (1, −3)
    u = propagate(S, d)
    Σ = cov(d)
    ref = propagate(S, center(d))                                # Core's one conversion: the centre and the (x, y) block
    @test geom(center(d)) == d.center && SMatrix{2, 2}(cov(center(d))) == SMatrix{2, 2}(Σ[1, 1], Σ[2, 1], Σ[1, 2], Σ[2, 2])
    @test geom(u) == geom(ref) && cov(u) == cov(ref)
    @test geom(u) ≈ S(SVector(d.center[1], d.center[2]))
    @test SMatrix{2, 2}(cov(u)) ≈ 4 * SMatrix{2, 2}(Σ[1, 1], Σ[2, 1], Σ[1, 2], Σ[2, 2])
end
