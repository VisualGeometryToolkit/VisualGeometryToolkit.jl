# VisualGeometryFeatures test items (ReTestItems). Shared fixtures: vgf_testsetup.jl.

@testitem "asift_detect API (CPU KA)" setup = [Fixtures] begin
    dets = asift_detect(IMG)                       # default :mxm NMS
    @test dets isa Vector
    @test !isempty(dets)
    # every planted blob is found at its COLMAP centre: a blob planted at
    # the 1-based pixel index (cy, cx) is centred on that pixel, whose
    # COLMAP centre is (cx - 0.5, cy - 0.5) — the top-left pixel's is
    # (0.5, 0.5). Sub-pixel: the detector localizes an isolated blob to
    # a few hundredths of a pixel (measured 2026-09-15).
    for (cy, cx, s, _) in BLOBS
        dmin = minimum(
            d -> hypot(
                d.frame.center[1] - (cx - 0.5),
                d.frame.center[2] - (cy - 0.5)
            ), dets
        )
        @test dmin < 0.5
    end
    # metadata contract
    d1 = first(dets)
    @test hasproperty(d1.metadata, :response)
    @test hasproperty(d1.metadata, :warp_idx)

    # pool ⊇ survivors; raw mode returns ImasFeatureKA
    pool = asift_detect(IMG; nms = :none, raw = true)
    @test pool isa Vector{ImasFeatureKA}
    @test length(pool) >= length(dets)

    # ellipse NMS mode runs
    dells = asift_detect(IMG; nms = :ellipse)
    @test 0 < length(dells) <= length(pool)
end

# asift_detect's output convention, pinned (documented in its docstring): COLMAP
# pixel centres. A blob centred on the pixel img[r, c] comes back at
# (c − 0.5, r − 0.5), and between pixel centres the quadratic sub-pixel
# refinement pulls it toward the nearest one — at σ = 4 by −0.007, −0.015,
# −0.021, −0.027 px at fractions 0.1–0.4, mirrored above 0.5, and −0.031 px at
# 0.49 and at 0.5 exactly (measured 2026-09-22). At 0.5 the DoG ties between
# the two pixels; the extremum test breaks the tie toward the first, which then
# refines. (Strict against all 26 neighbours, it rejected both
# and an unrefined coarser-octave detection was returned 0.5 px low.)
@testitem "asift_detect coordinates: COLMAP pixel centres, sub-pixel bias ≤ 0.04 px" setup = [Fixtures] begin
    # a dark isotropic Gaussian, σ = 4, on a Float32 canvas (no 8-bit quantisation)
    blob(r, c) = Float32[0.5 - 0.35 * exp(-((i - r)^2 + (j - c)^2) / 32) for i in 1:64, j in 1:64]
    # the error of the detection nearest where COLMAP puts a blob centred at
    # (r, c), and how many detections lie within a pixel of it
    function err(r, c)
        x, y = c - 0.5, r - 0.5
        dets = asift_detect(blob(r, c); device = KA.CPU())
        d = argmin(d -> hypot(d.center[1] - x, d.center[2] - y), dets)
        near = count(d -> hypot(d.center[1] - x, d.center[2] - y) < 1, dets)
        return d.center[1] - x, d.center[2] - y, near
    end
    # on a pixel centre: exact, wherever on the grid
    for (r, c) in ((32, 32), (20, 27), (41, 33))
        ex, ey, near = err(r, c)
        @test abs(ex) <= 0.005 && abs(ey) <= 0.005 && near == 1
    end
    # between pixel centres, along each axis in turn, the exact half-pixel tie
    # included: one detection, its bias under 0.04 px, the other axis exact
    for f in (0.1, 0.2, 0.3, 0.4, 0.49, 0.5, 0.51, 0.6, 0.7, 0.8, 0.9)
        ex, ey, near = err(32, 32 + f)
        @test abs(ex) <= 0.04 && abs(ey) <= 0.005 && near == 1
        ex, ey, near = err(32 + f, 32)
        @test abs(ey) <= 0.04 && abs(ex) <= 0.005 && near == 1
    end
    # both axes on the tie at once: a 4-way tie, still one refined detection
    let (ex, ey, near) = err(32.5, 32.5)
        @test abs(ex) <= 0.04 && abs(ey) <= 0.04 && near == 1
    end
end

@testitem "polarity filtering" setup = [Fixtures] begin
    dark = asift_detect(IMG; polarity = :dark, nms = :none, raw = true)
    light = asift_detect(IMG; polarity = :light, nms = :none, raw = true)
    @test all(f -> f.intensity > 0, dark)
    @test all(f -> f.intensity < 0, light)
    @test !isempty(dark) && !isempty(light)
    # planted valley at (row 70, col 96) is a DARK blob; peak at (40, 50) is LIGHT
    @test any(f -> hypot(f.x - 96, f.y - 70) < 2.0, dark)
    @test any(f -> hypot(f.x - 50, f.y - 40) < 2.0, light)

    # the gate runs BEFORE cross-warp NMS: with the light features out of the
    # pool the splat is pointwise ≤ the mixed pool's, so every dark survivor of
    # the mixed pool must survive the dark-only pool too (never fewer)
    key(f) = (f.x, f.y, f.warp_idx)
    for nms in (:mxm, :ellipse)
        mixed = asift_detect(IMG; polarity = :none, nms, raw = true)
        darkd = asift_detect(IMG; polarity = :dark, nms, raw = true)
        @test all(f -> f.intensity > 0, darkd)
        @test issubset(
            Set(key(f) for f in mixed if f.intensity > 0),
            Set(key(f) for f in darkd)
        )
        @test length(darkd) >= count(f -> f.intensity > 0, mixed)
    end
    @test_throws ArgumentError asift_detect(IMG; polarity = :bogus, nms = :none)
end

@testitem "scale-banded cross-warp NMS" setup = [Fixtures] begin
    key(f) = (f.x, f.y, f.warp_idx)
    keys_(v) = Set(key(f) for f in v)
    pool = asift_detect(IMG; nms = :none, raw = true)
    W, H = size(IMG, 2), size(IMG, 1)
    for nms in (:mxm, :ellipse)
        base = asift_detect(IMG; nms, raw = true)
        off = asift_detect(IMG; nms, band_octaves = 0.0, raw = true)
        @test keys_(off) == keys_(base)                       # banding off ≡ the whole-pool rule
        banded = asift_detect(IMG; nms, band_octaves = 1.0, band_reach = 1, raw = true)
        @test issubset(keys_(base), keys_(banded))            # a band only removes suppressors
        @test !isempty(banded)
        # NMS over the pool asift_detect returns keeps what the in-pipeline NMS
        # keeps (one set of kernels behind both)
        host = VisualGeometryFeatures.cross_warp_nms(
            pool, W, H; mode = nms,
            band_octaves = 1.0, band_reach = 1
        )
        @test keys_(host) == keys_(banded)
        host0 = VisualGeometryFeatures.cross_warp_nms(pool, W, H; mode = nms)
        @test keys_(host0) == keys_(base)
    end
    @test_throws ArgumentError asift_detect(IMG; nms = :ellipse, band_octaves = -1.0)
end

@testitem "cross-warp NMS: no 65535-feature cap" setup = [Fixtures] begin
    # Through `cross_warp_nms`, which runs the NMS kernels asift_detect! runs.
    # 70 000 features on a 4 px grid, each alone in its footprint, plus a
    # weaker duplicate of the first placed past index 65 535: every grid
    # feature survives and the duplicate is suppressed, not dropped by an
    # index cap (the old 16-bit packing kept only the first 65 535).
    nx, ny = 280, 250
    feat(x, y, I) = ImasFeatureKA(
        Float32(x), Float32(y), 1.0f0, I,
        0.25f0, 0.0f0, 0.25f0, Int32(1)
    )
    feats = [feat(4i, 4j, 0.1f0) for j in 0:(ny - 1) for i in 0:(nx - 1)]
    push!(feats, feat(0, 0, 0.05f0))
    for mode in (:mxm, :ellipse)
        kept = VisualGeometryFeatures.cross_warp_nms(feats, 4nx, 4ny; device = KA.CPU(), mode)
        @test length(kept) == nx * ny
        @test all(k -> k.intensity == 0.1f0, kept)
    end
end

@testitem "identity-only tilts ≡ plain SIFT" setup = [Fixtures] begin
    # a single identity warp is a plain DoG-SIFT detector: every feature has
    # warp_idx == 1 and a circular input-frame shape S = σ·I
    feats = asift_detect(IMG; tilts = [(1.0, 0.0)], nms = :none, raw = true)
    @test !isempty(feats)
    @test all(f -> f.warp_idx == 1, feats)
    @test all(f -> f.s_xy == 0 && isapprox(f.s_xx, f.s_yy; rtol = 1.0e-6), feats)
    @test all(f -> isapprox(f.s_xx, f.σ; rtol = 1.0e-5), feats)
end

@testitem "plan reuse + determinism" setup = [Fixtures] begin
    a = asift_detect(IMG; nms = :none, raw = true)
    b = asift_detect(IMG; nms = :none, raw = true)   # cached plan, same input
    @test length(a) == length(b)
    ka = sort([(f.x, f.y, f.σ) for f in a])
    kb = sort([(f.x, f.y, f.σ) for f in b])
    @test all(k1 == k2 for (k1, k2) in zip(ka, kb))
    clear_asift_plans!()
    @test isempty(VisualGeometryFeatures._ASIFT_PLAN_CACHE)
end

@testitem "CPU launches: static, dynamic inside a threaded loop, same features" setup = [Fixtures] begin
    using VisualGeometryFeatures: _CPUKernel, _in_threaded_region
    # the fallback, against Base's own error: `@threads :static` throws inside
    # any `@threads` loop (also on one thread), and the launch runs dynamically
    # instead
    ran = Symbol[]
    static() = (Threads.@threads :static for _ in 1:2; end; push!(ran, :static))
    _CPUKernel(static, () -> push!(ran, :dynamic))()
    @test ran == [:static]
    empty!(ran)
    inside = Bool[]
    Threads.@threads for _ in 1:1
        push!(inside, _in_threaded_region())
        _CPUKernel(static, () -> push!(ran, :dynamic))()
    end
    @test inside == [true] && ran == [:dynamic]
    # a static launch that loses the race to another loop: the error is caught
    # (the region check saw none, then the launch threw) and the launch runs
    # dynamically
    empty!(ran)
    entered = Base.Event(); release = Base.Event(); other = Ref{Task}()
    function racing()
        other[] = Threads.@spawn Threads.@threads for _ in 1:1
            notify(entered); wait(release)
        end
        wait(entered)                     # another task's loop is now running
        return static()
    end
    _CPUKernel(racing, () -> push!(ran, :dynamic))()
    notify(release); wait(other[])
    @test ran == [:dynamic]
    # any other error propagates
    @test_throws ArgumentError _CPUKernel(() -> throw(ArgumentError("x")), () -> nothing)()
    # the detector from inside a threaded loop and from concurrent tasks: the
    # features and covariances of a plain call, bit for bit (pool order aside)
    key(r) = sort(
        [
            (f.x, f.y, f.σ, f.intensity, f.s_xx, f.s_xy, f.s_yy, f.warp_idx, Tuple(c))
                for (f, c) in zip(r.features, r.cov)
        ]; by = x -> x[1:8]
    )
    det(nms) = asift_detect(
        IMG; device = KA.CPU(), nms, raw = true, uncertainty = FeatureCovariance()
    )
    for nms in (:none, :mxm)
        ref = key(det(nms))
        nested = Vector{Any}(undef, 2)
        Threads.@threads for i in 1:2
            nested[i] = key(det(nms))
        end
        @test all(isequal(ref, x) for x in nested)
        tasks = [Threads.@spawn key(det(nms)) for _ in 1:3]
        @test all(isequal(ref, fetch(t)) for t in tasks)
    end
end

@testitem "pyramid tile lists: active and masked tiles partition each level" setup = [Fixtures] begin
    using VisualGeometryFeatures: _tile_lists, _dilate_tiles
    triples(v) = Set((v[i], v[i + 1], v[i + 2]) for i in 1:3:length(v))
    Wo, Ho = 100, 70
    aw = [100, 40, 1]; ah = [70, 33, 70]
    act, msk = _tile_lists(Wo, Ho, aw, ah)
    A = triples(act); M = triples(msk)
    @test isempty(A ∩ M) && length(act) + length(msk) == 3 * 4 * 3 * 3
    # active: the tile's first pixel inside the warp's bounds
    @test all((32bx + 1 <= aw[l] && 32by + 1 <= ah[l]) for (bx, by, l) in A)
    @test !any((32bx + 1 <= aw[l] && 32by + 1 <= ah[l]) for (bx, by, l) in M)
    @test length(A) == 4 * 3 + 2 * 2 + 3
    # the dilation adds every in-grid 8-neighbour of an active tile, same warp
    D = triples(_dilate_tiles(act, Wo, Ho))
    @test A ⊆ D && all(0 <= bx < 4 && 0 <= by < 3 for (bx, by, _) in D)
    @test (2, 2, 2) in D && (3, 2, 2) ∉ D && (1, 0, 3) in D && (2, 0, 3) ∉ D
end

# The ComputeBackend/KABackend/CpuBackend/ka_device wrapper vocabulary was
# removed from VGC — devices are plain KernelAbstractions tokens now
# (KA.CPU(), CUDABackend()); the old re-export testset asserted names that
# no longer exist anywhere in the stack.
@testitem "backends (KernelAbstractions vocabulary)" setup = [Fixtures] begin
    @test KA.CPU() isa KA.Backend
end

@testitem "nn_match on KA.CPU() is match" setup = [Fixtures] begin
    # one matcher: device = nothing is KA.CPU(), and `match` is the same call
    d1 = rand(Float32, 40, 128); d2 = rand(Float32, 50, 128)
    m = match_key(VisualGeometryFeatures.match(d1, d2, MutualNN()))
    @test !isempty(m)
    @test match_key(nn_match(d1, d2, MutualNN())) == m
    @test match_key(nn_match(d1, d2, MutualNN(); device = KA.CPU())) == m
end

@testitem "gray_f32 / gray_u8 — the input ladders" setup = [Fixtures] begin
    using VisualGeometryFeatures: gray_f32, gray_u8
    f = rand(Gray{Float32}, 4, 5)
    @test gray_f32(f) === f                                  # untouched
    m = rand(Float32, 4, 5)
    @test gray_f32(m) === m                                  # a bare device-form matrix is returned as is
    rgb = rand(RGB{N0f8}, 4, 5)
    g = gray_f32(rgb)
    @test g isa Matrix{Gray{Float32}} && size(g) == (4, 5)
    u = rand(UInt8, 4, 5)
    @test gray_u8(u) === u
    g8 = rand(Gray{N0f8}, 4, 5)
    b = gray_u8(g8)
    @test b isa Matrix{UInt8} && b == reinterpret(UInt8, g8)
    @test pointer(b) != pointer(reinterpret(UInt8, g8))      # a copy the caller owns
    @test gray_u8(rgb) == reinterpret(UInt8, Gray{N0f8}.(rgb))
    @test gray_u8(f) isa Matrix{UInt8}                        # quantised, by design
end

@testitem "gray_u8 owns its bytes" setup = [Fixtures] begin
    # `gray_u8` used to `unsafe_wrap` a Gray{N0f8} Array's memory without
    # keeping the image alive, so a temporary's pixels were freed and reused
    # while the bytes were still read: 15 782 of 4 000 000 intact after one GC,
    # and in a fresh worker process this very body crashed it (2026-09-15).
    # Every colorant but Gray{N0f8} reaches it through exactly such a temporary.
    using VisualGeometryFeatures: gray_u8
    gray_temp() = gray_u8(Gray{N0f8}.(fill(Gray{N0f8}(0.5), 2000, 2000)))
    rgb_temp() = gray_u8(fill(RGB{N0f8}(0.5, 0.5, 0.5), 2000, 2000))
    for make in (gray_temp, rgb_temp)
        u = make()
        GC.gc(true)
        junk = [rand(UInt8, 2000, 2000) for _ in 1:12]   # reuse whatever was freed
        @test all(==(0x80), u)                          # N0f8(0.5) is the byte 0x80
    end
end

@testitem "batched_rows — the describe backends' chunking skeleton" setup = [Fixtures] begin
    using VisualGeometryFeatures: batched_rows
    calls = Int[]
    out = batched_rows(10, 4) do offset, n
        push!(calls, n)
        Float32[offset + i + 10j for i in 1:n, j in 0:2]   # n × 3, any shape with n*3 elements works
    end
    @test size(out) == (10, 3) && calls == [4, 4, 2]
    @test out[:, 1] == Float32.(1:10) && out[7, 3] == 27
    @test batched_rows((o, n) -> zeros(Float32, n, 2), 5, 0) |> size == (5, 2)   # max_batch <= 0: one chunk
    @test size(batched_rows((o, n) -> error("never"), 0, 4)) == (0, 0)
    # retry_single: a failing batch drops to single rows for the rest of the run
    seen = Int[]
    out = batched_rows(6, 4; retry_single = true) do offset, n
        n > 1 && error("arena full")
        push!(seen, offset + 1)
        Float32[offset + 1;;]
    end
    @test vec(out) == Float32.(1:6) && seen == collect(1:6)
end

@testitem "detector input keeps float precision" setup = [Fixtures] begin
    # `_asift_input` used to send every non-Gray colorant through
    # `Gray{N0f8}`, so `RGB{Float32}` silently lost 8 bits while
    # `Gray{Float32}` did not — the asymmetry existed only because `{<:Gray}`
    # is a more specific method. Fixed-point sources must still take the
    # UInt8 upload path.
    g8 = Gray{N0f8}.(fill(Gray(0.5f0), 4, 4))
    rgb8 = RGB{N0f8}.(fill(RGB(0.5f0, 0.5f0, 0.5f0), 4, 4))
    @test eltype(VisualGeometryFeatures._asift_input(g8)) === UInt8
    @test eltype(VisualGeometryFeatures._asift_input(rgb8)) === UInt8
    @test eltype(VisualGeometryFeatures._asift_input(Gray{Float32}.(g8))) === Float32

    fine = fill(RGB{Float32}(0.5f0, 0.5f0, 0.5f0), 4, 4)
    fine[1, 1] = RGB{Float32}(0.5f0 + 1.0f0 / 512, 0.5f0, 0.5f0)   # finer than N0f8's 1/255
    out = VisualGeometryFeatures._asift_input(fine)
    @test eltype(out) === Float32
    @test out[1, 1] != out[2, 2]          # would collapse to one byte if quantised
end

@testitem "ASIFT plan cache is bounded (LRU)" setup = [Fixtures] begin
    # An unbounded cache turns "N distinct resolutions" into "N resident
    # pyramids" — measured at 11.09 GiB for one 1600x1200 plan with the 25-tilt
    # schedule, which is how a 13-resolution corpus reached a 173 GiB pool.
    # Evicted plans are released eagerly via KA.unsafe_free!.
    clear_asift_plans!()
    ts1 = [(1.0, 0.0)]                       # single tilt: small, fast plans
    try
        set_max_cached_plans!(1)
        for (w, h) in ((64, 48), (48, 64), (80, 60), (64, 48))
            VisualGeometryFeatures._asift_plan(w, h, KA.CPU(), ts1, false)
            @test length(VisualGeometryFeatures._ASIFT_PLAN_CACHE) == 1
        end
        set_max_cached_plans!(3)
        clear_asift_plans!()
        for (i, (w, h)) in enumerate(((64, 48), (48, 64), (80, 60), (96, 72)))
            VisualGeometryFeatures._asift_plan(w, h, KA.CPU(), ts1, false)
            @test length(VisualGeometryFeatures._ASIFT_PLAN_CACHE) == min(i, 3)
        end
        @test_throws ArgumentError set_max_cached_plans!(0)
    finally
        set_max_cached_plans!(1)
        clear_asift_plans!()
    end
end

@testitem "stacked multi-gallery matcher ≡ per-pair match" setup = [Fixtures] begin
    # Hoisted from BlobBoards: per gallery, nn_match_stacked must reproduce
    # `match(gallery, features, strategy)` exactly — it is the same matcher with
    # the galleries as blocks of its rows: MutualNN reverse checks are
    # block-local, KnnMatch takes any k (it was capped at 4 by a register
    # top-4 kernel of its own).
    using Random: MersenneTwister
    using LinearAlgebra: normalize
    rng = MersenneTwister(42)
    mkdesc(n) = permutedims(reduce(hcat, [Float32.(normalize(randn(rng, 128))) for _ in 1:n]))
    gals = Dict("a" => mkdesc(40), "b" => mkdesc(25), "c" => mkdesc(60))
    F = mkdesc(80)
    key3(ms) = sort([(m.first, m.second, round(m.score; digits = 5)) for m in ms])

    md = nn_match_stacked(MutualNN(), gals, F; backend = KA.CPU(), min_similarity = 0.1)
    md3 = nn_match_stacked(KnnMatch(3), gals, F; backend = KA.CPU(), min_similarity = 0.1)
    md6 = nn_match_stacked(KnnMatch(6), gals, F; backend = KA.CPU(), min_similarity = 0.1)
    for (k, G) in gals
        @test key3(md[k]) == key3(VisualGeometryFeatures.match(G, F, MutualNN(); min_similarity = 0.1))
        @test key3(md3[k]) == key3(VisualGeometryFeatures.match(G, F, KnnMatch(3); min_similarity = 0.1))
        @test key3(md6[k]) == key3(VisualGeometryFeatures.match(G, F, KnnMatch(6); min_similarity = 0.1))
    end
    # Float64 galleries stack, and match, in Float64
    gals64 = Dict(k => Float64.(G) for (k, G) in gals)
    @test eltype(StackedDescriptorIndex(KA.CPU(), gals64).d_all) == Float64
    md64 = nn_match_stacked(MutualNN(), gals64, Float64.(F); backend = KA.CPU(), min_similarity = 0.1)
    for (k, G) in gals64
        @test match_key(md64[k]) == match_key(VisualGeometryFeatures.match(G, Float64.(F), MutualNN(); min_similarity = 0.1))
    end

    # Prebuilt index reuse is identical to per-call construction.
    idx = StackedDescriptorIndex(KA.CPU(), gals)
    md_idx = nn_match_stacked(
        MutualNN(), gals, F; backend = KA.CPU(),
        index = idx, min_similarity = 0.1
    )
    for k in keys(gals)
        @test key3(md_idx[k]) == key3(md[k])
    end

    # Chunking the score matrix must not change WHICH matches are selected.
    # Chunks are ROW ranges — a gallery may span several (one large target
    # is one gallery whose block alone can exceed the card) and the reverse
    # sweep merges its per-column best across them. `chunk_bytes = 1` forces
    # one row per chunk (the pathological split: every gallery in pieces);
    # 7 rows per chunk cuts galleries at arbitrary places; the default path
    # uses a single chunk here. Scores are compared with a tolerance, not
    # bitwise: splitting the GEMM changes cuBLAS's accumulation order, so
    # entries differ by ~1 ulp — the same order as the stacked-vs-per-pair
    # difference asserted above.
    pairs_of(ms) = sort([(m.first, m.second) for m in ms])
    for (strat, base) in ((MutualNN(), md), (KnnMatch(3), md3)), cb in (1, 7 * size(F, 1) * sizeof(Float32))
        split = nn_match_stacked(
            strat, gals, F; backend = KA.CPU(),
            min_similarity = 0.1, chunk_bytes = cb
        )
        for k in keys(gals)
            @test pairs_of(split[k]) == pairs_of(base[k])
            sb = Dict((m.first, m.second) => m.score for m in base[k])
            for m in split[k]
                @test isapprox(m.score, sb[(m.first, m.second)]; atol = 1.0f-5)
            end
        end
    end

    # A prebuilt index and `galleries` need NOT agree: the index defines the
    # matched set, `galleries` only the key type. BlobBoards depends on this —
    # it narrows its gallery dict as boards are accepted while keeping ONE
    # index over the full set, and restaging per call would cost a full
    # descriptor upload. Documented in the docstring; pinned here so an
    # assert-or-intersect "fix" cannot silently match fewer galleries.
    narrowed = Dict("a" => gals["a"])
    md_narrow = nn_match_stacked(
        MutualNN(), narrowed, F; backend = KA.CPU(),
        index = idx, min_similarity = 0.1
    )
    @test sort(collect(keys(md_narrow))) == sort(collect(keys(gals)))
    for k in keys(gals)
        @test key3(md_narrow[k]) == key3(md[k])
    end

    # resolve_one_to_one: best-first, each endpoint at most once; MutualNN
    # sets (already one-to-one) are returned with the same content.
    ms = md3["c"]
    r = resolve_one_to_one(ms, size(gals["c"], 1), size(F, 1))
    @test length(unique(m.first for m in r)) == length(r)
    @test length(unique(m.second for m in r)) == length(r)
    @test issorted([m.score for m in r]; rev = true)
    @test key3(resolve_one_to_one(md["a"], size(gals["a"], 1), size(F, 1))) == key3(md["a"])

    # resolve_one_to_one! respects pre-seeded masks and chains lists.
    used_a = falses(10); used_b = falses(10)
    used_a[3] = true
    kept = resolve_one_to_one!(
        used_a, used_b, [(3, 1), (2, 1), (2, 5)];
        a = c -> c[1], b = c -> c[2]
    )
    @test kept == [2]   # (3,1) blocked by seed; (2,1) takes a=2,b=1; (2,5) blocked by a=2

    # "Best" follows the metric: the lowest score wins for a distance. It sorted
    # by descending score regardless, so L2 kept each point's worst candidate.
    near = NNMatch(1, 2, 0.1f0, Inf32)
    far = NNMatch(1, 1, 0.9f0, Inf32)
    endpoints = r -> [(m.first, m.second) for m in r]
    @test endpoints(resolve_one_to_one([far, near], 1, 2; metric = L2())) == [(1, 2)]
    @test endpoints(resolve_one_to_one([far, near], 1, 2)) == [(1, 1)]
end

# The ratio rule is s_2nd / s_best while s_best > 0, and Inf32 once it is not:
# a Lowe ratio is a ratio of magnitudes, undefined for a non-positive best
# similarity, and Inf32 is the value no ratio test admits. Only min_similarity ≤ 0
# keeps such a match at all, so only it reaches the rule; the item above never
# does. Both matchers take it from the one `_ratio`, and this pins both branches
# through `match`, `nn_match_stacked` and the written-out reference.
@testitem "ratios and the ratio test when min_similarity ≤ 0" setup = [Fixtures] begin
    using Random: MersenneTwister
    VGF = VisualGeometryFeatures
    # Entries k/64 with |k| ≤ 64: every dot product is exact in Float32 whatever
    # the GEMM's blocking or summation order, a device's included, so the paths
    # must agree bit for bit — ratios too, not just indices and rounded scores.
    rng = MersenneTwister(5)
    D = 16
    q(A) = Float32.(A) ./ 64.0f0
    F = q(vcat([64 zeros(Int, 1, D - 1)], rand(rng, 1:64, 29, D)))  # e₁, then the positive orthant
    gals = Dict(
        "pos" => q(rand(rng, 1:64, 12, D)),                          # every score > 0
        "neg" => q(-rand(rng, 1:64, 12, D)),                         # every score < 0
        "mixed" => q(rand(rng, -64:64, 20, D)),                      # both signs
        "zero" => q(hcat([0], -rand(rng, 1:64, 1, D - 1))),          # best exactly 0 (at e₁), the rest < 0
    )
    # No ties where an index is chosen (a row's top 4, a gallery column's
    # best), so agreement cannot hang on tie-breaking.
    untied(v, n) = allunique(sort(vec(v); rev = true)[1:min(n, length(v))])
    for G in values(gals)
        S = G * F'
        @test all(i -> untied(S[i, :], 4), axes(S, 1))
        @test all(j -> untied(S[:, j], 2), axes(S, 2))
    end
    # The POSITIVE branch, unchanged: a positive best keeps the quotient, a
    # negative value included — a runner-up worse than orthogonal makes the best
    # MORE distinctive, and such a match belongs through a ratio test.
    pos = VGF.match(gals["pos"], F, NN(); min_similarity = -Inf)
    @test !isempty(pos) && all(m -> m.score > 0 && isfinite(m.ratio), pos)
    mixed = VGF.match(gals["mixed"], F, NN(); min_similarity = -Inf)
    @test any(m -> m.score > 0 && m.ratio < 0, mixed)          # positive best, negative runner-up
    @test all(m -> m.score > 0 ? isfinite(m.ratio) : m.ratio === Inf32, mixed)

    # The NON-POSITIVE branch: the ratio is undefined, so it is Inf32. Both
    # cases are present — every score negative, and a best of exactly 0.
    neg = VGF.match(gals["neg"], F, MutualNN(); min_similarity = -Inf)
    @test !isempty(neg) && all(m -> m.score < 0 && m.ratio === Inf32, neg)
    zero_match = only(VGF.match(gals["zero"], F, NN(); min_similarity = 0.0))
    @test zero_match.score == 0 && zero_match.ratio === Inf32

    # …and so no ratio test admits one. This is what the rule buys: the plain
    # quotient made a best of exactly 0 report −Inf, which satisfied EVERY
    # threshold, and made negative bests report s_2nd/s_best ≥ 1, which a
    # threshold above 1 admitted.
    for τ in (0.8, 1.0, 2.0)
        @test isempty(VGF.match(gals["neg"], F, NN(); min_similarity = -Inf, ratio_threshold = τ))
        @test isempty(VGF.match(gals["zero"], F, NN(); min_similarity = -Inf, ratio_threshold = τ))
    end

    for min_similarity in (0.0, -2.0, -Inf), strat in (NN(), MutualNN(), KnnMatch(3))
        st = nn_match_stacked(strat, gals, F; backend = KA.CPU(), min_similarity)
        for (k, G) in gals
            @test isequal(match_key(st[k]), match_key(VGF.match(G, F, strat; min_similarity)))
            @test isequal(match_key(VGF.match(G, F, strat; min_similarity)), ref_match(G * F', strat; min_similarity))
        end
    end
end

# The extrema pool grows instead of truncating: a plan sized far below what
# the frame produces returns exactly what a plan sized above it does.
@testitem "asift extrema pool: growth on overflow is lossless" setup = [Fixtures] begin
    # a dense field: many blobs across scales
    blobs = [
        (20 + 11i, 14 + 13j, 2.0 + 0.5 * ((i + j) % 5), 0.3 * (isodd(i + j) ? 1 : -1))
            for i in 0:12, j in 0:16
    ]
    # `asift_detect!` takes the plan's input form: H×W bytes (`gray_u8`)
    img = gray_u8(synth_blobs(200, 260, blobs))
    key(f) = (round(f.x; digits = 3), round(f.y; digits = 3), round(f.σ; digits = 3), f.warp_idx)
    tilts = VisualGeometryFeatures._imas_tilts_25()[1:5]   # growth is schedule-independent
    big = AsiftPlan(260, 200; device = KA.CPU(), tilts, capacity = 1 << 20)
    # two warps per chunk, so later chunks grow a pool that already holds features
    small = AsiftPlan(260, 200; device = KA.CPU(), tilts, capacity = 64, chunk = 2)
    pool_big = asift_detect!(big, img; nms = :none, intensity_thr = 0.01)
    pool_small = asift_detect!(small, img; nms = :none, intensity_thr = 0.01)
    @test length(pool_big) > 64                       # the small plan had to grow
    @test sort(key.(pool_small)) == sort(key.(pool_big))
    # the survivors of the cross-warp NMS agree as well, and a grown plan is
    # reusable on the next frame
    d_big = asift_detect!(big, img; nms = :ellipse, intensity_thr = 0.01)
    d_small = asift_detect!(small, img; nms = :ellipse, intensity_thr = 0.01)
    @test sort(key.(d_small)) == sort(key.(d_big))
    @test !isempty(d_big)
    # a growth whose allocation fails (9 × 2^60 Float32) leaves an EMPTY pool,
    # never released buffers, and the next frame grows it back
    pool_names = propertynames(VisualGeometryFeatures._pool_buffers(KA.CPU(), 0))
    @test_throws ArgumentError VisualGeometryFeatures._grow_pool!(small, 2^59 + 1, 100)
    @test all(nm -> isempty(getfield(small, nm)), pool_names)
    @test sort(key.(asift_detect!(small, img; nms = :none, intensity_thr = 0.01))) ==
        sort(key.(pool_big))
    # the pool buffers and the keypoint-covariance buffers (grown alongside them
    # when in use) are exactly the plan's non-const fields: `_grow_pool!` can
    # replace every one of them, and nothing else
    P = typeof(small)
    @test Set(nm for nm in fieldnames(P) if !isconst(P, nm)) ==
        Set(pool_names) ∪ Set((:rawcov, :featcov, :covh))
end

@testitem "ASIFT polarity: is_dark / is_light follow the response sign" setup = [Fixtures] begin
    # asift_detections wrote laplacian_scale_score = 0.0 — the field the polarity
    # helpers read — so every ASIFT detection was neither dark nor light.
    dets = asift_detect(IMG; nms = :none)
    @test any(is_dark, dets) && any(is_light, dets)
    @test all(d -> is_dark(d) == (d.metadata.response > 0), dets)
    @test all(d -> is_light(d) == (d.metadata.response < 0), dets)
    @test count(is_dark, dets) == count(d -> d.metadata.response > 0, dets)
end

@testitem "Detection(frame, response; polarity) vocabulary" setup = [Fixtures] begin
    using VisualGeometryCore: Disc
    f = Disc(10.0, 20.0, 3.0)
    @test Detection(f, 0.2; polarity = :dark).metadata.laplacian_scale_score == 1.0
    @test Detection(f, -0.2; polarity = :light).metadata.laplacian_scale_score == -1.0
    @test Detection(f, -0.2; polarity = :bright).metadata.laplacian_scale_score == -1.0
    @test isnan(Detection(f, 0.2).metadata.laplacian_scale_score)
    @test_throws ArgumentError Detection(f, 0.2; polarity = :drak)   # was NaN, silently
    # VGC's accessor (renamed from `frame`); the `.frame` property stays
    d = Detection(f, 0.2; polarity = :dark)
    @test feature_frame(d) === d.frame === f
    @test feature_frame(f) === f
end

@testitem "argument validation: bad input raises instead of running wrong" setup = [Fixtures] begin
    plan = AsiftPlan(160, 96; device = KA.CPU(), tilts = [(1.0, 0.0)])
    # float input is W×H [x,y]; a transposed frame used to be scrambled silently
    @test_throws DimensionMismatch asift_detect!(plan, rand(Float32, 96, 160))
    @test_throws DimensionMismatch asift_detect!(plan, rand(Float32, 150, 96))
    @test asift_detect!(plan, rand(Float32, 160, 96)) isa Vector{ImasFeatureKA}
    # each of these threw InexactError from inside a kernel
    @test_throws ArgumentError asift_detect(IMG; nms = :mxm, band_octaves = 1.0e-9)
    @test_throws ArgumentError asift_detect(IMG; nms = :ellipse, cutoff = Inf)
    @test_throws ArgumentError asift_detect(IMG; nms = :ellipse, cutoff = NaN)
    @test_throws ArgumentError asift_detect(IMG; nms = :mxm, window_half = -1)
    pool = asift_detect(IMG; nms = :none, raw = true)
    @test_throws ArgumentError VisualGeometryFeatures.cross_warp_nms(
        pool, 160, 96; mode = :ellipse, cutoff = Inf
    )
    # too small for one octave: a clear error, not "reducing over an empty collection"
    @test_throws r"too small for the SIFT pyramid" AsiftPlan(30, 30; device = KA.CPU(), upsample = false)
    @test_throws r"too small for the SIFT pyramid" AsiftPlan(14, 14; device = KA.CPU())
    @test AsiftPlan(32, 32; device = KA.CPU(), tilts = [(1.0, 0.0)], upsample = false) isa AsiftPlan
end

@testitem "device memory query: a backend without one warns, then runs unchunked" setup = [Fixtures] begin
    VGF = VisualGeometryFeatures
    @test (@test_logs (:warn, r"no free-memory query") VGF.device_free_bytes(FakeGPU())) ==
        typemax(Int)
    @test (@test_logs (:warn, r"no free-memory query") VGF._chunk_warps(FakeGPU(), 25, 1, 0)) == 25   # no query: all warps resident
    @test VGF._chunk_warps(KA.CPU(), 25, typemax(Int) ÷ 4, 0) == 1   # the budget clamps to one
    @test VGF._chunk_warps(KA.CPU(), 25, 1, 3) == 3                 # an explicit chunk is honoured
    @test VGF.device_free_bytes(KA.CPU()) > 0
end

# There is one matcher — the KernelAbstractions kernels, on the CPU for `match`
# and on any backend for `nn_match` — and these check it against the matching
# rules written out directly (`ref_match`: full stable sorts, no kernels). The
# host twin they replaced ran partialsortperm per row and column.
@testitem "the one matcher ≡ the reference rules" setup = [Fixtures] begin
    using Random: MersenneTwister
    using LinearAlgebra: normalize
    VGF = VisualGeometryFeatures
    rng = MersenneTwister(7)
    unit(n) = permutedims(reduce(hcat, [Float32.(normalize(randn(rng, 32))) for _ in 1:n]))
    d1 = unit(40); d2 = unit(55)
    for (metric, similarity) in ((CosineSimilarity(), true), (L2(), false))
        dm = VGF.pairwise_matrix(metric, d1, d2)
        for strat in (NN(), MutualNN(), KnnMatch(1), KnnMatch(3), KnnMatch(7)),
                kw in (
                    (;), (; min_similarity = 0.2), (; max_distance = 1.1), (; ratio_threshold = 0.9),
                    (; min_similarity = 0.1, max_distance = 1.2, ratio_threshold = 0.95),
                )
            ms = VGF.match(d1, d2, strat; metric, kw...)
            @test match_key(ms) == ref_match(dm, strat; similarity, kw...)
            # best first: descending similarity, ascending distance
            @test issorted([m.score for m in ms]; rev = similarity)
            # a precomputed matrix gives the same, and so does every keyword on nn_match
            @test match_key(VGF.match(d1, d2, strat; metric, dm, kw...)) == match_key(ms)
            @test match_key(nn_match(d1, d2, strat; device = KA.CPU(), metric, kw...)) == match_key(ms)
        end
    end
    # Float64 descriptors are matched in Float64 (a score is stored as Float32)
    D1 = Float64.(d1); D2 = Float64.(d2)
    for strat in (NN(), MutualNN(), KnnMatch(3))
        @test match_key(VGF.match(D1, D2, strat)) == ref_match(D1 * D2', strat)
    end
    # a ratio exactly at the threshold is rejected (strict <)
    r = VGF.match(d1, d2, NN())[1].ratio
    at_r = VGF.match(d1, d2, NN(); ratio_threshold = r)
    @test all(m -> m.ratio < r, at_r)
    @test match_key(at_r) == ref_match(VGF.pairwise_matrix(CosineSimilarity(), d1, d2), NN(); ratio_threshold = r)
    # one gallery row: no second neighbour, so ratio Inf32 and any ratio test
    # rejects (the device path once reported −Inf32 and kept every match)
    one = unit(1)
    @test all(m -> m.ratio == Inf32, VGF.match(d1, one, NN()))
    @test isempty(VGF.match(d1, one, NN(); ratio_threshold = 0.8))
    @test_throws DimensionMismatch VGF.match(d1, d2; dm = zeros(Float32, 3, 3))
end

# One pairwise_matrix serves every backend (it had a host twin whose L2 used a
# unit-norm shortcut that disagreed with the device for unnormalized input).
@testitem "pairwise_matrix: exact for any descriptors, in their own precision" setup = [Fixtures] begin
    using Random: MersenneTwister
    rng = MersenneTwister(3)
    a = 3.0f0 .* randn(rng, Float32, 20, 16)   # deliberately not normalized
    b = randn(rng, Float32, 25, 16)
    l2 = VisualGeometryFeatures.pairwise_matrix(L2(), a, b)
    @test l2 == VisualGeometryFeatures.pairwise_matrix(L2(), a, b, KA.CPU())
    exact = [sqrt(sum(abs2, a[i, :] .- b[j, :])) for i in 1:20, j in 1:25]
    @test isapprox(l2, exact; rtol = 1.0e-3)
    # Float32 stays Float32 and Float64 stays Float64 — never narrowed
    @test eltype(l2) == Float32
    @test eltype(VisualGeometryFeatures.pairwise_matrix(CosineSimilarity(), a, b)) == Float32
    A = Float64.(a); B = Float64.(b)
    @test eltype(VisualGeometryFeatures.pairwise_matrix(CosineSimilarity(), A, B)) == Float64
    l2_64 = VisualGeometryFeatures.pairwise_matrix(L2(), A, B)
    @test eltype(l2_64) == Float64
    exact64 = [sqrt(sum(abs2, A[i, :] .- B[j, :])) for i in 1:20, j in 1:25]
    @test isapprox(l2_64, exact64; rtol = 1.0e-10)
    @test VisualGeometryFeatures.pairwise_matrix(CosineSimilarity(), A, B) ≈ A * B'
end

@testitem "batched_rows(retry_single = true) does not retry an interrupt" setup = [Fixtures] begin
    using VisualGeometryFeatures: batched_rows
    calls = Ref(0)
    interrupt(offset, n) = (calls[] += 1; throw(InterruptException()))
    @test_throws InterruptException batched_rows(interrupt, 4, 2; retry_single = true)
    @test calls[] == 1   # the bare `catch` used to fall back to single rows and call again
end

@testitem "seed_sigma has one authority: the plan" setup = [Fixtures] begin
    img = gray_u8(IMG)
    ts = [(1.0, 0.0)]
    # the tiled blur caps the tap radius, so seed_sigma can only go a little above 1.6
    @test_throws ArgumentError AsiftPlan(160, 96; device = KA.CPU(), tilts = ts, seed_sigma = 2.0)
    plan = AsiftPlan(160, 96; device = KA.CPU(), tilts = ts, seed_sigma = 1.4)
    # the pyramid is blurred for the plan's value: asift_detect! has no keyword
    # that could restate it
    @test_throws MethodError asift_detect!(plan, img; seed_sigma = 1.4)
    mine = asift_detect!(plan, img; nms = :none)
    k(f) = (round(f.x; digits = 3), round(f.y; digits = 3), round(f.σ; digits = 3))
    # asift_detect builds (and caches) its plan with the same seed_sigma
    @test sort(k.(asift_detect(IMG; tilts = ts, seed_sigma = 1.4, nms = :none, raw = true))) ==
        sort(k.(mine))
    @test sort(k.(asift_detect(IMG; tilts = ts, nms = :none, raw = true))) != sort(k.(mine))
    clear_asift_plans!()
end

@testitem "Detection JSON round trip: every frame type, all metadata" setup = [Fixtures] begin
    using JSON3
    using VisualGeometryCore: Disc, OrientedDisc, EllipticFrame, AffineFrame
    meta = (response = 0.25, edge_score = 0.0, laplacian_scale_score = 1.0, warp_idx = 7)
    frames = (
        Disc(10.5, 20.25, 3.0), OrientedDisc(10.5, 20.25, 3.0, 0.7),
        EllipticFrame([10.5, 20.25], [2.0 0.3; 0.3 1.5]),
        AffineFrame([10.5, 20.25], [1.2 0.1; -0.2 0.9]),
    )
    for f in frames
        d = Detection(f, meta)
        # writing threw (the package's own `pairs` shadowed Base.pairs); reading
        # failed for every frame but Disc and dropped warp_idx
        r = JSON3.read(JSON3.write(d), Detection)
        @test r.frame == d.frame
        @test r.metadata == d.metadata
        @test typeof(r.metadata) == typeof(d.metadata)   # 0.0 and 1.0 stay Float64
    end
    # JSON3 reads through a Dict: documented fields first, the rest by name
    e = Detection(frames[1], (zeta = 1, response = 2.0, alpha = "a"))
    @test JSON3.read(JSON3.write(e), Detection).metadata == (response = 2.0, alpha = "a", zeta = 1)
    a = first(asift_detect(IMG))
    r = JSON3.read(JSON3.write(a), Detection)
    @test r.frame == a.frame && r.metadata == a.metadata && typeof(r) == typeof(a)
end

@testitem "nn_match_stacked has its docstring" setup = [Fixtures] begin
    # it sat above `const _SCORE_BUDGET`, so it documented the constant
    @test occursin("LOCAL indices", string(@doc nn_match_stacked))
end

@testitem "StructTypes.construct for quantities and Size2 is unambiguous" setup = [Fixtures] begin
    # untyped second arguments overlapped StructTypes' own construct methods
    # (identity, String, Nothing): construct(Size2, ::Size2) threw an
    # ambiguity error instead of returning its argument
    using StructTypes, JSON3
    using VisualGeometryCore: Size2, mm
    SQ = VisualGeometryFeatures.ScalarOrQuantity
    @test isempty(Test.detect_ambiguities(VisualGeometryFeatures; recursive = true))
    s = Size2(width = 3.0mm, height = 4.0mm)
    @test StructTypes.construct(Size2, s) === s
    @test StructTypes.construct(Size2, JSON3.read(JSON3.write(s))) == s
    @test StructTypes.construct(Size2, Dict{String, Any}("width" => 640, "height" => 480)) ==
        Size2(width = 640, height = 480)
    @test StructTypes.construct(SQ, 5) == 5
    @test StructTypes.construct(SQ, Dict{String, Any}("value" => 2.0, "unit" => "mm")) == 2.0mm
end

@testitem "match does not shadow Base.match" setup = [Fixtures] begin
    # exported, it made an unqualified match(r"...", s) ambiguous in every module
    # that did `using VisualGeometryFeatures`
    @test !Base.isexported(VisualGeometryFeatures, :match)
    @test Base.ispublic(VisualGeometryFeatures, :match)
    m = Module()
    @test Core.eval(
        m, Expr(
            :toplevel, :(using VisualGeometryFeatures),
            :(match(r"b", "abc"))
        )
    ) isa RegexMatch
end

@testitem "FileCache: the JLD2 extension's get-or-compute" begin
    # The extension did not load under Pkg.test until JLD2 joined the test
    # target; BlobBoards' descriptor gallery cache is this code.
    using JLD2
    @test Base.get_extension(VisualGeometryFeatures, :VisualGeometryFeaturesJLD2Ext) !== nothing
    k = cache_key("board", 3, 2.5)
    @test k == cache_key("board", 3, 2.5) && k != cache_key("board", 3, 2.6)
    @test open_cache(nothing) === nothing && close_cache(nothing) === nothing
    path = joinpath(mktempdir(), "cache.jld2")
    calls = Ref(0)
    c = open_cache(path)
    @test c isa FileCache
    @test get!(() -> (calls[] += 1; [1.0, 2.0]), c, k) == [1.0, 2.0]
    @test get!(() -> (calls[] += 1; nothing), c, k) == [1.0, 2.0]   # hit: f not called
    @test calls[] == 1
    close_cache(c)
    c = open_cache(path)                                          # persisted across handles
    @test get!(() -> error("recomputed"), c, k) == [1.0, 2.0]
    close_cache(c)
    # a path that cannot be opened warns and returns nothing; callers then compute
    # directly (JLD2 logs its own MmapIO fallback warning first, hence :any)
    @test (@test_logs (:warn, r"Cache unavailable") match_mode = :any open_cache(joinpath(path, "sub", "x.jld2"))) === nothing
end

@testitem "Makie extension: detections!, matches!, matches" setup = [Fixtures] begin
    # The extension did not load under Pkg.test until Makie joined the test
    # target. No backend is needed: these build scene graphs, nothing renders.
    using Makie: Makie, Figure, Axis
    using VisualGeometryCore: matches, matches!, Attributed, PrecMat, Point2
    @test Base.get_extension(VisualGeometryFeatures, :VisualGeometryFeaturesMakieExt) !== nothing
    dets = asift_detect(IMG; nms = :ellipse)
    ax = Axis(Figure()[1, 1])
    detections!(ax, dets; color_by = :response)
    @test !isempty(ax.scene.plots)
    @test detections!(ax, dets[1:0]) === nothing
    @test_throws ArgumentError detections!(ax, dets; color_by = :bogus)
    cs = [Point2(1.0 + i, 2.0) => Point2(3.0, 4.0 + i) for i in 1:10]
    fig, (a, b) = matches(rand(Gray{N0f8}, 20, 30), rand(Gray{N0f8}, 20, 30), cs)
    @test fig isa Figure && !isempty(a.scene.plots) && !isempty(b.scene.plots)
    # stochastic endpoints: one σ-ellipse per drawn cspond, at most max_display
    Λ = PrecMat{2, Float64}(2.0, 0.1, 0.1, 3.0)
    scs = [Attributed(Point2(5.0 + i, 5.0), Λ) => Attributed(Point2(6.0, 7.0 + i), Λ) for i in 1:40]
    fig2 = Figure()
    a2 = Axis(fig2[1, 1]); b2 = Axis(fig2[1, 2])
    matches!(a2, b2, scs; max_display = 7)
    @test count(p -> p isa Makie.Poly, a2.scene.plots) == 7
    @test_throws ArgumentError matches!(a2, b2, scs; max_display = 0)
end
