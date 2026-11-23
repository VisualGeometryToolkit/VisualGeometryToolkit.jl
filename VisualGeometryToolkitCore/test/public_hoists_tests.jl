# Public names for private functions a downstream package (RadiometricCamera) copied or reached by private name:
# each is the function its own callers in Core already use, so these pin the public contract, not new arithmetic.

@testitem "normalized_point: K⁻¹𝐩, the pixel-frame half of backproject" begin
    using VisualGeometryToolkitCore: warm_allocated
    using Test
    using StaticArrays
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: normalized_point, KannalaBrandt, CameraCalibrationMat, convert_eltype

    K = SMatrix{3, 3, Float64}(600, 0, 0, 2.5, 610, 0, 320, 240, 1)       # skew 2.5: K⁻¹ is not two divisions
    pin = CameraModel(CameraCalibrationMat(K))
    fish = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    pixels = [SVector(x, y) for x in 3.25:97.5:1270.0, y in 1.5:83.0:950.0]
    for p in pixels
        x = normalized_point(pin.intrinsics, p)
        @test x isa SVector{2, Float64}
        @test x ≈ (K \ SVector(p[1], p[2], 1.0))[SOneTo(2)] atol = 1.0e-14
        # the one map backproject applies: the lens's bearing at it, bit for bit
        @test isequal(backproject(fish, p), backproject(fish.lens, normalized_point(fish.intrinsics, p)))
        @test isequal(backproject(pin, p), backproject(pin.lens, normalized_point(pin.intrinsics, p)))
    end
    # the element type follows the pixel's (a Float32 device camera)
    @test normalized_point(convert_eltype(Float32, fish).intrinsics, SVector(100.0f0, 50.0f0)) isa SVector{2, Float32}
    @test warm_allocated(normalized_point, fish.intrinsics, SVector(1.0, 2.0)) == 0
end

@testitem "with_parameters(::MappedSurface): the calibration rebuilt, every other member kept" begin
    using Test, StaticArrays, Random
    using Rotations: RotMatrix, RotXYZ
    using ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, with_parameters, parameter_vector, grid_nodes, convert_eltype

    rng = Xoshiro(7)
    nodes = grid_nodes(4; lo = -0.6, hi = 0.6)
    X = EuclideanMap(RotMatrix{3}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.02, -0.01, 2.0))
    P = SimilarityMap(0.9, one(RotMatrix{3, Float64}), SVector(-0.4, -0.45, 0.0))
    rect = Rect2(-0.5, -0.5, 1.0, 1.0)
    rays = [(SVector(0.3randn(rng), 0.3randn(rng), 0.0), SVector(0.2randn(rng), 0.2randn(rng), 1.0)) for _ in 1:200]
    # the same hit: equal numbers (a signed zero equal to its opposite), a miss on both. Exact since Core 0.9.3: the
    # root gives the same bits at every call site (mapped_surface_determinism_tests.jl).
    same((ta, xa), (tb, xb)) = isnan(ta) ? isnan(tb) && all(isnan, xb) : ta == tb && xa == xb
    H = HeightField(nodes, 0.01 .* randn(rng, 16))
    T = ThinPlateSpline(nodes, 0.01 .* randn(rng, 16), 0.01 .* randn(rng, 16))
    for (C, Pc) in ((H, nothing), (H, P), (H ∘ Aspect(0.01) ∘ Skew(-0.003), nothing), (H ∘ T, P))
        s = MappedSurface(X, C, Pc, rect)
        θ = parameter_vector(C) .+ 0.003 .* randn(rng, length(parameter_vector(C)))
        s2 = with_parameters(s, θ)
        built = MappedSurface(X, with_parameters(C, θ), Pc, rect)        # the checked builder from the parts
        @test s2 isa MappedSurface && s2.frame == s.frame && s2.map.t1 == s.map.t1   # the frame and the pose kept
        @test all(same(intersection(s2, o, d), intersection(built, o, d)) for (o, d) in rays)
        @test count(r -> isfinite(intersection(s2, r...)[1]), rays) > 50
        @test all(same(intersection(with_parameters(s, parameter_vector(C)), o, d), intersection(s, o, d)) for (o, d) in rays)
    end

    # RadiometricCamera's former form, bit for bit: every HeightField member rebuilt by its own with_parameters
    s = MappedSurface(X, H, P, rect)
    z = 0.005 .* randn(rng, 16)
    rc(m) = m isa HeightField ? with_parameters(m, z) : m
    by_members = MappedSurface(VisualGeometryToolkitCore.Cameras._map_members(rc, s.map), s.frame)
    @test all(isequal(intersection(with_parameters(s, z), o, d), intersection(by_members, o, d)) for (o, d) in rays)

    # Dual heights: the hit's value is the plain one's, and it carries partials
    zd = ForwardDiff.Dual{:h}.(z, 1.0)
    o, d = first(r for r in rays if isfinite(intersection(with_parameters(s, z), r...)[1]))
    td, xd = intersection(with_parameters(s, zd), o, d)
    @test ForwardDiff.value(td) ≈ intersection(with_parameters(s, z), o, d)[1] atol = 1.0e-14
    # a Float32 surface takes Float32 parameters
    s32 = convert_eltype(Float32, s)
    @test with_parameters(s32, Float32.(z)) isa MappedSurface
end

@testitem "intersection_at: the hit's partials at a known hit, no root" begin
    using VisualGeometryToolkitCore: warm_allocated
    using Test, StaticArrays, Random
    using Rotations: RotMatrix, RotXYZ
    using ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, intersection_at, with_parameters, grid_nodes, convert_eltype, surface

    rng = Xoshiro(11)
    nodes = grid_nodes(5; lo = -0.6, hi = 0.6)
    z = 0.01 .* randn(rng, 25)
    X = EuclideanMap(RotMatrix{3}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.02, -0.01, 2.0))
    rect = Rect2(-0.5, -0.5, 1.0, 1.0)
    rays = [(SVector(0.3randn(rng), 0.3randn(rng), 0.0), SVector(0.2randn(rng), 0.2randn(rng), 1.0)) for _ in 1:300]
    partials_of(v) = reduce(hcat, [collect(ForwardDiff.partials(c)) for c in v])
    for C in (HeightField(nodes, z), HeightField(nodes, z) ∘ (Aspect(0.01) ∘ Skew(-0.003)))
        s = MappedSurface(X, C, nothing, rect)
        H = C isa HeightField ? C : C.t1
        hits = [(o, d, intersection(s, o, d)...) for (o, d) in rays]
        @test count(h -> isfinite(h[3]), hits) > 100
        # plain parts: the supplied hit and its ray parameter; a miss stays a miss
        for (o, d, t, x) in hits
            ta, xa = intersection_at(s, o, d, x)
            @test isfinite(t) ? (abs(ta - t) <= 64eps() && xa == x) : (isnan(ta) && all(isnan, xa))
        end
        # Dual heights (25 partials), a Dual ray, both: intersection's value and partials to rounding
        zd = SVector{25}(ntuple(i -> ForwardDiff.Dual{:h}(H.h[i], ntuple(j -> Float64(i == j), 25)), 25))
        dual_field(m) = m isa HeightField ? with_parameters(m, zd) : m
        sd = MappedSurface(VisualGeometryToolkitCore.Cameras._map_members(dual_field, s.map), s.frame)
        worst_value = 0.0; worst_partial = 0.0; scale = 0.0
        for (o, d, t, x) in hits
            od = SVector{3}(ntuple(i -> ForwardDiff.Dual{:r}(o[i], ntuple(j -> Float64(i == j), 3)), 3))
            for (surf, org) in ((sd, o), (s, od))
                tr, xr = intersection(surf, org, d)
                ta, xa = intersection_at(surf, org, d, x)
                if !isfinite(t)
                    @test isnan(ForwardDiff.value(ta)) && all(isnan ∘ ForwardDiff.value, xa)
                    continue
                end
                worst_value = max(worst_value, abs(ForwardDiff.value(ta) - ForwardDiff.value(tr)), maximum(abs, ForwardDiff.value.(xa) - ForwardDiff.value.(xr)))
                worst_partial = max(worst_partial, maximum(abs, partials_of(xa) - partials_of(xr)), maximum(abs, collect(ForwardDiff.partials(ta)) - collect(ForwardDiff.partials(tr))))
                scale = max(scale, maximum(abs, partials_of(xr)))
            end
        end
        @test worst_value <= 64eps() && worst_partial <= 1.0e-12 * max(scale, 1.0)
        o, d, _, x = first(h for h in hits if isfinite(h[3]))
        @test (@inferred intersection_at(sd, o, d, x))[1] isa ForwardDiff.Dual && warm_allocated(intersection_at, sd, o, d, x) == 0
        # a miss handed over as intersection returns it
        nan = SVector(NaN, NaN, NaN)
        @test all(isnan ∘ ForwardDiff.value, intersection_at(sd, o, d, nan)[2]) && isnan(intersection_at(s, o, d, nan)[1])
        # Float32
        s32 = convert_eltype(Float32, s); o32, d32 = SVector{3, Float32}(o), SVector{3, Float32}(d)
        t32, x32 = intersection(s32, o32, d32)
        @test intersection_at(s32, o32, d32, x32) isa Tuple{Float32, SVector{3, Float32}}
    end
    # a surface without a method for the known hit: intersection itself
    q = Quad(SVector(0.0, 0.0, 2.0), one(SMatrix{3, 3, Float64}), Rect2(-0.5, -0.5, 1.0, 1.0))
    o, d = SVector(0.1, 0.05, 0.0), SVector(0.02, -0.03, 1.0)
    @test isequal(intersection_at(q, o, d, intersection(q, o, d)[2]), intersection(q, o, d))
end

@testitem "bilinear_weights: the texel block and fractions bilinear_sample reads" begin
    using VisualGeometryToolkitCore: warm_allocated
    using Test, StaticArrays, Random
    using VisualGeometryToolkitCore: bilinear_sample, bilinear_weights

    rng = Xoshiro(3)
    img = randn(rng, 4, 8)                       # H = 4 rows, W = 8 columns (the texel centres exact in binary)
    # the reference: the 2×2 texel block between the row and column interpolation weights
    blend(img, (i, j, fx, fy)) = (SVector(1 - fy, fy)' * SMatrix{2, 2}(img[j:(j + 1), i:(i + 1)]) * SVector(1 - fx, fx))
    uvs = [SVector(rand(rng), rand(rng)) for _ in 1:500]
    append!(uvs, [SVector(-0.3, 0.5), SVector(1.4, 1.2), SVector(0.0, 0.0), SVector(1.0, 1.0), SVector(NaN, 0.4)])
    for uv in uvs
        w = bilinear_weights(uv, (8, 4))
        i, j, fx, fy = w
        @test 1 <= i <= 7 && 1 <= j <= 3 && 0 <= fx <= 1 && 0 <= fy <= 1
        @test blend(img, w) ≈ bilinear_sample(img, uv) atol = 8eps()      # the lookup's own arithmetic
    end
    # a texel centre: that texel, fraction 0 (or the block before it at the last centre, fraction 1)
    @test bilinear_weights(SVector(2.5 / 8, 1.5 / 4), (8, 4)) == (3, 2, 0.0, 0.0)
    @test bilinear_weights(SVector(7.5 / 8, 3.5 / 4), (8, 4)) == (7, 3, 1.0, 1.0)
    @test bilinear_weights(SVector(NaN, 0.1), (8, 4))[[1, 3]] == (1, 0.0)
    @test_throws ArgumentError bilinear_weights(SVector(0.5, 0.5), (1, 4))
    @test warm_allocated(bilinear_weights, SVector(0.3, 0.6), (8, 4)) == 0
end

@testitem "is_inside_texel_centres: exactly where bilinear_weights does not extend the edge" begin
    using VisualGeometryToolkitCore: warm_allocated
    using Test, StaticArrays, Random, ForwardDiff
    using VisualGeometryToolkitCore: bilinear_weights, is_inside_texel_centres

    # Unclamped, the block and fractions reproduce the position's 0-based centre coordinates exactly
    # (x − x0 is exact for x ∈ [x0, x0 + 1]); clamping moves them. So the predicate must equal "reproduces".
    reproduces(uv, (W, H)) = ((i, j, fx, fy) = bilinear_weights(uv, (W, H));
        (i - 1) + fx == uv[1] * W - one(eltype(uv)) / 2 && (j - 1) + fy == uv[2] * H - one(eltype(uv)) / 2)
    for T in (Float32, Float64), (W, H) in ((8, 4), (2, 2), (24, 30))
        rng = Xoshiro(11)
        uvs = [SVector{2, T}(T(1.4) * rand(rng, T) - T(0.2), T(1.4) * rand(rng, T) - T(0.2)) for _ in 1:2000]
        # the edges, the corners and every texel centre (dyadic for W, H powers of two: exact), and one ulp past each edge
        cs(n) = [(T(k) - T(0.5)) / T(n) for k in 1:n]
        for x in cs(W), y in cs(H); push!(uvs, SVector(x, y)); end
        lo = (T(0.5) / T(W), T(0.5) / T(H)); hi = ((T(W) - T(0.5)) / T(W), (T(H) - T(0.5)) / T(H))
        for x in (lo[1], hi[1], prevfloat(lo[1]), nextfloat(hi[1]), T(0.5)), y in (lo[2], hi[2], prevfloat(lo[2]), nextfloat(hi[2]), T(0.5))
            push!(uvs, SVector(x, y))
        end
        for uv in uvs
            @test is_inside_texel_centres(uv, (W, H)) == reproduces(uv, (W, H))
        end
        if ispow2(W) && ispow2(H)       # the boundary itself, exactly: centres and edges in, one ulp past out
            @test all(is_inside_texel_centres(SVector(x, y), (W, H)) for x in cs(W), y in cs(H))
            @test is_inside_texel_centres(SVector(lo...), (W, H)) && is_inside_texel_centres(SVector(hi...), (W, H))
            @test is_inside_texel_centres(SVector(lo[1], hi[2]), (W, H)) && is_inside_texel_centres(SVector(hi[1], lo[2]), (W, H))
            @test !is_inside_texel_centres(SVector(prevfloat(lo[1]), T(0.5)), (W, H))
            @test !is_inside_texel_centres(SVector(nextfloat(hi[1]), T(0.5)), (W, H))
            @test !is_inside_texel_centres(SVector(T(0.5), prevfloat(lo[2])), (W, H))
            @test !is_inside_texel_centres(SVector(T(0.5), nextfloat(hi[2])), (W, H))
        end
        @test !is_inside_texel_centres(SVector(T(NaN), T(0.5)), (W, H)) && !is_inside_texel_centres(SVector(T(0.5), T(Inf)), (W, H))
        @test !is_inside_texel_centres(SVector(zero(T), zero(T)), (W, H)) && !is_inside_texel_centres(SVector(one(T), one(T)), (W, H))
    end
    # on the value under a Dual, as bilinear_weights chooses its block
    d = ForwardDiff.Dual(0.3, 1.0)
    @test is_inside_texel_centres(SVector(d, 0.5), (8, 4)) && !is_inside_texel_centres(SVector(ForwardDiff.Dual(0.01, 1.0), 0.5), (8, 4))
    @test is_inside_texel_centres(SVector(0.5, 0.5), (8, 4)) isa Bool
    @test_throws ArgumentError is_inside_texel_centres(SVector(0.5, 0.5), (1, 4))
    @test warm_allocated(is_inside_texel_centres, SVector(0.3, 0.6), (8, 4)) == 0
end

@testitem "julia_blocks: a page's Julia examples" begin
    using Test
    using VisualGeometryToolkitCore: julia_blocks

    page = joinpath(mktempdir(), "page.md")
    write(page, """
    # A page

    ```julia-repl
    pkg> add Foo
    ```

    ```julia
    using Foo
    x = 1
    ```

    ```python
    print(1)
    ```

    ```julia
    # Illustrative
    y = 2
    ```
    """)
    @test julia_blocks(page) == ["using Foo\nx = 1\n", "# Illustrative\ny = 2\n"]
    @test julia_blocks(page) isa Vector{String}
    write(page, "no code here\n")
    @test isempty(julia_blocks(page))
end
