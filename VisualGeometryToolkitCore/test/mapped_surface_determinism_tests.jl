@testitem "board deformation: a ray's hit is the same bits at every call site" begin
    using Test, StaticArrays, Random
    using ForwardDiff        # loaded, as in every worker that differentiates a hit
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, chart, normal, grid_nodes

    # One surface and one ray have one hit, whatever code calls `intersection`. Two copies of the root compiled into
    # one caller (two surfaces of the same numbers hit in one function) were measured 2–3 ulp apart on a composed
    # calibration (1 ray in 200; Julia 1.12/1.13, --check-bounds=yes, as in this worker). Every call here is checked
    # bit for bit against one reference: 200 rays × 50 calls = 10⁴ calls a surface, from five call sites.
    rng = Xoshiro(7)
    nodes = grid_nodes(4; lo = -0.6, hi = 0.6)
    X = EuclideanMap(RotMatrix{3}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.02, -0.01, 2.0))
    P = SimilarityMap(0.9, one(RotMatrix{3, Float64}), SVector(-0.4, -0.45, 0.0))
    rect = Rect2(-0.5, -0.5, 1.0, 1.0)
    rays = [(SVector(0.3randn(rng), 0.3randn(rng), 0.0), SVector(0.2randn(rng), 0.2randn(rng), 1.0)) for _ in 1:200]
    H = HeightField(nodes, 0.01 .* randn(rng, 16))
    T = ThinPlateSpline(nodes, 0.01 .* randn(rng, 16), 0.01 .* randn(rng, 16))
    AS = Aspect(0.01) ∘ Skew(-0.003)

    bits(r::Tuple) = (reinterpret(UInt64, r[1]), reinterpret.(UInt64, Tuple(r[2])))
    bits(x::StaticVector) = reinterpret.(UInt64, Tuple(x))
    @noinline reference(s, o, d) = intersection(s, o, d)
    # the sites: two surfaces of the same numbers hit in one function, compared in place or returned; the same
    # surface twice; a closure over the surface; a comprehension's filter
    @noinline pair(a, b, o, d) = (intersection(a, o, d), intersection(b, o, d))
    @noinline agree(a, b, o, d) = bits(intersection(a, o, d)) == bits(intersection(b, o, d))
    @noinline twice(a, o, d) = (intersection(a, o, d), intersection(a, o + zero(o), d))

    for (name, C, Pc) in (("H", H, nothing), ("H ∘ Aspect ∘ Skew", H ∘ AS, nothing), ("H ∘ Aspect ∘ Skew, placed", H ∘ AS, P),
                          ("H ∘ T", H ∘ T, nothing), ("H ∘ T, placed", H ∘ T, P))
        s = MappedSurface(X, C, Pc, rect)
        s′ = MappedSurface(X, C, Pc, rect)          # the same numbers, another object
        ref = [bits(reference(s, o, d)) for (o, d) in rays]
        @test count(r -> isfinite(reinterpret(Float64, r[1])), ref) > 50
        calls = differ = 0
        for (k, (o, d)) in enumerate(rays)
            for _ in 1:10
                a, b = pair(s, s′, o, d)
                differ += (bits(a) != ref[k]) + (bits(b) != ref[k])
                agree(s, s′, o, d) || (differ += 1)
                a, b = twice(s′, o, d)
                differ += (bits(a) != ref[k]) + (bits(b) != ref[k])
                differ += count(r -> bits(r) != ref[k], map(t -> intersection(t, o, d), (s, s′)))
                calls += 9
            end
        end
        differ += count(k -> bits(intersection(s′, rays[k]...)) != ref[k], [k for k in eachindex(rays) if bits(intersection(s, rays[k]...)) == ref[k]])
        calls += 2length(rays)
        @test calls >= 10_000
        @test differ == 0
        differ == 0 || @info "$name: $differ of $calls calls differ from the reference bits"
        # the chart lookup and the normal at a hit: the same root, one answer
        hits = [reinterpret(Float64, collect(r[2])) for r in ref if isfinite(reinterpret(Float64, r[1]))]
        cref = [bits(chart(s, SVector{3}(x))) for x in hits]
        @test all(bits(chart(s′, SVector{3}(x))) == cref[i] && bits(chart(s, SVector{3}(x))) == cref[i] for (i, x) in enumerate(hits))
        nref = [bits(normal(s, SVector{3}(x))) for x in hits]
        @test all(bits(normal(s′, SVector{3}(x))) == nref[i] for (i, x) in enumerate(hits))
    end
end
