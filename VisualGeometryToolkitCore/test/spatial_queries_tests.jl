# SpatialQueries: allocation-free intersection and nearest-neighbour queries on
# SpatialIndexing's R-tree (src/geometry/spatial_queries.jl), checked against
# brute force on random point and rectangle sets.

@testitem "spatial_queries" begin
    using Test
    using VisualGeometryToolkitCore.SpatialQueries
    using SpatialIndexing: SpatialIndexing, RTree, SpatialElem
    using Random

    const SI = SpatialIndexing
    rng = Xoshiro(11)

    # Point or box elements with ids 1:n; bulk-loaded (`bulk_load!`, or
    # SpatialIndexing's `load!`) or inserted one by one, at capacities that give
    # a single-leaf root, a twig root and a three-level tree.
    function random_rects(rng, n; boxes::Bool)
        return map(1:n) do _
            x, y = 1000 * rand(rng), 800 * rand(rng)
            w, h = boxes ? (20 * rand(rng), 20 * rand(rng)) : (0.0, 0.0)
            SI.Rect((x, y), (x + w, y + h))
        end
    end
    function build(rects; bulk, capacity::Int)
        tree = RTree{Float64, 2}(Int, Nothing; leaf_capacity = capacity, branch_capacity = capacity)
        if bulk === :bulk_load!
            bulk_load!(tree, [SpatialElem(r, j, nothing) for (j, r) in enumerate(rects)])
        elseif bulk === :load!
            SI.load!(tree, eachindex(rects); convertel = j -> SpatialElem(rects[j], j, nothing))
        else
            for (j, r) in enumerate(rects)
                SI.insert!(tree, r, j, nothing)
            end
        end
        return tree
    end
    overlaps(a, b) = all(i -> a.low[i] <= b.high[i] && a.high[i] >= b.low[i], 1:2)
    function sqdist(r, p)
        s = 0.0
        for i in 1:2
            d = p[i] < r.low[i] ? r.low[i] - p[i] : (p[i] > r.high[i] ? p[i] - r.high[i] : 0.0)
            s += d * d
        end
        return s
    end
    hits(tree, q) = (out = Int[]; foreach_intersecting(e -> (push!(out, SI.id(e)); false), tree, q); sort!(out))

    # A counting callback over a captured Ref, and a batch of nearest queries
    # through one state: the hot loops of BlobBoards' association.
    struct CountHits
        n::Base.RefValue{Int}
    end
    (c::CountHits)(elem) = (c.n[] += 1; false)
    function intersect_batch(tree, queries, counter)
        for q in queries
            foreach_intersecting(counter, tree, q)
        end
        return counter.n[]
    end
    # The same with a do-block closure over local Refs and an array (BlobBoards'
    # tracking callback): `foreach_intersecting` must specialize on it.
    function intersect_batch_closure(tree, queries, skip)
        n = Ref(0)
        best = Ref(0)
        for q in queries
            foreach_intersecting(tree, q) do elem
                j = SpatialIndexing.id(elem)
                skip[j] && return false
                n[] += 1
                j > best[] && (best[] = j)
                false
            end
        end
        return n[] + best[]
    end
    function nearest_batch(state, tree, points)
        s = 0
        for p in points
            res = nearest_neighbor!(state, tree, p; max_sqdist = 400.0)
            res === nothing || (s += res[1])
        end
        return s
    end

    cases = [(n, boxes, bulk, cap) for n in (1, 7, 300, 3000), boxes in (false, true), bulk in (:bulk_load!, :load!, :insert!), cap in (8, 25)]
    @testset "n=$n boxes=$boxes bulk=$bulk capacity=$cap" for (n, boxes, bulk, cap) in cases
        rects = random_rects(rng, n; boxes)
        tree = build(rects; bulk, capacity = cap)
        @test length(tree) == n
        state = NearestNeighborState(tree)
        for _ in 1:100
            x, y = 1100 * rand(rng) - 50, 900 * rand(rng) - 50
            r = 60 * rand(rng)
            q = SI.Rect((x - r, y - r), (x + r, y + r))
            @test hits(tree, q) == findall(b -> overlaps(b, q), rects)

            d2 = [sqdist(b, (x, y)) for b in rects]
            dmin = minimum(d2)
            res = nearest_neighbor(tree, (x, y))
            # (≈: the query accumulates with muladd, which may or may not fuse)
            @test res !== nothing && res[2] ≈ dmin && d2[res[1]] ≈ dmin
            @test nearest_neighbor!(state, tree, (x, y)) == res
            # The radius gate is inclusive: nothing beyond it, the minimum at it.
            @test nearest_neighbor!(state, tree, (x, y); max_sqdist = r^2) ==
                (res[2] <= r^2 ? res : nothing)
            @test nearest_neighbor!(state, tree, (x, y); max_sqdist = res[2]) == res
            @test nearest_neighbor!(state, tree, (x, y); max_sqdist = prevfloat(res[2])) === nothing
        end
    end

    @testset "bulk_load! builds load!'s tree" begin
        # node for node: the same elements in the same traversal order, the
        # same per-level node counts
        order(tree) = (out = Int[]; foreach_intersecting(e -> (push!(out, SI.id(e)); false), tree, SI.Rect((-Inf, -Inf), (Inf, Inf))); out)
        for n in (1, 30, 700, 5000), cap in (8, 25), boxes in (false, true)
            rects = random_rects(rng, n; boxes)
            a = build(rects; bulk = :bulk_load!, capacity = cap)
            b = build(rects; bulk = :load!, capacity = cap)
            @test order(a) == order(b)
            @test a.nnodes_perlevel == b.nnodes_perlevel
            @test SI.check(a)
        end
        tree = build(random_rects(rng, 5; boxes = false); bulk = :bulk_load!, capacity = 8)
        @test_throws ArgumentError bulk_load!(tree, [SpatialElem(SI.Rect((0.0, 0.0), (0.0, 0.0)), 1, nothing)])
        @test isempty(bulk_load!(RTree{Float64, 2}(Int, Nothing), SpatialElem{Float64, 2, Int, Nothing}[]))
    end

    @testset "early exit" begin
        rects = random_rects(rng, 500; boxes = true)
        tree = build(rects; bulk = :bulk_load!, capacity = 8)
        everything = SI.Rect((-1.0e6, -1.0e6), (1.0e6, 1.0e6))
        seen = Ref(0)
        @test foreach_intersecting(e -> (seen[] += 1; seen[] == 3), tree, everything)
        @test seen[] == 3
        @test !foreach_intersecting(e -> false, tree, everything)
        @test !foreach_intersecting(e -> true, tree, SI.Rect((2.0e6, 2.0e6), (3.0e6, 3.0e6)))
    end

    @testset "empty tree" begin
        tree = RTree{Float64, 2}(Int, Nothing)
        @test nearest_neighbor(tree, (0.0, 0.0)) === nothing
        @test nearest_neighbor!(NearestNeighborState(tree), tree, (0.0, 0.0)) === nothing
        @test !foreach_intersecting(e -> true, tree, SI.Rect((0.0, 0.0), (1.0, 1.0)))
    end

    @testset "allocation-free" begin
        for (n, cap) in ((10, 25), (3000, 25), (3000, 8))
            tree = build(random_rects(rng, n; boxes = false); bulk = :bulk_load!, capacity = cap)
            queries = map(1:100) do _
                x, y = 1000 * rand(rng), 800 * rand(rng)
                SI.Rect((x - 30, y - 30), (x + 30, y + 30))
            end
            points = [(1000 * rand(rng), 800 * rand(rng)) for _ in 1:100]
            counter = CountHits(Ref(0))
            state = NearestNeighborState(tree)
            skip = falses(n)
            intersect_batch(tree, queries, counter); nearest_batch(state, tree, points)
            intersect_batch_closure(tree, queries, skip)
            @test (@allocated intersect_batch(tree, queries, counter)) == 0
            # the closure's two Refs, once per batch, not per query or element
            @test (@allocated intersect_batch_closure(tree, queries, skip)) <= 64
            @test (@allocated nearest_batch(state, tree, points)) == 0
        end
    end
end
