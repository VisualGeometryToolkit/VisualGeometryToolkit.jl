# ViewGraph and ObservationGraph — the label bookkeeping over Graphs.jl, the
# connectivity policies, the spanning-tree chain, the loop and orientation
# filters on rotation edges, the spanning-tree seed of rotation averaging, and
# the co-visibility of an observation graph.

@testitem "ViewGraph: structure and connectivity policies" begin
    using Test
    using VisualGeometryToolkitEstimation: ViewGraph, views, edgelist, adjacent_views, components, label_components,
        require_connected, keep_largest_component, articulation, bridges, induced

    # two components: a triangle 1-2-3 with a tail 3-4, and the pair 10-11
    e = Dict((1, 2) => :a, (2, 3) => :b, (1, 3) => :c, (3, 4) => :d, (10, 11) => :e)
    vg = ViewGraph(e)
    @test views(vg) == [1, 2, 3, 4, 10, 11]
    @test vg[3, 4] == :d && haskey(vg, 1, 3) && !haskey(vg, 3, 1)
    @test first.(edgelist(vg)) == [(1, 2), (1, 3), (2, 3), (3, 4), (10, 11)]
    @test ViewGraph(Dict(edgelist(vg))).edges == vg.edges                     # the edge list round-trips
    @test adjacent_views(vg, 3) == [1, 2, 4]
    @test components(vg) == [[1, 2, 3, 4], [10, 11]]
    @test label_components(vg) == Dict(1 => 1, 2 => 1, 3 => 1, 4 => 1, 10 => 2, 11 => 2)
    @test articulation(vg) == [3]                                             # 4 hangs on 3
    @test bridges(vg) == [(3, 4), (10, 11)]
    @test views(keep_largest_component(vg)) == [1, 2, 3, 4]
    @test views(induced(vg, [1, 2])) == [1, 2] && first.(edgelist(induced(vg, [1, 2]))) == [(1, 2)]
    err = try require_connected(vg; what = "sequence"); "" catch ex; sprint(showerror, ex) end
    @test occursin("sequence is not connected", err) && occursin("[1, 2, 3, 4], [10, 11]", err)
    @test require_connected(induced(vg, [1, 2, 3])) isa ViewGraph
    # refusals: a reversed pair, an unknown view, a repeated view
    @test_throws ArgumentError ViewGraph(Dict((2, 1) => :x))
    @test_throws ArgumentError ViewGraph([1, 2], Dict((1, 3) => :x))
    @test_throws ArgumentError ViewGraph([1, 1, 2], Dict((1, 2) => :x))
    # labels of any ordered type
    sg = ViewGraph(Dict(("a", "b") => 1, ("b", "c") => 2))
    @test components(sg) == [["a", "b", "c"]] && articulation(sg) == ["b"]
end

@testitem "ViewGraph: spanning tree, chain, loop and orientation filters" begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations
    using VisualGeometryToolkitEstimation: ViewGraph, spanning_tree, chain, triangles, filter_cycles,
        rotation_loop_angle, filter_by_orientation, edgelist
    using VisualGeometryToolkitCore: log_so3

    rng = Xoshiro(3)
    n = 6
    R = Dict(k => (k == 1 ? one(SMatrix{3, 3, Float64, 9}) : SMatrix{3, 3, Float64, 9}(RotMatrix(QuatRotation(normalize(randn(rng, 4)))))) for k in 1:n)
    j_from_i(i, j) = R[j]' * R[i]                                             # f_j = R_ij·f_i, d = R_k·f_k
    pairs = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 2)]              # a sequence with lag-2 pairs: triangles
    edge(i, j; inliers = 100) = (j_from_i = j_from_i(i, j), inliers = inliers)
    vg = ViewGraph(Dict(p => edge(p...; inliers = 100 + 10 * p[1]) for p in pairs))

    @testset "spanning tree: maximum weight, n - 1 edges" begin
        t = spanning_tree(vg; weight = e -> e.inliers)
        @test length(t) == n - 1
        @test (5, 6) in t                                                    # the heaviest edges win
        @test_throws ArgumentError spanning_tree(ViewGraph(Dict((1, 2) => (inliers = 0,))); weight = e -> e.inliers)
    end
    @testset "chain: exact on consistent edges, world_from_k" begin
        A = chain(vg, 1; weight = e -> e.inliers, identity = one(SMatrix{3, 3, Float64, 9}), inverse = transpose)
        @test sort!(collect(keys(A))) == collect(1:n)
        @test maximum(k -> norm(A[k] - R[k]), 1:n) < 1.0e-12                 # R_1 = I, so the gauges agree
        # a root other than 1: the same rotations up to that gauge
        B = chain(vg, 4; weight = e -> e.inliers, identity = one(SMatrix{3, 3, Float64, 9}), inverse = transpose)
        @test maximum(k -> norm(B[k] - R[4]' * R[k]), 1:n) < 1.0e-12
        @test_throws ArgumentError chain(vg, 99; identity = one(SMatrix{3, 3, Float64, 9}))
    end
    @testset "triangles and the cycle filter catch a wrong edge" begin
        @test triangles(vg) == [(i, i + 1, i + 2) for i in 1:(n - 2)]
        bad = Dict(p => edge(p...) for p in pairs)
        bad[(3, 4)] = (j_from_i = SMatrix{3, 3, Float64, 9}(RotMatrix(RotationVec(0.4, -0.2, 0.3))) * bad[(3, 4)].j_from_i, inliers = 100)
        bg = ViewGraph(bad)
        @test rotation_loop_angle(bg[2, 3], bg[3, 4], bg[2, 4]) > 0.3
        @test rotation_loop_angle(bg[1, 2], bg[2, 3], bg[1, 3]) < 1.0e-12
        kept = first.(edgelist(filter_cycles(bg; residual = rotation_loop_angle, tol = 1.0e-3)))
        # the corrupted edge goes, and with it (2, 4) and (3, 5): each lies only in a triangle through
        # (3, 4), so no consistent loop supports it (Theia's rule; the orientation filter is the precise one)
        @test kept == setdiff(sort(pairs), [(2, 4), (3, 4), (3, 5)])
        # an edge in no triangle stays unless keep_untested = false
        chain_only = ViewGraph(Dict(p => edge(p...) for p in [(1, 2), (2, 3)]))
        @test length(edgelist(filter_cycles(chain_only; residual = rotation_loop_angle, tol = 1.0e-3))) == 2
        @test isempty(edgelist(filter_cycles(chain_only; residual = rotation_loop_angle, tol = 1.0e-3, keep_untested = false)))
        # the orientation filter against the true rotations removes the same edge
        @test first.(edgelist(filter_by_orientation(bg, R; tol = 1.0e-3))) == setdiff(sort(pairs), [(3, 4)])
    end
end

@testitem "fit_rotation_averaging: the spanning-tree seed and the ViewGraph method" begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations
    using VisualGeometryToolkitEstimation: fit_rotation_averaging, ViewGraph
    using VisualGeometryToolkitCore: estimate, converged

    rng = Xoshiro(11)
    n = 8
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
    rel(i, j) = R[j]' * R[i]
    edges = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 3)]
    noisy = Dict(e => RotMatrix(RotationVec((1.0e-3 .* randn(rng, 3))...)) * rel(e...) for e in edges)
    prior = Dict(e => 50.0 + e[1] for e in edges)
    err(Rh) = maximum(k -> rotation_angle(RotMatrix{3}(SMatrix{3, 3}(Rh[k])' * R[k])), 1:n)

    spectral = fit_rotation_averaging(noisy, prior)
    tree = fit_rotation_averaging(noisy, prior; seed = :spanning_tree)
    @test converged(spectral) && converged(tree)
    @test err(estimate(tree)) < 5.0e-3
    @test maximum(k -> norm(Matrix(estimate(tree)[k]) - Matrix(estimate(spectral)[k])), 1:n) < 1.0e-8   # same optimum
    @test estimate(tree)[1] ≈ one(RotMatrix{3, Float64}) atol = 1.0e-12                                  # re-gauged
    # an explicit seed: the truth itself, any gauge
    G = RotMatrix(RotationVec(0.3, 0.1, -0.2))
    given = fit_rotation_averaging(noisy, prior; seed = Dict(k => G * R[k] for k in 1:n))
    @test maximum(k -> norm(Matrix(estimate(given)[k]) - Matrix(estimate(spectral)[k])), 1:n) < 1.0e-8
    @test_throws ArgumentError fit_rotation_averaging(noisy, prior; seed = Dict(1 => R[1]))
    @test_throws ArgumentError fit_rotation_averaging(noisy, prior; seed = :nonsense)
    # the default is unchanged: bit for bit the spectral fit
    @test estimate(fit_rotation_averaging(noisy, prior)) == estimate(spectral)
    # the ViewGraph method is the dictionary form
    vg = ViewGraph(Dict(e => (j_from_i = noisy[e], inliers = prior[e]) for e in edges))
    @test estimate(fit_rotation_averaging(vg; prior = e -> e.inliers)) == estimate(spectral)
end

@testitem "ObservationGraph: frames × boards, components, covisibility" begin
    using Test
    using VisualGeometryToolkitEstimation: ObservationGraph, ViewGraph, observers, observed, observed_in, observers_of,
        edgelist, components, label_components, require_connected, articulation, bridges, covisibility, views

    # frames 1-4 see boards "A" "B"; frame 4 alone links "C"; frames 7-8 see "Z" (a second component)
    obs = Dict((1, "A") => 10, (1, "B") => 11, (2, "A") => 12, (3, "B") => 13, (4, "B") => 14, (4, "C") => 15,
               (7, "Z") => 16, (8, "Z") => 17)
    og = ObservationGraph(obs)
    @test observers(og) == [1, 2, 3, 4, 7, 8] && observed(og) == ["A", "B", "C", "Z"]
    @test og[4, "C"] == 15 && haskey(og, 1, "A") && !haskey(og, 1, "C")
    @test observed_in(og, 1) == ["A", "B"] && observers_of(og, "B") == [1, 3, 4]
    @test ObservationGraph(Dict(edgelist(og))).edges == og.edges
    cs = components(og)
    @test cs[1] == (; observers = [1, 2, 3, 4], observed = ["A", "B", "C"])
    @test cs[2] == (; observers = [7, 8], observed = ["Z"])
    @test label_components(og).observed["Z"] == 2
    @test articulation(og) == (; observers = [1, 4], observed = ["A", "B", "Z"])
    @test (4, "C") in bridges(og) && (2, "A") in bridges(og)
    @test_throws ArgumentError require_connected(og)
    # integer labels on both sides do not collide
    ints = ObservationGraph(Dict((1, 1) => :x, (2, 1) => :y))
    @test observers(ints) == [1, 2] && observed(ints) == [1] && length(components(ints)) == 1

    # covisibility of the frames: shared-board counts; of the boards: shared-frame counts
    fv = covisibility(og)
    @test fv isa ViewGraph
    @test Dict(edgelist(fv)) == Dict((1, 2) => 1, (1, 3) => 1, (1, 4) => 1, (3, 4) => 1, (7, 8) => 1)
    bv = covisibility(og; side = :observed, weight = identity)
    @test Dict(edgelist(bv)) == Dict(("A", "B") => [1], ("B", "C") => [4])
    @test views(bv) == ["A", "B", "C", "Z"]
    @test_throws ArgumentError covisibility(og; side = :boards)
end
