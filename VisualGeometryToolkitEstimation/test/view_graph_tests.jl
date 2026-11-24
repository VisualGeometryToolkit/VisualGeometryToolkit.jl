# ViewGraph and ObservationGraph — the label bookkeeping over Graphs.jl, the
# connectivity policies, the spanning-tree chain, the loop and orientation
# filters on rotation edges, the spanning-tree seed of rotation averaging, and
# the co-visibility of an observation graph.

@testitem "ViewGraph: structure and connectivity policies" begin
    using Test
    using VisualGeometryToolkitEstimation: ViewGraph, views, edgelist, nedges, adjacent_views, components, label_components,
        require_connected, keep_largest_component, articulation, bridges, induced

    # two components: a triangle 1-2-3 with a tail 3-4, and the pair 10-11
    e = Dict((1, 2) => :a, (2, 3) => :b, (1, 3) => :c, (3, 4) => :d, (10, 11) => :e)
    vg = ViewGraph(e)
    @test views(vg) == [1, 2, 3, 4, 10, 11]
    @test vg[3, 4] == :d && haskey(vg, 1, 3) && !haskey(vg, 3, 1) && nedges(vg) == 5
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
    # equal sizes: by the smallest view; an isolated view is its own component; no cut, no bridge in a cycle
    tie = ViewGraph([7, 5, 6, 1, 2, 9], Dict((5, 6) => :x, (1, 2) => :y, (2, 7) => :z, (1, 7) => :w))
    @test components(tie) == [[1, 2, 7], [5, 6], [9]] && articulation(tie) == Int[] && bridges(tie) == [(5, 6)]
    @test label_components(tie) == Dict(1 => 1, 2 => 1, 7 => 1, 5 => 2, 6 => 2, 9 => 3)
end

@testitem "ViewGraph: spanning tree, chain, loop and orientation filters" begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations
    using VisualGeometryToolkitEstimation: ViewGraph, spanning_tree, chain, triangles, filter_cycles,
        rotation_loop_angle, filter_by_orientation, edgelist
    using VisualGeometryToolkitCore: log_so3, Attributed

    rng = Xoshiro(3)
    n = 6
    R = Dict(k => (k == 1 ? one(SMatrix{3, 3, Float64, 9}) : SMatrix{3, 3, Float64, 9}(RotMatrix(QuatRotation(normalize(randn(rng, 4)))))) for k in 1:n)
    rel(i, j) = R[j]' * R[i]                                                  # f_j = R_ij·f_i, d = R_k·f_k
    pairs = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 2)]              # a sequence with lag-2 pairs: triangles
    edge(i, j; weight = 100) = Attributed(rel(i, j), (; weight))
    vg = ViewGraph(Dict(p => edge(p...; weight = 100 + 10 * p[1]) for p in pairs))

    @testset "spanning tree: maximum weight, n - 1 edges" begin
        t = spanning_tree(vg)
        @test length(t) == n - 1
        @test (5, 6) in t                                                    # the heaviest edges win
        @test_throws ArgumentError spanning_tree(ViewGraph(Dict((1, 2) => Attributed(rel(1, 2), (; weight = 0)))))
    end
    @testset "chain: exact on consistent edges, world_from_k" begin
        A = chain(vg, 1; identity = one(SMatrix{3, 3, Float64, 9}), inverse = transpose)
        @test sort!(collect(keys(A))) == collect(1:n)
        @test maximum(k -> norm(A[k] - R[k]), 1:n) < 1.0e-12                 # R_1 = I, so the gauges agree
        # a root other than 1: the same rotations up to that gauge
        B = chain(vg, 4; identity = one(SMatrix{3, 3, Float64, 9}), inverse = transpose)
        @test maximum(k -> norm(B[k] - R[4]' * R[k]), 1:n) < 1.0e-12
        @test_throws ArgumentError chain(vg, 99; identity = one(SMatrix{3, 3, Float64, 9}))
    end
    @testset "triangles and the cycle filter catch a wrong edge" begin
        @test triangles(vg) == [(i, i + 1, i + 2) for i in 1:(n - 2)]
        bad = Dict(p => edge(p...) for p in pairs)
        bad[(3, 4)] = Attributed(SMatrix{3, 3, Float64, 9}(RotMatrix(RotationVec(0.4, -0.2, 0.3))) * bad[(3, 4)].value, (; weight = 100))
        bg = ViewGraph(bad)
        @test rotation_loop_angle(bg[2, 3], bg[3, 4], bg[2, 4]) > 0.3
        @test rotation_loop_angle(bg[1, 2], bg[2, 3], bg[1, 3]) < 1.0e-12
        kept = first.(edgelist(filter_cycles(bg; tol = 1.0e-3)))
        # the corrupted edge goes, and with it (2, 4) and (3, 5): each lies only in a triangle through
        # (3, 4), so no consistent loop supports it (Theia's rule; the orientation filter is the precise one)
        @test kept == setdiff(sort(pairs), [(2, 4), (3, 4), (3, 5)])
        # an edge in no triangle stays unless keep_untested = false
        chain_only = ViewGraph(Dict(p => edge(p...) for p in [(1, 2), (2, 3)]))
        @test length(edgelist(filter_cycles(chain_only; tol = 1.0e-3))) == 2
        @test isempty(edgelist(filter_cycles(chain_only; tol = 1.0e-3, keep_untested = false)))
        # the orientation filter against the true rotations removes the same edge
        @test first.(edgelist(filter_by_orientation(bg, R; tol = 1.0e-3))) == setdiff(sort(pairs), [(3, 4)])
    end
end

@testitem "rotation averaging: the spanning-tree seed and the ViewGraph method" begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations
    using VisualGeometryToolkitEstimation: RotationAveraging, ViewGraph
    using VisualGeometryToolkitCore: ClassDefault, EstimationProblem, solve, AbsoluteRotations
    include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))
    using VisualGeometryToolkitCore: successful_retcode, Attributed

    rng = Xoshiro(11)
    n = 8
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
    rel(i, j) = R[j]' * R[i]
    edges = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 3)]
    noisy = Dict(e => RotMatrix(RotationVec((1.0e-3 .* randn(rng, 3))...)) * rel(e...) for e in edges)
    weighted = Dict(e => Attributed(noisy[e], (; weight = 50.0 + e[1])) for e in edges)
    import VisualGeometryToolkitEstimation as VgtkEstimation

    spectral = rotation_averaging(weighted; scale = 1.0e-3)
    tree = rotation_averaging(weighted; seed = :spanning_tree, scale = 1.0e-3)
    @test successful_retcode(spectral) && successful_retcode(tree)
    @test max_rotation_error(tree.u, R, n) < 5.0e-3
    @test maximum(k -> norm(Matrix(tree.u[k]) - Matrix(spectral.u[k])), 1:n) < 1.0e-8   # same optimum
    @test tree.u[1] ≈ one(RotMatrix{3, Float64}) atol = 1.0e-12                                  # re-gauged
    # an explicit seed: the truth itself, any gauge
    G = RotMatrix(RotationVec(0.3, 0.1, -0.2))
    given = rotation_averaging(weighted; seed = Dict(k => G * R[k] for k in 1:n), scale = 1.0e-3)
    @test maximum(k -> norm(Matrix(given.u[k]) - Matrix(spectral.u[k])), 1:n) < 1.0e-8
    @test_throws ArgumentError rotation_averaging(weighted; seed = Dict(1 => R[1]), scale = 1.0e-3)
    @test_throws ArgumentError rotation_averaging(weighted; seed = :nonsense, scale = 1.0e-3)
    @test spectral.u[1] ≈ one(RotMatrix{3, Float64}) atol = 1.0e-12                              # one gauge rule
    # the ViewGraph method is the dictionary form
    @test (rotation_averaging(ViewGraph(weighted); scale = 1.0e-3)).u == spectral.u
    # the weights matter: unit weights give another (close) optimum
    @test (rotation_averaging(noisy; scale = 1.0e-3)).u != spectral.u

    # every seed refuses a disconnected graph, as the spectral one does
    split = Dict((1, 2) => rel(1, 2), (3, 4) => rel(3, 4))
    @test_throws ArgumentError rotation_averaging(split; seed = Dict(k => R[k] for k in 1:4), scale = 1.0e-3)
    @test_throws ArgumentError rotation_averaging(split; seed = :spanning_tree, scale = 1.0e-3)
    # a seed's extra frames are not the averaging's: the gauge is the first frame of the edges
    sub = Dict(e => noisy[e] for e in edges if e[1] >= 2)
    res = rotation_averaging(sub; seed = Dict(k => R[k] for k in 1:n), scale = 1.0e-3)
    @test sort!(collect(keys(res.u))) == collect(2:n)
    @test res.u[2] ≈ one(RotMatrix{3, Float64}) atol = 1.0e-12
    # a pair given as (j, i) is accepted by the spanning-tree seed, as by the spectral one
    flipped = Dict(((j, i) => noisy[(i, j)]' for (i, j) in edges if i == 1)..., (e => noisy[e] for e in edges if e[1] != 1)...)
    @test max_rotation_error((rotation_averaging(flipped; seed = :spanning_tree, scale = 1.0e-3)).u, R, n) < 5.0e-3
    # a view in no edge is a disconnected ViewGraph
    @test_throws ArgumentError rotation_averaging(ViewGraph(vcat(1:n, 99), Dict(e => noisy[e] for e in edges)); scale = 1.0e-3)
end

@testitem "chain: transforms of another type than the identity" begin
    using Test, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitEstimation: ViewGraph, chain
    A = SMatrix{2, 2}(2.0, 0.0, 0.0, 1.0)
    vg = ViewGraph(Dict((1, 2) => A, (2, 3) => A))
    out = chain(vg, 1; identity = I)                     # UniformScaling at the root, matrices elsewhere
    @test out[1] == I && out[2] ≈ inv(A) && out[3] ≈ inv(A) * inv(A)
    # views outside the root's component get no entry; a branching tree composes along each view's own path
    forest = ViewGraph(Dict((1, 2) => A, (1, 3) => 2A, (3, 4) => A, (7, 8) => A))
    out = chain(forest, 3; identity = I)
    @test sort!(collect(keys(out))) == [1, 2, 3, 4] && out[3] == I
    @test out[1] == I * (2A) && out[2] == out[1] * inv(A) && out[4] == I * inv(A)
end

@testitem "ObservationGraph: frames × boards, components, covisibility" begin
    using Test
    using VisualGeometryToolkitEstimation: ObservationGraph, ViewGraph, observers, observed, observed_in, observers_of,
        edgelist, nedges, components, label_components, require_connected, articulation, bridges, covisibility, views

    # frames 1-4 see boards "A" "B"; frame 4 alone links "C"; frames 7-8 see "Z" (a second component)
    obs = Dict((1, "A") => 10, (1, "B") => 11, (2, "A") => 12, (3, "B") => 13, (4, "B") => 14, (4, "C") => 15,
               (7, "Z") => 16, (8, "Z") => 17)
    og = ObservationGraph(obs)
    @test observers(og) == [1, 2, 3, 4, 7, 8] && observed(og) == ["A", "B", "C", "Z"]
    @test og[4, "C"] == 15 && haskey(og, 1, "A") && !haskey(og, 1, "C") && nedges(og) == 8
    @test observed_in(og, 1) == ["A", "B"] && observers_of(og, "B") == [1, 3, 4]
    @test ObservationGraph(Dict(edgelist(og))).edges == og.edges
    cs = components(og)
    @test cs[1] == (; observers = [1, 2, 3, 4], observed = ["A", "B", "C"])
    @test cs[2] == (; observers = [7, 8], observed = ["Z"])
    @test label_components(og).observed["Z"] == 2
    @test articulation(og) == (; observers = [1, 4], observed = ["A", "B", "Z"])
    @test bridges(og) == [(1, "A"), (1, "B"), (2, "A"), (3, "B"), (4, "B"), (4, "C"), (7, "Z"), (8, "Z")]   # each component a tree
    @test label_components(og) == (; observers = Dict(1 => 1, 2 => 1, 3 => 1, 4 => 1, 7 => 2, 8 => 2),
                                     observed = Dict("A" => 1, "B" => 1, "C" => 1, "Z" => 2))
    # equal vertex counts: by the smallest observer
    two = ObservationGraph(Dict((5, "P") => 1, (2, "Q") => 2))
    @test components(two) == [(; observers = [2], observed = ["Q"]), (; observers = [5], observed = ["P"])]
    @test_throws ArgumentError require_connected(og)
    # integer labels on both sides do not collide
    ints = ObservationGraph(Dict((1, 1) => :x, (2, 1) => :y))
    @test observers(ints) == [1, 2] && observed(ints) == [1] && length(components(ints)) == 1

    # covisibility of the frames: shared-board counts; of the boards: shared-frame counts
    fv = covisibility(og)
    @test fv isa ViewGraph
    @test Dict(p => e.weight for (p, e) in edgelist(fv)) == Dict((1, 2) => 1, (1, 3) => 1, (1, 4) => 1, (3, 4) => 1, (7, 8) => 1)
    @test fv[1, 2].value == ["A"]
    bv = covisibility(og; side = :observed)
    @test Dict(p => e.value for (p, e) in edgelist(bv)) == Dict(("A", "B") => [1], ("B", "C") => [4])
    @test views(bv) == ["A", "B", "C", "Z"]
    @test_throws ArgumentError covisibility(og; side = :boards)
end

@testitem "filtered rotation averaging: the cycle and orientation filters around the averaging" begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations
    using VisualGeometryToolkitEstimation: FilteredRotationAveraging, RotationAveraging, ViewGraph, edgelist, nedges
    using VisualGeometryToolkitCore: EstimationProblem, solve, AbsoluteRotations, ClassDefault
    include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))
    using VisualGeometryToolkitCore: Attributed

    rng = Xoshiro(23)
    n = 10
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
    pairs = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 3)]
    rel = Dict(p => RotMatrix(RotationVec((1.0e-3 .* randn(rng, 3))...)) * R[p[2]]' * R[p[1]] for p in pairs)
    rel[(4, 6)] = RotMatrix(RotationVec(0.8, -0.4, 0.3)) * rel[(4, 6)]                   # one wrong pair
    vg = ViewGraph(Dict(p => Attributed(rel[p], (; weight = 100.0)) for p in pairs))
    import VisualGeometryToolkitEstimation as VgtkEstimation

    # no filter: the plain fit on the whole graph
    plain = filtered_averaging(vg; scale = 1.0e-3)
    @test plain.metadata.graph === vg && plain.u == rotation_averaging(vg; scale = 1.0e-3).u
    # the orientation filter removes the wrong pair and refits on the rest
    o = filtered_averaging(vg; orientation_tol = deg2rad(2.0), scale = 1.0e-3)
    @test !haskey(o.metadata.graph, 4, 6) && nedges(o.metadata.graph) == length(pairs) - 1
    @test max_rotation_error(o.u, R, n) < 5.0e-3
    # the cycle filter removes it before averaging
    c = filtered_averaging(vg; cycle_tol = deg2rad(2.0), scale = 1.0e-3)
    @test !haskey(c.metadata.graph, 4, 6) && max_rotation_error(c.u, R, n) < 5.0e-3
    @test_throws ArgumentError filtered_averaging(vg; cycle_tol = -1.0, scale = 1.0e-3)
    @test_throws ArgumentError filtered_averaging(vg; orientation_tol = 0.0, scale = 1.0e-3)
    # the old names are gone, no aliases (retired in check_naming): the algorithms are the API
    @test !isdefined(VisualGeometryToolkitEstimation, Symbol("filtered_rotation" * "_averaging"))
    @test !isdefined(VisualGeometryToolkitEstimation, Symbol("bundle_refine" * "_relpose"))
    @test !isdefined(VisualGeometryToolkitEstimation, Symbol("fit_filtered" * "_rotation_averaging"))
    @test :FilteredRotationAveraging in names(VisualGeometryToolkitEstimation)
    @test :RelativePoseProblem in names(VisualGeometryToolkitEstimation)
end

# B2's Uncertain edges, `Attributed(Uncertain(R_ij, CovMat(Σ)), (; weight))`, are the edges the
# averaging accepts: the filters read their rotation (Core's `geom`), never the covariance (they
# made an `SMatrix` of the Uncertain value, a DimensionMismatch).
@testitem "the view-graph filters take Uncertain edges" begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations
    using VisualGeometryToolkitEstimation: FilteredRotationAveraging, filter_cycles, filter_by_orientation,
        rotation_loop_angle, ViewGraph, nedges
    using VisualGeometryToolkitCore: EstimationProblem, solve, AbsoluteRotations
    include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))
    using VisualGeometryToolkitCore: Attributed, Uncertain, CovMat

    rng = Xoshiro(29)
    n = 10
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
    pairs = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 3)]
    rel = Dict(p => RotMatrix(RotationVec((1.0e-3 .* randn(rng, 3))...)) * R[p[2]]' * R[p[1]] for p in pairs)
    rel[(4, 6)] = RotMatrix(RotationVec(0.8, -0.4, 0.3)) * rel[(4, 6)]                   # one wrong pair
    Σ = SMatrix{3, 3, Float64}(1.0e-6I)
    vg = ViewGraph(Dict(p => Attributed(Uncertain(rel[p], CovMat(Σ)), (; weight = 100.0)) for p in pairs))
    import VisualGeometryToolkitEstimation as VgtkEstimation

    # the residual and both filters on the Uncertain edges, and on bare Uncertain rotations
    @test rotation_loop_angle(vg[1, 2], vg[2, 3], vg[1, 3]) < 1.0e-2
    @test rotation_loop_angle(Uncertain(rel[(1, 2)], CovMat(Σ)), rel[(2, 3)], rel[(1, 3)]) < 1.0e-2
    c = filter_cycles(vg; tol = deg2rad(2.0))
    @test !haskey(c, 4, 6) && haskey(c, 4, 5)
    o = filter_by_orientation(vg, R; tol = deg2rad(2.0))
    @test !haskey(o, 4, 6) && nedges(o) == length(pairs) - 1
    @test nedges(filter_by_orientation(vg, Dict(k => Uncertain(R[k], CovMat(Σ)) for k in 1:n); tol = deg2rad(2.0))) == length(pairs) - 1
    # the filtered averaging at its tolerances: the wrong pair removed, the rest fitted (no scale: the covariances carry it)
    fo = filtered_averaging(vg; orientation_tol = deg2rad(2.0))
    @test !haskey(fo.metadata.graph, 4, 6) && max_rotation_error(fo.u, R, n) < 5.0e-3
    fc = filtered_averaging(vg; cycle_tol = deg2rad(2.0), orientation_tol = deg2rad(2.0))
    @test !haskey(fc.metadata.graph, 4, 6) && max_rotation_error(fc.u, R, n) < 5.0e-3
end
