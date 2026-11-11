# The surface of dual_hit.jl (generate_dual_hit.jl and test/mapped_surface_tests.jl build it alike): RadiometricCamera's
# board — 25 nodes on the page chart [0, 1] × [0, −1], placed with side 1 m, posed 1.2 m away.
dual_hit_rotation() = RotMatrix{3}(RotXYZ(0.1, -0.2, 0.05))
dual_hit_pose(x) = dual_hit_rotation() * x + SVector(0.02, -0.01, 1.2)
# the heights with one partial each (∂/∂hᵢ)
dual_hit_heights(h) = [ForwardDiff.Dual{Nothing}(h[i], ntuple(j -> Float64(i == j), length(h))...) for i in eachindex(h)]
function dual_hit_surface(h)
    nodes = [SVector(n[1], -n[2]) for n in grid_nodes(5; lo = 0.0, hi = 1.0)]
    P = SimilarityMap(1.0, one(RotMatrix{3, Float64}), SVector(-0.5, 0.5, 0.0))
    X = EuclideanMap(dual_hit_rotation(), SVector(0.02, -0.01, 1.2))
    return MappedSurface(X, HeightField(nodes, h), P, Rect2(-0.5, -0.5, 1.0, 1.0))
end
