# Shared helpers of the SplineLens tests (`setup = [SplineAxis]`).

@testsetup module SplineAxis
    using StaticArrays: SVector
    using LinearAlgebra: norm
    export principal_ray, far_edge_probe

    # The ray of a SplineLens camera's principal point (x = 0): `normalize(backward(0, 0))`.
    principal_ray(c) = (d = c.lens.backward(SVector(0.0, 0.0)); d / norm(d))

    # Whether the equivalence fixture's pixel `i` (test/fixtures/spline_lens_golden.jl) lies
    # on the image's far edge, x = W or y = H. The old grid's half-open box gave NaN there;
    # a SplineLens images the closed rectangle, so the golden comparisons skip these
    # probes and check that the lens images them.
    far_edge_probe(G, i) = G.pixels[1, i] == G.image[1] || G.pixels[2, i] == G.image[2]
end
