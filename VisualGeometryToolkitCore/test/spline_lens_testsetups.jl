# Shared helpers of the SplineLens tests (`setup = [SplineAxis]`).

@testsetup module SplineAxis
    using StaticArrays: SVector
    using LinearAlgebra: norm
    export principal_ray

    # The ray of a SplineLens camera's principal point (x = 0): `normalize(backward(0, 0))`.
    principal_ray(c) = (d = c.lens.backward(SVector(0.0, 0.0)); d / norm(d))
end
