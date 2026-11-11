# A chart-LM linearization state with rows only, shared by the chart LM's tests
# (hoisted_helpers_tests.jl, lm_driver_tests.jl). ReTestItems discovers
# `@testsetup` modules in `*_testsetups.jl` files; an item takes one with
# `setup = [CircleLmScene]`.

@testsetup module CircleLmScene
    using LinearAlgebra: norm
    using StaticArrays: SVector
    import VisualGeometryToolkitCore as VgtkCore
    export CircleLm, circle_points

    # A circle's residual rows on the chart x = (Δcx, Δcy, Δρ) about the centre (1, −2)
    # and radius 3: the distance of each point to the circle. The Jacobian is the
    # ForwardDiff default.
    struct CircleLm
        pts::Vector{SVector{2, Float64}}
    end
    function VgtkCore.residual_rows!(f, lm::CircleLm, x)
        c = SVector(1.0 + x[1], -2.0 + x[2]); ρ = 3.0 + x[3]
        for (i, p) in enumerate(lm.pts)
            f[i] = norm(p - c) - ρ
        end
        return f
    end

    # Twelve points on the circle of centre (1.5, −1.8) and radius 2.5.
    circle_points() = [SVector(1.5 + 2.5cos(θ), -1.8 + 2.5sin(θ)) for θ in range(0, 2π; length = 13)[1:12]]
end
