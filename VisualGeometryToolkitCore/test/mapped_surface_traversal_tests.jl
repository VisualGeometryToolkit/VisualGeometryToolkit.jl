@testitem "board deformation: the chart map's fused traversal agrees with its own call and transform_deriv" begin
    using Test, StaticArrays, Random
    using CoordinateTransformations: ComposedTransformation
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, transform_deriv
    import VisualGeometryToolkitCore as VgtkCore
    Cam = VgtkCore.Cameras

    # The MappedSurface root reads its chart map through `_map_value`/`_map_value_jacobian` (mapped_surface.jl) and a
    # calibration through `_calibration_value`/`_calibration_value_jacobian` (target_surfaces.jl): explicit fused
    # products, so one rounding order in every compiled copy. Every other caller reads the same map through its own
    # call and `transform_deriv` (CT's chain rule) and the calibration through `_apply`/`_point_jacobian`. The paths
    # must agree to rounding, member by member and composed, or they drift apart silently.
    rng = Xoshiro(11)
    nodes = grid_nodes(4; lo = -0.6, hi = 0.6)
    X = EuclideanMap(RotMatrix{3}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.02, -0.01, 2.0))
    P = SimilarityMap(0.9, RotMatrix{3}(RotXYZ(0.0, 0.0, 0.3)), SVector(-0.4, -0.45, 0.0))
    rect = Rect2(-0.5, -0.5, 1.0, 1.0)
    H = HeightField(nodes, 0.01 .* randn(rng, 16))
    T = ThinPlateSpline(nodes, 0.01 .* randn(rng, 16), 0.01 .* randn(rng, 16))
    E2 = EuclideanMap(RotMatrix{2}(0.02), SVector(0.004, -0.002))
    AS = Aspect(0.01) ∘ Skew(-0.003)
    pts = [SVector(0.8rand(rng) - 0.4, 0.8rand(rng) - 0.4) for _ in 1:50]

    near(a, b) = isapprox(a, b; rtol = 1.0e-12, atol = 1.0e-14)
    # each member of a composition at the point it is evaluated at, and the composition itself
    members(m, x) = m isa ComposedTransformation ? vcat(members(m.t2, x), members(m.t1, m.t2(x)), [(m, x)]) : [(m, x)]
    calibration_members(c, q) = c isa Cam._CalibrationComposition ?
        vcat(calibration_members(c.t2, q), calibration_members(c.t1, Cam._apply(c.t2, q)), [(c, q)]) : [(c, q)]
    kinds = Set{Any}()

    for C in (H, H ∘ AS, H ∘ E2, H ∘ T, H ∘ T ∘ AS, H ∘ Scale(0.01) ∘ AS)
        for Pc in (nothing, P)
            s = MappedSurface(X, C, Pc, rect)
            for p in pts
                for (m, x) in members(s.map, p)
                    y, J = Cam._map_value_jacobian(m, x)
                    @test near(y, m(x))
                    @test near(J, transform_deriv(m, x))
                    @test isequal(Cam._map_value(m, x), y)        # the point is the Jacobian pass's, operation for operation
                    push!(kinds, nameof(typeof(m)))
                end
            end
        end
        for p in pts, (c, q) in calibration_members(C, p)
            y, J = Cam._calibration_value_jacobian(c, q)
            @test near(y, Cam._apply(c, q))
            @test near(J, Cam._point_jacobian(c, q))
            @test isequal(Cam._calibration_value(c, q), y)
            push!(kinds, nameof(typeof(c)))
        end
    end
    # every member kind the traversal has a method for was reached
    @test issubset((:ComposedTransformation, :EuclideanMap, :SimilarityMap, :_PlainCalibration, :HeightField,
                    :ThinPlateSpline, :Aspect, :Skew, :Scale), kinds)
end
