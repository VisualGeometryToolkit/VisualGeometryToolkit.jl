# =============================================================================
# EuclideanMap's translation eltype is free: a physical pose holds FlexUnits'
# base-form quantities (metres), an inner-loop map plain numbers (Float64,
# Float32, ForwardDiff Duals). `ustrip(u, E)` gives the plain map in `u`;
# attaching is construction, `EuclideanMap(rotation(E), translation(E) .* u)`.
# A unitful map meeting plain numbers raises `DimensionError`, never promotes.
# =============================================================================

@testitem "euclidean map units" begin
    using Test
    using StaticArrays, Rotations, LinearAlgebra, ForwardDiff, Statistics
    using FlexUnits: Dimensions, FixRat32, Quantity
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: EuclideanMap, m, mm, ustrip, DimensionError, translation, rotation,
        relabel, CameraFrame, WorldFrame, CameraPose, Uncertain, CovMat, PrecMat, geom, has_units, with_unit,
        PlainEuclideanMap, UnitfulEuclideanMap

    Q = Quantity{Float64, Dimensions{FixRat32}}
    # within one unit in the last place, elementwise
    within_ulp(a, b) = all(abs(x - y) <= eps(abs(y)) for (x, y) in zip(a, b))

    R = RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.3))
    R2 = RotMatrix{3, Float64}(RotXYZ(-0.4, 0.05, 0.2))
    E = EuclideanMap(R, SVector(0.0628, 8.0293853, 0.062746506) .* m)
    E2 = EuclideanMap(R2, SVector(1.0, -2.0, 0.5) .* m)

    @testset "a physical map holds base-form quantities" begin
        @test has_units(E)
        @test E isa UnitfulEuclideanMap{3}
        @test !(E isa PlainEuclideanMap{3})
        @test eltype(translation(E)) === Q
        @test isbitstype(typeof(E))
        @test ustrip.(m, translation(E)) == SVector(0.0628, 8.0293853, 0.062746506)   # metres, exactly
        # a quantity in any length unit is stored in metres
        @test translation(EuclideanMap(R, SVector(62.8, 0.0, 0.0) .* mm))[1] ≈ 0.0628m
        # a translation must be a length
        @test_throws DimensionError EuclideanMap(R, SVector(1.0, 2.0, 3.0) .* (m * m))
    end

    @testset "ustrip gives the plain map in a unit; plain maps refuse it" begin
        P = ustrip(mm, E)
        @test !has_units(P)
        @test P isa PlainEuclideanMap{3}
        @test translation(P) ≈ SVector(62.8, 8029.3853, 62.746506)
        @test rotation(P) === rotation(E)
        @test_throws ArgumentError ustrip(mm, P)
        # attaching is construction
        A = EuclideanMap(rotation(P), translation(P) .* mm)
        @test has_units(A)
        @test within_ulp(ustrip.(m, translation(A)), ustrip.(m, translation(E)))
    end

    @testset "a unitful map acts on quantities only" begin
        y = E(SVector(1.0, 2, 3) .* m)
        @test eltype(y) <: Quantity
        @test ustrip.(m, y) == R * SVector(1.0, 2, 3) + ustrip.(m, translation(E))
        @test_throws DimensionError E(SVector(1.0, 2, 3))
    end

    @testset "a unitful map applied to a static point stays static and allocation-free" begin
        x = SVector(0.4, -1.5, 2.25) .* m
        apply(E, x) = E(x)
        apply(E, x)
        @test apply(E, x) isa SVector{3, Q}
        @test (@allocated apply(E, x)) == 0
        # the fast path is the generic `R * x + t`, bit for bit, in 2-D and 3-D,
        # over translations and points spanning six decades
        generic(E, x) = rotation(E) * x + translation(E)
        decades = (1.0e-3, 1.0e-1, 1.0, 1.0e1, 1.0e3)
        maps3 = [EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.3i, -0.7i, 1.1i)), SVector(sin(i), cos(3i), sin(5i)) .* (s * m))
            for i in 1:12, s in decades]
        points3 = [SVector(cos(i), sin(2i), cos(7i)) .* (s * m) for i in 1:40, s in decades]
        @test all(E1(x1) === generic(E1, x1) for E1 in maps3, x1 in points3)
        maps2 = [EuclideanMap(RotMatrix{2, Float64}(0.37i), SVector(sin(i), cos(3i)) .* (s * m)) for i in 1:12, s in decades]
        points2 = [SVector(cos(i), sin(2i)) .* (s * m) for i in 1:40, s in decades]
        @test all(E1(x1) === generic(E1, x1) for E1 in maps2, x1 in points2)
        # a Float32 rotation, a labeled map
        E32 = EuclideanMap(RotMatrix{3, Float32}(RotXYZ(0.1, -0.2, 0.3)), SVector(0.5, 0.25, -1.0) .* m; check = false)
        @test E32(x) === generic(E32, x)
        El = relabel(CameraFrame, WorldFrame, E)
        @test El(x) === generic(El, x)
        # a QuatRotation map (as IO builds every loaded pose, with the explicit
        # type constructor) and its inverse keep the quaternion's own product
        quat_maps = [
            EuclideanMap{3, Float64, QuatRotation{Float64}, SVector{3, Q}}(
                    QuatRotation(RotXYZ(0.3i, -0.7i, 1.1i)), SVector(sin(i), cos(3i), sin(5i)) .* (s * m)
                ) for i in 1:12, s in decades
        ]
        @test rotation(quat_maps[1]) isa QuatRotation{Float64}
        @test rotation(inv(quat_maps[1])) isa QuatRotation{Float64}
        @test all(E1(x1) === generic(E1, x1) for E1 in quat_maps, x1 in points3)
        @test all(E1(x1) === generic(E1, x1) for E1 in inv.(quat_maps), x1 in points3)
        @test apply(quat_maps[1], x) isa SVector{3, Q}
        # the public positional form keeps the rotation type it is given, for a
        # unitful and a plain translation alike (a saved pose round-trips bit-exactly)
        q = QuatRotation(RotXYZ(0.3, -0.7, 1.1))
        t_mm = SVector(1.0, 2.0, 3.0) .* mm
        Eq = EuclideanMap(q, t_mm)
        @test rotation(Eq) isa QuatRotation{Float64}
        @test rotation(Eq) === q
        @test rotation(inv(Eq)) isa QuatRotation{Float64}
        @test rotation(EuclideanMap(q, SVector(1.0, 2.0, 3.0))) === q
        @test rotation(EuclideanMap(q, [1.0, 2.0, 3.0])) === q
        Rm = RotMatrix{3, Float64}(RotXYZ(0.3, -0.7, 1.1))
        @test rotation(EuclideanMap(Rm, t_mm)) === Rm
        @test rotation(EuclideanMap(Rm, SVector(1.0, 2.0, 3.0))) isa RotMatrix{3, Float64}
        @test rotation(EuclideanMap(RotationVec(0.1, 0.2, 0.3), SVector(1.0, 2.0, 3.0))) isa RotMatrix{3, Float64}
        # a point of another dimension, or of mixed dimensions, is the generic DimensionError
        @test_throws DimensionError E(SVector(1.0, 2, 3) .* (m * m))
        @test_throws DimensionError E(SVector(1.0m, 2.0m, 3.0 * m * m))
    end

    @testset "composition and inversion stay unitful and concrete" begin
        C = E ∘ E2
        @test isconcretetype(typeof(C))
        @test eltype(translation(C)) === eltype(translation(E))
        I = inv(E)
        @test isconcretetype(typeof(I))
        @test eltype(translation(I)) === eltype(translation(E))
        @test ustrip.(m, translation(E ∘ I)) ≈ zeros(3) atol = 1.0e-15
        P = ustrip(m, E2)
        @test_throws DimensionError E ∘ P
        @test_throws DimensionError P ∘ E
    end

    @testset "plain translations keep their eltype under a Float64 rotation" begin
        F32 = EuclideanMap(R, SVector{3, Float32}(1, 2, 3))
        @test eltype(translation(F32)) === Float32
        @test isbitstype(typeof(F32))
        @test (@inferred F32(SVector(1.0, 2.0, 3.0))) isa SVector{3, Float64}
        @test (@inferred inv(F32)) isa EuclideanMap
        D = ForwardDiff.Dual{Nothing}(1.0, 1.0)
        FD = EuclideanMap(R, SVector(D, 2D, 3D))
        @test eltype(translation(FD)) === typeof(D)
        @test isbitstype(typeof(FD))
        @test (@inferred FD(SVector(1.0, 2.0, 3.0))) isa SVector{3, typeof(D)}
        # a Float64 vector keeps today's map exactly
        P64 = EuclideanMap(R, [1.0, 2.0, 3.0])
        @test translation(P64) === SVector(1.0, 2.0, 3.0)
        @test typeof(rotation(P64)) === RotMatrix{3, Float64, 9}
    end

    @testset "an uncertain pose: the covariance's translation block moves with the unit" begin
        A = SMatrix{6, 6, Float64}(reshape(range(0.1, 3.6; length = 36), 6, 6)) / 10
        Σ = A * A' + SMatrix{6, 6, Float64}(I)
        X = Uncertain(relabel(CameraFrame, WorldFrame, E), CovMat(Σ))
        @test has_units(X)
        Xs = ustrip(mm, X)
        @test !has_units(Xs)
        @test geom(Xs) isa CameraPose          # frames preserved
        Σs = SMatrix{6, 6, Float64}(Statistics.cov(Xs))
        @test Σs[1:3, 1:3] == Σ[1:3, 1:3]
        @test Σs[4:6, 4:6] == Σ[4:6, 4:6] .* 1.0e6
        @test Σs[1:3, 4:6] == Σ[1:3, 4:6] .* 1.0e3
        @test Σs[4:6, 1:3] == Σ[4:6, 1:3] .* 1.0e3
        Xb = with_unit(Xs, mm)
        @test has_units(Xb)
        @test geom(Xb) isa CameraPose
        @test rotation(geom(Xb)) === rotation(geom(X))
        @test within_ulp(ustrip.(m, translation(geom(Xb))), ustrip.(m, translation(geom(X))))
        @test within_ulp(SMatrix{6, 6, Float64}(Statistics.cov(Xb)), Σ)
        @test_throws ArgumentError with_unit(X, mm)
        @test_throws ArgumentError ustrip(mm, Xs)
        # a precision moves the opposite way
        Λ = inv(CovMat(Σ))
        Λs = SMatrix{6, 6, Float64}(VgtkCore.prec(ustrip(mm, Uncertain(E, Λ))))
        @test Λs[4:6, 4:6] == SMatrix{6, 6, Float64}(Λ)[4:6, 4:6] ./ 1.0e6
    end
end
