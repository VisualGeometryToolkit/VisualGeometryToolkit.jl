# =============================================================================
# Target calibration (geometry/cameras/target_calibration.jl): the kinds, their
# protocol, chains, lengths, and an uncertain calibration on a PlanarTarget.
# =============================================================================

@testitem "target calibration: the kinds and their protocol" begin
    using Test, StaticArrays, LinearAlgebra, Random, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, ThinPlateSplineSystem, tps_coefficients, tps_evaluate, bending_energy

    rng = Xoshiro(20261026)
    nodes = grid_nodes(4; lo = -0.1, hi = 0.1)                       # a 20 cm board's chart, 16 nodes
    A = Affine([1.002 0.001; -0.0015 0.998], [0.0004, -0.0003])
    T = ThinPlateSpline(nodes, 1.0e-4 .* randn(rng, 16), 1.0e-4 .* randn(rng, 16))
    H = HeightField(nodes, 1.0e-3 .* randn(rng, 16))
    kinds = (A, T, H, CalibrationChain(A, T), CalibrationChain(A, T, H), CalibrationChain(Rigid(), H))
    xs = [SVector(0.2 * rand(rng) - 0.1, 0.2 * rand(rng) - 0.1) for _ in 1:8]
    push!(xs, SVector(nodes[6]...))                                     # on a node

    @testset "with_parameters ∘ parameter_vector round trips" begin
        for c in kinds
            θ = parameter_vector(c)
            @test with_parameters(c, θ) == c
            θ′ = θ .+ 1.0e-5 .* randn(rng, length(θ))
            @test parameter_vector(with_parameters(c, θ′)) == θ′
            @test length(parameter_names(c)) == length(θ) && allunique(parameter_names(c))
            lo, hi = parameter_bounds(c)
            @test all(==(-Inf), lo) && all(==(Inf), hi)
            @test_throws ArgumentError with_parameters(c, [θ; 0.0])
        end
        @test parameter_names(A) == (:a11, :a12, :a21, :a22, :bx, :by)
        @test parameter_vector(A) == SVector(1.002, 0.001, -0.0015, 0.998, 0.0004, -0.0003)
        @test parameter_names(T)[[1, 16, 17, 32]] == (:fx1, :fx16, :fy1, :fy16)
        @test parameter_names(H) == ntuple(i -> Symbol(:h, i), 16)
        @test parameter_names(CalibrationChain(A, H))[[1, 6, 7, 22]] == (:affine_a11, :affine_by, :height_field_h1, :height_field_h16)
    end

    @testset "calibration_jacobian is ∂c(x)/∂θ (ForwardDiff)" begin
        for c in kinds, x in xs
            J = calibration_jacobian(c, x)
            Jad = ForwardDiff.jacobian(θ -> with_parameters(c, θ)(x), parameter_vector(c))
            @test size(J) == size(Jad)
            @test J ≈ Jad atol = 1.0e-13 rtol = 1.0e-12
        end
        @test size(calibration_jacobian(A, xs[1])) == (2, 6)
        @test size(calibration_jacobian(T, xs[1])) == (2, 32)
        @test size(calibration_jacobian(H, xs[1])) == (3, 16)
        @test size(calibration_jacobian(Rigid(), xs[1])) == (2, 0)
    end

    @testset "a chain is the composition of its members" begin
        C = CalibrationChain(A, T, H)
        for x in xs
            @test C(x) == H(T(A(x)))
            @test CalibrationChain(A, T)(x) == T(A(x))
        end
        @test parameter_vector(C) == vcat(parameter_vector(A), parameter_vector(T), parameter_vector(H))
        @test regularizer(C) == vcat(regularizer(A), regularizer(T), regularizer(H))
        @test_throws ArgumentError CalibrationChain(H, A)                       # a height field only last
        @test_throws ArgumentError CalibrationChain(A, A)                       # one member of each kind
        @test_throws ArgumentError CalibrationChain(A)                          # at least two
        @test_throws ArgumentError CalibrationChain(CalibrationChain(A, T), H)  # no nesting
    end

    @testset "the regularizer's rows are the bending energy" begin
        sys = ThinPlateSplineSystem(nodes)
        for _ in 1:5
            v, u = randn(rng, 16), randn(rng, 16)
            @test sum(abs2, regularizer(with_parameters(H, v))) ≈ bending_energy(sys, v) rtol = 1.0e-12
            @test sum(abs2, regularizer(with_parameters(T, [v; u]))) ≈ bending_energy(sys, v) + bending_energy(sys, u) rtol = 1.0e-12
        end
        affine_values = [1.0 + 2p[1] - 3p[2] for p in nodes]                 # no bending, to rounding
        @test sum(abs2, regularizer(with_parameters(H, affine_values))) < 1.0e-10 * sum(abs2, affine_values)
        @test isempty(regularizer(A)) && isempty(regularizer(Rigid()))
        @test length(regularizer(T)) == 32 && length(regularizer(H)) == 16
    end

    @testset "a height field's z is the thin-plate spline, bit for bit" begin
        # RadiometricCamera's board: a 5 × 5 height grid over [0, 1]²
        rc_nodes = grid_nodes(5; lo = 0.0, hi = 1.0)
        heights = [0.002 * sin(3p[1]) * cos(2p[2]) for p in rc_nodes]
        hf = HeightField(rc_nodes, heights)
        sys = ThinPlateSplineSystem(rc_nodes)
        w, a = tps_coefficients(sys, heights)
        for p in (SVector(0.37, 0.61), SVector(0.0, 0.0), SVector(0.25, 0.5), SVector(0.999, 0.001))
            q = hf(p)
            @test q[3] === tps_evaluate(sys.nodes, w, a, p)
            @test q[1] === p[1] && q[2] === p[2]
        end
    end

    @testset "Rigid, Affine and the identity" begin
        x = xs[1]
        @test Rigid()(x) === x
        @test Affine([1.0 0.0; 0.0 1.0], [0.0, 0.0])(x) == x
        @test A(x) ≈ SMatrix{2, 2}(1.002, -0.0015, 0.001, 0.998) * x + SVector(0.0004, -0.0003)
        @test ThinPlateSpline(nodes)(x) == x && HeightField(nodes)(x) == SVector(x[1], x[2], 0.0)
    end
end

@testitem "target calibration: lengths and an uncertain calibration" begin
    using Test, StaticArrays, LinearAlgebra, Statistics
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, mm, m, has_units
    using FlexUnits: DimensionError, ustrip

    pts = [SVector(0.0, 0.0), SVector(100.0, 0.0), SVector(0.0, 50.0), SVector(100.0, 50.0)]
    pts_mm = [p .* mm for p in pts]
    nodes_mm = [p .* mm for p in grid_nodes(3; lo = 0.0, hi = 100.0)]

    @testset "a physical calibration moves physical points" begin
        A = Affine([1.001 0.0; 0.0 0.999], [0.5, -0.25] .* mm)
        t = PlanarTarget(pts_mm; calibration = A)
        @test has_units(A) && has_units(t)
        @test parameter_vector(A)[5:6] ≈ [0.0005, -0.00025]                 # lengths in θ: metres
        P = points(t)
        @test all(p -> eltype(p) <: VgtkCore.Quantity, P)
        tp = ustrip(mm, t)
        @test !has_units(tp) && !has_units(calibration(tp))
        @test [ustrip.(mm, p) for p in P] ≈ points(tp)
        @test [ustrip.(mm, p) for p in points(VgtkCore.with_unit(tp, mm))] ≈ [ustrip.(mm, p) for p in P]
        @test VgtkCore.with_unit(tp, mm) == PlanarTarget(pts_mm; calibration = VgtkCore.with_unit(calibration(tp), mm))
        # one form: a plain calibration on lengths, or the reverse, is refused
        @test_throws DimensionError PlanarTarget(pts_mm; calibration = Affine([1.0 0.0; 0.0 1.0], [0.5, 0.0]))
        @test_throws DimensionError PlanarTarget(pts; calibration = A)
        @test_throws DimensionError A(pts[2])
        # Rigid has no lengths: it sits with either form
        @test points(PlanarTarget(pts_mm)) == pts_mm && points(PlanarTarget(pts)) == pts
    end

    @testset "the thin-plate kinds in lengths" begin
        H = HeightField(nodes_mm, fill(0.0mm, 9) .+ [0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0] .* mm)
        Hp = HeightField(grid_nodes(3; lo = 0.0, hi = 100.0), [0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0])
        t = PlanarTarget(pts_mm; calibration = H)
        @test points(ustrip(mm, t)) ≈ points(PlanarTarget(pts; calibration = Hp))
        @test ustrip.(mm, points(t)[2]) ≈ Hp(pts[2])
        T = ThinPlateSpline(nodes_mm, fill(0.01mm, 9), fill(-0.01mm, 9))
        @test ustrip.(mm, T(pts_mm[2])) ≈ pts[2] + SVector(0.01, -0.01)
        @test_throws DimensionError HeightField(nodes_mm, fill(0.1, 9))
        @test_throws DimensionError CalibrationChain(Affine([1.0 0.0; 0.0 1.0], [0.0, 0.0]), H)
    end

    @testset "Uncertain(calibration, Σ)" begin
        A = Affine([1.001 0.0; 0.0 0.999], [0.5, -0.25] .* mm)
        Σ = SMatrix{6, 6}(Diagonal([1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-10, 1.0e-10]))   # b's in m²
        u = Uncertain(A, CovMat(Σ))
        @test geom(u) === A && Statistics.cov(u) == Σ
        @test parameter_vector(u) == parameter_vector(A) && parameter_names(u) == parameter_names(A)
        @test calibration_jacobian(u, pts[2]) == calibration_jacobian(A, pts[2])
        t = PlanarTarget(pts_mm; calibration = u)
        @test calibration(t) === u && t isa PlanarTarget{typeof(A)}
        @test points(t) == points(PlanarTarget(pts_mm; calibration = A))
        # the covariance moves with the unit: the length entries in mm², the rest unchanged
        Σmm = Statistics.cov(calibration(ustrip(mm, t)))
        @test Σmm[1:4, 1:4] == Σ[1:4, 1:4] && Σmm[5:6, 5:6] ≈ 1.0e6 .* Σ[5:6, 5:6]
        @test Statistics.cov(calibration(VgtkCore.with_unit(ustrip(mm, t), mm))) ≈ Σ
        # a new parameter vector is a new calibration: no covariance
        @test calibration(with_parameters(t, parameter_vector(t))) == A
        @test_throws DimensionMismatch PlanarTarget(pts_mm; calibration = Uncertain(A, CovMat(SMatrix{2, 2}(1.0, 0, 0, 1.0))))
    end
end

@testitem "target calibration: planarity is in the type" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, PlanarCalibration, mm

    nodes = grid_nodes(3; lo = 0.0, hi = 100.0)
    A = Affine([1.0 0.001; 0.0 1.0], [0.1, 0.0])
    T = ThinPlateSpline(nodes, fill(0.01, 9), fill(-0.01, 9))
    H = HeightField(nodes, [0.0, 0.1, 0.0, 0.2, 0.0, -0.1, 0.0, 0.0, 0.3])
    pts = [SVector(0.0, 0.0), SVector(100.0, 0.0), SVector(0.0, 50.0), SVector(30.0, 40.0)]

    @testset "is_planar by type" begin
        @test is_planar(Rigid) && is_planar(typeof(A)) && is_planar(typeof(T)) && !is_planar(typeof(H))
        @test is_planar(typeof(CalibrationChain(A, T))) && is_planar(typeof(CalibrationChain(Rigid(), A)))
        @test !is_planar(typeof(CalibrationChain(A, T, H))) && !is_planar(typeof(CalibrationChain(A, H)))
        @test is_planar(A) && !is_planar(H) && is_planar(PlanarTarget(pts; calibration = T)) && !is_planar(PlanarTarget(pts; calibration = H))
        @test is_planar(Uncertain(A, CovMat(SMatrix{6, 6}(1.0I)))) && !is_planar(Uncertain(H, CovMat(SMatrix{9, 9}(1.0I))))
        @test (@inferred is_planar(typeof(CalibrationChain(A, T, H)))) === false
        @test A isa PlanarCalibration && T isa PlanarCalibration && CalibrationChain(A, T) isa PlanarCalibration
        @test !(H isa PlanarCalibration) && !(CalibrationChain(A, H) isa PlanarCalibration)
    end

    @testset "points: SVector{2} when planar, SVector{3} otherwise (inferred)" begin
        for c in (Rigid(), A, T, CalibrationChain(A, T))
            t = PlanarTarget(pts; calibration = c)
            P = @inferred points(t)
            @test eltype(P) === SVector{2, Float64}
        end
        for c in (H, CalibrationChain(A, H), CalibrationChain(A, T, H))
            t = PlanarTarget(pts; calibration = c)
            P = @inferred points(t)
            @test eltype(P) === SVector{3, Float64}
            @test P == [c(p) for p in pts]
        end
        tmm = PlanarTarget([p .* mm for p in pts]; calibration = VgtkCore.with_unit(H, mm))
        @test eltype(@inferred points(tmm)) <: SVector{3, <:VgtkCore.Quantity}
        # a method written for the plane refuses a height field by dispatch
        on_plane(t::PlanarTarget{<:PlanarCalibration}) = length(points(t))
        @test on_plane(PlanarTarget(pts; calibration = CalibrationChain(A, T))) == 4
        @test_throws MethodError on_plane(PlanarTarget(pts; calibration = H))
    end

    @testset "a rig's world points take a lifted target as it is" begin
        X = EuclideanMap(one(SMatrix{3, 3, Float64}), SVector(0.0, 0.0, 1000.0))
        rig = Rig{WorldFrame}(Dict(:b => PosedTarget(PlanarTarget(pts; calibration = H), X)))
        @test points(rig, :b) ≈ [H(p) + SVector(0.0, 0.0, 1000.0) for p in pts]
    end
end
