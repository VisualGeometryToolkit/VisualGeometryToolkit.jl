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

@testitem "target calibration: gauges on the target manifold" begin
    using Test, StaticArrays, LinearAlgebra, Random, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, TargetManifold, minimal_manifold, tangent_size, retract, recenter,
        retract_jacobian, action_jacobian, local_coordinates, restrict, gauge_names, gauge_directions, model

    rng = Xoshiro(7)
    nodes = grid_nodes(4; lo = -0.1, hi = 0.1)                         # 16 nodes: an in-plane spline has n = 32
    pts = [SVector(0.2 * rand(rng) - 0.1, 0.2 * rand(rng) - 0.1) for _ in 1:12]
    A = Affine([1.002 0.001; -0.0015 0.998], [0.0004, -0.0003])
    H = HeightField(nodes, 1.0e-3 .* randn(rng, 16))
    T = ThinPlateSpline(nodes, 1.0e-4 .* randn(rng, 16), 1.0e-4 .* randn(rng, 16))
    unit_cols(D) = D ./ norm.(eachcol(D))'

    @testset "Affine: translation, rotation and scale are fixed; stretch and shear stay" begin
        t = PlanarTarget(pts; calibration = A)
        @test gauge_names(A) == (:translation_x, :translation_y, :rotation, :scale) && gauge_names(t) == gauge_names(A)
        D = gauge_directions(A)
        a = [1.002, 0.001, -0.0015, 0.998]
        @test D[:, 1] == [0, 0, 0, 0, 1, 0] && D[:, 2] == [0, 0, 0, 0, 0, 1]
        @test D[:, 3] ≈ [0.0015, -0.998, 1.002, 0.001, 0, 0] / norm(a)    # δA = J A₀
        @test D[:, 4] ≈ [a; 0; 0] / norm(a)                                 # δA = A₀
        ch = minimal_manifold(t)
        @test ch isa TargetManifold && tangent_size(ch) == 2
        B = retract_jacobian(ch)
        @test B isa SMatrix{6, 2} && B' * B ≈ I && norm(B' * D) < 1.0e-14   # the fixed directions have zero tangent
        v = SVector{2}(1.0e-4 .* randn(rng, 2))
        Δθ = parameter_vector(retract(ch, v)) - parameter_vector(A)
        @test norm(D' * Δθ) < 1.0e-16 && local_coordinates(ch, retract(ch, v)) ≈ v
        @test action_jacobian(ch, pts[1]) ≈ calibration_jacobian(A, pts[1]) * B
        @test tangent_size(TargetManifold(t; free = :scale)) == 3            # a lone Symbol
        @test tangent_size(TargetManifold(t; free = (:scale, :rotation))) == 4
        ch6 = TargetManifold(t; free = gauge_names(A))                      # a known pose: everything free
        @test tangent_size(ch6) == 6 && retract_jacobian(ch6) == I
        @test_throws ArgumentError TargetManifold(t; free = :piston)
        # recenter keeps the slice and the type
        r = recenter(ch, v)
        @test typeof(r) === typeof(ch) && retract_jacobian(r) == B && model(r) == retract(ch, v)
        @test only(Base.return_types(recenter, (typeof(ch), typeof(v)))) === typeof(ch)
    end

    @testset "ThinPlateSpline: its whole affine part is fixed" begin
        t = PlanarTarget(pts; calibration = T)
        @test gauge_names(T) == (:fx_1, :fx_u, :fx_v, :fy_1, :fy_u, :fy_v)
        D = gauge_directions(T)
        u, w = [p[1] for p in nodes], [p[2] for p in nodes]
        o, z = ones(16), zeros(16)
        @test D == [[o; z] [u; z] [w; z] [z; o] [z; u] [z; w]]
        ch = minimal_manifold(t)
        @test tangent_size(ch) == 32 - 6
        B = retract_jacobian(ch)
        @test norm(B' * D) < 1.0e-13 && B' * B ≈ I
        @test tangent_size(TargetManifold(t; free = (:fx_u, :fy_v))) == 28
        @test tangent_size(TargetManifold(t; free = gauge_names(T))) == 32
    end

    @testset "HeightField: piston and the two tilts are fixed" begin
        t = PlanarTarget(pts; calibration = H)
        @test gauge_names(t) == (:piston, :tilt_x, :tilt_y)
        ch = minimal_manifold(t)
        @test tangent_size(ch) == 13
        D = gauge_directions(H)
        B = retract_jacobian(ch)
        @test norm(B' * D) < 1.0e-13 && B' * B ≈ I
        v = SVector{13}(1.0e-4 .* randn(rng, 13))
        δh = parameter_vector(retract(ch, v)) - parameter_vector(H)
        @test abs(sum(δh)) < 1.0e-17 && abs(sum(p[1] * x for (p, x) in zip(nodes, δh))) < 1.0e-17 &&
            abs(sum(p[2] * x for (p, x) in zip(nodes, δh))) < 1.0e-17
        @test size(action_jacobian(ch, pts[1])) == (3, 13)
        plane = with_parameters(t, parameter_vector(H) + [1.0e-4 + 2.0e-3 * p[1] - 1.0e-3 * p[2] for p in nodes])
        @test norm(local_coordinates(ch, plane)) < 1.0e-15                  # a plane of heights is invisible
        @test tangent_size(TargetManifold(t; free = :piston)) == 14
        @test tangent_size(TargetManifold(t; free = (:piston, :tilt_x, :tilt_y))) == 16
    end

    @testset "chains: the members' gauges, prefixed" begin
        C = CalibrationChain(A, T, H)
        t = PlanarTarget(pts; calibration = C)
        @test gauge_names(C) == (:affine_translation_x, :affine_translation_y, :affine_rotation, :affine_scale,
            :thin_plate_spline_fx_1, :thin_plate_spline_fx_u, :thin_plate_spline_fx_v,
            :thin_plate_spline_fy_1, :thin_plate_spline_fy_u, :thin_plate_spline_fy_v,
            :height_field_piston, :height_field_tilt_x, :height_field_tilt_y)
        ch = minimal_manifold(t)
        @test tangent_size(ch) == 6 + 32 + 16 - 13
        @test norm(retract_jacobian(ch)' * unit_cols(gauge_directions(C))) < 1.0e-13
        @test tangent_size(TargetManifold(t; free = :affine_scale)) == 6 + 32 + 16 - 12
    end

    @testset "restrict fixes only the gauge combinations inside the free span" begin
        tA = PlanarTarget(pts; calibration = A)
        chA = minimal_manifold(tA)
        # b alone: both translations lie in it and are gauges there — nothing is left
        @test tangent_size(restrict(chA, (:bx, :by))) == 0
        rb = restrict(TargetManifold(tA; free = (:translation_x, :translation_y)), (:bx, :by))
        @test rb isa TargetManifold && tangent_size(rb) == 2 && retract_jacobian(rb) == SMatrix{6, 2}(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1)
        # the review's table: a gauge with only a component on the free axes fixes nothing
        @test tangent_size(restrict(chA, (:a11,))) == 1
        @test tangent_size(restrict(chA, (:a11, :a12))) == 2
        tH = PlanarTarget(pts; calibration = H)
        @test tangent_size(restrict(minimal_manifold(tH), (:h1, :h2, :h3))) == 3
        @test tangent_size(restrict(minimal_manifold(tH), (:h1, :h2, :h3, :h4, :h5))) == 5
        # at a diagonal A the scale (a11, 0, 0, a22) lies in (a11, a22) and bx is a translation: one left
        tD = PlanarTarget(pts; calibration = Affine([1.002 0.0; 0.0 0.998], [0.0004, -0.0003]))
        rD = restrict(minimal_manifold(tD), (:a11, :a22, :bx))
        @test tangent_size(rD) == 1 && abs(dot(retract_jacobian(rD)[:, 1], [1.002, 0, 0, 0.998, 0, 0])) < 1.0e-14
        @test_throws ArgumentError restrict(chA, (:h1,))
        @test tangent_size(minimal_manifold(PlanarTarget(pts))) == 0           # Rigid: zero-dimensional
    end

    @testset "the gauge rank is scale-free: a physical spline on centimetre nodes" begin
        # the tilt columns are the node coordinates, so their size follows the
        # node units; the default manifold's tangent size must not
        mt = VgtkCore.m
        for s in (1.0e-2, 1.0e-9)                                        # 1e-9: where unnormalised columns lose rank
            ns = grid_nodes(4; lo = -s, hi = s)
            ns_m = [p .* mt for p in ns]
            pts_m = [p .* (s / 0.1) .* mt for p in pts]
            Hm = HeightField(ns_m, fill(0.0mt, 16))
            Tm = ThinPlateSpline(ns_m, fill(0.0mt, 16), fill(0.0mt, 16))
            @test tangent_size(minimal_manifold(PlanarTarget(pts_m; calibration = Hm))) == tangent_size(minimal_manifold(PlanarTarget(pts; calibration = H)))
            @test tangent_size(minimal_manifold(PlanarTarget(pts_m; calibration = Tm))) == tangent_size(minimal_manifold(PlanarTarget(pts; calibration = T)))
            Cm = CalibrationChain(Affine([1.0 0.0; 0.0 1.0], [0.0, 0.0] .* mt), Hm)
            @test tangent_size(minimal_manifold(PlanarTarget(pts_m; calibration = Cm))) == 6 + 16 - 7
        end
    end

    @testset "the basis: an SMatrix up to n = 18, a Matrix above" begin
        T3 = ThinPlateSpline(grid_nodes(3; lo = -0.1, hi = 0.1))                   # n = 18
        small = minimal_manifold(PlanarTarget(pts; calibration = T3))
        @test retract_jacobian(small) isa SMatrix{18, 12}
        vs = SVector{12}(1.0e-5 .* randn(rng, 12))
        @test only(Base.return_types(recenter, (typeof(small), typeof(vs)))) === typeof(small)
        @test action_jacobian(small, pts[1]) isa SMatrix{2, 12} && local_coordinates(small, retract(small, vs)) ≈ vs
        @test retract_jacobian(minimal_manifold(PlanarTarget(pts; calibration = T))) isa Matrix{Float64}   # n = 32
        T5 = ThinPlateSpline(grid_nodes(5; lo = -0.1, hi = 0.1))                   # n = 50
        big = minimal_manifold(PlanarTarget(pts; calibration = T5))
        B = retract_jacobian(big)
        @test B isa Matrix{Float64} && size(B) == (50, 44) && tangent_size(big) == 44 && B' * B ≈ I
        v = 1.0e-5 .* randn(rng, 44)
        r = recenter(big, v)
        @test typeof(r) === typeof(big) && only(Base.return_types(recenter, (typeof(big), Vector{Float64}))) === typeof(big)
        @test local_coordinates(big, model(r)) ≈ v
        J = action_jacobian(big, pts[1])
        @test J isa Matrix{Float64} && size(J) == (2, 44) && J ≈ calibration_jacobian(T5, pts[1]) * B
        @test norm(B' * gauge_directions(T5)) < 1.0e-13
    end
end

@testitem "target calibration: the default manifold has full rank with the poses free" begin
    # The review's harness: a pinhole, 4 poses about 0.6 m from a 20 × 16 cm
    # board, 30 design points, 3 × 3 nodes. The joint projection Jacobian over the
    # poses (a rotation vector and a translation each) and the default manifold's
    # coordinates has full column rank: no gauge is left free.
    using Test, StaticArrays, LinearAlgebra, Random, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, minimal_manifold, retract_jacobian, tangent_size

    rng = Xoshiro(1)
    design = [SVector(x, y) for x in range(-0.1, 0.1; length = 6) for y in range(-0.08, 0.08; length = 5)]
    nodes = grid_nodes(3; lo = -0.1, hi = 0.1)
    skew(ω) = SMatrix{3, 3}(0, ω[3], -ω[2], -ω[3], 0, ω[1], ω[2], -ω[1], 0)
    function rodrigues(ω)
        θ = sqrt(sum(abs2, ω) + 1.0e-300)
        K = skew(ω / θ)
        return one(SMatrix{3, 3, eltype(ω)}) + sin(θ) * K + (1 - cos(θ)) * K * K
    end
    lift(p::StaticVector{2}) = SVector(p[1], p[2], zero(p[1]))
    lift(p::StaticVector{3}) = SVector(p[1], p[2], p[3])
    F = 4
    R0 = [rodrigues(SVector(0.3randn(rng), 0.3randn(rng), 0.3randn(rng))) for _ in 1:F]
    t0 = [SVector(0.05randn(rng), 0.05randn(rng), 0.6 + 0.1rand(rng)) for _ in 1:F]
    function projections(c, θ, pose)
        cc = with_parameters(c, θ)
        out = promote_type(eltype(θ), eltype(pose))[]
        for f in 1:F
            R = rodrigues(SVector{3}(pose[(6f - 5):(6f - 3)])) * R0[f]
            t = t0[f] + SVector{3}(pose[(6f - 2):(6f)])
            for p in design
                X = R * lift(cc(p)) + t
                push!(out, X[1] / X[3], X[2] / X[3])
            end
        end
        return out
    end
    function nullity_default(c; free = ())
        θ0 = Vector(parameter_vector(c)); n = length(θ0)
        J = ForwardDiff.jacobian(z -> projections(c, z[1:n], z[(n + 1):end]), [θ0; fill(1.0e-12, 6F)])
        B = Matrix(retract_jacobian(VgtkCore.TargetManifold(PlanarTarget(design; calibration = c); free)))
        s = svdvals([J[:, 1:n] * B J[:, (n + 1):end]])
        return count(<(1.0e-9 * s[1]), s)
    end

    A = Affine([1.01 0.02; -0.015 0.99], [0.003, -0.002])
    T = ThinPlateSpline(nodes, 1.0e-4 .* randn(rng, 9), 1.0e-4 .* randn(rng, 9))
    H0 = HeightField(nodes)
    H = HeightField(nodes, 1.0e-3 .* randn(rng, 9))
    cases = (
        "Affine" => A, "ThinPlateSpline" => T, "HeightField flat" => H0, "HeightField h ≈ 1 mm" => H,
        "Chain(Affine, TPS at zero displacement)" => CalibrationChain(A, ThinPlateSpline(nodes)),
        "Chain(Affine, HeightField)" => CalibrationChain(A, H), "Chain(Affine, HeightField flat)" => CalibrationChain(A, H0),
        "Chain(TPS, HeightField)" => CalibrationChain(T, H),
    )
    for (name, c) in cases
        @testset "$name" begin
            @test nullity_default(c) == 0
        end
    end
    # the harness can see a gauge: with the Affine's freed, its four come back
    @test nullity_default(A; free = VgtkCore.gauge_names(A)) == 4
end

@testitem "target calibration: names, hashes, AD on lengths, an uncertain calibration" begin
    using Test, StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, gauge_names, mm, m, Attributed
    using FlexUnits: ustrip

    nodes = grid_nodes(3; lo = 0.0, hi = 100.0)
    A = Affine([1.001 0.0; 0.0 0.999], [0.5, -0.25])
    T = ThinPlateSpline(nodes, fill(0.01, 9), fill(-0.01, 9))
    H = HeightField(nodes, [0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0])
    C = CalibrationChain(A, T, H)
    pts = [SVector(0.0, 0.0), SVector(100.0, 0.0), SVector(30.0, 40.0)]

    @testset "a chain's names are inferred" begin
        @test (@inferred parameter_names(C)) == parameter_names(C) && length(parameter_names(C)) == 6 + 18 + 9
        @test (@inferred gauge_names(C)) == gauge_names(C)
        t = PlanarTarget(pts; calibration = C)
        @test (@inferred VgtkCore.parameters(t)).affine_a11 == 1.001
    end

    @testset "one tag registry: TARGET_CALIBRATION_TAGS" begin
        R = VgtkCore.TARGET_CALIBRATION_TAGS
        @test collect(keys(R)) == ["rigid", "affine", "thin_plate_spline", "height_field", "calibration_chain"]
        @test isempty(VgtkCore.tag_rule_violations(R))
        @test [VgtkCore.tag_of(R, c) for c in (Rigid(), A, T, H, C)] == collect(keys(R))
        @test R["affine"] === Affine && R["calibration_chain"] === CalibrationChain
        @test_throws ArgumentError R["warp"]
        @test parameter_names(C)[1] === :affine_a11 && parameter_names(C)[end] === :height_field_h9   # the chain prefixes are its tags
    end

    @testset "hash agrees with ==" begin
        for (x, y) in ((A, Affine([1.001 0.0; 0.0 0.999], [0.5, -0.25])), (T, ThinPlateSpline(nodes, fill(0.01, 9), fill(-0.01, 9))),
                       (H, HeightField(nodes, [0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0])), (C, CalibrationChain(A, T, H)),
                       (PlanarTarget(pts; calibration = H), PlanarTarget(pts; calibration = HeightField(nodes, [0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0]))))
            @test x == y && hash(x) == hash(y)
            @test length(Set([x, y])) == 1
        end
        @test hash(A) != hash(with_parameters(A, parameter_vector(A) .+ 1.0e-3))
    end

    @testset "ForwardDiff through a physical calibration" begin
        pmm = SVector(10.0, 20.0) .* mm
        for c in (VgtkCore.with_unit(A, mm), VgtkCore.with_unit(H, mm))
            J = ForwardDiff.jacobian(θ -> ustrip.(m, with_parameters(c, θ)(pmm)), Vector(parameter_vector(c)))
            @test J ≈ calibration_jacobian(c, pmm) atol = 1.0e-14
        end
    end

    @testset "an uncertain calibration: with_parameters returns the bare kind" begin
        u = Uncertain(A, CovMat(SMatrix{6, 6}(1.0e-6I)))
        θ = parameter_vector(A) .+ 1.0e-4
        @test with_parameters(u, θ) isa Affine && parameter_vector(with_parameters(u, θ)) == θ
        @test calibration(with_parameters(PlanarTarget(pts; calibration = u), θ)) isa Affine
    end

    @testset "an attributed target's calibration record does not shadow its calibration" begin
        a = Attributed(PlanarTarget(pts; calibration = H), (; calibration_record = (; tool = "a ruler", n_images = 3)))
        @test a.calibration === H && calibration(a.value) === H           # the field, forwarded: the target's kind
        @test a.calibration_record == (; tool = "a ruler", n_images = 3)  # the attribute: who calibrated it
    end
end
