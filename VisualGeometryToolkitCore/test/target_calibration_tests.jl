# =============================================================================
# Target calibration (geometry/cameras/target_calibration.jl): the kinds, their
# protocol, the compositions (CoordinateTransformations' `∘`: the affine
# factors in their canonical order, a height field after an in-plane kind),
# lengths, and an uncertain calibration on a PlanarTarget.
# =============================================================================

@testitem "target calibration: the kinds and their protocol" begin
    using Test, StaticArrays, LinearAlgebra, Random, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, ThinPlateSplineSystem, tps_coefficients, tps_evaluate, bending_energy,
        transform_deriv, transform_deriv_params
    using CoordinateTransformations: ComposedTransformation
    using Rotations: RotMatrix

    rng = Xoshiro(20261026)
    nodes = grid_nodes(4; lo = -0.1, hi = 0.1)                       # a 20 cm board's chart, 16 nodes
    AU = Aspect(0.002) ∘ Skew(-0.0015)                                 # the print error with a jointly estimated pose
    E = with_parameters(EuclideanMap(one(RotMatrix{2, Float64}), SVector(0.0, 0.0)), [0.3, 0.0004, -0.0003])
    K = E ∘ Scale(0.001) ∘ Aspect(0.002) ∘ Skew(-0.0015)              # a known pose: the whole affine
    T = ThinPlateSpline(nodes, 1.0e-4 .* randn(rng, 16), 1.0e-4 .* randn(rng, 16))
    H = HeightField(nodes, 1.0e-3 .* randn(rng, 16))
    kinds = (Aspect(0.002), Skew(-0.0015), Scale(0.001), E, AU, K, T, H, H ∘ AU, H ∘ T)
    xs = [SVector(0.2 * rand(rng) - 0.1, 0.2 * rand(rng) - 0.1) for _ in 1:8]
    push!(xs, SVector(nodes[6]...))                                     # on a node
    has_rigid(c) = any(l -> l isa EuclideanMap, VgtkCore.Cameras._members(c))

    @testset "with_parameters ∘ parameter_vector round trips" begin
        for c in kinds
            θ = parameter_vector(c)
            θ′ = θ .+ 1.0e-5 .* randn(rng, length(θ))
            if has_rigid(c)       # the angle goes through atan(sin, cos): to rounding
                @test all(x -> with_parameters(c, θ)(x) ≈ c(x), xs)
                @test parameter_vector(with_parameters(c, θ′)) ≈ θ′ rtol = 1.0e-15
            else
                @test with_parameters(c, θ) == c
                @test parameter_vector(with_parameters(c, θ′)) == θ′
            end
            @test length(parameter_names(c)) == length(θ) && allunique(parameter_names(c))
            lo, hi = parameter_bounds(c)
            @test all(==(-Inf), lo) && all(==(Inf), hi)
            @test_throws ArgumentError with_parameters(c, [θ; 0.0])
        end
        @test parameter_names(Aspect(0.0)) == (:a1,) && parameter_names(Skew(0.0)) == (:u12,) && parameter_names(Scale()) == (:log_scale,)
        @test parameter_names(E) == (:angle, :tx, :ty) && parameter_vector(E) ≈ [0.3, 0.0004, -0.0003]
        @test parameter_names(T)[[1, 16, 17, 32]] == (:fx1, :fx16, :fy1, :fy16)
        @test parameter_names(H) == ntuple(i -> Symbol(:h, i), 16)
        # CoordinateTransformations' order: the outer map's first
        @test parameter_names(H ∘ AU)[[1, 16, 17, 18]] == (:height_field_h1, :height_field_h16, :aspect_a1, :skew_u12)
        @test parameter_names(K) == (:euclidean_angle, :euclidean_tx, :euclidean_ty, :scale_log_scale, :aspect_a1, :skew_u12)
    end

    @testset "transform_deriv_params is ∂c(x)/∂θ, transform_deriv ∂c(x)/∂x (ForwardDiff)" begin
        for c in kinds, x in xs
            J = transform_deriv_params(c, x)
            Jad = ForwardDiff.jacobian(θ -> with_parameters(c, θ)(x), Vector(parameter_vector(c)))
            @test size(J) == size(Jad)
            @test J ≈ Jad atol = 1.0e-13 rtol = 1.0e-12
            @test transform_deriv(c, x) ≈ ForwardDiff.jacobian(c, x) atol = 1.0e-13 rtol = 1.0e-12
        end
        @test size(transform_deriv_params(AU, xs[1])) == (2, 2) && size(transform_deriv_params(K, xs[1])) == (2, 6)
        @test size(transform_deriv_params(T, xs[1])) == (2, 32)
        @test size(transform_deriv_params(H, xs[1])) == (3, 16)
        @test size(transform_deriv_params(H ∘ AU, xs[1])) == (3, 18)
        @test size(transform_deriv_params(Rigid(), xs[1])) == (2, 0)
    end

    @testset "a composition: CT's ∘, its protocol in CT's order" begin
        for X in (AU, T)
            C = H ∘ X
            @test C isa ComposedTransformation && C.t1 === H && C.t2 === X        # nothing merged
            for x in xs
                @test C(x) == H(X(x))
                # the order of the parameter vector is the order of CT's Jacobian: hcat(p1, m1 * p2)
                # (≈: the spline's nodal weights round differently where the compiler inlines them)
                @test transform_deriv_params(C, x) ≈ hcat(transform_deriv_params(H, X(x)), transform_deriv(H, X(x)) * transform_deriv_params(X, x)) rtol = 1.0e-13
            end
            @test parameter_vector(C) == vcat(parameter_vector(H), parameter_vector(X))
            @test regularizer(C) == vcat(regularizer(H), regularizer(X))
            @test with_parameters(C, parameter_vector(C)) == C && hash(with_parameters(C, parameter_vector(C))) == hash(C)
        end
        # every kind ∘ every other is a ComposedTransformation of the two, unmerged
        for a in (Rigid(), Aspect(0.1), Skew(0.1), Scale(0.1), E, T, H), b in (Rigid(), Aspect(0.1), Skew(0.1), Scale(0.1), E, T, H)
            a isa EuclideanMap && b isa EuclideanMap && continue                 # Core composes two rigid maps into one
            c = a ∘ b
            @test c isa ComposedTransformation && c.t1 === a && c.t2 === b
        end
        # a target takes the factors in their canonical order, a height field only outermost
        A1, U1, S1 = Aspect(0.1), Skew(0.1), Scale(0.1)
        for c in (U1 ∘ A1, A1 ∘ S1, S1 ∘ E, A1 ∘ A1, T ∘ A1, A1 ∘ T, A1 ∘ H, H ∘ Rigid(), Rigid() ∘ A1, H ∘ (T ∘ A1),
                  (H ∘ T) ∘ A1, H ∘ H, H ∘ U1 ∘ A1)
            @test_throws ArgumentError PlanarTarget(xs; calibration = c)
        end
        # a target stores a composition right-associated, whatever ∘ built
        for c in (E ∘ S1 ∘ A1 ∘ U1, E ∘ (S1 ∘ (A1 ∘ U1)), (E ∘ S1) ∘ (A1 ∘ U1), H ∘ A1 ∘ U1, H ∘ (A1 ∘ U1))
            s = calibration(PlanarTarget(xs; calibration = c))
            @test s == foldr(ComposedTransformation, VgtkCore.Cameras._members(c))
            @test parameter_vector(s) == parameter_vector(c) && parameter_names(s) == parameter_names(c)
        end
        # the retired AffineMap kind is refused by name
        e = try PlanarTarget(xs; calibration = VgtkCore.AffineMap(SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0), SVector(0.0, 0.0))); nothing catch err; err end
        @test e isa ArgumentError && occursin("Aspect ∘ Skew", e.msg)
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
        @test all(c -> isempty(regularizer(c)), (Rigid(), Aspect(0.1), Skew(0.1), Scale(0.1), E, K))
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

    @testset "Rigid, the factors and the identity" begin
        x = xs[1]
        @test Rigid()(x) === x
        @test Aspect{2}()(x) == x && Skew{2}()(x) == x && Scale()(x) == x
        @test Aspect(0.002)(x) == SVector(exp(0.002) * x[1], exp(-0.002) * x[2])
        @test Skew(-0.0015)(x) == SVector(x[1] - 0.0015 * x[2], x[2])
        @test Scale(0.001)(x) == exp(0.001) * x
        @test K(x) ≈ E(exp(0.001) * (SMatrix{2, 2}(exp(0.002), 0, 0, exp(-0.002)) * (SMatrix{2, 2}(1.0, 0, -0.0015, 1.0) * x)))
        @test ThinPlateSpline(nodes)(x) == x && HeightField(nodes)(x) == SVector(x[1], x[2], 0.0)
    end
end

@testitem "target calibration: the Iwasawa factors, N = 2 and 3" begin
    using Test, StaticArrays, LinearAlgebra, Random, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: transform_deriv, transform_deriv_params, convert_eltype, length_mask, gauge_names
    import Rotations

    rng = Xoshiro(20261104)
    # the derivative of an affine map's (vec M, b) along a parameter, read off
    # its parameter Jacobian (affine in x): b' = J(0), M' e_j = J(e_j) − J(0)
    function affine_tangents(c, N)
        J0 = transform_deriv_params(c, zero(SVector{N, Float64}))
        Je = [transform_deriv_params(c, SVector{N}(ntuple(i -> Float64(i == j), N))) - J0 for j in 1:N]
        return reduce(hcat, [vcat(vec(reduce(hcat, [Je[j][:, k] for j in 1:N])), J0[:, k]) for k in 1:size(J0, 2)])
    end

    for N in (2, 3)
        @testset "N = $N" begin
            L = N * (N - 1) ÷ 2
            a, u, k = SVector{N - 1}(1.0e-2 .* randn(rng, N - 1)), SVector{L}(1.0e-2 .* randn(rng, L)), 1.0e-2 * randn(rng)
            A, U, S = Aspect(a), Skew(u), Scale(k)
            @test A isa Aspect{N, Float64} && U isa Skew{N, Float64, L}
            @test length(parameter_vector(A)) == N - 1 && length(parameter_vector(U)) == L
            xs = [SVector{N}(randn(rng, N)) for _ in 1:6]
            for c in (A, U, S, A ∘ U, S ∘ A ∘ U), x in xs
                Jad = ForwardDiff.jacobian(θ -> with_parameters(c, θ)(x), Vector(parameter_vector(c)))
                @test transform_deriv_params(c, x) ≈ Jad atol = 1.0e-14 rtol = 1.0e-13
                @test transform_deriv(c, x) ≈ ForwardDiff.jacobian(c, x) atol = 1.0e-14 rtol = 1.0e-13
                # a finite difference too, central, h = 1e-6
                θ0 = Vector(parameter_vector(c))
                Jfd = reduce(hcat, [(with_parameters(c, θ0 + 1.0e-6 * e)(x) - with_parameters(c, θ0 - 1.0e-6 * e)(x)) / 2.0e-6
                                    for e in eachcol(Matrix{Float64}(I, length(θ0), length(θ0)))])
                @test transform_deriv_params(c, x) ≈ Jfd atol = 1.0e-9
            end
            # round trips, exact
            for c in (A, U, S, A ∘ U, S ∘ A ∘ U)
                θ = parameter_vector(c)
                @test with_parameters(c, θ) == c && parameter_vector(with_parameters(c, θ .+ 1.0e-3)) == θ .+ 1.0e-3
                @test isempty(gauge_names(c)) && !any(length_mask(c)) && isempty(regularizer(c))
            end
            # Σ dᵢ = 0 exactly; det = 1: exact for the skew, to rounding for the aspect
            @test sum(A.log_diagonal) == 0
            @test det(transform_deriv(U, xs[1])) == 1
            @test det(transform_deriv(A ∘ U, xs[1])) ≈ 1 atol = 8eps()
            # inverses
            for c in (A, U, S)
                @test all(x -> inv(c)(c(x)) ≈ x, xs)
            end
            # eltypes: Float32, and ForwardDiff duals through with_parameters
            @test convert_eltype(Float32, A) isa Aspect{N, Float32} && convert_eltype(Float32, U) isa Skew{N, Float32, L} &&
                convert_eltype(Float32, S) isa Scale{Float32}
            @test parameter_vector(convert_eltype(Float32, U)) == Float32.(u)
            d = ForwardDiff.Dual(0.01, 1.0)
            @test with_parameters(A, fill(d, N - 1)) isa Aspect{N, typeof(d)}
            # GAUGE-FREE: at the identity, Aspect ∘ Skew's tangents and the pose's (rotation,
            # translation) and scale's span the whole affine algebra, and are independent
            P = affine_tangents(Aspect{N}() ∘ Skew{N}(), N)
            rot = reduce(hcat, [vcat(vec(G), zeros(N)) for G in
                                (SMatrix{N, N}(ntuple(q -> ((r, s) = (mod1(q, N), (q - 1) ÷ N + 1); Float64((r, s) == (i, j)) - Float64((r, s) == (j, i))), N * N))
                                 for i in 1:N for j in (i + 1):N)])
            tra = vcat(zeros(N * N, N), Matrix{Float64}(I, N, N))
            sca = reshape([vec(Matrix{Float64}(I, N, N)); zeros(N)], :, 1)
            pose_scale = [rot tra sca]
            @test size(P, 2) == (N - 1) + L == N^2 + N - (L + N + 1)            # the count
            @test rank(P) == size(P, 2) && rank(pose_scale) == L + N + 1
            @test rank([pose_scale P]) == N^2 + N                               # independent: together, every affine direction
        end
    end

    @testset "N = 2: the known-pose composition spans the affine maps at any point" begin
        E = with_parameters(EuclideanMap(Rotations.RotMatrix{2}(0.0), SVector(0.0, 0.0)), [0.4, 0.01, -0.02])
        K = E ∘ Scale(0.02) ∘ Aspect(-0.03) ∘ Skew(0.05)
        @test rank(affine_tangents(K, 2)) == 6
    end
end

@testitem "target calibration: lengths and an uncertain calibration" begin
    using Test, StaticArrays, LinearAlgebra, Statistics
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, mm, m, has_units, transform_deriv_params
    using FlexUnits: DimensionError, ustrip
    using Rotations: RotMatrix

    pts = [SVector(0.0, 0.0), SVector(100.0, 0.0), SVector(0.0, 50.0), SVector(100.0, 50.0)]
    pts_mm = [p .* mm for p in pts]
    nodes_mm = [p .* mm for p in grid_nodes(3; lo = 0.0, hi = 100.0)]
    rigid(t) = EuclideanMap(RotMatrix{2}(0.01), t)
    AU = Aspect(0.001) ∘ Skew(0.0005)

    @testset "a physical calibration moves physical points" begin
        K = rigid(SVector(0.5, -0.25) .* mm) ∘ AU
        t = PlanarTarget(pts_mm; calibration = K)
        @test has_units(K) && has_units(t)
        @test parameter_vector(K)[2:3] ≈ [0.0005, -0.00025]                 # lengths in θ: metres
        P = points(t)
        @test all(p -> eltype(p) <: VgtkCore.Quantity, P)
        tp = ustrip(mm, t)
        @test !has_units(tp) && !has_units(calibration(tp))
        @test [ustrip.(mm, p) for p in P] ≈ points(tp)
        @test [ustrip.(mm, p) for p in points(VgtkCore.with_unit(tp, mm))] ≈ [ustrip.(mm, p) for p in P]
        @test [ustrip.(mm, p) for p in points(VgtkCore.with_unit(tp, mm))] ≈
            [ustrip.(mm, p) for p in points(PlanarTarget(pts_mm; calibration = VgtkCore.with_unit(calibration(tp), mm)))]
        # one form: a plain calibration on lengths, or the reverse, is refused
        @test_throws DimensionError PlanarTarget(pts_mm; calibration = rigid(SVector(0.5, 0.0)) ∘ AU)
        @test_throws DimensionError PlanarTarget(pts; calibration = K)
        # the dimensionless factors sit with either form
        @test [ustrip.(mm, p) for p in points(PlanarTarget(pts_mm; calibration = AU))] ≈ [ustrip.(mm, AU(p)) for p in pts_mm]
        @test points(PlanarTarget(pts; calibration = AU)) == [AU(p) for p in pts]
        @test points(PlanarTarget(pts_mm)) == pts_mm && points(PlanarTarget(pts)) == pts
        @test ustrip(mm, Aspect(0.1)) === Aspect(0.1) && VgtkCore.with_unit(Skew(0.1), mm) === Skew(0.1)
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
        @test_throws DimensionError PlanarTarget(pts_mm; calibration = H ∘ rigid(SVector(0.0, 0.0)) ∘ AU)
        @test [ustrip.(mm, p) for p in points(PlanarTarget(pts_mm; calibration = H ∘ AU))] ≈ [ustrip.(mm, H(AU(p))) for p in pts_mm]
    end

    @testset "Uncertain(calibration, Σ)" begin
        K = calibration(PlanarTarget(pts_mm; calibration = rigid(SVector(0.5, -0.25) .* mm) ∘ AU))   # t stored in metres
        Σ = SMatrix{5, 5}(Diagonal([1.0e-8, 1.0e-10, 1.0e-10, 1.0e-8, 1.0e-8]))   # t's in m²
        u = Uncertain(K, CovMat(Σ))
        @test geom(u) === K && Statistics.cov(u) == Σ
        @test parameter_vector(u) == parameter_vector(K) && parameter_names(u) == parameter_names(K)
        @test transform_deriv_params(u, pts[2]) == transform_deriv_params(K, pts[2])
        t = PlanarTarget(pts_mm; calibration = u)
        @test calibration(t) === u && t isa PlanarTarget{typeof(K)}
        @test points(t) == points(PlanarTarget(pts_mm; calibration = K))
        # the covariance moves with the unit: the length entries in mm², the rest unchanged
        Σmm = Statistics.cov(calibration(ustrip(mm, t)))
        @test Σmm[1, 1] == Σ[1, 1] && Σmm[4:5, 4:5] == Σ[4:5, 4:5] && Σmm[2:3, 2:3] ≈ 1.0e6 .* Σ[2:3, 2:3]
        @test Statistics.cov(calibration(VgtkCore.with_unit(ustrip(mm, t), mm))) ≈ Σ
        # a new parameter vector is a new calibration: no covariance
        @test !(calibration(with_parameters(t, parameter_vector(t))) isa VgtkCore.Attributed)
        @test_throws DimensionMismatch PlanarTarget(pts_mm; calibration = Uncertain(K, CovMat(SMatrix{2, 2}(1.0, 0, 0, 1.0))))
    end
end

@testitem "target calibration: planarity is in the type" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, PlanarCalibration, mm

    nodes = grid_nodes(3; lo = 0.0, hi = 100.0)
    A = Aspect(0.001) ∘ Skew(0.001)
    T = ThinPlateSpline(nodes, fill(0.01, 9), fill(-0.01, 9))
    H = HeightField(nodes, [0.0, 0.1, 0.0, 0.2, 0.0, -0.1, 0.0, 0.0, 0.3])
    pts = [SVector(0.0, 0.0), SVector(100.0, 0.0), SVector(0.0, 50.0), SVector(30.0, 40.0)]

    @testset "is_planar by type" begin
        @test is_planar(Rigid) && is_planar(typeof(A)) && is_planar(typeof(T)) && !is_planar(typeof(H))
        @test !is_planar(typeof(H ∘ T)) && !is_planar(typeof(H ∘ A))
        @test is_planar(A) && !is_planar(H) && is_planar(PlanarTarget(pts; calibration = T)) && !is_planar(PlanarTarget(pts; calibration = H))
        @test is_planar(Uncertain(A, CovMat(SMatrix{2, 2}(1.0I)))) && !is_planar(Uncertain(H, CovMat(SMatrix{9, 9}(1.0I))))
        @test (@inferred is_planar(typeof(H ∘ T))) === false
        @test (@inferred is_planar(typeof(H ∘ A))) === false
        @test A isa PlanarCalibration && T isa PlanarCalibration && Rigid() isa PlanarCalibration && Aspect(0.1) isa PlanarCalibration
        @test !(H isa PlanarCalibration) && !((H ∘ A) isa PlanarCalibration)
    end

    @testset "points: SVector{2} when planar, SVector{3} otherwise (inferred)" begin
        for c in (Rigid(), A, T)
            t = PlanarTarget(pts; calibration = c)
            P = @inferred points(t)
            @test eltype(P) === SVector{2, Float64}
        end
        for c in (H, H ∘ A, H ∘ T)
            t = PlanarTarget(pts; calibration = c)
            P = @inferred points(t)
            @test eltype(P) === SVector{3, Float64}
            @test P == [c(p) for p in pts]
        end
        tmm = PlanarTarget([p .* mm for p in pts]; calibration = VgtkCore.with_unit(H, mm))
        @test eltype(@inferred points(tmm)) <: SVector{3, <:VgtkCore.Quantity}
        # a method written for the plane refuses a height field by dispatch
        on_plane(t::PlanarTarget{<:PlanarCalibration}) = length(points(t))
        @test on_plane(PlanarTarget(pts; calibration = T)) == 4
        @test on_plane(PlanarTarget(pts; calibration = A)) == 4
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
        retract_jacobian, action_jacobian, local_coordinates, restrict, gauge_names, gauge_directions, model, transform_deriv_params

    rng = Xoshiro(7)
    nodes = grid_nodes(4; lo = -0.1, hi = 0.1)                         # 16 nodes: an in-plane spline has n = 32
    pts = [SVector(0.2 * rand(rng) - 0.1, 0.2 * rand(rng) - 0.1) for _ in 1:12]
    A = Aspect(0.002) ∘ Skew(-0.0015)
    H = HeightField(nodes, 1.0e-3 .* randn(rng, 16))
    T = ThinPlateSpline(nodes, 1.0e-4 .* randn(rng, 16), 1.0e-4 .* randn(rng, 16))
    unit_cols(D) = D ./ norm.(eachcol(D))'

    @testset "the affine factors: no gauges, every parameter free" begin
        t = PlanarTarget(pts; calibration = A)
        @test gauge_names(A) == () && gauge_names(t) == () && size(gauge_directions(A)) == (2, 0)
        ch = minimal_manifold(t)
        @test ch isa TargetManifold && tangent_size(ch) == 2 && retract_jacobian(ch) == I
        v = SVector{2}(1.0e-4 .* randn(rng, 2))
        @test local_coordinates(ch, retract(ch, v)) ≈ v
        @test action_jacobian(ch, pts[1]) ≈ transform_deriv_params(A, pts[1])
        @test_throws ArgumentError TargetManifold(t; free = :scale)        # nothing to free
        r = recenter(ch, v)
        @test typeof(r) === typeof(ch) && model(r) == retract(ch, v)
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

    @testset "a composition: the members' gauges, prefixed, the outer member's first" begin
        C = H ∘ T
        t = PlanarTarget(pts; calibration = C)
        @test gauge_names(C) == (:height_field_piston, :height_field_tilt_x, :height_field_tilt_y,
            :thin_plate_spline_fx_1, :thin_plate_spline_fx_u, :thin_plate_spline_fx_v,
            :thin_plate_spline_fy_1, :thin_plate_spline_fy_u, :thin_plate_spline_fy_v)
        ch = minimal_manifold(t)
        @test tangent_size(ch) == 16 + 32 - 9
        @test norm(retract_jacobian(ch)' * unit_cols(gauge_directions(C))) < 1.0e-13
        CA = H ∘ A
        tA = PlanarTarget(pts; calibration = CA)
        @test gauge_names(CA) == (:height_field_piston, :height_field_tilt_x, :height_field_tilt_y)
        @test tangent_size(minimal_manifold(tA)) == 16 + 2 - 3
    end

    @testset "restrict fixes only the gauge combinations inside the free span" begin
        tH = PlanarTarget(pts; calibration = H)
        @test tangent_size(restrict(minimal_manifold(tH), (:h1, :h2, :h3))) == 3
        @test tangent_size(restrict(minimal_manifold(tH), (:h1, :h2, :h3, :h4, :h5))) == 5
        tA = PlanarTarget(pts; calibration = A)
        @test tangent_size(restrict(minimal_manifold(tA), (:skew_u12,))) == 1
        @test_throws ArgumentError restrict(minimal_manifold(tA), (:h1,))
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
            @test tangent_size(minimal_manifold(PlanarTarget(pts_m; calibration = Hm ∘ A))) == 2 + 16 - 3
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
        @test J isa Matrix{Float64} && size(J) == (2, 44) && J ≈ transform_deriv_params(T5, pts[1]) * B
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
    using Rotations: RotMatrix

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

    A = Aspect(-0.01) ∘ Skew(0.02)
    T = ThinPlateSpline(nodes, 1.0e-4 .* randn(rng, 9), 1.0e-4 .* randn(rng, 9))
    H0 = HeightField(nodes)
    H = HeightField(nodes, 1.0e-3 .* randn(rng, 9))
    E = with_parameters(EuclideanMap(RotMatrix{2}(0.0), SVector(0.0, 0.0)), [0.02, 0.003, -0.002])
    cases = (
        "Aspect ∘ Skew" => A, "ThinPlateSpline" => T, "HeightField flat" => H0, "HeightField h ≈ 1 mm" => H,
        "HeightField ∘ Aspect ∘ Skew" => H ∘ A, "HeightField flat ∘ Aspect ∘ Skew" => H0 ∘ A,
        "HeightField ∘ ThinPlateSpline" => H ∘ T, "HeightField ∘ TPS at zero displacement" => H ∘ ThinPlateSpline(nodes),
    )
    for (name, c) in cases
        @testset "$name" begin
            @test nullity_default(c) == 0
        end
    end
    # the harness can see a gauge: the pose's directions, put back into the
    # composition, come back as exactly their count — the scale (1) and the rigid
    # member (3)
    @test nullity_default(Scale(0.01) ∘ A) == 1
    @test nullity_default(E ∘ Scale(0.01) ∘ A) == 4
end

@testitem "target calibration: names, hashes, AD on lengths, an uncertain calibration" begin
    using Test, StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, gauge_names, mm, m, Attributed, Uncertain, CovMat, transform_deriv_params
    using FlexUnits: ustrip
    using Rotations: RotMatrix

    nodes = grid_nodes(3; lo = 0.0, hi = 100.0)
    A = Aspect(0.001) ∘ Skew(0.0005)
    E = EuclideanMap(RotMatrix{2}(0.01), SVector(0.5, -0.25))
    T = ThinPlateSpline(nodes, fill(0.01, 9), fill(-0.01, 9))
    H = HeightField(nodes, [0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0])
    C = H ∘ T
    pts = [SVector(0.0, 0.0), SVector(100.0, 0.0), SVector(30.0, 40.0)]

    @testset "a composition's names are inferred" begin
        @test (@inferred parameter_names(C)) == parameter_names(C) && length(parameter_names(C)) == 9 + 18
        @test (@inferred gauge_names(C)) == gauge_names(C)
        t = PlanarTarget(pts; calibration = H ∘ A)
        @test (@inferred VgtkCore.parameters(t)).aspect_a1 == 0.001
        @test (@inferred parameter_names(E ∘ Scale(0.0) ∘ A)) == (:euclidean_angle, :euclidean_tx, :euclidean_ty, :scale_log_scale, :aspect_a1, :skew_u12)
    end

    @testset "one tag registry: TARGET_CALIBRATION_TAGS" begin
        R = VgtkCore.TARGET_CALIBRATION_TAGS
        @test collect(keys(R)) == ["rigid", "euclidean", "scale", "aspect", "skew", "thin_plate_spline", "height_field"]
        @test isempty(VgtkCore.tag_rule_violations(R))                      # `EuclideanMap`, the suffix `Map` stripped
        @test [VgtkCore.tag_of(R, c) for c in (Rigid(), E, Scale(), Aspect(0.0), Skew(0.0), T, H)] == collect(keys(R))
        @test R["euclidean"] === EuclideanMap && R["aspect"] === Aspect
        e = try R["affine"]; nothing catch err; err end                    # retired: an AffineMap is Aspect ∘ Skew (∘ …)
        @test e isa ArgumentError && occursin("renamed \"aspect\"", e.msg)
        @test VgtkCore.retired_tags(R) == ["affine" => "aspect"]
        @test_throws ArgumentError R["warp"]
        @test_throws ArgumentError R["calibration_chain"]                   # a composition has no tag
        @test_throws ArgumentError VgtkCore.calibration_tag(typeof(C))
        @test VgtkCore.calibration_tag(typeof(E)) == "euclidean" && VgtkCore.calibration_tag(typeof(Skew(0.0))) == "skew"
        @test parameter_names(C)[1] === :height_field_h1 && parameter_names(C)[end] === :thin_plate_spline_fy9   # the prefixes are the tags
    end

    @testset "Attributed: ==, isequal and hash by value and attributes" begin
        Σ(n) = CovMat(SMatrix{n, n, Float64}(1.0e-6I))
        uT, uT2 = Uncertain(T, Σ(18)), Uncertain(ThinPlateSpline(nodes, fill(0.01, 9), fill(-0.01, 9)), Σ(18))
        @test uT == uT2 && isequal(uT, uT2) && hash(uT) == hash(uT2) && length(Set([uT, uT2])) == 1
        @test uT != Uncertain(T, CovMat(SMatrix{18, 18, Float64}(2.0e-6I)))
        # whole targets compare equal, an uncertain thin-plate composition included
        t1 = PlanarTarget(pts; calibration = Uncertain(C, Σ(27)))
        t2 = PlanarTarget(copy(pts); calibration = Uncertain(HeightField(nodes, [0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0]) ∘
            ThinPlateSpline(nodes, fill(0.01, 9), fill(-0.01, 9)), Σ(27)))
        @test t1 == t2 && hash(t1) == hash(t2)
        a = Attributed(t1, (; calibration_record = (; tool = "x")))
        @test a == Attributed(t2, (; calibration_record = (; tool = "x"))) && a != Attributed(t2, (; calibration_record = (; tool = "y")))
    end

    @testset "Uncertain(Rigid(), Σ) is refused: a rigid calibration has no parameters" begin
        e = try Uncertain(Rigid(), CovMat(SMatrix{0, 0, Float64}())); nothing catch e; e end
        @test e isa ArgumentError && occursin("no parameters, so no covariance", e.msg)
    end

    @testset "hash agrees with ==" begin
        H2() = HeightField(nodes, [0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0])
        T2() = ThinPlateSpline(nodes, fill(0.01, 9), fill(-0.01, 9))
        A2() = Aspect(0.001) ∘ Skew(0.0005)
        for (x, y) in ((A, A2()), (T, T2()), (H, H2()), (C, H2() ∘ T2()), (H ∘ A, H2() ∘ A2()),
                       (PlanarTarget(pts; calibration = H), PlanarTarget(pts; calibration = H2())),
                       (PlanarTarget(pts; calibration = C), PlanarTarget(pts; calibration = H2() ∘ T2())),
                       (PlanarTarget(pts; calibration = H ∘ A), PlanarTarget(pts; calibration = H2() ∘ A2())))
            @test x == y && hash(x) == hash(y)
            @test length(Set([x, y])) == 1
        end
        @test hash(A) != hash(with_parameters(A, parameter_vector(A) .+ 1.0e-3))
        @test hash(C) != hash(with_parameters(C, parameter_vector(C) .+ 1.0e-3))
    end

    @testset "ForwardDiff through a physical calibration" begin
        pmm = SVector(10.0, 20.0) .* mm
        for c in (VgtkCore.with_unit(E ∘ A, mm), VgtkCore.with_unit(H, mm), VgtkCore.with_unit(H ∘ A, mm), VgtkCore.with_unit(H ∘ (E ∘ (Scale(0.01) ∘ A)), mm))
            J = ForwardDiff.jacobian(θ -> ustrip.(m, with_parameters(c, θ)(pmm)), Vector(parameter_vector(c)))
            @test J ≈ transform_deriv_params(c, pmm) atol = 1.0e-14
        end
    end

    @testset "an uncertain calibration: with_parameters returns the bare kind" begin
        u = Uncertain(A, CovMat(SMatrix{2, 2}(1.0e-6I)))
        θ = parameter_vector(A) .+ 1.0e-4
        @test with_parameters(u, θ) == with_parameters(A, θ) && parameter_vector(with_parameters(u, θ)) == θ
        @test !(calibration(with_parameters(PlanarTarget(pts; calibration = u), θ)) isa Attributed)
    end

    @testset "an attributed target's calibration record does not shadow its calibration" begin
        a = Attributed(PlanarTarget(pts; calibration = H), (; calibration_record = (; tool = "a ruler", n_images = 3)))
        @test a.calibration === H && calibration(a.value) === H           # the field, forwarded: the target's kind
        @test a.calibration_record == (; tool = "a ruler", n_images = 3)  # the attribute: who calibrated it
    end
end

@testitem "target calibration: a metre move keeps the thin-plate solver" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, m, mm, with_unit, surface
    using FlexUnits: ustrip
    using Rotations: RotMatrix
    # Lengths are stored in metres, so a strip to (or attachment of) metres leaves
    # the nodes and values as they are: the solver, the coefficients and the
    # bending factor are the same objects, not a refit.
    nodes = grid_nodes(5; lo = 0.0, hi = 0.2)
    T = with_unit(ThinPlateSpline(nodes, fill(1.0e-4, 25), fill(-1.0e-4, 25)), m)
    H = with_unit(HeightField(nodes, fill(1.0e-4, 25)), m)
    for c in (T, H)
        p = ustrip(m, c)
        @test p.system === c.system && p.bending === c.bending && !p.physical
        q = with_unit(p, m)
        @test q.system === c.system && q == c
        @test ustrip(mm, c).system !== c.system            # another unit moves the nodes: a refit
    end
    # surface() of a physical target strips its calibration to metres: no refit
    # (a 5 × 5 refit forms and inverts a 28 × 28 system and an eigendecomposition)
    pts = [SVector(x, y) for x in 0.0:0.05:0.2 for y in 0.0:0.05:0.2]
    X = EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 1.0))
    Hplain = ustrip(m, H)
    pt = with_unit(PosedTarget(PlanarTarget(pts; calibration = Hplain), X), m)
    surface(pt)
    @test (@allocated surface(pt)) < 16_384
end

@testitem "target calibration: is_affine, and the surface chart read off it" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, is_affine, transform_deriv
    using Rotations: RotMatrix
    nodes = grid_nodes(3; lo = 0.0, hi = 1.0)
    E = EuclideanMap(RotMatrix{2}(0.02), SVector(0.004, -0.002))
    K = E ∘ (Scale(0.01) ∘ (Aspect(0.01) ∘ Skew(-0.003)))
    H = HeightField(nodes, fill(1.0e-3, 9))
    T = ThinPlateSpline(nodes)
    @test is_affine(Rigid()) && is_affine(K) && is_affine(H) && is_affine(H ∘ K) && is_affine(Aspect(0.1))
    @test !is_affine(T) && !is_affine(H ∘ T)
    @test is_affine(typeof(H ∘ K))                                       # a function of the type
    # the chart's affine: the calibration's image of the design origin and its point Jacobian there
    o = SVector(0.0, 0.0)
    M, b = VgtkCore.Cameras._chart_affine(K)
    @test M ≈ transform_deriv(K, o) && b ≈ K(o) && b ≈ SVector(0.004, -0.002)
    @test VgtkCore.Cameras._chart_affine(H) == (SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0), SVector(0.0, 0.0))
    M2, b2 = VgtkCore.Cameras._chart_affine(H ∘ K)
    @test M2 == M && b2 == b
    @test_throws ArgumentError VgtkCore.Cameras._chart_affine(T)
    # a composition is refused for its in-plane kind
    e = try VgtkCore.Cameras._chart_affine(H ∘ T); nothing catch err; err end
    @test e isa ArgumentError && occursin("ThinPlateSpline", e.msg)
end

@testitem "target calibration: length_mask" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, length_mask, parameter_vector
    using Rotations: RotMatrix
    nodes = grid_nodes(2; lo = 0.0, hi = 1.0)
    E = EuclideanMap(RotMatrix{2}(0.0), SVector(0.0, 0.0))
    A = Aspect(0.0) ∘ Skew(0.0)
    T = ThinPlateSpline(nodes)
    H = HeightField(nodes)
    @test length_mask(Rigid()) == SVector{0, Bool}()
    @test length_mask(E) === SVector(false, true, true) === length_mask(typeof(E))
    @test length_mask(Scale()) === SVector(false) && length_mask(Aspect(SVector(0.0, 0.0))) === SVector(false, false)
    @test length_mask(Skew(SVector(0.0, 0.0, 0.0))) === SVector(false, false, false)
    @test length_mask(T) == trues(8) && length_mask(H) == trues(4)
    @test (@inferred length_mask(typeof(E ∘ Scale() ∘ A))) == [false, true, true, false, false, false]
    for C in (H ∘ A, H ∘ T)
        @test (@inferred length_mask(typeof(C))) == vcat(length_mask(H), length_mask(C.t2))
        @test length(length_mask(C)) == length(parameter_vector(C))
    end
end

@testitem "target calibration: with_unit of a non-length unit" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, with_unit, px
    using FlexUnits: DimensionError, ConversionError
    using Rotations: RotMatrix
    # A thin-plate kind refuses a unit that is not a length with FlexUnits'
    # DimensionError, as at 0.2's base (the metre shortcut checks the dimension
    # before it converts); a rigid member's translation raises FlexUnits' own
    # error for a non-length.
    nodes = grid_nodes(3; lo = 0.0, hi = 1.0)
    @test_throws DimensionError with_unit(ThinPlateSpline(nodes), px)
    @test_throws DimensionError with_unit(HeightField(nodes), px)
    @test_throws Union{DimensionError, ConversionError} with_unit(EuclideanMap(RotMatrix{2}(0.0), SVector(0.0, 0.0)) ∘ Aspect(0.0), px)
end
