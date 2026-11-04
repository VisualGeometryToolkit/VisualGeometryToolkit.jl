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
    using Rotations: Rotations, RotMatrix, Angle2d

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

    @testset "with_parameters ∘ parameter_vector round trips" begin
        for c in kinds
            θ = parameter_vector(c)
            # a step the calibration may take: along its manifold (a thin-plate kind's values keep no plane part)
            B = Matrix(VgtkCore.retract_jacobian(VgtkCore.TargetManifold(PlanarTarget(xs; calibration = c); known = (:pose,))))
            θ′ = θ .+ B * (1.0e-5 .* randn(rng, size(B, 2)))
            # exact for every kind: the rigid member stores its angle (an Angle2d), not a matrix
            @test with_parameters(c, θ) == c
            @test parameter_vector(with_parameters(c, θ′)) == θ′
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

    @testset "the rigid member's angle is its value: exact round trips" begin
        @test VgtkCore.rotation(E) isa Angle2d{Float64}
        mt = VgtkCore.m
        for _ in 1:2000
            θ = [π * (2 * rand(rng) - 1), 0.01 * randn(rng), 0.01 * randn(rng)]
            c = with_parameters(E, θ)
            @test parameter_vector(c) == θ
            s = calibration(PlanarTarget(xs; calibration = c))                       # what a target stores
            @test s == c && parameter_vector(s) == θ
            p = with_parameters(VgtkCore.with_unit(E, mt), θ)                                # physical: the translation in metres
            @test parameter_vector(p) == θ && parameter_vector(ustrip(mt, p)) == θ
            @test parameter_vector(with_parameters(K, [θ; 0.001; 0.002; -0.0015]))[1:3] == θ
        end
        # a map built from a matrix is converted once, where a target stores it: its angle becomes the value
        R = RotMatrix{2}(0.7)
        s = calibration(PlanarTarget(xs; calibration = EuclideanMap(R, SVector(0.001, 0.002))))
        @test VgtkCore.rotation(s) isa Angle2d{Float64} && VgtkCore.rotation(s).theta === Rotations.rotation_angle(R)
        @test SMatrix{2, 2}(VgtkCore.rotation(s)) ≈ R rtol = 1.0e-15
        @test parameter_vector(with_parameters(s, parameter_vector(s))) == parameter_vector(s)
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
        for c in (U1 ∘ A1, A1 ∘ S1, S1 ∘ E, A1 ∘ A1, A1 ∘ T, E ∘ T, T ∘ T, T ∘ H, A1 ∘ H, H ∘ Rigid(), Rigid() ∘ A1, T ∘ Rigid(),
                  H ∘ H, H ∘ U1 ∘ A1, H ∘ A1 ∘ T)
            @test_throws ArgumentError PlanarTarget(xs; calibration = c)
        end
        # the canonical print error, HeightField ∘ ThinPlateSpline ∘ Aspect ∘ Skew, in any association
        for c in (H ∘ T ∘ A1 ∘ U1, H ∘ (T ∘ (A1 ∘ U1)), (H ∘ T) ∘ (A1 ∘ U1), T ∘ A1 ∘ U1, H ∘ T ∘ E ∘ S1 ∘ A1 ∘ U1)
            s = calibration(PlanarTarget(xs; calibration = c))
            @test parameter_vector(s) == parameter_vector(c) && s == foldr(ComposedTransformation, VgtkCore.members(c))
        end
        # a target stores a composition right-associated, whatever ∘ built
        for c in (E ∘ S1 ∘ A1 ∘ U1, E ∘ (S1 ∘ (A1 ∘ U1)), (E ∘ S1) ∘ (A1 ∘ U1), H ∘ A1 ∘ U1, H ∘ (A1 ∘ U1))
            s = calibration(PlanarTarget(xs; calibration = c))
            @test s == foldr(ComposedTransformation, VgtkCore.members(c))
            @test parameter_vector(s) == parameter_vector(c) && parameter_names(s) == parameter_names(c)
        end
        # the structure, public: the members, the canonical order, a kind's planar type
        @test VgtkCore.members(H ∘ (A1 ∘ U1)) == VgtkCore.members((H ∘ A1) ∘ U1) == (H, A1, U1) && VgtkCore.members(A1) == (A1,)
        @test map(VgtkCore.calibration_tag, VgtkCore.TARGET_CALIBRATION_ORDER) ==
            ("height_field", "thin_plate_spline", "euclidean", "scale", "aspect", "skew")
        @test [length(VgtkCore.length_mask(VgtkCore.planar_kind(K, 4))) for K in VgtkCore.TARGET_CALIBRATION_ORDER] == [4, 8, 3, 1, 1, 1]
        @test length(VgtkCore.length_mask(VgtkCore.planar_kind(Rigid))) == 0
        # the Iwasawa parameters: the factors compose back to A; the Jacobian is the parameters' (central differences)
        for A in (SMatrix{2, 2}(1.001, -0.003, 0.002, 0.999), SMatrix{2, 2}(0.7, 2.1, -1.3, 0.4))
            θ, J = VgtkCore.iwasawa_parameters_with_jacobian(A)
            @test θ == VgtkCore.iwasawa_parameters(A)
            F = EuclideanMap(VgtkCore.Cameras.Angle2d(θ[1]), SVector(0.0, 0.0); check = false) ∘ Scale(θ[2]) ∘ Aspect(θ[3]) ∘ Skew(θ[4])
            @test F(SVector(0.3, -0.7)) ≈ A * SVector(0.3, -0.7) rtol = 1.0e-14
            g(a) = VgtkCore.iwasawa_parameters(SMatrix{2, 2}(a[1], a[3], a[2], a[4]))
            a0 = SVector(A[1, 1], A[1, 2], A[2, 1], A[2, 2])
            Jfd = reduce(hcat, [(h = 1.0e-6 * SVector(ntuple(i -> Float64(i == k), 4)); (g(a0 + h) - g(a0 - h)) / 2.0e-6) for k in 1:4])
            @test J ≈ Jfd rtol = 1.0e-7
        end
        @test_throws ArgumentError VgtkCore.iwasawa_parameters(SMatrix{2, 2}(1.0, 0.0, 0.0, -1.0))
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
        w, a = tps_coefficients(sys, Vector(parameter_vector(hf)))          # the stored heights: their plane part projected off
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
    using VisualGeometryToolkitCore: transform_deriv, transform_deriv_params, convert_eltype, length_mask
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
                @test !any(length_mask(c)) && isempty(regularizer(c))
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
        # (the field is projected in metres, its stored form; its plain form in mm is
        # the same geometry, not a field projected in millimetre numbers — the
        # kernel's scale moves a piston, see HeightField)
        Hp = ustrip(mm, H)
        t = PlanarTarget(pts_mm; calibration = H)
        @test points(ustrip(mm, t)) ≈ points(PlanarTarget(pts; calibration = Hp))
        @test ustrip.(mm, points(t)[2]) ≈ Hp(pts[2])
        @test VgtkCore.with_unit(Hp, mm) == H || parameter_vector(VgtkCore.with_unit(Hp, mm)) ≈ parameter_vector(H)
        Hmm = HeightField(grid_nodes(3; lo = 0.0, hi = 100.0), [0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0])
        # built from millimetre numbers, the field is projected in those: it differs from the metre one by a piston only
        dz(p) = ustrip(mm, H(SVector(p[1] * mm, p[2] * mm))[3]) - Hmm(p)[3]
        @test all(p -> abs(dz(p) - dz(pts[1])) < 1.0e-9, pts) && abs(dz(pts[1])) > 1.0e-6
        T = ThinPlateSpline(nodes_mm, fill(0.01mm, 9), fill(-0.01mm, 9))
        @test ustrip.(mm, T(pts_mm[2])) ≈ pts[2]            # a constant displacement is a translation: no part of a spline
        Tb = ThinPlateSpline(nodes_mm, [0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0] .* mm, fill(0.0mm, 9))
        Tp = ustrip(mm, Tb)
        @test ustrip.(mm, Tb(pts_mm[3])) ≈ Tp(pts[3]) && Tp(pts[3]) != pts[3]   # bending survives, in lengths as in plain numbers
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

@testitem "target calibration: the thin-plate kinds are pure bending; the target manifold" begin
    using Test, StaticArrays, LinearAlgebra, Random, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, TargetManifold, minimal_manifold, tangent_size, retract, recenter,
        retract_jacobian, action_jacobian, local_coordinates, restrict, model, transform_deriv_params, bending_energy,
        ThinPlateSplineSystem, tps_coefficients, tps_evaluate
    import Rotations

    rng = Xoshiro(7)
    nodes = grid_nodes(4; lo = -0.1, hi = 0.1)                         # 16 nodes: an in-plane spline has n = 32
    pts = [SVector(0.2 * rand(rng) - 0.1, 0.2 * rand(rng) - 0.1) for _ in 1:12]
    A = Aspect(0.002) ∘ Skew(-0.0015)
    sys = ThinPlateSplineSystem(nodes)
    M = [ones(16) [p[1] for p in nodes] [p[2] for p in nodes]]         # the plane fields 1, u, v at the nodes
    Aff = sys.coeff[17:19, :]                                          # the interpolant's affine coefficients: a(v) = Aff v
    plane_part(v) = norm(M * (Aff * v)) / norm(v)                      # the affine term at the nodes, relative
    fx, fy, h = 1.0e-4 .* randn(rng, 16), 1.0e-4 .* randn(rng, 16), 1.0e-3 .* randn(rng, 16)
    H = HeightField(nodes, h)
    T = ThinPlateSpline(nodes, fx, fy)

    @testset "the projection: exact, idempotent, the identity on an affine displacement" begin
        @test plane_part(fx) > 1.0e-3 && plane_part(h) > 1.0e-3                # what was supplied has an affine term
        @test plane_part(T.fx) < 1.0e-12 && plane_part(T.fy) < 1.0e-12 && plane_part(H.h) < 1.0e-12
        @test norm(H.a) < 1.0e-15 * norm(H.h) && norm(T.ax) < 1.0e-15 * norm(T.fx) && norm(T.ay) < 1.0e-15 * norm(T.fy)
        # the stored values are the supplied ones less their interpolant's affine term: the bending kept
        Π = I - M * Aff
        @test isapprox(T.fx, Π * fx; rtol = 1.0e-12) && isapprox(T.fy, Π * fy; rtol = 1.0e-12) && isapprox(H.h, Π * h; rtol = 1.0e-12)
        @test isapprox(H.w, tps_coefficients(sys, h)[1]; rtol = 1.0e-10)       # the same bending weights
        # idempotent, bit for bit: a spline rebuilt from its stored values is the same spline
        @test ThinPlateSpline(nodes, T.fx, T.fy) == T && HeightField(nodes, H.h) == H
        @test with_parameters(T, parameter_vector(T)) == T && with_parameters(H, parameter_vector(H)) == H
        @test parameter_vector(with_parameters(H, Π * h)) == Π * h || plane_part(Π * h) > 1.0e-12
        # RadiometricCamera's board heights — its spline's bending part at its chart nodes — are kept bit for bit
        rc = grid_nodes(5; lo = 0.0, hi = 1.0)
        src = ThinPlateSplineSystem(rc)
        wb, _ = tps_coefficients(src, 1.0e-3 .* randn(rng, 25))
        z = [tps_evaluate(src.nodes, wb, zero(SVector{3, Float64}), p) for p in rc]
        @test parameter_vector(HeightField(rc, z)) == z
        # an affine displacement is not the spline's: it is the identity (a plane of heights: flat)
        Ta = ThinPlateSpline(nodes, [1.0e-3 + 2.0e-3 * p[1] - 1.0e-3 * p[2] for p in nodes], [3.0e-3 * p[2] - 1.0e-4 for p in nodes])
        Ha = HeightField(nodes, [1.0e-4 + 2.0e-3 * p[1] - 1.0e-3 * p[2] for p in nodes])
        @test norm(parameter_vector(Ta)) < 1.0e-16 && norm(parameter_vector(Ha)) < 1.0e-16     # of displacements ~1e-3: rounding
        @test all(x -> Ta(x) ≈ x && norm(Ha(x) - SVector(x[1], x[2], 0.0)) < 1.0e-16, pts)
        # the bending energy does not see the affine term: unchanged by the projection
        @test sum(abs2, regularizer(H)) ≈ bending_energy(sys, h) rtol = 1.0e-10
        @test sum(abs2, regularizer(T)) ≈ bending_energy(sys, fx) + bending_energy(sys, fy) rtol = 1.0e-10
        # physical: the same projection in metres
        Hm = HeightField([p .* VgtkCore.m for p in nodes], h .* VgtkCore.m)
        @test parameter_vector(Hm) == parameter_vector(H)
    end

    @testset "the affine factors: every parameter free" begin
        t = PlanarTarget(pts; calibration = A)
        ch = minimal_manifold(t)
        @test ch isa TargetManifold && tangent_size(ch) == 2 && retract_jacobian(ch) == I
        v = SVector{2}(1.0e-4 .* randn(rng, 2))
        @test local_coordinates(ch, retract(ch, v)) ≈ v
        @test action_jacobian(ch, pts[1]) ≈ transform_deriv_params(A, pts[1])
        r = recenter(ch, v)
        @test typeof(r) === typeof(ch) && model(r) == retract(ch, v)
        @test only(Base.return_types(recenter, (typeof(ch), typeof(v)))) === typeof(ch)
    end

    @testset "the pose's gauges: EuclideanMap and Scale refused unless known" begin
        E = with_parameters(EuclideanMap(one(Rotations.RotMatrix{2, Float64}), SVector(0.0, 0.0)), [0.1, 0.002, -0.001])
        K = E ∘ Scale(0.001) ∘ A                                       # a migrated affine: all four factors
        tK, tS, tE = (PlanarTarget(pts; calibration = c) for c in (K, Scale(0.001) ∘ A, E))
        for t in (tK, tS, tE)
            @test_throws ArgumentError minimal_manifold(t)
            @test_throws ArgumentError TargetManifold(t)
        end
        msg = sprint(showerror, try minimal_manifold(tK) catch e; e end)
        @test occursin("gauge", msg) && occursin("EuclideanMap", msg) && occursin("Scale", msg) && occursin("known", msg)
        @test tangent_size(minimal_manifold(tS; known = (:length,))) == 3      # a ruler: the scale observable
        @test_throws ArgumentError minimal_manifold(tK; known = (:length,))    # the rigid member still needs the pose
        @test tangent_size(minimal_manifold(tK; known = (:pose,))) == 6        # a known pose: the whole affine
        @test tangent_size(minimal_manifold(tE; known = (:pose,))) == 3
        @test_throws ArgumentError minimal_manifold(tK; known = (:ruler,))
        @test tangent_size(restrict(minimal_manifold(tK; known = (:pose,)), (:skew_u12,))) == 1
    end

    @testset "ThinPlateSpline: 2K − 6 directions, all pure bending" begin
        t = PlanarTarget(pts; calibration = T)
        ch = minimal_manifold(t)
        @test tangent_size(ch) == 32 - 6
        B = retract_jacobian(ch)
        @test B' * B ≈ I && norm(Aff * B[1:16, :]) < 1.0e-12 && norm(Aff * B[17:32, :]) < 1.0e-12
        v = 1.0e-4 .* randn(rng, 26)
        @test local_coordinates(ch, retract(ch, v)) ≈ v
        @test plane_part(parameter_vector(retract(ch, v))[1:16]) < 1.0e-12
    end

    @testset "HeightField: K − 3 directions, all pure bending" begin
        t = PlanarTarget(pts; calibration = H)
        ch = minimal_manifold(t)
        @test tangent_size(ch) == 13
        B = retract_jacobian(ch)
        @test norm(Aff * B) < 1.0e-12 && B' * B ≈ I
        v = SVector{13}(1.0e-4 .* randn(rng, 13))
        δh = parameter_vector(retract(ch, v)) - parameter_vector(H)
        @test plane_part(δh) < 1.0e-12 && plane_part(parameter_vector(retract(ch, v))) < 1.0e-12
        @test local_coordinates(ch, retract(ch, v)) ≈ v
        @test size(action_jacobian(ch, pts[1])) == (3, 13)
        # the basis is reproducible: built twice, bit for bit
        @test retract_jacobian(minimal_manifold(PlanarTarget(pts; calibration = HeightField(nodes, h)))) == B
        # the public rule: the stored heights are pure bending, a plane added to them is not;
        # a field built on the system is the field built on its nodes
        @test VgtkCore.is_pure_bending(H.system, parameter_vector(H))
        @test !VgtkCore.is_pure_bending(H.system, parameter_vector(H) .+ 0.01 .* first.(H.system.nodes))
        @test HeightField(H.system, parameter_vector(H); project = false) == HeightField(nodes, parameter_vector(H); project = false)
    end

    @testset "a composition: its members' blocks" begin
        C = H ∘ T ∘ A
        ch = minimal_manifold(PlanarTarget(pts; calibration = C))
        @test tangent_size(ch) == 13 + 26 + 2
        B = retract_jacobian(ch)
        # orthonormal, pure bending in every spline's block, the affine factors' axes wholly free
        @test B' * B ≈ I && norm(Aff * B[1:16, :]) < 1.0e-12 && norm(Aff * B[17:32, :]) < 1.0e-12 &&
            norm(Aff * B[33:48, :]) < 1.0e-12 && B[49:50, :] * B[49:50, :]' ≈ I
        @test tangent_size(minimal_manifold(PlanarTarget(pts; calibration = H ∘ A))) == 16 + 2 - 3
    end

    @testset "restrict: the named parameters' directions that keep the constraint" begin
        tH = PlanarTarget(pts; calibration = H)
        # m named heights keep m − rank(the affine rows on them) directions
        for idx in ([1, 2, 5], [1, 2, 3], [1, 2, 3, 4, 5], [6, 7, 10, 11])
            r = restrict(minimal_manifold(tH), Tuple(Symbol(:h, i) for i in idx))
            @test tangent_size(r) == length(idx) - rank(Aff[:, idx])
            B = retract_jacobian(r)
            @test norm(Aff * B) < 1.0e-12 && norm(B[setdiff(1:16, idx), :]) == 0
        end
        tA = PlanarTarget(pts; calibration = A)
        @test tangent_size(restrict(minimal_manifold(tA), (:skew_u12,))) == 1
        @test_throws ArgumentError restrict(minimal_manifold(tA), (:h1,))
        @test tangent_size(minimal_manifold(PlanarTarget(pts))) == 0           # Rigid: zero-dimensional
    end

    @testset "the constraint's rank is scale-free: a physical spline on centimetre nodes" begin
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
        A5 = ThinPlateSplineSystem(grid_nodes(5; lo = -0.1, hi = 0.1)).coeff[26:28, :]
        @test norm(A5 * B[1:25, :]) < 1.0e-12 && norm(A5 * B[26:50, :]) < 1.0e-12
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
    # `ambient`: every stored parameter free (B = I), as the 0.4 kinds were
    # before their gauges were fixed — what the constraint removes
    function nullity_default(c; ambient = false, known = ())
        θ0 = Vector(parameter_vector(c)); n = length(θ0)
        J = ForwardDiff.jacobian(z -> projections(c, z[1:n], z[(n + 1):end]), [θ0; fill(1.0e-12, 6F)])
        B = ambient ? Matrix{Float64}(I, n, n) : Matrix(retract_jacobian(VgtkCore.TargetManifold(PlanarTarget(design; calibration = c); known)))
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
        "HeightField ∘ ThinPlateSpline ∘ Aspect ∘ Skew" => H ∘ T ∘ A,
    )
    for (name, c) in cases
        @testset "$name" begin
            @test nullity_default(c) == 0
        end
    end
    # The drop: with every stored parameter free, the canonical composition at
    # the design (a flat height field, a spline at zero displacement) is
    # rank-deficient by exactly the old gauge count — the spline's six affine
    # directions (four the pose's, two the aspect and skew again) and the height
    # field's three plane directions — and on its manifold by none. (Away from
    # the design the plane directions are near-gauges, observed only through
    # the bending: a height field of 1 mm leaves its piston alone exactly
    # unobservable, the tilts at singular values ~1e-7 of the largest.)
    T0 = ThinPlateSpline(nodes)
    @test nullity_default(H0 ∘ T0 ∘ A; ambient = true) == 6 + 3 && nullity_default(H0 ∘ T0 ∘ A) == 0
    @test nullity_default(H0 ∘ T; ambient = true) == 4 + 3          # (no aspect and skew: the spline's stretch and shear are observed)
    @test nullity_default(H0; ambient = true) == 3 && nullity_default(T; ambient = true) == 4
    @test nullity_default(H; ambient = true) == 1                    # the piston: a translation along the normal, at any height
    # the harness can see a gauge: the pose's directions, put back into the
    # composition, come back as exactly their count — the scale (1) and the rigid
    # member (3) — which is why the joint-pose manifold refuses them unless a
    # known length or pose is stated (the harness here estimates the poses, so
    # stating one is false and the gauges show)
    @test nullity_default(Scale(0.01) ∘ A; known = (:length,)) == 1
    @test nullity_default(E ∘ Scale(0.01) ∘ A; known = (:pose,)) == 4
    @test_throws ArgumentError nullity_default(Scale(0.01) ∘ A)
    @test_throws ArgumentError nullity_default(E ∘ Scale(0.01) ∘ A)
end

@testitem "target calibration: names, hashes, AD on lengths, an uncertain calibration" begin
    using Test, StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, mm, m, Attributed, Uncertain, CovMat, transform_deriv_params
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
        # no kind has gauges (0.5.0): the two gauge accessors are gone (their names retired, so spelled by parts)
        @test !any(s -> isdefined(VgtkCore, Symbol("gauge_" * s)), ("names", "directions"))
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
        # (the field built physical: pure bending in metres, its stored form — a unit move of a
        # millimetre-number field would carry the piston the scale adds, and a step projects it)
        Hm = HeightField([p .* mm for p in nodes], parameter_vector(H) .* mm)
        for c in (VgtkCore.with_unit(E ∘ A, mm), Hm, Hm ∘ A, Hm ∘ (VgtkCore.with_unit(E, mm) ∘ (Scale(0.01) ∘ A)))
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
    using VisualGeometryToolkitCore: grid_nodes, m, mm, with_unit, surface, Uncertain, CovMat
    using FlexUnits: ustrip
    using LinearAlgebra: I
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
    # an uncertain calibration's covariance is not part of the surface: it is
    # dropped before the strip, not rescaled (25 × 25) on every call
    Hu = Uncertain(H, CovMat(SMatrix{25, 25}(1.0e-8 * I)))
    ptu = PosedTarget(PlanarTarget([p .* m for p in pts]; calibration = Hu), with_unit(X, m))
    @test surface(ptu) == surface(pt)
    @test (@allocated surface(ptu)) < 16_384
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

@testitem "target calibration: a chart placement — a field on the board's dimensionless chart" begin
    # RadiometricCamera's board warping: a thin-plate height field over the
    # page's chart [0, 1]² (rows down: the design's y reflected), the page a
    # square of side s with its top-left corner at (x0, ytop) in the design
    # chart. On the chart's nodes its heights are pure bending, kept bit for
    # bit; on the page's design nodes (x0 + s n₁, ytop − s n₂) the same values
    # carry a piston −(ln s / s²) Σ wᵢ|mᵢ|² that the pure-bending projection
    # would remove. With a ChartPlacement the field stays on the chart and the
    # surface is the warped page, piston included.
    using Test, StaticArrays, LinearAlgebra, Random, ForwardDiff
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, ThinPlateSplineSystem, tps_coefficients, tps_evaluate, placement, surface,
        surface_point, intersection, minimal_manifold, retract_jacobian, retract, action_jacobian, tangent_size, with_unit, has_units
    using FlexUnits: DimensionError
    using Rotations: RotMatrix, RotXYZ

    rng = Xoshiro(11)
    mt, mm = VgtkCore.m, VgtkCore.mm
    rc = grid_nodes(5; lo = 0.0, hi = 1.0)                          # RadiometricCamera's HEIGHT_NODES
    src = ThinPlateSplineSystem(rc)
    hh = 1.0e-3 .* randn(rng, 25)                                  # its nodal heights
    w = SVector{25}(src.coeff[1:25, :] * hh)                       # board_height's weights (the affine part dropped)
    z = [tps_evaluate(src.nodes, w, zero(SVector{3, Float64}), n) for n in rc]   # board_height at the nodes: world units
    s, x0, ytop = 0.297, -0.05, 0.31
    world = [SVector(x0 + n[1] * s, ytop - n[2] * s) for n in rc]  # the page's design nodes (RadiometricCamera's posed_target)
    wsys = ThinPlateSplineSystem(world)
    ww, wa = tps_coefficients(wsys, z)                              # the warped page: the full interpolant, its piston kept
    truth(p) = tps_evaluate(wsys.nodes, ww, wa, p)
    # a real piston (what projecting on design nodes drops), −(ln s / s²) Σ wᵢ|mᵢ|² of the board_relief, and no tilt
    @test abs(wa[1]) > 1.0e-8 && norm(wa[2:3]) < 1.0e-12 * abs(wa[1]) + 1.0e-15
    @test wa[1] ≈ -(log(s) / s^2) * sum(ww[i] * sum(abs2, world[i] - SVector(x0, ytop)) for i in 1:25) * s^2 rtol = 1.0e-8

    chart = [SVector(n[1], -n[2]) for n in rc]                     # the chart's nodes, reflected (rows down)
    zc = z ./ s                                                     # heights in chart units
    hf = HeightField(chart, zc)
    P = ChartPlacement(SVector(x0, ytop), s)
    bd = Board(Size2(width = s, height = s), SVector(x0, ytop - s))
    design = [SVector(x0 + s * rand(rng), ytop - s * rand(rng)) for _ in 1:40]
    t = PlanarTarget(design; calibration = hf, board = bd, placement = P)
    X = EuclideanMap(RotMatrix{3}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.02, -0.01, 0.8))

    @testset "(a) RadiometricCamera's heights, bit for bit; its warped page to 1e-12" begin
        @test parameter_vector(hf) == zc                            # pure bending on the chart: not projected
        @test placement(t) == P && calibration(t) === hf
        pts = points(t)
        @test pts isa Vector{SVector{3, Float64}}
        @test all(q[SOneTo(2)] == p for (q, p) in zip(pts, design))   # the plane point exactly the design point
        @test maximum(abs(q[3] - truth(p)) for (q, p) in zip(pts, design)) <= 1.0e-12
        sf = surface(PosedTarget(t, X))
        @test sf isa HeightSurface
        @test maximum(norm(surface_point(sf, p) - X(SVector(p[1], p[2], truth(p)))) for p in design) <= 1.0e-12
        # a ray down the normal meets the warped page where the truth is
        for p in design[1:10]
            x = X(SVector(p[1], p[2], truth(p)))
            n = SMatrix{3, 3}(VgtkCore.rotation(X))[:, 3]
            @test norm(intersection(sf, x + 0.2 * n, -n)[2] - x) <= 1.0e-12
        end
        # without the placement, the same values on the design nodes lose the piston
        old = surface(PosedTarget(PlanarTarget(:page; calibration = HeightField(world, z), board = bd), X))
        @test abs(norm(surface_point(old, design[1]) - X(SVector(design[1][1], design[1][2], truth(design[1])))) - abs(wa[1])) < 1.0e-12
        # a height field alone leaves the quad as it is: the placed and the rigid target's quads agree bit for bit
        @test sf.quad == surface(PosedTarget(PlanarTarget(design; board = bd), X))
        # s = 1 and o = 0: the surface is the field's own
        t1 = PlanarTarget([SVector(0.5, -0.5)]; calibration = hf, placement = ChartPlacement(SVector(0.0, 0.0), 1.0))
        s1 = surface(PosedTarget(t1, X))
        @test s1.w == hf.w && s1.a == hf.a && s1.nodes == SVector{25}(chart)
    end

    @testset "(b) the reflected page: reflected chart nodes are pure bending, and the same page" begin
        sysc = ThinPlateSplineSystem(chart)
        Mc = [ones(25) [p[1] for p in chart] [p[2] for p in chart]]
        # No affine term on the reflected nodes, to rounding: coeff is inv([K P; Pᵀ 0]), LAPACK's
        # LU and inversion, whose last bits follow OpenBLAS's core type, so the plane part of zc
        # comes out at the order of κ·eps, not of eps. Measured (Julia 1.12.7), as a multiple
        # of eps·‖zc‖: 42.6 native (i9-13900K), Haswell and Zen; 44.3 Sandybridge; 51.6 Nehalem;
        # 129 Prescott; about 48 on GitHub's runner, past the earlier fixed 1e-14 (45 eps).
        # κ ≈ 365 here, so the bound 4κ·eps holds them all with a margin of 11.
        Kc = [[VgtkCore.tps_radial_basis(p, q) for p in chart, q in chart] Mc; Mc' zeros(3, 3)]
        @test norm(Mc * (sysc.coeff[26:28, :] * zc)) <= 4 * cond(Kc) * eps() * norm(zc)
        # the same chart translated into [0, 1]² (origin at the page's lower-left corner): the same points
        shifted = [SVector(n[1], 1 - n[2]) for n in rc]
        t2 = PlanarTarget(design; calibration = HeightField(shifted, zc), placement = ChartPlacement(SVector(x0, ytop - s), s))
        @test parameter_vector(calibration(t2)) == zc
        @test maximum(norm(a - b) for (a, b) in zip(points(t2), points(t))) <= 1.0e-12
    end

    @testset "(c) the placement adds no parameters: the pose + calibration Jacobian has full rank" begin
        ch = minimal_manifold(t)
        @test tangent_size(ch) == 25 - 3 && parameter_names(t) == parameter_names(hf)
        B = Matrix(retract_jacobian(ch))
        # action_jacobian is ∂points/∂θ · B, the placement's side in the height row
        J = ForwardDiff.jacobian(θ -> points(with_parameters(t, θ))[3], Vector(parameter_vector(t)))
        @test action_jacobian(ch, design[3]) ≈ J * B rtol = 1.0e-10
        # four poses about 0.8 m, a pinhole; poses (rotation vector, translation) and the manifold's coordinates
        F = 4
        R0 = [RotMatrix{3}(RotXYZ(0.3randn(rng), 0.3randn(rng), 0.3randn(rng))) for _ in 1:F]
        t0 = [SVector(0.05randn(rng), 0.05randn(rng), 0.8 + 0.1rand(rng)) for _ in 1:F]
        sk(ω) = SMatrix{3, 3}(0, ω[3], -ω[2], -ω[3], 0, ω[1], ω[2], -ω[1], 0)
        function proj(v, pose)
            tv = retract(ch, v)
            out = promote_type(eltype(v), eltype(pose))[]
            for f in 1:F
                ω = SVector{3}(pose[(6f - 5):(6f - 3)])
                R = (I + sk(ω) + sk(ω) * sk(ω) / 2) * SMatrix{3, 3}(R0[f])
                tt = t0[f] + SVector{3}(pose[(6f - 2):(6f)])
                for q in points(tv)
                    Y = R * q + tt
                    push!(out, Y[1] / Y[3], Y[2] / Y[3])
                end
            end
            return out
        end
        Jj = ForwardDiff.jacobian(x -> proj(x[1:22], x[23:end]), zeros(22 + 6F))
        sv = svdvals(Jj)
        @test count(<(1.0e-9 * sv[1]), sv) == 0
    end

    @testset "(d) units: the placement carries the lengths, the calibration is unit-free" begin
        Pm = ChartPlacement(SVector(x0, ytop) .* mt, s * mt)
        tm = PlanarTarget([p .* mt for p in design]; calibration = hf, placement = Pm, board = with_unit(bd, mt))
        @test has_units(tm) && has_units(Pm) && calibration(tm) === hf
        @test all(a == b .* mt for (a, b) in zip(points(tm), points(t)))
        @test ustrip(mt, tm) == t && with_unit(t, mt) == tm
        # the physical target's surface (in metres) is the plain one's
        Xm = EuclideanMap(VgtkCore.rotation(X), VgtkCore.translation(X) .* mt)
        @test surface(PosedTarget(tm, Xm)) == surface(PosedTarget(t, X))
        @test calibration(ustrip(mm, tm)) === hf && calibration(with_unit(t, mm)) === hf     # the calibration never moves
        tmm = with_unit(t, mm)                                       # numbers read as millimetres
        @test placement(tmm) == ChartPlacement(SVector(x0, ytop) .* mm, s * mm)
        @test maximum(norm(VgtkCore.strip_metres(a) - b ./ 1000) for (a, b) in zip(points(tmm), points(t))) <= 1.0e-15
        @test maximum(norm(a - b) for (a, b) in zip(points(ustrip(mm, tmm)), points(t))) <= 1.0e-15
        Pmm = ustrip(mm, Pm)
        @test !has_units(Pmm) && Pmm.side ≈ 1000s && Pmm.origin ≈ 1000 .* SVector(x0, ytop)
        @test with_unit(Pmm, mm).side ≈ Pm.side && with_unit(Pmm, mm).origin ≈ Pm.origin
        # one form: a physical placement on plain points, and a calibration with units, are refused
        @test_throws DimensionError PlanarTarget(design; calibration = hf, placement = Pm)
        @test_throws ArgumentError PlanarTarget([p .* mt for p in design]; calibration = with_unit(hf, mt), placement = Pm)
        @test_throws ArgumentError ChartPlacement(SVector(x0, ytop), s * mt)
        @test_throws ArgumentError ChartPlacement(SVector(x0, ytop), -s)
        @test_throws DimensionError ChartPlacement(SVector(x0, ytop) .* VgtkCore.px, s * VgtkCore.px)
        # an unresolved placed target takes its form from the placement
        @test has_units(PlanarTarget(:page; calibration = hf, placement = Pm)) && !has_units(PlanarTarget(:page; calibration = hf, placement = P))
        # equality and hashing see the placement
        @test t != PlanarTarget(design; calibration = hf, board = bd) && hash(t) == hash(PlanarTarget(design; calibration = hf, board = bd, placement = P))
        @test VgtkCore.resolve(PlanarTarget(:page; calibration = hf, placement = P), design) == PlanarTarget(:page; nominal = design, calibration = hf, placement = P)
    end
end

@testitem "target calibration: a physical composition's action applies each member in its own form" begin
    # ThinPlateSpline ∘ EuclideanMap ∘ Aspect ∘ Skew on lengths: the rigid member
    # translates in metres, so the composed action must not strip the design
    # point before the inner members act on it. Lengths are stored in metres,
    # so the physical target's action is the plain target's number for number.
    using Test, StaticArrays, Random
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: grid_nodes, action_jacobian, retract_jacobian, transform_deriv_params, TargetManifold, m
    using Rotations: Angle2d
    rng = Xoshiro(3)
    nodes = grid_nodes(3; lo = -0.1, hi = 0.1)
    fx, fy = 1.0e-4 .* randn(rng, 9), 1.0e-4 .* randn(rng, 9)
    ϑ, t = 0.02, SVector(0.003, -0.002)
    cp = ThinPlateSpline(nodes, fx, fy) ∘ EuclideanMap(Angle2d(ϑ), t) ∘ Aspect(0.002) ∘ Skew(-0.0015)
    cm = ThinPlateSpline([n .* m for n in nodes], fx .* m, fy .* m) ∘ EuclideanMap(Angle2d(ϑ), t .* m) ∘ Aspect(0.002) ∘ Skew(-0.0015)
    design = [SVector(x, y) for x in range(-0.1, 0.1; length = 4) for y in range(-0.08, 0.08; length = 3)]
    chp = TargetManifold(PlanarTarget(design; calibration = cp); known = (:pose,))
    chm = TargetManifold(PlanarTarget([p .* m for p in design]; calibration = cm); known = (:pose,))
    @test retract_jacobian(chm) == retract_jacobian(chp)
    for p in design
        Jp = action_jacobian(chp, p)
        @test Jp ≈ transform_deriv_params(cp, p) * retract_jacobian(chp) atol = 1.0e-14
        @test action_jacobian(chm, p .* m) ≈ Jp atol = 1.0e-15
    end
end
