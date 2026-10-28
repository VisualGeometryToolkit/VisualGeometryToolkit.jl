# The lens parameter chart (parameter_vector / with_parameters / names), the
# parameter Jacobian transform_deriv_params, and the lens record
# (lens_record / lens_from_record) — all derived from the leaves.
@testitem "lens_chart" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: transform_deriv_params, toml_value, distort_normalized, Even, Full

    function _numerical_param_jacobian(lens, x)
        θ = parameter_vector(lens)
        isempty(θ) && return zeros(2, 0)
        return FiniteDiff.finite_difference_jacobian(θ′ -> distort_normalized(with_parameters(lens, θ′), x), θ, Val(:central))
    end

    opencv = Lens(Perspective(), Additive(Radial((-0.2, 0.05, -0.01), (0.03, -0.004, 0.001)), Tangential(0.002, -0.001), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))
    lenses = Any[
        Pinhole(),
        Division(-0.1667, 0.0291),
        LiftLens(Radial((0.05, -0.01), (-0.2, 0.04, 0.003))),
        opencv,
        Lens(Perspective(), Additive(Radial((-0.2, 0.05)), Tangential(0.002, -0.001))),
        KannalaBrandt(-0.02, 0.003, -0.0004, 0.0001),
        KannalaBrandt(-0.02),
        Lens(Equidistant(), Additive(Radial((-0.02, 0.004, -0.0009, 0.0001)), Tangential(0.0006, -0.0004), ThinPrism((0.0002,), (-0.0001,)))),
        Lens(Equidistant(), Chain(Radial((0.1, -0.02, 0.003, 0.0, 0.0, 0.0001)), Additive(Tangential(0.0003, -0.0002), ThinPrism((0.001, 0.0), (-0.0005, 2.0e-5))))),
        Lens(DoubleSphere(-0.2, 0.6)),
        Lens(Unified(0.8)),
        Lens(ExtendedUnified(0.6, 1.1)),
        Lens(Unified(0.7), Additive(Radial((-0.1, 0.01)), Tangential(1.0e-3, -2.0e-3))),
        Lens(Perspective(), FieldOfView(0.9)),
        Lens(Perspective(), Chain(Additive(Radial((-0.1,)), Tangential(1.0e-3, 0.0)), Tilt(0.01, -0.02))),
        Scaramuzza(-0.1664, -0.0063, 0.035),
        Lens(Equidistant()), Lens(Stereographic()), Lens(Equisolid()), Lens(Orthographic()),
    ]

    @testset "lens parameter chart" begin
        for lens in lenses
            θ = parameter_vector(lens)
            @test θ isa SVector
            rt = with_parameters(lens, θ)
            @test typeof(rt) === typeof(lens) && parameter_vector(rt) == θ
            @test with_parameters(lens, collect(θ)) == lens                   # any vector
            @test length(parameter_names(lens)) == length(θ)
            @test allunique(parameter_names(lens))
            @test values(parameters(lens)) == Tuple(θ) && keys(parameters(lens)) == parameter_names(lens)
            if length(θ) > 0
                moved = with_parameters(lens, θ .+ 1.0e-3)
                @test distort_normalized(moved, SVector(0.3, -0.2)) != distort_normalized(lens, SVector(0.3, -0.2))
                @test_throws ArgumentError with_parameters(lens, θ[1:(end - 1)])
                l32 = with_parameters(lens, Float32.(θ))
                @test eltype(parameter_vector(l32)) === Float32
            else
                @test_throws ArgumentError with_parameters(lens, [1.0])
            end
        end
        # the generated names: per leaf, projection first, distortion depth-first
        @test parameter_names(Division(-0.1, 0.02)) == (:d1, :d2)
        @test parameter_names(LiftLens(Radial((0.05, -0.01), (-0.2, 0.04, 0.003)))) == (:k1, :k2, :d1, :d2, :d3)
        @test parameter_names(opencv) == (:k1, :k2, :k3, :d1, :d2, :d3, :p1, :p2, :sx1, :sx2, :sy1, :sy2)
        @test parameter_names(Lens(DoubleSphere(0.0, 0.5))) == (:xi, :alpha)
        @test parameter_names(Lens(Unified(0.7), Additive(Radial((-0.1,)), Tangential(0.0, 0.0)))) == (:xi, :k1, :p1, :p2)
        @test parameter_names(Scaramuzza(0.1, 0.0, 0.2)) == (:d1, :d2, :d3)
        @test parameter_names(Lens(Perspective(), FieldOfView(0.9))) == (:omega,)
        @test parameter_names(Pinhole()) == ()
        @test parameter_vector(Pinhole()) === SVector{0, Float64}()
        @test parameters(opencv).d3 == 0.001 && parameters(opencv).sy2 == -1.2e-3
        # names must be unique within a lens: two radial factors in one chain are refused
        @test_throws ArgumentError Lens(Perspective(), Chain(Radial((0.1,)), Radial((0.2,))))
        # structure is checked where it is built
        @test_throws ArgumentError Additive(Tangential(0.1, 0.0), Radial((0.1,)))          # a map only first
        @test_throws ArgumentError Chain(Radial((0.1,)), Tangential(0.0, 0.1))           # deltas only in an Additive
        @test_throws ArgumentError Lens(Perspective(), Radial{Full}((), (0.1,)))          # Full is a lift's basis
        @test_throws ArgumentError ThinPrism((0.1,), (0.1, 0.2))
        # the named models are the structures they name
        @test KannalaBrandt(-0.02, 0.003) isa KannalaBrandt{2, Float64}
        @test KannalaBrandt(-0.02, 0.003) === Lens(Equidistant(), Radial((-0.02, 0.003)))
        @test Division(-0.1) === LiftLens(Radial((), (-0.1,)))
        @test Scaramuzza(0.1, 0.2) isa Scaramuzza{2, Float64}
        # an alias with its number type free is still a subtype of its kind
        @test Division{2} <: LiftLens && Scaramuzza{3} <: LiftLens && KannalaBrandt{4} <: Lens{Equidistant}
        @test Pinhole() === Lens(Perspective())
        @test with_parameters(LiftLens(Radial((0.05, -0.01), (-0.2, 0.04, 0.003))), Float32.([1, 2, 3, 4, 5])) isa LiftLens{Radial{Even, 2, 3, Float32}}
    end

    @testset "the zero lens of a type" begin
        for lens in lenses
            z = zero(typeof(lens))
            @test typeof(z) === typeof(lens) && all(iszero, parameter_vector(z))
        end
        @test zero(KannalaBrandt{4}) === KannalaBrandt(0.0, 0.0, 0.0, 0.0)
        @test zero(Division{2}) === Division(0.0, 0.0)
        @test zero(Pinhole) === Pinhole()
        @test_throws ArgumentError zero(Division)
        @test_throws ArgumentError zero(KannalaBrandt)
    end

    @testset "transform_deriv_params vs central differences" begin
        for lens in lenses, x in (SVector(0.3, -0.2), SVector(-0.05, 0.4), SVector(0.7, 0.55))
            J = transform_deriv_params(lens, x)
            Jn = _numerical_param_jacobian(lens, x)
            @test J isa SMatrix{2}
            @test size(J) == size(Jn)
            isempty(J) && continue
            @test maximum(abs.(J .- Jn)) <= 1.0e-7 * max(1.0, maximum(abs.(Jn)))
        end
        # the axis (ρ = 0): finite, and the radial families' derivative vanishes there
        @test all(iszero, transform_deriv_params(Division(-0.2, 0.05), SVector(0.0, 0.0)))
        @test all(isfinite, transform_deriv_params(Lens(Perspective(), Additive(Radial((-0.2,)), Tangential(0.01, 0.0))), SVector(0.0, 0.0)))
        @test all(iszero, transform_deriv_params(KannalaBrandt(-0.02), SVector(0.0, 0.0)))
        @test all(iszero, transform_deriv_params(Scaramuzza(-0.2, 0.01), SVector(0.0, 0.0)))
        # Float32 points give Float32 Jacobians
        @test transform_deriv_params(Division(-0.2, 0.05), SVector{2, Float32}(0.3, 0.1)) isa SMatrix{2, 2, Float32}
    end

    @testset "lens record" begin
        # one canonical form per lens: keys are leaf tags, arities are array lengths
        @test lens_record(Pinhole()) == Dict("perspective" => Float64[])
        @test lens_record(KannalaBrandt(-0.043, 0.011)) == Dict("equidistant" => Float64[], "radial" => Any[[-0.043, 0.011], Float64[]])
        @test lens_record(Division(-0.31, 0.024)) == Dict("lift" => Dict("radial" => Any[Float64[], [-0.31, 0.024]]))
        @test lens_record(Scaramuzza(-0.17)) == Dict("lift" => Dict("radial_full" => Any[Float64[], [-0.17]]))
        @test lens_record(Lens(DoubleSphere(-0.2, 0.59))) == Dict("double_sphere" => [-0.2, 0.59])
        @test lens_record(Lens(Perspective(), Additive(Radial((-0.28,)), Tangential(1.0e-3, -5.0e-4)))) ==
            Dict("perspective" => Float64[], "add" => Any[Dict("radial" => Any[[-0.28], Float64[]]), Dict("tangential" => [1.0e-3, -5.0e-4])])
        @test toml_value(Division(-0.1)) == lens_record(Division(-0.1))
        # round trip: exact, the type determined by the record alone
        for lens in lenses
            back = lens_from_record(lens_record(lens))
            @test back === lens
            @test lens_from_record(Float32, lens_record(lens)) == with_parameters(lens, Float32.(parameter_vector(lens)))
        end
        # random structures and arities, random coefficients: bitwise, through a string
        rng = Xoshiro(7)
        rnd(n) = Tuple(randn(rng, n) .* 10.0 .^ rand(rng, -6:0, n))
        for _ in 1:50, P in 0:4, Q in 0:3
            rl = (
                LiftLens(Radial(rnd(P), rnd(Q))), LiftLens(Radial{Full}(rnd(P), rnd(Q))),
                Lens(Perspective(), Radial(rnd(P), rnd(Q))),
                Lens(Equidistant(), Additive(Radial(rnd(P), rnd(Q)), Tangential(rnd(2)...), ThinPrism(rnd(Q + 1), rnd(Q + 1)))),
                Lens(DoubleSphere(rnd(2)...), Chain(Radial(rnd(P), rnd(Q)), Tilt(rnd(2)...))),
            )
            for lens in rl
                s = repr(lens_record(lens))            # shortest round-trip printing of every Float64
                @test lens_from_record(eval(Meta.parse(s))) === lens
            end
        end
        # refusals name the offending key
        @test_throws ArgumentError lens_from_record(Dict("radial" => [[0.1], []]))                         # no projection
        @test_throws ArgumentError lens_from_record(Dict("perspective" => [], "equidistant" => []))        # two projections
        @test_throws ArgumentError lens_from_record(Dict("perspective" => [0.1]))                           # arity
        @test_throws ArgumentError lens_from_record(Dict("double_sphere" => [0.1]))
        @test_throws ArgumentError lens_from_record(Dict("perspective" => [], "radial" => [[0.1]]))        # not a pair
        @test_throws ArgumentError lens_from_record(Dict("perspective" => [], "tangential" => [0.1]))
        @test_throws ArgumentError lens_from_record(Dict("perspective" => [], "brown_conrady" => [0.1]))   # no alias tags
        @test_throws ArgumentError lens_from_record(Dict("perspective" => [], "radial" => [[0.1], []], "tangential" => [0.0, 0.0]))
        @test_throws ArgumentError lens_from_record(Dict("lift" => Dict("tangential" => [0.1, 0.0])))
        err = try
            lens_from_record(Dict("perspective" => [], "spline" => [1.0])); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("spline", err.msg)
    end
end

# The one parameter protocol: parameter_vector / parameter_names / parameter_bounds
# on lenses, targets, warps and LogicalIntrinsics. The golden values were
# recorded from the per-family getters and name functions this protocol replaced.
@testitem "parameter_protocol" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: LensManifold, retract, recenter, retract_jacobian, local_coordinates, tangent_size, model

    @testset "values are what the retired getters returned" begin
        opencv = Lens(Perspective(), Additive(Radial((-0.2, 0.05, -0.01), (0.03, -0.004, 0.001)), Tangential(0.002, -0.001), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))
        @test parameter_vector(opencv) == SVector(-0.2, 0.05, -0.01, 0.03, -0.004, 0.001, 0.002, -0.001, -5.5e-3, 7.0e-4, 2.2e-3, -1.2e-3)
        @test parameter_names(opencv) == (:k1, :k2, :k3, :d1, :d2, :d3, :p1, :p2, :sx1, :sx2, :sy1, :sy2)
        @test parameter_vector(Lens(DoubleSphere(-0.2, 0.6))) === SVector(-0.2, 0.6)
        @test eltype(parameter_vector(with_parameters(Division(-0.1, 0.02), Float32[-0.1, 0.02]))) === Float32
        @test parameter_vector(Pinhole()) === SVector{0, Float64}()
        t = PlanarTarget([SVector(0.0, 0.0), SVector(1.0, 0.0)])
        @test parameter_vector(t) === SVector{0, Float64}() && parameter_vector(Rigid()) === SVector{0, Float64}()
        @test parameter_names(t) === () && parameter_names(Rigid()) === ()
        @test parameters(t) == NamedTuple()
    end

    @testset "parameter_bounds" begin
        lo, hi = parameter_bounds(Lens(DoubleSphere(0.1, 0.5)))
        @test lo == SVector(-1.0, 0.0) && hi == SVector(Inf, 1.0)    # ξ > −1, α ∈ [0, 1]
        lo, hi = parameter_bounds(Lens(Unified(0.7), Additive(Radial((-0.1,)), Tangential(1.0e-3, 0.0))))
        @test lo == SVector(-1.0, -Inf, -Inf, -Inf) && all(==(Inf), hi)   # ξ > −1, the terms unbounded
        @test parameter_bounds(Lens(ExtendedUnified(0.5, 1.2))) == (SVector(0.0, 0.0), SVector(1.0, Inf))
        @test parameter_bounds(Lens(Perspective(), FieldOfView(0.9))) == (SVector(0.0), SVector(Float64(π)))
        @test parameter_bounds(KannalaBrandt(-0.02, 0.003)) == (SVector(-Inf, -Inf), SVector(Inf, Inf))
        lo32, hi32 = parameter_bounds(Lens(DoubleSphere(0.1f0, 0.5f0)))
        @test eltype(lo32) === Float32 && hi32[2] === 1.0f0
        @test parameter_bounds(Pinhole()) == (SVector{0, Float64}(), SVector{0, Float64}())
        t = PlanarTarget([SVector(0.0, 0.0)])
        @test parameter_bounds(t) == (SVector{0, Float64}(), SVector{0, Float64}())
        K = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 1.5, 1000.0, 0, 320.0, 240.0, 1)))
        lo, hi = parameter_bounds(K)
        @test lo == SVector(0.0, 0.0, -Inf, -Inf, -Inf) && all(==(Inf), hi)   # f > 0, aspect > 0
    end

    @testset "LogicalIntrinsics" begin
        K = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 1.5, 1000.0, 0, 320.0, 240.0, 1)))
        @test parameter_names(K) == (:f, :aspect, :skew, :cx, :cy)
        θ = parameter_vector(K)
        @test θ == SVector(800.0, 1.25, 1.5, 320.0, 240.0)
        @test with_parameters(K, θ) == K
        @test with_parameters(K, collect(θ)).K == K.K
        Kg = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 1.5, 804.0, 0, 320.0, 240.0, 1)))
        @test with_parameters(Kg, parameter_vector(Kg)).K ≈ Kg.K rtol = 1.0e-15   # f·aspect rounds in the last place
        @test with_parameters(K, θ).K_inv == K.K_inv
        @test_throws ArgumentError with_parameters(K, θ[1:4])
        K32 = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float32}(800, 0, 0, 1.5, 804, 0, 320, 240, 1)))
        @test eltype(parameter_vector(K32)) === Float32 && with_parameters(K32, parameter_vector(K32)) isa LogicalIntrinsics{Float32}
    end

    @testset "LensManifold reads parameter_bounds: the chart of each kind of bound" begin
        # ξ ∈ (−1, ∞): ξ = −1 + (ξ₀ + 1)·exp(v); α ∈ [0, 1], both ends models: the additive chart clamped (a box)
        ds = Lens(DoubleSphere(0.1, 0.5)); ch = LensManifold(ds); v = SVector(0.01, 0.2)
        @test parameter_vector(retract(ch, v)) == SVector(-1.0 + 1.1 * exp(0.01), 0.7)
        @test retract_jacobian(ch) == [1.1 0.0; 0.0 1.0]
        @test retract_jacobian(ch, v) ≈ [1.1 * exp(0.01) 0.0; 0.0 1.0] rtol = 1.0e-15
        @test retract_jacobian(ch, SVector(0.0, 0.6)) == [1.1 0.0; 0.0 0.0]                        # past α = 1: clamped, slope 0
        @test parameter_vector(retract(ch, SVector(0.0, 0.6)))[2] === 1.0
        @test local_coordinates(ch, retract(ch, v)) ≈ v rtol = 1.0e-13
        @test parameter_vector(model(LensManifold(Lens(DoubleSphere(0.0, 0.0))))) == SVector(0.0, 0.0)      # a closed end is a model: kept
        @test parameter_vector(model(LensManifold(Lens(DoubleSphere(-1.0, 0.5)))))[1] == -1.0 + 1.0e-6      # an open end anchors inside
        @test parameter_vector(model(LensManifold(Lens(DoubleSphere(0.0, 1.3))))) == SVector(0.0, 1.0)      # past a closed end: clamped onto it
        # ω ∈ [0, π): closed at 0 (the identity), open at π — ω = π − (π − ω₀)·exp(−v), clamped at 0
        fv = LensManifold(Lens(Perspective(), FieldOfView(0.9)))
        ω(d) = parameter_vector(retract(fv, SVector(d)))[1]
        @test ω(0.0) ≈ 0.9 rtol = 1.0e-15
        @test ω(0.3) ≈ π - (π - 0.9) * exp(-0.3) rtol = 1.0e-15
        @test ω(-50.0) === 0.0 && π - 1.0e-12 < ω(50.0) < π
        @test retract_jacobian(fv)[1, 1] ≈ π - 0.9 rtol = 1.0e-15
        @test retract_jacobian(fv, SVector(0.3))[1, 1] ≈ (ω(0.3 + 1.0e-6) - ω(0.3 - 1.0e-6)) / 2.0e-6 rtol = 1.0e-7
        @test retract_jacobian(fv, SVector(-50.0))[1, 1] == 0
        @test local_coordinates(fv, retract(fv, SVector(0.3)))[1] ≈ 0.3 rtol = 1.0e-12
        # ω = 0, the identity, is the anchor with slope π (not 10⁻⁶·π); ω and −ω are one map: −0.3 anchors at 0.3, the same lens
        f0 = LensManifold(Lens(Perspective(), FieldOfView(0.0)))
        @test parameter_vector(model(f0)) == SVector(0.0)
        @test retract_jacobian(f0)[1, 1] ≈ π rtol = 1.0e-15
        neg = Lens(Perspective(), FieldOfView(-0.3))
        @test parameter_vector(model(LensManifold(neg))) == SVector(0.3)
        X = SVector(0.4, -0.2, 1.0)
        @test project(model(LensManifold(neg)), X) ≈ project(neg, X) rtol = 1.0e-15
        @test local_coordinates(LensManifold(Lens(Perspective(), FieldOfView(0.3))), neg) == SVector(0.0)
        # β ≥ 0, closed at 0 (β = 0 a model): additive, never below 0; α ∈ [0, 1] (α = 0 the pinhole) anchored on it
        eu = LensManifold(Lens(ExtendedUnified(0.6, 1.2)))
        @test parameter_vector(retract(eu, SVector(0.0, -40.0)))[2] === 0.0
        @test parameter_vector(retract(eu, SVector(0.0, 0.5)))[2] == 1.2 + 0.5
        e0 = LensManifold(Lens(ExtendedUnified(0.0, 1.0)))
        @test parameter_vector(model(e0)) == SVector(0.0, 1.0) && retract_jacobian(e0) == [1.0 0.0; 0.0 1.0]
        kb = LensManifold(KannalaBrandt(-0.02, 0.003))
        @test parameter_vector(retract(kb, SVector(0.001, -0.002))) == SVector(-0.019, 0.001)
    end

    @testset "LensManifold stays inferred and allocation-free" begin
        ch = LensManifold(Lens(DoubleSphere(0.1, 0.5))); v = SVector(0.01, 0.2)
        @inferred retract(ch, v)
        @inferred recenter(ch, v)
        @inferred retract_jacobian(ch, v)
        @inferred local_coordinates(ch, retract(ch, v))
        retract(ch, v); recenter(ch, v)
        f1(ch, v) = @allocated retract(ch, v)
        f2(ch, v) = @allocated recenter(ch, v)
        f1(ch, v); f2(ch, v)
        @test f1(ch, v) == 0
        @test f2(ch, v) == 0
    end

    struct _OddBounds <: VgtkCore.AbstractLensModel end
    VgtkCore.parameter_names(::_OddBounds) = (:a, :b)
    VgtkCore.parameter_bounds(::_OddBounds) = (SVector(2.0, -Inf), SVector(0.0, Inf))

    @testset "LensManifold refuses a bound that is no interval" begin
        @test_throws ArgumentError LensManifold(_OddBounds())
        @test LensManifold(KannalaBrandt(-0.02)) isa LensManifold   # unbounded: accepted
    end
end

# convert_eltype: a lens, intrinsics, camera model or grid model rebuilt in
# another float type, and back.
@testitem "convert_eltype" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: convert_eltype, CentralGenericModel, NoncentralGenericModel, Division, CONTRACT_GENERICS

    @test convert_eltype in CONTRACT_GENERICS && Base.ispublic(VgtkCore, :convert_eltype) && !Base.isexported(VgtkCore, :convert_eltype)
    # within one Float32 rounding of the value (relative), back in Float64
    near(a, b) = all(abs.(a .- b) .<= eps(Float32) .* max.(abs.(b), 1))

    for lens in (KannalaBrandt(-0.04, 0.01), Lens(DoubleSphere(-0.2, 0.6)), Division(-0.1))
        l32 = convert_eltype(Float32, lens)
        @test eltype(parameter_vector(l32)) === Float32
        l64 = convert_eltype(Float64, l32)
        @test typeof(l64) === typeof(lens) && near(parameter_vector(l64), parameter_vector(lens))
    end

    intr = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(800.1, 0, 0, 1.5, 1000.3, 0, 320.7, 240.2, 1)))
    i32 = convert_eltype(Float32, intr)
    @test i32 isa LogicalIntrinsics{Float32} && eltype(i32.K_inv) === Float32
    i64 = convert_eltype(Float64, i32)
    @test i64 isa LogicalIntrinsics{Float64} && near(i64.K, intr.K) && near(i64.K_inv, intr.K_inv)

    cm = CameraModel(intr; lens = KannalaBrandt(-0.04, 0.01))
    c32 = convert_eltype(Float32, cm)
    @test c32 isa CameraModel{LogicalIntrinsics{Float32}} && eltype(parameter_vector(c32.lens)) === Float32
    u = SVector(400.5f0, 300.25f0)
    @test eltype(backproject(c32, u)) === Float32
    c64 = convert_eltype(Float64, c32)
    @test typeof(c64) === typeof(cm) && near(c64.intrinsics.K, cm.intrinsics.K)

    K = @SMatrix [300.0 0.0 320.0; 0.0 300.0 240.0; 0.0 0.0 1.0]
    src = CameraModel(CameraCalibrationMat(K); lens = Division(-0.1))
    g = CentralGenericModel(src, (640, 480); grid_size = (12, 10))
    g32 = convert_eltype(Float32, g)
    @test g32 isa CentralGenericModel{Float32}
    g64 = convert_eltype(Float64, g32)
    @test g64 isa CentralGenericModel{Float64} && all(near(a, b) for (a, b) in zip(g64.grid, g.grid))
    n = NoncentralGenericModel(src, (640, 480); grid_size = (12, 10))
    n32 = convert_eltype(Float32, n)
    @test n32 isa NoncentralGenericModel{Float32}
    @test all(near(a, b) for (a, b) in zip(convert_eltype(Float64, n32).directions, n.directions))

    # the kernels' private conversion forwards to it: a downstream lens's own
    # method is the one the batched and fused kernels apply
    @test VgtkCore._precision(Float32, KannalaBrandt(-0.04, 0.01)) === convert_eltype(Float32, KannalaBrandt(-0.04, 0.01))
    @test VgtkCore._precision(Float32, g).grid == g32.grid
    @test VgtkCore._precision(Float32, intr) === intr          # intrinsics are not converted inside a warp chain
    @eval struct _TaggedLens{T} <: VgtkCore.AbstractLensModel
        k::T
    end
    @eval VgtkCore.convert_eltype(::Type{T}, l::_TaggedLens) where {T <: AbstractFloat} = _TaggedLens(T(l.k + 1))
    @test VgtkCore._precision(Float32, _TaggedLens(1.0)) === _TaggedLens(2.0f0)
end

# The parameter protocol on a parametric camera (intrinsics, then lens) and the
# ParameterManifold on every lens type: round trips, the chart inside the
# bounds, and ForwardDiff through with_parameters against finite differences.
@testitem "parameter_protocol: CameraModel and every lens's manifold" begin
    using Test, StaticArrays, LinearAlgebra
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ParameterManifold, LensManifold, CONTRACT_GENERICS, retract, recenter, retract_jacobian,
        local_coordinates, tangent_size, model, restrict, action_jacobian, minimal_manifold, parameters

    @test all(f -> f in CONTRACT_GENERICS, (parameter_vector, parameter_names, parameter_bounds, with_parameters))
    opencv = Lens(Perspective(), Additive(Radial((-0.2, 0.05, -0.01), (0.03, -0.004, 0.001)), Tangential(0.002, -0.001), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))
    lenses = Any[
        Pinhole(), Division(-0.1667, 0.0291), LiftLens(Radial((0.05, -0.01), (-0.2, 0.04, 0.003))), opencv,
        KannalaBrandt(-0.02, 0.003, -0.0004, 0.0001),
        Lens(Equidistant(), Chain(Radial((0.1, -0.02, 0.003)), Additive(Tangential(0.0003, -0.0002), ThinPrism((0.001,), (-0.0005,))))),
        Lens(DoubleSphere(-0.2, 0.6)), Lens(Unified(0.8)), Lens(ExtendedUnified(0.6, 1.1)),
        Lens(Unified(0.7), Additive(Radial((-0.1, 0.01)), Tangential(1.0e-3, -2.0e-3))),
        Lens(Perspective(), FieldOfView(0.9)),
        Lens(Perspective(), Chain(Additive(Radial((-0.1,)), Tangential(1.0e-3, 0.0)), Tilt(0.01, -0.02))),
        Scaramuzza(-0.1664, -0.0063, 0.035), Lens(Stereographic()),
    ]
    K = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(400.0, 0, 0, 0.5, 395.0, 0, 320.0, 240.0, 1)))
    X = SVector(0.2, -0.1, 1.0)

    @testset "a CameraModel: the intrinsics' coordinates, then the lens's" begin
        for lens in lenses
            cam = CameraModel(K, lens)
            θ = parameter_vector(cam)
            @test θ isa SVector && θ == vcat(parameter_vector(K), parameter_vector(lens))
            @test parameter_names(cam) == (:f, :aspect, :skew, :cx, :cy, parameter_names(lens)...) && allunique(parameter_names(cam))
            lo, hi = parameter_bounds(cam)
            @test lo == vcat(parameter_bounds(K)[1], parameter_bounds(lens)[1]) && hi == vcat(parameter_bounds(K)[2], parameter_bounds(lens)[2])
            @test all(lo .<= θ .<= hi)
            rt = with_parameters(cam, θ)
            @test typeof(rt) === typeof(cam) && with_parameters(cam, collect(θ)) isa typeof(cam)
            @test parameter_vector(rt) ≈ θ rtol = 1.0e-15
            @test values(parameters(cam)) == Tuple(θ)
            @test_throws ArgumentError with_parameters(cam, θ[1:(end - 1)])
            # ForwardDiff through with_parameters and project, against central differences
            f(q) = project(with_parameters(cam, q), X)
            J = ForwardDiff.jacobian(f, θ)
            Jn = FiniteDiff.finite_difference_jacobian(q -> Vector(f(SVector{length(θ)}(q))), Vector(θ), Val(:central))
            @test J ≈ Jn rtol = 1.0e-6 atol = 1.0e-7
            # the manifold: minimal_manifold(camera), the chart's action Jacobian is the parameter Jacobian through it
            ch = minimal_manifold(cam)
            @test ch isa ParameterManifold && tangent_size(ch) == length(θ) && model(ch) == cam
            @test action_jacobian(ch, X) ≈ J * retract_jacobian(ch) rtol = 1.0e-12
            v = SVector{length(θ)}(ntuple(i -> 1.0e-3 * (-1)^i, length(θ)))
            @test local_coordinates(ch, retract(ch, v)) ≈ v rtol = 1.0e-8 atol = 1.0e-12
            @test parameter_vector(model(recenter(ch, v))) ≈ parameter_vector(retract(ch, v))
        end
        # skew (and aspect) fixed by name, through restrict
        cam = CameraModel(K, KannalaBrandt(-0.02, 0.003))
        r = restrict(minimal_manifold(cam), (:f, :cx, :cy, :k1, :k2))
        @test tangent_size(r) == 5
        moved = retract(r, SVector(0.1, 1.0, -1.0, 0.001, 0.0))
        @test parameters(moved).skew == 0.5 && parameters(moved).aspect ≈ 395 / 400 && parameters(moved).f ≈ 400 * exp(0.1)
        @test_throws ArgumentError restrict(minimal_manifold(cam), (:f, :nope))
        # the focal's chart keeps it positive
        @test parameters(retract(minimal_manifold(cam), SVector(-60.0, 0, 0, 0, 0, 0, 0))).f > 0
    end

    @testset "every lens's manifold stays inside its bounds and round-trips" begin
        for lens in lenses
            ch = LensManifold(lens)
            n = tangent_size(ch)
            n == 0 && continue
            lo, hi = parameter_bounds(lens)
            kinds = VisualGeometryToolkitCore.Cameras._bound_kinds(lens)
            for scale in (1.0e-3, 1.0, 30.0)
                v = SVector{n}(ntuple(i -> scale * (-1)^i, n))
                θ = parameter_vector(retract(ch, v))
                # never past a finite bound: on a closed end at most, strictly inside an open one
                @test all(map((t, a, b, k) -> (k[1] ? a <= t : a < t) && (k[2] ? t <= b : t < b), θ, lo, hi, kinds)) || any(isinf, θ)
            end
            v = SVector{n}(ntuple(i -> 0.01 * (-1)^i, n))
            @test local_coordinates(ch, retract(ch, v)) ≈ v rtol = 1.0e-8
            Jfd = ForwardDiff.jacobian(w -> parameter_vector(retract(ch, w)), v)
            @test retract_jacobian(ch, v) ≈ Jfd rtol = 1.0e-12
            @test retract_jacobian(ch) ≈ ForwardDiff.jacobian(w -> parameter_vector(retract(ch, w)), zero(v)) rtol = 1.0e-12
        end
    end
end

# A solver's search box on the parameter manifold (`ParameterManifold(x; box)`): keyed by the parameter names, within the domain,
# applied at the step (recenter), kept by restrict and by the manifold's NLLSsolver variable.
@testitem "ParameterManifold: a solver's box by name, applied at the step" begin
    using Test, StaticArrays
    import NLLSsolver
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ParameterManifold, parameter_box, retract, recenter, restrict, model, parameters, tangent_size

    K = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(1.2, 0, 0, 0, 1.8, 0, 0.01, -0.02, 1)))   # a chart camera
    cam = CameraModel(K, Lens(DoubleSphere(-0.2, 0.6)))
    names = parameter_names(cam)                                                     # f, aspect, skew, cx, cy, xi, alpha
    box = (; f = (0.1, 50.0), aspect = (0.1, 10.0), cx = (-0.3, 0.3), xi = (-0.95, 1.5), alpha = (-1.0e3, 1.0e3))
    ch = ParameterManifold(cam; box)

    @testset "the box by name, intersected with the domain; unnamed parameters keep their domain" begin
        lo, hi = parameter_box(ch)
        @test lo == SVector(0.1, 0.1, -Inf, -0.3, -Inf, -0.95, 0.0) && hi == SVector(50.0, 10.0, Inf, 0.3, Inf, 1.5, 1.0)
        @test parameter_box(ParameterManifold(cam)) === nothing
        @test model(ch) == model(ParameterManifold(cam))                                # the anchor as given
        @test_throws ArgumentError ParameterManifold(cam; box = (; nope = (0.0, 1.0)))
        @test_throws ArgumentError ParameterManifold(cam; box = (; alpha = (2.0, 3.0)))   # empty within α ∈ [0, 1]
    end

    @testset "a step inside the box is the unboxed step, bit for bit; one past it lands on the face" begin
        v = SVector(0.01, -0.02, 0.0, 0.003, 0.004, 0.01, -0.05)
        @test model(recenter(ch, v)) == model(recenter(ParameterManifold(cam), v))
        big = SVector(10.0, 0.0, 0.0, 1.0, 0.0, -50.0, 0.0)                          # f × e¹⁰, cx + 1, ξ toward −1
        p = parameters(model(recenter(ch, big)))
        @test p.f == 50.0 && p.cx == 0.3 && p.xi == -0.95 && p.cy == -0.02
        q = parameters(retract(ch, big))                                              # the chart alone: the domain only
        @test q.f > 50.0 && q.cx > 0.3 && -1.0 <= q.xi < -0.95
        @test parameter_box(recenter(ch, big)) == parameter_box(ch)                    # the box travels with the anchor
    end

    @testset "restrict keeps the box; the manifold as an NLLSsolver variable steps inside it" begin
        r = restrict(ch, (:f, :cx, :cy, :xi, :alpha))
        @test tangent_size(r) == 5
        u = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]                            # from index 4 (f, cx, cy, ξ, α): cx + 5
        n = NLLSsolver.update(r, u, 4)
        @test parameters(model(n)).cx == 0.3 && n == recenter(r, SVector(0.0, 5.0, 0.0, 0.0, 0.0))
        @test parameter_vector(model(NLLSsolver.update(r, zeros(5)))) ≈ parameter_vector(model(r)) rtol = 1.0e-15   # a zero step: the chart's round trip
    end
end

# A coordinate's scale (`ParameterManifold(x; scale)`): the coordinate v moves the chart by s·v. A pixel camera of a W × H image with
# the principal point (and the skew) scaled by the image size steps exactly like the camera of the normalized image [−½, ½]²: its
# action Jacobian is the normalized camera's, times (W, H) on the projected point.
@testitem "ParameterManifold: a coordinate's scale" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ParameterManifold, coordinate_scale, retract, recenter, restrict, model, parameters,
                                     retract_jacobian, action_jacobian, local_coordinates, parameter_vector
    W, H = 640, 480
    lens = KannalaBrandt(-0.03, 0.002, 0.0005, -0.0001)
    Kp = SMatrix{3, 3, Float64}(310.0, 0, 0, 0.4, 312.0, 0, 322.0, 236.0, 1)                  # pixels (a small skew)
    N = SMatrix{3, 3, Float64}(1 / W, 0, 0, 0, 1 / H, 0, -0.5, -0.5, 1)                         # pixels → the normalized image
    cam_px = CameraModel(LogicalIntrinsics(CameraCalibrationMat(Kp)), lens)
    cam_n = CameraModel(LogicalIntrinsics(CameraCalibrationMat(N * Kp)), lens)
    scale = (; skew = W, cx = W, cy = H)
    ch = ParameterManifold(cam_px; scale)
    @test coordinate_scale(ch) == SVector(1.0, 1.0, W, W, H, 1.0, 1.0, 1.0, 1.0) && coordinate_scale(ParameterManifold(cam_px)) === nothing
    @test_throws ArgumentError ParameterManifold(cam_px; scale = (; nope = 2.0))
    @test_throws ArgumentError ParameterManifold(cam_px; scale = (; cx = 0.0))
    # the coordinate v is the unscaled chart's coordinate s·v
    v = SVector(0.01, -0.02, 1.0e-4, 0.003, -0.002, 0.01, 0.0, -0.001, 0.0)
    @test model(recenter(ch, v)) == model(recenter(ParameterManifold(cam_px), coordinate_scale(ch) .* v))
    @test local_coordinates(ch, retract(ch, v)) ≈ v rtol = 1.0e-12
    @test coordinate_scale(recenter(ch, v)) == coordinate_scale(ch)                           # the scale travels with the anchor
    @test coordinate_scale(recenter(ParameterManifold(cam_px; box = (; f = (10.0, 1.0e4)), scale), v)) == coordinate_scale(ch)
    @test retract_jacobian(ch) ≈ retract_jacobian(ParameterManifold(cam_px)) * Diagonal(coordinate_scale(ch))
    @test retract_jacobian(ch, v) ≈ retract_jacobian(ParameterManifold(cam_px), coordinate_scale(ch) .* v) * Diagonal(coordinate_scale(ch))
    # the pixel camera with the image-size scale is the normalized camera's chart, column for column
    for X in (SVector(0.1, -0.2, 1.0), SVector(-0.8, 0.5, 0.6), SVector(1.5, 0.3, 0.4))
        Jp = action_jacobian(ch, X)
        Jn = action_jacobian(ParameterManifold(cam_n), X)
        @test Jp ≈ Diagonal(SVector(W, H)) * Jn rtol = 1.0e-10
    end
    # restrict keeps it
    r = restrict(ch, (:f, :cx, :cy, :k1))
    @test coordinate_scale(r.manifold) == coordinate_scale(ch)
    # Float32 scales keep a Float32 model's retraction in Float32 (a GPU kernel's camera)
    cam32 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float32}(Kp))), KannalaBrandt(-0.03f0, 0.002f0, 0.0005f0, -0.0001f0))
    ch32 = ParameterManifold(cam32; scale = (; skew = Float32(W), cx = Float32(W), cy = Float32(H)))
    @test eltype(coordinate_scale(ch32)) === Float32 && eltype(coordinate_scale(ch)) === Float64
    @test eltype(parameter_vector(retract(ch32, Float32.(v)))) === Float32
end
