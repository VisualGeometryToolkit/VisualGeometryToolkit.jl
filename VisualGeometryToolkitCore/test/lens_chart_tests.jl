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
        @test lo == SVector(-Inf, 0.0) && hi == SVector(Inf, 1.0)
        @test (lo[2], hi[2]) == (0.0, 1.0)                           # α's bound
        lo, hi = parameter_bounds(Lens(Unified(0.7), Additive(Radial((-0.1,)), Tangential(1.0e-3, 0.0))))
        @test all(==(-Inf), lo) && all(==(Inf), hi) && length(lo) == 4
        lo32, hi32 = parameter_bounds(Lens(DoubleSphere(0.1f0, 0.5f0)))
        @test eltype(lo32) === Float32 && hi32[2] === 1.0f0
        @test parameter_bounds(Pinhole()) == (SVector{0, Float64}(), SVector{0, Float64}())
        t = PlanarTarget([SVector(0.0, 0.0)])
        @test parameter_bounds(t) == (SVector{0, Float64}(), SVector{0, Float64}())
        K = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 1.5, 1000.0, 0, 320.0, 240.0, 1)))
        lo, hi = parameter_bounds(K)
        @test all(==(-Inf), lo) && all(==(Inf), hi) && length(lo) == 5
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

    @testset "LensManifold reads parameter_bounds, unchanged values" begin
        ds = Lens(DoubleSphere(0.1, 0.5)); ch = LensManifold(ds); v = SVector(0.01, 0.2)
        @test parameter_vector(retract(ch, v)) == SVector(0.11, 0.549833997312478)
        @test retract_jacobian(ch) == [1.0 0.0; 0.0 0.25]
        @test retract_jacobian(ch, v) == [1.0 0.0; 0.0 0.24751657271185995]
        @test local_coordinates(ch, retract(ch, v)) == SVector(0.009999999999999995, 0.20000000000000018)
        @test parameter_vector(model(LensManifold(Lens(DoubleSphere(0.0, 0.0))))) == SVector(0.0, 1.0e-6)   # a boundary value anchors inside
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
    VgtkCore.parameter_bounds(::_OddBounds) = (SVector(0.0, -Inf), SVector(2.0, Inf))

    @testset "LensManifold refuses a bound it cannot honor" begin
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
