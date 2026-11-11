# LensSpec: a lens family by name and term counts (cameras/lens_spec.jl).
@testitem "lens_spec" begin
    using Test, TOML
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: toml_value, free_names

    # spec, expected parameter names
    cases = Any[
        (LensSpec(:division; coefficients = 3), (:d1, :d2, :d3)),
        (LensSpec(:kannala_brandt; coefficients = 4), (:k1, :k2, :k3, :k4)),
        (LensSpec(:brown_conrady; radial = 3, tangential = true), (:k1, :k2, :k3, :p1, :p2)),
        (LensSpec(:brown_conrady; radial = 2), (:k1, :k2)),
        (LensSpec(:rational; numerator = 3, denominator = 3), (:k1, :k2, :k3, :d1, :d2, :d3)),
        (LensSpec(:perspective), ()),
        (LensSpec(:equidistant), ()),
        (LensSpec(:stereographic), ()),
        (LensSpec(:equisolid), ()),
        (LensSpec(:orthographic), ()),
        (LensSpec(:double_sphere), (:xi, :alpha)),
        (LensSpec(:unified), (:xi,)),
        (LensSpec(:extended_unified), (:alpha, :beta)),
    ]

    @testset "lens(spec): names and reference values" begin
        for (spec, names) in cases
            L = lens(spec)
            @test L isa AbstractLensModel
            @test parameter_names(L) == names
            spec.family === :extended_unified || @test all(iszero, parameter_vector(L))
            @test free_names(spec) == names
        end
        @test lens(LensSpec(:division; coefficients = 3)) == Division(0.0, 0.0, 0.0)
        @test lens(LensSpec(:kannala_brandt; coefficients = 2)) == KannalaBrandt(0.0, 0.0)
        @test lens(LensSpec(:brown_conrady; radial = 2, tangential = true)) ==
            Lens(Perspective(), Additive(Radial((0.0, 0.0)), Tangential(0.0, 0.0)))
        @test lens(LensSpec(:brown_conrady; radial = 2)) == Lens(Perspective(), Radial((0.0, 0.0)))
        @test lens(LensSpec(:rational; numerator = 1, denominator = 2)) == Lens(Perspective(), Radial((0.0,), (0.0, 0.0)))
        @test lens(LensSpec(:equidistant)) == Lens(Equidistant())
    end

    @testset "parametric projections: the member equal to perspective" begin
        using StaticArrays, Random
        rng = MersenneTwister(7)
        rays = [SVector(0.8 * randn(rng), 0.8 * randn(rng), 1.0) for _ in 1:20]
        P = Lens(Perspective())
        for f in (:double_sphere, :unified, :extended_unified)
            L = lens(LensSpec(f))
            @test maximum(maximum(abs, VisualGeometryToolkitCore.Cameras.project(L, X) - VisualGeometryToolkitCore.Cameras.project(P, X)) for X in rays) <= 1.0e-12
        end
        @test lens(LensSpec(:extended_unified)) == Lens(ExtendedUnified(0.0, 1.0))
        # a non-reference value keeps its number: the record form
        @test LensSpec(Lens(DoubleSphere(-0.2, 0.6))).family === :record
        @test lens(LensSpec(Lens(DoubleSphere(-0.2, 0.6)))) == Lens(DoubleSphere(-0.2, 0.6))
    end

    @testset "the named form is eltype-agnostic and keeps the sign of a zero" begin
        # a Float32 lens at the reference values is the named form, and lens(spec) is Float64 either way
        s32 = LensSpec(Division(0.0f0, 0.0f0))
        @test s32 == LensSpec(:division; coefficients = 2)
        @test eltype(parameter_vector(lens(s32))) === Float64
        @test LensSpec(KannalaBrandt(0.0f0, 0.0f0, 0.0f0)) == LensSpec(:kannala_brandt; coefficients = 3)
        # a -0.0 coefficient is a value: not the reference, so the record form keeps its sign
        neg = LensSpec(Division(-0.0, 0.0))
        @test neg.family === :record
        @test signbit(parameter_vector(lens(neg))[1])
        @test neg != LensSpec(:division; coefficients = 2)
    end

    @testset "the inner constructor validates" begin
        @test_throws ArgumentError LensSpec(:division, (; coefficients = 0), nothing, nothing)
        @test_throws ArgumentError LensSpec(:division, (;), nothing, nothing)
        @test_throws ArgumentError LensSpec(:division, (; coefficients = 2), (:d9,), nothing)
        @test_throws ArgumentError LensSpec(:fish, (;), nothing, nothing)
        @test_throws ArgumentError LensSpec(:record, (; coefficients = 2), nothing, Dict{String, Any}())
        @test LensSpec(:division, (; coefficients = 2), [:d2], nothing) == LensSpec(:division; coefficients = 2, fixed = (:d2,))
    end

    @testset "round trips" begin
        for (spec, _) in cases
            @test LensSpec(lens(spec)) == spec
            @test LensSpec(toml_value(spec)) == spec
            @test LensSpec(TOML.parse(sprint(TOML.print, toml_value(spec)))) == spec
        end
        @test toml_value(LensSpec(:division; coefficients = 3)) == Dict{String, Any}("lens" => "division", "coefficients" => 3)
        withfixed = LensSpec(:kannala_brandt; coefficients = 4, fixed = (:k3, :k4))
        @test LensSpec(TOML.parse(sprint(TOML.print, toml_value(withfixed)))) == withfixed
        @test toml_value(withfixed)["fixed"] == ["k3", "k4"] && !haskey(toml_value(withfixed), "free")   # saves `fixed` only
        @test LensSpec(lens(withfixed); fixed = (:k3, :k4)) == withfixed
        @test LensSpec(:division; coefficients = 3) != LensSpec(:division; coefficients = 2)
        @test hash(LensSpec(:division; coefficients = 3)) == hash(LensSpec(:division; coefficients = 3))
    end

    @testset "record escape hatch" begin
        L = Lens(Perspective(), Additive(ThinPrism((0.001,), (-0.002,))) ∘ Radial((-0.2, 0.05)))
        spec = LensSpec(Dict("record" => lens_record(L)))
        @test lens(spec) == L
        @test LensSpec(L) == spec
        @test LensSpec(toml_value(spec)) == spec
        @test LensSpec(TOML.parse(sprint(TOML.print, toml_value(spec)))) == spec
        @test toml_value(spec)["lens"] isa AbstractDict
        # a named structure at non-reference values is not its named form: the numbers are kept
        D = Division(-0.1, 0.02)
        @test lens(LensSpec(D)) == D
        @test lens(LensSpec(Lens(Perspective(), Radial((-0.1,))))) == Lens(Perspective(), Radial((-0.1,)))
        @test_throws ArgumentError LensSpec(Dict("record" => Dict("fish" => [1.0])))
    end

    @testset "refusals" begin
        e = try
            LensSpec(:division; radial = 3)
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("coefficients", e.msg)
        @test_throws ArgumentError LensSpec(:division)
        @test_throws ArgumentError LensSpec(:division; coefficients = 0)
        @test_throws ArgumentError LensSpec(:division; coefficients = 2.5)
        @test_throws ArgumentError LensSpec(:brown_conrady; radial = 2, tangential = 1)
        @test_throws ArgumentError LensSpec(:perspective; coefficients = 1)
        e = try
            LensSpec(:fish)
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test all(f -> occursin(f, e.msg), ("division", "kannala_brandt", "brown_conrady", "rational", "equidistant"))
        @test_throws ArgumentError LensSpec(Dict("lens" => "division"))
        @test_throws ArgumentError LensSpec(Dict("coefficients" => 3))
    end

    @testset "fixed: the coefficients held; free_names the complement" begin
        @test free_names(LensSpec(:kannala_brandt; coefficients = 4, fixed = (:k3, :k4))) == (:k1, :k2)
        @test free_names(LensSpec(:kannala_brandt; coefficients = 4, fixed = [:k3, :k4])) == (:k1, :k2)
        @test free_names(LensSpec(:kannala_brandt; coefficients = 4, fixed = (:k4, :k3))) == (:k1, :k2)   # the chart's order
        @test free_names(LensSpec(:kannala_brandt; coefficients = 4)) == (:k1, :k2, :k3, :k4)
        # the retired spellings are refused, naming `fixed`
        @test_throws r"give `fixed`" LensSpec(:kannala_brandt; coefficients = 4, free = (:k1, :k2))
        @test_throws r"give `fixed`" LensSpec(KannalaBrandt(0.0, 0.0); free = (:k1,))
        @test_throws r"give \"fixed\"" LensSpec(Dict("lens" => "kannala_brandt", "coefficients" => 4, "free" => ["k1"]))
        e = try
            LensSpec(:kannala_brandt; coefficients = 2, fixed = (:k1, :k9))
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin(":k9", e.msg) && occursin("its parameters are k1, k2", e.msg)
        @test_throws ArgumentError LensSpec(:division; coefficients = 2, fixed = (:d1, :d1))
        # a group holds a family's coefficients; a name held twice through it is refused
        @test free_names(LensSpec(:brown_conrady; radial = 3, tangential = true, fixed = (:p,))) == (:k1, :k2, :k3)
        @test_throws r"twice" LensSpec(:brown_conrady; radial = 3, fixed = (:k, :k2))
    end

    @testset "free_parameter_names: names, groups, the gauge" begin
        fpn = VisualGeometryToolkitCore.free_parameter_names
        intr = (:f, :aspect, :skew, :cx, :cy)
        @test fpn(intr, ()) == (:f, :aspect, :cx, :cy)                       # skew is the gauge: never free
        @test fpn(intr, (:skew,)) == (:f, :aspect, :cx, :cy)                 # naming it is accepted, a no-op
        @test fpn(intr, intr) == ()                                          # fixed = parameter_names(model) holds all
        @test fpn(intr, [:cx, :cy]) == (:f, :aspect)
        rat = (:f, :cx, :cy, :k1, :k2, :k3, :d1, :d2, :p1, :p2)
        @test fpn(rat, (:k,)) == (:f, :cx, :cy, :d1, :d2, :p1, :p2)          # a group: k1…k3
        @test fpn(rat, (:d, :k3)) == (:f, :cx, :cy, :k1, :k2, :p1, :p2)
        @test fpn(rat, (:p2, :f)) == (:cx, :cy, :k1, :k2, :k3, :d1, :d2, :p1)   # the names' order, not fixed's
        @test fpn((:k1, :k10, :k2), (:k,)) == ()                             # every digit run is the group's
        @test fpn((:sx1, :sy1, :s1), (:s,)) == (:sx1, :sy1)                  # a stem takes digits only
        @test_throws r"no parameter or group" fpn(rat, (:q,))
        @test_throws r"no parameter or group" fpn(rat, (:k1x,))
        @test_throws r"twice" fpn(rat, (:k1, :k1))
        @test_throws r"twice" fpn(rat, (:k, :k1))
        @test_throws ArgumentError fpn(rat, ("k1",))                          # symbols only
        @test fpn(LensSpec(:division; coefficients = 2) |> lens, (:d1,)) == (:d2,)   # the model form
    end

    @testset "show" begin
        @test sprint(show, LensSpec(:division; coefficients = 3)) == "LensSpec(:division; coefficients = 3)"
        @test sprint(show, LensSpec(:brown_conrady; radial = 3, tangential = true)) == "LensSpec(:brown_conrady; radial = 3, tangential = true)"
        @test sprint(show, LensSpec(:equidistant)) == "LensSpec(:equidistant)"
        @test sprint(show, LensSpec(:division; coefficients = 2, fixed = (:d2,))) == "LensSpec(:division; coefficients = 2, fixed = (:d2,))"
        # the record form lists the record's keys, sorted
        rec = LensSpec(Division(-0.1, 0.02))
        @test sprint(show, rec) == "LensSpec(record = {" * join(sort(collect(keys(rec.record))), ", ") * "})"
        @test startswith(sprint(show, rec), "LensSpec(record = {")
        @test endswith(sprint(show, LensSpec(Division(-0.1, 0.02); fixed = (:d2,))), ", fixed = (:d2,))")
    end
end

@testitem "camera_model_spec" begin
    using Test, TOML
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: toml_value, SplineLensSpec, intrinsics_kind

    roundtrip(spec) = CameraModelSpec(TOML.parse(sprint(TOML.print, toml_value(spec))))

    @testset "metric lens spec" begin
        spec = CameraModelSpec(LensSpec(:division; coefficients = 2))
        @test intrinsics_kind(spec) === LogicalIntrinsics
        @test (@inferred intrinsics_kind(spec)) === LogicalIntrinsics
        d = toml_value(spec)
        @test d["intrinsics"] == "metric" && d["lens"] == "division" && d["coefficients"] == 2
        @test roundtrip(spec) == spec
        @test CameraModelSpec(Dict("lens" => "division", "coefficients" => 2)) == spec   # intrinsics defaults to metric
        rec = CameraModelSpec(LensSpec(Division(0.1, 0.0)))
        @test roundtrip(rec) == rec
        held = CameraModelSpec(LensSpec(:kannala_brandt; coefficients = 4, fixed = (:k3, :k4)))
        @test roundtrip(held) == held
        @test hash(roundtrip(spec)) == hash(spec)
    end

    @testset "a parameterized intrinsics kind is normalized to its wrapper" begin
        lens = LensSpec(:division; coefficients = 2)
        spec = CameraModelSpec(lens; intrinsics = LogicalIntrinsics{Float64})
        @test spec == CameraModelSpec(lens)
        @test intrinsics_kind(spec) === LogicalIntrinsics
        @test roundtrip(spec) == spec
        @test hash(spec) == hash(CameraModelSpec(lens))
        pspec = CameraModelSpec(lens; intrinsics = ProjectiveIntrinsics{Float64})
        @test intrinsics_kind(pspec) === ProjectiveIntrinsics
        @test roundtrip(pspec) == pspec
    end

    @testset "the inner constructor normalizes a parameterized intrinsics kind too" begin
        lens = LensSpec(:division; coefficients = 2)
        spec = CameraModelSpec{LogicalIntrinsics{Float64}, LensSpec}(lens)
        @test intrinsics_kind(spec) === LogicalIntrinsics
        @test spec == CameraModelSpec(lens)
        @test intrinsics_kind(CameraModelSpec{ProjectiveIntrinsics{Float32}, LensSpec}(lens)) === ProjectiveIntrinsics
    end

    @testset "projective lens spec" begin
        spec = CameraModelSpec(LensSpec(:division; coefficients = 2); intrinsics = ProjectiveIntrinsics)
        @test intrinsics_kind(spec) === ProjectiveIntrinsics
        d = toml_value(spec)
        @test d["intrinsics"] == "projective" && d["lens"] == "division" && d["coefficients"] == 2
        back = roundtrip(spec)
        @test back == spec
        @test intrinsics_kind(back) === ProjectiveIntrinsics
        @test back != CameraModelSpec(LensSpec(:division; coefficients = 2))   # the kind is part of the spec
        @test hash(back) == hash(spec)
    end

    @testset "spline lens spec" begin
        for kw in (NamedTuple(), (; grid_size = (9, 7)), (; calib_area = (0, 0, 640, 480)), (; grid_size = (9, 7), calib_area = (0, 0, 640, 480)))
            spec = CameraModelSpec(:spline; kw...)
            @test intrinsics_kind(spec) === LogicalIntrinsics
            @test spec.model isa SplineLensSpec
            @test toml_value(spec)["type"] == "spline"
            @test Set(keys(toml_value(spec))) == Set(vcat("type", String[String(k) for k in keys(kw)]))
            @test roundtrip(spec) == spec
        end
        @test_throws ArgumentError CameraModelSpec(:spline; grid_size = (0, 7))
        @test_throws ArgumentError CameraModelSpec(:spline; calib_area = (1, 2, 3))
        @test_throws ArgumentError CameraModelSpec(:nonsense)
        @test_throws ArgumentError CameraModelSpec(Dict("type" => "spline", "bogus" => 1))
        # the config template's `intrinsics = "metric"` left beside an uncommented `type`:
        # redundant but true (a spline lens is metric), so accepted; any other kind is refused
        g = CameraModelSpec(Dict("type" => "spline", "intrinsics" => "metric", "grid_size" => [9, 7]))
        @test g == CameraModelSpec(:spline; grid_size = (9, 7))
        e = try
            CameraModelSpec(Dict("type" => "spline", "intrinsics" => "projective")); nothing
        catch err
            err
        end
        @test e isa ArgumentError && occursin("metric", e.msg) && occursin("projective", e.msg)
        # the grid's table type before Core 0.9 is refused, naming the type it is now
        e = try
            CameraModelSpec(Dict("type" => "central" * "_generic", "grid_size" => [9, 7])); nothing
        catch err
            err
        end
        @test e isa ArgumentError && occursin("\"spline\"", e.msg)
    end

    @testset "spline spec hash" begin
        a = CameraModelSpec(:spline; grid_size = (9, 7), calib_area = (0, 0, 640, 480))
        b = CameraModelSpec(:spline; grid_size = [9, 7], calib_area = [0, 0, 640, 480])
        @test a == b && hash(a) == hash(b)
        @test hash(a) != hash(CameraModelSpec(:spline; grid_size = (9, 8), calib_area = (0, 0, 640, 480)))
        @test hash(CameraModelSpec(:spline)) == hash(CameraModelSpec(:spline))
        @test hash(SplineLensSpec(; grid_size = (9, 7))) == hash(SplineLensSpec(; grid_size = [9, 7]))
        @test hash(roundtrip(a)) == hash(a)
    end

    @testset "refusals" begin
        e = try
            CameraModelSpec(Dict("intrinsics" => "spherical", "lens" => "division", "coefficients" => 2)); nothing
        catch err
            err
        end
        @test e isa ArgumentError && occursin("metric, projective", e.msg)
    end

    @testset "show" begin
        @test sprint(show, CameraModelSpec(LensSpec(:division; coefficients = 2))) == "CameraModelSpec(LensSpec(:division; coefficients = 2))"
        @test sprint(show, CameraModelSpec(:spline; grid_size = (9, 7))) == "CameraModelSpec(:spline; grid_size = (9, 7))"
    end
end
