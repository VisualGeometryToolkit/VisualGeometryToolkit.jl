# LensSpec: a lens family by name and term counts (cameras/lens_spec.jl).
@testitem "lens_spec" begin
    using Test, TOML
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: toml_value, free_names

    # name, spec, expected parameter names, the structure's lens type
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

    @testset "round trips" begin
        for (spec, _) in cases
            @test LensSpec(lens(spec)) == spec
            @test LensSpec(toml_value(spec)) == spec
            @test LensSpec(TOML.parse(sprint(TOML.print, toml_value(spec)))) == spec
        end
        @test toml_value(LensSpec(:division; coefficients = 3)) == Dict{String, Any}("lens" => "division", "coefficients" => 3)
        withfree = LensSpec(:kannala_brandt; coefficients = 4, free = (:k1, :k2))
        @test LensSpec(TOML.parse(sprint(TOML.print, toml_value(withfree)))) == withfree
        @test LensSpec(lens(withfree); free = (:k1, :k2)) == withfree
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

    @testset "free" begin
        @test free_names(LensSpec(:kannala_brandt; coefficients = 4, free = (:k1, :k2))) == (:k1, :k2)
        @test free_names(LensSpec(:kannala_brandt; coefficients = 4, free = [:k1, :k2])) == (:k1, :k2)
        e = try
            LensSpec(:kannala_brandt; coefficients = 2, free = (:k1, :k9))
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("k9", e.msg) && occursin("chart", e.msg) && occursin("(:k1, :k2)", e.msg)
        @test_throws ArgumentError LensSpec(:division; coefficients = 2, free = (:d1, :d1))
    end

    @testset "show" begin
        @test sprint(show, LensSpec(:division; coefficients = 3)) == "LensSpec(:division; coefficients = 3)"
        @test sprint(show, LensSpec(:brown_conrady; radial = 3, tangential = true)) == "LensSpec(:brown_conrady; radial = 3, tangential = true)"
        @test sprint(show, LensSpec(:equidistant)) == "LensSpec(:equidistant)"
        @test sprint(show, LensSpec(:division; coefficients = 2, free = (:d1,))) == "LensSpec(:division; coefficients = 2, free = (:d1,))"
    end
end

@testitem "camera_model_spec" begin
    using Test, TOML
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: toml_value, GenericGridSpec, intrinsics_kind

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
        free = CameraModelSpec(LensSpec(:kannala_brandt; coefficients = 4, free = (:k1, :k2)))
        @test roundtrip(free) == free
        @test hash(roundtrip(spec)) == hash(spec)
    end

    @testset "generic grid spec" begin
        for kw in (NamedTuple(), (; grid_size = (9, 7)), (; calib_area = (0, 0, 640, 480)), (; grid_size = (9, 7), calib_area = (0, 0, 640, 480)))
            spec = CameraModelSpec(:central_generic; kw...)
            @test intrinsics_kind(spec) === LogicalIntrinsics
            @test spec.model isa GenericGridSpec
            @test toml_value(spec)["type"] == "central_generic"
            @test Set(keys(toml_value(spec))) == Set(vcat("type", String[String(k) for k in keys(kw)]))
            @test roundtrip(spec) == spec
        end
        @test_throws ArgumentError CameraModelSpec(:central_generic; grid_size = (0, 7))
        @test_throws ArgumentError CameraModelSpec(:central_generic; calib_area = (1, 2, 3))
        @test_throws ArgumentError CameraModelSpec(:nonsense)
        @test_throws ArgumentError CameraModelSpec(Dict("type" => "central_generic", "bogus" => 1))
    end

    @testset "refusals" begin
        e = try
            CameraModelSpec(Dict("intrinsics" => "spherical", "lens" => "division", "coefficients" => 2)); nothing
        catch err
            err
        end
        @test e isa ArgumentError && occursin("metric, projective", e.msg)
        if !isdefined(VisualGeometryToolkitCore, :ProjectiveIntrinsics)
            e = try
                CameraModelSpec(Dict("intrinsics" => "projective", "lens" => "division", "coefficients" => 2)); nothing
            catch err
                err
            end
            @test e isa ArgumentError && occursin("ProjectiveIntrinsics", e.msg)
        end
    end

    @testset "show" begin
        @test sprint(show, CameraModelSpec(LensSpec(:division; coefficients = 2))) == "CameraModelSpec(LensSpec(:division; coefficients = 2))"
        @test sprint(show, CameraModelSpec(:central_generic; grid_size = (9, 7))) == "CameraModelSpec(:central_generic; grid_size = (9, 7))"
    end
end
