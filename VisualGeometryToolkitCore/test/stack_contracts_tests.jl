# stack_contracts_tests.jl — the stack's quality contracts (src/quality/
# tag_registry.jl, src/quality/stack_contracts.jl) and the gates that run them
# (ext/VisualGeometryToolkitCoreQualityExt): the tag rule, the registry, the source
# and config scans, the dependency allow-list, private names across packages
# and the contract generics.

@testitem "stack contracts: snake_case" begin
    using Test
    using VisualGeometryToolkitCore: snake_case
    for (name, tag) in (
            :FixedS2 => "fixed_s2", :SmoothTLS => "smooth_tls", :WelschHQNewton => "welsch_hq_newton",
            :LogPerDecade => "log_per_decade", :StuTuBCD => "stu_tu_bcd", :GmHqEps => "gm_hq_eps",
            :ExhaustiveSampling => "exhaustive_sampling", :StudentTUniform => "student_t_uniform",
            :GemanMcclureShape => "geman_mcclure_shape", :SimFitPP => "sim_fit_pp",
            :NoLocalOptimization => "no_local_optimization",
        )
        @test snake_case(name) == tag
    end
    # a digit stays with its letters; an uppercase letter after one starts a word
    @test snake_case(:Homography4pt) == "homography4pt"
    @test snake_case(:Irls2D) == "irls2_d"
    @test snake_case("TLS") == "tls"
    @test snake_case(:lower_case) == "lower_case"
end

@testitem "stack contracts: tag registry" begin
    using Test
    using VisualGeometryToolkitCore: TagRegistry, tag_of, expected_tag, tag_rule_violations, tag_family,
        tag_paths, retired_tags, tag_registries
    abstract type Lo end
    struct KeepModel <: Lo end
    struct WeightedRefine{N} <: Lo end
    struct TLSShape end
    struct GemanMcclureShape end

    r = TagRegistry(
        :model_update, "", "keep_model" => KeepModel, "weighted_refine" => WeightedRefine, "none" => nothing;
        retired = ("keep" => "keep_model",), paths = [r"^lo\.model_update$"]
    )
    @test r["keep_model"] === KeepModel
    @test r["none"] === nothing
    @test collect(keys(r)) == ["keep_model", "weighted_refine", "none"]
    @test collect(r) == ["keep_model" => KeepModel, "weighted_refine" => WeightedRefine, "none" => nothing]
    @test length(r) == 3 && haskey(r, "none") && !haskey(r, "keep")
    @test get(r, "missing", 0) == 0
    @test tag_family(r) === :model_update && tag_paths(r) == [r"^lo\.model_update$"]
    @test retired_tags(r) == ["keep" => "keep_model"]
    # a retired tag names its successor; an unknown tag lists the known ones
    err(f) = try
        f()
        nothing
    catch e
        e
    end
    e = err(() -> r["keep"])
    @test e isa ArgumentError && e.msg == "tag \"keep\" was renamed \"keep_model\""
    e = err(() -> r["kept"])
    @test e isa ArgumentError && occursin("\"keep_model\", \"weighted_refine\", \"none\"", e.msg)
    # the writer side: a value, a type, a parameterization, an instance
    @test tag_of(r, nothing) == "none"
    @test tag_of(r, KeepModel) == "keep_model"
    @test tag_of(r, WeightedRefine{2}) == "weighted_refine"
    @test tag_of(r, KeepModel()) == "keep_model"
    @test_throws ArgumentError tag_of(r, TLSShape)
    @test isempty(tag_rule_violations(r))

    # the rule: snake_case of the name, the family suffix stripped
    losses = TagRegistry(:loss, "Shape", "tls" => TLSShape, "geman_mcclure" => GemanMcclureShape)
    @test isempty(tag_rule_violations(losses))
    @test expected_tag(losses, TLSShape) == "tls"
    bad = TagRegistry(
        :loss, "Shape", "TLS" => TLSShape, "gm" => GemanMcclureShape, "nothing" => nothing;
        retired = ("old" => "gone",)
    )
    v = tag_rule_violations(bad)
    @test length(v) == 4
    @test any(occursin("\"TLS\"", x) && occursin("should be \"tls\"", x) for x in v)
    @test any(occursin("should be \"none\"", x) for x in v)
    @test any(occursin("\"gone\"", x) for x in v)
    # the most specific registered supertype wins
    nested = TagRegistry(:lo, "", "lo" => Lo, "keep_model" => KeepModel)
    @test tag_of(nested, KeepModel) == "keep_model" && tag_of(nested, WeightedRefine{1}) == "lo"
    # structural errors at construction: a duplicate, a tag both current and retired
    @test_throws ArgumentError TagRegistry(:x, "", "a" => KeepModel, "a" => TLSShape)
    @test_throws ArgumentError TagRegistry(:x, "", "a" => KeepModel; retired = ("a" => "b",))
    # discovery: every registry constant of a module tree
    @eval module RegistryHolder
    using VisualGeometryToolkitCore: TagRegistry
    const LOSSES = TagRegistry(:loss, "", "none" => nothing)
    module Inner
    using VisualGeometryToolkitCore: TagRegistry
    const MODES = TagRegistry(:mode, "", "none" => nothing)
    end
    end
    found = tag_registries(RegistryHolder)
    @test sort(first.(found)) == sort(["$(RegistryHolder).LOSSES", "$(RegistryHolder.Inner).MODES"])
end

@testitem "stack contracts: source and config scans" begin
    using Test
    using VisualGeometryToolkitCore: TagRegistry, tag_table_violations, config_tag_violations, private_name_violations
    struct KeepModel end
    tree(files) = (dir = mktempdir(); foreach(files) do (rel, src)
        mkpath(dirname(joinpath(dir, rel)))
        write(joinpath(dir, rel), src)
    end; dir)

    # tag tables: a Dict of types and a string paired with a type, outside a registry
    dir = tree((
        "a.jl" => "const T = Dict{String, Type}(\"keep\" => KeepModel)\n",
        "b.jl" => "const U = Dict(\n    \"fixed\" => FixedScale,\n    \"other\" => Estimate{2},\n)\n",
        "c.jl" => """
        # "keep" => KeepModel in a comment
        \"\"\"
        A docstring: Dict{String, Type} and "keep" => KeepModel.
        \"\"\"
        const R = TagRegistry(:lo, "",
            "keep_model" => KeepModel,   # registered
            "weighted_refine" => WeightedRefine)
        spec = ("cutoff" => Float32, "band_reach" => Int)   # converters, not tags
        x = "a" => KeepModel(1)                              # a call, not a type
        s = "a # b" * "c"
        """,
    ))
    v = tag_table_violations(dir)
    @test length(v) == 3
    @test startswith(v[1], "a.jl:1:") && startswith(v[2], "b.jl:2:") && startswith(v[3], "b.jl:3:")
    @test isempty(tag_table_violations(joinpath(dir, "nowhere")))

    # config tags: every string at a registry's paths is a current tag
    reg = TagRegistry(
        :model_update, "", "keep_model" => KeepModel, "none" => nothing;
        retired = ("keep" => "keep_model",), paths = [r"lo\.model_update", r"arms\[\d+\]\.update"]
    )
    cfg = tree((
        "default.toml" => "[lo]\nmodel_update = \"keep_model\"\nother = \"keep\"\n",
        "sub/arms.toml" => "[[arms]]\nupdate = \"keep\"\n[[arms]]\nupdate = \"refit\"\n[[arms]]\nupdate = \"none\"\n",
        "notes.txt" => "update = \"keep\"\n",
    ))
    v = config_tag_violations([reg], [cfg])
    @test length(v) == 2
    @test occursin("arms[1].update", v[1]) && occursin("was renamed \"keep_model\"", v[1])
    @test occursin("arms[2].update", v[2]) && occursin("unknown model_update tag \"refit\"", v[2])
    @test config_tag_violations(["NAME" => reg], [cfg]) == v
    # a path matches whole: a longer key is another key
    @test isempty(config_tag_violations([reg], [tree(("a.toml" => "[lo]\nmodel_update_note = \"keep\"\n",))]))
    @test only(config_tag_violations([reg], [tree(("bad.toml" => "x = \n",))])) |> s -> occursin("does not parse", s)

    # private names: `_` names reached from another stack package, imported or qualified
    src = tree((
        "a.jl" => """
        using VisualGeometryToolkitCore: public_name, _private_one,
            _private_two   # a continued list
        import VisualGeometryToolkitEstimation.Sub: _sub_private
        x = RANSACScoringDoneRight._qualified(1)
        y = VisualGeometryToolkitCore.Cameras._nested
        # VisualGeometryToolkitCore._in_a_comment
        s = "VisualGeometryToolkitCore._in_a_string"
        z = MyOwnPackage._own
        """,
        "b.jl" => "const R = VisualGeometryToolkitEstimation\nimport RANSACScoringDoneRight as S\nR._via_const(); S._via_as()\n",
    ))
    v = private_name_violations(src, "MyOwnPackage")
    @test sort(last.(split.(v, ": "))) == sort([
        "VisualGeometryToolkitCore._private_one", "VisualGeometryToolkitCore._private_two",
        "VisualGeometryToolkitEstimation.Sub._sub_private", "RANSACScoringDoneRight._qualified",
        "VisualGeometryToolkitCore.Cameras._nested", "VisualGeometryToolkitEstimation._via_const", "RANSACScoringDoneRight._via_as",
    ])
    @test startswith(only(filter(x -> occursin("_private_two", x), v)), "a.jl:1:")
    # a package's own names are its own
    @test !any(occursin("_qualified", x) for x in private_name_violations(src, "RANSACScoringDoneRight"))
end

@testitem "stack contracts: dependency allow-list" begin
    using Test
    using VisualGeometryToolkitCore: stack_dependencies, stack_dependency_violations
    t = stack_dependencies()
    bb = ["VisualGeometryToolkitCore", "VisualGeometryToolkitIO", "VisualGeometryToolkitFeatures", "VisualGeometryToolkitEstimation", "RANSACScoringDoneRight", "LocalOptimizationDoneRight"]
    @test t == Dict(
        "VisualGeometryToolkitCore" => String[],
        "VisualGeometryToolkitIO" => ["VisualGeometryToolkitCore"],
        "VisualGeometryToolkitFeatures" => ["VisualGeometryToolkitCore"],
        "VisualGeometryToolkitEval" => ["VisualGeometryToolkitCore", "VisualGeometryToolkitIO", "VisualGeometryToolkitEstimation"],
        "PoseLib" => ["VisualGeometryToolkitCore"],
        "VisualGeometryToolkitEstimation" => ["VisualGeometryToolkitCore"],
        "RANSACScoringDoneRight" => ["VisualGeometryToolkitCore"],
        "LocalOptimizationDoneRight" => ["VisualGeometryToolkitCore", "VisualGeometryToolkitEstimation", "PoseLib"],
        "BlobBoards" => bb,
        "accv2026" => ["LocalOptimizationDoneRight", "RANSACScoringDoneRight", "VisualGeometryToolkitEval", "VisualGeometryToolkitEstimation", "VisualGeometryToolkitCore", "VisualGeometryToolkitIO", "PoseLib"],
        "neurips2026" => ["RANSACScoringDoneRight", "VisualGeometryToolkitEval", "VisualGeometryToolkitEstimation", "VisualGeometryToolkitCore"],
        "bmvc2026" => vcat(bb[1:3], "VisualGeometryToolkitEval", bb[4:end], "BlobBoards"),
    )
    # every row names stack packages only, never itself
    @test all(all(p -> haskey(t, p) && p != k, v) for (k, v) in t)

    proj = joinpath(mktempdir(), "Project.toml")
    write(proj, """
    name = "VisualGeometryToolkitEstimation"
    [deps]
    VisualGeometryToolkitCore = "96820d6b-1dad-4e77-ad4a-8276f2ec1ce7"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    [weakdeps]
    PoseLib = "a8c7e0b4-3d2f-4f1c-9a1e-7b2d9c8f4a3b"
    [extras]
    RANSACScoringDoneRight = "e23d4225-f201-4c3a-a297-7efae5f2deeb"
    [targets]
    test = ["RANSACScoringDoneRight", "Test"]
    """)
    @test stack_dependency_violations(proj, "VisualGeometryToolkitEstimation") ==
        ["weakdeps: PoseLib", "extras: RANSACScoringDoneRight", "targets.test: RANSACScoringDoneRight"]
    @test stack_dependency_violations(proj, "LocalOptimizationDoneRight") == ["extras: RANSACScoringDoneRight", "targets.test: RANSACScoringDoneRight"]
    @test_throws ArgumentError stack_dependency_violations(proj, "NotAStackProject")
    # a project kept elsewhere supplies its own row, as a dictionary or a TOML file
    row = Dict("SomeProject" => ["VisualGeometryToolkitCore", "VisualGeometryToolkitEstimation", "RANSACScoringDoneRight"])
    @test stack_dependency_violations(proj, "SomeProject"; extra = row) == ["weakdeps: PoseLib"]
    rows = joinpath(mktempdir(), "rows.toml")
    write(rows, "SomeProject = [\"VisualGeometryToolkitCore\", \"PoseLib\", \"RANSACScoringDoneRight\"]\n")
    @test isempty(stack_dependency_violations(proj, "SomeProject"; extra = rows))
    @test stack_dependencies(; extra = rows) == merge(t, Dict("SomeProject" => ["VisualGeometryToolkitCore", "PoseLib", "RANSACScoringDoneRight"]))
    # an extra row adds a project, never replaces a published row
    @test_throws ArgumentError stack_dependencies(; extra = Dict("VisualGeometryToolkitEstimation" => String[]))
    # this package
    @test isempty(stack_dependency_violations(joinpath(pkgdir(VisualGeometryToolkitCore), "Project.toml"), "VisualGeometryToolkitCore"))
end

@testitem "stack contracts: contract generics" begin
    using Test
    using VisualGeometryToolkitCore: CONTRACT_GENERICS, contract_generic_violations, centroid, snake_case
    # every contract is a function this package owns, listed once
    @test all(f -> f isa Function && Base.moduleroot(parentmodule(f)) === VisualGeometryToolkitCore, CONTRACT_GENERICS)
    @test allunique(CONTRACT_GENERICS)
    @test centroid in CONTRACT_GENERICS && !(snake_case in CONTRACT_GENERICS)
    # the camera maps and the Manifold protocol are extension points
    for n in (:project, :backproject, :tangent_size, :retract, :recenter, :local_coordinates,
              :retract_jacobian, :action_jacobian, :zero_tangent, :restrict, :tangent_part,
              :transform_covariance, :minimal_manifold, :model)
        @test getfield(VisualGeometryToolkitCore, n) in CONTRACT_GENERICS
    end
    # this package extends no other stack package's functions
    @test isempty(contract_generic_violations(VisualGeometryToolkitCore))

    # a package that adds a method to a contract, and one to a non-contract
    @eval module Downstream
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitCore
    struct Thing end
    VgtkCore.centroid(::Thing) = 0
    VgtkCore.snake_case(::Thing) = "thing"
    end
    try
        v = contract_generic_violations(Downstream)
        @test length(v) == 1 && startswith(only(v), "VisualGeometryToolkitCore.snake_case (")
    finally
        for f in (centroid, snake_case), m in methods(f)
            m.module === Downstream && Base.delete_method(m)
        end
    end
    @test isempty(contract_generic_violations(Downstream))
end

@testitem "stack contracts: gate modes" begin
    using Test, Aqua, ExplicitImports, JET
    Q = Base.get_extension(VisualGeometryToolkitCore, :VisualGeometryToolkitCoreQualityExt)
    # a test set that keeps what is recorded and hands nothing to its parent
    mutable struct Kept <: Test.AbstractTestSet
        description::String
        results::Vector{Any}
        Kept(desc; kw...) = new(desc, Any[])
    end
    Test.record(ts::Kept, r) = (push!(ts.results, r); r)
    Test.finish(ts::Kept) = ts
    kept(mode, found) = (@testset Kept "probe" begin
        Q._gate(mode, "probe", found)
    end).results

    @test only(kept(:report, () -> ["a finding"])) isa Test.Broken
    @test only(kept(:report, () -> String[])) isa Test.Pass
    @test only(kept(:enforce, () -> ["a finding"])) isa Test.Fail
    @test only(kept(:enforce, () -> String[])) isa Test.Pass
    @test isempty(kept(:off, () -> error("never called")))
    # a collector that throws is one finding
    @test only(kept(:enforce, () -> error("boom"))) isa Test.Fail
    @test Q._findings(() -> throw(ArgumentError("boom"))) == ["ArgumentError: boom"]
    @test only(kept(:sometimes, () -> String[])) isa Test.Error
end
