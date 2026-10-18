# =============================================================================
# quality_gates_tests.jl — the package-wide quality gates.
#
# 1. Aqua: stale deps, undefined exports, unbound args, project extras, compat
#    bounds, piracies, ambiguities, persistent tasks.
# 2. ExplicitImports: no implicit import, no stale explicit import, and every
#    explicit import taken from the module that owns the name.
# 3. No method references an undefined global — every method DEFINED here,
#    extensions of upstream functions included (a name an upstream makes
#    `public` stops resolving through a bare `using`). A module loads fine with
#    such a method and fails only when the line runs — the suite never runs
#    most of this package (dataset readers, figure emitters). Gensym'd names
#    (`##…`, from macros such as `Threads.@threads`) are not globals and are
#    skipped.
# 4. No function owned here shares its name with a VisualGeometryToolkitCore
#    function: a method for this package's types extends the owner's function.
# 5. Every exported and public name has a docstring.
# 6. Calling conventions: VisualGeometryToolkitCore's vendored argorder_check.jl
#    (types positional and before the data, no precision keyword, no public
#    function that mutates an argument without `!`). It finds nothing here, so
#    there is no .argorder_allow; an allowlist, if one is ever needed, only
#    shrinks.
# 7. JET, as a RATCHET: `JET_BASELINE` only ever decreases. It is 0; the two
#    reports it once had were HDF5 handle reads union-split over
#    Dataset/Datatype/Group, narrowed where the handle is indexed.
# =============================================================================

@testitem "quality_gates" setup = [GlobalRefScan] begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using Aqua, ExplicitImports
    using VisualGeometryToolkitEval
    import VisualGeometryToolkitCore

    const _QG_M = VisualGeometryToolkitEval
    const _QG_MODS = (_QG_M, _QG_M.Stats, _QG_M.Tuning, _QG_M.Evaluation, _QG_M.Experiments,
                      _QG_M.Render, _QG_M.Pools, _QG_M.Datasets, _QG_M.DataArtifacts,
                      _QG_M.PaperProjects)

    @testset "quality gates" begin
        # The stack contracts (VisualGeometryToolkitCore quality/stack_contracts.jl),
        # enforced: the dependency allow-list, no private name of another stack
        # package, methods on other stack packages' functions only on their
        # CONTRACT_GENERICS.
        @testset "stack contracts" begin
            root = pkgdir(_QG_M)
            @test isempty(VgtkCore.stack_dependency_violations(joinpath(root, "Project.toml"), "VisualGeometryToolkitEval"))
            @test isempty(vcat(VgtkCore.private_name_violations(joinpath(root, "src"), "VisualGeometryToolkitEval"),
                               VgtkCore.private_name_violations(joinpath(root, "ext"), "VisualGeometryToolkitEval")))
            @test isempty(VgtkCore.contract_generic_violations(_QG_M))
        end
        @testset "Aqua" begin
            Aqua.test_stale_deps(_QG_M)
            Aqua.test_undefined_exports(_QG_M)
            Aqua.test_unbound_args(_QG_M)
            Aqua.test_project_extras(_QG_M)
            Aqua.test_deps_compat(_QG_M)
            Aqua.test_piracies(_QG_M)
            Aqua.test_ambiguities(_QG_M)
            Aqua.test_persistent_tasks(_QG_M)
        end

        @testset "ExplicitImports" begin
            @test check_no_implicit_imports(_QG_M) === nothing
            @test check_no_stale_explicit_imports(_QG_M) === nothing
            @test check_all_explicit_imports_via_owners(_QG_M) === nothing
        end

        @testset "no method references an undefined global" begin
            bad = undefined_globals(methods_defined_in(Set(_QG_MODS)))
            isempty(bad) || @info "undefined globals" bad
            @test isempty(bad)
        end

        # Every exported and `public` name is documented. Under Julia 1.12 a
        # blank or comment line between a docstring and its definition, or a
        # docstring inside an `if`, silently documents nothing.
        @testset "every public name has a docstring" begin
            undocumented = [string(nameof(m), ".", n) for m in _QG_MODS for n in names(m)
                            if n !== nameof(m) && !(getfield(m, n) isa Module) && !Docs.hasdoc(m, n)]
            isempty(undocumented) || @info "undocumented public names" undocumented
            @test isempty(undocumented)
        end

        @testset "no function shadows a VisualGeometryToolkitCore function" begin
            shadows = String[]
            for mod in _QG_MODS, n in names(mod; all = true)
                isdefined(mod, n) || continue
                f = getfield(mod, n)
                (f isa Function && parentmodule(f) === mod && !startswith(string(n), "#")) || continue
                isdefined(VisualGeometryToolkitCore, n) && getfield(VisualGeometryToolkitCore, n) isa Function &&
                    push!(shadows, "$(nameof(mod)).$n")
            end
            @test isempty(shadows)
        end
    end
end

@testitem "quality: calling conventions" begin
    using Test
    using VisualGeometryToolkitEval
    import VisualGeometryToolkitCore

    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "argorder_check.jl"))
    allow = read_allowlist(joinpath(pkgdir(VisualGeometryToolkitEval), ".argorder_allow"))
    bad = [f for f in check(VisualGeometryToolkitEval) if (f[1], f[2]) ∉ allow]
    isempty(bad) || foreach(f -> @warn("argument-order finding", f), bad)
    @test isempty(bad)
end

@testitem "quality: JET" begin
    using Test
    using JET
    using VisualGeometryToolkitEval

    JET_BASELINE = 0
    reports = JET.get_reports(JET.report_package(VisualGeometryToolkitEval; target_modules = (VisualGeometryToolkitEval,)))
    length(reports) > JET_BASELINE &&
        foreach(r -> @info("JET", report = first(replace(sprint(show, r), "\n" => " "), 240)), reports)
    @test length(reports) <= JET_BASELINE
end

# The evaluation layer sits BELOW the paper packages: VisualGeometryToolkitEval
# evaluates any scorer and any local optimizer through VisualGeometryToolkitCore's
# contracts (and VisualGeometryToolkitEstimation's loop), so it may not depend on the
# scorer package (RansacScoringDoneRight) or the optimizer package
# (LocalOptimizationDoneRight) — not as a dependency, a weak dependency, a test
# extra or a source, and no file of src/, ext/ or test/ loads either (K+,
# 2026-10-09: the composition evaluator moved here from
# LocalOptimizationDoneRight on that condition).
@testitem "quality: no paper-layer dependency" begin
    using Test, TOML
    using VisualGeometryToolkitEval

    root = pkgdir(VisualGeometryToolkitEval)
    proj = TOML.parsefile(joinpath(root, "Project.toml"))
    forbidden = Dict("RansacScoringDoneRight" => "e23d4225-f201-4c3a-a297-7efae5f2deeb",
                     "LocalOptimizationDoneRight" => "87a2e6a3-7afc-471f-bd1b-aa3740a29cd9")
    for (name, uuid) in forbidden
        for section in ("deps", "weakdeps", "extras", "sources", "compat")
            table = get(proj, section, Dict{String,Any}())
            @test !haskey(table, name)
            @test !(uuid in string.(values(table)))
        end
        @test !(name in get(get(proj, "targets", Dict{String,Any}()), "test", String[]))
    end
    # no file loads either (`using`/`import` of the module, qualified or not)
    loads = Regex("^\\s*(using|import)\\s+[^#]*\\b(" * join(keys(forbidden), "|") * ")\\b")
    offenders = String[]
    for dir in filter(d -> isdir(joinpath(root, d)), ["src", "ext", "test"]), (d, _, fs) in walkdir(joinpath(root, dir)), f in fs
        endswith(f, ".jl") || continue
        for (i, line) in enumerate(eachline(joinpath(d, f)))
            occursin(loads, line) && push!(offenders, "$(relpath(joinpath(d, f), root)):$i")
        end
    end
    @test isempty(offenders) || (@info "paper-layer loads" offenders; false)
end
