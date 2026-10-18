# =============================================================================
# VisualGeometryToolkitCoreQualityExt — the stack's static quality gates
# =============================================================================
#
# `test_quality_gates` (stub and docstring in src/quality/test_helpers.jl): the
# Aqua / ExplicitImports / JET ratchets every stack package's quality test used
# to write out for itself, and the stack-contract gates (tag rule, tag tables,
# config tags, the dependency allow-list, public cross-package names, contract
# generics; functions in src/quality/). Loads with the four test packages,
# never with the package.
# =============================================================================
module VisualGeometryToolkitCoreQualityExt
    import VisualGeometryToolkitCore as VgtkCore

using Test: Test, @test, @test_broken, @testset
using Aqua: Aqua
using ExplicitImports: check_no_implicit_imports, check_all_explicit_imports_via_owners,
    check_no_stale_explicit_imports, check_all_qualified_accesses_are_public,
    check_all_explicit_imports_are_public
using JET: JET
using TOML: TOML
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, tag_registries, tag_rule_violations,
    tag_table_violations, config_tag_violations, stack_dependency_violations,
    private_name_violations, contract_generic_violations
import VisualGeometryToolkitCore: test_quality_gates

# Aqua hunts only methods whose parent module IS the module it is given, so the
# piracy gate walks every submodule.
function _modules_of(m::Module)
    subs = [getfield(m, n) for n in names(m; all = true) if isdefined(m, n) && !Base.isdeprecated(m, n)]
    return vcat(m, [_modules_of(x) for x in subs if x isa Module && x !== m && parentmodule(x) === m]...)
end

# The root modules M loads itself: Base, Core, M and its Project [deps].
function _own_roots(M::Module)
    deps = get(TOML.parsefile(joinpath(pkgdir(M), "Project.toml")), "deps", Dict{String, Any}())
    return Set{Symbol}([:Base, :Core, nameof(M), Symbol.(keys(deps))...])
end

# A KernelAbstractions LAUNCH report: a no-method report whose every call
# signature is `kwcall(::NamedTuple{(:ndrange,)}, ::KA.Kernel, …)`. Not a file,
# not a message pattern — any other report in a kernel file counts.
_launch_sig(t) = t isa DataType && t <: Tuple && length(t.parameters) >= 3 &&
    t.parameters[1] === typeof(Core.kwcall) && t.parameters[2] <: NamedTuple{(:ndrange,)} &&
    t.parameters[3] <: VgtkCore.KA.Kernel
_ka_launch(r) = r isa JET.MethodErrorReport && all(_launch_sig, r.t isa Type ? (r.t,) : r.t)

# The filter's controls: a real launch on an abstract GPU backend is filtered; a
# plain no-method report and an `ndrange` kwcall on a non-kernel still count.
# The backend is a GPU no package implements, so the report is the same whether
# or not a test item sharing this worker loaded CUDA (whose `CUDABackend` makes
# a launch on the abstract `KA.GPU` resolvable).
struct _DecoyGPU <: VgtkCore.KA.GPU end
_decoy_launch(b, r, M) = VgtkCore._row_kernel!(b)(VgtkCore._sampson_residual_row!, r, M, r, r, r; ndrange = 1)
_decoy_kw(x::Int; ndrange) = x
_decoy_plain(x) = x + "a"

# -----------------------------------------------------------------------------
# The stack-contract gates: each collects findings (strings), then reports or
# enforces them by its mode.
# -----------------------------------------------------------------------------

const _GATE_MODES = (:report, :enforce, :off)

# A finding collector that throws reports the error as its one finding (the
# ExplicitImports checks, an unknown project in the allow-list).
function _findings(f)
    return try
        collect(String, f())
    catch e
        e isa InterruptException && rethrow()
        [sprint(showerror, e)]
    end
end

# `:enforce` — `@test` the findings are empty, each logged on failure;
# `:report` — each finding logged and the gate recorded broken (empty: an
# ordinary test that holds); `:off` — nothing.
function _gate(mode::Symbol, what::AbstractString, f)
    mode in _GATE_MODES || throw(ArgumentError("gate mode for $what must be one of $_GATE_MODES, got :$mode"))
    mode === :off && return nothing
    found = _findings(f)
    isempty(found) || @warn "$what: $(length(found)) finding(s) ($mode)" findings = join(found, "\n")
    if mode === :enforce || isempty(found)
        @test isempty(found)
    else
        @test_broken isempty(found)
    end
    return found
end

function _stack_gates(
        M::Module; tag_rules, tag_tables, config_tags, config_dirs, registries,
        stack_dependencies, stack_name, stack_extra, public_names, third_party_public, qualified_ignore, contract_generics
    )
    dir = pkgdir(M)
    @testset "stack contracts" begin
        _gate(tag_rules, "tag rule", () -> [v for (_, r) in tag_registries(M) for v in tag_rule_violations(r)])
        _gate(tag_tables, "tag tables outside TagRegistry", () -> tag_table_violations(joinpath(dir, "src")))
        dirs = config_dirs === nothing ? filter(isdir, [joinpath(dir, "configs")]) : config_dirs
        regs = vcat(Any[last(r) for r in tag_registries(M)], collect(Any, registries))
        _gate(config_tags, "config tags", () -> config_tag_violations(regs, dirs))
        _gate(stack_dependencies, "stack dependency allow-list",
            () -> stack_dependency_violations(joinpath(dir, "Project.toml"), stack_name; extra = stack_extra))
        # The two ExplicitImports checks throw on a finding: the error is the finding.
        _gate(third_party_public, "explicit imports are public", () -> (check_all_explicit_imports_are_public(M); String[]))
        _gate(third_party_public, "qualified accesses are public",
            () -> (check_all_qualified_accesses_are_public(M; ignore = Tuple(something(qualified_ignore, ()))); String[]))
        _gate(public_names, "private names across stack packages",
            () -> vcat(private_name_violations(joinpath(dir, "src"), stack_name), private_name_violations(joinpath(dir, "ext"), stack_name)))
        _gate(contract_generics, "methods on other stack packages' non-contract functions", () -> contract_generic_violations(M))
    end
    return nothing
end

function test_quality_gates(
        M::Module; jet_bound::Integer, treat_as_own = [], stale_ignore = (),
        qualified_ignore = nothing, persistent_tasks::Bool = true,
        tag_rules::Symbol = :enforce, tag_tables::Symbol = :enforce, config_tags::Symbol = :enforce,
        config_dirs = nothing, registries = (), stack_dependencies::Symbol = :enforce,
        stack_name::AbstractString = string(nameof(M)), stack_extra = nothing, public_names::Symbol = :enforce,
        third_party_public::Symbol = :report, contract_generics::Symbol = :enforce
    )
    return @testset "quality gates: $M" begin
        _stack_gates(
            M; tag_rules, tag_tables, config_tags, config_dirs, registries, stack_dependencies,
            stack_name, stack_extra, public_names, third_party_public, qualified_ignore, contract_generics
        )
        @testset "Aqua" begin
            Aqua.test_undefined_exports(M)
            Aqua.test_unbound_args(M)
            Aqua.test_stale_deps(M)
            Aqua.test_deps_compat(M)
            Aqua.test_project_extras(M)
            # Loading can pull KernelAbstractions/GPUArrays, so the 60 s default
            # times out on a cold depot.
            persistent_tasks && Aqua.test_persistent_tasks(M; tmax = 300)
            # In-process rather than `Aqua.test_ambiguities` (a subprocess): only
            # pairs with a method of M, and only among the modules M loads itself —
            # Base, Core, M and its [deps]. A pair touching anything else (a test
            # item's module, rooted at Main; a package another item in this worker
            # loaded, e.g. CUDA) is what a fresh process would not see, so the
            # result does not depend on which items share the worker. A dropped
            # pair is logged.
            own = _own_roots(M)
            all_ambiguities = Test.detect_ambiguities(M; recursive = true)
            ambiguities = filter(p -> all(m -> nameof(Base.moduleroot(parentmodule(m))) in own, p), all_ambiguities)
            length(ambiguities) == length(all_ambiguities) ||
                @info "ambiguities involving modules M does not load (not the package's)" pairs = setdiff(all_ambiguities, ambiguities)
            @test isempty(ambiguities)
            for m in _modules_of(M)
                Aqua.test_piracies(m; treat_as_own)
            end
        end
        @testset "ExplicitImports" begin
            @test check_no_implicit_imports(M) === nothing
            # Names are imported from the module that owns them — a re-export can move or vanish.
            @test check_all_explicit_imports_via_owners(M) === nothing
            @test check_no_stale_explicit_imports(M; ignore = Tuple(stale_ignore)) === nothing
            qualified_ignore === nothing ||
                @test check_all_qualified_accesses_are_public(M; ignore = Tuple(qualified_ignore)) === nothing
        end
        @testset "JET" begin
            launch = JET.get_reports(JET.report_call(_decoy_launch, (_DecoyGPU, AbstractVector, Any)))
            @test !isempty(launch) && all(_ka_launch, launch)
            for decoy in (JET.report_call(x -> _decoy_kw(x; ndrange = 2), (String,)), JET.report_call(_decoy_plain, (Int,)))
                rs = JET.get_reports(decoy)
                @test !isempty(rs) && !any(_ka_launch, rs)
            end
            reports = JET.get_reports(JET.report_package(M; target_modules = (M,), toplevel_logger = nothing))
            counted = filter(!_ka_launch, reports)
            if length(counted) > jet_bound
                for r in counted
                    @info "JET" report = first(replace(sprint(show, r), "\n" => " "), 240)
                end
            end
            @test length(counted) <= jet_bound
        end
    end
end

end # module VisualGeometryToolkitCoreQualityExt
