# =============================================================================
# Quality gates: Aqua, ExplicitImports, JET
#
# Added in 0.8.0 with the package-cleanup audit, which found (and removed) a
# dataset layer nobody here used, 31 stale imports, two stale dependencies,
# pirated Base methods and an export that shadowed Base.match. These keep them
# from coming back. Every exclusion below says why it exists.
# =============================================================================

@testitem "Aqua" begin
    using Aqua
    import VisualGeometryToolkitCore as VgtkCore
    Aqua.test_all(
        VisualGeometryToolkitFeatures;
        # its own item below: it precompiles a throwaway package, the slowest check
        persistent_tasks = false,
        # Glue by design, not new piracy. Detection is an alias of VgtkCore's
        # Attributed, so its constructors, property forwarding and geometry
        # conversions extend VgtkCore functions on VgtkCore types. The fix is to move
        # the type into VgtkCore. (The JSON serialization of VgtkCore frames, Size2 and
        # FlexUnits quantities left for VisualGeometryToolkitIO, as explicit
        # functions rather than JSON.jl methods.) A pirated method on anything
        # else fails this test. (Distributions'
        # MvNormal left the list with its serializer and Detection conversions,
        # which nothing called.)
        piracies = (;
            treat_as_own = [
                VgtkCore.Attributed, VgtkCore.AbstractDetection, VgtkCore.FeatureFrame,
                VgtkCore.Disc, VgtkCore.OrientedDisc, VgtkCore.EllipticFrame, VgtkCore.AffineFrame, VgtkCore.Ellipse,
            ],
        ),
    )
end

@testitem "ExplicitImports" begin
    using ExplicitImports
    # every name says which package it comes from
    @test ExplicitImports.check_no_implicit_imports(VisualGeometryToolkitFeatures) === nothing
    # an import nothing reads is where the last refactor's residue hides (31 had
    # accumulated, most from the dataset layer that moved out)
    @test ExplicitImports.check_no_stale_explicit_imports(VisualGeometryToolkitFeatures) === nothing
end

@testitem "JET" begin
    using JET
    # RATCHET: a ceiling that may only go down; lower it whenever reports are
    # fixed. The remaining reports are false positives JET cannot see through:
    # kernel launches on an abstract KernelAbstractions.Backend in the device
    # matchers (matching_ka.jl, matching_stacked_ka.jl). 13 → 7 when the host
    # cross-warp NMS kernels were folded into the pool ones and the stacked
    # matcher's reverse buffers became one `nothing`-or-NamedTuple. Measure it
    # under Pkg.test, not a bare REPL: the test target loads more packages.
    BASELINE = 7
    reports = JET.get_reports(
        JET.report_package(
            VisualGeometryToolkitFeatures;
            target_modules = (VisualGeometryToolkitFeatures,)
        )
    )
    if length(reports) != BASELINE
        for r in reports
            @info "JET" report = first(replace(sprint(show, r), "\n" => " "), 300)
        end
    end
    @test length(reports) <= BASELINE
end

@testitem "Aqua: persistent tasks" begin
    # Precompiles a throwaway package that loads this one, to catch background
    # tasks that would hang precompilation. It is Aqua's slowest check, so it
    # runs on its own worker instead of holding up the others.
    using Aqua
    # Needs Aqua ≥ 0.8.18 (the compat floor): 0.8.17's walk of every dependency's
    # [deps] threw on LogExpFunctions 1.0.1, which lists ChainRulesCore in both
    # [deps] and [weakdeps], so this was skipped from 2026-09-17 to 2026-09-21.
    Aqua.test_persistent_tasks(VisualGeometryToolkitFeatures)
end

@testitem "quality: stack contracts" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitFeatures
    # The stack contracts, enforced (VisualGeometryToolkitCore quality/stack_contracts.jl):
    # no private name of another stack package, and methods on other stack packages'
    # functions only on their CONTRACT_GENERICS.
    root = pkgdir(VisualGeometryToolkitFeatures)
    dirs = filter(isdir, [joinpath(root, "src"), joinpath(root, "ext")])
    @test isempty(reduce(vcat, [VgtkCore.private_name_violations(d, "VisualGeometryToolkitFeatures") for d in dirs]))
    @test isempty(VgtkCore.contract_generic_violations(VisualGeometryToolkitFeatures))
end
