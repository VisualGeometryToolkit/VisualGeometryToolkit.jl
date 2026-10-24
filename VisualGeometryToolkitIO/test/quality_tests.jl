# =============================================================================
# Tooling gates — what keeps the package clean between audits
# =============================================================================
#
# Aqua, ExplicitImports and JET run in the DEFAULT suite: each catches a class
# of rot no behavioural test here can see — a dep that stopped being used, an
# export that names nothing, a compat entry that went missing, an upstream name
# silently arriving through `using`, a schema field reaching typed code as a
# 7-way JSON3 Union.
#
# JET used to live out of suite (`test/jet_ratchet.jl`) because this package
# declared `julia = "1.10"` and JET requires 1.12. That floor was never real —
# every VisualGeometryToolkitCore it admits requires 1.12 — and out of suite the
# ratchet rotted from its recorded 0 to 23 reports without anything noticing.
# The floor is now the true one, and the gate runs with everything else.
#
# Three items, so the gates spread over workers; each loads only its own tool.

@testitem "quality: Aqua" begin
    using Aqua

    # Run the checks INDIVIDUALLY: `test_all` aborts at the first failure and
    # hides every finding behind it.
    Aqua.test_stale_deps(VisualGeometryToolkitIO)
    Aqua.test_piracies(VisualGeometryToolkitIO)
    Aqua.test_undefined_exports(VisualGeometryToolkitIO)
    Aqua.test_unbound_args(VisualGeometryToolkitIO)
    Aqua.test_project_extras(VisualGeometryToolkitIO)
    Aqua.test_deps_compat(VisualGeometryToolkitIO)
    Aqua.test_persistent_tasks(VisualGeometryToolkitIO)
    # Ambiguities are ON here, against the usual advice to leave them off:
    # they succeed today (2.1 s), and they are the specific guard against the
    # shape this package keeps growing — a `where`-parameterized fallback
    # sitting beside concrete sibling methods (`frame_name`, `lens_name`,
    # `lens_from_params`, `_colmap_intrinsics`, `_colmap_lens`). If an
    # UPSTREAM ambiguity ever lands here, this is the first check to disable,
    # with the offending pair named in a comment.
    Aqua.test_ambiguities(VisualGeometryToolkitIO)
end

@testitem "quality: ExplicitImports" begin
    using ExplicitImports

    @test check_no_implicit_imports(VisualGeometryToolkitIO) === nothing
    @test check_no_stale_explicit_imports(VisualGeometryToolkitIO) === nothing
    @test check_no_self_qualified_accesses(VisualGeometryToolkitIO) === nothing

    # The two allowlists below are UPSTREAM PRIVACY, not this package's debt:
    # each name is the documented way to do its job but is not declared
    # `public` by its owner. They are enumerated rather than switched off so
    # that a NEW non-public dependency fails this testset.
    #
    #   params — Rotations' quaternion accessor.
    @test check_all_explicit_imports_are_public(
        VisualGeometryToolkitIO;
        ignore = (:params,)
    ) === nothing

    #   JSON3.read/write/pretty/Array — JSON3's entire user-facing surface is
    #                                   undeclared; nothing else to call.
    #   SQLite.DB, DBInterface.execute/close! — likewise for the sqlite handle.
    @test check_all_qualified_accesses_are_public(
        VisualGeometryToolkitIO;
        ignore = (:Array, :DB, :close!, :execute, :pretty, :read, :write)
    ) === nothing
end

@testitem "quality: JET" begin
    using JET

    # THE RATCHET: `JET_BASELINE` only ever DECREASES. It is 0, which is a real
    # claim about this package and not a lucky measurement — every report JET
    # has raised here came from an UNTYPED READ: almost always JSON3's property
    # access returning a 7-way `Union{Nothing, Bool, Float64, Int64, String,
    # JSON3.Array, JSON3.Object}` (once, iterating an `AbstractString`, which
    # yields `Any` — canonical.jl narrows the char). Those are fixed at the READ BOUNDARY with a typeassert per
    # schema field (`d.frame::String`, `d.poses::JSON3.Array`, `sv.rms::Real`),
    # where the expected type is a documented property of the schema and the
    # assert doubles as validation. An assert inside a CALLEE is union-split at
    # its caller and reports anyway — narrow where the field is read. If this
    # starts failing, look for a new `d.<field>` that reaches a typed function
    # without an assert; do NOT add annotations downstream.
    JET_BASELINE = 0
    reports = JET.get_reports(
        JET.report_package(VisualGeometryToolkitIO; target_modules = (VisualGeometryToolkitIO,))
    )
    if length(reports) > JET_BASELINE
        for r in reports
            @info "JET" report = first(replace(sprint(show, r), "\n" => " "), 240)
        end
    end
    @test length(reports) <= JET_BASELINE
end
