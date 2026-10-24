# =============================================================================
# Tooling gates — what keeps the package clean between audits
# =============================================================================
#
# Aqua, ExplicitImports and JET run in the DEFAULT suite: each catches a class
# of rot no behavioural test here can see — a dep that stopped being used, an
# export that names nothing, a compat entry that went missing, an upstream name
# silently arriving through `using`, a schema field reaching typed code as an
# untyped JSON value.
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

    #   SQLite.DB, DBInterface.execute/close! — the sqlite handle's user-facing
    #                                          surface is undeclared; nothing
    #                                          else to call.
    @test check_all_qualified_accesses_are_public(
        VisualGeometryToolkitIO;
        ignore = (:DB, :close!, :execute)
    ) === nothing
end

@testitem "quality: JET" begin
    using JET

    # The ratchet: `JET_BASELINE` only ever decreases. It is 0, which is a real
    # claim about this package and not a lucky measurement. Almost every report
    # JET has raised here came from an untyped read: a field of a parsed JSON
    # object (with JSON3, a 7-way `Union`; with JSON.jl, `Any`), or iterating an
    # `AbstractString`, which yields `Any` (canonical.jl narrows the char).
    # Those are fixed at the read boundary with a typeassert per schema field
    # (`d.frame::String`, `d.poses::Vector{Any}`, `sv.rms::Real`), where the
    # expected type is a documented property of the schema and the assert
    # doubles as validation. An assert inside a callee is union-split at its
    # caller and reports anyway, so narrow where the field is read. If this
    # starts failing, look for a new `d.<field>` that reaches a typed function
    # without an assert; do not add annotations downstream.
    #
    # The other source is splatting a variable-length argument list into a
    # call Julia cannot infer: Julia 1.13 reported `join(string.(xs), '\t')`
    # over a splatted tuple in the TRC writer, which now prints field by field.
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
