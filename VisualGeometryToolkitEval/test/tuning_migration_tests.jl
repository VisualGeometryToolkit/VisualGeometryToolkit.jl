# A format-6 tuning sidecar (before the efficiency-constant names, 2026-11-11)
# reads through its version step, and a new one is written with the new names.
# The fixtures in data/tuning_v6/ were written by the code that wrote format 6
# (scripts/make_tuning_v6_fixture.jl): what an old record IS, not this package's
# idea of it.

@testitem "tuning_migration" begin
    using Test, TOML, JLD2
    using VisualGeometryToolkitCore: thrown_message, OutlierHalfwidth, EfficiencyConstant, TLSShape,
        GemanMcclureShape, WelschShape, CauchyShape, decision_width
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitEval as VgtkEval
    using VisualGeometryToolkitEval.Tuning: load_tuning_artifact, tuning_hash, EfficiencyConstantWindow
    using VisualGeometryToolkitEval.Evaluation: ComposedSpec, Roster, SoloSpec, deployed_budget_settings
    using VisualGeometryToolkitEstimation: Irls

    F = VgtkEval.engine_fixtures()
    dir = joinpath(@__DIR__, "data", "tuning_v6")
    expected = TOML.parsefile(joinpath(dir, "expected.toml"))
    old = "scale" * "_multiple"                     # the v6 name (retired: never a literal)
    shapes = Dict("HW +Irls(m)" => TLSShape(), "HW +GM(m)" => GemanMcclureShape(),
                  "HW +Welsch(m)" => WelschShape(), "HW +Cauchy(m)" => CauchyShape())
    # the fixture's roster, under the current names
    arms = [ComposedSpec("HW +Irls", :hw, OutlierHalfwidth(), OutlierHalfwidth(), _ -> Irls{TLSShape}());
            [ComposedSpec(b, :hw, OutlierHalfwidth(), EfficiencyConstant(), c -> Irls{typeof(k)}(; efficiency_constant = c))
             for (b, k) in sort!(collect(shapes); by = first)]]
    roster = Roster(p -> arms, (p; role = :refinement) -> SoloSpec[], p -> ComposedSpec[])
    rt = F.roster_tuning()
    v6 = joinpath(dir, "composed.jld2")

    @testset "the fixture is a format-6 file under the old names" begin
        @test JLD2.load(v6, "tuning_format") == 6
        @test haskey(JLD2.load(v6, "grid_specs"), old)
        @test haskey(JLD2.load(v6, "tuning_spec"), old * "_window")
    end

    @testset "an old record reads, in memory, and is never rewritten" begin
        bytes = read(v6)
        art = load_tuning_artifact(v6)
        @test read(v6) == bytes
        @test art.format == 6 && art.recorded_hash == expected["tuning_hash"]
        # the spec and the grids under the new names: the record the current code writes for the same run
        @test art.spec == rt.spec && art.hash == tuning_hash(rt.spec) != art.recorded_hash
        @test art.spec.scheme.efficiency_constant_window == EfficiencyConstantWindow(1.0, 3.3)
        @test art.grid_specs == rt.grids && haskey(art.grids, :efficiency_constant)
        @test art["arms"] == [a.base for a in arms] && art["mode"] == "miscal"
    end

    @testset "results unchanged: each arm deploys the multiple the old code deployed" begin
        bud = Dict(deployed_budget_settings(roster, v6))
        for (b, k) in shapes
            e = expected["arms"][b]
            @test bud[b].lo == Irls{typeof(k)}(; efficiency_constant = e["plain"])
            @test bud[b].kind === EfficiencyConstant() && bud[b].ref == bud[b].sel
        end
        gm = expected["arms"]["HW +GM(m)"]
        @test gm["plain"] === decision_width(GemanMcclureShape(), gm["stored"]) != gm["stored"]   # the half-weight currency
        @test all(expected["arms"][b]["plain"] === expected["arms"][b]["stored"] for b in keys(shapes) if b != "HW +GM(m)")
    end

    @testset "axis names in the data are renamed too" begin
        err = try deployed_budget_settings(roster, joinpath(dir, "clipped.jld2")) catch e; e end
        @test err isa ErrorException && occursin("HW +GM(m): efficiency_constant_hi", err.msg)
        @test !occursin(old, err.msg)
    end

    @testset "a new record is written with the new names" begin
        mktempdir() do tmp
            p = joinpath(tmp, "new.jld2")
            F.write_roster_sidecar(p, rt, arms; code = (VgtkEval,), mode = "miscal")
            @test JLD2.load(p, "tuning_format") == VgtkEval.ARTIFACT_FORMATS.tuning.version == 7
            @test Set(keys(JLD2.load(p, "grid_specs"))) == Set(["outlier_halfwidth", "loss_width", "efficiency_constant"])
            spec = JLD2.load(p, "tuning_spec")
            @test haskey(spec, "efficiency_constant_window") && haskey(spec, "efficiency_constant")
            @test !any(occursin(old, k) for k in keys(spec))
            art = load_tuning_artifact(p)
            @test art.format == 7 && art.hash == art.recorded_hash == tuning_hash(rt.spec)
            # a format-7 constant is plain already: deployed as it is stored
            m = rt.values[:efficiency_constant][3]
            @test Dict(deployed_budget_settings(roster, p))["HW +GM(m)"].lo == Irls{GemanMcclureShape}(; efficiency_constant = m)
        end
    end

    @testset "a format-6 file whose spec was altered is refused" begin
        mktempdir() do tmp
            forged = joinpath(tmp, "forged.jld2")
            cp(v6, forged)
            JLD2.jldopen(forged, "r+") do f
                delete!(f, "tuning_hash"); f["tuning_hash"] = "0000000000000000"
            end
            m = thrown_message(() -> load_tuning_artifact(forged); shown = true)
            @test occursin(forged, m) && occursin("hashes to", m) && occursin(expected["tuning_hash"], m)
        end
    end

    @testset "the version step renames a coarse-fine artifact's axes" begin
        tree = Dict{String,Any}("schema" => "vgtk_eval/tuning/v6",
            "tuning_spec" => JLD2.load(v6, "tuning_spec"), "tuning_hash" => JLD2.load(v6, "tuning_hash"),
            "grid_specs" => JLD2.load(v6, "grid_specs"), "axes" => [old, "loss_width"],
            "selected" => Dict(old => 2.0, "loss_width" => 0.004), "boundary_hit" => Dict(old => false),
            "param_name" => old)
        t = VgtkIO.migrate_record(tree)
        @test t["schema"] == "vgtk_eval/tuning/v7"
        @test t["axes"] == ["efficiency_constant", "loss_width"] && t["param_name"] == "efficiency_constant"
        @test t["selected"] == Dict("efficiency_constant" => 2.0, "loss_width" => 0.004)
        @test t["boundary_hit"] == Dict("efficiency_constant" => false)
        @test VgtkIO.migrate_record(t) == t                               # a current record comes back as it is
    end
end
