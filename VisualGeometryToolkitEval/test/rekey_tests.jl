@testitem "rekey" begin
# The re-key map of a tag rename: renames are scoped by key path, the old key is
# today's hash and the new key the hash of the renamed table, an unrenamed config
# keeps its key, and the written map reads back as old → new.
using Test, TOML
using VisualGeometryToolkitEval.Experiments: TagRename, rename_tags, rekey_map, write_rekey_map, read_rekey_map
using VisualGeometryToolkitEval: _content_key

renames = [TagRename(r"(^|\.)nuisance\.scale$", "estimate", "estimated"),
           TagRename(r"(^|\.)selectors\[\d+\]$", "ig", "student_t_uniform"),
           TagRename(r"(^|\.)window\.type$", "all", "exhaustive")]
t = Dict("nuisance" => Dict("scale" => "estimate", "nu" => "estimate"),
         "selectors" => ["ig", "msac"], "window" => Dict("type" => "fixed"), "n" => 3)
r, applied = rename_tags(t, renames)
@test r["nuisance"]["scale"] == "estimated" && r["nuisance"]["nu"] == "estimate"   # scoped by path
@test r["selectors"] == ["student_t_uniform", "msac"] && r["window"]["type"] == "fixed" && r["n"] == 3
@test sort(first.(applied)) == ["nuisance.scale", "selectors[1]"]
@test t["nuisance"]["scale"] == "estimate"                                          # input untouched
@test_throws ArgumentError rename_tags(Dict("a" => "x"), [TagRename(r"a", "x", "y"), TagRename(r"^a$", "x", "z")])

dir = mktempdir()
moved = joinpath(dir, "a.toml"); write(moved, "selectors = [\"ig\"]\n")
kept = joinpath(dir, "b.toml"); write(kept, "selectors = [\"msac\"]\n")
bad = joinpath(dir, "c.toml"); write(bad, "x = \n")
rows = rekey_map([moved, kept, bad]; renames, keyed_table = TOML.parsefile, key = _content_key)
@test rows[1].moved && rows[1].old_key == _content_key(TOML.parsefile(moved)) &&
      rows[1].new_key == _content_key(Dict("selectors" => ["student_t_uniform"]))
@test !rows[2].moved && rows[2].old_key == rows[2].new_key
@test !isempty(rows[3].error)
p = write_rekey_map(joinpath(dir, "_rekey", "map.toml"), rows; project = "test", renames, commits = Dict("X" => "abc"))
@test read_rekey_map(p) == Dict(rows[1].old_key => rows[1].new_key)
end

@testitem "migrate_keyed_dirs: the retired <kind>-<key> layout to the step's <kind>/<key>" begin
    using Test
    import VisualGeometryToolkitEval as VgtkEval
    const E = VgtkEval.Experiments
    E.set_project_root!(mktempdir())
    root = mktempdir()
    withenv("VGTK_RESULTS_ROOT" => root, "VGTK_SCRATCH" => "1") do               # scratch: the records need no git checkout
        rec = Dict{String, Any}("case" => "a")
        k = E.run_key(rec)
        old = joinpath(root, "calibrate", "refine-" * k); mkpath(old); write(joinpath(old, "parameters.json"), "{}")
        E.write_provenance(old, Dict("run_key" => k); driver = "d")
        other = joinpath(root, "calibrate", "refine-" * "0"^16); mkpath(other)
        E.write_provenance(other, Dict("run_key" => "f"^16); driver = "d")          # its record names another key: left
        plain = joinpath(root, "calibrate", "notes"); mkpath(plain)
        moves = (@test_logs (:warn, r"left in place") E.migrate_keyed_dirs(root))
        @test moves == [old => joinpath(root, "calibrate", "refine", k)] && isdir(old)   # dry run: nothing moved
        @test (@test_logs (:warn, r"left in place") E.migrate_keyed_dirs(root; dry_run = false)) == moves
        @test !isdir(old) && isdir(other) && isdir(plain)
        s = E.Step("calibrate", "refine", rec; output = "parameters.json")
        @test E.step_dir(s) == joinpath(root, "calibrate", "refine", k) && isfile(E.step_output(s))
        # an occupied destination is refused, nothing moved
        again = joinpath(root, "calibrate", "refine-" * k); mkpath(again); E.write_provenance(again, Dict("run_key" => k); driver = "d")
        @test_throws r"destination\(s\) exist, nothing moved" E.migrate_keyed_dirs(root; dry_run = false)
        @test isdir(again)
    end
    E.set_project_root!("")
end
