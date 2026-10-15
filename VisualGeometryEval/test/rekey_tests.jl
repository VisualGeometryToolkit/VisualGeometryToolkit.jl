@testitem "rekey" begin
# The re-key map of a tag rename: renames are scoped by key path, the old key is
# today's hash and the new key the hash of the renamed table, an unrenamed config
# keeps its key, and the written map reads back as old → new.
using Test, TOML
using VisualGeometryEval.Experiments: TagRename, rename_tags, rekey_map, write_rekey_map, read_rekey_map
using VisualGeometryEval.Tuning: _toml_hash

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
rows = rekey_map([moved, kept, bad]; renames, keyed_table = TOML.parsefile, key = _toml_hash)
@test rows[1].moved && rows[1].old_key == _toml_hash(TOML.parsefile(moved)) &&
      rows[1].new_key == _toml_hash(Dict("selectors" => ["student_t_uniform"]))
@test !rows[2].moved && rows[2].old_key == rows[2].new_key
@test !isempty(rows[3].error)
p = write_rekey_map(joinpath(dir, "_rekey", "map.toml"), rows; project = "test", renames, commits = Dict("X" => "abc"))
@test read_rekey_map(p) == Dict(rows[1].old_key => rows[1].new_key)
end
