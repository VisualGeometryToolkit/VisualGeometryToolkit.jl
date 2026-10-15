# =============================================================================
# Re-keying — the old-key → new-key map of a tag rename, computed BEFORE the
# rename lands, so no keyed result moves without a record of where it went.
#
# A run key is a content hash of a config-derived table (`run_key` here; a
# project may key differently, e.g. a hash of its own `config_dict`). Renaming a
# TOML tag (`"estimate"` → `"estimated"`) changes that table, so the key moves
# although no number does. `rekey_map` takes each config, builds the exact table
# the project's key function hashes (with the CURRENT code), applies the tag
# renames to that table, and hashes both: the old key is today's, the new key is
# the one the renamed code will compute. The map is written once per project and
# rename (`write_rekey_map`), and the identity probes after the rename assert
# `new key == map[old key]` and that every key not in the map is unchanged.
# =============================================================================

"""
    TagRename(path::Regex, old, new)

One tag rename: a string VALUE `old` whose dotted key path matches `path`
becomes `new`. The path of a value is its keys joined by `.`, an array element
adding `[i]` (`"selectors[1]"`, `"geometry.local_optimization.type"`), so a
rename can be scoped to the family that owns the tag: `"fixed"` is one tag in
`nuisance.scale` and another in `window.type`.
"""
struct TagRename
    path::Regex
    old::String
    new::String
end

"""
    rename_tags(table, renames) -> (renamed, applied)

A deep copy of `table` with every matching rename applied, and the list of
`path => (old, new)` substitutions made. A value matched by two renames is an
`ArgumentError` (the renames must be unambiguous).
"""
function rename_tags(table::AbstractDict, renames::AbstractVector{TagRename})
    applied = Pair{String, Tuple{String, String}}[]
    return _rename(table, renames, "", applied), applied
end

function _rename(d::AbstractDict, renames, path, applied)
    out = Dict{String, Any}()
    for (k, v) in d
        out[string(k)] = _rename(v, renames, isempty(path) ? string(k) : "$path.$k", applied)
    end
    return out
end
_rename(v::AbstractVector, renames, path, applied) =
    Any[_rename(x, renames, "$path[$i]", applied) for (i, x) in enumerate(v)]
function _rename(v::AbstractString, renames, path, applied)
    hits = [r for r in renames if r.old == v && occursin(r.path, path)]
    isempty(hits) && return v
    length(hits) == 1 || throw(ArgumentError("value \"$v\" at $path matches $(length(hits)) renames"))
    push!(applied, path => (v, hits[1].new))
    return hits[1].new
end
_rename(v, _renames, _path, _applied) = v

"""
    rekey_map(configs; renames, keyed_table, key) -> Vector{NamedTuple}

For each config path in `configs`: `keyed_table(path)` is the table the
project's key function hashes, computed with the current code, and `key(table)`
that hash. Returns one row per config — `config`, `old_key`, `new_key`, `moved`
and the `applied` substitutions. A config whose `keyed_table` throws is
recorded with `error` set and no keys (it is not keyed by this function).
"""
function rekey_map(configs; renames::AbstractVector{TagRename}, keyed_table, key)
    rows = NamedTuple[]
    for c in configs
        t = try
            keyed_table(c)
        catch e
            push!(rows, (config = string(c), old_key = "", new_key = "", moved = false,
                applied = Pair{String, Tuple{String, String}}[], error = sprint(showerror, e)))
            continue
        end
        renamed, applied = rename_tags(t, renames)
        old, new = key(t), key(renamed)
        push!(rows, (config = string(c), old_key = old, new_key = new, moved = old != new, applied, error = ""))
    end
    return rows
end

"""
    write_rekey_map(path, rows; project, renames, commits = Dict()) -> path

Write a `rekey_map` as TOML: the project, the renames, the commits it was
computed at, and per config its old and new keys and the substitutions.
"""
function write_rekey_map(path::AbstractString, rows; project::AbstractString,
                         renames::AbstractVector{TagRename}, commits::AbstractDict = Dict{String, String}())
    d = Dict{String, Any}(
        "project" => project,
        "commits" => Dict{String, Any}(string(k) => string(v) for (k, v) in commits),
        "renames" => [Dict("path" => r.path.pattern, "old" => r.old, "new" => r.new) for r in renames],
        "configs" => [Dict{String, Any}(
            "config" => r.config, "old_key" => r.old_key, "new_key" => r.new_key, "moved" => r.moved,
            "applied" => [Dict("path" => first(a), "old" => last(a)[1], "new" => last(a)[2]) for a in r.applied],
            "error" => r.error) for r in rows],
    )
    mkpath(dirname(path))
    open(io -> TOML.print(io, d; sorted = true), path, "w")
    return path
end

"""
    read_rekey_map(path) -> Dict{String,String}

The old-key → new-key map of a written re-key file (moved keys only).
"""
read_rekey_map(path::AbstractString) =
    Dict{String, String}(c["old_key"] => c["new_key"] for c in TOML.parsefile(path)["configs"] if c["moved"])
