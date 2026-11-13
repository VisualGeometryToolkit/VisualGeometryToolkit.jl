# =============================================================================
# VGIoSchemaCases — the deletion cases of the validator's spec tables, and the Python schema tooling
# =============================================================================
#
# Shared by test/json_schema_tests.jl (the published JSON Schemas refuse what the
# validator refuses) and the validator's own deletion tests (v2_validation_tests.jl).

@testsetup module VGIoSchemaCases
    using VisualGeometryToolkitIO: VisualGeometryToolkitIO
    const VgtkIO = VisualGeometryToolkitIO

    export required_paths, getpath, without, with_key, plain_doc, pixi_python, schema_dir

    # The pixi `py` environment's python, or `nothing` when it is not installed or lacks `module`.
    function pixi_python(module_ = "jsonschema")
        python = joinpath(pkgdir(VgtkIO), ".pixi", "envs", "py", "bin", "python")
        isfile(python) && success(pipeline(`$python -c "import $(module_)"`; stdout = devnull, stderr = devnull)) || return nothing
        return python
    end
    schema_dir() = joinpath(pkgdir(VgtkIO), "docs", "src", "schemas")

    # A parsed document as plain mutable Dicts (String keys), the form the cases are edited in.
    plain_doc(x::AbstractDict) = Dict{String, Any}(String(k) => plain_doc(v) for (k, v) in x)
    plain_doc(x::AbstractVector) = Any[plain_doc(v) for v in x]
    plain_doc(x) = x

    # THE DELETION CASES: the paths of the required keys a document holds,
    # following its kind's spec table (validate.jl) — every `_Obj`'s required
    # keys, into the first element of every array, through the shape an
    # `_Either` or `_IfKey` selects. Deleting each in turn must be refused.
    function required_paths(spec, doc)
        out = Vector{Vector{Any}}()
        walk!(out, spec, doc, Any[])
        return out
    end
    walk!(out, s::VgtkIO._CheckedSpec, x, at) = walk!(out, s.inner, x, at)
    walk!(out, s::VgtkIO._DetectionSpec, x, at) = walk!(out, s.outer, x, at)   # (its records' shape is its header's)
    function walk!(out, s::VgtkIO._Obj, x, at)
        x isa AbstractDict || return nothing
        for (k, sk) in s.required
            haskey(x, String(k)) || continue
            push!(out, [at; String(k)])
            walk!(out, sk, x[String(k)], [at; String(k)])
        end
        for (k, sk) in s.optional
            (k !== :provenance && haskey(x, String(k))) && walk!(out, sk, x[String(k)], [at; String(k)])
        end
        return nothing
    end
    walk!(out, s::VgtkIO._Vec, x, at) = (x isa AbstractVector && !isempty(x)) ? walk!(out, s.elem, x[1], [at; 1]) : nothing
    walk!(out, s::VgtkIO._OrNull, x, at) = x === nothing ? nothing : walk!(out, s.inner, x, at)
    walk!(out, s::VgtkIO._Either, x, at) = foreach(p -> haskey(x, String(p.first)) && walk!(out, p.second, x, at), s.alts)
    walk!(out, s::VgtkIO._IfKey, x, at) = x isa AbstractDict ? walk!(out, haskey(x, String(s.key)) ? s.then : s.otherwise, x, at) : nothing
    walk!(out, s, x, at) = nothing

    getpath(x, path) = foldl((v, k) -> v[k], path; init = x)
    without(doc, path) = (d = deepcopy(doc); delete!(getpath(d, path[1:(end - 1)]), path[end]); d)
    with_key(doc, path, v) = (d = deepcopy(doc); getpath(d, path[1:(end - 1)])[path[end]] = v; d)
end
