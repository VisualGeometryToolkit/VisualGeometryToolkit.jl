# =============================================================================
# GlobalRefScan — find methods whose lowered code names a global that is not
# defined. Such a method loads fine and fails only when the line runs (the
# 2026-09-21 export demotions left several). Shared by the datasets item and
# the package-wide quality gate.
# =============================================================================

@testsetup module GlobalRefScan
    export methods_defined_in, undefined_globals

    globalrefs(x::GlobalRef) = (x,)
    globalrefs(x::Expr) = Iterators.flatten(map(globalrefs, x.args))
    globalrefs(x) = ()

    # Every method DEFINED in `mods` — including methods these modules add to
    # other packages' functions (`VisualGeometryToolkitCore.loss_family(::X) = …`) and
    # type constructors, which a walk over the modules' own names never reaches:
    # every function and type of every loaded module (submodules too) is visited.
    function methods_defined_in(mods)
        ms = Method[]
        seen = Base.IdSet{Any}(); visited = Base.IdSet{Module}()
        function visit(M::Module)
            M in visited && return
            push!(visited, M)
            for n in names(M; all = true)
                # A deprecated binding (another package's, e.g. GeometryBasics'
                # `Tesselation`) is not read: under `--depwarn=error` reading it throws.
                (isdefined(M, n) && !Base.isdeprecated(M, n)) || continue
                v = getfield(M, n)
                if v isa Module
                    parentmodule(v) === M && v !== M && visit(v)
                elseif (v isa Function || v isa Type) && !(v in seen)
                    push!(seen, v)
                    for m in methods(v)
                        m.module in mods && push!(ms, m)
                    end
                end
            end
        end
        foreach(visit, Base.loaded_modules_array())
        return unique(ms)
    end

    # "name at file:line" for every undefined global the methods reference.
    function undefined_globals(ms)
        bad = String[]
        for m in ms
            ci = try
                Base.uncompressed_ast(m)
            catch
                continue
            end
            for stmt in ci.code, x in globalrefs(stmt)
                startswith(String(x.name), "#") && continue
                isdefined(x.mod, x.name) || push!(bad, "$(x.name) at $(m.file):$(m.line)")
            end
        end
        return unique!(bad)
    end
end
