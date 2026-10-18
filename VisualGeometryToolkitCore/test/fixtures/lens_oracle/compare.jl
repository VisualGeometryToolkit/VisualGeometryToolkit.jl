# Compares the redesigned lens families against the oracle recorded from the
# pre-redesign zoo (generate.jl). Returns, per case and quantity, the worst
# disagreement — in units in the last place for finite pairs — and the NaN-mask
# mismatches. Shared by test/lens_oracle_tests.jl and interactive checks.
using VisualGeometryToolkitCore, StaticArrays, TOML
const _VGC = VisualGeometryToolkitCore

_pairs(v, n) = [SVector{n}(v[i:(i + n - 1)]) for i in 1:n:length(v)]
_ulps(a, b, ::Type{T} = Float64) where {T} = (a == b) ? 0.0 : abs(a - b) / eps(T(max(abs(a), abs(b), floatmin(T))))

# The pre-redesign Brown–Conrady always carried twelve coefficients
# (k1…k6, p1, p2, s1…s4); its migrated lens carries only the non-zero ones,
# under the generated names. The old parameter Jacobian's columns, by new name:
const _BC_OLD = (:k1, :k2, :k3, :k4, :k5, :k6, :p1, :p2, :s1, :s2, :s3, :s4)
const _BC_NEW_TO_OLD = Dict(:k1 => :k1, :k2 => :k2, :k3 => :k3, :d1 => :k4, :d2 => :k5, :d3 => :k6,
    :p1 => :p1, :p2 => :p2, :sx1 => :s1, :sx2 => :s2, :sy1 => :s3, :sy2 => :s4)
function _old_columns(name, lens, np)
    names = lens_parameter_names(lens)
    startswith(name, "opencv") || return collect(1:length(names))
    return [findfirst(==(_BC_NEW_TO_OLD[n]), _BC_OLD) for n in names]
end

function _cmp(got, want, ::Type{T} = Float64) where {T}
    worst, worst_abs, nanmismatch = 0.0, 0.0, 0
    for (g, w) in zip(got, want), (x, y) in zip(g, w)
        if isnan(x) || isnan(y)
            nanmismatch += isnan(x) != isnan(y)
        else
            worst = max(worst, _ulps(x, y, T)); worst_abs = max(worst_abs, abs(x - y))
        end
    end
    return (; ulps = worst, abs = worst_abs, nan = nanmismatch)
end

function compare_oracle(path = joinpath(@__DIR__, "lens_oracle.toml"))
    o = TOML.parsefile(path)
    out = Dict{String, Any}()
    for (name, d) in o
        name == "generator" && continue
        lens = lens_from_record(d["lens"])
        rays = _pairs(d["rays"], 3); pts = _pairs(d["points"], 2); persp = _pairs(d["persp"], 2)
        r = Dict{String, Any}()
        r["project"] = _cmp([project(lens, X) for X in rays], _pairs(d["project"], 2))
        r["backproject"] = _cmp([backproject(lens, x) for x in pts], _pairs(d["backproject"], 3))
        r["distort"] = _cmp([distort_normalized(lens, x) for x in persp], _pairs(d["distort"], 2))
        r["undistort"] = _cmp([undistort_normalized(lens, x) for x in pts], _pairs(d["undistort"], 2))
        r["jac"] = _cmp([vec(transform_deriv(lens, X)) for X in rays[1:8]], _pairs(d["jac"], 6))
        np = Int(d["nparams"])
        if np > 0
            cols = _old_columns(name, lens, np)
            old = [reshape(J, 2, np)[:, cols] for J in _pairs(d["jac_params"], 2np)]
            r["jac_params"] = _cmp([transform_deriv_params(lens, X) for X in rays[1:8]], old)
        end
        l32 = _VGC._precision(Float32, lens)
        r["project32"] = _cmp([Float64.(project(l32, SVector{3, Float32}(X))) for X in rays[1:16]], _pairs(d["project32"], 2), Float32)
        r["backproject32"] = _cmp([Float64.(backproject(l32, SVector{2, Float32}(x))) for x in pts[1:min(16, end)]], _pairs(d["backproject32"], 3), Float32)
        r["monotone_radius"] = _cmp([[Float64(_VGC.monotone_radius(lens))]], [[d["monotone_radius"]]])
        r["view_angle_limit"] = _cmp([[Float64(view_angle_limit(lens))]], [[d["view_angle_limit"]]])
        out[name] = r
    end
    return out
end
