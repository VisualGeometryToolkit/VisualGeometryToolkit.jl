# =============================================================================
# The chain-equivalence corpus: the calibrations and lenses whose outputs the
# goldens in this folder pin, and the goldens' line format.
# =============================================================================
#
# Included by `generate.jl` (which wrote the goldens once, on the code before
# the two chain types were replaced by CoordinateTransformations' `∘`) and by
# the equivalence test items of VisualGeometryToolkitCore and
# VisualGeometryToolkitIO, which rebuild the corpus on the current code and
# compare. The corpus is built from the parameters stored in
# `corpus_params.txt`, never from a random stream, so a reader needs no RNG.
#
# The ONE place the composition is spelled is the `compose` argument: a
# function of the members IN APPLICATION ORDER (innermost first) that returns
# their composition. Everything else here is independent of how a composition
# is represented.
#
# Golden line format, one value per line:
#
#     key<TAB>type<TAB>dims<TAB>payload
#
# type `f`: Float64s, `repr` (shortest round trip, so bit-exact), column-major
# over `dims` (comma-separated; empty for a scalar); `b`: Bools as 0/1; `s`:
# symbols or names, space-separated; `t`: one text, `escape_string`d.

module ChainEquivalenceCorpus

using StaticArrays: SVector, SMatrix
using LinearAlgebra: normalize
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: Rigid, Affine, ThinPlateSpline, HeightField, Lens, Perspective, Equidistant,
    Radial, Tangential, ThinPrism, Tilt, Additive, grid_nodes, with_unit, m

export golden_line, read_goldens, golden_value, target_cases, lens_cases, design_points, physical_points,
    rays, image_points, jacobian_count, KIND_LETTERS

# -----------------------------------------------------------------------------
# The line format
# -----------------------------------------------------------------------------

_dims(x::AbstractArray) = join(size(x), ",")
golden_line(key::AbstractString, x::Bool) = string(key, "\tb\t\t", Int(x))
golden_line(key::AbstractString, x::Real) = string(key, "\tf\t\t", repr(Float64(x)))
golden_line(key::AbstractString, x::AbstractArray{Bool}) = string(key, "\tb\t", _dims(x), "\t", join(Int.(vec(x)), " "))
golden_line(key::AbstractString, x::AbstractArray{<:Real}) =
    string(key, "\tf\t", _dims(x), "\t", join((repr(Float64(v)) for v in vec(x)), " "))
golden_line(key::AbstractString, x::Tuple{Vararg{Symbol}}) = string(key, "\ts\t", length(x), "\t", join(String.(x), " "))
golden_line(key::AbstractString, x::AbstractString) = string(key, "\tt\t\t", escape_string(x))

"""
    read_goldens(path) -> Dict{String, Any}

Every line of a golden file: Float64 arrays (a scalar for empty dims), Bool
arrays, `Tuple`s of `Symbol`s and `String`s, by key.
"""
function read_goldens(path::AbstractString)
    out = Dict{String, Any}()
    for line in eachline(path)
        isempty(line) && continue
        key, type, dims, payload = split(line, '\t'; limit = 4)
        sz = isempty(dims) ? () : Tuple(parse.(Int, split(dims, ',')))
        out[String(key)] = if type == "f"
            v = isempty(payload) ? Float64[] : parse.(Float64, split(payload, ' '))
            isempty(sz) ? only(v) : reshape(v, sz)
        elseif type == "b"
            v = isempty(payload) ? Bool[] : parse.(Int, split(payload, ' ')) .== 1
            isempty(sz) ? only(v) : reshape(v, sz)
        elseif type == "s"
            isempty(payload) ? () : Tuple(Symbol.(split(payload, ' ')))
        else
            unescape_string(payload)
        end
    end
    return out
end
golden_value(g::AbstractDict, key) = haskey(g, key) ? g[key] : error("chain equivalence: no golden \"$key\"")

# -----------------------------------------------------------------------------
# Inputs: points, rays and image points (deterministic grids, no RNG)
# -----------------------------------------------------------------------------

# 200 design points in [−0.02, 0.12]² (the nodes span [0, 0.1]²): a 20 × 10
# lattice, irrationally offset so no point sits on a node.
design_points() = [SVector(-0.02 + 0.14 * (i + 0.3183) / 20, -0.02 + 0.14 * (j + 0.2718) / 10) for j in 0:9 for i in 0:19]
physical_points() = [p .* m for p in design_points()]
const jacobian_count = 10

# 200 unit rays: a perspective lens's within 40° of the axis, a fisheye's within 95°.
function rays(maxangle::Real)
    out = SVector{3, Float64}[]
    for j in 0:9, i in 0:19
        θ = maxangle * (j + 0.5) / 10
        ψ = 2π * (i + 0.37) / 20
        push!(out, SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ)))
    end
    return out
end
# 200 image points in the projection's frame, |x| up to `rmax`.
image_points(rmax::Real) = [SVector(rmax * (j + 0.5) / 10 * cos(2π * (i + 0.61) / 20), rmax * (j + 0.5) / 10 * sin(2π * (i + 0.61) / 20)) for j in 0:9 for i in 0:19]

# -----------------------------------------------------------------------------
# The target calibrations
# -----------------------------------------------------------------------------

const KIND_LETTERS = (R = :rigid, A = :affine, T = :thin_plate_spline, H = :height_field)

# The four kinds from the stored parameters, plain (lengths as plain metres) or
# physical (the same lengths in metres).
function kinds(P::AbstractDict; physical::Bool = false)
    nodes = grid_nodes(3; lo = 0.0, hi = 0.1)
    A = Affine(SMatrix{2, 2}(P["param/affine/A"]), P["param/affine/b"])
    T = ThinPlateSpline(nodes, P["param/thin_plate_spline/fx"], P["param/thin_plate_spline/fy"])
    H = HeightField(nodes, P["param/height_field/h"])
    ks = (R = Rigid(), A = A, T = T, H = H)
    return physical ? map(c -> c isa Rigid ? c : with_unit(c, m), ks) : ks
end

# Every composition the chain rules admit: two or more members, distinct
# kinds, a HeightField only last. Names spell the members in application
# order (`A_T_H` is an Affine, then a ThinPlateSpline, then a HeightField).
function admitted_orders()
    inplane = (:R, :A, :T)
    out = Vector{Vector{Symbol}}()
    perms(xs, k) = k == 0 ? [Symbol[]] : [[x; rest] for x in xs for rest in perms(filter(!=(x), xs), k - 1)]
    for k in 1:3, p in perms(collect(inplane), k)
        k >= 2 && push!(out, p)
        push!(out, [p; :H])
    end
    return out
end

"""
    target_cases(P, compose; physical = false) -> Vector{Pair{String, Any}}

The four single kinds (`"R"`, `"A"`, `"T"`, `"H"`) and every admitted
composition (`"A_T_H"`, …), `compose(members)` building each from its members
in application order.
"""
function target_cases(P::AbstractDict, compose; physical::Bool = false)
    ks = kinds(P; physical)
    out = Pair{String, Any}[String(k) => getfield(ks, k) for k in (:R, :A, :T, :H)]
    for order in admitted_orders()
        push!(out, join(String.(order), "_") => compose(Tuple(getfield(ks, k) for k in order)))
    end
    return out
end

# -----------------------------------------------------------------------------
# The lenses
# -----------------------------------------------------------------------------

# OpenCV's projectPoints distortion for an n-vector (n = 4, 5, 8, 12, 14), as
# VisualGeometryToolkitIO's `opencv_lens` builds it: the radial, tangential and
# thin-prism terms summed at one point, then the tilt.
function opencv(v::AbstractVector, compose)
    n = length(v)
    k = n == 4 ? (v[1], v[2]) : (v[1], v[2], v[5])
    d = n >= 8 ? (v[6], v[7], v[8]) : ()
    terms = (Radial(k, d), Tangential(v[3], v[4]))
    n >= 12 && (terms = (terms..., ThinPrism((v[9], v[10]), (v[11], v[12]))))
    add = Additive(terms...)
    return n == 14 ? Lens(Perspective(), compose((add, Tilt(v[13], v[14])))) : Lens(Perspective(), add)
end
# Aria's Fisheye624 (COLMAP RAD_TAN_THIN_PRISM_FISHEYE), as VisualGeometryToolkitIO
# assembles it from COLMAP's 12 extra parameters: the radial factor, then the
# tangential and thin-prism deltas of the radially distorted point.
fisheye624(e::AbstractVector, compose) =
    Lens(Equidistant(), compose((Radial(Tuple(e[1:6])), Additive(Tangential(e[8], e[7]), ThinPrism((e[9], e[10]), (e[11], e[12]))))))

"""
    lens_cases(P, compose) -> Vector{Pair{String, Any}}

OpenCV's 4-, 5-, 8-, 12- and 14-coefficient lenses and COLMAP's Fisheye624,
from the stored coefficients.
"""
lens_cases(P::AbstractDict, compose) = Pair{String, Any}[
    ("opencv$n" => opencv(P["param/opencv$n"], compose) for n in (4, 5, 8, 12, 14))...,
    "fisheye624" => fisheye624(P["param/fisheye624"], compose),
]

end # module
