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
# What differs between the code the goldens were written on and the code
# that reads them is spelled in ONE place, an ADAPTER (a NamedTuple) that each
# reader supplies — see `adapter fields` below. Everything else here is
# independent of how a calibration, a composition or a Jacobian is spelled.
#
# Adapter fields:
#   affine(A, b)                  the in-plane affine print error from A (2 × 2) and b
#   compose_calibrations(members) their composition, members IN APPLICATION ORDER
#                                 (innermost first)
#   compose_distortions(stages)   the same for a lens's distortion stages
#   jacobian_theta(c, x)          a calibration's parameter Jacobian at x
#   jacobian_x(c, x)              its point Jacobian at x
#   ct_order, lens_ct_order       whether a calibration's / a lens's composed
#                                 parameters are in CoordinateTransformations'
#                                 order (outer map first) rather than the
#                                 goldens' application order
#   retired                       the target cases the code refuses (names)
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
using VisualGeometryToolkitCore: Rigid, ThinPlateSpline, HeightField, Lens, Perspective, Equidistant,
    Radial, Tangential, ThinPrism, Tilt, Additive, grid_nodes, with_unit, m

export golden_line, read_goldens, golden_value, target_cases, lens_cases, design_points, physical_points,
    rays, image_points, jacobian_count, KIND_LETTERS, case_members, target_blocks, lens_blocks, block_permutation, ulps,
    expected_core_value, compare_value, scaled_comparison, MAX_ULPS

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
function kinds(P::AbstractDict, V; physical::Bool = false)
    nodes = grid_nodes(3; lo = 0.0, hi = 0.1)
    A = V.affine(SMatrix{2, 2}(P["param/affine/A"]), P["param/affine/b"])
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
    target_cases(P, V; physical = false) -> Vector{Pair{String, Any}}

The four single kinds (`"R"`, `"A"`, `"T"`, `"H"`) and every composition the
chain rules of the goldens' code admitted (`"A_T_H"`, …), the adapter's
`compose_calibrations(members)` building each from its members in application
order.
"""
function target_cases(P::AbstractDict, V; physical::Bool = false)
    ks = kinds(P, V; physical)
    out = Pair{String, Any}[String(k) => getfield(ks, k) for k in (:R, :A, :T, :H)]
    for order in admitted_orders()
        push!(out, join(String.(order), "_") => V.compose_calibrations(Tuple(getfield(ks, k) for k in order)))
    end
    return out
end

# A case's members in application order (`"A_H"` → `[:A, :H]`).
case_members(name::AbstractString) = Symbol.(split(name, '_'))

# Per kind: (parameters, gauge directions, regularizer rows) — the blocks a
# composition's vectors stack, one per member.
const _KIND_BLOCKS = (R = (0, 0, 0), A = (6, 4, 0), T = (18, 6, 18), H = (9, 3, 9))
target_blocks(name::AbstractString, which::Int) = [_KIND_BLOCKS[k][which] for k in case_members(name)]

"""
    block_permutation(sizes, reversed) -> Vector{Int}

The indices of a vector stacked from blocks of `sizes` (in application order)
in the order the code under test stacks them: the same order, or the blocks
reversed (CoordinateTransformations' order, the outer map first). `new = old[p]`.
"""
function block_permutation(sizes::AbstractVector{Int}, reversed::Bool)
    offsets = cumsum([0; sizes[1:(end - 1)]])
    blocks = [collect((o + 1):(o + n)) for (o, n) in zip(offsets, sizes)]
    return reduce(vcat, reversed ? reverse(blocks) : blocks; init = Int[])
end

"""
    ulps(a, b) -> Int128

The distance between two Float64s in units in the last place (0 for equal
values, `-0.0` and `0.0` included, and for two NaNs; `typemax` when one is NaN
and the other is not).
"""
function ulps(a::Float64, b::Float64)
    (isnan(a) || isnan(b)) && return isnan(a) && isnan(b) ? Int128(0) : typemax(Int128)
    o(x) = (i = Int128(reinterpret(Int64, x)); i < 0 ? -(i - Int128(typemin(Int64))) : i)
    return abs(o(a) - o(b))
end

# -----------------------------------------------------------------------------
# What the code at hand must reproduce of a golden
# -----------------------------------------------------------------------------

const MAX_ULPS = 4
const _TARGET_PARAMETER_FIELDS = ("names", "theta", "bounds_lo", "bounds_hi", "length_mask", "with_parameters/theta",
    "with_unit_mm/theta", "ustrip_mm/theta", "manifold/retract")
const _LENS_PARAMETER_FIELDS = ("names", "theta", "bounds_lo", "bounds_hi", "float32/theta", "with_parameters/theta", "manifold/retract")

"""
    expected_core_value(key, golden, V) -> value | :absent

The golden value of `key` as the code of adapter `V` must reproduce it: a
composition's parameter-indexed vectors, Jacobian columns, gauge columns and
regularizer rows permuted from the goldens' application order to the code's
(`block_permutation`); the calibration tag of a composition `:absent` when the
code has none (`V.composition_tags == false`); every other value as stored.
"""
function expected_core_value(key::AbstractString, g, V)
    parts = split(key, '/'; limit = 3)
    family, name, field = parts[1], parts[2], parts[3]
    if family == "target"
        composed = length(case_members(name)) > 1
        field == "tag" && composed && !V.composition_tags && return :absent
        p = block_permutation(target_blocks(name, 1), V.ct_order)
        q = block_permutation(target_blocks(name, 2), V.ct_order)
        r = block_permutation(target_blocks(name, 3), V.ct_order)
        field in _TARGET_PARAMETER_FIELDS && return g[p]
        field == "gauge_names" && return g[q]
        field == "gauge_directions" && return g[p, q]
        field in ("jacobian_theta", "manifold/action") && return g[:, p, :]
        field == "regularizer" && return g[r]
        return g
    end
    blocks = lens_blocks(name)
    blocks === nothing && return g
    p = block_permutation(blocks, V.lens_ct_order)
    field in _LENS_PARAMETER_FIELDS && return g[p]
    field == "jacobian_theta" && return g[:, p, :]
    return g
end

"""
    scaled_comparison(key, V) -> Bool

Whether a golden is compared within `MAX_ULPS` ulps of its array's largest
entry rather than entry by entry: a composition's target-manifold action and
retraction when the code stacks its parameters in CT's order. Both run through
the manifold's basis `B`, an SVD of the gauge directions in θ's order; permuting
θ permutes that SVD's input, which moves the rounding of `B` (by ~1e-16 of
its scale: an entry that is zero to rounding then has no ulp distance).
"""
function scaled_comparison(key::AbstractString, V)
    parts = split(key, '/'; limit = 3)
    return parts[1] == "target" && V.ct_order && length(case_members(parts[2])) > 1 &&
        parts[3] in ("manifold/action", "manifold/retract")
end

"""
    compare_value(expected, actual; scaled = false) -> (ok::Bool, ulps::Int128, why::String)

Names, texts and flags exactly; Float64 arrays of the same shape within
`MAX_ULPS` entry by entry (`ulps` the largest distance found, 0 when bit for
bit identical). With `scaled`, within `MAX_ULPS` ulps of the array's largest
entry instead (`scaled_comparison`): for values whose rounding moves with the
order of an SVD's input, where an entry that is zero to rounding has no ulp.
"""
function compare_value(e, a; scaled::Bool = false)
    if e isa AbstractArray{Float64} || e isa Float64
        a isa Union{AbstractArray{<:Real}, Real} || return (false, Int128(0), "a $(typeof(a)) where Float64s are pinned")
        size(e) == size(a) || return (false, Int128(0), "size $(size(a)), the golden's $(size(e))")
        d = isempty(e) ? Int128(0) : maximum(ulps.(Float64.(e), Float64.(a)))
        if scaled && !isempty(e)
            δ, s = maximum(abs.(Float64.(e) .- Float64.(a))), maximum(abs, e)
            ok = δ <= MAX_ULPS * eps(s)
            return (ok, d, ok ? "" : "off by $(δ), $(δ / eps(s)) ulps of the array's largest entry $(s)")
        end
        return (d <= MAX_ULPS, d, d <= MAX_ULPS ? "" : "$(d) ulps")
    end
    ea = a isa AbstractArray ? collect(a) : a
    return (isequal(e, ea), Int128(0), isequal(e, ea) ? "" : "$(repr(ea)), the golden's $(repr(e))")
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
    lens_cases(P, V) -> Vector{Pair{String, Any}}

OpenCV's 4-, 5-, 8-, 12- and 14-coefficient lenses and COLMAP's Fisheye624,
from the stored coefficients.
"""
lens_cases(P::AbstractDict, V) = Pair{String, Any}[
    ("opencv$n" => opencv(P["param/opencv$n"], V.compose_distortions) for n in (4, 5, 8, 12, 14))...,
    "fisheye624" => fisheye624(P["param/fisheye624"], V.compose_distortions),
]

# The parameter blocks of a lens case's composed distortion, per stage in
# application order (the projections here have none); `nothing` when the
# distortion is not composed.
lens_blocks(name::AbstractString) = name == "opencv14" ? [12, 2] : name == "fisheye624" ? [6, 6] : nothing

end # module
