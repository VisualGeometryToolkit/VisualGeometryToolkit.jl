# =============================================================================
# The radiometric model's VGC values: vgio/vignetting/v1, vgio/environment_light/v1, vgio/blur/v1
# =============================================================================
#
# {"schema": "vgio/vignetting/v1", "pixel_frame": "colmap",
#  "vignetting": {"type": "tensor_product", "degree": [p, q], "knots_x": […], "knots_y": […],
#                 "control_points": [[P₁₁, …, P₁ny], …, [Pnx1, …]]}            (row i: x index i)
#              | {"type": "radial", "degree": p, "center": [cx, cy], "radius": r, "control_points": [P₁, …]}
#              | {"type": "log", "spline": {a tensor_product or radial body}}}
#      VGC's TensorProductVignetting, RadialVignetting, LogVignetting: positions in COLMAP pixels (the
#      fields' own convention), V unitless; a tensor surface's knots are stated (rebuilt exactly); a
#      radial profile's are its count's and degree's on [0, 1].
#
# {"schema": "vgio/environment_light/v1", "length_unit": "mm", "center": [x, y, z], "radius": r,
#  "map": {"type": "spherical_harmonics", "basis": "real_orthonormal_no_condon_shortley", "l_max": L,
#          "coefficients": [[a₁, …, a_J], …]}}                              (one row per channel, J = (L+1)²)
#      VGC's EnvironmentLight of a SphericalHarmonicMap: the centre and radius lengths in `length_unit`
#      (metres as quantities in memory; a plain light is refused — `with_unit(light, m)`), the
#      coefficients in the caller's radiance units; the basis convention stated, so a file is never read
#      in another one (VGC `sh_basis`: real, orthonormal, no Condon–Shortley phase, j = l² + l + m + 1).
#
# {"schema": "vgio/blur/v1",
#  "blurs": [{"name": "R", "unit": "px^2" | "mm^2" | …, "covariance": [[…], […]] | {"diagonal": […]} | {"isotropic": λ}}, …]}
#      name → VGC GaussianBlur, the density N(0, Σ): its covariance in the storage it holds (a ScaledIdentity, SDiagonal
#      or SMatrix: IO's covariance encoding, `_cov_wire`), in the unit each block states — px² on a sensor, a length
#      unit squared on a print (IO's length rule: a decimal variance exact, a computed one within an ulp) — so one
#      file may hold both. The unit is recorded, the blur's frame tag is not.
#
# Each body is a VGC value on IO's style, so another kind's record can nest it: `JSON.lower(style, v)` and
# `JSON.lift(style, AbstractVignetting, record)` (the `vignetting` body), `JSON.lift(style, EnvironmentLight, record)`
# (centre, radius in the style's length unit, map; no `schema`/`length_unit`) and `JSON.lift(style, GaussianBlur,
# record)` (one blur's `{"unit", "covariance"}`, its unit stated: px², or a print's in the style's length unit squared
# on write). The kinds' save/load go through the same methods.
#
# Every number crosses by the kind's rule: positions and coefficients as they are; lengths, and a print blur's
# variances, by IO's decimal-exact length rule — a decimal file number reads back exactly, a value computed in code
# (σ² of a σ) may move by an ulp, as every file length does (units.md, "What is exact").

const _SH_BASIS = "real_orthonormal_no_condon_shortley"

const _TENSOR_BODY = _Obj(:type => _Tok("tensor_product"), :degree => _Vec(_PosInt(), 2), :knots_x => _Vec(_Real()),
    :knots_y => _Vec(_Real()), :control_points => _Vec(_Vec(_Real())))
const _RADIAL_BODY = _Obj(:type => _Tok("radial"), :degree => _PosInt(), :center => _Vec(_Real(), 2), :radius => _Real(),
    :control_points => _Vec(_Real()))
# A body is told by its `type`; a log field holds a spline body.
struct _VignettingBody <: _Spec end
function _validate(::_VignettingBody, x, file, at)
    x isa AbstractDict && haskey(x, :type) || _refuse(file, at, "a vignetting is an object with a `type` (tensor_product, radial, log)")
    t = x[:type]
    t == "tensor_product" && return _validate(_TENSOR_BODY, x, file, at)
    t == "radial" && return _validate(_RADIAL_BODY, x, file, at)
    t == "log" && return _validate(_Obj(:type => _Tok("log"), :spline => _VignettingBody()), x, file, at)
    _refuse(file, _at(at, :type), "$(repr(t)) is not a vignetting type (tensor_product, radial, log)")
end
const _VIGNETTING_SPEC = _Obj(
    Pair{Symbol, _Spec}[:schema => _schema_tok("vignetting"), :pixel_frame => _Tok("colmap"), :vignetting => _VignettingBody()],
    Pair{Symbol, _Spec}[_PROVENANCE...],
)

const _ENVIRONMENT_LIGHT_SPEC = _Obj(
    Pair{Symbol, _Spec}[
        :schema => _schema_tok("environment_light"), :length_unit => _LENGTH_UNIT,
        :center => _Vec(_Real(), 3), :radius => _Real(),
        :map => _Checked(
            _Obj(:type => _Tok("spherical_harmonics"), :basis => _Tok(_SH_BASIS), :l_max => _Int(), :coefficients => _Vec(_Vec(_Real()))),
            m -> all(r -> length(r) == (m[:l_max] + 1)^2, m[:coefficients]) ? nothing :
                "every channel has (l_max + 1)² = $((m[:l_max] + 1)^2) coefficients",
        ),
    ],
    Pair{Symbol, _Spec}[_PROVENANCE...],
)

const _BLUR_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[:schema => _schema_tok("blur"),
            :blurs => _Vec(_Obj(:name => _NonEmpty(), :unit => _Named(s -> s isa AbstractString && _blur_unit(s) !== nothing, "px^2 or a length unit squared (mm^2, …)"),
                :covariance => _Cov(2)))],
        Pair{Symbol, _Spec}[_PROVENANCE...],
    ),
    x -> _unique_check([b[:name] for b in x[:blurs]], "blur names"),
)

# --- vignetting ---------------------------------------------------------------

# The `vignetting` body, on the style (positions in COLMAP pixels: the fields' own convention, no length).
JSON.lower(::VgioStyle, v::TensorProductVignetting) = (
    type = "tensor_product", degree = collect(spline_degree(v)),
    knots_x = Float64.(collect(knot_vectors(v)[1])), knots_y = Float64.(collect(knot_vectors(v)[2])),
    control_points = [Float64[control_points(v)[i, j] for j in axes(control_points(v), 2)] for i in axes(control_points(v), 1)],
)
JSON.lower(::VgioStyle, v::RadialVignetting) = (
    type = "radial", degree = only(spline_degree(v)), center = Float64.(collect(VgtkCore.center(v))),
    radius = Float64(field_radius(v)), control_points = Float64.(collect(vec(control_points(v)))),
)
JSON.lower(s::VgioStyle, v::LogVignetting) = (type = "log", spline = JSON.lower(s, log_spline(v)))
JSON.lower(::VgioStyle, v::AbstractVignetting) = throw(ArgumentError("save: a $(typeof(v)) has no vgio/vignetting body (tensor_product, radial, log)"))

function JSON.lift(s::VgioStyle, ::Type{AbstractVignetting}, b::_Record)
    b.type == "log" && return LogVignetting(JSON.lift(s, AbstractVignetting, b.spline::_Record))
    if b.type == "tensor_product"
        P = reduce(vcat, (permutedims(Float64[Float64(x::Real) for x in r]) for r in b.control_points::Vector{Any}))
        return TensorProductVignetting(Float64[Float64(x::Real) for x in b.knots_x], Float64[Float64(x::Real) for x in b.knots_y], P;
            degree = (Int(b.degree[1]), Int(b.degree[2])))
    end
    P = Float64[Float64(x::Real) for x in b.control_points]
    v = RadialVignetting(length(P), SVector(Float64(b.center[1]), Float64(b.center[2])), Float64(b.radius); degree = Int(b.degree))
    return with_control_points(v, P)
end

"""
    save(path, v::AbstractVignetting; provenance = (;))

Write a VGC vignetting field — a `TensorProductVignetting`, `RadialVignetting`
or `LogVignetting` — as a `vgio/vignetting/v1` file: its type, degree, knots
(a tensor surface's, stated) or centre and radius (a radial profile's), and
control points, positions in COLMAP pixels. [`load`](@ref) gives the same field
back, bit for bit.
"""
function save(path::AbstractString, v::AbstractVignetting; provenance::NamedTuple = (;))
    s = VgioStyle()   # (no length declared)
    body = (schema = _schema("vignetting"), pixel_frame = "colmap", vignetting = JSON.lower(s, v))
    _write_store(path, _VIGNETTING_SPEC, _with_provenance(body, provenance); style = s)
    return nothing
end

"""
    load(AbstractVignetting, path) -> AbstractVignetting

Read a `vgio/vignetting/v1` file: the vignetting field [`save`](@ref) wrote.
"""
load(::Type{AbstractVignetting}, path::AbstractString) =
    JSON.lift(VgioStyle(), AbstractVignetting, _read_store(path, "vignetting", _VIGNETTING_SPEC).vignetting::_Record)

# --- the environment light ------------------------------------------------------

"""
    save(path, light::EnvironmentLight; length_unit = mm, provenance = (;))

Write a VGC `EnvironmentLight` of a `SphericalHarmonicMap` as a
`vgio/environment_light/v1` file: its centre and radius in `length_unit`
(decimal-exact; a plain light — no units — is an `ArgumentError` naming
`with_unit(light, m)`), and its coefficients per channel with the basis
convention stated. [`load`](@ref) gives the same light back.
"""
function save(path::AbstractString, light::EnvironmentLight{<:SphericalHarmonicMap}; length_unit::Units = mm, provenance::NamedTuple = (;))
    s = VgioStyle(; length_unit)
    body = merge((schema = _schema("environment_light"), length_unit = unit_string(length_unit)), JSON.lower(s, light))
    _write_store(path, _ENVIRONMENT_LIGHT_SPEC, _with_provenance(body, provenance); style = s)
    return nothing
end

# The light's body (no `schema`, no `length_unit`: the enclosing record's): its centre and radius in the style's
# length unit, decimal-exact; a plain light is refused, naming `with_unit(light, m)`.
function JSON.lower(s::VgioStyle, light::EnvironmentLight{<:SphericalHarmonicMap})
    has_units(light) || throw(ArgumentError(
        "save: the light's centre and radius are lengths; give them units, `with_unit(light, m)` (metres in memory)"))
    a = coefficients(light)
    return (
        center = [_leaf_length(c, s.length_unit) for c in VgtkCore.center(light)], radius = _leaf_length(VgtkCore.radius(light), s.length_unit),
        map = (type = "spherical_harmonics", basis = _SH_BASIS, l_max = max_degree(light),
               coefficients = [Float64[a[c, j] for j in axes(a, 2)] for c in axes(a, 1)]),
    )
end
function JSON.lift(s::VgioStyle{<:Units}, ::Type{EnvironmentLight}, d::_Record)
    rows = d.map.coefficients::Vector{Any}
    a = reduce(vcat, (permutedims(Float64[Float64(x::Real) for x in r]) for r in rows))
    return EnvironmentLight(SphericalHarmonicMap(a, Int(d.map.l_max)), _lengths_from_json(Val(3), d.center::Vector{Any}, s.length_unit),
        read_length(d.radius::Real, s.length_unit))
end

"""
    load(EnvironmentLight, path) -> EnvironmentLight

Read a `vgio/environment_light/v1` file: the light [`save`](@ref) wrote, its
centre and radius lengths in metres.
"""
function load(::Type{EnvironmentLight}, path::AbstractString)
    d = _read_store(path, "environment_light", _ENVIRONMENT_LIGHT_SPEC)
    return JSON.lift(VgioStyle(; length_unit = parse_unit(d.length_unit::String)), EnvironmentLight, d::_Record)
end

# --- blurs ------------------------------------------------------------------------

"""
    save(path, blurs::AbstractDict{Symbol, <:GaussianBlur}; length_unit = mm, provenance = (;))

Write named Gaussian blurs (a camera's in-focus blur per channel, `:R`, `:G`,
`:B`; a print's blur) as a `vgio/blur/v1` file: each one's covariance in the
storage it holds (`{"isotropic": λ}` for a round one, `{"diagonal": […]}`, or
rows) and the unit it states, `"px^2"` for a blur on a sensor (plain, or px
quantities) and `length_unit` squared (`"mm^2"`) for one in lengths (m²
quantities). A plain blur labeled with a frame that is not a pixel frame is an
`ArgumentError` (its lengths need units). The file records the unit, not the
blur's frame tag. [`load`](@ref) gives the blurs back in their storage and
unit, each entry as the file's length rule reads it: a decimal variance exactly,
a variance computed in code (σ² of a σ) to within an ulp, as for every file
length (units.md).
"""
function save(path::AbstractString, blurs::AbstractDict{Symbol, <:GaussianBlur}; length_unit::Units = mm, provenance::NamedTuple = (;))
    s = VgioStyle(; length_unit)
    body = (schema = _schema("blur"), blurs = [merge((name = String(k),), JSON.lower(s, blurs[k])) for k in sort!(collect(keys(blurs)))])
    _write_store(path, _BLUR_SPEC, _with_provenance(body, provenance); style = s)
    return nothing
end

"""
    load(Dict{Symbol, GaussianBlur}, path) -> Dict{Symbol, GaussianBlur}

Read a `vgio/blur/v1` file: name → `GaussianBlur`, each in the storage and the unit its block states.
"""
function load(::Type{Dict{Symbol, GaussianBlur}}, path::AbstractString)
    d = _read_store(path, "blur", _BLUR_SPEC)
    s = VgioStyle()
    return Dict{Symbol, GaussianBlur}(Symbol(b.name::String) => JSON.lift(s, GaussianBlur, b::_Record) for b in d.blurs::Vector{Any})
end

# One blur's block, `{"unit": …, "covariance": …}`: the covariance in the storage it holds, in the unit the block
# states. `"px^2"` is a blur on a sensor: plain numbers (or px quantities, stripped) as they are. `"<length>^2"` is a
# print's: m² quantities in memory, each entry in the length unit squared through IO's covariance length rule
# (`_length_cov_json` / `_length_cov_from_json` at length power 2: the decimal shift, never a multiplication by the
# scale). The lift reads the stated unit — px² by its name, an area as a registered length unit squared — and the
# record states the unit only, never the blur's frame tag (a blur is read back unlabeled).
const _PX2_NAME = "px^2"
function JSON.lower(s::VgioStyle{<:Units}, b::GaussianBlur)
    Σ = cov(b)
    if !has_units(b)
        Σ isa CovMat{2, <:Any, <:Any, <:Union{UnlabeledFrame{2}, PixelFrame, ImageFrame}} || throw(ArgumentError(
            "save: a plain blur is in px²; one labeled with a frame that is not a pixel frame holds lengths — give it units, `with_unit(b, m)`"))
        return (unit = _PX2_NAME, covariance = _cov_json(Σ))
    end
    d = dimension(Σ[1, 1])
    d == dimension(px) && return (unit = _PX2_NAME, covariance = _cov_json(cov(ustrip(px, b))))   # px quantities
    d == dimension(m^2) || throw(ArgumentError("save: a blur's covariance is px² or a length squared, got the dimension $d"))
    return (unit = string(unit_string(s.length_unit), "^2"), covariance = _length_cov_json(cov(ustrip(m, b)), s.length_unit, (1, 1)))
end
function JSON.lift(::VgioStyle, ::Type{GaussianBlur}, b::_Record)
    u = _blur_unit(b.unit::String)
    u === nothing && throw(ArgumentError("a blur's unit is px^2 or a length unit squared (mm^2, …), got $(repr(b.unit))"))
    u isa _PxSquared && return GaussianBlur(_cov_matrix(b.covariance, 2))
    return with_unit(GaussianBlur(_length_cov_from_json(b.covariance, 2, u, (1, 1))), m)
end
# The unit a blur block states, if it is one: `_PxSquared()` for exactly "px^2", the length unit of "<length>^2", or nothing.
struct _PxSquared end
# Dimensionless units are told by name, not by `==`: FlexUnits equates every dimensionless unit of scale 1 (px, px³, pd²).
function _blur_unit(s::AbstractString)
    s == _PX2_NAME && return _PxSquared()
    c = match(r"^(.+)\^2$", s)
    c === nothing && return nothing
    u = try
        parse_unit(c.captures[1]::AbstractString)
    catch e
        e isa ArgumentError || rethrow()
        return nothing
    end
    return dimension(u) == dimension(m) ? u : nothing
end
