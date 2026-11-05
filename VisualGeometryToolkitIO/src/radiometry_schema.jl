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
# {"schema": "vgio/blur/v1", "unit": "px",
#  "blurs": [{"name": "R", "moments": [[…], […]] | {"diagonal": […]} | {"isotropic": λ}}, …]}
#      name → VGC GaussianBlur on a sensor: its second-moment matrix in px² in the storage it holds
#      (a ScaledIdentity, SDiagonal or SMatrix: the covariance shapes, `_cov_wire`).

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
        Pair{Symbol, _Spec}[:schema => _schema_tok("blur"), :unit => _Tok("px"),
            :blurs => _Vec(_Obj(:name => _NonEmpty(), :moments => _Cov(2)))],
        Pair{Symbol, _Spec}[_PROVENANCE...],
    ),
    x -> _unique_check([b[:name] for b in x[:blurs]], "blur names"),
)

# --- vignetting ---------------------------------------------------------------

_vignetting_json(v::TensorProductVignetting) = (
    type = "tensor_product", degree = collect(spline_degree(v)),
    knots_x = Float64.(collect(knot_vectors(v)[1])), knots_y = Float64.(collect(knot_vectors(v)[2])),
    control_points = [Float64[control_points(v)[i, j] for j in axes(control_points(v), 2)] for i in axes(control_points(v), 1)],
)
_vignetting_json(v::RadialVignetting) = (
    type = "radial", degree = only(spline_degree(v)), center = Float64.(collect(VgtkCore.center(v))),
    radius = Float64(field_radius(v)), control_points = Float64.(collect(vec(control_points(v)))),
)
_vignetting_json(v::LogVignetting) = (type = "log", spline = _vignetting_json(log_spline(v)))
_vignetting_json(v::AbstractVignetting) = throw(ArgumentError("save: a $(typeof(v)) has no vgio/vignetting body (tensor_product, radial, log)"))

function _vignetting_from_json(b)
    b.type == "log" && return LogVignetting(_vignetting_from_json(b.spline))
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
    body = (schema = _schema("vignetting"), pixel_frame = "colmap", vignetting = _vignetting_json(v))
    _write_store(path, _VIGNETTING_SPEC, _with_provenance(body, provenance))
    return nothing
end

"""
    load(AbstractVignetting, path) -> AbstractVignetting

Read a `vgio/vignetting/v1` file: the vignetting field [`save`](@ref) wrote.
"""
load(::Type{AbstractVignetting}, path::AbstractString) = _vignetting_from_json(_read_store(path, "vignetting", _VIGNETTING_SPEC).vignetting)

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
    has_units(light) || throw(ArgumentError(
        "save: the light's centre and radius are lengths; give them units, `with_unit(light, m)` (metres in memory)"))
    a = coefficients(light)
    body = (
        schema = _schema("environment_light"), length_unit = unit_string(length_unit),
        center = [write_length(c, length_unit) for c in VgtkCore.center(light)], radius = write_length(VgtkCore.radius(light), length_unit),
        map = (type = "spherical_harmonics", basis = _SH_BASIS, l_max = max_degree(light),
               coefficients = [Float64[a[c, j] for j in axes(a, 2)] for c in axes(a, 1)]),
    )
    _write_store(path, _ENVIRONMENT_LIGHT_SPEC, _with_provenance(body, provenance))
    return nothing
end

"""
    load(EnvironmentLight, path) -> EnvironmentLight

Read a `vgio/environment_light/v1` file: the light [`save`](@ref) wrote, its
centre and radius lengths in metres.
"""
function load(::Type{EnvironmentLight}, path::AbstractString)
    d = _read_store(path, "environment_light", _ENVIRONMENT_LIGHT_SPEC)
    u = parse_unit(d.length_unit::String)
    rows = d.map.coefficients::Vector{Any}
    a = reduce(vcat, (permutedims(Float64[Float64(x::Real) for x in r]) for r in rows))
    return EnvironmentLight(SphericalHarmonicMap(a, Int(d.map.l_max)), _lengths_from_json(Val(3), d.center::Vector{Any}, u),
        read_length(d.radius::Real, u))
end

# --- blurs ------------------------------------------------------------------------

"""
    save(path, blurs::AbstractDict{Symbol, <:GaussianBlur}; provenance = (;))

Write named Gaussian blurs on a sensor (a camera's in-focus blur per channel,
`:R`, `:G`, `:B`) as a `vgio/blur/v1` file: each one's second-moment matrix in
px², in the storage it holds (`{"isotropic": λ}` for a round one,
`{"diagonal": […]}`, or rows). [`load`](@ref) gives the same blurs back.
"""
function save(path::AbstractString, blurs::AbstractDict{Symbol, <:GaussianBlur}; provenance::NamedTuple = (;))
    body = (schema = _schema("blur"), unit = "px",
        blurs = [(name = String(k), moments = _cov_wire((x, n) -> x, moment_matrix(blurs[k]), (0, 0))) for k in sort!(collect(keys(blurs)))])
    _write_store(path, _BLUR_SPEC, _with_provenance(body, provenance))
    return nothing
end

"""
    load(Dict{Symbol, GaussianBlur}, path) -> Dict{Symbol, GaussianBlur}

Read a `vgio/blur/v1` file: name → `GaussianBlur`, each in the storage its file states.
"""
function load(::Type{Dict{Symbol, GaussianBlur}}, path::AbstractString)
    d = _read_store(path, "blur", _BLUR_SPEC)
    return Dict{Symbol, GaussianBlur}(Symbol(b.name::String) => GaussianBlur(_cov_matrix(b.moments, 2)) for b in d.blurs::Vector{Any})
end
