# =============================================================================
# The radiometric model's VGC values: vgio/vignetting/v2.0, vgio/environment_light/v2.0, vgio/blur/v2.0
# =============================================================================
#
# Each file opens with the envelope (validate.jl: `units`, `conventions`, `generator`).
#
# {"schema": "vgio/vignetting/v2.0", <envelope>,
#  "vignetting": {"type": "tensor_product", "degree": [p, q], "knots_x": […], "knots_y": […],
#                 "control_points": [[P₁₁, …, P₁ny], …, [Pnx1, …]]}            (row i: x index i)
#              | {"type": "radial", "degree": p, "center": [cx, cy], "radius": r, "chart": "linear" | "projective",
#                 "control_points": [P₁, …]}
#              | {"type": "log", "spline": {a tensor_product or radial body}}}
#      VGC's TensorProductVignetting, RadialVignetting, LogVignetting: positions in pixels of the file's
#      `units.pixel` (any registered pixel frame; VGC's fields hold COLMAP pixels, so a file in another
#      frame moves them by its offset on the way in), V unitless; a tensor surface's knots are stated
#      (rebuilt exactly); a radial profile's are its count's, degree's and chart's (VGC's `RADIAL_CHARTS`,
#      always stated: a declaration has no default).
#
# {"schema": "vgio/environment_light/v2.0", <envelope>, "center": [x, y, z], "radius": r,
#  "map": {"type": "spherical_harmonics", "basis": "real_orthonormal_no_condon_shortley", "max_degree": L,
#          "coefficients": [[a₁, …, a_J], …]}}                              (one row per channel, J = (L+1)²)
#      VGC's EnvironmentLight of a SphericalHarmonicMap: the centre and radius lengths in `units.length`
#      (metres as quantities in memory; a plain light is refused — `with_unit(light, m)`), the
#      coefficients in `units.radiance` ("relative": known up to scale); the basis convention stated, so a
#      file is never read in another one (VGC `sh_basis`: real, orthonormal, no Condon–Shortley phase,
#      j = l² + l + m + 1). `max_degree` is Core's name (`max_degree(light)`).
#
# {"schema": "vgio/blur/v2.0", <envelope>,
#  "blurs": [{"name": "R", "pixel_frame": "colmap", "covariance": {"isotropic": λ}},           a blur on a sensor (px²)
#            {"name": "print", "frame": "target_plane", "covariance": [[…], […]]}, …]}       a blur on a print (units.length²)
#      name → VGC GaussianBlur, the density N(0, Σ), its covariance in the storage it holds (a ScaledIdentity,
#      SDiagonal or SMatrix: IO's covariance encoding) and its FRAME, the plane it lives on, which its `CovMat`
#      carries and reads back with: a pixel frame (`pixel_frame`: PixelFrame{O}, or `frame` "image:<id>",
#      ImageFrame{id}) — px², as stored — or a target's plane (`frame` "target_plane", TargetPlaneFrame) — a
#      length squared, `units.length`² on disk (IO's length rule: a decimal variance exact, a computed one
#      within an ulp), m² quantities in memory. The unit is the field's power of a declared base, never a
#      per-blur string (`blur/v1` wrote one, and no frame; no published file holds a v1 file, which is refused).
#
# Each body is a VGC value on IO's style, so another kind's record can nest it: `JSON.lower(style, v)` and
# `JSON.lift(style, AbstractVignetting, record)` (the `vignetting` body), `JSON.lift(style, EnvironmentLight, record)`
# (centre, radius in the style's length unit, map) and `JSON.lift(style, GaussianBlur, record)` (one blur's
# `{"pixel_frame" | "frame", "covariance"}`; an IO 0.9 body, `{"unit", "covariance"}`, as another package's record of
# an earlier version embeds it, reads as the unlabeled blur it held). The kinds' save/load go through the same methods.
#
# Every number crosses by the kind's rule: positions and coefficients as they are; lengths, and a print blur's
# variances, by IO's decimal-exact length rule — a decimal file number reads back exactly, a value computed in code
# (σ² of a σ) may move by an ulp, as every file length does (units.md, "What is exact").

const _SH_BASIS = "real_orthonormal_no_condon_shortley"

# A body is told by its `type`, the field's tag in VGC's `VIGNETTING_TAGS`; a log field holds a spline body.
struct _VignettingBody <: _Spec end
_vignetting_type(T::Type{<:AbstractVignetting}) = _Tok(tag_of(VIGNETTING_TAGS, T))
_vignetting_body(T::Type{<:TensorProductVignetting}) = _Obj(:type => _vignetting_type(T), :degree => _Vec(_PosInt(), 2),
    :knots_x => _Vec(_Real()), :knots_y => _Vec(_Real()), :control_points => _Vec(_Vec(_Real())))
_vignetting_body(T::Type{<:RadialVignetting}) = _Obj(
    Pair{Symbol, _Spec}[:type => _vignetting_type(T), :degree => _PosInt(), :center => _Vec(_Real(), 2), :radius => _Real(),
                        :chart => _Named(c -> c in String.(RADIAL_CHARTS), "a radial chart ($(join(String.(RADIAL_CHARTS), ", ")))"),
                        :control_points => _Vec(_Real())],
    Pair{Symbol, _Spec}[])
_vignetting_body(T::Type{<:LogVignetting}) = _Obj(:type => _vignetting_type(T), :spline => _VignettingBody())
_vignetting_tags() = join(keys(VIGNETTING_TAGS), ", ")
function _validate(::_VignettingBody, x, file, at)
    x isa AbstractDict && haskey(x, :type) || _refuse(file, at, "a vignetting is an object with a `type` ($(_vignetting_tags()))")
    t = x[:type]
    t isa AbstractString || _refuse(file, _at(at, :type), "$(repr(t)) is not a vignetting type ($(_vignetting_tags()))")
    # the registry's lookup: an unknown tag lists the known ones, a retired one names its successor
    T = try
        VIGNETTING_TAGS[t]
    catch err
        err isa ArgumentError || rethrow()
        _refuse(file, _at(at, :type), err.msg)
    end
    return _validate(_vignetting_body(T), x, file, at)
end
const _VIGNETTING_SPEC = _kind_spec("vignetting", Pair{Symbol, _Spec}[:vignetting => _VignettingBody()])

const _ENVIRONMENT_LIGHT_SPEC = _kind_spec(
    "environment_light",
    Pair{Symbol, _Spec}[
        :center => _Vec(_Real(), 3), :radius => _Real(),
        :map => _CheckedSpec(
            _Obj(:type => _Tok("spherical_harmonics"), :basis => _Tok(_SH_BASIS), :max_degree => _Int(), :coefficients => _Vec(_Vec(_Real()))),
            m -> all(r -> length(r) == (m[:max_degree] + 1)^2, m[:coefficients]) ? nothing :
                "every channel has (max_degree + 1)² = $((m[:max_degree] + 1)^2) coefficients",
        ),
    ],
)

# A blur's plane: a pixel frame (`pixel_frame`), or `frame` — a view's image frame
# (`image:<id>`) or a target's plane (`target_plane`).
const _TARGET_PLANE_NAME = "target_plane"
_blur_frame_ok(s) = s == _TARGET_PLANE_NAME || (startswith(s, "image:") && length(s) > 6)
const _BLUR_RECORD = _Either(Pair{Symbol, _Spec}[
    :pixel_frame => _Obj(:name => _NonEmpty(), :pixel_frame => _PIXEL_FRAME, :covariance => _Cov(2)),
    :frame => _Obj(:name => _NonEmpty(), :frame => _Named(_blur_frame_ok, "a blur's plane (\"target_plane\", \"image:<id>\")"), :covariance => _Cov(2)),
])
const _BLUR_SPEC = _kind_spec("blur", Pair{Symbol, _Spec}[:blurs => _Vec(_BLUR_RECORD)];
                              check = x -> _unique_check([b[:name] for b in x[:blurs]], "blur names"))

# --- vignetting ---------------------------------------------------------------

# A matrix stored as its rows (a tensor surface's control points, a light's coefficients per channel, a panorama's gains
# per frame), read back as a Matrix{Float64}.
_rows_matrix(rows) = reduce(vcat, (permutedims(Float64[Float64(x::Real) for x in r]) for r in rows))
# …and written: its rows, each a Vector{Float64}.
_matrix_rows(A::AbstractMatrix) = [Float64[A[i, j] for j in axes(A, 2)] for i in axes(A, 1)]

# The `vignetting` body, on the style: positions in the style's pixel frame (VGC's fields hold COLMAP pixels), no length.
function JSON.lower(s::VgioStyle, v::TensorProductVignetting)
    c = _vignetting_offset(s, CANONICAL_PIXEL_FRAME, _pixel_frame(s))
    return (
        type = tag_of(VIGNETTING_TAGS, v), degree = collect(spline_degree(v)),
        knots_x = Float64.(collect(knot_vectors(v)[1])) .+ c[1], knots_y = Float64.(collect(knot_vectors(v)[2])) .+ c[2],
        control_points = _matrix_rows(control_points(v)),
    )
end
# A radial profile states its chart always (a declaration has no default).
function JSON.lower(s::VgioStyle, v::RadialVignetting)
    c = _vignetting_offset(s, CANONICAL_PIXEL_FRAME, _pixel_frame(s))
    return (type = tag_of(VIGNETTING_TAGS, v), degree = only(spline_degree(v)), center = Float64.(collect(VgtkCore.center(v)) .+ c),
            radius = Float64(field_radius(v)), chart = String(radial_chart(v)), control_points = Float64.(collect(vec(control_points(v)))))
end
# The offset of a position written in the pixel frame `G` for one held in `F`
# (frames differ by a translation): `x_G = x_F + offset`.
_vignetting_offset(::VgioStyle, F::Type{<:PixelFrame}, G::Type{<:PixelFrame}) = (o = EuclideanMap(F, G)(SVector(0.0, 0.0)); SVector(o[1], o[2]))
JSON.lower(s::VgioStyle, v::LogVignetting) = (type = tag_of(VIGNETTING_TAGS, v), spline = JSON.lower(s, log_spline(v)))
JSON.lower(::VgioStyle, v::AbstractVignetting) = throw(ArgumentError("save: a $(typeof(v)) has no vgio/vignetting body ($(_vignetting_tags()))"))

JSON.lift(s::VgioStyle, ::Type{AbstractVignetting}, b::Record) = _lift_vignetting(s, VIGNETTING_TAGS[b.type::String], b)
_lift_vignetting(s::VgioStyle, ::Type{<:LogVignetting}, b::Record) = LogVignetting(JSON.lift(s, AbstractVignetting, b.spline::Record))
function _lift_vignetting(s::VgioStyle, ::Type{<:TensorProductVignetting}, b::Record)
    c = _vignetting_offset(s, _pixel_frame(s), CANONICAL_PIXEL_FRAME)
    P = _rows_matrix(b.control_points::Vector{Any})
    return TensorProductVignetting(Float64[Float64(x::Real) + c[1] for x in b.knots_x], Float64[Float64(x::Real) + c[2] for x in b.knots_y], P;
        degree = (Int(b.degree[1]), Int(b.degree[2])))
end
function _lift_vignetting(s::VgioStyle, ::Type{<:RadialVignetting}, b::Record)
    c = _vignetting_offset(s, _pixel_frame(s), CANONICAL_PIXEL_FRAME)
    P = Float64[Float64(x::Real) for x in b.control_points]
    chart = Symbol(b.chart::String)
    v = RadialVignetting(length(P), SVector(Float64(b.center[1]) + c[1], Float64(b.center[2]) + c[2]), Float64(b.radius); degree = Int(b.degree), chart)
    return with_control_points(v, P)
end

"""
    save(path, v::AbstractVignetting; provenance = (;))

Write a VGC vignetting field — a `TensorProductVignetting`, `RadialVignetting`
or `LogVignetting` — as a `vgio/vignetting/v2.0` file: its type, degree, knots
(a tensor surface's, stated) or centre, radius and chart (a radial profile's), and
control points, positions in COLMAP pixels (the file's `units.pixel`).
[`load`](@ref) gives the same field back, bit for bit.
"""
function save(path::AbstractString, v::AbstractVignetting; provenance::NamedTuple = (;))
    s = VgioStyle()
    _write_kind(path, "vignetting", (vignetting = JSON.lower(s, v),), provenance; style = s)
    return nothing
end

"""
    load(AbstractVignetting, path) -> AbstractVignetting

Read a `vgio/vignetting/v2.0` file: the vignetting field [`save`](@ref) wrote,
its positions moved from the file's `units.pixel` into COLMAP pixels (VGC's
fields' own). A v1 file (with or without its `chart`, which IO 0.9 added inside
v1) is read through its version step.
"""
function load(::Type{AbstractVignetting}, path::AbstractString)
    d = _read_store(path, "vignetting")
    return JSON.lift(VgioStyle(d), AbstractVignetting, d.vignetting::Record)
end

# --- the environment light ------------------------------------------------------

"""
    save(path, light::EnvironmentLight; length_unit = mm, provenance = (;))

Write a VGC `EnvironmentLight` of a `SphericalHarmonicMap` as a
`vgio/environment_light/v2.0` file: its centre and radius in `length_unit`
(decimal-exact; a plain light — no units — is an `ArgumentError` naming
`with_unit(light, m)`), and its coefficients per channel with the basis
convention stated, in `units.radiance` ("relative"). [`load`](@ref) gives the
same light back.
"""
function save(path::AbstractString, light::EnvironmentLight{<:SphericalHarmonicMap}; length_unit::Units = mm, provenance::NamedTuple = (;))
    s = VgioStyle(; length_unit)
    _write_kind(path, "environment_light", JSON.lower(s, light), provenance; style = s)
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
        map = (type = "spherical_harmonics", basis = _SH_BASIS, max_degree = max_degree(light),
               coefficients = _matrix_rows(a)),
    )
end
function JSON.lift(s::VgioStyle{<:Units}, ::Type{EnvironmentLight}, d::Record)
    a = _rows_matrix(d.map.coefficients::Vector{Any})
    return EnvironmentLight(SphericalHarmonicMap(a, Int(d.map.max_degree)), _lengths_from_json(Val(3), d.center::Vector{Any}, s.length_unit),
        read_length(d.radius::Real, s.length_unit))
end

"""
    load(EnvironmentLight, path) -> EnvironmentLight

Read a `vgio/environment_light/v2.0` file: the light [`save`](@ref) wrote, its
centre and radius lengths in metres. A v1 file is read through its version step.
"""
function load(::Type{EnvironmentLight}, path::AbstractString)
    d = _read_store(path, "environment_light")
    return JSON.lift(VgioStyle(d), EnvironmentLight, d::Record)
end

# --- blurs ------------------------------------------------------------------------

"""
    save(path, blurs::AbstractDict{Symbol, <:GaussianBlur}; length_unit = mm, provenance = (;))

Write named Gaussian blurs (a camera's in-focus blur per channel, `:R`, `:G`,
`:B`; a print's blur) as a `vgio/blur/v2.0` file: each one's covariance in the
storage it holds (`{"isotropic": λ}` for a round one, `{"diagonal": […]}`, or
rows) and the plane it lives on, its covariance's frame — a pixel frame
(`PixelFrame{O}`, an `ImageFrame{id}`: px², plain numbers or px quantities) or
a target's plane (`TargetPlaneFrame`: m² quantities, written in `length_unit`²).
An unlabeled blur is an `ArgumentError` (name the plane it lives on: `GaussianBlur(CovMat{2, …, F}(Σ))`),
and so is a plain one on a target's plane (its lengths need units).
[`load`](@ref) gives the blurs back with their frames, in their storage, each
entry as the file's length rule reads it: a decimal variance exactly, a
variance computed in code (σ² of a σ) to within an ulp, as for every file length
(units.md).
"""
function save(path::AbstractString, blurs::AbstractDict{Symbol, <:GaussianBlur}; length_unit::Units = mm, provenance::NamedTuple = (;))
    s = VgioStyle(; length_unit)
    _write_kind(path, "blur", (blurs = JSON.lower(s, blurs),), provenance; style = s)
    return nothing
end

"""
    load(Dict{Symbol, GaussianBlur}, path) -> Dict{Symbol, GaussianBlur}

Read a `vgio/blur/v2.0` file: name → `GaussianBlur`, each in the storage its
block states, labeled with its plane (a pixel frame's px², a target plane's m²
quantities).
"""
function load(::Type{Dict{Symbol, GaussianBlur}}, path::AbstractString)
    d = _read_store(path, "blur")
    return JSON.lift(VgioStyle(d), Dict{Symbol, GaussianBlur}, d.blurs)
end

# Named blurs as a list body, `[{"name": …, <plane>, "covariance": …}, …]`: the `blurs` of a `vgio/blur` file, and
# the same body nested in another package's record (a camera's per-channel in-focus blur), which lowers and lifts it
# through these two methods so the body is written once. One block per blur, in name order (`sort` of the symbols).
# The lift refuses a repeated name (a file's validator does too; a nested body has no validator).
JSON.lower(s::VgioStyle{<:Units}, blurs::AbstractDict{Symbol, <:GaussianBlur}) =
    [merge((name = String(k),), JSON.lower(s, blurs[k])) for k in sort!(collect(keys(blurs)))]
function JSON.lift(s::VgioStyle, ::Type{Dict{Symbol, GaussianBlur}}, body::Vector)   # a parsed list (not JSON's lazy one)
    blurs = Dict{Symbol, GaussianBlur}()
    for b in body
        k = Symbol(b.name::String)
        haskey(blurs, k) && throw(ArgumentError("blurs: the name $(repr(String(k))) is repeated"))
        blurs[k] = JSON.lift(s, GaussianBlur, b::Record)
    end
    return blurs
end

# One blur's block: its plane and its covariance in the storage it holds. A blur on a pixel frame (or a view's image
# frame) is px² — plain numbers, or px quantities stripped — as it is; a blur on a target's plane is m² quantities in
# memory, each entry in the style's length unit squared through IO's covariance length rule (`_length_cov_json` at
# length power 2: the decimal shift, never a multiplication by the scale).
_blur_frame(::Type{PixelFrame{O}}) where {O} = (pixel_frame = String(O),)
_blur_frame(::Type{ImageFrame{id}}) where {id} = (frame = "image:$(id)",)
_blur_frame(::Type{VgtkCore.TargetPlaneFrame}) = (frame = _TARGET_PLANE_NAME,)
_blur_frame(::Type{F}) where {F} = throw(ArgumentError(
    "save: a blur names the plane it lives on — a pixel frame, an ImageFrame{id} or the TargetPlaneFrame — through its covariance's frame; got $(F)"))
_cov_frame(::CovMat{N, T, S, F}) where {N, T, S, F} = F
function JSON.lower(s::VgioStyle{<:Units}, b::GaussianBlur)
    Σ = cov(b)
    F = _cov_frame(Σ)
    plane = _blur_frame(F)
    if F === VgtkCore.TargetPlaneFrame
        has_units(b) || throw(ArgumentError("save: a blur on a target's plane holds lengths — give it units, `with_unit(b, m)`"))
        d = dimension(Σ[1, 1])
        d == dimension(m^2) || throw(ArgumentError("save: a print blur's covariance is a length squared, got the dimension $d"))
        return merge(plane, (covariance = _length_cov_json(cov(ustrip(m, b)), s.length_unit, (1, 1)),))
    end
    has_units(b) || return merge(plane, (covariance = _cov_json(Σ),))
    dimension(Σ[1, 1]) == dimension(px) || throw(ArgumentError("save: a blur on a pixel plane is px², got the dimension $(dimension(Σ[1, 1]))"))
    return merge(plane, (covariance = _cov_json(cov(ustrip(px, b))),))
end
function JSON.lift(s::VgioStyle, ::Type{GaussianBlur}, b::Record)
    haskey(b, :unit) && return _blur_v09(s, b)        # an IO 0.9 body embedded in another package's earlier record
    if haskey(b, :pixel_frame)
        F = pixel_frame_from_name(b.pixel_frame::String)
        return GaussianBlur(relabel(F, CovMat(_cov_matrix(b.covariance, 2))))
    end
    f = b.frame::String
    if f == _TARGET_PLANE_NAME
        return with_unit(GaussianBlur(relabel(VgtkCore.TargetPlaneFrame, CovMat(_length_cov_from_json(b.covariance, 2, s.length_unit, (1, 1))))), m)
    end
    return GaussianBlur(relabel(ImageFrame{Symbol(f[7:end])}, CovMat(_cov_matrix(b.covariance, 2))))
end
# An IO 0.9 blur body, `{"unit": "px^2" | "<length>^2", "covariance"}`, unlabeled as that version read it.
const _PX2_NAME = "px^2"
function _blur_v09(::VgioStyle, b::Record)
    u = _blur_unit(b.unit::String)
    u === nothing && throw(ArgumentError("an IO 0.9 blur's unit is px^2 or a length unit squared (mm^2, …), got $(repr(b.unit))"))
    u isa _PxSquared && return GaussianBlur(_cov_matrix(b.covariance, 2))
    return with_unit(GaussianBlur(_length_cov_from_json(b.covariance, 2, u, (1, 1))), m)
end
# The unit an IO 0.9 blur block states, if it is one: `_PxSquared()` for exactly "px^2", the length unit of "<length>^2", or nothing.
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
