# =============================================================================
# vgio/radial_profile/v1.0 ↔ VGC `BinnedRadialProfile` — a measured radial profile, binned (a flat field by field angle)
# =============================================================================
#
# {"schema": "vgio/radial_profile/v1.0", <envelope>,
#  "profile": {"type": "binned_radial_profile",
#              "chart": {"type": "equisolid", "center": [cx, cy], "radius": r, "focal_length": f},   (pixels: units.pixel)
#              "edges": [θ0, θ1, …, θB],          (the chart's coordinate: equisolid in units.angle, rad; B+1 increasing)
#              "values": [[v1, …, vB], …],         (one row per channel; unitless)
#              "counts": [n1, …, nB]}}             (pixels per bin; bin b, 1-based, is [edges[b], edges[b+1]))
#
# A MEASURED profile, binned rather than fitted: the vignetting bodies are fitted fields (splines), so this is its own
# body; its CHART is Core's one radial-chart vocabulary (`RADIAL_CHARTS`: linear, projective, equisolid), the
# equisolid chart alone stating its focal length. The profile's centre and radius are COLMAP pixels in the value and
# written in the file's pixel frame (moved by the style, as the vignetting body's positions). The body is embeddable
# (`JSON.lower(::VgioStyle, ::BinnedRadialProfile)`, `JSON.lift(style, BinnedRadialProfile, body)`).

const _RADIAL_CHART = _Named(c -> Symbol(c) in VgtkCore.RADIAL_CHARTS, "a radial chart ($(join(VgtkCore.RADIAL_CHARTS, ", ")))")
const _PROFILE_CHART = _IfKey(:focal_length,
    _Obj(:type => _Named(==("equisolid"), "the equisolid chart (the one that states a focal length)"), :center => _XY, :radius => _Real(),
         :focal_length => _Real()),
    _Obj(:type => _RADIAL_CHART, :center => _XY, :radius => _Real()))
function _radial_profile_check(x)
    B = length(x[:edges]) - 1
    B >= 1 || return "edges: at least two (B + 1 edges for B bins)"
    all(i -> x[:edges][i] < x[:edges][i + 1], 1:B) || return "edges: strictly increasing"
    isempty(x[:values]) && return "values: one row per channel, at least one"
    all(r -> length(r) == B, x[:values]) || return "values: every channel has one value per bin ($(B))"
    length(x[:counts]) == B || return "counts: one pixel count per bin ($(B))"
    all(>=(0), x[:counts]) || return "counts: each ≥ 0"
    (x[:chart][:type] == "equisolid") == haskey(x[:chart], :focal_length) || return "chart: the equisolid chart states its focal length, and only it"
    return nothing
end
const _RADIAL_PROFILE_BODY = _CheckedSpec(_Obj(
    :type => _Tok("binned_radial_profile"), :chart => _PROFILE_CHART, :edges => _Vec(_Real()),
    :values => _Vec(_Vec(_Real())), :counts => _Vec(_Int())), _radial_profile_check)
const _RADIAL_PROFILE_SPEC = _kind_spec("radial_profile", Pair{Symbol, _Spec}[:profile => _RADIAL_PROFILE_BODY])

function JSON.lower(s::VgioStyle, p::BinnedRadialProfile)
    c = VgtkCore.center(p) + _vignetting_offset(s, CANONICAL_PIXEL_FRAME, _pixel_frame(s))
    chart = merge((type = String(radial_chart(p)), center = Float64[c[1], c[2]], radius = field_radius(p)),
                  p.focal_length === nothing ? (;) : (focal_length = p.focal_length,))
    return (type = "binned_radial_profile", chart = chart, edges = p.edges, values = _matrix_rows(p.values), counts = p.counts)
end
function JSON.lift(s::VgioStyle, ::Type{BinnedRadialProfile}, b::Record)
    ch = b.chart
    c = SVector(Float64(ch.center[1]::Real), Float64(ch.center[2]::Real)) + _vignetting_offset(s, _pixel_frame(s), CANONICAL_PIXEL_FRAME)
    return BinnedRadialProfile(Symbol(ch.type::String), c, Float64(ch.radius::Real), Float64[Float64(e::Real) for e in b.edges],
                               _rows_matrix(b.values::Vector{Any}), Int[Int(n::Integer) for n in b.counts];
                               focal_length = haskey(ch, :focal_length) ? Float64(ch.focal_length::Real) : nothing)
end

"""
    save(path, p::BinnedRadialProfile; provenance = (;))

Write VisualGeometryToolkitCore's `BinnedRadialProfile` — a measured radial
profile (a flat field binned by field angle) — as a `vgio/radial_profile/v1.0`
file: its chart (one of `RADIAL_CHARTS`: its centre and image-circle radius in
pixels, the equisolid chart's focal length), the `B + 1` bin edges in the chart's
coordinate (radians for equisolid), the `channels × B` values (one row per
channel) and the `B` pixel counts.
"""
function save(path::AbstractString, p::BinnedRadialProfile; provenance::NamedTuple = (;))
    s = VgioStyle()
    _write_kind(path, "radial_profile", (profile = JSON.lower(s, p),), provenance; style = s)
    return nothing
end

"""
    load(BinnedRadialProfile, path) -> BinnedRadialProfile

Read a `vgio/radial_profile/v1.0` file ([`save`](@ref)'s): the profile exactly
as written, its centre in COLMAP pixels (a file in another pixel frame moved
into them).
"""
function load(::Type{BinnedRadialProfile}, path::AbstractString)
    d = _read_store(path, "radial_profile", _RADIAL_PROFILE_SPEC)
    return JSON.lift(VgioStyle(d), BinnedRadialProfile, d.profile::Record)
end
