# =============================================================================
# vgio/blur/v2.0 ↔ named VGC `GaussianBlur`s
# =============================================================================
#
# The file opens with the envelope (validate.jl: `units`, `conventions`, `generator`).
#
# {"schema": "vgio/blur/v2.0", <envelope>,
#  "blurs": [{"name": "R", "pixel_frame": "colmap", "covariance": {"isotropic": λ}},           a blur on a sensor (px²)
#            {"name": "print", "frame": "target:89b", "covariance": [[…], […]]}, …]}         a blur on a print (units.length²)
#      name → VGC GaussianBlur, the density N(0, Σ), its covariance in the storage it holds (a ScaledIdentity,
#      SDiagonal or SMatrix: IO's covariance encoding) and its FRAME, the plane it lives on, which its `CovMat`
#      carries and reads back with: a pixel frame (`pixel_frame`: PixelFrame{O}, or `frame` "image:<id>",
#      ImageFrame{id}) — px², as stored — or a target's plane (`frame` "target:<id>", TargetPlaneFrame{id}) — a
#      length squared, `units.length`² on disk (IO's length rule: a decimal variance exact, a computed one
#      within an ulp), m² quantities in memory. The unit is the field's power of a declared base, never a
#      per-blur string (`blur/v1` wrote one, and no frame; no published file holds a v1 file, which is refused).
#
# The body is a VGC value on IO's style, so another kind's record can nest it: `JSON.lower(style, blur)` and
# `JSON.lift(style, GaussianBlur, record)` (one blur's `{"pixel_frame" | "frame", "covariance"}`; an IO 0.9 body,
# `{"unit", "covariance"}`, as another package's record of an earlier version embeds it, reads as the unlabeled blur
# it held). The kind's save/load go through the same methods.
#
# A print blur's variances cross by IO's decimal-exact length rule — a decimal file number reads back exactly, a value
# computed in code (σ² of a σ) may move by an ulp, as every file length does (units.md, "What is exact").
#
# (The radiometric kinds this file held up to IO 0.10 — vignetting, environment_light — are VisualRadiometryToolkitIO's
# with the types they hold: validate.jl, `_MOVED_KINDS`.)

# A blur's plane: a pixel frame (`pixel_frame`), or `frame` — a view's image frame
# (`image:<id>`) or a target's plane (`target:<id>`, the target named as its 3-D frame is).
const _BLUR_FRAME_RE = r"^(image|target):.+$"
const _BLUR_RECORD = _Either(Pair{Symbol, _Spec}[
    :pixel_frame => _Obj(:name => _NonEmpty(), :pixel_frame => _PIXEL_FRAME, :covariance => _Cov(2)),
    :frame => _Obj(:name => _NonEmpty(), :frame => _Named(Base.Fix1(occursin, _BLUR_FRAME_RE), "a blur's plane (\"target:<id>\", \"image:<id>\")"), :covariance => _Cov(2)),
])
const _BLUR_SPEC = _kind_spec("blur", Pair{Symbol, _Spec}[:blurs => _Vec(_BLUR_RECORD)];
                              check = x -> _unique_check([b[:name] for b in x[:blurs]], "blur names"), rule = "blur names are unique")

# --- blurs ------------------------------------------------------------------------

"""
    save(path, blurs::AbstractDict{Symbol, <:GaussianBlur}; length_unit = mm, provenance = (;))

Write named Gaussian blurs (a camera's in-focus blur per channel, `:R`, `:G`,
`:B`; a print's blur) as a `vgio/blur/v2.0` file: each one's covariance in the
storage it holds (`{"isotropic": λ}` for a round one, `{"diagonal": […]}`, or
rows) and the plane it lives on, its covariance's frame — a pixel frame
(`PixelFrame{O}`, an `ImageFrame{id}`: px², plain numbers or px quantities) or
a target's plane (`TargetPlaneFrame{id}`, the target named: m² quantities, written in `length_unit`²).
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
_blur_frame(::Type{VgtkCore.TargetPlaneFrame{id}}) where {id} = (frame = "target:$(id)",)
_blur_frame(::Type{F}) where {F} = throw(ArgumentError(
    "save: a blur names the plane it lives on — a pixel frame, an ImageFrame{id} or a target's TargetPlaneFrame{id} (the bare TargetPlaneFrame names no target) — through its covariance's frame; got $(F)"))
_cov_frame(::CovMat{N, T, S, F}) where {N, T, S, F} = F
function JSON.lower(s::VgioStyle{<:Units}, b::GaussianBlur)
    Σ = cov(b)
    F = _cov_frame(Σ)
    plane = _blur_frame(F)
    if F <: VgtkCore.TargetPlaneFrame
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
    if startswith(f, "target:")
        P = VgtkCore.TargetPlaneFrame{Symbol(f[8:end])}
        return with_unit(GaussianBlur(relabel(P, CovMat(_length_cov_from_json(b.covariance, 2, s.length_unit, (1, 1))))), m)
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
