# =============================================================================
# The frame grammar — VGC 3D frame types ↔ their serialized names
# =============================================================================
#
# "world" ↔ WorldFrame, "camera_optical" ↔ CameraFrame, "view:<id>" ↔
# ViewFrame{id}, "target:<id>" ↔ TargetFrame{id}, "body:<id>" ↔ BodyFrame{id};
# ids are Symbols. Every frame name any vgio kind stores goes through it, so a
# file's frames read back as types.

"""
    frame_name(F::Type{<:Frame}) -> String

The serialized name of a VGC 3D frame type: `"world"`, `"camera_optical"`
(VGC's `CameraFrame`, a single camera's optical frame), `"view:<id>"`,
`"target:<id>"`, `"body:<id>"` (VGC's `BodyFrame{id}`, a rigid body tracked or
mounted as a unit — a board rig, a camera rig). Inverse of
[`frame_from_name`](@ref).
"""
frame_name(::Type{WorldFrame}) = "world"
frame_name(::Type{CameraFrame}) = "camera_optical"         # a single camera's optical frame
frame_name(::Type{ViewFrame{id}}) where {id} = "view:$(id)"
frame_name(::Type{TargetFrame{id}}) where {id} = "target:$(id)"
frame_name(::Type{BodyFrame{id}}) where {id} = "body:$(id)"             # a rigid body: a tracked or mounted rig
# Symmetric with the lens refusal: an unhandled frame type names itself and
# the fix, instead of surfacing as a MethodError from inside whichever `save`
# happened to reach it.
frame_name(::Type{F}) where {F <: Frame} = throw(
    ArgumentError(
        "no vgio serialization for frame type $F — supply its id type parameter " *
            "(e.g. `ViewFrame{:cam}`, not `ViewFrame`), or add a `frame_name`/" *
            "`frame_from_name` pair in frame_names.jl"
    )
)

"""
    frame_from_name(s::AbstractString) -> Type{<:Frame}

Parse a serialized frame name back to the VGC frame type; the id becomes a
`Symbol` (a board uid that happens to look numeric is still an identity, not
a number).
"""
function frame_from_name(s::AbstractString)
    s == "world" && return WorldFrame
    s == "camera_optical" && return CameraFrame
    for (prefix, F) in (("view:", ViewFrame), ("target:", TargetFrame), ("body:", BodyFrame))
        (startswith(s, prefix) && length(s) > length(prefix)) && return F{Symbol(s[(length(prefix) + 1):end])}
    end
    throw(
        ArgumentError(
            "unknown frame name $(repr(s)) — expected \"world\", \"camera_optical\", \"view:<id>\", \"target:<id>\" or \"body:<id>\""
        )
    )
end

const _FRAME_NAME = _Named(
    s -> try
        frame_from_name(s)
        true
    catch
        false
    end,
    "a frame name (\"world\", \"camera_optical\", \"view:<id>\", \"target:<id>\", \"body:<id>\")",
)

