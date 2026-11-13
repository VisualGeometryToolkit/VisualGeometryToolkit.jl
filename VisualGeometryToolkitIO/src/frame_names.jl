# =============================================================================
# The frame grammar — VGC 3D frame types ↔ their serialized names
# =============================================================================
#
# "world" ↔ WorldFrame, "camera" ↔ CameraFrame (a name mirrors its type; IO 0.9 named it "camera_optical", read by the version steps), "view:<id>" ↔
# ViewFrame{id}, "target:<id>" ↔ TargetFrame{id}, "body:<id>" ↔ BodyFrame{id};
# ids are Symbols. Every frame name any vgio kind stores goes through it, so a
# file's frames read back as types.

"""
    frame_name(F::Type{<:Frame}) -> String

The serialized name of a VGC 3D frame type: `"world"`, `"camera"`
(VGC's `CameraFrame`, a single camera's optical frame), `"view:<id>"`,
`"target:<id>"`, `"body:<id>"` (VGC's `BodyFrame{id}`, a rigid body tracked or
mounted as a unit — a board rig, a camera rig). Inverse of
[`frame_from_name`](@ref).
"""
frame_name(::Type{WorldFrame}) = "world"
frame_name(::Type{CameraFrame}) = "camera"         # a single camera's optical frame
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
# The grammar's names: the fixed frames, and the prefixes of the frames with an id.
const _FIXED_FRAMES = ("world" => WorldFrame, "camera" => CameraFrame)
const _ID_FRAMES = ("view:" => ViewFrame, "target:" => TargetFrame, "body:" => BodyFrame)
function frame_from_name(s::AbstractString)
    for (name, F) in _FIXED_FRAMES
        s == name && return F
    end
    for (prefix, F) in _ID_FRAMES
        (startswith(s, prefix) && length(s) > length(prefix)) && return F{Symbol(s[(length(prefix) + 1):end])}
    end
    throw(
        ArgumentError(
            "unknown frame name $(repr(s)) — expected \"world\", \"camera\", \"view:<id>\", \"target:<id>\" or \"body:<id>\""
        )
    )
end

# The grammar as a pattern (the published JSON Schema's; `frame_from_name` is the reader).
const _FRAME_NAME_RE = Regex("^($(join(first.(_FIXED_FRAMES), '|'))|($(join(first.(_ID_FRAMES), '|'))).+)\$")
const _FRAME_NAME = _Named(
    Base.Fix1(occursin, _FRAME_NAME_RE),
    "a frame name (\"world\", \"camera\", \"view:<id>\", \"target:<id>\", \"body:<id>\")",
)

