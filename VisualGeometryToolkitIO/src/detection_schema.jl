# =============================================================================
# vgio/detection/v1.0 ↔ detections per image (VGC `Attributed{<:FeatureFrame}` / `Attributed{<:Uncertain{<:FeatureFrame}}`)
# =============================================================================
#
# {"schema": "vgio/detection/v1.0", <envelope>,
#  "detection": {"frame": "point_frame", "number_type": "float64",
#                "covariance": {"storage": "full", "number_type": "float64"},          (absent: plain frames)
#                "attributes": {"label": "symbol", "window": "point2", "F": "float64", "covariance_status": "symbol"}},
#  "images": [{"image": "DSC00001.ARW",
#              "detections": [{"center": [512.25, 300.5], "covariance": [[0.01, 0.0], [0.0, 0.02]],
#                              "attributes": {"label": "corner", …, "covariance_status": "propagated"}},
#                             {"center": [80.0, 90.0], "attributes": {"label": "unknown", …, "covariance_status": "unclassified"}}]}]}
#
# A detector's output per image, unassociated (the terminology's DETECTION): each
# detection is its frame through THE frame codec (core_json.jl: `center` and the
# shape; an uncertain frame's `covariance` over its `frame_dof` parameters,
# centre first) and its attributes. `detection` declares ONCE the type every
# detection of the file has — the frame type (the tag rule), its number type,
# the covariance's storage and number type, and each attribute's type — so a
# file reads back exact by type, an image without detections included.
#
# AN ABSENT COVARIANCE is stated by its producer: a record of an uncertain file
# omits `covariance` exactly when the producer has none (in memory, every entry
# not finite), and then its attributes MUST hold `covariance_status`, the
# producer's reason. It reads back as the NaN covariance in the declared
# storage. No `null`, no size beside it, no default.
#
# Pixel coordinates are in `units.pixel`; a file is written in colmap (VGC's
# detectors' convention) and a reader moves another frame's centres into it.

# The attribute types a file declares, by name (each a type a detector's attributes hold).
const _ATTRIBUTE_TYPES = (("float64", Float64), ("float32", Float32), ("int", Int), ("bool", Bool),
                          ("string", String), ("symbol", Symbol), ("point2", Point2{Float64}))
const _NUMBER_TYPES = (("float64", Float64), ("float32", Float32))
const _COV_STORAGES = ("full", "diagonal", "isotropic")
_type_name(table, T, what) = (i = findfirst(p -> p[2] === T, table); i === nothing ?
    throw(ArgumentError("a detection file's $(what) is one of $(join((repr(p[2]) for p in table), ", ")), got $(T)")) : table[i][1])
_type_of(table, name) = table[findfirst(p -> p[1] == name, table)][2]

_attribute_json(x::Symbol) = String(x)
_attribute_json(x::Point2) = Float64[x[1], x[2]]
_attribute_json(x::AbstractFloat) = Float64(x)   # a Float32 exactly, as its Float64
_attribute_json(x) = x
_attribute_from(::Type{Float64}, x) = Float64(x::Real)
_attribute_from(::Type{Float32}, x) = Float32(x::Real)
_attribute_from(::Type{Int}, x) = Int(x::Integer)
_attribute_from(::Type{Bool}, x) = x::Bool
_attribute_from(::Type{String}, x) = String(x::AbstractString)
_attribute_from(::Type{Symbol}, x) = Symbol(x::AbstractString)
_attribute_from(::Type{Point2{Float64}}, x) = Point2{Float64}(Float64(x[1]::Real), Float64(x[2]::Real))

# A covariance's storage, by the matrix its CovMat holds.
_storage_name(::Type{<:CovMat{N, T, M}}) where {N, T, M} = M <: Diagonal ? "diagonal" : M <: ScaledIdentity ? "isotropic" : "full"
_storage_name(::Type{<:PrecMat{N, T, M}}) where {N, T, M} = _storage_name(CovMat{N, T, M})   # written as its covariance
_cov_eltype(::Type{<:Union{CovMat{N, T}, PrecMat{N, T}}}) where {N, T} = T
# The covariance of storage `storage` and number type `T`, from the wire's matrix `m` (or all NaN: none stated).
function _stored_cov(storage::AbstractString, ::Type{T}, ::Val{N}, m) where {T, N}
    storage == "diagonal" && return CovMat(Diagonal(SVector{N, T}(m === nothing ? ntuple(_ -> T(NaN), N) : m.diag)))
    storage == "isotropic" && return CovMat(ScaledIdentity{N}(m === nothing ? T(NaN) : T(m.λ)))
    return CovMat(m === nothing ? SMatrix{N, N, T}(ntuple(_ -> T(NaN), N * N)) : SMatrix{N, N, T}(m))
end

# The `detection` declaration of a detection type `D`.
function _detection_header(::Type{D}) where {D}
    isconcretetype(D) && D <: Attributed || throw(ArgumentError(
        "a detection file holds detections of one concrete type (a StructArray's, or a Vector of one); got $(D)"))
    V, A = fieldtype(D, :value), fieldtype(D, :attributes)
    A <: NamedTuple || throw(ArgumentError("a detection's attributes are a NamedTuple, got $(A)"))
    F = V <: Uncertain ? fieldtype(V, :value) : V
    F <: FeatureFrame || throw(ArgumentError("a detection is an Attributed feature frame, plain or Uncertain; got $(V)"))
    h = (frame = tag_of(_FRAME_TAGS, F), number_type = _type_name(_NUMBER_TYPES, eltype(F), "number type"))
    if V <: Uncertain
        C = fieldtype(V, :attributes)
        h = merge(h, (covariance = (storage = _storage_name(C), number_type = _type_name(_NUMBER_TYPES, _cov_eltype(C), "covariance number type")),))
    end
    attrs = NamedTuple{fieldnames(A)}(Tuple(_type_name(_ATTRIBUTE_TYPES, T, "attribute type") for T in fieldtypes(A)))   # in the value's order
    return merge(h, (attributes = attrs,))
end
# …and the type it declares.
function _detection_type(h)
    T = _type_of(_NUMBER_TYPES, h.number_type::String)
    F = _FRAME_TAGS[h.frame::String]{T}
    names = Tuple(Symbol(k) for k in keys(h.attributes))
    A = NamedTuple{names, Tuple{(_type_of(_ATTRIBUTE_TYPES, h.attributes[k]::String) for k in keys(h.attributes))...}}
    haskey(h, :covariance) || return Attributed{F, A}
    C = typeof(_stored_cov(h.covariance.storage::String, _type_of(_NUMBER_TYPES, h.covariance.number_type::String), Val(frame_dof(F)), nothing))
    return Attributed{Uncertain{F, C}, A}
end

# One detection's record, on the style.
function _detection_json(s::VgioStyle, d::Attributed)
    v, a = getfield(d, :value), getfield(d, :attributes)
    attrs = (attributes = map(_attribute_json, a),)
    v isa Uncertain || return merge(JSON.lower(s, v), attrs)
    Σ = cov(v)
    all(isfinite, Σ) && return merge(JSON.lower(s, v), attrs)
    any(isfinite, Σ) && throw(ArgumentError("a detection's covariance is finite, or none (every entry NaN, its `covariance_status` stated); got $(Σ)"))
    haskey(a, :covariance_status) || throw(ArgumentError(
        "a detection without a covariance (not finite) states why: its attributes hold `covariance_status` (the producer's reason)"))
    return merge(JSON.lower(s, geom(v)), attrs)
end

# …and back: a record of the file's type `D`, its centres moved by `offset` into colmap.
function _detection_from(s::VgioStyle, ::Type{Attributed{V, A}}, x, offset) where {V, A}
    a = A(Tuple(_attribute_from(fieldtype(A, k), _get_key(x.attributes, String(k))) for k in fieldnames(A)))
    F = V <: Uncertain ? fieldtype(V, :value) : V
    f = JSON.lift(s, F, x) + SVector{2, eltype(F)}(offset)
    V <: Uncertain || return Attributed(f, a)
    C = fieldtype(V, :attributes)
    N, Tc = frame_dof(F), _cov_eltype(C)
    m = _has_key(x, "covariance") ? _cov_matrix(_lower(x.covariance), N) : nothing
    Σ = _stored_cov(_storage_name(C), Tc, Val(N), m)
    Σ isa C || throw(ArgumentError("a detection's covariance is stored $(_storage_name(typeof(Σ))), the file declares $(_storage_name(C))"))
    return Attributed(Uncertain(f, Σ), a)
end

# --- the table ----------------------------------------------------------------

struct _Bool <: _Spec end
_validate(::_Bool, x, file, at) = (x isa Bool || _refuse(file, at, "expected true or false"); nothing)
const _ANY_STRING = _Named(_ -> true, "a string")
_attribute_spec(name) = name in ("float64", "float32") ? _Real() : name == "int" ? _Int() : name == "bool" ? _Bool() :
    name == "string" ? _ANY_STRING : name == "symbol" ? _NonEmpty() : _XY

const _DETECTION_HEADER = _Obj(
    Pair{Symbol, _Spec}[
        :frame => _Named(t -> haskey(_FRAME_TAGS, t), "a frame type ($(join(keys(_FRAME_TAGS), ", ")))"),
        :number_type => _Named(t -> any(p -> p[1] == t, _NUMBER_TYPES), "a number type (float64, float32)"),
        :attributes => _CheckedSpec(_Free(), x -> begin
            bad = [k for (k, v) in x if !(v isa AbstractString && any(p -> p[1] == v, _ATTRIBUTE_TYPES))]
            isempty(bad) ? nothing : "the attribute $(first(bad))'s type is one of $(join(first.(_ATTRIBUTE_TYPES), ", "))"
        end),
    ],
    Pair{Symbol, _Spec}[:covariance => _Obj(
        :storage => _Named(t -> t in _COV_STORAGES, "a covariance storage (full, diagonal, isotropic)"),
        :number_type => _Named(t -> any(p -> p[1] == t, _NUMBER_TYPES), "a number type (float64, float32)"),
    )],
)

# A detection record's table, from the file's declaration.
function _detection_record_spec(h)
    shape = Dict("point_frame" => Pair{Symbol, _Spec}[], "disc" => Pair{Symbol, _Spec}[:sigma => _Real()],
                 "oriented_disc" => Pair{Symbol, _Spec}[:sigma => _Real(), :angle => _Real()],
                 "elliptic_frame" => Pair{Symbol, _Spec}[:shape_matrix => _Vec(_Vec(_Real(), 2), 2)],
                 "affine_frame" => Pair{Symbol, _Spec}[:affine_matrix => _Vec(_Vec(_Real(), 2), 2)])[h[:frame]]
    attrs = _Obj(Pair{Symbol, _Spec}[Symbol(k) => _attribute_spec(v) for (k, v) in h[:attributes]], Pair{Symbol, _Spec}[])
    dof = frame_dof(_FRAME_TAGS[h[:frame]])
    cov = haskey(h, :covariance) ? Pair{Symbol, _Spec}[:covariance => _cov_spec(dof)] : Pair{Symbol, _Spec}[]
    return _Obj(Pair{Symbol, _Spec}[:center => _XY, shape..., :attributes => attrs], cov)
end

# The file's table: the envelope and the declaration, then every detection by the declaration.
struct _DetectionSpec <: _Spec
    outer::_Spec
end
function _validate(s::_DetectionSpec, x, file, at)
    _validate(s.outer, x, file, at)
    h = _getkey(x, :detection)
    rec = _detection_record_spec(h)
    uncertain = _haskey(h, :covariance)
    for (i, im) in enumerate(_getkey(x, :images)), (j, d) in enumerate(_getkey(im, :detections))
        where_ = _at(at, "images[$(i)].detections[$(j)]")
        _validate(rec, d, file, where_)
        uncertain && !_haskey(d, :covariance) && !haskey(h[:attributes], "covariance_status") && _refuse(file, where_,
            "a detection without a covariance states why: the file declares a `covariance_status` attribute")
        if uncertain && _haskey(d, :covariance)
            c = _getkey(d, :covariance)
            st = c isa AbstractVector ? "full" : _haskey(c, :diagonal) ? "diagonal" : "isotropic"
            st == h[:covariance][:storage] || _refuse(file, _at(where_, "covariance"), "stored $(st), the file declares $(h[:covariance][:storage])")
        end
    end
    return nothing
end

const _DETECTION_SPEC = _DetectionSpec(_kind_spec("detection",
    Pair{Symbol, _Spec}[
        :detection => _DETECTION_HEADER,
        :images => _CheckedSpec(_Vec(_Obj(:image => _NonEmpty(), :detections => _Vec(_Free()))),
                                x -> _unique_check([im[:image] for im in x], "image names")),
    ]))

# --- save / load --------------------------------------------------------------

"""
    save(path, detections::AbstractDict{<:AbstractString, <:AbstractVector{<:AbstractDetection}}; provenance = (;))

Write detections per image — a detector's output, `image name => detections`
(VisualGeometryToolkitFeatures' `detect_foerstner`, ASIFT's; a `StructArray` or a
`Vector`) — as a `vgio/detection/v1.0` file, images sorted by name, pixels in
colmap (the detectors' convention). Every detection is of one concrete type
`Attributed{F, A}` or `Attributed{Uncertain{F, C}, A}`, `F` a feature frame of
Float64 or Float32, `A` a NamedTuple of attributes of the types Float64,
Float32, Int, Bool, String, Symbol and `Point2{Float64}`; the file declares that
type once. An uncertain frame's covariance is over its `frame_dof` parameters,
centre first; a detection whose covariance is not finite (every entry NaN: its
producer has none) is written without one and must carry the attribute
`covariance_status`, the producer's reason. Anything else is an
`ArgumentError`, nothing written.
"""
function save(path::AbstractString, detections::AbstractDict{<:AbstractString, <:AbstractVector{<:AbstractDetection}};
              provenance::NamedTuple = (;))
    isempty(detections) && throw(ArgumentError("save: a detection file holds the detections of at least one image"))
    Ds = unique(eltype(v) for v in values(detections))
    length(Ds) == 1 || throw(ArgumentError("save: a detection file holds one detection type; its images hold $(join(Ds, ", "))"))
    s = VgioStyle()
    images = [(image = String(k), detections = [_detection_json(s, d) for d in detections[k]]) for k in sort!(collect(keys(detections)))]
    _write_kind(path, "detection", (detection = _detection_header(only(Ds)), images = images), provenance; style = s)
    return nothing
end

"""
    load(Dict{String, <:AbstractVector{<:AbstractDetection}}, path) -> Dict{String, Vector{D}}

Read a `vgio/detection/v1.0` file ([`save`](@ref)'s): per image name, its
detections as a `Vector{D}`, `D` the detection type the file declares (a
`StructArray` written reads back as a `Vector` of its element type), centres in
colmap pixels (a file in another pixel frame moved into it). A detection the
file holds without a covariance (its `covariance_status` stated) reads back with
the NaN covariance in the declared storage.
"""
function load(::Type{<:AbstractDict{String, <:AbstractVector{<:AbstractDetection}}}, path::AbstractString)
    d = _read_store(path, "detection", _DETECTION_SPEC)
    s = VgioStyle(d)
    D = _detection_type(d.detection)
    o = EuclideanMap(_pixel_frame(s), CANONICAL_PIXEL_FRAME)(SVector(0.0, 0.0))
    return Dict{String, Vector{D}}(
        String(im.image::String) => D[_detection_from(s, D, x, o) for x in im.detections::Vector{Any}] for im in d.images::Vector{Any}
    )
end
