# =============================================================================
# src/datasets/roma.jl — the RoMa / SP+LG consolidated-csponds reader.
#
# Input: `<dir>/<scene>.jld2`, written by `consolidate_csponds`
# (src/datasets/consolidate.jl: plain pixel records, `format = 2`), or a file
# of its first version, which serialized the pair's `rig::StereoRig`, the
# matcher's own point types and the GT config (no `format` key; read through
# the legacy typemap below). No SNN filter: a RoMa correspondence
# carries a match confidence, stored as `snn = 1 − confidence` (lower = better,
# the samplers' order), not a ratio test.
#
# The first-version files store `VisualGeometryFeatures.GTConfig`. Nothing here
# needs it resolved: the rebuild below never reads `gt_config`, and loading
# with it unresolvable rebuilt identical RomaPairs (checked 2026-09-15 on one
# file per dataset).
# =============================================================================

"""
    RomaPair <: StereoRawPair

One raw RoMa / SP+LG pair, in pixels: csponds, `snn = 1 − confidence`, the
consolidated GT inlier mask, the pixel calibrations, image sizes and the GT
relative pose, with its source scene ([`pair_scene`](@ref)).
"""
struct RomaPair <: StereoRawPair
    src_id   :: String
    dst_id   :: String
    scene    :: String
    csponds  :: StructArray             # uncalibrated pixel csponds
    snn      :: Vector{Float32}         # sampler sort key (lower = better): 1 - confidence
    gt_inliers :: BitVector             # the consolidated inlier_mask
    K1       :: SMatrix{3,3,Float64,9}
    K2       :: SMatrix{3,3,Float64,9}
    size1    :: Size2{Int}
    size2    :: Size2{Int}
    R_gt     :: SMatrix{3,3,Float64,9}
    t_gt     :: SVector{3,Float64}
end

# The first-version files predate the current VGC types (Unitful-era, before the
# frame tags), so JLD2 reconstructs every entry opaquely; the reader rebuilds a
# RomaPair field by field from the reconstructed properties and never touches a
# unit conversion. Layout (verified 2026-10-06 on one file of each of the
# scannet1500/eth3d splg and roma and seven_scenes splg trees):
#   rig.{source,target}::BoundedCamera{ camera.model.intrinsics.K.data::NTuple{9},
#   extent.{width,height} } ; rig.extrinsics.{R.mat.data, t.data} ;
#   csponds: a StructVector (or a reconstructed wrapper with
#   .components.{first,second}) of Point2f (SP+LG) or Uncertain{Point2f} (RoMa,
#   whose covariance the pair does not keep).
#
# The RoMa endpoints were written as VisualGeometryCore's former `Uncertain`
# STRUCT (fields `value`, `matrix`). `Uncertain` is now the alias
# `Attributed{V, <:Union{CovMat, PrecMat}}`, which that layout does not match,
# so the typemap reads each as a `_LegacyUncertain` — its point, in Float64 —
# and the rebuild reads that point exactly as `centroid` read the struct's.
#
# The files also store Unitful `Quantity` values (mm lengths in the GTConfig and
# intrinsics metadata) whose names can resolve onto current types and force an
# impossible convert during the read. The typemap reads each as an opaque
# `_LegacyQuantity`; the rebuild reads none of them.
struct _LegacyQuantity
    val::Float64
end
JLD2.rconvert(::Type{_LegacyQuantity}, nt::NamedTuple{(:val,), Tuple{Float64}}) = _LegacyQuantity(nt.val)
struct _LegacyUncertain
    value::SVector{2,Float64}
end
JLD2.rconvert(::Type{_LegacyUncertain}, nt::NamedTuple) = _LegacyUncertain(SVector{2,Float64}(nt.value))
const _LEGACY_TYPEMAP = Dict{String,Any}(
    "Unitful.Quantity" => JLD2.Upgrade(_LegacyQuantity),
    "Quantity"         => JLD2.Upgrade(_LegacyQuantity),
    "VisualGeometryCore.Uncertain" => JLD2.Upgrade(_LegacyUncertain),
)

"""
    load_roma_pairs(path) -> Vector{RomaPair}

Every pair of one consolidated RoMa / SP+LG csponds scene file
(`<dir>/<scene>.jld2`), in pixels.
"""
function load_roma_pairs(path::AbstractString)
    format = JLD2.jldopen(f -> haskey(f, "format") ? f["format"] : 1, path, "r")
    format == CONSOLIDATED_FORMAT &&
        return RomaPair[_roma_pair_v2(e) for e in JLD2.jldopen(f -> f["pairs"], path, "r")]
    format == 1 || error("$path: consolidated csponds format $format; this reader knows 1 and $CONSOLIDATED_FORMAT")
    return RomaPair[_roma_pair(e) for e in JLD2.jldopen(f -> f["pairs"], path, "r"; typemap = _LEGACY_TYPEMAP)]
end

# A record of `consolidate_csponds` (format 2): plain fields, pixels.
function _roma_pair_v2(e)::RomaPair
    cf, ct = _legacy_cols(e.csponds)
    cs_pix = VisualGeometryCore.csponds([SVector{2,Float64}(p) for p in cf], [SVector{2,Float64}(p) for p in ct])
    return RomaPair(first(splitext(e.source_name)), first(splitext(e.target_name)), e.scene, cs_pix,
                    Float32.(1.0f0 .- e.confidence), BitVector(e.inlier_mask), e.K1, e.K2,
                    Size2{Int}(e.size1...), Size2{Int}(e.size2...), e.R_gt, e.t_gt)
end

_legacy_p2(p) = SVector{2,Float64}(centroid(p))
_legacy_p2(p::_LegacyUncertain) = p.value
_legacy_cols(cs::StructArrays.StructArray) =
    (StructArrays.component(cs, :first), StructArrays.component(cs, :second))
_legacy_cols(cs) = (cs.components.first, cs.components.second)
function _roma_pair(entry)::RomaPair
    rig = entry.rig
    _K(cam)  = SMatrix{3,3,Float64,9}(cam.camera.model.intrinsics.K.data)
    _sz(cam) = Size2(cam.extent.width, cam.extent.height)
    R_gt = SMatrix{3,3,Float64,9}(rig.extrinsics.R.mat.data)
    t_gt = SVector{3,Float64}(rig.extrinsics.t.data)
    cf, ct = _legacy_cols(entry.csponds)
    cs_pix = VisualGeometryCore.csponds([_legacy_p2(p) for p in cf], [_legacy_p2(p) for p in ct])
    # PROSAC order, lower = better: invert the match confidence (higher = better)
    snn = Float32.(1.0f0 .- entry.confidence)
    return RomaPair(first(splitext(entry.source_name)), first(splitext(entry.target_name)),
                    entry.scene, cs_pix, snn, entry.inlier_mask,
                    _K(rig.source), _K(rig.target), _sz(rig.source), _sz(rig.target),
                    R_gt, t_gt)
end

# Per-pair source-scene label (the pool's `scene_names`).
Pools.pair_scene(p::RomaPair) = p.scene
