"""
    PoseLib

Julia bindings to [PoseLib](https://github.com/PoseLib/PoseLib), the C++ library
of minimal solvers and robust refiners for camera pose estimation.

Loading this package extends VisualGeometryToolkitCore:

- Its minimal solvers gain a PoseLib backend, `PoseLibSolver()`: the
  four-point homography, the seven-point fundamental matrix, the
  perspective-three-point (P3P) absolute pose and the five-point essential
  matrix. The eight-point essential matrix, `essential_dlt`, has no native
  variant, so its default routes here.
- PoseLib's robust Levenberg–Marquardt refiners are exported:
  [`refine_homography`](@ref), [`refine_fundamental`](@ref),
  [`refine_relpose`](@ref) and [`refine_absolute_pose`](@ref). Each takes a
  seed model and its correspondences, and returns the refined model in the
  caller's frame.

The C++ library and its flat C wrapper ship prebuilt as this package's
`PoseLib` artifact, on Linux only for now. The manual's "Platforms and
binaries" page says how to run against a library built from source.
"""
module PoseLib
    import VisualGeometryToolkitCore as VgtkCore

using StaticArrays: MVector, SVector, SMatrix, StaticVector, StaticMatrix
using Rotations: RotMatrix
using Artifacts: @artifact_str
using Libdl: dlopen, dlext
using CompilerSupportLibraries_jll: CompilerSupportLibraries_jll  # libstdc++/libgcc_s, as the recipe declares
using FlexUnits: Quantity

using VisualGeometryToolkitCore: VisualGeometryToolkitCore,
    AbstractLossShape, GaussianShape, TLSShape, HuberShape, CauchyShape,
    HomographyMat, FundMat, EssentialMat, Extrinsics3, FixedModels,
    EuclideanMap, UnlabeledFrame, UnitfulEuclideanMap, ustrip,
    PoseLibSolver, Vec3, Point3, Cspond, sources, targets, hasattr,
    sign_normalize, is_valid_solution,
    homography_4pt, fundamental_7pt, p3p,
    essential_5pt, essential_dlt, motion

# Every `ccall` goes to `libposelib_jl`, the library in this package's `PoseLib`
# artifact, resolved at load time so a relocated depot or an `Overrides.toml`
# entry (docs: Developer notes) is honoured; nothing here probes. The artifact
# is Linux-only until the `PoseLib_jll` release (docs: "Platforms and
# binaries"). Elsewhere the package still loads — so does everything that
# depends on it — and [`is_available`](@ref) is false: the first call that
# needs the library raises the reason (`_require_library`).
libposelib_jl::String = ""
const _UNAVAILABLE = "PoseLib.jl ships binaries for Linux only until the PoseLib_jll release, so the PoseLib library is not available on this platform; see https://visualgeometrytoolkit.github.io/PoseLib/stable/guide/binaries/"

function __init__()
    @static if Sys.islinux()
        _bind_library!(joinpath(artifact"PoseLib", "lib", "libposelib_jl.$(dlext)"))
    else
        _bind_library!(nothing)
    end
    return nothing
end

# Bind the library at `path` (opened now), or mark it unavailable (`nothing`):
# what `__init__` does per platform, and the switch the suite marks the library
# unavailable with.
function _bind_library!(path::Union{Nothing, AbstractString})
    path === nothing || dlopen(path)
    global libposelib_jl = path === nothing ? "" : String(path)
    return nothing
end

"""
    is_available() -> Bool

Whether the PoseLib library is loaded on this platform (the package's
binaries are Linux-only until the PoseLib_jll release). Without it, every
function that calls into the library raises an error stating why; the
package itself loads everywhere.
"""
is_available() = !isempty(libposelib_jl)

# The guard before every `ccall`: the library, or the reason it is missing.
@inline _require_library() = (is_available() || _library_unavailable(); nothing)
@noinline _library_unavailable() = error(_UNAVAILABLE)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Pack N SVector{3} into a flat NTuple for ccall (x,y,z,x,y,z,...)
@inline function _pack_points3(pts::NTuple{N, <:StaticVector{3, Float64}}) where N
    ntuple(i -> pts[div(i-1, 3) + 1][mod(i-1, 3) + 1], Val(3N))
end

# Read pose (R col-major + t) from MVector at offset. Absolute-pose (p3p)
# output — world-to-camera, so the labeled `Extrinsics3` is the honest type.
@inline function _read_pose_from_buf(buf, offset::Int)
    R = SMatrix{3,3,Float64,9}(
        buf[offset+1], buf[offset+2], buf[offset+3],
        buf[offset+4], buf[offset+5], buf[offset+6],
        buf[offset+7], buf[offset+8], buf[offset+9])
    t = Vec3{Float64}(buf[offset+10], buf[offset+11], buf[offset+12])
    Extrinsics3(R, t)
end

# The C wrapper takes each side of the correspondences as ONE contiguous
# point-major buffer (`M·n` and `N·n` doubles). Packing both columns, read
# through VGC's `sources`/`targets` protocol in one sweep, is the only per-call
# copy, whichever Cspond container is supplied (`Csponds`, `StructArray{<:Pair}`,
# `Vector{<:Pair}`).
@inline function _pack(cs::AbstractVector{<:Cspond{<:StaticVector{M}, <:StaticVector{N}}}) where {M, N}
    src = sources(cs); dst = targets(cs)
    Base.require_one_based_indexing(src, dst)
    n = length(cs)
    a = Vector{Float64}(undef, M * n); b = Vector{Float64}(undef, N * n)
    @inbounds for i in 1:n
        u = src[i]; v = dst[i]
        for j in 1:M; a[M * (i - 1) + j] = u[j]; end
        for j in 1:N; b[N * (i - 1) + j] = v[j]; end
    end
    return a, b
end

# Per-correspondence 3×2 matrices (the backprojection Jacobians `M₁`/`M₂`) into
# one contiguous `6·n` buffer, point-major. StaticArrays stores column-major and
# so does Eigen, so a StaticMatrix's LINEAR index is already the wire order and
# the C side maps each 6-double block straight onto an
# `Eigen::Matrix<double,3,2>` — no transpose anywhere in the crossing.
@inline function _pack_tangent(Ms::AbstractVector{<:StaticMatrix{3, 2}})
    Base.require_one_based_indexing(Ms)
    a = Vector{Float64}(undef, 6 * length(Ms))
    @inbounds for i in eachindex(Ms)
        M = Ms[i]
        for j in 1:6
            a[6 * (i - 1) + j] = M[j]
        end
    end
    return a
end

# Read 3x3 matrix from MVector at offset
@inline function _read_mat33_from_buf(buf, offset::Int)
    SMatrix{3,3,Float64,9}(
        buf[offset+1], buf[offset+2], buf[offset+3],
        buf[offset+4], buf[offset+5], buf[offset+6],
        buf[offset+7], buf[offset+8], buf[offset+9])
end

# =============================================================================
# homography_4pt  (PoseLib backend)
# =============================================================================

"""
    homography_4pt(::PoseLibSolver, s₁, s₂, s₃, s₄, d₁, d₂, d₃, d₄) -> Union{HomographyMat, Nothing}

PoseLib's four-point homography solver, the `PoseLibSolver` backend of
VisualGeometryToolkitCore's `homography_4pt` kernel. The points are 2-vectors:
sources `sᵢ` and targets `dᵢ` with `dᵢ ≃ H sᵢ`. Returns the homography, sign
normalized, or `nothing` for a degenerate sample. The public entry point is
`solve(Homography4pt(PoseLibSolver()), cs, idx)`.
"""
function VgtkCore.homography_4pt(::PoseLibSolver,
                        s₁::StaticVector{2,T}, s₂::StaticVector{2,T},
                        s₃::StaticVector{2,T}, s₄::StaticVector{2,T},
                        d₁::StaticVector{2,T}, d₂::StaticVector{2,T},
                        d₃::StaticVector{2,T}, d₄::StaticVector{2,T}) where T
    o = one(Float64)
    x1 = (SVector(Float64(s₁[1]), Float64(s₁[2]), o),
           SVector(Float64(s₂[1]), Float64(s₂[2]), o),
           SVector(Float64(s₃[1]), Float64(s₃[2]), o),
           SVector(Float64(s₄[1]), Float64(s₄[2]), o))
    x2 = (SVector(Float64(d₁[1]), Float64(d₁[2]), o),
           SVector(Float64(d₂[1]), Float64(d₂[2]), o),
           SVector(Float64(d₃[1]), Float64(d₃[2]), o),
           SVector(Float64(d₄[1]), Float64(d₄[2]), o))
    p1 = _pack_points3(x1)
    p2 = _pack_points3(x2)
    H_buf = Ref(ntuple(_ -> 0.0, Val(9)))
    _require_library()
    n = ccall((:poselib_homography_4pt, libposelib_jl), Cint,
              (Ref{NTuple{12,Float64}}, Ref{NTuple{12,Float64}},
               Ref{NTuple{9,Float64}}, Cint),
              Ref(p1), Ref(p2), H_buf, 1)
    n > 0 || return nothing
    H = SMatrix{3,3,Float64,9}(H_buf[]...)
    all(isfinite, H) || return nothing
    sign_normalize(HomographyMat{Float64}(Tuple(H)))
end

function VgtkCore.homography_4pt(::PoseLibSolver, cs, idx::AbstractVector{<:Integer})
    u1 = cs.first; u2 = cs.second
    @inbounds VgtkCore.homography_4pt(PoseLibSolver(),
        u1[idx[1]], u1[idx[2]], u1[idx[3]], u1[idx[4]],
        u2[idx[1]], u2[idx[2]], u2[idx[3]], u2[idx[4]])
end

# =============================================================================
# fundamental_7pt  (PoseLib backend)
# =============================================================================

function VgtkCore.fundamental_7pt(::PoseLibSolver, cs, idx::AbstractVector{<:Integer})
    u1 = cs.first; u2 = cs.second
    o = one(Float64)
    x1 = ntuple(Val(7)) do k
        @inbounds SVector(Float64(u1[idx[k]][1]), Float64(u1[idx[k]][2]), o)
    end
    x2 = ntuple(Val(7)) do k
        @inbounds SVector(Float64(u2[idx[k]][1]), Float64(u2[idx[k]][2]), o)
    end
    p1 = _pack_points3(x1)
    p2 = _pack_points3(x2)
    F_buf = MVector{27, Float64}(undef)
    _require_library()
    n = ccall((:poselib_relpose_7pt, libposelib_jl), Cint,
              (Ref{NTuple{21,Float64}}, Ref{NTuple{21,Float64}},
               Ptr{Float64}, Cint),
              Ref(p1), Ref(p2), pointer(F_buf), 3)
    FixedModels{3, FundMat{Float64}}(n) do i
        FundMat{Float64}(Tuple(_read_mat33_from_buf(F_buf, 9*(i-1))))
    end
end

# =============================================================================
# p3p  (PoseLib backend + convenience default)
# =============================================================================

function VgtkCore.p3p(::PoseLibSolver,
             x::AbstractVector{S},
             X::AbstractVector{T}) where {S <: StaticVector{3,Float64},
                                          T <: StaticVector{3,Float64}}
    length(x) == 3 && length(X) == 3 || return nothing
    px = _pack_points3((Vec3{Float64}(x[1]),   Vec3{Float64}(x[2]),   Vec3{Float64}(x[3])))
    pX = _pack_points3((Point3{Float64}(X[1]), Point3{Float64}(X[2]), Point3{Float64}(X[3])))
    pose_buf = MVector{48, Float64}(undef)
    _require_library()
    n = ccall((:poselib_p3p, libposelib_jl), Cint,
              (Ref{NTuple{9,Float64}}, Ref{NTuple{9,Float64}},
               Ptr{Float64}, Cint),
              Ref(px), Ref(pX), pointer(pose_buf), 4)
    FixedModels{4, Extrinsics3}(n) do i
        _read_pose_from_buf(pose_buf, 12*(i-1))
    end
end

# No-backend p3p default now lives in VisualGeometryToolkitCore (NativeSolver, its
# port of this library's p3p.cc). Select this backend explicitly:
# p3p(PoseLibSolver(), rays, X).

# =============================================================================
# essential_5pt  (PoseLib only — backend method + no-backend default)
# =============================================================================

# Task-local ccall marshalling buffers for the hot-path 5-pt solve. The
# per-call `Ref(p1)`/`Ref(p2)` plus the `pointer`-forced heap `MVector{90}`
# cost ~3.9 KB of garbage per solve — and the RVG RANSAC loop calls this once
# per hypothesis (~10³ per pair). Task-local (not global) so concurrent
# `Threads.@threads` eval loops each get their own buffers. Marshalling only:
# the C call receives bitwise-identical inputs and writes the same outputs,
# so results are unchanged (gated by the RVG 120-config fingerprint grid).
mutable struct _Relpose5ptBuffers
    p1::Base.RefValue{NTuple{15,Float64}}
    p2::Base.RefValue{NTuple{15,Float64}}
    E::Vector{Float64}                       # 10 models × 9 (col-major 3×3 each)
    # Hoisted compaction temp of the FixedModels builder (VGC types.jl): the
    # generic `FixedModels{N,M}(make, n)` allocates its `MVector{N,M}` temp
    # per call (~2.5 KB); reusing it here makes the whole solve heap-free.
    models::MVector{10, EssentialMat{Float64}}
end
_Relpose5ptBuffers() = _Relpose5ptBuffers(Ref{NTuple{15,Float64}}(),
                                          Ref{NTuple{15,Float64}}(),
                                          Vector{Float64}(undef, 90),
                                          MVector{10, EssentialMat{Float64}}(undef))

@inline _relpose5pt_buffers() =
    get!(task_local_storage(), :poselib_relpose5pt_buffers) do
        _Relpose5ptBuffers()
    end::_Relpose5ptBuffers

# Five bearings per side, as an `NTuple` or as the `SVector{5}` VGC's
# `solve(Essential5pt(PoseLibSolver()), cs, idx)` builds. `Tuple(::SVector)` is
# free, so both reach the same packing.
const _Bearings5 = Union{NTuple{5, <:StaticVector{3, Float64}},
                         StaticVector{5, <:StaticVector{3, Float64}}}

function VgtkCore.essential_5pt(::PoseLibSolver, x1::_Bearings5, x2::_Bearings5)
    bufs = _relpose5pt_buffers()
    bufs.p1[] = _pack_points3(Tuple(x1))
    bufs.p2[] = _pack_points3(Tuple(x2))
    E_buf = bufs.E
    _require_library()
    n = ccall((:poselib_relpose_5pt, libposelib_jl), Cint,
              (Ref{NTuple{15,Float64}}, Ref{NTuple{15,Float64}},
               Ptr{Float64}, Cint),
              bufs.p1, bufs.p2, E_buf, 10)
    # Verbatim FixedModels builder semantics (VGC `FixedModels{N,M}(make, n)`:
    # keep candidates satisfying `is_valid_solution`, pad with zero(M)) with the
    # MVector compaction temp drawn from the task-local buffers instead of a
    # fresh per-call allocation. Same predicate, same order, same padding —
    # results are element-for-element what the generic builder returns.
    valid = bufs.models
    n_valid = 0
    @inbounds for i in 1:Int(n)
        m = EssentialMat{Float64}(Tuple(_read_mat33_from_buf(E_buf, 9*(i-1))))
        is_valid_solution(m) || continue
        n_valid += 1
        valid[n_valid] = m
    end
    z = zero(EssentialMat{Float64})
    nv = n_valid   # closure-capture a NON-reassigned binding (no Box)
    FixedModels{10, EssentialMat{Float64}}(nv,
        ntuple(i -> i <= nv ? (@inbounds valid[i]) : z, Val(10)))
end

# Bearings only, as in VGC's `Essential5pt`: a 2-vector endpoint is a
# MethodError here, never lifted to z = 1. Columns are read through the
# `sources`/`targets` protocol so every Cspond container works (a plain
# `Vector{<:Pair}` has no `.first` column).
function VgtkCore.essential_5pt(::PoseLibSolver,
        cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{3}}},
        idx::AbstractVector{<:Integer})
    u1 = sources(cs); u2 = targets(cs)
    x1 = @inbounds ntuple(k -> Vec3{Float64}(u1[idx[k]]), Val(5))
    x2 = @inbounds ntuple(k -> Vec3{Float64}(u2[idx[k]]), Val(5))
    VgtkCore.essential_5pt(PoseLibSolver(), x1, x2)
end

# No-backend essential_5pt default now lives in VisualGeometryToolkitCore
# (NativeSolver port of relpose_5pt.cc). Select this backend explicitly:
# essential_5pt(PoseLibSolver(), x1, x2).

# =============================================================================
# essential_dlt  (PoseLib only — backend method + no-backend default)
# =============================================================================

function VgtkCore.essential_dlt(::PoseLibSolver,
        cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{3}}})
    n = length(cs)
    @assert n >= 8
    buf1, buf2 = _pack(cs)
    E_buf = MVector{9, Float64}(undef)
    _require_library()
    ccall((:poselib_essential_8pt, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}),
          buf1, buf2, n, pointer(E_buf))
    all(isfinite, E_buf) || return nothing
    EssentialMat{Float64}(Tuple(E_buf))
end

@inline VgtkCore.essential_dlt(cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{3}}}) =
    VgtkCore.essential_dlt(PoseLibSolver(), cs)

# =============================================================================
# refine_relpose  (robust-loss LM refinement of a relative pose — PoseLib LO)
# =============================================================================

export refine_relpose
public is_available
export LeZachTruncation

# Relative-pose currency: a cam1→cam2 motion is NOT a world-to-camera
# extrinsics, so `Extrinsics3`'s WorldFrame→CameraFrame labels would be
# semantically wrong here. An UNLABELED concrete alias (same layout as the
# retired VGC `Pose3`). VGC's view-identity frames (`ViewFrame{id}`) could
# label it; that migration has not been done.
public Rigid3

"""
    Rigid3

The rigid motion `x ↦ R·x + t` that the pose refiners take as their seed and
return: a VisualGeometryToolkitCore `EuclideanMap` in 3D with a `RotMatrix`
rotation, a `Vec3` translation and unlabeled frames.
`EuclideanMap(RotMatrix{3}(R), Vec3(t))` builds one.

For [`refine_relpose`](@ref) it maps the first camera's frame to the second's.
For [`refine_absolute_pose`](@ref) it maps the world frame to the camera's.
A labeled pose, such as the `Extrinsics3` that P3P returns, is converted with
VisualGeometryToolkitCore's `unlabel`.
"""
const Rigid3 = EuclideanMap{
    3, Float64, RotMatrix{3, Float64, 9}, Vec3{Float64},
    UnlabeledFrame{3}, UnlabeledFrame{3},
}

# -----------------------------------------------------------------------------
# The loss, typed: a VisualGeometryToolkitCore loss shape (or the Le–Zach marker),
# mapped to PoseLib's `BundleOptions::LossType` enum (PoseLib/types.h) ONLY at
# the ccall boundary. The integer enum never leaves this file.
#
# THE WIDTH. Each PoseLib loss is built from one scalar `loss_scale`
# (PoseLib/robust/robust_loss.h), and for every loss bound here that scalar IS
# the `loss_width` w of VisualGeometryToolkitCore's loss-shape closed forms,
# s = r²:
#
#   TruncatedLoss(w)        ρ = min(s, w²),  weight = s < w²         ≡ ρ(TLSShape(), s, w²)
#   TruncatedLossLeZach(w)  ρ = min(s, w²)  (TR-IRLS weights, μ = ½) — the TLS width
#   HuberLoss(w)            ρ = s ≤ w² ? s : w(2r − w), weight = min(1, w/r)  ≡ ρ(HuberShape(), s, w²)
#   CauchyLoss(w)           ρ = w²·log1p(s/w²), weight = 1/(1 + s/w²)          ≡ ρ(CauchyShape(), s, w²)
#   TrivialLoss             ρ = s, no width                                    ≡ ρ(GaussianShape(), s, ·)
#
# So no factor, square or ½ is applied: `_poselib_loss_scale` is the identity
# on the width, and test/test_refine.jl pins it behaviourally (a truncation at
# w, not w² or w/2; Huber's elbow at w). PoseLib's TruncatedCauchyLoss is not
# bound: no stack caller uses it.
# -----------------------------------------------------------------------------

"""
    LeZachTruncation()

The truncated loss `ρ = min(r², w²)`, optimized by the iteratively reweighted
least-squares scheme of Le and Zach (International Conference on 3D Vision,
2021). This is PoseLib's `TRUNCATED_LE_ZACH` loss.

The loss itself is VisualGeometryToolkitCore's `TLSShape()` at `loss_width = w`.
`LeZachTruncation()` is a marker, not a loss shape: it selects how the same
loss is optimized. Any refiner takes it as its `loss` keyword.
"""
struct LeZachTruncation end

# PoseLib `BundleOptions::LossType` enum values (PoseLib/types.h).
_poselib_loss_type(::GaussianShape)    = Int32(0)   # TRIVIAL
_poselib_loss_type(::TLSShape)         = Int32(1)   # TRUNCATED
_poselib_loss_type(::HuberShape)       = Int32(2)   # HUBER
_poselib_loss_type(::CauchyShape)      = Int32(3)   # CAUCHY
_poselib_loss_type(::LeZachTruncation) = Int32(5)   # TRUNCATED_LE_ZACH

# PoseLib's `loss_scale` from the stack's `loss_width` — the identity (header
# above). The Gaussian loss has no width: it takes none, and PoseLib ignores
# the scale it is handed.
_poselib_loss_scale(::GaussianShape, ::Nothing) = 1.0
_poselib_loss_scale(::GaussianShape, loss_width::Real) = throw(ArgumentError(
    "GaussianShape() has no loss width; omit `loss_width`"))
_poselib_loss_scale(loss, ::Nothing) = throw(ArgumentError(
    "$(loss) needs a `loss_width` (in the unit of the residual)"))
_poselib_loss_scale(::Union{TLSShape, HuberShape, CauchyShape, LeZachTruncation}, loss_width::Real) =
    Float64(loss_width)

# Unit-norm gate for a bearing column. Matches VisualGeometryToolkitEstimation's
# `_assert_unit_bearings` (`problems/cspond.jl`) in tolerance and in what the
# message has to tell the caller — the stack settled this shape, so this
# binding follows it rather than inventing a second one.
#
# The gate exists because `StaticVector{3}` is a NECESSARY condition, not a
# guarantee: a z = 1 chart point wears 3-vector clothing and even returns the
# right answer for every ray the chart can represent, so the exclusion of rays
# at and past 90° survives the type and merely stops being visible. The
# tolerance is not delicate — a chart point has norm ≥ 1, about 1.0185 on a
# narrow field and 1.41 at 45° off axis, and equals a unit bearing only on the
# optical axis, where the two vectors are the same vector.
#
# Checked here and not in an inner loop: a binding entry point is the boundary,
# and this is one sweep over data the caller just built.
function _assert_unit_bearings(fs, which::AbstractString)
    worst = 0.0; at = 0
    @inbounds for i in eachindex(fs)
        d = abs(sqrt(fs[i][1]^2 + fs[i][2]^2 + fs[i][3]^2) - 1)
        d > worst && (worst = d; at = i)
    end
    worst <= 1.0e-6 || throw(ArgumentError(
        "refine_relpose needs UNIT bearings; the $which at index $at has norm " *
            "$(round(sqrt(fs[at][1]^2 + fs[at][2]^2 + fs[at][3]^2), digits = 6)). " *
            "A z = 1 chart point satisfies the 3-vector signature but is not a " *
            "bearing — its norm is 1 only on the optical axis, ~1.41 at 45° off " *
            "it, and it does not exist at all at 90°. Back-project with the " *
            "camera: `backproject_with_jacobian(camera, u)`."
    ))
    return nothing
end

"""
    refine_relpose(pose::UnitfulEuclideanMap{3}, args...; kw...)

This form is refused with an `ArgumentError`. A calibrated relative pose is
determined only up to scale, so its translation is a direction, not a length:
there is no unit to strip at entry or to reattach at exit. Hand over the plain
seed, `ustrip(m, pose)` converted to a [`Rigid3`](@ref).
"""
refine_relpose(::UnitfulEuclideanMap{3}, args...; kw...) = throw(ArgumentError(
    "refine_relpose: a relative pose is up to scale, so its translation is a direction, not a " *
        "length; supply the plain seed, `convert(PoseLib.Rigid3, ustrip(m, pose))`"
))

"""
    refine_relpose(pose::Rigid3, cs::…StaticVector{2}…; max_iter, loss, loss_width)

This form is refused with an `ArgumentError`. Calibrated 2-vectors are
coordinates on the z = 1 plane (`GnomonicFrame`). That plane has no point for a
ray at or beyond 90° from the optical axis, which excludes fisheye and
omnidirectional cameras. Relative pose is calibrated geometry, so in this stack
it takes unit bearings (unit rays).

Use one of the two supported forms:

- `refine_relpose(pose, cs)`, with `cs` of unit bearings `f₁ => f₂` that carry
  their back-projection Jacobians `M₁` and `M₂`. This is PoseLib's
  tangent-Sampson refiner.
- `refine_relpose(pose, K1, K2, cs)`, with pixel points and the two
  intrinsic matrices. PoseLib back-projects through the cameras and runs the
  same refiner.

`backproject_with_jacobian(camera, u)` returns a bearing and its Jacobian
together.
"""
function refine_relpose(::Rigid3,
                        cs::AbstractVector{<:Cspond{<:StaticVector{2}, <:StaticVector{2}}};
                        kw...)
    throw(ArgumentError(
        "refine_relpose takes unit BEARINGS (3-vectors). Got " *
            "$(eltype(sources(cs))) => $(eltype(targets(cs))). Calibrated 2-vectors " *
            "imply z = 1 — the gnomonic chart — which does not exist for a ray at " *
            "the horizon. Either back-project to bearings and attach the Jacobians, " *
            "`refine_relpose(pose, withattrs(Csponds(f₁, f₂); M₁, M₂); max_iter, loss_width)` " *
            "from `backproject_with_jacobian(camera, u)`, or hand over pixels and " *
            "the intrinsics, `refine_relpose(pose, K1, K2, cs; max_iter, loss_width)`."
    ))
end

"""
    refine_relpose(pose::Rigid3, K1, K2, cs; max_iter, loss = TLSShape(), loss_width) -> Rigid3

Robust Levenberg–Marquardt refinement of a relative pose from pixel
correspondences and the two cameras' intrinsics.

`pose` is the seed [`Rigid3`](@ref), mapping the first camera's frame to the
second's. `cs` holds the correspondences `x1 => x2`, each point in its own
image's frame. `K1` and `K2` are the intrinsic matrices of those frames: pixel
matrices for pixel points, or matrices in whatever frame the points are in.
Returns the refined `Rigid3`.

`loss_width` is in the unit of the points. PoseLib back-projects through each
camera, so one width serves both images even when their focal lengths differ.
`loss` and `loss_width` are as in [`refine_homography`](@ref). `max_iter`, the
iteration budget, has no default.

Each `K` reaches PoseLib as its `SIMPLE_PINHOLE` camera model, with focal
length `(K₁₁ + K₂₂)/2` and principal point `(K₁₃, K₂₃)`. Skew and any
difference between `K₁₁` and `K₂₂` are dropped, so the model is exact only for
square pixels and zero skew.
"""
function refine_relpose(pose::Rigid3, K1::AbstractMatrix, K2::AbstractMatrix,
                        cs::AbstractVector{<:Cspond{<:StaticVector{2}, <:StaticVector{2}}};
                        max_iter::Integer, loss = TLSShape(),
                        loss_width::Union{Real, Nothing} = nothing)
    X1, X2 = _pack(cs)
    R0 = pose.R; t0 = pose.t
    Rin = Vector{Float64}(undef, 9)
    @inbounds for j in 1:3, i in 1:3
        Rin[3(j-1)+i] = R0[i, j]            # column-major (Eigen layout)
    end
    tin  = Float64[t0[1], t0[2], t0[3]]
    cam1 = Float64[(K1[1,1] + K1[2,2]) / 2, K1[1,3], K1[2,3]]   # focal, cx, cy
    cam2 = Float64[(K2[1,1] + K2[2,2]) / 2, K2[1,3], K2[2,3]]
    R_out = MVector{9, Float64}(undef); t_out = MVector{3, Float64}(undef)
    _require_library()
    ccall((:poselib_refine_relpose_cam, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          X1, X2, length(cs), Rin, tin, cam1, cam2, _poselib_loss_scale(loss, loss_width), max_iter, _poselib_loss_type(loss),
          pointer(R_out), pointer(t_out))
    return Rigid3(_read_mat33_from_buf(R_out, 0),
                 Vec3{Float64}(t_out[1], t_out[2], t_out[3]))
end

"""
    refine_relpose(pose::Rigid3, cs; max_iter, loss = TLSShape(), loss_width) -> Rigid3

Robust Levenberg–Marquardt refinement of a relative pose from unit bearings,
through PoseLib's tangent-Sampson refiner.

`pose` is the seed [`Rigid3`](@ref), mapping the first camera's frame to the
second's. `cs` holds the correspondences `f₁ => f₂` of unit bearings
(`StaticVector{3}`). They carry their 3×2 back-projection Jacobians as the
attribute columns `M₁` and `M₂`, attached with `withattrs(cs; M₁, M₂)`.
VisualGeometryToolkitCore's `backproject_with_jacobian(camera, u)` returns
each bearing together with its Jacobian. Returns the refined `Rigid3`.

No image plane is involved, so a ray at or beyond 90° from the optical axis is
an ordinary input. A 3-vector that is not of unit length is refused with an
`ArgumentError`: a point on the z = 1 plane lifted to `(x, y, 1)` has the right
type but is not a bearing.

# The unit of `loss_width`

`loss_width` is the width `w` of the loss on the tangent-Sampson residual

    r = (f₂ᵀ E f₁) / √(‖M₂ᵀ E f₁‖² + ‖M₁ᵀ Eᵀ f₂‖²)

The denominator is the gradient norm of the epipolar constraint with respect
to the coordinates that `M` differentiates against. The residual `r` is in the
unit of those coordinates, and `loss_width` must be given in the same unit.
The Jacobians set the unit; the call cannot.

| `M` is | `r` and `loss_width` are in |
|---|---|
| `∂f/∂u` from `backproject_with_jacobian` | pixels of `u` |
| an orthonormal basis of the tangent plane of the sphere at `f` | radians |

A width in the wrong unit is off by a factor of the focal length. Nothing
reports this: a truncated loss simply keeps the wrong inliers.

All arguments are in the caller's frame, and the refined pose comes back in it.
"""
function refine_relpose(pose::Rigid3,
                        cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{3}}};
                        max_iter::Integer, loss = TLSShape(),
                        loss_width::Union{Real, Nothing} = nothing)
    (hasattr(cs, :M₁) && hasattr(cs, :M₂)) || throw(ArgumentError(
        "refine_relpose on bearings needs the backprojection Jacobians as attribute " *
        "columns: withattrs(cs; M₁, M₂), from backproject_with_jacobian. They set " *
        "the unit of `loss_width`, so there is no defensible default for them."))
    # The 3-vector signature is a NECESSARY condition, not a guarantee: without
    # this the chart would slip through in 3-vector clothing and the 90° exclusion would
    # survive, just invisibly.
    _assert_unit_bearings(sources(cs), "source")
    _assert_unit_bearings(targets(cs), "target")
    D1, D2 = _pack(cs)
    J1 = _pack_tangent(cs.M₁); J2 = _pack_tangent(cs.M₂)
    R0 = pose.R; t0 = pose.t
    Rin = Vector{Float64}(undef, 9)
    @inbounds for j in 1:3, i in 1:3
        Rin[3(j-1)+i] = R0[i, j]            # column-major (Eigen layout)
    end
    tin = Float64[t0[1], t0[2], t0[3]]
    R_out = MVector{9, Float64}(undef); t_out = MVector{3, Float64}(undef)
    _require_library()
    ccall((:poselib_refine_relpose_bearing, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint,
           Ptr{Float64}, Ptr{Float64},
           Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          D1, D2, J1, J2, length(cs), Rin, tin,
          _poselib_loss_scale(loss, loss_width), max_iter, _poselib_loss_type(loss),
          pointer(R_out), pointer(t_out))
    return Rigid3(_read_mat33_from_buf(R_out, 0),
                 Vec3{Float64}(t_out[1], t_out[2], t_out[3]))
end

# =============================================================================
# refine_fundamental  (robust-loss LM refinement of a fundamental matrix)
# =============================================================================

export refine_fundamental

"""
    refine_fundamental(F, cs; max_iter, loss = TLSShape(), loss_width) -> FundMat

Robust Levenberg–Marquardt refinement of a fundamental matrix through
PoseLib's `refine_fundamental`. It minimizes the Sampson error over the
Bartoli–Sturm parametrization of `F`, which keeps the matrix at rank 2.

`F` is the seed 3×3 fundamental matrix. `cs` holds the correspondences
`x1 => x2` of 2D image points, in the frame `F` is expressed in (pixels for a
pixel-frame `F`). `loss_width` is the width of the loss on the Sampson
residual, in the same unit. `loss` and `loss_width` are as in
[`refine_homography`](@ref). Returns the refined `FundMat`.
"""
function refine_fundamental(F::AbstractMatrix,
                            cs::AbstractVector{<:Cspond{<:StaticVector{2}, <:StaticVector{2}}};
                            max_iter::Integer, loss = TLSShape(),
                            loss_width::Union{Real, Nothing} = nothing)
    @assert size(F) == (3, 3)
    X1, X2 = _pack(cs)
    Fin = Vector{Float64}(undef, 9)
    @inbounds for j in 1:3, i in 1:3
        Fin[3(j-1)+i] = F[i, j]                # column-major (Eigen layout)
    end
    F_out = MVector{9, Float64}(undef)
    _require_library()
    ccall((:poselib_refine_fundamental, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64},
           Float64, Cint, Cint, Ptr{Float64}),
          X1, X2, length(cs), Fin, _poselib_loss_scale(loss, loss_width), max_iter, _poselib_loss_type(loss),
          pointer(F_out))
    return FundMat{Float64}(Tuple(_read_mat33_from_buf(F_out, 0)))
end

# =============================================================================
# refine_homography  (robust-loss LM refinement of a homography)
# =============================================================================

export refine_homography
export refine_absolute_pose

"""
    refine_absolute_pose(pose::Rigid3, cs::…StaticVector{2}…; max_iter, loss, loss_width)

This form is refused with an `ArgumentError`. Its targets would be calibrated
points on the z = 1 plane (`GnomonicFrame`), with a loss width divided by the
focal length. That plane has no point for a ray at or beyond 90° from the
optical axis.

Use `refine_absolute_pose(pose, K, cs)` instead: the same correspondences in
pixels, the intrinsic matrix `K`, and `loss_width` in pixels.

PoseLib has no absolute-pose refiner on bearings. Its residual is a
reprojection error in the image, so the pixel form is the only one, and it
reaches PoseLib as a `SIMPLE_PINHOLE` camera. That model cannot project a ray
at or beyond 90° either, so this package offers no correct absolute-pose
refinement for a fisheye camera.
"""
function refine_absolute_pose(::Rigid3,
                              cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{2}}};
                              kw...)
    throw(ArgumentError(
        "refine_absolute_pose no longer takes calibrated 2-vectors. Got " *
            "$(eltype(sources(cs))) => $(eltype(targets(cs))); the targets were " *
            "points on the z = 1 chart, which does not exist for a ray at the " *
            "horizon, with the loss width pre-divided by the focal length. Supply the same " *
            "correspondences as PIXELS with their intrinsics and a PIXEL loss width: " *
            "`refine_absolute_pose(pose, K, cs; max_iter, loss_width)`."
    ))
end

"""
    refine_absolute_pose(pose::Rigid3, K, cs; max_iter, loss = TLSShape(), loss_width) -> Rigid3

Robust Levenberg–Marquardt refinement of an absolute pose from 3D-to-2D
correspondences, through PoseLib's `bundle_adjust`. The residual is the
reprojection error `project(camera, R X + t) − u`.

`pose` is the seed [`Rigid3`](@ref), mapping the world frame to the camera's.
`K` is the intrinsic matrix of the frame the image points are in. `cs` holds
the correspondences `X => u` of 3D points to 2D image points. Returns the
refined `Rigid3`.

`loss_width` is the width of the loss on the reprojection residual, in the
unit of `u` and `K` (pixels for a pixel-frame `K`). `loss` and `loss_width`
are as in [`refine_homography`](@ref). Only the six degrees of freedom of the
pose are refined; `K` stays fixed.

`K` reaches PoseLib as its `SIMPLE_PINHOLE` camera model, with focal length
`(K₁₁ + K₂₂)/2` and principal point `(K₁₃, K₂₃)`. Skew and any difference
between `K₁₁` and `K₂₂` are dropped, so the model is exact only for square
pixels and zero skew.
"""
function refine_absolute_pose(pose::Rigid3, K::AbstractMatrix,
                              cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{2}}};
                              max_iter::Integer, loss = TLSShape(),
                              loss_width::Union{Real, Nothing} = nothing)
    @assert size(K) == (3, 3)
    Xb, xb = _pack(cs)
    R0 = pose.R; t0 = pose.t
    Rin = Vector{Float64}(undef, 9)
    @inbounds for j in 1:3, i in 1:3
        Rin[3(j-1)+i] = R0[i, j]            # column-major (Eigen layout)
    end
    tin = Float64[t0[1], t0[2], t0[3]]
    cam = Float64[(K[1,1] + K[2,2]) / 2, K[1,3], K[2,3]]        # focal, cx, cy
    R_out = MVector{9, Float64}(undef); t_out = MVector{3, Float64}(undef)
    _require_library()
    ccall((:poselib_bundle_adjust_cam, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          xb, Xb, length(cs), Rin, tin, cam,
          _poselib_loss_scale(loss, loss_width), max_iter, _poselib_loss_type(loss),
          pointer(R_out), pointer(t_out))
    return Rigid3(_read_mat33_from_buf(R_out, 0),
                 Vec3{Float64}(t_out[1], t_out[2], t_out[3]))
end

# -----------------------------------------------------------------------------
# Physical poses and world points: one strip at entry, one reattachment at exit
# -----------------------------------------------------------------------------
#
# The layering (VisualGeometryToolkitCore `conventions/units.md`): the ccall
# boundary takes plain Float64, so a physical seed and physical world points
# are stripped in metres once (`ustrip(m, ·)` of a base-form quantity is its
# stored value, exactly), the plain refinement runs, and the refined pose is
# read back in metres once (scale 1, exact). Mixed forms are refused.

const _PhysicalPoints3 = AbstractVector{<:Cspond{<:StaticVector{3, <:Quantity}, <:StaticVector{2}}}
const _PlainPoints3 = AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{2}}}

_metres_point(X::StaticVector{3, <:Quantity}) = SVector{3, Float64}(VgtkCore.strip_metres(X))

"""
    refine_absolute_pose(pose::UnitfulEuclideanMap{3}, K, cs; max_iter, loss = TLSShape(), loss_width)

The physical form of [`refine_absolute_pose`](@ref): `pose` a world→camera
pose whose translation is a length (FlexUnits quantities) and `cs` the
correspondences `X => u` with the world points `X` lengths. Both are stripped in
metres once, the plain refinement runs, and the refined pose comes back with its
translation in metres: equal, after `ustrip(m, ·)`, to the plain refinement's
on the inputs' `ustrip(m, ·)`. `u`, `K` and `loss_width` stay in pixels. A
labeled seed (an estimate's `CameraExtrinsics`, world → camera) is read
unlabeled; the refined pose comes back unlabeled. A physical seed with plain
points, or the reverse, is an `ArgumentError`.
"""
function refine_absolute_pose(pose::UnitfulEuclideanMap{3}, K::AbstractMatrix, cs::_PhysicalPoints3; kwargs...)
    seed = convert(Rigid3, VgtkCore.unlabel(ustrip(VgtkCore.m, pose)))   # a labeled pose (an estimate's CameraExtrinsics) too
    plain = [_metres_point(first(c)) => last(c) for c in cs]
    refined = refine_absolute_pose(seed, K, plain; kwargs...)
    # rebuilt through the checked constructor (the refined rotation is verified, as
    # before), then read in metres
    return VgtkCore.with_unit(EuclideanMap(refined.R, refined.t), VgtkCore.m)
end

refine_absolute_pose(::UnitfulEuclideanMap{3}, ::AbstractMatrix, ::_PlainPoints3; kwargs...) = throw(ArgumentError(
    "refine_absolute_pose: the seed is physical but the world points are plain; attach their unit (X .* m)"
))
refine_absolute_pose(::Rigid3, ::AbstractMatrix, ::_PhysicalPoints3; kwargs...) = throw(ArgumentError(
    "refine_absolute_pose: the world points are physical but the seed is plain; attach its unit, " *
        "EuclideanMap(pose.R, pose.t .* m)"
))

"""
    PoseLib.bundle_adjust(pose, K, cs; max_iter, loss, loss_width)

An alias of [`refine_absolute_pose`](@ref) under PoseLib's own name for
absolute-pose refinement. It is not exported. It takes
`refine_absolute_pose`'s arguments (the pose first, then the `X => u`
correspondences), not upstream's `(x, X, pose)` order.
"""
const bundle_adjust = refine_absolute_pose

"""
    refine_homography(H, cs; max_iter, loss = TLSShape(), loss_width) -> HomographyMat

Robust Levenberg–Marquardt refinement of a homography through PoseLib's
`refine_homography`. It minimizes the symmetric transfer error: the forward
residual `x2 − π(H x1)` and the backward residual `x1 − π(H⁻¹ x2)`, each
robustified per point, where `π` divides by the third coordinate.

`H` is the seed 3×3 homography. `cs` holds the correspondences `x1 => x2` of
2D image points, in the frame `H` is expressed in (pixels for a pixel-frame
`H`). `loss_width` is the width of the loss on a one-sided transfer residual,
in the same unit. Returns the refined `HomographyMat`.

`loss` is one of VisualGeometryToolkitCore's loss shapes, or
[`LeZachTruncation`](@ref)`()`:

| `loss` | `ρ(r)` | note |
|---|---|---|
| `TLSShape()` (default) | `min(r², w²)` | truncated quadratic: inliers have `r < w`, outliers get zero weight |
| `HuberShape()` | `r²` up to `w`, linear beyond | |
| `CauchyShape()` | `w² log(1 + r²/w²)` | |
| `GaussianShape()` | `r²` | plain least squares; takes no `loss_width` |
| `LeZachTruncation()` | `min(r², w²)` | the truncated loss under Le and Zach's scheme |

`loss_width` is the width `w` of VisualGeometryToolkitCore's closed form for
that shape. PoseLib's own `loss_scale` equals it for every loss bound here, so
it reaches PoseLib unchanged. `max_iter`, the iteration budget, has no default.
"""
function refine_homography(H::AbstractMatrix,
                           cs::AbstractVector{<:Cspond{<:StaticVector{2}, <:StaticVector{2}}};
                           max_iter::Integer, loss = TLSShape(),
                           loss_width::Union{Real, Nothing} = nothing)
    @assert size(H) == (3, 3)
    X1, X2 = _pack(cs)
    Hin = Vector{Float64}(undef, 9)
    @inbounds for j in 1:3, i in 1:3
        Hin[3(j-1)+i] = H[i, j]                # column-major (Eigen layout)
    end
    H_out = MVector{9, Float64}(undef)
    _require_library()
    ccall((:poselib_refine_homography, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64},
           Float64, Cint, Cint, Ptr{Float64}),
          X1, X2, length(cs), Hin, _poselib_loss_scale(loss, loss_width), max_iter, _poselib_loss_type(loss),
          pointer(H_out))
    return HomographyMat{Float64}(Tuple(_read_mat33_from_buf(H_out, 0)))
end

# =============================================================================
# motion(::HomographyMat) — moved to VisualGeometryToolkitCore
# =============================================================================
# The Malis–Vargas homography decomposition is now a NATIVE method in
# VisualGeometryToolkitCore (src/geometry/fundmat.jl), ported off the C++
# `poselib_motion_from_homography` (validated bit-for-bit against it). PoseLib no
# longer defines `motion(::HomographyMat)`, so the VGC method is authoritative.
# The C wrapper symbol stays in the shared library but is unused from Julia.


end # module PoseLib
