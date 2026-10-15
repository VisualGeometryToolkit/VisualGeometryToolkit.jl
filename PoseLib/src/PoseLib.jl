"""
    PoseLib

Julia bindings to the PoseLib C++ minimal-solver library.

Extends `VisualGeometryCore` with PoseLib-backed methods. Loading this
package adds:

- Backend overloads (`::PoseLibSolver`) for shared solvers:
  `homography_4pt`, `fundamental_7pt`, `p3p`, `essential_5pt`.
- A PoseLib-only solver (no native variant): `essential_dlt`, whose
  convenience default routes to the PoseLib backend when no
  `AbstractSolverBackend` is supplied.
- Robust-loss LM refinement, model first and correspondences as one
  Cspond vector: `refine_relpose`, `refine_fundamental`,
  `refine_homography`, `refine_absolute_pose`. The loss is a
  VisualGeometryCore loss shape (`TLSShape()`, `HuberShape()`,
  `CauchyShape()`, `GaussianShape()`) or `LeZachTruncation()`, at the
  `loss_width` the shape's own closed forms use.
  The calibrated pair each take the caller's own units, and REFUSE the z = 1
  chart: `refine_relpose(pose, cs)` on unit BEARINGS carrying their `M₁`/`M₂`
  backprojection Jacobians (PoseLib's tangent Sampson — no chart, so a ray at
  or beyond 90° is ordinary), and `refine_absolute_pose(pose, K, cs)` on pixel
  points with a pixel loss width. Handing either a calibrated 2-vector raises
  an `ArgumentError` naming the form to use. Each docstring names the unit its
  `loss_width` is in, and what sets it.

# Binary

The C++ library and its flat C wrapper (`wrapper/julia_wrapper.cpp`) ship
prebuilt as this package's own `PoseLib` artifact (`Artifacts.toml`; Linux only
until the `PoseLib_jll` release), built from
the recipe `yggdrasil/P/PoseLib/build_tarballs.jl` by
`scripts/build_binaries.jl`. To run against a library built from source,
override the artifact in the depot's `Overrides.toml` (see the README,
"Building the wrapper from source").
"""
module PoseLib

using StaticArrays: MVector, SVector, SMatrix, StaticVector, StaticMatrix
using Rotations: RotMatrix
using Artifacts: @artifact_str
using Libdl: dlopen, dlext
using CompilerSupportLibraries_jll: CompilerSupportLibraries_jll  # libstdc++/libgcc_s, as the recipe declares

using VisualGeometryCore: VisualGeometryCore,
    AbstractLossShape, GaussianShape, TLSShape, HuberShape, CauchyShape,
    HomographyMat, FundMat, EssentialMat, Extrinsics3, FixedModels,
    EuclideanMap, UnlabeledFrame,
    PoseLibSolver, Vec3, Point3, Cspond, sources, targets, hasattr,
    sign_normalize, is_valid_solution,
    homography_4pt, fundamental_7pt, p3p,
    essential_5pt, essential_dlt, motion

# Every `ccall` goes to `libposelib_jl`, the library in this package's `PoseLib`
# artifact, resolved at load time so a relocated depot or an `Overrides.toml`
# entry (README, "Building the wrapper from source") is honoured; nothing here
# probes. The artifact is Linux-only until the `PoseLib_jll` release (README,
# "Installation"); elsewhere loading stops here, with that reason.
libposelib_jl::String = ""

function __init__()
    @static if Sys.islinux()
        global libposelib_jl = joinpath(artifact"PoseLib", "lib", "libposelib_jl.$(dlext)")
        dlopen(libposelib_jl)
    else
        error("PoseLib.jl binaries are Linux-only until the PoseLib_jll release; see README")
    end
    return nothing
end

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
    homography_4pt(::PoseLibSolver, s₁, s₂, s₃, s₄, d₁, d₂, d₃, d₄)

PoseLib backend for 4-point homography. Takes SVector{2} (inhomogeneous) points.
"""
function VisualGeometryCore.homography_4pt(::PoseLibSolver,
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
    n = ccall((:poselib_homography_4pt, libposelib_jl), Cint,
              (Ref{NTuple{12,Float64}}, Ref{NTuple{12,Float64}},
               Ref{NTuple{9,Float64}}, Cint),
              Ref(p1), Ref(p2), H_buf, 1)
    n > 0 || return nothing
    H = SMatrix{3,3,Float64,9}(H_buf[]...)
    all(isfinite, H) || return nothing
    sign_normalize(HomographyMat{Float64}(Tuple(H)))
end

function VisualGeometryCore.homography_4pt(::PoseLibSolver, cs, idx::AbstractVector{<:Integer})
    u1 = cs.first; u2 = cs.second
    @inbounds VisualGeometryCore.homography_4pt(PoseLibSolver(),
        u1[idx[1]], u1[idx[2]], u1[idx[3]], u1[idx[4]],
        u2[idx[1]], u2[idx[2]], u2[idx[3]], u2[idx[4]])
end

# =============================================================================
# fundamental_7pt  (PoseLib backend)
# =============================================================================

function VisualGeometryCore.fundamental_7pt(::PoseLibSolver, cs, idx::AbstractVector{<:Integer})
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

function VisualGeometryCore.p3p(::PoseLibSolver,
             x::AbstractVector{S},
             X::AbstractVector{T}) where {S <: StaticVector{3,Float64},
                                          T <: StaticVector{3,Float64}}
    length(x) == 3 && length(X) == 3 || return nothing
    px = _pack_points3((Vec3{Float64}(x[1]),   Vec3{Float64}(x[2]),   Vec3{Float64}(x[3])))
    pX = _pack_points3((Point3{Float64}(X[1]), Point3{Float64}(X[2]), Point3{Float64}(X[3])))
    pose_buf = MVector{48, Float64}(undef)
    n = ccall((:poselib_p3p, libposelib_jl), Cint,
              (Ref{NTuple{9,Float64}}, Ref{NTuple{9,Float64}},
               Ptr{Float64}, Cint),
              Ref(px), Ref(pX), pointer(pose_buf), 4)
    FixedModels{4, Extrinsics3}(n) do i
        _read_pose_from_buf(pose_buf, 12*(i-1))
    end
end

# No-backend p3p default now lives in VisualGeometryCore (NativeSolver, its
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

function VisualGeometryCore.essential_5pt(::PoseLibSolver, x1::_Bearings5, x2::_Bearings5)
    bufs = _relpose5pt_buffers()
    bufs.p1[] = _pack_points3(Tuple(x1))
    bufs.p2[] = _pack_points3(Tuple(x2))
    E_buf = bufs.E
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
function VisualGeometryCore.essential_5pt(::PoseLibSolver,
        cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{3}}},
        idx::AbstractVector{<:Integer})
    u1 = sources(cs); u2 = targets(cs)
    x1 = @inbounds ntuple(k -> Vec3{Float64}(u1[idx[k]]), Val(5))
    x2 = @inbounds ntuple(k -> Vec3{Float64}(u2[idx[k]]), Val(5))
    VisualGeometryCore.essential_5pt(PoseLibSolver(), x1, x2)
end

# No-backend essential_5pt default now lives in VisualGeometryCore
# (NativeSolver port of relpose_5pt.cc). Select this backend explicitly:
# essential_5pt(PoseLibSolver(), x1, x2).

# =============================================================================
# essential_dlt  (PoseLib only — backend method + no-backend default)
# =============================================================================

function VisualGeometryCore.essential_dlt(::PoseLibSolver,
        cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{3}}})
    n = length(cs)
    @assert n >= 8
    buf1, buf2 = _pack(cs)
    E_buf = MVector{9, Float64}(undef)
    ccall((:poselib_essential_8pt, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}),
          buf1, buf2, n, pointer(E_buf))
    all(isfinite, E_buf) || return nothing
    EssentialMat{Float64}(Tuple(E_buf))
end

@inline VisualGeometryCore.essential_dlt(cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{3}}}) =
    VisualGeometryCore.essential_dlt(PoseLibSolver(), cs)

# =============================================================================
# refine_relpose  (robust-loss LM refinement of a relative pose — PoseLib LO)
# =============================================================================

export refine_relpose
export LeZachTruncation

# Relative-pose currency: a cam1→cam2 motion is NOT a world-to-camera
# extrinsics, so `Extrinsics3`'s WorldFrame→CameraFrame labels would be
# semantically wrong here. File-local UNLABELED concrete alias (same layout
# as the retired VGC `Pose3`). VGC's view-identity frames (`ViewFrame{id}`)
# could label it; that migration has not been done.
const Rigid3 = EuclideanMap{
    3, Float64, RotMatrix{3, Float64, 9}, Vec3{Float64},
    UnlabeledFrame{3}, UnlabeledFrame{3},
}

# -----------------------------------------------------------------------------
# The loss, typed: a VisualGeometryCore loss shape (or the Le–Zach marker),
# mapped to PoseLib's `BundleOptions::LossType` enum (PoseLib/types.h) ONLY at
# the ccall boundary. The integer enum never leaves this file.
#
# THE WIDTH. Each PoseLib loss is built from one scalar `loss_scale`
# (PoseLib/robust/robust_loss.h), and for every loss bound here that scalar IS
# the `loss_width` w of VisualGeometryCore's loss-shape closed forms,
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

The truncated loss `ρ = min(r², w²)` (VisualGeometryCore's `TLSShape` at
`loss_width = w`) optimized by the TR-IRLS scheme of Le & Zach (3DV 2021),
PoseLib's `TRUNCATED_LE_ZACH`. A marker, not a loss shape: the LOSS is
`TLSShape`'s, the marker selects the optimization scheme.
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

# Unit-norm gate for a bearing column. Matches RobustVisualGeometry's
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
    refine_relpose(pose::Rigid3, cs::…StaticVector{2}…; max_iter, loss, loss_width)

**Refused.** Calibrated 2-vectors are coordinates on the z = 1 chart
(`GnomonicFrame`), and that chart has no point for a ray at or beyond 90° — the
rays this stack exists to calibrate. Relative pose is calibrated geometry, so
it takes unit bearings (unit rays).

Use one of the two forms that do exist:

- `refine_relpose(pose, cs)` with `cs` of **unit bearings** `f₁ => f₂` carrying
  their `M₁`/`M₂` backprojection Jacobians — PoseLib's tangent-Sampson refiner.
- `refine_relpose(pose, K1, K2, cs)` with **pixel** points and the two
  intrinsics, which back-projects through the cameras inside PoseLib and
  reduces to exactly the same refiner.

`backproject_with_jacobian(camera, u)` produces the bearing and its Jacobian
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

**Camera-aware** robust-loss refinement (truncated by default: MSAC) (matches the RANSAC-Scoring reference):
`pose` is the seed `Rigid3`, `cs` the correspondences `x1 => x2` in each
image's frame, and `K1`/`K2` the intrinsics of THAT frame — pixels for pixel
`K`, or any frame `K` is expressed in (LocalOptimizationDoneRight's refit layer supplies
the pool frame's `Kc = T·K`). `loss_width` is the loss's width in the unit of that
same frame. PoseLib calibrates internally through the cameras, so one width is
valid even when the two focals differ. `loss` and `loss_width` are as in
[`refine_homography`](@ref); `max_iter` (the LM iteration budget) is the
caller's configuration and has no default.

Each `K` reaches C++ as a SIMPLE_PINHOLE camera: focal `(K₁₁ + K₂₂)/2` and
principal point `(K₁₃, K₂₃)`; the skew and any `K₁₁ ≠ K₂₂` are dropped, so the
model is exact only for a square-pixel, zero-skew `K`.
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

**Bearing** robust-loss (truncated by default: MSAC) Levenberg–Marquardt refinement of a relative
pose, through PoseLib's tangent-Sampson refiner
(`refine_relpose(d1, d2, M1, M2, &pose, opt)`, `bundle.cc:228` —
`FixCameraRelativePoseRefiner`). `pose` is the seed `Rigid3` and `cs` the
correspondences `f₁ => f₂` of **unit bearings** (`StaticVector{3}`), carrying
their 3×2 backprojection Jacobians as the attribute COLUMNS `M₁`/`M₂`
(`withattrs(cs; M₁, M₂)`) — the pair
`VisualGeometryCore.backproject_with_jacobian(model, u)` returns. Returns the
refined `Rigid3`.

This is the calibrated entry point with no chart: a ray at or beyond 90° is an
ordinary bearing here, where the z = 1 chart (`GnomonicFrame`) has no
coordinate to name it with — which is why `StaticVector{2}` endpoints are
refused, and why a 3-vector that is not unit-norm is refused too (a chart point
lifted to `(x, y, 1)` satisfies the signature but is not a bearing).

# The unit of `loss_width`

`loss_width` is the loss's width w (for the default truncation: inliers
`r < w`, outliers zero-weighted) on
the tangent-Sampson residual

    r = (f₂ᵀ E f₁) / √(‖M₂ᵀ E f₁‖² + ‖M₁ᵀ Eᵀ f₂‖²)

whose denominator is exactly `‖∇C‖` of the epipolar constraint taken with
respect to the coordinate `M` DIFFERENTIATES AGAINST. So `r` carries that
coordinate's unit, and `loss_width` must be given in the same one: **`M` picks the unit
and the call cannot.**

| `M` is | `r` and `loss_width` are in |
|---|---|
| `∂f/∂u` from `backproject_with_jacobian` (the stack's producer) | **pixels** of `u` |
| an orthonormal basis of the tangent plane `T_f S²` | **radians** |

Bearings are unitless, so nothing else can set the scale: supplying a width scaled as
if the bearings carried one is wrong by a factor of the focal length, and fails
silently — a truncated loss simply keeps the wrong inliers.

# Frames

Binding layer: `pose`, `cs`, `M₁`/`M₂` and `loss_width` are all in the caller's
frame and the refined `Rigid3` comes back in it. Nothing here normalizes; the
Jacobians are the caller's own, and whichever camera produced them fixes both
the frame and the width's unit.
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

Robust-loss (truncated by default: MSAC) Levenberg–Marquardt refinement of a
fundamental matrix via
PoseLib's `refine_fundamental` (Sampson error, optimized over the Bartoli–Sturm
SVD factorization of `F`).  `F` is the seed 3×3 fundamental matrix, `cs` the
correspondences `x1 => x2` of 2D image points (same coordinate frame as `F` —
pixels for a pixel-frame `F`), and `loss_width` the loss's width on the
Sampson residual.  `loss` and `loss_width` are as in
[`refine_homography`](@ref).  Returns the refined `FundMat`.
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

**Refused.** The 2-vectors were coordinates on the z = 1 chart
(`GnomonicFrame`) — "calibrated image points" — with the loss width pre-divided by the focal
length. PnP is calibrated geometry, and the chart has no point for a ray at or
beyond 90°, so this shape is the one the stack is removing.

Use `refine_absolute_pose(pose, K, cs)`: the same correspondences as **pixels**
with their intrinsics, and `loss_width` in pixels. Nothing is pre-divided, and the frame
the caller hands over is the frame the result comes back in.

# The gap this leaves, stated rather than papered over

PoseLib has no absolute-pose refiner on bearings — every `bundle_adjust`
overload in `bundle.h` takes `Point2D`, because the residual is a reprojection
into an image, not an angle on the sphere. So the pixel form above is the only
absolute-pose path here, and it crosses as SIMPLE_PINHOLE, which cannot project
a ray at or beyond 90° either. A fisheye caller therefore has no correct
absolute-pose refinement in this package — which was equally true before, and
is now visible instead of hidden behind a chart that silently mis-measured it.
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

**Camera-aware** absolute-pose bundle adjustment — PoseLib's
`bundle_adjust(x, X, &image, opt)` (`bundle.cc:105`, the `Image` overload),
whose `AbsolutePoseRefiner` forms the residual as
`project(camera, R X + t) − x` through the camera's OWN projection. `pose` is
the seed world→camera pose, `K` the intrinsics of the frame `cs`'s image points
are in, and `cs` the correspondences `X => u` of 3D points to 2D image points
in that same frame. Returns the refined `Rigid3`.

`loss_width` is the loss's width on that reprojection residual, **in the
units of `u` and `K` — pixels for a pixel-frame `K`**; `loss` and `loss_width`
are as in [`refine_homography`](@ref). Nothing is pre-divided on the caller's behalf; this is the only
absolute-pose form, the calibrated z = 1 one having been withdrawn.

`K` is NOT refined: `BundleOptions` leaves `refine_focal_length`,
`refine_principal_point` and `refine_extra_params` false, so only the 6-DOF
pose moves.

`K` reaches C++ as a SIMPLE_PINHOLE camera: focal `(K₁₁ + K₂₂)/2` and principal
point `(K₁₃, K₂₃)`; the skew and any `K₁₁ ≠ K₂₂` are dropped, so the model is
exact only for a square-pixel, zero-skew `K` — the same lossy crossing
[`refine_relpose`](@ref) documents.

Binding layer: `pose`, `cs`, `K` and `loss_width` are all in the caller's frame
and the result comes back in it; nothing here normalizes.
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
    ccall((:poselib_bundle_adjust_cam, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          xb, Xb, length(cs), Rin, tin, cam,
          _poselib_loss_scale(loss, loss_width), max_iter, _poselib_loss_type(loss),
          pointer(R_out), pointer(t_out))
    return Rigid3(_read_mat33_from_buf(R_out, 0),
                 Vec3{Float64}(t_out[1], t_out[2], t_out[3]))
end

"Upstream-faithful alias (PoseLib names absolute-pose refinement `bundle_adjust`); unexported. Takes `refine_absolute_pose`'s arguments (pose first, then the `X => x` csponds), not upstream's `(x, X, pose)`."
const bundle_adjust = refine_absolute_pose

"""
    refine_homography(H, cs; max_iter, loss = TLSShape(), loss_width) -> HomographyMat

Robust-loss (truncated by default: MSAC) Levenberg–Marquardt refinement of a
homography via
PoseLib's `refine_homography` (symmetric transfer error — forward
`|x2 - π(H x1)|` and backward `|x1 - π(adj(H) x2)|`, each side robustified
per point).  `H` is the seed 3×3 homography, `cs` the correspondences
`x1 => x2` of 2D image points (same coordinate frame as `H` — pixels for a
pixel-frame `H`), and `loss_width` the loss's width on the one-sided transfer
residual.  Returns the refined `HomographyMat`.

`loss` is a VisualGeometryCore loss shape — `TLSShape()` (the default: the
truncation `min(r², w²)`, inliers `r < w` and outliers zero-weighted),
`HuberShape()`, `CauchyShape()`, `GaussianShape()` (plain least squares, which
takes no `loss_width`) — or [`LeZachTruncation`](@ref)`()`, the same
truncation optimized by the Le & Zach TR-IRLS scheme. `loss_width` is the
width `w` of that shape's closed forms (`ρ(shape, r², w²)`); PoseLib's
`loss_scale` equals it for every bound loss, so it crosses unchanged.
`max_iter`, the LM iteration budget, is the caller's configuration: it has no
default.
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
    ccall((:poselib_refine_homography, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64},
           Float64, Cint, Cint, Ptr{Float64}),
          X1, X2, length(cs), Hin, _poselib_loss_scale(loss, loss_width), max_iter, _poselib_loss_type(loss),
          pointer(H_out))
    return HomographyMat{Float64}(Tuple(_read_mat33_from_buf(H_out, 0)))
end

# =============================================================================
# motion(::HomographyMat) — moved to VisualGeometryCore
# =============================================================================
# The Malis–Vargas homography decomposition is now a NATIVE method in
# VisualGeometryCore (src/geometry/fundmat.jl), ported off the C++
# `poselib_motion_from_homography` (validated bit-for-bit against it). PoseLib no
# longer defines `motion(::HomographyMat)`, so the VGC method is authoritative.
# The C wrapper symbol stays in the shared library but is unused from Julia.


end # module PoseLib
