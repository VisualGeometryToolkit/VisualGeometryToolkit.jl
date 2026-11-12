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

The C++ library and its flat C wrapper ship prebuilt as the binary package
`PoseLib_jll`. The manual's "Platforms and binaries" page says how to run
against a library built from source.
"""
module PoseLib
    import VisualGeometryToolkitCore as VgtkCore

using StaticArrays: MVector, SVector, SMatrix, StaticVector, StaticMatrix
using Rotations: RotMatrix
using PoseLib_jll: PoseLib_jll, libposelib_jl
using FlexUnits: Quantity

using VisualGeometryToolkitCore: VisualGeometryToolkitCore,
    AbstractLossShape, GaussianShape, TLSShape, HuberShape, CauchyShape,
    Irls, Triggs, EfficiencyConstant, FitResult, AbstractOptimizationStrategy,
    HomographyMat, FundMat, EssentialMat, Extrinsics3, FixedModels,
    EuclideanMap, UnlabeledFrame, UnitfulEuclideanMap, ustrip,
    PoseLibSolver, Vec3, Point3, Cspond, sources, targets, hasattr,
    sign_normalize, is_valid_solution,
    homography_4pt, fundamental_7pt, p3p,
    essential_5pt, essential_dlt, motion

# Every `ccall` goes to `libposelib_jl`, the library product of `PoseLib_jll`.
# On a platform Yggdrasil does not build for, the package still loads — so does
# everything that depends on it — and [`is_available`](@ref) is false: the first
# call that needs the library raises the reason (`_require_library`).
const _UNAVAILABLE = "PoseLib_jll has no build of the PoseLib library for this platform; see https://visualgeometrytoolkit.github.io/PoseLib/stable/guide/binaries/"

"""
    is_available() -> Bool

Whether `PoseLib_jll` provides the PoseLib library on this platform. Without
it, every function that calls into the library raises an error stating why;
the package itself loads everywhere.
"""
is_available() = PoseLib_jll.is_available()

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

"""
    fundamental_7pt(::PoseLibSolver, cs, idx) -> FixedModels{3, FundMat{Float64}}

PoseLib's seven-point fundamental-matrix solver, the `PoseLibSolver` backend of
VisualGeometryToolkitCore's `fundamental_7pt` kernel. `cs` holds 2D image
correspondences `x1 => x2` in the caller's frame and `idx` names the seven of
the sample. Returns every real solution, up to three, in that frame; a
degenerate sample returns fewer, possibly none. The public entry point is
`solve(Fundamental7pt(PoseLibSolver()), cs, idx)`.
"""
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

"""
    p3p(::PoseLibSolver, x, X) -> Union{FixedModels{4, Extrinsics3}, Nothing}

PoseLib's perspective-three-point solver, the `PoseLibSolver` backend of
VisualGeometryToolkitCore's `p3p` kernel. `x` holds three unit bearings
(`StaticVector{3, Float64}`) and `X` the three world points they see. Returns
every real solution, up to four, as world-to-camera extrinsics in the frame of
`X` (plain `Float64` points; the translation is in their unit); a degenerate sample returns fewer, possibly none, and inputs of
any length other than three return `nothing`. The public entry point is
`solve(P3P(PoseLibSolver()), cs, idx)` with `cs` the correspondences `X => x`.
"""
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
const _Float64Bearings5 = Union{NTuple{5, <:StaticVector{3, Float64}},
                         StaticVector{5, <:StaticVector{3, Float64}}}

"""
    essential_5pt(::PoseLibSolver, x1, x2) -> FixedModels{10, EssentialMat{Float64}}
    essential_5pt(::PoseLibSolver, cs, idx) -> FixedModels{10, EssentialMat{Float64}}

PoseLib's five-point essential-matrix solver, the `PoseLibSolver` backend of
VisualGeometryToolkitCore's `essential_5pt` kernel. `x1` and `x2` are five unit
bearings each (an `NTuple` or an `SVector{5}` of `StaticVector{3, Float64}`);
the second form reads the five correspondences `cs[idx]` of bearings
`f₁ => f₂`. A 2-vector is a `MethodError`, never lifted to the z = 1 plane.
Returns every real solution that VisualGeometryToolkitCore's
`is_valid_solution` accepts, up to ten; a degenerate sample returns fewer, possibly
none. The public entry point is `solve(Essential5pt(PoseLibSolver()), cs, idx)`.
"""
function VgtkCore.essential_5pt(::PoseLibSolver, x1::_Float64Bearings5, x2::_Float64Bearings5)
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

"""
    essential_dlt(::PoseLibSolver, cs) -> Union{EssentialMat{Float64}, Nothing}
    essential_dlt(cs) -> Union{EssentialMat{Float64}, Nothing}

PoseLib's eight-point essential matrix: the direct linear transform over eight
or more correspondences `cs` of unit bearings `f₁ => f₂`, every one taken as an
inlier. VisualGeometryToolkitCore has no native eight-point solver, so the form
without a backend runs this one. An algorithm, so it can only be degenerate:
it returns `nothing` when PoseLib's solution is not finite, but a degenerate
set (a pure rotation, repeated points) can also give a finite, meaningless `E`
that nothing flags. Fewer than eight correspondences throw an `ArgumentError`
naming the count.
"""
function VgtkCore.essential_dlt(::PoseLibSolver,
        cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{3}}})
    n = length(cs)
    n >= 8 || throw(ArgumentError("essential_dlt needs at least 8 correspondences, got $n"))
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

# The refiners' 3×3 seed check: an `ArgumentError` naming the size it got, run
# before the matrix is read under `@inbounds` into the ccall's buffer.
_require_3x3(A::AbstractMatrix, name::String) =
    size(A) == (3, 3) || throw(ArgumentError("$name must be 3×3, got $(join(size(A), '×'))"))

# =============================================================================
# refine_relpose  (robust-loss LM refinement of a relative pose — PoseLib LO)
# =============================================================================

export refine_relpose
public is_available
export LeZach

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
# The strategy, typed: the stack's robust-estimation contract names a refiner's
# loss as its strategy, `optimization_strategy`, and its width by `scale` or `width`.
# Each strategy PoseLib runs maps to PoseLib's `BundleOptions::LossType` enum
# (PoseLib/types.h) ONLY at the ccall boundary; the integer enum never leaves
# this file.
#
# PoseLib's Levenberg–Marquardt weighs each residual by its loss weight ρ′ alone
# (robust/optim/jacobian_accumulator.h: J'WJ with W = loss.weight(r²), no ρ″
# term), so its strategy is `Irls{K}`. A `Triggs{K}` strategy (the ρ″-corrected
# LM) is refused: PoseLib does not run it.
#
# THE WIDTH. Each PoseLib loss is built from one scalar `loss_scale`
# (PoseLib/robust/robust_loss.h), and for every loss bound here that scalar IS
# the width w of VisualGeometryToolkitCore's loss-shape closed forms, s = r²:
#
#   TruncatedLoss(w)        ρ = min(s, w²),  weight = s < w²         ≡ ρ(TLSShape(), s, w²)
#   TruncatedLossLeZach(w)  ρ = min(s, w²)  (TR-IRLS weights, μ = ½) — the TLS width
#   HuberLoss(w)            ρ = s ≤ w² ? s : w(2r − w), weight = min(1, w/r)  ≡ ρ(HuberShape(), s, w²)
#   CauchyLoss(w)           ρ = w²·log1p(s/w²), weight = 1/(1 + s/w²)          ≡ ρ(CauchyShape(), s, w²)
#   TrivialLoss             ρ = s, no width                                    ≡ least squares (`nothing`)
#
# So no factor, square or ½ is applied: `width = w` reaches PoseLib unchanged,
# and test/test_refine.jl pins it behaviourally (a truncation at w, not w² or
# w/2; Huber's elbow at w). `scale = σ` runs at `efficiency_constant · σ`.
# PoseLib's TruncatedCauchyLoss is not bound: no stack caller uses it.
# -----------------------------------------------------------------------------

"""
    LeZach() <: AbstractOptimizationStrategy

PoseLib's own optimization strategy, given as `optimization_strategy = LeZach()`
or selected from TOML as `type = "le_zach"` (its tag in [`POSELIB_STRATEGY_TAGS`](@ref)):
the truncated loss `ρ = min(r², w²)` (VisualGeometryToolkitCore's `TLSShape()`),
optimized by the iteratively reweighted least-squares scheme of Le and Zach
(International Conference on 3D Vision, 2021). This is PoseLib's
`TRUNCATED_LE_ZACH` loss. It selects how the truncated loss is optimized, so it
takes the same `width` as `Irls{TLSShape}`. Like TLS it has no efficiency
constant, so it takes `width`, not `scale`. A member of the stack's one
optimization-strategy family: its loss family is TLS's, its path IRLS and its
label `TLS-LZ`.
"""
struct LeZach <: AbstractOptimizationStrategy end

# The registry of PoseLib's strategies, through VisualGeometryToolkitCore's one
# machinery (`@lo_registry`). LeZach has no fields, so its `[optimization_strategy.le_zach]`
# defaults table is empty.
const _POSELIB_STRATEGY_DEFAULTS = Dict{String, Any}("le_zach" => Dict{String, Any}())

#   type     tag        path    scale    family   label        shape     parametric
const _POSELIB_STRATEGY_ROWS = (
    (LeZach, "le_zach", :irls,  nothing, nothing, raw"TLS-LZ", TLSShape, false),
)

"""
    POSELIB_STRATEGY_TAGS

PoseLib's optimization strategies' registry (a VisualGeometryToolkitCore
`TagRegistry`, built by its `@lo_registry`): `"le_zach"` → [`LeZach`](@ref).
A caller that reads strategies from TOML merges it with the core's,
`merge(VisualGeometryToolkitCore.STRATEGY_TAGS, PoseLib.POSELIB_STRATEGY_TAGS)` (or with
VisualGeometryToolkitEstimation's `OPTIMIZATION_STRATEGY_TAGS`).
"""
const POSELIB_STRATEGY_TAGS = VisualGeometryToolkitCore.@lo_registry _POSELIB_STRATEGY_ROWS defaults = _POSELIB_STRATEGY_DEFAULTS

public POSELIB_STRATEGY_TAGS

# THE strategy table: the one place a ruling on which strategies PoseLib runs is
# a one-line change. A strategy → PoseLib's `BundleOptions::LossType`.
_poselib_loss_type(::Nothing) = Int32(0)                      # TRIVIAL: least squares
_poselib_loss_type(::Irls{TLSShape}) = Int32(1)               # TRUNCATED
_poselib_loss_type(::Irls{HuberShape}) = Int32(2)             # HUBER
_poselib_loss_type(::Irls{CauchyShape}) = Int32(3)            # CAUCHY
_poselib_loss_type(::LeZach) = Int32(5)                       # TRUNCATED_LE_ZACH
_poselib_loss_type(::Irls{GaussianShape}) = throw(ArgumentError(
    "Irls{GaussianShape}: least squares is `optimization_strategy = nothing`, which takes no `scale` or `width`"))
_poselib_loss_type(::Irls{K}) where {K} = throw(ArgumentError(
    "PoseLib has no $(nameof(K)) loss; its refiners run Irls{TLSShape}, Irls{HuberShape}, Irls{CauchyShape}, " *
        "LeZach(), or `nothing` (least squares)"))
_poselib_loss_type(::Triggs) = throw(ArgumentError(
    "PoseLib's refiners take no Triggs strategy: their Levenberg–Marquardt weighs each residual by ρ′ alone " *
        "(no ρ″ term), so give Irls{K} (K = TLSShape, HuberShape or CauchyShape)"))
_poselib_loss_type(lo) = throw(ArgumentError(
    "`optimization_strategy` must be Irls{TLSShape}, Irls{HuberShape}, Irls{CauchyShape}, LeZach() or `nothing` " *
        "(least squares); got $(repr(lo))"))

# The loss shape a strategy deploys, whose efficiency constant converts a scale to a width.
_strategy_shape(::Irls{K}) where {K} = K()
_strategy_shape(::LeZach) = TLSShape()

# The efficiency constant of a strategy at the residual's codimension `d`: the
# strategy's own (`Irls{K}`'s width mode, `EfficiencyConstant(c)`), else the shape's
# default, `efficiency_constant(shape, d)`. An absolute width in the strategy
# is refused: it is the `width` keyword's.
_strategy_constant(lo::Irls, d::Int) = _strategy_constant(lo.width, _strategy_shape(lo), d)
_strategy_constant(lo::LeZach, d::Int) = _strategy_constant(nothing, _strategy_shape(lo), d)
_strategy_constant(c::EfficiencyConstant{Float64}, _shape, _d::Int) = c.value
_strategy_constant(::Nothing, shape, d::Int) =
    hasmethod(VgtkCore.efficiency_constant, Tuple{Type{typeof(shape)}}) ? Float64(VgtkCore.efficiency_constant(shape, d)) :
        throw(ArgumentError("$(nameof(typeof(shape))) has no efficiency constant (a truncation): give `width = w`, " *
            "or the strategy's own efficiency constant"))
_strategy_constant(w, _shape, _d::Int) = throw(ArgumentError(
    "PoseLib takes an absolute width as the `width` keyword, not in the strategy (got $(repr(w)))"))

# Whether a strategy carries its own efficiency constant (then a `width` would
# silently override it).
_carries_constant(lo::Irls) = lo.width !== nothing
_carries_constant(::LeZach) = false

"""
    PoseLibRefineMetadata

The metadata of a PoseLib refiner's [`FitResult`](@extref VisualGeometryToolkitCore.FitResult):
the robust-estimation contract's scale record. `scale` is the noise scale the
refiner was given and `scale_source` where it came from (`ScaleGiven` for
`scale = σ`, `ScaleUpstream` for `scale = result`; `ScaleNone`, `scale = NaN`,
for `width = w` and for least squares); `width` is the loss width PoseLib ran
at (`NaN` for least squares). `noise_scale(result)` hands the scale on.
"""
struct PoseLibRefineMetadata
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    width::Float64
end

public PoseLibRefineMetadata

# A placeholder for a keyword that must be given: an `ArgumentError` naming it,
# rather than an `UndefKeywordError` (a caller of the retired `loss` keyword is
# told to give `optimization_strategy`).
struct _Required end

# The keywords a refiner refuses, by name, and the required strategy.
function _check_refine_keywords(fname::Symbol, lo, kw)
    for k in keys(kw)
        k === :loss && throw(ArgumentError(
            "$fname: the `loss` keyword is retired; give the strategy as `optimization_strategy` " *
                "(Irls{TLSShape}(), Irls{HuberShape}(), Irls{CauchyShape}(), LeZach(), or `nothing` for least squares)"))
        k === :local_optimization && throw(ArgumentError(
            "$fname: the `local_optimization` keyword is renamed; give the strategy as `optimization_strategy`"))
        k === :loss_width && throw(ArgumentError(
            "$fname: the `loss_width` keyword is retired; give `width = w` (or `scale = σ`)"))
        throw(ArgumentError("$fname: unsupported keyword `$k`"))
    end
    lo isa _Required && throw(ArgumentError(
        "$fname needs `optimization_strategy`: Irls{TLSShape}(), Irls{HuberShape}(), Irls{CauchyShape}(), LeZach(), " *
            "or `nothing` for least squares"))
    return nothing
end

# The width PoseLib runs at, and the metadata recording it, from the strategy
# and the scale input; `d` is the codimension of the refiner's residual.
function _refine_width(::Nothing, scale, width, ::Int)
    (scale === nothing && width === nothing) || throw(ArgumentError(
        "`optimization_strategy = nothing` is least squares, which has no width: omit `scale` and `width`"))
    # PoseLib ignores the scale of its trivial loss; 1.0 is what it has always been handed.
    return 1.0, PoseLibRefineMetadata(NaN, VgtkCore.ScaleNone, NaN)
end
function _refine_width(lo, scale, width, d::Int)
    _poselib_loss_type(lo)                                   # the strategy is admitted first
    si = VgtkCore.scale_input(scale, width)
    if isnan(si.scale)                                       # `width = w`
        _carries_constant(lo) && throw(ArgumentError(
            "the strategy carries its own efficiency constant, which `width` would override; give `scale` instead, " *
                "or a strategy without one"))
        w = Float64(si.width)
        return w, PoseLibRefineMetadata(NaN, si.source, w)
    end
    s = Float64(si.scale)
    w = _strategy_constant(lo, d) * s
    return w, PoseLibRefineMetadata(s, si.source, w)
end

# The refined model as a FitResult: PoseLib's iteration count and why it
# stopped (it exhausts the budget only at `iterations == max_iter`). PoseLib
# reports no residual summary, so `rms` is `NaN`; `n` is the number of
# correspondences refined over.
_refine_result(model, iterations::Integer, max_iter::Integer, n::Integer, meta::PoseLibRefineMetadata) =
    FitResult(model, iterations < max_iter ? :converged : :max_iterations, iterations, NaN, n, meta)

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
    refine_relpose(pose::Rigid3, cs::…StaticVector{2}…; max_iter, optimization_strategy, width)

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
            "`refine_relpose(pose, withattrs(Csponds(f₁, f₂); M₁, M₂); max_iter, optimization_strategy, width)` " *
            "from `backproject_with_jacobian(camera, u)`, or hand over pixels and " *
            "the intrinsics, `refine_relpose(pose, K1, K2, cs; max_iter, optimization_strategy, width)`."
    ))
end

# A K as PoseLib's SIMPLE_PINHOLE camera: (focal, cx, cy), the focal the mean of K₁₁ and K₂₂ — the one statement of
# the mapping, for every binding that hands a K over and for `calibrated_loss_width`.
_pinhole_focal(K::AbstractMatrix) = (K[1, 1] + K[2, 2]) / 2
_simple_pinhole(K::AbstractMatrix) = Float64[_pinhole_focal(K), K[1, 3], K[2, 3]]

"""
    calibrated_loss_width(K1, K2, width) -> Float64

The calibrated (bearing-unit) loss width that PoseLib's relative-pose refiner
works at when it is handed `width` in the unit of the points together with the
intrinsics `K1` and `K2` ([`refine_relpose`](@ref)`(pose, K1, K2, cs; width)`):
`width · ½(1/f₁ + 1/f₂)`, PoseLib's camera-aware rule (`robust.cc`), with each
`fᵢ = (Kᵢ₁₁ + Kᵢ₂₂)/2`, the focal length of the `SIMPLE_PINHOLE` camera the
binding hands over.

A refiner on calibrated data that must deploy at the same operating point as
PoseLib's pixel refiner converts its width with this, so the two cannot drift
apart when the camera mapping changes.
"""
calibrated_loss_width(K1::AbstractMatrix, K2::AbstractMatrix, width::Real) =
    width * (0.5 * (1.0 / _pinhole_focal(K1) + 1.0 / _pinhole_focal(K2)))

public calibrated_loss_width

"""
    refine_relpose(pose::Rigid3, K1, K2, cs; max_iter, optimization_strategy, scale | width) -> FitResult{Rigid3}

Robust Levenberg–Marquardt refinement of a relative pose from pixel
correspondences and the two cameras' intrinsics.

`pose` is the seed [`Rigid3`](@ref), mapping the first camera's frame to the
second's. `cs` holds the correspondences `x1 => x2`, each point in its own
image's frame. `K1` and `K2` are the intrinsic matrices of those frames: pixel
matrices for pixel points, or matrices in whatever frame the points are in.
Returns a `FitResult` whose `estimate` is the refined `Rigid3`, as a `FitResult`.

The width is in the unit of the points. PoseLib back-projects through each
camera, so one width serves both images even when their focal lengths differ.
`max_iter`, `optimization_strategy`, `scale` and `width`, and the result, are as
in [`refine_homography`](@ref); the residual's codimension is 1.

Each `K` reaches PoseLib as its `SIMPLE_PINHOLE` camera model, with focal
length `(K₁₁ + K₂₂)/2` and principal point `(K₁₃, K₂₃)`. Skew and any
difference between `K₁₁` and `K₂₂` are dropped, so the model is exact only for
square pixels and zero skew.

It ends in the local minimum its seed leads to, as described for
[`refine_homography`](@ref).
"""
function refine_relpose(pose::Rigid3, K1::AbstractMatrix, K2::AbstractMatrix,
                        cs::AbstractVector{<:Cspond{<:StaticVector{2}, <:StaticVector{2}}};
                        max_iter::Integer, optimization_strategy = _Required(),
                        scale = nothing, width = nothing, kw...)
    _check_refine_keywords(:refine_relpose, optimization_strategy, kw)
    w, meta = _refine_width(optimization_strategy, scale, width, 1)
    X1, X2 = _pack(cs)
    R0 = pose.R; t0 = pose.t
    Rin = Vector{Float64}(undef, 9)
    @inbounds for j in 1:3, i in 1:3
        Rin[3(j-1)+i] = R0[i, j]            # column-major (Eigen layout)
    end
    tin  = Float64[t0[1], t0[2], t0[3]]
    cam1 = _simple_pinhole(K1)
    cam2 = _simple_pinhole(K2)
    R_out = MVector{9, Float64}(undef); t_out = MVector{3, Float64}(undef)
    _require_library()
    iterations = ccall((:poselib_refine_relpose_cam, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          X1, X2, length(cs), Rin, tin, cam1, cam2, w, max_iter, _poselib_loss_type(optimization_strategy),
          pointer(R_out), pointer(t_out))
    model = Rigid3(_read_mat33_from_buf(R_out, 0),
                   Vec3{Float64}(t_out[1], t_out[2], t_out[3]))
    return _refine_result(model, iterations, max_iter, length(cs), meta)
end

"""
    refine_relpose(pose::Rigid3, cs; max_iter, optimization_strategy, scale | width) -> FitResult{Rigid3}

Robust Levenberg–Marquardt refinement of a relative pose from unit bearings,
through PoseLib's tangent-Sampson refiner.

`pose` is the seed [`Rigid3`](@ref), mapping the first camera's frame to the
second's. `cs` holds the correspondences `f₁ => f₂` of unit bearings
(`StaticVector{3}`). They carry their 3×2 back-projection Jacobians as the
attribute columns `M₁` and `M₂`, attached with `withattrs(cs; M₁, M₂)`.
VisualGeometryToolkitCore's `backproject_with_jacobian(camera, u)` returns
each bearing together with its Jacobian. Returns a `FitResult` whose `estimate` is the refined `Rigid3`, as a `FitResult`.

No image plane is involved, so a ray at or beyond 90° from the optical axis is
an ordinary input. A 3-vector that is not of unit length is refused with an
`ArgumentError`: a point on the z = 1 plane lifted to `(x, y, 1)` has the right
type but is not a bearing.

# The unit of the width

`width` is the width `w` of the loss on the tangent-Sampson residual

    r = (f₂ᵀ E f₁) / √(‖M₂ᵀ E f₁‖² + ‖M₁ᵀ Eᵀ f₂‖²)

The denominator is the gradient norm of the epipolar constraint with respect
to the coordinates that `M` differentiates against. The residual `r` is in the
unit of those coordinates, and `width` (or `scale`) must be given in the same unit.
The Jacobians set the unit; the call cannot.

| `M` is | `r` and `width` are in |
|---|---|
| `∂f/∂u` from `backproject_with_jacobian` | pixels of `u` |
| an orthonormal basis of the tangent plane of the sphere at `f` | radians |

A width in the wrong unit is off by a factor of the focal length. Nothing
reports this: a truncated loss simply keeps the wrong inliers.

All arguments are in the caller's frame, and the refined pose comes back in it.
It ends in the local minimum its seed leads to, as described for
[`refine_homography`](@ref).
"""
function refine_relpose(pose::Rigid3,
                        cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{3}}};
                        max_iter::Integer, optimization_strategy = _Required(),
                        scale = nothing, width = nothing, kw...)
    _check_refine_keywords(:refine_relpose, optimization_strategy, kw)
    w, meta = _refine_width(optimization_strategy, scale, width, 1)
    (hasattr(cs, :M₁) && hasattr(cs, :M₂)) || throw(ArgumentError(
        "refine_relpose on bearings needs the backprojection Jacobians as attribute " *
        "columns: withattrs(cs; M₁, M₂), from backproject_with_jacobian. They set " *
        "the unit of the width, so there is no defensible default for them."))
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
    iterations = ccall((:poselib_refine_relpose_bearing, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint,
           Ptr{Float64}, Ptr{Float64},
           Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          D1, D2, J1, J2, length(cs), Rin, tin,
          w, max_iter, _poselib_loss_type(optimization_strategy),
          pointer(R_out), pointer(t_out))
    model = Rigid3(_read_mat33_from_buf(R_out, 0),
                   Vec3{Float64}(t_out[1], t_out[2], t_out[3]))
    return _refine_result(model, iterations, max_iter, length(cs), meta)
end

# =============================================================================
# refine_fundamental  (robust-loss LM refinement of a fundamental matrix)
# =============================================================================

export refine_fundamental

"""
    refine_fundamental(F, cs; max_iter, optimization_strategy, scale | width) -> FitResult{FundMat}

Robust Levenberg–Marquardt refinement of a fundamental matrix through
PoseLib's `refine_fundamental`. It minimizes the Sampson error over the
Bartoli–Sturm parametrization of `F`, which keeps the matrix at rank 2.

`F` is the seed 3×3 fundamental matrix. `cs` holds the correspondences
`x1 => x2` of 2D image points, in the frame `F` is expressed in (pixels for a
pixel-frame `F`). The width is that of the loss on the Sampson residual, in
the same unit (codimension 1). `max_iter`, `optimization_strategy`, `scale` and
`width`, and the result, are as in [`refine_homography`](@ref). Returns a
`FitResult` whose `estimate` is the refined `FundMat`. It ends in the
local minimum its seed leads to, as described for
[`refine_homography`](@ref).
"""
function refine_fundamental(F::AbstractMatrix,
                            cs::AbstractVector{<:Cspond{<:StaticVector{2}, <:StaticVector{2}}};
                            max_iter::Integer, optimization_strategy = _Required(),
                            scale = nothing, width = nothing, kw...)
    _check_refine_keywords(:refine_fundamental, optimization_strategy, kw)
    w, meta = _refine_width(optimization_strategy, scale, width, 1)
    _require_3x3(F, "F")
    X1, X2 = _pack(cs)
    Fin = Vector{Float64}(undef, 9)
    @inbounds for j in 1:3, i in 1:3
        Fin[3(j-1)+i] = F[i, j]                # column-major (Eigen layout)
    end
    F_out = MVector{9, Float64}(undef)
    _require_library()
    iterations = ccall((:poselib_refine_fundamental, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64},
           Float64, Cint, Cint, Ptr{Float64}),
          X1, X2, length(cs), Fin, w, max_iter, _poselib_loss_type(optimization_strategy),
          pointer(F_out))
    model = FundMat{Float64}(Tuple(_read_mat33_from_buf(F_out, 0)))
    return _refine_result(model, iterations, max_iter, length(cs), meta)
end

# =============================================================================
# refine_homography  (robust-loss LM refinement of a homography)
# =============================================================================

export refine_homography
export refine_absolute_pose

"""
    refine_absolute_pose(pose::Rigid3, cs::…StaticVector{2}…; max_iter, optimization_strategy, width)

This form is refused with an `ArgumentError`. Its targets would be calibrated
points on the z = 1 plane (`GnomonicFrame`), with a loss width divided by the
focal length. That plane has no point for a ray at or beyond 90° from the
optical axis.

Use `refine_absolute_pose(pose, K, cs)` instead: the same correspondences in
pixels, the intrinsic matrix `K`, and the width in pixels.

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
            "`refine_absolute_pose(pose, K, cs; max_iter, optimization_strategy, width)`."
    ))
end

"""
    refine_absolute_pose(pose::Rigid3, K, cs; max_iter, optimization_strategy, scale | width) -> FitResult{Rigid3}

Robust Levenberg–Marquardt refinement of an absolute pose from 3D-to-2D
correspondences, through PoseLib's C++ absolute-pose bundle adjustment. The residual is the
reprojection error `project(camera, R X + t) − u`.

`pose` is the seed [`Rigid3`](@ref), mapping the world frame to the camera's.
`K` is the intrinsic matrix of the frame the image points are in. `cs` holds
the correspondences `X => u` of 3D points to 2D image points. Returns the
refined `Rigid3`, as a `FitResult`.

The width is that of the loss on the reprojection residual, in the unit of
`u` and `K` (pixels for a pixel-frame `K`; codimension 2). `max_iter`,
`optimization_strategy`, `scale` and `width`, and the result,
are as in [`refine_homography`](@ref). Only the six degrees of freedom of the
pose are refined; `K` stays fixed.

`K` reaches PoseLib as its `SIMPLE_PINHOLE` camera model, with focal length
`(K₁₁ + K₂₂)/2` and principal point `(K₁₃, K₂₃)`. Skew and any difference
between `K₁₁` and `K₂₂` are dropped, so the model is exact only for square
pixels and zero skew.

It ends in the local minimum its seed leads to, as described for
[`refine_homography`](@ref).
"""
function refine_absolute_pose(pose::Rigid3, K::AbstractMatrix,
                              cs::AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{2}}};
                              max_iter::Integer, optimization_strategy = _Required(),
                              scale = nothing, width = nothing, kw...)
    _check_refine_keywords(:refine_absolute_pose, optimization_strategy, kw)
    w, meta = _refine_width(optimization_strategy, scale, width, 2)
    _require_3x3(K, "K")
    Xb, xb = _pack(cs)
    R0 = pose.R; t0 = pose.t
    Rin = Vector{Float64}(undef, 9)
    @inbounds for j in 1:3, i in 1:3
        Rin[3(j-1)+i] = R0[i, j]            # column-major (Eigen layout)
    end
    tin = Float64[t0[1], t0[2], t0[3]]
    cam = _simple_pinhole(K)
    R_out = MVector{9, Float64}(undef); t_out = MVector{3, Float64}(undef)
    _require_library()
    iterations = ccall((:poselib_bundle_adjust_cam, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          xb, Xb, length(cs), Rin, tin, cam,
          w, max_iter, _poselib_loss_type(optimization_strategy),
          pointer(R_out), pointer(t_out))
    model = Rigid3(_read_mat33_from_buf(R_out, 0),
                   Vec3{Float64}(t_out[1], t_out[2], t_out[3]))
    return _refine_result(model, iterations, max_iter, length(cs), meta)
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
    refine_absolute_pose(pose::UnitfulEuclideanMap{3}, K, cs; max_iter, optimization_strategy, width)

The physical form of [`refine_absolute_pose`](@ref): `pose` a world→camera
pose whose translation is a length (FlexUnits quantities) and `cs` the
correspondences `X => u` with the world points `X` lengths. Both are stripped in
metres once, the plain refinement runs, and the refined pose comes back with its
translation in metres, as a `FitResult`: equal, after `ustrip(m, ·)`, to the plain refinement's
on the inputs' `ustrip(m, ·)`, with the plain refinement's `retcode`,
`iterations` and scale record. `u`, `K` and the width stay in pixels. A
labeled seed (an estimate's `CameraExtrinsics`, world → camera) is read
unlabeled; the refined pose comes back unlabeled. A physical seed with plain
points, or the reverse, is an `ArgumentError`.
"""
function refine_absolute_pose(pose::UnitfulEuclideanMap{3}, K::AbstractMatrix, cs::_PhysicalPoints3; kwargs...)
    seed = convert(Rigid3, VgtkCore.unlabel(ustrip(VgtkCore.m, pose)))   # a labeled pose (an estimate's CameraExtrinsics) too
    plain = [_metres_point(first(c)) => last(c) for c in cs]
    r = refine_absolute_pose(seed, K, plain; kwargs...)
    refined = r.estimate
    # rebuilt through the checked constructor (the refined rotation is verified, as
    # before), then read in metres; the rest of the result is the plain refinement's
    pose_m = VgtkCore.with_unit(EuclideanMap(refined.R, refined.t), VgtkCore.m)
    return FitResult(pose_m, r.retcode, r.iterations, r.rms, r.n, r.metadata)
end

refine_absolute_pose(::UnitfulEuclideanMap{3}, ::AbstractMatrix, ::_PlainPoints3; kwargs...) = throw(ArgumentError(
    "refine_absolute_pose: the seed is physical but the world points are plain; attach their unit (X .* m)"
))
refine_absolute_pose(::Rigid3, ::AbstractMatrix, ::_PhysicalPoints3; kwargs...) = throw(ArgumentError(
    "refine_absolute_pose: the world points are physical but the seed is plain; attach its unit, " *
        "EuclideanMap(pose.R, pose.t .* m)"
))

"""
    refine_homography(H, cs; max_iter, optimization_strategy, scale | width) -> FitResult{HomographyMat}

Robust Levenberg–Marquardt refinement of a homography through PoseLib's
`refine_homography`. It minimizes the symmetric transfer error: the forward
residual `x2 − π(H x1)` and the backward residual `x1 − π(H⁻¹ x2)`, each
robustified per point, where `π` divides by the third coordinate.

`H` is the seed 3×3 homography. `cs` holds the correspondences `x1 => x2` of
2D image points, in the frame `H` is expressed in (pixels for a pixel-frame
`H`). The width is that of the loss on a one-sided transfer residual, in the
same unit. Returns a `FitResult` whose `estimate` is the refined
`HomographyMat`.

`optimization_strategy` is required, like `max_iter`: the robust estimate's
strategy is the caller's configuration (its strategy or TOML table), never a
default here. PoseLib's Levenberg–Marquardt weighs each residual by its loss
weight `ρ′` alone, so its strategies are `Irls{K}` (VisualGeometryToolkitCore's);
a `Triggs{K}` (the `ρ″`-corrected LM) is refused, as is any other shape:

| `optimization_strategy` | `ρ(r)` | note |
|---|---|---|
| `Irls{TLSShape}()` | `min(r², w²)` | truncated quadratic: inliers have `r < w`, outliers get zero weight |
| `Irls{HuberShape}()` | `r²` up to `w`, linear beyond | |
| `Irls{CauchyShape}()` | `w² log(1 + r²/w²)` | |
| [`LeZach`](@ref)`()` | `min(r², w²)` | the truncated loss under Le and Zach's scheme |
| `nothing` | `r²` | plain least squares; takes no `scale` or `width` |

A robust strategy takes exactly one of:

- `width = w`, the width `w` of VisualGeometryToolkitCore's closed form for
  that shape. PoseLib's own `loss_scale` equals it for every loss bound here,
  so it reaches PoseLib unchanged;
- `scale = σ` (or an earlier robust result, read by `noise_scale`): the loss
  runs at `w = c·σ`, with `c` the strategy's efficiency constant, else
  VisualGeometryToolkitCore's `efficiency_constant(shape, d)` at the
  residual's codimension `d` (2 here). TLS and Le–Zach have no efficiency
  constant (a truncation), so they take `width`.

The retired `loss` and `loss_width` keywords are refused with an
`ArgumentError` naming their replacements. `max_iter` is PoseLib's
Levenberg–Marquardt iteration budget; it has no default (an `Irls` strategy's
own `max_iter`, its reweighting count, is not read by PoseLib).

The result records PoseLib's `iterations` and its `retcode`: `:converged` when
PoseLib stopped on its own tolerances, `:max_iterations` when it used the whole
budget. Its metadata ([`PoseLibRefineMetadata`](@ref)) is the scale record:
`noise_scale(result)` hands a given scale on to the next stage. PoseLib reports
no residuals, weights or cost, so `rms` is `NaN` and `n` is the number of
correspondences.

A refiner ends in the local minimum its seed leads to: under a truncated loss,
a correspondence whose residual at the seed exceeds the width gets no weight,
so a seed must be close enough that its inliers fall within the width.
Without the library on this platform ([`is_available`](@ref) false), a call
that reaches it raises an error that says so; a refused form (calibrated
2-vectors, mixed units, non-unit bearings) raises its own `ArgumentError` first.

```jldoctest
julia> using PoseLib, VisualGeometryToolkitCore

julia> H = HomographyMat([1.0 0.0 5.0; 0.0 1.0 -3.0; 0.0 0.0 1.0]);

julia> x1 = [Point2(a, b) for a in 0.0:100.0:300.0 for b in 0.0:100.0:300.0];

julia> x2 = [Point2(x[1] + 5.0, x[2] - 3.0) for x in x1];

julia> r = refine_homography(HomographyMat([1.0 0.0 6.0; 0.0 1.0 -3.0; 0.0 0.0 1.0]), Csponds(x1, x2);
                              max_iter = 25, optimization_strategy = Irls{TLSShape}(), width = 2.0);

julia> H1 = estimate(r); H1 / H1[3, 3] ≈ H
true

julia> retcode(r), iterations(r) < 25
(:converged, true)
```
"""
function refine_homography(H::AbstractMatrix,
                           cs::AbstractVector{<:Cspond{<:StaticVector{2}, <:StaticVector{2}}};
                           max_iter::Integer, optimization_strategy = _Required(),
                           scale = nothing, width = nothing, kw...)
    _check_refine_keywords(:refine_homography, optimization_strategy, kw)
    w, meta = _refine_width(optimization_strategy, scale, width, 2)
    _require_3x3(H, "H")
    X1, X2 = _pack(cs)
    Hin = Vector{Float64}(undef, 9)
    @inbounds for j in 1:3, i in 1:3
        Hin[3(j-1)+i] = H[i, j]                # column-major (Eigen layout)
    end
    H_out = MVector{9, Float64}(undef)
    _require_library()
    iterations = ccall((:poselib_refine_homography, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64},
           Float64, Cint, Cint, Ptr{Float64}),
          X1, X2, length(cs), Hin, w, max_iter, _poselib_loss_type(optimization_strategy),
          pointer(H_out))
    model = HomographyMat{Float64}(Tuple(_read_mat33_from_buf(H_out, 0)))
    return _refine_result(model, iterations, max_iter, length(cs), meta)
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
