"""
    PoseLib

Julia bindings to [PoseLib](https://github.com/PoseLib/PoseLib), the C++ library
of minimal solvers and robust refiners for camera pose estimation.

Loading this package extends VisualGeometryToolkitCore's `solve(problem, algorithm)`
interface with PoseLib's backend, `PoseLibSolver()`:

- The minimal solvers: `solve(MinimalProblem(M, cs, idx), solver(PoseLibSolver()))`
  runs PoseLib's four-point homography (`Homography4pt`), seven-point
  fundamental matrix (`Fundamental7pt`), perspective-three-point absolute pose
  (`P3P`) and five-point essential matrix (`Essential5pt`).
- The eight-point essential matrix:
  `solve(EstimationProblem(EssentialMat, cs), DLT(PoseLibSolver()))`, which has
  no native variant.
- PoseLib's robust Levenberg–Marquardt refines of a homography, a fundamental
  matrix, a relative pose and an absolute pose:
  `solve(RefineProblem(cs, guess[, p]), LevenbergMarquardt(PoseLibSolver();
  optimization_strategy); stops = LmStops(; max_iter), scale | width)`, the
  guess a `HomographyMat`, a `FundMat` or an `EuclideanMap{3}` and the context
  `p` the intrinsics where PoseLib needs them (the `PoseLibSolver` method of
  VisualGeometryToolkitCore's `lm_backend_refine` states each form). The
  result is a `FitResult` with the refined model, in the caller's frame, as `u`.

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
    Irls, Triggs, EfficiencyConstant, FitResult, AbstractOptimizationStrategy, ClassDefault, SolveStats,
    LevenbergMarquardt, LevenbergDamping, LmStops, NullParameters, PlainEuclideanMap,
    HomographyMat, FundMat, EssentialMat, CameraExtrinsics, FixedModels,
    EuclideanMap, UnitfulEuclideanMap, ustrip,
    PoseLibSolver, Vec3, Point3, Correspondence, sources, targets, hasattr,
    Homography4pt, Fundamental7pt, Essential5pt, P3P,
    sign_normalize, is_valid_solution,
    homography_4pt, fundamental_7pt, p3p,
    essential_5pt, essential_8pt

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

# Unit-bearing correspondences (`StaticVector{3}` each side) and plane correspondences (`StaticVector{2}` each side),
# in any Correspondence container.
const _BearingPairs = AbstractVector{<:Correspondence{<:StaticVector{3}, <:StaticVector{3}}}
const _PlanePairs = AbstractVector{<:Correspondence{<:StaticVector{2}, <:StaticVector{2}}}

# Pack N SVector{3} into a flat NTuple for ccall (x,y,z,x,y,z,...)
@inline function _pack_points3(pts::NTuple{N, <:StaticVector{3, Float64}}) where N
    ntuple(i -> pts[div(i-1, 3) + 1][mod(i-1, 3) + 1], Val(3N))
end

# Read pose (R col-major + t) from MVector at offset. Absolute-pose (p3p)
# output — world-to-camera, so the labeled `CameraExtrinsics` is the honest type;
# the C buffers' bits, unchecked.
@inline _read_pose_from_buf(buf, offset::Int) =
    CameraExtrinsics(_plain_pose(_read_mat33_from_buf(buf, offset),
                                 Vec3{Float64}(buf[offset+10], buf[offset+11], buf[offset+12]); check = false))

# The unlabeled pose a refine returns, `PlainEuclideanMap{3, Float64}` with a `RotMatrix{3, Float64}` rotation and a
# `Vec3{Float64}` translation, from a rotation matrix and a translation (the rotation checked unless `check = false`).
_plain_pose(R::SMatrix{3, 3, Float64, 9}, t::Vec3{Float64}; check::Bool = true) = EuclideanMap(RotMatrix{3, Float64}(R), t; check)

# The C wrapper takes each side of the correspondences as one contiguous
# point-major block (`M·n` and `N·n` doubles). Both blocks live in ONE buffer,
# the sources first, the targets from `M·n + 1` on (`_targets_at`), each handed
# to the C call as a pointer under `GC.@preserve`: one allocation. Packing both
# columns, read through VGC's `sources`/`targets` protocol in one sweep, is the
# only per-call copy, whichever Correspondence container is supplied
# (`Correspondences`, `StructArray{<:Pair}`, `Vector{<:Pair}`); `source` maps each
# source point to its plain form as it is packed (a physical world point's metres).
@inline function _pack(cs::AbstractVector{<:Correspondence{<:StaticVector{M}, <:StaticVector{N}}}, source::F = identity) where {M, N, F}
    src = sources(cs); dst = targets(cs)
    Base.require_one_based_indexing(src, dst)
    n = length(cs)
    a = Vector{Float64}(undef, (M + N) * n)
    @inbounds for i in 1:n
        u = source(src[i]); v = dst[i]
        for j in 1:M; a[M * (i - 1) + j] = u[j]; end
        for j in 1:N; a[M * n + N * (i - 1) + j] = v[j]; end
    end
    return a
end
# Where the targets' block of a packed buffer starts: after `M` doubles per correspondence.
@inline _targets_at(buf::Vector{Float64}, M::Int, n::Int) = pointer(buf, M * n + 1)

# The per-correspondence 3×2 matrices (the backprojection Jacobians `M₁` and
# `M₂`) into ONE buffer, each side a contiguous `6·n` block, point-major, `M₂`'s
# from `6n + 1` on. StaticArrays stores column-major and so does Eigen, so a
# StaticMatrix's LINEAR index is already the wire order and the C side maps each
# 6-double block straight onto an `Eigen::Matrix<double,3,2>` — no transpose
# anywhere in the crossing.
@inline function _pack_tangent(M1s::AbstractVector{<:StaticMatrix{3, 2}}, M2s::AbstractVector{<:StaticMatrix{3, 2}})
    Base.require_one_based_indexing(M1s, M2s)
    n = length(M1s)
    a = Vector{Float64}(undef, 12 * n)
    @inbounds for i in 1:n
        A = M1s[i]; B = M2s[i]
        for j in 1:6
            a[6 * (i - 1) + j] = A[j]
            a[6 * n + 6 * (i - 1) + j] = B[j]
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
`solve(MinimalProblem(HomographyMat, cs, idx), Homography4pt(PoseLibSolver()))`.
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
`solve(MinimalProblem(FundMat, cs, idx), Fundamental7pt(PoseLibSolver()))`.
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
    p3p(::PoseLibSolver, x, X) -> Union{FixedModels{4, model_type(P3P())}, Nothing}

PoseLib's perspective-three-point solver, the `PoseLibSolver` backend of
VisualGeometryToolkitCore's `p3p` kernel. `x` holds three unit bearings
(`StaticVector{3, Float64}`) and `X` the three world points they see. Returns
every real solution, up to four, as world-to-camera extrinsics in the frame of
`X` (plain `Float64` points; the translation is in their unit); a degenerate sample returns fewer, possibly none, and inputs of
any length other than three return `nothing`. The public entry point is
`solve(MinimalProblem(CameraExtrinsics, cs, idx), P3P(PoseLibSolver()))` with `cs` the correspondences
`X => x`.
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
    FixedModels{4, VgtkCore.model_type(P3P())}(n) do i
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
# `solve(MinimalProblem(EssentialMat, cs, idx), Essential5pt(PoseLibSolver()))` builds. `Tuple(::SVector)` is
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
none. The public entry point is
`solve(MinimalProblem(EssentialMat, cs, idx), Essential5pt(PoseLibSolver()))`.
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
# `sources`/`targets` protocol so every Correspondence container works (a plain
# `Vector{<:Pair}` has no `.first` column).
function VgtkCore.essential_5pt(::PoseLibSolver,
        cs::_BearingPairs, idx::AbstractVector{<:Integer})
    u1 = sources(cs); u2 = targets(cs)
    x1 = @inbounds ntuple(k -> Vec3{Float64}(u1[idx[k]]), Val(5))
    x2 = @inbounds ntuple(k -> Vec3{Float64}(u2[idx[k]]), Val(5))
    VgtkCore.essential_5pt(PoseLibSolver(), x1, x2)
end

# No-backend essential_5pt default now lives in VisualGeometryToolkitCore
# (NativeSolver port of relpose_5pt.cc). Select this backend explicitly:
# essential_5pt(PoseLibSolver(), x1, x2).

# =============================================================================
# essential_8pt  (PoseLib only — the backend of DLT(PoseLibSolver()) on an essential matrix)
# =============================================================================

"""
    essential_8pt(::PoseLibSolver, cs) -> Union{EssentialMat{Float64}, Nothing}

PoseLib's eight-point essential matrix, the `PoseLibSolver` backend of
VisualGeometryToolkitCore's `essential_8pt` kernel: the direct linear transform
over eight or more correspondences `cs` of unit bearings `f₁ => f₂`, every one
taken as an inlier. VisualGeometryToolkitCore has no native eight-point solver,
so this is the only backend. The public entry point is
`solve(EstimationProblem(EssentialMat, cs), DLT(PoseLibSolver()))`, whose
`FitResult` holds the matrix as `u` (`nothing` when degenerate).

An algorithm, so it can only be degenerate: it returns `nothing` when PoseLib's
solution is not finite, but a degenerate set (a pure rotation, repeated points)
can also give a finite, meaningless `E` that nothing flags. Fewer than eight
correspondences throw an `ArgumentError` naming the count. Correspondences that
are not bearings (2-vectors on the z = 1 plane) are a `MethodError`, never lifted.
"""
function VgtkCore.essential_8pt(::PoseLibSolver, cs::_BearingPairs)
    n = length(cs)
    n >= 8 || throw(ArgumentError("DLT(PoseLibSolver()) of an essential matrix needs at least 8 correspondences, got $n"))
    buf = _pack(cs)
    E_buf = MVector{9, Float64}(undef)
    _require_library()
    GC.@preserve buf ccall((:poselib_essential_8pt, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}),
          pointer(buf), _targets_at(buf, 3, n), n, pointer(E_buf))
    all(isfinite, E_buf) || return nothing
    EssentialMat{Float64}(Tuple(E_buf))
end
# Any other data (calibrated 2-vectors, a z = 1 chart) has no 8-point: the pair's MethodError, stated here so that
# Core's fallback for a backend without one does not claim PoseLib lacks it.
VgtkCore.essential_8pt(backend::PoseLibSolver, cs) = throw(MethodError(VgtkCore.essential_8pt, (backend, cs)))

# =============================================================================
# The refines: LevenbergMarquardt(PoseLibSolver()) on raw data (lm_backend_refine)
# =============================================================================

public is_available
export LeZach

# -----------------------------------------------------------------------------
# The strategy, typed: the stack's robust-estimation contract names a refine's
# loss as its strategy, `optimization_strategy` (the LevenbergMarquardt's field),
# and its width by `scale` or `width` (solve keywords). Each strategy PoseLib
# runs maps to PoseLib's `BundleOptions::LossType` enum (PoseLib/types.h) ONLY
# at the ccall boundary; the integer enum never leaves this file.
#
# PoseLib's Levenberg–Marquardt weighs each residual by its loss weight ρ′ alone
# (robust/optim/jacobian_accumulator.h: J'WJ with W = loss.weight(r²), no ρ″
# term), so its strategy is `Irls{K}`. A `Triggs{K}` strategy (the ρ″-corrected
# LM) is refused: PoseLib does not run it. Its damping is λ·I
# (`LevenbergDamping()`, `default_damping(::PoseLibSolver)`).
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
# w/2; Huber's elbow at w). `scale = σ` runs at `efficiency_constant(strategy, d) · σ`, the strategy's own constant,
# else VisualGeometryToolkitCore's at the residual's codimension d (`_deployed_width`).
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
    "PoseLib has no $(nameof(K)) loss; its refines run Irls{TLSShape}, Irls{HuberShape}, Irls{CauchyShape}, " *
        "LeZach(), or `nothing` (least squares)"))
_poselib_loss_type(::Triggs) = throw(ArgumentError(
    "PoseLib's refines take no Triggs strategy: their Levenberg–Marquardt weighs each residual by ρ′ alone " *
        "(no ρ″ term), so give Irls{K} (K = TLSShape, HuberShape or CauchyShape)"))
_poselib_loss_type(::ClassDefault) = throw(ArgumentError(
    _WHO * " needs `optimization_strategy`: Irls{TLSShape}(), Irls{HuberShape}(), " *
        "Irls{CauchyShape}(), LeZach(), or `nothing` for least squares (PoseLib's refines have no class default)"))
_poselib_loss_type(lo) = throw(ArgumentError(
    "`optimization_strategy` must be Irls{TLSShape}, Irls{HuberShape}, Irls{CauchyShape}, LeZach() or `nothing` " *
        "(least squares); got $(repr(lo))"))

# Whether a strategy carries its own efficiency constant (then a `width` would
# silently override it).
_carries_constant(lo::Irls) = lo.width !== nothing
_carries_constant(::LeZach) = false

"""
    PoseLibRefineMetadata

The metadata of a PoseLib refine's [`FitResult`](@extref VisualGeometryToolkitCore.FitResult):
the robust-estimation contract's scale record. `scale` is the noise scale the
refine was given and `scale_source` where it came from (`ScaleGiven` for
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

# PoseLib's Levenberg–Marquardt, `LevenbergMarquardt(PoseLibSolver(); …)`: its damping is PoseLib's λ·I.
VgtkCore.default_damping(::PoseLibSolver) = LevenbergDamping()

# The iteration budget PoseLib runs, from the solve's `stops`: PoseLib's wrapper sets `max_iter` and nothing else, so
# a stop setting is APPLIED or REFUSED, never ignored. Every other field must be `===` to `LmStops()`'s (the defaults
# of record): `stops === LmStops(; max_iter = stops.max_iter)`, which also covers a field added to LmStops later; the
# `ArgumentError` names the field that differs.
function _poselib_max_iter(stops::LmStops)
    stops === LmStops(; max_iter = stops.max_iter) || throw(_unapplied_stop(stops))
    stops.max_iter <= typemax(Cint) || throw(ArgumentError("PoseLib's max_iter is a C int, ≤ $(typemax(Cint)); got $(stops.max_iter)"))
    return stops.max_iter
end
@noinline function _unapplied_stop(stops::LmStops)
    d = LmStops()
    for (name, set, default) in (("ftol", stops.ftol, d.ftol), ("xtol", stops.xtol, d.xtol), ("xtol_rel", stops.xtol_rel, d.xtol_rel),
                                 ("stationarity_tol", stops.stationarity_tol, d.stationarity_tol),
                                 ("max_time", stops.max_time, d.max_time), ("residual_resolution", stops.residual_resolution, d.residual_resolution))
        set === default || return ArgumentError(string("PoseLib's Levenberg–Marquardt applies only `max_iter` of its LmStops ",
            "(its tolerances are PoseLib's own); `", name, " = ", set, "` cannot be applied: leave it at its default"))
    end
    return ArgumentError("PoseLib's Levenberg–Marquardt applies only `max_iter` of its LmStops (its tolerances are " *
        "PoseLib's own): leave every other field at its default")
end

# The algorithm's strategy and damping, admitted (a missing strategy is named first); returns the strategy.
function _check_poselib_alg(alg::LevenbergMarquardt)
    lo = alg.optimization_strategy
    lo === nothing || _poselib_loss_type(lo)
    alg.damping isa LevenbergDamping || throw(ArgumentError(
        "PoseLib's Levenberg–Marquardt damps by λ·I: its damping is LevenbergDamping() (the default); got $(repr(alg.damping))"))
    return lo
end

# The width PoseLib runs at, and the metadata recording it, from the admitted strategy `lo` and the solve's scale
# input; `d` is the codimension of the refine's residual.
function _deployed_width(::Nothing, scale, width, ::Int)
    (scale === nothing && width === nothing) || throw(ArgumentError(
        "`optimization_strategy = nothing` is least squares, which has no width: omit `scale` and `width`"))
    # PoseLib ignores the scale of its trivial loss; 1.0 is what it has always been handed.
    return 1.0, PoseLibRefineMetadata(NaN, VgtkCore.ScaleNone, NaN)
end
function _deployed_width(lo, scale, width, d::Int)
    si = VgtkCore.scale_input(scale, width)
    if isnan(si.scale)                                       # `width = w`
        _carries_constant(lo) && throw(ArgumentError(
            "the strategy carries its own efficiency constant, which `width` would override; give `scale` instead, " *
                "or a strategy without one"))
        w = Float64(si.width)
        return w, PoseLibRefineMetadata(NaN, si.source, w)
    end
    s = Float64(si.scale)
    w = VgtkCore.efficiency_constant(lo, d) * s   # the strategy's own constant, else the table's (Core)
    return w, PoseLibRefineMetadata(s, si.source, w)
end

# The refined model as a FitResult: PoseLib's iteration count (`stats.nsteps`) and why it stopped (it exhausts the
# budget only at `iterations == max_iter`), as its termination record: `(Success, NoRule)` before the cap (PoseLib
# reports no stop rule, so none is claimed), `(MaxIters, IterationCap)` at it. PoseLib reports no residual summary, so
# `rms` is `NaN` and `resid` `nothing`; `n` is the number of correspondences refined over (every one handed over:
# PoseLib reports no weights, so it is not VisualGeometryToolkitEstimation's weighted support). `original` is the C
# wrapper's raw output: its return value (PoseLib's `BundleStats.iterations`) and its output buffers as written.
_poselib_result(model, original, max_iter::Int, n::Integer, meta::PoseLibRefineMetadata) =
    FitResult(model, (original.iterations < max_iter ? (VgtkCore.ReturnCode.Success, VgtkCore.NoRule) :
                                                              (VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap))...,
              SolveStats(; nsteps = original.iterations), NaN, n, meta; original)

# A guess pose in the type a refine returns (Float64 matrix rotation and translation), so the result's type never
# depends on the input's.
_returned_pose(pose::PlainEuclideanMap{3}) = _plain_pose(SMatrix{3, 3, Float64, 9}(pose.R), Vec3{Float64}(pose.t))

# Fewer correspondences than the model's minimal sample (`solver`'s `sample_size`): nothing to refine.
_too_few(cs, solver) = length(cs) < VgtkCore.sample_size(solver)
# The guess `model`, unrefined: `(InitialFailure, NoRule)` after 0 iterations and `n = 0` (no C call: `original` is
# `nothing`), as VisualGeometryToolkitCore's LevenbergMarquardt reports it.
_unrefined(model, meta::PoseLibRefineMetadata) =
    FitResult(model, VgtkCore.ReturnCode.InitialFailure, VgtkCore.NoRule, SolveStats(), NaN, 0, meta)

# The guess's and K's 3×3 check: an `ArgumentError` naming the size it got, run before the matrix is read into the
# ccall's buffer.
_require_3x3(A::AbstractMatrix, name::String) =
    size(A) == (3, 3) || throw(ArgumentError("$name must be 3×3, got $(join(size(A), '×'))"))

# A K as PoseLib's SIMPLE_PINHOLE camera: (focal, cx, cy), the focal the mean of K₁₁ and K₂₂ — the one statement of
# the mapping, for every refine that hands a K over and for `calibrated_loss_width`.
_pinhole_focal(K::AbstractMatrix) = (K[1, 1] + K[2, 2]) / 2
_simple_pinhole(K::AbstractMatrix) = Float64[_pinhole_focal(K), K[1, 3], K[2, 3]]

"""
    calibrated_loss_width(K1, K2, width) -> Float64

The calibrated (bearing-unit) loss width that PoseLib's relative-pose refine
works at when it is handed `width` in the unit of the points together with the
intrinsics `K1` and `K2` (a refine of pixel correspondences with the
context `(; K1, K2)`): `width · ½(1/f₁ + 1/f₂)`, PoseLib's camera-aware rule
(`robust.cc`), with each `fᵢ = (Kᵢ₁₁ + Kᵢ₂₂)/2`, the focal length of the
`SIMPLE_PINHOLE` camera handed over.

A refine on calibrated data that must deploy at the same operating point as
PoseLib's pixel refine converts its width with this, so the two cannot drift
apart when the camera mapping changes.
"""
calibrated_loss_width(K1::AbstractMatrix, K2::AbstractMatrix, width::Real) =
    width * (0.5 * (1.0 / _pinhole_focal(K1) + 1.0 / _pinhole_focal(K2)))

public calibrated_loss_width

# -----------------------------------------------------------------------------
# The refines (VisualGeometryToolkitCore's `lm_backend_refine`, the backend contract generic its
# `solve(RefineProblem(data, u0[, p]), LevenbergMarquardt(PoseLibSolver(); …))` forwards to)
# -----------------------------------------------------------------------------

const _PhysicalPoints3 = AbstractVector{<:Correspondence{<:StaticVector{3, <:Quantity}, <:StaticVector{2}}}
const _PlainPoints3 = AbstractVector{<:Correspondence{<:StaticVector{3, <:Real}, <:StaticVector{2}}}
const _WHO = "LevenbergMarquardt(PoseLibSolver())"
const _WHO_RELPOSE = _WHO * " of a relative pose from pixels"
const _WHO_ABSPOSE = _WHO * " of an absolute pose"
# The bearing gate's remedy (VisualGeometryToolkitCore's `assert_unit_bearings`): how to build a bearing.
const _BEARING_REMEDY = "Back-project with the camera: `backproject_with_jacobian(camera, u)`."

# The C wrapper's 3×3 in- and output: one column-major (Eigen) block of nine doubles.
const _Mat33 = SMatrix{3, 3, Float64, 9}

# A 3×3 matrix (or a rotation) into the C wrapper's column-major (Eigen) buffer, and a translation into its buffer:
# small heap vectors handed to `ccall` (which the compiler does not keep on the heap where they do not escape).
_colmajor9(A) = collect(Float64, vec(_Mat33(A)))
_c_translation(t) = collect(Float64, SVector{3, Float64}(t))

"""
    lm_backend_refine(::PoseLibSolver, u0, data, p, alg; stops, scale, width) -> FitResult

PoseLib's robust Levenberg–Marquardt refines, the `PoseLibSolver` backend of
VisualGeometryToolkitCore's `lm_backend_refine`. The public entry point is the
solve it forwards from:

```julia
solve(RefineProblem(data, guess[, p]), LevenbergMarquardt(PoseLibSolver(); optimization_strategy);
      stops = LmStops(; max_iter), scale | width) -> FitResult
```

The guess's type and the data's shape select PoseLib's residual:

| guess | `data` | context `p` | PoseLib's residual (codimension) |
|---|---|---|---|
| `HomographyMat` | `x1 => x2`, 2D image points | none | the symmetric transfer error `x2 − π(H x1)`, `x1 − π(H⁻¹ x2)` (2) |
| `FundMat` | `x1 => x2`, 2D image points | none | the Sampson error, over the Bartoli–Sturm parametrization of `F` (rank 2) (1) |
| `EuclideanMap{3}`, a relative pose | `x1 => x2`, 2D points of two views | `(; K1, K2)` | the Sampson error through the two cameras (1) |
| `EuclideanMap{3}`, a relative pose | `f₁ => f₂`, unit bearings with Jacobians `M₁`, `M₂` | none | the tangent-Sampson error (1) |
| `EuclideanMap{3}`, an absolute pose | `X => u`, 3D points to 2D image points | `(; K)` | the reprojection error `project(camera, R X + t) − u` (2) |

A relative pose maps the first camera's frame to the second's; an absolute pose
the world frame to the camera's. A pose guess is a plain `EuclideanMap{3}` (a
labeled one, such as the `CameraExtrinsics` P3P returns, is read unlabeled), except
the physical absolute pose below. The refined model comes back in the caller's
frame: a `HomographyMat{Float64}`, a `FundMat{Float64}`, or an unlabeled
`EuclideanMap{3}` with a `RotMatrix{3, Float64}` rotation and a `Vec3{Float64}`
translation (metres for a physical absolute pose).

The settings: `optimization_strategy` (the algorithm's field) is required, one
of `Irls{TLSShape}()`, `Irls{HuberShape}()`, `Irls{CauchyShape}()`,
[`LeZach`](@ref)`()` or `nothing` (least squares; it takes no `scale` or
`width`). The damping is PoseLib's λ·I, `LevenbergDamping()`, the backend's
default; any other is refused. `stops` is VisualGeometryToolkitCore's
`LmStops`: PoseLib applies its `max_iter` (default 1000) and refuses any other
field that is not `===` to `LmStops()`'s, naming it — PoseLib's own tolerances
apply. Exactly one of `width = w` (the loss width, in the unit of the residual)
or `scale = σ` (a noise scale, or an earlier robust result) is given for a
robust strategy; with `scale` the loss runs at the strategy's efficiency
constant, else `efficiency_constant(shape, d)` at the codimension `d`, times σ.
TLS and Le–Zach have no efficiency constant, so they take `width`.

The width is in the unit of the residual: the points' unit (pixels for pixel
points), or for bearings the unit their Jacobians set.

# The two cameras' and the camera's intrinsics

Each `K` (of the context, in the frame of the points: a pixel matrix for pixel
points) reaches PoseLib as its `SIMPLE_PINHOLE` camera model, with focal length
`(K₁₁ + K₂₂)/2` and principal point `(K₁₃, K₂₃)`. Skew and any difference between
`K₁₁` and `K₂₂` are dropped, so the model is exact only for square pixels and
zero skew. Of a relative pose PoseLib back-projects through each camera, so one
width serves both images even when their focal lengths differ. Of an absolute
pose only the six degrees of freedom of the pose are refined; `K` stays fixed.

# Bearings, and the unit of their width

On unit bearings `f₁ => f₂` (`StaticVector{3}`) the correspondences carry their
3×2 back-projection Jacobians as the attribute columns `M₁` and `M₂`
(`withattrs(cs; M₁, M₂)`; VisualGeometryToolkitCore's
`backproject_with_jacobian(camera, u)` returns each bearing with its Jacobian).
The residual is

    r = (f₂ᵀ E f₁) / √(‖M₂ᵀ E f₁‖² + ‖M₁ᵀ Eᵀ f₂‖²)

in the unit of the coordinates the Jacobians differentiate against, and the
width must be given in that unit: pixels of `u` for `M = ∂f/∂u` from
`backproject_with_jacobian`, radians for an orthonormal basis of the sphere's
tangent plane at `f`. A width in the wrong unit is off by a factor of the focal
length, and nothing reports it: a truncated loss simply keeps the wrong inliers.
No image plane is involved, so a ray at or beyond 90° from the optical axis is
an ordinary input. A bearing that is not finite or not of unit length is
refused with an `ArgumentError`: a point on the z = 1 plane lifted to
`(x, y, 1)` has the right type but is not a bearing.

# The result

A `FitResult`: `u` the refined model; `retcode` and `stop_rule`
`(ReturnCode.Success, NoRule)` when PoseLib stopped on its own tolerances,
`(ReturnCode.MaxIters, IterationCap)` when it used the whole budget;
`stats.nsteps` PoseLib's iteration count; `rms` `NaN` and `resid` `nothing`
(PoseLib reports no residuals); `n` the correspondences refined over; `metadata`
the scale record ([`PoseLibRefineMetadata`](@ref)); `original` the C wrapper's
raw output, its return value (PoseLib's `BundleStats.iterations`) and the output
buffers as written, column-major (`H` or `F`; a pose's `R` and `t`). Fewer
correspondences than the model's minimal sample (4, 7, 5, 3) return the guess
unrefined, `(InitialFailure, NoRule)`, `n = 0`, without calling PoseLib
(`original` `nothing`). A refine ends in the local minimum its guess leads to.

# Refused forms

Each is an `ArgumentError` naming the supported form:

- a relative pose on 2D points without the cameras' intrinsics (calibrated
  2-vectors on the z = 1 plane, which has no point for a ray at or beyond 90°
  from the optical axis); an absolute pose on 3D-to-2D correspondences without
  its `K` (the same plane). PoseLib has no absolute-pose refiner on bearings,
  and its `SIMPLE_PINHOLE` camera cannot project a ray at or beyond 90° either,
  so this package offers no correct absolute-pose refinement for a fisheye
  camera;
- a physical relative pose (its translation is a direction, up to scale: there
  is no unit to strip; hand over `ustrip(m, pose)`);
- a physical absolute-pose guess with plain world points, or the reverse;
- a `K` that is not 3×3.

Any other combination (a stray context, another model) is a `MethodError`.

# A physical absolute pose

A guess whose translation is a length (FlexUnits quantities; a labeled pose, an
estimate's `CameraExtrinsics`, is read unlabeled) with world points `X` that are
lengths: both are stripped in metres once, the plain refinement runs, and the
refined pose comes back with its translation in metres: equal, after
`ustrip(m, ·)`, to the plain refinement's on the inputs' `ustrip(m, ·)`, with
its termination record and scale record. """
VgtkCore.lm_backend_refine(::PoseLibSolver, ::Any, ::Any, ::Any, ::Any)

# The homography and the fundamental matrix: one body (the model type, its minimal solver, the C function, the
# residual's codimension and the name of the output buffer in `original`).
for (M, solver, cfun, d, out) in ((:HomographyMat, :Homography4pt, :poselib_refine_homography, 2, :H),
                                  (:FundMat, :Fundamental7pt, :poselib_refine_fundamental, 1, :F))
    @eval function VgtkCore.lm_backend_refine(::PoseLibSolver, G::$M, cs::_PlanePairs, ::NullParameters,
                                              alg::LevenbergMarquardt; stops::LmStops, scale, width)
        w, meta = _deployed_width(_check_poselib_alg(alg), scale, width, $d)
        max_iter = _poselib_max_iter(stops)
        _too_few(cs, $solver(PoseLibSolver())) && return _unrefined($M{Float64}(Tuple(_Mat33(G))), meta)
        X = _pack(cs)
        G_out = MVector{9, Float64}(undef)
        _require_library()
        iterations = GC.@preserve X ccall(($(QuoteNode(cfun)), libposelib_jl), Cint,
              (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Float64, Cint, Cint, Ptr{Float64}),
              pointer(X), _targets_at(X, 2, length(cs)), length(cs), _colmajor9(G), w, max_iter, _poselib_loss_type(alg.optimization_strategy), G_out)
        original = NamedTuple{(:iterations, $(QuoteNode(out)))}((iterations, SVector(G_out)))
        return _poselib_result($M{Float64}(Tuple(G_out)), original, max_iter, length(cs), meta)
    end
end

# The relative pose on pixels with the two cameras.
function VgtkCore.lm_backend_refine(::PoseLibSolver, pose::PlainEuclideanMap{3}, cs::_PlanePairs, p::NamedTuple,
                                    alg::LevenbergMarquardt; stops::LmStops, scale, width)
    w, meta = _deployed_width(_check_poselib_alg(alg), scale, width, 1)
    max_iter = _poselib_max_iter(stops)
    (; K1, K2) = VgtkCore.read_context(p, (:K1, :K2), (;), _WHO_RELPOSE)
    _too_few(cs, Essential5pt(PoseLibSolver())) && return _unrefined(_returned_pose(pose), meta)
    X = _pack(cs)
    R_out = MVector{9, Float64}(undef); t_out = MVector{3, Float64}(undef)
    _require_library()
    iterations = GC.@preserve X ccall((:poselib_refine_relpose_cam, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          pointer(X), _targets_at(X, 2, length(cs)), length(cs), _colmajor9(pose.R), _c_translation(pose.t), _simple_pinhole(K1), _simple_pinhole(K2),
          w, max_iter, _poselib_loss_type(alg.optimization_strategy), R_out, t_out)
    return _pose_result(SVector(R_out), SVector(t_out), iterations, max_iter, length(cs), meta)
end

# The relative pose on unit bearings with their Jacobians.
function VgtkCore.lm_backend_refine(::PoseLibSolver, pose::PlainEuclideanMap{3}, cs::_BearingPairs, ::NullParameters,
                                    alg::LevenbergMarquardt; stops::LmStops, scale, width)
    w, meta = _deployed_width(_check_poselib_alg(alg), scale, width, 1)
    max_iter = _poselib_max_iter(stops)
    (hasattr(cs, :M₁) && hasattr(cs, :M₂)) || throw(ArgumentError(
        "a relative pose on bearings needs the backprojection Jacobians as attribute " *
        "columns: withattrs(cs; M₁, M₂), from backproject_with_jacobian. They set " *
        "the unit of the width, so there is no defensible default for them."))
    # The 3-vector signature is a NECESSARY condition, not a guarantee: without this the chart would slip through in
    # 3-vector clothing and the 90° exclusion would survive, just invisibly.
    VgtkCore.assert_unit_bearings(sources(cs), "source", "a relative pose on bearings"; remedy = _BEARING_REMEDY)
    VgtkCore.assert_unit_bearings(targets(cs), "target", "a relative pose on bearings"; remedy = _BEARING_REMEDY)
    _too_few(cs, Essential5pt(PoseLibSolver())) && return _unrefined(_returned_pose(pose), meta)
    D = _pack(cs); J = _pack_tangent(cs.M₁, cs.M₂); n = length(cs)
    R_out = MVector{9, Float64}(undef); t_out = MVector{3, Float64}(undef)
    _require_library()
    iterations = GC.@preserve D J ccall((:poselib_refine_relpose_bearing, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint,
           Ptr{Float64}, Ptr{Float64},
           Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          pointer(D), _targets_at(D, 3, n), pointer(J), pointer(J, 6n + 1), n, _colmajor9(pose.R), _c_translation(pose.t),
          w, max_iter, _poselib_loss_type(alg.optimization_strategy), R_out, t_out)
    return _pose_result(SVector(R_out), SVector(t_out), iterations, max_iter, length(cs), meta)
end

# The absolute pose on pixels with the camera.
VgtkCore.lm_backend_refine(::PoseLibSolver, pose::PlainEuclideanMap{3}, cs::_PlainPoints3, p::NamedTuple,
                           alg::LevenbergMarquardt; stops::LmStops, scale, width) =
    _absolute_pose(pose, cs, p, alg, stops, scale, width, identity)

# The layering (VisualGeometryToolkitCore `conventions/units.md`): the ccall boundary takes plain Float64, so a
# physical guess and physical world points are stripped in metres once (`ustrip(m, ·)` of a base-form quantity is its
# stored value, exactly; the points as they are packed), the plain refinement runs, and the refined pose is read back
# in metres once (scale 1, exact), rebuilt through the checked constructor (the refined rotation is verified, as
# before); the rest of the result is the plain refinement's. A labeled guess (an estimate's CameraExtrinsics) is read
# unlabeled.
function VgtkCore.lm_backend_refine(::PoseLibSolver, pose::UnitfulEuclideanMap{3}, cs::_PhysicalPoints3, p::NamedTuple,
                                    alg::LevenbergMarquardt; stops::LmStops, scale, width)
    plain = _absolute_pose(VgtkCore.unlabel(ustrip(VgtkCore.m, pose)), cs, p, alg, stops, scale, width, _metres_point)
    return VgtkCore.map_estimate(refined -> VgtkCore.with_unit(EuclideanMap(refined.R, refined.t), VgtkCore.m), plain)
end

_metres_point(X::StaticVector{3, <:Quantity}) = SVector{3, Float64}(VgtkCore.strip_metres(X))

# `world` maps each world point to its plain Float64 form as it is packed (`identity` for plain points).
function _absolute_pose(pose, cs, p, alg, stops::LmStops, scale, width, world::W) where {W}   # `W`: specialized on `world`
    w, meta = _deployed_width(_check_poselib_alg(alg), scale, width, 2)
    max_iter = _poselib_max_iter(stops)
    (; K) = VgtkCore.read_context(p, (:K,), (;), _WHO_ABSPOSE)
    _require_3x3(K, "K")
    _too_few(cs, P3P(PoseLibSolver())) && return _unrefined(_returned_pose(pose), meta)
    X = _pack(cs, world)             # the world points, then the image points
    R_out = MVector{9, Float64}(undef); t_out = MVector{3, Float64}(undef)
    _require_library()
    iterations = GC.@preserve X ccall((:poselib_bundle_adjust_cam, libposelib_jl), Cint,
          (Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Float64, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
          _targets_at(X, 3, length(cs)), pointer(X), length(cs), _colmajor9(pose.R), _c_translation(pose.t), _simple_pinhole(K),
          w, max_iter, _poselib_loss_type(alg.optimization_strategy), R_out, t_out)
    return _pose_result(SVector(R_out), SVector(t_out), iterations, max_iter, length(cs), meta)
end

# A refined pose from the C wrapper's output, read into static values where the buffers were written (rotation
# column-major, translation), so the buffers do not escape the refine.
@inline _pose_result(R::SVector{9, Float64}, t::SVector{3, Float64}, iterations, max_iter, n, meta) =
    _poselib_result(_plain_pose(_Mat33(Tuple(R)), Vec3{Float64}(t)), (; iterations, R, t), max_iter, n, meta)

# -----------------------------------------------------------------------------
# The refused forms
# -----------------------------------------------------------------------------

# A physical relative pose: up to scale, so its translation has no unit to strip.
VgtkCore.lm_backend_refine(::PoseLibSolver, ::UnitfulEuclideanMap{3}, ::Union{_PlanePairs, _BearingPairs}, _p, ::LevenbergMarquardt; kw...) =
    throw(ArgumentError("a relative pose is up to scale, so its translation is a direction, not a length; supply the " *
        "plain guess, `ustrip(m, pose)`"))

# A relative pose on 2D points without the cameras: calibrated 2-vectors, the z = 1 chart.
VgtkCore.lm_backend_refine(::PoseLibSolver, ::PlainEuclideanMap{3}, cs::_PlanePairs, ::NullParameters, ::LevenbergMarquardt; kw...) =
    throw(ArgumentError(
        "a relative pose takes pixels with the two cameras' intrinsics, or unit BEARINGS (3-vectors). Got " *
            "$(eltype(sources(cs))) => $(eltype(targets(cs))) without intrinsics. Calibrated 2-vectors " *
            "imply z = 1 — the gnomonic chart — which does not exist for a ray at the horizon. Either hand over " *
            "pixels and the intrinsics, `RefineProblem(cs, pose, (; K1, K2))`, or back-project to bearings and attach " *
            "the Jacobians, `RefineProblem(withattrs(Correspondences(f₁, f₂); M₁, M₂), pose)` from " *
            "`backproject_with_jacobian(camera, u)`."))

# An absolute pose without its K: the targets would be calibrated points on the z = 1 chart.
VgtkCore.lm_backend_refine(::PoseLibSolver, ::EuclideanMap{3}, cs::AbstractVector{<:Correspondence{<:StaticVector{3}, <:StaticVector{2}}},
                           ::NullParameters, ::LevenbergMarquardt; kw...) =
    throw(ArgumentError(
        "an absolute pose takes PIXELS with the camera's intrinsics. Got $(eltype(sources(cs))) => " *
            "$(eltype(targets(cs))) without them: the targets would be points on the z = 1 chart, which does not " *
            "exist for a ray at the horizon. Supply the same correspondences as pixels, their intrinsics and a pixel " *
            "width: `RefineProblem(cs, pose, (; K))`."))

# Mixed units of an absolute pose.
VgtkCore.lm_backend_refine(::PoseLibSolver, ::UnitfulEuclideanMap{3}, ::_PlainPoints3, ::NamedTuple, ::LevenbergMarquardt; kw...) =
    throw(ArgumentError("an absolute pose: the guess is physical but the world points are plain; attach their unit (X .* m)"))
VgtkCore.lm_backend_refine(::PoseLibSolver, ::PlainEuclideanMap{3}, ::_PhysicalPoints3, ::NamedTuple, ::LevenbergMarquardt; kw...) =
    throw(ArgumentError("an absolute pose: the world points are physical but the guess is plain; attach its unit, " *
        "EuclideanMap(pose.R, pose.t .* m)"))

# Any other guess, data or context (a stray context, another model): the tuple's MethodError, stated here so that
# Core's fallback for a backend without a refine does not claim PoseLib has none.
VgtkCore.lm_backend_refine(backend::PoseLibSolver, u0, data, p, alg; kw...) =
    throw(MethodError(VgtkCore.lm_backend_refine, (backend, u0, data, p, alg)))

# =============================================================================
# motion(::HomographyMat) — moved to VisualGeometryToolkitCore
# =============================================================================
# The Malis–Vargas homography decomposition is now a NATIVE method in
# VisualGeometryToolkitCore (src/geometry/fundmat.jl), ported off the C++
# `poselib_motion_from_homography` (validated bit-for-bit against it). PoseLib no
# longer defines `motion(::HomographyMat)`, so the VGC method is authoritative.
# The C wrapper symbol stays in the shared library but is unused from Julia.


end # module PoseLib
