# =============================================================================
# RANSAC Rotation Problem — a pure rotation between two calibrated views
# =============================================================================
#
# `RotationProblem` is the `MinimalSolveProblem{RotMatrix}` alias: the rotation
# `R` with `f₂ ∥ R f₁` between the unit bearings of two views of a camera that
# only rotated (a panorama sweep, a rotating rig) — the INFINITE HOMOGRAPHY of
# a calibrated camera, on rays rather than on the z = 1 chart, so the whole
# field of a fisheye takes part (past 90°). Its construction, minimal solve
# (VGC `Rotation2pt`: `kabsch` on two pairs), sampler and RANSAC wiring are the
# family's (cspond.jl), patterned on the essential problem: the data are unit
# bearings from the cameras' own back-projection carrying their per-view
# back-projection Jacobians as the `M₁`/`M₂` attribute columns, which put every
# residual and threshold in PIXELS.
#
# ONE Sampson machinery. The constraint is the homography's transfer form on
# bearings — VGC's `constraint_with_jacobians(TransferForm(H), f₁, f₂; M₁, M₂)`
# with `H = R`: the transferred ray in `f₂`'s tangent plane, `G_x` the two-sided
# pixel Jacobian (both images noisy, Σ_g = P(R M₁M₁ᵀRᵀ + M₂M₂ᵀ)Pᵀ). Scoring
# whitens it through VGC's `whiten` (the default `whitened_residual`, behind a
# hemisphere gate), the chart LM and the lifts read VGC's transfer rows with the
# whitening frozen at the anchor (`transfer_whitening!` / `transfer_rows!` /
# `transfer_jacobian_rows!`, their bearing methods) — exactly the homography's
# chart rows (homography.jl) on bearings. Nothing here restates g/‖∇g‖.
#
# The chart is the rotation block of VGC's `LeftRigidManifold6` (translation
# fixed at zero: `restrict(…, 1:3)`), the left exp perturbation R(ω) = exp(ω)·R.
# The refit is the family's chart LM (`refine!` → `_refine!(::NonlinLsq, …)`);
# the HQ/GNC lifts (LocalOptimizationDoneRight) read the same rows through the
# generic provider (manifold_providers.jl).
# =============================================================================


codimension(::Type{<:MinimalSolveProblem{<:RotMatrix}}) = 2    # f₂ × R f₁ = 0: two equations in f₂'s tangent plane
chart_dof(::Type{<:RotationProblem}) = 3

"""
    RotationProblem(camera1, camera2, pixel_cs; kw...)

A `RotationProblem` from PIXEL correspondences and the cameras of the two views
(the same camera twice for a rotating camera, `RotationProblem(cam, cam, cs)`):
each view back-projected with its own camera through
`backproject_with_jacobian`, the Jacobians landing with their endpoints by
construction (as [`EssentialProblem`](@ref)`(camera1, camera2, pixel_cs)`). The
attribute columns of `pixel_cs` (the `:score` prior) travel along.
"""
function RotationProblem(camera1, camera2, pixel_cs; kw...)
    fm₁ = map(u -> VgtkCore.backproject_with_jacobian(camera1, u), sources(pixel_cs))
    fm₂ = map(u -> VgtkCore.backproject_with_jacobian(camera2, u), targets(pixel_cs))
    cols = merge(VgtkCore.attrs_of(pixel_cs), (M₁ = last.(fm₁), M₂ = last.(fm₂)))
    return RotationProblem(VgtkCore.Csponds(first.(fm₁), first.(fm₂), cols); kw...)
end

"""
    RotationProblem(bearings::Csponds; sampler = nothing, coincident_tol = COINCIDENT_PX)

A `RotationProblem` from UNIT-BEARING correspondences `f₁ => f₂` (3-vectors from
the cameras' back-projection) carrying their 3×2 back-projection Jacobians as
the attribute columns `M₁`/`M₂` (`withattrs(cs; M₁, M₂)`), which put the
residual, the threshold and the sample gate's `coincident_tol` in PIXELS — so,
as for the essential problem, there is no default for them. A `:score` column
is the PROSAC prior. Unit norm and finiteness are checked once, here.
"""
function RotationProblem(bearings::VgtkCore.Csponds; sampler::Union{Nothing, AbstractSampler} = nothing,
                         coincident_tol::Real = VgtkCore.COINCIDENT_PX)
    (eltype(sources(bearings)) <: StaticArrays.StaticVector{3} && eltype(targets(bearings)) <: StaticArrays.StaticVector{3}) ||
        throw(ArgumentError("RotationProblem takes unit BEARINGS (3-vectors), got $(eltype(sources(bearings))) => " *
                            "$(eltype(targets(bearings))); use `backproject_with_jacobian(camera, u)`"))
    _assert_unit_bearings(sources(bearings), "source", "RotationProblem")
    _assert_unit_bearings(targets(bearings), "target", "RotationProblem")
    (VgtkCore.hasattr(bearings, :M₁) && VgtkCore.hasattr(bearings, :M₂)) || throw(ArgumentError(
        "RotationProblem requires the back-projection Jacobians as attribute columns: `withattrs(cs; M₁ = ..., M₂ = ...)`, " *
            "from `backproject_with_jacobian(camera, u)` — they put the residual and the threshold in PIXELS"))
    _assert_finite_jacobians(bearings.M₁, "M₁", "RotationProblem")
    _assert_finite_jacobians(bearings.M₂, "M₂", "RotationProblem")
    smplr = _problem_sampler(bearings, 2, sampler)
    cs = _check_noise_support(RotMatrix, _canonicalize_pairs(Float64, bearings))
    return MinimalSolveProblem{RotMatrix{3, Float64, 9}, 2}(cs, nothing, cs, nothing, VgtkCore.Rotation2pt(), nothing, smplr;
                                                            coincident_tol)
end

# A Pair vector cannot carry the Jacobian columns: refused as the essential problem refuses it.
RotationProblem(::AbstractVector{<:Pair{<:StaticArrays.StaticVector{3}, <:StaticArrays.StaticVector{3}}}; kw...) =
    throw(ArgumentError("RotationProblem needs the back-projection Jacobians, which a vector of Pairs cannot carry: " *
                        "`withattrs(Csponds(f₁, f₂); M₁, M₂)`, from `backproject_with_jacobian(camera, u)`"))

# The sample gate: the bearing coincidence gate of VGC (a sample's two bearings
# the same detection in either view, the angle converted to pixels through M —
# keyed on the essential matrix's type, whose bearing form it is), at the
# problem's pixel tolerance. A parallel or antiparallel pair is the minimal
# solver's own refusal (`Rotation2pt`: count 0).
test_sample(p::RotationProblem, idx::AbstractVector{Int}) =
    test_sample(EssentialMat, p.cs, idx; tol_px = p.coincident_tol)

# The infinite homography of a rotation: the transfer form's matrix.
@inline _transfer_homography(R) = HomographyMat(Tuple(SMatrix{3, 3}(R)))   # eltype-generic (ForwardDiff through the chart)

# The scoring constraint — VGC's bearing transfer form at H = R (its rounding
# scale is the exact-zero rule: no 5th value).
@inline function constraint_with_jacobians(p::RotationProblem, R::RotMatrix, i::Int)
    @inbounds return VgtkCore.constraint_with_jacobians(
        VgtkCore.TransferForm(_transfer_homography(R)), sources(p.cs)[i], targets(p.cs)[i]; M₁ = p.cs.M₁[i], M₂ = p.cs.M₂[i])
end

# The whitened residual: VGC's default (`whiten` of the constraint above, in
# pixels) behind the hemisphere gate — the transferred ray must be in f₂'s
# hemisphere, since an antipodal pair also has a small tangent residual.
@inline function whitened_residual(p::RotationProblem, R::RotMatrix, i::Int)
    @inbounds dot(R * sources(p.cs)[i], targets(p.cs)[i]) > 0 || return Inf
    return VgtkCore.whitened_distance(VgtkCore.whitened_constraint_residual(constraint_with_jacobians(p, R, i)))   # VGC's default, behind the gate
end

# The sub-detector half-width refusal is the bearing problems' (essential.jl
# `_check_scoring` on `_BearingProblem`), in the same pixel unit.

# =============================================================================
# The chart — the rotation block of LeftRigidManifold6, and the transfer rows
# =============================================================================

const _ROTATION_BLOCK = SVector(1, 2, 3)
_chart(::RotationProblem, R::RotMatrix) =
    VgtkCore.restrict(LeftRigidManifold6(SMatrix{3, 3, Float64, 9}(R), zero(SVector{3, Float64})), _ROTATION_BLOCK)
_model_at(::RotationProblem, ch::VgtkCore.SubsetManifold) = RotMatrix{3, Float64, 9}(SMatrix{3, 3, Float64, 9}(model(ch).R))

# The rotation at chart coordinates `a`, and its directions ∂R/∂ω_q there (the
# first nine rows of the chart's retraction Jacobian, column-major vec(R)).
@inline _rotation_at(ch, a) = _transfer_homography(retract(ch, a).R)
@inline function _rotation_directions(ch, a)
    Jr = retract_jacobian(ch, a)
    return ntuple(q -> SMatrix{3, 3, Float64, 9}(ntuple(k -> Jr[k, q], Val(9))), Val(3))
end

# The per-point whitening blocks frozen at the chart's anchor (the homography's
# policy), one buffer per (task, n) under the one reuse policy.
_rotation_whitening_buffer(n::Int) = task_buffers(
    Vector{SMatrix{2, 2, Float64, 4}}, () -> Vector{SMatrix{2, 2, Float64, 4}}(undef, n), (:rvg_rotation_W, n)
)
function _lm_state(p::RotationProblem, ch, sqrtw)
    W = _rotation_whitening_buffer(data_size(p))
    transfer_whitening!(W, _rotation_at(ch, VgtkCore.zero_tangent(ch)), sources(p.cs), targets(p.cs); M₁ = p.cs.M₁, M₂ = p.cs.M₂)
    return ChartLm(ch, p, sqrtw, W)
end

function residual_rows!(r::AbstractVector, lm::ChartLm{<:VgtkCore.SubsetManifold, <:RotationProblem}, a)
    cs = lm.problem.cs
    return transfer_rows!(r, _rotation_at(lm.ch, a), lm.aux, sources(cs), targets(cs), lm.sqrtw)
end
function residual_jacobian_rows!(J::AbstractMatrix, lm::ChartLm{<:VgtkCore.SubsetManifold, <:RotationProblem}, a)
    cs = lm.problem.cs
    return transfer_jacobian_rows!(J, _rotation_at(lm.ch, a), _rotation_directions(lm.ch, a), lm.aux, sources(cs), targets(cs), lm.sqrtw)
end
