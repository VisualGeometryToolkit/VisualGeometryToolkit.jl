# =============================================================================
# RANSAC Radial-Homography Problem — one-sided division model (H, k, c)
# =============================================================================
#
# `RadialHomographyProblem` is the `MinimalSolveProblem` instantiation for
# VisualGeometryToolkitCore's `RadialHomography` model class (Wadenbäck, Valtonen
# Örnhag, Edstedt, "Radially Distorted Homographies, Revisited", 3DV 2026 —
# the HomLib `get_one_sided` port): a plane-induced homography `H` composed
# with a one-parameter division-model back-projection on the TARGET side,
# about a FIXED center.
#
# Structural differences from `HomographyProblem`:
#
#   1. COORDINATE NORMALIZATION IS FIXED-CENTER on the target side. The
#      division model is defined about its center — a centroid-centered
#      Hartley frame would relocate the center and leave the model class.
#      The target frame is Hartley ABOUT THE CENTER (translate the
#      distortion center to the origin, scale to mean distance √2 from it);
#      sources keep the ordinary centroid Hartley frame
#      (`CspondNormalization(RadialHomography, cs; center)`). The center is
#      problem DATA (like the camera of `AbsolutePoseProblem`): the
#      constructor requires it.
#   2. THE MINIMAL SOLVE RUNS IN THE NORMALIZED FRAME (the 5-pt kernel's
#      division model is origin-centered) and every candidate is
#      composed back out of it, so problem-level models live in the DATA frame
#      with center and k in pixel units — self-contained for consumers.
#   3. RESIDUALS ARE MEASUREMENT-SPACE REPROJECTIONS `‖dᵢ − transfer(m, sᵢ)‖`
#      in pixels (`radial_transfer` is ray-based, so predictions survive the
#      hemisphere boundary for k < 0); a ray outside a k > 0 model's
#      representable field gets a large finite sentinel residual.
#
# LO support: the 4-arg `refine!(lo_ws, p, θ, w)` is a JOINT 9-dof chart LM —
# sl(3) ⊕ k about the seed — on the weighted measurement-space reprojection
# (see the fit section below), driven by the posterior-weight IRLS outer
# loops (`Irls`, LocalOptimizationDoneRight's `StudentTUniform`) with their best-cost tracking; the chart-lift family
# (HQ/GNC) rides the same kernels through its `ManifoldLiftProvider`.
#
#   4. TWO HYPOTHESIS PATHS. `minimal = :radial` (the default) samples five
#      correspondences and solves (H, k) with the 5-point kernel — whose
#      redundant-equation root pick fails on boards small enough that k is
#      not identifiable from one board. `minimal = :homography` samples FOUR
#      and solves the plain homography (`Homography4pt`): the hypothesis is a
#      radial homography with no distortion term, scored with the same
#      reprojection residual, and the local optimization estimates (H, k) on
#      its consensus — the global search for H paired with the (H, k)
#      optimizer. The distorted-away correspondences a plane homography
#      cannot hold enter at the re-score, once k is in the model. In front of
#      the nonlinear step, k is seeded by the closed-form least squares in k
#      given H (`_radial_k_seed`) — the linear fit, never a zero start.
# =============================================================================

# `K` is the minimal sample size — 5 for the radial 5-point kernel, 4 for
# plain-homography hypotheses (`minimal = :homography`, below). A partial
# application, as the cspond.jl aliases.
const RadialHomographyProblem = MinimalSolveProblem{VgtkCore.RadialHomography{T}, K, T} where {T <: AbstractFloat, K}

codimension(::Type{<:MinimalSolveProblem{<:VgtkCore.RadialHomography}}) = 2  # d_g = 2: reprojection pair

# The fixed-center normalization for RadialHomography is VisualGeometryToolkitCore's
# `CspondNormalization(RadialHomography, cs; center)`.

# The minimal sample size of each hypothesis path — the one statement of it
# (the constructor's `K` and `find_radial_homography`'s early exit).
function _radial_sample_size(minimal::Symbol)
    minimal === :radial && return 5
    minimal === :homography && return 4
    throw(ArgumentError("minimal must be :radial or :homography, got :$minimal"))
end

"""
    RadialHomographyProblem(pairs; center, sampler=nothing, backend=NativeSolver(),
                            extra_check=nothing, minimal=:radial)

Construct a radially distorted homography problem (one-sided division
model on the target side) from 2D-2D point correspondences.

- `center`: the distortion center in TARGET pixels, REQUIRED — the target
  side is normalized about it (Hartley, mean distance √2). Data, not
  configuration: the division model is meaningless without its center.
- `extra_check`: `true` (the default when unset) keeps only the cubic root
  consistent with the redundant 4.5-pt equation per minimal sample (the
  reference's RANSAC setting); `false` scores all real roots. It selects a
  root of the 5-point kernel, so supplying it with `minimal = :homography` is an
  `ArgumentError` rather than a setting that silently does nothing.
- `minimal = :radial`: the hypothesis path — `:radial` samples five
  correspondences for the 5-point (H, k) kernel; `:homography` samples four
  for the plain homography, the (H, k) local optimization supplying k on the
  consensus (header, point 4).
- `pairs` (and its `:score` prior)/`sampler`/`backend`: as [`HomographyProblem`](@ref).
"""
function RadialHomographyProblem(
        pairs;
        center,
        sampler::Union{Nothing, AbstractSampler} = nothing,
        backend::AbstractSolverBackend = NativeSolver(),
        extra_check::Union{Nothing, Bool} = nothing,
        minimal::Symbol = :radial
    )
    K = _radial_sample_size(minimal)
    minimal === :homography && extra_check !== nothing && throw(
        ArgumentError(
            "extra_check selects a root of the radial 5-point kernel; it has no " *
                "meaning for minimal = :homography (four-point hypotheses)"
        )
    )
    slv = minimal === :radial ? VgtkCore.Homography5ptRadial(backend, something(extra_check, true)) :
        VgtkCore.Homography4pt(backend)
    return _make_centered_problem(VgtkCore.RadialHomography, K, pairs, slv, sampler, center, nothing)
end

# The model class of a hypothesis: a plain-homography hypothesis (the 4-point
# path) is a radial homography with no distortion term, in the solver's
# origin-centred frame like the kernel's own output; `solve` (cspond.jl
# `_candidate`) composes it back into the data frame.
_model_class(::RadialHomographyProblem, m::VgtkCore.RadialHomography) = m
_model_class(::RadialHomographyProblem{T}, H::HomographyMat) where {T <: AbstractFloat} =
    VgtkCore.RadialHomography{T}(HomographyMat{T}(Tuple(H)), zero(T))

# The sample gate on the normalized view, SCALE-FREE as H's and F's
# (`_COINCIDENT_NORMALIZED`, cspond.jl): the target distinctness at the same
# constant in the fixed-centre normalized frame (VGC's source separation is that
# constant already). Until 2026-09-30 it was 1 px carried in by the
# normalization's scale — a claim about the caller's units the H/F gate stopped
# making. The model gate is the family's (cspond.jl `test_model`).
test_sample(p::RadialHomographyProblem, idx::AbstractVector{Int}) =
    VgtkCore.test_sample(model_type(p), p.cs_hartley, idx; target_tol = _COINCIDENT_NORMALIZED)

# The consensus gate on the data frame, as for H.
test_consensus(
    p::RadialHomographyProblem, m::VgtkCore.RadialHomography,
    mask::BitVector
) =
    VgtkCore.test_consensus(m, p.cs, mask)

# Extent-consensus hook (BoundedProblem): quad winding on the H part (the
# division back-projection is orientation-preserving about its center) PLUS
# every extent corner in the model's field and front hemisphere — VGC's one
# edge rule, `radial_in_front`: an (H, k) hypothesis that pushes a board corner
# to or past the 90° circle spans ≥ 180° across a planar target and is
# degenerate. The uncalibrated analog of the pose path's corner-cheirality gate.
@inline function _check_extent(
        et::WithExtent{<:HyperRectangle{2}}, _p,
        m::VgtkCore.RadialHomography, _mask
    )
    is_orientation_preserving(homography(m), et.extent) || return false
    return all(s -> VgtkCore.radial_in_front(m, s), rect_corners(et.extent))
end

# =============================================================================
# The scoring residual — measurement-space reprojection distance (pixels)
# =============================================================================

# The reprojection distance of correspondence `i`, Jacobian-free
# (`residuals!` is VisualGeometryToolkitCore's loop over it). A k > 0 model can
# place a predicted ray outside its representable field (`radial_transfer` →
# nothing): VGC's one out-of-field residual (`OUT_OF_FIELD_RESIDUAL`), finite so
# the scoring sweep's sort and squares stay well-defined while the point is
# buried among the outliers.
@inline function whitened_residual(
        p::RadialHomographyProblem, m::VgtkCore.RadialHomography, i::Int
    )
    d̂ = VgtkCore.radial_transfer(m, @inbounds sources(p.cs)[i])
    d̂ === nothing && return VgtkCore.OUT_OF_FIELD_RESIDUAL
    v = @inbounds targets(p.cs)[i]
    return Float64(norm(SVector(d̂[1] - v[1], d̂[2] - v[2])))
end

# =============================================================================
# The chart — sl(3) ⊕ k (9 dof, codim 2), measurement-space reprojection rows
# =============================================================================
#
# The house chart shape (H: sl(3)), extended by one Euclidean dimension for k:
# VGC's `minimal_manifold(::RadialHomography)`, the product of
# HomographyManifold8 about H₀ and the additive k, embedded back into a
# RadialHomography with its fixed distortion centre. The weighted objective is
# the TRUE measurement-space reprojection
#
#   Σᵢ wᵢ ‖dᵢ − project(m, T₋c·H(a)·s̃ᵢ)‖²,  H(a) = H₀(I + Σ aᵩGᵩ),  k = k₀ + a₉
#
# — no DLT, no alternation, and no dehomogenized "undistorted point" anywhere:
# the projection stays in ray form, so back-hemisphere geometry needs no
# special-casing. The rows are VGC's radial transfer rows: one field-edge
# predicate for the rows, their Jacobian and the scorer's `radial_transfer`; an
# out-of-field ray is a constant residual with a ZERO Jacobian row (charged,
# never rewarded, for leaving the field). The chart-LM refit and the HQ/GNC
# lift share the chart and the rows (interface.jl `ChartLm`).
# =============================================================================

chart_dof(::Type{<:RadialHomographyProblem}) = 9

_radial_chart(H₀, k₀, c) = minimal_manifold(
    VgtkCore.RadialHomography{Float64}(HomographyMat{Float64}(Tuple(H₀)), Float64(k₀), c)
)

# The refit's model in the problem's type (`RadialHomography{T}`, as `solve`
# returns): a Float32 problem's LO result converts back into its workspace.
function _model_at(::RadialHomographyProblem{T}, ch) where {T <: AbstractFloat}
    m = model(ch)
    return VgtkCore.RadialHomography{T}(
        HomographyMat{T}(Tuple(homography(m))), T(division_coeff(m)), Point2{T}(center(m)[1], center(m)[2])
    )
end

# √w-weighted radial-transfer rows in the (H, k) chart at `xv` (VGC's `residual_rows!`).
function residual_rows!(f, lm::ChartLm{<:Any, <:RadialHomographyProblem}, xv)
    cs = lm.problem.cs
    return radial_transfer_rows!(f, retract(lm.ch, SVector{9}(xv)), sources(cs), targets(cs), lm.sqrtw)
end

# The chart's ambient directions are [vec H; k] (10×9): the first eight columns'
# vec H part is dH (VGC's `retract_directions`); the rows function adds
# the k column itself.
function residual_jacobian_rows!(J, lm::ChartLm{<:Any, <:RadialHomographyProblem}, xv)
    x = SVector{9}(xv)
    dH = VgtkCore.retract_directions(lm.ch, x, Val(8))                 # the sl(3) columns' vec H part
    return radial_transfer_jacobian_rows!(J, retract(lm.ch, x), dH, sources(lm.problem.cs), lm.sqrtw)   # 2n × 9: sl(3) directions + k
end

"""
    _radial_k_seed(u, v, H, c, weights) -> k

The closed-form weighted least squares in k given H — the LINEAR fit that
seeds the joint (H, k) step for a hypothesis carrying no distortion term (a
plain-homography hypothesis, `minimal = :homography`). An observation's
division-model lift `(d − c ; 1 + k‖d − c‖²)` is proportional to the ray
`(x, y, z) = T₋c·H·s̃`, so with `λ = ((d − c)·(x, y)) / (x² + y²)` from the
first two components the third reads `k·r² = λz − 1`: one unknown, linear.
Scale-free in H (λz is). Zero weights are skipped; a set with no radius
returns 0.
"""
function _radial_k_seed(u, v, H::SMatrix{3, 3, Float64, 9}, c::SVector{2, Float64}, weights)
    num = 0.0; den = 0.0
    @inbounds for i in eachindex(u)
        wi = Float64(weights[i])
        wi > 0 || continue
        x, y, z = radial_ray(H, c, u[i])                     # VGC's centred ray T₋c·H·s̃
        ρ² = x * x + y * y
        ρ² > 1.0e-12 || continue
        dx = Float64(v[i][1]) - c[1]; dy = Float64(v[i][2]) - c[2]
        r² = dx * dx + dy * dy
        λ = (dx * x + dy * y) / ρ²
        num += wi * r² * (λ * z - 1)
        den += wi * r² * r²
    end
    return den > 0 ? num / den : 0.0
end

"""
    _radial_seed_chart(p, θ, weights) -> chart

THE seed of every (H, k) local optimization — the refit (`_refit_chart`) and
the HQ/GNC lift (its provider's `lift_init`): VGC's sl(3) ⊕ k chart about the
unit-Frobenius H of `θ`. A hypothesis carrying no distortion term (a
plain-homography hypothesis) gets the linear k on the correspondences
`weights` admits (`_radial_k_seed`) — never a zero start.
"""
function _radial_seed_chart(p::RadialHomographyProblem, θ::VgtkCore.RadialHomography, weights)
    H₀ = unit_normalize(SMatrix{3, 3, Float64, 9}(homography(θ)))
    iszero(division_coeff(θ)) || return _radial_chart(H₀, division_coeff(θ), center(θ))
    c = SVector{2, Float64}(center(θ)[1], center(θ)[2])
    return _radial_chart(H₀, _radial_k_seed(sources(p.cs), targets(p.cs), H₀, c, weights), center(θ))
end
_refit_chart(p::RadialHomographyProblem, θ, weights) = _radial_seed_chart(p, θ, weights)
