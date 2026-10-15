# =============================================================================
# RANSAC Radial-Camera Problem — the 1-D radial camera of one planar board
# =============================================================================
#
# `RadialCameraProblem` estimates VisualGeometryCore's `RadialCamera` — the
# 2×3 block P with the RAW centred pixel of plane point X̃ on the radial line
# through P·X̃: rows 1–2 of the plane homography, invariant to focal length
# and to any radially symmetric distortion about the centre — from
# plane → pixel correspondences of a board of known geometry. It is the
# uncalibrated verification model that needs no distortion parameter: one
# linear equation per correspondence (`constraint_with_jacobians`), five
# determine P (`radial_camera_5pt`, the null vector — no roots to pick), and
# the residual is the perpendicular pixel distance to the predicted radial
# line, codimension 1. Points near the centre are weakly constrained (a ray's
# angular resolution scales as 1/r) — the trade for distortion invariance.
#
# Sampson whitening of the algebraic row is exact here (only the pixel is
# uncertain), so `residuals!` reports g/‖∂g/∂x‖ directly. The LO step is the
# chart LM on the sphere chart of vec(P) (below).
# The centre is problem DATA (required), as for the radial homography. The
# problem owns ONE normalization, `CspondNormalization(RadialCamera, cs; center)`
# (Hartley at the plane centroid, Hartley about the centre on the image side —
# the strategy `fit_radial_camera` solves in too), built at construction: the
# minimal solves run on the normalized view and their cameras come back to
# pixels about the centre (`p.context`). Scoring and the LO stay on the RAW pixels
# (`RadialCamera` carries its centre; the constraint subtracts it).
# =============================================================================

# The `MinimalSolveProblem` instantiation for VGC's `RadialCamera` (bounded
# free parameters, as the cspond.jl aliases): `cs` the canonical plane => pixel
# data (RAW pixels, attribute columns kept), `cs_hartley` = `cs_solve` its
# normalized view (the minimal solve's input), `solver` the 5-point kernel about
# the normalized frame's origin, `context` the distortion centre in pixels.
const RadialCameraProblem = MinimalSolveProblem{VisualGeometryCore.RadialCamera{Float64}, 5}

"""
    RadialCameraProblem(pairs; center, sampler = nothing)

The radial-camera problem from `plane_point => raw_pixel` pairs of one board.
`center` — the distortion centre in pixels — is REQUIRED (data, not
configuration); a `Csponds` `:score` attribute column drives PROSAC sampling,
and every attribute column travels with the data.
"""
function RadialCameraProblem(
        pairs;
        center,
        sampler::Union{Nothing, AbstractSampler} = nothing,
    )
    slv = VisualGeometryCore.RadialCamera5pt(; center = SVector(0.0, 0.0))
    return _make_centered_problem(
        VisualGeometryCore.RadialCamera, 5, pairs, slv, sampler, center, Point2{Float64}(center[1], center[2])
    )
end

codimension(::Type{<:MinimalSolveProblem{<:VisualGeometryCore.RadialCamera}}) = 1   # one line-distance per correspondence

# The radial camera's datum: a camera composed back into pixels (`_candidate`,
# cspond.jl — the minimal solve's roots and `fit_radial_camera`'s solution)
# carries the problem's centre EXACTLY, not the composition's rounding of it.
_with_datum(p::RadialCameraProblem, P::VisualGeometryCore.RadialCamera) =
    VisualGeometryCore.RadialCamera(radial_rows(P), p.context)

# The constraint of one correspondence on its RAW pixel (VGC's; the camera
# carries its centre and the constraint subtracts it). `residuals!` is
# VisualGeometryCore's default through it: the perpendicular distance of the
# pixel from its predicted radial line — the Sampson distance of the scalar
# constraint with the pixel the only uncertain quantity, exact — by VGC's one
# whitening, degeneracy rule included (0 / ±Inf where the gradient vanishes).
@inline constraint_with_jacobians(p::RadialCameraProblem, P::VisualGeometryCore.RadialCamera, i::Int) =
    @inbounds VisualGeometryCore.constraint_with_jacobians(P, sources(p.cs)[i], targets(p.cs)[i])
provides_solver_covariance(::Type{<:RadialCameraProblem}) = true   # predictive scoring (covariance.jl)

# =============================================================================
# The chart — VGC's SphereManifold on vec(P) (5 dof, codim 1)
# =============================================================================
#
# P is homogeneous in ℝ⁶, so the chart about the anchor is
# P(a) = normalize(P₀ + Σᵢ aᵢBᵢ) with B₁..B₅ an orthonormal basis of vec(P₀)'s
# complement (one Householder reflection taking e₁ to P₀ supplies it) — the
# scale direction deleted without fixing an entry: VGC's
# `minimal_manifold(::RadialCamera)`, the centre carried by the embedding. The
# residual of correspondence i is the perpendicular distance of its centred
# pixel from the predicted radial line,
#
#     dᵢ = g / ‖G_x‖,   g = x₁(p₂·X̃) − x₂(p₁·X̃),   G_x = (p₂·X̃, −p₁·X̃)
#
# with (x₁, x₂) the pixel MINUS the model's centre (`constraint_with_jacobians`
# subtracts it, so raw pixels go in) — Sampson, exact: only the pixel is
# uncertain. A degenerate correspondence (exactly vanishing gradient; ±Inf
# under the rule) contributes a zero row, as a zero weight would. No SVD
# anywhere in the RANSAC loop: the minimal solve is closed form and the refit
# is this chart. The chart-LM refit and the HQ/GNC lift share the chart and the
# rows (interface.jl `ChartLm`).
# =============================================================================

chart_dof(::Type{<:RadialCameraProblem}) = 5

# A camera with no finite, non-zero direction has no chart: the refit declines.
function _refit_chart(::RadialCameraProblem, θ::VisualGeometryCore.RadialCamera, _weights)
    nv = norm(SVector{6, Float64}(radial_rows(θ)))
    (isfinite(nv) && nv > 0) || return nothing
    return minimal_manifold(θ)
end

# √w-weighted point–radial-line distances in the chart at `a` (VGC's `residual_rows!`).
function residual_rows!(f, lm::ChartLm{<:Any, <:RadialCameraProblem}, a)
    cs = lm.problem.cs; plane = sources(cs); pixel = targets(cs); sqrtw = lm.sqrtw
    P = retract(lm.ch, SVector{5}(a[1], a[2], a[3], a[4], a[5]))
    @inbounds for i in 1:length(cs)
        g, G_x, G_θ = VisualGeometryCore.constraint_with_jacobians(P, plane[i], pixel[i])
        d = first(whitened_rows(SampsonWhitening(), g, G_x, G_θ))
        f[i] = isinf(d) ? 0.0 : sqrtw[i] * d
    end
    return f
end

# ∂d/∂a: VGC's signed-Sampson derivative (`signed_sampson_jacobian`) along the
# chart directions A = ∂vec(P)/∂a (the sphere's `retract_jacobian`, the
# normalization included). The constraint's own second derivative ∂G_x/∂vec(P)
# is closed form: G_x = (u, −v) with u = p₂·X̃, v = p₁·X̃.
function residual_jacobian_rows!(J, lm::ChartLm{<:Any, <:RadialCameraProblem}, a)
    cs = lm.problem.cs; plane = sources(cs); pixel = targets(cs); sqrtw = lm.sqrtw
    av = SVector{5}(a[1], a[2], a[3], a[4], a[5])
    P = retract(lm.ch, av)
    A = retract_jacobian(lm.ch, av)                                  # 6×5
    @inbounds for i in 1:length(cs)
        X = plane[i]
        X̃ = SVector(X[1], X[2], 1.0)
        g, G_x, G_θ = VisualGeometryCore.constraint_with_jacobians(P, X, pixel[i])
        ∂G_x = SMatrix{2, 6, Float64, 12}(
            0.0, -X̃[1], X̃[1], 0.0, 0.0, -X̃[2], X̃[2], 0.0, 0.0, -X̃[3], X̃[3], 0.0
        )                                                              # rows ∂u, −∂v over vec(P)
        row = signed_sampson_jacobian(g, G_x, A' * G_θ, ∂G_x * A, iszero(sum(abs2, G_x)))
        for q in 1:5
            J[i, q] = sqrtw[i] * row[q]
        end
    end
    return J
end

# A radial camera fixes directions only: there is no quad to wind, so the
# extent gate is the inner problem's consensus — nothing to add.
@inline _check_extent(::WithExtent{<:HyperRectangle{2}}, _p, ::VisualGeometryCore.RadialCamera, _mask) = true
