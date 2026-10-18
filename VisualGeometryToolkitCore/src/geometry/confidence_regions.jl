# =============================================================================
# Confidence regions in the image domain — the reader-facing form of the
# propagated model covariances (uncertainty thread, supplement figures).
#
# Two constructions, both PURE GEOMETRY (Makie-independent; the plotting layer
# consumes them through convert_arguments in src/plotting/convert.jl):
#
# • POINT-TRANSFER ELLIPSE (homography): x' = π(H·x̃) carries the 2×2
#   covariance J_H·Σ_θ·J_Hᵀ; `confidence_ellipse` turns any 2×2 covariance
#   into the existing `Ellipse` primitive at a χ²₂ level set.
#
# • EPIPOLAR-LINE BAND (F / E): the line l = F·x̃ carries Σ_l = A·Σ_θ·Aᵀ,
#   A = ∂l/∂vec(F) = (x̃ᵀ ⊗ I₃). The level-k family of lines sweeps the region
#
#       B_k = { x : (x̃ᵀ l)² ≤ k² · x̃ᵀ Σ_l x̃ }
#
#   whose boundary is the point conic C = l·lᵀ − k²·Σ_l (x̃ᵀCx̃ = 0) — the
#   Hartley–Zisserman envelope-of-lines construction (MVG 2nd ed. §5.2.6,
#   Result 5.7: "the conic C = l̄l̄ᵀ − k²Σ_l is the envelope of the lines at
#   Mahalanobis distance k from l̄"), generically a hyperbola hugging the
#   nominal line. Level conventions (verified numerically in
#   confidence_regions_tests.jl):
#     k² = χ²₂(α): the band swept by the α-mass of the LINE family (2 line
#         dof) — a random line lies ENTIRELY inside B_k with probability ≥ α;
#         this is the convention the supplement figures use;
#     k² = χ²₁(α): the pointwise band — at each FIXED image point x on the
#         nominal line's normal section, the fraction of sampled lines
#         going by on either side within the band is α.
#   Both quadratic forms are invariant to the projective scale of (l, Σ_l)
#   as long as they come from the SAME representative of F — which they do
#   (Σ_l is pushed from Σ_θ at that representative).
# =============================================================================

using Distributions: Chisq
import Distributions: quantile as _dquantile

"""
    confidence_ellipse(μ, Σ; level=0.95) -> Ellipse

The `level` confidence ellipse of a 2D Gaussian N(`μ`, `Σ`):
`{x : (x−μ)ᵀ Σ⁻¹ (x−μ) = χ²₂(level)}`, as the standard `Ellipse` primitive
(so it plots through the existing `poly!`/`convert_arguments` layer).

It is `Ellipse(μ, Symmetric(χ²₂(level)·Σ))`, so a `Σ` that is not positive
definite (singular, indefinite, or NaN) is refused with that constructor's
`ArgumentError`, never clamped into a degenerate ellipse.
"""
function confidence_ellipse(μ, Σ::AbstractMatrix; level::Real = 0.95)
    k2 = _dquantile(Chisq(2), level)
    return Ellipse(Point2{Float64}(μ[1], μ[2]), Symmetric(k2 * SMatrix{2, 2, Float64, 4}(Σ)))
end

"""
    LineConfidenceBand

Confidence region of an uncertain homogeneous 2D line: nominal line `l`,
line covariance `Σl` (3×3, from pushing a model covariance through the
line construction), and squared Mahalanobis level `k2`. Region:
`{x : (x̃ᵀl)² ≤ k2·x̃ᵀΣl x̃}`; boundary conic `C = l·lᵀ − k2·Σl`.
Plot with `lines!(ax, band, rect)` (Makie layer) or sample the boundary with
[`band_boundary`](@ref).
"""
struct LineConfidenceBand
    l::SVector{3, Float64}
    Σl::SMatrix{3, 3, Float64, 9}
    k2::Float64
end

"Boundary point conic C = l·lᵀ − k²·Σ_l of the band."
band_conic(b::LineConfidenceBand) =
    SMatrix{3, 3, Float64, 9}(b.l * b.l' - b.k2 * b.Σl)

"Signed band membership: `x̃ᵀCx̃ ≤ 0` ⇔ inside."
function in_band(b::LineConfidenceBand, x)
    x̃ = SVector{3, Float64}(x[1], x[2], 1.0)
    return dot(x̃, band_conic(b) * x̃) <= 0
end

"""
    epipolar_band(F::AbstractFundMat, Σθ::AbstractMatrix(9×9), u₁;
                  level=0.95, dof=2) -> LineConfidenceBand

Confidence band of the epipolar line `l = F·ũ₁` in image 2 under the ambient
model covariance `Σθ` (vec(F) column-major). `dof=2` gives the whole-line
(H–Z envelope) convention, `dof=1` the pointwise convention — see the file
header. Uncertainty of `u₁` itself is not included (model-only band; fold a
4×4 point covariance into Σ_l externally if needed).
"""
function epipolar_band(
        F::AbstractFundMat, Σθ::AbstractMatrix, u₁;
        level::Real = 0.95, dof::Int = 2
    )
    s̃ = SVector{3, Float64}(u₁[1], u₁[2], 1.0)
    l = SMatrix{3, 3, Float64, 9}(F) * s̃
    A = kron(transpose(s̃), SMatrix{3, 3, Float64, 9}(I))         # 3×9, l = A·vec(F)
    Σl = SMatrix{3, 3, Float64, 9}(Symmetric(A * Matrix(Σθ) * A'))
    return LineConfidenceBand(l, Σl, _dquantile(Chisq(dof), level))
end

"""
    band_boundary(b::LineConfidenceBand, rect::HyperRectangle{2}; n=201)
        -> (upper::Vector{Point2{Float64}}, lower::Vector{Point2{Float64}})

Sample the two boundary branches of the band inside `rect`, as offsets along
the nominal line's normal over `n` stations of the clipped nominal segment.
Stations where the band is UNBOUNDED (the near-degeneracy blow-up: the
normal-section quadratic loses its positive curvature or its real roots) emit
`(NaN, NaN)` so polylines break there instead of lying. Returns empty vectors
when the nominal line misses `rect`.
"""
function band_boundary(b::LineConfidenceBand, rect; n::Int = 201)
    upper = Point2{Float64}[]; lower = Point2{Float64}[]
    seg = clip_to_rect(Line2D{Float64}(b.l), rect)
    seg === nothing && return upper, lower
    p1 = SVector{2, Float64}(seg.points[1]); p2 = SVector{2, Float64}(seg.points[2])
    d = p2 - p1
    L = norm(d)
    L < eps() && return upper, lower
    n̂ = SVector{2, Float64}(-d[2], d[1]) / L
    ñ = SVector{3, Float64}(n̂[1], n̂[2], 0.0)
    C = band_conic(b)
    nan = Point2{Float64}(NaN, NaN)
    for t in range(0.0, 1.0; length = n)
        p = p1 + t * d
        p̃ = SVector{3, Float64}(p[1], p[2], 1.0)
        a = dot(ñ, C * ñ)
        bq = 2 * dot(p̃, C * ñ)
        c = dot(p̃, C * p̃)
        disc = bq^2 - 4a * c
        if a <= 0 || disc < 0
            push!(upper, nan); push!(lower, nan)   # unbounded section
            continue
        end
        r = sqrt(disc)
        s1 = (-bq - r) / (2a); s2 = (-bq + r) / (2a)
        push!(lower, Point2{Float64}(p + s1 * n̂))
        push!(upper, Point2{Float64}(p + s2 * n̂))
    end
    return upper, lower
end
