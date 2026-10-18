# =============================================================================
# Synthetic Conic/Ellipse Points
# =============================================================================

"""
    ellipse_points(; cx=50.0, cy=50.0, a=20.0, b=10.0, angle=0.3,
                    n=100, outlier_frac=0.0, sigma=0.0, seed=42)

Generate synthetic points on an ellipse with optional noise and outliers.

Points are uniformly sampled in angle around the ellipse. Outliers are
uniformly distributed within the bounding box of the ellipse (expanded by 2×).

# Returns
- `points::Vector{Point2{Float64}}`: Point set (inliers + outliers, shuffled)
- `is_inlier::BitVector`: Mask indicating true inlier points

# Arguments
- `cx, cy`: Ellipse center
- `a, b`: Semi-major and semi-minor axes
- `angle`: Rotation angle (radians)
- `n::Int=100`: Number of inlier points
- `outlier_frac::Real=0.0`: Fraction of total that are outliers
- `sigma::Real=0.0`: Gaussian noise std added to points
- `seed::Int=42`: RNG seed
"""
function ellipse_points(;
        cx::Real = 50.0, cy::Real = 50.0,
        a::Real = 20.0, b::Real = 10.0,
        angle::Real = 0.3,
        n::Int = 100,
        outlier_frac::Real = 0.0,
        sigma::Real = 0.0,
        seed::Int = 42
    )
    rng = MersenneTwister(seed)
    σ = Float64(sigma)
    cxf, cyf = Float64(cx), Float64(cy)
    af, bf = Float64(a), Float64(b)
    θ = Float64(angle)
    cosθ, sinθ = cos(θ), sin(θ)

    # Uniformly spaced angles
    angles = range(0, 2π, length = n + 1)[1:n]
    pts_in = [
        Point2{Float64}(
            cxf + af * cos(t) * cosθ - bf * sin(t) * sinθ + σ * randn(rng),
            cyf + af * cos(t) * sinθ + bf * sin(t) * cosθ + σ * randn(rng)
        ) for t in angles
    ]

    # Outliers in expanded bounding box (2× the ellipse extent)
    n_total = round(Int, n / (1 - Float64(outlier_frac)))
    n_outliers = n_total - n

    if n_outliers == 0
        return (pts_in, trues(n))
    end

    extent = max(af, bf) * 2.0
    pts_out = [
        Point2{Float64}(
            cxf + extent * (2 * rand(rng) - 1),
            cyf + extent * (2 * rand(rng) - 1)
        ) for _ in 1:n_outliers
    ]

    pts_all = vcat(pts_in, pts_out)
    is_inlier = vcat(trues(n), falses(n_outliers))

    perm = randperm(rng, n_total)
    return (pts_all[perm], is_inlier[perm])
end
