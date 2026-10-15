# =============================================================================
# src/pools/synthetic_pool.jl — THE synthetic E pool pair of the engine tests
# (this package's and the LO engine's): a small calibrated (K = I) scene built
# directly in the pool frame, perturbed candidate essentials as the pool's
# models (stored as FundMat, the E pool layout), their Sampson distances and
# optimistic per-axis pose errors.
# =============================================================================

"""
    synthetic_e_pool_pair(rng = MersenneTwister(7); n = 30, sibling_counts = vcat([2, 2], fill(1, 23)),
                          noise = 0.0, δ_rot = 0.1, t_jitter = 0.05, gt_variance = 1.0e-4)
        -> (pair::PoolPair{FundMat{Float64}}, residuals::Matrix{Float32})

A calibrated two-view pool pair in the pool frame (K = I, identity maps): `n`
correspondences of points 3–5 units ahead under the fixed GT motion
(rotation 0.15 rad about y, unit translation along (0.8, 0.1, 0.3)), target
coordinates jittered by `noise`·N(0, 1) when `noise > 0`. One minimal sample
(5 indices) per entry of `sibling_counts`, carrying that many sibling models:
each the GT rotated by U(−δ_rot/2, δ_rot/2) about z, the translation jittered
by `t_jitter`·N(0, I₃) and renormalized. Every model's optimistic (Δr, Δt),
all-inlier GT and the stated `gt_variance`; the residual matrix holds every
model's Sampson distances.
"""
function synthetic_e_pool_pair(rng::AbstractRNG = MersenneTwister(7); n::Integer = 30,
                               sibling_counts::AbstractVector{<:Integer} = vcat([2, 2], fill(1, 23)),
                               noise::Real = 0.0, δ_rot::Real = 0.1, t_jitter::Real = 0.05,
                               gt_variance::Real = 1.0e-4)
    φ = 0.15
    R = SMatrix{3,3,Float64,9}(RotY(φ))
    t = SVector{3,Float64}(normalize(SA[0.8, 0.1, 0.3]))
    src = Vector{SVector{2,Float64}}(undef, n); dst = Vector{SVector{2,Float64}}(undef, n)
    for i in 1:n
        X = SA[2.0 * (rand(rng) - 0.5), 1.5 * (rand(rng) - 0.5), 3.0 + 2.0 * rand(rng)]
        Y = R * X + t
        jitter = noise > 0 ? noise .* SA[randn(rng), randn(rng)] : SA[0.0, 0.0]
        src[i] = SVector{2,Float64}(X[1] / X[3], X[2] / X[3])
        dst[i] = SVector{2,Float64}(Y[1] / Y[3] + jitter[1], Y[2] / Y[3] + jitter[2])
    end
    M = sum(sibling_counts)
    models = Vector{FundMat{Float64}}(undef, M)
    samples = Matrix{Int}(undef, 5, M)
    er = Vector{Float64}(undef, M); et = Vector{Float64}(undef, M)
    j = 1
    for k in sibling_counts
        idx = rand(rng, 1:n, 5)
        for _ in 1:k
            δ = δ_rot * (rand(rng) - 0.5)
            Rp = R * SMatrix{3,3,Float64,9}(RotZ(δ))
            tp = normalize(t + t_jitter * SA[randn(rng), randn(rng), randn(rng)])
            E = SMatrix{3,3,Float64,9}(EssentialMat(Rp, tp))
            models[j] = FundMat{Float64}(Tuple(E))
            samples[:, j] = idx
            er[j], et[j] = optimistic_pose_error(E, R, t)
            j += 1
        end
    end
    D = Float32[sampson_distance(models[k], src[i], dst[i]) for i in 1:n, k in 1:M]
    K = SMatrix{3,3,Float64,9}(I)
    pair = PoolPair{FundMat{Float64}}(; csponds = csponds(src, dst), K1 = K, K2 = K, R_gt = R, t_gt = t,
                                      models, sample_indices = samples, errors_r = er, errors_t = et,
                                      gt_inliers = trues(n), gt_variance = Float64(gt_variance),
                                      source_map = K, target_map = K)
    return pair, D
end
