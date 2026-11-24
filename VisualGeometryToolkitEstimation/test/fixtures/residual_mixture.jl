# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The residual mixture's test-only scorer and draws (residual_mixture_tests.jl, and the golden records of
# ci/golden/fixtures/est_other_fits.jl). No estimator is called; include-once safe (an identical struct redefinition is
# a no-op).
import VisualGeometryToolkitCore as VgtkCore

# A scorer with a uniform outlier model, test-only: its consensus set is the
# residuals below its threshold (inliers first, each part in index order), its
# per-point scores the squared residuals (what `ResidualMixture` seeds the
# posterior from), and its outlier half-width. Scale prior: the default.
struct HalfwidthScoring{DG} <: VgtkCore.AbstractScoring
    threshold::Float64
    outlier_halfwidth::Float64
end
HalfwidthScoring(dg::Integer; threshold, outlier_halfwidth) =
    HalfwidthScoring{Int(dg)}(Float64(threshold), Float64(outlier_halfwidth))
mutable struct HalfwidthScoringWorkspace
    order::Vector{Int}
    k::Int
end
VgtkCore.codimension(::HalfwidthScoring{DG}) where {DG} = DG
VgtkCore.outlier_halfwidth(s::HalfwidthScoring) = s.outlier_halfwidth
VgtkCore.scoring_workspace(::HalfwidthScoring, n::Int, _d_g::Int) =
    HalfwidthScoringWorkspace(Vector{Int}(undef, n), 0)
VgtkCore.inlier_count(::HalfwidthScoring, W::HalfwidthScoringWorkspace) = W.k
VgtkCore.consensus_order(::HalfwidthScoring, W::HalfwidthScoringWorkspace) = W.order
function VgtkCore.score!(W::HalfwidthScoringWorkspace, s::HalfwidthScoring, scores, r, n;
                         sample = nothing, bound = -Inf)
    sq_threshold = s.threshold^2
    k = 0
    for i in 1:n
        scores[i] = Float64(r[i])^2
        scores[i] < sq_threshold && (k += 1; W.order[k] = i)
    end
    j = k
    for i in 1:n
        scores[i] < sq_threshold || (j += 1; W.order[j] = i)
    end
    W.k = k
    return Float64(k)
end

# Codimension-2 reprojection residual NORMS: Rayleigh inliers at σ, gross
# outliers uniform over the ±halfwidth square (the mixture's outlier support);
# the residuals and their inlier labels.
function reprojection_residuals(rng; n_in, n_out, σ, halfwidth)
    r_in = [σ * hypot(randn(rng), randn(rng)) for _ in 1:n_in]
    r_out = [hypot(halfwidth * (2rand(rng) - 1), halfwidth * (2rand(rng) - 1)) for _ in 1:n_out]
    return vcat(r_in, r_out), vcat(trues(n_in), falses(n_out))
end
