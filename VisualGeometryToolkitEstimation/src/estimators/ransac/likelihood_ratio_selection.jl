# =============================================================================
# LikelihoodRatioSelection — the refine of several starts, the most likely kept
# =============================================================================

"""
    LikelihoodRatioSelection(refine::LocalOptimization)

A [`RefineKind`](@extref VisualGeometryToolkitCore.RefineKind): model selection between refines. Each candidate start
of `solve(RefineProblem(f, (θ₁, θ₂, …), p), LikelihoodRatioSelection(refine))` is refined by the SAME seeded local
optimization, `solve(RefineProblem(f, θᵢ, p), refine)`, so the candidates' scores at their optima are one scorer's
(their difference a log-likelihood ratio for a marginal scorer). The kept candidate is the found model of highest
score (`metadata.quality` of its local optimization), the first on a tie: a later candidate replaces an earlier only
by a strictly higher score. With no found model the first candidate's record is returned, `selected = 0`.

The result is the kept refine's (`u`, termination record, `rms`, `n`, `resid`) with `metadata = (; candidates,
selected)`: every candidate's own result (recorded by the front end, in start order) and the kept one's index. A
caller that rules on the candidates (a branch that refined back into another's basin) reads them there. Every
candidate is a model of the problem's family; a candidate set is a tuple.
"""
struct LikelihoodRatioSelection{A <: LocalOptimization} <: AbstractEstimationAlgorithm
    refine::A
end
estimator_kind(::LikelihoodRatioSelection) = RefineKind()

function __solve(prob::RefineProblem{<:AbstractRansacProblem, <:Tuple}, alg::LikelihoodRatioSelection)
    isempty(prob.u0) && throw(ArgumentError("LikelihoodRatioSelection: no candidate start"))
    rs = map(θ -> solve(RefineProblem(prob.f, θ, prob.p), alg.refine), prob.u0)
    i = 0
    for (k, r) in enumerate(rs)
        found_model(r) || continue
        (i == 0 || r.metadata.quality > rs[i].metadata.quality) && (i = k)
    end
    r = rs[max(i, 1)]
    return FitResult(r.u, r.retcode, r.stop_rule, r.stats, r.rms, r.n, (; candidates = rs, selected = i); resid = r.resid)
end
