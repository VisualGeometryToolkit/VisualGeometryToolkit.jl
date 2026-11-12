# =============================================================================
# A minimal-sample point the model does not reach (its residual Inf: a detection
# a lattice pose leaves unassigned, RadiometricCamera's grid index, 2026-11-12)
# is not in the consensus set, and a result's estimated scale is finite. Before,
# the threshold scorers forced every sample point into the set, its Inf residual
# made the set's scale Inf, and the result recorded it as ScaleEstimated.
# =============================================================================

@testitem "ransac: a spurious minimal-sample point (Inf residual) is not in the consensus set; an estimated scale is finite" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: FixedModels, MsacScoring, RansacScoring, estimate, inlier_mask, noise_scale, scale_source,
        ScaleEstimated, ScaleNone, metadata
    using VisualGeometryToolkitEstimation: AbstractRansacProblem, ransac, RansacConfig

    # A 1-D location, its model a sample's mean over the points it reaches; point 1 is reached by no model (Inf).
    struct _SpuriousPointProblem <: AbstractRansacProblem{2}
        x::Vector{Float64}
        sampler::VgtkEstimation.ExhaustiveSampler
    end
    _SpuriousPointProblem(x) = _SpuriousPointProblem(x, VgtkEstimation.ExhaustiveSampler(length(x), 2))
    VgtkCore.data_size(p::_SpuriousPointProblem) = length(p.x)
    VgtkCore.codimension(::Type{_SpuriousPointProblem}) = 1
    VgtkCore.model_type(::Type{_SpuriousPointProblem}) = Float64
    VgtkEstimation.sampler(p::_SpuriousPointProblem) = p.sampler
    function VgtkCore.solve(p::_SpuriousPointProblem, idx::AbstractVector{Int})
        reached = [i for i in idx if i != 1]
        return FixedModels{1, Float64}(1, (sum(p.x[reached]) / length(reached),))
    end
    VgtkCore.whitened_residual(p::_SpuriousPointProblem, m::Float64, i::Int) = i == 1 ? Inf : p.x[i] - m

    # point 2 sits at the truth (0): the sample (1, 2) gives the exact model, and a forced-in point 1 was a full
    # quality point of bonus, so that sample won; every pair is tried (the exhaustive schedule)
    x = [0.0, 0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5]
    p = _SpuriousPointProblem(x)
    # (RANSAC's count ties every model within the threshold, so its winner is the first such sample: only the set is checked)
    for sc in (MsacScoring(p; threshold = 1.0), RansacScoring(p; threshold = 1.0))
        r = ransac(p, sc; config = RansacConfig(max_trials = 100, min_trials = 1))
        m = inlier_mask(r)
        @test !m[1] && all(m[2:end])                                   # the 11 reached points, not the spurious one
        @test r.dof == 11 - 2                                          # ν = (n_in − m)·d_g over the set's own points
        @test scale_source(r) === ScaleEstimated && isfinite(noise_scale(r)) && noise_scale(r) < 1.0
        @test !metadata(r).nonfinite_scale
        if sc isa MsacScoring
            @test estimate(r) == 0.0 && noise_scale(r) ≈ sqrt(sum(abs2, x[2:end]) / 9)
        end
    end

    # the invariant at the result: an estimated scale is finite; a non-finite one is dropped to ScaleNone (NaN) and recorded
    @test isequal(VgtkEstimation._estimated_scale(9, 0.25), (0.5, ScaleEstimated, false))
    @test isequal(VgtkEstimation._estimated_scale(9, Inf), (NaN, ScaleNone, true))
    @test isequal(VgtkEstimation._estimated_scale(9, NaN), (NaN, ScaleNone, true))
    @test isequal(VgtkEstimation._estimated_scale(0, 1.0), (NaN, ScaleNone, false))
    # a metadata built without the field (every earlier constructor) records no drop
    md = VgtkEstimation.RansacMetadata{Float64, typeof(MsacScoring(p; threshold = 1.0))}(zeros(2), ones(2), 1.0, 0.5, ScaleEstimated, 2, 1.0, VgtkCore.ReturnCode.Default, VgtkCore.NoRule)
    @test !md.nonfinite_scale
end
