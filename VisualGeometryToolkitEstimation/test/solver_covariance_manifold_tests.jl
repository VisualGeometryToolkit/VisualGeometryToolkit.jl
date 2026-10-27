@testitem "solver_covariance_manifold" setup = [EpipolarScenes, HomographyScenes] begin
    import VisualGeometryToolkitEstimation as VgtkEstimation
# =============================================================================
# solver_covariance for E / F / H — the one constraint covariance
#
# For the matrix models it returns the UNIT-SCALE ambient 9×9 shape propagated from
# the covariance in the minimal chart (estimators/ransac/covariance.jl: VgtkCore
# implicit_covariance on the constraint rows). Contract tests here, plus the minimal-E smoke
# through VgtkCore's Monte Carlo calibration harness (`VisualGeometryToolkitCore.Synthetic`).
# =============================================================================

using Test
using LinearAlgebra
using Random
using StableRNGs
using StaticArrays
using VisualGeometryToolkitCore: solver_covariance, provides_solver_covariance   # the constraint-problem contract (VgtkCore)
using VisualGeometryToolkitEstimation: BoundedProblem, RelposeBundleProblem
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: dof, model, estimate, converged, retcode, FixedModels, Uncertain, CovMat, WelschShape,
    transform_covariance, EuclideanManifold
const VgtkCore = VisualGeometryToolkitCore

using VisualGeometryToolkitCore.Synthetic: sample_homography_scene, sample_twoview_scene, projective_distance,
    make_point_covs, perturb_pairs, essential_trial, epipolar_band_trial,
    run_calibration, is_calibrated, run_containment   # VgtkCore's MC calibration harness (public)

# The E problems take the calibrated points in ECAM's pixels (`ecs`,
# EpipolarScenes), so the gate's PIXEL tolerance stays meaningful.

# Provider-contract helper: the delivered hypothesis among the enumerated
# solutions closest to the ground truth (pipeline pattern: providers are
# evaluated at a DELIVERED hypothesis; local to this contract test — the
# calibration harness itself carries no root-selection machinery).
_closest_to(models, truth) = argmin(m -> projective_distance(m, truth), models)

# Ambient Σ̃ from a provider must be symmetric PSD, rank ≤ chart dof, and its
# propagation to the minimal chart must round-trip (Σ lies in the model tangent).
function _check_provider_contract(Σ9, mdl, D)
    @test Σ9 isa SMatrix{9, 9, Float64, 81}
    S = Symmetric(Matrix(Σ9))
    @test norm(Matrix(Σ9) - Matrix(Σ9)') < 1.0e-10 * (1 + norm(Σ9))
    λ = eigvals(S)
    @test λ[1] > -1.0e-9 * max(λ[end], 1.0)                 # PSD
    @test count(>(1.0e-9 * max(λ[end], 1.0)), λ) <= D       # rank ≤ dof
    ch = minimal_manifold(mdl)
    Σv = transform_covariance(EuclideanManifold(ch), ch, Matrix(Σ9))
    @test Matrix(transform_covariance(ch, EuclideanManifold(ch), Σv)) ≈ Matrix(Σ9) rtol = 1.0e-6
    return Σv
end

@testset "solver_covariance — HomographyProblem (chart-domain, 8-dof)" begin
    rng = StableRNG(31)
    H, cs = sample_homography_scene(rng)
    csn = perturb_pairs(rng, cs, 0.5, make_point_covs(rng, 4))
    p = HomographyProblem(csn)
    idx = collect(1:4)
    models = VgtkEstimation.solve(p, idx)
    Ĥ = first(models)
    Σ9 = solver_covariance(p, Ĥ, idx)
    @test Σ9 !== nothing
    Σv = _check_provider_contract(Σ9, Ĥ, 8)
    # matches VgtkCore's implicit covariance in the minimal chart directly
    ch = VgtkCore.minimal_manifold(Ĥ)
    Σv_direct = VgtkCore.implicit_covariance(ch, csn)
    @test Matrix(Σv) ≈ Σv_direct rtol = 1.0e-8
end

@testset "solver_covariance — FundMatProblem (replaces the `nothing` gap)" begin
    rng = StableRNG(32)
    got = 0
    for _ in 1:6
        _, _, F, _, cs_px, _ = sample_twoview_scene(rng, 7)
        csn = perturb_pairs(
            rng, cs_px, 0.5,
            make_point_covs(rng, 7)
        )
        p = FundMatProblem(csn)
        idx = collect(1:7)
        models = VgtkEstimation.solve(p, idx)
        length(models) == 0 && continue
        F̂ = first(models)
        Σ9 = solver_covariance(p, F̂, idx)
        Σ9 === nothing && continue
        _check_provider_contract(Σ9, F̂, 7)
        got += 1
    end
    @test got ≥ 4
end

@testset "solver_covariance — EssentialProblem (5-pt, bearing units)" begin
    rng = StableRNG(33)
    got = 0
    for _ in 1:6
        _, _, _, E, _, cs_nm = sample_twoview_scene(rng, 5)
        csn = perturb_pairs(
            rng, cs_nm, 0.5 / 600.0,
            make_point_covs(rng, 5)
        )
        p = ecs(first.(csn), last.(csn))
        idx = collect(1:5)
        models = VgtkEstimation.solve(p, idx)
        length(models) == 0 && continue
        Ê = _closest_to(collect(models), E)
        Σ9 = solver_covariance(p, Ê, idx)
        Σ9 === nothing && continue
        _check_provider_contract(Σ9, Ê, 5)
        got += 1
    end
    @test got ≥ 4
end

# G·Σ·Gᵀ for a vector constraint's whitened model gradient G (d_g ≥ 2) or the
# scalar one's (a 9-vector for E/F).
_model_term(G::AbstractVector, Σ) = dot(G, Σ * G)
_model_term(G::AbstractMatrix, Σ) = G * Σ * G'

@testset "a covariance follows the scale of its hypothesis" begin
    # H, F and E are homogeneous: the model term of the predictive variance,
    # G_w·Σ̃·G_wᵀ, must not depend on the representative scored. (Until
    # 2026-09-22 the E and F providers returned the covariance of the CHART's
    # representative — ‖E‖ = √2 — whatever scale the solver returned, ‖E‖ = 1:
    # a 2× model term for every production E hypothesis.)
    rng = StableRNG(35)
    _, _, _, _, cpx, cnm = sample_twoview_scene(rng, 9)
    _, csh = sample_homography_scene(rng; npts = 8)
    cases = (
        (ecs(first.(cnm), last.(cnm)), 5),
        (FundMatProblem(cpx), 7),
        (HomographyProblem(csh), 4),
    )
    for (p, k) in cases
        θ = first(VgtkEstimation.solve(p, collect(1:k)))
        Σ = solver_covariance(p, θ, collect(1:k))
        for c in (0.3, -2.0)
            θc = typeof(θ)(Tuple(c .* SMatrix{3, 3, Float64, 9}(θ)))
            Σc = solver_covariance(p, θc, collect(1:k))
            @test Matrix(Σc) ≈ c^2 .* Matrix(Σ) rtol = 1.0e-9
            for i in (k + 1):length(p.cs)
                _, G, _ = VgtkCore.whiten(VgtkEstimation.constraint_with_jacobians(p, θ, i)...)
                _, Gc, _ = VgtkCore.whiten(VgtkEstimation.constraint_with_jacobians(p, θc, i)...)
                @test _model_term(Gc, Σc) ≈ _model_term(G, Σ) rtol = 1.0e-9
            end
        end
    end
end

# (Predictive scoring with the solver covariance — the Student-t + uniform
# marginal's `Predictive` mode on these providers — is LocalOptimizationDoneRight's test.)

@testset "MC calibration smoke — E through the harness (tracked protocol)" begin
    rng = StableRNG(34)
    # Chart-domain full-dof NEES for the tracked 5-point E solution is
    # heavy-tailed on the generic narrow-FOV family (constraint-variety
    # curvature — a property of the minimal PROBLEM, quantified in the
    # secondary table of eval/validation); the OUTPUT-domain epipolar-band
    # containment is near-nominal (the headline statistic). Pin both facts.
    rc = run_calibration(essential_trial(σ_px = 0.5), rng, 600, 5; cond_cap = 1.0e8)
    @test !is_calibrated(rc)
    rb = run_containment(epipolar_band_trial(:E; σ_px = 0.5), rng, 800)
    @test abs(rb.rates[1] - 0.95) < 0.03           # pointwise χ²₁
    @test rb.rates[2] > 0.9                        # whole-line χ²₂
end
end
