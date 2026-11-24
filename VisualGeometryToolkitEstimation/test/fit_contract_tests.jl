# The robust-estimation contract on the reweighted fits (B2): ReweightedLeastSquares on a linear model, a rotation, a
# rigid alignment and a robot-world calibration, and RotationAveraging. One scale input (exactly one of `scale = σ`,
# `scale = <upstream result>`, `width = w`), the width `efficiency_constant ×
# scale` at the fit's codimension, the scale record, the weights and their
# kind, and the refusals.

@testsetup module FitContractScenes
    using Random, LinearAlgebra, StaticArrays
    using Rotations: RotMatrix, RotXYZ, rotation_angle
    export contract_linear, contract_rotation, contract_averaging, contract_points, contract_handeye, CONTRACT_FITS
    export Xoshiro, normalize, norm, SVector, SMatrix, RotMatrix, RotXYZ, rotation_angle,
        Correspondences, EuclideanMap, rotation, translation, VgtkCore, VgtkEstimation
    using VisualGeometryToolkitCore: Correspondences, EuclideanMap, rotation, translation, EstimationProblem, solve,
        AbsoluteRotations
    using VisualGeometryToolkitEstimation: ReweightedLeastSquares, RotationAveraging, LinearModel, RobotWorldHandEyeCalibration
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation

    # the scenes, each drawn from the caller's RNG: the suite's fixture file, shared with the golden records
    include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "contract_scenes.jl"))
    contract_linear(; seed = 1) = contract_linear(Xoshiro(seed))
    contract_rotation(; seed = 2) = contract_rotation(Xoshiro(seed))
    contract_points(; seed = 3, n = 60) = contract_points(Xoshiro(seed); n)
    # robot-world hand-eye: N frames × M markers, a little pose noise (1e-4 rad / 1e-4); its robust fit whitens by the
    # given σ_rot = σ_trans = σ (1e-2), so scale and width are in whitened units and the clean residuals sit well inside
    # the widths the contract items use
    contract_handeye(; seed = 4, N = 12, M = 2, noise = 1.0e-4, σ = 1.0e-2) = (contract_handeye(Xoshiro(seed); N, M, noise), σ)
    # fit(keywords) for each of the five fits, on clean data (the hand-eye at its whitening σ): the algorithm's fields
    # (`optimization_strategy`) to the algorithm, the rest (the scale input, a stray keyword) to the solve
    function _solve_fit(prob, alg, kw)
        a = NamedTuple(k => v for (k, v) in pairs(kw) if k === :optimization_strategy)
        r = NamedTuple(k => v for (k, v) in pairs(kw) if k !== :optimization_strategy)
        return solve(prob, alg(; a...); r...)
    end
    const CONTRACT_FITS = (
        (:linear, 1, kw -> _solve_fit(EstimationProblem(LinearModel, contract_linear()), ReweightedLeastSquares, kw)),
        (:rotation, 2, kw -> _solve_fit(EstimationProblem(RotMatrix{3}, contract_rotation()), ReweightedLeastSquares, kw)),
        (:rotation_averaging, 3, kw -> _solve_fit(EstimationProblem(AbsoluteRotations, contract_averaging()), RotationAveraging, kw)),
        (:rigid_alignment, 3, kw -> _solve_fit(EstimationProblem(EuclideanMap{3}, first(contract_points())), ReweightedLeastSquares, kw)),
        (:robot_world_hand_eye, 6, kw -> let (markers, σ) = contract_handeye()
            _solve_fit(EstimationProblem(RobotWorldHandEyeCalibration, markers, (; σ_rot = σ, σ_trans = σ)), ReweightedLeastSquares, kw)
        end),
    )
end

@testitem "fit contract: the scale input, refusals" setup = [FitContractScenes] begin
    using Test
    using VisualGeometryToolkitCore: Irls, Triggs
    using VisualGeometryToolkitCore: HuberShape, WelschShape, GaussianShape, TLSShape
    for (name, _, fit) in CONTRACT_FITS
        @testset "$name" begin
            huber = (; optimization_strategy = Irls{HuberShape}())
            @test_throws ArgumentError fit(huber)                                        # neither scale nor width
            @test_throws ArgumentError fit((; huber..., scale = 0.01, width = 0.02))      # both
            @test_throws ArgumentError fit((; huber..., scale = 0.0))                     # not positive
            @test_throws ArgumentError fit((; huber..., scale = -1.0))
            @test_throws ArgumentError fit((; huber..., width = Inf))
            @test_throws ArgumentError fit((; optimization_strategy = nothing, scale = 0.01))   # L2 ignores a scale: refused
            @test_throws ArgumentError fit((; optimization_strategy = nothing, width = 0.01))
            @test_throws ArgumentError fit((; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0), scale = 0.01))
            @test_throws ArgumentError fit((; optimization_strategy = Irls{GaussianShape}(), scale = 0.01))
            @test_throws ArgumentError fit((; optimization_strategy = Irls{TLSShape}(), scale = 0.01))  # TLS: no efficiency constant
            @test_throws ArgumentError fit((; optimization_strategy = Irls{HuberShape}(loss_width = 0.1), scale = 0.01))
            # an upstream result that recorded no scale is refused (its NaN is never handed on)
            l2 = fit((; optimization_strategy = nothing))
            @test_throws ArgumentError fit((; huber..., scale = l2))
            wonly = fit((; huber..., width = 0.05))
            @test_throws ArgumentError fit((; huber..., scale = wonly))
            # the old keywords are gone: no keyword of the solve
            @test_throws MethodError fit((; irls = Irls{HuberShape}(), scale = 0.01))
            @test_throws MethodError fit((; local_optimization = Irls{HuberShape}(), scale = 0.01))
            @test_throws MethodError fit((; efficiency_constant = 2.0, scale = 0.01))
            @test_throws MethodError fit((; loss = HuberShape(), scale = 0.01))
        end
    end
end

@testitem "fit contract: the scale record, the width rule, the weights and their kind" setup = [FitContractScenes] begin
    using Test
    using VisualGeometryToolkitEstimation: FitMetadata, RotationAveragingMetadata
    using VisualGeometryToolkitCore: Irls
    # each metadata through its outer constructor (the strategy type given, the weights' type inferred)
    _remade(m::FitMetadata{L}) where {L} = FitMetadata{L}(m.weights, m.residuals, m.scale, m.scale_source, m.loss_width)
    _remade(m::RotationAveragingMetadata{K, L}) where {K, L} =
        RotationAveragingMetadata{K, L}(m.edges, m.weights, m.residuals, m.scale, m.scale_source, m.loss_width, m.graph)
    using VisualGeometryToolkitCore: HuberShape, WelschShape, TLSShape, noise_scale, scale_source, weights, weight_kind,
        inlier_mask, ScaleGiven, ScaleUpstream, ScaleNone, BinaryWeights, IrlsWeights, efficiency_constant
    for (name, d, fit) in CONTRACT_FITS
        @testset "$name" begin
            r = fit((; optimization_strategy = Irls{WelschShape}(), scale = 0.02))
            # given: the scale recorded unchanged, the width c(K, d) × scale at the fit's codimension
            @test noise_scale(r) === 0.02 && scale_source(r) === ScaleGiven
            @test r.metadata.loss_width ≈ efficiency_constant(WelschShape, d) * 0.02
            @test weight_kind(r) === IrlsWeights()
            @test weights(r) isa Vector{Float64} && all(w -> 0 <= w <= 1, weights(r))
            @test_throws ArgumentError inlier_mask(r)                          # an IRLS weight has no cut of its own
            @test inlier_mask(r, 0.5) == (weights(r) .> 0.5)
            # the hand-off: an upstream result's scale, recorded as such
            u = fit((; optimization_strategy = Irls{HuberShape}(), scale = r))
            @test noise_scale(u) === 0.02 && scale_source(u) === ScaleUpstream
            @test u.metadata.loss_width ≈ efficiency_constant(HuberShape, d) * 0.02
            # the strategy's own constant is the plain multiple
            c = fit((; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = 0.02))
            @test c.metadata.loss_width ≈ 0.06
            # width only: ScaleNone, NaN scale, the width recorded
            wonly = fit((; optimization_strategy = Irls{HuberShape}(), width = 0.05))
            @test scale_source(wonly) === ScaleNone && wonly.metadata.loss_width === 0.05
            @test_throws ArgumentError noise_scale(wonly)
            # least squares: lazy unit weights, binary, no scale
            l2 = fit((; optimization_strategy = nothing))
            @test weight_kind(l2) === BinaryWeights()
            @test weights(l2) == VgtkCore.unit_weights(length(weights(l2))) && !(weights(l2) isa Vector)
            @test all(inlier_mask(l2))
            @test scale_source(l2) === ScaleNone && isnan(l2.metadata.loss_width)
            @test_throws ArgumentError noise_scale(l2)
            # type-stable accessors
            for x in (r, u, wonly, l2)
                @test (@inferred scale_source(x)) isa VgtkCore.ScaleSource
                @inferred weights(x)
                @inferred weight_kind(x)
            end
            @test (@inferred noise_scale(r)) === 0.02
            # the metadata constructors are inferable (the robust record and the least-squares one)
            # n and rms are the stack's one weighted support/rms rule (unit priors here: the solve weights are the weights)
            if name !== :robot_world_hand_eye
                sq = name === :rotation_averaging ? [abs2(r.metadata.residuals[e]) for e in r.metadata.edges] : abs2.(r.metadata.residuals)
                @test (r.n, r.rms) === VgtkCore.weighted_support_rms(weights(r), sq)
            end
            if name !== :robot_world_hand_eye                         # its metadata is _robot_world_hand_eye_result's
                for x in (r, l2)
                    m = x.metadata
                    @test (@inferred _remade(m)) isa typeof(m)
                end
            end
        end
    end
end

@testitem "rigid alignment: 5 gross outliers move the L2 alignment, not the robust one" setup = [FitContractScenes] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using VisualGeometryToolkitCore: Irls, ClassDefault
    using VisualGeometryToolkitCore: HuberShape, WelschShape, weights, withattrs
    using VisualGeometryToolkitCore: EstimationProblem, solve
    using VisualGeometryToolkitEstimation: ReweightedLeastSquares, RobotWorldHandEyeCalibration
    include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))
    # a walk of 150 camera centres (5 mm noise per axis, metres) and five
    # centres a registration placed 0.9–2.4 m wrong (icra2026 failure B)
    rng = Xoshiro(44)
    R = RotMatrix(RotXYZ(0.05, -0.1, 1.2)); t = SVector(1.0, -2.0, 0.3)
    s = range(0, 6π; length = 150)
    walk = [SVector(2cos(u), 1.5sin(u), 0.3sin(3u)) for u in s]
    σ = 0.005
    gt = [R * a + t + σ .* randn(rng, SVector{3, Float64}) for a in walk]
    bad = [44, 52, 57, 60, 63]
    for (i, d) in zip(bad, (0.9, 1.6, 2.2, 2.2, 2.4))
        gt[i] += d .* normalize(randn(rng, SVector{3, Float64}))
    end
    cs = Correspondences(walk, gt)
    err(fit) = (rotation_angle(RotMatrix{3, Float64}(rotation(fit.u) * R')), norm(translation(fit.u) - t))
    l2 = rigid_fit(cs)
    @test retcode_symbol(l2) === :solved
    clean = rigid_fit(Correspondences(walk[setdiff(1:150, bad)], gt[setdiff(1:150, bad)]))
    e_clean = err(clean)
    e_l2 = err(l2)
    @test e_l2[1] > 10 * max(e_clean[1], 1.0e-4) || e_l2[2] > 10 * max(e_clean[2], 1.0e-3)   # the L2 alignment moves
    for K in (HuberShape, WelschShape)
        rob = rigid_fit(cs; optimization_strategy = Irls{K}(), scale = σ)
        @test retcode_symbol(rob) === :converged
        e = err(rob)
        @test e[1] < 2.0e-3 && e[2] < 2.0e-3                                           # beyond tolerance: no
        @test maximum(weights(rob)[bad]) < 0.1
        @test rob.n == (K === WelschShape ? count(>(0), weights(rob)) : 150)
    end
    # prior weights drop pairs; fewer than three, or collinear, is refused
    @test_throws ArgumentError rigid_fit(Correspondences(walk[1:2], gt[1:2]))
    line = [SVector(Float64(k), 0.0, 0.0) for k in 1:5]
    @test_throws ArgumentError rigid_fit(Correspondences(line, line))
    p = ones(150); p[bad] .= 0
    @test err(rigid_fit(withattrs(cs; weights = p)))[1] ≈ e_clean[1] atol = 1.0e-12
end

@testitem "robot-world fit: a robust fit through gross observations" setup = [FitContractScenes, PoseScenes] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StableRNGs
    using VisualGeometryToolkitCore: Irls, Triggs, ClassDefault
    using VisualGeometryToolkitCore: HuberShape, WelschShape, log_so3, weights, weight_kind, noise_scale, scale_source,
        IrlsWeights, BinaryWeights, ScaleGiven, ScaleNone
    using VisualGeometryToolkitCore: EstimationProblem, solve
    using VisualGeometryToolkitEstimation: ReweightedLeastSquares, RobotWorldHandEyeCalibration
    include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))
    rng = StableRNG(21)
    N, M = 30, 2
    (; Y, X, A) = robot_world_hand_eye_scene(rng; N, M, Y = (t_scale = 0.5,), X = (ang_max = 0.5, t_scale = 0.3), A = (t_scale = 1.0,))
    σ_R, σ_t = 2.0e-3, 2.0e-3
    markers = Dict(m => Pair{typeof(Y), typeof(Y)}[] for m in 1:M)
    for i in 1:N, m in 1:M
        push!(markers[m], A[i] => perturb_pose(rng, marker_pose(Y, A[i], X[m]); σ_R, σ_t))
    end
    # five gross observations of marker 1: a pose placed tens of degrees and a metre wrong
    for i in (4, 9, 15, 22, 27)
        a, b = markers[1][i]
        markers[1][i] = a => perturb_pose(rng, b; σ_R = 0.8, σ_t = 1.0)
    end
    errY(fit) = (norm(log_so3(rotation(fit.u.Y)' * rotation(Y))), norm(translation(fit.u.Y) - translation(Y)))
    l2 = rwhe_fit(markers)
    @test weight_kind(l2) === BinaryWeights() && scale_source(l2) === ScaleNone
    # Huber (monotone: never zeroes a weight) from the poisoned least-squares seed. A redescending loss at the
    # same fixed width may reject a whole marker at that seed: then an ArgumentError naming the monotone loss
    try
        rwhe_fit(markers; σ_rot = σ_R, σ_trans = σ_t, optimization_strategy = Irls{WelschShape}(), scale = 1.0)
    catch err
        @test err isa ArgumentError && occursin("HuberShape", err.msg)
    end
    for K in (HuberShape,)
        rob = rwhe_fit(markers; σ_rot = σ_R, σ_trans = σ_t, optimization_strategy = Irls{K}(), scale = 1.0)
        @test retcode_symbol(rob) === :converged
        @test weight_kind(rob) === IrlsWeights() && noise_scale(rob) === 1.0 && scale_source(rob) === ScaleGiven
        @test length(weights(rob)) == N * M
        @test maximum(weights(rob)[[4, 9, 15, 22, 27]]) < 0.2                   # marker 1's block comes first
        eR, et = errY(rob)
        lR, lt = errY(l2)
        @test eR < 5.0e-3 && et < 5.0e-3
        @test eR < lR / 5 || et < lt / 5
        @test all(isfinite, rob.metadata.cov) && rob.metadata.min_eigval > 0
    end
    # the robust fit needs its whitening; a Triggs is a refine's
    @test_throws ArgumentError rwhe_fit(markers; optimization_strategy = Irls{HuberShape}(), scale = 1.0)
    @test_throws ArgumentError rwhe_fit(markers; σ_rot = σ_R, σ_trans = σ_t,
                                                       optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0), scale = 1.0)
    @test_throws ArgumentError rwhe_fit(markers; scale = 1.0)
    @test_throws MethodError rwhe_fit(markers; local_optimization = Irls{HuberShape}())   # no keyword of the solve
end

@testitem "fit contract: the default strategy is the TOML table's (shape and max_iter)" setup = [FitContractScenes] begin
    using Test
    using VisualGeometryToolkitEstimation: optimization_strategy_from_toml, _dflt
    using VisualGeometryToolkitCore: Irls
    using VisualGeometryToolkitCore: HuberShape, weights
    for (name, _, fit) in CONTRACT_FITS
        name in (:rigid_alignment, :robot_world_hand_eye) && continue   # their default is least squares (no table strategy)
        section = name === :rotation_averaging ? "rotation_averaging" : "reweighted_least_squares.$name"
        lo = optimization_strategy_from_toml(_dflt(section, "optimization_strategy"); class = Symbol(section))
        @test lo isa Irls{HuberShape}                                   # Huber where accuracy matters (owner's ruling)
        @test lo.max_iter == (name === :rotation_averaging ? 100 : 50)  # the table's, not a bare Irls's 25
        a = fit((; scale = 0.02)); b = fit((; optimization_strategy = lo, scale = 0.02))
        @test a.u == b.u && a.stats.nsteps == b.stats.nsteps && weights(a) == weights(b)
    end
end
