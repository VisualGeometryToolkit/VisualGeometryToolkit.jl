# The robust-estimation contract on the reweighted fits (B2): fit_linear,
# fit_rotation, fit_rotation_averaging, fit_rigid_alignment and
# fit_robot_world_hand_eye. One scale input (exactly one of `scale = σ`,
# `scale = <upstream result>`, `width = w`), the width `efficiency_constant ×
# scale` at the fit's codimension, the scale record, the weights and their
# kind, and the refusals.

@testsetup module FitContractScenes
    using Random, LinearAlgebra, StaticArrays
    using Rotations: RotMatrix, RotXYZ, rotation_angle
    export contract_linear, contract_rotation, contract_averaging, contract_points, contract_handeye, CONTRACT_FITS
    export Xoshiro, normalize, norm, SVector, SMatrix, RotMatrix, RotXYZ, rotation_angle,
        Correspondences, EuclideanMap, rotation, translation, estimate, VgtkCore, VgtkEstimation
    using VisualGeometryToolkitCore: Correspondences, EuclideanMap, rotation, translation, estimate
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation

    function contract_linear(; seed = 1)
        rng = Xoshiro(seed)
        A = randn(rng, 60, 3); x0 = randn(rng, 3)
        return A, A * x0 .+ 1.0e-3 .* randn(rng, 60)
    end
    function contract_rotation(; seed = 2)
        rng = Xoshiro(seed)
        R = RotMatrix(RotXYZ(0.2, -0.4, 0.9))
        as = [normalize(randn(rng, SVector{3, Float64})) for _ in 1:40]
        return Correspondences(as, [R * a for a in as])
    end
    function contract_averaging()
        R = Dict(k => RotMatrix(RotXYZ(0.3k, -0.2k, 0.1k^2)) for k in 1:6)
        return Dict((i, j) => SMatrix{3, 3, Float64, 9}(R[j]' * R[i]) for i in 1:6 for j in (i + 1):6)
    end
    function contract_points(; seed = 3, n = 60)
        rng = Xoshiro(seed)
        R = RotMatrix(RotXYZ(0.1, 0.5, -0.3)); t = SVector(0.4, -1.0, 2.0)
        as = [SVector{3, Float64}(4 .* randn(rng, 3)) for _ in 1:n]
        return Correspondences(as, [R * a + t for a in as]), R, t
    end
    # robot-world hand-eye: N frames × M markers, a little pose noise (1e-4 rad / 1e-4); its robust fit whitens by the
    # given σ_rot = σ_trans = σ (1e-2), so scale and width are in whitened units and the clean residuals sit well inside
    # the widths the contract items use
    function contract_handeye(; seed = 4, N = 12, M = 2, noise = 1.0e-4, σ = 1.0e-2)
        rng = Xoshiro(seed)
        rot(s) = SMatrix{3, 3, Float64, 9}(RotMatrix(RotXYZ((s .* randn(rng, 3))...)))
        E(R, t) = EuclideanMap(R, SVector{3, Float64}(t))
        Y = E(rot(1.0), 0.5 .* randn(rng, 3)); X = [E(rot(0.5), 0.3 .* randn(rng, 3)) for _ in 1:M]
        A = [E(rot(1.0), randn(rng, 3)) for _ in 1:N]
        markers = Dict(m => [let B = inv(Y) ∘ A[i] ∘ X[m]
                                 A[i] => E(SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(noise .* randn(rng, SVector{3, Float64}))) * rotation(B),
                                           translation(B) + noise .* randn(rng, SVector{3, Float64}))
                             end for i in 1:N] for m in 1:M)
        return markers, σ
    end
    # fit(scale/width keywords...) for each of the five fits, on clean data (the hand-eye at its whitening σ)
    const CONTRACT_FITS = (
        (:fit_linear, 1, kw -> VgtkEstimation.fit_linear(contract_linear()...; kw...)),
        (:fit_rotation, 2, kw -> VgtkEstimation.fit_rotation(contract_rotation(); kw...)),
        (:fit_rotation_averaging, 3, kw -> VgtkEstimation.fit_rotation_averaging(contract_averaging(); kw...)),
        (:fit_rigid_alignment, 3, kw -> VgtkEstimation.fit_rigid_alignment(first(contract_points()); kw...)),
        (:fit_robot_world_hand_eye, 6, kw -> let (markers, σ) = contract_handeye()
            VgtkEstimation.fit_robot_world_hand_eye(markers; σ_rot = σ, σ_trans = σ, kw...)
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
            # the old keywords are gone
            @test_throws MethodError fit((; irls = Irls{HuberShape}(), scale = 0.01))
            e = try fit((; local_optimization = Irls{HuberShape}(), scale = 0.01)); nothing; catch err; err; end
            @test e isa ArgumentError && occursin("`local_optimization` is retired", e.msg) && occursin("optimization_strategy", e.msg)
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
        RotationAveragingMetadata{K, L}(m.edges, m.weights, m.residuals, m.scale, m.scale_source, m.loss_width)
    using VisualGeometryToolkitCore: HuberShape, WelschShape, TLSShape, noise_scale, scale_source, weights, weight_kind,
        inlier_mask, ScaleGiven, ScaleUpstream, ScaleNone, BinaryWeights, IrlsWeights, efficiency_constant
    for (name, d, fit) in CONTRACT_FITS
        @testset "$name" begin
            r = fit((; optimization_strategy = Irls{WelschShape}(), scale = 0.02))
            # given: the scale recorded unchanged, the width c(K, d) × scale at the fit's codimension
            @test noise_scale(r) === 0.02 && scale_source(r) === ScaleGiven
            @test r.loss_width ≈ efficiency_constant(WelschShape, d) * 0.02
            @test weight_kind(r) === IrlsWeights()
            @test weights(r) isa Vector{Float64} && all(w -> 0 <= w <= 1, weights(r))
            @test_throws ArgumentError inlier_mask(r)                          # an IRLS weight has no cut of its own
            @test inlier_mask(r, 0.5) == (weights(r) .> 0.5)
            # the hand-off: an upstream result's scale, recorded as such
            u = fit((; optimization_strategy = Irls{HuberShape}(), scale = r))
            @test noise_scale(u) === 0.02 && scale_source(u) === ScaleUpstream
            @test u.loss_width ≈ efficiency_constant(HuberShape, d) * 0.02
            # the strategy's own constant is the plain multiple
            c = fit((; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = 0.02))
            @test c.loss_width ≈ 0.06
            # width only: ScaleNone, NaN scale, the width recorded
            wonly = fit((; optimization_strategy = Irls{HuberShape}(), width = 0.05))
            @test scale_source(wonly) === ScaleNone && isnan(wonly.scale) && wonly.loss_width === 0.05
            @test_throws ArgumentError noise_scale(wonly)
            # least squares: lazy unit weights, binary, no scale
            l2 = fit((; optimization_strategy = nothing))
            @test weight_kind(l2) === BinaryWeights()
            @test weights(l2) == VgtkCore.unit_weights(length(weights(l2))) && !(weights(l2) isa Vector)
            @test all(inlier_mask(l2))
            @test scale_source(l2) === ScaleNone && isnan(l2.loss_width)
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
            if name !== :fit_robot_world_hand_eye
                sq = name === :fit_rotation_averaging ? [abs2(r.residuals[e]) for e in r.edges] : abs2.(r.residuals)
                @test (r.n, r.rms) === VgtkEstimation._weighted_support_rms(weights(r), sq)
            end
            if name !== :fit_robot_world_hand_eye                     # its metadata is _robot_world_hand_eye_result's
                for x in (r, l2)
                    m = x.metadata
                    @test (@inferred _remade(m)) isa typeof(m)
                end
            end
        end
    end
end

@testitem "fit_rigid_alignment: 5 gross outliers move the L2 alignment, not the robust one" setup = [FitContractScenes] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using VisualGeometryToolkitEstimation: fit_rigid_alignment
    using VisualGeometryToolkitCore: Irls
    using VisualGeometryToolkitCore: HuberShape, WelschShape, weights, withattrs, retcode
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
    err(fit) = (rotation_angle(RotMatrix{3, Float64}(rotation(estimate(fit)) * R')), norm(translation(estimate(fit)) - t))
    l2 = fit_rigid_alignment(cs)
    @test retcode_symbol(l2) === :solved
    clean = fit_rigid_alignment(Correspondences(walk[setdiff(1:150, bad)], gt[setdiff(1:150, bad)]))
    e_clean = err(clean)
    e_l2 = err(l2)
    @test e_l2[1] > 10 * max(e_clean[1], 1.0e-4) || e_l2[2] > 10 * max(e_clean[2], 1.0e-3)   # the L2 alignment moves
    for K in (HuberShape, WelschShape)
        rob = fit_rigid_alignment(cs; optimization_strategy = Irls{K}(), scale = σ)
        @test retcode_symbol(rob) === :converged
        e = err(rob)
        @test e[1] < 2.0e-3 && e[2] < 2.0e-3                                           # beyond tolerance: no
        @test maximum(weights(rob)[bad]) < 0.1
        @test rob.n == (K === WelschShape ? count(>(0), weights(rob)) : 150)
    end
    # prior weights drop pairs; fewer than three, or collinear, is refused
    @test_throws ArgumentError fit_rigid_alignment(Correspondences(walk[1:2], gt[1:2]))
    line = [SVector(Float64(k), 0.0, 0.0) for k in 1:5]
    @test_throws ArgumentError fit_rigid_alignment(Correspondences(line, line))
    p = ones(150); p[bad] .= 0
    @test err(fit_rigid_alignment(withattrs(cs; weights = p)))[1] ≈ e_clean[1] atol = 1.0e-12
end

@testitem "fit_robot_world_hand_eye: a robust fit through gross observations" setup = [FitContractScenes, PoseScenes] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StableRNGs
    using VisualGeometryToolkitEstimation: fit_robot_world_hand_eye
    using VisualGeometryToolkitCore: Irls, Triggs
    using VisualGeometryToolkitCore: HuberShape, WelschShape, log_so3, weights, weight_kind, noise_scale, scale_source,
        IrlsWeights, BinaryWeights, ScaleGiven, ScaleNone, retcode
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
    errY(fit) = (norm(log_so3(rotation(estimate(fit).Y)' * rotation(Y))), norm(translation(estimate(fit).Y) - translation(Y)))
    l2 = fit_robot_world_hand_eye(markers)
    @test weight_kind(l2) === BinaryWeights() && scale_source(l2) === ScaleNone
    # Huber (monotone: never zeroes a weight) from the poisoned least-squares seed. A redescending loss at the
    # same fixed width may reject a whole marker at that seed: then an ArgumentError naming the monotone loss
    try
        fit_robot_world_hand_eye(markers; σ_rot = σ_R, σ_trans = σ_t, optimization_strategy = Irls{WelschShape}(), scale = 1.0)
    catch err
        @test err isa ArgumentError && occursin("HuberShape", err.msg)
    end
    for K in (HuberShape,)
        rob = fit_robot_world_hand_eye(markers; σ_rot = σ_R, σ_trans = σ_t, optimization_strategy = Irls{K}(), scale = 1.0)
        @test retcode_symbol(rob) === :converged
        @test weight_kind(rob) === IrlsWeights() && noise_scale(rob) === 1.0 && scale_source(rob) === ScaleGiven
        @test length(weights(rob)) == N * M
        @test maximum(weights(rob)[[4, 9, 15, 22, 27]]) < 0.2                   # marker 1's block comes first
        eR, et = errY(rob)
        lR, lt = errY(l2)
        @test eR < 5.0e-3 && et < 5.0e-3
        @test eR < lR / 5 || et < lt / 5
        @test all(isfinite, rob.cov) && rob.min_eigval > 0
    end
    # the robust fit needs its whitening; a Triggs is a refine's
    @test_throws ArgumentError fit_robot_world_hand_eye(markers; optimization_strategy = Irls{HuberShape}(), scale = 1.0)
    @test_throws ArgumentError fit_robot_world_hand_eye(markers; σ_rot = σ_R, σ_trans = σ_t,
                                                       optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0), scale = 1.0)
    @test_throws ArgumentError fit_robot_world_hand_eye(markers; scale = 1.0)
    @test_throws ArgumentError fit_robot_world_hand_eye(markers; local_optimization = Irls{HuberShape}())   # retired, naming optimization_strategy
end

@testitem "fit contract: the default strategy is the TOML table's (shape and max_iter)" setup = [FitContractScenes] begin
    using Test
    using VisualGeometryToolkitEstimation: optimization_strategy_from_toml, _dflt
    using VisualGeometryToolkitCore: Irls
    using VisualGeometryToolkitCore: HuberShape
    for (name, _, fit) in CONTRACT_FITS
        name in (:fit_rigid_alignment, :fit_robot_world_hand_eye) && continue   # their default is least squares (no table strategy)
        lo = optimization_strategy_from_toml(_dflt(String(name), "optimization_strategy"); class = name)
        @test lo isa Irls{HuberShape}                                   # Huber where accuracy matters (owner's ruling)
        @test lo.max_iter == (name === :fit_rotation_averaging ? 100 : 50)  # the table's, not a bare Irls's 25
        a = fit((; scale = 0.02)); b = fit((; optimization_strategy = lo, scale = 0.02))
        @test estimate(a) == estimate(b) && a.iterations == b.iterations && a.weights == b.weights
    end
end
