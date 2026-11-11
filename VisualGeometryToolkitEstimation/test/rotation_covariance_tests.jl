# The per-edge covariance of a relative rotation (the owner's ruling): a
# RotationProblem whose pixels carry their keypoints' covariances (Uncertain),
# whitened through Core's Sampson layer on every route; `uncertain_rotation`,
# its fit's rotation with the absolute covariance noise_scale² × solver_covariance
# in the so(3) chart, against a Monte Carlo of the pipeline under ANISOTROPIC
# keypoint noise; the chart the averaging reads it in; and fit_rotation_averaging
# whitening each Uncertain edge by it (scale upstream, a weak edge down-weighted).

@testsetup module RotationEdgeScenes
    using Random, LinearAlgebra, StaticArrays
    using Rotations: RotMatrix, RotXYZ
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CameraModel, px, Uncertain, CovMat, csponds
    export edge_camera, edge_scene, anisotropic_covariances, jitter, uncertain_pixels

    edge_camera() = CameraModel(800.0px, SA[320.0px, 240.0px])
    # n directions inside both views (±0.35 rad off axis), their exact pixels in views 1 and 2
    function edge_scene(rng, cam, R; n = 40)
        dirs = [normalize(SVector(0.7 * (rand(rng) - 0.5), 0.5 * (rand(rng) - 0.5), 1.0)) for _ in 1:n]
        return [VgtkCore.project(cam, d) for d in dirs], [VgtkCore.project(cam, R * d) for d in dirs]
    end
    # one anisotropic 2×2 covariance per keypoint (px²): σ_major/σ_minor up to 4, at a random angle, times s²
    function anisotropic_covariances(rng, n; s = 0.5)
        map(1:n) do _
            θ = π * rand(rng); a = 1 + 3 * rand(rng)
            Q = SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ))
            s^2 * Q * SMatrix{2, 2}(a, 0.0, 0.0, 1 / a) * Q'
        end
    end
    # a pixel moved by a draw of N(0, Σ)
    jitter(rng, u, Σ) = typeof(u)(u + SMatrix{2, 2}(cholesky(Symmetric(Matrix(Σ))).L) * SVector(randn(rng), randn(rng)))
    uncertain_pixels(us, Σs) = [Uncertain(u, CovMat(SMatrix{2, 2}(Σ))) for (u, Σ) in zip(us, Σs)]
end

@testitem "rotation edge: Uncertain pixels whiten every route of RotationProblem" setup = [RotationEdgeScenes] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using Rotations: RotMatrix, RotXYZ
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: csponds, whitened_residual, exact
    using VisualGeometryToolkitCore: measurement_covariance
    using VisualGeometryToolkitEstimation: RotationProblem, solver_covariance

    rng = Xoshiro(4)
    cam = edge_camera(); R = RotMatrix(RotXYZ(0.02, -0.15, 0.01))
    u1, u2 = edge_scene(rng, cam, R; n = 12)
    Σ1 = anisotropic_covariances(rng, 12); Σ2 = anisotropic_covariances(rng, 12)
    pu = RotationProblem(cam, cam, csponds(uncertain_pixels(u1, Σ1), uncertain_pixels(u2, Σ2)))
    pp = RotationProblem(cam, cam, csponds(u1, u2))
    @test pu.cs.Σ₁ == Σ1 && pu.cs.Σ₂ == Σ2                                     # the keypoints' covariances, Core's noise columns
    @test measurement_covariance(pu, 3) == VgtkCore.pair_covariance(Σ1[3], Σ2[3])
    @test measurement_covariance(pp, 3) === I
    # the whitened residual at a perturbed rotation is the Mahalanobis one: g over G_x·Σ_x·G_xᵀ
    Q = RotMatrix(RotXYZ(0.021, -0.149, 0.0102))
    for i in (1, 5, 9)
        g, Gx, _ = VgtkEstimation.constraint_with_jacobians(pu, Q, i)
        S = Gx * measurement_covariance(pu, i) * Gx'
        @test norm(whitened_residual(pu, Q, i)) ≈ sqrt(dot(g, S \ g)) rtol = 1.0e-10
    end
    # plain pixels: no noise columns, the problem it always was
    @test !VgtkCore.hasattr(pp.cs, :Σ₁) && !VgtkCore.hasattr(pp.cs, :Σ₂)
    @test solver_covariance(pu, R, collect(1:12)) != solver_covariance(pp, R, collect(1:12))
    # an exact side is allowed (the other carries the noise); both exact is refused (Core's rule)
    @test RotationProblem(cam, cam, csponds(exact.(u1), uncertain_pixels(u2, Σ2))) isa RotationProblem
    @test_throws ArgumentError RotationProblem(cam, cam, csponds(exact.(u1), exact.(u2)))
end

@testitem "rotation edge: uncertain_rotation matches a Monte Carlo of find_rotation under anisotropic keypoint noise" setup = [RotationEdgeScenes] begin
    using Test, Random, LinearAlgebra, StaticArrays, Statistics
    using Rotations: RotMatrix, RotXYZ
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: csponds, log_so3, estimate, value_of, MsacScoring, Uncertain, CovMat,
        noise_scale, scale_source, ScaleEstimated
    using VisualGeometryToolkitEstimation: RotationProblem, find_rotation, uncertain_rotation, solver_covariance
    using StableRNGs: StableRNG

    rng = Xoshiro(3)
    cam = edge_camera(); R = RotMatrix(RotXYZ(0.02, -0.15, 0.01))
    n = 40
    u1, u2 = edge_scene(rng, cam, R; n)
    Σ1 = anisotropic_covariances(rng, n); Σ2 = anisotropic_covariances(rng, n)
    # the covariance at the truth, unit variance factor (the covariances are the noise)
    p0 = RotationProblem(cam, cam, csponds(uncertain_pixels(u1, Σ1), uncertain_pixels(u2, Σ2)))
    exact_fit = find_rotation(cam, cam, csponds(uncertain_pixels(u1, Σ1), uncertain_pixels(u2, Σ2)); rng = StableRNG(1))
    e0 = uncertain_rotation(p0, exact_fit; scale = 1.0)
    @test e0 isa Uncertain
    @test SMatrix{3, 3}(value_of(e0)) ≈ SMatrix{3, 3}(R) atol = 1.0e-10
    ΣR = SMatrix{3, 3}(cov(e0))
    @test ΣR ≈ ΣR' && all(>(0), eigvals(Symmetric(Matrix(ΣR))))
    @test @inferred(uncertain_rotation(p0, exact_fit; scale = 1.0)) isa Uncertain
    # Monte Carlo of the pipeline: keypoints drawn from their covariances, find_rotation (its scorer's threshold in
    # whitened units; its LO the whitened chart LM), the error in the left chart log(R̂·Rᵀ)
    trials = 300
    runs = map(1:trials) do t
        cs = csponds(uncertain_pixels([jitter(rng, u, Σ) for (u, Σ) in zip(u1, Σ1)], Σ1),
                     uncertain_pixels([jitter(rng, u, Σ) for (u, Σ) in zip(u2, Σ2)], Σ2))
        p = RotationProblem(cam, cam, cs)
        res = find_rotation(cam, cam, cs; scoring = MsacScoring(p; threshold = 4.0), rng = StableRNG(t))
        (log_so3(SMatrix{3, 3}(estimate(res)) * SMatrix{3, 3}(R)'), p, res)
    end
    ω = first.(runs)
    C = cov(reduce(hcat, ω); dims = 2)
    # find_rotation's scorer ORIGINATES the scale, in whitened units (B4's record): ≈ 1 when the covariances are the noise;
    # uncertain_rotation reads it by default (scale = result)
    @test all(r -> scale_source(r[3]) === ScaleEstimated, runs)
    scales = [noise_scale(r[3]) for r in runs]
    @test 0.6 < median(scales) < 1.4
    _, p1, r1 = runs[1]
    @test SMatrix{3, 3}(cov(uncertain_rotation(p1, r1))) == SMatrix{3, 3}(cov(uncertain_rotation(p1, r1; scale = noise_scale(r1))))
    @test SMatrix{3, 3}(cov(uncertain_rotation(p1, r1))) ≈ noise_scale(r1)^2 * SMatrix{3, 3}(cov(uncertain_rotation(p1, r1; scale = 1.0))) rtol = 1.0e-12
    rel(A) = norm(C - A) / norm(A)
    # 300 trials: a sample covariance entry within ~3·√(2/300) ≈ 25 % of the diagonal scale
    @test rel(ΣR) < 0.25
    @test all(0.7 .< diag(C) ./ diag(ΣR) .< 1.4)
    # the covariances matter: the isotropic model (each keypoint's mean variance) misses the Monte Carlo by more
    iso(Σs) = [SMatrix{2, 2}(tr(Σ) / 2 * I) for Σ in Σs]
    piso = RotationProblem(cam, cam, csponds(uncertain_pixels(u1, iso(Σ1)), uncertain_pixels(u2, iso(Σ2))))
    Σiso = SMatrix{3, 3}(cov(uncertain_rotation(piso, exact_fit; scale = 1.0)))
    @test rel(ΣR) < rel(Σiso)
    # the scale is the variance factor: Σ_R scales by its square; a width is no scale
    @test SMatrix{3, 3}(cov(uncertain_rotation(p0, exact_fit; scale = 2.0))) ≈ 4 * ΣR rtol = 1.0e-12
    @test_throws ArgumentError uncertain_rotation(p0, exact_fit; scale = 0.0)
end

@testitem "rotation edge: the chart of Σ_R is the averaging's (left perturbation, transported by R_j)" setup = [RotationEdgeScenes] begin
    using Test, Random, LinearAlgebra, StaticArrays, Statistics
    using Rotations: RotMatrix, QuatRotation
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: SO3Manifold, retract, retract_jacobian, exp_so3, log_so3, propagate
    using VisualGeometryToolkitEstimation: RotationProblem

    # RotationProblem's chart is SO3Manifold's LEFT exp, R(δ) = exp(δ)·R: the chart Σ_R is in
    rng = Xoshiro(2)
    R = SMatrix{3, 3}(RotMatrix(QuatRotation(normalize(randn(rng, 4)))))
    δ = SVector(1.0e-3, -2.0e-3, 0.5e-3)
    @test SMatrix{3, 3}(retract(SO3Manifold(R), δ)) ≈ SMatrix{3, 3}(exp_so3(δ)) * R atol = 1.0e-14
    # the 9×9 vec(R) form is the chart's pushed forward, and reads back (_rotation_chart_covariance)
    L = @SMatrix [1.0 0.0 0.0; 0.3 2.0 0.0; -0.2 0.4 0.5]; Σ = 1.0e-4 * L * L'
    Σ9 = propagate(retract_jacobian(SO3Manifold(R)), Σ)
    @test VgtkEstimation._rotation_chart_covariance(SMatrix{9, 9}(Σ9), R) ≈ Σ rtol = 1.0e-10
    # the averaging's edge residual r = log(R_j·R_ij·R_iᵀ), its whitening W = whitening_factor(R_j, Σ): with R_ij drawn in
    # the LEFT chart at covariance Σ (the pairwise fit's), W·r is unit white noise at the true frames ...
    Ri = SMatrix{3, 3}(RotMatrix(QuatRotation(normalize(randn(rng, 4)))))
    Rj = SMatrix{3, 3}(RotMatrix(QuatRotation(normalize(randn(rng, 4)))))
    Rij = Rj' * Ri
    W = only(VgtkEstimation._edge_whitening([Σ], [(1, 2)], Dict(1 => Ri, 2 => Rj)))
    draw() = L * SVector(randn(rng), randn(rng), randn(rng)) * 1.0e-2
    left = [W * log_so3(Rj * SMatrix{3, 3}(exp_so3(draw())) * Rij * Ri') for _ in 1:4000]
    @test norm(cov(reduce(hcat, left); dims = 2) - I) < 0.15
    # ... and a RIGHT-chart draw (R_ij·exp(δ)) is not: the transport is the chart's
    right = [W * log_so3(Rj * Rij * SMatrix{3, 3}(exp_so3(draw())) * Ri') for _ in 1:4000]
    @test norm(cov(reduce(hcat, right); dims = 2) - I) > 0.5
end

@testitem "rotation edge: the averaging of uncertain_rotation edges — scale upstream, a weak edge down-weighted" setup = [RotationEdgeScenes] begin
    using Test, Random, LinearAlgebra, StaticArrays, Statistics
    using Rotations: RotMatrix, RotXYZ, QuatRotation, rotation_angle
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: csponds, estimate, value_of, Attributed, Uncertain, CovMat, HuberShape, MsacScoring,
        noise_scale, scale_source, ScaleUpstream, ScaleGiven, ScaleNone, ScaleEstimated
    using VisualGeometryToolkitEstimation: RotationProblem, find_rotation, uncertain_rotation, fit_rotation_averaging, Irls
    using StableRNGs: StableRNG

    cam = edge_camera()
    nf = 5
    edges = [(i, j) for i in 1:nf for j in (i + 1):min(nf, i + 2)]
    weak = (2, 3)
    # pixel noise per edge: the weak edge's keypoints 8× noisier (its Σ says so)
    s(e) = e == weak ? 4.0 : 0.5
    err(est, R) = maximum(k -> rotation_angle(RotMatrix{3}(SMatrix{3, 3}(est[k])' * R[k])), 1:nf)
    function trial(seed)
        rng = Xoshiro(seed)
        # small turns between frames so every pair overlaps; frame k's directions into the common frame: d = R_k·f_k
        R = Dict(k => SMatrix{3, 3}(RotMatrix(RotXYZ(0.03k, -0.04k, 0.02k))) for k in 1:nf)
        R = Dict(k => R[1]' * R[k] for k in 1:nf)                              # the gauge: frame 1 the identity
        fits = map(edges) do (i, j)
            Rij = R[j]' * R[i]                                                 # f_j = R_ij·f_i
            u1, u2 = edge_scene(rng, cam, RotMatrix{3}(Rij); n = 30)
            Σ1 = anisotropic_covariances(rng, 30; s = s((i, j))); Σ2 = anisotropic_covariances(rng, 30; s = s((i, j)))
            cs = csponds(uncertain_pixels([jitter(rng, u, Σ) for (u, Σ) in zip(u1, Σ1)], Σ1),
                         uncertain_pixels([jitter(rng, u, Σ) for (u, Σ) in zip(u2, Σ2)], Σ2))
            p = RotationProblem(cam, cam, cs)
            res = find_rotation(cam, cam, cs; scoring = MsacScoring(p; threshold = 4.0), rng = StableRNG(seed))
            (p, res)
        end
        unc = Dict(e => uncertain_rotation(p, res) for (e, (p, res)) in zip(edges, fits))   # at each fit's own scale
        plain = Dict(e => SMatrix{3, 3}(estimate(res)) for (e, (_, res)) in zip(edges, fits))
        a = fit_rotation_averaging(unc; optimization_strategy = Irls{HuberShape}())
        b = fit_rotation_averaging(plain; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3)
        return err(estimate(a), R), err(estimate(b), R), a, b, unc, fits
    end
    res = [trial(k) for k in 1:10]
    @test mean(r[1] for r in res) < mean(r[2] for r in res)                   # the covariance-weighted average is the better one
    _, _, a, b, unc, fits = res[1]
    # the weak edge's covariance is the wider (8× noise: ~64× variance), so it weighs less
    @test tr(SMatrix{3, 3}(cov(unc[weak]))) > 20 * tr(SMatrix{3, 3}(cov(unc[(1, 2)])))
    # scale_source at each stage: the pairwise scorer estimates it (whitened units) → the edges' absolute covariances →
    # the averaging's upstream
    @test all(f -> scale_source(f[2]) === ScaleEstimated, fits)
    @test noise_scale(a) == 1.0 && scale_source(a) == ScaleUpstream
    @test scale_source(b) == ScaleGiven
    w = fit_rotation_averaging(unc; optimization_strategy = Irls{HuberShape}(), width = 2.0)
    @test scale_source(w) == ScaleNone && w.loss_width == 2.0
    @test scale_source(fit_rotation_averaging(unc; optimization_strategy = nothing)) == ScaleNone
    # absolute covariances take no scale; a graph mixing Uncertain and plain edges is refused
    @test_throws r"give no `scale`" fit_rotation_averaging(unc; optimization_strategy = Irls{HuberShape}(), scale = 1.0)
    mixed = Dict{Tuple{Int, Int}, Any}(unc); mixed[weak] = value_of(unc[weak])
    @test_throws r"every edge its covariance" fit_rotation_averaging(mixed; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3)
    @test @inferred(fit_rotation_averaging(unc; optimization_strategy = Irls{HuberShape}())) isa VgtkCore.FitResult
end

@testitem "rotation averaging: Uncertain edges at a known covariance — the synthetic checks" begin
    using Test, Random, LinearAlgebra, StaticArrays, Statistics
    using Rotations: RotMatrix, RotationVec, QuatRotation, rotation_angle
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Attributed, Uncertain, CovMat, estimate, HuberShape, retract_jacobian, SO3Manifold, propagate
    using VisualGeometryToolkitEstimation: fit_rotation_averaging, Irls

    n = 7
    edges = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 2)]
    weak = (3, 4)
    σ, σw = 0.004, 0.04                                              # rad per axis: a typical edge, the weak one
    err(e, R) = maximum(k -> rotation_angle(RotMatrix{3}(SMatrix{3, 3}(e[k])' * R[k])), 1:n)
    function trial(seed)
        rng = Xoshiro(seed)
        R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
        s(e) = e == weak ? σw : σ
        rel = Dict(e => RotMatrix(RotationVec((s(e) .* randn(rng, 3))...)) * (R[e[2]]' * R[e[1]]) for e in edges)
        uni = fit_rotation_averaging(rel; optimization_strategy = Irls{HuberShape}(), scale = σ)
        unc = fit_rotation_averaging(Dict(e => Attributed(Uncertain(rel[e], CovMat(SMatrix{3, 3}(s(e)^2 * I))), (; weight = 1.0))
                                          for e in edges); optimization_strategy = Irls{HuberShape}())
        return err(estimate(uni), R), err(estimate(unc), R), uni, unc
    end
    res = [trial(k) for k in 1:20]
    @test mean(r[2] for r in res) < mean(r[1] for r in res)          # the covariance-weighted average is the better one
    # the weak edge's information relative to a typical edge: the uniform run's Huber weight ratio, the
    # Uncertain run's loss weight × (σ/σw)²
    _, _, uni, unc = res[1]
    q = findfirst(==(weak), unc.edges); t = findfirst(==((1, 2)), unc.edges)
    @test unc.weights[q] * (σ / σw)^2 / unc.weights[t] < uni.weights[q] / uni.weights[t]

    # absolute σ²·I edges are the plain edges at scale σ (to rounding: the whitened path is another evaluation order)
    rng = Xoshiro(9)
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
    rel = Dict(e => RotMatrix(RotationVec((σ .* randn(rng, 3))...)) * (R[e[2]]' * R[e[1]]) for e in edges)
    plain = fit_rotation_averaging(rel; scale = σ)
    eye = fit_rotation_averaging(Dict(e => Uncertain(rel[e], CovMat(SMatrix{3, 3}(σ^2 * I))) for e in edges))
    @test maximum(k -> norm(SMatrix{3, 3}(estimate(plain)[k]) - SMatrix{3, 3}(estimate(eye)[k])), 1:n) < 1.0e-10
    @test plain.weights ≈ eye.weights rtol = 1.0e-10
    # a 9×9 vec(R) covariance (solver_covariance's form) is read into the chart: J Σω Jᵀ gives back Σω
    Σω = SMatrix{3, 3}(Diagonal(SVector(1.0, 2.0, 3.0))) * σ^2
    nine = Dict(e => Uncertain(rel[e], CovMat(propagate(retract_jacobian(SO3Manifold(SMatrix{3, 3}(rel[e]))), Σω))) for e in edges)
    three = Dict(e => Uncertain(rel[e], CovMat(Σω)) for e in edges)
    a = fit_rotation_averaging(nine); b = fit_rotation_averaging(three)
    @test maximum(k -> norm(SMatrix{3, 3}(estimate(a)[k]) - SMatrix{3, 3}(estimate(b)[k])), 1:n) < 1.0e-9
    @test_throws ArgumentError fit_rotation_averaging(Dict(e => Uncertain(rel[e], CovMat(SMatrix{2, 2}(1.0I))) for e in edges))
end
