# The per-edge covariance of a relative rotation (the owner's ruling, B2-cov):
# RotationProblem's solver_covariance (the constraint sandwich, the implicit-
# function covariance of its whitened-constraint estimator) against a Monte
# Carlo of that estimator, and fit_rotation_averaging whitening each Uncertain
# edge by it.

@testitem "rotation covariance: solver_covariance(::RotationProblem) matches a Monte Carlo of its estimator" begin
    using Test, Random, LinearAlgebra, StaticArrays, Statistics
    using Rotations: RotMatrix, RotXYZ
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CameraModel, px, csponds, log_so3, exp_so3, whitened_residual,
        retract_jacobian, SO3Manifold, propagate, provides_solver_covariance
    using VisualGeometryToolkitEstimation: RotationProblem, solver_covariance

    rng = Xoshiro(3)
    cam = CameraModel(800.0px, SA[320.0px, 240.0px])
    R = RotMatrix(RotXYZ(0.02, -0.15, 0.01))
    n = 40
    # directions inside both views (±0.35 rad off axis), their exact pixels
    dirs = [normalize(SVector(0.7 * (rand(rng) - 0.5), 0.5 * (rand(rng) - 0.5), 1.0)) for _ in 1:n]
    u1 = [VgtkCore.project(cam, d) for d in dirs]
    u2 = [VgtkCore.project(cam, R * d) for d in dirs]
    p0 = RotationProblem(cam, cam, csponds(u1, u2))
    @test provides_solver_covariance(typeof(p0))
    Σ9 = solver_covariance(p0, R, collect(1:n))
    @test size(Σ9) == (9, 9)
    J = retract_jacobian(SO3Manifold(SMatrix{3, 3}(R)))
    Σω = propagate(J' / 2, Σ9)                                     # the so(3) chart: Jᵀ Σ J / 4
    @test propagate(J, Σω) ≈ Σ9 rtol = 1.0e-8                       # the vec(R) covariance IS the chart's, pushed forward

    # Monte Carlo of the estimator the problem's rows define: the whitened constraint
    # least squares over every point (Gauss–Newton in the left chart, finite-difference rows)
    σ = 0.5                                                        # px per coordinate
    function fit_gn(p)
        Rh = SMatrix{3, 3}(R)
        res(Q) = reduce(vcat, [whitened_residual(p, RotMatrix{3, Float64}(Q), i) for i in 1:n])
        for _ in 1:6
            r0 = res(Rh)
            Jc = reduce(hcat, [(res(SMatrix{3, 3}(exp_so3(1.0e-7 * SVector(ntuple(k -> k == c ? 1.0 : 0.0, 3)))) * Rh) - r0) / 1.0e-7
                               for c in 1:3])
            δ = -(Jc \ r0)
            Rh = SMatrix{3, 3}(exp_so3(SVector{3}(δ))) * Rh
        end
        return log_so3(Rh * SMatrix{3, 3}(R)')
    end
    trials = 300
    jit(u) = typeof(u)(u + σ * SVector(randn(rng), randn(rng)))     # the pixels, the camera's point type
    ω = [fit_gn(RotationProblem(cam, cam, csponds(jit.(u1), jit.(u2)))) for _ in 1:trials]
    C = cov(reduce(hcat, ω); dims = 2)
    # 300 trials: each entry of a sample covariance within ~3·√(2/300) ≈ 25 % of the diagonal scale
    @test norm(C - σ^2 * Σω) / norm(σ^2 * Σω) < 0.25
    @test all(diag(C) ./ diag(σ^2 * Σω) .> 0.7) && all(diag(C) ./ diag(σ^2 * Σω) .< 1.4)
end

@testitem "rotation averaging: Uncertain edges, a weak edge weighs less than at a uniform scale" begin
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
                                          for e in edges); optimization_strategy = Irls{HuberShape}(), scale = 1.0)
        return err(estimate(uni), R), err(estimate(unc), R), uni, unc
    end
    res = [trial(k) for k in 1:20]
    @test mean(r[2] for r in res) < mean(r[1] for r in res)          # the covariance-weighted average is the better one
    # the weak edge's information relative to a typical edge: the uniform run's Huber weight ratio, the
    # Uncertain run's loss weight × (σ/σw)²
    _, _, uni, unc = res[1]
    q = findfirst(==(weak), unc.edges); t = findfirst(==((1, 2)), unc.edges)
    @test unc.weights[q] * (σ / σw)^2 / unc.weights[t] < uni.weights[q] / uni.weights[t]

    # Σ = I edges are the plain edges (to rounding: the whitened path is another evaluation order)
    rng = Xoshiro(9)
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
    rel = Dict(e => RotMatrix(RotationVec((σ .* randn(rng, 3))...)) * (R[e[2]]' * R[e[1]]) for e in edges)
    plain = fit_rotation_averaging(rel; scale = σ)
    eye = fit_rotation_averaging(Dict(e => Uncertain(rel[e], CovMat(SMatrix{3, 3}(1.0I))) for e in edges); scale = σ)
    @test maximum(k -> norm(SMatrix{3, 3}(estimate(plain)[k]) - SMatrix{3, 3}(estimate(eye)[k])), 1:n) < 1.0e-10
    # a 9×9 vec(R) covariance (solver_covariance's form) is read into the chart: J Σω Jᵀ gives back Σω
    Σω = SMatrix{3, 3}(Diagonal(SVector(1.0, 2.0, 3.0))) * σ^2
    nine = Dict(e => Uncertain(rel[e], CovMat(propagate(retract_jacobian(SO3Manifold(SMatrix{3, 3}(rel[e]))), Σω))) for e in edges)
    three = Dict(e => Uncertain(rel[e], CovMat(Σω)) for e in edges)
    a = fit_rotation_averaging(nine; scale = 1.0); b = fit_rotation_averaging(three; scale = 1.0)
    @test maximum(k -> norm(SMatrix{3, 3}(estimate(a)[k]) - SMatrix{3, 3}(estimate(b)[k])), 1:n) < 1.0e-9
    @test_throws ArgumentError fit_rotation_averaging(Dict(e => Uncertain(rel[e], CovMat(SMatrix{2, 2}(1.0I))) for e in edges); scale = 1.0)
end
