# RotationProblem / find_rotation — the pure rotation between two views of a
# calibrated fisheye on unit bearings (the infinite homography): the minimal
# solver, the gates, the residual against VGC's bearing transfer constraint, the
# chart rows against ForwardDiff, the chart-LM refit lowering the Sampson cost
# below the kabsch start, and the recovery past 90° with outliers.

@testsetup module RotationScene
using Random: randn, rand, randperm
using LinearAlgebra: normalize
using StaticArrays: SA, SVector
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, project, backproject_with_jacobian, Csponds
export FISHEYE, rotation_scene

# A fisheye whose field reaches 100° off axis (an equidistant-based KB lens, f = 300 px).
const FISHEYE = CameraModel(300.0px, SA[600.0px, 600.0px]; lens = KannalaBrandt(-0.02))

# n correspondences of a pure rotation R between two views of FISHEYE, each ray within
# `θmax` of both optical axes; pixel noise σ (px), a fraction `outl` replaced by random pixels.
function rotation_scene(rng, R, n; σ = 0.5, outl = 0.0, θmax = deg2rad(100))
    u₁ = SVector{2, Float64}[]; u₂ = SVector{2, Float64}[]; θs = Float64[]
    while length(u₁) < n
        d = normalize(randn(rng, SVector{3, Float64}))
        (acos(clamp(d[3], -1, 1)) < θmax && acos(clamp((R * d)[3], -1, 1)) < θmax) || continue
        a = project(FISHEYE, d); b = project(FISHEYE, R * d)
        (all(isfinite, a) && all(isfinite, b)) || continue
        push!(u₁, a + σ * randn(rng, SVector{2, Float64})); push!(u₂, b + σ * randn(rng, SVector{2, Float64}))
        push!(θs, acos(clamp(d[3], -1, 1)))
    end
    bad = randperm(rng, n)[1:round(Int, outl * n)]
    for i in bad
        while true
            c = SVector(600 + 500 * (2rand(rng) - 1), 600 + 500 * (2rand(rng) - 1))
            all(isfinite, backproject_with_jacobian(FISHEYE, c)[1]) && (u₂[i] = c; break)
        end
    end
    return Csponds(u₁, u₂), θs, bad
end
end

@testitem "RotationProblem: minimal solver, gates, residual" setup = [RotationScene] begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations, Statistics
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, project, backproject_with_jacobian, Csponds,
        withattrs, sources, targets, estimate, converged, angle_between
    using VisualGeometryToolkitEstimation: RotationProblem, find_rotation
    import VisualGeometryToolkitEstimation as VgtkEstimation
    rng = Xoshiro(21)
    R = RotMatrix(RotationVec(0.3, -0.5, 0.2))
    cs, θs, _ = rotation_scene(rng, R, 60; σ = 0.0)
    p = RotationProblem(FISHEYE, FISHEYE, cs)
    @test VgtkEstimation.codimension(p) == 2 && VgtkEstimation.minimal_sample_size(p) == 2 && VgtkEstimation.chart_dof(p) == 3
    @test maximum(θs) > deg2rad(90)                                    # the field reaches past 90°
    sol = VgtkEstimation.solve(p, [1, 2])
    @test sol.count == 1 && norm(SMatrix{3, 3}(first(sol)) - R) < 1.0e-9   # exact on two clean pairs
    r = zeros(60); VgtkEstimation.residuals!(r, p, first(sol))
    @test maximum(r) < 1.0e-5                                           # pixels
    allocs(r, p, θ) = (VgtkEstimation.residuals!(r, p, θ); @allocated VgtkEstimation.residuals!(r, p, θ))
    @test allocs(r, p, first(sol)) == 0
    # the residual is the Sampson distance of the joint measurement (Σ = I) to VGC's bearing transfer constraint, in pixels
    i = 7; Rp = RotMatrix(RotationVec(0.3, -0.5, 0.21))
    g, G_x, _ = VgtkEstimation.constraint_with_jacobians(p, Rp, i)
    @test VgtkEstimation.whitened_residual(p, Rp, i) ≈ sqrt(g' * ((G_x * G_x') \ g)) rtol = 1.0e-10
    # an antipodal transfer is an outlier however small its tangent residual
    @test VgtkEstimation.whitened_residual(p, RotMatrix(RotationVec(π, 0.0, 0.0)) * first(sol), i) == Inf ||
          dot(RotMatrix(RotationVec(π, 0.0, 0.0)) * first(sol) * sources(p.cs)[i], targets(p.cs)[i]) > 0
    # the coincidence gate refuses a sample of one detection twice
    cs2 = Csponds(vcat(sources(cs)[1:2], [sources(cs)[1]]), vcat(targets(cs)[1:2], [targets(cs)[1]]))
    p2 = RotationProblem(FISHEYE, FISHEYE, cs2)
    @test !VgtkEstimation.test_sample(p2, [1, 3]) && VgtkEstimation.test_sample(p2, [1, 2])
    # parallel bearings: the minimal solver's refusal
    f = backproject_with_jacobian(FISHEYE, SA[600.0, 600.0])
    pb = RotationProblem(withattrs(Csponds([f[1], f[1], f[1]], [f[1], f[1], f[1]]); M₁ = [f[2], f[2], f[2]], M₂ = [f[2], f[2], f[2]]))
    @test VgtkEstimation.solve(pb, [1, 2]).count == 0
    # refusals: a Pair vector, 2-vectors, non-unit bearings
    @test_throws ArgumentError RotationProblem([f[1] => f[1], f[1] => f[1]])
    @test_throws ArgumentError RotationProblem(withattrs(Csponds([2 * f[1], f[1]], [f[1], f[1]]); M₁ = [f[2], f[2]], M₂ = [f[2], f[2]]))
    @test_throws ArgumentError RotationProblem(Csponds([f[1], f[1]], [f[1], f[1]]))
end

@testitem "RotationProblem: chart rows against ForwardDiff; the refit lowers the Sampson cost" setup = [RotationScene] begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations, Statistics
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, project, backproject_with_jacobian, Csponds,
        withattrs, sources, targets, estimate, converged, angle_between
    using VisualGeometryToolkitEstimation: RotationProblem, find_rotation
    import VisualGeometryToolkitEstimation as VgtkEstimation
    import ForwardDiff
    using VisualGeometryToolkitCore: kabsch, zero_tangent, residual_rows!, residual_jacobian_rows!
    rng = Xoshiro(22)
    R = RotMatrix(RotationVec(-0.4, 0.25, 0.6))
    # anisotropic fisheye noise: the periphery's pixels subtend different angles radially and tangentially
    cs, _, _ = rotation_scene(rng, R, 200; σ = 1.0)
    p = RotationProblem(FISHEYE, FISHEYE, cs)
    R0 = RotMatrix{3, Float64, 9}(kabsch(Csponds(sources(p.cs), targets(p.cs))) )   # the unweighted kabsch: the LM's start
    ch = VgtkEstimation._chart(p, R0)
    n = 200; sqrtw = rand(rng, 2n)
    lm = VgtkEstimation._lm_state(p, ch, sqrtw)
    a0 = SVector(1.0e-3, -2.0e-3, 5.0e-4)
    J = zeros(2n, 3); residual_jacobian_rows!(J, lm, a0)
    rows(a) = (out = zeros(eltype(a), 2n); residual_rows!(out, lm, SVector{3}(a)); out)
    @test J ≈ ForwardDiff.jacobian(rows, Vector(a0)) rtol = 1.0e-8
    # the chart-LM refit (refine!) from the kabsch start lowers the Sampson cost
    cost(Rx) = (r = zeros(n); VgtkEstimation.residuals!(r, p, Rx); sum(abs2, r))
    ws = VgtkEstimation.refine_workspace(typeof(p), n)
    R1 = VgtkEstimation.refine!(ws, p, R0, ones(n))
    @test R1 isa RotMatrix
    @test cost(R1) < cost(R0)
    @test Rotations.rotation_angle(RotMatrix{3}(R1' * R)) < 3.0e-3                  # at the noise floor (1 px on 200 pairs)
    refit_allocs(ws, p, R0, w) = (VgtkEstimation.refine!(ws, p, R0, w); @allocated VgtkEstimation.refine!(ws, p, R0, w))
    @test refit_allocs(ws, p, R0, ones(n)) == 0   # allocation-free: the refit's buffers are the workspace's and the task's
end

@testitem "find_rotation: a fisheye rotation past 90° with 30% outliers" setup = [RotationScene] begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations, Statistics
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, project, backproject_with_jacobian, Csponds,
        withattrs, sources, targets, estimate, converged, angle_between
    using VisualGeometryToolkitEstimation: RotationProblem, find_rotation
    import VisualGeometryToolkitEstimation as VgtkEstimation
    rng = Xoshiro(23)
    for trial in 1:3
        R = RotMatrix(QuatRotation(normalize(SVector(1.0, 0.3randn(rng), 0.3randn(rng), 0.3randn(rng)))))
        cs, θs, bad = rotation_scene(rng, R, 300; σ = 0.5, outl = 0.3)
        res = find_rotation(FISHEYE, FISHEYE, cs; rng = Xoshiro(trial))
        @test converged(res)
        @test rad2deg(Rotations.rotation_angle(RotMatrix{3}(estimate(res)' * R))) < 0.05
        inl = res.inlier_mask
        good = setdiff(1:300, bad)
        @test count(inl[good]) > 0.8 * length(good)      # 1 px MSAC on a 2-D residual of 0.5 px per coordinate in each view: P(χ²₂ < 4) = 0.86
        @test count(inl[bad]) < 0.05 * length(bad) + 2
        @test any(θs[findall(inl)] .> deg2rad(90))                       # inliers past 90° took part
    end
    @test find_rotation(FISHEYE, FISHEYE, Csponds([SA[600.0, 600.0]], [SA[600.0, 600.0]])) === nothing
    @test_throws ArgumentError find_rotation(FISHEYE, FISHEYE, rotation_scene(rng, one(RotMatrix{3}), 20)[1]; threshold = 0.01)
end
