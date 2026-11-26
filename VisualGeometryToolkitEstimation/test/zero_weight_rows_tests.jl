# R22: contract test — the chart rows' zero-weight rule (`residual_rows!`, `refine!`)
# A zero-weight row is exactly 0 (0.11.4). Every chart refit — the LO's `refine!`, a `refine_*`'s least squares and
# both Triggs paths — runs `_chart_refit`, which builds the rows as √w·e. A datum at weight 0 whose own residual is
# Inf or NaN in Float64 gave 0·Inf = NaN: the LM started at a NaN cost and handed back its seed. The rule, in that
# one place: a row (and its Jacobian rows) whose √w is 0 is exactly 0, whatever the problem's row is; a row at a
# nonzero √w is the problem's own, bit for bit.

@testitem "zero-weight rows: the ellipse and hand-eye refits (red before 0.11.4)" setup = [ConicScenes, PoseScenes] begin
    using VisualGeometryToolkitCore: solve, RefineProblem
    using VisualGeometryToolkitEstimation: LocalOptimization
    using Test, LinearAlgebra, StaticArrays, StableRNGs
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: Point2, Ellipse, EuclideanMap, RigidExpManifold6, residual_rows!, rotation, translation
    using VisualGeometryToolkitEstimation: EllipseProblem, HandEyeProblem, data_size, ChartLm

    # the rows of datum k at the anchor of θ, at unit weight (the scenario's precondition)
    function rows_at(p, θ, k)
        ch = VgtkEstimation._chart(p, VgtkEstimation._as_model(p, θ))
        rows, _ = VgtkEstimation._lm_size(p)
        r = residual_rows!(zeros(rows), VgtkEstimation._refit_state(p, ch, ones(rows)), zeros(SVector{VgtkEstimation.chart_dof(typeof(p)), Float64}))
        c = rows ÷ data_size(p)
        return r[(c * (k - 1) + 1):(c * k)]
    end
    zero_last(n) = (w = ones(n); w[n] = 0.0; w)
    centre(e) = SVector(e.center[1], e.center[2])

    @testset "ellipse: a NaN point at weight 0" begin
        pn, _ = ellipse_scene(StableRNG(91), 100; σ = 1.0, outl = 0.0)
        p = EllipseProblem(vcat(pn, [Point2(NaN, NaN)]))
        e₀ = Ellipse(Point2((centre(E₀) .+ SVector(2.0, -1.5))...), E₀.a + 3, E₀.b - 2, E₀.θ + 0.01)
        @test !all(isfinite, rows_at(p, e₀, 101))
        e₁ = VgtkCore.refine!(VgtkEstimation.refit_workspace(typeof(p), 101), p, e₀, zero_last(101))
        @test e₁ isa Ellipse{Float64}
        @test norm(centre(e₁) - centre(E₀)) < 0.5 && abs(e₁.a - E₀.a) < 0.5 && abs(e₁.b - E₀.b) < 0.5 && abs(e₁.θ - E₀.θ) < 0.01
    end

    @testset "the Triggs LO (the robustified LM): a NaN point, weighed 0 by its non-finite seed residual" begin
        pn, _ = ellipse_scene(StableRNG(92), 100; σ = 1.0, outl = 0.0)
        p = EllipseProblem(vcat(pn, [Point2(NaN, NaN)]))
        e₀ = Ellipse(Point2((centre(E₀) .+ SVector(2.0, -1.5))...), E₀.a + 3, E₀.b - 2, E₀.θ + 0.01)
        r = solve(RefineProblem(p, e₀), LocalOptimization(VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 2.5); scoring = VgtkCore.MsacScoring(p; threshold = 5.0)))
        e₁ = r.u
        @test norm(centre(e₁) - centre(E₀)) < 0.5 && abs(e₁.a - E₀.a) < 0.5 && abs(e₁.b - E₀.b) < 0.5 && abs(e₁.θ - E₀.θ) < 0.01
    end

    @testset "hand-eye: a NaN motion at weight 0" begin
        sc = robot_world_hand_eye_scene(StableRNG(19); N = 8, Y = (ang_max = π / 2,), X = (ang_max = π / 2,), A = (ang_max = π / 2,))
        Xtrue = only(sc.X); A = sc.A
        B = [marker_pose(sc.Y, a, Xtrue) for a in A]
        bad = EuclideanMap(rotation(B[1]), NaN .* translation(B[1]))         # B's own type, a NaN translation
        p = HandEyeProblem(vcat(A, [A[1]]) .=> vcat(B, [bad]))
        clean = HandEyeProblem(A .=> B)
        n = data_size(p)
        @test !all(isfinite, rows_at(p, Xtrue, n))
        cost(X) = VgtkCore.rows_cost(ChartLm(RigidExpManifold6(rotation(X), translation(X)), clean, ones(6 * data_size(clean))),
                                     zero(SVector{6, Float64}), 6 * data_size(clean))
        Xpert = retract_pose(Xtrue, (0.4, -0.32, 0.24, 0.1, -0.05, 0.08))
        Xref = VgtkCore.refine!(nothing, p, Xpert, zero_last(n))
        @test Xref !== nothing
        @test Xref !== nothing && cost(Xref) < 0.5 * cost(Xpert)
    end
end

@testitem "zero-weight rows: PnP, bearing pose and refine_* (guards; not reachable as red)" setup = [PixelPoseScenes] begin
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, successful_retcode
    # These rows were finite already, so the rule changes nothing here: PnP's rows give an invalid point (a non-finite
    # e included) the finite out-of-field residual; BearingPoseProblem refuses non-finite data, and a point at the camera
    # centre or far away has a finite tangent residual (its caller `weights` column enters the row as √w·sw·e, 0 for a
    # finite e); `refine_*` runs at unit weight except PnP's own invalid-anchor zeroing. They stay correct at weight 0.
    using Test, LinearAlgebra, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: EuclideanMap, CameraModel, Correspondences, withattrs, project
    using VisualGeometryToolkitEstimation: AbsolutePoseProblem, BearingPoseProblem
    using Rotations: RotMatrix, RotXYZ
    E = EuclideanMap(RotMatrix(RotXYZ(0.3, -0.2, 0.1)), SVector(0.1, -0.05, 2.0))
    E₀ = EuclideanMap(RotMatrix(RotXYZ(0.33, -0.18, 0.12)), SVector(0.13, -0.08, 2.05))
    cam = CameraModel(; focal = 800.0, pp = (640.0, 480.0))
    Xs = [E.R' * (SVector(0.3sin(k), 0.3cos(2k), 2.0 + 0.5sin(3k)) - E.t) for k in 1:30]
    perr(F) = maximum(pose_errors(F, E))   # PixelPoseScenes' rotation and translation errors, the larger
    zero_last(n) = (w = ones(n); w[n] = 0.0; w)
    @testset "PnP: a NaN pixel at weight 0" begin
        cs = vcat([X => SVector{2, Float64}(project(cam, E.R * X + E.t)) for X in Xs], [Xs[1] => SVector(NaN, NaN)])
        F = VgtkCore.refine!(nothing, AbsolutePoseProblem(cam, cs), E₀, zero_last(31))
        @test F !== nothing && perr(F) < 1.0e-8
    end
    fs = [normalize(E.R * X + E.t) for X in Xs]
    Xc = -(E.R' * E.t)                                    # the camera centre
    @testset "bearing pose: a point at the camera centre at weight 0" begin
        p = BearingPoseProblem(Correspondences(vcat(Xs, [Xc]), vcat(fs, [SVector(0.0, 0.0, 1.0)])))
        F = VgtkCore.refine!(nothing, p, E₀, zero_last(31))
        @test F !== nothing && perr(F) < 1.0e-8
    end
    @testset "the AbsolutePoseProblem refine on bearings: a caller weight 0" begin
        cs = withattrs(Correspondences(vcat(Xs, [Xc]), vcat(fs, [SVector(0.0, 0.0, 1.0)])); weights = zero_last(31))
        r = solve(RefineProblem(BearingPoseProblem(cs), E₀), LevenbergMarquardt(; optimization_strategy = nothing))
        @test successful_retcode(r) && perr(r.u) < 1.0e-8
    end
end
