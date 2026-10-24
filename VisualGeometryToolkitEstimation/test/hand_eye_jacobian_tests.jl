@testitem "hand_eye_jacobian" setup = [PoseScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
using Test
using LinearAlgebra
using StaticArrays
using Random
using StableRNGs
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: rotation, translation   # VGC accessors (public)
using VisualGeometryToolkitCore: RigidExpManifold6   # the hand-eye refit chart (public)
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: data_size, ChartLm

# Hand-eye uses an exact ForwardDiff Jacobian (constraint_with_jacobians' Gθ is only
# the near-optimum GN approximation), so we validate the FLM refit end-to-end:
# synthesize a solvable AX=XB scene, perturb the true X, and check the
# posterior-weighted residual drops after the pose-LM refit.
@testset "Hand-eye LO (NonlinLsq FLM) reduces residual" begin
    rng = StableRNG(19)
    sc = robot_world_hand_eye_scene(rng; N = 8, Y = (ang_max = π / 2,), X = (ang_max = π / 2,), A = (ang_max = π / 2,))
    Xtrue = only(sc.X); A = sc.A
    B = [marker_pose(sc.Y, a, Xtrue) for a in A]
    problem = HandEyeProblem(A .=> B)
    n = data_size(problem)
    w = ones(n)

    hecost(X) = VgtkCore.rows_cost(
        ChartLm(RigidExpManifold6(rotation(X), translation(X)), problem, repeat(w; inner = 6)), zero(SVector{6, Float64}), 6n
    )

    Xpert = retract_pose(Xtrue, (0.4, -0.32, 0.24, 0.1, -0.05, 0.08))
    Xref = VgtkEstimation.refine!(nothing, problem, Xpert, w)

    @test Xref !== nothing
    @test Xref isa EuclideanMap
    c0 = hecost(Xpert); c1 = hecost(Xref)
    @test c1 < c0
    @test c1 < 0.5 * c0
end
end
