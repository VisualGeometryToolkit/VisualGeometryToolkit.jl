# Zhang — the IAC regression from plane homographies (calibration.jl)
@testitem "Zhang" begin
    using Test
    using StaticArrays: SVector, SMatrix, @SMatrix
    using Rotations: RotXYZ
    using VisualGeometryToolkitCore: HomographyMat, LogicalIntrinsics, Size2, Attributed,
        focal_length, μm, px, ReturnCode, ClosedForm
    using VisualGeometryToolkitCore: solve, EstimationProblem, Zhang
    import VisualGeometryToolkitCore as VgtkCore

    # the closed form's model (its FitResult's `u`); the context is the principal point and, optionally, the pitch
    zhang(Hs; center) = solve(EstimationProblem(LogicalIntrinsics, Hs, (; center)), Zhang()).u
    zhang(Hs, pitch; center) = solve(EstimationProblem(LogicalIntrinsics, Hs, (; center, pitch)), Zhang()).u

    # the synthetic views (`zhang_views`, `zhang_homography`, `ZHANG_CENTER`): the suite's fixture file, shared with
    # the golden records
    include(joinpath(pkgdir(VgtkCore), "test", "fixtures", "zhang_views.jl"))
    const _c = ZHANG_CENTER

    @testset "Zhang" begin
        fx, fy = 1226.3, 1219.7
        Hs = zhang_views(fx, fy)

        @testset "free fx, fy recovery" begin
            intr = zhang(Hs; center = _c)
            @test SMatrix{3, 3}(intr.K)[1:2, 3] == _c                     # pixels in, pixels out
            @test_throws ArgumentError solve(EstimationProblem(LogicalIntrinsics, Hs), Zhang())   # the centre is required (the context lacks it)
            # the result record: one closed-form step over every homography
            r = solve(EstimationProblem(LogicalIntrinsics, Hs, (; center = _c)), Zhang())
            @test r.u == intr
            @test r.retcode === ReturnCode.Success && r.stop_rule === ClosedForm
            @test r.stats.nsteps == 0 && isnan(r.rms) && r.n == length(Hs)
            f̂x, f̂y = focal_length(intr)
            @test f̂x ≈ fx atol = 1.0e-6
            @test f̂y ≈ fy atol = 1.0e-6
            @test intr isa LogicalIntrinsics{Float64}
        end

        @testset "weights are per-homography vote strengths, carried as Attributed" begin
            # a garbage homography with zero weight must not perturb the solve
            junk = zhang_homography(
                (@SMatrix [500.0 0 0; 0 900.0 0; 0 0 1.0]),
                RotXYZ(0.3, 0.2, 0.1), SVector(0.1, 0.1, 2.0)
            )
            voted = [[Attributed(H, (; weight = 10)) for H in Hs]; Attributed(junk, (; weight = 0))]   # integer votes
            intr = zhang(voted; center = _c)
            f̂x, f̂y = focal_length(intr)
            @test f̂x ≈ fx atol = 1.0e-6
            @test f̂y ≈ fy atol = 1.0e-6
            # a bare homography weighs 1; Attributed and bare mix
            @test focal_length(zhang([Hs[1:3]; [Attributed(H, (; weight = 1.0)) for H in Hs[4:end]]]; center = _c))[1] ≈ fx atol = 1.0e-6
            pitch = Size2(width = 1.0μm / px, height = 1.0μm / px)
            @test zhang(voted[1:end-1], pitch; center = _c) isa LogicalIntrinsics{Float64}
        end

        @testset "known pitch → single f (dispatch on Size2)" begin
            pitch = Size2(width = 3.9μm / px, height = 4.0μm / px)
            τ = 3.9 / 4.0                      # fy/fx = pitch.width/pitch.height
            f0 = 1300.0
            Hs_τ = zhang_views(f0, τ * f0)
            intr = zhang(Hs_τ, pitch; center = _c)
            @test SMatrix{3, 3}(intr.K)[1:2, 3] == _c
            f̂x, f̂y = focal_length(intr)
            @test f̂x ≈ f0 atol = 1.0e-6
            @test f̂y / f̂x ≈ τ atol = 1.0e-12    # aspect is imposed, not estimated
        end

        @testset "degenerate: fronto-parallel views throw" begin
            K = @SMatrix [fx 0.0 _c[1]; 0.0 fy _c[2]; 0.0 0.0 1.0]
            flat = [
                zhang_homography(K, RotXYZ(0.0, 0.0, ψ), SVector(0.1ψ, -0.2ψ, 2.0 + ψ))
                    for ψ in (0.0, 0.3, -0.4, 0.7)
            ]   # rotations about the optical axis only
            @test_throws ArgumentError zhang(flat; center = _c)
        end

        @testset "input validation" begin
            @test_throws ArgumentError zhang(HomographyMat{Float64}[]; center = _c)
            @test_throws ArgumentError solve(EstimationProblem(LogicalIntrinsics, Hs, (; center = _c, weights = ones(length(Hs)))), Zhang())   # no weights vector
            @test_throws ArgumentError zhang([Attributed(Hs[1], (; weight = -1.0)); Hs[2:end]]; center = _c)
            @test_throws ArgumentError zhang([Attributed(Hs[1], (; weight = Inf)); Hs[2:end]]; center = _c)
        end
    end
end
