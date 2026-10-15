# zhang_intrinsics — Zhang IAC regression from plane homographies (calibration.jl)
@testitem "zhang_intrinsics" begin
    using Test
    using StaticArrays: SVector, SMatrix, @SMatrix
    using Rotations: RotXYZ
    using VisualGeometryCore: HomographyMat, LogicalIntrinsics, Size2,
        zhang_intrinsics, focal_length, μm, px

    _hom(K, R, t) = HomographyMat{Float64}(
        Tuple(SMatrix{3, 3, Float64}(K * hcat(R[:, 1], R[:, 2], t)))
    )

    _c = SVector(640.0, 480.0)
    function _synth_views(fx, fy)
        K = @SMatrix [fx 0.0 _c[1]; 0.0 fy _c[2]; 0.0 0.0 1.0]      # raw pixels, principal point _c
        angles = [
            (0.4, -0.3, 0.2), (-0.5, 0.2, -0.1), (0.2, 0.5, 0.3),
            (-0.3, -0.4, 0.1), (0.6, 0.1, -0.2), (0.1, -0.6, 0.4),
        ]
        ts = [
            SVector(0.3, -0.2, 2.5), SVector(-0.4, 0.1, 3.0), SVector(0.2, 0.3, 2.2),
            SVector(-0.1, -0.3, 2.8), SVector(0.5, 0.2, 3.5), SVector(0.0, 0.4, 2.0),
        ]
        return [_hom(K, RotXYZ(a...), t) for (a, t) in zip(angles, ts)]
    end

    @testset "zhang_intrinsics" begin
        fx, fy = 1226.3, 1219.7
        Hs = _synth_views(fx, fy)

        @testset "free fx, fy recovery" begin
            intr = zhang_intrinsics(Hs; center = _c)
            @test SMatrix{3, 3}(intr.K)[1:2, 3] == _c                     # pixels in, pixels out
            @test_throws UndefKeywordError zhang_intrinsics(Hs)
            f̂x, f̂y = focal_length(intr)
            @test f̂x ≈ fx atol = 1.0e-6
            @test f̂y ≈ fy atol = 1.0e-6
            @test intr isa LogicalIntrinsics{Float64}
        end

        @testset "weights are per-homography vote strengths" begin
            # a garbage homography with zero weight must not perturb the solve
            junk = _hom(
                (@SMatrix [500.0 0 0; 0 900.0 0; 0 0 1.0]),
                RotXYZ(0.3, 0.2, 0.1), SVector(0.1, 0.1, 2.0)
            )
            intr = zhang_intrinsics(vcat(Hs, [junk]); center = _c, weights = vcat(fill(10.0, length(Hs)), 0.0))
            f̂x, f̂y = focal_length(intr)
            @test f̂x ≈ fx atol = 1.0e-6
            @test f̂y ≈ fy atol = 1.0e-6
        end

        @testset "known pitch → single f (dispatch on Size2)" begin
            pitch = Size2(width = 3.9μm / px, height = 4.0μm / px)
            τ = 3.9 / 4.0                      # fy/fx = pitch.width/pitch.height
            f0 = 1300.0
            Hs_τ = _synth_views(f0, τ * f0)
            intr = zhang_intrinsics(Hs_τ, pitch; center = _c)
            @test SMatrix{3, 3}(intr.K)[1:2, 3] == _c
            f̂x, f̂y = focal_length(intr)
            @test f̂x ≈ f0 atol = 1.0e-6
            @test f̂y / f̂x ≈ τ atol = 1.0e-12    # aspect is imposed, not estimated
        end

        @testset "degenerate: fronto-parallel views throw" begin
            K = @SMatrix [fx 0.0 _c[1]; 0.0 fy _c[2]; 0.0 0.0 1.0]
            flat = [
                _hom(K, RotXYZ(0.0, 0.0, ψ), SVector(0.1ψ, -0.2ψ, 2.0 + ψ))
                    for ψ in (0.0, 0.3, -0.4, 0.7)
            ]   # rotations about the optical axis only
            @test_throws ArgumentError zhang_intrinsics(flat; center = _c)
        end

        @testset "input validation" begin
            @test_throws ArgumentError zhang_intrinsics(HomographyMat{Float64}[]; center = _c)
            @test_throws ArgumentError zhang_intrinsics(Hs; center = _c, weights = [1.0])
            @test_throws ArgumentError zhang_intrinsics(Hs; center = _c, weights = [-1.0; ones(length(Hs) - 1)])
            @test_throws ArgumentError zhang_intrinsics(Hs; center = _c, weights = [Inf; ones(length(Hs) - 1)])
        end
    end
end
