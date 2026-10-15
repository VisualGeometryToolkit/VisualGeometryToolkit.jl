@testitem "cheirality" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using StructArrays
    using VisualGeometryCore
    using VisualGeometryCore: test_cheirality, HomographyMat

    @testset "test_cheirality — uniform-sign rule" begin

        # Canonical "board in front of camera" homography for a world y-up board
        # mapped to a y-down image. SMatrix constructor is column-major:
        # col1 = (f/d, 0, 0), col2 = (0, -f/d, 0), col3 = (cx, cy, 1.0). The
        # resulting matrix rows are (f/d, 0, cx), (0, -f/d, cy), (0, 0, 1).
        #
        #   image_u = (f/d) · x_board + cx
        #   image_v = -(f/d) · y_board + cy
        #   det(H)  = -(f/d)² < 0  (2D orientation-reversing by construction)
        f, d = 1000.0, 500.0
        cx, cy = 800.0, 600.0
        H_mat = SMatrix{3, 3, Float64, 9}(
            f / d, 0.0, 0.0,
            0.0, -f / d, 0.0,
            cx, cy, 1.0
        )

        # 2D sample correspondences for a world y-up board ↔ y-down image.
        board_pts = [
            SA[-20.0, -15.0], SA[25.0, -18.0],
            SA[18.0, 22.0], SA[-22.0, 17.0],
        ]
        image_pts = [
            let w = H_mat * SA[p[1], p[2], 1.0]
                PerspectiveMap()(w)
            end
                for p in board_pts
        ]
        H = HomographyMat{Float64}(Tuple(H_mat))

        # Wrap into the cspond StructArray layout test_cheirality expects.
        cs = StructArray((first = board_pts, second = image_pts))

        @testset "object→image (det H < 0) succeeds" begin
            @test det(SMatrix{3, 3, Float64, 9}(H)) < 0
            @test test_cheirality(H, cs, 1:4)
            # Invariant under overall sign flip of H:
            H_flipped = HomographyMat{Float64}(Tuple(-SMatrix{3, 3, Float64, 9}(H)))
            @test test_cheirality(H_flipped, cs, 1:4)
        end

        @testset "image→image (det H > 0) succeeds" begin
            # Pure similarity: H = diag(s, s, 1), det > 0.
            s = 1.5
            H_sim = HomographyMat{Float64}(
                (
                    s, 0.0, 0.0,
                    0.0, s, 0.0,
                    10.0, 20.0, 1.0,
                )
            )
            dst = [
                let w = SMatrix{3, 3, Float64, 9}(H_sim) * SA[p[1], p[2], 1.0]
                    PerspectiveMap()(w)
                end
                    for p in board_pts
            ]
            cs_sim = StructArray((first = board_pts, second = dst))
            @test det(SMatrix{3, 3, Float64, 9}(H_sim)) > 0
            @test test_cheirality(H_sim, cs_sim, 1:4)
            # Invariant under overall sign flip of H:
            H_sim_flipped = HomographyMat{Float64}(Tuple(-SMatrix{3, 3, Float64, 9}(H_sim)))
            @test test_cheirality(H_sim_flipped, cs_sim, 1:4)
        end

        @testset "mixed-sign λ (bow-tie in depth) fails" begin
            # Construct a contrived H where the projective depth h₃·[u;1] crosses
            # zero within the span of the sample points — some points map to
            # w > 0, others to w < 0, producing mixed sign λᵢ that no H-scalar
            # flip can reconcile.
            H_bad = HomographyMat{Float64}(
                (
                    1.0, 0.0, 1.0,    # h₃₁ = 1  (grows with x)
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0,
                )
            )  # h₃₃ = 0 → sign flips at x = 0
            @test !test_cheirality(H_bad, cs, 1:4)
        end

        @testset "mask form (consensus-set check)" begin
            # The mask-form method is what RANSAC's `test_consensus` calls — it
            # walks the inlier BitVector in place, no `findall` allocation.
            # Same H_bad as above; verify mask form reaches the same verdict
            # as the indexed form.
            H_bad = HomographyMat{Float64}(
                (
                    1.0, 0.0, 1.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0,
                )
            )
            @test !test_cheirality(H_bad, cs, trues(4))                 # all 4 inliers
            @test !test_cheirality(H_bad, cs, BitVector([true, true, true, false]))

            # A BitVector that picks only same-sign points (the first two have
            # x < 0 → w < 0; both negative → no straddle) → cheirality holds.
            @test test_cheirality(H_bad, cs, BitVector([true, false, false, true]))

            # Good H succeeds regardless of which inliers we pick.
            @test test_cheirality(H, cs, trues(4))
            @test test_cheirality(H, cs, BitVector([true, false, true, false]))
        end

        @testset "HomographyMap entry point" begin
            # The HomographyMap-typed entry is the idiomatic form; should agree
            # with the HomographyMat form across both index and mask selectors.
            hm = HomographyMap(H)
            @test test_cheirality(hm, cs, 1:4)
            @test test_cheirality(hm, cs, trues(4))

            H_bad = HomographyMat{Float64}(
                (
                    1.0, 0.0, 1.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0,
                )
            )
            @test !test_cheirality(HomographyMap(H_bad), cs, 1:4)
            @test !test_cheirality(HomographyMap(H_bad), cs, trues(4))
        end
    end
end
