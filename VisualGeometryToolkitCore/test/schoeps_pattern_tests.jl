# =============================================================================
# Schöps et al.'s tagged star pattern as a PlanarTarget (geometry/cameras/
# targets.jl): the feature order and positions against camera_calibration's
# FeatureDetectorTaggedPattern (GetCorners, IsValidFeatureCoord) and its known
# geometry, transcribed; the units; the refusals.
# =============================================================================

@testitem "SchoepsPattern: Schöps' feature order and geometry" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: SchoepsPattern, mm, m, ustrip, points, has_units

    # camera_calibration (puzzlepaint/camera_calibration), transcribed:
    #   PatternData::IsValidFeatureCoord(x, y): 0 ≤ x ≤ squares_x − 2, 0 ≤ y ≤ squares_y − 2, and for every tag NOT
    #     (x ≥ tag.x − 1 && y ≥ tag.y − 1 && x ≤ tag.x − 1 + tag.width && y ≤ tag.y − 1 + tag.height)
    #   FeatureDetectorTaggedPattern::GetCorners: for y in 0:squares_y − 2, for x in 0:squares_x − 2, each valid (x, y) the next
    #     feature id (0-based) → feature_id_to_position[id] = (x, y)
    #   calibration.cc / dense_initialization.cc: the pattern point of a feature is cell_length_in_meters · (x, y, 0)
    function reference(sx, sy, tags)
        valid(x, y) = 0 <= x <= sx - 2 && 0 <= y <= sy - 2 &&
            !any(t -> x >= t.x - 1 && y >= t.y - 1 && x <= t.x - 1 + t.width && y <= t.y - 1 + t.height, tags)
        ids = Tuple{Int, Int}[]
        for y in 0:(sy - 2), x in 0:(sx - 2)
            valid(x, y) && push!(ids, (x, y))
        end
        return ids
    end
    # the reference's 3D point of a position, and the design chart's: a half turn about x and a shift of one square
    to_chart(q, sy, s) = SVector(q[1] + s, (sy - 1) * s - q[2])

    @testset "against the reference: $(sx) × $(sy)" for (sx, sy, tags) in (
            (30, 30, [(; x = 13, y = 13, width = 4, height = 4, index = 0)]),          # a 1 m Siemens board: 29² − 5² features
            (25, 36, [(; x = 10, y = 16, width = 4, height = 4, index = 3)]),          # an A4 board: 24 × 35 − 5²
            (6, 5, [(; x = 2, y = 1, width = 2, height = 2, index = 1)]),
            (12, 9, [(; x = 0, y = 0, width = 2, height = 3, index = 0), (; x = 8, y = 5, width = 4, height = 4, index = 1)]),
            (4, 3, NamedTuple[]))
        s = 0.012
        ref = reference(sx, sy, tags)
        t = PlanarTarget(SchoepsPattern(; squares_x = sx, squares_y = sy, square_length = s, apriltags = tags))
        ps = points(t)
        @test length(ps) == length(ref)
        # feature id k is point k + 1, at the reference's point moved into the design chart
        @test all(isapprox(ps[k + 1], to_chart(s .* SVector(ref[k + 1]...), sy, s); atol = 1.0e-15) for k in eachindex(ref) .- 1)
        # and the motion is rigid: every distance is the reference's
        i, j = 1, length(ps)
        @test norm(ps[i] - ps[j]) ≈ s * norm(SVector(ref[i]...) - SVector(ref[j]...))
    end

    @testset "counts and the first points by hand" begin
        siemens = SchoepsPattern(; squares_x = 30, squares_y = 30, square_length = 1.0, apriltags = [(; x = 13, y = 13, width = 4, height = 4)])
        @test length(points(PlanarTarget(siemens))) == 816
        p = SchoepsPattern(; squares_x = 6, squares_y = 5, square_length = 1.0, apriltags = [(; x = 2, y = 1, width = 2, height = 2)])
        ps = points(PlanarTarget(p))
        # 5 × 4 inner corners; gx ∈ 1:3, gy ∈ 0:2 are next to the tag: row gy = 0 keeps gx = 0, 4; the last row all five
        @test length(ps) == 20 - 9
        @test ps[1] == SVector(1.0, 4.0) && ps[2] == SVector(5.0, 4.0) && ps[3] == SVector(1.0, 3.0) && ps[end] == SVector(5.0, 1.0)
    end

    @testset "lengths, stored in metres" begin
        p = SchoepsPattern(; squares_x = 6, squares_y = 5, square_length = 10mm, apriltags = [(; x = 2, y = 1, width = 2, height = 2)])
        t = PlanarTarget(p)
        @test has_units(points(t)[1][1]) && ustrip.(mm, points(t)[1]) ≈ SVector(10.0, 40.0)
        @test p.square_length == 0.01m
    end

    @testset "refusals" begin
        tag = [(; x = 2, y = 1, width = 2, height = 2)]
        @test_throws ArgumentError SchoepsPattern(; squares_x = 1, squares_y = 5, square_length = 1.0, apriltags = tag)
        @test_throws ArgumentError SchoepsPattern(; squares_x = 6, squares_y = 5, square_length = 0.0, apriltags = tag)
        @test_throws ArgumentError SchoepsPattern(; squares_x = 6, squares_y = 5, square_length = 3VisualGeometryToolkitCore.px, apriltags = tag)   # not a length
        @test_throws ArgumentError SchoepsPattern(; squares_x = 3, squares_y = 5, square_length = 1.0, apriltags = tag)   # off the pattern
        @test_throws ArgumentError SchoepsPattern(; squares_x = 6, squares_y = 5, square_length = 1.0, apriltags = [(; x = -1, y = 1, width = 2, height = 2)])
    end
end
