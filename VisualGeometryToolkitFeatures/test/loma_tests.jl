# LoMa's model-free half (src/loma.jl): the descriptor graph's input (the detection selection is spread_tests.jl's). The
# networks themselves are downloaded on first use and are not run here.

@testitem "LoMaDescriptor's input: an image of any size, resampled to the graph's 784² as PanoramaStitching did" begin
    using VisualGeometryToolkitFeatures: LOMA_IMAGE_SIZE
    using VisualGeometryToolkitCore: image_warp, image_warp!, AffineMap, Rect
    using StaticArrays: SMatrix, SVector
    using Random: MersenneTwister
    import VisualGeometryToolkitFeatures as VgtkFeatures
    n = LOMA_IMAGE_SIZE
    @test n == 784
    # at the graph's size, the image as it is (the Float32 channelview itself)
    img = rand(MersenneTwister(1), Float32, 3, n, n)
    @test VgtkFeatures._loma_input(img) === img
    @test VgtkFeatures._loma_input(Float64.(img)) == img
    # another size: each channel by the COLMAP scaling of the network's pixels onto the image's, Float64 math,
    # Float32 taps — PanoramaStitching's `detect_features` (≤ 0.6), on an H × W × 3 frame, bit for bit
    Hf, Wf = 1000, 1500
    toned = rand(MersenneTwister(2), Hf, Wf, 3)
    bm = image_warp(AffineMap(SMatrix{2, 2, Float64, 4}(Wf / n, 0.0, 0.0, Hf / n), SVector(0.0, 0.0)), Rect(0, 0, n, n))
    ref = Array{Float32}(undef, 3, n, n); plane = Matrix{Float32}(undef, n, n)
    for c in 1:3
        ref[c, :, :] .= image_warp!(plane, bm, view(toned, :, :, c))
    end
    got = VgtkFeatures._loma_input(PermutedDimsArray(toned, (3, 1, 2)))
    @test size(got) == (3, n, n) && reinterpret(UInt32, got) == reinterpret(UInt32, ref)
    # an upscale too, and a channel count other than 3 is refused
    @test size(VgtkFeatures._loma_input(rand(Float32, 3, 100, 120))) == (3, n, n)
    # an upscale's border: a 392² image (x ↦ x/2) puts the network's last pixel centre 783.5 at 391.75, past the
    # image's last centre 391.5; the tap clamps to the border pixel (replicate), never blends in the fill
    small = rand(MersenneTwister(3), Float32, 3, 392, 392) .+ 0.5f0          # in [0.5, 1.5): a zero fill would show
    up = VgtkFeatures._loma_input(small)
    for c in 1:3
        @test up[c, end, end] ≈ small[c, end, end] rtol = 1.0f-6             # the corner is the corner pixel
        @test up[c, 1, 1] ≈ small[c, 1, 1] rtol = 1.0f-6
        @test up[c, end, 1] ≈ small[c, end, 1] rtol = 1.0f-6
        @test up[c, 1, end] ≈ small[c, 1, end] rtol = 1.0f-6
    end
    @test minimum(up) >= 0.5f0 - 1.0f-6                                       # no fill anywhere
    # the last row is the image's last row, resampled along it only (the vertical tap clamped)
    row = VgtkFeatures._loma_input(repeat(small[:, end:end, :]; outer = (1, 392, 1)))
    @test up[:, end, :] ≈ row[:, end, :] rtol = 1.0f-6
    # and the interior is the unpadded bilinear warp's, unchanged
    bm2 = image_warp(AffineMap(SMatrix{2, 2, Float64, 4}(392 / n, 0.0, 0.0, 392 / n), SVector(0.0, 0.0)), Rect(0, 0, n, n))
    inner = image_warp!(Matrix{Float32}(undef, n, n), bm2, small[1, :, :])
    @test up[1, 2:(n - 1), 2:(n - 1)] ≈ inner[2:(n - 1), 2:(n - 1)] rtol = 1.0f-5
    @test minimum(inner[end, :]) < 0.5f0                                       # the old path did blend the fill there
    @test_throws ArgumentError VgtkFeatures._loma_input(rand(Float32, 1, n, n))
    # the keypoint chart reads the image's own size: a detection needs no rescaling
    using VisualGeometryToolkitFeatures: Detection
    using VisualGeometryToolkitCore: Disc, Point2
    d = [Detection(Disc(Point2(750.25, 333.5), 2.0), (response = 1.0,))]
    sx, sy = n / Wf, n / Hf
    ds = [Detection(Disc(Point2(sx * 750.25, sy * 333.5), 2.0), (response = 1.0,))]
    @test VgtkFeatures._loma_keypoints(d, (3, Hf, Wf)) ≈ VgtkFeatures._loma_keypoints(ds, (3, n, n))
end
