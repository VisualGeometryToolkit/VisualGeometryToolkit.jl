# LoMa's model-free halves (src/loma.jl): the strongest-detection selection and the descriptor graph's input. The
# networks themselves are downloaded on first use and are not run here.

@testitem "strongest: the n largest |response|, in detector order, ties to the earlier" begin
    using VisualGeometryToolkitFeatures: strongest, Detection
    using VisualGeometryToolkitCore: Disc, Point2
    det(r, k) = Detection(Disc(Point2(Float64(k), 0.0), 1.0), (response = r,))
    rs = [0.1, -0.5, 0.3, 0.5, -0.05, 0.3, 0.2]
    ds = [det(r, k) for (k, r) in enumerate(rs)]
    @test getproperty.(strongest(ds, 3), :response) == [-0.5, 0.3, 0.5]           # signed: |−0.5| ties 0.5; detector order
    @test getproperty.(strongest(ds, 4), :response) == [-0.5, 0.3, 0.5, 0.3]      # the tie at 0.3: both, in order
    @test strongest(ds, 7) === ds && strongest(ds, 100) === ds && isempty(strongest(ds, 0))
    @test_throws ArgumentError strongest(ds, -1)
    # PanoramaStitching's copy (features.jl ≤ 0.6): a stable sort by |response|, the first n, back in detector order
    ps(d, n) = d[sort!(sortperm(abs.(getproperty.(d, :response)); rev = true)[1:min(end, n)])]
    for n in 0:7
        @test strongest(ds, n) == ps(ds, n)
    end
end

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
    @test_throws ArgumentError VgtkFeatures._loma_input(rand(Float32, 1, n, n))
    # the keypoint chart reads the image's own size: a detection needs no rescaling
    using VisualGeometryToolkitFeatures: Detection
    using VisualGeometryToolkitCore: Disc, Point2
    d = [Detection(Disc(Point2(750.25, 333.5), 2.0), (response = 1.0,))]
    sx, sy = n / Wf, n / Hf
    ds = [Detection(Disc(Point2(sx * 750.25, sy * 333.5), 2.0), (response = 1.0,))]
    @test VgtkFeatures._loma_keypoints(d, (3, Hf, Wf)) ≈ VgtkFeatures._loma_keypoints(ds, (3, n, n))
end
