# load_linear_image: a PNG as linear radiance — the stored samples (no gamma
# re-encoding), the sRGB decode against the standard formula, the box
# downsample against hand-computed blocks.

@testitem "linear image: a PNG read as its stored samples" begin
    using VisualGeometryToolkitFeatures: load_linear_image
    using PNGFiles: PNGFiles
    using Colors: RGB, Gray
    using FixedPointNumbers: N0f8, N0f16
    mktempdir() do dir
        # 8-bit RGB, saved WITH a gAMA chunk (file_gamma = 1, as BlobBoards writes
        # its boards): PNGFiles' default load re-encodes it, ours must not.
        codes = [UInt8(17 * (i + 3j)) % UInt8 for i in 1:3, j in 1:4]
        img = [RGB{N0f8}(reinterpret(N0f8, codes[i, j]), reinterpret(N0f8, UInt8(255) - codes[i, j]), N0f8(0.5)) for i in 1:3, j in 1:4]
        p = joinpath(dir, "rgb8.png")
        PNGFiles.save(p, img; file_gamma = 1.0)
        x = load_linear_image(p)
        @test x isa Array{Float64, 3} && size(x) == (3, 4, 3)
        @test x[:, :, 1] == codes ./ 255
        @test x[:, :, 2] == (255 .- codes) ./ 255
        @test all(==(Float64(N0f8(0.5))), x[:, :, 3])
        default = PNGFiles.load(p)                       # the trap this reader exists for
        @test Float64.(getfield.(default, :r)) != x[:, :, 1]

        # 16-bit gray: replicated to three channels, normalized to [0, 1]
        g = [Gray{N0f16}(reinterpret(N0f16, UInt16(1000 * i + 7 * j))) for i in 1:2, j in 1:3]
        q = joinpath(dir, "gray16.png")
        PNGFiles.save(q, g)
        y = load_linear_image(q)
        @test size(y) == (2, 3, 3)
        @test all(y[:, :, c] == [(1000 * i + 7 * j) / 65535 for i in 1:2, j in 1:3] for c in 1:3)

        # the options compose: decode, then average in linear space
        z = load_linear_image(p; srgb = true, downsample = 2)
        @test size(z) == (1, 2, 3)
        lin(c) = (v = c / 255; v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055)^2.4)
        @test z[1, 1, 1] ≈ (lin(codes[1, 1]) + lin(codes[2, 1]) + lin(codes[1, 2]) + lin(codes[2, 2])) / 4 rtol = 1.0e-15
    end
end

@testitem "linear image: the sRGB decode is IEC 61966-2-1's" begin
    using VisualGeometryToolkitFeatures: srgb_to_linear
    # the standard's piecewise formula, and values computed from it independently
    @test srgb_to_linear(0.0) == 0.0
    @test srgb_to_linear(1.0) == 1.0
    @test srgb_to_linear(0.04045) == 0.04045 / 12.92                     # the linear segment, through its end
    @test srgb_to_linear(0.5) ≈ 0.21404114048223255 rtol = 1.0e-15       # ((0.5 + 0.055)/1.055)^2.4
    @test srgb_to_linear(200 / 255) ≈ ((200 / 255 + 0.055) / 1.055)^2.4 rtol = 1.0e-15
    # continuous at the switch (the two segments meet to ~1e-8: the standard's rounded constants)
    @test abs(srgb_to_linear(nextfloat(0.04045)) - srgb_to_linear(0.04045)) < 1.0e-7
    @test issorted(srgb_to_linear.(range(0, 1; length = 1001)))
end

@testitem "linear image: the box downsample" begin
    using VisualGeometryToolkitFeatures: box_downsample
    A = reshape(Float64[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16], 4, 4, 1)
    # hand-computed 2 × 2 block means
    @test box_downsample(A, 2)[:, :, 1] == [(1 + 2 + 5 + 6) / 4 (3 + 4 + 7 + 8) / 4; (9 + 10 + 13 + 14) / 4 (11 + 12 + 15 + 16) / 4]
    # the remainder past the last whole block is dropped
    B = reshape(collect(1.0:25.0), 5, 5, 1)
    @test size(box_downsample(B, 2)) == (2, 2, 1)
    @test box_downsample(B, 2)[1, 1, 1] == (1 + 2 + 6 + 7) / 4
    # per channel, and f = 1 is the identity
    C = cat(A, 2 .* A; dims = 3)
    @test box_downsample(C, 2)[:, :, 2] == 2 .* box_downsample(A, 2)[:, :, 1]
    @test box_downsample(C, 1) == C
    @test_throws ArgumentError box_downsample(C, 0)
end
