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

@testitem "linear image: the sRGB encode is IEC 61966-2-1's, the decode's inverse" begin
    using VisualGeometryToolkitFeatures: srgb_to_linear, linear_to_srgb
    # the standard's piecewise OETF, exactly at its breakpoint and on each segment
    @test linear_to_srgb(0.0) == 0.0
    @test linear_to_srgb(0.0031308) == 12.92 * 0.0031308                  # the linear segment, through its end
    @test linear_to_srgb(nextfloat(0.0031308)) == 1.055 * nextfloat(0.0031308)^(1 / 2.4) - 0.055
    @test linear_to_srgb(0.5) == 1.055 * 0.5^(1 / 2.4) - 0.055
    @test abs(linear_to_srgb(nextfloat(0.0031308)) - linear_to_srgb(0.0031308)) < 1.0e-7   # the segments meet
    @test linear_to_srgb(1.0) ≈ 1.0 atol = 1.0e-15
    # the decode's breakpoint is the encode's image of it, to the standard's rounded constants (0.04045 / 12.92
    # is 0.0031308049…, just past the encode's 0.0031308, so it is encoded on the power segment)
    @test srgb_to_linear(0.04045) > 0.0031308
    @test linear_to_srgb(srgb_to_linear(0.04045)) ≈ 0.04045 rtol = 1.0e-6
    # OETF ∘ EOTF and EOTF ∘ OETF round-trip on both segments
    x = range(0, 1; length = 2001)
    @test maximum(abs.(linear_to_srgb.(srgb_to_linear.(x)) .- x)) < 1.0e-12
    @test maximum(abs.(srgb_to_linear.(linear_to_srgb.(x)) .- x)) < 1.0e-12
    # the value's precision is kept: a Float32 image encodes in Float32
    @test linear_to_srgb(0.25f0) isa Float32 && linear_to_srgb(0.001f0) == 12.92f0 * 0.001f0
end

@testitem "linear image: srgb_image, the one display encoding" begin
    using VisualGeometryToolkitFeatures: srgb_image, linear_to_srgb
    using Colors: RGB, Gray, red, green, blue, gray
    using FixedPointNumbers: N0f8, N0f16
    img = zeros(2, 3, 3)
    img[1, 1, :] .= (0.1, 0.2, 0.3)
    img[1, 2, :] .= (2.0, -0.5, 0.05)            # over and under the display range: clamped
    img[2, 1, 2] = NaN                           # no value: the background
    img[2, 2, :] .= (0.02, 0.04, 0.08)
    q8(x) = reinterpret(N0f8, round(UInt8, 255 * linear_to_srgb(clamp(x, 0, 1))))
    # exposure, then each plane's white balance, NaN → background, clamp, sRGB, 8 bits
    out = srgb_image(img; exposure = 2.0, white_balance = (0.5, 1.0, 2.0), background = 0.25)
    @test out isa Matrix{RGB{N0f8}} && size(out) == (2, 3)
    @test (red(out[1, 1]), green(out[1, 1]), blue(out[1, 1])) == (q8(0.1), q8(0.4), q8(1.2))
    @test (red(out[1, 2]), green(out[1, 2]), blue(out[1, 2])) == (q8(2.0), q8(-1.0), q8(0.2))
    @test green(out[2, 1]) == q8(0.25) && red(out[2, 1]) == q8(0.0)
    # the defaults: exposure 1, no white balance, black background, no tone curve
    d = srgb_image(img)
    @test (red(d[2, 2]), green(d[2, 2]), blue(d[2, 2])) == (q8(0.02), q8(0.04), q8(0.08)) && green(d[2, 1]) == q8(0.0)
    # 16 bits on request
    h = srgb_image(N0f16, img)
    @test h isa Matrix{RGB{N0f16}} && red(h[1, 1]) == reinterpret(N0f16, round(UInt16, 65535 * linear_to_srgb(0.1)))
    # a tone curve, applied to the exposed, white-balanced value before the clamp
    t = srgb_image(img; exposure = 2.0, tone = x -> x / (1 + x))
    @test red(t[1, 2]) == q8(4.0 / 5.0) && green(t[1, 1]) == q8(0.4 / 1.4)
    # one plane (a matrix or an H × W × 1 image) is gray
    g = srgb_image(img[:, :, 2]; exposure = 2.0)
    @test g isa Matrix{Gray{N0f8}} && gray(g[1, 1]) == q8(0.4)
    @test srgb_image(img[:, :, 2:2]; exposure = 2.0) == g
    # the white balance has a factor per plane; the exposure and the factors are positive and finite
    @test_throws ArgumentError srgb_image(img; white_balance = (1.0, 1.0))
    @test_throws ArgumentError srgb_image(img; exposure = 0)
    @test_throws ArgumentError srgb_image(img; white_balance = (1.0, NaN, 1.0))
    @test_throws ArgumentError srgb_image(zeros(2, 2, 2))
    # GPU first: a device image encodes on the device
    cuda = try
        mod = Base.require(Main, :CUDA)
        (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing      # no functional GPU here: recorded as Broken, not as a success
    else
        dimg = Base.invokelatest(getproperty(cuda, :CuArray), img)
        dout = Base.invokelatest(srgb_image, dimg; exposure = 2.0, white_balance = (0.5, 1.0, 2.0), background = 0.25)
        @test !(dout isa Array) && Base.invokelatest(Array, dout) == out
    end
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
    # 2-D, and the element type's own float: an integer image averages in Float64, a Float32 one in Float32
    @test box_downsample(A[:, :, 1], 2) == box_downsample(A, 2)[:, :, 1]
    @test box_downsample(Int.(A), 2) == box_downsample(A, 2) && eltype(box_downsample(Float32.(A), 2)) === Float32
    @test_throws ArgumentError box_downsample(zeros(2, 2, 2, 2), 1)
end

@testitem "linear image: the box downsample of a Gray{Float32} patch is BlobBoards' supersampling filter, bit for bit" begin
    using VisualGeometryToolkitFeatures: box_downsample
    using Colors: Gray
    using Random: MersenneTwister
    # BlobBoards' `_box_downsample` (canonicalize.jl, ≤ 0.13): a Float32 accumulation of the S × S block, scaled by 1/S²
    function bb_box_downsample(hi::AbstractMatrix, S::Int)
        P = size(hi, 1) ÷ S
        out = Matrix{Gray{Float32}}(undef, P, P)
        invn = 1.0f0 / (S * S)
        for j in 1:P, i in 1:P
            acc = 0.0f0
            for jj in 1:S, ii in 1:S
                acc += Float32(hi[(i - 1) * S + ii, (j - 1) * S + jj])
            end
            out[i, j] = Gray{Float32}(acc * invn)
        end
        return out
    end
    for S in (2, 3, 4)
        hi = Gray{Float32}.(rand(MersenneTwister(S), Float32, 16S, 16S))
        lo = box_downsample(hi, S)
        @test lo isa Matrix{Gray{Float32}} && reinterpret(UInt32, Float32.(lo)) == reinterpret(UInt32, Float32.(bb_box_downsample(hi, S)))
    end
end

@testitem "gaussian_blur: unit mass, the separable Gaussian, the masked (normalized) convolution" begin
    using Test
    using VisualGeometryToolkitFeatures: gaussian_blur
    # a constant image stays constant everywhere (the border renormalization); σ = 0 is a copy
    A = fill(0.7, 30, 40, 2)
    @test all(isapprox.(gaussian_blur(A, 2.5), 0.7; rtol = 1.0e-14))
    @test gaussian_blur(A, 0) == A
    # an impulse far from the border spreads to the normalized Gaussian, separable
    B = zeros(41, 41, 1); B[21, 21, 1] = 1
    σ = 2.0; G = gaussian_blur(B, σ)[:, :, 1]
    g = [exp(-t^2 / (2σ^2)) for t in -20:20]; g[abs.(-20:20) .> 6] .= 0; g ./= sum(g)
    @test G ≈ g * g' rtol = 1.0e-12
    @test sum(G) ≈ 1 rtol = 1.0e-12
    # masked: the outside never bleeds in; zero outside the mask; a constant inside stays constant
    C = fill(1.0, 20, 20, 1); C[:, 11:end, 1] .= 100.0
    m = zeros(20, 20); m[:, 1:10] .= 1
    D = gaussian_blur(C, 3.0, m)
    @test all(isapprox.(D[:, 1:10, 1], 1.0; rtol = 1.0e-14))
    @test all(iszero, D[:, 11:end, 1])
    @test_throws DimensionMismatch gaussian_blur(C, 1.0, zeros(3, 3))
    @test_throws ArgumentError gaussian_blur(C, -1.0)
end

@testitem "the one Gaussian taps builder: each user's truncation, bit for bit" begin
    # gaussian_blur (3σ, renormalized per pixel at the border) and ASIFT's pyramid
    # (4σ, at least one tap) and σ_aa pre-blur (4σ, at most 16 taps), unit sum,
    # build their taps with `_gaussian_profile`; these are the formulas they had.
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using Test
    function unit(σ::Float32, r)
        k = [i == 0 ? 1.0f0 : exp(-Float32(i)^2 / (2.0f0 * σ^2)) for i in 0:r]
        return k ./= (k[1] + 2.0f0 * sum(@view k[2:end]))
    end
    for s in (0.0, 0.3, 0.8, 1.2263, 1.6, 2.5, 3.9, 4.01, 7.7)
        σ = Float32(s)
        @test VgtkFeatures.gaussian_taps(s) == unit(σ, max(1, ceil(Int, 4.0f0 * σ)))
        @test VgtkFeatures.sigma_aa_taps(s) == unit(σ, min(ceil(Int, 4.0f0 * σ), 16))
    end
    for σ in (0.5, 1, 1.7, 3 // 2, 6.25)
        r = ceil(Int, 3σ)
        @test [reverse(VgtkFeatures._gaussian_profile(σ, 3)[2:end]); VgtkFeatures._gaussian_profile(σ, 3)] ==
              [exp(-t^2 / (2σ^2)) for t in (-r):r]
    end
end
