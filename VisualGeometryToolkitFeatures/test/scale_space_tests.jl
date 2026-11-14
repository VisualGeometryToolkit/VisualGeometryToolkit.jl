# gaussian_scale_space: ASIFT's pyramid factored out. ASIFT itself is pinned bit for bit across the factoring (the
# hashes below were recorded on the CPU at 654a136f, before it), and the standalone scale space is level for level the
# untilted warp's.

@testsetup module ScaleSpaceFixture
using Random: Xoshiro, rand, randn
using SHA: sha256
export golden_img, bits_hash

# Smooth texture, 40 Gaussian blobs of both polarities and noise: every octave holds extrema.
function golden_img(H, W)
    rng = Xoshiro(2026)
    img = [0.5f0 + 0.3f0 * sinpi(r / 17) * cospi(c / 23) for r in 1:H, c in 1:W]
    for _ in 1:40
        a, b, s, v = rand(rng) * W, rand(rng) * H, 1.5 + 6rand(rng), rand(rng) < 0.5 ? 0.35f0 : -0.35f0
        for r in 1:H, c in 1:W
            img[r, c] += v * exp(-((c - a)^2 + (r - b)^2) / (2s^2))
        end
    end
    return clamp.(img, 0.0f0, 1.0f0) .+ 0.02f0 .* randn(rng, Float32, H, W)
end

bits_hash(x::AbstractVector{Float32}) = bytes2hex(sha256(reinterpret(UInt8, x)))
end

@testitem "asift_detect: bit-identical across the scale-space factoring (CPU)" setup = [ScaleSpaceFixture] begin
    import KernelAbstractions as KA
    using Statistics: cov
    img = golden_img(150, 200)
    feat(fs) = bits_hash([Float32(getfield(f, k)) for f in fs for k in fieldnames(typeof(f))])
    g1 = asift_detect(img; device = KA.CPU(), raw = true)
    @test length(g1) == 87
    @test feat(g1) == "19bee0b8eed210064cb55e46f1a7f2d0a9c3039deca0cecd590cf0c228978f02"
    g2 = asift_detect(img; device = KA.CPU(), raw = true, tilts = :untilted, nms = :none)
    @test length(g2) == 70
    @test feat(g2) == "c96560f1da83c94cb89ecfda9ebedb81e1ec9a4b3dd4921b0ff1cfe793783da7"
    g3 = asift_detect(img; device = KA.CPU(), raw = true, uncertainty = FeatureCovariance())
    @test feat(g3.features) == feat(g1)
    @test bits_hash([Float32(x) for c in g3.cov for x in Matrix(c)]) ==
        "72cd90e0d3fde944580677448c4e1b6ed9bbe1e798124cb787f0bf15768463eb"
    plan = AsiftPlan(200, 150; device = KA.CPU(), tilts = :untilted)
    asift_detect!(plan, permutedims(img))
    @test bits_hash(reduce(vcat, [vec(Array(g)) for g in plan.space.levels])) ==
        "5d55e800f4ad4720a60cbfc3dc4913d30f128ec0d1fd777060ce9bf35d906688"
end

@testitem "gaussian_scale_space: the untilted warp's levels, bit for bit; level_sigma" setup = [ScaleSpaceFixture] begin
    import KernelAbstractions as KA
    import VisualGeometryToolkitFeatures as VgtkFeatures
    img = golden_img(150, 200)
    for upsample in (true, false)
        plan = AsiftPlan(200, 150; device = KA.CPU(), tilts = :untilted, upsample)
        asift_detect!(plan, permutedims(img))
        space = gaussian_scale_space(img; device = KA.CPU(), upsample)
        @test space isa GaussianScaleSpace
        @test space.dims == plan.space.dims
        @test length(space.levels) == length(plan.space.levels)
        @test all(Array(a) == Array(b) for (a, b) in zip(space.levels, plan.space.levels))
        # fewer octaves: the same first octaves
        few = gaussian_scale_space(img; device = KA.CPU(), upsample, nb_octaves = 2)
        @test length(few.levels) == 2 && all(Array(few.levels[o]) == Array(space.levels[o]) for o in 1:2)
    end
    space = gaussian_scale_space(img; device = KA.CPU())
    @test VgtkFeatures.level_sigma(space, 1, 1) ≈ 0.8          # octave −1: 1.6 upsampled px
    @test VgtkFeatures.level_sigma(space, 2, 4) ≈ 3.2          # octave 0, level S + 1: 2·1.6
    @test_throws ArgumentError gaussian_scale_space(img; nb_octaves = 99)
end

@testitem "gaussian_scale_space: CUDA matches the CPU" setup = [ScaleSpaceFixture] begin
    import KernelAbstractions as KA
    using CUDA
    if CUDA.functional()
        img = golden_img(150, 200)
        cpu = gaussian_scale_space(img; device = KA.CPU())
        gpu = gaussian_scale_space(img; device = CUDA.CUDABackend())
        @test all(isapprox(Array(a), Array(b); rtol = 1.0e-5, atol = 1.0e-6) for (a, b) in zip(cpu.levels, gpu.levels))
    else
        @test_skip "no functional CUDA device"
    end
end
