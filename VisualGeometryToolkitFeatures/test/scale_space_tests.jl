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
    # The tilted detection's bits depend on the CPU's floating-point evaluation (its vector instruction set), so its
    # pins are per CPU family, a (features, covariances) pair each; a run reproduces ONE pair:
    #   pair 1  recorded at 654a136f, before the factoring, on the machine the factoring was made on; reproduced by the
    #           cluster's nodes poseidon and hephaestus;
    #   pair 2  the cluster's node nereus and GitHub's hosted runners (the same bits at Features 0.9.8 and 0.9.9).
    # A machine that matches neither pair is a third family, or a change of the detector: the failure below prints its
    # two hashes — check on a machine of pair 1 or 2 that the detector is unchanged there, then add the pair with its
    # machine. The untilted detection and the pyramid's levels below have one pin, the same bits on all of these machines.
    feature_pins = ("19bee0b8eed210064cb55e46f1a7f2d0a9c3039deca0cecd590cf0c228978f02",
                    "0893189cf2f9021937394a7926014bf6cd3da9e257de0f38dc038d35f5e3aec6")
    covariance_pins = ("72cd90e0d3fde944580677448c4e1b6ed9bbe1e798124cb787f0bf15768463eb",
                       "6bd868d734e131a50e048629e419f0565890961e1fe256202bf172d2439ff3b7")
    machine = findfirst(==(feat(g1)), feature_pins)
    machine === nothing && @error "asift_detect on this CPU matches no recorded pair (see the comment above)" features = feat(g1)
    @test machine !== nothing
    g2 = asift_detect(img; device = KA.CPU(), raw = true, tilts = :untilted, nms = :none)
    @test length(g2) == 70
    @test feat(g2) == "c96560f1da83c94cb89ecfda9ebedb81e1ec9a4b3dd4921b0ff1cfe793783da7"
    g3 = asift_detect(img; device = KA.CPU(), raw = true, uncertainty = FeatureCovariance())
    @test feat(g3.features) == feat(g1)
    covariances = bits_hash([Float32(x) for c in g3.cov for x in Matrix(c)])
    machine === nothing && @error "… and its covariances' hash" covariances
    @test machine !== nothing && covariances == covariance_pins[machine]
    plan = AsiftPlan(200, 150; device = KA.CPU(), tilts = :untilted)
    asift_detect!(plan, permutedims(img))
    @test bits_hash(reduce(vcat, [vec(Array(g)) for g in plan.space.levels])) ==
        "5d55e800f4ad4720a60cbfc3dc4913d30f128ec0d1fd777060ce9bf35d906688"
end

@testitem "asift_detect: upsample, nb_scales and nb_octaves at their defaults are the detector's bits before them (CPU)" setup = [ScaleSpaceFixture] begin
    import KernelAbstractions as KA
    import VisualGeometryToolkitFeatures as VgtkFeatures
    # `upsample` was a Bool and `nb_scales`/`nb_octaves` were not keywords of asift_detect (Features ≤ 0.9.11's
    # pre-release main, c4fc2168): spelled out at their defaults, and `upsample` as the Bool, they are the pins above,
    # recorded before them — the untilted detection's and the pyramid's on every machine, the tilted one's per family
    img = golden_img(150, 200)
    feat(fs) = bits_hash([Float32(getfield(f, k)) for f in fs for k in fieldnames(typeof(f))])
    ref = asift_detect(img; device = KA.CPU(), raw = true)
    for kw in ((; upsample = 1, nb_scales = 3, nb_octaves = VgtkFeatures.sift_nb_octaves(200, 150; upsample = 1)),
               (; upsample = true), (; upsample = 1))
        @test feat(asift_detect(img; device = KA.CPU(), raw = true, tilts = :untilted, nms = :none, kw...)) ==
            "c96560f1da83c94cb89ecfda9ebedb81e1ec9a4b3dd4921b0ff1cfe793783da7"
        @test feat(asift_detect(img; device = KA.CPU(), raw = true, kw...)) == feat(ref)
    end
    plan = AsiftPlan(200, 150; device = KA.CPU(), tilts = :untilted, upsample = 1, nb_scales = 3)
    asift_detect!(plan, permutedims(img))
    @test bits_hash(reduce(vcat, [vec(Array(g)) for g in plan.space.levels])) ==
        "5d55e800f4ad4720a60cbfc3dc4913d30f128ec0d1fd777060ce9bf35d906688"
    # one plan for `true` and 1
    clear_asift_plans!()
    asift_detect(img; device = KA.CPU(), tilts = :untilted, upsample = true)
    asift_detect(img; device = KA.CPU(), tilts = :untilted, upsample = 1, nb_scales = 3)
    @test length(VgtkFeatures._ASIFT_PLAN_CACHE) == 1
    clear_asift_plans!()
end

@testitem "asift_detect: upsample a count, nb_scales and nb_octaves pick the pyramid" setup = [ScaleSpaceFixture] begin
    import KernelAbstractions as KA
    import VisualGeometryToolkitFeatures as VgtkFeatures
    img = golden_img(150, 200)
    for (up, S) in ((2, 3), (2, 5), (3, 3))
        plan = AsiftPlan(200, 150; device = KA.CPU(), tilts = :untilted, upsample = up, nb_scales = S)
        @test plan.space.first_octave == -up && plan.space.nb_scales == S
        @test plan.space.dims[1] == (200 << up, 150 << up)
        # the same coarsest octave whatever the upsampling
        @test length(plan.space.dims) == VgtkFeatures.sift_nb_octaves(200, 150; upsample = 1) + up - 1
        @test last(plan.space.dims) == (50, 37)
    end
    # the searched scales halve with each upsampling: a refined σ is no finer than the first DoG slice's,
    # seed_sigma·2^(0.5/S)·2^(−upsample)
    for (up, S) in ((1, 3), (2, 3), (2, 5))
        f = asift_detect(img; device = KA.CPU(), raw = true, tilts = :untilted, upsample = up, nb_scales = S)
        @test !isempty(f) && minimum(x -> x.σ, f) >= 1.6 * exp2(0.5 / S - up) - 1.0e-4
    end
    # an octave more than the default, at most
    no = VgtkFeatures.sift_nb_octaves(200, 150; upsample = 1)
    @test length(AsiftPlan(200, 150; device = KA.CPU(), tilts = :untilted, nb_octaves = no + 1).space.dims) == no + 1
    @test_throws ArgumentError AsiftPlan(200, 150; device = KA.CPU(), tilts = :untilted, nb_octaves = no + 2)
    @test_throws ArgumentError asift_detect(img; device = KA.CPU(), tilts = :untilted, nb_octaves = 0)
    @test_throws ArgumentError AsiftPlan(200, 150; device = KA.CPU(), upsample = -1)
    @test_throws ArgumentError AsiftPlan(200, 150; device = KA.CPU(), nb_scales = 0)
    @test_throws ArgumentError AsiftPlan(200, 150; device = KA.CPU(), upsample = 5)
    # a seed σ below the input's blur in the first octave's pixels: refused at one upsampling (1.0 there), the
    # upsampled image itself as the first level at two (2.0 there)
    @test_throws ArgumentError AsiftPlan(200, 150; device = KA.CPU(), tilts = :untilted, seed_sigma = 0.9)
    @test AsiftPlan(200, 150; device = KA.CPU(), tilts = :untilted, upsample = 2).space.taps[1] |> Array == [1.0f0, 0.0f0]
    # a refused nb_scales leaves the cached plan resident
    clear_asift_plans!()
    asift_detect(img; device = KA.CPU(), tilts = :untilted)
    @test_throws ArgumentError asift_detect(img; device = KA.CPU(), tilts = :untilted, nb_scales = 99)
    @test length(VgtkFeatures._ASIFT_PLAN_CACHE) == 1
    # the covariances and the continuous refinement on a twice upsampled pyramid
    r = asift_detect(img; device = KA.CPU(), raw = true, upsample = 2, uncertainty = FeatureCovariance(),
                     refinement = ContinuousRefinement())
    @test !isempty(r.features) && all(c -> all(isfinite, Matrix(c)), r.cov)
    clear_asift_plans!()
end

@testitem "gaussian_scale_space: the untilted warp's levels, bit for bit; level_sigma" setup = [ScaleSpaceFixture] begin
    import KernelAbstractions as KA
    import VisualGeometryToolkitFeatures as VgtkFeatures
    img = golden_img(150, 200)
    for upsample in (true, false, 2)
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
    @test VgtkFeatures.level_sigma(gaussian_scale_space(img; device = KA.CPU(), upsample = 2), 1, 1) ≈ 0.4   # octave −2
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
