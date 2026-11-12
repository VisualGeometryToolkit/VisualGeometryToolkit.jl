# Public names for private functions a downstream package (RadiometricCamera) copied or reached by private name:
# each is the function its own callers in Core already use, so these pin the public contract, not new arithmetic.

@testitem "normalized_point: K⁻¹𝐩, the pixel-frame half of backproject" begin
    using Test
    using StaticArrays
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: normalized_point, KannalaBrandt, CameraCalibrationMat, convert_eltype
    allocs(f, args...) = (f(args...); @allocated f(args...))

    K = SMatrix{3, 3, Float64}(600, 0, 0, 2.5, 610, 0, 320, 240, 1)       # skew 2.5: K⁻¹ is not two divisions
    pin = CameraModel(CameraCalibrationMat(K))
    fish = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    pixels = [SVector(x, y) for x in 3.25:97.5:1270.0, y in 1.5:83.0:950.0]
    for p in pixels
        x = normalized_point(pin.intrinsics, p)
        @test x isa SVector{2, Float64}
        @test x ≈ (K \ SVector(p[1], p[2], 1.0))[SOneTo(2)] atol = 1.0e-14
        # the one map backproject applies: the lens's bearing at it, bit for bit
        @test isequal(backproject(fish, p), backproject(fish.lens, normalized_point(fish.intrinsics, p)))
        @test isequal(backproject(pin, p), backproject(pin.lens, normalized_point(pin.intrinsics, p)))
    end
    # the element type follows the pixel's (a Float32 device camera)
    @test normalized_point(convert_eltype(Float32, fish).intrinsics, SVector(100.0f0, 50.0f0)) isa SVector{2, Float32}
    @test allocs(normalized_point, fish.intrinsics, SVector(1.0, 2.0)) == 0
end

@testitem "with_parameters(::MappedSurface): the calibration rebuilt, every other member kept" begin
    using Test, StaticArrays, Random
    using Rotations: RotMatrix, RotXYZ
    using ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, with_parameters, parameter_vector, grid_nodes, convert_eltype

    rng = Xoshiro(7)
    nodes = grid_nodes(4; lo = -0.6, hi = 0.6)
    X = EuclideanMap(RotMatrix{3}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.02, -0.01, 2.0))
    P = SimilarityMap(0.9, one(RotMatrix{3, Float64}), SVector(-0.4, -0.45, 0.0))
    rect = Rect2(-0.5, -0.5, 1.0, 1.0)
    rays = [(SVector(0.3randn(rng), 0.3randn(rng), 0.0), SVector(0.2randn(rng), 0.2randn(rng), 1.0)) for _ in 1:200]
    H = HeightField(nodes, 0.01 .* randn(rng, 16))
    T = ThinPlateSpline(nodes, 0.01 .* randn(rng, 16), 0.01 .* randn(rng, 16))
    for (C, Pc) in ((H, nothing), (H, P), (H ∘ Aspect(0.01) ∘ Skew(-0.003), nothing), (H ∘ T, P))
        s = MappedSurface(X, C, Pc, rect)
        θ = parameter_vector(C) .+ 0.003 .* randn(rng, length(parameter_vector(C)))
        s2 = with_parameters(s, θ)
        built = MappedSurface(X, with_parameters(C, θ), Pc, rect)        # the checked builder from the parts
        @test s2 isa MappedSurface && s2.frame == s.frame && s2.map.t1 == s.map.t1   # the frame and the pose kept
        @test all(isequal(intersection(s2, o, d), intersection(built, o, d)) for (o, d) in rays)
        @test count(r -> isfinite(intersection(s2, r...)[1]), rays) > 50
        @test all(isequal(intersection(with_parameters(s, parameter_vector(C)), o, d), intersection(s, o, d)) for (o, d) in rays)
    end

    # RadiometricCamera's former form, bit for bit: every HeightField member rebuilt by its own with_parameters
    s = MappedSurface(X, H, P, rect)
    z = 0.005 .* randn(rng, 16)
    rc(m) = m isa HeightField ? with_parameters(m, z) : m
    by_members = MappedSurface(VisualGeometryToolkitCore.Cameras._map_members(rc, s.map), s.frame)
    @test all(isequal(intersection(with_parameters(s, z), o, d), intersection(by_members, o, d)) for (o, d) in rays)

    # Dual heights: the hit's value is the plain one's, and it carries partials
    zd = ForwardDiff.Dual{:h}.(z, 1.0)
    o, d = first(r for r in rays if isfinite(intersection(with_parameters(s, z), r...)[1]))
    td, xd = intersection(with_parameters(s, zd), o, d)
    @test ForwardDiff.value(td) ≈ intersection(with_parameters(s, z), o, d)[1] atol = 1.0e-14
    # a Float32 surface takes Float32 parameters
    s32 = convert_eltype(Float32, s)
    @test with_parameters(s32, Float32.(z)) isa MappedSurface
end

@testitem "bilinear_weights: the texel block and fractions bilinear_sample reads" begin
    using Test, StaticArrays, Random
    using VisualGeometryToolkitCore: bilinear_sample, bilinear_weights
    allocs(f, args...) = (f(args...); @allocated f(args...))

    rng = Xoshiro(3)
    img = randn(rng, 4, 8)                       # H = 4 rows, W = 8 columns (the texel centres exact in binary)
    blend(img, (i, j, fx, fy)) = (1 - fx) * (1 - fy) * img[j, i] + fx * (1 - fy) * img[j, i + 1] +
        (1 - fx) * fy * img[j + 1, i] + fx * fy * img[j + 1, i + 1]
    uvs = [SVector(rand(rng), rand(rng)) for _ in 1:500]
    append!(uvs, [SVector(-0.3, 0.5), SVector(1.4, 1.2), SVector(0.0, 0.0), SVector(1.0, 1.0), SVector(NaN, 0.4)])
    for uv in uvs
        w = bilinear_weights(uv, (8, 4))
        i, j, fx, fy = w
        @test 1 <= i <= 7 && 1 <= j <= 3 && 0 <= fx <= 1 && 0 <= fy <= 1
        @test blend(img, w) ≈ bilinear_sample(img, uv) atol = 8eps()      # the lookup's own arithmetic
    end
    # a texel centre: that texel, fraction 0 (or the block before it at the last centre, fraction 1)
    @test bilinear_weights(SVector(2.5 / 8, 1.5 / 4), (8, 4)) == (3, 2, 0.0, 0.0)
    @test bilinear_weights(SVector(7.5 / 8, 3.5 / 4), (8, 4)) == (7, 3, 1.0, 1.0)
    @test bilinear_weights(SVector(NaN, 0.1), (8, 4))[[1, 3]] == (1, 0.0)
    @test_throws ArgumentError bilinear_weights(SVector(0.5, 0.5), (1, 4))
    @test allocs(bilinear_weights, SVector(0.3, 0.6), (8, 4)) == 0
end

@testitem "julia_blocks: a page's Julia examples" begin
    using Test
    using VisualGeometryToolkitCore: julia_blocks

    page = joinpath(mktempdir(), "page.md")
    write(page, """
    # A page

    ```julia-repl
    pkg> add Foo
    ```

    ```julia
    using Foo
    x = 1
    ```

    ```python
    print(1)
    ```

    ```julia
    # Illustrative
    y = 2
    ```
    """)
    @test julia_blocks(page) == ["using Foo\nx = 1\n", "# Illustrative\ny = 2\n"]
    @test julia_blocks(page) isa Vector{String}
    write(page, "no code here\n")
    @test isempty(julia_blocks(page))
end
