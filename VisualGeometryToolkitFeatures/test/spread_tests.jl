# spread_strongest (src/spread.jl): SSC against the authors' C++ and against their sequential sweep, CPU against CUDA.
# test/data/ssc_bailo.toml is the output of the authors' `ssc` (ANMS-Codes, C++/include/anms.h, MIT) compiled with a
# cv::KeyPoint shim, on Float32 inputs stable-sorted by score as their demo does. `ssc_sequential` below is that
# function's loop, line for line, the oracle on inputs the fixture does not hold.

@testsetup module SpreadFixture
using TOML: TOML
using VisualGeometryToolkitFeatures: Detection
using VisualGeometryToolkitCore: Disc, Point2
export SSC_FIXTURE, ssc_sequential, point_detections

const SSC_FIXTURE = TOML.parsefile(joinpath(@__DIR__, "data", "ssc_bailo.toml"))

point_detections(x, y, s) = [Detection(Disc(Point2(Float64(x[i]), Float64(y[i])), 1.0), (response = Float64(s[i]),)) for i in eachindex(x)]

# The authors' C++ `ssc`, sequentially, on points stable-sorted by |score| descending: the kept input indices, ascending.
function ssc_sequential(x, y, s, cols, rows, n, tolerance)
    order = sortperm(abs.(s); rev = true)          # stable: ties keep the earlier index first
    M = length(s)
    exp1 = rows + cols + 2n
    exp2 = 4 * Int64(cols) + 4 * Int64(n) + 4 * Int64(rows) * n + Int64(rows)^2 + Int64(cols)^2 - 2 * Int64(rows) * cols +
           4 * Int64(rows) * cols * n
    exp3 = sqrt(Float64(exp2)); exp4 = Float64(n - 1)
    high = Int(max(-round((exp1 + exp3) / exp4), -round((exp1 - exp3) / exp4)))
    low = max(1, floor(Int, sqrt(M / n)))
    K = Float32(n); t = Float32(tolerance)
    kmin = round(Int, K - K * t, RoundNearestTiesAway); kmax = round(Int, K + K * t, RoundNearestTiesAway)
    prev = -1; result = Int[]
    while true
        width = low + (high - low) ÷ 2
        (width == prev || low > high) && break
        empty!(result)
        c = width / 2.0
        ncols = floor(Int, cols / c); nrows = floor(Int, rows / c)
        covered = falses(nrows + 1, ncols + 1)
        for i in order
            row = floor(Int, y[i] / c); col = floor(Int, x[i] / c)
            covered[row + 1, col + 1] && continue
            push!(result, i)
            r = floor(Int, width / c)
            covered[max(row - r, 0) + 1:min(row + r, nrows) + 1, max(col - r, 0) + 1:min(col + r, ncols) + 1] .= true
        end
        kmin <= length(result) <= kmax && break
        length(result) < kmin ? (high = width - 1) : (low = width + 1)
        prev = width
    end
    return sort(result)
end
end

@testitem "spread_strongest: the authors' C++ ssc's selections, exactly" setup = [SpreadFixture] begin
    using VisualGeometryToolkitFeatures: spread_strongest
    import KernelAbstractions as KA
    @test length(SSC_FIXTURE["cases"]) == 6
    for c in SSC_FIXTURE["cases"]
        ds = point_detections(c["x"], c["y"], c["score"])
        ref = sort(c["selected"] .+ 1)
        got = spread_strongest(ds, (c["height"], c["width"]), c["n"]; tolerance = c["tolerance"], device = KA.CPU())
        @test (c["name"], got) == (c["name"], ds[ref])
        # the in-test oracle is the C++ too
        @test (c["name"], ssc_sequential(c["x"], c["y"], c["score"], c["width"], c["height"], c["n"], c["tolerance"])) == (c["name"], ref)
    end
end

@testitem "spread_strongest: the sequential sweep's set on random inputs, ties and clusters; the count in the band" setup = [SpreadFixture] begin
    using VisualGeometryToolkitFeatures: spread_strongest
    using Random: Xoshiro
    import KernelAbstractions as KA
    band(n, tol) = (round(Int, n - n * tol, RoundNearestTiesAway), round(Int, n + n * tol, RoundNearestTiesAway))
    inband(k, n, tol) = band(n, tol)[1] <= k <= band(n, tol)[2]
    rng = Xoshiro(11)
    for trial in 1:40
        W, H = rand(rng, 50:900), rand(rng, 50:700)
        M = rand(rng, 50:3000)
        # clusters as well as uniform positions; scores with ties (quantized) or smooth gradients (long dependency chains)
        cx, cy = rand(rng) * W, rand(rng) * H
        clustered = rand(rng, Bool)
        x = clustered ? clamp.(cx .+ 30 .* randn(rng, M), 0, W) : W .* rand(rng, M)
        y = clustered ? clamp.(cy .+ 30 .* randn(rng, M), 0, H) : H .* rand(rng, M)
        kind = trial % 3
        s = kind == 0 ? rand(rng, M) .- 0.5 : kind == 1 ? rand(rng, -3:3, M) ./ 3 : x ./ W .+ y ./ H
        n = rand(rng, 2:max(2, M ÷ 3)); tol = rand(rng, (0.0, 0.05, 0.1, 0.3))
        ds = point_detections(x, y, s)
        got = spread_strongest(ds, (H, W), n; tolerance = tol, device = KA.CPU())
        ref = ssc_sequential(x, y, s, W, H, n, tol)
        if inband(length(ref), n, tol)          # the reference's search landed in the band: its set, exactly
            @test (trial, got) == (trial, ds[ref])
        else                                    # it missed (its integer widths jump over the band): the band
            @test (trial, length(got)) == (trial, clamp(length(got), band(n, tol)...))
        end
    end
    # the count within the tolerance, also where the reference's integer search misses the band (n = 2048 here: its
    # last covering keeps 2391), and at no tolerance
    x, y, s = 1000 .* rand(rng, 20_000), 700 .* rand(rng, 20_000), randn(rng, 20_000)
    ds = point_detections(x, y, s)
    @test !inband(length(ssc_sequential(x, y, s, 1000, 700, 2048, 0.02)), 2048, 0.02)
    for n in (100, 500, 2048), tol in (0.0, 0.02, 0.1)
        k = length(spread_strongest(ds, (700, 1000), n; tolerance = tol, device = KA.CPU()))
        @test (n, tol, inband(k, n, tol)) == (n, tol, true)
    end
end

@testitem "spread_strongest: edge cases, determinism, CPU == CUDA" setup = [SpreadFixture] begin
    using VisualGeometryToolkitFeatures: spread_strongest
    using Random: Xoshiro
    using CUDA: CUDA
    import KernelAbstractions as KA
    rng = Xoshiro(3)
    x, y, s = 640 .* rand(rng, 5000), 480 .* rand(rng, 5000), randn(rng, 5000)
    ds = point_detections(x, y, s)
    cpu = KA.CPU()
    @test spread_strongest(ds, (480, 640), 5000; device = cpu) === ds
    @test isempty(spread_strongest(ds, (480, 640), 0; device = cpu))
    @test spread_strongest(ds, (480, 640), 1; device = cpu) == [ds[argmax(abs.(s))]]
    @test spread_strongest(ds, (3, 480, 640), 300; device = cpu) == spread_strongest(ds, (480, 640), 300; device = cpu)
    @test_throws ArgumentError spread_strongest(ds, (480, 640), -1; device = cpu)
    @test_throws ArgumentError spread_strongest(ds, (480, 640), 10; tolerance = 1.0, device = cpu)
    @test_throws ArgumentError spread_strongest(point_detections([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, NaN, 0.5]), (8, 8), 2; device = cpu)
    # the same set run to run, in detector order
    a = spread_strongest(ds, (480, 640), 300; device = cpu)
    @test all(_ -> spread_strongest(ds, (480, 640), 300; device = cpu) == a, 1:5)
    @test issorted(map(d -> findfirst(e -> e === d, ds), a))
    if CUDA.functional()
        gpu = CUDA.CUDABackend()
        for n in (2, 50, 300, 1000)
            @test (n, spread_strongest(ds, (480, 640), n; device = gpu)) == (n, spread_strongest(ds, (480, 640), n; device = cpu))
        end
    end
end

@testitem "spread_strongest: a bright and a dim half, both covered" begin
    using VisualGeometryToolkitFeatures: spread_strongest, asift_detect
    using VisualGeometryToolkitCore: centroid
    import KernelAbstractions as KA
    # blobs on a grid; the right half's contrast a tenth of the left's
    H, W = 256, 512
    img = fill(0.5f0, H, W)
    for cy in 16:32:H, cx in 16:32:W
        a = cx < W / 2 ? 0.4f0 : 0.04f0
        for j in max(1, cx - 12):min(W, cx + 12), i in max(1, cy - 12):min(H, cy + 12)
            img[i, j] -= a * exp(-Float32((i - cy)^2 + (j - cx)^2) / (2 * 4.0f0^2))
        end
    end
    ds = asift_detect(img; device = KA.CPU(), tilts = [(1.0, 0.0)], intensity_thr = 0.005)
    left(d) = centroid(d)[1] < W / 2
    @test count(left, ds) > 0 && count(!left, ds) > 0
    n = 40
    top = ds[sortperm(abs.(getproperty.(ds, :response)); rev = true)[1:n]]
    @test count(!left, top) == 0                         # the global top n: all in the bright half
    kept = spread_strongest(ds, (H, W), n; device = KA.CPU())
    # spread: the dim half holds a share (the bright half's blobs also put extrema in the gaps between them), from its
    # top to its bottom
    dim = filter(!left, kept)
    @test length(dim) >= length(kept) ÷ 4 && count(left, kept) >= length(kept) ÷ 4
    @test minimum(d -> centroid(d)[2], dim) < H / 4 && maximum(d -> centroid(d)[2], dim) > 3H / 4
end
