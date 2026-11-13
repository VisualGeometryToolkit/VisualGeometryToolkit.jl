# detect_foerstner (the Förstner interest operator, after ip_fop.m) against
# synthetic images with known sub-pixel truth. Positions are COLMAP: the pixel
# (r, c) covers [c − 1, c] × [r − 1, r], its centre at (c − ½, r − ½).

@testsetup module FoerstnerFixture
using Random: AbstractRNG
export render, matlab_imresize2, reference_unmap, blob_grid

# The image of the indicator-like function f(x, y), each pixel the mean of
# ss × ss samples over its area (anti-aliased, so sub-pixel truth is in the image).
render(f, H, W; ss = 8) =
    [sum(f(c - 1 + (i - 0.5) / ss, r - 1 + (j - 0.5) / ss) for i in 1:ss, j in 1:ss) / ss^2 for r in 1:H, c in 1:W]

# Dots of radius 2.5 px over the whole frame, centres at sub-pixel offsets,
# within 8 px of every border.
function blob_grid(H, W; nx = 6, ny = 5)
    truth = [
        (x + 0.137 * i, y + 0.291 * j)
            for (i, x) in enumerate(range(8.0, W - 9.0; length = nx)) for (j, y) in enumerate(range(8.0, H - 9.0; length = ny))
    ]
    return truth, render((x, y) -> any((x - a)^2 + (y - b)^2 < 2.5^2 for (a, b) in truth) ? 0.8 : 0.2, H, W)
end

# MATLAB's imresize(g, 2, 'bilinear'), the reference's doubling (ip_fop.m line
# 172): doubled pixel j samples input j/2 + 1/4 (1-based), the border replicated.
# Test-only: it shows what the reference's grid and its back-map do.
function matlab_imresize2(g)
    H, W = size(g)
    taps(n) = [(clamp(floor(Int, j / 2 + 0.25), 1, n), clamp(floor(Int, j / 2 + 0.25) + 1, 1, n), (j / 2 + 0.25) - floor(j / 2 + 0.25)) for j in 1:(2n)]
    rt, ct = taps(H), taps(W)
    return [
        (1 - rt[i][3]) * ((1 - ct[j][3]) * g[rt[i][1], ct[j][1]] + ct[j][3] * g[rt[i][1], ct[j][2]]) +
            rt[i][3] * ((1 - ct[j][3]) * g[rt[i][2], ct[j][1]] + ct[j][3] * g[rt[i][2], ct[j][2]])
            for i in 1:(2H), j in 1:(2W)
    ]
end

# ip_fop.m lines 340–350: the reference's map from a doubled 1-based coordinate
# back to the input (n the input size), here returned in COLMAP.
reference_unmap(x′, n) = ((1 - n) / (1 - 2n) * x′ + (1 - 2) * n / (1 - 2n)) - 0.5
end

@testitem "detect_foerstner: circular points over the whole frame (COLMAP, < 0.01 px)" setup = [FoerstnerFixture] begin
    using VisualGeometryToolkitCore: Point2
    using LinearAlgebra: norm
    truth, img = blob_grid(100, 140)
    for T in (Float32, Float64)
        points = detect_foerstner(T, img)
        @test length(points) == length(truth)
        @test all(d -> d.label === :circular, points)
        err = [minimum(d -> norm(d.center - Point2(t)), points) for t in truth]
        @test maximum(err) < 0.01                       # half a pixel off would be 0.5
        @test all(d -> d.window isa Point2{Float64} && norm(d.window - d.center) < 1, points)
    end
end

@testitem "detect_foerstner: the reference's back-map is biased, the exact inverse is not" setup = [FoerstnerFixture] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    const VGF = VisualGeometryToolkitFeatures
    H, W = 100, 140
    truth, img = blob_grid(H, W)
    # the reference's sampling grid (imresize), its own windows and estimates
    found = VGF._foerstner_windows(
        KA.CPU(), matlab_imresize2(img); method = :foerstner, noise_sigma = 2.0, derivative = :gaussian2d,
        integration = :box, derivative_sigma = 1.0, integration_sigma = 1.41, precision_threshold = 0.5,
        roundness_threshold = 0.3
    )
    exact_err, reference_err, predicted = Float64[], Float64[], Float64[]
    for t in truth
        # the circular estimate (rows 8, 9: doubled 1-based row, column) nearest the truth
        k = argmin(k -> norm((found.estimates[9, k] / 2 - 0.25, found.estimates[8, k] / 2 - 0.25) .- t), axes(found.estimates, 2))
        r′, c′ = found.estimates[8, k], found.estimates[9, k]
        exact = (c′ / 2 - 0.25, r′ / 2 - 0.25)              # imresize's inverse, x = x′/2 + 1/4, in COLMAP
        reference = (reference_unmap(c′, W), reference_unmap(r′, H))
        push!(exact_err, maximum(abs, exact .- t))
        bias = reference .- t
        push!(reference_err, maximum(abs, bias))
        # the predicted bias ((n + 1)/2 − x)/(2n − 1), x 1-based
        expect = (((W + 1) / 2 - (t[1] + 0.5)) / (2W - 1), ((H + 1) / 2 - (t[2] + 0.5)) / (2H - 1))
        push!(predicted, maximum(abs, bias .- expect))
    end
    @test maximum(exact_err) < 0.01
    @test maximum(reference_err) > 0.2                  # near the borders
    @test minimum(reference_err) < maximum(reference_err) / 4   # toward the centre it vanishes
    @test maximum(predicted) < 0.01                     # the bias is the predicted one
end

@testitem "detect_foerstner: corners" setup = [FoerstnerFixture] begin
    using VisualGeometryToolkitCore: Point2, Ellipse, geom
    using Statistics: cov
    using LinearAlgebra: norm, eigvals
    t = (30.37, 27.81)
    img = render((x, y) -> 0.2 + 0.6 * xor(x > t[1], y > t[2]), 56, 64)   # a checkerboard X-junction
    for method in (:foerstner, :koethe)
        d = only(detect_foerstner(Float64, img; method))
        @test d.label === :corner
        @test d.F < 1
        e = d.center - Point2(t)
        # the window lands up to a few doubled pixels off the vertex of the
        # X-junction (its measure is flat there), which biases the estimate:
        # within the reported covariance, not to 0.01 px
        @test norm(e) < 0.15
        @test e' * (cov(d.value) \ e) < 13.8            # χ²₂(0.999)
        ell = Ellipse(d.value; sigma = 1)
        @test ell.center == d.center && ell.a ≈ sqrt(maximum(eigvals(Matrix(cov(d.value)))))
    end
end

@testitem "detect_foerstner: options, precision and arguments" setup = [FoerstnerFixture] begin
    using VisualGeometryToolkitCore: Point2
    using LinearAlgebra: norm
    using Statistics: cov
    truth, img = blob_grid(60, 70; nx = 2, ny = 2)
    p32, p64 = detect_foerstner(img), detect_foerstner(Float64, img)
    @test length(p32) == length(p64) == 4
    @test maximum(norm(a.center - b.center) for (a, b) in zip(p32, p64)) < 1.0e-3
    for kw in ((; derivative = :gaussian1d), (; integration = :gaussian), (; method = :koethe))
        p = detect_foerstner(Float64, img; kw...)
        @test all(d -> d.label === :circular, p) && length(p) == 4
    end
    # the labels from the statistic (ip_fop.m line 429), and an unknown row:
    # its window centre, a NaN covariance (the reference's noclass)
    VGF = VisualGeometryToolkitFeatures
    estimates = zeros(14, 3)
    estimates[[3, 6, 10, 13], :] .= 1.0                 # N⁻¹ = I for both models
    estimates[7, :] .= (1.0, 50.0, 10.0)                # Ω_corner
    estimates[14, :] .= (50.0, 1.0, 10.0)               # Ω_circular: F = 0.02, 50, 1
    found = (; centres = [CartesianIndex(11, 21), CartesianIndex(31, 41), CartesianIndex(51, 61)], estimates, redundancy = 167, upscale = 2)
    rows = VGF._foerstner_points(found, VGF._foerstner_classes(found, 0.999), VGF._undouble)
    @test [d.label for d in rows] == [:corner, :circular, :unknown]
    @test [d.F for d in rows] ≈ [0.02, 50.0, 1.0]
    u = rows[3]
    @test u.center == u.window == Point2(30.0, 25.0) && all(isnan, cov(u.value))
    @test isempty(detect_foerstner(fill(0.5f0, 40, 40)))
    @test_throws ArgumentError detect_foerstner(img; method = :harris)
    @test_throws ArgumentError detect_foerstner(img; derivative = :sobel)
    @test_throws ArgumentError detect_foerstner(img; integration = :median)
    @test_throws ArgumentError detect_foerstner(img; significance_level = 1)
    @test_throws ArgumentError detect_foerstner(img; noise = 0)
    @test_throws ArgumentError detect_foerstner(img; noise = Inf)
    @test all(d -> d.covariance_source === :a_posteriori, p64)
    @test all(d -> d.covariance_source === :propagated, detect_foerstner(Float64, img; noise = 2.0))
end

# The kept line 423 (the circular point's residual sum takes two terms from the
# corner model's residuals), against the circular model's own residual sum, by
# the coverage of the reported covariance on noisy dots (the docs page's numbers
# come from the 400-trial version of this).
@testitem "detect_foerstner: the circular point's covariance, line 423 against its derivation" setup = [FoerstnerFixture] begin
    import KernelAbstractions as KA
    using Random: Xoshiro, randn
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: SVector, SMatrix
    const VGF = VisualGeometryToolkitFeatures
    rng = Xoshiro(7)
    nees_kept, nees_derived = Float64[], Float64[]
    for _ in 1:100
        t = (24 + rand(rng), 24 + rand(rng))
        img = render((x, y) -> (x - t[1])^2 + (y - t[2])^2 < 2.5^2 ? 0.8 : 0.2, 48, 48) .+ 0.01 .* randn(rng, 48, 48)
        doubled = zeros(96, 96)
        VGF._launch!(VGF._upsample_doubled!, KA.CPU(), doubled, reshape(img, 48, 48, 1); ndrange = size(doubled))
        f = VGF._foerstner_windows(
            KA.CPU(), doubled; method = :foerstner, noise_sigma = 2.55, derivative = :gaussian2d, integration = :box,
            derivative_sigma = 1.0, integration_sigma = 1.41, precision_threshold = 0.5, roundness_threshold = 0.3
        )
        k = argmin(k -> norm(((f.centres[k][2] - 1) / 2, (f.centres[k][1] - 1) / 2) .- t), eachindex(f.centres))
        e = f.estimates[:, k]
        rw, cw = Tuple(f.centres[k])
        derived = 0.0                                   # Σ (xᵢ − x̂)ᵀ Γ⊥ᵢ (xᵢ − x̂), the circular model's own residuals
        for c in (cw - f.ρ):(cw + f.ρ), r in (rw - f.ρ):(rw + f.ρ)
            a, b, d = f.gamma[r, c, 1], f.gamma[r, c, 2], f.gamma[r, c, 3]
            er, ec = r - e[8], c - e[9]
            derived += b * er^2 - 2d * er * ec + a * ec^2
        end
        err = SVector((e[9] - 1) / 2 - t[1], (e[8] - 1) / 2 - t[2])
        Ninv = SMatrix{2, 2}(e[13], e[12], e[11], e[10]) / 4      # (x, y) order
        push!(nees_kept, err' * ((e[14] / f.redundancy * Ninv) \ err))
        push!(nees_derived, err' * ((derived / f.redundancy * Ninv) \ err))
    end
    mean(v) = sum(v) / length(v)
    # neither is calibrated (nominal mean 2); the kept form is the closer
    @test mean(nees_kept) < mean(nees_derived)
    @test count(<(5.99), nees_kept) > count(<(5.99), nees_derived)   # 95% coverage
end

# The propagated covariance is σn² J Jᵀ with J the derivative of the estimate with respect to every input pixel:
# against J by central differences of the detector itself (Float64), for a circular point in the interior, one
# whose derivative support crosses the image border (the dot at x = 5.8 puts the window centre near doubled column
# 12.6; the second suppression keeps none below 12, so the window starts at column 6 or 7 and its derivative taps,
# radius 7, read doubled columns 1 and 2 (the doubling's clamp) and the zero padding beyond), and a corner. The
# pixels perturbed are those the window can read: (ρ + r)/2 = 6.5 input pixels for its doubled half-width ρ = 6 and
# the taps' radius r = 7, plus 2 for the doubling and the rounding of the window centre. Central
# differences err by O(ε²) (the estimate is smooth in the pixels while the window stays) and by rounding ≈ eps/ε of
# the estimate's scale: at ε = 1e-5 both are ≤ 1e-9 of J relative, so 1e-6 holds them with room and is far below
# any real error (one missed pixel of the support is a percent).
@testitem "detect_foerstner: the propagated covariance is σn² J Jᵀ (central differences)" setup = [FoerstnerFixture] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    using Statistics: cov
    using VisualGeometryToolkitCore: Point2
    σn = 2.0
    scenes = (
        ((24.31, 23.77), render((x, y) -> (x - 24.31)^2 + (y - 23.77)^2 < 2.5^2 ? 0.8 : 0.2, 48, 48), :circular),
        ((5.8, 30.4), render((x, y) -> (x - 5.8)^2 + (y - 30.4)^2 < 2.5^2 ? 0.8 : 0.2, 48, 48), :circular),
        ((20.37, 21.81), render((x, y) -> 0.2 + 0.6 * ((x > 20.37) & (y > 21.81)), 40, 44), :corner),   # an L corner
    )
    reach = cld(6 + 7, 2) + 2
    for (t, img, label) in scenes
        points = detect_foerstner(Float64, img; device = KA.CPU(), noise = σn)
        k = argmin(k -> norm(points[k].center - Point2(t)), eachindex(points))
        d = points[k]
        @test d.label === label && d.covariance_source === :propagated
        c = round.(Int, (d.window[2] + 0.5, d.window[1] + 0.5))     # the window centre's input (row, column)
        ε = 1.0e-5
        Σfd = zeros(2, 2)
        for q in CartesianIndices(img)
            maximum(abs.(Tuple(q) .- c)) <= reach || continue
            moved = map((-ε, ε)) do h
                im = copy(img); im[q] += h
                p = detect_foerstner(Float64, im; device = KA.CPU())
                only(filter(e -> e.window == d.window, p)).center
            end
            j = (moved[2] - moved[1]) / 2ε
            Σfd .+= (σn / 255)^2 .* (j * j')
        end
        @test norm(Matrix(cov(d.value)) - Σfd) <= 1.0e-6 * norm(Σfd)
    end
end

# The Förstner study's Monte Carlo (Part 2): one dot geometry in each of n × n tiles, each with its own iid noise of
# known σn, NEES about the tiles' empirical mean (the noise-only spread: the estimator's own bias is not the noise's)
# and the χ²₂ 95 % coverage, at σi = 1.41 and 3. Bounds from the sampling alone, 4 standard errors at m = n² trials:
# NEES is the mean of χ²₂ (variance 4), so ±4·2/√m; the coverage a binomial share, ±4·√(0.95·0.05/m). The suite runs
# 16 × 16 tiles; the study's 32 × 32 (45 s on the CPU) runs with VGTK_BENCHMARK_TESTS=1.
@testsetup module FoerstnerMonteCarlo
import KernelAbstractions as KA
using Random: Xoshiro, randn
using LinearAlgebra: norm
using Statistics: cov
using VisualGeometryToolkitCore: Point2
using VisualGeometryToolkitFeatures: detect_foerstner
export foerstner_calibration

# (σd, σi) => (; m, mean_e2, cov95_2) of the circular points found, one per tile; `render` is FoerstnerFixture's and
# `mc_summary` UncFixtures' (another setup's names are the item's, passed in).
function foerstner_calibration(n, render, mc_summary; tile = 30, σn = 2.0, settings = ((1.0, 1.41), (1.0, 3.0)))
    o = (14.31, 13.77)                                   # the dot centre in its tile
    clean = render(tile * n, tile * n) do x, y
        a, b = x - tile * fld(x, tile) - o[1], y - tile * fld(y, tile) - o[2]
        a^2 + b^2 < 2.5^2 ? 0.8 : 0.2
    end
    rng = Xoshiro(11)
    return map(settings) do (σd, σi)
        img = clean .+ (σn / 255) .* randn(rng, size(clean))
        points = filter(d -> d.label === :circular, detect_foerstner(Float64, img; device = KA.CPU(), noise = σn,
                                                                     derivative_sigma = σd, integration_sigma = σi))
        est, Σs = Vector{Float64}[], Matrix{Float64}[]
        for ty in 0:(n - 1), tx in 0:(n - 1)
            t = Point2(tile * tx + o[1], tile * ty + o[2])
            i = findfirst(d -> norm(d.center - t) < 1.5, points)
            i === nothing && continue
            push!(est, [points[i].center[1] - t[1], points[i].center[2] - t[2], 0.0])
            Σ = zeros(3, 3); Σ[1:2, 1:2] .= Matrix(cov(points[i].value)); Σ[3, 3] = 1
            push!(Σs, Σ)
        end
        s = mc_summary(est, Σs)
        (σd, σi) => (; m = length(est), s.mean_e2, s.cov95_2)
    end
end
end

@testitem "detect_foerstner: the propagated covariance is calibrated (NEES₂ ≈ 2, χ² 95 % coverage)" setup = [FoerstnerFixture, UncFixtures, FoerstnerMonteCarlo] begin
    n = 16
    for (_, r) in foerstner_calibration(n, render, mc_summary)
        @test r.m == n^2
        @test abs(r.mean_e2 - 2) < 4 * 2 / sqrt(r.m)
        @test abs(r.cov95_2 - 0.95) < 4 * sqrt(0.95 * 0.05 / r.m)
    end
end

@testitem "detect_foerstner: the propagated covariance is calibrated, the study's 32 × 32 tiles" setup = [FoerstnerFixture, UncFixtures, FoerstnerMonteCarlo] skip = get(ENV, "VGTK_BENCHMARK_TESTS", "") != "1" begin
    n = 32
    for (_, r) in foerstner_calibration(n, render, mc_summary)
        @test r.m == n^2
        @test abs(r.mean_e2 - 2) < 4 * 2 / sqrt(r.m)
        @test abs(r.cov95_2 - 0.95) < 4 * sqrt(0.95 * 0.05 / r.m)
    end
end

# GPU: the same result as the CPU, within Float32 rounding. Runs only with a
# functional CUDA device (CUDA_VISIBLE_DEVICES=0); records a skip otherwise.
@testitem "detect_foerstner: CUDA matches the CPU" setup = [FoerstnerFixture] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    using Statistics: cov
    cuda = try
        m = Base.require(Main, :CUDA)
        (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing        # CUDA is in the test environment: no functional GPU here
    else
        truth, img = blob_grid(100, 140)
        img = [img render((x, y) -> 0.2 + 0.6 * xor(x > 30.37, y > 50.2), 100, 64)]   # and a corner
        backend = Base.invokelatest(getproperty(cuda, :CUDABackend))
        for (T, tol) in ((Float32, 1.0e-3), (Float64, 1.0e-9)), noise in (nothing, 2.0)
            c = detect_foerstner(T, img; device = KA.CPU(), noise)
            g = Base.invokelatest(detect_foerstner, T, img; device = backend, noise)
            @test length(c) == length(g) && [d.label for d in c] == [d.label for d in g]
            @test any(d -> d.label === :corner, c)
            @test maximum(norm(a.center - b.center) for (a, b) in zip(c, g)) < tol
            @test maximum(maximum(abs.(cov(a.value) .- cov(b.value)) ./ abs.(cov(a.value))) for (a, b) in zip(c, g)) < sqrt(tol)
        end
    end
end
