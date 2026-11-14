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
    # the default window (ρ = 6, ν ≈ 12): this junction (F ≈ 0.16) is unknown at 0.999 (no ellipse of its NaN covariance), a corner at 0.99
    for method in (:foerstner, :koethe)
        u = only(detect_foerstner(Float64, img; method))
        @test u.label === :unknown && all(isnan, cov(u.value))
        @test_throws ArgumentError Ellipse(u.value; sigma = 1)
        d = only(detect_foerstner(Float64, img; method, significance_level = 0.99))
        @test d.label === :corner && d.F == u.F
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
    taps = hcat(VGF._derivative_taps(:gaussian2d, 2.0)...)                     # derivative_sigma 1, doubled
    found = (; centres = [CartesianIndex(11, 21), CartesianIndex(31, 41), CartesianIndex(51, 61)], estimates, ρ = 6, redundancy = 167, upscale = 2, taps)
    rows = VGF._foerstner_points(found, VGF._foerstner_classes(found, 0.999), VGF._undouble)
    @test [d.label for d in rows] == [:corner, :circular, :unknown]
    @test [d.F for d in rows] ≈ [0.02, 50.0, 1.0]
    u = rows[3]
    @test u.center == u.window == Point2(30.0, 25.0) && all(isnan, cov(u.value))
    @test u.covariance_status === :unclassified && rows[1].covariance_status === rows[2].covariance_status === :a_posteriori
    @test isempty(detect_foerstner(fill(0.5f0, 40, 40)))
    @test_throws ArgumentError detect_foerstner(img; method = :harris)
    @test_throws ArgumentError detect_foerstner(img; derivative = :sobel)
    @test_throws ArgumentError detect_foerstner(img; integration = :median)
    @test_throws ArgumentError detect_foerstner(img; significance_level = 1)
    @test_throws ArgumentError detect_foerstner(img; noise = 0)
    @test_throws ArgumentError detect_foerstner(img; noise = Inf)
    @test all(d -> d.covariance_status === :a_posteriori, p64)
    @test all(d -> d.covariance_status === :propagated, detect_foerstner(Float64, img; noise = 2.0))
end

# The kept line 423 (the circular point's residual sum takes two terms from the
# corner model's residuals), against the circular model's own residual sum, by
# the coverage of the reported covariance on noisy dots (the docs page's numbers
# come from the 400-trial version of this).
@testitem "detect_foerstner: the F-test at the residual sums' effective degrees of freedom" setup = [FoerstnerFixture] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: Disc, Point2
    using Random: Xoshiro, randn
    using Statistics: mean, var
    VGF = VisualGeometryToolkitFeatures
    # white noise: each residual sum is the quadratic form Iᵀ Q I, E = σ² tr Q and ν = (tr Q)²/tr Q² (`_residual_dof`)
    σn, level = 0.5, 0.999
    img = Float32.(0.5 .+ (σn / 255) .* randn(Xoshiro(3), 600, 600))
    σ = 6.0                                                     # a seed's scale: ρ = 22 doubled px
    seeds = [Detection(Disc(Point2(x, y), σ), (; precision = 1.0)) for x in 30.0:24:570 for y in 30.0:24:570]   # disjoint windows
    n = length(seeds)
    g = VGF.ka_upload(Float32, KA.CPU(), img)
    doubled = KA.allocate(KA.CPU(), Float32, 2 .* size(g)...)
    VGF._launch!(VGF._upsample_doubled!, KA.CPU(), doubled, reshape(g, size(g)..., 1); ndrange = size(doubled))
    found = VGF._foerstner_seeded(KA.CPU(), doubled, seeds; derivative = :gaussian2d, derivative_sigma = 1.0)
    @test length(found.centres) == n && all(==(22), found.ρ)
    dof = VGF._residual_dof(found.taps, 22)
    # E[Ω] = σ² tr Q, within 4 standard errors of the mean (Var Ω = 2(σ² tr Q)²/ν); the corner sum's variance gives ν
    # within 4 standard errors (a sample variance's relative SE is √(2/(n − 1)))
    for (row, d) in ((7, dof.corner), (14, dof.circular))
        Ω = found.estimates[row, :]
        expected = (σn / 255)^2 * d.E
        @test abs(mean(Ω) - expected) < 4 * expected * sqrt(2 / d.ν / n)
    end
    Ωc = found.estimates[7, :]
    @test abs(2mean(Ωc)^2 / var(Ωc) - dof.corner.ν) < 4 * dof.corner.ν * sqrt(2 / (n - 1))
    @test dof.corner.ν < (2 * 22 + 1)^2 / 10                    # far below the pixel count (2ρ + 1)² − 2 = 2023
    # the image-wide default window (ρ = 6, ν ≈ 12; seeds at σ = 1.3 give it), where the neglected two parameters weigh
    # most: the corner sum's ν within 4 standard errors, and at most the stated rate of noise windows labelled
    p = 2 * (1 - level)                                         # the labelled rate on noise, both sides
    small = [Detection(Disc(Point2(x, y), 1.3), (; precision = 1.0)) for x in 20.0:10:580 for y in 20.0:10:580]
    f6 = VGF._foerstner_seeded(KA.CPU(), doubled, small; derivative = :gaussian2d, derivative_sigma = 1.0)
    ν6 = VGF._residual_dof(f6.taps, 6).corner.ν
    Ω6 = f6.estimates[7, :]
    @test all(==(6), f6.ρ) && 11 < ν6 < 13 && abs(2mean(Ω6)^2 / var(Ω6) - ν6) < 4 * ν6 * sqrt(2 / (length(small) - 1))
    m = length(small)
    @test count(!=(0), VGF._foerstner_classes(f6, level)) <= m * p + 4 * sqrt(m * p * (1 - p))
    # the labels: at most the stated rate (1 − level per side) of noise windows is labelled, 4 binomial SEs over it;
    # at the pixel count it was about a fifth per side
    rows = detect_foerstner(Float64, img, seeds; device = KA.CPU(), significance_level = level)
    @test count(d -> d.label !== :unknown, rows) <= n * p + 4 * sqrt(n * p * (1 - p))
    # a dot is still a circular point and a corner a corner (the suite's synthetic scenes elsewhere)
    truth, dots = blob_grid(60, 70; nx = 2, ny = 2)
    @test all(d -> d.label === :circular, detect_foerstner(Float64, dots; device = KA.CPU()))
end

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
        @test d.label === label && d.covariance_status === :propagated
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
        # a Poisson–Gaussian pixel variance α·x + σ² weighs each column by its own pixel's variance, J diag(v) Jᵀ
        α = 4 * (σn / 255)^2
        pg = VisualGeometryToolkitFeatures._detect_foerstner(Float64, img, (max.(α .* img, 0) .+ (σn / 255)^2, :propagated); device = KA.CPU())
        dpg = only(filter(e -> e.window == d.window, pg))
        Σpg = zeros(2, 2)
        for q in CartesianIndices(img)
            maximum(abs.(Tuple(q) .- c)) <= reach || continue
            moved = map((-ε, ε)) do h
                im = copy(img); im[q] += h
                only(filter(e -> e.window == d.window, detect_foerstner(Float64, im; device = KA.CPU()))).center
            end
            j = (moved[2] - moved[1]) / 2ε
            Σpg .+= (max(α * img[q], 0) + (σn / 255)^2) .* (j * j')
        end
        @test norm(Matrix(cov(dpg.value)) - Σpg) <= 1.0e-6 * norm(Σpg)
    end
end

# A RAW's green cells under the frame's sensor model: fed only under a gain (the owner's ruling A), each pixel's variance
# the uncertain image's (Core's one law), the status by the gain's source; read noise alone or none left a-posteriori,
# bit for bit the detector on the plain green plane; with K → ∞ (no Poisson part) the covariance is the iid one of the
# same σ, to the α·x term's share.
@testitem "detect_foerstner on a LinearRaw: the sensor model's gain, its source, no gain left a-posteriori" setup = [NoisyMosaic] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: LinearRaw, green_plane, sensor_fixture
    using VisualGeometryToolkitCore: value_of
    using Statistics: var
    using Statistics: cov
    using Random: Xoshiro
    using Test
    import VisualGeometryToolkitCore
    const SensorSource = VisualGeometryToolkitCore.SensorSource
    # a bright dot in the green cells (photosite (i, j) at cell ((i + 1) ÷ 2, (j + 1) ÷ 2))
    dot(i, j) = hypot((j + 1) ÷ 2 - 20.3, (i + 1) ÷ 2 - 18.7) < 3 ? 6000.0 : 600.0
    m = noisy_mosaic(Xoshiro(9), dot, 80, 80; electrons_per_dn = 2.5, read_dn = 3.0)
    plain = detect_foerstner(Float64, LinearRaw(m; mode = :green_cells); device = KA.CPU())
    @test any(d -> d.label === :circular, plain) && all(d -> d.covariance_status in (:a_posteriori, :unclassified), plain)
    for (source, status) in ((SensorSource.measured, :propagated_measured), (SensorSource.estimated, :propagated_estimated))
        model = pg_model(2.5, 3.0; source)
        p = detect_foerstner(Float64, LinearRaw(m; mode = :green_cells, sensor_model = model); sensor_model = model, device = KA.CPU())
        @test any(d -> d.label === :circular, p) && all(d -> d.covariance_status in (status, :unclassified), p)
    end
    # no gain: read noise alone, or none — the a-posteriori covariance, the plain detector's numbers bit for bit
    for nogain in (sensor_fixture(; read_noise = (3.0, 3.0, 3.0, 3.0)).model, sensor_fixture().model)
        a = detect_foerstner(Float64, LinearRaw(m; mode = :green_cells, sensor_model = nogain); sensor_model = nogain, device = KA.CPU())
        @test [cov(d.value) for d in a] == [cov(d.value) for d in detect_foerstner(Float64, Array(green_plane(LinearRaw(m; mode = :green_cells))); device = KA.CPU())]
    end
    # a gain whose LinearRaw holds no variances (linearised without the model): refused
    @test_throws ArgumentError detect_foerstner(Float64, LinearRaw(m; mode = :green_cells); sensor_model = pg_model(2.5, 3.0), device = KA.CPU())
    # no Poisson part: K = 1e12 electrons per code leaves σ² alone, the iid case of σn = 255·σ
    big_model = pg_model(1.0e12, 3.0)
    Lb = LinearRaw(m; mode = :green_cells, sensor_model = big_model)
    x = map(value_of, green_plane(Lb))
    σ² = minimum(var, green_plane(Lb))
    big = detect_foerstner(Float64, Lb; sensor_model = big_model, device = KA.CPU())
    iid = detect_foerstner(Float64, Array(x); noise = 255 * sqrt(σ²), noise_sigma = 2.0, device = KA.CPU())
    @test length(big) == length(iid)
    for (p, q) in zip(big, iid)
        p.label === :unknown && continue
        @test isapprox(Matrix(cov(p.value)), Matrix(cov(q.value)); rtol = 1.0e-6)   # α·x/σ² ≤ 1e-8 here
    end
end

# The Förstner study's Monte Carlo (Part 2): one dot geometry in each of n × n tiles, each with its own iid noise of
# known σn, NEES about the tiles' empirical mean (the noise-only spread: the estimator's own bias is not the noise's)
# and the χ²₂ 95 % coverage, at σi = 1.41 and 3. Bounds from the sampling alone, 4 standard errors at m = n² trials:
# NEES is the mean of χ²₂ (variance 4), so ±4·2/√m; the coverage a binomial share, ±4·√(0.95·0.05/m). The study's
# 32 × 32 tiles (seconds on the CPU).
@testsetup module FoerstnerMonteCarlo
import KernelAbstractions as KA
using Random: Xoshiro, randn
using LinearAlgebra: norm
using Statistics: cov
using VisualGeometryToolkitCore: Point2
using VisualGeometryToolkitFeatures: detect_foerstner
export foerstner_calibration

# (σd, σi) => (; m, mean_e2, cov95_2) of the circular points found, one per tile. `render` (FoerstnerFixture) and
# `mc_summary` (UncFixtures) are arguments: a setup module does not see the other setups.
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

@testitem "detect_sfop and the seeded detect_foerstner on a LinearRaw: the green cells under the frame's sensor model" setup = [NoisyMosaic] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: LinearRaw, green_plane, value_of, Disc
    using Statistics: cov
    using Random: Xoshiro
    using Test
    # two bright dots in the green cells (photosite (i, j) at cell ((i + 1) ÷ 2, (j + 1) ÷ 2))
    dot(i, j) = any(hypot((j + 1) ÷ 2 - a, (i + 1) ÷ 2 - b) < 3 for (a, b) in ((60.3, 58.7), (85.6, 81.2))) ? 6000.0 : 600.0
    m = noisy_mosaic(Xoshiro(9), dot, 280, 280; electrons_per_dn = 2.5, read_dn = 3.0)   # 140 × 140 cells: SFOP's 3 octaves
    model = pg_model(2.5, 3.0)
    L = LinearRaw(m; mode = :green_cells, sensor_model = model)
    x = Array(map(value_of, green_plane(L)))
    # SFOP on the green cells' values, as on the matrix
    seeds = detect_sfop(L; device = KA.CPU(), noise_sigma = 2.0)
    @test isequal(collect(seeds), collect(detect_sfop(x; device = KA.CPU(), noise_sigma = 2.0))) && length(seeds) >= 2
    # the seeded windows under the sensor model: the image-wide call's numbers at the same windows, the model's covariance with them
    found = detect_foerstner(Float64, L; sensor_model = model, device = KA.CPU(), integration_sigma = 2.0)
    again = detect_foerstner(Float64, L, [Detection(Disc(d.window, 2.0), (; precision = 1.0)) for d in found]; sensor_model = model, device = KA.CPU())
    @test length(again) == length(found) && all(isequal(a, b) for (a, b) in zip(again, found))
    @test any(d -> d.covariance_status === :propagated_measured, again)
    rows = detect_foerstner(Float64, L, seeds; sensor_model = model, device = KA.CPU())
    @test count(d -> d.label === :circular, rows) >= 2
end

@testitem "detect_foerstner: the propagated covariance is calibrated (NEES₂ ≈ 2, χ² 95 % coverage)" setup = [FoerstnerFixture, UncFixtures, FoerstnerMonteCarlo] begin
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
    import VisualGeometryToolkitCore
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
            # the covariances of the classified rows (an :unknown row has none: NaN)
            @test maximum(maximum(abs.(cov(a.value) .- cov(b.value)) ./ abs.(cov(a.value))) for (a, b) in zip(c, g) if a.label !== :unknown) < sqrt(tol)
            # the seeded windows, one half-width per seed (the F-test reads them on the host)
            seeds = [Detection(VisualGeometryToolkitCore.Disc(d.window, σ), (; precision = 1.0)) for (d, σ) in zip(c, Iterators.cycle((2.0, 3.0)))]
            cs = detect_foerstner(T, img, seeds; device = KA.CPU(), noise)
            gs = Base.invokelatest(detect_foerstner, T, img, seeds; device = backend, noise)
            @test length(cs) == length(gs) && [d.label for d in cs] == [d.label for d in gs]
            @test maximum(norm(a.center - b.center) for (a, b) in zip(cs, gs) if a.label !== :unknown) < tol
        end
    end
end

# gaussian_gradient: the package's one Gaussian-derivative gradient (Förstner's), with gaussian_blur's mask semantics.
@testitem "gaussian_gradient: a linear ramp's gradient, Förstner's correlation bit for bit, gaussian_blur's mask" setup = [FoerstnerFixture] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitFeatures: gaussian_gradient
    const VGF = VisualGeometryToolkitFeatures
    H, W = 60, 70
    α, β = 0.013, -0.007                                  # ∂/∂column, ∂/∂row per pixel
    ramp = [0.3 + α * c + β * r for r in 1:H, c in 1:W]
    for σ in (1.0, 2.0, 2.8)
        gx, gy = gaussian_gradient(ramp, σ)
        @test size(gx) == size(gy) == (H, W) && eltype(gx) == Float64
        d, c = VGF._derivative_taps(:gaussian2d, σ)
        r = length(d) ÷ 2
        inner = (r + 1):(H - r), (r + 1):(W - r)
        # the taps' first moment Σ x·d[x] is 1 less the Gaussian's second-moment mass beyond the truncation ⌈3.3σ⌉:
        # 2(cφ(c) + 1 − Φ(c)) = 0.0124 at c = 3.3 (the sampling's error, e^(−2π²σ²) ≤ 3e-9 at σ ≥ 1, is far below); the
        # across taps sum to 1, so a ramp's gradient is its slope times that moment
        @test all(isapprox.(gx[inner...], α; rtol = 0.0125)) && all(isapprox.(gy[inner...], β; rtol = 0.0125))
        # the correlation Förstner ran before: rows with (d, c), columns with (c, d), zero padded — the same bits
        tmp = zeros(H, W, 2); ref = zeros(H, W, 2)
        VGF._launch!(VGF._correlate_rows!, KA.CPU(), tmp, reshape(ramp, H, W, 1), [d c]; ndrange = size(tmp))
        VGF._launch!(VGF._correlate_columns!, KA.CPU(), ref, tmp, [c d]; ndrange = size(ref))
        @test gy == ref[:, :, 1] && gx == ref[:, :, 2]
    end
    # :gaussian1d: no smoothing across; an integer image is read as float
    gx1, gy1 = gaussian_gradient(round.(Int, 100 .* ramp), 1.5; derivative = :gaussian1d)
    @test eltype(gx1) == Float64
    # the mask (gaussian_blur's normalized convolution): no bleed from outside, 0 outside, the plain gradient deep inside
    disc = [hypot(r - 30.5, c - 35.5) < 22 ? 1.0 : 0.0 for r in 1:H, c in 1:W]
    wild = ifelse.(disc .> 0, ramp, 1.0e6)                 # a huge value outside the mask must not enter
    mx, my = gaussian_gradient(wild, 2.0, disc)
    nx, ny = gaussian_gradient(ramp, 2.0, disc)
    @test mx == nx && my == ny
    @test all(iszero, mx[disc .== 0]) && all(iszero, my[disc .== 0])
    px, py = gaussian_gradient(ramp, 2.0)
    r2 = length(first(VGF._derivative_taps(:gaussian2d, 2.0))) ÷ 2
    deep = [hypot(r - 30.5, c - 35.5) < 22 - sqrt(2) * r2 - 1 for r in 1:H, c in 1:W]   # the whole square stencil (its corners at √2·r) inside the mask
    # there D = 1 to rounding (the unit-sum taps over ones) and ∂D = 0, so the quotient is N's gradient: rounding only
    @test maximum(abs.(mx[deep] .- px[deep])) <= 64 * eps() * maximum(abs, px)
    @test_throws ArgumentError gaussian_gradient(ramp, 0.0)
    @test_throws ArgumentError gaussian_gradient(ramp, 1.0; derivative = :sobel)
    @test_throws DimensionMismatch gaussian_gradient(ramp, 1.0, ones(3, 3))
end

# GPU: gaussian_gradient on the device against the CPU. The two run the same correlations in the same order; a GPU may
# fuse each multiply-add, so each of the two correlations differs by at most its 2r + 1 terms' rounding: per pixel
# |Δ| ≤ 2(2r + 1)·eps(T)·max|img|·Σ|d|·Σ|c|. Runs only with a functional CUDA device; records a skip otherwise.
@testitem "gaussian_gradient: CUDA matches the CPU" setup = [FoerstnerFixture] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitFeatures: gaussian_gradient
    const VGF = VisualGeometryToolkitFeatures
    cuda = try
        m = Base.require(Main, :CUDA)
        (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing
    else
        cu = getproperty(cuda, :cu)
        img = Float32.(render((x, y) -> (x - 31.2)^2 + (y - 22.7)^2 < 64 ? 0.8 : 0.2, 48, 64))
        disc = Float32.([hypot(r - 24, c - 32) < 20 for r in 1:48, c in 1:64])
        for σ in (1.0f0, 2.5f0), mask in (nothing, disc)
            cx, cy = gaussian_gradient(img, σ, mask)
            gx, gy = Base.invokelatest(gaussian_gradient, cu(img), σ, mask === nothing ? nothing : cu(mask))
            d, c = VGF._derivative_taps(:gaussian2d, σ)
            r = length(d) ÷ 2
            bound = 2 * (2r + 1) * eps(Float32) * maximum(abs, img) * sum(abs, d) * sum(abs, c)
            if mask === nothing
                @test maximum(abs.(Array(gx) .- cx)) <= bound && maximum(abs.(Array(gy) .- cy)) <= bound
            else
                # g = (∂N·D − N·∂D)/D², each of N, ∂N, D, ∂D within `bound` (D's and ∂D's with max|mask| = 1): to first order
                # |Δg| ≤ (bound·D + |∂N|·bound + bound·|∂D| + |N|·bound)/D² + 2|g|·bound/D, per pixel, plus the quotient's
                # own rounding, a few eps(T)·|g|
                H, W = size(img)
                n = VGF._separable_correlate(KA.CPU(), reshape(Float64.(img .* disc), H, W, 1), [c d c], [c c d])
                dd = VGF._separable_correlate(KA.CPU(), reshape(Float64.(disc), H, W, 1), [c d c], [c c d])
                inside = disc .> 0
                for (k, gpu, cpu) in ((3, gx, cx), (2, gy, cy))
                    N, ∂N, D, ∂D = n[:, :, 1], n[:, :, k], dd[:, :, 1], dd[:, :, k]
                    allowed = @. (bound * D + abs(∂N) * bound + bound * abs(∂D) + abs(N) * bound) / D^2 +
                                 2 * abs(cpu) * bound / D + 4 * eps(Float32) * abs(cpu)
                    @test all((abs.(Array(gpu) .- cpu) .<= allowed)[inside])
                end
            end
        end
    end
end
