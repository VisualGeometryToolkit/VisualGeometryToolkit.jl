# The image circle MEASURED from photographs (`detect_image_circle`): synthetic frames of a bright disc of known centre
# and radius on a dark surround (each frame darkened by a "scene" region over a different part of the rim: no single
# frame shows the whole rim), and rims of known outer edge and knee (a sharp stop, a linear ramp) with clutter,
# saturation and noise. The coverage is the RANSAC ellipse through every angle's log-brightness drops on log-polar
# profiles of the frames' maximum about the image centre. The fit is the VisualGeometryToolkitEstimation extension.

@testsetup module ImageCircleFixtures
    using Statistics: mean
    using Random: Xoshiro
    export cover, disc_frames, rim_frames
    # the fraction of pixel (x, y) (1-based column, row) inside the disc of centre (cx, cy) and radius ρ, COLMAP pixels,
    # on a 4 × 4 subpixel grid
    cover(x, y, cx, cy, ρ) = mean((x - 1 + (i - 0.5) / 4 - cx)^2 + (y - 1 + (j - 0.5) / 4 - cy)^2 <= ρ^2 for i in 1:4, j in 1:4)
    # six frames of the disc, each darkened over one part of the rim
    function rim_frames(W, H, c, r)
        disc = [cover(u, v, c..., r) for v in 1:H, u in 1:W]
        rng = Xoshiro(7)
        return map(1:6) do k
            φ = 2π * (k - 1) / 6
            scene = [cover(u, v, c[1] + r * cos(φ), c[2] + r * sin(φ), 45.0) for v in 1:H, u in 1:W]   # dark over one part of the rim
            Float32.(clamp.(0.01 .+ 0.75 .* disc .* (1 .- scene) .+ 0.01 .* randn(rng, H, W), 0, 1))
        end
    end
    # n frames of the whole disc, with noise
    function disc_frames(W, H, c, r, n)
        disc = [cover(u, v, c..., r) for v in 1:H, u in 1:W]
        rng = Xoshiro(7)
        return [Float32.(clamp.(0.01 .+ 0.75 .* disc .+ 0.01 .* randn(rng, H, W), 0, 1)) for _ in 1:n]
    end
end

# A rim of known outer edge `re` and inner knee `rk` for the rim profile (`detect_image_circle`'s `rim`, `rim_margin`): the
# interior a quadratic vignetting, a linear ramp from the knee to the edge (its ends rounded over `foot` px when given), a
# dark floor beyond (`re == rk`: a sharp stop,
# its 1-px partial band by 4 × 4 supersampling); noise σ; scene clutter (six dark discs inside the knee, a different set
# per frame); and in the first frame a saturated patch of radius `sat_radius` px, `sat_at` px right of the centre.
@testsetup module RimFixtures
    using Random: Xoshiro
    using Statistics: mean
    export rim_synth
    function rim_synth(W, H, c, re, rk; nframes = 3, fl = 0.01, level = 0.8, vig = 0.15, σ = 0.004, sat_at = 0.6rk, sat_radius = 12,
                       foot = 0.0, seed = 7)
        rng = Xoshiro(seed)
        # `foot` > 0: the ramp's knee and foot rounded over `foot` px (the ramp's clamps softened, x ↦ foot·log(1 + eˣ/ᶠᵒᵒᵗ)),
        # a soft tail into the floor like the 14 mm's
        soft(x) = foot * log1p(exp(x / foot))
        prof(r) = foot > 0 ? fl + (level - fl) * (1 - vig * (min(r, rk) / re)^2) * (soft(re - r) - soft(rk - r)) / (re - rk) :
                  r >= re ? fl : r <= rk ? fl + (level - fl) * (1 - vig * (r / re)^2) :
                  fl + (level - fl) * (1 - vig * (rk / re)^2) * (re - r) / (re - rk)
        base = re == rk ?
            [mean(prof(hypot(u - 1 + (i - 0.5) / 4 - c[1], v - 1 + (j - 0.5) / 4 - c[2])) for i in 1:4, j in 1:4) for v in 1:H, u in 1:W] :
            [prof(hypot(u - 0.5 - c[1], v - 0.5 - c[2])) for v in 1:H, u in 1:W]
        return map(1:nframes) do k
            img = copy(base)
            for _ in 1:6
                φ, ρ, s = 2π * rand(rng), (rk - 25) * sqrt(rand(rng)), 8 + 10rand(rng)
                cx, cy = c[1] + ρ * cos(φ), c[2] + ρ * sin(φ)
                for v in 1:H, u in 1:W
                    hypot(u - 0.5 - cx, v - 0.5 - cy) < s && (img[v, u] *= 0.3)
                end
            end
            img = clamp.(img .+ σ .* randn(rng, H, W), 0, 1)
            if k == 1
                for v in 1:H, u in 1:W
                    hypot(u - 0.5 - (c[1] + sat_at), v - 0.5 - c[2]) < sat_radius && (img[v, u] = 1.0)
                end
            end
            Float32.(img)
        end
    end
end

@testitem "detect_image_circle: the coverage from frames that each show part of the rim" setup = [ImageCircleFixtures] begin
    using Random: Xoshiro
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation   # the fit's extension
    W, H, c, r = 360, 240, (181.3, 118.7), 100.0
    a = 0.1 * hypot(W, H)   # the outlier half-width: a literal of the caller's
    frames = rim_frames(W, H, c, r)
    # no single frame shows the whole rim: a point just inside it is dark in each frame
    rim(φ) = CartesianIndex(round(Int, c[2] + (r - 3) * sin(φ) + 0.5), round(Int, c[1] + (r - 3) * cos(φ) + 0.5))
    @test all(f -> any(φ -> f[rim(φ)] < 0.3, range(0, 2π; length = 73)), frames)
    res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a)          # the default model: an ellipse
    e = res.circle
    @test e isa VgtkCore.Ellipse{Float64}
    # the centre to a few hundredths of a pixel; the axes within half a pixel of the true rim: the largest log drop of
    # the stop's 1-px partial-coverage band, its 2 px span's middle on the 1-px radial grid
    @test hypot(e.center[1] - c[1], e.center[2] - c[2]) < 0.1 && abs(e.a - r) < 0.5 && abs(e.b - r) < 0.5
    # its uncertainty: the inliers' scatter through the fit's covariance, a few hundredths of a pixel here, and it covers
    # the error (the true radius within 3σ of the axes plus the half-pixel bias of the band)
    @test all(s -> 0 < s < 0.1, res.center_scale) && all(s -> 0 < s < 0.1, res.axes_scale)
    @test size(res.covariance) == (5, 5)
    rc = detect_image_circle(Xoshiro(1), VgtkCore.Circle, frames; outlier_halfwidth = a)   # the model type leads, after the RNG
    @test rc.circle isa VgtkCore.Circle{Float64} && hypot(rc.circle.center[1] - c[1], rc.circle.center[2] - c[2]) < 0.1 &&
        abs(rc.circle.r - r) < 0.5 && rc.covariance === nothing && all(isnan, rc.axes_scale)   # the circle problem provides no covariance
    @test detect_image_circle(VgtkCore.Circle, frames; outlier_halfwidth = a).circle isa VgtkCore.Circle &&
        detect_image_circle(frames; outlier_halfwidth = a).circle isa VgtkCore.Ellipse
    @test VgtkCore.converged(res.fit) && length(res.contour) == count(==(:ok), res.rim.reason) <= count(VgtkCore.inlier_mask(res.fit))
    # the contour is the rim's points (per angle the inlier candidate nearest the coverage), within a pixel of the true
    # circle; the scene regions inside the disc do not move it
    @test all(p -> abs(hypot(p[1] - c[1], p[2] - c[2]) - r) < 1.0, res.contour)
    @test res.rim isa VgtkFeatures.RimProfile && res.rim.statistics === VgtkFeatures.EmpiricalNoise()
    # `nothing` when the lens lights the whole frame: no rim, no candidate
    @test detect_image_circle([fill(0.8f0, H, W)]; outlier_halfwidth = a) === nothing
    # frames may stream (a generator)
    @test detect_image_circle(Xoshiro(1), (f for f in frames); outlier_halfwidth = a).circle == e
    @test_throws DimensionMismatch detect_image_circle([frames[1], frames[1][1:end-1, :]]; outlier_halfwidth = a)
    @test_throws ArgumentError detect_image_circle(Matrix{Float32}[]; outlier_halfwidth = a)
    @test res.cut == false
    @test res.rim_gap < π                                                                     # the rim seen all round the centre
end

@testitem "detect_image_circle: a disc cut by the frame (the rim in two arcs)" setup = [ImageCircleFixtures] begin
    # an APS-C lens on the full frame: the lit disc reaches past the top and bottom of the frame; the angles there have no
    # drop, and the coverage comes from the two arcs
    using Random: Xoshiro
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, r = 360, 240, (181.3, 118.7), 140.0
    a = 0.1 * hypot(W, H)
    disc = [cover(u, v, c..., r) for v in 1:H, u in 1:W]
    frames = disc_frames(W, H, c, r, 3)
    @test disc[1, round(Int, c[1])] > 0.9 && disc[H, round(Int, c[1])] > 0.9 && disc[1, 1] == 0      # cut at the top and bottom, corners dark
    res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a)
    @test res !== nothing && res.cut
    e = res.circle
    @test e isa VgtkCore.Ellipse{Float64}                                                        # the model asked for, cut or not
    @test hypot(e.center[1] - c[1], e.center[2] - c[2]) < 0.5 && abs(e.a - r) < 1.0
    @test detect_image_circle(Xoshiro(1), VgtkCore.Circle, frames; outlier_halfwidth = a).circle isa VgtkCore.Circle
    @test any(p -> p[1] < c[1], res.contour) && any(p -> p[1] > c[1], res.contour)              # both arcs
    @test all(p -> abs(hypot(p[1] - c[1], p[2] - c[2]) - r) < 1.5, res.contour)                # ≤ 1.5 px: the arcs' ends meet the frame
    @test res.rim_gap < π                                                                     # two opposite arcs span the centre
end

@testitem "detect_image_circle: one arc of rim (a single dark corner) — a mask, extrapolated away from the arc" setup = [ImageCircleFixtures] begin
    # the disc covers three corners of the frame; only the fourth is dark, so the rim is one short arc, near the corner:
    # the band reaches all of it only past the half diagonal (rim_band's outer 0.25). The circle is a MASK, its boundary right
    # along the arc, its centre and radius not determined (rim_gap > π says so)
    using Random: Xoshiro
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, r = 360, 240, (120.0, 90.0), 260.0
    a = 0.1 * hypot(W, H)
    noise = Xoshiro(7)
    one = [Float32(0.01 + 0.75 * cover(u, v, c..., r) + 0.01 * randn(noise)) for v in 1:H, u in 1:W]
    res = detect_image_circle(Xoshiro(1), VgtkCore.Circle, [one]; outlier_halfwidth = a, rim_band = (0.8, 0.25))
    @test res !== nothing && res.circle isa VgtkCore.Circle{Float64}
    @test res.rim_gap > π                                                                     # one arc: the rest is extrapolated
    # along the observed arc the true rim is within 2 px of the fitted boundary
    truth_rim = [(c[1] + r * cos(φ), c[2] + r * sin(φ)) for φ in (atan(p[2] - c[2], p[1] - c[1]) for p in res.contour)]
    fitted = res.circle
    @test all(q -> abs(hypot(q[1] - fitted.center[1], q[2] - fitted.center[2]) - fitted.r) < 2.0, truth_rim)
    # within the default band (to the corners) only the arc's middle is in reach: still a mask, one arc
    near = detect_image_circle(Xoshiro(1), VgtkCore.Circle, [one]; outlier_halfwidth = a)
    @test near === nothing || near.rim_gap > π
end

@testitem "detect_image_circle: a hot pixel in the dark surround moves nothing" setup = [ImageCircleFixtures] begin
    # a hot pixel is one pixel of a bin's dozens: the bin median ignores it
    using Random: Xoshiro
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, r = 360, 240, (181.3, 118.7), 80.0
    a = 0.1 * hypot(W, H)
    frames = disc_frames(W, H, c, r, 3)
    clean = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a).circle
    frames[2][30, 40] = 0.9f0                                               # the hot pixel: 30 px from the border, ~90 px off the rim
    res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a)
    @test hypot(res.circle.center[1] - clean.center[1], res.circle.center[2] - clean.center[2]) < 0.05 && abs(res.circle.a - clean.a) < 0.05
    @test hypot(res.circle.center[1] - c[1], res.circle.center[2] - c[2]) < 0.1 && abs(res.circle.a - r) < 0.5
end

@testitem "detect_image_circle: the hull fit's keywords are deprecated and ignored; every keyword is checked" setup = [ImageCircleFixtures] begin
    using Random: Xoshiro
    using Test
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, r = 360, 240, (181.3, 118.7), 80.0
    a = 0.1 * hypot(W, H)
    frames = disc_frames(W, H, c, r, 2)
    res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a)
    for kw in ((; min_hole_area = 8), (; smoothing_weight = 5e-4), (; border_margin = 5))
        dep = @test_deprecated detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, kw...)
        # under --depwarn=error the warning is the error @test_deprecated checks; otherwise the call returns, unchanged
        Base.JLOptions().depwarn == 2 || @test dep.circle == res.circle
    end
    @test_throws UndefKeywordError detect_image_circle(Xoshiro(1), frames)                 # outlier_halfwidth is required
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = 0)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, rim_angles = 4)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, rim_band = (1.0, 0.0))
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, significance = 1)
    unread = (error("a frame was read") for _ in 1:1)                                       # every keyword before a frame
    err = try
        detect_image_circle(unread; outlier_halfwidth = -1)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("outlier_halfwidth", err.msg)
end

@testitem "detect_image_circle on RAW frames: the green cells, the circle in raw pixels" setup = [ImageCircleFixtures] begin
    # A RAW frame is measured on its green cells (LinearRaw{:green_cells}, the geometry image) and the circle taken to the
    # raw's pixels by the cells' raw_frame_map (Core's similarity action on an Ellipse); the keywords are in green-cell
    # pixels (the resolution measured on: only the result is mapped). The same scene measured on the full-resolution frames
    # agrees to the cell's size (the green cells average two photosites: the rim's band, ≲ 2 raw px).
    using Random: Xoshiro
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    using VisualGeometryToolkitCore: RawMosaic, LinearRaw, raw_frame_map
    using StaticArrays: SVector
    W, H, c, r = 360, 240, (181.3, 118.7), 100.0
    a = 0.1 * hypot(W, H)
    frames = rim_frames(W, H, c, r)
    md = (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing, iso = nothing,
          model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = "")
    mosaic(f) = RawMosaic(round.(UInt16, f .* 65535), UInt8[0 1; 3 2], (0.0, 0.0, 0.0, 0.0), 65535.0, (0, 0), (H, W), md)
    mosaics = mosaic.(frames)
    cells = [LinearRaw(m; mode = :green_cells) for m in mosaics]
    S = VgtkCore.unlabel(raw_frame_map(first(cells)))
    @test VgtkCore.scale(S) == 2
    full = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a).circle
    # the estimate on the cells' plane, taken to the raw by hand: what the RAW method does
    planes = [L.image[:, :, 1] for L in cells]
    by_hand = detect_image_circle(Xoshiro(1), planes; outlier_halfwidth = a / 2)
    green = S(by_hand.circle)
    @test green isa VgtkCore.Ellipse{Float64}
    @test hypot(green.center[1] - full.center[1], green.center[2] - full.center[2]) < 1.0
    @test abs(green.a - full.a) < 2.0 && abs(green.b - full.b) < 2.0                                  # one green cell = 2 raw px
    @test hypot(green.center[1] - c[1], green.center[2] - c[2]) < 1.0 && abs(green.a - r) < 2.0
    # the RAW method: the same, bit for bit, with the same keywords (green-cell px: only the result is mapped)
    raw = detect_image_circle(Xoshiro(1), cells; outlier_halfwidth = a / 2)
    @test raw.circle === green
    @test raw.contour == [VgtkCore.Point2{Float64}(S(SVector{2, Float64}(p[1], p[2]))) for p in by_hand.contour] && raw.cut == by_hand.cut && raw.rim_gap === by_hand.rim_gap
    @test VgtkCore.estimate(raw.fit) === green && VgtkCore.inlier_mask(raw.fit) == VgtkCore.inlier_mask(by_hand.fit)
    @test all(raw.axes_scale .≈ 2 .* by_hand.axes_scale) && all(raw.center_scale .≈ 2 .* by_hand.center_scale)   # raw px: the cells' scale 2
    # a mosaic is read as its green cells, and frames may stream; a cell image of another mode is refused
    @test detect_image_circle(Xoshiro(1), (m for m in mosaics); outlier_halfwidth = a / 2).circle === green
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [LinearRaw(m; mode = :bin2) for m in mosaics]; outlier_halfwidth = a / 2)
    @test detect_image_circle(Xoshiro(1), VgtkCore.Circle, cells; outlier_halfwidth = a).circle isa VgtkCore.Circle{Float64}
    # one kind of frame, one sensor window
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [cells[1], planes[2]]; outlier_halfwidth = a)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [planes[1], cells[2]]; outlier_halfwidth = a)
    shifted = LinearRaw(RawMosaic(round.(UInt16, frames[2] .* 65535), UInt8[0 1; 3 2], (0.0, 0.0, 0.0, 0.0), 65535.0, (0, 0), (H + 2, W), md;
                                  window_origin = (2, 0)); mode = :green_cells)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [cells[1], shifted]; outlier_halfwidth = a)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [:not_an_image]; outlier_halfwidth = a)
end

@testitem "detect_image_circle on RAW frames on the device: the host result, bit for bit" setup = [ImageCircleFixtures] begin
    using Random: Xoshiro
    using Adapt: adapt
    using CUDA: CUDA
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    using VisualGeometryToolkitCore: RawMosaic, LinearRaw
    if CUDA.functional()
        W, H, c, r = 360, 240, (181.3, 118.7), 100.0
        a = 0.1 * hypot(W, H)
        md = (; model = "synthetic")
        host = [RawMosaic(round.(UInt16, f .* 65535), UInt8[0 1; 3 2], (0.0, 0.0, 0.0, 0.0), 65535.0, (0, 0), (H, W), md) for f in rim_frames(W, H, c, r)]
        device = [adapt(CUDA.CuArray, m) for m in host]
        @test LinearRaw(first(device); mode = :green_cells).image isa CUDA.CuArray   # the cell kernel ran on the device
        h = detect_image_circle(Xoshiro(1), host; outlier_halfwidth = a)
        d = detect_image_circle(Xoshiro(1), device; outlier_halfwidth = a)
        @test d.circle === h.circle && d.contour == h.contour && d.rim_gap === h.rim_gap && d.cut == h.cut
        d2 = detect_image_circle(Xoshiro(1), (LinearRaw(m; mode = :green_cells) for m in device); outlier_halfwidth = a)
        @test d2.circle === h.circle
    else
        @test_skip CUDA.functional()                         # no functional GPU here: recorded as Broken
    end
end


@testitem "detect_image_circle: the extension hooks name the package to load" begin
    # the fit and its covariance are a package extension (VisualGeometryToolkitCore's `require_extension`): without it
    # loaded the call stops before any work with an ArgumentError naming the package; with it loaded, a fallback reached
    # by arguments no method of the extension takes raises their MethodError (a loaded extension cannot be unloaded, so
    # the not-loaded message is checked on an extension that does not exist)
    using Test
    import VisualGeometryToolkitEstimation
    import VisualGeometryToolkitFeatures as VgtkFeatures
    import VisualGeometryToolkitCore as VgtkCore
    @test Base.get_extension(VgtkFeatures, :VisualGeometryToolkitFeaturesEstimationExt) !== nothing
    @test_throws MethodError VgtkFeatures._fit_image_circle(Int, Float64[]; outlier_halfwidth = 1.0, rng = nothing)
    @test_throws MethodError VgtkFeatures._coverage_covariance(1, Float64[])
    e = try
        VgtkCore.require_extension(() -> VgtkFeatures._missing_extension("SomePackage", "a part"), VgtkFeatures, :NoSuchExtension)
    catch err
        err
    end
    @test e isa ArgumentError && occursin("a part needs `using SomePackage`", e.msg)
end

# The rim's measurement on rims of known outer edge and knee (RimFixtures), 640 × 480 frames, the default band (from a
# fifth of the half diagonal, 80 px, to the corners).
@testitem "detect_image_circle: the coverage of a soft ramp (clutter, saturation, noise) and of a sharp stop" setup = [RimFixtures] begin
    using Random: Xoshiro
    using Statistics: median
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, re, rk = 640, 480, (321.3, 240.7), 150.0, 120.0
    ramp = detect_image_circle(Xoshiro(1), rim_synth(W, H, c, re, rk; sat_at = 95.0, sat_radius = 20); outlier_halfwidth = 50)
    rim = ramp.rim
    @test all(<=(1 / rim.radii[end] + 1e-12), diff(log.(rim.radii)))          # Δ(ln r) ≤ 1 / r_max: ≤ 1 px at the band's end
    @test sum(rim.bin_samples) >= 2π * last(rim.radii) && minimum(rim.bin_samples) >= 3   # ≤ 1 px of arc; ≥ 3 rows a bin
    @test count(==(:saturated), rim.bin_reason) > 0                           # the saturated patch's bins, marked
    # the largest log drop of a linear ramp to the floor is at its foot: the coverage is where the light ends, within the
    # drop's 2 px span inside the edge (the span's middle; the floor's level, 1/80 of the interior, sets the foot)
    cov = ramp.circle
    @test hypot(cov.center[1] - c[1], cov.center[2] - c[2]) < 0.25 && re - 4 < cov.a <= re && re - 4 < cov.b <= re
    @test all(s -> 0 < s < 0.25, ramp.axes_scale) && all(s -> 0 < s < 0.25, ramp.center_scale)
    # the chosen edges agree with the coverage; the scene's clutter (dark discs inside the knee) loses the vote
    @test median(abs.(filter(!isnan, rim.edge) .- (cov.a + cov.b) / 2)) < 1.5      # the spans' middles on the 1-px grid
    @test count(==(:ok), rim.reason) >= 0.9 * length(rim.angles)
    sharp = detect_image_circle(Xoshiro(1), rim_synth(W, H, c, re, re; sat_at = 95.0, sat_radius = 20); outlier_halfwidth = 50)
    # a sharp stop: the drop is the stop's 1-px partial band, the coverage within a pixel of the stop
    @test hypot(sharp.circle.center[1] - c[1], sharp.circle.center[2] - c[2]) < 0.25 && abs(sharp.circle.a - re) < 1 && abs(sharp.circle.b - re) < 1
    # a fixed angle count, and an initial model of the caller's (a principal point, a radius whose band holds the rim)
    fixed = detect_image_circle(Xoshiro(1), rim_synth(W, H, c, re, re); outlier_halfwidth = 50, rim_angles = 72)
    @test length(fixed.rim.angles) == 72 && abs(fixed.circle.a - re) < 1
    init = VgtkFeatures.Ellipse(VgtkFeatures.Point2(318.0, 243.0), 200.0, 200.0, 0.0)
    given = detect_image_circle(Xoshiro(1), rim_synth(W, H, c, re, re); outlier_halfwidth = 50, initial = init, rim_band = (0.5, 0.2))
    @test given.rim.initial == init && abs(given.circle.a - re) < 1
end

@testitem "detect_image_circle: the rim under a sensor model (RAW frames, Uncertain cells)" setup = [RimFixtures] begin
    # a Poisson-Gaussian sensor (gain 2 e⁻/code, read noise 2 codes, black 512, white 16383) photographing the ramp with a
    # DARK floor: read as LinearRaw cells with the model, each bin's noise is the model's (the maximum of the frames'
    # noise); read without it, the data's. The result records which; the coverage agrees.
    using Random: Xoshiro
    using Statistics: median
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitCore: RawMosaic, LinearRaw, sensor_fixture
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, re, rk = 640, 480, (321.3, 240.7), 150.0, 120.0
    K, σ0, black, white = 2.0, 2.0, 512.0, 16383.0
    md = (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing,
          iso = nothing, model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = "")
    rng = Xoshiro(3)
    mosaics = map(rim_synth(W, H, c, re, rk; σ = 0.0, fl = 0.0, sat_at = 95.0, sat_radius = 20)) do f
        e = f .* (white - black) .* K
        codes = black .+ (e .+ sqrt.(e) .* randn(rng, size(e))) ./ K .+ σ0 .* randn(rng, size(e))
        RawMosaic(round.(UInt16, clamp.(codes, 0, white)), UInt8[0 1; 3 2], (black, black, black, black), white, (0, 0), (H, W), md)
    end
    model = sensor_fixture(; gain = K, read_noise = (σ0, σ0, σ0, σ0)).model
    m = detect_image_circle(Xoshiro(1), [LinearRaw(x; mode = :green_cells, sensor_model = model) for x in mosaics]; outlier_halfwidth = 25)
    e = detect_image_circle(Xoshiro(1), [LinearRaw(x; mode = :green_cells) for x in mosaics]; outlier_halfwidth = 25)
    @test m.rim.statistics === VgtkFeatures.SensorModelNoise() && e.rim.statistics === VgtkFeatures.EmpiricalNoise()
    @test any(isfinite, m.rim.profile_scale) && all(isnan, e.rim.profile_scale)
    # the model against the data: the empirical noise (differences 2 px apart) over the model's, recorded per angle
    # (the bilinear sampling's averaging and the maximum's law under light move it from 1: a check, reported)
    @test 0.2 < median(filter(isfinite, m.rim.model_scatter)) < 2 && all(isnan, e.rim.model_scatter)
    # the coverage in raw pixels at the ramp's foot, to the cells' grid: one cell (2 raw px) either way, plus the drop's
    # 2-cell span (4 raw px) inward
    @test hypot(m.circle.center[1] - c[1], m.circle.center[2] - c[2]) < 1 && re - 6 < min(m.circle.a, m.circle.b) && max(m.circle.a, m.circle.b) <= re + 2
    @test abs(m.circle.a - e.circle.a) < 1 && abs(m.circle.b - e.circle.b) < 1
end

@testitem "detect_image_circle: the rim's kernels on the device (GPU)" setup = [RimFixtures] begin
    using Random: Xoshiro
    using CUDA: CUDA
    using KernelAbstractions: CPU
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    if CUDA.functional()
        W, H, c = 640, 480, (321.3, 240.7)
        frames = rim_synth(W, H, c, 150.0, 120.0; sat_at = 95.0, sat_radius = 20)
        kw = (; outlier_halfwidth = 50, rim_angles = 360)
        h = detect_image_circle(Xoshiro(1), frames; kw..., device = CPU())
        d = detect_image_circle(Xoshiro(1), frames; kw..., device = CUDA.CUDABackend())
        # the same warp and median kernels: the bins agree to Float32 rounding of the warp's math, the reasons exactly
        @test d.rim.bin_reason == h.rim.bin_reason
        @test isequal(isnan.(d.rim.profile), isnan.(h.rim.profile))
        @test maximum(abs, filter(!isnan, d.rim.profile .- h.rim.profile)) < 1e-5
        @test abs(d.circle.a - h.circle.a) < 0.05 && abs(d.circle.center[1] - h.circle.center[1]) < 0.05
    else
        @test_skip CUDA.functional()                         # no functional GPU here: recorded as Broken
    end
end

@testitem "detect_image_circle: the coverage's uncertainty at a full-frame radius" begin
    # the axes' σ through Core's ellipse chart (retract_jacobian of EllipseManifold) at a full-frame radius, where the
    # form's entries are ~1/a² (3e-7 at a = 1800 px): a finite-difference Jacobian with an absolute 1e-6 step made the form
    # indefinite and the eigenvalue's square root a DomainError on the 14 mm (Features 0.9.3's candidate, 2026-11-14)
    using Random: Xoshiro
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    e = VgtkCore.Ellipse(VgtkCore.Point2(2133.7, 1425.0), 1800.0, 1700.0, 0.3)
    rng = Xoshiro(2)
    σ = 0.5
    pts = map(range(0, 2π; length = 401)[1:400]) do t
        s, c = sincos(e.θ)
        p = VgtkCore.Point2(e.center[1] + e.a * cos(t) * c - e.b * sin(t) * s, e.center[2] + e.a * cos(t) * s + e.b * sin(t) * c)
        p + σ * VgtkCore.Point2(randn(rng), randn(rng))
    end
    u = VgtkFeatures._coverage_uncertainty(e, pts)
    # the centre's σ ≈ σ·√(2/n) and the axes' ≈ σ·√(2/n)·√2 (the isotropic-noise circle's), to a factor 2
    @test all(s -> 0.5 * σ * sqrt(2 / 400) < s < 2 * σ * sqrt(4 / 400), u.center_scale)
    @test all(s -> isfinite(s) && 0.5 * σ * sqrt(2 / 400) < s < 2 * σ * sqrt(4 / 400), u.axes_scale)
end

@testitem "detect_image_circle: a disc centred on the image centre (the log-polar origin): no ties on the radial grid" setup = [ImageCircleFixtures] begin
    # every angle's edge is at the same grid radius when the disc is centred on the origin: the candidates are refined to
    # the log drop's sub-sample peak, so the fit has a residual and the coverage its uncertainty (an exact circle at the
    # origin with rms 0 was the 14 mm's failure, 2026-11-14)
    using Random: Xoshiro
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, r = 360, 240, 100.0
    res = detect_image_circle(Xoshiro(1), disc_frames(W, H, (W / 2, H / 2), r, 3); outlier_halfwidth = 0.1 * hypot(W, H))
    @test res.fit.rms > 0 && all(s -> s > 0, res.axes_scale)
    @test hypot(res.circle.center[1] - W / 2, res.circle.center[2] - H / 2) < 0.1 && abs(res.circle.a - r) < 0.5
    @test count(==(:ok), res.rim.reason) >= 0.9 * length(res.rim.angles)
end

@testitem "detect_image_circle: a ramp centred exactly on the image centre, at any radius (no ring of the sampling grid)" setup = [RimFixtures] begin
    # the 14 mm's failure (Features 0.9.3's candidate, 2026-11-14): a drop measured over "the samples to ≥ 2 px" changed
    # its sample count where the log grid's spacing crosses 2/n px (Δr = 2/3 px at r ≈ ⅔·r_max), and every wedge locked
    # onto that radius about ANY origin. The drop is now across exactly 2 px at every radius (the log-polar map's inverse
    # and differential). A 14 mm-like ramp (13 % of the radius wide, its foot a soft tail, an APS-C rim cut by the frame)
    # centred EXACTLY on the default start, its foot at, inside and beyond the old crossover: the coverage follows the
    # rim, one offset from it at every radius (the span-count rule measured −0.1, −4.0 and +1.9 px at 0, 4 and 8 px past
    # the crossover: a ring at the crossover, and this item fails 10 of its 28 tests), and the start (sub-pixel, ±25 px) moves nothing. The linear ramp (a hard
    # foot) at the crossovers Δr = 2/3 and 1/2 px is at its foot as the soft ramp item's.
    using Random: Xoshiro
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H = 800, 600
    c = (W / 2, H / 2)
    rmax = hypot(W, H) / 2
    offsets = map((0.0, 4.0, 8.0)) do d
        re = 2rmax / 3 + d
        frames = rim_synth(W, H, c, re, 0.87re; sat_radius = 0, foot = 4.0, σ = 0.001)
        r0 = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = 50)
        cov = r0.circle
        @test hypot(cov.center[1] - c[1], cov.center[2] - c[2]) < 0.5
        # the start moves the coverage within its own uncertainty (3σ, plus a quarter pixel)
        σx, σy = r0.center_scale
        σa = maximum(r0.axes_scale)
        for off in ((0.3, -0.4), (25.0, -15.0), (-25.0, 15.0))
            init = VgtkFeatures.Ellipse(VgtkFeatures.Point2(c[1] + off[1], c[2] + off[2]), rmax, rmax, 0.0)
            r1 = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = 50, initial = init).circle
            @test abs(r1.center[1] - cov.center[1]) <= 3σx + 0.25 && abs(r1.center[2] - cov.center[2]) <= 3σy + 0.25
            @test abs(max(r1.a, r1.b) - max(cov.a, cov.b)) <= 3σa + 0.25 && abs(min(r1.a, r1.b) - min(cov.a, cov.b)) <= 3σa + 0.25
        end
        (cov.a + cov.b) / 2 - re
    end
    @test maximum(offsets) - minimum(offsets) < 1.5
    for re in (2rmax / 3, rmax / 2)
        res = detect_image_circle(Xoshiro(1), rim_synth(W, H, c, re, 0.87re; sat_radius = 0); outlier_halfwidth = 50, rim_angles = 180)
        cov = res.circle
        @test hypot(cov.center[1] - c[1], cov.center[2] - c[2]) < 0.25
        @test re - 4 < min(cov.a, cov.b) && max(cov.a, cov.b) <= re + 0.5
        @test res.fit.rms > 0 && all(s -> 0 < s < 0.25, res.axes_scale)
    end
end
