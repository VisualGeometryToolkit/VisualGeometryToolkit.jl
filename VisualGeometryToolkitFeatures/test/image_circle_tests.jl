# The image circle MEASURED from photographs (`detect_image_circle`, moved from RadiometricCamera with these tests):
# synthetic frames of a bright disc of known centre and radius on a dark surround, each frame darkened by a "scene" region
# over a different part of the rim (no single frame shows the whole rim), with noise. The circle comes back to a fraction
# of a pixel from the convex hull of the dark regions' contours. The fit is the VisualGeometryToolkitEstimation extension.

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

@testitem "detect_image_circle: the circle from frames that each show part of the rim" setup = [ImageCircleFixtures] begin
    using Random: Xoshiro
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation   # the fit's extension
    W, H, c, r = 360, 240, (181.3, 118.7), 100.0
    a = 0.1 * hypot(W, H)   # the outlier half-width: a literal of the caller's
    frames = rim_frames(W, H, c, r)
    # no single frame shows the whole rim: a point just inside it is dark in each frame
    rim(φ) = CartesianIndex(round(Int, c[2] + (r - 3) * sin(φ) + 0.5), round(Int, c[1] + (r - 3) * cos(φ) + 0.5))
    @test all(f -> any(φ -> f[rim(φ)] < 0.3, range(0, 2π; length = 73)), frames)

    res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, min_hole_area = 8)          # the default model: an ellipse
    e = res.circle
    @test e isa VgtkCore.Ellipse{Float64}
    # the centre to a fraction of a pixel; the axes ≈ 1 px outside the true rim: the hull follows the darkest-level copy of
    # the surround, on the dark side of the rim's partial-coverage band
    @test hypot(e.center[1] - c[1], e.center[2] - c[2]) < 0.1 && 0 < e.a - r < 1.5 && 0 < e.b - r < 1.5
    rc = detect_image_circle(Xoshiro(1), VgtkCore.Circle, frames; outlier_halfwidth = a, min_hole_area = 8)   # the model type leads, after the RNG
    @test rc.circle isa VgtkCore.Circle{Float64} && rc.contour == res.contour
    @test hypot(rc.circle.center[1] - c[1], rc.circle.center[2] - c[2]) < 0.1 && 0 < rc.circle.r - r < 1.5
    @test detect_image_circle(VgtkCore.Circle, frames; outlier_halfwidth = a, min_hole_area = 8).circle isa VgtkCore.Circle &&
        detect_image_circle(frames; outlier_halfwidth = a, min_hole_area = 8).circle isa VgtkCore.Ellipse
    @test VgtkCore.converged(res.fit) && count(VgtkCore.inlier_mask(res.fit)) == length(res.contour)
    # the contour is the hull's vertices: on pixel corners, within a pixel of the true circle; the scene regions inside the
    # disc do not move it
    @test all(p -> isinteger(p[1]) && isinteger(p[2]) && abs(hypot(p[1] - c[1], p[2] - c[2]) - r) < 1.5, res.contour)
    # `nothing` (as find_circle returns it) when no dark region has a boundary off the frame's border
    @test detect_image_circle([fill(0.8f0, H, W)]; outlier_halfwidth = a, min_hole_area = 8) === nothing
    # frames may stream (a generator)
    @test detect_image_circle(Xoshiro(1), (f for f in frames); outlier_halfwidth = a, min_hole_area = 8).circle == e
    @test_throws DimensionMismatch detect_image_circle([frames[1], frames[1][1:end-1, :]]; outlier_halfwidth = a, min_hole_area = 8)
    @test_throws ArgumentError detect_image_circle(Matrix{Float32}[]; outlier_halfwidth = a, min_hole_area = 8)
    @test res.cut == false
    @test res.rim_gap < π                                                                     # the rim seen all round the centre
end

@testitem "detect_image_circle: a disc cut by the frame (the rim in two arcs)" setup = [ImageCircleFixtures] begin
    # an APS-C lens on the full frame: the lit disc reaches past the top and bottom of the frame; the hull closes the cuts
    # along the border, those vertices are dropped, and the circle comes from the two arcs
    using Random: Xoshiro
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, r = 360, 240, (181.3, 118.7), 140.0
    a = 0.1 * hypot(W, H)
    disc = [cover(u, v, c..., r) for v in 1:H, u in 1:W]
    frames = disc_frames(W, H, c, r, 3)
    @test disc[1, round(Int, c[1])] > 0.9 && disc[H, round(Int, c[1])] > 0.9 && disc[1, 1] == 0      # cut at the top and bottom, corners dark
    res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, min_hole_area = 8)
    @test res !== nothing && res.cut
    e = res.circle
    @test e isa VgtkCore.Ellipse{Float64}                                                        # the model asked for, cut or not
    @test hypot(e.center[1] - c[1], e.center[2] - c[2]) < 0.5 && 0 < e.a - r < 1.5                   # ≈ 1 px outward, as above
    @test detect_image_circle(Xoshiro(1), VgtkCore.Circle, frames; outlier_halfwidth = a, min_hole_area = 8).circle isa VgtkCore.Circle
    @test all(p -> 5 < p[1] < W - 5 && 5 < p[2] < H - 5, res.contour)                           # no vertex within border_margin of the border
    @test any(p -> p[1] < c[1], res.contour) && any(p -> p[1] > c[1], res.contour)              # both arcs
    @test all(p -> abs(hypot(p[1] - c[1], p[2] - c[2]) - r) < 1.5, res.contour)
    @test res.rim_gap < π                                                                     # two opposite arcs span the centre
end

@testitem "detect_image_circle: one arc of rim (a single dark corner) — a mask, extrapolated away from the arc" setup = [ImageCircleFixtures] begin
    # the disc covers three corners of the frame; only the fourth is dark, so the rim is one short arc. The circle is a MASK:
    # returned, its boundary right along the arc, its centre and radius not determined (rim_gap > π says so)
    using Random: Xoshiro
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, r = 360, 240, (120.0, 90.0), 260.0
    a = 0.1 * hypot(W, H)
    one = [Float32(0.01 + 0.75 * cover(u, v, c..., r)) for v in 1:H, u in 1:W]
    res = detect_image_circle(Xoshiro(1), VgtkCore.Circle, [one]; outlier_halfwidth = a, min_hole_area = 8)
    @test res !== nothing && res.circle isa VgtkCore.Circle{Float64}
    @test res.rim_gap > π                                                                     # one arc: the rest is extrapolated
    # along the observed arc the true rim is within 2 px of the fitted boundary
    arc = res.contour[VgtkCore.inlier_mask(res.fit)]
    truth_rim = [(c[1] + r * cos(φ), c[2] + r * sin(φ)) for φ in (atan(p[2] - c[2], p[1] - c[1]) for p in arc)]
    fitted = res.circle
    @test all(q -> abs(hypot(q[1] - fitted.center[1], q[2] - fitted.center[2]) - fitted.r) < 2.0, truth_rim)
    # an ellipse cannot be fitted to it: the arc's hull vertices are a near-straight staircase, so every 5-point sample holds
    # 3 collinear points and the RANSAC finds no model: `nothing`
    @test detect_image_circle(Xoshiro(1), [one]; outlier_halfwidth = a, min_hole_area = 8) === nothing
end

# A hot pixel in the dark surround: it punches a 1-pixel hole in the surround's dark region, whose corners lie far outside
# the rim and past the border margin. Kept, they are hull vertices: the hull bridges past the rim, drops the rim's vertices
# near it and the rim is seen only on one side of the bridge. `min_hole_area` drops the small holes and the circle comes
# back from the rim alone.
@testitem "detect_image_circle: a hot pixel in the dark surround (min_hole_area)" setup = [ImageCircleFixtures] begin
    using Random: Xoshiro
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, r = 360, 240, (181.3, 118.7), 80.0
    a = 0.1 * hypot(W, H)
    frames = disc_frames(W, H, c, r, 3)
    frames[2][30, 40] = 0.9f0                                               # the hot pixel: 30 px from the border, ~90 px off the rim
    off_rim(res) = maximum(p -> abs(hypot(p[1] - c[1], p[2] - c[2]) - r), res.contour)
    kept = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, min_hole_area = 0)   # every hole kept: the failure
    @test off_rim(kept) > 50 && kept.rim_gap > 1
    res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, min_hole_area = 8)
    @test off_rim(res) < 1.5 && res.rim_gap < 0.5
    @test hypot(res.circle.center[1] - c[1], res.circle.center[2] - c[2]) < 0.1 && 0 < res.circle.a - r < 1.5 && 0 < res.circle.b - r < 1.5
    @test_throws UndefKeywordError detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a)            # min_hole_area is required
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, min_hole_area = -1)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = 0, min_hole_area = 8)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, min_hole_area = 8, smoothing_weight = -1)
end

@testitem "detect_image_circle on RAW frames: the green cells, the circle in raw pixels" setup = [ImageCircleFixtures] begin
    # A RAW frame is measured on its green cells (LinearRaw{:green_cells}, the geometry image) and the circle taken to the
    # raw's pixels by the cells' raw_frame_map (Core's similarity action on an Ellipse); the keywords are in green-cell
    # pixels (the resolution measured on: only the result is mapped). The same scene measured on the full-resolution frames
    # agrees to the map's exactness (the green cells average two photosites: the rim's partial-coverage band, ≲ 1 raw px).
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
    full = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, min_hole_area = 8).circle
    # the estimate on the cells' plane, taken to the raw by hand: what the RAW method does
    planes = [L.image[:, :, 1] for L in cells]
    by_hand = detect_image_circle(Xoshiro(1), planes; outlier_halfwidth = a / 2, min_hole_area = 2)
    green = S(by_hand.circle)
    @test green isa VgtkCore.Ellipse{Float64}
    @test hypot(green.center[1] - full.center[1], green.center[2] - full.center[2]) < 1.0
    @test abs(green.a - full.a) < 2.0 && abs(green.b - full.b) < 2.0                                  # one green cell = 2 raw px
    @test hypot(green.center[1] - c[1], green.center[2] - c[2]) < 1.0 && 0 < green.a - r < 3.0
    # the RAW method: the same, bit for bit, with the same keywords (green-cell px: only the result is mapped)
    raw = detect_image_circle(Xoshiro(1), cells; outlier_halfwidth = a / 2, min_hole_area = 2)
    @test raw.circle === green
    @test raw.contour == [VgtkCore.Point2{Float64}(S(SVector{2, Float64}(p[1], p[2]))) for p in by_hand.contour] && raw.cut == by_hand.cut && raw.rim_gap === by_hand.rim_gap
    @test VgtkCore.estimate(raw.fit) === green && VgtkCore.inlier_mask(raw.fit) == VgtkCore.inlier_mask(by_hand.fit)
    # a mosaic is read as its green cells, and frames may stream; a cell image of another mode is refused
    @test detect_image_circle(Xoshiro(1), (m for m in mosaics); outlier_halfwidth = a / 2, min_hole_area = 2).circle === green
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [LinearRaw(m; mode = :bin2) for m in mosaics]; outlier_halfwidth = a / 2, min_hole_area = 2)
    @test detect_image_circle(Xoshiro(1), VgtkCore.Circle, cells; outlier_halfwidth = a, min_hole_area = 8).circle isa VgtkCore.Circle{Float64}
    # one kind of frame, one sensor window
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [cells[1], planes[2]]; outlier_halfwidth = a, min_hole_area = 8)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [planes[1], cells[2]]; outlier_halfwidth = a, min_hole_area = 8)
    shifted = LinearRaw(RawMosaic(round.(UInt16, frames[2] .* 65535), UInt8[0 1; 3 2], (0.0, 0.0, 0.0, 0.0), 65535.0, (0, 0), (H + 2, W), md;
                                  window_origin = (2, 0)); mode = :green_cells)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [cells[1], shifted]; outlier_halfwidth = a, min_hole_area = 8)
    @test_throws ArgumentError detect_image_circle(Xoshiro(1), [:not_an_image]; outlier_halfwidth = a, min_hole_area = 8)
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
        h = detect_image_circle(Xoshiro(1), host; outlier_halfwidth = a, min_hole_area = 8)
        d = detect_image_circle(Xoshiro(1), device; outlier_halfwidth = a, min_hole_area = 8)
        @test d.circle === h.circle && d.contour == h.contour && d.rim_gap === h.rim_gap && d.cut == h.cut
        d2 = detect_image_circle(Xoshiro(1), (LinearRaw(m; mode = :green_cells) for m in device); outlier_halfwidth = a, min_hole_area = 8)
        @test d2.circle === h.circle
    else
        @test_skip CUDA.functional()                         # no functional GPU here: recorded as Broken
    end
end

@testitem "detect_image_circle: the L0 smoothing (the ImageSmooth extension)" setup = [ImageCircleFixtures] begin
    using Random: Xoshiro
    using ImageSmooth: ImageSmooth
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    W, H, c, r = 360, 240, (181.3, 118.7), 100.0
    frames = disc_frames(W, H, c, r, 2)
    a = 0.1 * hypot(W, H)
    # Under --depwarn=error (the release's strict run) this one check runs in a child Julia with --depwarn=yes: ImageSmooth
    # 0.1.1 (its latest release, 2021) calls ImageBase's deprecated `fdiff!` inside `L0Smooth`, a third-party
    # deprecation that the strict run would turn into an error here. Features calls only ImageSmooth's public `smooth!`;
    # every other item stays strict. The child reads the same frames (Float32, H × W each) and prints the circle.
    circle = if Base.JLOptions().depwarn == 2
        mktempdir() do dir
            path = joinpath(dir, "frames.bin")
            open(io -> foreach(f -> write(io, f), frames), path, "w")
            code = """
                using Random: Xoshiro
                using ImageSmooth: ImageSmooth
                using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
                using VisualGeometryToolkitFeatures: detect_image_circle
                H, W, n = parse.(Int, ARGS[2:4])
                frames = open(io -> [read!(io, Matrix{Float32}(undef, H, W)) for _ in 1:n], ARGS[1])
                res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = $a, min_hole_area = 8, smoothing_weight = 5e-4)
                println(join(repr.((res.circle.center[1], res.circle.center[2], res.circle.a)), " "))
                """
            cmd = `$(Base.julia_cmd()) --depwarn=yes --startup-file=no --project=$(Base.active_project()) -e $code $path $H $W $(length(frames))`
            parse.(Float64, split(readchomp(cmd)))
        end
    else
        res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = a, min_hole_area = 8, smoothing_weight = 5e-4)
        (res.circle.center[1], res.circle.center[2], res.circle.a)
    end
    # on a sharp rim the smoothing does not move the circle (≈ 1 px outward, as unsmoothed)
    @test hypot(circle[1] - c[1], circle[2] - c[2]) < 0.5 && 0 < circle[3] - r < 2.0
end

@testitem "detect_image_circle: the extension hooks name the package to load" begin
    # the fit and the smoothing are package extensions (VisualGeometryToolkitCore's `require_extension`): without one
    # loaded the call stops before any work with an ArgumentError naming the package; with it loaded, a fallback reached
    # by arguments no method of the extension takes raises their MethodError (a loaded extension cannot be unloaded, so
    # the not-loaded message is checked on an extension that does not exist)
    using Test
    import VisualGeometryToolkitEstimation, ImageSmooth   # both extensions load
    import VisualGeometryToolkitFeatures as VgtkFeatures
    import VisualGeometryToolkitCore as VgtkCore
    @test Base.get_extension(VgtkFeatures, :VisualGeometryToolkitFeaturesEstimationExt) !== nothing
    @test Base.get_extension(VgtkFeatures, :VisualGeometryToolkitFeaturesImageSmoothExt) !== nothing
    @test_throws MethodError VgtkFeatures._fit_image_circle(Int, Float64[]; outlier_halfwidth = 1.0, rng = nothing)
    @test_throws MethodError VgtkFeatures._l0_smooth!(nothing, nothing, 1)
    e = try
        VgtkCore.require_extension(() -> VgtkFeatures._missing_extension("SomePackage", "a part"), VgtkFeatures, :NoSuchExtension)
    catch err
        err
    end
    @test e isa ArgumentError && occursin("a part needs `using SomePackage`", e.msg)
end
