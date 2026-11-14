# Opt-in, real data: `detect_image_circle` on the paper's image-circle sets, every frame RAW through Features' one RAW
# path (green cells; the result in raw px). Skipped unless the tiers are under VGTK_DATA_ROOT (dgx01's
# /raid/data/jpr/rc-vgtk-root). The regression values are this method's on 2026-11-14 (Features 0.9.3's candidate, dgx01
# /raid/data/jpr/imgcircle/study2_wip.tsv at 03e48b7e); each tolerance is 3σ of the fit's own uncertainty (`axes_scale`,
# `center_scale`) plus one green cell (2 raw px, the radial grid the edges are sampled on), each axis on its own.
# Cross-checks: RC's MEASURED hull of the same frames (the run logs of 2026-11-13), each axis within 1 %; the synthetic
# field stop's truth to the lit pixels' outer boundary (its item).

@testitem "detect_image_circle on the fisheye's f/4, f/22 and f/32 sets and the 14 mm, and its start invariance (real data, opt-in)" begin
    using Random: Xoshiro
    using LibRaw_jll: LibRaw_jll                               # the RAW decoder (Features' LibRaw extension)
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    using Test
    pd = joinpath(get(ENV, "VGTK_DATA_ROOT", ""), "paper-data")
    sets = (
        # name, files, reference (cx, cy, a, b) raw px, RC's measured hull (a, b) raw px (nothing: no trusted hull)
        ("fisheye f/4 (DT 8mm F4 SAM wide open)", [joinpath(pd, "defocus/fisheye/siemens-stars/$n.ARW") for n in ("DSC04845", "DSC04846", "DSC04847", "DSC04848", "DSC04853")],
         (2123.150, 1428.695, 1086.709, 1076.045), (1084.69, 1081.40)),
        ("fisheye f/22 (AprilGrid 10 cm)", [joinpath(pd, "geometric-accuracy/fisheye/april-grid/$n.ARW") for n in ("DSC06999", "DSC07081", "DSC07182", "DSC07283", "DSC07365")],
         (2136.817, 1435.989, 1339.845, 1316.823), (1334.27, 1320.53)),
        ("fisheye f/32 (Siemens stars; the bottom's rim hidden by the camera mount)", [joinpath(pd, "geometric-accuracy/fisheye/siemens-stars/$n.ARW") for n in ("DSC03644", "DSC03674", "DSC03704", "DSC03734", "DSC03764")],
         # the owner-verified coverage (2026-11-14: this detector's fit, inspected on the rim incl. the mount side; the
         # earlier reference 2136.266, 1435.866, 1334.820, 1312.278 sat 3.5 px high)
         (2136.997, 1439.331, 1333.768, 1315.245), (1339.39, 1311.72)),
        ("14 mm (APS-C on the full frame: a ~230 px ramp)", [joinpath(pd, "vignetting/14mm/diffuser/DSC04161.ARW")],
         (2131.650, 1425.813, 1718.921, 1717.064), nothing),
    )
    if !all(s -> all(isfile, s[2]), sets)
        @test_skip "the paper-data tier is not under VGTK_DATA_ROOT ($(repr(get(ENV, "VGTK_DATA_ROOT", ""))))"
    else
        for (name, files, ref, hull) in sets
            res = detect_image_circle(Xoshiro(1), [load_raw(f; mode = :green_cells) for f in files]; outlier_halfwidth = 500)
            c = res.circle
            σa, σb = res.axes_scale
            σx, σy = res.center_scale
            @info "detect_image_circle (real data)" name c res.axes_scale res.center_scale bins = length(res.rim.angles) support = count(==(:ok), res.rim.reason)
            # the axes in either order (θ may name either one a)
            A, B = max(c.a, c.b), min(c.a, c.b)
            Ra, Rb = max(ref[3], ref[4]), min(ref[3], ref[4])
            @test abs(c.center[1] - ref[1]) <= 3σx + 2
            @test abs(A - Ra) <= 3max(σa, σb) + 2 && abs(B - Rb) <= 3max(σa, σb) + 2
            @test abs(c.center[2] - ref[2]) <= 3σy + 2
            # RC's measured hull, each axis within 1 % (a sanity bound: the hull is the darkest MSER level's boundary, the
            # coverage the outer step's steepest log drop, 3–8 px apart on these sets)
            hull === nothing || @test abs(A - max(hull...)) <= 0.01 * max(hull...) && abs(B - min(hull...)) <= 0.01 * min(hull...)
        end
        # the default start, exactly the frame centre (the log-polar origin), against a start 29 px off it: the same
        # coverage within 3σ plus one green cell (a candidate rule tied to the sampling grid made a ring at a fixed
        # radius about ANY origin and the default start followed it: 2128.0, 1424.0, r 1707.4 on the 14 mm, 2026-11-14)
        cells = [load_raw(f; mode = :green_cells) for f in sets[4][2]]
        r0 = detect_image_circle(Xoshiro(1), cells; outlier_halfwidth = 500)
        H, W = 2 .* size(first(cells).saturated)
        init = VisualGeometryToolkitFeatures.Ellipse(VisualGeometryToolkitFeatures.Point2(W / 2 + 25, H / 2 - 15), hypot(W, H) / 2, hypot(W, H) / 2, 0.0)
        r1 = detect_image_circle(Xoshiro(1), cells; outlier_halfwidth = 500, initial = init)
        σx, σy = max.(r0.center_scale, r1.center_scale)
        σa = maximum((r0.axes_scale..., r1.axes_scale...))
        @test abs(r0.circle.center[1] - r1.circle.center[1]) <= 3σx + 2 && abs(r0.circle.center[2] - r1.circle.center[2]) <= 3σy + 2
        @test abs(max(r0.circle.a, r0.circle.b) - max(r1.circle.a, r1.circle.b)) <= 3σa + 2 &&
              abs(min(r0.circle.a, r0.circle.b) - min(r1.circle.a, r1.circle.b)) <= 3σa + 2
    end
end

@testitem "detect_image_circle on the synthetic field stop: the truth (opt-in)" begin
    using Random: Xoshiro
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation
    using Test
    dir = joinpath(get(ENV, "VGTK_DATA_ROOT", ""), "geometric_accuracy/detector_bias/data/v8/circle_field_stop")
    if !isfile(joinpath(dir, "field_stop.json"))
        @test_skip "the synthetic tier is not under VGTK_DATA_ROOT"
    else
        # field_stop.json: centre (2141.0, 1442.1), radius 1309.6 (COLMAP px), a flat field at 0.5 inside, 0 outside
        frames = [Float32.(load_linear_image(joinpath(dir, "F$k.png"))[:, :, 2]) for k in 1:3]
        res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = 1000)
        c = res.circle
        @test hypot(c.center[1] - 2141.0, c.center[2] - 1442.1) < 0.5
        # the frames are a noise-free BINARY disc (a pixel is lit when its centre is inside): where its light ends is the
        # outer boundary of the lit pixels, between the continuous stop and the stop + half a pixel's diagonal (√2/2 px,
        # the farthest a lit pixel's outer corner reaches beyond the circle), to the fit's own uncertainty
        for (axis, σ) in zip((c.a, c.b), res.axes_scale)
            @test 1309.6 - σ <= axis <= 1309.6 + sqrt(2) / 2 + σ
        end
    end
end
