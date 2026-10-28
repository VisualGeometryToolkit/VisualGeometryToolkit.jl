# Schöps et al.'s camera_calibration files (intrinsics0.yaml, dataset.bin),
# read through the typed `load` — fixtures in test/fixtures/schoeps/.
#
# The models are checked against an INDEPENDENT evaluation of the reference's
# B-spline surface (the four cubic weights of generic_models' util.h written
# out, on the grid parsed here straight from the YAML text): the reader's grid
# layout and the model's evaluation agree with the reference at every pixel.

@testsetup module SchoepsReference
    export schoeps_raw_grid, schoeps_surface, schoeps_grid_coordinate, FIXTURES

    const FIXTURES = joinpath(@__DIR__, "fixtures", "schoeps")

    # the `gw × gh` grid under `tag`, as the reference stores it: entry y·gw + x
    function schoeps_raw_grid(path, tag, gw, gh)
        txt = read(path, String)
        body = match(Regex("^$(tag)\\s*:\\s*\\[(.*?)\\]", "ms"), txt).captures[1]
        nums = parse.(Float64, split(body, r"[,\s]+"; keepempty = false))
        return [nums[(3 * (y * gw + x) + 1):(3 * (y * gw + x) + 3)] for x in 0:(gw - 1), y in 0:(gh - 1)]   # [x+1, y+1]
    end
    # util.h's uniform cubic weights at the local coordinate x ∈ [3, 4)
    bfac(x) = (-1 / 6 * (x - 4)^3, 1 / 2 * x^3 - 11 / 2 * x^2 + 39 / 2 * x - 131 / 6, -1 / 2 * x^3 + 5 * x^2 - 16 * x + 50 / 3, 1 / 6 * (x - 3)^3)
    # EvalUniformCubicBSplineSurface at the grid coordinate (gx, gy) ∈ [1, gw − 2] × [1, gh − 2]
    function schoeps_surface(cp, gx, gy)
        x = gx + 2; y = gy + 2
        ix = min(floor(Int, x), size(cp, 1) - 1); iy = min(floor(Int, y), size(cp, 2) - 1)
        ax, bx, cx, dx = bfac(x - (ix - 3))
        at(i, j) = cp[i + 1, j + 1]
        rows = [ax * at(ix - 3, iy - 4 + k) + bx * at(ix - 2, iy - 4 + k) + cx * at(ix - 1, iy - 4 + k) + dx * at(ix, iy - 4 + k) for k in 1:4]
        ay, by, cy, dy = bfac(y - (iy - 3))
        return ay * rows[1] + by * rows[2] + cy * rows[3] + dy * rows[4]
    end
    # pixel (corner convention) → grid coordinate, the reference's PixelCornerConvToGridPoint
    schoeps_grid_coordinate(px, py; gw, gh, cminx, cminy, cmaxx, cmaxy) =
        (1 + (gw - 3) * (px - cminx) / (cmaxx + 1 - cminx), 1 + (gh - 3) * (py - cminy) / (cmaxy + 1 - cminy))
end

@testitem "schoeps: central intrinsics0.yaml" setup = [VGIoFixtures, SchoepsReference] begin
    using VisualGeometryToolkitCore: backproject, CentralGenericModel, NoncentralGenericModel
    using VisualGeometryToolkitIO: SCHOEPS_PIXEL_FRAME
    path = joinpath(FIXTURES, "central_intrinsics0.yaml")
    cam = load(CentralGenericModel, path)
    @test cam isa CentralGenericModel{Float64}
    @test (cam.width, cam.height) == (64, 48)
    @test (cam.calib_min_x, cam.calib_min_y, cam.calib_max_x, cam.calib_max_y) == (4, 3, 59, 44)
    @test size(cam.grid) == (4, 5)                       # [gy, gx]
    @test SCHOEPS_PIXEL_FRAME === PixelFrame{:colmap}     # the corner convention, read as stored

    cp = schoeps_raw_grid(path, "grid", 5, 4)
    for gy in 1:4, gx in 1:5                             # the layout: entry y·gw + x lands at [gy, gx]
        @test collect(cam.grid[gy, gx]) ≈ cp[gx, gy] ./ norm(cp[gx, gy]) atol = 1.0e-14
    end
    # The model at pixels across the calibrated area against the reference's surface.
    worst = 0.0
    for px in range(4.0, 59.9; length = 9), py in range(3.0, 44.9; length = 7)
        gx, gy = schoeps_grid_coordinate(px, py; gw = 5, gh = 4, cminx = 4, cminy = 3, cmaxx = 59, cmaxy = 44)
        ref = schoeps_surface(cp, gx, gy); ref = ref ./ norm(ref)
        f = backproject(cam, SVector(px, py))
        worst = max(worst, norm(collect(f) - ref))
    end
    @test worst < 1.0e-12

    # The typed load is an assertion of intent, and the YAML states no lengths.
    @test_throws ArgumentError load(NoncentralGenericModel, path; file_unit = m)
    @test_throws ArgumentError load(CentralGenericModel, path; file_unit = m)
    # A vgio round trip of the YAML-read model is exact.
    mktempdir() do dir
        out = joinpath(dir, "cam.json")
        save(out, cam)
        @test load(CentralGenericModel, out).grid == cam.grid
        @test_throws ArgumentError load(CentralGenericModel, out; file_unit = m)
    end
end

@testitem "schoeps: noncentral intrinsics0.yaml" setup = [VGIoFixtures, SchoepsReference] begin
    using VisualGeometryToolkitCore: backproject, NoncentralGenericModel
    path = joinpath(FIXTURES, "noncentral_intrinsics0.yaml")
    @test_throws ArgumentError load(NoncentralGenericModel, path)              # the unit is the caller's to state
    @test_throws ArgumentError load(NoncentralGenericModel, path; file_unit = px)
    cam_m = load(NoncentralGenericModel, path; file_unit = m)
    cam_mm = load(NoncentralGenericModel, path; file_unit = mm)
    P = schoeps_raw_grid(path, "point_grid", 5, 4)
    D = schoeps_raw_grid(path, "direction_grid", 5, 4)
    @test collect(cam_m.points[2, 3]) == P[3, 2]                               # metres as stored
    @test collect(cam_mm.points[2, 3]) ≈ P[3, 2] ./ 1000 rtol = 1.0e-15        # millimetres read into metres
    worst_o = 0.0; worst_d = 0.0
    for px in range(4.0, 59.9; length = 7), py in range(3.0, 44.9; length = 5)
        gx, gy = schoeps_grid_coordinate(px, py; gw = 5, gh = 4, cminx = 4, cminy = 3, cmaxx = 59, cmaxy = 44)
        o = schoeps_surface(P, gx, gy); d = schoeps_surface(D, gx, gy); d = d ./ norm(d)
        ray = backproject(cam_m, SVector(px, py))
        worst_d = max(worst_d, norm(collect(ray.direction) - d))
        # the ray's origin is a point on the reference's line (the model may slide it along the line)
        δ = collect(ray.origin) - o
        worst_o = max(worst_o, norm(δ - (δ' * d) * d))
    end
    @test worst_d < 1.0e-12
    @test worst_o < 1.0e-12
end

@testitem "schoeps: dataset.bin to TargetPoints2D" setup = [VGIoFixtures, SchoepsReference] begin
    using VisualGeometryToolkitCore: TargetPoints2D, convert_frame
    using VisualGeometryToolkitIO: SCHOEPS_PIXEL_FRAME
    path = joinpath(FIXTURES, "dataset.bin")
    @test_throws ArgumentError load(TargetPoints2D, path)                     # the target is the caller's to state
    obs = load(TargetPoints2D, path; target = :pattern)
    @test obs isa TargetPoints2D{SCHOEPS_PIXEL_FRAME}
    @test sort(collect(keys(obs))) == [1, 2]
    @test obs.images == Dict(1 => "image000.png", 2 => "image001.png")
    @test [(p.target, p.index, p.xy, p.weight) for p in obs[1]] ==
        [(:pattern, 1, SVector(0.5, 0.5), 1.0), (:pattern, 8, SVector(10.25, 20.75), 1.0), (:pattern, 42, SVector(63.5, 47.5), 1.0)]
    @test only(obs[2]).xy == SVector(32.0, 24.0) && only(obs[2]).index == 4
    # The corner convention's (0.5, 0.5) is the top-left pixel's centre: (0, 0) in OpenCV's frame.
    @test first(convert_frame(PixelFrame{:opencv}, obs)[1]).xy == SVector(0.0, 0.0)
    # Not a dataset.bin, and not a target beside a vgio file.
    mktempdir() do dir
        out = joinpath(dir, "points.json")
        @test_throws ArgumentError load(TargetPoints2D, joinpath(FIXTURES, "central_intrinsics0.yaml"); target = :pattern)
        bad = joinpath(dir, "v1.bin"); b = read(path); b[14] = 0x01; write(bad, b)     # version 1
        @test_throws ArgumentError load(TargetPoints2D, bad; target = :pattern)
        # a corrupt length or a truncated file: an ArgumentError naming it, never a short read or an EOFError
        b = read(path)
        nname = Int(ntoh(reinterpret(UInt32, b[31:34])[1]))         # image set 1's name length (big-endian), after the 30-byte header
        nf_at = 35 + nname                                            # its feature count
        for (desc, c) in (
                "truncated in the features" => b[1:(nf_at + 3 + 6)],
                "truncated in the header" => b[1:20],
                "a name length past the end" => (c = copy(b); c[31:34] .= 0x7f; c),
                "a feature count past the end" => (c = copy(b); c[nf_at:(nf_at + 3)] .= 0x7f; c),
            )
            p2 = joinpath(dir, "bad.bin"); write(p2, c)
            e = try load(TargetPoints2D, p2; target = :pattern); nothing catch e; e end
            @test e isa ArgumentError && occursin("calib_data", e.msg)
        end
    end
end

@testitem "content dispatch: a UTF-8 byte-order mark is not content" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: NoncentralGenericModel
    # a vgio file an editor saved with a BOM is read as the vgio file it is, not sent to the Schöps YAML parser
    src = joinpath(@__DIR__, "fixtures", "units", "generic_camera.json")
    mktempdir() do dir
        p = joinpath(dir, "bom.json")
        write(p, vcat(UInt8[0xef, 0xbb, 0xbf], read(src)))
        @test VisualGeometryToolkitIO._file_form(p) === :json
        a, b = load(NoncentralGenericModel, p), load(NoncentralGenericModel, src)
        @test a.directions == b.directions && a.points == b.points
    end
end
