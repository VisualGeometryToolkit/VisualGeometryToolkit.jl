# =============================================================================
# An old central grid body EMBEDDED in another record
# =============================================================================
#
# RadiometricCamera's `rc/calibration_state/v1` stored its camera as the old
# Schöps/vgio central grid body (the old central type token, the pixel
# lattice's `direction_grid` and `present`, a nominal pinhole, a domain) with no
# schema of its own. `camera_from_json` reads such a body through the one
# conversion every old central grid takes (`_central_grid_body`, migrate.jl), so
# it is the camera the whole-file migration of the same body gives, bit for bit.
#
# fixtures/old/calibration_state_camera_body.json is `camera.camera` of
# rc_synthetic_2026-11-10's
# geometric_accuracy/detector_bias/data/render_inputs/fisheye_20261027_eta2.0_v7.state.json
# (the body alone; every number as the file holds it). The old NON-CENTRAL body
# reads the same way, through `_noncentral_grid_body`, its line points in the
# enclosing record's length unit:
# fixtures/old/calibration_state_noncentral_camera_body.json is `camera.camera` of
# rc-results-0.8's 20261110-baseline/refine-94c55707e60ca234/state_it24.json
# (an rc/calibration_state/v1 record in mm; the body alone, every number as the file holds it).

@testsetup module EmbeddedGridFixtures
    using JSON
    using VisualGeometryToolkitCore: PixelFrame, BoundedCamera, model, extent, mm
    import VisualGeometryToolkitIO
    export whole_file_camera, embedded_cameras

    # the body as an old grid FILE (vgio/generic_camera/v3, the colmap frame every old grid's numbers are in; a
    # non-central one's line points in the enclosing record's `length_unit`), read by the whole-file migration path:
    # a central one as the store of its one camera, a non-central one through its steps
    function whole_file_camera(text::AbstractString; length_unit = nothing)
        t = JSON.parse(text; dicttype = Dict{String, Any})
        t["schema"] = "vgio/generic_camera/v3"
        t["pixel_frame"] = "colmap"
        length_unit === nothing || (t["length_unit"] = length_unit)
        p = joinpath(mktempdir(), "embedded.json")
        write(p, JSON.json(t))
        return VisualGeometryToolkitIO.load(BoundedCamera, p)
    end
    # the body as an enclosing record holds it, read by both camera_from_json forms
    function embedded_cameras(body; length_unit = mm)
        m, ext = VisualGeometryToolkitIO.camera_from_json(body, PixelFrame{:colmap}; length_unit)
        return (; m, ext, bc = VisualGeometryToolkitIO.camera_from_json(BoundedCamera, body, PixelFrame{:colmap}; length_unit))
    end
end

@testitem "an embedded old central grid body reads as the whole-file migration's camera" setup = [VGIoFixtures, EmbeddedGridFixtures] begin
    using VisualGeometryToolkitCore: model, extent, lens_coverage, SplineLens, PixelFrame, Ellipse
    using VisualGeometryToolkitIO: camera_from_json, same_value
    path = joinpath(@__DIR__, "fixtures", "old", "calibration_state_camera_body.json")
    text = read(path, String)
    body = VisualGeometryToolkitIO._parse_vgio_text(text, path)
    @test body.type == "CentralGenericModel" && haskey(body, :direction_grid) && haskey(body, :present)
    ref = whole_file_camera(text)
    r = embedded_cameras(body)
    @test model(ref).lens isa SplineLens && lens_coverage(ref) isa Ellipse
    # bit-identical: every number compared with `===` (no length in a central camera; length_ulps = 0 regardless)
    @test same_value(r.m, model(ref); length_ulps = 0) && r.ext == extent(ref)
    @test same_value(r.bc, ref; length_ulps = 0)
    # the old body's own refusal (not the non-central body's missing "spline")
    short = JSON.parse(text; dicttype = Dict{String, Any}); delete!(short, "direction_grid")
    e = try camera_from_json(VisualGeometryToolkitIO._parse_vgio_text(JSON.json(short), path), PixelFrame{:colmap}); nothing catch e; e end
    @test e isa ArgumentError && occursin("old grid body", e.msg) && occursin("direction_grid", e.msg)
end

@testitem "an embedded old non-central grid body reads as the whole-file migration's camera" setup = [VGIoFixtures, EmbeddedGridFixtures] begin
    using VisualGeometryToolkitCore: model, extent, lens_coverage, PixelFrame, Ellipse, NoncentralSplineModel
    using VisualGeometryToolkitIO: same_value
    path = joinpath(@__DIR__, "fixtures", "old", "calibration_state_noncentral_camera_body.json")
    text = read(path, String)
    body = VisualGeometryToolkitIO._parse_vgio_text(text, path)
    @test body.type == "NoncentralGenericModel" && haskey(body, :point_grid) && haskey(body, :present)
    ref = whole_file_camera(text; length_unit = "mm")      # the record it came from declares mm
    r = embedded_cameras(body; length_unit = mm)
    @test model(ref) isa NoncentralSplineModel && lens_coverage(ref) isa Ellipse
    # bit-identical: the line points too (length_ulps = 0), both read from the same numbers in the same unit
    @test same_value(r.m, model(ref); length_ulps = 0) && r.ext == extent(ref)
    @test same_value(r.bc, ref; length_ulps = 0)
    # the points are in the enclosing record's unit: read in m they are 1000 times the mm reading
    in_m = embedded_cameras(body; length_unit = m)
    @test model(in_m.bc).points.controls[findfirst(c -> all(isfinite, c), model(in_m.bc).points.controls)] ≈
        1000 .* model(r.bc).points.controls[findfirst(c -> all(isfinite, c), model(r.bc).points.controls)]
end

# The render-input states on the data tier (rc_synthetic_2026-11-10; the directory in VGTK_RC_RENDER_INPUTS, else skipped).
@testitem "RC's render-input states: their cameras read as the whole-file migration's" setup = [VGIoFixtures, EmbeddedGridFixtures] skip = !isdir(get(ENV, "VGTK_RC_RENDER_INPUTS", "")) begin
    using VisualGeometryToolkitCore: model, extent, SplineLens
    using VisualGeometryToolkitIO: same_value
    dir = ENV["VGTK_RC_RENDER_INPUTS"]
    files = sort!(filter(f -> endswith(f, ".state.json") && (occursin("_v7", f) || endswith(f, "_v8.state.json") || occursin("_refined_v8", f)), readdir(dir)))
    @test length(files) == 3
    for f in files
        state = VisualGeometryToolkitIO._parse_vgio_text(read(joinpath(dir, f), String), f)
        @test state.schema == "rc/calibration_state/v1" && state.pixel_frame == "colmap"
        body = state.camera.camera
        @test body.type == "CentralGenericModel"
        ref = whole_file_camera(JSON.json(body))
        r = embedded_cameras(body)
        @test model(r.bc).lens isa SplineLens
        @test same_value(r.m, model(ref); length_ulps = 0) && r.ext == extent(ref)
        @test same_value(r.bc, ref; length_ulps = 0)
    end
end
