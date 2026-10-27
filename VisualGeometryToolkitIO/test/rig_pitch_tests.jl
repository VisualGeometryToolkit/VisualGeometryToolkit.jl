# =============================================================================
# rig/v5: a sensor bounded by its physical Sensor carries the pixel pitch
# =============================================================================
#
# The camera body of a rig sensor gains `"pitch": [width, height]`, a length per
# pixel in the rig's `length_unit`, iff the camera is bounded by a `Sensor`;
# absent, the body and the identity are the bare extent's, byte for byte (every
# other fixture and content_id pin covers that). Additive in rig/v5, unreleased.

@testitem "rig/v5: the pitch travels with the camera body" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, BoundedCamera, PosedCamera, StereoRig, Sensor, sensor, model, μm, CMOS_SENSORS, image_circle
    using VisualGeometryToolkitIO: content_id

    src = joinpath(@__DIR__, "fixtures", "units", "rig_pitch.json")
    rig = load(Rig, src)

    @testset "the reader restores the Sensor bound" begin
        s = sensor(rig[1])
        @test s isa Sensor
        @test extent(rig[1]) == Size2(2448, 2048)
        @test ustrip(μm, s.pitch.width) ≈ 3.45 && ustrip(μm, s.pitch.height) ≈ 3.45
        @test ustrip(m, s.pitch.width) === ustrip(m, VgtkIO.read_length(0.00345, mm))   # the metre value nearest the decimal
        @test sensor(rig[2]) === nothing                                 # a bare extent stays one
        @test image_circle(rig[1], 6.0mm).a ≈ 6.0e-3 / (2 * 3.45e-6)     # the datasheet route reads the loaded pitch
    end

    @testset "load(save(rig)): the bound, the bytes and the identity" begin
        mktempdir() do dir
            out = joinpath(dir, "rig.json")
            save(out, rig)
            @test read(out, String) == read(src, String)
            back = load(Rig, out)
            @test sensor(back[1]) == sensor(rig[1])
            @test content_id(back) == content_id(rig) == content_id(src)
            # a value built in code (the CMOS_SENSORS pitch, 3.45μm) writes and hashes as its file
            fish = BoundedCamera(rig[1].camera, CMOS_SENSORS["Sony"]["IMX250"])
            built = StereoRig(fish, rig[2]; ids = (:fisheye, :cam))
            p2 = joinpath(dir, "built.json")
            save(p2, built)
            @test read(p2, String) == read(src, String)
            @test content_id(built) == content_id(src)
            # the identity covers the pitch: the same rig bounded by its bare extent differs
            bare = StereoRig(BoundedCamera(rig[1].camera, extent(rig[1])), rig[2]; ids = (:fisheye, :cam))
            @test content_id(bare) != content_id(rig)
            p3 = joinpath(dir, "bare.json")
            save(p3, bare)
            @test !occursin("pitch", read(p3, String))
            @test sensor(load(Rig, p3)[1]) === nothing
        end
    end

    @testset "a pitch that is not two positive lengths per pixel is refused" begin
        text = read(src, String)
        mktempdir() do dir
            for (bad, why) in (("[\n                    -0.00345,", "negative"), ("[\n                    \"3.45um\",", "a string"))
                p = joinpath(dir, "bad.json")
                write(p, replace(text, "[\n                    0.00345," => bad; count = 1))
                @test_throws ArgumentError load(Rig, p)
            end
            p = joinpath(dir, "short.json")
            write(p, replace(text, "0.00345,\n                    0.00345" => "0.00345"; count = 1))
            @test_throws ArgumentError load(Rig, p)
            # the error names the key, as every refusal does
            msg = try
                load(Rig, joinpath(dir, "bad.json")); ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("pitch", msg)
        end
    end
end
