# COLMAP binary round trip through pycolmap, the reference implementation: the
# wire layer writes a model, pycolmap loads it and writes it again, and the wire
# layer reads pycolmap's output back — every value must come back identical.
#
# It runs when the pixi `py` environment is installed (`pixi install -e py`,
# which brings pycolmap) and calls that environment's python directly, so the
# suite never resolves or downloads an environment. Without it the check is
# recorded as SKIPPED (a broken test), never silently absent — it used to live
# in test/verify_pycolmap.jl, which runtests.jl never included.

@testitem "COLMAP binary round trip through pycolmap" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: ColmapPoint3D

    python = joinpath(dirname(@__DIR__), ".pixi", "envs", "py", "bin", "python")
    if !(isfile(python) && success(pipeline(`$python -c "import pycolmap"`; stdout = devnull, stderr = devnull)))
        @test_skip "pycolmap is not installed in the pixi py environment"
    else
        model = ColmapModel(
            [
                ColmapCamera(
                    1, :OPENCV, 4240, 2832,
                    [
                        2461.431099350077, 2460.563045263503, 2140.159365282542,
                        1400.300904420802, -0.0912, 0.0473, 1.1e-4, -2.3e-4,
                    ]
                ),
                ColmapCamera(2, :PINHOLE, 640, 480, [800.25, 805.75, 320.5, 240.5]),
                ColmapCamera(
                    3, :THIN_PRISM_FISHEYE, 6048, 4032,
                    [3411.42, 3410.02, 3041.29, 2014.07, 0.21047, 0.21102, -5.36e-6, 0.000515, -0.158, 0.407, -8.46e-5, 0.000861]
                ),
            ],
            [
                ColmapImage(
                    1, SVector(0.9128709291752769, 0.18257418583505536, 0.36514837167011072, 0.0),
                    SVector(251.90139647689938, -144.38003547869317, 904.1085203725466),
                    1, "DSC04520.JPG",
                    [(12.5, 40.25, Int64(7)), (600.0, 10.0, Int64(-1))]
                ),
                ColmapImage(2, SVector(1.0, 0.0, 0.0, 0.0), SVector(0.0, 0.0, 1000.0), 2, "sub dir/second.jpg"),
            ],
            [ColmapPoint3D(7, SVector(1.25, -2.5, 3.75), SVector{3, UInt8}(10, 20, 30), 0.5, [(Int32(1), Int32(0))])],
            MM_PX
        )
        mktempdir() do tmp
            dir_out, dir_back = mkpath(joinpath(tmp, "out")), mkpath(joinpath(tmp, "back"))
            save(dir_out, model)
            script = joinpath(tmp, "roundtrip.py")
            write(
                script, """
                import sys, pycolmap
                rec = pycolmap.Reconstruction(sys.argv[1])
                rec.write_binary(sys.argv[2])
                """
            )
            @test success(`$python $script $dir_out $dir_back`)
            back = load(ColmapModel, dir_back; contract = MM_PX)
            cams = Dict(c.camera_id => c for c in back.cameras)
            for c0 in model.cameras
                c1 = cams[c0.camera_id]
                @test (c1.model, c1.width, c1.height) == (c0.model, c0.width, c0.height)
                @test c1.params == c0.params
            end
            ims = Dict(i.image_id => i for i in back.images)
            for i0 in model.images
                i1 = ims[i0.image_id]
                @test i1.qvec == i0.qvec && i1.tvec == i0.tvec
                @test i1.name == i0.name && i1.camera_id == i0.camera_id
                @test i1.points2D == i0.points2D
            end
            p0, p1 = only(model.points3D), only(back.points3D)
            @test (p1.point3D_id, p1.xyz, p1.rgb, p1.error, p1.track) == (p0.point3D_id, p0.xyz, p0.rgb, p0.error, p0.track)
        end
    end
end
