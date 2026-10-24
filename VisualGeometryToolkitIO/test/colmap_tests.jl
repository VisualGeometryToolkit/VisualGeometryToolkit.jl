# =============================================================================
# the COLMAP wire layer — validation, camera assembly, model round trips
# =============================================================================

@testitem "wire validation" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    @test_throws ArgumentError ColmapCamera(1, :PINHOLE, 100, 100, [1.0, 2.0])   # wrong arity
    @test_throws ArgumentError ColmapCamera(1, :NOT_A_MODEL, 100, 100, Float64[])
    # an embedded pose: a non-unit quaternion and an unknown unit are refused, never repaired
    pj(q) = JSON3.read(JSON3.write((rotation = collect(q), translation = [0.0, 0.0, 0.0])))
    @test_throws ArgumentError VgtkIO.pose_from_json(pj(2 .* q0), "mm")
    @test_throws ArgumentError VgtkIO.pose_from_json(pj(q0), "smoot")
    # An empty image name is representable in binary but NOT in text (NAME is
    # the last field on the line), so the wire type refuses it rather than let
    # the two formats disagree.
    @test_throws ArgumentError ColmapImage(Int32(1), q0, SVector(0.0, 0.0, 0.0), Int32(1), "")
    @test_throws ArgumentError colmap_image(_mkpose(), 1, 1, "")
end

@testitem "camera assembly" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(2461.4, 0, 0, 0, 2460.6, 0, 2140.2, 1400.3, 1))
    pin = CameraModel(K; lens = Pinhole(), image_origin = :colmap)
    c = colmap_camera(pin, 1, 4240, 2832)
    @test c.model === :PINHOLE
    @test camera_model(c).intrinsics.K ≈ pin.intrinsics.K

    bc = CameraModel(K; lens = opencv_lens([-0.1, 0.05, 1.0e-4, -2.0e-4]), image_origin = :colmap)
    c2 = colmap_camera(bc, 2, 4240, 2832)
    @test c2.model === :OPENCV
    back = camera_model(c2)
    @test back.lens === bc.lens
    @test parameters(back.lens).k1 == -0.1 && parameters(back.lens).p2 == -2.0e-4

    bc6 = CameraModel(K; lens = opencv_lens([-0.1, 0.05, 1.0e-4, -2.0e-4, 0.01, 0.0, 0.0, 0.002]), image_origin = :colmap)
    c3 = colmap_camera(bc6, 3, 4240, 2832)
    @test c3.model === :FULL_OPENCV
    @test parameters(camera_model(c3).lens).d3 == 0.002

    c32 = camera_model(Float32, c2)
    @test eltype(c32.intrinsics.K) === Float32
    @test eltype(lens_parameters(c32.lens)) === Float32

    # a thin prism on a perspective view has no COLMAP model: loud, not truncated
    bcs = CameraModel(K; lens = opencv_lens([-0.1, 0.05, 1.0e-4, -2.0e-4, 0.01, 0.0, 0.0, 0.0, -5.5e-4, 7.0e-6, -2.2e-3, 1.2e-4]), image_origin = :colmap)
    @test_throws ArgumentError colmap_camera(bcs, 4, 4240, 2832)

    # every COLMAP 4.0 model reads, and writes back as a model of the same function
    for (model, params, as) in (
            (:SIMPLE_PINHOLE, [800.0, 320.0, 240.0], :PINHOLE),
            (:SIMPLE_RADIAL, [800.0, 320.0, 240.0, -0.1], :OPENCV),
            (:RADIAL, [800.0, 320.0, 240.0, -0.1, 0.02], :OPENCV),
            (:SIMPLE_RADIAL_FISHEYE, [800.0, 320.0, 240.0, -0.05], :OPENCV_FISHEYE),
            (:RADIAL_FISHEYE, [800.0, 320.0, 240.0, -0.05, 0.004], :OPENCV_FISHEYE),
            (:FOV, [800.0, 810.0, 320.0, 240.0, 0.9], :FOV),
            (:SIMPLE_DIVISION, [800.0, 320.0, 240.0, -0.2], :DIVISION),
            (:DIVISION, [800.0, 810.0, 320.0, 240.0, -0.2], :DIVISION),
            (:SIMPLE_FISHEYE, [800.0, 320.0, 240.0], :FISHEYE),
            (:FISHEYE, [800.0, 810.0, 320.0, 240.0], :FISHEYE),
            (:RAD_TAN_THIN_PRISM_FISHEYE, [800.0, 810.0, 320.0, 240.0, 0.1, -0.02, 0.003, 0.0, 0.0, 1.0e-4, 3.0e-4, -2.0e-4, 1.0e-3, 0.0, -5.0e-4, 2.0e-5], :RAD_TAN_THIN_PRISM_FISHEYE),
        )
        cam = camera_model(ColmapCamera(1, model, 640, 480, params))
        w = colmap_camera(cam, 1, 640, 480)
        @test w.model === as
        back = camera_model(w)
        @test back.intrinsics.K == cam.intrinsics.K
        # the same function — the same structure when the model is the read one
        model === as && @test back.lens === cam.lens
        for X in (SVector(0.3, -0.2, 1.0), SVector(-0.4, 0.5, 1.0), SVector(0.05, 0.6, 0.7))
            @test VgtkCore.project(back, X) == VgtkCore.project(cam, X)
        end
    end

    # skew has no COLMAP parameter either: refused, not silently dropped
    Ks = CameraCalibrationMat(SMatrix{3, 3, Float64}(2461.4, 0, 0, 1.5, 2460.6, 0, 2140.2, 1400.3, 1))
    @test_throws ArgumentError colmap_camera(CameraModel(Ks; lens = Pinhole(), image_origin = :colmap), 5, 4240, 2832)
end

@testitem "THIN_PRISM_FISHEYE (every ETH3D DSLR camera) assembles and round-trips" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    # ETH3D courtyard, dslr_calibration_jpg camera 0, in COLMAP's order:
    # fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1
    p = [3411.42, 3410.02, 3041.29, 2014.07, 0.21047, 0.21102, -5.36e-6, 0.000515, -0.158, 0.407, -8.46e-5, 0.000861]
    row = ColmapCamera(0, :THIN_PRISM_FISHEYE, 6048, 4032, p)
    cam = camera_model(row)
    @test cam.lens === Lens(Equidistant(), Additive(Radial((p[5], p[6], p[9], p[10])), Tangential(p[7], p[8]), ThinPrism((p[11],), (p[12],))))
    @test cam.intrinsics.K == SMatrix{3, 3, Float64}(p[1], 0, 0, 0, p[2], 0, p[3], p[4], 1)
    # VGC's order is the leaves' (k1, k2, k3, k4, p1, p2, sx1, sy1): k3, k4 move ahead of p1, p2
    @test collect(VgtkCore.lens_parameters(cam.lens)) == p[[5, 6, 9, 10, 7, 8, 11, 12]]
    # The camera projects as COLMAP evaluates the model (models.h, ImgFromCam
    # + Distortion), computed here independently of VGC's composition: the
    # equidistant angle, then the additive radial + tangential + thin-prism
    # bundle on it. A permuted coefficient moves these pixels.
    function colmap_thin_prism_fisheye(X)
        r = hypot(X[1], X[2])
        θ = atan(r, X[3])
        u, v = θ * X[1] / r, θ * X[2] / r
        k1, k2, p1, p2, k3, k4, sx1, sy1 = p[5:12]
        r2 = u^2 + v^2
        radial = k1 * r2 + k2 * r2^2 + k3 * r2^3 + k4 * r2^4
        du = u * radial + 2p1 * u * v + p2 * (r2 + 2u^2) + sx1 * r2
        dv = v * radial + 2p2 * u * v + p1 * (r2 + 2v^2) + sy1 * r2
        return SVector(p[1] * (u + du) + p[3], p[2] * (v + dv) + p[4])
    end
    for X in (SVector(0.3, -0.2, 1.0), SVector(-0.5, 0.4, 1.0), SVector(0.05, 0.6, 0.8))
        x = VgtkCore.project(cam, X)
        @test SVector(x[1], x[2]) ≈ colmap_thin_prism_fisheye(X) atol = 1.0e-8
    end
    # the writer is the reader's inverse: every coefficient back in its slot, exactly
    back = colmap_camera(cam, 0, 6048, 4032)
    @test colmap_model(cam.lens) === :THIN_PRISM_FISHEYE
    @test back.model === :THIN_PRISM_FISHEYE && back.params == p
    for format in (:binary, :text)
        mktempdir() do dir
            save(dir, ColmapModel([row], ColmapImage[], VgtkIO.ColmapPoint3D[], MM_PX); format)
            r = load(ColmapModel, dir).cameras[1]
            @test r.model === :THIN_PRISM_FISHEYE && r.params == p
            @test camera_model(r).lens == cam.lens
        end
    end
    # single precision end to end, as for the other models
    c32 = camera_model(Float32, row)
    @test eltype(lens_parameters(c32.lens)) === Float32 && eltype(c32.intrinsics.K) === Float32
    @test collect(VgtkCore.lens_parameters(c32.lens)) == Float32.(p[[5, 6, 9, 10, 7, 8, 11, 12]])
end

@testitem "colmap model round trip (binary + text)" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 0, 805.0, 0, 320.5, 240.5, 1))
    cam = CameraModel(K; lens = opencv_lens([-0.1, 0.05, 1.0e-4, -2.0e-4]), image_origin = :colmap)
    pose = _mkpose()
    model = ColmapModel(
        [colmap_camera(cam, 1, 640, 480)],
        [
            let im = colmap_image(pose, 1, 1, "DSC04520.JPG")
                ColmapImage(
                    im.image_id, im.qvec, im.tvec, im.camera_id, im.name,
                    [(12.5, 40.25, Int64(7)), (600.0, 10.0, Int64(-1))]
                )
            end,
        ],
        [
            VgtkIO.ColmapPoint3D(
                7, SVector(1.0, 2.0, 3.0),
                SVector{3, UInt8}(10, 20, 30), 0.5,
                [(Int32(1), Int32(0))]
            ),
        ],
        MM_PX
    )

    for format in (:binary, :text)
        mktempdir() do dir
            save(dir, model; format)
            back = load(ColmapModel, dir)
            @test back.contract == MM_PX
            # EXACT in both formats: binary stores the doubles, and the text
            # writer prints each in Julia's shortest round-trippable form, so
            # parsing it back is the identical Float64 — `≈` would hide a
            # writer that lost digits.
            @test back.cameras[1].model === :OPENCV
            @test back.cameras[1].params == model.cameras[1].params
            @test back.images[1].qvec == model.images[1].qvec
            @test back.images[1].tvec == model.images[1].tvec
            @test back.images[1].name == "DSC04520.JPG"
            @test back.images[1].points2D == model.images[1].points2D
            @test back.points3D[1].xyz == model.points3D[1].xyz
            @test back.points3D[1].track == model.points3D[1].track

            pose_back = rigid_transform(back.images[1], back.contract)
            @test pose_back.t ≈ pose.t atol = 1.0e-9

            # sidecar-less dirs demand an explicit contract
            rm(joinpath(dir, VgtkIO.UNITS_SIDECAR))
            @test_throws ArgumentError load(ColmapModel, dir)
            @test load(ColmapModel, dir; contract = MM_PX).contract == MM_PX
        end
    end
end

@testitem "colmap image names may contain spaces" setup = [VGIoFixtures] begin
    # The text writer puts NAME last on the line and COLMAP's own reader
    # getlines the tail, so a name with spaces is legal on both sides — and
    # image names are PATHS, so they occur in real datasets. Tokenizing the
    # line and taking the 10th field truncated "sub dir/img 01.jpg" to "sub",
    # silently, in the text format only. Interior runs of spaces must survive
    # too, which is why the third case is double-spaced.
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 0, 800.0, 0, 320.5, 240.5, 1))
    cam = colmap_camera(CameraModel(K; lens = Pinhole(), image_origin = :colmap), 1, 640, 480)
    for nm in ("plain.jpg", "sub dir/img 01.jpg", "a  b.jpg")
        model = ColmapModel(
            [cam], [colmap_image(_mkpose(), 1, 1, nm)],
            ColmapPoint3D[], MM_PX
        )
        for format in (:binary, :text)
            mktempdir() do dir
                save(dir, model; format)
                @test load(ColmapModel, dir).images[1].name == nm
            end
        end
    end
end

@testitem "UnknownScale: an SfM model's world has no unit" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    # A PhotoTourism-style model: its scale is the reconstruction's own. It
    # is read as such only when someone SAYS so, its translations stay
    # plain numbers, and every metric assembly refuses.
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 0, 800.0, 0, 320.5, 240.5, 1))
    cam = colmap_camera(CameraModel(K; lens = Pinhole(), image_origin = :colmap), 1, 640, 480)
    im = ColmapImage(1, q0, SVector(0.25, -1.5, 3.0), 1, "a.jpg")
    pt = VgtkIO.ColmapPoint3D(1, SVector(1.0, 2.0, 3.0), SVector{3, UInt8}(0, 0, 0), 0.1, Tuple{Int32, Int32}[])
    mktempdir() do dir
        save(dir, ColmapModel([cam], [im], [pt], UnknownScale()))
        sidecar = joinpath(dir, VgtkIO.UNITS_SIDECAR)
        @test JSON3.read(read(sidecar, String)).world == "unknown"   # the model states it
        back = load(ColmapModel, dir)
        @test back.contract === UnknownScale()
        @test back.images[1].tvec == SVector(0.25, -1.5, 3.0)       # plain numbers, unscaled
        @test back.images[1].qvec == q0
        @test back.points3D[1].xyz == SVector(1.0, 2.0, 3.0)
        # anything metric refuses, and says why
        err = try
            rigid_transform(back.images[1], back.contract)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("scale is unknown", err.msg)
        @test_throws ArgumentError colmap_image(_mkpose(), 2, 1, "b.jpg"; contract = UnknownScale())
        # a caller claiming a unit contradicts the sidecar
        @test_throws ArgumentError load(ColmapModel, dir; contract = MM_PX)
        # a model from another tool has no sidecar: never a default, the caller states it
        rm(sidecar)
        @test_throws ArgumentError load(ColmapModel, dir)
        @test load(ColmapModel, dir; contract = UnknownScale()).contract === UnknownScale()
    end
    # nor can a metric model be read as scale-free
    mktempdir() do dir
        save(dir, ColmapModel([cam], [im], [pt], MM_PX))
        @test_throws ArgumentError load(ColmapModel, dir; contract = UnknownScale())
        @test rigid_transform(load(ColmapModel, dir).images[1], MM_PX).t ≈ [0.25, -1.5, 3.0]
    end
end
