# =============================================================================
# EnvelopeCases — one value of every vgio kind, written by its own writer
# =============================================================================
#
# The envelope tests (envelope_tests.jl) and the validator's deletion cases
# (v2_validation_tests.jl) run over a file of every kind.

@testsetup module EnvelopeCases
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitIO: load, save
    using VisualGeometryToolkitCore: mm, m, px, CameraModel, CameraCalibrationMat, KannalaBrandt, Size2, EuclideanMap,
        BoundedCamera, PosedCamera, Rig, PosedTarget, PlanarTarget, PointTarget, AbstractTarget, Trajectory, TargetPoses,
        TargetHomographies, HomographyMap, HomographyMat, TargetObservations, CaptureSettings, ExposureSettings, LensSettings,
        DeviceIdentity, Attributed, AbstractVignetting, RadialVignetting, LogVignetting, EnvironmentLight, SphericalHarmonicMap,
        GaussianBlur, PixelFrame, WorldFrame, CameraFrame, BodyFrame, TargetFrame, ViewFrame, Uncertain, CovMat, relabel,
        correspondences, withattrs, with_control_points, with_unit, sign_normalize, parse_unit
    using StaticArrays: SVector, SMatrix
    using Rotations: QuatRotation
    using JSON: JSON
    export KINDS, write_every_kind, tomutable, refusal

    ml(t) = SVector{length(t)}(map(x -> VgtkIO.read_length(x, mm), Tuple(t)))
    pose(t) = EuclideanMap(QuatRotation(0.9128709291752769, 0.18257418583505536, 0.36514837167011072, 0.0, false), ml(t))

    # one value of every kind, with the type `load` takes
    function values_()
        h = repeat("ab", 32)
        K = CameraCalibrationMat(SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 320.0, 240.0, 1))
        cam = BoundedCamera(CameraModel(K; lens = KannalaBrandt(0.1, -0.02, 0.003, -0.0004), pixel_frame = :colmap), Size2(640, 480))
        ident = EuclideanMap(one(QuatRotation{Float64}), ml((0.0, 0.0, 0.0)))
        pts = [ml((0.0, 0.0)), ml((62.8, 0.0)), ml((0.0, 40.25))]
        sec = parse_unit("s")
        H = HomographyMap(sign_normalize(HomographyMat{Float64}((812.3, 37.9, 0.0001, -41.7, 805.1, -0.0002, 1210.25, 640.5, 0.6))))
        return [
            "transforms" => (Vector{EuclideanMap}, [relabel(BodyFrame{:camera_rig}, CameraFrame, pose((251.9, -144.4, 904.1)))]),
            "cameras" => (Dict{Symbol, BoundedCamera}, Dict(:cam => cam)),
            "rig" => (Rig, Rig{ViewFrame{:a}}((BoundedCamera(PosedCamera(model_of(cam), relabel(ViewFrame{:a}, ViewFrame{:a}, ident)), Size2(640, 480)),))),
            "target" => (AbstractTarget, PointTarget([ml((1.0, 2.0, 3.0)), ml((-13.9, 0.0, 5.5))])),
            "trajectory" => (Trajectory, Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => pose((1.0, 2.0, 900.0)));
                images = Dict(1 => "a.png"), times = Dict(1 => 0.5), time_origin = "2026-05-27T09:30:00+02:00")),
            "target_poses" => (TargetPoses, TargetPoses{CameraFrame}(Dict(1 => Dict(:a => pose((1.0, 2.0, 900.0)))); images = Dict(1 => "a.png"), sensors = "cam")),
            "target_homographies" => (TargetHomographies, TargetHomographies{PixelFrame{:colmap}}(Dict(3 => Dict(:a => H)); images = Dict(3 => "a.png"), sensors = "cam", targets = [h])),
            "observations" => (TargetObservations, TargetObservations{PixelFrame{:julia}}(
                Dict(2 => Dict(:a => withattrs(correspondences([1, 3], [SVector(640.25, 480.5), SVector(1.0, 2.0)]); weights = [1.0, 0.5])));
                images = Dict(2 => "a.png"), sensors = "cam", targets = [h])),
            "capture_settings" => (CaptureSettings, Attributed(
                CaptureSettings(Dict(1 => (exposure = ExposureSettings(0.01 * sec, 4.0, 100), lens_settings = LensSettings(VgtkIO.read_length(14.0, mm))));
                    body = DeviceIdentity(; make = "Sony", model = "ILME-FX3"), lens = DeviceIdentity(; model = "14mm")),
                (; identity_sources = (body = "exif", lens = "exif"), sensor = "cam"))),
            "vignetting" => (AbstractVignetting, LogVignetting(with_control_points(RadialVignetting(6, SVector(320.0, 240.0), 400.0), [0.5 + 0.01i for i in 1:6]))),
            "environment_light" => (EnvironmentLight, with_unit(EnvironmentLight(SphericalHarmonicMap(ones(3, 4), 1), SVector(0.1, -0.2, 0.3), 1.5), m)),
            "blur" => (Dict{Symbol, GaussianBlur}, Dict(:R => GaussianBlur(relabel(PixelFrame{:colmap}, CovMat(SMatrix{2, 2, Float64}(0.5, 0.1, 0.1, 0.4)))))),
        ]
    end
    model_of(c) = VgtkCore.model(c)
    const KINDS = first.(values_())

    # Write one file of every kind into `dir`: kind => (path, type, value).
    function write_every_kind(dir)
        out = Dict{String, Tuple{String, Any, Any}}()
        for (kind, (T, v)) in values_()
            p = joinpath(dir, "$(kind).json")
            save(p, v)
            out[kind] = (p, T, v)
        end
        return out
    end

    tomutable(x::AbstractDict) = Dict{String, Any}(String(k) => tomutable(v) for (k, v) in x)
    tomutable(x::AbstractVector) = Any[tomutable(v) for v in x]
    tomutable(x) = x

    # `nothing` if `load(T, path)` reads the edited tree `t`, else the refusal's message (an ArgumentError, required)
    function refusal(dir, name, T, t)
        p = joinpath(dir, "edited_$(name).json")
        write(p, JSON.json(t))
        return try
            load(T, p)
            nothing
        catch e
            e isa ArgumentError ? e.msg : rethrow()
        end
    end
end
