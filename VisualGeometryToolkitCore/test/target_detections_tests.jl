@testitem "target detections" begin
    using VisualGeometryToolkitCore: thrown_message
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: m, mm, sources, targets, withattrs, has_units, with_unit
    using StaticArrays
    using LinearAlgebra: I
    using Statistics: cov
    using FlexUnits: DimensionError

    # TargetDetections{F} (captures.jl): per frame, per target, csponds(index,
    # detection), the dimension from the frame, one form per container; its
    # stamps a FrameStamps; converted from TargetPoints2D/3D.

    Σ2 = CovMat(SMatrix{2, 2, Float64}(0.25I))
    Σ3 = CovMat(SMatrix{3, 3, Float64}(1.0e-6I))   # m²
    pix(i, x) = csponds(i, x)

    @testset "FrameStamps" begin
        s = FrameStamps([1, 2]; sensors = "cam", images = Dict(1 => "a.png"), times = Dict(2 => 0.5), time_origin = "2026-11-05T00:00:00")
        @test s.sensors == Dict(1 => "cam", 2 => "cam") && s.images == Dict(1 => "a.png") && s.times[2] == 0.5
        @test s == FrameStamps([1, 2]; sensors = "cam", images = Dict(1 => "a.png"), times = Dict(2 => 0.5), time_origin = "2026-11-05T00:00:00")
        @test hash(s) == hash(FrameStamps([1, 2]; sensors = "cam", images = Dict(1 => "a.png"), times = Dict(2 => 0.5), time_origin = "2026-11-05T00:00:00"))
        @test s != FrameStamps([1, 2]; sensors = "cam")
        @test_throws ArgumentError FrameStamps([1]; images = Dict(2 => "b.png"))   # a frame with no value
        @test_throws ArgumentError FrameStamps([1]; times = Dict(1 => 0.0))         # times need their origin
    end

    @testset "pixels: plain with weights, Uncertain" begin
        cs = withattrs(pix([1, 4], [SVector(1.0, 2.0), SVector(3.0, 4.0)]); weights = [1.0, 0.5])
        td = TargetDetections{PixelFrame{:opencv}}(Dict(1 => Dict(:b => cs)); sensors = "cam", images = Dict(1 => "a.png"), targets = ["ab"])
        C = Csponds{Int, SVector{2, Float64}, Vector{Int}, Vector{SVector{2, Float64}}, @NamedTuple{weights::Vector{Float64}}}
        @test td isa TargetDetections{PixelFrame{:opencv}, C}                 # one concrete set type
        @test td[1][:b][2] == (4 => SVector(3.0, 4.0)) && td[1][:b].weights == [1.0, 0.5]
        @test (td.sensors, td.images, td.targets) == (Dict(1 => "cam"), Dict(1 => "a.png"), ["ab"])
        @test td.stamps isa FrameStamps && length(td) == 1 && collect(keys(td)) == [1]
        @test !has_units(td)
        c = convert_frame(PixelFrame{:colmap}, td)
        @test c isa TargetDetections{PixelFrame{:colmap}} && c[1][:b][1] == (1 => SVector(1.5, 2.5))
        @test c[1][:b].weights == [1.0, 0.5] && convert_frame(PixelFrame{:opencv}, c) == td
        @test c != td                                                         # equal numbers, different frames
        @test td != TargetDetections{PixelFrame{:opencv}}(Dict(1 => Dict(:b => withattrs(cs; weights = [1.0, 1.0]))); sensors = "cam", images = Dict(1 => "a.png"), targets = ["ab"])

        tu = TargetDetections{PixelFrame{:colmap}}(Dict(3 => Dict(:b => pix([2], [Uncertain(SVector(1.0, 2.0), Σ2)]))))
        @test tu[3][:b][1].second isa UncertainPixel && cov(tu[3][:b][1].second) == Σ2
        @test cov(convert_frame(PixelFrame{:opencv}, tu)[3][:b][1].second) == Σ2   # a translation keeps the covariance
        @test_throws MethodError ustrip(mm, td)                               # pixels are not lengths
    end

    @testset "validation" begin
        P = PixelFrame{:colmap}
        @test_throws ArgumentError TargetDetections{P}(Dict(1 => Dict(:b => pix([0], [SVector(1.0, 2.0)]))))            # 1-based
        @test_throws ArgumentError TargetDetections{P}(Dict(1 => Dict(:b => pix([2, 2], [SVector(1.0, 2.0), SVector(1.0, 2.0)]))))   # unique
        @test_throws ArgumentError TargetDetections{P}(Dict(1 => Dict(:b => pix([1], [SVector(1.0, 2.0, 3.0)]))))       # dimension
        @test_throws ArgumentError TargetDetections{P}(Dict(1 => Dict(:b => pix(Int[], SVector{2, Float64}[]))))        # absent, not empty
        @test_throws ArgumentError TargetDetections{P}(Dict(1 => Dict(:b => [SVector(1.0, 2.0)])))                       # not pairs
        @test_throws ArgumentError TargetDetections{P}(                                                                 # plain + Uncertain
            Dict(1 => Dict(:b => pix([1], [SVector(1.0, 2.0)]), :c => pix([1], [Uncertain(SVector(1.0, 2.0), Σ2)])))
        )
        @test_throws ArgumentError TargetDetections{P}(                                                                 # columns differ
            Dict(1 => Dict(:b => pix([1], [SVector(1.0, 2.0)]), :c => withattrs(pix([1], [SVector(1.0, 2.0)]); weights = [1.0])))
        )
        @test_throws ArgumentError TargetDetections{P}(Dict(1 => Dict(:b => pix([1], [SVector(1.0, 2.0)]))); images = Dict(2 => "x.png"))
        @test_throws ArgumentError TargetDetections{P}(Dict{Int, Any}(); targets = ["a", "a"])
        @test_throws ArgumentError TargetDetections{UnlabeledFrame{3}}(Dict{Int, Any}())
        @test_throws ArgumentError TargetDetections{UnlabeledFrame{2}}(Dict{Int, Any}())                                 # not a pixel frame
        @test_throws DimensionError TargetDetections{WorldFrame}(                                                       # plain + physical
            Dict(1 => Dict(:a => pix([1], [SVector(1.0, 2.0, 3.0)]), :b => pix([1], [SVector(1.0, 2.0, 3.0) .* m])))
        )
        e = TargetDetections{P}(Dict{Int, Any}())
        @test isempty(e) && e == TargetDetections{P}(Dict{Int, Any}())
    end

    @testset "3D points: units" begin
        td = TargetDetections{WorldFrame}(Dict(7 => Dict(:body => pix([3, 1], [SVector(1.0, 2.0, 3.0) .* m, SVector(4.0, 5.0, 6.0) .* m]))); sensors = "mocap")
        @test has_units(td) && td.sensors[7] == "mocap"
        pl = ustrip(mm, td)
        @test !has_units(pl) && pl[7][:body][1] == (3 => SVector(1000.0, 2000.0, 3000.0))
        @test with_unit(pl, mm) == td && with_unit(ustrip(m, td), m) == td
        tu = TargetDetections{WorldFrame}(Dict(7 => Dict(:body => pix([3], [Uncertain(SVector(1.0, 2.0, 3.0) .* m, Σ3)]))); sensors = "mocap")
        @test has_units(tu) && cov(tu[7][:body][1].second) == Σ3                     # m², as supplied
        @test cov(ustrip(mm, tu)[7][:body][1].second) ≈ SMatrix{3, 3, Float64}(1.0I)  # mm²
        @test with_unit(ustrip(mm, tu), mm) == tu
        plain = TargetDetections{WorldFrame}(Dict(7 => Dict(:body => pix([3], [SVector(1.0, 2.0, 3.0)]))))
        @test !has_units(plain) && plain[7][:body][1].second isa SVector{3, Float64}
    end

    @testset "from TargetPoints2D / TargetPoints3D" begin
        o = TargetPoints2D{PixelFrame{:opencv}}(
            Dict(
                1 => [(target = :b, index = 1, xy = SVector(1.0, 2.0), weight = 1.0), (target = :c, index = 2, xy = SVector(3.0, 4.0), weight = 0.5),
                    (target = :b, index = 5, xy = SVector(5.0, 6.0), weight = 0.25)],
                2 => [(target = :b, index = 1, xy = SVector(1.5, 2.5), weight = 1.0)],
            );
            sensors = "cam", images = Dict(1 => "a.png", 2 => "b.png"), targets = ["ab"],
        )
        td = TargetDetections(o)
        @test td isa TargetDetections{PixelFrame{:opencv}}
        @test sources(td[1][:b]) == [1, 5] && targets(td[1][:b]) == [SVector(1.0, 2.0), SVector(5.0, 6.0)]
        @test td[1][:b].weights == [1.0, 0.25] && td[1][:c].weights == [0.5]           # the bundle weight, a column
        @test (td.images, td.sensors, td.targets) == (o.images, o.sensors, o.targets)
        @test td == TargetDetections{PixelFrame{:opencv}}(
            Dict(
                1 => Dict(:b => withattrs(pix([1, 5], [SVector(1.0, 2.0), SVector(5.0, 6.0)]); weights = [1.0, 0.25]),
                    :c => withattrs(pix([2], [SVector(3.0, 4.0)]); weights = [0.5])),
                2 => Dict(:b => withattrs(pix([1], [SVector(1.5, 2.5)]); weights = [1.0])),
            );
            sensors = "cam", images = Dict(1 => "a.png", 2 => "b.png"), targets = ["ab"],
        )
        mixed = TargetPoints2D{PixelFrame{:opencv}}(
            Dict(1 => [(target = :b, index = 1, xy = SVector(1.0, 2.0), weight = 1.0), (target = :b, index = 2, xy = Uncertain(SVector(3.0, 4.0), Σ2), weight = 0.5)])
        )
        @test_throws ArgumentError TargetDetections(mixed)                             # one form per container
        p3 = TargetPoints3D{WorldFrame}(
            Dict(7 => [(target = :body, index = 3, xyz = Uncertain(SVector(1.0, 2.0, 3.0) .* m, Σ3))]);
            sensors = "mocap", times = Dict(7 => 1.5), time_origin = "2026-11-05T00:00:00",
        )
        t3 = TargetDetections(p3)
        @test t3 isa TargetDetections{WorldFrame} && has_units(t3) && !hasattr(t3[7][:body], :weights)
        @test cov(t3[7][:body][1].second) == Σ3 && (t3.times, t3.time_origin) == (p3.times, p3.time_origin)
    end

    @testset "conversion refusals name the frame, target and index" begin
        dup = TargetPoints2D{PixelFrame{:opencv}}(
            Dict(4 => [(target = :b, index = 2, xy = SVector(1.0, 2.0), weight = 1.0), (target = :b, index = 2, xy = SVector(3.0, 4.0), weight = 1.0)])
        )
        s = thrown_message(() -> TargetDetections(dup))
        @test occursin("frame 4, target b, index 2", s) && occursin("load(TargetPoints2D, path)", s)
        mixed = TargetPoints3D{WorldFrame}(
            Dict(
                1 => [(target = :body, index = 1, xyz = SVector(1.0, 2.0, 3.0) .* m)],
                2 => [(target = :body, index = 3, xyz = Uncertain(SVector(1.0, 2.0, 3.0) .* m, Σ3))],
            )
        )
        s = thrown_message(() -> TargetDetections(mixed))
        @test occursin("frame 2, target body, index 3 is Uncertain", s) && occursin("frame 1, target body, index 1 is plain", s)
        @test occursin("load(TargetPoints3D, path)", s)
    end

    @testset "Csponds equality is of its pairs" begin
        # a Csponds is the vector of its pairs: `==`, `isequal` and `hash` ignore
        # its columns (TargetDetections compares them itself)
        cs = pix([1, 2], [SVector(1.0, 2.0), SVector(3.0, 4.0)])
        w = withattrs(cs; weights = [1.0, 0.5])
        @test w == cs && isequal(w, cs) && hash(w) == hash(cs) && hash(w) == hash(collect(w))
    end

    @testset "isequal and hash agree with ==" begin
        P = PixelFrame{:colmap}
        fr() = Dict(1 => Dict(:b => withattrs(pix([1, 2], [SVector(1.0, 2.0), SVector(3.0, 4.0)]); weights = [1.0, 0.5])))
        a = TargetDetections{P}(fr(); sensors = "cam", targets = ["ab"])
        b = TargetDetections{P}(fr(); sensors = "cam", targets = ["ab"])
        @test a == b && isequal(a, b) && hash(a) == hash(b)
        others = (
            TargetDetections{P}(fr(); sensors = "other", targets = ["ab"]),                  # stamps
            TargetDetections{P}(fr(); sensors = "cam", targets = ["cd"]),                    # targets
            TargetDetections{PixelFrame{:opencv}}(fr(); sensors = "cam", targets = ["ab"]),   # pixel frame
            TargetDetections{P}(                                                             # columns
                Dict(1 => Dict(:b => withattrs(pix([1, 2], [SVector(1.0, 2.0), SVector(3.0, 4.0)]); weights = [1.0, 1.0]))); sensors = "cam", targets = ["ab"],
            ),
        )
        for o in others
            @test a != o && !isequal(a, o) && hash(a) != hash(o)
        end
        d = Dict(a => 1)
        @test d[b] == 1 && all(o -> !haskey(d, o), others)
        @test length(Set([a, b, others...])) == 1 + length(others)
    end
end
