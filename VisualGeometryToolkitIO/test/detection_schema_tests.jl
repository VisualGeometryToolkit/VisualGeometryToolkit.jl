# vgio/detection/v1.0 (detection_schema.jl): detections per image through the one frame codec — exact by type for
# every frame type, plain and uncertain, every covariance storage, Float64 and Float32; an absent covariance stated by
# the producer's `covariance_status`; the table's refusals.

@testitem "detection: Förstner-like point frames, an absent covariance stated, exact by type" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: validate
    using VisualGeometryToolkitCore: PointFrame, Point2, CovMat, Uncertain, Attributed, AbstractDetection
    nan2 = CovMat(SMatrix{2, 2, Float64, 4}(NaN, NaN, NaN, NaN))
    det(x, y, Σ, label, st) = Attributed(Uncertain(PointFrame(x, y), Σ), (; label, window = Point2(round(x), round(y)), F = 0.31, covariance_status = st))
    D1 = [det(512.25, 300.5, CovMat(SMatrix{2, 2}(0.01, 0.002, 0.002, 0.02)), :corner, :propagated),
          det(80.0, 90.0, nan2, :unknown, :unclassified)]
    D2 = typeof(D1)()                                     # an image without detections keeps its type
    dets = Dict("DSC00002.ARW" => D1, "DSC00001.ARW" => D2)
    mktempdir() do dir
        p = joinpath(dir, "detections.json")
        save(p, dets)
        @test validate(p) == "detection"
        b = load(Dict{String, <:AbstractVector{<:AbstractDetection}}, p)
        @test b isa Dict{String, Vector{eltype(D1)}}
        @test isequal(b, dets)
        d = JSON.parsefile(p)
        @test d["schema"] == "vgio/detection/v1.0" && d["units"]["pixel"] == "colmap"
        @test d["detection"] == Dict("frame" => "point_frame", "number_type" => "float64",
                                     "covariance" => Dict("storage" => "full", "number_type" => "float64"),
                                     "attributes" => Dict("label" => "symbol", "window" => "point2", "F" => "float64", "covariance_status" => "symbol"))
        @test collect(keys(d["detection"]["attributes"])) == ["label", "window", "F", "covariance_status"]   # the value's order
        r = d["images"][2]["detections"]
        @test r[1]["covariance"] == [[0.01, 0.002], [0.002, 0.02]] && !haskey(r[2], "covariance")
        @test r[2]["attributes"]["covariance_status"] == "unclassified" && !occursin("null", read(p, String))
        # a NaN covariance without its stated reason is refused, nothing written
        bad = Dict("a" => [Attributed(Uncertain(PointFrame(1.0, 2.0), nan2), (; label = :unknown))])
        q = joinpath(dir, "bad.json")
        @test_throws ArgumentError save(q, bad)
        @test !isfile(q)
        part = Dict("a" => [Attributed(Uncertain(PointFrame(1.0, 2.0), CovMat(SMatrix{2, 2}(NaN, 0.0, 0.0, 1.0))), (; covariance_status = :x))])
        @test_throws ArgumentError save(q, part)
    end
end

@testitem "detection: every frame type, plain and uncertain, every storage, Float32" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: VgioStyle
    using VisualGeometryToolkitCore: PointFrame, Disc, OrientedDisc, EllipticFrame, AffineFrame, Point2, CovMat, Uncertain,
        Attributed, AbstractDetection, frame_dof, ScaledIdentity
    using LinearAlgebra: Diagonal, I
    s = VgioStyle()
    frames = (PointFrame(10.5, 20.25), Disc(Point2(1.0, 2.0), 3.0), OrientedDisc(1.0, 2.0, 3.0, 0.5),
              EllipticFrame(Point2(10.5, 20.25), SMatrix{2, 2}(2.0, 0.3, 0.3, 1.5)), AffineFrame(Point2(1.0, 2.0), SMatrix{2, 2}(1.2, -0.2, 0.1, 0.9)))
    attrs = (; response = 0.05, edge_score = 3.5f0, warp_idx = 4, dark = true, note = "", label = :blob)
    mktempdir() do dir
        p = joinpath(dir, "d.json")
        for f in frames, T in (Float64, Float32)
            g = JSON.lift(s, typeof(f).name.wrapper{T}, JSON.parse(JSON.json(f; style = s)))
            n = frame_dof(g)
            values = Any[Attributed(g, attrs)]
            for Σ in (CovMat(SMatrix{n, n}(Matrix(0.25 * I(n)) .+ 0.01)), CovMat(Diagonal(SVector{n}(0.5 .+ (1:n) ./ 8))),
                      CovMat(ScaledIdentity{n}(0.375)), CovMat(Diagonal(SVector{n, Float32}(Float32.(0.5 .+ (1:n) ./ 8)))))
                push!(values, Attributed(Uncertain(g, Σ), attrs))
            end
            for v in values
                dets = Dict("img.png" => [v, v])
                save(p, dets)
                b = load(Dict{String, <:AbstractVector{<:AbstractDetection}}, p)
                @test b == dets && typeof(b["img.png"]) == Vector{typeof(v)}
            end
        end
        # one detection type per file
        @test_throws ArgumentError save(p, Dict{String, AbstractVector{<:AbstractDetection}}("a" => [Attributed(frames[1], (; r = 1.0))], "b" => [Attributed(frames[2], (; r = 1.0))]))
        # an attribute of a type the file cannot state
        @test_throws ArgumentError save(p, Dict("a" => [Attributed(frames[1], (; r = 1 // 2))]))
        # a covariance not over frame_dof (ASIFT's 3×3 on an EllipticFrame)
        @test_throws ArgumentError save(p, Dict("a" => [Attributed(Uncertain(frames[4], CovMat(SMatrix{3, 3}(Matrix(1.0I(3))))), (;))]))
    end
end

@testitem "detection: the table's refusals on read; another pixel frame moved into colmap" setup = [VGIoFixtures] begin
    using LinearAlgebra: I
    using VisualGeometryToolkitIO: validate
    using VisualGeometryToolkitCore: PointFrame, CovMat, Uncertain, Attributed, AbstractDetection
    dets = Dict("a" => [Attributed(Uncertain(PointFrame(10.0, 20.0), CovMat(SMatrix{2, 2}(1.0, 0.0, 0.0, 2.0))), (; label = :corner))])
    mktempdir() do dir
        p = joinpath(dir, "d.json")
        save(p, dets)
        q = joinpath(dir, "e.json")
        function refused(f!)
            d = JSON.parsefile(p; dicttype = Dict{String, Any})
            f!(d)
            write(q, JSON.json(d))
            return try load(Dict{String, <:AbstractVector{<:AbstractDetection}}, q); false catch e; e isa ArgumentError end
        end
        rec(d) = d["images"][1]["detections"][1]
        @test refused(d -> (rec(d)["cov"] = pop!(rec(d), "covariance")))                         # the retired key
        @test refused(d -> (d["detection"]["frame"] = "keypoint"))                                # an unknown frame type
        @test refused(d -> (d["detection"]["attributes"]["label"] = "complex"))                   # an attribute type outside the set
        @test refused(d -> (d["detection"]["covariance"]["storage"] = "diagonal"))                # not in its declared storage
        @test refused(d -> (rec(d)["covariance"] = nothing))                                      # no null covariance
        @test refused(d -> pop!(rec(d), "covariance"))                                            # absent, no covariance_status declared
        @test refused(d -> (rec(d)["covariance"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))   # not frame_dof
        @test refused(d -> push!(d["images"], d["images"][1]))                                    # an image twice
        @test !refused(identity)
        # a file in OpenCV pixels reads into colmap (+0.5)
        d = JSON.parsefile(p; dicttype = Dict{String, Any})
        d["units"]["pixel"] = "opencv"
        write(q, JSON.json(d))
        b = load(Dict{String, <:AbstractVector{<:AbstractDetection}}, q)
        c(x) = getfield(getfield(x, :value), :value).center
        @test c(only(b["a"])) == c(only(dets["a"])) .+ 0.5
    end
end
