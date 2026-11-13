# =============================================================================
# core_json.jl: quantities, Size2 and THE feature-frame codec on the style (JSON.lower / JSON.lift); the retired
# UnitContract layout's units object (migrate_units)
# =============================================================================

@testitem "codec: a quantity is a number in the style's declared unit; its field tag says what it is" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO, VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CovMat, Uncertain
    using LinearAlgebra: Diagonal, I
    using VisualGeometryToolkitIO: VgioStyle
    using VisualGeometryToolkitCore: dpi, pt
    s = VgioStyle(; length_unit = mm, length_roles = (pattern = pt,))
    sm = VgioStyle(; length_unit = m)
    @test JSON.lower(s, 2.5mm) === 2.5 && JSON.lower(sm, 2.5mm) === 0.0025
    @test JSON.lower(s, 640px) === 640 && JSON.lower(s, 640.5px) === 640.5 && JSON.lower(s, 300dpi) === 300
    @test JSON.lower(s, 0.5mm / px) === 0.5                       # a pitch is a length (px is dimensionless)
    @test_throws ArgumentError JSON.lower(s, 1.0 * parse_unit("s"))   # no declared unit for a time in a quantity field
    @test_throws ArgumentError JSON.lower(s, 1.0mm^2)
    # the tag: what a number IS on read (a pixel is dimensionless); a role for a length only
    for (q, tags) in ((VgtkIO.read_length(2.5, mm), (; quantity = :length)), (640.0px, (; quantity = :pixel)),
                      (300.0dpi, (; quantity = :density)), (VgtkIO.read_length(2.12598425197, pt), (; quantity = :length, role = :pattern)))
        x = JSON.lower(s, q, tags)
        @test JSON.lift(s, Quantity, x, tags) == q
        @test JSON.lift(s, Quantity, JSON.parse(JSON.json(x)), tags) == q
    end
    @test JSON.lower(s, VgtkIO.read_length(2.12598425197, pt), (; role = :pattern)) === 2.12598425197   # decimal-exact in its role
    @test_throws ArgumentError JSON.lift(s, Quantity, 2.5, (;))                        # a quantity's field states its tag
    @test_throws ArgumentError JSON.lift(s, Quantity, 2.5, (; quantity = :angle))
    @test_throws ArgumentError JSON.lift(s, Quantity, 640, (; quantity = :pixel, role = :pattern))
    @test_throws ArgumentError JSON.lift(s, Quantity, 1.0, (; quantity = :length, role = :ruler))   # an undeclared role
    @test_throws ArgumentError JSON.lower(s, 640px, (; quantity = :length))           # written as what it is not
    for bad in ("2.5", true, Dict("value" => 640, "unit" => "px"))
        e = try JSON.lift(s, Quantity, bad, (; quantity = :pixel)); nothing catch e; e end
        @test e isa ArgumentError
    end
    # a quantity left in a record is lowered by the writer's style: the file's units
    @test JSON.json((width = 210.0mm, dpi = 300dpi); style = s) == """{"width":210.0,"dpi":300}"""
end

@testitem "codec: Size2 is {width, height}, plain or tagged" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO, VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CovMat, Uncertain
    using LinearAlgebra: Diagonal, I
    using VisualGeometryToolkitIO: VgioStyle
    using VisualGeometryToolkitCore: pt
    s, sm = VgioStyle(), VgioStyle(; length_unit = m)
    @test JSON.json(Size2(640, 480); style = s) == """{"width":640,"height":480}"""
    @test JSON.json(Size2(width = 640px, height = 480px); style = s) == """{"width":640,"height":480}"""
    @test JSON.json(Size2(width = 210.0mm, height = 297.0mm); style = sm) == """{"width":0.21,"height":0.297}"""
    rt(x, st, tags) = JSON.lift(st, Size2, JSON.parse(JSON.json(JSON.lower(st, x, tags))), tags)
    @test rt(Size2(640, 480), s, (;)) === Size2(640, 480)
    @test rt(Size2(width = 640.0px, height = 480.0px), s, (; quantity = :pixel)) == Size2(width = 640.0px, height = 480.0px)
    a4 = Size2(width = VgtkIO.read_length(210, mm), height = VgtkIO.read_length(297, mm))
    @test rt(a4, s, (; quantity = :length)) === a4 && rt(a4, sm, (; quantity = :length)) === a4
    sp = VgioStyle(; length_roles = (pattern = pt,))
    b = Size2(width = VgtkIO.read_length(17.5, pt), height = VgtkIO.read_length(125.25, pt))
    @test JSON.lower(sp, b, (; quantity = :length, role = :pattern)) == (width = 17.5, height = 125.25)
    @test rt(b, sp, (; quantity = :length, role = :pattern)) === b
    @test_throws ArgumentError JSON.lower(s, Size2(width = 640px, height = 3.0mm), (; quantity = :pixel))
    @test_throws ArgumentError JSON.lower(s, Size2(width = 640px, height = 480px), (; quantity = :length))
end

@testitem "codec: the retired UnitContract layout as the 0.10 units object (migrate_units)" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO, VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CovMat, Uncertain
    using LinearAlgebra: Diagonal, I
    using VisualGeometryToolkitIO: migrate_units, VgioStyle, UnitContract
    using VisualGeometryToolkitCore: dpi, pt
    u = migrate_units(JSON.parse("""{"world": "mm", "density": "dpi", "lengths": {"pattern": "pt"}}"""))
    @test u == Dict{String, Any}("length" => "mm", "time" => "s", "angle" => "rad", "pixel" => "colmap", "density" => "dpi",
                                 "radiance" => "relative", "length_roles" => Dict{String, Any}("pattern" => "pt"))
    @test migrate_units(Dict("world" => "m"))["density"] == "dpi" && isempty(migrate_units(Dict("world" => "m"))["length_roles"])
    # the record it declares reads back as the contract the old layout meant
    s = VgioStyle((units = JSON.parse(JSON.json(u)),))
    @test s.contract == UnitContract(world = mm, density = dpi, lengths = (pattern = pt,))
    @test_throws ArgumentError migrate_units(Dict("world" => "px"))
    @test_throws ArgumentError migrate_units(Dict("world" => "mm", "density" => "mm"))
    @test_throws ArgumentError migrate_units(JSON.parse("""{"world": "mm", "lengths": {"world": "pt"}}"""))
end

@testitem "codec: THE frame codec — every FeatureFrame a case, exact by type (Float64 and Float32)" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO, VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CovMat, Uncertain
    using LinearAlgebra: Diagonal, I
    using VisualGeometryToolkitIO: VgioStyle
    using VisualGeometryToolkitCore: PointFrame, Disc, OrientedDisc, EllipticFrame, AffineFrame, Point2, frame_dof
    s = VgioStyle()
    frames = (PointFrame(10.5, 20.25), Disc(Point2(1.0, 2.0), 3.0), OrientedDisc(1.0, 2.0, 3.0, 0.5),
              EllipticFrame(Point2(10.5, 20.25), SMatrix{2, 2}(2.0, 0.3, 0.3, 1.5)),
              AffineFrame(Point2(1.0, 2.0), SMatrix{2, 2}(1.2, -0.2, 0.1, 0.9)))
    # the bytes: centre, then the shape under ASCII keys, matrices in rows
    @test JSON.json(frames[1]; style = s) == """{"center":[10.5,20.25]}"""
    @test JSON.json(frames[2]; style = s) == """{"center":[1.0,2.0],"sigma":3.0}"""
    @test JSON.json(frames[3]; style = s) == """{"center":[1.0,2.0],"sigma":3.0,"angle":0.5}"""
    @test JSON.json(frames[4]; style = s) == """{"center":[10.5,20.25],"shape_matrix":[[2.0,0.3],[0.3,1.5]]}"""
    @test JSON.json(frames[5]; style = s) == """{"center":[1.0,2.0],"affine_matrix":[[1.2,0.1],[-0.2,0.9]]}"""
    for f in frames, T in (Float64, Float32)
        g = JSON.lift(s, typeof(f).name.wrapper{T}, JSON.parse(JSON.json(f; style = s)))   # (a Float32 frame of the same numbers)
        @test g isa typeof(f).name.wrapper{T}
        F = typeof(g)
        back = JSON.lift(s, F, JSON.parse(JSON.json(g; style = s)))
        @test back === g
        # the uncertain frame: its covariance over frame_dof parameters, centre first, in every storage
        n = frame_dof(g)
        for Σ in (CovMat(SMatrix{n, n}(Matrix(1.0I(n)) .* 0.25 .+ 0.01)), CovMat(Diagonal(SVector{n}(0.5 .+ (1:n) ./ 8))), CovMat(VgtkCore.ScaledIdentity{n}(0.375)))
            u = Uncertain(g, Σ)
            ub = JSON.lift(s, Uncertain{F}, JSON.parse(JSON.json(u; style = s)))
            @test ub == u && typeof(ub) == typeof(u)
            @test VgtkCore.center(ub) == VgtkCore.center(u)       # the one detection → observation conversion agrees
        end
        @test_throws ArgumentError JSON.lower(s, Uncertain(g, CovMat(SMatrix{n + 1, n + 1}(Matrix(1.0I(n + 1))))))   # not frame_dof
        @test_throws ArgumentError JSON.lower(s, Uncertain(g, CovMat(SMatrix{n, n}(fill(NaN, n, n)))))              # none: a detection's status
    end
    # a frame of lengths, through the tags (its numbers in the role's unit)
    d = Disc(Point2(VgtkIO.read_length(17.5, mm), VgtkIO.read_length(12.25, mm)), VgtkIO.read_length(3.0, mm))
    tags = (; quantity = :length)
    @test JSON.lower(VgioStyle(; length_unit = m), d, tags) == (center = [0.0175, 0.01225], sigma = 0.003)
    @test JSON.lift(s, Disc, JSON.parse(JSON.json(JSON.lower(s, d, tags))), tags) == d
    @test_throws ArgumentError JSON.lower(s, d, (; quantity = :pixel))
end
