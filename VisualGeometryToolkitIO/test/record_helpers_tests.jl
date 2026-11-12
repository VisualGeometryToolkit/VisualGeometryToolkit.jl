@testitem "Record, snap_integer, MATRIX_LAYOUT: the record helpers other packages' lifts and writers use" begin
    # BlobBoards' lifts restated `Record` and `snap_integer` (its Integer and
    # Rational methods beside the float one), CaliBlobs' session record the matrix
    # layout: one public copy each, here.
    using Test
    import JSON
    using VisualGeometryToolkitIO: VisualGeometryToolkitIO, Record, snap_integer, MATRIX_LAYOUT
    @test Base.ispublic(VisualGeometryToolkitIO, :Record) && Base.ispublic(VisualGeometryToolkitIO, :snap_integer)
    @test JSON.parse("{\"a\": 1}") isa Record
    @test (; a = 1) isa Record
    @test !(Dict("a" => 1) isa Record)                    # no property access
    @test snap_integer(640.0) === 640 && snap_integer(0.5) === 0.5
    @test snap_integer(2.0^60) === 2.0^60                 # past the exact integers of a Float64: kept
    @test snap_integer(Int32(3)) === 3 && snap_integer(UInt8(7)) === 7 && snap_integer(big(5)) === 5
    @test snap_integer(true) === true && snap_integer(false) === false   # a flag is not a number: as it is
    @test snap_integer(typemax(UInt64)) === typemax(UInt64)              # past Int: kept, written exactly
    @test snap_integer(big(2)^70) == big(2)^70
    @test snap_integer(6 // 2) === 3 && snap_integer(1 // 2) === 0.5
    @test snap_integer("rows") == "rows"                  # anything else as it is
    @test MATRIX_LAYOUT == "rows" && Base.ispublic(VisualGeometryToolkitIO, :MATRIX_LAYOUT)
end

@testitem "load and save are contract generics: another package's kinds extend the two verbs" begin
    # BlobBoards' boards and patterns are read and written through IO's two verbs
    # (`load(::Type{T}, path)`, `save(path, x)`): IO declares them extension points.
    using Test
    import VisualGeometryToolkitIO as VgtkIO
    @test VgtkIO.load in VgtkIO.CONTRACT_GENERICS
    @test VgtkIO.save in VgtkIO.CONTRACT_GENERICS
    @test VgtkIO.migrate_schema in VgtkIO.CONTRACT_GENERICS
end

@testitem "with_provenance and metres: a writer's provenance and a file length as plain metres, for another package's kind" begin
    # RadiometricCamera's session records called both by their former private names.
    using Test
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: mm, m, μm, inch, ustrip
    @test Base.ispublic(VgtkIO, :with_provenance) && Base.ispublic(VgtkIO, :metres)

    body = (schema = "vgio/x/v1", length_unit = "mm", value = 1)
    @test VgtkIO.with_provenance(body, (;)) === body                    # empty: the body as it is
    p = (tool = "rc", commit = "abc")
    b = VgtkIO.with_provenance(body, p)
    @test b == merge(body, (provenance = p,)) && last(keys(b)) === :provenance

    # read_length stripped: decimal-exact for a power-of-ten unit, a Float64
    @test VgtkIO.metres(13.9, mm) === 0.0139 && VgtkIO.metres(62.8, mm) === 0.0628
    @test VgtkIO.metres(5, μm) === 5.0e-6 && VgtkIO.metres(2, m) === 2.0
    for (x, u) in ((13.9, mm), (1.3, mm), (0.25, inch), (7, m))
        @test VgtkIO.metres(x, u) === Float64(ustrip(m, VgtkIO.read_length(x, u)))
    end
end
