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
    @test snap_integer(Int32(3)) === 3
    @test snap_integer(6 // 2) === 3 && snap_integer(1 // 2) === 0.5
    @test snap_integer("rows") == "rows"                  # anything else as it is
    @test MATRIX_LAYOUT == "rows" && Base.ispublic(VisualGeometryToolkitIO, :MATRIX_LAYOUT)
end
