# =============================================================================
# write_record / read_record — the vgio file boundary for another package's kind
# =============================================================================

@testitem "write_record / read_record: an owner's kind, its VGC values on the style, its version steps on load" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    import JSON
    using VisualGeometryToolkitCore: EuclideanMap, mm, m
    using Rotations: RotMatrix, RotXYZ
    # a downstream owner's version step (a declared extension point)
    VgtkIO.migrate_schema(::Val{Symbol("toy/thing/v1")}, t::AbstractDict) = merge(t, Dict("schema" => "toy/thing/v2", "n" => t["count"]))
    mktempdir() do dir
        p = joinpath(dir, "thing.json")
        X = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.3)),[0.01, 0.02, 0.5] .* m)
        VgtkIO.write_record(p, (schema = "toy/thing/v2", length_unit = "mm", pose = X, n = 3); style = VgtkIO.VgioStyle(; length_unit = mm))
        text = read(p, String)
        @test startswith(text, "{\n    \"schema\"") && endswith(text, "}")          # the vgio shape
        r = VgtkIO.read_record(p)
        @test r.schema == "toy/thing/v2" && r.n === Int64(3)
        @test r.pose.translation == [10.0, 20.0, 500.0]                              # the style's unit
        Y = JSON.lift(VgtkIO.VgioStyle(; length_unit = mm), EuclideanMap{3}, r.pose)
        @test Y.R ≈ X.R && Y.t == X.t
        # an earlier version, read through the owner's step in memory (the file untouched)
        write(p, """{"schema": "toy/thing/v1", "count": 7}""")
        r = VgtkIO.read_record(p)
        @test r.schema == "toy/thing/v2" && r.n == 7
        @test occursin("toy/thing/v1", read(p, String))
        # a non-finite number is refused by its field, unless the file is a report that allows it
        @test_throws ArgumentError VgtkIO.write_record(joinpath(dir, "bad.json"), (; a = (; b = NaN)))
        @test !isfile(joinpath(dir, "bad.json"))
        q = joinpath(dir, "report.json")
        VgtkIO.write_record(q, (; a = (; b = NaN, c = Inf)); allownan = true)
        e = try VgtkIO.read_record(q); nothing catch e; e end
        @test e isa ArgumentError && occursin("a.b", e.msg)
        r = VgtkIO.read_record(q; allownan = true)
        @test isnan(r.a.b) && r.a.c == Inf
    end
end
