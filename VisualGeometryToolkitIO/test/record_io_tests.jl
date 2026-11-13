# =============================================================================
# write_record / read_record — the vgio file boundary for another package's kind
# =============================================================================

@testitem "write_record / read_record: an owner's kind, its VGC values on the style, its version steps on load" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    import JSON
    import VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EuclideanMap, mm, m
    using Rotations: RotMatrix, RotXYZ
    # a downstream owner's version step (a declared extension point)
    VgtkIO.migrate_schema(::Val{Symbol("toy/thing/v1")}, t::AbstractDict) = merge(t, Dict("schema" => "toy/thing/v2", "n" => t["count"]))
    mktempdir() do dir
        p = joinpath(dir, "thing.json")
        X = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.3)),[0.01, 0.02, 0.5] .* m)
        VgtkIO.write_record(p, (schema = "toy/thing/v2.0", pose = X, n = 3); style = VgtkIO.VgioStyle(; length_unit = mm))
        text = read(p, String)
        @test startswith(text, "{\n    \"schema\"") && endswith(text, "}")          # the vgio shape
        r = VgtkIO.read_record(p)
        @test r.schema == "toy/thing/v2.0" && r.n === Int64(3)
        # the envelope every vgio file has, after the schema: the generator, the style's units, the fixed conventions
        has(x, k) = k in Symbol.(keys(x))
        @test Symbol.(collect(keys(r)))[1:4] == [:schema, :generator, :units, :conventions]
        @test r.generator.name == "VisualGeometryToolkitIO" && r.generator.version == string(pkgversion(VgtkIO)) && !has(r.generator, :caller)
        @test r.units.length == "mm" && r.units.pixel == "colmap" && r.units.time == "s" && r.units.angle == "rad" &&
              r.units.density == "dpi" && r.units.radiance == "relative" && isempty(r.units.length_roles)
        @test r.conventions.rotation_encoding == "quaternion_wxyz_hamilton" && r.conventions.matrix_layout == VgtkIO.MATRIX_LAYOUT &&
              r.conventions.index_base == 1 && r.conventions.grid_layout == "row_major"
        @test r.pose.translation == [10.0, 20.0, 500.0]                              # the style's unit
        s = VgtkIO.VgioStyle(r)                                                      # the style the record declares
        Y = JSON.lift(s, EuclideanMap{3}, r.pose)
        @test Y.R ≈ X.R && Y.t == X.t
        # a record states none of the envelope itself
        for k in (:generator, :units, :conventions)
            @test_throws ArgumentError VgtkIO.write_record(joinpath(dir, "env.json"), merge((schema = "toy/thing/v2.0",), NamedTuple{(k,)}(((;),))))
        end
        @test_throws ArgumentError VgtkIO.write_record(joinpath(dir, "env.json"), (; n = 1))        # nor lacks its schema
        @test !isfile(joinpath(dir, "env.json"))
        # the calling package, named by `with_caller`, in the generator; IO itself is not its own caller
        VgtkIO.with_caller(VisualGeometryToolkitCore) do
            VgtkIO.write_record(p, (schema = "toy/thing/v2.0", n = 4))
        end
        g = VgtkIO.read_record(p).generator
        @test g.caller.name == "VisualGeometryToolkitCore" && g.caller.version == string(pkgversion(VisualGeometryToolkitCore))
        VgtkIO.with_caller(() -> VgtkIO.write_record(p, (schema = "toy/thing/v2.0", n = 4)), VgtkIO)
        @test !has(VgtkIO.read_record(p).generator, :caller)
        # …and in every vgio kind a writer saves inside it
        VgtkIO.with_caller(VisualGeometryToolkitCore) do
            save(joinpath(dir, "poses.json"), [VisualGeometryToolkitCore.relabel(VisualGeometryToolkitCore.WorldFrame, VisualGeometryToolkitCore.CameraFrame, X)])
        end
        @test JSON.parsefile(joinpath(dir, "poses.json"))["generator"]["caller"]["name"] == "VisualGeometryToolkitCore"
        @test VgtkIO.validate(joinpath(dir, "poses.json")) == "transforms"
        # an earlier version, read through the owner's step in memory (the file untouched)
        write(p, """{"schema": "toy/thing/v1", "count": 7}""")
        r = VgtkIO.read_record(p)
        @test r.schema == "toy/thing/v2" && r.n == 7
        @test occursin("toy/thing/v1", read(p, String))
        # the same from JSON text, and a step's refusal names the file and its schema
        r = VgtkIO.read_record(IOBuffer("""{"schema": "toy/thing/v1", "count": 5}"""))
        @test r.schema == "toy/thing/v2" && r.n == 5
        write(p, """{"schema": "toy/thing/v1"}""")              # a v1 record always held `count`
        e = try VgtkIO.read_record(p); nothing catch e; e end
        @test e isa ArgumentError && occursin(p, e.msg) && occursin("toy/thing/v1", e.msg) && occursin("count", e.msg)
        # a non-finite number is refused by its field, unless the file is a report that allows it
        e = try VgtkIO.write_record(joinpath(dir, "bad.json"), (schema = "toy/report/v1.0", a = (; b = NaN))); nothing catch e; e end
        @test e isa ArgumentError && occursin("a.b", e.msg)
        @test !isfile(joinpath(dir, "bad.json"))
        q = joinpath(dir, "report.json")
        VgtkIO.write_record(q, (schema = "toy/report/v1.0", a = (; b = NaN, c = Inf)); allownan = true)
        e = try VgtkIO.read_record(q); nothing catch e; e end
        @test e isa ArgumentError && occursin("a.b", e.msg)
        r = VgtkIO.read_record(q; allownan = true)
        @test isnan(r.a.b) && r.a.c == Inf
    end
end
