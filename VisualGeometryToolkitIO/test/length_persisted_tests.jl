# =============================================================================
# The writer's lengths are their persisted values (IO 0.10.1)
# =============================================================================
#
# The design of record (ruling 1: a length's identity is its metre value at 15
# significant digits, hashed as JCS; spec §5: a value built in code and the
# value loaded from the file written from it hash the same) holds only if a
# file written from a computed, full-precision length reads back as that
# length's PERSISTED value (`_persisted`, content_id's length leaf) when it
# cannot read back as the length itself. In 0.10.0
# the writer's last resort, for a value no file number reads back to (about one
# full-precision metre value in eight in mm), was the nearest file number, and
# for one in ninety (probe 15472: 2 300 of 200 000) that read back on the other
# side of a 15th-digit rounding boundary: the rig, the file written from it and
# the rig loaded back had two identities.

@testitem "lengths: a written length reads back as its persisted value" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using Random: Xoshiro
    using VisualGeometryToolkitCore: cm, μm, inch, ft, pt
    km = parse_unit("km")
    back(t, u) = ustrip(m, VgtkIO.read_length(VgtkIO.write_length(t * m, u), u))
    # the probe's example: 0.10.0 wrote -167.4265305809965 mm, which read back as
    # -0.1674265305809965 m, whose persisted value is not t's
    t = -0.16742653058099652
    @test back(t, mm) === VgtkIO._persisted(t)
    @test VgtkIO._persisted(back(t, mm)) === VgtkIO._persisted(t)
    rng = Xoshiro(1)
    ts = [(rand(rng) - 0.5) * 0.6 for _ in 1:200_000]        # computed metre values, full precision
    for u in (mm, cm, μm, m, km, inch, ft, pt)
        # the identity of the value read back is the value's own
        @test count(t -> VgtkIO._persisted(back(t, u)) !== VgtkIO._persisted(t), ts) == 0
        # write ∘ read ∘ write: the second write is the first, byte for byte
        @test count(ts) do t
            c = VgtkIO.write_length(t * m, u)
            repr(VgtkIO.write_length(VgtkIO.read_length(c, u), u)) != repr(c)
        end == 0
    end
    for u in (mm, cm, μm, m, km)
        # a decimal unit: the value itself whenever a file number reads back to it, else its persisted value
        @test count(t -> !(back(t, u) === t || back(t, u) === VgtkIO._persisted(t)), ts) == 0
    end
    @test count(t -> back(t, mm) !== t, ts) > 0                 # the batch holds values no mm number reads back to
end

@testitem "content_id: a rig of computed lengths, its file and the rig loaded back have one identity" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using Random: Xoshiro
    using VisualGeometryToolkitCore: Rig, PointTarget, PosedTarget, WorldFrame, inch, pt
    using VisualGeometryToolkitIO: content_id
    rng = Xoshiro(7)
    # full-precision lengths, as a solver computes them (not decimals read from a file)
    len3() = SVector{3}(((rand(rng, 3) .- 0.5) .* 0.6)...) .* m
    pose() = EuclideanMap{3, Float64, QuatRotation{Float64}, typeof(len3())}(QuatRotation(q0[1], q0[2], q0[3], q0[4], false), len3())
    rig = Rig{WorldFrame}(Dict(Symbol("t", i) => PosedTarget(PointTarget([len3() for _ in 1:4]), pose()) for i in 1:40))
    id = content_id(rig)
    mktempdir() do dir
        for u in (mm, inch, pt)
            p = joinpath(dir, "rig_$(unit_string(u)).json")
            save(p, rig; length_unit = u)
            back = load(Rig, p)
            @test content_id(p) == id
            @test content_id(back) == id
            q = joinpath(dir, "again_$(unit_string(u)).json")
            save(q, back; length_unit = u)
            @test read(q, String) == read(p, String)               # write ∘ read ∘ write is byte-stable
        end
    end
end
