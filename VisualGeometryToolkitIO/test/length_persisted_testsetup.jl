# The `PersistedLengthClaims` test setup (length_persisted_tests.jl): the claims of IO 0.10.1's writer over a batch of
# computed metre values, one function for the default suite's small batch and the benchmark suite's 200 000.

@testsetup module PersistedLengthClaims
    using Test
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: m, mm, cm, μm, inch, ft, pt, parse_unit
    using FlexUnits: ustrip
    export persisted_length_claims

    back(t, u) = ustrip(m, VgtkIO.read_length(VgtkIO.write_length(t * m, u), u))
    function persisted_length_claims(ts)
        for u in (mm, cm, μm, m, parse_unit("km"), inch, ft, pt)
            # the identity of the value read back is the value's own
            @test count(t -> VgtkIO._persisted(back(t, u)) !== VgtkIO._persisted(t), ts) == 0
            # write ∘ read ∘ write: the second write is the first, byte for byte
            @test count(ts) do t
                c = VgtkIO.write_length(t * m, u)
                repr(VgtkIO.write_length(VgtkIO.read_length(c, u), u)) != repr(c)
            end == 0
        end
        for u in (mm, cm, μm, m, parse_unit("km"))
            # a decimal unit: the value itself whenever a file number reads back to it, else its persisted value
            @test count(t -> !(back(t, u) === t || back(t, u) === VgtkIO._persisted(t)), ts) == 0
        end
        @test count(t -> back(t, mm) !== t, ts) > 0                 # the batch holds values no mm number reads back to
        return nothing
    end
end
