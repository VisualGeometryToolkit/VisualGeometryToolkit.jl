# =============================================================================
# canonical_json / sha256_hex — the encoding content identities are minted from
# =============================================================================

@testitem "canonical_json" setup = [VGIoFixtures] begin
    tree = Dict("b" => 1, "a" => [1.0, 2.5, "x", nothing, true], "c" => Dict("z" => 3, "y" => -0.0))
    s = canonical_json(tree)
    @test s == "{\"a\":[1,2.5,\"x\",null,true],\"b\":1,\"c\":{\"y\":0,\"z\":3}}"   # sorted, tight, integral floats bare
    @test canonical_json(1.0e16) == "1.0e16"                # past 2^53: stays a float
    @test canonical_json(9.007199254740992e15) == "9007199254740992"
    @test canonical_json("q\"\\\n\t\x01") == "\"q\\\"\\\\\\n\\t\\u0001\""
    @test_throws ArgumentError canonical_json(NaN)
    @test_throws ArgumentError canonical_json(Inf)
    @test_throws ArgumentError canonical_json(:sym)          # not lowered: the caller's job
    io = IOBuffer(); canonical_json(io, Dict("k" => Int64(2))); @test String(take!(io)) == "{\"k\":2}"
    # the identity is stable across insertion order
    @test canonical_json(Dict("x" => 1, "y" => 2)) == canonical_json(Dict("y" => 2, "x" => 1))
end

@testitem "canonical identity" setup = [VGIoFixtures] begin
    # a content identity is Core's sha256_hex of the canonical encoding (the digest itself is Core's, tested there)
    using VisualGeometryToolkitCore: sha256_hex
    @test length(sha256_hex(canonical_json(Dict("a" => 1)))) == 64
    @test sha256_hex(canonical_json(Dict("x" => 1, "y" => 2))) == sha256_hex(canonical_json(Dict("y" => 2, "x" => 1)))
end
