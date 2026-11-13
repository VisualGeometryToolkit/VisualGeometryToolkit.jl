# =============================================================================
# canonical_json / sha256_hex — RFC 8785 (JCS), the encoding content identities are minted from
# =============================================================================
#
# The vectors are RFC 8785's own: Appendix B (the IEEE-754 → ECMAScript number
# table), §3.2.3 (member sorting by UTF-16 code units) and §3.2.4 (the
# serialization example), so any JCS implementation reproduces these bytes.

@testitem "canonical_json: RFC 8785 Appendix B numbers" begin
    using Test
    using VisualGeometryToolkitCore: canonical_json, sha256_hex
    bits(h) = reinterpret(Float64, parse(UInt64, h; base = 16))
    # (IEEE-754 bits, the JCS text), RFC 8785 Appendix B
    for (h, s) in (
            ("0000000000000000", "0"), ("8000000000000000", "0"),                  # ±0 is 0
            ("0000000000000001", "5e-324"), ("8000000000000001", "-5e-324"),      # the smallest subnormal
            ("7fefffffffffffff", "1.7976931348623157e+308"), ("ffefffffffffffff", "-1.7976931348623157e+308"),
            ("4340000000000000", "9007199254740992"), ("c340000000000000", "-9007199254740992"),   # ±2^53
            ("4430000000000000", "295147905179352830000"),                      # 2^68: no exponent below 1e21
            ("44b52d02c7e14af5", "9.999999999999997e+22"), ("44b52d02c7e14af6", "1e+23"),
            ("44b52d02c7e14af7", "1.0000000000000001e+23"),
            ("444b1ae4d6e2ef4e", "999999999999999700000"), ("444b1ae4d6e2ef4f", "999999999999999900000"),
            ("444b1ae4d6e2ef50", "1e+21"),                                       # the exponent's lower bound
            ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"), ("3eb0c6f7a0b5ed8d", "0.000001"),   # its upper bound
            ("41b3de4355555553", "333333333.3333332"), ("41b3de4355555554", "333333333.33333325"),
            ("41b3de4355555555", "333333333.3333333"), ("41b3de4355555556", "333333333.3333334"),
            ("41b3de4355555557", "333333333.33333343"),
            ("becbf647612f3696", "-0.0000033333333333333333"),
            ("43143ff3c1cb0959", "1424953923781206.2"),
        )
        @test canonical_json(bits(h)) == s
    end
    # NaN and ±Infinity have no JSON form (Appendix B: "N/A")
    @test_throws ArgumentError canonical_json(bits("7fffffffffffffff"))
    @test_throws ArgumentError canonical_json(bits("7ff0000000000000"))
    @test_throws ArgumentError canonical_json(-Inf)
    # the boundaries of the plain form, ECMA-262 Number::toString: 21 integer digits, 6 leading zeros
    @test canonical_json(1.0e20) == "100000000000000000000"
    @test canonical_json(1.5e-6) == "0.0000015"
    @test canonical_json(1.5e-7) == "1.5e-7"
    @test canonical_json(1.0) == "1" && canonical_json(-2.5) == "-2.5" && canonical_json(1.0e-5) == "0.00001"
    @test canonical_json(123.456e10) == "1234560000000"
    @test canonical_json(5.0e-7) == "5e-7"
    # an integer is the double it is: exact up to 2^53, refused past it
    @test canonical_json(Int64(2)^53) == "9007199254740992"
    @test_throws ArgumentError canonical_json(Int64(2)^53 + 1)
end

@testitem "canonical_json: RFC 8785 strings, sorting and the §3.2.4 example" begin
    using Test
    using VisualGeometryToolkitCore: canonical_json, sha256_hex
    # §3.2.3: members sorted by their names' UTF-16 code units (an astral character's surrogates
    # sort before U+FB33, unlike its code point), names and strings escaped minimally
    sorting = JSON.parse("""{
        "\\u20ac": "Euro Sign", "\\r": "Carriage Return", "\\ufb33": "Hebrew Letter Dalet With Dagesh",
        "1": "One", "\\ud83d\\ude00": "Emoji: Grinning Face", "\\u0080": "Control",
        "\\u00f6": "Latin Small Letter O With Diaeresis"}""")
    @test canonical_json(sorting) ==
        "{\"\\r\":\"Carriage Return\",\"1\":\"One\",\"\u0080\":\"Control\",\"ö\":\"Latin Small Letter O With Diaeresis\"," *
        "\"€\":\"Euro Sign\",\"😀\":\"Emoji: Grinning Face\",\"\ufb33\":\"Hebrew Letter Dalet With Dagesh\"}"
    # §3.2.4 (Figure 1 → its canonical form)
    example = JSON.parse("""{
        "numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],
        "string": "\\u20ac\$\\u000F\\u000aA'\\u0042\\u0022\\u005c\\\\\\"\\/",
        "literals": [null, true, false]}""")
    @test canonical_json(example) ==
        "{\"literals\":[null,true,false],\"numbers\":[333333333.3333333,1e+30,4.5,0.002,1e-27]," *
        "\"string\":\"€\$\\u000f\\nA'B\\\"\\\\\\\\\\\"/\"}"
    # every control character escapes as \u00xx (lowercase hex) unless it has a short form
    @test canonical_json("q\"\\\n\t\x01\b\f\r\x1f") == "\"q\\\"\\\\\\n\\t\\u0001\\b\\f\\r\\u001f\""
    @test canonical_json("\x7f/é") == "\"\x7f/é\""                       # DEL, the solidus and non-ASCII as they are
    tree = Dict("b" => 1, "a" => [1.0, 2.5, "x", nothing, true], "c" => Dict("z" => 3, "y" => -0.0))
    @test canonical_json(tree) == "{\"a\":[1,2.5,\"x\",null,true],\"b\":1,\"c\":{\"y\":0,\"z\":3}}"
    @test_throws ArgumentError canonical_json(:sym)          # not lowered: the caller's job
    io = IOBuffer(); canonical_json(io, Dict("k" => Int64(2))); @test String(take!(io)) == "{\"k\":2}"
    @test canonical_json(Dict("x" => 1, "y" => 2)) == canonical_json(Dict("y" => 2, "x" => 1))
end

@testitem "canonical identity" begin
    using Test
    using VisualGeometryToolkitCore: canonical_json, sha256_hex
    # a content identity is sha256_hex of the canonical encoding
    @test sha256_hex(canonical_json(Dict("a" => 1))) == "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"   # sha256 of {"a":1}
    @test sha256_hex(canonical_json(Dict("x" => 1, "y" => 2))) == sha256_hex(canonical_json(Dict("y" => 2, "x" => 1)))
end
