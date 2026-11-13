# =============================================================================
# canonical_json / sha256_hex — RFC 8785 (JCS), the encoding content identities are minted from
# =============================================================================
#
# The vectors are RFC 8785's own: Appendix B (the IEEE-754 → ECMAScript number
# table), §3.2.3 (member sorting by UTF-16 code units) and §3.2.4 (the
# serialization example), so any JCS implementation reproduces these bytes.

# canonical_json is VisualGeometryToolkitCore's (src/base/canonical.jl, RFC 8785; its Appendix B, §3.2.3 and §3.2.4
# vectors are Core's canonical_tests): IO re-exports it, and its content identities hash it.
@testitem "canonical identity" setup = [VGIoFixtures] begin
    # a content identity is Core's sha256_hex of the canonical encoding (the digest itself is Core's, tested there)
    using VisualGeometryToolkitCore: sha256_hex
    @test sha256_hex(canonical_json(Dict("a" => 1))) == "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"   # sha256 of {"a":1}
    @test sha256_hex(canonical_json(Dict("x" => 1, "y" => 2))) == sha256_hex(canonical_json(Dict("y" => 2, "x" => 1)))
end
