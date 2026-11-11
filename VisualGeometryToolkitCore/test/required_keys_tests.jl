@testitem "required_key, is_finite_real: a required key refused by name, a checked number" begin
    # The record readers each wrote them (VisualGeometryToolkitIO's `_need` and
    # `_isreal`, CaliBlobs' session record's `_need` and `_isnum`): one copy each,
    # beside check_keys.
    using Test
    using VisualGeometryToolkitCore: required_key, is_finite_real, thrown_message
    d = Dict(:fx => 800.0, "name" => "cam")
    @test required_key(d, :fx, "cameras.json: left") === 800.0
    @test required_key(d, "name", "x") == "cam"
    msg = thrown_message(() -> required_key(d, :fy, "cameras.json: left"))
    @test occursin("cameras.json: left", msg) && occursin("`fy`", msg) && occursin("required", msg)
    @test is_finite_real(1) && is_finite_real(2.5) && is_finite_real(1 // 3)
    @test !is_finite_real(true) && !is_finite_real(NaN) && !is_finite_real(Inf) && !is_finite_real("1") && !is_finite_real(nothing)
    @test !is_finite_real(big(10)^400)                  # finite, but no Float64 holds it
end

@testitem "free_names is a contract generic: a calibrator's own lens target extends it" begin
    using VisualGeometryToolkitCore: CONTRACT_GENERICS, free_names
    @test free_names in CONTRACT_GENERICS
end
