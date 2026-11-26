# The golden framework's own tests (bits.jl's comparison under a FieldMap): run from the worktree root with
#   JULIA_LOAD_PATH="@:$PWD:@stdlib" julia --project=VisualGeometryToolkitEstimation/test ci/golden/framework_tests.jl
using Test
using VisualGeometryToolkitCore
include(joinpath(@__DIR__, "bits.jl"))

module RetagFixture
struct ImageMeasure end
struct SolidAngleMeasure end
end

@testset "FieldMap retagged: a recorded Symbol against the new singleton's tag" begin
    old = bit_record((; weight = :image))
    new = bit_record((; weight = RetagFixture.ImageMeasure()))
    @test only(t for (p, t, _) in old if p == "weight") == "Symbol"
    @test only(t for (p, t, _) in new if p == "weight") == "singleton"
    m = FieldMap(; retyped = [""], retagged = ["weight" => "Measure"])
    @test isempty(compare_record(old, new; map = m))                                   # the tag rule: ImageMeasure → "image"
    # a singleton of another tag is a difference, never matched
    other = bit_record((; weight = RetagFixture.SolidAngleMeasure()))
    @test !isempty(compare_record(old, other; map = m))
    @test any(d -> occursin("retagged", d), compare_record(old, other; map = m))
    # the suffix is the declared one: without it the tag is "image_measure", which is not the recorded Symbol
    @test !isempty(compare_record(old, new; map = FieldMap(; retyped = [""], retagged = ["weight" => ""])))
    # an undeclared Symbol → singleton change is a difference
    @test !isempty(compare_record(old, new; map = FieldMap(; retyped = [""])))
    # a retagged path still requires a Symbol on the recorded side
    @test !isempty(compare_record(bit_record((; weight = "image")), new; map = m))
end
