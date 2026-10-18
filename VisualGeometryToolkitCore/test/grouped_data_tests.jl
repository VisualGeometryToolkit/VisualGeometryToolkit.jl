# Grouped data: the block order of a dict of groups and its inverse split.
@testitem "grouped_data" begin
    using Test
    using VisualGeometryToolkitCore: block_order, split_blocks

    g = Dict(3 => [1.0], 1 => [2.0, 3.0], 2 => Float64[])
    @test block_order(g) == [1, 2, 3]
    @test block_order(g, Dict(1 => :a, 2 => :b, 3 => :c)) == [1, 2, 3]
    @test_throws ArgumentError block_order(g, Dict(1 => :a, 2 => :b))
    @test_throws ArgumentError block_order(g, Dict(1 => :a, 2 => :b, 4 => :c))

    v = collect(1.0:5.0)
    d = split_blocks(v, [:a, :b, :c], (2, 0, 3))
    @test d == Dict(:a => [1.0, 2.0], :b => Float64[], :c => [3.0, 4.0, 5.0])
    @test d isa Dict{Symbol, Vector{Float64}}
    @test_throws DimensionMismatch split_blocks(v, [:a, :b], (2, 2))
    @test_throws DimensionMismatch split_blocks(v, [:a, :b], (2, 2, 1))
end
