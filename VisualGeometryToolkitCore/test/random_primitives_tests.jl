@testitem "random_primitives: rand(Line2D), rand(Plane3D) through Random's sampler, never ambiguous with an RNG package" begin
    using Test, Random
    using StableRNGs: StableRNG
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: Line2D, Plane3D
    import CUDA   # its `rand(rng::CUDA.RNG, T::Type)` met the old `rand(::AbstractRNG, ::Type{Line2D})` methods ambiguously

    # the ambiguity this guards: Random.rand's methods that involve a Core primitive, with CUDA loaded
    core_primitive(m) = occursin(r"Line2D|Plane3D", string(m.sig))
    @test isempty(filter(((a, b),) -> core_primitive(a) || core_primitive(b),
        Test.detect_ambiguities(Random; recursive = true)))
    @test !any(m -> m.sig <: Tuple{typeof(rand), Random.AbstractRNG, Type{<:Union{Line2D, Plane3D}}}, methods(rand))

    rng = StableRNG(7)
    for _ in 1:50
        l = rand(rng, Line2D)
        @test l isa Line2D{Float64}
        c = l.coeffs
        @test norm(c[1:2]) ≈ 1 && 0 <= -c[3] < 1
        p = rand(rng, Plane3D{Float32})
        @test p isa Plane3D{Float32}
        @test norm(p.coeffs[1:3]) ≈ 1 && -1 <= p.coeffs[4] < 1
    end
    @test rand(Line2D) isa Line2D{Float64} && rand(Plane3D) isa Plane3D{Float64}
    @test rand(StableRNG(1), Line2D{Float64}) == rand(StableRNG(1), Line2D)   # one sampler for both spellings
    @test length(rand(rng, Plane3D, 3)) == 3
end
