# quality_splat_tests.jl — VisualGeometryToolkitCore's `check_splat`: a call that
# splats a collection whose length is data (Features 0.8.3's
# `zip(map(_identity_fields, ids)...)`, one `zip` per capture size) is found
# from the typed code, and a fixed-length splat is not. Core itself is checked by
# its `splat` gate in "quality: gates".

@testitem "quality: check_splat finds planted data-length splats" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: check_splat, SplatViolation
    import VisualGeometryToolkitCore as VgtkCore

    dir = mktempdir()
    file = joinpath(dir, "src", "planted.jl")
    mkpath(dirname(file))
    write(file, """
    using StaticArrays: SVector
    vec_splat(v::Vector{Int}) = max(v...)                      # a Vector
    gen_splat(n::Int) = max((i for i in 1:n)...)               # a generator
    bare_splat(rows::AbstractVector) = tuple(rows...)          # a bare name, abstract
    tuple_splat(t::NTuple{3, Int}) = max(t...)
    named_splat(x::NamedTuple{(:a, :b), Tuple{Int, Int}}) = tuple(x...)
    static_splat(v::SVector{3, Float64}) = max(v...)
    pair_splat(p::Pair{Int, Int}) = max(p...)
    _pad(k) = (k, k)
    returned_splat(k::Int) = max(_pad(k)...)                   # a call returning a Tuple
    """)
    M = Module(:SplatPlanted)
    Base.include(M, file)
    toml(s) = write(joinpath(dir, "DUPLICATION.toml"), s)

    found = check_splat(M; root = dir)
    @test all(v -> v isa SplatViolation && v.check === :splat && v.file == "src/planted.jl", found)
    @test sort!([v.name for v in found]) == ["bare_splat", "gen_splat", "vec_splat"]
    byname = Dict(v.name => v for v in found)
    @test (byname["vec_splat"].line, byname["vec_splat"].callee, byname["vec_splat"].type) == (2, "max", "Vector{Int64}")
    @test startswith(byname["gen_splat"].type, "Base.Generator")
    @test (byname["bare_splat"].line, byname["bare_splat"].callee, byname["bare_splat"].type) == (4, "tuple", "AbstractVector")
    @test occursin("src/planted.jl:2  [splat] vec_splat: max(…::Vector{Int64}...)", sprint(show, byname["vec_splat"]))

    # an allow-list entry silences its function; one that matches nothing, or names a gone file, is a finding
    toml("""
    [[splat]]
    file = "src/planted.jl"
    function = "vec_splat"
    reason = "the test's"
    """)
    @test sort!([v.name for v in check_splat(M; root = dir)]) == ["bare_splat", "gen_splat"]
    toml("""
    [[splat]]
    file = "src/planted.jl"
    function = "tuple_splat"
    reason = "a Tuple's splat needs none"

    [[splat]]
    file = "src/gone.jl"
    function = "f"
    reason = "its file was deleted"
    """)
    stale = filter(v -> v.check === :stale_exception, check_splat(M; root = dir))
    @test sort!([v.name for v in stale]) == ["f", "tuple_splat"]
    @test all(v -> v.file == "DUPLICATION.toml" && occursin("matches nothing", sprint(show, v)), stale)
    toml("""
    [[splat]]
    file = "src/planted.jl"
    function = "vec_splat"
    """)
    @test_throws ArgumentError check_splat(M; root = dir)     # an entry without its reason
    rm(joinpath(dir, "DUPLICATION.toml"))

    # a module that is not a package needs its root
    @test_throws ArgumentError check_splat(M)
    # the gate modes
    @test_throws ArgumentError VgtkCore.test_splat(M; root = dir, mode = :warn)
    @test VgtkCore.test_splat(M; root = dir, mode = :off) === nothing
end
