# =============================================================================
# Every file under test/ runs. ReTestItems discovers `*_tests.jl` item files
# and `*_testsetup.jl` setup files only: a file named any other way (the old
# `test_*.jl` convention, say) would sit in the directory and never run, and an
# item file whose item is not named after it breaks the `test_args` name
# filter. runtests.jl is the one exception.
# =============================================================================

@testitem "suite_layout" begin
    using Test

    stray = String[]
    for (dir, _, files) in walkdir(@__DIR__), f in files
        endswith(f, ".jl") || continue
        path = joinpath(dir, f)
        rel = relpath(path, @__DIR__)
        if rel == "runtests.jl" || endswith(f, "_testsetup.jl")
            continue
        elseif endswith(f, "_tests.jl")
            stem = f[1:(end - length("_tests.jl"))]
            occursin("@testitem \"$stem\"", read(path, String)) ||
                push!(stray, "$rel: no @testitem \"$stem\"")
        else
            push!(stray, "$rel: neither *_tests.jl nor *_testsetup.jl")
        end
    end
    @test isempty(stray)
end
