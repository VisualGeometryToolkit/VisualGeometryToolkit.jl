# Where a driver's numbers come from (src/reproduce/number_provenance.jl): the
# checkout's commit for numbers a driver computes itself, as a PaperNumbers
# `commit` field.

@testitem "number_provenance" setup = [GitFixture] begin
    using Test
    using DrWatson: gitdescribe
    using VisualGeometryToolkitEval.Reproduce: checkout_commit, PaperNumbers
    dir = gitrepo()
    described = gitdescribe(dir)
    @test checkout_commit(dir) == described                      # a sha: nothing to sanitize
    write(joinpath(dir, "a.txt"), "edited\n")
    @test checkout_commit(dir) == described * "-dirty"           # an uncommitted tree says so
    git!(dir, "tag", "-a", "v0;1", "-m", "a tag holding the provenance comment's separator")
    @test occursin(';', gitdescribe(dir))
    @test checkout_commit(dir) == replace(gitdescribe(dir), ';' => '_')
    @test checkout_commit(mktempdir()) == "unknown"              # outside a repository
    nums = PaperNumbers(; driver = "scripts/a.jl", config = "configs/a.toml", run_key = "none",
                        commit = checkout_commit(dir))           # a field PaperNumbers accepts
    @test length(nums) == 0
end
