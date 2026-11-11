# Where a driver's numbers come from (src/reproduce/number_provenance.jl): the
# checkout's commit for numbers a driver computes itself, as a PaperNumbers
# `commit` field, and a run's recorded provenance for numbers read from it.

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

# A run's number provenance: its recorded run_key, gitcommit and config_source,
# each only where the run recorded it, the config made project-relative.
@testitem "number_provenance: a run's provenance" begin
    using Test
    using VisualGeometryToolkitEval.Reproduce: number_provenance, PaperNumbers, add_number!, write_numbers
    root = mktempdir()
    run_dir = mkpath(joinpath(root, "results", "cell-3f2a"))
    write(joinpath(run_dir, "_provenance.toml"), """
          run_key = "3f2a"
          gitcommit = "v0.9;x-2-gabc1234-dirty"
          config_source = "$(joinpath(root, "configs", "run.toml"))"
          driver = "scripts/run.jl"
          """)
    p = number_provenance(run_dir; root)
    @test p == (; run_key = "3f2a", commit = "v0.9_x-2-gabc1234-dirty", config = "configs/run.toml")
    write(joinpath(run_dir, "_provenance.toml"), "gitcommit = \"abc1234\"\nconfig_source = \"configs/run.toml\"\n")
    @test number_provenance(run_dir; root) == (; commit = "abc1234", config = "configs/run.toml")   # relative kept
    nums = PaperNumbers(; driver = "scripts/fig.jl", config = "configs/fig.toml", run_key = "3f2a",
                        number_provenance(run_dir; root)...)
    add_number!(nums, "medianError", 1.5; digits = 1)
    text = read(write_numbers(joinpath(root, "n.tex"), nums), String)
    @test occursin("config=configs/run.toml; run=3f2a; commit=abc1234", text)
    @test_throws ErrorException number_provenance(mktempdir(); root)                           # no record
end
