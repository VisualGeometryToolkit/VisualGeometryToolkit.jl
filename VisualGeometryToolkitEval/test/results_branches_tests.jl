# Results branches (src/reproduce/results_branches.jl): a full reproduction
# becomes a branch `results/<UTC stamp>-<commit>` of the paper's repository, in
# a worktree of its own, carrying the paper-facing assets, RUN.toml and
# COMPARISON.md; accept_results merges it into the base branch and tags it;
# remove_results_worktree removes the worktree (and the branch on request).
# The paper is a throwaway git repository; the user's checkout is compared
# before and after every step.

@testitem "results_branches" setup = [GitFixture] begin
    using Test
    using Dates: DateTime
    using TOML: TOML
    using VisualGeometryToolkitCore: sha256_hex
    using VisualGeometryToolkitEval.Reproduce

    git(dir, args...) = strip(read(`git -C $dir $args`, String))

    # The paper project: a subdirectory of its repository (cvpr2027 of radiometric-camera).
    repo = gitrepo()
    paper = joinpath(repo, "proj")
    numbers(path; err, speed, frames, extra = nothing) = begin
        nums = PaperNumbers(; driver = "scripts/fit.jl", config = "configs/fit.toml", run_key = "k1", commit = "c0ffee1")
        add_number!(nums, "fitErrMedian", err; digits = 2, better = :lower)
        add_number!(nums, "fitSpeed", speed; better = :higher)
        add_number!(nums, "fitFrames", frames)
        extra === nothing || add_number!(nums, "fitShare", extra; digits = 1, better = :none)
        write_numbers(path, nums)
    end
    mkpath(joinpath(paper, "paper", "latex")); mkpath(joinpath(paper, "paper", "fig"))
    numbers(joinpath(paper, "paper", "latex", "numbers.tex"); err = 1.5, speed = 10, frames = 61, extra = 0.5)
    write(joinpath(paper, "paper", "latex", "table.tex"), "\\fitErrMedian & \\fitSpeed \\\\\n")
    write(joinpath(paper, "paper", "fig", "teaser.png"), UInt8[0x89, 0x50, 0x4e, 0x47, 1, 2, 3])
    git!(repo, "add", "-A"); git!(repo, "commit", "-q", "-m", "paper")
    base = git(repo, "symbolic-ref", "--short", "HEAD")
    head = git(repo, "rev-parse", "HEAD")

    # The user's checkout as the user sees it: HEAD, the branch, the status and every file's bytes.
    snapshot() = (git(repo, "rev-parse", "HEAD"), git(repo, "symbolic-ref", "--short", "HEAD"),
                  git(repo, "status", "--porcelain", "--untracked-files=all"),
                  Dict(relpath(joinpath(d, f), repo) => read(joinpath(d, f))
                       for (d, _, fs) in walkdir(repo) if !occursin("/.git", d * "/") for f in fs))
    before = snapshot()

    # The reproduction's renders, outside the paper: the lower-better median fell, the
    # higher-better speed fell, the frames held, the share has no direction; the figure changed.
    out = mktempdir()
    numbers(joinpath(out, "numbers.tex"); err = 1.2, speed = 8, frames = 61, extra = 0.7)
    cp(joinpath(paper, "paper", "latex", "table.tex"), joinpath(out, "table.tex"))
    write(joinpath(out, "teaser.png"), UInt8[0x89, 0x50, 0x4e, 0x47, 9, 9, 9])
    @test occursin("better=lower", read(joinpath(out, "numbers.tex"), String))
    @test_throws ArgumentError add_number!(PaperNumbers(; driver = "d", config = "c", run_key = "k", commit = "x"),
                                           "a", 1; better = :smaller)
    # a table row: one direction for every cell, or one per column
    let nums = PaperNumbers(; driver = "d", config = "c", run_key = "k", commit = "x"), f = tempname()
        add_table_row!(nums, "rowA", ["err" => 1, "n" => 2]; better = ["err" => :lower])
        add_table_row!(nums, "rowB", ["err" => 1, "n" => 2]; better = :higher)
        text = read(write_numbers(f, nums), String)
        @test occursin(r"provenance \\rowAErr: .*; better=lower\n", text) && !occursin(r"provenance \\rowAN: .*better", text)
        @test occursin(r"provenance \\rowBN: .*; better=higher\n", text)
        @test_throws ArgumentError add_table_row!(nums, "rowC", ["err" => 1]; better = ["typo" => :lower])
        @test_throws ArgumentError add_table_row!(nums, "rowC", ["err" => 1]; better = ["err" => :up])
    end

    tier = mktempdir()
    write(joinpath(tier, "TIER.toml"), "name = \"rc_baselines\"\nversion = 3\n")
    scratch = joinpath(mktempdir(), "worktrees")
    assets = [joinpath(out, "numbers.tex") => "paper/latex/numbers.tex",
              joinpath(out, "table.tex") => "paper/latex/table.tex",
              joinpath(out, "teaser.png") => "paper/fig/teaser.png"]
    run = results_branch(paper, assets; scratch_root = scratch, stamp = DateTime(2026, 11, 11, 2, 15, 30),
                         run_keys = ["fit-k1", "figures-k2"], slurm_jobs = ["7181", "7182_0"], gpus = ["B200", "B200"],
                         machine = "dgx01", wall_times = ["fit" => 12.5, "figures" => 3.0],
                         tiers = ["rc_baselines" => tier, "rc_photos" => repeat("ab", 20)])

    # the branch and its worktree, named by the UTC stamp and the evaluated commit
    @test run.branch == "results/20261111_021530-" * head[1:7]
    @test dirname(run.worktree) == scratch && isdir(run.worktree)
    @test git(repo, "rev-parse", "--verify", "refs/heads/" * run.branch) == run.commit
    @test git(run.worktree, "symbolic-ref", "--short", "HEAD") == run.branch
    @test git(repo, "rev-parse", run.branch * "^") == head            # one commit on the base
    @test snapshot() == before                                          # the user's checkout is untouched

    # what the branch carries: the assets at the paper's paths, the manifest, the comparison
    files = split(git(repo, "diff", "--name-only", head, run.branch), '\n')
    @test Set(files) == Set(["proj/RUN.toml", "proj/COMPARISON.md", "proj/paper/latex/numbers.tex",
                             "proj/paper/fig/teaser.png", "proj/comparison/base/paper/fig/teaser.png"])
    @test read(joinpath(run.worktree, "proj", "paper", "fig", "teaser.png")) == read(joinpath(out, "teaser.png"))

    # the manifest
    m = TOML.parsefile(joinpath(run.worktree, "proj", "RUN.toml"))
    @test m["branch"] == run.branch && m["stamp"] == "20261111_021530"
    @test m["commit"] == head && m["dirty"] == false
    @test m["base"] == base && m["base_commit"] == head
    @test m["run_keys"] == ["fit-k1", "figures-k2"] && m["slurm_jobs"] == ["7181", "7182_0"]
    @test m["machine"] == "dgx01" && m["gpus"] == ["B200", "B200"]
    @test m["wall_times"] == Dict("fit" => 12.5, "figures" => 3.0)
    @test m["tiers"]["rc_baselines"] == Dict("hash" => sha256_hex(read(joinpath(tier, "TIER.toml"))), "of" => "TIER.toml sha256")
    @test m["tiers"]["rc_photos"] == Dict("hash" => repeat("ab", 20), "of" => "artifact git-tree-sha1")
    @test m["assets"] == ["proj/paper/latex/numbers.tex", "proj/paper/latex/table.tex", "proj/paper/fig/teaser.png"]
    @test_throws ArgumentError results_branch(paper, assets; scratch_root = scratch, tiers = ["t" => mktempdir()])

    # the comparison: a direction per number, CHANGED without one
    verdict = Dict(r.name => r for r in run.numbers)
    @test verdict["fitErrMedian"].verdict == "IMPROVED" && verdict["fitErrMedian"].delta == "-0.30"
    @test verdict["fitSpeed"].verdict == "DEGRADED" && verdict["fitSpeed"].delta == "-2"
    @test verdict["fitFrames"].verdict == "UNCHANGED"
    @test verdict["fitShare"].verdict == "CHANGED" && verdict["fitShare"].better == "none"
    md = read(joinpath(run.worktree, "proj", "COMPARISON.md"), String)
    @test occursin(r"\| `\\fitErrMedian` \|.*\| `1.50` \| `1.20` \| -0.30 \| lower \| IMPROVED \|", md)
    @test occursin("paper/fig/teaser.png", md) && occursin("![new](paper/fig/teaser.png)", md)
    @test occursin("![base](comparison/base/paper/fig/teaser.png)", md)
    @test snapshot() == before

    # acceptance without sign-off is refused, and changes nothing
    @test_throws ErrorException accept_results(paper, run.branch; replace_submitted = false)
    @test snapshot() == before && isempty(git(repo, "tag", "--list", "accepted/*"))

    # acceptance: one merge commit on the base of the assets alone, tagged accepted/<stamp>
    acc = accept_results(paper, run.branch; replace_submitted = true)
    @test acc.tag == "accepted/20261111_021530"
    @test git(repo, "rev-parse", acc.tag * "^{commit}") == acc.commit == git(repo, "rev-parse", base)
    @test split(git(repo, "rev-list", "--parents", "-n", "1", acc.commit)) == [acc.commit, head, run.commit]
    @test Set(split(git(repo, "diff", "--name-only", head, acc.commit), '\n')) ==
          Set(["proj/paper/latex/numbers.tex", "proj/paper/fig/teaser.png"])
    # the user's checkout is on the base, which moved to the merge: the accepted assets, nothing else
    after = snapshot()
    @test after[1] == acc.commit && after[2] == base && isempty(after[3])
    @test after[4]["proj/paper/fig/teaser.png"] == read(joinpath(out, "teaser.png"))
    @test after[4]["proj/paper/latex/numbers.tex"] == read(joinpath(out, "numbers.tex"))
    @test Set(keys(after[4])) == Set(keys(before[4]))
    @test_throws ErrorException accept_results(paper, run.branch; replace_submitted = true)   # the tag exists

    # removal: the worktree goes, the branch stays; a rejected run's branch on request
    remove_results_worktree(paper, run.branch)
    @test !isdir(run.worktree) && !isempty(git(repo, "branch", "--list", run.branch))
    rejected = results_branch(paper, assets; scratch_root = scratch, stamp = DateTime(2026, 11, 12))
    @test rejected.branch == "results/20261112_000000-" * acc.commit[1:7]
    remove_results_worktree(paper, rejected.branch; delete_branch = true)
    @test !isdir(rejected.worktree) && isempty(git(repo, "branch", "--list", rejected.branch))
    @test_throws ArgumentError remove_results_worktree(paper, base)
    @test snapshot() == after
end
