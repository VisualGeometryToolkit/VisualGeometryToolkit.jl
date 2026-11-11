# Results branches (src/reproduce/results_branches.jl): a full reproduction
# becomes a branch `results/<UTC stamp>-<commit>` of the paper's repository, in
# a worktree of its own, carrying the paper-facing assets, RUN.toml and
# COMPARISON.md; accept_results merges it into the base branch and tags it;
# remove_results_worktree removes the worktree (and the branch on request).
# The paper is a throwaway git repository; the user's checkout is compared
# before and after every step, and no step leaves a worktree or a scratch
# directory behind that it did not mean to.

@testitem "results_branches" setup = [GitFixture] begin
    using Test
    using Dates: DateTime
    using TOML: TOML
    using VisualGeometryToolkitCore: sha256_hex
    using VisualGeometryToolkitEval.Reproduce

    git(dir, args...) = strip(read(`git -C $dir $args`, String))
    worktrees(repo) = count(startswith("worktree "), split(git(repo, "worktree", "list", "--porcelain"), '\n'))

    numbers(path; err, speed, frames = 61, extra = 0.5, len = "5\\,mm", fresh = nothing) = begin
        nums = PaperNumbers(; driver = "scripts/fit.jl", config = "configs/fit.toml", run_key = "k1", commit = "c0ffee1")
        add_number!(nums, "fitErrMedian", err; digits = 2, better = :lower)
        add_number!(nums, "fitSpeed", speed; better = "higher")                 # a String direction
        frames === nothing || add_number!(nums, "fitFrames", frames)
        add_number!(nums, "fitShare", extra; digits = 1, better = :none)
        add_number!(nums, "fitLen", len; better = :lower)                       # a typeset value with its unit
        fresh === nothing || add_number!(nums, "fitFresh", fresh)
        write_numbers(path, nums)
    end

    # A paper project in a subdirectory of its repository (cvpr2027 of radiometric-camera),
    # its base commit, and a reproduction's renders outside it.
    function fixture()
        repo = gitrepo()
        paper = joinpath(repo, "proj")
        mkpath(joinpath(paper, "paper", "latex")); mkpath(joinpath(paper, "paper", "fig"))
        numbers(joinpath(paper, "paper", "latex", "numbers.tex"); err = 1.5, speed = 10)
        write(joinpath(paper, "paper", "latex", "table.tex"), "\\fitErrMedian & \\fitSpeed \\\\\n")
        write(joinpath(paper, "paper", "fig", "teaser.png"), UInt8[0x89, 0x50, 0x4e, 0x47, 1, 2, 3])
        git!(repo, "add", "-A"); git!(repo, "commit", "-q", "-m", "paper")
        # the renders: the lower-better median fell, the higher-better speed fell, the frames held,
        # the share has no direction, the length changed its unit; the figure changed
        out = mktempdir()
        numbers(joinpath(out, "numbers.tex"); err = 1.2, speed = 8, extra = 0.7, len = "5\\,cm")
        cp(joinpath(paper, "paper", "latex", "table.tex"), joinpath(out, "table.tex"))
        write(joinpath(out, "teaser.png"), UInt8[0x89, 0x50, 0x4e, 0x47, 9, 9, 9])
        assets = [joinpath(out, "numbers.tex") => "paper/latex/numbers.tex",
                  joinpath(out, "table.tex") => "paper/latex/table.tex",
                  joinpath(out, "teaser.png") => "paper/fig/teaser.png"]
        scratch = joinpath(mktempdir(), "worktrees")
        return (; repo, paper, out, assets, scratch, base = git(repo, "symbolic-ref", "--short", "HEAD"),
                head = git(repo, "rev-parse", "HEAD"))
    end
    # The user's checkout as the user sees it: HEAD, the branch, the status and every file's bytes.
    snapshot(repo) = (git(repo, "rev-parse", "HEAD"), git(repo, "symbolic-ref", "--short", "HEAD"),
                      git(repo, "status", "--porcelain", "--untracked-files=all"),
                      Dict(relpath(joinpath(d, f), repo) => read(joinpath(d, f))
                           for (d, _, fs) in walkdir(repo) if !occursin("/.git", d * "/") for f in fs))
    complete(f; kw...) = results_branch(f.paper, f.assets; scratch_root = f.scratch, gpus = ["B200"], wall_times = ["fit" => 1.0],
                                        tiers = ["photos" => repeat("ab", 20)], kw...)

    @testset "a number's direction" begin
        @test_throws ArgumentError add_number!(PaperNumbers(; driver = "d", config = "c", run_key = "k", commit = "x"),
                                               "a", 1; better = :smaller)
        @test_throws ArgumentError add_number!(PaperNumbers(; driver = "d", config = "c", run_key = "k", commit = "x"),
                                               "a", 1; better = "up")
        # a table row: one direction for every cell (a Symbol or a String), or one per column
        nums = PaperNumbers(; driver = "d", config = "c", run_key = "k", commit = "x")
        add_table_row!(nums, "rowA", ["err" => 1, "n" => 2]; better = ["err" => :lower])
        add_table_row!(nums, "rowB", ["err" => 1, "n" => 2]; better = :higher)
        add_table_row!(nums, "rowD", ["err" => 1]; better = "lower")
        text = read(write_numbers(tempname(), nums), String)
        @test occursin(r"provenance \\rowAErr: .*; better=lower\n", text) && !occursin(r"provenance \\rowAN: .*better", text)
        @test occursin(r"provenance \\rowBN: .*; better=higher\n", text)
        @test occursin(r"provenance \\rowDErr: .*; better=lower\n", text)
        @test_throws ArgumentError add_table_row!(nums, "rowC", ["err" => 1]; better = ["typo" => :lower])
        @test_throws ArgumentError add_table_row!(nums, "rowC", ["err" => 1]; better = ["err" => :up])
        @test_throws ArgumentError add_table_row!(nums, "rowC", ["err" => 1]; better = "up")
    end

    @testset "run, comparison, acceptance and removal" begin
        f = fixture()
        (; repo, paper, out, assets, scratch, base, head) = f
        before = snapshot(repo)
        tier = mktempdir()
        write(joinpath(tier, "TIER.toml"), "name = \"rc_baselines\"\nversion = 3\n")
        run = results_branch(paper, assets; scratch_root = scratch, stamp = DateTime(2026, 11, 11, 2, 15, 30),
                             run_keys = ["fit-k1", "figures-k2"], slurm_jobs = ["7181", "7182_0"], gpus = ["B200", "B200"],
                             machine = "dgx01", wall_times = ["fit" => 12.5, "figures" => 3.0],
                             tiers = ["rc_baselines" => tier, "rc_photos" => repeat("ab", 20)])

        # the branch and its worktree, named by the UTC stamp and the evaluated commit
        @test run.branch == "results/20261111_021530-" * head[1:7]
        @test dirname(run.worktree) == scratch && isdir(run.worktree) && readdir(scratch) == [basename(run.worktree)]
        @test git(repo, "rev-parse", "--verify", "refs/heads/" * run.branch) == run.commit
        @test git(run.worktree, "symbolic-ref", "--short", "HEAD") == run.branch
        @test git(repo, "rev-parse", run.branch * "^") == head            # one commit on the base
        @test snapshot(repo) == before && worktrees(repo) == 2            # the user's checkout is untouched

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
        @test m["missing"] == String[]
        @test_throws ArgumentError results_branch(paper, assets; scratch_root = scratch, tiers = ["t" => mktempdir()])
        @test_throws ArgumentError results_branch(paper, assets; scratch_root = scratch, wall_times = ["fit" => "slow"])
        @test worktrees(repo) == 2

        # the comparison: a direction per number, CHANGED without one or when the unit changed
        verdict = Dict(r.name => r for r in run.numbers)
        @test verdict["fitErrMedian"].verdict == "IMPROVED" && verdict["fitErrMedian"].delta == "-0.30"
        @test verdict["fitSpeed"].verdict == "DEGRADED" && verdict["fitSpeed"].delta == "-2"
        @test verdict["fitFrames"].verdict == "UNCHANGED"
        @test verdict["fitShare"].verdict == "CHANGED" && verdict["fitShare"].better == "none"
        @test verdict["fitLen"].verdict == "CHANGED" && verdict["fitLen"].better == "lower"
        md = read(joinpath(run.worktree, "proj", "COMPARISON.md"), String)
        @test occursin(r"\| `\\fitErrMedian` \|.*\| `1.50` \| `1.20` \| -0.30 \| lower \| IMPROVED \|", md)
        @test occursin("paper/fig/teaser.png", md) && occursin("![new](paper/fig/teaser.png)", md)
        @test occursin("![base](comparison/base/paper/fig/teaser.png)", md)
        @test occursin("against `$base` at $(head[1:7])", md)                       # the base it compared against
        @test snapshot(repo) == before

        # acceptance without sign-off is refused by deploy_assets, and changes nothing
        @test_throws "differ from the submitted" accept_results(paper, run.branch; replace_submitted = false, scratch_root = scratch)
        @test snapshot(repo) == before && isempty(git(repo, "tag", "--list", "accepted/*"))
        @test worktrees(repo) == 2 && readdir(scratch) == [basename(run.worktree)]

        # acceptance: one merge commit on the base of the assets alone, tagged accepted/<stamp>
        acc = accept_results(paper, run.branch; replace_submitted = true, scratch_root = scratch)
        @test acc.tag == "accepted/20261111_021530"
        @test git(repo, "rev-parse", acc.tag * "^{commit}") == acc.commit == git(repo, "rev-parse", base)
        @test split(git(repo, "rev-list", "--parents", "-n", "1", acc.commit)) == [acc.commit, head, run.commit]
        @test Set(split(git(repo, "diff", "--name-only", head, acc.commit), '\n')) ==
              Set(["proj/paper/latex/numbers.tex", "proj/paper/fig/teaser.png"])
        @test worktrees(repo) == 2 && readdir(scratch) == [basename(run.worktree)]
        # the user's checkout is on the base, which moved to the merge: the accepted assets, nothing else
        after = snapshot(repo)
        @test after[1] == acc.commit && after[2] == base && isempty(after[3])
        @test after[4]["proj/paper/fig/teaser.png"] == read(joinpath(out, "teaser.png"))
        @test after[4]["proj/paper/latex/numbers.tex"] == read(joinpath(out, "numbers.tex"))
        @test Set(keys(after[4])) == Set(keys(before[4]))
        @test_throws "is accepted" accept_results(paper, run.branch; replace_submitted = true, scratch_root = scratch)

        # removal: the worktree goes, the branch stays
        remove_results_worktree(paper, run.branch)
        @test !isdir(run.worktree) && !isempty(git(repo, "branch", "--list", run.branch)) && worktrees(repo) == 1

        # a second run: a macro dropped (REMOVED), one new in the existing file (NEW); rejected, its branch deleted
        numbers(joinpath(out, "numbers.tex"); err = 1.2, speed = 8, extra = 0.7, len = "5\\,cm", frames = nothing, fresh = 3)
        rejected = complete(f; stamp = DateTime(2026, 11, 12))
        @test rejected.branch == "results/20261112_000000-" * acc.commit[1:7]
        verdict = Dict(r.name => r.verdict for r in rejected.numbers)
        @test verdict["fitFrames"] == "REMOVED" && verdict["fitFresh"] == "NEW" && verdict["fitErrMedian"] == "UNCHANGED"
        remove_results_worktree(paper, rejected.branch; delete_branch = true)
        @test !isdir(rejected.worktree) && isempty(git(repo, "branch", "--list", rejected.branch)) && worktrees(repo) == 1
        @test_throws ArgumentError remove_results_worktree(paper, base)
        @test snapshot(repo)[[1, 2, 3]] == after[[1, 2, 3]]
    end

    @testset "the evaluated commit, the manifest's gaps" begin
        f = fixture()
        git!(f.repo, "commit", "-q", "--allow-empty", "-m", "later")
        write(joinpath(f.repo, "a.txt"), "edited\n")                               # a dirty checkout
        dirty = complete(f; stamp = DateTime(2026, 11, 13))
        m = TOML.parsefile(joinpath(dirty.worktree, "proj", "RUN.toml"))
        @test m["dirty"] == true && m["commit"] == git(f.repo, "rev-parse", "HEAD")
        # a given commit is resolved (a name, a short hash, DrWatson's "-dirty" record)
        named = complete(f; commit = "HEAD~1", stamp = DateTime(2026, 11, 14))
        @test named.branch == "results/20261114_000000-" * f.head[1:7]
        m = TOML.parsefile(joinpath(named.worktree, "proj", "RUN.toml"))
        @test m["commit"] == f.head && m["dirty"] == false
        recorded = complete(f; commit = f.head[1:9] * "-dirty", stamp = DateTime(2026, 11, 15))
        m = TOML.parsefile(joinpath(recorded.worktree, "proj", "RUN.toml"))
        @test m["commit"] == f.head && m["dirty"] == true
        @test_throws ErrorException complete(f; commit = "nosuchcommit", stamp = DateTime(2026, 11, 16))
        # no tiers, no wall times: a warning, and the manifest names what it lacks
        bare = @test_logs (:warn, r"tiers.*wall_times") match_mode = :any results_branch(f.paper, f.assets;
                   scratch_root = f.scratch, gpus = ["B200"], stamp = DateTime(2026, 11, 17))
        @test TOML.parsefile(joinpath(bare.worktree, "proj", "RUN.toml"))["missing"] == ["tiers", "wall_times"]
        # the GPUs and the Slurm job, detected when not given
        seen = Reproduce._detected_gpus()
        detected = withenv("SLURM_JOB_ID" => "4242") do
            results_branch(f.paper, f.assets; scratch_root = f.scratch, stamp = DateTime(2026, 11, 17, 1),
                           tiers = ["photos" => repeat("ab", 20)], wall_times = ["fit" => 1.0])
        end
        m = TOML.parsefile(joinpath(detected.worktree, "proj", "RUN.toml"))
        @test m["slurm_jobs"] == ["4242"] && m["gpus"] == seen
        @test m["missing"] == (isempty(seen) ? ["gpus"] : String[])
        if !isempty(seen)                                     # a machine with GPUs: CUDA_VISIBLE_DEVICES restricts them
            @test withenv(Reproduce._detected_gpus, "CUDA_VISIBLE_DEVICES" => "0") == seen[1:1]
        end
        @test worktrees(f.repo) == 6
    end

    @testset "a failure mid-way leaves no branch and no worktree" begin
        f = fixture()
        before = snapshot(f.repo)
        src = first(f.assets[3])
        chmod(src, 0o000)                                     # unreadable: the copy into the worktree fails
        try
            @test_throws Base.IOError complete(f; stamp = DateTime(2026, 11, 18))
        finally
            chmod(src, 0o644)
        end
        @test worktrees(f.repo) == 1 && isempty(git(f.repo, "branch", "--list", "results/*"))
        @test !ispath(f.scratch) || isempty(readdir(f.scratch))
        @test snapshot(f.repo) == before
        rerun = complete(f; stamp = DateTime(2026, 11, 18))                     # the same name, now made
        @test rerun.branch == "results/20261118_000000-" * f.head[1:7] && worktrees(f.repo) == 2
    end

    @testset "acceptance and the user's checkout" begin
        # the base checked out with an accepted asset edited: the fast-forward is refused,
        # the base stays, no tag, no merge worktree
        f = fixture()
        run = complete(f; stamp = DateTime(2026, 11, 19))
        teaser = joinpath(f.paper, "paper", "fig", "teaser.png")
        write(teaser, UInt8[7, 7, 7])
        before = snapshot(f.repo)
        @test_throws "would be overwritten" accept_results(f.paper, run.branch; replace_submitted = true, scratch_root = f.scratch)
        @test snapshot(f.repo) == before && git(f.repo, "rev-parse", f.base) == f.head
        @test isempty(git(f.repo, "tag", "--list", "accepted/*"))
        @test worktrees(f.repo) == 2 && readdir(f.scratch) == [basename(run.worktree)]
        # the base checked out with an unrelated file edited: the base moves, the edit survives
        write(teaser, UInt8[0x89, 0x50, 0x4e, 0x47, 1, 2, 3])
        write(joinpath(f.repo, "a.txt"), "my notes\n")
        acc = accept_results(f.paper, run.branch; replace_submitted = true, scratch_root = f.scratch)
        @test git(f.repo, "rev-parse", "HEAD") == acc.commit == git(f.repo, "rev-parse", f.base)
        @test read(joinpath(f.repo, "a.txt"), String) == "my notes\n"
        @test read(teaser) == read(joinpath(f.out, "teaser.png"))

        # the checkout on another branch: untouched, while the base moves
        g = fixture()
        run = complete(g; stamp = DateTime(2026, 11, 20))
        git!(g.repo, "checkout", "-q", "-b", "drafting")
        before = snapshot(g.repo)
        acc = accept_results(g.paper, run.branch; replace_submitted = true, scratch_root = g.scratch)
        @test snapshot(g.repo) == before
        @test git(g.repo, "rev-parse", g.base) == acc.commit && git(g.repo, "rev-parse", acc.tag * "^{commit}") == acc.commit
        @test worktrees(g.repo) == 2 && readdir(g.scratch) == [basename(run.worktree)]

        # the base checked out in another worktree of the user's (the checkout itself on another branch):
        # that worktree fast-forwards with the accepted assets, the checkout is untouched
        w = fixture()
        run = complete(w; stamp = DateTime(2026, 11, 20, 1))
        git!(w.repo, "checkout", "-q", "-b", "drafting")
        other = joinpath(mktempdir(), "base-checkout")
        git!(w.repo, "worktree", "add", "-q", other, w.base)
        before = snapshot(w.repo)
        acc = accept_results(w.paper, run.branch; replace_submitted = true, scratch_root = w.scratch)
        @test snapshot(w.repo) == before
        after_other = snapshot(other)
        @test after_other[1] == acc.commit == git(w.repo, "rev-parse", w.base) && after_other[2] == w.base
        @test isempty(after_other[3])                                              # moved and clean, not stale
        @test after_other[4]["proj/paper/fig/teaser.png"] == read(joinpath(w.out, "teaser.png"))
        @test worktrees(w.repo) == 3 && readdir(w.scratch) == [basename(run.worktree)]
        git!(w.repo, "worktree", "remove", other)

        # a base that moved since the run: the acceptance warns about the assets it changed
        h = fixture()
        run = complete(h; stamp = DateTime(2026, 11, 21))
        write(joinpath(h.paper, "paper", "fig", "teaser.png"), UInt8[5, 5])
        git!(h.repo, "commit", "-q", "-am", "a later figure")
        @test_logs (:warn, r"moved since the run.*paper/fig/teaser.png") match_mode = :any accept_results(h.paper, run.branch;
                   replace_submitted = true, scratch_root = h.scratch)
    end
end

@testitem "results_branches: one git process for the assets, the identity read once" setup = [GitFixture] begin
    using Test
    using VisualGeometryToolkitEval.Reproduce: Reproduce
    repo = gitrepo()
    write(joinpath(repo, "b.bin"), UInt8[0x00, 0x0a, 0xff, 0x0a])          # binary, newlines inside and at the end
    write(joinpath(repo, "e.txt"), "")
    git!(repo, "add", "-A"); git!(repo, "commit", "-q", "-m", "two")
    @test Reproduce._git_blobs(repo, ["HEAD:b.bin", "HEAD:e.txt", "HEAD:a.txt"]) == [UInt8[0x00, 0x0a, 0xff, 0x0a], UInt8[], Vector{UInt8}("a\n")]
    @test isempty(Reproduce._git_blobs(repo, String[]))
    @test_throws "HEAD:nope.txt does not exist" Reproduce._git_blobs(repo, ["HEAD:a.txt", "HEAD:nope.txt"])
    # no configured identity: the package's own; a repository's own name and email: none added
    withenv("GIT_CONFIG_GLOBAL" => "/dev/null", "GIT_CONFIG_NOSYSTEM" => "1", "GIT_CONFIG_SYSTEM" => "/dev/null") do
        @test Reproduce._commit_identity(repo) == ("-c", "user.name=VisualGeometryToolkitEval", "-c", "user.email=vgtk@localhost")
        git!(repo, "config", "user.name", "R")
        @test Reproduce._commit_identity(repo) == Reproduce._commit_identity(repo) != ()   # a name alone is not an identity
        git!(repo, "config", "user.email", "r@r")
        @test Reproduce._commit_identity(repo) == ()
    end
end
