# The shared data/results binding, provenance record, keyed cache and data
# policy (src/experiments/data.jl, provenance.jl, data_policy.jl).

@testitem "experiments_data" setup = [GitFixture] begin
    import VisualGeometryToolkitEval as VgtkEval
    using Test, TOML
    using DrWatson: gitdescribe, produce_or_load
    using Pkg.Artifacts: create_artifact, bind_artifact!
    using VisualGeometryToolkitEval
    const E = VgtkEval.Experiments

    @testset "root ladder" begin
        dir = mktempdir()
        withenv("VGTK_DATA_ROOT" => nothing, "VGTK_RESULTS_ROOT" => nothing) do
            E.set_project_root!(dir)
            @test E.project_root() == dir
            @test_throws ErrorException E.data_root()
            @test_throws ErrorException E.results_root()
            @test !E.is_primary_copy()
            # the link path itself, unresolved (accv2026: its paths are spelled under it)
            tgt = mktempdir()
            mkpath(joinpath(dir, "data")); symlink(tgt, joinpath(dir, "data", "exp_raw"))
            @test E.data_root() == joinpath(dir, "data", "exp_raw")
            @test E.is_primary_copy()
            # resolved (bmvc2026)
            E.set_project_root!(dir; resolve_links = true)
            @test E.data_root() == realpath(tgt)
            # the fallback rung after the link
            rm(joinpath(dir, "data", "exp_raw"))
            E.set_project_root!(dir; data_fallback = () -> tgt)
            @test E.data_root() == tgt
            E.set_project_root!(dir; data_fallback = () -> error("no preference"))
            @test_throws ErrorException E.data_root()
            # results: created only when the project asks
            E.set_project_root!(dir; create_results = true)
            @test E.results_root() == joinpath(dir, "data", "exp_pro") && isdir(E.results_root())
            withenv("VGTK_DATA_ROOT" => "/x/data", "VGTK_RESULTS_ROOT" => "/x/res") do
                E.set_project_root!(dir)
                @test E.data_root() == "/x/data" && E.results_root() == "/x/res"
            end
            E.set_project_root!("")
        end
    end

    @testset "artifact view" begin
        dir = mktempdir()
        toml = joinpath(dir, "Artifacts.toml")
        h1 = create_artifact(d -> (mkpath(joinpath(d, "run1")); write(joinpath(d, "run1", "a.json"), "{}")))
        h2 = create_artifact(d -> (mkpath(joinpath(d, "run1")); write(joinpath(d, "run1", "b.json"), "[]")))
        bind_artifact!(toml, "tier_a", h1; lazy = true,
                       download_info = [("https://zenodo.org/records/1/files/tier_a.tar.gz", "0"^64)])
        bind_artifact!(toml, "tier_b", h2; lazy = true,
                       download_info = [("https://zenodo.org/records/ZENODO_RECORD_X/files/tier_b.tar.gz", "0"^64)])
        withenv("VGTK_DATA_ROOT" => nothing) do
            E.set_project_root!(dir; artifacts = E.ArtifactView(toml; mount = n -> "sub", default = ["tier_a"]))
            root = E.data_root()
            @test root == joinpath(dir, "data", "exp_raw")              # the view IS the project's datadir("exp_raw")
            @test E.is_artifact_view(root) && !E.is_primary_copy()
            @test isfile(joinpath(root, "sub", "run1", "a.json")) && islink(joinpath(root, "sub", "run1", "a.json"))
            @test !ispath(joinpath(root, "sub", "run1", "b.json"))
            # a further tier links into the same directories (per-file links)
            @test E.data_root(; tiers = ["tier_b"]) == root             # installed already: no download
            @test isfile(joinpath(root, "sub", "run1", "b.json"))
            @test E.link_tree!(E.install_artifact("tier_a", toml), joinpath(root, "sub")) == 0   # idempotent
            @test_throws ErrorException E.assert_primary_copy("a producer")
            # an uninstalled artifact behind a placeholder URL is refused with the way out
            bind_artifact!(toml, "tier_c", Base.SHA1("1"^40); lazy = true, force = true,
                           download_info = [("https://zenodo.org/records/ZENODO_RECORD_X/files/c.tar.gz", "0"^64)])
            err = try E.install_artifact("tier_c", toml); nothing catch e; e end
            @test err isa ErrorException && occursin("not published", err.msg)
            @test_throws ErrorException E.install_artifact("tier_zz", toml)
            E.set_project_root!("")
        end
    end

    @testset "input write guard" begin
        data = mktempdir(); res = joinpath(data, "results", "proj"); mkpath(res)
        withenv("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => res) do
            @test_throws ErrorException E.assert_not_input(joinpath(data, "pools", "new"))
            @test E.assert_not_input(joinpath(data, "pools", "new"); producer = true) === nothing
            @test E.assert_not_input(joinpath(res, "cell", "run")) === nothing        # results nested in the data root
            @test E.assert_not_input(mktempdir()) === nothing
            # every writer is guarded
            @test_throws ErrorException E.write_provenance(joinpath(data, "x"); driver = "t")
            @test_throws ErrorException E.cached_result(_ -> Dict("a" => 1), "p", Dict("k" => 1), joinpath(data, "c"))
            @test !ispath(joinpath(data, "x")) && !ispath(joinpath(data, "c"))
        end
        withenv("VGTK_DATA_ROOT" => nothing) do
            E.set_project_root!(mktempdir())
            @test E.assert_not_input(mktempdir()) === nothing                         # no data root bound: nothing to guard
            E.set_project_root!("")
        end
    end

    @testset "provenance record" begin
        repo = gitrepo()
        data = mktempdir(); res = mktempdir()
        withenv("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => res) do
            E.set_project_root!(repo)
            out = joinpath(res, "cell", "run-abc")
            p = E.write_provenance(out, Dict("run_key" => "abc", "n_frames" => 3, "note" => nothing);
                                   driver = "drv.jl", inputs = [joinpath(data, "pools", "P"), joinpath(res, "cv", "op.jld2"), "/elsewhere/f"])
            raw = TOML.parsefile(p)
            # DrWatson's tagged dict, flat: its own field names
            @test occursin(r"^[0-9a-f]{40}$", raw["gitcommit"]) && !haskey(raw, "gitpatch") && !haskey(raw, "uncommitted_files")
            @test raw["driver"] == "drv.jl" && raw["run_key"] == "abc" && raw["n_frames"] == 3 && raw["note"] == "nothing"
            @test all(haskey(raw, k) for k in ("timestamp", "hostname", "julia"))
            # inputs relative to their roots, the roots beside them
            @test raw["inputs"]["data"] == ["pools/P"] && raw["inputs"]["data_root"] == data
            @test raw["inputs"]["results"] == ["cv/op.jld2"] && raw["inputs"]["absolute"] == ["/elsewhere/f"]
            @test E.provenance_inputs(E.read_provenance(out)) ==
                  [joinpath(data, "pools", "P"), joinpath(res, "cv", "op.jld2"), "/elsewhere/f"]
            @test_throws ArgumentError E.write_provenance(out, Dict("gitcommit" => "x"); driver = "d")
            # a dirty tree: gitcommit suffixed, the patch inline under DrWatson's name
            write(joinpath(repo, "a.txt"), "changed\n")
            raw = TOML.parsefile(E.write_provenance(out; driver = "d"))
            @test endswith(raw["gitcommit"], "-dirty") && occursin("changed", raw["gitpatch"])
            @test raw["uncommitted_files"] == ["a.txt"]
            git!(repo, "checkout", "-q", "--", "a.txt")
            # …and in a git WORKTREE too (DrWatson's gitpatch drops it there)
            wt = joinpath(mktempdir(), "wt")
            git!(repo, "worktree", "add", "-q", wt)
            write(joinpath(wt, "a.txt"), "in a worktree\n")
            E.set_project_root!(wt)
            raw = TOML.parsefile(E.write_provenance(joinpath(res, "wt-run"); driver = "d"))
            @test endswith(raw["gitcommit"], "-dirty") && occursin("in a worktree", raw["gitpatch"])
            @test raw["uncommitted_files"] == ["a.txt"]
            E.set_project_root!(repo)
            # write_run_records: _config.toml + the record
            E.write_run_records(out, Dict{String,Any}("experiment" => Dict{String,Any}("dataset" => "d"));
                                driver = "d", config_source = "c.toml", key = "k1", inputs = [joinpath(data, "pools", "P")])
            rec = E.read_provenance(out)
            @test rec["run_key"] == "k1" && isfile(joinpath(out, "_config.toml"))
            @test_throws ErrorException E.write_run_records(joinpath(data, "pools", "Q"), Dict{String,Any}(); driver = "g",
                                                            config_source = "c.toml", key = "k")
            @test E.write_run_records(joinpath(data, "pools", "Q"), Dict{String,Any}(); driver = "g",
                                      config_source = "c.toml", key = "k", producer = true) isa String
            E.set_project_root!("")
        end
    end

    @testset "provenance back-compat" begin
        # VisualGeometryToolkitEval ≤ 0.1: flat, absolute inputs, the patch beside it
        d1 = mktempdir()
        write(joinpath(d1, "_provenance.toml"), """
            config_source = "/p/c.toml"
            driver = "run_sequential_cv.jl"
            gitcommit = "0123456789abcdef0123456789abcdef01234567-dirty"
            inputs = ["/data/pools/P"]
            run_key = "a7453bd32767eaf0"
            timestamp = "2026-10-01T00:00:00"
            """)
        write(joinpath(d1, "_dirty.patch"), "diff --git a/x b/x\n")
        r1 = E.read_provenance(d1)
        @test r1["run_key"] == "a7453bd32767eaf0" && r1["gitpatch"] == "diff --git a/x b/x\n"
        @test E.provenance_inputs(r1) == ["/data/pools/P"]
        # bmvc2026: [provenance] (git) + [config] (payload)
        d2 = mktempdir()
        write(joinpath(d2, "_provenance.toml"), """
            [config]
            frame_limit = 0
            n_frames = 12
            run_key = "1234"

            [provenance]
            created = "2026-08-17T10:00:00"
            driver = "go_bmvc2026.jl"
            gitcommit = "v0.9.0-3-g0123456"
            gitpatch = "diff"
            """)
        r2 = E.read_provenance(d2)
        @test r2["n_frames"] == 12 && r2["driver"] == "go_bmvc2026.jl" && r2["timestamp"] == "2026-08-17T10:00:00"
        @test r2["gitcommit"] == "v0.9.0-3-g0123456" && !haskey(r2, "created") && !haskey(r2, "config")
        @test_throws ErrorException E.read_provenance(mktempdir())
    end

    @testset "run_complete and commit drift" begin
        repo = gitrepo()
        git!(repo, "tag", "-a", "v0.1.0", "-m", "release")          # an ANNOTATED tag: describe names it
        write(joinpath(repo, "a.txt"), "b\n"); git!(repo, "commit", "-qam", "two")
        E.set_project_root!(repo)
        out = mktempdir()
        @test !E.run_complete(out; sidecar = "r.jld2")
        write(joinpath(out, "r.jld2"), "x")
        @test !E.run_complete(out; sidecar = "r.jld2")                 # no record: not complete
        E.write_provenance(out; driver = "d")
        rec = E.read_provenance(out)["gitcommit"]
        @test startswith(rec, "v0.1.0-1-g")                           # the annotated-tag form
        # same commit: no warning (the retired split('-')[1] parse compared "v0.1.0" with a sha and always warned)
        @test (@test_logs E.run_complete(out; sidecar = "r.jld2"))
        @test E.commit_drift(Dict("gitcommit" => rec * "-dirty")) === nothing
        @test E.commit_drift(Dict("gitcommit" => gitdescribe(repo; warn = false))) === nothing
        @test E.commit_drift(Dict{String,Any}()) === nothing
        # an exact tag names its commit
        git!(repo, "checkout", "-q", "v0.1.0")
        @test E.commit_drift(Dict("gitcommit" => "v0.1.0")) === nothing
        git!(repo, "checkout", "-q", "-")
        # a new commit: complete, but warned
        write(joinpath(repo, "a.txt"), "c\n"); git!(repo, "commit", "-qam", "three")
        @test E.commit_drift(Dict("gitcommit" => rec)) isa Tuple
        @test (@test_logs (:warn, r"DIFFERENT commit") E.run_complete(out; sidecar = "r.jld2"))
        E.set_project_root!("")
    end

    @testset "keyed cache" begin
        repo = gitrepo()
        E.set_project_root!(repo)
        cfg = Dict{String,Any}("dataset" => "260527", "system" => "bb", "level" => "none", "pose_run" => "abc")
        @test E.legacy_hash_key(cfg) == string(hash(cfg))
        dir = mktempdir()
        # the file DrWatson's produce_or_load(…; filename = hash, prefix) wrote is found, not recomputed
        _, f0 = produce_or_load(c -> Dict("v" => 1), cfg, dir; filename = hash, prefix = "pose", tag = false, verbose = false)
        data, f1 = E.cached_result(c -> error("must not recompute"), "pose", cfg, dir)
        @test f1 == f0 && data["v"] == 1
        # a new config computes, tags and saves; force keeps the previous version (DrWatson safesave's _#1)
        c2 = merge(cfg, Dict("level" => "o10"))
        d2, f2 = E.cached_result(c -> Dict("v" => 2), "pose", c2, dir)
        @test d2["v"] == 2 && isfile(f2) && E.cached_result(c -> error(), "pose", c2, dir)[1]["gitcommit"] isa String
        d3, _ = E.cached_result(c -> Dict("v" => 3), "pose", c2, dir; force = true)
        @test d3["v"] == 3 && isfile(replace(f2, ".jld2" => "_#1.jld2"))
        E.set_project_root!("")
    end

    @testset "data policy" begin
        root = mktempdir(); mkpath(joinpath(root, "scripts", "slurm")); mkpath(joinpath(root, "src"))
        write(joinpath(root, "src", "ok.jl"), """
            # a comment may name /mnt/data/x
            const OUT = joinpath(results_root(), "cell", "budget-" * run_key(cfg))
            """)
        write(joinpath(root, "src", "bad.jl"), """
            const POOL = "/mnt/data/accv2026/data/P"
            mkpath(joinpath(data_root(), "derived"))
            const T = joinpath(EXP_DATA_ROOT, "results", "lo_timing")
            const STAMP = Dates.format(Dates.now(), "yyyy")
            const TAG = "miscal-sel-v5"
            const SMOKE = budget_cell_dir(cfg) * "-smoke"
            f = JLD2.load(oracle_floors_path(), "floors")
            const SIMS = datadir("sims", "budget-\$(key)-smoke")
            run_cli("--smoke")
            const POOLS = artifact"pt_e_pools"
            p = Artifacts.@artifact_str("heb_pools")
            sc = load_tuning_artifact(rec["scorer_artifact"])
            """)
        write(joinpath(root, "scripts", "slurm", "x.sbatch"), "#SBATCH --output=/home/u/logs/%j.out\n# /home/fine\n")
        v = E.data_policy_violations(root)
        @test sort([(x.rule, x.file, x.line) for x in v]) ==
              sort([("D1", "src/bad.jl", 1), ("D2", "src/bad.jl", 2), ("D2", "src/bad.jl", 3), ("D3", "src/bad.jl", 4),
                    ("D3", "src/bad.jl", 5), ("D3", "src/bad.jl", 6), ("D3", "src/bad.jl", 7),
                    ("D4", "src/bad.jl", 10), ("D4", "src/bad.jl", 11),
                    ("D1", "scripts/slurm/x.sbatch", 1)])
        # a log line's `timestamp` is not a wall-clock stamp in a path; real stamps still are
        write(joinpath(root, "src", "stamps.jl"), """
            timestamp = Dates.format(now(), "yyyy-mm-dd")
            run_stamp = Dates.format(now(), "yyyy")
            f(; stamp = now()) = 1
            ts = Dates.format(Dates.now(), "HH") * "_TimeStamp"
            """)
        v3 = E.data_policy_violations(root)
        @test sort([x.line for x in v3 if x.file == "src/stamps.jl"]) == [2, 3, 4]
        allow = [E.DataPolicyAllow("D3", "src/bad.jl", "miscal-sel-v5"), E.DataPolicyAllow("D1", "src/ok.jl", "nothing here")]
        v2 = E.data_policy_violations(root; allow)
        @test length(v2) == length(v3) - 1 + 1
        @test any(x -> x.rule == "STALE" && x.file == "src/ok.jl", v2)
        @test !any(x -> x.rule == "D3" && x.line == 5, v2)
    end
end
