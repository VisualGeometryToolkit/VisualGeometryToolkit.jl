# The shared data/results binding, provenance record, keyed cache and data
# policy (src/experiments/data.jl, provenance.jl, data_policy.jl).

@testitem "experiments_data" setup = [GitFixture] begin
    import VisualGeometryToolkitEval as VgtkEval
    import VisualGeometryToolkitCore as VgtkCore
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
            # one tier, as data_root(; tier) and input_path ask for it: installed through the same resolver, a view tier
            # being one the project's catalog lists (the runner's gate, `catalog_view_mount`)
            mkpath(joinpath(dir, "configs"))
            write(joinpath(dir, "configs", "data_artifacts.toml"), "[artifacts.tier_e]\nmount_dir = \"sub\"\n[artifacts.tier_f]\nmount_dir = \"sub\"\n")
            h5 = create_artifact(d -> (mkpath(joinpath(d, "run1")); write(joinpath(d, "run1", "e.json"), "1")))
            bind_artifact!(toml, "tier_e", h5; lazy = true, download_info = [("https://zenodo.org/records/1/files/tier_e.tar.gz", "0"^64)])
            @test !ispath(joinpath(root, "sub", "run1", "e.json"))
            @test E.data_root(; tier = "tier_e") == joinpath(root, "sub") && isfile(joinpath(root, "sub", "run1", "e.json"))   # where the view mounts it
            h6 = create_artifact(d -> (mkpath(joinpath(d, "run1")); write(joinpath(d, "run1", "f.json"), "2")))
            bind_artifact!(toml, "tier_f", h6; lazy = true, download_info = [("https://zenodo.org/records/1/files/tier_f.tar.gz", "0"^64)])
            cf = Dict{String,Any}("inputs" => Dict{String,Any}("f" => Dict{String,Any}("tier" => "tier_f", "path" => "run1/f.json")))
            @test E.input_path(cf, "f") == joinpath(root, "sub", "run1", "f.json")
            @test E.data_root(; tier = "no_such_artifact") == root                 # not the view's: only looked up
            @test_throws ErrorException E.assert_primary_copy("a producer")
            # an uninstalled artifact behind a placeholder URL is refused with the way out
            bind_artifact!(toml, "tier_c", Base.SHA1("1"^40); lazy = true, force = true,
                           download_info = [("https://zenodo.org/records/ZENODO_RECORD_X/files/c.tar.gz", "0"^64)])
            err = try E.install_artifact("tier_c", toml); nothing catch e; e end
            @test err isa ErrorException && occursin("not published", err.msg)
            bind_artifact!(toml, "tier_d", Base.SHA1("2"^40); lazy = true, force = true,     # the same on Hugging Face
                           download_info = [("https://huggingface.co/datasets/lab/data/resolve/HF_REVISION_X/d.tar.gz", "0"^64)])
            err = try E.install_artifact("tier_d", toml); nothing catch e; e end
            @test err isa ErrorException && occursin("not published", err.msg)
            @test VgtkCore._published("https://huggingface.co/datasets/lab/data/resolve/$("ab"^20)/d.tar.gz") &&
                  VgtkCore._published("https://zenodo.org/records/12345/files/c.tar.gz?download=1")
            @test_throws ErrorException E.install_artifact("tier_zz", toml)
            @test E.install_artifact === VgtkCore.install_artifact   # one implementation, Core's (hoisted for BlobBoards)
            # the Hugging Face token: HF_TOKEN first, else the file `hf auth login` writes under HF_HOME
            hf_home = mktempdir()
            withenv("HF_TOKEN" => nothing, "HF_HOME" => hf_home) do
                @test VgtkCore.huggingface_token() === nothing
                write(joinpath(hf_home, "token"), "hf_file\n")
                @test VgtkCore.huggingface_token() == "hf_file"
                withenv(() -> (@test VgtkCore.huggingface_token() == "hf_env"), "HF_TOKEN" => " hf_env ")
            end
            E.set_project_root!("")
        end
    end

    @testset "one tier resolver, read and write: the view's mount only under the view; producers never install" begin
        dir = mktempdir(); toml = joinpath(dir, "Artifacts.toml")
        h = create_artifact(d -> (mkpath(joinpath(d, "run1")); write(joinpath(d, "run1", "e.json"), "1")))
        bind_artifact!(toml, "tier_e", h; lazy = true, download_info = [("https://zenodo.org/records/1/files/tier_e.tar.gz", "0"^64)])
        bind_artifact!(toml, "tier_n", Base.SHA1("3"^40); lazy = true,              # never installed: a placeholder URL
                       download_info = [("https://zenodo.org/records/ZENODO_RECORD_X/files/n.tar.gz", "0"^64)])
        mkpath(joinpath(dir, "configs"))
        write(joinpath(dir, "configs", "data_artifacts.toml"), "[artifacts.tier_e]\nmount_dir = \"sub\"\n[artifacts.tier_n]\nmount_dir = \"n\"\n")
        write(joinpath(dir, "configs", "e.toml"), "[inputs.e]\ntier = \"tier_e\"\npath = \"run1/e.json\"\n")
        sitedir = mktempdir(); envdir = mktempdir()
        write(joinpath(dir, "configs", "reproduce.toml"), """
            [project]
            name = "p"
            workdir = ".."
            [sites.m]
            tiers = { tier_e = "$sitedir" }
            [[stage]]
            name = "s"
            command = ["true"]
            config = "configs/e.toml"
            """)
        cfg = Dict{String,Any}("n" => 1); k = E.run_key(cfg)
        withenv("VGTK_DATA_ROOT" => nothing, "VGTK_SITE" => nothing, "VGTK_DATA_ROOT_TIER_E" => nothing, "VGTK_DATA_ROOT_TIER_N" => nothing) do
            E.set_project_root!(dir; artifacts = E.ArtifactView(toml; mount = n -> n == "tier_e" ? "sub" : "n"))
            root = E.data_root()                                                     # the view
            # S1: the runner's missing_inputs and input_path resolve a view tier to the same directory
            @test E.input_path(TOML.parsefile(joinpath(dir, "configs", "e.toml")), "e") == joinpath(root, "sub", "run1", "e.json")
            g = VgtkEval.Reproduce.load_stage_graph(joinpath(dir, "configs", "reproduce.toml"))
            @test isempty(VgtkEval.Reproduce.missing_inputs(g, ["s"]; data_root = root, env = Dict{String,String}()))
            @test first(VgtkEval.tier_data_root("tier_e", root; env = Dict{String,String}(),
                        view_mount = VgtkEval.catalog_view_mount(root, joinpath(dir, "configs", "data_artifacts.toml")))) == joinpath(root, "sub")
            # S3: a producer never installs and never writes into a view tier; with its root bound it writes there
            @test_throws r"tier tier_n is published read-only by the artifact view; bind VGTK_DATA_ROOT_TIER_N" E.keyed_data_dir(cfg, "tier_n", "k")
            @test_throws r"tier tier_e is published read-only" E.keyed_data_dir(cfg, "tier_e", "k")
            run = joinpath(mktempdir(), "run"); mkpath(run); E.write_provenance(run, Dict("run_key" => k); driver = "d")
            @test_throws r"tier tier_e is published read-only" E.promote_run(run, "tier_e")
            # a tier the view does not publish: the data root IS the view, read-only, so producing it is refused too
            @test_throws r"the data root is an artifact view \(read-only\); bind VGTK_DATA_ROOT or VGTK_DATA_ROOT_LOCAL_TIER" E.keyed_data_dir(cfg, "local_tier", "k")
            @test_throws r"the data root is an artifact view" E.promote_run(run, "local_tier")
            @test_throws r"the data root is an artifact view" withenv(() -> E.keyed_data_dir(cfg, "local_tier", "k"), "VGTK_DATA_ROOT" => root)
            @test_throws r"the data root is an artifact view" withenv(() -> E.keyed_data_dir(cfg, "local_tier", "k"),
                                                                      "VGTK_DATA_ROOT_LOCAL_TIER" => joinpath(root, "sub"))   # a root inside the view
            writable = mktempdir()
            @test withenv(() -> E.keyed_data_dir(cfg, "local_tier", "k"), "VGTK_DATA_ROOT_LOCAL_TIER" => writable) == joinpath(writable, "local_tier", "k-" * k)
            withenv("VGTK_DATA_ROOT_TIER_N" => envdir) do
                @test E.keyed_data_dir(cfg, "tier_n", "k") == joinpath(envdir, "tier_n", "k-" * k)
            end
            # S4: env beats site beats the view, for reading and producing
            withenv("VGTK_SITE" => "m") do
                @test E.data_root(; tier = "tier_e") == sitedir
                @test E.keyed_data_dir(cfg, "tier_e", "k") == joinpath(sitedir, "tier_e", "k-" * k)
                withenv("VGTK_DATA_ROOT_TIER_E" => envdir) do
                    @test E.data_root(; tier = "tier_e") == envdir && E.keyed_data_dir(cfg, "tier_e", "k") == joinpath(envdir, "tier_e", "k-" * k)
                end
            end
            # the graph's data_env names the tier variables, for the drivers as for the runner (its own reader, load_stage_graph)
            g_text = read(joinpath(dir, "configs", "reproduce.toml"), String)
            write(joinpath(dir, "configs", "reproduce.toml"), replace(g_text, "name = \"p\"" => "name = \"p\"\ndata_env = \"P_DATA\""; count = 1))
            withenv("VGTK_SITE" => "m", "P_DATA_TIER_E" => envdir) do
                @test E.data_root(; tier = "tier_e") == envdir
                @test_throws r"refusing to write .*P_DATA_TIER_E" E.assert_not_input(joinpath(envdir, "x"))
            end
            write(joinpath(dir, "configs", "reproduce.toml"), g_text)
            @test E.data_root(; tier = "tier_e") == joinpath(root, "sub")
            # S2: the mount applies only under the view: a VGTK_DATA_ROOT, an author's own data/exp_raw, a fallback
            flat = mktempdir(); mkpath(joinpath(flat, "run1")); write(joinpath(flat, "run1", "e.json"), "1")
            withenv("VGTK_DATA_ROOT" => flat) do
                @test E.data_root(; tier = "tier_e") == flat && E.input_path(TOML.parsefile(joinpath(dir, "configs", "e.toml")), "e") == joinpath(flat, "run1", "e.json")
                @test E.keyed_data_dir(cfg, "tier_e", "k") == joinpath(flat, "tier_e", "k-" * k)          # unmoved
                @test isempty(VgtkEval.Reproduce.missing_inputs(g, ["s"]; data_root = flat, env = Dict{String,String}()))
            end
            own = mktempdir(); proj2 = mktempdir(); mkpath(joinpath(proj2, "data")); symlink(own, joinpath(proj2, "data", "exp_raw"))
            E.set_project_root!(proj2; artifacts = E.ArtifactView(toml; mount = _ -> "sub"))
            @test E.data_root(; tier = "tier_e") == joinpath(proj2, "data", "exp_raw")
            fb = mktempdir()
            E.set_project_root!(mktempdir(); data_fallback = () -> fb, artifacts = E.ArtifactView(toml; mount = _ -> "sub"))
            @test E.data_root(; tier = "tier_e") == fb && E.keyed_data_dir(cfg, "tier_e", "k") == joinpath(fb, "tier_e", "k-" * k)
            E.set_project_root!("")
        end
    end

    @testset "tier_path: a path in a tier, under its tier's root" begin
        data = mktempdir(); derived = mktempdir()
        withenv("VGTK_DATA_ROOT" => data, "VGTK_DATA_ROOT_DERIVED" => derived, "VGTK_SITE" => nothing) do
            E.set_project_root!(mktempdir())
            @test E.tier_path("derived", "derived", "pool-k", "pool.jld2") == joinpath(derived, "derived", "pool-k", "pool.jld2")
            @test E.tier_path("boards", "a4.toml") == joinpath(data, "a4.toml")               # no tier root: the data root
            @test E.tier_path("boards") == data
            E.set_project_root!("")
        end
    end

    @testset "a scratch run never completes, caches or promotes a run of record, nor the reverse" begin
        out = mktempdir(); write(joinpath(out, "r.jld2"), "x")
        withenv("VGTK_SCRATCH" => nothing) do
            E.set_project_root!(mktempdir())
            E.write_provenance(out; driver = "d")
            @test E.run_complete(out; sidecar = "r.jld2")
            @test !withenv(() -> (@test_logs (:info, r"its code changed") E.run_complete(out; sidecar = "r.jld2")), "VGTK_SCRATCH" => "1")
            s = mktempdir(); write(joinpath(s, "r.jld2"), "x")
            withenv(() -> E.write_provenance(s; driver = "d"), "VGTK_SCRATCH" => "1")
            @test withenv(() -> E.run_complete(s; sidecar = "r.jld2"), "VGTK_SCRATCH" => "1")
            @test !(@test_logs (:info, r"its code changed") E.run_complete(s; sidecar = "r.jld2"))
            cfg = Dict{String,Any}("n" => 2)
            @test withenv(() -> E.run_key(cfg), "VGTK_SCRATCH" => "1") != E.run_key(cfg)
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
            # the sensor model's provenance (spec §5): Core's component_sources, per distinct model with its frame count
            measured = VgtkCore.sensor_fixture(; black = (512.0, 512.0, 512.0, 512.0), gain = 2.0, read_noise = (1.5, 1.5, 1.5, 1.5),
                                               source = VgtkCore.SensorSource.measured).model
            raw = TOML.parsefile(E.write_provenance(out; driver = "d", sensor_model = [measured, measured, VgtkCore.SensorModel()]))
            sm = raw["sensor_model"]
            @test length(sm) == 2
            m1 = only(filter(r -> r["frames"] == 2, sm)); m0 = only(filter(r -> r["frames"] == 1, sm))
            @test (m1["black"], m1["noise"], m1["defects"], m1["reason"], m1["specified_override"]) == ("measured", "measured", "absent", "Matched", false)
            @test (m0["black"], m0["noise"], m0["defects"], m0["reason"]) == ("file", "absent", "absent", "NoCharacterisation")
            @test only(TOML.parsefile(E.write_provenance(out; driver = "d", sensor_model = measured))["sensor_model"])["frames"] == 1
            @test_throws ArgumentError E.write_provenance(out, Dict("sensor_model" => "x"); driver = "d")   # the writer's own
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
        # a new commit that changes no code tree (this repository has no src): the same code, complete, no warning
        write(joinpath(repo, "a.txt"), "c\n"); git!(repo, "commit", "-qam", "three")
        @test E.commit_drift(Dict("gitcommit" => rec)) isa Tuple
        @test (@test_logs E.run_complete(out; sidecar = "r.jld2"))
        E.set_project_root!("")
    end

    @testset "run_complete and the keyed cache reuse a result only while its code trees are the current ones" begin
        repo = gitrepo()
        mkpath(joinpath(repo, "src")); mkpath(joinpath(repo, "docs"))
        write(joinpath(repo, "src", "M.jl"), "module M end\n"); write(joinpath(repo, "docs", "x.md"), "x\n")
        git!(repo, "add", "-A"); git!(repo, "commit", "-qm", "code")
        E.set_project_root!(repo)
        out = mktempdir(); write(joinpath(out, "r.jld2"), "x")
        E.write_provenance(out; driver = "d")
        trees = E.read_provenance(out)["code_trees"]
        @test trees == E.current_code_trees(repo) && collect(keys(trees)) == ["src"]
        @test (@test_logs E.run_complete(out; sidecar = "r.jld2"))
        # a docs-only commit: no tree changed
        write(joinpath(repo, "docs", "x.md"), "y\n"); git!(repo, "commit", "-qam", "docs")
        @test (@test_logs E.run_complete(out; sidecar = "r.jld2"))
        # src changes: not complete, the tree named
        write(joinpath(repo, "src", "M.jl"), "module M\nf() = 1\nend\n"); git!(repo, "commit", "-qam", "src")
        @test !(@test_logs (:info, r"rerunning .*its code changed") E.run_complete(out; sidecar = "r.jld2"))
        chg = VgtkEval.code_tree_changes(trees, E.current_code_trees(repo))
        @test length(chg) == 1 && occursin(r"^src: [0-9a-f]{10} → [0-9a-f]{10}$", chg[1])
        E.write_provenance(out; driver = "d")                         # the rerun records the new trees
        @test (@test_logs E.run_complete(out; sidecar = "r.jld2"))
        # a record from before the trees were kept: runs once more
        p = joinpath(out, "_provenance.toml"); d = TOML.parsefile(p); delete!(d, "code_trees"); open(io -> TOML.print(io, d), p, "w")
        @test !(@test_logs (:info, r"its code changed") E.run_complete(out; sidecar = "r.jld2"))
        # the keyed cache: computed once, reused, computed again when src changes (the old version kept)
        dir = mktempdir(); cfg = Dict{String,Any}("a" => 1)
        d1, f1 = E.cached_result(c -> Dict{String,Any}("v" => 1), "r", cfg, dir)
        @test d1["code_trees"] == E.current_code_trees(repo)
        @test E.cached_result(c -> error("must not recompute"), "r", cfg, dir)[1]["v"] == 1
        write(joinpath(repo, "src", "M.jl"), "module M\nf() = 2\nend\n"); git!(repo, "commit", "-qam", "src again")
        d2, _ = @test_logs (:info, r"recomputing .*its code changed") match_mode = :any E.cached_result(c -> Dict{String,Any}("v" => 2), "r", cfg, dir)
        @test d2["v"] == 2 && isfile(replace(f1, ".jld2" => "_#1.jld2"))
        E.set_project_root!("")
    end

    @testset "a run of record of uncommitted code is refused (the runner's refusal); a scratch run is warned and marked" begin
        repo = gitrepo()
        mkpath(joinpath(repo, "src")); write(joinpath(repo, "src", "M.jl"), "module M end\n")
        git!(repo, "add", "-A"); git!(repo, "commit", "-qm", "code")
        E.set_project_root!(repo)
        out = mktempdir(); write(joinpath(out, "r.jld2"), "x")
        write(joinpath(repo, "src", "M.jl"), "module M\nf() = 1\nend\n")              # an uncommitted edit
        @test_throws r"a run of record runs committed code, and src has uncommitted changes" E.write_provenance(out; driver = "d")
        @test !isfile(joinpath(out, "_provenance.toml"))
        dir = mktempdir()
        @test_throws r"a run of record runs committed code" E.cached_result(c -> error("must not compute"), "r", Dict("a" => 1), dir)
        @test isempty(readdir(dir))
        write(joinpath(repo, "src", "N.jl"), "# untracked\n"); git!(repo, "checkout", "-q", "--", "src/M.jl")
        @test_throws r"src has uncommitted changes" E.write_provenance(out; driver = "d")                 # an untracked file too
        # a scratch run records it, warned, its tree marked: never reused once the code is committed (or reverted)
        withenv("VGTK_SCRATCH" => "1") do
            @test E.scratch_run()
            @test_logs (:warn, r"scratch run records .* of uncommitted code") match_mode = :any E.write_provenance(out; driver = "d")
            # never reused, not even by a scratch run of the same marked tree (other edits carry the same mark)
            @test !(@test_logs (:info, r"its code changed") match_mode = :any E.run_complete(out; sidecar = "r.jld2"))
        end
        @test endswith(E.read_provenance(out)["code_trees"]["src"], "+uncommitted")
        rm(joinpath(repo, "src", "N.jl"))
        @test !(@test_logs (:info, r"its code changed") E.run_complete(out; sidecar = "r.jld2"))
        # code in no git checkout cannot be named: refused, and a scratch run records "uncommitted"
        loose = mktempdir(); mkpath(joinpath(loose, "src")); write(joinpath(loose, "src", "M.jl"), "module M end\n")
        E.set_project_root!(loose)
        @test_throws r"src has uncommitted changes \(or no git checkout\)" E.write_provenance(joinpath(loose, "run"); driver = "d")
        withenv("VGTK_SCRATCH" => "1") do
            @test_logs (:warn, r"scratch run") match_mode = :any E.write_provenance(joinpath(loose, "run"); driver = "d")
        end
        @test E.read_provenance(joinpath(loose, "run"))["code_trees"]["src"] == "uncommitted"
        # no src: the record names none of the project's own code, which is warned
        E.set_project_root!(gitrepo())
        @test_logs (:warn, r"records no tree of the project's own code") match_mode = :any E.write_provenance(mktempdir(); driver = "d")
        E.set_project_root!("")
    end

    @testset "run_key folds files, upstream keys and parts; keyed data dirs, promote_run, verify_inputs, require_input's sha256" begin
        data = mktempdir(); res = mktempdir()
        withenv("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => res) do
            E.set_project_root!(mktempdir())
            cfg = Dict{String,Any}("experiment" => Dict{String,Any}("dataset" => "d", "matcher" => "m", "problem" => "p"), "n" => 3)
            k0 = E.run_key(cfg)
            @test E.run_key(cfg; files = nothing, upstream = nothing, parts = nothing) == k0       # nothing folded: the config's key
            f = joinpath(data, "board.toml"); write(f, "a\n")
            kf = E.run_key(cfg; files = Dict("board" => f))
            @test kf != k0 && E.run_key(cfg; files = Dict("board" => f)) == kf
            write(f, "b, edited\n")
            @test E.run_key(cfg; files = Dict("board" => f)) != kf                                 # the file's content keys it
            ku = E.run_key(cfg; upstream = Dict("poses" => "0123456789abcdef"))
            @test ku != k0 && ku != E.run_key(cfg; upstream = Dict("poses" => "fedcba9876543210"))
            kp = E.run_key(cfg; parts = (; frame_limit = 10))
            @test kp != k0 && kp == E.run_key(cfg; parts = Dict("frame_limit" => 10)) && kp != E.run_key(cfg; parts = (; frame_limit = 0))
            @test_throws r"keyed_with" E.run_key(merge(cfg, Dict{String,Any}("keyed_with" => 1)))
            @test basename(E.keyed_cell_dir(cfg, "e", "pose"; parts = (; frame_limit = 10))) == "pose-" * kp
            # a project laid out by its own config, not by pool: [experiment] layout, or a base under the results root
            lay = Dict{String,Any}("experiment" => Dict{String,Any}("layout" => ["mocap", "cam0"]), "n" => 3)
            @test E.experiment_dir(lay) == joinpath(res, "mocap", "cam0")
            @test E.keyed_cell_dir(lay, "selfcal", "run") == joinpath(res, "mocap", "cam0", "selfcal", "run-" * E.run_key(lay))
            b = joinpath(res, "absolute_pose_experiments", "set1")
            @test E.keyed_cell_dir(cfg, "tune_radial_camera", "run"; base = b) == joinpath(b, "tune_radial_camera", "run-" * k0)
            @test E.keyed_cell_dir(cfg, "figures", "fig_sensitivity"; base = res) == joinpath(res, "figures", "fig_sensitivity-" * k0)   # a run over several datasets
            @test_throws r"is not under the results root" E.keyed_cell_dir(cfg, "x", "run"; base = mktempdir())
            @test_throws r"is not under the results root" E.keyed_cell_dir(cfg, "x", "run"; base = joinpath(res, "..", "elsewhere"))
            for bad in (String[], ["a", ".."], ["/abs"], ["a/b"], [""], "mocap")
                @test_throws r"layout" E.experiment_dir(Dict{String,Any}("experiment" => Dict{String,Any}("layout" => bad)))
            end
            p = joinpath(mktempdir(), "lay.toml")
            write(p, "[experiment]\nlayout = [\"mocap\", \"cam0\"]\n")
            @test E.load_run_config(p)["experiment"]["layout"] == ["mocap", "cam0"]               # no pool fields needed
            write(p, "[experiment]\ndataset = \"d\"\n")
            @test_throws r"lacks `matcher` \(or a `layout`\)" E.load_run_config(p)
            # a producer's keyed directory under the data root
            @test E.keyed_data_dir(cfg, "annotations", "train") == joinpath(data, "annotations", "train-" * k0)
            # producer and consumer agree under a tier root set elsewhere: one path of record, `<tier>/<kind>-<key>`
            derived = mktempdir()
            withenv("VGTK_DATA_ROOT_DERIVED" => derived) do
                out = E.keyed_data_dir(cfg, "derived", "pool")
                @test out == joinpath(derived, "derived", "pool-" * k0)
                mkpath(out); write(joinpath(out, "pool.jld2"), "p")
                c = Dict{String,Any}("inputs" => Dict{String,Any}("pool" => Dict{String,Any}("tier" => "derived", "path" => "derived/pool-" * k0)))
                @test E.input_path(c, "pool") == out
            end
            # a run of record made an input of record
            run = joinpath(res, "fig", "run-" * k0); mkpath(run); write(joinpath(run, "r.csv"), "1\n")
            @test_throws r"no _provenance.toml" E.promote_run(run, "runs_of_record")
            E.write_provenance(run, Dict("run_key" => k0); driver = "d")
            dst = E.promote_run(run, "runs_of_record")
            @test dst == joinpath(data, "runs_of_record", "run-" * k0) && read(joinpath(dst, "r.csv"), String) == "1\n"
            @test E.promote_run(run, "runs_of_record") == dst                                       # the same run: left as it is
            other = joinpath(res, "x", "run-" * k0); mkpath(other); E.write_provenance(other, Dict("run_key" => "other"); driver = "d")
            @test_throws r"holds another run" E.promote_run(other, "runs_of_record")
            # runs without a run_key: the same run again is left as it is; another run of that name is refused
            nokey = joinpath(res, "fig", "nokey"); mkpath(nokey); E.write_provenance(nokey, Dict("what" => "a"); driver = "d")
            dst2 = E.promote_run(nokey, "runs_of_record")
            @test E.promote_run(nokey, "runs_of_record") == dst2
            nokey2 = joinpath(res, "x", "nokey"); mkpath(nokey2); E.write_provenance(nokey2, Dict("what" => "b"); driver = "d")
            @test_throws r"holds another run \(another record; this run has no run_key\)" E.promote_run(nokey2, "runs_of_record")
            # the inputs of a config, verified as the runner verifies them
            h = first(split(read(`sha256sum $f`, String)))
            c2 = merge(cfg, Dict{String,Any}("inputs" => Dict{String,Any}(
                "board" => Dict{String,Any}("tier" => "boards", "path" => "board.toml", "sha256" => h),
                "runs" => Dict{String,Any}("tier" => "runs_of_record", "path" => "runs_of_record/run-" * k0))))
            @test E.verify_inputs(c2) == Dict("board" => f, "runs" => dst)
            c2["inputs"]["board"]["sha256"] = "0"^64
            @test_throws r"input board \(board.toml\): sha256 expected 0{64}, actual [0-9a-f]{64}" E.verify_inputs(c2)
            c2["inputs"]["runs"]["path"] = "runs_of_record/gone"
            @test_throws r"input runs \(runs_of_record\): runs_of_record/gone is not under" E.verify_inputs(c2)
            # require_input: with a sha256, the content checked
            @test E.require_input(f; what = "board", provenance = "p", sha256 = h) == f
            @test E.require_input(f; what = "board", provenance = "p", sha256 = uppercase(h)) == f
            @test_throws r"refused: input \[board\] .* sha256 expected 0{64}, actual" E.require_input(f; what = "board", provenance = "p", sha256 = "0"^64)
            @test E.require_input(f; what = "board", provenance = "p") == f                         # absent: unchecked
            # the runner's check (inspect_input): a sha256 that is no SHA-256 is a config error; absent, the provenance recipe
            @test_throws r"is no SHA-256" E.require_input(f; what = "board", provenance = "p", sha256 = "abc")
            @test_throws r"MISSING INPUT \[board\]: .*gone\n  provenance: p" E.require_input(joinpath(dirname(f), "gone"); what = "board", provenance = "p")
            E.set_project_root!("")
        end
    end

    @testset "data_root(; tier): the environment, then the stage graph's [sites.<site>.tiers], then the data root" begin
        data = mktempdir(); res = mktempdir(); proj = mktempdir(); mocap = mktempdir()
        mkpath(joinpath(proj, "configs")); mkpath(joinpath(proj, "rel", "mocap"))
        write(joinpath(proj, "configs", "reproduce.toml"), """
            [project]
            name = "p"
            workdir = ".."
            [sites.m]
            tiers = { mocap = "rel/mocap", abs = "$mocap" }
            [[stage]]
            name = "s"
            command = ["true"]
            """)
        withenv("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => res, "VGTK_SITE" => nothing, "VGTK_DATA_ROOT_MOCAP" => nothing) do
            E.set_project_root!(proj)
            @test E.data_root(; tier = "mocap") == data                                            # no site: the data root
            withenv("VGTK_SITE" => "m") do
                @test E.data_root(; tier = "mocap") == joinpath(proj, "rel", "mocap")              # the site's, relative to the workdir
                @test E.data_root(; tier = "abs") == mocap && E.data_root(; tier = "other") == data
                withenv("VGTK_DATA_ROOT_MOCAP" => mocap) do
                    @test E.data_root(; tier = "mocap") == mocap                                   # the environment first
                    # verify_inputs per tier; the tier root is input: no write under it
                    write(joinpath(mocap, "take.csv"), "1\n")
                    h = first(split(read(`sha256sum $(joinpath(mocap, "take.csv"))`, String)))
                    c = Dict{String,Any}("inputs" => Dict{String,Any}("take" => Dict{String,Any}("tier" => "mocap", "path" => "take.csv", "sha256" => h)))
                    @test E.verify_inputs(c) == Dict("take" => joinpath(mocap, "take.csv"))
                    @test E.input_path(c, "take") == joinpath(mocap, "take.csv")            # the one reader
                    @test_throws r"declares no \[inputs.gone\]" E.input_path(c, "gone")
                    c["inputs"]["take"]["sha256"] = "0"^64
                    @test_throws r"sha256 expected 0{64}" E.input_path(c, "take")
                    c["inputs"]["take"]["sha256"] = h
                    @test_throws r"is not under" E.verify_inputs(c; root = data)
                    @test_throws r"input tier \(VGTK_DATA_ROOT_MOCAP" E.assert_not_input(joinpath(mocap, "out"))
                end
            end
            E.set_project_root!("")
        end
    end

    @testset "a scratch run is kept apart: marked, keyed apart, and refused as an input, upstream or promoted run" begin
        data = mktempdir(); res = mktempdir()
        withenv("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => res, "VGTK_SCRATCH" => nothing) do
            E.set_project_root!(mktempdir())
            cfg = Dict{String,Any}("experiment" => Dict{String,Any}("dataset" => "d", "matcher" => "m", "problem" => "p"), "n" => 1)
            rec_dir = E.keyed_cell_dir(cfg, "e", "run")
            E.write_provenance(rec_dir, Dict("run_key" => E.run_key(cfg)); driver = "d")
            @test E.read_provenance(rec_dir)["scratch"] === false && !VgtkEval.scratch_record(rec_dir)
            scr_dir, scr_key, scr_file = withenv("VGTK_SCRATCH" => "1") do
                d = E.keyed_cell_dir(cfg, "e", "run")
                E.write_provenance(d, Dict("run_key" => E.run_key(cfg)); driver = "d")
                _, f = E.cached_result(c -> Dict{String,Any}("v" => 1), "r", cfg, joinpath(res, "cache"))
                d, E.run_key(cfg), f
            end
            # marked, and never the directory, key or cache file of the run of record
            @test E.read_provenance(scr_dir)["scratch"] === true && VgtkEval.scratch_record(scr_dir)
            @test VgtkEval.scratch_record(joinpath(scr_dir, "anything.csv"))                  # a file in a scratch run's directory
            @test scr_dir != rec_dir && scr_key != E.run_key(cfg)
            @test occursin("r-scratch_", basename(scr_file))
            _, f2 = E.cached_result(c -> Dict{String,Any}("v" => 2), "r", cfg, joinpath(res, "cache"))
            @test f2 != scr_file && E.cached_result(c -> error("cached"), "r", cfg, joinpath(res, "cache"))[1]["scratch"] === false
            # refused to a run of record: as an upstream run, an input of its record, a promoted run, a config input
            @test_throws r"scratch run's output .*never an upstream run of record" E.run_key(cfg; upstream = Dict("poses" => scr_dir))
            @test E.run_key(cfg; upstream = Dict("poses" => rec_dir)) == E.run_key(cfg; upstream = Dict("poses" => E.run_key(cfg)))
            @test_throws r"a run of record reads no scratch output" E.write_provenance(joinpath(res, "next"); driver = "d", inputs = [scr_dir])
            @test_throws r"promote_run: .* is a scratch run's output" E.promote_run(scr_dir, "runs_of_record")
            mkpath(joinpath(data, "runs")); cp(scr_dir, joinpath(data, "runs", "s"))
            c = Dict{String,Any}("inputs" => Dict{String,Any}("s" => Dict{String,Any}("tier" => "runs", "path" => "runs/s")))
            @test_throws r"input s \(runs/s\): a scratch run's output" E.verify_inputs(c)
            @test_throws r"a scratch run's output" E.input_path(c, "s")
            # an input nested deeper in a scratch run's directory is its output too
            mkpath(joinpath(scr_dir, "sub")); write(joinpath(scr_dir, "sub", "x.csv"), "1\n")
            @test VgtkEval.scratch_record(joinpath(scr_dir, "sub", "x.csv")) && !VgtkEval.scratch_record(joinpath(rec_dir, "sub", "y.csv"))
            mkpath(joinpath(data, "runs", "s", "sub")); write(joinpath(data, "runs", "s", "sub", "x.csv"), "1\n")
            cn = Dict{String,Any}("inputs" => Dict{String,Any}("x" => Dict{String,Any}("tier" => "runs", "path" => "runs/s/sub/x.csv")))
            @test_throws r"input x \(runs/s/sub/x.csv\): a scratch run's output" E.verify_inputs(cn)
            # a directory holding one kind of record never takes the other
            @test_throws r"holds the record of a run of record, and this is a scratch run" withenv(() -> E.write_provenance(rec_dir; driver = "d"), "VGTK_SCRATCH" => "1")
            @test_throws r"holds the record of a scratch run, and this is a run of record" E.write_provenance(scr_dir; driver = "d")
            @test E.write_provenance(rec_dir; driver = "d") isa String                                   # the same kind: rewritten
            withenv("VGTK_SCRATCH" => "1") do                                                      # a scratch run may read one
                @test E.input_path(c, "s") == joinpath(data, "runs", "s")
                @test E.run_key(cfg; upstream = Dict("poses" => scr_dir)) isa String
            end
            E.set_project_root!("")
        end
    end

    @testset "cached_result through another codec; write_run_records with a payload" begin
        E.set_project_root!(mktempdir())
        dir = mktempdir(); cfg = Dict{String,Any}("a" => 1)
        codec = (; ext = ".toml", load = TOML.parsefile, save = (f, d) -> open(io -> TOML.print(io, d), f, "w"))
        d1, f1 = E.cached_result(c -> Dict{String,Any}("pose" => [1.0, 2.0]), "he", cfg, dir; codec...)
        @test endswith(f1, ".toml") && TOML.parsefile(f1)["pose"] == [1.0, 2.0] && haskey(TOML.parsefile(f1), "code_trees")
        @test E.cached_result(c -> error("cached"), "he", cfg, dir; codec...)[1]["pose"] == [1.0, 2.0]
        d3, _ = E.cached_result(c -> Dict{String,Any}("pose" => [3.0]), "he", cfg, dir; force = true, codec...)
        @test d3["pose"] == [3.0] && isfile(replace(f1, ".toml" => "_#1.toml"))                   # the previous version kept
        E.cached_result(c -> Dict{String,Any}("pose" => [4.0]), "he", cfg, dir; force = true, codec...)
        # DrWatson safesave's numbering: the newest previous version is _#1, the older renumbered up
        @test TOML.parsefile(replace(f1, ".toml" => "_#1.toml"))["pose"] == [3.0] && TOML.parsefile(replace(f1, ".toml" => "_#2.toml"))["pose"] == [1.0, 2.0]
        out = joinpath(mktempdir(), "run")
        E.write_run_records(out, Dict{String,Any}("experiment" => Dict{String,Any}("dataset" => "d")); driver = "d",
                            config_source = "c.toml", key = "k", fields = Dict("folds" => 5, "replayed" => "detection_cv"))
        r = E.read_provenance(out)
        @test r["folds"] == 5 && r["replayed"] == "detection_cv" && r["run_key"] == "k"
        @test_throws r"are the records' own" E.write_run_records(out, Dict{String,Any}(); driver = "d", config_source = "c",
                                                                  key = "k", fields = Dict("run_key" => "x"))
        E.set_project_root!("")
    end

    @testset "keyed cache" begin
        repo = gitrepo()
        E.set_project_root!(repo)
        cfg = Dict{String,Any}("dataset" => "260527", "system" => "bb", "level" => "none", "pose_run" => "abc")
        @test E.legacy_hash_key(cfg) == string(hash(cfg))
        dir = mktempdir()
        # the file DrWatson's produce_or_load(…; filename = hash, prefix) wrote is found; it keeps no code trees (it
        # predates them), so it is computed once more (the old version kept), then reused
        _, f0 = produce_or_load(c -> Dict("v" => 1), cfg, dir; filename = hash, prefix = "pose", tag = false, verbose = false)
        data, f1 = @test_logs (:info, r"recomputing") match_mode = :any E.cached_result(c -> Dict("v" => 1), "pose", cfg, dir)
        @test f1 == f0 && data["v"] == 1 && isfile(replace(f0, ".jld2" => "_#1.jld2"))
        @test E.cached_result(c -> error("must not recompute"), "pose", cfg, dir)[1]["v"] == 1
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
            floors = JLD2.load(oracle_floors_path(keyed_cell_dir(cfg, "floors", "oracle")), "floors")
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
            g = JLD2.load(oracle_floors_path(results_root()), "floors")
            h = JLD2.load(oracle_floors_path(keyed_cell_dir(cfg, "floors", "oracle")), "floors")
            """)
        write(joinpath(root, "scripts", "slurm", "x.sbatch"), "#SBATCH --output=/home/u/logs/%j.out\n# /home/fine\n")
        v = E.data_policy_violations(root)
        @test sort([(x.rule, x.file, x.line) for x in v]) ==
              sort([("D1", "src/bad.jl", 1), ("D2", "src/bad.jl", 2), ("D2", "src/bad.jl", 3), ("D3", "src/bad.jl", 4),
                    ("D3", "src/bad.jl", 5), ("D3", "src/bad.jl", 6), ("D3", "src/bad.jl", 7),
                    ("D4", "src/bad.jl", 10), ("D4", "src/bad.jl", 11), ("D3", "src/bad.jl", 13),
                    ("D1", "scripts/slurm/x.sbatch", 1)])
        # a log line's `timestamp` is not a wall-clock stamp in a path; real stamps still are
        write(joinpath(root, "src", "stamps.jl"), """
            timestamp = Dates.format(now(), "yyyy-mm-dd")
            run_stamp = Dates.format(now(), "yyyy")
            f(; stamp = now()) = 1
            ts = Dates.format(Dates.now(), "HH") * "_TimeStamp"
            stamps = [Dates.format(now(), "yyyy")]
            was_stamped = Dates.format(now(), "HH")
            u = Dates.format(now(), "HH") * "_UTCStamp"
            stampDir = Dates.format(now(), "yyyy")
            RUN_STAMPS = Dates.format(now(), "yyyy")
            timestamps = Dates.format(now(), "yyyy")
            Timestamp = now()
            """)
        v3 = E.data_policy_violations(root)
        @test sort([x.line for x in v3 if x.file == "src/stamps.jl"]) == [2, 3, 4, 5, 6, 7, 8, 9]
        allow = [E.DataPolicyAllow("D3", "src/bad.jl", "miscal-sel-v5"), E.DataPolicyAllow("D1", "src/ok.jl", "nothing here")]
        v2 = E.data_policy_violations(root; allow)
        @test length(v2) == length(v3) - 1 + 1
        @test any(x -> x.rule == "STALE" && x.file == "src/ok.jl", v2)
        @test !any(x -> x.rule == "D3" && x.file == "src/bad.jl" && x.line == 5, v2)
    end
end
