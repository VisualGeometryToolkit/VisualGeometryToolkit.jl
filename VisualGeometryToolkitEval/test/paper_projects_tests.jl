# The paper-project check (src/paper_projects/violations.jl): a good fixture
# project — the template's files, with its example assets in place — has no
# violation, and one breach of each rule is reported under its rule.

@testitem "paper_projects" begin
    using Test
    using VisualGeometryToolkitEval.PaperProjects
    using VisualGeometryToolkitEval.PaperProjects: _populate!
    using VisualGeometryToolkitEval.Experiments: DataPolicyAllow

    # The template's files plus the assets its example stage would write. Not
    # a git repository: L3 reads the project's .gitignore.
    function good()
        root = mktempdir()
        _populate!(root; name = "Demo", papers = ["paper/main.tex", "paper/supp.tex"])
        figs = joinpath(root, "paper", "figures")
        write(joinpath(figs, "example_fit.tex"), "\\begin{tikzpicture}\\end{tikzpicture}\n")
        write(joinpath(figs, "example_fit_caption.tex"), "\\caption{x}\n")
        write(joinpath(figs, "example_numbers.tex"), "\\newcommand{\\ExampleSlope}{2.01}\n\\newcommand{\\ExampleN}{40}\n")
        return root
    end
    check(root; kw...) = paper_project_violations(root; papers = ["paper/main.tex", "paper/supp.tex"], kw...)
    rules(v) = sort!(unique([x.rule for x in v]))
    edit(f, root, rel) = (p = joinpath(root, rel); write(p, f(read(p, String))))
    add(root, rel, text) = (p = joinpath(root, rel); mkpath(dirname(p)); write(p, text))

    @testset "the template is compliant" begin
        root = good()
        @test isempty(check(root))
        @test isempty(check(root; checks = (PAPER_PROJECT_CHECKS..., :contract)))
        # the conventions' pieces are all there
        for p in ("Project.toml", "README.md", "REPRODUCE.md", "CLAUDE.md", ".gitignore", "Artifacts.toml",
                  "configs/example.toml", "configs/data_artifacts.toml", "scripts/example_stage.jl",
                  "src/Demo.jl", "test/runtests.jl", "paper/main.tex", "paper/supp.tex", "data/README.md")
            @test isfile(joinpath(root, p))
        end
        for d in ("plots", "data/exp_raw", "data/exp_pro", "data/sims", "paper/figures")
            @test isdir(joinpath(root, d))
        end
        proj = PaperProjects.TOML.parsefile(joinpath(root, "Project.toml"))
        @test proj["name"] == "Demo" && proj["compat"]["VisualGeometryToolkitEval"] == (v = pkgversion(parentmodule(PaperProjects)); v.major == 0 ? "0.$(v.minor)" : string(v.major))
        @test issubset(("DrWatson", "VisualGeometryToolkitEval"), keys(proj["deps"]))
    end

    @testset "layout" begin
        root = good(); rm(joinpath(root, "README.md"))
        @test rules(check(root)) == ["L1"]
        root = good(); edit(s -> replace(s, r"VisualGeometryToolkitEval = .*\n" => ""), root, "Project.toml")
        @test rules(check(root)) == ["L2"]
        root = good(); edit(s -> replace(s, "/data/exp_raw\n" => "", "/data/exp_pro\n" => ""), root, ".gitignore")
        v = check(root)
        @test rules(v) == ["L3"] && length(v) == 2
        root = good(); edit(s -> replace(s, "paper_project_violations(projectdir()" => "isempty(("), root, "test/runtests.jl")
        @test "L4" in rules(check(root))
    end

    @testset "data policy" begin
        root = good()
        add(root, "src/paths.jl", "const RAW = \"/mnt/data/x\"\nconst P = artifact\"pools\"\n")
        v = check(root)
        @test rules(v) == ["D1", "D4"]
        @test all(x -> x.check === :data_policy && x.file == "src/paths.jl", v)
        # a deferred violation is allowlisted; a stale entry is reported
        v2 = check(root; allow = [DataPolicyAllow("D1", "src/paths.jl", "/mnt/data"),
                                  DataPolicyAllow("D3", "src/paths.jl", "nothing")])
        @test sort(rules(v2)) == ["D4", "STALE"]
    end

    @testset "asset index" begin
        root = good()
        edit(s -> replace(s, "\\end{document}" => "\\includegraphics{figures/hand.png}\n\\end{document}"), root, "paper/main.tex")
        v = check(root)
        @test rules(v) == ["missing_asset", "unindexed_asset"]
        @test all(x -> x.check === :reproduce && x.file == "paper/main.tex" && x.line > 0, v)
        # the asset declared static with provenance clears both but the missing file
        write(joinpath(root, "paper", "figures", "hand.png"), "png")
        edit(s -> s * "| — | `paper/figures/hand.png` | static | — | — | — | drawn by hand |\n", root, "REPRODUCE.md")
        @test isempty(check(root))
    end

    @testset "configs of record" begin
        stage(body) = """
            using DrWatson
            # reproduce: writes paper/figures/x.tex
            $body
            write_run_records(dir, cfg; driver = "x", config_source = p, key = run_key(cfg))
            run_stages("x", STAGES)
            """
        root = good(); add(root, "scripts/noconfig.jl", stage("cfg = Dict(\"a\" => 1)"))
        @test rules(check(root)) == ["C1"]
        # a config named by its name (a project builder loads it) is a config of record
        root = good(); add(root, "scripts/byname.jl", stage("cfg = builder(\"example\")"))
        @test isempty(check(root))
        root = good(); add(root, "scripts/missing.jl", stage("cfg = TOML.parsefile(projectdir(\"configs\", \"nope.toml\"))"))
        @test rules(check(root)) == ["C3"]
        for (line, what) in (("rng = StableRNG(42)", "seed"), ("const SEED = 7", "seed"),
                             ("scenes = [\"brandenburg\", \"buckingham\", \"colosseum\"]", "names"),
                             ("ws = [0.1, 0.2, 0.5, 1.0]", "table"), ("frames = 1:200", "range"),
                             ("for t in 0.1:0.1:2.0", "range"), ("pool = joinpath(data_root(), \"pools/heb.jld2\")", "path"),
                             ("x = \"/srv/data/raw\"", "path"))
            root = good()
            add(root, "scripts/const.jl", stage("cfg = TOML.parsefile(projectdir(\"configs\", \"example.toml\"))\n$line"))
            v = check(root)
            @test rules(v) == ["C2"]
            @test only(v).line == 4
        end
        # structure, not data: none of these is a constant of record
        root = good()
        add(root, "scripts/fine.jl", stage("""
            cfg = TOML.parsefile(projectdir("configs", "example.toml"))
            for i in 1:3; end
            xs = v[1:n]
            out = joinpath(dir, "counts.jld2")
            include(srcdir("helpers.jl"))
            @info "wrote" path = "a/b.jld2"
            \"\"\"
                docstring naming [1, 2, 3, 4] and "pools/x.jld2"
            \"\"\"
            f() = nothing   # seed = 3
            """))
        @test isempty(check(root))
        # a justified constant opts out with a reason; a reasonless opt-out is X1
        root = good()
        add(root, "scripts/optout.jl", stage("""
            cfg = TOML.parsefile(projectdir("configs", "example.toml"))
            # paper-project: allow C2 the solver's published test vector, not a choice of ours
            ref = [1.0, 2.0, 3.0, 4.0]
            rng = StableRNG(1)   # paper-project: allow C2
            """))
        v = check(root)
        @test rules(v) == ["X1"] && only(v).check === :allow
        # an allowlist entry for a configs rule
        root = good(); add(root, "scripts/noconfig.jl", stage("cfg = Dict(\"a\" => 1)"))
        @test isempty(check(root; allow = [DataPolicyAllow("C1", "scripts/noconfig.jl", "reads no config")]))
        # a script that produces nothing is not held to the rules
        root = good(); add(root, "scripts/look.jl", "x = [0.1, 0.2, 0.3, 0.4]\nprintln(x)\n")
        @test isempty(check(root))
    end

    @testset "provenance" begin
        root = good()
        add(root, "scripts/bare.jl", """
            cfg = TOML.parsefile(projectdir("configs", "example.toml"))
            jldsave(joinpath(results_root(), "x", "y.jld2"); a = 1)
            """)
        @test rules(check(root)) == ["P1", "P2"]
        # through an included prelude, the shared writer counts
        root = good()
        add(root, "src/writer.jl", "record!(d, cfg) = write_run_records(d, cfg; driver = \"x\", config_source = \"c\", key = run_key(cfg))\n")
        add(root, "scripts/via.jl", """
            include(srcdir("writer.jl"))
            cfg = TOML.parsefile(projectdir("configs", "example.toml"))
            run_stages("via", [])
            """)
        @test isempty(check(root))
        # a generator of the index without the stage runner
        root = good()
        edit(s -> replace(s, "run_stages(\"example_stage\", STAGES)" => "foreach(st -> st.run(), STAGES)"),
             root, "scripts/example_stage.jl")
        @test rules(check(root)) == ["S1"]
    end

    @testset "data catalog" begin
        root = good()
        write(joinpath(root, "Artifacts.toml"), """
            [pools]
            git-tree-sha1 = "$(repeat("a", 40))"
            lazy = true
            [[pools.download]]
            sha256 = "$(repeat("b", 64))"
            url = "https://zenodo.org/records/ZENODO_RECORD_DATA/files/pools.tar.gz?download=1"
            """)
        v = check(root)
        @test rules(v) == ["A1"] && any(x -> occursin("pools: bound but not catalogued", x.message), v)
        edit(s -> s * """
            [artifacts.pools]
            kind = "tree"
            dir = "pools"
            mount = "data"
            record = "data"
            required = true
            what = "the pools"
            provenance = "scripts/example_stage.jl"
            """, root, "configs/data_artifacts.toml")
        @test isempty(check(root))
        rm(joinpath(root, "configs", "data_artifacts.toml"))
        @test rules(check(root)) == ["A1"]
    end

    @testset "selection and display" begin
        root = good(); rm(joinpath(root, "README.md"))
        add(root, "src/paths.jl", "const RAW = \"/mnt/data/x\"\n")
        @test rules(check(root; checks = (:layout,))) == ["L1"]
        @test rules(check(root; checks = (:data_policy,))) == ["D1"]
        v = only(check(root; checks = (:data_policy,)))
        @test sprint(show, v) == "src/paths.jl:1: [data_policy/D1] const RAW = \"/mnt/data/x\""
    end
end
