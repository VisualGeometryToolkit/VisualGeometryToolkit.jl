# The paper-project check (src/paper_projects/violations.jl): a good fixture
# project — the template's files, with its example assets in place — has no
# violation, and one breach of each rule is reported under its rule.

@testitem "paper_projects" begin
    using Test
    using VisualGeometryToolkitEval.PaperProjects
    using VisualGeometryToolkitEval.PaperProjects: _populate!
    using VisualGeometryToolkitEval.Experiments: DataPolicyAllow
    import VisualGeometryToolkitCore as VgtkCore
    # the item's artifact store: a throwaway one (never the depot, which a cluster job refuses)
    store_saved = get(ENV, "VGTK_ARTIFACT_STORE", nothing)
    ENV["VGTK_ARTIFACT_STORE"] = mktempdir()

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
        # L5: the new-machine step is there and is DataArtifacts' fetch_data!
        root = good(); rm(joinpath(root, "scripts", "fetch_data.jl"))
        @test rules(check(root)) == ["L5"]
        root = good(); edit(s -> replace(s, "fetch_data!(project_catalog())" => "nothing"), root, "scripts/fetch_data.jl")
        @test rules(check(root)) == ["L5"]
        # L6: nothing tracked under data/ but its README
        root = good()
        add(root, "data/exp_raw/frames/a.png", "png")
        git(args...) = run(pipeline(`git -C $root -c user.name=t -c user.email=t@t -c commit.gpgsign=false $args`; stdout = devnull, stderr = devnull))
        git("init", "-q"); git("add", "."); git("add", "-f", "data/exp_raw/frames/a.png"); git("commit", "-q", "-m", "x")
        v = filter(x -> x.rule == "L6", check(root))
        @test [x.file for x in v] == [joinpath("data", "exp_raw", "frames", "a.png")]
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
        # D1 beyond the code: tests, examples, configs, docs, provenance (comments and text included)
        root = good()
        add(root, "configs/site.toml", "# the data live in /cluster/projects/x/data\nroot = \"rel\"\n")
        add(root, "test/paths.jl", "dir = \"/raid/me/data\"\n")
        add(root, "docs/notes.md", "Run it on /home/someone/data.\n")
        add(root, "examples/demo.jl", "x = 1   # https://example.org/home/page is a URL, not a path\n")
        v = check(root)
        @test rules(v) == ["D1"] && sort([x.file for x in v]) == ["configs/site.toml", "docs/notes.md", "test/paths.jl"]
        # D5: an artifact or download API of the project's own
        root = good()
        add(root, "src/fetch.jl", "using Downloads\nf = Downloads.download(url)\nt = install_artifact(\"a\", toml)\n")
        add(root, "scripts/get.jl", "using Pkg.Artifacts: ensure_artifact_installed\n")
        v = check(root)
        @test rules(v) == ["D5"] && length(v) == 3
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
        # the config path from the command line, cli_option(ARGS, "config"), is a config of record
        root = good(); add(root, "scripts/cliopt.jl", stage("cfg = load_mine(cli_option(ARGS, \"config\"))"))
        @test isempty(check(root))
        root = good(); add(root, "scripts/cliopt2.jl", stage("cfg = load_mine(cli_option(args, \"config\", \"x\"))"))
        @test isempty(check(root))
        root = good(); add(root, "scripts/missing.jl", stage("cfg = TOML.parsefile(projectdir(\"configs\", \"nope.toml\"))"))
        @test rules(check(root)) == ["C3"]
        for (line, what) in (("rng = StableRNG(42)", "seed"), ("const SEED = 7", "seed"),
                             ("scenes = [\"brandenburg\", \"buckingham\", \"colosseum\"]", "names"),
                             ("ws = [0.1, 0.2, 0.5, 1.0]", "table"), ("frames = 1:200", "range"),
                             ("for t in 0.1:0.1:2.0", "range"), ("pool = joinpath(data_root(), \"pools/heb.jld2\")", "path"),
                             ("x = \"/data/raw\"", "path"))
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
        edit(s -> replace(s, r"run_stages\(\"example_stage\", .*\)$"m => "foreach(st -> st.run(), STAGES)"),
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

    @testset "storage: inputs catalogued, one name one tree, the view's links into the store" begin
        catalogued(root) = edit(s -> s * """
            [artifacts.pools]
            kind = "tree"
            dir = "pools"
            mount = "data"
            record = "data"
            required = true
            what = "the pools"
            provenance = "scripts/example_stage.jl"
            """, root, "configs/data_artifacts.toml")
        bind(root, h) = write(joinpath(root, "Artifacts.toml"), """
            [pools]
            git-tree-sha1 = "$h"
            lazy = true
            [[pools.download]]
            sha256 = "$(repeat("b", 64))"
            url = "https://zenodo.org/records/ZENODO_RECORD_DATA/files/pools.tar.gz?download=1"
            """)
        # A2: a config input names a catalogued tier
        root = good(); catalogued(root); bind(root, repeat("a", 40))
        add(root, "configs/run.toml", "[inputs.p]\ntier = \"pools\"\npath = \"heb.jld2\"\n")
        @test isempty(check(root))
        add(root, "configs/run.toml", "[inputs.p]\ntier = \"other\"\npath = \"x\"\n[inputs.q]\npath = \"y\"\n")
        v = check(root)
        @test rules(v) == ["A2"] && length(v) == 2 && all(x -> x.file == "configs/run.toml", v)
        # A3: the same name, another tree, in another project's bindings or in the store's name record
        root = good(); catalogued(root); bind(root, repeat("a", 40))
        other = mktempdir(); bind(other, repeat("c", 40))
        v = check(root; stack_artifacts = [joinpath(other, "Artifacts.toml")])
        @test rules(v) == ["A3"] && occursin(repeat("c", 40), only(v).message)
        bind(other, repeat("a", 40))
        @test isempty(check(root; stack_artifacts = [joinpath(other, "Artifacts.toml")]))
        store = ENV["VGTK_ARTIFACT_STORE"]
        foreach(h -> (mkpath(joinpath(store, "names", "pools")); touch(joinpath(store, "names", "pools", h))), (repeat("a", 40), repeat("d", 40)))
        @test rules(check(root)) == ["A3"]
        rm(joinpath(store, "names"); recursive = true)
        # A4: the artifact view's links resolve into the store
        root = good(); catalogued(root); bind(root, repeat("a", 40))
        view = joinpath(root, "data", "exp_raw")
        write(joinpath(view, ".artifact_view"), "view\n")
        tree = joinpath(store, repeat("a", 40)); mkpath(tree); write(joinpath(tree, "heb.jld2"), "x")
        symlink(joinpath(tree, "heb.jld2"), joinpath(view, "heb.jld2"))
        @test isempty(check(root))
        depot = mktempdir(); write(joinpath(depot, "old.jld2"), "y")
        symlink(joinpath(depot, "old.jld2"), joinpath(view, "old.jld2"))
        v = check(root)
        @test rules(v) == ["A4"] && only(v).file == joinpath("data", "exp_raw", "old.jld2")
        # … and the cluster without a store is reported, not thrown
        withenv("VGTK_ARTIFACT_STORE" => nothing, "VGTK_SITE" => nothing, "NODE_CACHE" => mktempdir()) do
            @test any(x -> x.rule == "A4" && occursin("refused", x.message), check(root))
        end
    end

    @testset "produced entries: a derived input of record, named by its producer, never bound" begin
        import VisualGeometryToolkitEval.Experiments as E
        import VisualGeometryToolkitEval.DataArtifacts as DA
        using TOML: TOML
        root = good()
        edit(s -> s * """
            [artifacts.pools]
            kind = "tree"
            dir = "pools"
            mount = "data"
            record = "data"
            required = true
            what = "the pools"
            provenance = "scripts/example_stage.jl"
            [artifacts.cells_of_record]
            kind = "produced"
            producer = "make_cells"
            covers = ["demo/cells"]
            tier = "derived"
            what = "the scored cells"
            provenance = "regenerated from the pools by the stage make_cells"
            """, root, "configs/data_artifacts.toml")
        write(joinpath(root, "Artifacts.toml"), """
            [pools]
            git-tree-sha1 = "$(repeat("a", 40))"
            lazy = true
            [[pools.download]]
            sha256 = "$(repeat("b", 64))"
            url = "https://zenodo.org/records/ZENODO_RECORD_DATA/files/pools.tar.gz?download=1"
            """)
        add(root, "configs/reproduce.toml", """
            [project]
            name = "demo"
            workdir = ".."
            [[stage]]
            name = "make_cells"
            command = ["julia", "scripts/example_stage.jl"]
            config = "configs/make_cells.toml"
            """)
        add(root, "configs/make_cells.toml", "[inputs.p]\ntier = \"pools\"\npath = \"heb.jld2\"\n")
        cell = "[inputs.c]\ntier = \"derived\"\npath = \"demo/cells/budget\"\n"
        add(root, "configs/run.toml", cell)
        storage(v) = sort!(unique([x.rule for x in v if x.rule in ("A1", "A2")]))
        @test isempty(storage(check(root)))                        # covered by the produced entry, its producer sound
        cat = DA.project_catalog(root)
        @test DA.required_artifacts(cat) == ["pools"] && isempty(DA.artifact_parts(cat, "cells_of_record"))
        @test isempty(DA.catalog_violations(cat))
        io = IOBuffer(); DA.fetch_data!(cat; argv = ["--list"], io)
        @test occursin(r"cells_of_record +produced by stage make_cells", String(take!(io)))
        # the configs stay as they are, so their run keys do: the check reads the covered config, never rewrites it, and
        # its key is the one computed with no catalog at all
        @test read(joinpath(root, "configs", "run.toml"), String) == cell
        k = E.run_key(TOML.parsefile(joinpath(root, "configs", "run.toml")))
        root2 = good()
        foreach(f -> cp(joinpath(root, f), joinpath(root2, f); force = true),
                ("Artifacts.toml", "configs/data_artifacts.toml", "configs/reproduce.toml", "configs/make_cells.toml", "configs/run.toml"))
        rm(joinpath(root, "configs", "data_artifacts.toml"))
        @test E.run_key(TOML.parsefile(joinpath(root, "configs", "run.toml"))) == k
        # an input outside `covers`, or of another tier, is not covered
        @test isempty(storage(check(root2)))
        add(root2, "configs/run.toml", "[inputs.c]\ntier = \"derived\"\npath = \"demo/cellsX\"\n")
        @test storage(check(root2)) == ["A2"]
        add(root2, "configs/run.toml", "[inputs.c]\ntier = \"other\"\npath = \"demo/cells/budget\"\n")
        @test storage(check(root2)) == ["A2"]
        add(root2, "configs/run.toml", cell)
        # the producer: a stage of the graph, reading published or produced inputs only, no cycle
        edit(s -> replace(s, "producer = \"make_cells\"" => "producer = \"nowhere\""), root2, "configs/data_artifacts.toml")
        v = check(root2); @test storage(v) == ["A1"] && any(x -> occursin("no stage", x.message), v)
        edit(s -> replace(s, "producer = \"nowhere\"" => "producer = \"make_cells\""), root2, "configs/data_artifacts.toml")
        add(root2, "configs/make_cells.toml", "[inputs.p]\ntier = \"unpublished\"\npath = \"raw\"\n")
        v = check(root2); @test "A1" in storage(v) && any(x -> occursin("neither published nor produced", x.message), v)
        add(root2, "configs/make_cells.toml", "[inputs.p]\ntier = \"derived\"\npath = \"demo/cells/prior\"\n")
        v = check(root2); @test storage(v) == ["A1"] && any(x -> occursin("cycle", x.message), v)
        add(root2, "configs/make_cells.toml", "[inputs.p]\ntier = \"pools\"\npath = \"heb.jld2\"\n")
        # its own keys only, and never bound
        edit(s -> replace(s, "producer = \"make_cells\"" => "producer = \"make_cells\"\nrequired = true"), root2, "configs/data_artifacts.toml")
        @test any(m -> occursin("not `required`", m), DA.catalog_violations(DA.project_catalog(root2)))
        edit(s -> replace(s, "required = true\n" => "", "tier = \"derived\"\n" => ""), root2, "configs/data_artifacts.toml")
        @test any(m -> occursin("no `tier`", m), DA.catalog_violations(DA.project_catalog(root2)))   # a path is its tier's
    end

    @testset "release (opt-in): public hosts only" begin
        root = good()
        @test rules(check(root; checks = (:release,))) == ["R1"]      # the template's stub record is not published yet
        edit(s -> replace(s, r"\[records\.[^\n]*\]" => m -> m * "\nhost = \"huggingface\"\nrepo = \"lab/data\"") , root, "configs/data_artifacts.toml")
        write(joinpath(root, "Artifacts.toml"), """
            [pools]
            git-tree-sha1 = "$(repeat("a", 40))"
            lazy = true
            [[pools.download]]
            sha256 = "$(repeat("b", 64))"
            url = "file:///tmp/pools.tar.gz"
            """)
        v = check(root; checks = (:release,))
        @test rules(v) == ["R1", "R2", "R3"]
        # a platform-specific entry (`[[name]]`, one table per platform) is read, not thrown on
        write(joinpath(root, "Artifacts.toml"), """
            [[pools]]
            arch = "x86_64"
            os = "linux"
            git-tree-sha1 = "$(repeat("a", 40))"
            [[pools.download]]
            sha256 = "$(repeat("b", 64))"
            url = "https://zenodo.org/records/ZENODO_RECORD_DATA/files/pools.tar.gz?download=1"
            """)
        @test count(x -> x.rule == "R3", check(root; checks = (:release,))) == 1
        @test !any(x -> x.rule == "A3", check(root; checks = (:storage,), stack_artifacts = [joinpath(root, "Artifacts.toml")]))
    end

    @testset "selection and display" begin
        root = good(); rm(joinpath(root, "README.md"))
        add(root, "src/paths.jl", "const RAW = \"/mnt/data/x\"\n")
        @test rules(check(root; checks = (:layout,))) == ["L1"]
        @test rules(check(root; checks = (:data_policy,))) == ["D1"]
        v = only(check(root; checks = (:data_policy,)))
        @test sprint(show, v) == "src/paths.jl:1: [data_policy/D1] const RAW = \"/mnt/data/x\""
    end
    store_saved === nothing ? delete!(ENV, "VGTK_ARTIFACT_STORE") : (ENV["VGTK_ARTIFACT_STORE"] = store_saved)
end

@testitem "paper_projects: paper_figures_dir" begin
    # The directory a paper reads its generated assets from — `figures/` beside
    # its main .tex, under the project root — once for every paper project
    # (RANSACScoringDoneRight's neurips2026 and radiometric-camera's experiments
    # each defined `paper_figdir()`; the template wrote it out per driver).
    using VisualGeometryToolkitEval.PaperProjects: paper_figures_dir
    using VisualGeometryToolkitEval.Experiments: set_project_root!
    root = mktempdir()
    @test paper_figures_dir(; root) == joinpath(root, "paper", "figures")
    @test paper_figures_dir(joinpath("papers", "icra", "main.tex"); root) == joinpath(root, "papers", "icra", "figures")
    @test paper_figures_dir(joinpath("paper", "main.tex"); root = "") == joinpath("paper", "figures")
    set_project_root!(root)
    try
        @test paper_figures_dir() == joinpath(root, "paper", "figures")
    finally
        set_project_root!("")
    end
end

@testitem "paper_projects: one layout for the template and the data policy's scan" begin
    # The template's folders and the folders D1 reads derive from one list (Experiments' _PROJECT_LAYOUT), so neither
    # drifts from the other; the lists are the ones each held by hand before.
    import VisualGeometryToolkitEval as VgtkEval
    E, P = VgtkEval.Experiments, VgtkEval.PaperProjects
    @test Set(P.PAPER_PROJECT_TEMPLATE) == Set(Any["configs", "scripts", "src", "test", "plots", "data" => ["exp_raw", "exp_pro", "sims"]])
    @test E._layout_dirs(:code) == ["src", "scripts"]
    @test E._MACHINE_PATH_DIRS == ["test", "examples", "configs", "docs", "provenance"]
    @test E._layout_dirs(:markdown) == ["", "data"]
    @test allunique(l.dir for l in E._PROJECT_LAYOUT) && all(l -> l.scan in (:code, :text, :markdown, :none), E._PROJECT_LAYOUT)
    # every folder the template makes is read by the scan or declared unread
    @test all(t -> (t isa Pair ? first(t) : t) in (l.dir for l in E._PROJECT_LAYOUT), P.PAPER_PROJECT_TEMPLATE)
    # a machine path in a :text folder's file and in data/README.md is found; one under plots/ (renders) is not read
    root = mktempdir()
    for (f, text) in ("configs/a.toml" => "root = \"/raid/x\"\n", "data/README.md" => "see /raid/y\n", "plots/p.tex" => "/raid/z\n",
                      "src/m.jl" => "x = 1\n")
        mkpath(dirname(joinpath(root, f))); write(joinpath(root, f), text)
    end
    @test sort([v.file for v in E.data_policy_violations(root)]) == sort([joinpath("configs", "a.toml"), joinpath("data", "README.md")])
end
