# The paper-project template end to end (src/paper_projects/template.jl):
# new_paper_project generates a DrWatson project, a fresh Julia instantiates it
# against THIS checkout of VisualGeometryToolkitEval, runs its example stage and its
# suite — which asserts zero paper-project violations — and then one injected
# breach of each check is detected.

@testitem "paper_project_template" timeout = 3600 begin
    using Test
    using Pkg
    using SHA: sha256
    using VisualGeometryToolkitEval
    using VisualGeometryToolkitEval.PaperProjects

    root = joinpath(mktempdir(), "DemoPaper")
    prev = Base.active_project()
    offline = Pkg.OFFLINE_MODE[]
    Pkg.offline(true)                              # the depot has DrWatson; no network
    try
        new_paper_project(root; papers = ["paper/main.tex"])
    finally
        Pkg.offline(offline)
    end
    @test Base.active_project() == prev            # initialize_project's activation is undone
    @test isfile(joinpath(root, "scripts", "example_stage.jl")) && !isfile(joinpath(root, "scripts", "intro.jl"))
    @test isdir(joinpath(root, ".git"))
    # `.tex` keeps `text` and loses git's built-in tex diff driver, whose regex libgit2 cannot compile on macOS
    # (an illegal byte sequence that failed `LibGit2.add!` on every .tex path); the paper is committed
    attrs = read(joinpath(root, ".gitattributes"), String)
    @test !occursin("diff=tex", attrs) && occursin(r"^\*\.tex\s+text\s*$"m, attrs)
    @test success(`git -C $root ls-files --error-unmatch paper/main.tex`) && isempty(read(`git -C $root status --porcelain`, String))
    # Before its stage has run, the paper reads assets that do not exist yet.
    @test Set(x.rule for x in paper_project_violations(root)) == Set(["missing_asset"])

    # A fresh process: dev this checkout, instantiate, run the stages, run the suite.
    vgeval = pkgdir(VisualGeometryToolkitEval)
    # This checkout, and every dependency the running environment tracks by path
    # (an unreleased upstream checkout the code of this checkout needs: under a
    # released stack there is none, and only this checkout is developed).
    own_deps = keys(Pkg.TOML.parsefile(joinpath(vgeval, "Project.toml"))["deps"])
    tracked = [d.source for d in values(Pkg.dependencies()) if d.is_tracking_path && d.name in own_deps]
    paths = [vgeval; tracked]
    # The results branch of a real change: the first run deploys into the paper, which is then committed
    # (a clean base); a second commit changes the config (n) and the driver (one number more), and the
    # rerun with --results-branch compares its renders against that base.
    script = """
        using Pkg
        Pkg.develop([PackageSpec(path = p) for p in $(repr(paths))])
        Pkg.instantiate()
        include(joinpath($(repr(root)), "scripts", "example_stage.jl"))
        include(joinpath($(repr(root)), "test", "runtests.jl"))
        git(args...) = run(`git -C $(repr(root)) -c user.name=Demo -c user.email=demo@example.org \$args`)
        git("add", "-A"); git("commit", "-q", "-m", "The paper with its example assets")
        write_edit(f, p) = write(p, f(read(p, String)))
        write_edit(s -> replace(s, "n = 40" => "n = 60"), joinpath($(repr(root)), "configs", "example.toml"))
        write_edit(s -> replace(s, "    write_numbers(MACROS, nums)" =>
            "    add_number!(nums, \\"ExampleIntercept\\", f[\\"intercept\\"]; digits = 2, better = :none)\\n    write_numbers(MACROS, nums)"),
            joinpath($(repr(root)), "scripts", "example_stage.jl"))
        git("commit", "-q", "-am", "More samples; the intercept as a number")
        empty!(ARGS); push!(ARGS, "--results-branch")
        include(joinpath($(repr(root)), "scripts", "example_stage.jl"))
        """
    env = copy(ENV)
    for k in ("VGTK_DATA_ROOT", "VGTK_RESULTS_ROOT", "VGTK_RESULTS_WORKTREES", "JULIA_PROJECT")
        delete!(env, k)
    end
    env["JULIA_PKG_OFFLINE"] = "true"
    env["JULIA_LOAD_PATH"] = "@:@stdlib"
    cmd = setenv(`$(Base.julia_cmd()) --startup-file=no --project=$root -e $script`, env)
    log = joinpath(mktempdir(), "template.log")
    ok = open(io -> success(pipeline(cmd; stdout = io, stderr = io)), log, "w")
    ok || @info "the generated project's run failed" log = read(log, String)
    @test ok
    @test isfile(joinpath(root, "Manifest.toml"))
    # the stage wrote the paper's assets and the run's records, keyed
    for f in ("example_fit.tex", "example_fit_caption.tex", "example_numbers.tex")
        @test isfile(joinpath(root, "paper", "figures", f))
    end
    runs = readdir(joinpath(root, "data", "exp_pro", "example"))
    @test length(runs) == 2 && all(startswith("example-"), runs)    # n = 40, then n = 60: two run keys
    for r in runs
        @test isfile(joinpath(root, "data", "exp_pro", "example", r, "_provenance.toml"))
        @test isfile(joinpath(root, "data", "exp_pro", "example", r, "_config.toml"))
    end
    @test isempty(paper_project_violations(root))
    # --results-branch: the run's paper assets on a results branch in a worktree beside the project
    branch = only(split(read(`git -C $root branch --list '--format=%(refname:short)' 'results/*'`, String)))
    worktree = joinpath(dirname(root), "DemoPaper-results", replace(branch, "/" => "-"))
    @test isfile(joinpath(worktree, "RUN.toml")) && isfile(joinpath(worktree, "COMPARISON.md"))
    @test isfile(joinpath(worktree, "paper", "figures", "example_numbers.tex"))
    @test strip(read(`git -C $root symbolic-ref --short HEAD`, String)) != branch
    # the renders stay out of the paper: the checkout is clean and its paper still holds the base's numbers
    @test isempty(read(`git -C $root status --porcelain`, String))
    @test occursin("\\newcommand{\\ExampleN}{40}", read(joinpath(root, "paper", "figures", "example_numbers.tex"), String))
    @test occursin("\\newcommand{\\ExampleN}{60}", read(joinpath(worktree, "paper", "figures", "example_numbers.tex"), String))
    # RUN.toml: the evaluated commit, clean; the data tier of the example (its synthetic draws, by sha256)
    manifest = Pkg.TOML.parsefile(joinpath(worktree, "RUN.toml"))
    head = strip(read(`git -C $root rev-parse HEAD`, String))
    @test manifest["commit"] == head && manifest["dirty"] == false
    @test manifest["base_commit"] == head
    key = only(manifest["run_keys"])
    samples = joinpath(root, "data", "exp_pro", "example", "example-" * key, "samples.tsv")
    @test manifest["tiers"] == Dict("example_samples" => Dict("hash" => bytes2hex(open(sha256, samples)),
                                                              "of" => "samples.tsv sha256"))
    @test !("tiers" in manifest["missing"])
    # COMPARISON.md: a real change against a committed base
    verdicts = Dict(r.captures[1] => (strip(r.captures[2]), strip(r.captures[3]), strip(r.captures[4]))
                    for r in eachmatch(r"^\| `\\(\w+)` \|[^|]*\|([^|]*)\|([^|]*)\|[^|]*\|[^|]*\| (\w+) \|$"m,
                                       read(joinpath(worktree, "COMPARISON.md"), String)))
    @test verdicts["ExampleN"] == ("`40`", "`60`", "CHANGED")
    @test verdicts["ExampleIntercept"][[1, 3]] == ("", "NEW")
    @test verdicts["ExampleSlope"][3] == "CHANGED"                 # other draws, another fitted slope

    # One injected breach of each check, each detected under its check.
    function breach(f)
        dst = joinpath(mktempdir(), "DemoPaper")
        cp(root, dst)
        f(dst)
        return Set(x.check for x in paper_project_violations(dst))
    end
    edit(f, p) = write(p, f(read(p, String)))
    stage = (d) -> joinpath(d, "scripts", "example_stage.jl")
    @test breach(d -> rm(joinpath(d, "README.md"))) == Set([:layout])
    @test breach(d -> edit(s -> s * "const RAW = \"/mnt/data/x\"\n", joinpath(d, "src", "DemoPaper.jl"))) == Set([:data_policy])
    @test breach(d -> edit(s -> replace(s, "\\end{document}" => "\\input{figures/unlisted}\n\\end{document}"),
                           joinpath(d, "paper", "main.tex"))) == Set([:reproduce])
    @test breach(d -> edit(s -> replace(s, "rng = StableRNG(s[\"seed\"])" => "rng = StableRNG(1)"), stage(d))) == Set([:configs])
    @test breach(d -> edit(s -> replace(s, r"    write_run_records\(.*\n" => ""), stage(d))) == Set([:provenance])
    @test breach(d -> write(joinpath(d, "Artifacts.toml"),
        "[x]\ngit-tree-sha1 = \"$(repeat("a", 40))\"\nlazy = true\n")) == Set([:catalog])
end
