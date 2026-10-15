# The paper-project template end to end (src/paper_projects/template.jl):
# new_paper_project generates a DrWatson project, a fresh Julia instantiates it
# against THIS checkout of VisualGeometryEval, runs its example stage and its
# suite — which asserts zero paper-project violations — and then one injected
# breach of each check is detected.

@testitem "paper_project_template" timeout = 3600 begin
    using Test
    using Pkg
    using VisualGeometryEval
    using VisualGeometryEval.PaperProjects

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
    # Before its stage has run, the paper reads assets that do not exist yet.
    @test Set(x.rule for x in paper_project_violations(root)) == Set(["missing_asset"])

    # A fresh process: dev this checkout, instantiate, run the stages, run the suite.
    vgeval = pkgdir(VisualGeometryEval)
    # This checkout, and every dependency the running environment tracks by path
    # (an unreleased upstream checkout the code of this checkout needs: under a
    # released stack there is none, and only this checkout is developed).
    own_deps = keys(Pkg.TOML.parsefile(joinpath(vgeval, "Project.toml"))["deps"])
    tracked = [d.source for d in values(Pkg.dependencies()) if d.is_tracking_path && d.name in own_deps]
    paths = [vgeval; tracked]
    script = """
        using Pkg
        Pkg.develop([PackageSpec(path = p) for p in $(repr(paths))])
        Pkg.instantiate()
        include(joinpath($(repr(root)), "scripts", "example_stage.jl"))
        include(joinpath($(repr(root)), "test", "runtests.jl"))
        """
    env = copy(ENV)
    for k in ("CAU_DATA_ROOT", "CAU_RESULTS_ROOT", "JULIA_PROJECT")
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
    @test length(runs) == 1 && startswith(only(runs), "example-")
    run_dir = joinpath(root, "data", "exp_pro", "example", only(runs))
    @test isfile(joinpath(run_dir, "_provenance.toml")) && isfile(joinpath(run_dir, "_config.toml"))
    @test isempty(paper_project_violations(root))

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
