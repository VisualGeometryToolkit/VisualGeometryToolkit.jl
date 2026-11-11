# The runner (src/reproduce/runner.jl, src/reproduce/run_results_branch.jl): a project's stage graph of record on a
# site. The site profiles shipped (dgx01, desktop, the template) load; the graph is validated and ordered; --only
# selects a subset with its prerequisites (or alone); array stages take their items from the config's cases or a
# list, sized and timed from it; the inputs are checked first and reported as data; the dry run prints the job
# graph for a Slurm and a local site; the local backend runs a project's sbatch launchers under bash, records a
# manifest and reuses a finished stage by its run key; the Slurm backend chains its jobs with afterok (through a
# stand-in sbatch) and reuses a completed or in-flight job; status reads a launch back; the results-branch stage
# makes the paper's results branch from a manifest.

@testsetup module RunnerFixture
    export runner_project, fake_bin
    # A paper project in a git repository: a stage graph of record, two experiment configs, launchers that record
    # what they ran (into the results root) and a data root holding one of the two inputs.
    function runner_project()
        repo = mktempdir()
        g(args...) = run(pipeline(`git -C $repo -c user.name=t -c user.email=t@t -c commit.gpgsign=false $args`; stdout = devnull, stderr = devnull))
        g("init", "-q")
        proj = mkpath(joinpath(repo, "proj"))
        mkpath(joinpath(proj, "configs", "experiments")); mkpath(joinpath(proj, "slurm")); mkpath(joinpath(proj, "scripts"))
        write(joinpath(proj, "configs", "experiments", "chrome.toml"), """
            [inputs.photos]
            tier = "raw_photos"
            path = "photos/chrome"
            [inputs.board]
            tier = "boards"
            path = "boards/a4.toml"
            [cases.zeta]
            pipeline = { lm = { max_time = 3600 } }
            [cases.alpha]
            pipeline = { lm = { max_time = 7200 } }
            [cases.mid]
            pipeline = { lm = { max_time = 60 } }
            """)
        write(joinpath(proj, "configs", "figures.toml"), """
            [runs]
            experiments = ["configs/experiments/texture_small.toml"]
            [chrome.a]
            config = "configs/experiments/chrome.toml"
            [chrome.b]
            config = "configs/experiments/gone.toml"
            """)
        for s in ("small", "large")
            write(joinpath(proj, "configs", "experiments", "texture_$s.toml"), "[inputs.sheet]\ntier = \"raw_photos\"\npath = \"photos/sheet_$s\"\n")
        end
        write(joinpath(proj, "scripts", "chrome.jl"), "# the driver\n")
        rec = raw"""echo "$(basename "$0") task=${SLURM_ARRAY_TASK_ID:-} cases=${CASES:-} sets=${SETS:-} exp=${EXPERIMENT:-} gpu=${CUDA_VISIBLE_DEVICES-unset} submit=${SLURM_SUBMIT_DIR:-}" >> "$VGTK_RESULTS_ROOT/ran.txt" """
        for n in ("setup", "prepare", "chrome", "texture")
            write(joinpath(proj, "slurm", "$n.sbatch"), "#!/bin/bash\n#SBATCH --output=logs/x_$(n)_%A_%a.log\nset -eu\n$rec\n")
        end
        write(joinpath(proj, "slurm", "broken.sbatch"), "#!/bin/bash\necho 'ERROR: no such input photos/x' >&2\nexit 1\n")
        write(joinpath(proj, "configs", "reproduce.toml"), """
            [project]
            name = "paper"
            workdir = ".."
            julia_project = "."
            code = ["scripts", "slurm"]

            [sites.desktop]
            data_root = "$(joinpath(repo, "data"))"
            results_root = "$(joinpath(repo, "results"))"

            [[stage]]
            name = "setup"
            what = "the environment"
            sbatch = "slurm/setup.sbatch"
            gpu = true

            [[stage]]
            name = "chrome_prepare"
            after = ["setup"]
            sbatch = "slurm/prepare.sbatch"
            config = "configs/experiments/chrome.toml"
            env = { EXPERIMENT = "chrome" }
            job_name = "paper-prepare-chrome"

            [[stage]]
            name = "chrome"
            after = ["chrome_prepare"]
            sbatch = "slurm/chrome.sbatch"
            driver = "scripts/chrome.jl"
            config = "configs/experiments/chrome.toml"
            gpu = true
            array = { from = "cases", select_env = "CASES", max_time = "pipeline.lm.max_time", time_margin_s = 600 }

            [[stage]]
            name = "texture"
            after = ["chrome"]
            sbatch = "slurm/texture.sbatch"
            config = "configs/experiments/texture_{item}.toml"
            gpu = true
            array = { items = ["small", "large"], select_env = "SETS" }

            [[stage]]
            name = "broken"
            after = ["setup"]
            sbatch = "slurm/broken.sbatch"
            default = false

            [[stage]]
            name = "figures"
            after = ["texture"]
            command = ["julia", "scripts/figures.jl"]
            inputs_from = [{ config = "configs/figures.toml", key = "runs.experiments" }, { config = "configs/figures.toml", key = "chrome.*.config" }]
            default = false

            [[stage]]
            name = "pdf"
            after = ["texture"]
            command = ["make", "-C", "paper"]
            time = "00:30:00"

            [[stage]]
            name = "results"
            after = ["texture"]
            results_branch = true

            [results_branch]
            project = "."
            assets = [{ from = "plots/*.png", to = "paper/fig" }]
            """)
        mkpath(joinpath(repo, "data", "photos", "chrome"))           # one input there; the board and the sheets are not
        g("add", "-A"); g("commit", "-q", "-m", "paper")
        return (; repo, proj, graph = joinpath(proj, "configs", "reproduce.toml"),
                data = joinpath(repo, "data"), results = joinpath(repo, "results"))
    end

    # A directory of stand-ins for sbatch and sacct: sbatch records its argv and prints the next job id; sacct prints
    # the states in $FAKE_SACCT.
    function fake_bin()
        bin = mktempdir()
        write(joinpath(bin, "sbatch"), raw"""#!/bin/bash
            n=$(( $(cat "$FAKE_SBATCH_LOG" 2>/dev/null | wc -l) + 101 ))
            echo "$n $* EXP=${EXPERIMENT:-} CASES=${CASES:-}" >> "$FAKE_SBATCH_LOG"
            echo "$n"
            """)
        write(joinpath(bin, "sacct"), raw"""#!/bin/bash
            cat "$FAKE_SACCT"
            """)
        chmod(joinpath(bin, "sbatch"), 0o755); chmod(joinpath(bin, "sacct"), 0o755)
        return bin
    end
end

@testitem "reproduce_runner" setup = [RunnerFixture, GitFixture] begin
    using Test
    using TOML: TOML
    using VisualGeometryToolkitEval.Reproduce
    using VisualGeometryToolkitEval.Reproduce: SITES_DIR

    nojobs(ids) = Dict{String, Vector{String}}()
    run_main(args; env = ENV, job_states = nojobs) = (io = IOBuffer(); code = reproduce_main(args; io, env, job_states); (code, String(take!(io))))
    env0 = filter(p -> !(first(p) in ("VGTK_DATA_ROOT", "VGTK_RESULTS_ROOT", "CASES", "SETS")), Dict(ENV))

    @testset "the shipped sites and the template" begin
        @test shipped_sites() == ["desktop", "dgx01"]
        dgx = load_site("dgx01")
        @test dgx.backend === :slurm && dgx.partition == "gpu" && dgx.gres == "gpu:1" && dgx.cpus_per_task == 28 && dgx.mem_per_cpu == "8G"
        desk = load_site("desktop")
        @test desk.backend === :local && desk.cuda_visible_devices == "0" && desk.cpus_per_task == 0
        tpl = load_site(joinpath(SITES_DIR, "_template.toml"))                    # every key documented there loads
        @test tpl.name == "mymachine" && tpl.backend === :slurm && tpl.extra == String[]
        @test_throws r"unknown site \"nowhere\".*desktop, dgx01" load_site("nowhere")
        d = mktempdir()
        write(joinpath(d, "mine.toml"), "[site]\nbackend = \"local\"\n[local]\ncuda_visible_devices = \"1\"\n")
        @test load_site(joinpath(d, "mine.toml")).name == "mine"                  # by path, named by the file
        @test load_site("mine"; dirs = [d]).cuda_visible_devices == "1"           # by name, from a project's sites
        write(joinpath(d, "bad.toml"), "[site]\nbackend = \"pbs\"\n")
        @test_throws r"backend = \"pbs\"" load_site(joinpath(d, "bad.toml"))
        write(joinpath(d, "bad.toml"), "[site]\nbackend = \"local\"\n[slurm]\nqueue = \"x\"\n")
        @test_throws r"unknown key slurm.queue" load_site(joinpath(d, "bad.toml"))
    end

    @testset "the graph: validated, in dependency order; --only with and without prerequisites" begin
        f = runner_project()
        g = load_stage_graph(f.graph)
        @test g.name == "paper" && g.workdir == normpath(f.proj * "/")
        @test [s.name for s in g.stages] == ["setup", "chrome_prepare", "chrome", "texture", "broken", "figures", "pdf", "results"]
        @test select_stages(g) == ["setup", "chrome_prepare", "chrome", "texture", "pdf", "results"]     # default: not broken
        @test select_stages(g; only = ["texture"]) == ["setup", "chrome_prepare", "chrome", "texture"]
        @test select_stages(g; only = ["texture"], deps = false) == ["texture"]
        @test select_stages(g; only = ["broken", "chrome_prepare"]) == ["setup", "chrome_prepare", "broken"]
        @test_throws r"unknown stage \"nope\"" select_stages(g; only = ["nope"])
        bad(s) = (p = joinpath(mktempdir(), "g.toml"); write(p, "[project]\nname = \"p\"\n" * s); p)
        @test_throws r"dependency cycle" load_stage_graph(bad("[[stage]]\nname = \"a\"\nafter = [\"b\"]\ncommand = [\"true\"]\n[[stage]]\nname = \"b\"\nafter = [\"a\"]\ncommand = [\"true\"]\n"))
        @test_throws r"no such stage" load_stage_graph(bad("[[stage]]\nname = \"a\"\nafter = [\"z\"]\ncommand = [\"true\"]\n"))
        @test_throws r"exactly one of" load_stage_graph(bad("[[stage]]\nname = \"a\"\nsbatch = \"x\"\ncommand = [\"true\"]\n"))
        @test_throws r"unknown key nodes" load_stage_graph(bad("[[stage]]\nname = \"a\"\ncommand = [\"true\"]\nnodes = 2\n"))
        @test_throws r"needs \[results_branch\] assets" load_stage_graph(bad("[[stage]]\nname = \"a\"\nresults_branch = true\n"))
    end

    @testset "array items: the config's cases (sorted, selectable) or a list; --time from the cases" begin
        f = runner_project()
        g = load_stage_graph(f.graph)
        chrome, texture = g.stages[3], g.stages[4]
        @test stage_items(g, chrome; env = env0) == ["alpha", "mid", "zeta"]
        @test stage_items(g, chrome; env = merge(env0, Dict("CASES" => "zeta,alpha"))) == ["alpha", "zeta"]
        @test_throws r"unknown case\(s\) omega" stage_items(g, chrome; env = merge(env0, Dict("CASES" => "omega")))
        @test stage_items(g, texture; env = env0) == ["small", "large"]
        @test stage_items(g, texture; env = merge(env0, Dict("SETS" => "large"))) == ["large"]
        @test stage_items(g, g.stages[1]; env = env0) == [""]
        p = plan_reproduction(g, load_site("dgx01"); env = env0, job_states = nojobs)
        @test p.stages[3].minutes == cld(7200 + 600, 60)                         # the largest case's max_time + margin
        @test any(==("--time=130"), p.stages[3].command)
    end

    @testset "missing inputs: reported as data (path, tier, used by), before anything runs" begin
        f = runner_project()
        g = load_stage_graph(f.graph)
        rows = missing_inputs(g, select_stages(g); data_root = f.data, env = env0)
        @test [r.path for r in rows] == ["boards/a4.toml", "photos/sheet_large", "photos/sheet_small"]
        @test rows[1].tier == "boards" && rows[1].used_by == ["chrome_prepare (configs/experiments/chrome.toml)", "chrome (configs/experiments/chrome.toml)"]
        @test rows[2].used_by == ["texture (configs/experiments/texture_large.toml)"]
        @test length(missing_inputs(g, select_stages(g); data_root = "", env = env0)) == 4   # no data root: every input
        # a stage that runs other drivers' configs: their inputs too, through the paths its own config lists
        rows = missing_inputs(g, ["figures"]; data_root = f.data, env = env0)
        @test [(r.path, r.tier) for r in rows] == [("boards/a4.toml", "boards"), ("configs/experiments/gone.toml", "config of record"),
                                                   ("photos/sheet_small", "raw_photos")]
        @test rows[1].used_by == ["figures (configs/experiments/chrome.toml, via configs/figures.toml)"]
        rm(joinpath(f.proj, "configs", "experiments", "texture_large.toml"))
        r = only(filter(r -> r.tier == "config of record", missing_inputs(g, ["texture"]; data_root = f.data, env = env0)))
        @test r.path == "configs/experiments/texture_large.toml"
        # the command line: a launch with a missing input runs nothing, exit 3, the report on its output
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop"]; env = env0)
        @test code == 3 && occursin("missing inputs: 3", out) && occursin("path\ttier\tused by", out) && occursin("nothing ran", out)
        @test !isfile(joinpath(f.results, "ran.txt")) && !isdir(joinpath(f.results, "_reproduce"))
    end

    @testset "the dry run: the job graph on a Slurm site and on a local one" begin
        f = runner_project()
        code, out = run_main(["--graph=$(f.graph)", "--site=dgx01", "--dry-run", "--data-root=$(f.data)", "--results-root=$(f.results)"]; env = env0)
        @test code == 0
        @test occursin("reproduce paper on dgx01 (slurm): 6 stage(s)", out)
        @test occursin("[1] setup\n", out) && occursin("[3] chrome  array of 3: alpha mid zeta", out)
        # each stage's own launcher, the site's options, the dependencies on this launch's stages
        @test occursin("submit   sbatch --parsable --partition=gpu --gres=gpu:1 --cpus-per-task=28 --mem-per-cpu=8G slurm/setup.sbatch", out)
        @test occursin("EXPERIMENT=chrome sbatch --parsable --dependency=afterok:\$setup --partition=gpu --cpus-per-task=28 --mem-per-cpu=8G --job-name=paper-prepare-chrome slurm/prepare.sbatch", out)
        @test occursin("CASES=alpha,mid,zeta sbatch --parsable --dependency=afterok:\$chrome_prepare --partition=gpu --gres=gpu:1 --cpus-per-task=28 --mem-per-cpu=8G --array=0-2 --time=130 slurm/chrome.sbatch", out)
        @test occursin("SETS=small,large sbatch --parsable --dependency=afterok:\$chrome", out) && occursin("--array=0-1 slurm/texture.sbatch", out)
        @test occursin("--job-name=paper-pdf --time=00:30:00 --output=logs/paper_pdf_%j.log '--wrap=make -C paper'", out)
        @test occursin("results_branch_of_run(ARGS[1])'", out)
        @test occursin("driver   julia --project=. scripts/chrome.jl --config=configs/experiments/chrome.toml", out)
        @test occursin("missing inputs: 3", out) && occursin("dry run: nothing submitted or run", out)
        @test !occursin("broken", out)
        @test !isdir(joinpath(f.results, "_reproduce"))                           # a dry run writes nothing
        # the local site: the same launchers under bash, device 0 for the GPU stages, the roots from the project's [sites.desktop]
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--dry-run", "--only=chrome"]; env = env0)
        @test code == 0 && occursin("reproduce paper on desktop (local): 3 stage(s)", out)
        @test occursin("data     $(f.data)", out)
        @test occursin("run      CUDA_VISIBLE_DEVICES=0 SLURM_SUBMIT_DIR=$(normpath(f.proj * "/")) bash slurm/setup.sbatch", out)
        @test occursin("run      SLURM_ARRAY_TASK_ID={0..2} CASES=alpha,mid,zeta CUDA_VISIBLE_DEVICES=0", out)
        @test occursin("CUDA_VISIBLE_DEVICES='' EXPERIMENT=chrome", out)          # a CPU stage sees no device
        # a GPU limit throttles every GPU array
        d = mktempdir(); write(joinpath(d, "shared.toml"), "[site]\nbackend = \"slurm\"\n[slurm]\ngpu_limit = 2\n")
        code, out = run_main(["--graph=$(f.graph)", "--site=$(joinpath(d, "shared.toml"))", "--dry-run", "--results-root=$(f.results)"]; env = env0)
        @test occursin("--array=0-2%2", out)
    end

    @testset "the local backend: runs the launchers, writes the manifest, reuses a finished stage" begin
        f = runner_project()
        mkpath(joinpath(f.data, "boards")); write(joinpath(f.data, "boards", "a4.toml"), "")
        mkpath(joinpath(f.data, "photos", "sheet_small")); mkpath(joinpath(f.data, "photos", "sheet_large"))
        args = ["--graph=$(f.graph)", "--site=desktop", "--only=texture"]
        code, out = run_main(args; env = env0)
        @test code == 0
        ran = readlines(joinpath(f.results, "ran.txt"))
        @test length(ran) == 1 + 1 + 3 + 2
        @test startswith(ran[1], "setup.sbatch task= ") && occursin("gpu=0", ran[1])
        @test occursin("prepare.sbatch task= cases= sets= exp=chrome gpu= ", ran[2])
        @test occursin("chrome.sbatch task=2 cases=alpha,mid,zeta", ran[5])
        @test occursin("texture.sbatch task=1 cases= sets=small,large", ran[7])
        ms = run_manifests(joinpath(f.results, "_reproduce"))
        m = TOML.parsefile(only(ms))
        @test m["project"] == "paper" && m["site"] == "desktop" && m["backend"] == "local" && !m["dirty"]
        @test m["commit"] == strip(read(`git -C $(f.repo) rev-parse HEAD`, String))
        @test m["data_root"] == f.data && m["results_root"] == f.results
        @test sort(collect(keys(m["configs"]))) == ["configs/experiments/chrome.toml", "configs/experiments/texture_large.toml", "configs/experiments/texture_small.toml"]
        @test all(r -> r["state"] == "completed", values(m["stages"])) && m["stages"]["chrome"]["tasks"] == Dict("0" => "completed", "1" => "completed", "2" => "completed")
        # again: every stage reused by its run key, nothing rerun
        code, out = run_main(args; env = env0)
        @test code == 0 && count("reused", out) == 4 && length(readlines(joinpath(f.results, "ran.txt"))) == 7
        # a changed config keys its stages anew: the stages that read it rerun, the rest are reused
        open(joinpath(f.proj, "configs", "experiments", "texture_small.toml"), "a") do io; println(io, "# changed"); end
        code, out = run_main(args; env = env0)
        @test code == 0 && count("reused", out) == 3 && length(readlines(joinpath(f.results, "ran.txt"))) == 9
        # a failing stage: stops the run, exit 4, its log and first error line; status reads it back
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=broken", "--no-deps"]; env = env0)
        @test code == 4 && occursin("stage broken failed", out) && occursin("ERROR: no such input photos/x", out)
        code, out = run_main(["status", "--graph=$(f.graph)", "--site=desktop"]; env = env0)
        @test code == 0 && occursin(r"broken\s+failed", out) && occursin("error ERROR: no such input photos/x", out)
        @test occursin("logs/paper_broken.log", out)
    end

    @testset "the Slurm backend: chained with afterok, reused when completed or in flight" begin
        f = runner_project()
        mkpath(joinpath(f.data, "boards")); write(joinpath(f.data, "boards", "a4.toml"), "")
        mkpath(joinpath(f.data, "photos", "sheet_small")); mkpath(joinpath(f.data, "photos", "sheet_large"))
        bin = fake_bin()
        log, sacct = joinpath(mktempdir(), "sbatch.log"), joinpath(mktempdir(), "sacct.txt")
        env = merge(env0, Dict("PATH" => bin * ":" * ENV["PATH"], "FAKE_SBATCH_LOG" => log, "FAKE_SACCT" => sacct,
                               "VGTK_DATA_ROOT" => f.data, "VGTK_RESULTS_ROOT" => f.results))
        args = ["--graph=$(f.graph)", "--site=dgx01", "--only=texture"]
        code, out = run_main(args; env, job_states = nojobs)
        code == 0 || @info out
        @test code == 0
        lines = readlines(log)
        @test length(lines) == 4
        @test startswith(lines[1], "101 --parsable --partition=gpu --gres=gpu:1") && endswith(lines[1], "slurm/setup.sbatch EXP= CASES=")
        @test startswith(lines[2], "102 --parsable --dependency=afterok:101 ") && occursin("EXP=chrome", lines[2])
        @test startswith(lines[3], "103 --parsable --dependency=afterok:102 ") && occursin("--array=0-2 --time=130 slurm/chrome.sbatch", lines[3]) &&
              endswith(lines[3], "CASES=alpha,mid,zeta")
        @test startswith(lines[4], "104 --parsable --dependency=afterok:103 ")
        m = TOML.parsefile(only(run_manifests(joinpath(f.results, "_reproduce"))))
        @test [m["stages"][s]["job_id"] for s in ("setup", "chrome_prepare", "chrome", "texture")] == ["101", "102", "103", "104"]
        # the next launch: setup and the prepare completed, the chrome array still running → texture is not resubmitted
        # either (its key is the same and its job is pending); the pdf stage is new and waits on the in-flight job
        write(sacct, "101|COMPLETED\n102|COMPLETED\n103_0|COMPLETED\n103_[1-2]|RUNNING\n104_[0-1]|PENDING\n")
        code, out = run_main(["--graph=$(f.graph)", "--site=dgx01", "--only=pdf"]; env, job_states = nothing)
        @test code == 0 && occursin(r"setup\s+reused", out) && occursin(r"chrome\s+103  \(in flight", out)
        lines = readlines(log)
        @test length(lines) == 5 && startswith(lines[5], "105 --parsable --dependency=afterok:104 ") && occursin("--wrap=make -C paper", lines[5])
        # status: each job's task states and, for a failed task, its log and first error line
        write(sacct, "101|COMPLETED\n102|COMPLETED\n103_0|COMPLETED\n103_1|FAILED\n103_2|COMPLETED\n104_[0-1]|CANCELLED by 0\n105|PENDING\n")
        mkpath(joinpath(f.proj, "logs"))
        write(joinpath(f.proj, "logs", "x_chrome_103_1.log"), "start\nERROR: LoadError: out of memory\n")
        code, out = run_main(["status", "--graph=$(f.graph)", "--site=dgx01"]; env, job_states = nothing)
        @test code == 0 && occursin(r"chrome\s+103\s+COMPLETED 2, FAILED 1", out)
        @test occursin("logs/x_chrome_103_1.log", out) && occursin("error ERROR: LoadError: out of memory", out)
        @test occursin(r"pdf\s+105\s+PENDING 1", out)
    end

    @testset "dependency-free: a bare julia (the stdlibs only) runs it by path, before any environment exists" begin
        f = runner_project()
        src = pkgdir(VisualGeometryToolkitEval, "src")
        boot = "include(ARGS[1]); include(ARGS[2]); exit(reproduce_main(ARGS[3:end]))"
        cmd = `$(Base.julia_cmd()) --startup-file=no -e $boot $(joinpath(src, "cli.jl")) $(joinpath(src, "reproduce", "runner.jl"))
               --graph=$(f.graph) --site=dgx01 --dry-run --results-root=$(f.results)`
        out = read(addenv(cmd, "JULIA_LOAD_PATH" => "@stdlib", "JULIA_PROJECT" => ""), String)
        @test occursin("reproduce paper on dgx01 (slurm): 6 stage(s)", out) && occursin("slurm/chrome.sbatch", out)
    end

    @testset "the results-branch stage: the renders of the graph's assets on a results branch" begin
        f = runner_project()
        git!(f.repo, "checkout", "-q", "-b", "main")
        mkpath(joinpath(f.proj, "paper", "fig")); write(joinpath(f.proj, "paper", "fig", "a.png"), UInt8[1, 2, 3])
        git!(f.repo, "add", "-A"); git!(f.repo, "commit", "-q", "-m", "submitted")
        mkpath(joinpath(f.proj, "plots")); write(joinpath(f.proj, "plots", "a.png"), UInt8[1, 2, 4]); write(joinpath(f.proj, "plots", "a.txt"), "x")
        g = load_stage_graph(f.graph)
        @test results_branch_assets(g) == [joinpath(f.proj, "plots", "a.png") => "paper/fig/a.png"]
        p = plan_reproduction(g, load_site("desktop"); env = env0, job_states = nojobs, only = ["setup"])
        recs = Dict{String, Any}("setup" => Dict{String, Any}("key" => p.stages[1].key, "job_id" => "", "state" => "completed"))
        mkpath(dirname(p.manifest))
        open(io -> TOML.print(io, Dict{String, Any}("workdir" => g.workdir, "graph" => relpath(g.path, g.workdir), "commit" => p.commit,
                                                   "dirty" => false, "site" => "desktop", "stages" => recs)), p.manifest, "w")
        r = redirect_stdout(devnull) do
            results_branch_of_run(p.manifest; scratch_root = joinpath(mktempdir(), "wt"))
        end
        @test startswith(r.branch, "results/") && TOML.parsefile(p.manifest)["results_branch"] == r.branch
        run_toml = TOML.parsefile(joinpath(r.worktree, "proj", "RUN.toml"))
        @test run_toml["run_keys"] == [p.stages[1].key] && run_toml["machine"] == "desktop" && run_toml["assets"] == ["proj/paper/fig/a.png"]
        @test read(joinpath(r.worktree, "proj", "paper", "fig", "a.png")) == UInt8[1, 2, 4]
    end
end
