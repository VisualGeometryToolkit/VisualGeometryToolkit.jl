# The runner (src/reproduce/runner.jl, src/reproduce/run_results_branch.jl): a project's stage graph of record on a
# site. The site profiles shipped (dgx01, desktop, the template) load; the graph is validated and ordered; --only
# selects a subset with its prerequisites (or alone); array stages take their items from the config's cases or a
# list, sized and timed from it; the inputs are checked first and reported as data; the dry run prints the job
# graph for a Slurm and a local site; the local backend runs a project's sbatch launchers under bash, records a
# manifest and reuses a finished stage by its run key; the Slurm backend submits each stage through its marker
# wrapper and chains its jobs with afterok (through a stand-in sbatch); reuse and status read the completion markers
# and the queue (stand-in squeue), never accounting (a stand-in sacct that fails as on a cluster without it); the
# job-name prefix labels every job and keys nothing; markers are written atomically; the results-branch stage makes
# the paper's results branch from a manifest.

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

    # A directory of stand-ins for sbatch, squeue and sacct: sbatch records its argv and prints the next job id;
    # squeue prints the lines of $FAKE_SQUEUE (`<id>[_<task>]|<state>|<reason>`) of the job given by -j; sacct fails
    # as it does on a cluster that keeps no accounting (dgx01).
    function fake_bin()
        bin = mktempdir()
        write(joinpath(bin, "sbatch"), raw"""#!/bin/bash
            n=$(( $(cat "$FAKE_SBATCH_LOG" 2>/dev/null | wc -l) + 101 ))
            echo "$n $* EXP=${EXPERIMENT:-} CASES=${CASES:-}" >> "$FAKE_SBATCH_LOG"
            echo "$n"
            """)
        write(joinpath(bin, "squeue"), raw"""#!/bin/bash
            while [ $# -gt 0 ]; do [ "$1" = -j ] && id=$2; shift; done
            grep -E "^${id}(_|[|])" "$FAKE_SQUEUE" || { echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1; }
            """)
        write(joinpath(bin, "sacct"), raw"""#!/bin/bash
            echo "Slurm accounting storage is disabled" >&2
            exit 1
            """)
        foreach(n -> chmod(joinpath(bin, n), 0o755), ("sbatch", "squeue", "sacct"))
        return bin
    end
end

@testitem "reproduce_runner" setup = [RunnerFixture, GitFixture] begin
    using Test
    using TOML: TOML
    using VisualGeometryToolkitEval.Reproduce
    using VisualGeometryToolkitEval.Reproduce: SITES_DIR

    noqueue(ids) = Dict{String, Dict{Int, String}}()
    run_main(args; env = ENV, queue_states = noqueue, job_states = nothing) =
        (io = IOBuffer(); code = reproduce_main(args; io, env, queue_states, job_states); (code, String(take!(io))))
    # the caller's environment without the roots, the selections and any Slurm job's variables (the suite may run
    # inside one: its SLURM_ARRAY_JOB_ID would name every stand-in job's markers)
    env0 = filter(p -> !(first(p) in ("VGTK_DATA_ROOT", "VGTK_RESULTS_ROOT", "CASES", "SETS")) && !startswith(first(p), "SLURM_"), Dict(ENV))

    @testset "the shipped sites and the template" begin
        @test shipped_sites() == ["desktop", "dgx01"]
        dgx = load_site("dgx01")
        @test dgx.backend === :slurm && dgx.partition == "gpu" && dgx.gres == "gpu:1" && dgx.cpus_per_task == 28 && dgx.mem_per_cpu == "8G"
        desk = load_site("desktop")
        @test desk.backend === :local && desk.cuda_visible_devices == "0" && desk.cpus_per_task == 0
        tpl = load_site(joinpath(SITES_DIR, "_template.toml"))                    # every key documented there loads
        @test tpl.name == "mymachine" && tpl.backend === :slurm && tpl.extra == String[] && tpl.job_name_prefix == ""
        @test_throws r"unknown site \"nowhere\".*desktop, dgx01" load_site("nowhere")
        d = mktempdir()
        write(joinpath(d, "mine.toml"), "[site]\nbackend = \"local\"\n[local]\ncuda_visible_devices = \"1\"\n")
        @test load_site(joinpath(d, "mine.toml")).name == "mine"                  # by path, named by the file
        @test load_site("mine"; dirs = [d]).cuda_visible_devices == "1"           # by name, from a project's sites
        write(joinpath(d, "bad.toml"), "[site]\nbackend = \"pbs\"\n")
        @test_throws r"backend = \"pbs\"" load_site(joinpath(d, "bad.toml"))
        write(joinpath(d, "bad.toml"), "[site]\nbackend = \"local\"\n[slurm]\nqueue = \"x\"\n")
        @test_throws r"unknown key slurm.queue" load_site(joinpath(d, "bad.toml"))
        write(joinpath(d, "bad.toml"), "[site]\nbackend = \"slurm\"\n[slurm]\njob_name_prefix = \"a b\"\n")
        @test_throws r"job_name_prefix = \"a b\": letters, digits" load_site(joinpath(d, "bad.toml"))
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
        p = plan_reproduction(g, load_site("dgx01"); env = env0, queue_states = noqueue)
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

    @testset "the local backend inherits no Slurm job: launched from inside one, a stage sees only its own variables" begin
        f = runner_project()
        mkpath(joinpath(f.data, "boards")); write(joinpath(f.data, "boards", "a4.toml"), "")
        mkpath(joinpath(f.data, "photos", "sheet_small")); mkpath(joinpath(f.data, "photos", "sheet_large"))
        job = Dict("SLURM_ARRAY_TASK_ID" => "4", "SLURM_SUBMIT_DIR" => "/elsewhere/submit", "SLURM_JOB_ID" => "7553",
                   "SLURM_ARRAY_JOB_ID" => "7553", "SLURM_CPUS_PER_TASK" => "96", "SLURM_JOB_NAME" => "vgtk-suites")
        code, out = withenv(job...) do
            run_main(["--graph=$(f.graph)", "--site=desktop", "--only=texture"]; env = merge(env0, job))
        end
        @test code == 0
        ran = readlines(joinpath(f.results, "ran.txt"))
        @test startswith(ran[1], "setup.sbatch task= ")                                     # no job's task id
        @test occursin("prepare.sbatch task= cases= sets= exp=chrome gpu= ", ran[2])
        @test !any(l -> occursin("/elsewhere/submit", l) || occursin("task=4", l), ran)
        @test occursin("submit=$(normpath(f.proj * "/"))", ran[2])                         # the project's, never the job's
        @test occursin("chrome.sbatch task=2 cases=alpha,mid,zeta", ran[5])                  # an array's own task ids
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
        mk = stage_markers(m["stages"]["chrome"]["markers"])                     # each task's completion marker, too
        @test sort(collect(keys(mk))) == [0, 1, 2] && all(d -> d["state"] == "completed" && d["exit_status"] == 0 && d["commit"] == m["commit"], values(mk))
        @test m["stages"]["chrome"]["markers"] == stage_markers_dir(only(ms), "chrome")
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

    @testset "the Slurm backend: through the marker wrapper, chained with afterok; reuse and status from the markers, never sacct" begin
        f = runner_project()
        mkpath(joinpath(f.data, "boards")); write(joinpath(f.data, "boards", "a4.toml"), "")
        mkpath(joinpath(f.data, "photos", "sheet_small")); mkpath(joinpath(f.data, "photos", "sheet_large"))
        bin = fake_bin()
        log, queue = joinpath(mktempdir(), "sbatch.log"), joinpath(mktempdir(), "squeue.txt")
        write(queue, "")
        env = merge(env0, Dict("PATH" => bin * ":" * ENV["PATH"], "FAKE_SBATCH_LOG" => log, "FAKE_SQUEUE" => queue,
                               "VGTK_DATA_ROOT" => f.data, "VGTK_RESULTS_ROOT" => f.results))
        launch(args) = run_main(args; env, queue_states = nothing, job_states = nothing)    # the stand-ins on PATH
        status(args...; job_states = nothing) =
            run_main(["status", "--graph=$(f.graph)", "--site=dgx01", args...]; env, queue_states = nothing, job_states)
        code, out = launch(["--graph=$(f.graph)", "--site=dgx01", "--only=texture"])
        code == 0 || @info out
        @test code == 0
        m1 = only(run_manifests(joinpath(f.results, "_reproduce")))
        m = TOML.parsefile(m1)
        @test [m["stages"][s]["job_id"] for s in ("setup", "chrome_prepare", "chrome", "texture")] == ["101", "102", "103", "104"]
        wrapper(s, mf = m1) = joinpath(stage_markers_dir(mf, s), "marker_wrapper.sbatch")
        lines = readlines(log)
        @test length(lines) == 4
        @test startswith(lines[1], "101 --parsable --partition=gpu --gres=gpu:1") && endswith(lines[1], "$(wrapper("setup")) EXP= CASES=")
        @test startswith(lines[2], "102 --parsable --dependency=afterok:101 ") && occursin("EXP=chrome", lines[2])
        @test startswith(lines[3], "103 --parsable --dependency=afterok:102 ") && occursin("--array=0-2 --time=130 $(wrapper("chrome"))", lines[3]) &&
              endswith(lines[3], "CASES=alpha,mid,zeta")
        @test startswith(lines[4], "104 --parsable --dependency=afterok:103 ")
        # the wrapper: the launcher's #SBATCH directives first, then a copy of the launcher as submitted
        @test startswith(read(wrapper("chrome"), String), "#!/bin/bash\n#SBATCH --output=logs/x_chrome_%A_%a.log\n")
        snap = joinpath(stage_markers_dir(m1, "chrome"), "chrome.sbatch")
        @test read(snap, String) == read(joinpath(f.proj, "slurm", "chrome.sbatch"), String)

        # the jobs, run as Slurm runs them (the wrapper under bash in the submit directory, the job's variables set)
        function runjob(s, id; task = nothing, out = devnull)
            e = merge(env, Dict("SLURM_JOB_ID" => task === nothing ? id : string(9000 + task), "SLURM_SUBMIT_DIR" => f.proj))
            task === nothing || merge!(e, Dict("SLURM_ARRAY_JOB_ID" => id, "SLURM_ARRAY_TASK_ID" => string(task)))
            return run(pipeline(Cmd(`bash $(wrapper(s))`; env = e, dir = f.proj, ignorestatus = true); stdout = out, stderr = out)).exitcode
        end
        @test runjob("setup", "101") == 0 && runjob("chrome_prepare", "102") == 0 && runjob("chrome", "103"; task = 0) == 0
        mk = only(values(stage_markers(stage_markers_dir(m1, "setup"))))
        @test mk["state"] == "completed" && mk["exit_status"] == 0 && mk["job_id"] == "101" && mk["commit"] == m["commit"] &&
              mk["key"] == m["stages"]["setup"]["key"] && mk["stage"] == "setup" && !haskey(mk, "signal")
        @test stage_markers(stage_markers_dir(m1, "chrome"))[0]["job_id"] == "103"            # an array's job id, not its task's
        @test filter(startswith("."), readdir(stage_markers_dir(m1, "setup"))) == String[]     # no temporary file left

        # the next launch: setup and the prepare completed (their markers), chrome's tasks 1 and 2 still queued and
        # texture pending → not resubmitted; the pdf stage is new and waits on the in-flight texture job
        write(queue, "103_1|RUNNING|None\n103_2|PENDING|Priority\n104_0|PENDING|Dependency\n104_1|PENDING|Dependency\n")
        code, out = launch(["--graph=$(f.graph)", "--site=dgx01", "--only=pdf"])
        @test code == 0 && occursin(r"setup\s+reused", out) && occursin(r"chrome_prepare\s+reused", out) && occursin(r"chrome\s+103  \(in flight", out)
        lines = readlines(log)
        @test length(lines) == 5 && startswith(lines[5], "105 --parsable --dependency=afterok:104 ")
        m2 = last(run_manifests(joinpath(f.results, "_reproduce")))
        @test occursin("--output=logs/paper_pdf_%j.log $(wrapper("pdf", m2))", lines[5]) && !occursin("--wrap", lines[5])
        @test occursin("\nmake -C paper &\n", read(wrapper("pdf", m2), String))
        @test TOML.parsefile(m2)["stages"]["chrome"]["markers"] == stage_markers_dir(m1, "chrome")   # an in-flight job's own markers

        # chrome task 1 fails (its marker says so); task 2 dies without a marker and leaves the queue (a node failure);
        # texture stays pending; sacct fails, as without accounting
        write(snap, "#!/bin/bash\necho start\necho 'ERROR: LoadError: out of memory'\nexit 3\n")
        mkpath(joinpath(f.proj, "logs"))
        @test runjob("chrome", "103"; task = 1, out = joinpath(f.proj, "logs", "x_chrome_103_1.log")) == 3
        @test stage_markers(stage_markers_dir(m1, "chrome"))[1]["exit_status"] == 3
        write(queue, "104_0|PENDING|Dependency\n104_1|PENDING|Dependency\n105|PENDING|Dependency\n")
        code, out = status("--manifest=$m1")
        @test code == 0
        @test occursin(r"setup\s+101\s+completed \(completed 1\)", out)
        @test occursin(r"chrome\s+103\s+failed \(completed 1, failed 1, unknown 1\)", out)
        @test occursin("logs/x_chrome_103_1.log", out) && occursin("error ERROR: LoadError: out of memory", out)
        @test occursin(r"texture\s+104\s+in flight \(PENDING 2\)", out)
        @test !occursin("sacct", out)                                                    # no accounting: nothing from it
        code, out = status()                                                             # the last launch
        @test code == 0 && occursin(r"setup\s+reused", out) && occursin(r"pdf\s+105\s+in flight \(PENDING 1\)", out)
        # a task that left no marker and the queue is unknown, never finished; where accounting exists its state is
        # shown beside it, as information only
        r = TOML.parsefile(m1)["stages"]["chrome"]
        @test stage_run_state(r, Dict("103" => Dict(2 => "RUNNING"))).state == "failed"
        rm(joinpath(stage_markers_dir(m1, "chrome"), "task-1.toml"))
        @test stage_run_state(r).state == "unknown" && stage_run_state(r, Dict("103" => Dict(1 => "RUNNING", 2 => "PENDING"))).state == "in flight"
        @test stage_run_state(r, Dict("103" => Dict(-1 => "PENDING"))).state == "in flight"     # a range squeue does not split
        acct(ids) = Dict("103" => ["COMPLETED", "NODE_FAIL"])
        code, out = status("--manifest=$m1"; job_states = acct)
        @test occursin(r"chrome\s+103\s+unknown \(completed 1, unknown 2\) \[sacct: COMPLETED, NODE_FAIL\]", out)
        write(snap, read(joinpath(f.proj, "slurm", "chrome.sbatch")))

        # the launch after: chrome never completed → submitted again; texture, pending on a dependency that can never
        # be satisfied, is not depended on but submitted after it; the finished stages are reused from their markers
        # alone (sacct fails)
        write(queue, "104_0|PENDING|DependencyNeverSatisfied\n104_1|PENDING|DependencyNeverSatisfied\n")
        code, out = launch(["--graph=$(f.graph)", "--site=dgx01", "--only=texture"])
        @test code == 0 && occursin(r"setup\s+reused", out) && occursin(r"chrome_prepare\s+reused", out)
        lines = readlines(log)
        @test length(lines) == 7 && startswith(lines[6], "106 --parsable --partition=gpu") && occursin("--array=0-2", lines[6])   # its prerequisites done
        @test startswith(lines[7], "107 --parsable --dependency=afterok:106 ")
        # a texture job genuinely pending behind an in-flight chrome is depended on, not resubmitted
        write(queue, "106_0|RUNNING|None\n106_1|RUNNING|None\n106_2|RUNNING|None\n107_0|PENDING|Dependency\n107_1|PENDING|Dependency\n")
        code, out = launch(["--graph=$(f.graph)", "--site=dgx01", "--only=texture"])
        @test code == 0 && occursin(r"chrome\s+106  \(in flight", out) && occursin(r"texture\s+107  \(in flight", out) && length(readlines(log)) == 7
        # but not behind a stage this launch submits again
        write(queue, "107_0|PENDING|Dependency\n107_1|PENDING|Dependency\n")
        code, out = launch(["--graph=$(f.graph)", "--site=dgx01", "--only=texture"])
        lines = readlines(log)
        @test code == 0 && length(lines) == 9 && startswith(lines[9], "109 --parsable --dependency=afterok:108 ")
    end

    @testset "the job-name prefix: every Slurm job named <prefix><name>, no run key changed" begin
        f = runner_project()
        write(joinpath(f.proj, "slurm", "chrome.sbatch"), "#!/bin/bash\n#SBATCH --job-name=paper-chrome\n#SBATCH --output=logs/x_chrome_%A_%a.log\nset -eu\n")
        g = load_stage_graph(f.graph)
        stamp = "20261111_000000_000"                       # one launch's (the results stage's command names its manifest)
        plain = plan_reproduction(g, load_site("dgx01"); env = env0, queue_states = noqueue, stamp)
        marked = plan_reproduction(g, load_site("dgx01"); env = env0, queue_states = noqueue, stamp, job_name_prefix = "leg-gate-")
        @test [st.key for st in plain.stages] == [st.key for st in marked.stages]
        name(p, s) = only(filter(startswith("--job-name="), p.stages[findfirst(st -> st.spec.name == s, p.stages)].command))
        @test name(marked, "setup") == "--job-name=leg-gate-paper-setup"                  # no name anywhere: <project>-<stage>
        @test name(marked, "chrome_prepare") == "--job-name=leg-gate-paper-prepare-chrome" # the stage's job_name
        @test name(marked, "chrome") == "--job-name=leg-gate-paper-chrome"                # the launcher's own #SBATCH --job-name
        @test name(marked, "pdf") == "--job-name=leg-gate-paper-pdf"
        @test count(a -> startswith(a, "--job-name="), plain.stages[1].command) == 0      # no prefix: the names as they were
        # from the site profile, and --job-name-prefix over it; the dry run shows it
        d = mktempdir(); write(joinpath(d, "rel.toml"), "[site]\nbackend = \"slurm\"\n[slurm]\npartition = \"gpu\"\njob_name_prefix = \"release-\"\n")
        site = load_site(joinpath(d, "rel.toml"))
        @test site.job_name_prefix == "release-"
        @test name(plan_reproduction(g, site; env = env0, queue_states = noqueue), "setup") == "--job-name=release-paper-setup"
        code, out = run_main(["--graph=$(f.graph)", "--site=$(joinpath(d, "rel.toml"))", "--dry-run", "--job-name-prefix=leg-gate-",
                              "--results-root=$(f.results)"]; env = env0)
        @test code == 0 && occursin("--job-name=leg-gate-paper-chrome", out) && !occursin("release-", out)
        @test_throws r"letters, digits" plan_reproduction(g, site; env = env0, queue_states = noqueue, job_name_prefix = "a;b")
    end

    @testset "completion markers are written atomically" begin
        f = runner_project()
        d = mktempdir(); p = joinpath(d, "task-0.toml")
        Reproduce._write_atomic(io -> print(io, "state = \"completed\"\n"), p)
        @test TOML.parsefile(p)["state"] == "completed" && readdir(d) == ["task-0.toml"]
        # a writer that fails part-way leaves the marker that was there, and no temporary file
        @test_throws ErrorException Reproduce._write_atomic(io -> (print(io, "state = \"fai"); flush(io); error("disk full")), p)
        @test read(p, String) == "state = \"completed\"\n" && readdir(d) == ["task-0.toml"]
        # the job's wrapper: written to a temporary file in the marker directory, then renamed into place (a stand-in
        # mv records its arguments); a TERM (scancel, the time limit) reaches the stage and is recorded as failed
        g = load_stage_graph(f.graph)
        plan = plan_reproduction(g, load_site("dgx01"); env = env0, queue_states = noqueue, only = ["setup"], results_root = f.results)
        dir = stage_markers_dir(plan.manifest, "setup")
        w = Reproduce._marker_wrapper(plan, plan.stages[1], dir)
        bin = mktempdir(); mvlog = joinpath(bin, "mv.log")
        write(joinpath(bin, "mv"), "#!/bin/bash\necho \"\$@\" >> $mvlog\nexec /bin/mv \"\$@\"\n"); chmod(joinpath(bin, "mv"), 0o755)
        e = merge(env0, Dict("PATH" => bin * ":" * ENV["PATH"], "VGTK_RESULTS_ROOT" => f.results, "SLURM_JOB_ID" => "7"))
        @test run(Cmd(`bash $w`; env = e, dir = f.proj, ignorestatus = true)).exitcode == 0
        src, dst = split(only(readlines(mvlog)))[end-1:end]
        @test dirname(src) == dir && startswith(basename(src), ".task-0.") && dst == joinpath(dir, "task-0.toml")
        @test stage_markers(dir; job_id = "7")[0]["state"] == "completed" && isempty(stage_markers(dir; job_id = "8"))
        write(joinpath(dir, "setup.sbatch"), "#!/bin/bash\ntouch \"$dir/started\"\nexec sleep 60\n")
        proc = run(pipeline(Cmd(`bash $w`; env = e, dir = f.proj, ignorestatus = true); stdout = devnull, stderr = devnull); wait = false)
        t0 = time()
        while !isfile(joinpath(dir, "started")) && time() - t0 < 30
            sleep(0.1)
        end
        kill(proc, Base.SIGTERM); wait(proc)
        mk = stage_markers(dir)[0]
        @test proc.exitcode == 143 && mk["state"] == "failed" && mk["exit_status"] == 143 && mk["signal"] == "TERM"
    end

    @testset "the Slurm backend's hazards: a requeued task, a failing squeue, the launcher's own SLURM_*, unnamed jobs, --signal, noexec" begin
        f = runner_project()
        mkpath(joinpath(f.data, "boards")); write(joinpath(f.data, "boards", "a4.toml"), "")
        bin = fake_bin()
        log, queue = joinpath(mktempdir(), "sbatch.log"), joinpath(mktempdir(), "squeue.txt")
        write(queue, "")
        # a stand-in sbatch, before the fixture's on PATH, that also records the SLURM_* variables it was run with
        spy = mktempdir(); envlog = joinpath(spy, "env.log")
        write(joinpath(spy, "sbatch"), "#!/bin/bash\nenv | grep '^SLURM_' | sort | tr '\\n' ' ' >> $envlog; echo >> $envlog\nexec $(joinpath(bin, "sbatch")) \"\$@\"\n")
        chmod(joinpath(spy, "sbatch"), 0o755)
        # launched from inside an array task of another job (a test suite under sbatch)
        parent = Dict("SLURM_ARRAY_TASK_ID" => "5", "SLURM_ARRAY_JOB_ID" => "99", "SLURM_JOB_ID" => "99", "SLURM_CONF" => "/etc/slurm/x.conf")
        env = merge(env0, parent, Dict("PATH" => spy * ":" * bin * ":" * ENV["PATH"], "FAKE_SBATCH_LOG" => log, "FAKE_SQUEUE" => queue,
                                       "VGTK_DATA_ROOT" => f.data, "VGTK_RESULTS_ROOT" => f.results))
        launch(args; env = env) = run_main(args; env, queue_states = nothing, job_states = nothing)
        code, out = launch(["--graph=$(f.graph)", "--site=dgx01", "--only=chrome_prepare"])
        @test code == 0
        m1 = only(run_manifests(joinpath(f.results, "_reproduce")))
        # sbatch sees none of the launching job's SLURM_* (SLURM_CONF, the cluster's configuration, kept)
        @test readlines(envlog) == ["SLURM_CONF=/etc/slurm/x.conf ", "SLURM_CONF=/etc/slurm/x.conf "]
        # a job no stage and no launcher names is named <project>-<stage>, never after the wrapper; a named one keeps its name
        lines = readlines(log)
        @test occursin(" --job-name=paper-setup ", lines[1])
        @test count("--job-name=", lines[2]) == 1 && occursin(" --job-name=paper-prepare-chrome ", lines[2])
        @test TOML.parsefile(m1)["stages"]["setup"]["key"] ==
              plan_reproduction(load_stage_graph(f.graph), load_site("dgx01"); env, queue_states = noqueue, only = ["setup"],
                                stamp = "x").stages[1].key                                         # the run key is the planned command's

        # a task preempted and requeued under the same job id: its first attempt's TERM left a "failed" marker, and the
        # queue holds it again → in flight, depended on, not submitted a second time
        sdir = stage_markers_dir(m1, "setup")
        mkpath(sdir); write(joinpath(sdir, "task-0.toml"), "job_id = \"101\"\nstate = \"failed\"\nexit_status = 143\nsignal = \"TERM\"\n")
        r = TOML.parsefile(m1)["stages"]["setup"]
        @test stage_run_state(r).state == "failed"
        @test stage_run_state(r, Dict("101" => Dict(0 => "PENDING"))).state == "in flight"
        @test stage_run_state(r, Dict("101" => Dict(0 => "RUNNING"))).tasks == Dict(0 => "RUNNING")
        @test stage_run_state(r, Dict("101" => Dict(0 => "COMPLETING"))).state == "failed"        # finishing: its marker
        write(queue, "101|PENDING|BeginTime\n102|PENDING|Dependency\n")
        code, out = launch(["--graph=$(f.graph)", "--site=dgx01", "--only=chrome_prepare"])
        @test code == 0 && occursin(r"setup\s+101  \(in flight", out) && length(readlines(log)) == 2

        # squeue failing for any other reason than an unknown job aborts the launch: nothing is submitted
        @test slurm_queue_states(["101"]; env) == Dict("101" => Dict(0 => "PENDING"))
        @test slurm_queue_states(["555"]; env) == Dict{String, Dict{Int, String}}()                  # "Invalid job id specified": gone
        down = mktempdir()
        write(joinpath(down, "squeue"), "#!/bin/bash\necho 'slurm_load_jobs error: Socket timed out on send/recv operation' >&2\nexit 1\n")
        chmod(joinpath(down, "squeue"), 0o755)
        envdown = merge(env, Dict("PATH" => down * ":" * env["PATH"]))
        @test_throws r"squeue -j 101 failed.*Socket timed out" slurm_queue_states(["101"]; env = envdown)
        code, out = launch(["--graph=$(f.graph)", "--site=dgx01", "--only=chrome_prepare"]; env = envdown)
        @test code == 4 && occursin("Socket timed out", out) && length(readlines(log)) == 2
        code, out = run_main(["status", "--graph=$(f.graph)", "--site=dgx01"]; env = envdown, queue_states = nothing)
        @test code == 4 && occursin("squeue -j", out)

        # the wrapper: the attempt is recorded (SLURM_RESTART_COUNT); the launcher's copy is run through its
        # interpreter, so a copy that cannot be executed (a noexec results root) still runs
        g = load_stage_graph(f.graph)
        plan = plan_reproduction(g, load_site("dgx01"); env = env0, queue_states = noqueue, only = ["setup"], results_root = f.results)
        dir = stage_markers_dir(plan.manifest, "setup")
        w = Reproduce._marker_wrapper(plan, plan.stages[1], dir)
        snap = joinpath(dir, "setup.sbatch")
        @test !Sys.isexecutable(snap)
        chmod(snap, 0o644)
        e = merge(env0, Dict("VGTK_RESULTS_ROOT" => f.results, "SLURM_JOB_ID" => "7", "SLURM_RESTART_COUNT" => "1"))
        @test run(Cmd(`bash $w`; env = e, dir = f.proj, ignorestatus = true)).exitcode == 0
        mk = stage_markers(dir)[0]
        @test mk["state"] == "completed" && mk["restart_count"] == 1
        @test Reproduce._interpreted("/r/x.sbatch", "#!/usr/bin/env bash\necho\n") == "/usr/bin/env bash /r/x.sbatch"
        @test Reproduce._interpreted("/r/x.sbatch", "echo\n") == "/bin/bash /r/x.sbatch"

        # a launcher's --signal directive now signals the wrapper: it is forwarded to the launcher, which handles it,
        # and recorded in the marker
        write(joinpath(f.proj, "slurm", "setup.sbatch"), """
            #!/bin/bash
            #SBATCH --signal=B:USR1@60
            trap 'touch "$dir/got_usr1"; exit 0' USR1
            touch "$dir/started"
            while true; do sleep 0.1; done
            """)
        @test Reproduce._directive_signals(["#SBATCH --signal=B:USR1@60", "#SBATCH --signal SIGUSR2", "#SBATCH --signal=TERM@30"]) == ["USR1", "USR2"]
        rm(joinpath(dir, "task-0.toml"))
        w = Reproduce._marker_wrapper(plan, plan.stages[1], dir)
        proc = run(pipeline(Cmd(`bash $w`; env = e, dir = f.proj, ignorestatus = true); stdout = devnull, stderr = devnull); wait = false)
        t0 = time()
        while !isfile(joinpath(dir, "started")) && time() - t0 < 30
            sleep(0.1)
        end
        kill(proc, Base.SIGUSR1); wait(proc)
        @test proc.exitcode == 0 && isfile(joinpath(dir, "got_usr1"))
        mk = get(stage_markers(dir), 0, Dict{String, Any}())
        @test get(mk, "state", "") == "completed" && get(mk, "signal", "") == "USR1"
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
        p = plan_reproduction(g, load_site("desktop"); env = env0, queue_states = noqueue, only = ["setup"])
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
