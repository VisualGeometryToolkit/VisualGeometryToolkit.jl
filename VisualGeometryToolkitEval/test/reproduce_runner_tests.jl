# The runner (src/reproduce/runner.jl, src/reproduce/run_results_branch.jl): a project's stage graph of record on a
# site. The site profiles shipped (dgx01, desktop, the template) load; the graph is validated and ordered; --only
# selects a subset with its prerequisites (or alone); array stages take their items from the config's cases or a
# list, sized and timed from it; the inputs are checked first and reported as data; the dry run prints the job
# graph for a Slurm and a local site; the local backend runs a project's sbatch launchers under bash, records a
# manifest and reuses a finished stage by its run key, which folds in its upstream stages' keys (an upstream stage
# keyed anew reruns every stage after it); the Slurm backend submits each stage through its marker
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
            precompile = false      # its launcher runs no julia

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
            precompile = false      # its launcher runs no julia
            array = { from = "cases", select_env = "CASES", max_time = "pipeline.lm.max_time", time_margin_s = 600 }

            [[stage]]
            name = "texture"
            after = ["chrome"]
            sbatch = "slurm/texture.sbatch"
            config = "configs/experiments/texture_{item}.toml"
            gpu = true
            precompile = false      # its launcher runs no julia
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
        @test dgx.backend === :slurm && dgx.partition == "gpu" && dgx.gres == "gpu:1" && dgx.cpus_per_task == 2 && dgx.mem_per_cpu == "8480M"
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
        @test_throws r"unknown key qos" load_stage_graph(bad("[[stage]]\nname = \"a\"\ncommand = [\"true\"]\nqos = \"high\"\n"))
        @test_throws r"needs \[results_branch\] assets" load_stage_graph(bad("[[stage]]\nname = \"a\"\nresults_branch = true\n"))
    end

    @testset "array items: the config's cases (sorted, selectable) or a list; --time from the cases" begin
        f = runner_project()
        g = load_stage_graph(f.graph)
        chrome, texture = g.stages[3], g.stages[4]
        @test stage_items(g, chrome; env = env0) == ["alpha", "mid", "zeta"]
        @test stage_items(g, chrome; env = merge(env0, Dict("CASES" => "zeta,alpha"))) == ["alpha", "zeta"]
        @test_throws r"CASES names unknown value\(s\) omega" stage_items(g, chrome; env = merge(env0, Dict("CASES" => "omega")))
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

    @testset "an input's sha256: verified when given (another content refused, with its id, expected and actual); absent, unchecked" begin
        f = runner_project()
        g = load_stage_graph(f.graph)
        board = joinpath(f.data, "boards", "a4.toml")
        mkpath(dirname(board)); write(board, "board of record\n")
        actual = first(split(read(`sha256sum $board`, String)))
        @test VisualGeometryToolkitEval.file_sha256(board) == actual
        @test VisualGeometryToolkitEval.file_sha256(board) == actual                             # the cached hash: the same
        cfg = joinpath(f.proj, "configs", "experiments", "chrome.toml")
        text = read(cfg, String)
        declare(h) = write(cfg, replace(text, "path = \"boards/a4.toml\"" => "path = \"boards/a4.toml\"\nsha256 = \"$h\""))
        # absent: unchecked, as before
        @test isempty(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0))
        # the content of record: accepted
        declare(actual)
        @test isempty(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0))
        declare(uppercase(actual))                                                                # hex in either case
        @test isempty(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0))
        # another content: refused, naming the input, both hashes; the launch runs nothing (exit 2), the dry run neither
        declare(actual)
        write(board, "a board edited by hand\n")
        other = first(split(read(`sha256sum $board`, String)))
        @test_throws r"input board of configs/experiments/chrome.toml \(boards/a4.toml, used by chrome_prepare.*sha256 expected [0-9a-f]{64}, actual [0-9a-f]{64}" missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0)
        e = try missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0); nothing catch err; err end
        @test e isa ArgumentError && occursin("expected $actual, actual $other", e.msg)
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=chrome_prepare"]; env = env0)
        @test code == 2 && occursin("refused: an input is not the data of record", out) && !isfile(joinpath(f.results, "ran.txt"))
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=chrome_prepare", "--dry-run"]; env = env0)
        @test code == 2 && occursin("expected $actual", out)
        # a directory's sha256 is its tree digest (tree_sha256): the sorted (path, file sha256) list, globs excluded
        photos = joinpath(f.data, "photos", "chrome")
        mkpath(joinpath(photos, "sub")); write(joinpath(photos, "a.ARW"), "a"); write(joinpath(photos, "sub", "b.ARW"), "b")
        write(joinpath(photos, "index.log"), "rebuilt by the reader")
        shell(ex) = first(split(read(Cmd(["bash", "-c", "cd '$photos' && find . -type f | sed 's|^./||' | grep -v -E '$ex' | LC_ALL=C sort | " *
                                          "while read f; do printf '%s\\t%s\\n' \"\$f\" \"\$(sha256sum \"\$f\" | cut -c1-64)\"; done | sha256sum"]), String)))
        full_digest, kept = VisualGeometryToolkitEval.tree_sha256(photos), VisualGeometryToolkitEval.tree_sha256(photos; exclude = ["*.log"])
        @test full_digest == shell("^\$") && kept == shell("\\.log\$") && full_digest != kept
        @test VisualGeometryToolkitEval.tree_sha256(photos; exclude = ["**/b.ARW", "*.log"]) == shell("b\\.ARW\$|\\.log\$")
        @test VisualGeometryToolkitEval.tree_sha256(photos; exclude = ["**/index.log"]) == kept                 # `**/` includes no directory
        tabbed = mktempdir(); write(joinpath(tabbed, "a\tb.txt"), "x")
        @test_throws r"holds a tab or a newline" VisualGeometryToolkitEval.tree_sha256(tabbed)
        dirdecl(h; excl = "") = write(cfg, replace(text, "path = \"photos/chrome\"" => "path = \"photos/chrome\"\nsha256 = \"$h\"$excl"))
        dirdecl(kept; excl = "\nsha256_exclude = [\"*.log\"]")
        @test isempty(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0))
        write(joinpath(photos, "index.log"), "rebuilt again")                                       # excluded: no change
        @test isempty(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0))
        write(joinpath(photos, "sub", "b.ARW"), "b, edited")
        @test_throws r"input photos .*sha256 expected [0-9a-f]{64}, actual" missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0)
        dirdecl(full_digest)                                                                         # the log counts without the glob
        @test_throws r"input photos .*sha256 expected" missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0)
        write(cfg, replace(text, "path = \"boards/a4.toml\"" => "path = \"boards/a4.toml\"\nsha256 = \"$actual\"\nsha256_exclude = [\"x\"]"))
        write(board, "board of record\n")
        @test_throws r"sha256_exclude is for a directory" missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0)
        declare("abc")
        @test_throws r"is no SHA-256" missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0)
        # a missing input with a sha256 is reported missing, as any
        declare(actual); rm(board)
        @test [r.path for r in missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0)] == ["boards/a4.toml"]
    end

    @testset "per-tier input roots: the environment, then the site's [sites.<site>.tiers], then the data root" begin
        f = runner_project()
        alt = mktempdir(); mkpath(joinpath(alt, "boards")); write(joinpath(alt, "boards", "a4.toml"), "board\n")
        g = load_stage_graph(f.graph)
        paths(rows) = [r.path for r in rows]
        # the fallback: the one data root, which lacks the board
        @test paths(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0, site = "desktop")) == ["boards/a4.toml"]
        # the environment's VGTK_DATA_ROOT_<TIER>
        @test VisualGeometryToolkitEval.tier_variable("boards") == "VGTK_DATA_ROOT_BOARDS"
        @test VisualGeometryToolkitEval.tier_variable("blobcalib_202609") == "VGTK_DATA_ROOT_BLOBCALIB_202609"
        envb = merge(env0, Dict("VGTK_DATA_ROOT_BOARDS" => alt))
        @test isempty(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = envb, site = "desktop"))
        # the site's table (a relative root is relative to the workdir); the environment comes first
        text = read(f.graph, String)
        write(f.graph, replace(text, "[sites.desktop]\n" => "[sites.desktop]\ntiers = { boards = \"$alt\" }\n"))
        g = load_stage_graph(f.graph)
        @test isempty(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0, site = "desktop"))
        @test paths(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0, site = "dgx01")) == ["boards/a4.toml"]   # another site's
        empty_root = mktempdir()
        @test paths(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, site = "desktop",
                                   env = merge(env0, Dict("VGTK_DATA_ROOT_BOARDS" => empty_root)))) == ["boards/a4.toml"]
        mkpath(joinpath(f.proj, "altrel", "boards")); write(joinpath(f.proj, "altrel", "boards", "a4.toml"), "board\n")
        write(f.graph, replace(text, "[sites.desktop]\n" => "[sites.desktop]\ntiers = { boards = \"altrel\" }\n"))
        g = load_stage_graph(f.graph)
        @test isempty(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0, site = "desktop"))
        # a named tier root that is not there is reported as such
        rows = missing_inputs(g, ["chrome_prepare"]; data_root = f.data, site = "desktop",
                              env = merge(env0, Dict("VGTK_DATA_ROOT_BOARDS" => "/nonexistent/boards-root")))
        @test only(rows).tier == "boards (root /nonexistent/boards-root, from VGTK_DATA_ROOT_BOARDS, is not there)"
        write(f.graph, replace(text, "[sites.desktop]\n" => "[sites.desktop]\ntiers = { boards = \"/nonexistent/b\" }\n"))
        g = load_stage_graph(f.graph)
        @test occursin("from [sites.desktop.tiers], is not there", only(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0, site = "desktop")).tier)
        # the sha256 of an input under its tier's root: another content refused
        write(f.graph, replace(text, "[sites.desktop]\n" => "[sites.desktop]\ntiers = { boards = \"$alt\" }\n"))
        g = load_stage_graph(f.graph)
        cfg = joinpath(f.proj, "configs", "experiments", "chrome.toml")
        write(cfg, replace(read(cfg, String), "path = \"boards/a4.toml\"" => "path = \"boards/a4.toml\"\nsha256 = \"$("0"^64)\""))
        @test_throws r"input board .*sha256 expected 0{64}, actual" missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0, site = "desktop")
        # the jobs resolve as the launch did: the site's tier roots and the site in their environment
        plan = plan_reproduction(g, load_site("desktop"); env = env0, queue_states = noqueue, only = ["setup"])
        @test plan.base_env["VGTK_DATA_ROOT_BOARDS"] == alt && plan.base_env["VGTK_SITE"] == "desktop"
        plan = plan_reproduction(g, load_site("desktop"); env = merge(env0, Dict("VGTK_DATA_ROOT_BOARDS" => empty_root)), queue_states = noqueue, only = ["setup"])
        @test plan.base_env["VGTK_DATA_ROOT_BOARDS"] == empty_root                                    # the caller's first
        @test_throws r"unknown key sites.desktop.tier " load_stage_graph(begin
            write(f.graph, replace(text, "[sites.desktop]\n" => "[sites.desktop]\ntier = 1\n")); f.graph end)
    end

    @testset "per-stage resources: a stage's cpus_per_task, mem_per_cpu and gres over the site's" begin
        f = runner_project()
        text = read(f.graph, String)
        write(f.graph, replace(text, "name = \"setup\"\n" => "name = \"setup\"\ncpus_per_task = 6\nmem_per_cpu = \"4G\"\ngres = \"gpu:4\"\n";
                               count = 1) |> t -> replace(t, "name = \"pdf\"\n" => "name = \"pdf\"\ngres = \"gpu:1\"\n"))
        g = load_stage_graph(f.graph)
        plan = plan_reproduction(g, load_site("dgx01"); env = env0, queue_states = noqueue)
        cmd(n) = plan.stages[findfirst(st -> st.spec.name == n, plan.stages)].command
        @test occursin("--partition=gpu --gres=gpu:4 --cpus-per-task=6 --mem-per-cpu=4G", join(cmd("setup"), " "))
        @test occursin("--gres=gpu:1 --cpus-per-task=2 --mem-per-cpu=8480M", join(cmd("pdf"), " "))      # gres: a GPU stage
        @test occursin("--partition=gpu --cpus-per-task=2 --mem-per-cpu=8480M", join(cmd("chrome_prepare"), " "))   # the site's
        local_plan = plan_reproduction(g, load_site("desktop"); env = env0, queue_states = noqueue, only = ["setup"])
        @test local_plan.stages[1].env["SLURM_CPUS_PER_TASK"] == "6" && local_plan.stages[1].env["CUDA_VISIBLE_DEVICES"] == "0"
        write(f.graph, replace(text, "name = \"setup\"\n" => "name = \"setup\"\ncpus_per_task = \"many\"\n"; count = 1))
        @test_throws r"cpus_per_task = \"many\" is not a count" load_stage_graph(f.graph)
    end

    @testset "a stage's key holds its device, never its CPU count" begin
        f = runner_project()
        d = mktempdir()
        key(g, site, name) = (p = plan_reproduction(g, load_site(site; dirs = [d]); env = env0, queue_states = noqueue, only = [name]);
                              p.stages[findfirst(st -> st.spec.name == name, p.stages)].key)
        for c in (2, 8)
            write(joinpath(d, "slurm$c.toml"), "[site]\nname = \"s\"\nbackend = \"slurm\"\n[slurm]\npartition = \"gpu\"\ngres = \"gpu:1\"\ncpus_per_task = $c\n")
            write(joinpath(d, "local$c.toml"), "[site]\nname = \"l\"\nbackend = \"local\"\n[local]\ncuda_visible_devices = \"0\"\ncpus_per_task = $c\n")
        end
        g = load_stage_graph(f.graph)
        for st in ("setup", "chrome")                                              # a GPU stage, a GPU array
            @test key(g, "slurm2", st) == key(g, "slurm8", st) && key(g, "local2", st) == key(g, "local8", st)
        end
        text = read(f.graph, String)                                               # setup on the CPU: another key
        write(f.graph, replace(text, "sbatch = \"slurm/setup.sbatch\"\ngpu = true\n" => "sbatch = \"slurm/setup.sbatch\"\n"; count = 1))
        cpu = load_stage_graph(f.graph)
        @test !cpu.stages[1].gpu
        @test key(cpu, "slurm2", "setup") != key(g, "slurm2", "setup") && key(cpu, "local2", "setup") != key(g, "local2", "setup")
        # a stage's `nice`: its Slurm priority, a label for the scheduler like the CPU count, keying no run
        write(f.graph, replace(text, "name = \"chrome\"\n" => "name = \"chrome\"\nnice = 100\n"; count = 1))
        niced = load_stage_graph(f.graph)
        p = plan_reproduction(niced, load_site("slurm2"; dirs = [d]); env = env0, queue_states = noqueue, only = ["chrome"])
        @test any(==("--nice=100"), p.stages[end].command) && !any(a -> startswith(a, "--nice"), p.stages[end - 1].command)
        @test key(niced, "slurm2", "chrome") == key(g, "slurm2", "chrome") && key(niced, "local2", "chrome") == key(g, "local2", "chrome")
        write(f.graph, replace(text, "name = \"chrome\"\n" => "name = \"chrome\"\nnice = -5\n"; count = 1))
        @test_throws r"nice = -5 is not a count" load_stage_graph(f.graph)
    end

    @testset "a GPU stage on Slurm compiles first, in a CPU-only job its GPU job runs after" begin
        f = runner_project()
        text = read(f.graph, String)
        write(f.graph, replace(text, "gpu = true\nprecompile = false      # its launcher runs no julia\n" =>
                                     "gpu = true\njulia_flags = [\"--depwarn=error\"]\n"; count = 1))     # setup compiles
        g = load_stage_graph(f.graph)
        @test g.stages[1].precompile && g.stages[1].julia_flags == ["--depwarn=error"] && !g.stages[3].precompile
        p = plan_reproduction(g, load_site("dgx01"); env = env0, queue_states = noqueue, only = ["chrome_prepare"])
        pre = join(p.stages[1].precompile, " ")
        @test startswith(pre, "sbatch --parsable --partition=gpu --cpus-per-task=2 --mem-per-cpu=8480M --job-name=paper-setup-precompile")
        @test !occursin("--gres", pre) && occursin("--depwarn=error -e", pre) && occursin("Pkg.precompile()", pre)
        @test isempty(p.stages[2].precompile)                                       # a CPU stage: none
        @test isempty(plan_reproduction(g, load_site("desktop"); env = env0, queue_states = noqueue, only = ["setup"]).stages[1].precompile)
        # not keyed: the same key with or without it
        k(gr) = plan_reproduction(gr, load_site("dgx01"); env = env0, queue_states = noqueue, only = ["setup"]).stages[1].key
        kon = k(g)
        write(f.graph, text)
        @test k(load_stage_graph(f.graph)) == kon
        write(f.graph, replace(text, "gpu = true\nprecompile = false      # its launcher runs no julia\n" =>
                                     "gpu = true\njulia_flags = [\"--depwarn=error\"]\n"; count = 1))
        # the dry run shows it, and the GPU job after it
        code, out = run_main(["--graph=$(f.graph)", "--site=dgx01", "--only=setup", "--dry-run", "--results-root=$(f.results)"]; env = env0)
        @test code == 0 && occursin("compile  sbatch --parsable --partition=gpu --cpus-per-task=2", out)
        @test occursin("sbatch --parsable --dependency=afterok:\$setup_precompile --partition=gpu --gres=gpu:1", out)
        # submitted: the compile first, the GPU job after it (afterok), its id recorded
        bin = fake_bin(); sb = joinpath(mktempdir(), "sbatch.log")
        env = merge(env0, Dict("PATH" => bin * ":" * ENV["PATH"], "FAKE_SBATCH_LOG" => sb, "FAKE_SQUEUE" => "/dev/null"))
        run(pipeline(`git -C $(f.repo) -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -qam precompile`; stdout = devnull))
        code, out = run_main(["--graph=$(f.graph)", "--site=dgx01", "--only=setup", "--results-root=$(f.results)", "--data-root=$(f.data)"]; env)
        @test code == 0 && occursin(r"setup\s+101  \(precompile, CPU only\)", out)
        l = readlines(sb)
        @test length(l) == 2 && startswith(l[1], "101 ") && !occursin("--gres", l[1]) && occursin("Pkg.precompile()", l[1])
        @test startswith(l[2], "102 --parsable --dependency=afterok:101 ") && occursin("--gres=gpu:1", l[2])
        m = TOML.parsefile(last(run_manifests(joinpath(f.results, "_reproduce"))))
        @test m["stages"]["setup"]["precompile_job_id"] == "101" && m["stages"]["setup"]["job_id"] == "102"
    end

    @testset "the runner bootstrap a project carries is generated from Eval's one template" begin
        @test isfile(Reproduce.REPRODUCE_SCRIPT_TEMPLATE)
        t = reproduce_script(; julia_project = "cvpr2027")
        @test occursin("--project=cvpr2027 -e", t) && occursin("-- --graph=cvpr2027/configs/reproduce.toml \"\$@\"", t)
        @test occursin("bash cvpr2027/scripts/reproduce/reproduce.sh --site=", t) && !occursin("@@", t)
        @test_throws r"not a path relative" reproduce_script(; julia_project = "/abs")
        @test_throws r"not a path relative" reproduce_script(; julia_project = "a b")
        # written into a repository and run from inside it: the runner's command line, from the repository root
        f = runner_project()
        p = write_reproduce_script(f.repo; julia_project = "proj")
        @test p == joinpath(f.repo, "proj", "scripts", "reproduce", "reproduce.sh") && Sys.isexecutable(p)
        @test read(p, String) == reproduce_script(; julia_project = "proj")
        @test success(`bash -n $p`)
        stub = joinpath(mktempdir(), "julia")
        write(stub, "#!/bin/bash\necho \"cwd=\$(pwd)\"\nfor a in \"\$@\"; do case \$a in --*|status) echo \"arg \$a\";; esac; done\n"); chmod(stub, 0o755)
        out = read(Cmd(`bash $p --site=desktop --dry-run`; env = merge(env0, Dict("JULIA" => stub)), dir = joinpath(f.proj, "slurm")), String)
        @test occursin("cwd=$(realpath(f.repo))", out) && occursin("arg --project=proj", out)
        @test occursin("arg --graph=proj/configs/reproduce.toml", out) && occursin("arg --site=desktop", out) && occursin("arg --dry-run", out)
    end

    @testset "unpublished inputs, a scratch input refused, distributed stages and validated resources" begin
        f = runner_project()
        cfg = joinpath(f.proj, "configs", "experiments", "chrome.toml")
        mkpath(joinpath(f.data, "boards")); write(joinpath(f.data, "boards", "a4.toml"), "")
        write(cfg, read(cfg, String) * "\n[inputs.capture]\ntier = \"unpublished\"\npath = \"captures/2026-09\"\n")
        g = load_stage_graph(f.graph)
        rows = missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0, site = "desktop")
        @test only(rows).path == "captures/2026-09" && only(rows).tier == "unpublished (never fetched: provide it on this machine)"
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=chrome_prepare", "--dry-run"]; env = env0)
        @test code == 0 && occursin("unpublished inputs: 1 (tier unpublished: never fetched", out) && occursin("ABSENT \tcaptures/2026-09", out)
        mkpath(joinpath(f.data, "captures", "2026-09"))
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=chrome_prepare", "--dry-run"]; env = env0)
        @test code == 0 && occursin("present\tcaptures/2026-09\tchrome_prepare (configs/experiments/chrome.toml)", out)
        @test Reproduce.UNPUBLISHED_TIER == "unpublished"
        # a scratch launch keys its stages apart from the runs of record
        k(e) = plan_reproduction(g, load_site("desktop"); env = e, queue_states = noqueue, only = ["setup"]).stages[1].key
        @test k(env0) != k(merge(env0, Dict("VGTK_SCRATCH" => "1"))) && k(env0) == k(merge(env0, Dict("VGTK_SCRATCH" => "0")))
        # a scratch run's output is no input of record
        write(joinpath(f.data, "captures", "2026-09", "_provenance.toml"), "scratch = true\n")
        @test_throws r"input capture of .*a scratch run's output" missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = env0, site = "desktop")
        # ... but it is an input of a scratch launch, as of a scratch driver (`verify_inputs`)
        @test isempty(missing_inputs(g, ["chrome_prepare"]; data_root = f.data, env = merge(env0, Dict("VGTK_SCRATCH" => "1")), site = "desktop"))
        # a distributed stage: ntasks and nodes; resources validated
        text = read(f.graph, String)
        write(f.graph, replace(text, "name = \"setup\"\n" => "name = \"setup\"\nntasks = 4\nnodes = 2\n"; count = 1))
        plan = plan_reproduction(load_stage_graph(f.graph), load_site("dgx01"); env = env0, queue_states = noqueue, only = ["setup"])
        @test occursin("--mem-per-cpu=8480M --ntasks=4 --nodes=2", join(plan.stages[1].command, " "))
        for (k, v) in (("gres", "\"gpu 4\""), ("mem_per_cpu", "\"lots\""), ("mem_per_cpu", "8"), ("ntasks", "-1"))
            write(f.graph, replace(text, "name = \"setup\"\n" => "name = \"setup\"\n$k = $v\n"; count = 1))
            @test_throws ArgumentError load_stage_graph(f.graph)
        end
        write(f.graph, replace(text, "name = \"setup\"\n" => "name = \"setup\"\nmem_per_cpu = \"4000M\"\ngres = \"gpu:b200:2\"\n"; count = 1))
        @test load_stage_graph(f.graph) isa StageGraph
        write(f.graph, replace(text, "name = \"setup\"\n" => "name = \"setup\"\nmem_per_cpu = \"lots\"\n"; count = 1))
        @test_throws r"mem_per_cpu = \"lots\" is not a Slurm memory" load_stage_graph(f.graph)
    end

    @testset "the dry run: the job graph on a Slurm site and on a local one" begin
        f = runner_project()
        code, out = run_main(["--graph=$(f.graph)", "--site=dgx01", "--dry-run", "--data-root=$(f.data)", "--results-root=$(f.results)"]; env = env0)
        @test code == 0
        @test occursin("reproduce paper on dgx01 (slurm): 6 stage(s)", out)
        @test occursin("[1] setup\n", out) && occursin("[3] chrome  array of 3: alpha mid zeta", out)
        # each stage's own launcher, the site's options, the dependencies on this launch's stages
        @test occursin("submit   sbatch --parsable --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem-per-cpu=8480M slurm/setup.sbatch", out)
        @test occursin("EXPERIMENT=chrome sbatch --parsable --dependency=afterok:\$setup --partition=gpu --cpus-per-task=2 --mem-per-cpu=8480M --job-name=paper-prepare-chrome slurm/prepare.sbatch", out)
        @test occursin("CASES=alpha,mid,zeta sbatch --parsable --dependency=afterok:\$chrome_prepare --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem-per-cpu=8480M --array=0-2 --time=130 slurm/chrome.sbatch", out)
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

    @testset "arrays: over a list or a table of the config, and a loop over values inside each task" begin
        # chrome_index_grid's shape: one task per chrome set (a list of the config), each running every variant (a
        # table of the config) in turn, the variant in VARIANT
        f = runner_project()
        write(joinpath(f.proj, "configs", "experiments", "grid.toml"), """
            [grid]
            sets = ["s2", "s1", "s3"]
            [variants.b]
            x = 1
            [variants.a]
            x = 2
            """)
        write(joinpath(f.proj, "slurm", "grid.sbatch"), raw"""#!/bin/bash
            echo "grid task=${SLURM_ARRAY_TASK_ID:-} variant=${VARIANT:-} sets=${SETS:-}" >> "$VGTK_RESULTS_ROOT/ran.txt"
            [ "${VARIANT:-}" != "fail" ]
            """)
        write(f.graph, read(f.graph, String) * """

            [[stage]]
            name = "grid"
            after = ["setup"]
            sbatch = "slurm/grid.sbatch"
            config = "configs/experiments/grid.toml"
            array = { from = "grid.sets", select_env = "SETS", each = { from = "variants", env = "VARIANT" } }
            default = false
            """)
        git!(f.repo, "add", "-A"); git!(f.repo, "commit", "-q", "-m", "grid")
        _stage_index(g, n) = findfirst(st -> st.name == n, g.stages)
        g = load_stage_graph(f.graph)
        s = g.stages[_stage_index(g, "grid")]
        @test stage_items(g, s; env = env0) == ["s2", "s1", "s3"]                                  # a list: in its order
        @test stage_items(g, s; env = merge(env0, Dict("SETS" => "s3,s2"))) == ["s3", "s2"]
        @test_throws r"SETS names unknown value\(s\) s9" stage_items(g, s; env = merge(env0, Dict("SETS" => "s9")))
        @test stage_each(g, s, "s1"; env = env0) == ["a", "b"]                                     # a table: sorted
        @test stage_each(g, s, "s1"; env = merge(env0, Dict("VARIANT" => "b"))) == ["b"]
        @test stage_each(g, g.stages[1], ""; env = env0) == String[]
        # the local backend: each task runs every value in turn
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=grid"]; env = env0)
        @test code == 0
        ran = readlines(joinpath(f.results, "ran.txt"))
        @test ran[2:end] == ["grid task=$t variant=$v sets=s2,s1,s3" for t in 0:2 for v in ("a", "b")]
        # the values are keyed: another selection is another run
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=grid"]; env = env0)
        @test occursin(r"grid\s+reused", out)
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=grid"]; env = merge(env0, Dict("VARIANT" => "b")))
        @test code == 0 && !occursin(r"grid\s+reused", out) && readlines(joinpath(f.results, "ran.txt"))[end] == "grid task=2 variant=b sets=s2,s1,s3"
        # the dry run names the loop
        code, out = run_main(["--graph=$(f.graph)", "--site=dgx01", "--only=grid", "--dry-run", "--results-root=$(f.results)"]; env = env0)
        @test occursin("[2] grid  array of 3: s2 s1 s3", out) && occursin("each     VARIANT over variants: a b (every task)", out)
        @test occursin("--array=0-2", out)
        # on Slurm the marker wrapper loops: run here as the job of task 1, it runs both values and writes the marker
        bin = fake_bin(); sb = joinpath(mktempdir(), "sbatch.log")
        env = merge(env0, Dict("PATH" => bin * ":" * ENV["PATH"], "FAKE_SBATCH_LOG" => sb, "FAKE_SQUEUE" => "/dev/null"))
        code, out = run_main(["--graph=$(f.graph)", "--site=dgx01", "--only=grid", "--no-deps", "--results-root=$(f.results)", "--data-root=$(f.data)"]; env)
        @test code == 0
        wrapper = only(filter(endswith("marker_wrapper.sbatch"), split(read(sb, String))))
        n = length(readlines(joinpath(f.results, "ran.txt")))
        job = Dict("SLURM_ARRAY_TASK_ID" => "1", "SLURM_JOB_ID" => "7", "VGTK_RESULTS_ROOT" => f.results, "PATH" => ENV["PATH"])
        @test success(Cmd(`bash $wrapper`; env = job, dir = f.proj))
        @test readlines(joinpath(f.results, "ran.txt"))[(n + 1):end] == ["grid task=1 variant=a sets=", "grid task=1 variant=b sets="]
        @test stage_markers(dirname(wrapper))[1]["state"] == "completed"
        # a failing value fails the task, and stops it
        write(joinpath(f.proj, "configs", "experiments", "grid.toml"), "[grid]\nsets = [\"s1\"]\n[variants.fail]\n[variants.z]\n")
        git!(f.repo, "commit", "-qam", "fail")
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=grid", "--no-deps"]; env = env0)
        @test code == 4 && readlines(joinpath(f.results, "ran.txt"))[end] == "grid task=0 variant=fail sets=s1"
        # the declaration is checked
        bad(a) = (write(f.graph, replace(read(f.graph, String), r"^array = \{ from = \"grid.sets\".*$"m => a)); f.graph)
        @test_throws r"array.each takes" load_stage_graph(bad("array = { from = \"grid.sets\", each = { from = \"variants\" } }"))
        @test_throws r"array.each.env = \"1x\" is no variable" load_stage_graph(bad("array = { from = \"grid.sets\", each = { from = \"variants\", env = \"1x\" } }"))
        @test_throws r"items = \[...\] or from" load_stage_graph(bad("array = { from = \"grid.sets\", items = [\"a\"] }"))
        g2 = load_stage_graph(bad("array = { from = \"grid.nope\" }"))
        @test_throws r"the config has no `grid.nope`" stage_items(g2, g2.stages[_stage_index(g2, "grid")]; env = env0)
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

    @testset "an upstream stage keyed anew reruns every stage after it, in one launch or in launches of their own" begin
        f = runner_project()
        mkpath(joinpath(f.data, "boards")); write(joinpath(f.data, "boards", "a4.toml"), "")
        mkpath(joinpath(f.data, "photos", "sheet_small")); mkpath(joinpath(f.data, "photos", "sheet_large"))
        nran() = (p = joinpath(f.results, "ran.txt"); isfile(p) ? length(readlines(p)) : 0)
        # a launch: its exit code, its output and how many tasks it ran (setup 1, chrome_prepare 1, chrome 3, texture 2)
        launch(only; deps = true) = (n = nran(); (c, o) = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=$only",
                                     (deps ? () : ("--no-deps",))...]; env = env0); (c, o, nran() - n))
        chrome_changed() = open(io -> println(io, "# changed"), joinpath(f.proj, "configs", "experiments", "chrome.toml"), "a")
        code, _, ran = launch("texture")
        @test code == 0 && ran == 1 + 1 + 3 + 2
        # chrome's config changed: chrome_prepare and chrome read it and rerun; texture reads neither file, but its
        # input is chrome's output, so it reruns too (it was reused, on chrome's old output); setup is reused
        chrome_changed()
        code, out, ran = launch("texture")
        @test code == 0 && count("reused", out) == 1 && occursin(r"setup\s+reused", out) && ran == 1 + 3 + 2
        # the same across launches of one stage each (`--no-deps`): texture's key folds the key of chrome's last completed
        # run, whose output it reads. Before chrome reruns, that output is unchanged: texture is reused
        chrome_changed()
        code, out, ran = launch("texture"; deps = false)
        @test code == 0 && occursin(r"texture\s+reused", out) && ran == 0
        @test launch("chrome_prepare"; deps = false)[[1, 3]] == (0, 1)
        @test launch("chrome"; deps = false)[[1, 3]] == (0, 3)
        code, out, ran = launch("texture"; deps = false)
        @test code == 0 && !occursin("reused", out) && ran == 2
        # and unchanged, everything is reused again
        code, out, ran = launch("texture")
        @test code == 0 && count("reused", out) == 4 && ran == 0
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
        @test startswith(lines[3], "103 --parsable --dependency=afterok:102 ") && occursin("--array=0-2 --time=130 --job-name=paper-chrome $(wrapper("chrome"))", lines[3]) &&
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
        run(`kill -USR1 $(getpid(proc))`); wait(proc)
        @test proc.exitcode == 0 && isfile(joinpath(dir, "got_usr1"))
        mk = get(stage_markers(dir), 0, Dict{String, Any}())
        @test get(mk, "state", "") == "completed" && get(mk, "signal", "") == "USR1"
    end

    @testset "code identity: a stage reruns when a code tree changed, naming it; docs reuse it; uncommitted code is refused" begin
        f = runner_project()
        gitin(dir) = (args...) -> run(pipeline(`git -C $dir -c user.name=t -c user.email=t@t -c commit.gpgsign=false $args`; stdout = devnull, stderr = devnull))
        g! = gitin(f.proj)
        # a stack package loaded from a path, in a repository of its own, and a released one, in the project's environment
        pkg = mktempdir(); p! = gitin(pkg)
        p!("init", "-q")
        mkpath(joinpath(pkg, "src")); mkpath(joinpath(pkg, "docs"))
        write(joinpath(pkg, "Project.toml"), "name = \"StackPkg\"\nuuid = \"00000000-0000-0000-0000-000000000001\"\n")
        write(joinpath(pkg, "src", "StackPkg.jl"), "module StackPkg end\n"); write(joinpath(pkg, "docs", "index.md"), "# StackPkg\n")
        p!("add", "-A"); p!("commit", "-q", "-m", "pkg")
        write(joinpath(f.proj, "Project.toml"), """
            [deps]
            StackPkg = "00000000-0000-0000-0000-000000000001"
            VisualGeometryToolkitCore = "00000000-0000-0000-0000-000000000002"
            """)
        core_tree = "0123456789abcdef0123456789abcdef01234567"
        write(joinpath(f.proj, "Manifest.toml"), """
            manifest_format = "2.0"
            [[deps.StackPkg]]
            path = "$pkg"
            uuid = "00000000-0000-0000-0000-000000000001"
            [[deps.VisualGeometryToolkitCore]]
            deps = ["Example"]
            git-tree-sha1 = "$core_tree"
            uuid = "00000000-0000-0000-0000-000000000002"
            [[deps.Example]]
            git-tree-sha1 = "fedcba9876543210fedcba9876543210fedcba98"
            uuid = "00000000-0000-0000-0000-000000000003"
            [[deps.NotLoaded]]
            path = "/nowhere"
            uuid = "00000000-0000-0000-0000-000000000004"
            """)
        g!("add", "-A"); g!("commit", "-q", "-m", "the environment")
        ct = VisualGeometryToolkitEval.code_trees(f.proj; code = ["scripts", "slurm"])
        @test sort(collect(keys(ct.trees))) == ["StackPkg", "VisualGeometryToolkitCore", "scripts", "slurm"] && isempty(ct.dirty)
        @test ct.trees["VisualGeometryToolkitCore"] == core_tree                 # a released one: its Manifest tree
        @test ct.trees["scripts"] == strip(read(`git -C $(f.proj) rev-parse HEAD:./scripts`, String))
        @test startswith(ct.trees["StackPkg"], "src:") && occursin(" Project.toml:", ct.trees["StackPkg"])
        ran() = length(readlines(joinpath(f.results, "ran.txt")))
        args = ["--graph=$(f.graph)", "--site=desktop", "--only=setup"]
        code, out = run_main(args; env = env0)
        @test code == 0 && ran() == 1
        rec() = TOML.parsefile(last(run_manifests(joinpath(f.results, "_reproduce"))))
        @test rec()["stages"]["setup"]["code_trees"] == ct.trees && rec()["code_trees"] == ct.trees
        # a docs-only commit, in the project and in the package: no tree changed, the stage is reused
        mkpath(joinpath(f.proj, "docs")); write(joinpath(f.proj, "docs", "x.md"), "x\n"); g!("add", "-A"); g!("commit", "-q", "-m", "docs")
        write(joinpath(pkg, "docs", "index.md"), "# StackPkg, documented\n"); p!("commit", "-qam", "docs")
        code, out = run_main(args; env = env0)
        @test code == 0 && occursin(r"setup\s+reused", out) && ran() == 1
        # the package's src changes: the stage reruns, the package named
        write(joinpath(pkg, "src", "StackPkg.jl"), "module StackPkg\nf() = 1\nend\n"); p!("commit", "-qam", "src")
        code, out = run_main(args; env = env0)
        @test code == 0 && occursin(r"setup\s+rerun: code changed: StackPkg: src [0-9a-f]{10} → [0-9a-f]{10}", out) && ran() == 2
        @test !occursin("VisualGeometryToolkitCore", out)
        code, out = run_main(args; env = env0)
        @test code == 0 && occursin(r"setup\s+reused", out) && ran() == 2
        # the project's own code changes: named by its path
        write(joinpath(f.proj, "scripts", "chrome.jl"), "# the driver, changed\n"); g!("commit", "-qam", "driver")
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--dry-run", "--only=setup"]; env = env0)
        @test code == 0 && occursin("RERUN (code changed: scripts: ", out)
        code, out = run_main(args; env = env0)
        @test code == 0 && occursin("rerun: code changed: scripts: ", out) && ran() == 3
        # uncommitted code is refused, in the project or in a package; the dry run shows it; a dirty paper is no code
        write(joinpath(pkg, "src", "StackPkg.jl"), "module StackPkg\nf() = 2\nend\n")
        code, out = run_main(args; env = env0)
        @test code == 2 && occursin("a run of record runs committed code, and StackPkg has uncommitted changes", out) && ran() == 3
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--dry-run", "--only=setup"]; env = env0)
        @test code == 0 && occursin("DIRTY: StackPkg (a launch is refused)", out)
        p!("checkout", "-q", "--", "src/StackPkg.jl")
        write(joinpath(f.proj, "slurm", "new.sbatch"), "#!/bin/bash\n")             # untracked, in a code path
        code, out = run_main(args; env = env0)
        @test code == 2 && occursin("slurm has uncommitted changes", out)
        rm(joinpath(f.proj, "slurm", "new.sbatch"))
        write(joinpath(f.proj, "docs", "x.md"), "y\n")                              # tracked, not code
        code, out = run_main(args; env = env0)
        @test code == 0 && occursin(r"setup\s+reused", out)
        g!("checkout", "-q", "--", "docs/x.md")
        # a launch recorded before the trees were kept: its stage reruns once, then is reused
        m = last(run_manifests(joinpath(f.results, "_reproduce"))); d = TOML.parsefile(m)
        delete!(d["stages"]["setup"], "code_trees"); open(io -> TOML.print(io, d), m, "w")
        code, out = run_main(args; env = env0)
        @test code == 0 && occursin("rerun: code changed: no code trees recorded", out) && ran() == 4
        code, out = run_main(args; env = env0)
        @test code == 0 && occursin(r"setup\s+reused", out) && ran() == 4
        # a scratch launch runs uncommitted code with a warning (docs/src/reproduce.md), under a key of its own
        write(joinpath(pkg, "src", "StackPkg.jl"), "module StackPkg\nf() = 3\nend\n")
        scratch = merge(env0, Dict("VGTK_SCRATCH" => "1"))
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--dry-run", "--only=setup"]; env = scratch)
        @test code == 0 && occursin("DIRTY: StackPkg (a scratch launch: run with a warning)", out)
        code, out = @test_logs (:warn, r"a scratch launch runs uncommitted code") match_mode = :any run_main(args; env = scratch)
        @test code == 0 && ran() == 5
        @test occursin("src:", rec()["stages"]["setup"]["code_trees"]["StackPkg"]) && occursin("+uncommitted", rec()["stages"]["setup"]["code_trees"]["StackPkg"])
        code, out = run_main(args; env = env0)                                      # a run of record still refuses it
        @test code == 2 && occursin("StackPkg has uncommitted changes", out) && ran() == 5
        p!("checkout", "-q", "--", "src/StackPkg.jl")
        # the rule itself
        changes = VisualGeometryToolkitEval.code_tree_changes
        @test isempty(changes(Dict("a" => "x"), Dict("a" => "x"))) && isempty(changes(Dict{String, Any}(), Dict{String, String}()))
        @test changes(Dict("a" => "x", "b" => "y"), Dict("a" => "x", "c" => "z")) == ["b: no longer loaded", "c: new"]
        @test changes(Dict("P" => "src:1 Project.toml:2"), Dict("P" => "src:1 Project.toml:3 ext:4")) ==
              ["P: Project.toml 2 → 3", "P: ext none → 4"]
        # the registered stack packages are the published rows of the stack's allow-list
        allow = TOML.parsefile(joinpath(pkgdir(VisualGeometryToolkitEval), "..", "VisualGeometryToolkitCore", "configs", "stack_dependencies.toml"))
        @test Set(VisualGeometryToolkitEval._STACK_PACKAGES) == Set(k for k in keys(allow) if isuppercase(first(k)))
    end

    @testset "code identity on Slurm: a stage in flight under other code stops the launch, nothing submitted twice" begin
        f = runner_project()
        bin = fake_bin()
        log, queue = joinpath(mktempdir(), "sbatch.log"), joinpath(mktempdir(), "squeue.txt")
        write(queue, "")
        env = merge(env0, Dict("PATH" => bin * ":" * ENV["PATH"], "FAKE_SBATCH_LOG" => log, "FAKE_SQUEUE" => queue,
                               "VGTK_DATA_ROOT" => f.data, "VGTK_RESULTS_ROOT" => f.results))
        launch(args) = run_main(args; env, queue_states = nothing, job_states = nothing)    # the stand-ins on PATH
        args = ["--graph=$(f.graph)", "--site=dgx01", "--only=setup"]
        code, out = launch(args)
        @test code == 0 && length(readlines(log)) == 1                                      # setup: job 101
        # job 101 still runs; the same code: depended on, not submitted again
        write(queue, "101|RUNNING|None\n")
        code, out = launch(args)
        @test code == 0 && occursin(r"setup\s+101  \(in flight", out) && length(readlines(log)) == 1
        # a commit changes the code (scripts/) while job 101 runs: the launch stops (exit 4), naming the job and the
        # tree, and submits nothing
        write(joinpath(f.proj, "scripts", "chrome.jl"), "# the driver, changed\n")
        run(pipeline(`git -C $(f.proj) -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -qam driver`; stdout = devnull, stderr = devnull))
        nman = length(run_manifests(joinpath(f.results, "_reproduce")))
        code, out = launch(args)
        @test code == 4 && occursin("job 101 of an earlier launch is still in flight, under other code (scripts: ", out)
        @test length(readlines(log)) == 1 && length(run_manifests(joinpath(f.results, "_reproduce"))) == nman
        code, out = launch(["--graph=$(f.graph)", "--site=dgx01", "--only=setup", "--dry-run"])          # the dry run says so too
        @test code == 4 && occursin("still in flight, under other code", out)
        # once the job has left the queue it reruns under the new code (its marker never written: unknown)
        write(queue, "")
        code, out = launch(args)
        @test code == 0 && length(readlines(log)) == 2 && startswith(readlines(log)[2], "102 ")
    end

    @testset "a launch that identifies none of the project's code is refused; a job's log found by its default name" begin
        f = runner_project()
        write(joinpath(f.proj, "configs", "nocode.toml"), replace(read(f.graph, String), r"^code = .*$"m => ""))
        run(pipeline(`git -C $(f.proj) -c user.name=t -c user.email=t@t -c commit.gpgsign=false add -A`; stdout = devnull))
        run(pipeline(`git -C $(f.proj) -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -qm nocode`; stdout = devnull))
        g = joinpath(f.proj, "configs", "nocode.toml")
        code, out = run_main(["--graph=$g", "--site=desktop", "--only=setup"]; env = env0)
        @test code == 2 && occursin("none of the project's own code is identified", out) && !isfile(joinpath(f.results, "ran.txt"))
        code, out = run_main(["--graph=$g", "--site=desktop", "--only=setup", "--dry-run"]; env = env0)
        @test code == 0 && occursin("NO PROJECT CODE", out)
        # %x in a launcher's --output is the job's name, which is <project>-<stage> when nothing names it
        write(joinpath(f.proj, "slurm", "setup.sbatch"), "#!/bin/bash\n#SBATCH --output=logs/%x_%j.log\ntrue\n")
        mkpath(joinpath(f.proj, "logs"))
        write(joinpath(f.proj, "logs", "a-decoy_101.log"), ""); write(joinpath(f.proj, "logs", "leg-paper-setup_101.log"), "")
        sg = load_stage_graph(f.graph)
        @test Reproduce._slurm_log(sg, sg.stages[1], "101", 0) == joinpath(f.proj, "logs", "leg-paper-setup_101.log")
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

    @testset "the results-branch stage: the renders of the graph's assets on a results branch, named in the launch's manifest" begin
        f = runner_project()
        git!(f.repo, "checkout", "-q", "-b", "main")
        mkpath(joinpath(f.proj, "paper", "fig")); write(joinpath(f.proj, "paper", "fig", "a.png"), UInt8[1, 2, 3])
        git!(f.repo, "add", "-A"); git!(f.repo, "commit", "-q", "-m", "submitted")
        mkpath(joinpath(f.proj, "plots")); write(joinpath(f.proj, "plots", "a.png"), UInt8[1, 2, 4]); write(joinpath(f.proj, "plots", "a.txt"), "x")
        mkpath(joinpath(f.data, "boards")); write(joinpath(f.data, "boards", "a4.toml"), "")
        mkpath(joinpath(f.data, "photos", "sheet_small")); mkpath(joinpath(f.data, "photos", "sheet_large"))
        g = load_stage_graph(f.graph)
        @test results_branch_assets(g) == [joinpath(f.proj, "plots", "a.png") => "paper/fig/a.png"]
        # a local launch: the results-branch stage runs in a julia that loads this package (this suite's environment),
        # writes the branch name into the launch's manifest, and the runner's own later rewrites keep it
        wt = mktempdir()
        env = merge(env0, Dict("JULIA" => joinpath(Sys.BINDIR, Base.julia_exename()),
                               "JULIA_LOAD_PATH" => join(["@"; Base.load_path(); "@stdlib"], Sys.iswindows() ? ';' : ':'),
                               "VGTK_RESULTS_WORKTREES" => wt))
        code, out = run_main(["--graph=$(f.graph)", "--site=desktop", "--only=results"]; env)
        @test code == 0
        m = TOML.parsefile(only(run_manifests(joinpath(f.results, "_reproduce"))))
        @test m["stages"]["results"]["state"] == "completed"
        branch = get(m, "results_branch", "")
        @test startswith(branch, "results/") && !isempty(readchomp(`git -C $(f.repo) branch --list $branch`))
        tree = joinpath(wt, only(filter(d -> isfile(joinpath(wt, d, "proj", "RUN.toml")), readdir(wt))))
        run_toml = TOML.parsefile(joinpath(tree, "proj", "RUN.toml"))
        @test run_toml["run_keys"] == sort([String(r["key"]) for (n, r) in m["stages"] if n != "results"])   # the stages before it
        @test run_toml["machine"] == "desktop" && run_toml["assets"] == ["proj/paper/fig/a.png"]
        @test read(joinpath(tree, "proj", "paper", "fig", "a.png")) == UInt8[1, 2, 4]
    end
end

# `code_trees` runs git several times per path; a driver asks before every cell. It answers once per process while
# nothing git reads changed, and a change — an edit, an untracked or deleted file, a commit, a staging, in the project
# or in a package it loads from a path — is seen at once.
@testitem "code_trees: once per process while nothing changed, every change seen" begin
    using Test
    using VisualGeometryToolkitEval: code_trees
    git(dir, args...) = run(pipeline(`git -C $dir -c user.name=t -c user.email=t@t -c commit.gpgsign=false $args`; stdout = devnull, stderr = devnull))
    pkg = mktempdir(); git(pkg, "init", "-q")
    mkpath(joinpath(pkg, "src")); write(joinpath(pkg, "src", "P.jl"), "module P end\n")
    write(joinpath(pkg, "Project.toml"), "name = \"P\"\nuuid = \"00000000-0000-0000-0000-000000000001\"\n")
    git(pkg, "add", "-A"); git(pkg, "commit", "-q", "-m", "p")
    repo = mktempdir(); git(repo, "init", "-q")
    mkpath(joinpath(repo, "src")); write(joinpath(repo, "src", "a.jl"), "a\n")
    write(joinpath(repo, "Project.toml"), "[deps]\nP = \"00000000-0000-0000-0000-000000000001\"\n")
    write(joinpath(repo, "Manifest.toml"), "manifest_format = \"2.0\"\n[[deps.P]]\npath = \"$pkg\"\nuuid = \"00000000-0000-0000-0000-000000000001\"\n")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "r")
    # a git that counts its runs, first on PATH
    bin = mktempdir(); calls = joinpath(bin, "calls")
    write(joinpath(bin, "git"), "#!/bin/sh\necho x >> '$calls'\nexec '$(Sys.which("git"))' \"\$@\"\n"); chmod(joinpath(bin, "git"), 0o755)
    ran() = isfile(calls) ? countlines(calls) : 0
    withenv("PATH" => bin * ":" * ENV["PATH"]) do
        a = code_trees(repo)
        @test ran() > 0 && isempty(a.dirty) && sort(collect(keys(a.trees))) == ["P", "src"]
        n = ran()
        @test code_trees(repo) == a && ran() == n                                   # nothing changed: no git
        write(joinpath(repo, "src", "a.jl"), "b\n")                                 # an edit
        @test code_trees(repo).dirty == ["src"] && ran() > n
        n = ran(); @test code_trees(repo).dirty == ["src"] && ran() == n
        git(repo, "commit", "-qam", "edit")                                         # committed: clean, another tree
        c = code_trees(repo)
        @test isempty(c.dirty) && c.trees["src"] != a.trees["src"]
        write(joinpath(pkg, "src", "new.jl"), "x\n")                                # untracked, in a package's src
        @test code_trees(repo).dirty == ["P"]
        rm(joinpath(pkg, "src", "new.jl"))
        @test code_trees(repo) == c
        git(pkg, "rm", "-q", "--cached", "src/P.jl")                                # the index alone changes
        @test code_trees(repo).dirty == ["P"]
        git(pkg, "add", "src/P.jl")
        @test code_trees(repo) == c
        write(joinpath(pkg, "src", "P.jl"), "module P\nf() = 1\nend\n"); git(pkg, "commit", "-qam", "f")
        @test isempty(code_trees(repo).dirty) && code_trees(repo).trees["P"] != c.trees["P"]
        git(repo, "checkout", "-q", "HEAD~1")                                       # a checkout
        @test code_trees(repo).trees["src"] == a.trees["src"]
    end
end
