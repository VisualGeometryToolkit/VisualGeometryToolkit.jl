# The configuration-of-record layer (src/experiments/run_records.jl).

@testitem "run_records" begin
    import VisualGeometryToolkitEval as VgtkEval
    using Test, TOML
    using VisualGeometryToolkitEval
    const RR = VgtkEval.Experiments

    @testset "run_key: effective config minus the render section" begin
        cfg = Dict{String,Any}("experiment" => Dict{String,Any}("dataset" => "pt", "matcher" => "sift",
                                                                "problem" => "E", "pool_dir" => "pools"),
                               "arm" => Dict{String,Any}("scale_multiple" => 1.5, "arms" => "a,b"),
                               "render" => Dict{String,Any}("xticks" => [1, 10, 100], "legend" => "l"))
        key = RR.run_key(cfg)
        @test key isa String && length(key) == 16 && all(isxdigit, key)   # SHA-256 of the sorted TOML, Julia-version stable
        # a render knob edit re-keys nothing; a data knob edit does
        c2 = deepcopy(cfg); c2["render"]["xticks"] = [5]
        @test RR.run_key(c2) == key
        c3 = deepcopy(cfg); delete!(c3, "render")
        @test RR.run_key(c3) == key
        c4 = deepcopy(cfg); c4["arm"]["scale_multiple"] = 2.0
        @test RR.run_key(c4) != key
        # order-independent and TOML round-trip stable (the on-disk _config.toml names its directory)
        io = IOBuffer(); TOML.print(io, cfg; sorted = true)
        back = TOML.parse(String(take!(io)))
        @test RR.run_key(back) == key
        @test "render" in RR.RUN_KEY_EXCLUDED
    end

    @testset "load_run_config sanity" begin
        dir = mktempdir()
        good = joinpath(dir, "good.toml")
        write(good, "[experiment]\ndataset = \"pt\"\nmatcher = \"sift\"\nproblem = \"E\"\npool_dir = \"p\"\n")
        @test RR.load_run_config(good)["experiment"]["problem"] == "E"
        bad = joinpath(dir, "bad.toml"); write(bad, "[experiment]\ndataset = \"pt\"\n")
        @test_throws ErrorException RR.load_run_config(bad)
        @test_throws ErrorException RR.load_run_config(joinpath(dir, "missing.toml"))
    end

    @testset "load_run_config refuses unknown keys in the sections we own" begin
        dir = mktempdir(); n = Ref(0)
        head = "[experiment]\ndataset = \"pt\"\nmatcher = \"sift\"\nproblem = \"E\"\npool_dir = \"p\"\n"
        load(extra) = (n[] += 1; p = joinpath(dir, "c$(n[]).toml"); write(p, head * extra);
                       RR.load_run_config(p))
        # Known keys of an owned section load, unknown ones throw. Until
        # 2026-10-03 only [tuning.<name>] was checked, so a [render] typo was
        # doubly invisible: ignored, AND excluded from the run key, so the
        # driver reused the previous run's keyed directory and its old panel.
        @test load("[render]\nymin = \"0\"\nymax = \"30\"\n") isa AbstractDict
        @test_throws ArgumentError load("[render]\nymim = \"0\"\n")
        @test_throws ArgumentError load("[protocol]\ncv_sections = \"heb\"\n")
        # A NESTED table is checked in its own right — a check on [render]
        # alone sees `main` as one known key and nothing inside it.
        @test load("[render.main]\npanel = \"p\"\npanel_med = \"m\"\n") isa AbstractDict
        @test_throws ArgumentError load("[render.main]\npanel_meed = \"m\"\n")
        # A section this package does NOT own is the project's: not key-checked
        # ([source.recon_special] is keyed by scene name, so a fixed vocabulary
        # would be wrong).
        @test load("[source]\nanything_at_all = 1\n") isa AbstractDict
        @test !any(((sec, _),) -> sec == "source", RR.RUN_CONFIG_KEYS)
        # The sharp pairing this check exists for: [render] is excluded from
        # the key, so nothing else would ever catch a typo in it.
        @test "render" in RR.RUN_KEY_EXCLUDED
        @test any(((sec, _),) -> sec == "render", RR.RUN_CONFIG_KEYS)
    end

    @testset "project root binding" begin
        dir = mktempdir()
        RR.set_project_root!(dir)
        @test RR.project_root() == dir
        withenv("VGTK_DATA_ROOT" => nothing, "VGTK_RESULTS_ROOT" => nothing) do
            @test_throws ErrorException RR.data_root()
            mkpath(joinpath(dir, "data", "exp_raw")); mkpath(joinpath(dir, "data", "exp_pro"))
            @test RR.data_root() == joinpath(dir, "data", "exp_raw")
            @test RR.results_root() == joinpath(dir, "data", "exp_pro")
        end
        withenv("VGTK_RESULTS_ROOT" => "/tmp/x") do
            @test RR.results_root() == "/tmp/x"
            cfg = Dict("experiment" => Dict("dataset" => "pt", "matcher" => "sift", "problem" => "E"))
            @test RR.experiment_dir(cfg) == joinpath("/tmp/x", "pt", "sift", "E")
        end
        # the result sidecar is the caller's to name: there is no default
        @test_throws UndefKeywordError RR.run_complete(dir)
        @test !RR.run_complete(dir; sidecar = "result.jld2")
        RR.set_project_root!("")
    end
end
