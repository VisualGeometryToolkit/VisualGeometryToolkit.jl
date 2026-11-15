# The configuration-of-record layer (src/experiments/run_records.jl).

@testitem "run_records" begin
    import VisualGeometryToolkitEval as VgtkEval
    using Test, TOML
    using VisualGeometryToolkitEval
    const RR = VgtkEval.Experiments

    @testset "run_key: effective config minus the render section" begin
        cfg = Dict{String,Any}("experiment" => Dict{String,Any}("dataset" => "pt", "matcher" => "sift",
                                                                "problem" => "E", "pool_dir" => "pools"),
                               "arm" => Dict{String,Any}("efficiency_constant" => 1.5, "arms" => "a,b"),
                               "render" => Dict{String,Any}("xticks" => [1, 10, 100], "legend" => "l"))
        key = RR.run_key(cfg)
        @test key isa String && length(key) == 16 && all(isxdigit, key)   # SHA-256 of the canonical JSON, Julia-version stable
        # a render knob edit re-keys nothing; a data knob edit does
        c2 = deepcopy(cfg); c2["render"]["xticks"] = [5]
        @test RR.run_key(c2) == key
        c3 = deepcopy(cfg); delete!(c3, "render")
        @test RR.run_key(c3) == key
        c4 = deepcopy(cfg); c4["arm"]["efficiency_constant"] = 2.0
        @test RR.run_key(c4) != key
        # order-independent and TOML round-trip stable (the on-disk _config.toml names its directory)
        io = IOBuffer(); TOML.print(io, cfg; sorted = true)
        back = TOML.parse(String(take!(io)))
        @test RR.run_key(back) == key
        @test "render" in RR.RUN_KEY_EXCLUDED
    end

    @testset "run_key: the stack's canonical JSON — key order and number spelling do not move it" begin
        a = TOML.parse("n = 3\nw = 1.0\n[arm]\nefficiency_constant = 1.5\nscenes = [\"x\", \"y\"]\n")
        b = TOML.parse("[arm]\nscenes = [\"x\", \"y\"]\nefficiency_constant = 15e-1\n\n[other]\n")
        delete!(b, "other"); b["w"] = 1e0; b["n"] = 3.0          # 3 and 3.0, 1.0 and 1e0: one number each
        @test RR.run_key(a) == RR.run_key(b)
        @test RR.run_key(a) == VgtkEval._content_key(a) == VgtkEval._content_key(TOML.parse(sprint(io -> TOML.print(io, a))))
        @test VgtkEval._content_key(a) == bytes2hex(VgtkEval.sha256(VgtkEval.canonical_json(a)))[1:16]   # Core's one encoding
        a["n"] = 3.5
        @test RR.run_key(a) != RR.run_key(b)                      # a value change moves it
        @test_throws ArgumentError RR.run_key(Dict{String,Any}("w" => Inf))   # an unencodable value fails loudly
    end

    @testset "run_key: an integer's spelling does not move it (hex, octal, binary: TOML's unsigned integers)" begin
        hex = TOML.parse("[arm]\nransac_seed = 0x0b10b5eed\nmask = 0o17\nbits = [0b101, 7]\n")
        dec = TOML.parse("[arm]\nransac_seed = 2970312429\nmask = 15\nbits = [5, 7.0]\n")
        @test hex["arm"]["ransac_seed"] isa Unsigned                     # TOML's hex literal is unsigned
        @test RR.run_key(hex) == RR.run_key(dec) == VgtkEval._content_key(dec)
        @test RR.run_key(TOML.parse(sprint(TOML.print, hex))) == RR.run_key(hex)   # TOML writes it back as hex
        # a loaded config holds the Int64 it is keyed by, so a run computes with what its key says
        p = joinpath(mktempdir(), "c.toml")
        write(p, "[experiment]\nlayout = [\"x\"]\n[arm]\nransac_seed = 0x0b10b5eed\nbits = [0b101, 7]\n")
        loaded = RR.load_run_config(p)
        @test loaded["arm"]["ransac_seed"] === 2970312429 && loaded["arm"]["bits"] == [5, 7] && loaded["arm"]["bits"] isa Vector{Int64}
        tree = TOML.parse("n = 3\n[a]\nv = [1, 2]\n")
        @test VgtkEval._lower_integers(tree, "") === tree                                  # nothing to lower: the tree itself
        @test VgtkEval._content_key(Dict{String,Any}("b" => true)) != VgtkEval._content_key(Dict{String,Any}("b" => 1))   # a Bool stays a Bool
        other = deepcopy(hex); other["arm"]["ransac_seed"] += 1
        @test RR.run_key(other) != RR.run_key(hex)                       # its value still moves it
        # an integer out of range is refused by its key path, never hashed to something
        refusal(cfg) = try RR.run_key(cfg); nothing catch e; e end
        e = refusal(TOML.parse("[arm]\nransac_seed = 0xffffffffffffffff\n"))
        @test e isa ArgumentError && occursin("arm.ransac_seed", e.msg)
        e = refusal(TOML.parse("[arm]\nbits = [1, 0x8000000000000000]\n"))
        @test e isa ArgumentError && occursin("arm.bits[2]", e.msg)
        e = refusal(Dict{String,Any}("arm" => Dict{String,Any}("huge" => big(2)^70)))
        @test e isa ArgumentError && occursin("arm.huge", e.msg)
        e = refusal(Dict{String,Any}("arm" => Dict{String,Any}("seeds.b" => typemin(Int64))))   # a quoted key, quoted
        @test e isa ArgumentError && occursin("arm.\"seeds.b\"", e.msg)
        e = refusal(Dict{String,Any}("n" => 2^53 + 1))                     # an Int64 past Core's bound: named too
        @test e isa ArgumentError && occursin("`n`", e.msg)
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
            # a step's directory is DrWatson's datadir("exp_pro", experiment, step, key) under the results root
            s = RR.Product(joinpath("pt", "sift", "E"), "cv", Dict("n" => 1); output = "result.json", code = nothing)
            @test RR.product_dir(s) == joinpath("/tmp/x", "pt", "sift", "E", "cv", RR.run_key(Dict("n" => 1)))
            @test !RR.produced(s)
        end
        RR.set_project_root!("")
    end
end

@testitem "run_records: config_of_record, a cell's step" begin
    # The configs of record of a project (configs/<name>.toml, loaded as run
    # configs) and the keyed step of one cell under its experiment's
    # results: `<results root>/<experiment>/<step>/<run_key>`.
    using VisualGeometryToolkitEval.Experiments: config_of_record, Product, product_dir, run_key,
        set_project_root!
    root = mktempdir()
    mkpath(joinpath(root, "configs"))
    write(joinpath(root, "configs", "cell.toml"), """
        [experiment]
        dataset = "d"
        matcher = "m"
        problem = "E"
        pool_dir = "pools"
        [render]
        main = "x"
        """)
    cfg = config_of_record("cell"; root)
    @test cfg["experiment"]["problem"] == "E"
    @test_throws ErrorException config_of_record("absent"; root)
    withenv("VGTK_RESULTS_ROOT" => joinpath(root, "results")) do
        dir = product_dir(Product(joinpath("d", "m", "E", "fig2"), "budget", cfg; output = "budget.json", code = nothing))
        @test dir == joinpath(root, "results", "d", "m", "E", "fig2", "budget", run_key(cfg))
        @test startswith(dir, joinpath(root, "results"))
    end
    set_project_root!(root)
    try
        @test config_of_record("cell") == cfg
    finally
        set_project_root!("")
    end
end
