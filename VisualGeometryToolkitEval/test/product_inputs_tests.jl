# Another project's run of record as an input (src/code_identity.jl `product_input`, Experiments' `input_path`, the
# runner's `missing_inputs`, PaperProjects' A1/A2) and a results-mount tier read under the results root.

@testitem "product inputs: another project's keyed product, read where it lives" setup = [GitFixture] begin
    using Test, TOML
    import VisualGeometryToolkitEval as VgtkEval
    const E = VgtkEval.Experiments
    const R = VgtkEval.Reproduce
    # bmvc2026's results: one product of record and its record
    theirs = mkpath(joinpath(mktempdir(), "bmvc2026"))
    key = "3f9a0c1d2e4b5a67"
    pdir = mkpath(joinpath(theirs, "calibration", "hand_eye", key))
    write(joinpath(pdir, "hand_eye_bb.json"), "{\"X\": 1}")
    write(joinpath(pdir, "_provenance.toml"), "run_key = \"$key\"\n")
    digest = VgtkEval.file_sha256(joinpath(pdir, "hand_eye_bb.json"))
    t = Dict{String, Any}("project" => "bmvc2026", "product" => "calibration/hand_eye", "key" => key,
                          "file" => "hand_eye_bb.json", "sha256" => digest)
    @test VgtkEval.is_product_input(t) && !VgtkEval.is_product_input(Dict("path" => "x"))
    @test VgtkEval.product_input_path(t) == joinpath("calibration", "hand_eye", key, "hand_eye_bb.json")
    @test_throws r"takes project, product, key" VgtkEval.product_input_path(merge(t, Dict("path" => "x")))
    @test_throws r"`<experiment>/<name>`" VgtkEval.product_input_path(merge(t, Dict("product" => "hand_eye")))
    @test_throws r"names its `key`" VgtkEval.product_input_path(merge(t, Dict("key" => "../k")))

    # the one check: the environment's root, the site's, a tier's; the record's key; the content
    env = Dict("VGTK_RESULTS_ROOT_BMVC2026" => theirs)
    x = VgtkEval.product_input("he", t; env)
    @test x.state === :present && x.path == joinpath(pdir, "hand_eye_bb.json") && x.source == "VGTK_RESULTS_ROOT_BMVC2026"
    x = VgtkEval.product_input("he", t; site_results = p -> p == "bmvc2026" ? theirs : nothing)
    @test x.state === :present && occursin("site", x.source)
    x = VgtkEval.product_input("he", t; env = Dict("VGTK_RESULTS_ROOT_BMVC2026" => mktempdir()), tier_root = () -> theirs)
    @test x.state === :present && x.root == theirs                     # the first root that holds it
    x = VgtkEval.product_input("he", t)
    @test x.state === :missing && occursin("no root names bmvc2026's results", x.problem) && x.rel == VgtkEval.product_input_path(t)
    # a tier root is resolved only when the roots before it do not hold the product
    @test VgtkEval.product_input("he", t; env, tier_root = () -> error("not read")).state === :present
    @test VgtkEval.product_input("he", t; env = Dict("R_BMVC2026" => theirs), prefix = "R").state === :present   # a graph's results_env
    @test VgtkEval.product_input("he", merge(t, Dict("sha256" => "0"^64)); env).state === :refused
    write(joinpath(pdir, "_provenance.toml"), "run_key = \"other\"\n")
    x = VgtkEval.product_input("he", t; env)
    @test x.state === :refused && occursin("names run key \"other\"", x.problem)
    write(joinpath(pdir, "_provenance.toml"), "run_key = \"$key\"\n")

    # a driver: input_path through the environment and through the active site profile ({project} the producer's name)
    proj = gitrepo(); mkpath(joinpath(proj, "configs"))
    cfg = Dict{String, Any}("inputs" => Dict{String, Any}("he" => t))
    withenv("VGTK_RESULTS_ROOT_BMVC2026" => theirs, "VGTK_SITE" => nothing) do
        E.set_project_root!(proj)
        @test E.input_path(cfg, "he") == joinpath(pdir, "hand_eye_bb.json")
        @test E.project_results_root("bmvc2026") === nothing                 # the site's rung alone: none here
    end
    site = joinpath(mktempdir(), "lab.toml")
    write(site, "[site]\nbackend = \"local\"\n[storage]\nresults_root = \"$(dirname(theirs))/{project}\"\n")
    withenv("VGTK_RESULTS_ROOT_BMVC2026" => nothing, "VGTK_SITE" => site) do
        @test E.project_results_root("bmvc2026") == theirs
        @test E.input_path(cfg, "he") == joinpath(pdir, "hand_eye_bb.json")
        @test E.input_path(cfg, "he"; root = theirs) == joinpath(pdir, "hand_eye_bb.json")
    end
    withenv("VGTK_RESULTS_ROOT_BMVC2026" => nothing, "VGTK_SITE" => nothing) do
        @test_throws r"input he \(bmvc2026\): .*no root names bmvc2026's results" E.input_path(cfg, "he")
    end
    E.set_project_root!("")
end

@testitem "results-mount tiers: a run of record published as a tier, read under the results root" setup = [GitFixture] begin
    using Test, TOML
    import VisualGeometryToolkitEval as VgtkEval
    const E = VgtkEval.Experiments
    const R = VgtkEval.Reproduce
    proj = gitrepo(); mkpath(joinpath(proj, "configs"))
    write(joinpath(proj, "configs", "data_artifacts.toml"), """
        [artifacts.bmvc2026_results]
        kind = "tree"
        mount = "results"
        mount_dir = "bmvc2026"
        dir = "."
        """)
    mount = VgtkEval.catalog_results_mount(joinpath(proj, "configs", "data_artifacts.toml"))
    @test mount("bmvc2026_results") == "bmvc2026" && mount("raw") === nothing
    @test VgtkEval.catalog_results_mount(joinpath(proj, "nothing.toml")) === nothing
    @test VgtkEval.tier_data_root("bmvc2026_results", "/d"; env = Dict{String, String}(), results_mount = mount,
                                  results_root = "/r") == ("/r/bmvc2026", "results")
    @test_throws r"published under the results" VgtkEval.tier_data_root("bmvc2026_results", "/d"; env = Dict{String, String}(),
                                                                        results_mount = mount, results_root = "/r", write = true)
    results, data = mktempdir(), mktempdir()
    key = "3f9a0c1d2e4b5a67"
    pdir = mkpath(joinpath(results, "bmvc2026", "calibration", "hand_eye", key))
    write(joinpath(pdir, "poses.json"), "[]"); write(joinpath(pdir, "_provenance.toml"), "run_key = \"$key\"\n")
    cfg = Dict{String, Any}("inputs" => Dict{String, Any}(
        "poses" => Dict{String, Any}("tier" => "bmvc2026_results", "path" => "calibration/hand_eye/$key/poses.json"),
        "he" => Dict{String, Any}("project" => "bmvc2026", "product" => "calibration/hand_eye", "key" => key,
                                  "file" => "poses.json", "tier" => "bmvc2026_results")))
    withenv("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => results, "VGTK_SITE" => nothing,
            "VGTK_RESULTS_ROOT_BMVC2026" => nothing, "VGTK_DATA_ROOT_BMVC2026_RESULTS" => nothing) do
        E.set_project_root!(proj)
        @test E.data_root(; tier = "bmvc2026_results") == joinpath(results, "bmvc2026")
        @test E.input_path(cfg, "poses") == joinpath(pdir, "poses.json")
        @test E.input_path(cfg, "he") == joinpath(pdir, "poses.json")          # the product input falls back to its tier
        E.set_project_root!("")
    end
    # the runner's launch check: the same roots (the results root, the site's results root of the producer)
    write(joinpath(proj, "configs", "c.toml"), sprint(TOML.print, cfg))
    write(joinpath(proj, "configs", "reproduce.toml"),
          "[project]\nname = \"icra\"\nworkdir = \"..\"\n[[stage]]\nname = \"s\"\ncommand = [\"true\"]\nconfig = \"configs/c.toml\"\n")
    g = R.load_stage_graph(joinpath(proj, "configs", "reproduce.toml"))
    e = Dict{String, String}()
    @test isempty(R.missing_inputs(g, ["s"]; data_root = data, env = e, results_root = results))
    rows = R.missing_inputs(g, ["s"]; data_root = data, env = e, results_root = mktempdir())
    @test sort([r.path for r in rows]) == sort(["bmvc2026:" * joinpath("calibration", "hand_eye", key, "poses.json"),
                                                "calibration/hand_eye/$key/poses.json"])
    site = joinpath(mktempdir(), "lab.toml")
    write(site, "[site]\nbackend = \"local\"\n[storage]\nresults_root = \"$results/{project}\"\n")
    rows = R.missing_inputs(g, ["s"]; data_root = data, env = e, results_root = mktempdir(), site_profile = R.load_site(site))
    @test [r.path for r in rows] == ["calibration/hand_eye/$key/poses.json"]   # the product found under the site's root
end

@testitem "paper_projects: A2 accepts a product input, A1 a producer of another project" setup = [GitFixture] begin
    using Test
    import VisualGeometryToolkitEval as VgtkEval
    P = VgtkEval.PaperProjects
    # the other project: its graph names stage pt_e_pools
    other = mktempdir(); mkpath(joinpath(other, "configs"))
    write(joinpath(other, "Artifacts.toml"), "")
    write(joinpath(other, "configs", "reproduce.toml"), "[project]\nname = \"accv2026\"\n[[stage]]\nname = \"pt_e_pools\"\ncommand = [\"true\"]\n")
    root = mktempdir(); mkpath(joinpath(root, "configs"))
    write(joinpath(root, "configs", "data_artifacts.toml"), """
        [artifacts.pools_of_record]
        kind = "produced"
        producer = "accv2026:pt_e_pools"
        covers = ["pools"]
        tier = "derived"
        what = "the pools"
        provenance = "regenerated by accv2026's stage pt_e_pools"
        [artifacts.ghost]
        kind = "produced"
        producer = "accv2026:no_such_stage"
        covers = ["ghost"]
        tier = "derived"
        what = "x"
        provenance = "x"
        """)
    write(joinpath(root, "configs", "cell.toml"), """
        [inputs.he]
        project = "bmvc2026"
        product = "calibration/hand_eye"
        key = "3f9a0c1d2e4b5a67"
        [inputs.bad]
        project = "bmvc2026"
        product = "hand_eye"
        key = "k"
        [inputs.pools]
        tier = "derived"
        path = "pools/x.jld2"
        """)
    v = P.paper_project_violations(root; checks = (:storage, :catalog), stack_artifacts = [joinpath(other, "Artifacts.toml")])
    a2 = [x.message for x in v if x.rule == "A2"]
    @test length(a2) == 1 && occursin("[inputs.bad]", only(a2)) && occursin("<experiment>/<name>", only(a2))
    a1 = [x.message for x in v if x.rule == "A1"]
    @test any(m -> occursin("ghost", m) && occursin("no_such_stage", m) && occursin("is no stage of accv2026", m), a1)
    @test !any(m -> occursin("pools_of_record", m), a1)
    v = P.paper_project_violations(root; checks = (:catalog,))
    @test any(x -> x.rule == "A1" && occursin("no project \"accv2026\" among stack_artifacts", x.message), v)
end
