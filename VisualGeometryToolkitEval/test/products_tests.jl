# The keyed product (src/experiments/products.jl): one cache, DrWatson's produce_or_load, keyed by only what a product reads.

# produce_or_load keyed by only what a product reads
@testitem "products" setup = [GitFixture] begin
    import VisualGeometryToolkitEval as VgtkEval
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using Test, TOML, StaticArrays
    const E = VgtkEval.Experiments

    # a committed project (its src a code tree) and a results root of its own
    function project()
        repo = gitrepo()
        mkpath(joinpath(repo, "src")); write(joinpath(repo, "src", "M.jl"), "module M end\n")
        git!(repo, "add", "-A"); git!(repo, "commit", "-qm", "code")
        E.set_project_root!(repo)
        return repo
    end
    pose(t) = VgtkCore.relabel(VgtkCore.WorldFrame, VgtkCore.ViewFrame{:cam},
                               VgtkCore.with_unit(VgtkCore.EuclideanMap(SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1), SVector(t, 0.0, 0.0)), VgtkCore.m))
    x(p) = VgtkCore.ustrip(VgtkCore.m, p).t
    args = (; driver = "scripts/x.jl", config_source = "configs/x.toml")

    @testset "computed once, then loaded; the record through VisualGeometryToolkitIO; DrWatson's layout" begin
        repo = project()
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
            s = E.Product("calibrate", "start", Dict("pipeline" => Dict("grid" => Dict("size" => [35, 25]))); output = "poses.json")
            @test s.key == E.run_key(Dict("pipeline" => Dict("grid" => Dict("size" => [35, 25]))))
            @test E.product_dir(s) == joinpath(E.results_root(), "calibrate", "start", s.key)    # datadir("exp_pro", experiment, product, key)
            @test !E.produced(s)
            n = Ref(0)
            dir = E.produce(d -> (n[] += 1; [pose(1.0), pose(2.0)]), s; args...)
            @test dir == E.product_dir(s) && n[] == 1 && E.produced(s)
            @test E.produce(d -> (n[] += 1; error("must not run")), s; args...) == dir && n[] == 1   # loaded, not produced
            ps = E.load_output(Vector{VgtkCore.EuclideanMap}, s)                               # VisualGeometryToolkitIO.load
            @test length(ps) == 2 && x(ps[2]) ≈ SVector(2.0, 0.0, 0.0)
            @test TOML.parsefile(joinpath(dir, "_config.toml"))["pipeline"]["grid"]["size"] == [35, 25]
            prov = E.read_provenance(dir)
            @test prov["run_key"] == s.key && prov["driver"] == "scripts/x.jl" && haskey(prov, "code_trees")
            # a Saved keeps its keywords (VisualGeometryToolkitIO.save's)
            s2 = E.Product("calibrate", "poses", Dict("a" => 1); output = "poses.json")
            E.produce(d -> E.Saved([pose(3.0)]; provenance = (; source = "test")), s2; args...)
            @test x(E.load_output(Vector{VgtkCore.EuclideanMap}, s2)[1]) ≈ SVector(3.0, 0.0, 0.0)
            # force produces again
            E.produce(d -> (n[] += 1; [pose(1.0)]), s; args..., force = true)
            @test n[] == 2
        end
        E.set_project_root!("")
    end

    @testset "the key is only what the product reads; an upstream product's key folds in" begin
        project()
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
            start(size) = E.Product("calibrate", "start", Dict("grid" => Dict("size" => size)); output = "s.json")
            refine(st, iters) = E.Product("calibrate", "refine", Dict("lm" => Dict("max_iter" => iters)); output = "r.json", upstream = ("start" => st,))
            a, b = start([35, 25]), start([35, 25])
            @test a.key == b.key && start([36, 25]).key != a.key
            # a refine-only change: the same start, another refinement
            @test refine(a, 24).key != refine(a, 50).key && start([35, 25]).key == a.key
            # a start change: the refinement re-keys with it
            @test refine(start([36, 25]), 24).key != refine(a, 24).key
            # an upstream given by its key is the same fold
            @test E.Product("calibrate", "refine", Dict("lm" => Dict("max_iter" => 24)); output = "r.json", upstream = ("start" => a.key,)).key == refine(a, 24).key
            @test_throws ArgumentError E.Product("calibrate", "re/fine", Dict(); output = "r.json")
            @test_throws ArgumentError E.Product("../x", "refine", Dict(); output = "r.json")
            @test_throws ArgumentError E.Product("x", "refine", Dict(); output = "../r.json")
            @test E.product_dir(E.Product("ETH/sift/H/fig4", "cv", Dict(); output = "r.json")) ==
                  joinpath(E.results_root(), "ETH", "sift", "H", "fig4", "cv", E.run_key(Dict()))   # a deeper layout
        end
        E.set_project_root!("")
    end

    @testset "a product that writes its own output; a failure never reads as produced; a failed save is an error" begin
        project()
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
            s = E.Product("x", "detect", Dict("k" => 1); output = "summary.tsv")
            dir = E.produce(s; args...) do d
                mkpath(joinpath(d, "detections")); write(joinpath(d, "detections", "a.tsv"), "1\n")
                write(joinpath(d, "summary.tsv"), "n\n1\n"); nothing                            # the file it writes last
            end
            @test E.produced(s) && isfile(joinpath(dir, "detections", "a.tsv"))
            # the inputs as a function: listed once the product ran, recorded in its provenance
            w = E.Product("x", "detect", Dict("k" => 5); output = "summary.tsv")
            inp = joinpath(mktempdir(), "photos"); mkpath(inp)
            E.produce(d -> (write(joinpath(d, "summary.tsv"), "n\n"); nothing), w; args..., inputs = () -> [inp])
            @test E.provenance_inputs(E.read_provenance(E.product_dir(w))) == [inp]
            @test E.produce(d -> error("must not run"), w; args..., inputs = () -> error("a produced product lists no inputs")) == E.product_dir(w)
            # f returns nothing without writing its output: an error, and no provenance
            t = E.Product("x", "detect", Dict("k" => 2); output = "summary.tsv")
            @test_throws ErrorException E.produce(d -> nothing, t; args...)
            @test !isfile(joinpath(E.product_dir(t), "_provenance.toml")) && !E.produced(t)
            # f throws after writing its output: the product is not produced (the provenance is written last)
            u = E.Product("x", "detect", Dict("k" => 3); output = "summary.tsv")
            @test_throws "crash" E.produce(d -> (write(joinpath(d, "summary.tsv"), "partial"); error("crash")), u; args...)
            @test !E.produced(u)
            # a product produced, then forced and failing: its old provenance is gone, so it never reads as produced
            @test_throws "crash" E.produce(d -> error("crash"), s; args..., force = true)
            @test !E.produced(s)
            # a record VisualGeometryToolkitIO cannot save: produce_or_load swallows the error, produce throws
            v = E.Product("x", "fit", Dict("k" => 4); output = "fit.json")
            @test_throws r"was not saved" E.produce(d -> 42, v; args...)
            @test !E.produced(v)
        end
        E.set_project_root!("")
    end

    @testset "reused only while its code trees are the current ones; a scratch run keys apart" begin
        repo = project()
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
            s = E.Product("x", "fit", Dict("k" => 1); output = "fit.json")
            n = Ref(0)
            E.produce(d -> (n[] += 1; [pose(1.0)]), s; args...)
            write(joinpath(repo, "a.txt"), "docs only\n"); git!(repo, "commit", "-qam", "docs")
            @test E.produced(s)                                                                  # no code tree changed
            write(joinpath(repo, "src", "M.jl"), "module M\nf() = 1\nend\n"); git!(repo, "commit", "-qam", "src")
            @test !(@test_logs (:info, r"its code changed") E.produced(s))
            E.produce(d -> (n[] += 1; [pose(1.0)]), s; args...)
            @test n[] == 2 && E.produced(s)
            sc = withenv(() -> E.Product("x", "fit", Dict("k" => 1); output = "fit.json"), "VGTK_SCRATCH" => "1")
            @test sc.key != s.key                                                                # never a run of record's directory
            # uncommitted code: a run of record refuses before computing
            write(joinpath(repo, "src", "M.jl"), "module M\nf() = 2\nend\n")
            @test_throws r"a run of record runs committed code" E.produce(d -> error("must not compute"), E.Product("x", "fit", Dict("k" => 9); output = "fit.json"); args...)
        end
        E.set_project_root!("")
    end

    @testset "the retired names are refused by name with their replacement" begin
        for (f, new) in ((E.fresher, "Stage(name, desc; products, run)"), (E.run_complete, "produced(product)"), (E.cached_result, "produce(f, Product"),
                         (E.legacy_hash_key, "run_key"), (E.keyed_cell_dir, "product_dir(Product"), (E.keyed_data_dir, "promote_run(product_dir"),
                         (E.experiment_dir, "Product(joinpath("))
            e = try f(Dict(), "a", "b"); nothing catch err; err end
            @test e isa ArgumentError && occursin("is retired", e.msg) && occursin(new, e.msg)
        end
    end
end
