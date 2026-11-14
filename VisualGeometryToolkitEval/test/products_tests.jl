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
            @test s.key == E.run_key(Dict("pipeline" => Dict("grid" => Dict("size" => [35, 25]))); code = E.current_code_trees(E.project_root()))
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
            # a Produced keeps the save's keywords (VisualGeometryToolkitIO.save's) and the inputs f read
            s2 = E.Product("calibrate", "poses", Dict("a" => 1); output = "poses.json")
            inp = mktempdir()
            E.produce(d -> E.Produced([pose(3.0)]; inputs = [inp], provenance = (; source = "test")), s2; args...)
            @test inp in E.provenance_inputs(E.read_provenance(E.product_dir(s2)))
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
            # an upstream given by its key is the same fold while it is not produced (a bare key folds no output hash)
            @test E.Product("calibrate", "refine", Dict("lm" => Dict("max_iter" => 24)); output = "r.json", upstream = ("start" => a.key,)).key == refine(a, 24).key
            @test_throws ArgumentError E.Product("calibrate", "re/fine", Dict(); output = "r.json")
            @test_throws ArgumentError E.Product("../x", "refine", Dict(); output = "r.json")
            @test_throws ArgumentError E.Product("x", "refine", Dict(); output = "../r.json")
            @test_throws ArgumentError E.Product("x", "refine", Dict(); output = ".")      # never the product's whole directory
            @test E.product_dir(E.Product("ETH/sift/H/fig4", "cv", Dict(); output = "r.json")) ==
                  joinpath(E.results_root(), "ETH", "sift", "H", "fig4", "cv", E.run_key(Dict(); code = E.current_code_trees(E.project_root())))   # a deeper layout
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
            # the inputs f read, returned with nothing (f wrote its output): recorded in its provenance
            w = E.Product("x", "detect", Dict("k" => 5); output = "summary.tsv")
            inp = joinpath(mktempdir(), "photos"); mkpath(inp)
            E.produce(d -> (write(joinpath(d, "summary.tsv"), "n\n"); E.Produced(; inputs = [inp])), w; args...)
            @test E.provenance_inputs(E.read_provenance(E.product_dir(w))) == [inp]
            @test E.produce(d -> error("must not run"), w; args...) == E.product_dir(w)
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
            # … and a forced re-production whose save fails never leaves the earlier output standing as this one's
            E.produce(d -> [pose(1.0)], v; args...)
            @test E.produced(v)
            @test_throws r"was not saved" E.produce(d -> 42, v; args..., force = true)
            @test !E.produced(v) && !isfile(E.product_output(v))
        end
        E.set_project_root!("")
    end

    @testset "the code it runs keys it: a code change is another product, the old one kept; a scratch run keys apart" begin
        repo = project()
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
            fit() = E.Product("x", "fit", Dict("k" => 1); output = "fit.json")
            s = fit()
            @test s.key == E.run_key(Dict("k" => 1); code = E.current_code_trees(repo))       # the code trees in the key
            n = Ref(0)
            E.produce(d -> (n[] += 1; [pose(1.0)]), s; args...)
            write(joinpath(repo, "a.txt"), "docs only\n"); git!(repo, "commit", "-qam", "docs")
            @test fit().key == s.key && E.produced(fit())                                       # no code tree changed: the same product
            write(joinpath(repo, "src", "M.jl"), "module M\nf() = 1\nend\n"); git!(repo, "commit", "-qam", "src")
            s2 = fit()
            @test s2.key != s.key && !E.produced(s2) && E.produced(s)                           # another product; the old one stays
            E.produce(d -> (n[] += 1; [pose(1.0)]), s2; args...)
            @test n[] == 2 && E.produced(s2) && isdir(E.product_dir(s))
            sc = withenv(() -> E.Product("x", "fit", Dict("k" => 1); output = "fit.json"), "VGTK_SCRATCH" => "1")
            @test sc.key != s.key                                                                # never a run of record's directory
            # uncommitted code: a run of record refuses before computing
            write(joinpath(repo, "src", "M.jl"), "module M\nf() = 2\nend\n")
            @test_throws r"a run of record runs committed code" E.produce(d -> error("must not compute"), E.Product("x", "fit", Dict("k" => 9); output = "fit.json"); args...)
        end
        E.set_project_root!("")
    end

    # a record whose save writes part of its file and then fails (a full disk)
    struct Boom end
    VgtkIO.save(file::AbstractString, ::Boom; kwargs...) = (write(file, "partial"); error("disk full"))

    @testset "an upstream produced again with another output re-keys its consumers; with the same output it does not" begin
        project()
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
            up = E.Product("x", "simulate", Dict("n" => 3); output = "samples.tsv")
            fit() = E.Product("x", "fit", Dict{String, Any}(); output = "fit.toml", upstream = ("simulate" => up,))
            samples(text) = d -> (write(joinpath(d, "samples.tsv"), text); nothing)
            E.produce(samples("1\n2\n"), up; args...)
            haskey(E.read_provenance(E.product_dir(up)), "output_sha256") || error("no output hash recorded")
            n = Ref(0)
            fitted = d -> (n[] += 1; write(joinpath(d, "fit.toml"), "n = $(n[])\n"); nothing)
            f1 = fit(); E.produce(fitted, f1; args...)
            @test E.produced(f1) && n[] == 1
            E.produce(samples("1\n5\n"), up; args..., force = true)                    # the same key, another output
            f2 = fit()
            @test f2.key != f1.key && !E.produced(f2)                                         # the consumer is produced again
            E.produce(fitted, f2; args...)
            @test n[] == 2
            E.produce(samples("1\n5\n"), up; args..., force = true)                    # produced again, the same output
            @test fit().key == f2.key && E.produced(fit())
            # a consumer built BEFORE its upstream is produced again: `produced` checks the upstream's outputs as it read them
            f3 = fit()
            E.produce(samples("9\n9\n"), up; args..., force = true)
            @test !E.produced(f3)
        end
        E.set_project_root!("")
    end

    @testset "a save that throws leaves no output and the product not produced (atomic: a temporary file renamed)" begin
        project()
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
            s = E.Product("x", "boom", Dict("k" => 1); output = "boom.json")
            @test_throws Exception E.produce(d -> Boom(), s; args...)                          # its save writes a part, then throws
            @test !E.produced(s) && !ispath(E.product_output(s))
            @test isempty(filter(f -> startswith(f, ".partial-"), readdir(E.product_dir(s))))
        end
        E.set_project_root!("")
    end

    @testset "a .jld2 output: a String- or a Symbol-keyed Dict saved and read back as it was (never VisualGeometryToolkitIO's)" begin
        project()
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
            ds = Dict{String, Any}("v" => [1.0, 2.0], "name" => "a")
            s = E.Product("x", "cv", Dict("k" => 1); output = "r.jld2")
            E.produce(d -> ds, s; args...)
            @test E.produced(s) && E.load_output(Dict{String, Any}, s) == ds
            dy = Dict{Symbol, Int}(:a => 1, :b => 2)
            t = E.Product("x", "cv", Dict("k" => 2); output = "r.jld2")
            E.produce(d -> dy, t; args...)
            @test E.load_output(Dict{Symbol, Int}, t) == dy
            # a concrete Dict reads back as the Dict{String, Any} a cached_result caller expects
            dv = Dict("err" => [0.1, 0.2])
            u = E.Product("x", "cv", Dict("k" => 3); output = "r.jld2")
            E.produce(d -> dv, u; args...)
            @test E.load_output(Dict{String, Any}, u) == Dict{String, Any}("err" => [0.1, 0.2])
        end
        E.set_project_root!("")
    end

    @testset "a directory output: produced while it is there, removed whole when produced again" begin
        project()
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
            s = E.Product("x", "model", Dict("k" => 1); output = "model")
            write_model(names) = d -> (m = mkpath(joinpath(d, "model")); foreach(f -> write(joinpath(m, f), f), names); nothing)
            E.produce(write_model(["a.txt", "b.txt"]), s; args...)
            @test E.produced(s)
            E.produce(write_model(["a.txt"]), s; args..., force = true)
            @test E.produced(s) && readdir(E.product_output(s)) == ["a.txt"]
        end
        E.set_project_root!("")
    end

    @testset "the retired names are refused by name with their replacement" begin
        for (f, new) in ((E.fresher, "Stage(name, desc; items, product, run)"), (E.run_complete, "produced(product)"), (E.cached_result, "produce(_ -> f(cfg), p"),
                         (E.legacy_hash_key, "run_key"), (E.keyed_cell_dir, "product_dir(Product"), (E.keyed_data_dir, "promote_run(product_dir"),
                         (E.experiment_dir, "Product(joinpath("))
            e = try f(Dict(), "a", "b"); nothing catch err; err end
            @test e isa ArgumentError && occursin("is retired", e.msg) && occursin(new, e.msg)
        end
    end
end
