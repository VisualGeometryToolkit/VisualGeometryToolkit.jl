# The published-data catalog (src/DataArtifacts.jl): the catalog checks, the
# derived file lists, the author-side pack/bind and the consumer fetch into
# the artifact view — hermetic: a throwaway project, a throwaway artifacts
# directory (never the depot) and file:// bindings.

@testitem "data_artifacts" begin
    using Test, TOML
    using Artifacts: with_artifacts_directory
    using VisualGeometryEval
    const E = VisualGeometryEval.Experiments
    const D = VisualGeometryEval.DataArtifacts

    @testset "split_scenes" begin
        @test length(D.split_scenes("heb:validation")) == 5 && length(D.split_scenes("heb:test")) == 5
        @test isempty(intersect(D.split_scenes("phototourism:validation"), D.split_scenes("phototourism:test")))
        @test_throws ArgumentError D.split_scenes("heb")
        @test_throws ArgumentError D.split_scenes("heb:train")
    end

    @testset "pt_scene_files: the reader's path rule" begin
        t = VisualGeometryEval.Datasets.PtRawTrees(; matches_tutorial_val = "mv", matches_tutorial_train = "mt",
            matches_ef_test = "me", recon_tutorial = "rt", recon_ef_test = "re",
            recon_special = Dict("reichstag" => "special/reichstag"))
        f = VisualGeometryEval.Datasets.pt_scene_files("sacre_coeur", t)
        @test f == [joinpath("mv", "sacre_coeur", "matches.h5"), joinpath("mv", "sacre_coeur", "match_conf.h5"),
                    (joinpath("rt", "sacre_coeur", "dense", "sparse", "$b.bin") for b in ("cameras", "images", "points3D"))...]
        @test VisualGeometryEval.Datasets.pt_scene_files("reichstag", t)[3] == joinpath("special", "reichstag", "cameras.bin")
        @test_throws ArgumentError VisualGeometryEval.Datasets.pt_scene_files("atlantis", t)
    end

    # A project: an author's data and results trees, two configs of record, a catalog.
    proj = mktempdir()
    author_data, author_results = mktempdir(), mktempdir()
    hv, ht = D.split_scenes("heb:validation"), D.split_scenes("heb:test")
    for s in vcat(hv, ht)
        mkpath(joinpath(author_data, "raw", "heb"))
        write(joinpath(author_data, "raw", "heb", "$(s)_homographies.h5"), s)
    end
    mkpath(joinpath(author_results, "stored", "sweep", "sub"))
    write(joinpath(author_results, "stored", "sweep", "SHA256SUMS"), "x\n")
    write(joinpath(author_results, "stored", "sweep", "sub", "zeta.jld2"), "z")
    mkpath(joinpath(author_data, "extra"))
    write(joinpath(author_data, "extra", "notes.txt"), "n")
    mkpath(joinpath(proj, "configs"))
    write(joinpath(proj, "configs", "heb_pools.toml"), """
        [experiment]
        dataset = "heb"
        matcher = "sift"
        problem = "H"
        pool_dir = "pools/heb"
        [source]
        raw_dir = "raw/heb"
        """)
    write(joinpath(proj, "configs", "sweep.toml"), """
        [experiment]
        dataset = "heb"
        matcher = "sift"
        problem = "H"
        pool_dir = "pools/heb"
        [stored]
        dir = "stored/sweep"
        """)
    catalog = joinpath(proj, "configs", "data_artifacts.toml")
    write(catalog, """
        [catalog]
        artifacts_toml_env = "VGE_TEST_ARTIFACTS_TOML"

        [records.inputs]
        title = "inputs"
        placeholder = "ZENODO_RECORD_INPUTS"

        [artifacts.t_heb_val]
        kind = "pool_sources"
        config = "heb_pools"
        splits = ["heb:validation"]
        mount = "data"
        record = "inputs"
        required = true
        what = "HEB validation"
        provenance = "test"

        [artifacts.t_stored]
        kind = "stored"
        config = "sweep"
        mount = "results"
        record = "inputs"
        required = true
        what = "stored"
        provenance = "test"

        [artifacts.t_extra]
        kind = "tree"
        dir = "extra"
        mount = "data"
        record = "inputs"
        required = false
        what = "extra"
        provenance = "test"
        """)

    withenv("CAU_DATA_ROOT" => author_data, "CAU_RESULTS_ROOT" => author_results,
            "VGE_TEST_ARTIFACTS_TOML" => nothing) do
        E.set_project_root!(proj)
        cat = D.load_catalog(catalog)
        @test cat.artifacts_toml == joinpath(proj, "Artifacts.toml")
        @test D.required_artifacts(cat) == ["t_heb_val", "t_stored"]
        @test D.catalog_violations(cat) == ["no Artifacts.toml at $(joinpath(proj, "Artifacts.toml"))"]

        @testset "derived file lists" begin
            a = D.artifact_files(cat, "t_heb_val")
            @test a.mount == "data" && a.root == author_data
            @test a.files == sort([joinpath("raw", "heb", "$(s)_homographies.h5") for s in hv])
            @test D.artifact_files(cat, "t_stored").files ==
                  [joinpath("stored", "sweep", "SHA256SUMS"), joinpath("stored", "sweep", "sub", "zeta.jld2")]
            @test D.artifact_files(cat, "t_extra").files == [joinpath("extra", "notes.txt")]
            # a project kind
            k = Dict("tree" => (cfg, e, root) -> ["extra/notes.txt", "extra/notes.txt"])
            @test D.artifact_files(cat, "t_extra"; kinds = k).files == ["extra/notes.txt"]
            @test_throws ErrorException D.artifact_files(cat, "t_extra"; kinds = Dict("tree" => (c, e, r) -> ["nope"]))
            @test_throws ErrorException D.artifact_files(cat, "t_zz")
        end

        @testset "pack, bind, check" begin
            stage = mktempdir()
            built = D.pack_artifacts(cat; stage, record_ids = Dict("inputs" => "12345"),
                                     local_toml = joinpath(stage, "local.toml"))
            @test sort([b.name for b in built]) == ["t_extra", "t_heb_val", "t_stored"]
            @test all(b -> isfile(b.tarball) && occursin(r"^[0-9a-f]{64}$", b.sha), built)
            bound = TOML.parsefile(cat.artifacts_toml)
            @test only(bound["t_stored"]["download"])["url"] == "https://zenodo.org/records/12345/files/t_stored.tar.gz?download=1"
            # the bindings must name the catalog's record: an id the catalog does not carry is flagged
            @test length(D.catalog_violations(cat)) == 3
            withid = D.DataCatalog(cat.path, cat.artifacts_toml, cat.configs,
                                   Dict{String,Any}("inputs" => Dict{String,Any}("placeholder" => "ZENODO_RECORD_INPUTS", "id" => "12345")),
                                   cat.artifacts)
            @test isempty(D.catalog_violations(withid))
            @test D.zenodo_url(withid, "t_extra") == "https://zenodo.org/records/12345/files/t_extra.tar.gz?download=1"
            @test D.zenodo_url(withid, "t_extra"; record_ids = Dict("inputs" => "9")) == "https://zenodo.org/records/9/files/t_extra.tar.gz?download=1"
            # placeholder URLs satisfy the check; a file:// URL in the committed bindings does not
            D.pack_artifacts(cat; stage)      # reuses the staged tarballs (listings current)
            @test occursin("ZENODO_RECORD_INPUTS", only(TOML.parsefile(cat.artifacts_toml)["t_extra"]["download"])["url"])
            @test isempty(D.catalog_violations(cat))
            withenv("VGE_TEST_ARTIFACTS_TOML" => joinpath(stage, "local.toml")) do
                @test !isempty(D.catalog_violations(D.load_catalog(catalog)))
            end
            # the same tree hashes when repacked from scratch
            built2 = D.pack_artifacts(cat, ["t_extra"]; stage = mktempdir())
            @test only(built2).hash == only(filter(b -> b.name == "t_extra", built)).hash

            @testset "fetch into a fresh machine's project" begin
                fresh = mktempdir()
                mkpath(joinpath(fresh, "configs"))
                cp(catalog, joinpath(fresh, "configs", "data_artifacts.toml"))
                withenv("CAU_DATA_ROOT" => nothing, "CAU_RESULTS_ROOT" => nothing,
                        "VGE_TEST_ARTIFACTS_TOML" => joinpath(stage, "local.toml")) do
                    with_artifacts_directory(mktempdir()) do      # never the depot
                        E.set_project_root!(fresh)
                        fcat = D.load_catalog(joinpath(fresh, "configs", "data_artifacts.toml"))
                        r = D.fetch_artifacts!(fcat)
                        @test [x.name for x in r] == ["t_heb_val", "t_stored"]
                        view = joinpath(fresh, "data", "exp_raw")
                        @test E.data_root() == view && E.is_artifact_view(view)
                        @test all(s -> isfile(joinpath(view, "raw", "heb", "$(s)_homographies.h5")), hv)
                        @test !ispath(joinpath(view, "raw", "heb", "$(first(ht))_homographies.h5"))
                        @test read(joinpath(E.results_root(), "stored", "sweep", "sub", "zeta.jld2"), String) == "z"
                        @test all(x -> x.linked == 0, D.fetch_artifacts!(fcat))       # idempotent
                        @test only(D.fetch_artifacts!(fcat, ["t_extra"])).linked == 1
                        # an author's own data/exp_raw is never written
                        own = mktempdir()
                        rm(view; recursive = true); symlink(own, view)
                        @test only(D.fetch_artifacts!(fcat, ["t_heb_val"])).dst === nothing
                        @test isempty(readdir(own))
                    end
                end
                # an unpublished record is refused with the way out
                withenv("CAU_DATA_ROOT" => nothing, "VGE_TEST_ARTIFACTS_TOML" => nothing) do
                    with_artifacts_directory(mktempdir()) do
                        E.set_project_root!(mktempdir())
                        err = try D.fetch_artifacts!(cat, ["t_extra"]); nothing catch e; e end
                        @test err isa ErrorException && occursin("not published", err.msg)
                    end
                end
            end
        end
        E.set_project_root!("")
    end

    # An entry too large for one upload, bound one artifact per scene plus its meta part.
    @testset "split_by = scene" begin
        proj = mktempdir()
        author_data = mktempdir()
        sc = sort(vcat(D.split_scenes("heb:validation"), D.split_scenes("heb:test")))
        pd = joinpath(author_data, "pools", "heb")
        mkpath(pd)
        write(joinpath(pd, "_config.toml"), "c"); write(joinpath(pd, "_provenance.toml"), "p")
        write(joinpath(pd, "_dirty.patch"), "never shipped")
        for s in sc
            write(joinpath(pd, "$s.jld2"), s); write(joinpath(pd, "$s.residuals.jld2"), s^2)
        end
        mkpath(joinpath(proj, "configs"))
        write(joinpath(proj, "configs", "heb_pools.toml"), """
            [experiment]
            dataset = "heb"
            matcher = "sift"
            problem = "H"
            pool_dir = "pools/heb"
            [source]
            raw_dir = "raw/heb"
            """)
        catalog = joinpath(proj, "configs", "data_artifacts.toml")
        write(catalog, """
            [catalog]
            artifacts_toml_env = "VGE_TEST_ARTIFACTS_TOML"
            [records.inputs]
            title = "inputs"
            placeholder = "ZENODO_RECORD_INPUTS"
            id = "12345"
            [artifacts.t_pools]
            kind = "pools"
            config = "heb_pools"
            splits = ["heb:validation", "heb:test"]
            split_by = "scene"
            mount = "data"
            record = "inputs"
            required = true
            what = "pools"
            provenance = "test"
            """)
        withenv("CAU_DATA_ROOT" => author_data, "VGE_TEST_ARTIFACTS_TOML" => nothing) do
            E.set_project_root!(proj)
            cat = D.load_catalog(catalog)
            parts = D.artifact_parts(cat, "t_pools")
            @test parts == vcat(["t_pools_meta"], ["t_pools_$s" for s in sc])
            @test D.artifact_files(cat, "t_pools_meta").files == ["pools/heb/_config.toml", "pools/heb/_provenance.toml"]
            @test D.artifact_files(cat, "t_pools_$(sc[1])").files == ["pools/heb/$(sc[1]).jld2", "pools/heb/$(sc[1]).residuals.jld2"]
            # the whole entry = the union of its parts; no source-tree diff
            whole = D.artifact_files(cat, "t_pools").files
            @test whole == sort(reduce(vcat, (D.artifact_files(cat, p).files for p in parts)))
            @test !any(endswith("_dirty.patch"), whole)
            @test D.zenodo_url(cat, "t_pools_meta") == "https://zenodo.org/records/12345/files/t_pools_meta.tar.gz?download=1"
            @test_throws ErrorException D.artifact_files(cat, "t_pools_atlantis")
            stage = mktempdir()
            built = D.pack_artifacts(cat; stage, local_toml = joinpath(stage, "local.toml"))
            @test sort([b.name for b in built]) == sort(parts)
            @test sort(collect(keys(TOML.parsefile(cat.artifacts_toml)))) == sort(parts)
            @test isempty(D.catalog_violations(cat))
            # a missing part binding is flagged
            b = TOML.parsefile(cat.artifacts_toml); delete!(b, "t_pools_meta")
            open(io -> TOML.print(io, b), cat.artifacts_toml, "w")
            @test D.catalog_violations(cat) == ["t_pools_meta: catalogued but not bound"]
            bad = D.DataCatalog(cat.path, cat.artifacts_toml, cat.configs, cat.records,
                                Dict{String,Any}(cat.artifacts..., "t_tree" => Dict{String,Any}("kind" => "tree", "dir" => "pools",
                                    "split_by" => "scene", "mount" => "data", "record" => "inputs", "required" => false,
                                    "what" => "x", "provenance" => "test")))
            @test any(contains("t_tree: split_by needs kind pool_sources | pools"), D.catalog_violations(bad))
            D.pack_artifacts(cat; stage)
            fresh = mktempdir(); mkpath(joinpath(fresh, "configs"))
            cp(catalog, joinpath(fresh, "configs", "data_artifacts.toml"))
            cp(joinpath(proj, "configs", "heb_pools.toml"), joinpath(fresh, "configs", "heb_pools.toml"))
            withenv("CAU_DATA_ROOT" => nothing, "VGE_TEST_ARTIFACTS_TOML" => joinpath(stage, "local.toml")) do
                with_artifacts_directory(mktempdir()) do
                    E.set_project_root!(fresh)
                    fcat = D.load_catalog(joinpath(fresh, "configs", "data_artifacts.toml"))
                    r = D.fetch_artifacts!(fcat)
                    @test [x.name for x in r] == parts
                    view = joinpath(fresh, "data", "exp_raw")
                    @test all(s -> read(joinpath(view, "pools", "heb", "$s.residuals.jld2"), String) == s^2, sc)
                    @test isfile(joinpath(view, "pools", "heb", "_config.toml"))
                    @test !ispath(joinpath(view, "pools", "heb", "_dirty.patch"))
                    @test [x.name for x in D.fetch_artifacts!(fcat, ["t_pools_$(sc[2])"])] == ["t_pools_$(sc[2])"]
                end
            end
            E.set_project_root!("")
        end
    end

    # A scene over the part cap: sub-parts by the rule, chunks reassembled on fetch.
    @testset "max_part_bytes: sub-parts" begin
        proj = mktempdir()
        author_data = mktempdir()
        sc = sort(D.split_scenes("heb:validation"))
        big = first(sc)
        pd = joinpath(author_data, "pools", "heb")
        mkpath(pd)
        write(joinpath(pd, "_config.toml"), "c"); write(joinpath(pd, "_provenance.toml"), "p")
        payload = rand(UInt8, 1000)
        for s in sc
            write(joinpath(pd, "$s.jld2"), s)
            write(joinpath(pd, "$s.residuals.jld2"), s == big ? payload : Vector{UInt8}(s))
        end
        mkpath(joinpath(proj, "configs"))
        write(joinpath(proj, "configs", "heb_pools.toml"), """
            [experiment]
            dataset = "heb"
            matcher = "sift"
            problem = "H"
            pool_dir = "pools/heb"
            [source]
            raw_dir = "raw/heb"
            """)
        catalog = joinpath(proj, "configs", "data_artifacts.toml")
        write(catalog, """
            [catalog]
            artifacts_toml_env = "VGE_TEST_ARTIFACTS_TOML"
            [records.inputs]
            title = "inputs"
            placeholder = "ZENODO_RECORD_INPUTS"
            id = "12345"
            [artifacts.t_pools]
            kind = "pools"
            config = "heb_pools"
            splits = ["heb:validation"]
            split_by = "scene"
            max_part_bytes = 300
            mount = "data"
            record = "inputs"
            required = true
            what = "pools"
            provenance = "test"
            """)
        withenv("CAU_DATA_ROOT" => author_data, "VGE_TEST_ARTIFACTS_TOML" => nothing) do
            E.set_project_root!(proj)
            cat = D.load_catalog(catalog)
            parts = D.artifact_parts(cat, "t_pools")
            # 1000 bytes over a 300-byte cap: 4 chunks of 250; the light file rides with chunk 1
            subs = ["t_pools_$(big)_part$i" for i in 1:4]
            @test parts == vcat(["t_pools_meta"], subs, ["t_pools_$s" for s in sc if s != big])
            p1 = D.artifact_files(cat, subs[1])
            @test p1.files == ["pools/heb/$big.jld2", "pools/heb/$big.residuals.jld2.chunk1of4"]
            @test [p.len for p in p1.pieces] == [length(big), 250]
            @test all(p -> sum(q -> q.len, D.artifact_files(cat, p).pieces) <= 300, parts)
            chunks = reduce(vcat, (D.artifact_files(cat, p).pieces for p in subs))
            @test sort([(c.offset, c.len) for c in chunks if c.src == "pools/heb/$big.residuals.jld2"]) ==
                  [(0, 250), (250, 250), (500, 250), (750, 250)]
            stage = mktempdir()
            built = D.pack_artifacts(cat; stage, local_toml = joinpath(stage, "local.toml"))
            @test sort([b.name for b in built]) == sort(parts)
            @test isempty(D.catalog_violations(cat))
            @test D.pack_artifacts(cat, [subs[2]]; stage)[1].hash == only(b for b in built if b.name == subs[2]).hash   # reused, same tree
            bad = D.DataCatalog(cat.path, cat.artifacts_toml, cat.configs, cat.records,
                                Dict{String,Any}(cat.artifacts..., "t_x" => Dict{String,Any}("kind" => "tree", "dir" => "pools",
                                    "max_part_bytes" => 10, "mount" => "data", "record" => "inputs", "required" => false,
                                    "what" => "x", "provenance" => "test")))
            @test any(contains("t_x: max_part_bytes needs split_by"), D.catalog_violations(bad))
            fresh = mktempdir(); mkpath(joinpath(fresh, "configs"))
            cp(catalog, joinpath(fresh, "configs", "data_artifacts.toml"))
            cp(joinpath(proj, "configs", "heb_pools.toml"), joinpath(fresh, "configs", "heb_pools.toml"))
            withenv("CAU_DATA_ROOT" => nothing, "VGE_TEST_ARTIFACTS_TOML" => joinpath(stage, "local.toml")) do
                with_artifacts_directory(mktempdir()) do
                    E.set_project_root!(fresh)
                    fcat = D.load_catalog(joinpath(fresh, "configs", "data_artifacts.toml"))
                    @test D.artifact_parts(fcat, "t_pools") == parts          # a consumer: from the bindings
                    r = D.fetch_artifacts!(fcat)
                    @test [x.name for x in r] == parts
                    view = joinpath(fresh, "data", "exp_raw")
                    @test read(joinpath(view, "pools", "heb", "$big.residuals.jld2")) == payload   # reassembled
                    @test !islink(joinpath(view, "pools", "heb", "$big.residuals.jld2"))
                    @test read(joinpath(view, "pools", "heb", "$big.jld2"), String) == big
                    @test all(x -> x.linked == 0, D.fetch_artifacts!(fcat))                       # idempotent
                    @test read(joinpath(view, "pools", "heb", "$big.residuals.jld2")) == payload
                end
            end
            E.set_project_root!("")
        end
    end
end
