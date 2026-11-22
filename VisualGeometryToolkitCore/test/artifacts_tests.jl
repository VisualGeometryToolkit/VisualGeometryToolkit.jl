# The artifact store and the one install of a published artifact (src/base/artifacts.jl), hoisted from
# VisualGeometryToolkitEval's Experiments (which re-exports install_artifact) for BlobBoards, which has no Eval
# dependency: the store's resolution order and its refusal on the cluster, the install's refusals and its hash checks,
# the per-tree lock across processes, a single file in the store, the depot → store migration, and the Hugging Face
# token read from the environment first, then from the file `hf auth login` writes.

@testsetup module StoreFixture
    import VisualGeometryToolkitCore as VgtkCore
    using Pkg.Artifacts: create_artifact, archive_artifact, bind_artifact!
    export NO_STORE, published_tree
    # The store variables unset: every rung below is the test's own; no Pkg server (the trees are file:// tarballs,
    # which Pkg would otherwise first ask the server for).
    const NO_STORE = ("VGTK_ARTIFACT_STORE" => nothing, "VGTK_SITE" => nothing, "NODE_CACHE" => nothing,
                      "JULIA_PKG_SERVER" => "")
    # A published tree: built in a scratch store, archived to a tarball and bound with its file:// URL (the sha256
    # given, or the tarball's own). Returns the tree hash and the Artifacts.toml.
    function published_tree(dir, name, files; sha = nothing, toml = joinpath(dir, "Artifacts.toml"))
        scratch = joinpath(dir, "build-$name")
        tarball = joinpath(dir, "$name.tar.gz")
        h = VgtkCore.with_artifact_store(scratch) do
            h = create_artifact(d -> foreach(((f, s),) -> (mkpath(dirname(joinpath(d, f))); write(joinpath(d, f), s)), files))
            (h, archive_artifact(h, tarball))
        end
        bind_artifact!(toml, name, h[1]; lazy = true, force = true,
                       download_info = [("file://" * tarball, something(sha, h[2]))])
        return h[1], toml
    end
end

@testitem "install_artifact: the one artifact install (refusals, the Hugging Face token)" setup = [StoreFixture] begin
    using Test, TOML
    import VisualGeometryToolkitCore as VgtkCore
    using Pkg.Artifacts: create_artifact, bind_artifact!
    using Base: SHA1
    dir = mktempdir()
    toml = joinpath(dir, "Artifacts.toml")
    write(toml, """
        [tier_c]
        git-tree-sha1 = "$("1"^40)"
        lazy = true

            [[tier_c.download]]
            url = "https://zenodo.org/records/ZENODO_RECORD_X/files/c.tar.gz"
            sha256 = "$("0"^64)"
        """)
    withenv(NO_STORE..., "VGTK_ARTIFACT_STORE" => joinpath(dir, "store")) do
        err = try VgtkCore.install_artifact("tier_zz", toml); nothing catch e; e end
        @test err isa ErrorException && occursin("unknown artifact 'tier_zz'", err.msg)
        err = try VgtkCore.install_artifact("tier_c", toml); nothing catch e; e end
        @test err isa ErrorException && occursin("not published", err.msg)
    end
    hf_home = mktempdir()
    withenv("HF_TOKEN" => nothing, "HF_HOME" => hf_home) do
        @test VgtkCore.huggingface_token() === nothing
        write(joinpath(hf_home, "token"), "hf_file\n")
        @test VgtkCore.huggingface_token() == "hf_file"
        withenv(() -> (@test VgtkCore.huggingface_token() == "hf_env"), "HF_TOKEN" => " hf_env ")
    end
end

@testitem "artifact_store: env → site profile → preference → depot, the depot refused by the site or on the cluster" setup = [StoreFixture] begin
    using Test, TOML
    import VisualGeometryToolkitCore as VgtkCore
    using Pkg.Artifacts: create_artifact, bind_artifact!
    using Base: SHA1
    dir = mktempdir()
    site_file = joinpath(dir, "mysite.toml")
    write(site_file, "[site]\nbackend = \"local\"\n[storage]\nartifact_store = \"$(joinpath(dir, "site-store"))\"\n")
    # a preference of this package, in an environment on the load path (LocalPreferences.toml beside a Project.toml
    # that names the package): what `Preferences.set_preferences!` writes
    prefenv = joinpath(dir, "prefenv")
    mkpath(prefenv)
    write(joinpath(prefenv, "Project.toml"), "[extras]\nVisualGeometryToolkitCore = \"96820d6b-1dad-4e77-ad4a-8276f2ec1ce7\"\n")
    write(joinpath(prefenv, "LocalPreferences.toml"), "[VisualGeometryToolkitCore]\nartifact_store = \"$(joinpath(dir, "pref-store"))\"\n")
    saved = VgtkCore._SITE_STORE[]
    try
        withenv(NO_STORE...) do
            # the depot, with nothing else
            s = VgtkCore.artifact_store()
            @test s.source === :depot && s.path == joinpath(first(DEPOT_PATH), "artifacts")
            # … refused on the cluster (a job there sets NODE_CACHE)
            withenv("NODE_CACHE" => joinpath(dir, "node")) do
                err = try VgtkCore.artifact_store(); nothing catch e; e end
                @test err isa ErrorException && occursin("refused", err.msg) && occursin("VGTK_ARTIFACT_STORE", err.msg)
            end
            # … refused where the site profile says so ([storage] refuse_depot), NODE_CACHE or not: a profile file
            # read here, a name through the registered reader
            refusing = joinpath(dir, "refusing.toml")
            write(refusing, "[site]\nbackend = \"slurm\"\n[storage]\nrefuse_depot = true\n")
            VgtkCore._SITE_STORE[] = nothing
            withenv("VGTK_SITE" => refusing) do
                err = try VgtkCore.artifact_store(); nothing catch e; e end
                @test err isa ErrorException && occursin("refuse_depot", err.msg) && occursin("VGTK_ARTIFACT_STORE", err.msg)
            end
            VgtkCore.set_site_artifact_store!(site -> site == "cluster" ? (; artifact_store = "", refuse_depot = true) : nothing)
            withenv("VGTK_SITE" => "cluster") do
                err = try VgtkCore.artifact_store(); nothing catch e; e end
                @test err isa ErrorException && occursin("refuse_depot", err.msg)
            end
            # a site that does not refuse it, and a name no profile has: the depot
            withenv(() -> (@test VgtkCore.artifact_store().source === :depot), "VGTK_SITE" => "laptop")
            write(joinpath(dir, "lenient.toml"), "[site]\nbackend = \"local\"\n[storage]\nrefuse_depot = false\n")
            VgtkCore._SITE_STORE[] = nothing
            withenv(() -> (@test VgtkCore.artifact_store().source === :depot), "VGTK_SITE" => joinpath(dir, "lenient.toml"))
            # … a profile that states it decides, NODE_CACHE or not; one that does not leaves it to NODE_CACHE
            withenv(() -> (@test VgtkCore.artifact_store().source === :depot),
                    "VGTK_SITE" => joinpath(dir, "lenient.toml"), "NODE_CACHE" => joinpath(dir, "node"))
            write(joinpath(dir, "silent.toml"), "[site]\nbackend = \"local\"\n")
            err = withenv(() -> (try VgtkCore.artifact_store(); nothing catch e; e end),
                          "VGTK_SITE" => joinpath(dir, "silent.toml"), "NODE_CACHE" => joinpath(dir, "node"))
            @test err isa ErrorException && occursin("NODE_CACHE is set", err.msg)
            # a profile file's `~/…` store is the home's
            write(joinpath(dir, "home.toml"), "[site]\nbackend = \"local\"\n[storage]\nartifact_store = \"~/vgtk-store-test\"\n")
            withenv(() -> (@test VgtkCore.artifact_store() == VgtkCore.ArtifactStore(joinpath(homedir(), "vgtk-store-test"), :site)),
                    "VGTK_SITE" => joinpath(dir, "home.toml"))
            write(joinpath(dir, "bad.toml"), "[site]\nbackend = \"local\"\n[storage]\nrefuse_depot = \"yes\"\n")
            err = withenv(() -> (try VgtkCore.artifact_store(); nothing catch e; e end), "VGTK_SITE" => joinpath(dir, "bad.toml"))
            @test err isa ErrorException && occursin("is not true or false", err.msg)
            # the preference over the depot (and on the cluster)
            pushfirst!(LOAD_PATH, prefenv)
            try
                @test VgtkCore.artifact_store() == VgtkCore.ArtifactStore(joinpath(dir, "pref-store"), :preference)
                withenv(() -> (@test VgtkCore.artifact_store().source === :preference), "NODE_CACHE" => joinpath(dir, "node"))
                # the site profile over the preference: a profile file read here, a name through the registered reader
                VgtkCore._SITE_STORE[] = nothing
                withenv("VGTK_SITE" => site_file) do
                    @test VgtkCore.artifact_store() == VgtkCore.ArtifactStore(joinpath(dir, "site-store"), :site)
                end
                withenv("VGTK_SITE" => "mysite") do
                    @test VgtkCore.artifact_store().source === :preference      # a name, and no reader: no rung
                    VgtkCore.set_site_artifact_store!(site -> site == "mysite" ? joinpath(dir, "named-store") : nothing)
                    @test VgtkCore.artifact_store() == VgtkCore.ArtifactStore(joinpath(dir, "named-store"), :site)
                end
                # a profile that names no store leaves it to the next rung
                withenv(() -> (@test VgtkCore.artifact_store().source === :preference), "VGTK_SITE" => "other")
                # a reader's (; artifact_store, refuse_depot): the store, or "" for the next rung
                VgtkCore.set_site_artifact_store!(site -> (; artifact_store = site == "mysite" ? joinpath(dir, "nt-store") : "",
                                                           refuse_depot = true))
                withenv(() -> (@test VgtkCore.artifact_store() == VgtkCore.ArtifactStore(joinpath(dir, "nt-store"), :site)),
                        "VGTK_SITE" => "mysite")
                withenv(() -> (@test VgtkCore.artifact_store().source === :preference), "VGTK_SITE" => "other")
                # the environment over everything
                withenv("VGTK_SITE" => "mysite", "VGTK_ARTIFACT_STORE" => joinpath(dir, "env-store")) do
                    @test VgtkCore.artifact_store() == VgtkCore.ArtifactStore(joinpath(dir, "env-store"), :env)
                end
            finally
                popfirst!(LOAD_PATH)
            end
        end
    finally
        VgtkCore._SITE_STORE[] = saved
    end
end

@testitem "install_artifact: into the store, hash-verified, atomic, one name one tree" setup = [StoreFixture] begin
    using Test, TOML
    import VisualGeometryToolkitCore as VgtkCore
    using Pkg.Artifacts: create_artifact, bind_artifact!
    using Base: SHA1
    dir = mktempdir()
    store = joinpath(dir, "store")
    h, toml = published_tree(dir, "tier_a", ["run1/a.json" => "{}", "b.txt" => "b"])
    withenv(NO_STORE..., "VGTK_ARTIFACT_STORE" => store) do
        tree = VgtkCore.install_artifact("tier_a", toml)
        @test tree == joinpath(store, bytes2hex(h.bytes)) && read(joinpath(tree, "run1", "a.json"), String) == "{}"
        @test !isdir(joinpath(first(DEPOT_PATH), "artifacts", bytes2hex(h.bytes)))   # never the depot
        @test VgtkCore.install_artifact("tier_a", toml) == tree                      # present: no download
        @test isfile(joinpath(store, "names", "tier_a", bytes2hex(h.bytes)))
        @test isempty(VgtkCore.store_name_clashes())
        # a tarball whose sha256 is not the bound one: refused, nothing left in the store
        hb, _ = published_tree(dir, "tier_b", ["x" => "x"]; sha = "0"^64, toml)
        err = try VgtkCore.install_artifact("tier_b", toml); nothing catch e; e end
        @test err isa ErrorException && occursin("tier_b", err.msg)
        @test !isdir(joinpath(store, bytes2hex(hb.bytes)))
        # a tree that does not hash to its git-tree-sha1 (another tree's tarball under this hash): refused
        meta = TOML.parsefile(toml)
        meta["tier_c"] = Dict("git-tree-sha1" => "3"^40, "lazy" => true, "download" => meta["tier_a"]["download"])
        open(io -> TOML.print(io, meta), toml, "w")
        err = try VgtkCore.install_artifact("tier_c", toml); nothing catch e; e end
        @test err isa ErrorException && !isdir(joinpath(store, "3"^40))
        # only the tree, the name record, install locks and Pkg's cache tag: no partial download left behind
        @test all(f -> f in (bytes2hex(h.bytes), "names", "CACHEDIR.TAG") || endswith(f, ".pid"), readdir(store))
        # the same name bound to another tree: two data under one name
        h2, toml2 = published_tree(dir, "tier_a", ["run1/a.json" => "{\"v\": 2}"]; toml = joinpath(dir, "Other.toml"))
        VgtkCore.install_artifact("tier_a", toml2)
        @test VgtkCore.store_name_clashes() == Dict("tier_a" => sort([bytes2hex(h.bytes), bytes2hex(h2.bytes)]))
    end
end

@testitem "install_artifact: one process installs a tree, another waits for its lock" setup = [StoreFixture] begin
    using Test, TOML
    import VisualGeometryToolkitCore as VgtkCore
    using Pkg.Artifacts: create_artifact, bind_artifact!
    using Base: SHA1
    dir = mktempdir()
    store = joinpath(dir, "store")
    h, toml = published_tree(dir, "tier_a", ["a" => "a"])
    lockfile = joinpath(store, "$(bytes2hex(h.bytes)).install.lock")
    mkpath(store)
    ready = joinpath(dir, "held")
    # another process holds the tree's install lock for 3 s, as install_artifact takes it: the lock directory, made
    # atomically, with its owner "<host> <pid> <time>"
    code = "mkdir($(repr(lockfile))); write(joinpath($(repr(lockfile)), \"owner\"), \"\$(gethostname()) \$(getpid()) \$(time())\"); " *
           "touch($(repr(ready))); sleep(3); rm($(repr(lockfile)); recursive = true)"
    # a bare julia (no test flags, no inherited project or load path: the standard libraries only)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    errlog = joinpath(dir, "holder.err")
    env = filter(kv -> !(first(kv) in ("JULIA_PROJECT", "JULIA_LOAD_PATH")), ENV)
    p = run(pipeline(setenv(`$julia --startup-file=no -e $code`, env); stdout = devnull, stderr = errlog); wait = false)
    t0 = time()
    while !isfile(ready) && process_running(p) && time() - t0 < 120
        sleep(0.05)
    end
    isfile(ready) || @error "the lock holder did not start" log = read(errlog, String)
    @test isfile(ready)
    withenv(NO_STORE..., "VGTK_ARTIFACT_STORE" => store) do
        t = @elapsed tree = VgtkCore.install_artifact("tier_a", toml)
        @test t > 1 && isfile(joinpath(tree, "a"))          # it waited for the lock, then installed
    end
    wait(p)
    @test success(p) && !isfile(lockfile)
end

@testitem "with_artifact_store: Pkg's override for the block, nested into the same store, restored" setup = [StoreFixture] begin
    using Test, TOML
    import VisualGeometryToolkitCore as VgtkCore
    using Pkg.Artifacts: create_artifact, bind_artifact!
    using Base: SHA1
    import Artifacts
    dir = mktempdir()
    VgtkCore.with_artifact_store(joinpath(dir, "a")) do
        @test Artifacts.artifacts_dirs() == [joinpath(dir, "a")]
        VgtkCore.with_artifact_store(joinpath(dir, "a")) do              # nested into the same store
            @test Artifacts.artifacts_dirs() == [joinpath(dir, "a")]
        end
        @test Artifacts.artifacts_dirs() == [joinpath(dir, "a")]       # still set after the inner block
        @test_throws ErrorException VgtkCore.with_artifact_store(() -> nothing, joinpath(dir, "b"))
        # a task the block spawns enters the same store without the lock its parent holds, and is refused another
        # store at once (it would wait for ever on that lock)
        @test fetch(Threads.@spawn VgtkCore.with_artifact_store(() -> Artifacts.artifacts_dirs(), joinpath(dir, "a"))) ==
              [joinpath(dir, "a")]
        t = Threads.@spawn try VgtkCore.with_artifact_store(() -> nothing, joinpath(dir, "b")); :entered catch; :refused end
        @test fetch(t) === :refused
    end
    @test Artifacts.ARTIFACTS_DIR_OVERRIDE[] === nothing
    # a task outside the block never runs unlocked in its store: it waits for the block, then sets its own override
    started, release = Channel{Nothing}(1), Channel{Nothing}(1)
    holder = Threads.@spawn VgtkCore.with_artifact_store(joinpath(dir, "a")) do
        put!(started, nothing); take!(release)
    end
    take!(started)
    other = Threads.@spawn VgtkCore.with_artifact_store(() -> Artifacts.artifacts_dirs(), joinpath(dir, "a"))
    sleep(0.2)
    @test !istaskdone(other)                                       # waiting on the lock, not inside the block
    put!(release, nothing); wait(holder)
    @test fetch(other) == [joinpath(dir, "a")]
    @test Artifacts.ARTIFACTS_DIR_OVERRIDE[] === nothing
end

@testitem "store_file: a single file in the store, verified, atomic" setup = [StoreFixture] begin
    using Test, TOML
    import VisualGeometryToolkitCore as VgtkCore
    using Pkg.Artifacts: create_artifact, bind_artifact!
    using Base: SHA1
    dir = mktempdir()
    src = joinpath(dir, "w.onnx"); write(src, "weights")
    digest = open(VgtkCore.sha256_hex, src)
    store = VgtkCore.ArtifactStore(joinpath(dir, "store"), :env)
    p = VgtkCore.store_file("file://$src", digest, "w.onnx"; depot_dir = joinpath(dir, "scratch"), store)
    @test p == joinpath(dir, "store", "files", digest, "w.onnx") && read(p, String) == "weights"
    rm(src)
    @test VgtkCore.store_file("file://$src", digest, "w.onnx"; depot_dir = "", store) == p   # stored: no download
    # a mismatch is an error and leaves no file
    write(src, "other")
    @test_throws r"sha256 .* expected" VgtkCore.store_file("file://$src", "0"^64, "x.onnx"; depot_dir = "", store)
    @test !ispath(joinpath(dir, "store", "files", "0"^64, "x.onnx"))
    # the depot store: the caller's own directory
    q = VgtkCore.store_file("file://$src", open(VgtkCore.sha256_hex, src), "y.onnx"; depot_dir = joinpath(dir, "scratch"),
                            store = VgtkCore.ArtifactStore(joinpath(dir, "depot"), :depot))
    @test q == joinpath(dir, "scratch", "y.onnx") && read(q, String) == "other"
end

@testitem "migrate_artifacts: the depot's trees into the store, hash-checked" setup = [StoreFixture] begin
    using Test, TOML
    import VisualGeometryToolkitCore as VgtkCore
    using Pkg.Artifacts: create_artifact, bind_artifact!
    using Base: SHA1
    dir = mktempdir()
    depot = joinpath(dir, "depot")
    store = VgtkCore.ArtifactStore(joinpath(dir, "store"), :env)
    toml = joinpath(dir, "Artifacts.toml")
    pushfirst!(DEPOT_PATH, depot)
    try
        h1 = create_artifact(d -> write(joinpath(d, "a"), "a"))           # into the (test's) depot
        h2 = create_artifact(d -> write(joinpath(d, "b"), "b"))
        for (n, h) in ("t1" => h1, "t2" => h2)
            bind_artifact!(toml, n, h; lazy = true, download_info = [("https://zenodo.org/records/1/files/$n.tar.gz", "0"^64)])
        end
        bind_artifact!(toml, "t3", SHA1("4"^40); lazy = true, download_info = [("https://zenodo.org/records/1/files/t3.tar.gz", "0"^64)])
        # a depot tree that no longer hashes to its name is left alone
        f = joinpath(depot, "artifacts", bytes2hex(h2.bytes), "b"); chmod(f, 0o644); write(f, "changed")
        r = Dict(VgtkCore.migrate_artifacts(toml; store))
        @test r == Dict("t1" => :migrated, "t2" => :mismatch, "t3" => :absent)
        @test read(joinpath(store.path, bytes2hex(h1.bytes), "a"), String) == "a" && !isdir(joinpath(store.path, bytes2hex(h2.bytes)))
        @test Dict(VgtkCore.migrate_artifacts(toml; store, names = ["t1"], remove = true)) == Dict("t1" => :present)
        @test !isdir(joinpath(depot, "artifacts", bytes2hex(h1.bytes)))     # removed from the depot once stored
        withenv(NO_STORE..., "VGTK_ARTIFACT_STORE" => store.path) do       # and installed from the store, no download
            @test VgtkCore.install_artifact("t1", toml) == joinpath(store.path, bytes2hex(h1.bytes))
        end
        @test_throws ArgumentError VgtkCore.migrate_artifacts(toml; store = VgtkCore.ArtifactStore(depot, :depot))
    finally
        popfirst!(DEPOT_PATH)
    end
end

@testitem "install_built_artifact: a tree built here, into the store, its hash checked" setup = [StoreFixture] begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    using Pkg.Artifacts: create_artifact
    using Base: SHA1
    dir = mktempdir()
    build!(d) = (write(joinpath(d, "lib.so"), "built"); mkpath(joinpath(d, "share")); write(joinpath(d, "share", "x"), "y"))
    # the tree's hash, as a pinned Artifacts.toml states it
    hash = VgtkCore.with_artifact_store(() -> create_artifact(build!), joinpath(dir, "scratch"))
    store = VgtkCore.ArtifactStore(joinpath(dir, "store"), :env)
    calls = Ref(0)
    counted!(d) = (calls[] += 1; build!(d))
    tree = VgtkCore.install_built_artifact(counted!, "built_tree", hash; store)
    @test tree == joinpath(store.path, bytes2hex(hash.bytes)) && read(joinpath(tree, "lib.so"), String) == "built"
    @test isfile(joinpath(store.path, "names", "built_tree", bytes2hex(hash.bytes)))
    # present: not built again
    @test VgtkCore.install_built_artifact(counted!, "built_tree", hash; store) == tree && calls[] == 1
    # a build that gives another tree is refused, and leaves nothing in the store
    wrong = SHA1("1"^40)
    err = try VgtkCore.install_built_artifact(d -> write(joinpath(d, "other"), "z"), "wrong_tree", wrong; store); nothing catch e; e end
    @test err isa ErrorException && occursin("not the pinned $("1"^40)", err.msg)
    @test sort([f for f in readdir(store.path) if isdir(joinpath(store.path, f))]) == sort([bytes2hex(hash.bytes), "names"])
    @test !isdir(joinpath(store.path, "names", "wrong_tree"))
end

@testitem "install lock: a directory made atomically (NFS-safe), a stale holder's taken over, released after" setup = [StoreFixture] begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    dir = mktempdir()
    lock = joinpath(dir, "k.install.lock")
    # held for the block, released after it, and after an error in it
    @test VgtkCore._install_lock(() -> (isdir(lock) && isfile(joinpath(lock, "owner"))), dir, "k")
    @test !ispath(lock)
    @test_throws ErrorException VgtkCore._install_lock(() -> error("in the block"), dir, "k")
    @test !ispath(lock)
    # a holder of this host that is gone: taken over at once
    p = run(`$(Base.julia_cmd()) --startup-file=no -e "exit()"`)
    mkdir(lock); write(joinpath(lock, "owner"), "$(gethostname()) $(getpid(p)) 0")
    @test !VgtkCore._process_alive(getpid(p)) && VgtkCore._process_alive(getpid())
    t = @elapsed VgtkCore._install_lock(() -> nothing, dir, "k")
    @test t < 5 && !ispath(lock)
    # a live holder of another host: waited for until its heartbeat is `stale_age` old
    mkdir(lock); write(joinpath(lock, "owner"), "elsewhere 1 0")
    t = @elapsed VgtkCore._install_lock(() -> nothing, dir, "k"; stale_age = 1.5, poll = 0.1)
    @test t >= 1.4 && !ispath(lock)
    # a live holder of this host (this process, another task) is waited for, never broken; its heartbeat keeps it fresh
    order = String[]
    holder = @async VgtkCore._install_lock(dir, "k"; stale_age = 0.5, heartbeat = 0.1) do
        push!(order, "held"); sleep(1.5); push!(order, "released")
    end
    while isempty(order); sleep(0.01); end
    VgtkCore._install_lock(() -> push!(order, "second"), dir, "k"; stale_age = 0.5, poll = 0.05)
    wait(holder)
    @test order == ["held", "released", "second"]
    # two waiters that judge one lock stale: one moves it aside, the other's move fails quietly
    mkdir(lock); VgtkCore._break_lock(lock); VgtkCore._break_lock(lock)
    @test !ispath(lock) && isempty(readdir(dir))
end
