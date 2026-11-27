# The digest cache (src/code_identity.jl: file_sha256's tier across processes, digest_cache_dir): a large file's hash
# kept while its size, mtime, ctime and inode are unchanged. A miss hashes and writes the entry, a hit reads none of the
# data (a forged entry shows it), any change of the file misses, and in a tree only the changed file is read again;
# `VGTK_DIGEST_CACHE=off` hashes in full; processes that hash at once agree and leave whole entries; a cache that cannot
# be written is no cache; a pin the hash does not match is still refused; a small file is never kept.

@testitem "digest_cache" begin
    using Test
    using TOML: TOML
    import VisualGeometryToolkitCore   # pkgdir: Core's canonical.jl, which a bare julia includes before src/cli.jl
    using VisualGeometryToolkitEval: VisualGeometryToolkitEval, digest_cache_dir, tree_sha256, file_sha256, inspect_input
    import VisualGeometryToolkitEval as VgtkEval
    # every file kept, however small and new (the fixtures are both); restored at the end, for the worker's other items
    saved = (VgtkEval._DIGEST_CACHE_MIN_BYTES[], VgtkEval._DIGEST_CACHE_MIN_AGE[])
    VgtkEval._DIGEST_CACHE_MIN_BYTES[], VgtkEval._DIGEST_CACHE_MIN_AGE[] = 0, 0.0
    try
    cold() = lock(() -> empty!(VgtkEval._FILE_SHA256), VgtkEval._FILE_SHA256_LOCK)   # a fresh process's view: no process cache
    entries(dir) = isdir(dir) ? sort(filter(endswith(".toml"), readdir(dir; join = true))) : String[]
    entry(dir, f) = only(filter(e -> TOML.parsefile(e)["path"] == realpath(f), entries(dir)))
    function forge!(e, h)          # the entry kept as is, its hash replaced: a hit returns it unread
        d = TOML.parsefile(e); d["sha256"] = h
        open(io -> TOML.print(io, d), e, "w")
    end
    fake = repeat("ab", 32)
    sha(f) = first(split(read(`sha256sum $f`, String)))
    function tree()
        t = mktempdir()
        mkpath(joinpath(t, "sub")); write(joinpath(t, "a.bin"), "aaaa"); write(joinpath(t, "sub", "b.bin"), "bbbb")
        return t
    end

    @testset "the directory: VGTK_DIGEST_CACHE, else XDG_CACHE_HOME, else ~/.cache; off for none" begin
        @test digest_cache_dir(Dict("VGTK_DIGEST_CACHE" => "/c/d")) == "/c/d"
        @test digest_cache_dir(Dict("XDG_CACHE_HOME" => "/x")) == "/x/vgtk/input-digests"
        @test digest_cache_dir(Dict{String, String}()) == joinpath(homedir(), ".cache", "vgtk", "input-digests")
        @test all(v -> digest_cache_dir(Dict("VGTK_DIGEST_CACHE" => v)) === nothing, ("off", "OFF", "0", "false", "no"))
        @test_throws r"an absolute path" digest_cache_dir(Dict("VGTK_DIGEST_CACHE" => "rel/dir"))
    end

    @testset "a file: miss, then hit without reading; any change misses" begin
        cache, d = mktempdir(), mktempdir()
        f = joinpath(d, "pool.bin"); write(f, "pool of record\n")
        withenv("VGTK_DIGEST_CACHE" => cache) do
            cold(); @test file_sha256(f) == sha(f)                                      # a miss writes the entry
            e = only(entries(cache))
            @test TOML.parsefile(e)["path"] == realpath(f) && TOML.parsefile(e)["sha256"] == sha(f)
            forge!(e, fake); cold()
            @test file_sha256(f) == fake                                                  # a hit: the data unread
            l = joinpath(mktempdir(), "link"); symlink(f, l); cold()
            @test file_sha256(l) == fake                                                  # by its real path
            for change in (() -> write(f, "pool OF RECORD\n"),                            # same size
                           () -> write(f, "another size\n"))
                forge!(e, fake); change(); cold()
                @test file_sha256(f) == sha(f) != fake
            end
            # a same-size rewrite whose mtime is set back: only its ctime moved
            ref = tempname(); run(`touch -r $f $ref`)                                     # ref keeps f's times
            forge!(e, fake); write(f, "ANOTHER SIZE\n"); run(`touch -r $ref $f`); cold()
            @test mtime(f) == mtime(ref) && filesize(f) == 13 && file_sha256(f) == sha(f) != fake
            # a file changed just now is not kept: its entry stays as it was
            VgtkEval._DIGEST_CACHE_MIN_AGE[] = 60.0
            forge!(e, fake); write(f, "changed now\n"); cold()
            @test file_sha256(f) == sha(f) && TOML.parsefile(e)["sha256"] == fake
            VgtkEval._DIGEST_CACHE_MIN_AGE[] = 0.0
        end
    end

    @testset "a tree: only the changed file is read again" begin
        cache, t = mktempdir(), tree()
        a, b = joinpath(t, "a.bin"), joinpath(t, "sub", "b.bin")
        withenv("VGTK_DIGEST_CACHE" => cache) do
            cold(); want = tree_sha256(t)
            @test length(entries(cache)) == 2
            forge!(entry(cache, a), fake); forge!(entry(cache, b), fake); cold()
            forged = tree_sha256(t)
            @test forged != want                                                          # both hits
            write(a, "AAAA"); cold()
            @test file_sha256(a) == sha(a) && file_sha256(b) == fake                     # a read again, b not
            # the pin is checked as before: a kept hash other than the content of record is refused
            tbl = Dict{String, Any}("path" => basename(t), "sha256" => want)
            x = inspect_input("pool", tbl, dirname(t))
            @test x.state === :refused && occursin("expected $want", x.problem)
        end
    end

    @testset "off: every file hashed in full, nothing read or written" begin
        cache, t = mktempdir(), tree()
        withenv(() -> (cold(); tree_sha256(t)), "VGTK_DIGEST_CACHE" => cache)
        foreach(e -> forge!(e, fake), entries(cache))
        withenv("VGTK_DIGEST_CACHE" => "off", "XDG_CACHE_HOME" => cache) do
            cold(); @test file_sha256(joinpath(t, "a.bin")) == sha(joinpath(t, "a.bin"))
        end
        @test length(entries(cache)) == 2 && !isdir(joinpath(cache, "vgtk"))
    end

    @testset "a small file is never kept" begin
        cache, t = mktempdir(), tree()
        VgtkEval._DIGEST_CACHE_MIN_BYTES[] = 5
        try withenv(() -> (cold(); tree_sha256(t)), "VGTK_DIGEST_CACHE" => cache) finally VgtkEval._DIGEST_CACHE_MIN_BYTES[] = 0 end
        @test isempty(entries(cache))
    end

    @testset "a cache that cannot be written is no cache" begin
        cache, t = mktempdir(), tree()
        chmod(cache, 0o555)
        try
            if !(try touch(joinpath(cache, "probe")); true catch; false end)   # root writes anyway: nothing to show
                withenv("VGTK_DIGEST_CACHE" => joinpath(cache, "digests")) do
                    cold(); @test file_sha256(joinpath(t, "a.bin")) == sha(joinpath(t, "a.bin"))
                end
            end
        finally
            chmod(cache, 0o755)
        end
    end

    @testset "concurrent processes on a cold cache agree and leave whole entries" begin
        cache, t = mktempdir(), tree()
        for i in 1:30; write(joinpath(t, "sub", "f$i.bin"), rand(UInt8, 4096)); end
        cold(); want = withenv(() -> tree_sha256(t), "VGTK_DIGEST_CACHE" => "off")
        src = pkgdir(VisualGeometryToolkitEval, "src")
        canonical = pkgdir(VisualGeometryToolkitCore, "src", "base", "canonical.jl")
        code = "foreach(include, ARGS[1:2]); _DIGEST_CACHE_MIN_BYTES[] = 0; _DIGEST_CACHE_MIN_AGE[] = 0.0; print(tree_sha256(ARGS[3]))"
        env = merge(Dict(ENV), Dict("JULIA_LOAD_PATH" => "@stdlib", "JULIA_PROJECT" => "", "VGTK_DIGEST_CACHE" => cache))
        cmds = [setenv(`$(Base.julia_cmd()) --startup-file=no -e $code $canonical $(joinpath(src, "cli.jl")) $t`, env) for _ in 1:4]
        @test all(==(want), fetch.([Threads.@spawn read(c, String) for c in cmds]))
        @test length(readdir(cache)) == 32 && all(e -> length(TOML.parsefile(e)["sha256"]) == 64, entries(cache))  # no temporary left
        foreach(rm, entries(cache))
        @test all(==(want), withenv(() -> (cold(); fetch.([Threads.@spawn tree_sha256(t) for _ in 1:8])), "VGTK_DIGEST_CACHE" => cache))
        @test length(readdir(cache)) == 32
    end
    finally
        VgtkEval._DIGEST_CACHE_MIN_BYTES[], VgtkEval._DIGEST_CACHE_MIN_AGE[] = saved
    end
end
