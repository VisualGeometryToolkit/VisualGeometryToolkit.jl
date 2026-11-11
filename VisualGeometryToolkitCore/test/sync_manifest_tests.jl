@testitem "sync_manifest: the stack's git hooks" begin
    # The one copy of the hooks every stack repository installed as its own
    # .githooks/ (RANSACScoringDoneRight and BlobBoards carried byte-identical
    # copies): the shell hook ships in githooks/, re-resolves through
    # `sync_manifest`, and includes src/quality/sync_manifest.jl BY PATH, so the
    # file must stand alone (Base only).
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: sync_manifest, install_githooks, githooks_dir

    hooks = githooks_dir()
    @test isfile(joinpath(hooks, "sync-manifest"))
    @test Sys.isexecutable(joinpath(hooks, "sync-manifest"))
    for h in ("post-checkout", "post-merge", "post-rewrite")
        @test islink(joinpath(hooks, h)) && readlink(joinpath(hooks, h)) == "sync-manifest"
    end
    # the hook finds the Julia side beside it
    script = normpath(joinpath(hooks, "..", "src", "quality", "sync_manifest.jl"))
    @test isfile(script)
    @test occursin("src/quality/sync_manifest.jl", read(joinpath(hooks, "sync-manifest"), String))

    # dependency-free: the file defines the entry in a bare module
    m = Module(:SyncManifestByPath)
    Base.include(m, script)
    @test isdefined(m, :sync_manifest) && isdefined(m, :install_githooks)

    @test_throws ArgumentError sync_manifest("resolve")

    # install_githooks points a repository's hooks at the copy
    repo = mktempdir()
    run(`git -C $repo init -q`)
    @test install_githooks(repo) == hooks
    @test readchomp(`git -C $repo config --get core.hooksPath`) == hooks
    @test_throws ArgumentError install_githooks(repo; hooks = mktempdir())

    # an instantiate of an empty project creates its Manifest and reports it
    proj = mktempdir()
    write(joinpath(proj, "Project.toml"), "name = \"Empty\"\nuuid = \"0c4f8b8e-7a4c-4a39-9d8e-1f2e3d4c5b6a\"\n")
    # Pkg (a standard library, not a dependency of the test environment) as sync_manifest loads it
    Pkg = Base.require(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))
    out = IOBuffer()
    res = Base.invokelatest(Pkg.activate, () -> sync_manifest("instantiate"; io = out), proj)
    @test res.created && isempty(res.moved) && res.seconds >= 0
    @test isfile(joinpath(proj, "Manifest.toml"))
    @test startswith(String(take!(out)), "Manifest created")
end
