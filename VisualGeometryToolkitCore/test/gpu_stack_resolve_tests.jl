# The package resolves beside the GPU stack its downstream projects hold. Core 0.9.5's weak `[compat] CUDA = "5.11.3"`
# (a floor raised to the gated resolution) needed CUDA_Runtime_Discovery 2, which no cuDNN 1.x accepts, so no
# environment holding cuDNN 1 (BlobBoards, CaliBlobs) could resolve Core 0.9.5 nor any package on it. Resolved from the
# registries in a child process, nothing precompiled.

@testitem "a GPU stack of cuDNN 1.x resolves with VisualGeometryToolkitCore" begin
    using Test, TOML
    import VisualGeometryToolkitCore as VgtkCore
    env = mktempdir()
    write(joinpath(env, "Project.toml"), """
        [deps]
        CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
        VisualGeometryToolkitCore = "96820d6b-1dad-4e77-ad4a-8276f2ec1ce7"
        cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

        [compat]
        cuDNN = "1"

        [sources]
        VisualGeometryToolkitCore = {path = $(repr(pkgdir(VgtkCore)))}
        """)
    log = joinpath(env, "resolve.log")
    # The child's load path is stated, not inherited: `Pkg.test()` runs this suite with `JULIA_LOAD_PATH = "@:<test dir>"`
    # (no `@stdlib`), and a child inheriting it cannot load Pkg (neither `env` nor the test project lists it). `@` is the
    # resolved project `env`, `@stdlib` the stdlibs Pkg and its dependencies come from: the default load path less
    # `@v#.#`, so no user default environment enters the resolve.
    load_path = join(["@", "@stdlib"], Sys.iswindows() ? ';' : ':')
    cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$env -e 'using Pkg; Pkg.resolve()'`,
                 "JULIA_PKG_PRECOMPILE_AUTO" => "0", "JULIA_LOAD_PATH" => load_path)
    ok = success(pipeline(cmd; stdout = log, stderr = log))
    ok || print(read(log, String))
    @test ok
    # the Manifest read as TOML: an entry's `deps = [...]` array precedes its `version`, so no line pattern spans it
    @test ok && startswith(only(TOML.parsefile(joinpath(env, "Manifest.toml"))["deps"]["cuDNN"])["version"], "1.")
end
