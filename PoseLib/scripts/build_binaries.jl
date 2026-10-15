# Build PoseLib.jl's prebuilt library from the recipe of record,
# `yggdrasil/P/PoseLib/build_tarballs.jl`, and bind the tarballs in
# `Artifacts.toml`.
#
# Run from an environment holding BinaryBuilder (never the package's own), with
# a depot of your choosing:
#
#     JULIA_DEPOT_PATH=<scratch-depot>:<your-depot> \
#         julia --project=<bb-env> scripts/build_binaries.jl --workdir=<dir> [options] [triplets...]
#
#   --workdir=DIR     the recipe is copied to DIR/recipe and built there; the
#                     tarballs (and a SHA256SUMS) land in DIR/products
#   --skip-build      bind the tarballs already in DIR/products
#   --url-base=URL    download base (default: the GitHub release of this
#                     library version, see `default_url_base`)
#   --artifacts=PATH  the Artifacts.toml to write (default: the package's own)
#   --all             every platform the recipe lists (Yggdrasil's set)
#   triplets          the platforms to build and bind (default: `LINUX`)
#
# Until PoseLib_jll is registered the package publishes Linux binaries only, so
# the default is `LINUX`; the recipe's own platform list stays Yggdrasil's.
# With --all, the macOS platforms need the macOS SDK licence accepted:
# BINARYBUILDER_AUTOMATIC_APPLE=true.

using Pkg, SHA, Base.BinaryPlatforms
using Pkg.Artifacts: bind_artifact!

const ROOT = dirname(@__DIR__)
const RECIPE = joinpath(ROOT, "yggdrasil", "P", "PoseLib")
const NAME = "PoseLib"

# Every Linux platform the recipe targets (glibc and musl).
const LINUX = [
    "x86_64-linux-gnu-cxx11", "x86_64-linux-musl-cxx11",
    "aarch64-linux-gnu-cxx11", "aarch64-linux-musl-cxx11",
    "i686-linux-gnu-cxx11", "i686-linux-musl-cxx11",
    "armv6l-linux-gnueabihf-cxx11", "armv6l-linux-musleabihf-cxx11",
    "armv7l-linux-gnueabihf-cxx11", "armv7l-linux-musleabihf-cxx11",
    "powerpc64le-linux-gnu-cxx11", "riscv64-linux-gnu-cxx11",
]

recipe_version() =
    VersionNumber(match(r"^version = v\"([^\"]+)\""m,
                        read(joinpath(RECIPE, "build_tarballs.jl"), String))[1])

default_url_base(v) =
    "https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/releases/download/$(NAME)-binaries-v$(v)"

function main(args)
    opts = Dict{String,String}()
    triplets = String[]
    for a in args
        m = match(r"^--([a-z-]+)(?:=(.*))?$", a)
        m === nothing ? push!(triplets, a) : (opts[m[1]] = something(m[2], ""))
    end
    everything = haskey(opts, "all")
    everything && !isempty(triplets) && error("--all and explicit triplets are exclusive")
    isempty(triplets) && !everything && append!(triplets, LINUX)
    workdir = abspath(get(opts, "workdir") do
        error("--workdir=DIR is required")
    end)
    v = recipe_version()
    url_base = rstrip(get(opts, "url-base", default_url_base(v)), '/')
    artifacts_toml = abspath(get(opts, "artifacts", joinpath(ROOT, "Artifacts.toml")))
    products = joinpath(workdir, "products")

    # The recipe compiles its bundled copies; they must be the package's own.
    for f in ("wrapper/julia_wrapper.cpp", "LICENSE")
        read(joinpath(RECIPE, "bundled", f)) == read(joinpath(ROOT, f)) ||
            error("yggdrasil/P/PoseLib/bundled/$f differs from $f; refresh the bundle first")
    end

    if !haskey(opts, "skip-build")
        recipe = joinpath(workdir, "recipe")
        rm(recipe; force = true, recursive = true)
        mkpath(workdir)
        cp(RECIPE, recipe)
        rm(joinpath(recipe, "build"); force = true, recursive = true)
        rm(joinpath(recipe, "products"); force = true, recursive = true)
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project())
               build_tarballs.jl --verbose $(join(triplets, ","))`
        run(setenv(Cmd(cmd; dir = recipe), ENV))
        mkpath(products)
        for f in readdir(joinpath(recipe, "products"))
            startswith(f, "$(NAME).v$(v).") && endswith(f, ".tar.gz") &&
                cp(joinpath(recipe, "products", f), joinpath(products, f); force = true)
        end
    end

    tripletof(f) = chop(f; head = length("$(NAME).v$(v)."), tail = length(".tar.gz"))
    tarballs = sort(filter(f -> startswith(f, "$(NAME).v$(v).") && endswith(f, ".tar.gz") &&
                                (everything || tripletof(f) in triplets),
                           readdir(products)))
    isempty(tarballs) && error("no $(NAME).v$(v).*.tar.gz in $products")
    everything || length(tarballs) == length(triplets) ||
        error("missing tarballs in $products for: ",
              join(setdiff(triplets, map(tripletof, tarballs)), ", "))

    isfile(artifacts_toml) && rm(artifacts_toml)
    sums = String[]
    for f in tarballs
        path = joinpath(products, f)
        triplet = tripletof(f)
        platform = parse(Platform, triplet)
        sha = bytes2hex(open(sha256, path))
        tree = mktempdir() do dir
            Pkg.PlatformEngines.unpack(path, dir)
            Base.SHA1(Pkg.GitTools.tree_hash(dir))
        end
        bind_artifact!(artifacts_toml, NAME, tree; platform,
                       download_info = [("$url_base/$f", sha)])
        push!(sums, "$sha  $f")
        println(rpad(triplet, 34), " git-tree-sha1 ", bytes2hex(tree.bytes),
                "  sha256 ", sha, "  ", filesize(path), " B")
    end
    write(joinpath(products, "SHA256SUMS"), join(sums, "\n") * "\n")
    println("wrote ", artifacts_toml, " (", length(tarballs), " platforms)")
end

main(ARGS)
