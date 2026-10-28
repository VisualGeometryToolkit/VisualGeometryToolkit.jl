# detect_mser against VLFeat's libvl (VLFeat 1b9075f, vl/mser.c).
#
#   julia --project=VisualGeometryToolkitFeatures/validation \
#       VisualGeometryToolkitFeatures/validation/mser_vlfeat.jl [--freeze] [libvl.so] [image]
#
# libvl defaults to $VLFEAT_ROOT/bin/glnxa64/libvl.so (VLFEAT_ROOT defaults to
# ~/src/vlfeat); the real image to the BlobBoards Office fisheye frame, read as
# 8-bit gray and averaged over 4 × 4 blocks (4240 × 2832 → 1060 × 708). libvl is
# not a dependency of the package: this script is the comparison, and
# `--freeze` writes test/data/mser_vlfeat.toml, VLFeat's output on the small
# images, which the package suite checks detect_mser against without libvl.
#
# What is compared, per image, polarity and parameter set:
#   - the number of regions;
#   - the regions as PIXEL SETS: each VLFeat seed filled as vl_erfill fills it
#     (ported below), against detect_mser's `pixels`. The component tree is
#     unique, so the sets must be identical; the pixel a tree names a region by
#     is not (VLFeat's union-find and ImageMorphology's MaxTree pick different
#     pixels of a plateau), so seed coincidence is only counted;
#   - the 5 × 11 image with one bright pixel at (row 2, column 7): VLFeat's seed
#     must be 0-based linear index (7 − 1)·5 + (2 − 1) = 31, ours (2, 7).
# The image is handed to libvl as Julia stores it, column-major with
# dims = (H, W): no transpose or permutation anywhere.

using VisualGeometryToolkitFeatures
import VisualGeometryToolkitFeatures as VgtkFeatures
using FileIO: load
using ImageCore: Gray, N0f8, channelview
using BenchmarkTools: @belapsed
using TOML: TOML
using Printf: @sprintf

const POSITIONAL = filter(a -> !startswith(a, "--"), ARGS)
const LIBVL = get(POSITIONAL, 1, joinpath(get(ENV, "VLFEAT_ROOT", joinpath(homedir(), "src", "vlfeat")), "bin", "glnxa64", "libvl.so"))
const REAL_IMAGE = get(POSITIONAL, 2, "/mnt/informatik/Datasets/20251028_BlobBoards_Office/sonyfx3_sigma8mm_fisheye_f8/jpeg/DSC03004.JPG")
const FREEZE = "--freeze" in ARGS

# -----------------------------------------------------------------------------
# libvl. The parameter setters are inline in mser.h, so the filter's fields are
# written directly; their offsets (x86-64 layout of struct _VlMserFilt) are
# checked against vl_mser_new's defaults before any use.
# -----------------------------------------------------------------------------
const OFF_NEL, OFF_R, OFF_MER, OFF_NMER = 16, 72, 88, 100
const OFF_DELTA, OFF_MAX_AREA, OFF_MIN_AREA, OFF_MAX_VAR, OFF_MIN_DIV = 144, 152, 160, 168, 176

field(f, T, off) = unsafe_load(Ptr{T}(f + off))
setfield(f, T, off, v) = unsafe_store!(Ptr{T}(f + off), T(v))

function vlfeat_seeds(levels::Matrix{UInt8}; delta = 5, min_area = nothing, max_area = 0.75,
        max_variation = 0.25, min_diversity = 0.2)
    h, w = size(levels)
    f = ccall((:vl_mser_new, LIBVL), Ptr{Cvoid}, (Cint, Ptr{Cint}), 2, Cint[h, w])
    try
        @assert field(f, Cint, OFF_NEL) == h * w
        @assert field(f, Cint, OFF_DELTA) == 5 && field(f, Cdouble, OFF_MAX_AREA) == 0.75 &&
            field(f, Cdouble, OFF_MIN_AREA) == 3.0 / (h * w) && field(f, Cdouble, OFF_MAX_VAR) == 0.25 &&
            field(f, Cdouble, OFF_MIN_DIV) == 0.2
        # vl_mser_process reads r[0].height before pixel 0 is added (vl/mser.c
        # lines 557–559) from vl_malloc'd memory: zero the forest so the seeds
        # do not depend on what the allocator left there.
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), field(f, Ptr{Cvoid}, OFF_R), 0, 16 * h * w)
        setfield(f, Cint, OFF_DELTA, delta)
        setfield(f, Cdouble, OFF_MAX_AREA, max_area)
        min_area === nothing || setfield(f, Cdouble, OFF_MIN_AREA, min_area)
        setfield(f, Cdouble, OFF_MAX_VAR, max_variation)
        setfield(f, Cdouble, OFF_MIN_DIV, min_diversity)
        ccall((:vl_mser_process, LIBVL), Cvoid, (Ptr{Cvoid}, Ptr{UInt8}), f, levels)
        n = field(f, Cint, OFF_NMER)
        mer = field(f, Ptr{UInt32}, OFF_MER)
        return [Int(unsafe_load(mer, k)) for k in 1:n]
    finally
        ccall((:vl_mser_delete, LIBVL), Cvoid, (Ptr{Cvoid},), f)
    end
end

# vl_mser.c (the MATLAB driver): dark on bright on the image, bright on dark
# on its inverse ~v = 255 − v.
vlfeat_mser(levels, polarity; kw...) = vlfeat_seeds(polarity === :dark ? levels : map(~, levels); kw...)

# toolbox/mser/vl_erfill.c, ported line for line: the 0-based seed `s` filled
# over the pixels <= its value (>= for an inverted, i.e. bright, seed), with
# VLFeat's neighbour order. Its quirk is kept: the seed is not marked visited,
# so it is listed twice; the returned SET is what is compared.
function erfill(levels::Matrix{UInt8}, s::Int, invert::Bool)
    dims = size(levels)
    strides = (1, dims[1])
    visited = falses(length(levels))
    members = [s]
    value = levels[s + 1]
    last_expanded = 0
    while last_expanded < length(members)
        index = members[last_expanded += 1]
        subs = (index % strides[2], index ÷ strides[2])   # dimension 1 first, as the C
        nsubs = [-1, -1]
        while true
            good = true
            nindex = 0
            for k in 1:2
                t = nsubs[k] + subs[k]
                good &= 0 <= t < dims[k]
                nindex += t * strides[k]
            end
            if good && nindex != index &&
                    ((!invert && levels[nindex + 1] <= value) || (invert && levels[nindex + 1] >= value)) &&
                    !visited[nindex + 1]
                visited[nindex + 1] = true
                push!(members, nindex)
            end
            k = 1
            done = false
            while (nsubs[k] += 1) > 1
                nsubs[k] = -1
                k += 1
                k > 2 && (done = true; break)
            end
            done && break
        end
    end
    return sort!(unique(members))
end

# -----------------------------------------------------------------------------
# Images
# -----------------------------------------------------------------------------
function real_image(path)
    g = reinterpret(UInt8, Gray{N0f8}.(load(path)))
    h, w = size(g) .÷ 4
    return [UInt8(round(Int, sum(Int, @view g[(4i - 3):4i, (4j - 3):4j]) / 16)) for i in 1:h, j in 1:w]
end

# Plateaus and ties: a 6 × 8 grid of levels from {0, 40, …, 240}, each a 4 × 4 block.
blocks() = [UInt8(40 * ((7 * (i ÷ 4) + 3 * (j ÷ 4) + (i ÷ 4) * (j ÷ 4)) % 7)) for i in 0:23, j in 0:31]
# Nested regions, some touching the border: concentric rings and an offset square.
function nested()
    img = [UInt8(clamp(30 * round(Int, hypot(i - 14, j - 20) / 4), 0, 255)) for i in 1:28, j in 1:36]
    img[1:6, 30:36] .= 0x46
    img[3:4, 32:33] .= 0x0a
    return img
end
# A bright disc on a dark surround, the fisheye image circle's shape, with a dark hole.
function disc_surround()
    img = [hypot(i - 24.5, j - 32.5) < 20 ? 0xb4 : 0x0c for i in 1:48, j in 1:64]
    img[20:23, 30:34] .= 0x3c
    return img
end
function asymmetric()
    img = fill(0x10, 5, 11)
    img[2, 7] = 0xf0
    return img
end
# Smooth random bumps (a fixed LCG, no Random dependency): ties come from quantization.
function bumps(h, w, seed)
    state = UInt64(seed)
    rnd() = (state = state * 6364136223846793005 + 1442695040888963407; (state >> 33) / 2.0^31)
    centres = [(1 + rnd() * h, 1 + rnd() * w, 2 + 6rnd(), 200rnd() - 100) for _ in 1:12]
    return [UInt8(clamp(round(Int, 128 + sum(a * exp(-((i - y)^2 + (j - x)^2) / (2s^2)) for (y, x, s, a) in centres)), 0, 255))
            for i in 1:h, j in 1:w]
end

const PARAMS = [
    "defaults" => (;),
    "delta2_var05_div05" => (; delta = 2, max_variation = 0.5, min_diversity = 0.5),
    "delta10_nodiv" => (; delta = 10, min_diversity = 1.0, max_variation = 1.0),
    "all_areas" => (; min_area = 0.0, max_area = 1.0),
]

# -----------------------------------------------------------------------------
# One comparison
# -----------------------------------------------------------------------------
function compare(name, levels, polarity, pname, params)
    seeds = vlfeat_mser(levels, polarity; params...)
    theirs = Set(erfill(levels, s, polarity === :bright) for s in seeds)
    mine = detect_mser(levels; params..., dark_on_bright = polarity === :dark, bright_on_dark = polarity === :bright)
    ours = Set(sort!(LinearIndices(levels)[r.pixels] .- 1) for r in mine)
    same_seed = count(r -> LinearIndices(levels)[r.seed] - 1 in seeds, mine)
    # our region's level is the level VLFeat's erfill fills at
    levels_ok = all(r -> levels[r.seed] == r.level, mine)
    ok = length(seeds) == length(mine) && theirs == ours && levels_ok
    return (; name, polarity, pname, size = size(levels), nvl = length(seeds), nours = length(mine), ok, same_seed, seeds)
end

function main()
    images = [
        "asymmetric_5x11" => asymmetric(), "blocks_24x32" => blocks(), "nested_28x36" => nested(),
        "disc_surround_48x64" => disc_surround(),
        ("bumps_$(k)_$(h)x$(w)" => bumps(h, w, k) for (k, (h, w)) in enumerate([(20, 30), (33, 17), (40, 40), (25, 60)]))...,
    ]
    real = isfile(REAL_IMAGE) ? real_image(REAL_IMAGE) : nothing
    if real !== nothing
        push!(images, "real_crop_64x96" => real[301:364, 501:596])
        push!(images, "real_$(size(real, 1))x$(size(real, 2))" => real)
    end

    # The index mapping, on its own: one bright pixel at (2, 7) of a 5 × 11 image.
    a = asymmetric()
    s = only(vlfeat_mser(a, :bright; min_area = 0.0))
    r = only(detect_mser(a; min_area = 0.0, dark_on_bright = false))
    mapping_ok = s == 31 && r.seed == CartesianIndex(2, 7) && LinearIndices(a)[r.seed] - 1 == s &&
        Tuple.(only(r.contours)) == [(6.0, 1.0), (7.0, 1.0), (7.0, 2.0), (6.0, 2.0)]

    rows = []
    for (name, img) in images, polarity in (:dark, :bright), (pname, params) in PARAMS
        push!(rows, compare(name, img, polarity, pname, params))
    end

    lines = ["| image | polarity | parameters | VLFeat | ours | pixel sets identical | seeds coincide |", "|---|---|---|---:|---:|---|---:|"]
    for x in rows
        push!(lines, "| $(x.name) | $(x.polarity) | $(x.pname) | $(x.nvl) | $(x.nours) | $(x.ok ? "yes" : "NO") | $(x.same_seed)/$(x.nours) |")
    end
    println(join(lines, "\n"))
    println("\nindex mapping (5×11, bright pixel at (2, 7)): VLFeat seed $s, ours $(Tuple(r.seed)) → ", mapping_ok ? "ok" : "WRONG")
    println("all identical: ", all(x -> x.ok, rows))

    if real !== nothing
        nreg = length(detect_mser(real))
        for polarity in (:dark, :bright)
            t_ours = @belapsed detect_mser($real; dark_on_bright = $(polarity === :dark), bright_on_dark = $(polarity === :bright))
            t_vl = @belapsed vlfeat_mser($real, $polarity)
            println(@sprintf("timing %d×%d, %s, defaults: detect_mser %.1f ms, libvl %.1f ms",
                size(real)..., polarity, 1e3t_ours, 1e3t_vl))
        end
        println(@sprintf("  (%d regions in all) both polarities: detect_mser %.1f ms", nreg, 1e3 * @belapsed(detect_mser($real))))
        t_tree = @belapsed VgtkFeatures.MaxTree($real; connectivity = 2, rev = true)
        println(@sprintf("  of which one MaxTree: %.1f ms", 1e3t_tree))
    end

    if FREEZE
        small = filter(x -> !startswith(x.name, "real_") || x.name == "real_crop_64x96", rows)
        imgs = Dict(images)
        fixture = Dict(
            "source" => "VLFeat 1b9075f libvl.so, vl_mser_process; written by validation/mser_vlfeat.jl --freeze",
            "images" => Dict(n => Dict("size" => collect(size(imgs[n])), "bytes" => bytes2hex(vec(imgs[n])))
                             for n in unique(x.name for x in small)),
            "cases" => [Dict("image" => x.name, "polarity" => String(x.polarity), "parameters" => x.pname,
                             "seeds" => x.seeds) for x in small],
            "parameters" => Dict(p => Dict(String(k) => v for (k, v) in pairs(nt)) for (p, nt) in PARAMS),
        )
        path = joinpath(@__DIR__, "..", "test", "data", "mser_vlfeat.toml")
        mkpath(dirname(path))
        open(io -> TOML.print(io, fixture; sorted = true), path, "w")
        println("froze ", length(small), " cases to ", path)
    end
    return all(x -> x.ok, rows) && mapping_ok
end

main() || exit(1)
