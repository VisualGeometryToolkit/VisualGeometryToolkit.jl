# Can a KernelAbstractions kernel read a mosaic's channelview as it is, and at what bandwidth?
# One cell kernel (LinearRaw's: four photosites per cell, linear_value, the bin2/green-cell stores,
# the saturation flag), four ways of reading a cell's photosites:
#
#   view 4×h×w      channelview(m)[k, i, j]  (Base.ReshapedArray of a PermutedDimsArray of a reshape)
#   permuted 2×2×h×w parent(channelview(m))[di + 1, dj + 1, i, j]  (the PermutedDimsArray inside it)
#   packed 4×h×w    the view materialised once into a contiguous array, [k, i, j]
#   held matrix     parent(m)[2i − 1 + di, 2j − 1 + dj]  (LinearRaw's kernel: the view's index rule, no wrapper)
#
# for an even mosaic and for a window with an odd start (its held array a SubArray), on the GPU when one
# is functional and on the CPU backend: compiles or not, the output against the held-matrix kernel, the
# time (BenchmarkTools, min of 50) and the effective bandwidth (bytes read + written per launch).
#
#   julia --project=VisualGeometryToolkitCore/benchmark --startup-file=no VisualGeometryToolkitCore/benchmark/raw_channelview.jl
using BenchmarkTools, CUDA
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
import VisualGeometryToolkitCore as VgtkCore
const adapt = VgtkCore.Adapt.adapt   # Core's Adapt (the benchmark environment holds no Adapt of its own)
using VisualGeometryToolkitCore: RawMosaic, LinearRaw, channelview, bayer_levels, linear_value, OnBackend

const QUAD = ((0, 0), (1, 0), (0, 1), (1, 1))
struct View end;     struct Permuted end;     struct Packed end;     struct Held end
@inline photosite(::Union{View, Packed}, a, k, i, j) = @inbounds a[k, i, j]
@inline photosite(::Permuted, a, k, i, j) = @inbounds a[QUAD[k][1] + 1, QUAD[k][2] + 1, i, j]
@inline photosite(::Held, a, k, i, j) = @inbounds a[2i - 1 + QUAD[k][1], 2j - 1 + QUAD[k][2]]

@kernel function cells_kernel!(img, sat, @Const(a), how, planes, sites, black, span, gain::Float64, white::Float64)
    i, j = @index(Global, NTuple)
    @inbounds begin
        x = ntuple(k -> photosite(how, a, k, i, j), Val(4))
        v = ntuple(k -> linear_value(x[k], black[k], span[k]), Val(4))
        VgtkCore._store_planes!(img, i, j, v, planes, gain, 1)
        sat[i, j] = any(k -> x[k] >= white, sites)
    end
end

@kernel function copy_kernel!(out, @Const(a))
    I = @index(Global, Cartesian)
    @inbounds out[I] = a[I]
end

function run(backend, toarr, H, W, window)
    host = RawMosaic(UInt16[(i * 7919 + j * 104729) % 16000 + 200 for i in 1:H, j in 1:W], UInt8[0 1; 3 2],
                     (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 0), (H, W), (;))
    full = adapt(toarr, host)
    m = window ? view(full, 2:H, 2:W) : full
    v = channelview(m)
    h, w = size(v, 2), size(v, 3)
    packed = toarr(collect(channelview(window ? view(host, 2:H, 2:W) : host)))
    black, span = bayer_levels(m)
    planes, sites = ((1,), (2, 3), (4,)), (1, 2, 3, 4)          # bin2 of an RGGB cell (GBRG from an odd window: same layout use)
    img = similar(parent(m), Float64, h, w, 3); sat = similar(parent(m), Bool, h, w)
    bytes = 2 * 4h * w + 8 * h * w * 3 + h * w
    launch(how, a) = (OnBackend(backend)(cells_kernel!)(img, sat, a, how, planes, sites, black, span, 1.0, 16383.0; ndrange = (h, w));
                      KA.synchronize(backend))
    ref = (launch(Held(), parent(m)); Array(img))
    rows = String[]
    for (name, how, a) in (("view 4×h×w (channelview)", View(), v), ("permuted 2×2×h×w (its parent)", Permuted(), parent(v)),
                           ("packed 4×h×w", Packed(), packed), ("held matrix + index rule (LinearRaw's kernel)", Held(), parent(m)))
        row = try
            launch(how, a)
            same = Array(img) == ref
            t = @belapsed $launch($how, $a) samples = 50 evals = 1
            "$(round(t * 1e3; digits = 3)) ms | $(round(bytes / t / 1e9; digits = 1)) GB/s | $(same ? "= held" : "DIFFERS")"
        catch e
            "does not compile/run | — | $(nameof(typeof(e))): $(replace(first(sprint(showerror, e), 140), '\n' => ' ', '|' => '/'))"
        end
        push!(rows, "| $(nameof(typeof(backend))) | $(window ? "window (2:H, 2:W)" : "even, whole") | $name | $row |")
    end
    # a plain copy through the view (Cartesian), and a broadcast over it
    extra = String[]
    for (what, f) in (("copy kernel over the view", () -> (o = similar(packed); copy_kernel!(backend)(o, v; ndrange = size(v)); KA.synchronize(backend); Array(o) == Array(packed))),
                      ("broadcast linear_value.(view, …)", () -> (r = linear_value.(v, toarr(reshape(collect(black), 4, 1, 1)), toarr(reshape(collect(span), 4, 1, 1))); typeof(r).name.name)),
                      ("adapt(Array, view) == host view", () -> adapt(Array, v) == channelview(window ? view(host, 2:H, 2:W) : host)))
        push!(extra, "$(nameof(typeof(backend))) $(window ? "window" : "whole"): $what: " *
              try string(f()) catch e "$(nameof(typeof(e))): $(replace(first(sprint(showerror, e), 140), '\n' => ' '))" end)
    end
    return rows, extra, typeof(v)
end

H, W = 2848, 4256     # the FX3's visible area, rounded to whole cells
CUDA.allowscalar(false)
println("| backend | mosaic | input | time (min of 50) | effective bandwidth | output |")
println("|---|---|---|---|---|---|")
notes = String[]
backends = CUDA.functional() ? [(CUDA.CUDABackend(), CUDA.CuArray), (KA.CPU(), Array)] : [(KA.CPU(), Array)]
for (backend, toarr) in backends
    for window in (false, true)
        rows, extra, T = run(backend, toarr, H, W, window)
        foreach(println, rows)
        append!(notes, extra); push!(notes, "view type: $T")
    end
end
println(); foreach(println, notes)
