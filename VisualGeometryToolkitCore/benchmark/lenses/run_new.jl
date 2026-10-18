# The redesigned lens families over a batch: pointwise! (broadcast) with the
# Serial / Threaded / Auto policies on the host and one broadcast kernel on CUDA.
#   CUDA_VISIBLE_DEVICES=3 julia --project=. -t 32 run_new.jl new.toml
using VisualGeometryToolkitCore, StaticArrays, BenchmarkTools, CUDA, Random, TOML
const VGC = VisualGeometryToolkitCore
_t(b) = minimum(b).time / 1e6
function _points(lens, N, ::Type{T}) where {T}
    rng = Xoshiro(1)
    ρmax = min(Float64(VGC.monotone_radius(lens)), 1.2) * 0.9
    [(r = ρmax * sqrt(rand(rng)); a = 2π * rand(rng); SVector{2, T}(r * cos(a), r * sin(a))) for _ in 1:N]
end
lenses = [
    "opencv5" => Lens(Perspective(), Additive(Radial((-0.2817, 0.0712, -0.0061)), Tangential(0.00091, -0.00042))),
    "kannala_brandt" => KannalaBrandt(-0.043, 0.011, -0.0083, 0.0015),
    "thin_prism_fisheye" => Lens(Equidistant(), Additive(Radial((-0.02, 0.004, -0.0009, 0.0001)), Tangential(0.0006, -0.0004), ThinPrism((0.0002,), (-0.0001,)))),
    "division1" => Division(-0.31), "division3" => Division(-0.2, 0.03, -0.004),
    "scaramuzza" => Scaramuzza(-0.17, -0.006, 0.035),
    "double_sphere" => Lens(DoubleSphere(-0.2, 0.59)),
]
out = Dict{String, Any}()
for T in (Float32, Float64)
    res = Dict{String, Any}()
    for (name, lens64) in lenses
        lens = VGC._precision(T, lens64)
        pts = _points(lens64, 10^6, T); o = similar(pts)
        r = Dict{String, Float64}()
        for (pname, exec) in (("serial", VGC.Serial()), ("threaded", VGC.Threaded()))
            r["cpu_$(pname)_undistort"] = _t(@benchmark undistort_normalized!($o, $lens, $pts; exec = $exec))
            r["cpu_$(pname)_distort"] = _t(@benchmark distort_normalized!($o, $lens, $pts; exec = $exec))
        end
        if CUDA.functional()
            cp = CuArray(pts); co = similar(cp)
            r["gpu_undistort"] = _t(@benchmark CUDA.@sync undistort_normalized!($co, $lens, $cp))
            r["gpu_distort"] = _t(@benchmark CUDA.@sync distort_normalized!($co, $lens, $cp))
        end
        res[name] = r
    end
    out[string(T)] = res
end
# the crossover of Serial and Threaded, by batch length (Auto's THREADED_MIN_LENGTH)
cross = Dict{String, Any}()
for (name, lens) in lenses[[1, 4]], n in (256, 1024, 4096, 16384, 65536)
    pts = _points(lens, n, Float64); o = similar(pts)
    cross["$(name)_$(n)_serial"] = _t(@benchmark undistort_normalized!($o, $lens, $pts; exec = VGC.Serial()))
    cross["$(name)_$(n)_threaded"] = _t(@benchmark undistort_normalized!($o, $lens, $pts; exec = VGC.Threaded()))
end
out["crossover"] = cross
open(io -> TOML.print(io, out; sorted = true), ARGS[1], "w")
