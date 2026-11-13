# =============================================================================
# IMAS feature record + the IMAS-25 tilt schedule.
# =============================================================================
#
# Back-projection itself runs on-device in driver.jl (`_bbackproject!`).
# =============================================================================

"""
    ImasFeatureKA

One raw feature of the ASIFT detector, as [`asift_detect!`](@ref) and
`asift_detect(img; raw = true)` return it, before conversion to a
[`Detection`](@ref) by [`asift_detections`](@ref). All fields are in the input
image's frame:

- `x`, `y`: the refined centre in 0-indexed OpenCV pixel coordinates (top-left
  pixel centre at (0, 0), 0.5 less than the COLMAP coordinates of a `Detection`);
- `σ`: the characteristic scale, in input pixels;
- `intensity`: the DoG response at the extremum, positive for a dark blob and
  negative for a light one;
- `s_xx`, `s_xy`, `s_yy`: the symmetric 2×2 shape factor of the elliptic frame,
  back-projected from the tilted warp it was found in;
- `warp_idx`: the index of that warp in the plan's tilt schedule (1 is the
  identity).

```jldoctest
julia> using VisualGeometryToolkitFeatures

julia> f = ImasFeatureKA(10.0, 20.0, 2.0, -0.1, 2.0, 0.0, 2.0, 1);

julia> d = only(asift_detections([f]));

julia> d.center, is_light(d)
([10.5, 20.5], true)
```
"""
struct ImasFeatureKA
    x::Float32; y::Float32; σ::Float32; intensity::Float32
    s_xx::Float32; s_xy::Float32; s_yy::Float32
    warp_idx::Int32
end

"""
    tilt_schedule(name::Symbol) -> Vector{Tuple{Float64, Float64}}
    tilt_schedule(tilts) -> Vector{Tuple{Float64, Float64}}

The simulated camera tilts ASIFT detects under, as `(t, φ)` pairs (tilt and
longitude in radians), by name or as given (`tilts`, any collection of pairs):

- `:imas25` — IMAS-25 (the default of [`asift_detect`](@ref) and
  [`AsiftPlan`](@ref)): the identity, 8 tilts of 2.62 and 16 of 5.18 at evenly
  spaced longitudes. Affine invariance for wide-baseline views.
- `:untilted` — the identity warp alone, `[(1.0, 0.0)]`: plain DoG-SIFT. Its
  detections localize best: on real panorama frames a tilted warp's detections
  land about 0.95 px rms from the untilted view's of the same structure, the
  untilted's about 0.3 px from debiased SIFT, so for views that differ by a
  rotation (a panorama) it is the better schedule.

Any other name is an `ArgumentError`.
"""
function tilt_schedule(name::Symbol)
    name === :untilted && return [(1.0, 0.0)]
    name === :imas25 || throw(ArgumentError("tilt_schedule: the schedules are :imas25 and :untilted, got :$name"))
    ts = Tuple{Float64, Float64}[(1.0, 0.0)]
    for k in 0:7
        push!(ts, (2.62, π * k / 8))
    end
    for k in 0:15
        push!(ts, (5.18, π * k / 16))
    end
    return ts
end
tilt_schedule(tilts) = collect(Tuple{Float64, Float64}, tilts)
