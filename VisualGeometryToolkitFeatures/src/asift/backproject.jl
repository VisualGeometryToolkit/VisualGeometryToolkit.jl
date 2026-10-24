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
- `s_xx`, `s_xy`, `s_yy`: the symmetric 2×2 shape matrix of the elliptic frame,
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

function _imas_tilts_25()
    ts = Tuple{Float64, Float64}[(1.0, 0.0)]
    for k in 0:7
        push!(ts, (2.62, π * k / 8))
    end
    for k in 0:15
        push!(ts, (5.18, π * k / 16))
    end
    return ts
end
