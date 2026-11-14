# =============================================================================
# The image circle of a lens, MEASURED from photographs (moved from RadiometricCamera)
# =============================================================================
#
# The frames' per-pixel maximum, resampled on a log-polar grid about the image centre (Core's warp), each angle's
# significant drops in log brightness the rim's candidates, and the RANSAC ellipse or circle through them
# (VisualGeometryToolkitEstimation's `find_ellipse` / `find_circle`: the weak-dependency extension
# VisualGeometryToolkitFeaturesEstimationExt): the lens COVERAGE, where its light ends (image_circle_rim.jl). A RAW frame
# is measured on its green cells and the result taken to the raw's pixels by the cells' `raw_frame_map`.
# =============================================================================

"""
    detect_image_circle([rng,] [T = Ellipse,] frames; outlier_halfwidth, initial = nothing, rim_band = (0.8, 0.0),
                        rim_angles = :auto, significance = 0.01,
                        device = default_device())
        -> (; circle::T, fit, contour, cut, rim_gap, covariance, center_scale, axes_scale, rim::RimProfile) | nothing

The image circle of a lens MEASURED from photographs `frames` — an iterable of images of one size (a vector, or a
generator that streams them: one camera, one lens, one focus, so the circle is the same in all of them) — in the frames'
COLMAP pixel frame: the lens COVERAGE, where its light ends. No lens model predicts it (a model's fold says where the
MODEL is valid, not where the field stop or the lens's own image circle cuts the light). The lit region is not
necessarily round: it is fitted as the model type `T`, Core's `Ellipse` (the default) or `Circle`.

The fit is VisualGeometryToolkitEstimation's `find_ellipse` / `find_circle`, through this package's extension on it:
`using VisualGeometryToolkitEstimation` first (without it the call is an `ArgumentError` naming the package).

The frames are gray images (anything `Colors.gray` reads: `Gray`, a real) or camera RAW images, never both:
- a `LinearRaw` of mode `:green_cells` or a `RawMosaic` (read as `LinearRaw(mosaic; mode = :green_cells)`) is
  measured on its GREEN CELLS, the geometry image, its saturated cells excluded, and the result is taken to the raw's
  pixels by the cells' `raw_frame_map` (Core's similarity action on an `Ellipse` or a `Circle`). The keywords in pixels
  are in GREEN-CELL pixels, the resolution measured on (half the raw's). Every frame must have the same map (one sensor,
  one window), and a cell image on a device is read to the host. Cells read under a sensor model with a variance law
  (an `Uncertain` image) carry each value's variance, and the rim's noise is then the MODEL's
  ([`SensorModelNoise`](@ref)); otherwise it is the data's ([`EmpiricalNoise`](@ref)). The type decides;
- any other image is measured as it is (its white, 1, is its saturation), and its result is in its own pixels.

Each frame lights a different part of the rim, and a scene's own edges move between frames; their union of light does
the measuring:

1. the per-pixel MAXIMUM over the frames (a region dark in one frame is filled by another; a scene edge is washed out),
   saturated where any frame is;
2. resampled by VisualGeometryToolkitCore's log-polar warp (`logpolar_map`, `image_warp!`) about the initial model's
   centre — `initial`, by default the image centre with the half diagonal for radius — over the radii
   `(1 − rim_band[1])·min(axes)` … `(1 + rim_band[2])·max(axes)` (the default: 0.2…1 of the half diagonal, every rim
   inside the frame down to a fifth of it), at most 1 px apart in radius (`Δ(ln r) ≤ 1/r_max`) and in arc
   (`Δθ ≤ 1/r_max`) at the outer end, so every pixel of the band contributes. The sample rows are grouped into angle
   bins and every (angle bin, radius) reduced to its MEDIAN (one KernelAbstractions kernel), both on `device`. A bin is
   skipped when a sample's bilinear taps leave the frame or more than half its samples are saturated. The angle bins
   (`rim_angles = :auto`) are chosen from the image: of the binnings doubling from 45 (8°) to 3 sample rows per bin,
   the one whose coverage is the most precise (the smallest standard error of its mean radius, the inliers' rms over
   √(wedges with an inlier), the rms no finer than the edges' own median uncertainty nor the radial grid's spacing over
   √12, with at least 20 such wedges). Finer bins give more wedges but noisier medians and, in a
   textured scene, the scene's edges among the candidates;
3. per angle, the CANDIDATES: every EDGE of the profile's OUTER ENVELOPE — the brightest it gets at or beyond each
   radius: where the light ends it does not come back, while a scene edge or a dark object inside the disc is followed
   by light again and leaves the envelope alone. Every radius is in pixels through the log-polar map, its differential
   and its inverse (Core's `transform_deriv`, `inv`): the drop at a radius r is across exactly 2 px about it (the
   bilinear taps' support; its ends r ± 1 px through the inverse map, the profile interpolated between columns), so it
   is continuous in r. An edge is a maximal run of significant drops; its candidate is the steepest drop in LOG
   brightness per pixel from the run's start to the next edge's (the log floored at the angle's darkest level or its
   noise), at its sub-sample peak in the grid coordinate taken to pixels by the map. A log drop is a ratio, independent
   of the exposure: a blurred sharp stop's steepest log drop is at the stop, a soft ramp's at its foot, where it meets
   the floor; a rim that falls in two steps (lit, a dim band, the
   floor) gives both (owner ruling 2026-11-14). A drop is significant when it exceeds its noise √(σ(r − 1)² + σ(r + 1)²)
   (σ each bin's noise at its pixels, rows × arc from the map's differential: the sensor model's, or
   the data's) by z = Φ⁻¹(1 − α/m), `α = significance` the family-wise level over the
   angle's m drops: the floor's noise alone makes an edge in an angle with probability at most α;
4. the model: `find_ellipse(candidates; scoring, rng)` for `Ellipse`, `find_circle(…)` for `Circle` — RANSAC with the
   robustified-LM local optimization, scored by the Student-t + uniform marginal (the inlier scale marginalized; the
   outliers uniform over ±`outlier_halfwidth`; its priors the stack's default, VisualGeometryToolkitEstimation's
   `scoring_from_toml` of `student_t_uniform`), on ONE candidate per angle, so the support counts angles and an angle
   with many edges does not dominate: first each angle's OUTERMOST edge (where its light ends); then each angle's edge
   NEAREST that model, any of its edges (stray light, or a step off the model, gives way), fitted
   again: the coverage. Per angle that edge, an inlier, is the rim's edge there (`rim.edge`); an angle whose rim is
   hidden (an object against the lens) is an outlier.

Returns `nothing` when no model is found (fewer than 5 candidates, or no consensus: a lens that lights the whole frame
has no rim) or the model contains the whole frame (its light covers the sensor). Otherwise:
- `circle`: the lens coverage, an `Ellipse{Float64}` or a `Circle{Float64}` (`T`), COLMAP pixels (a RAW frame's: raw
  pixels);
- `fit`: the second RANSAC's `RansacEstimate`, on one candidate per angle that has any (each angle's edge nearest the
  first model, in angle order); `inlier_mask(fit)` indexes those points. For RAW frames the RANSAC ran
  on the green cells: its estimate is mapped to raw pixels (Core's `map_estimate`, `estimate(fit) == circle`), but its
  `rms` and every residual or scale in its metadata stay in GREEN-CELL pixels (raw pixels ÷ 2);
- `contour`: the rim's points, `Vector{Point2{Float64}}`, in `circle`'s pixels: per angle bin with one, the inlier
  candidate nearest the coverage (`rim.edge`);
- `cut`: whether the coverage reaches past the frame (an APS-C lens on the full frame: the rim is seen in arcs);
- `rim_gap`: the largest angle, radians, between consecutive inliers around the coverage's centre. Below π the rim is
  seen on all sides of the centre and the centre and radius are measured; above π the observed rim is one arc, and the
  boundary away from it is the model's extrapolation;
- `covariance`: the coverage's covariance in Core's `ambient_coordinates` of the MEASURED plane (an ellipse's centre and
  form `(Q₁₁, Q₁₂, Q₂₂)`; a circle's centre and radius): `s²·solver_covariance` on the inliers, `s²` their squared
  distances to the model along their rays over n − dof (`nothing` when singular, and for a `Circle`: its problem
  provides no solver covariance); `center_scale`, `axes_scale` the standard deviations of the centre and of the
  semi-axes `(a, b)` in `circle`'s pixels (`NaN` without a covariance);
- `rim`: the measurement ([`RimProfile`](@ref)): the profiles, every angle's candidates and edge, the noise statistics.

# Keywords
- `outlier_halfwidth` (required, pixels > 0): the Student-t + uniform scorer's outlier half-width; a literal of the
  caller's, no computed default. 1000 full-resolution px held one basin on every set measured, for 750–2000.
- `initial`: the initial model (the result's pixels): its centre is the log-polar origin, its axes scale the band. By
  default the image centre with the half diagonal (no prior: a camera's principal point is a better centre when known).
- `rim_band` = `(inner, outer)`, fractions of the initial model's axes: the radii resampled, from `(1 − inner)·min(axes)`
  to `(1 + outer)·max(axes)`. The default `(0.8, 0.0)`, with the default initial (the half diagonal), resamples from a
  fifth of the half diagonal out to the corners. The outer end is derived: the corners are the farthest any light reaches
  on the sensor. The inner end is a stated ASSUMPTION, not a fit: the coverage reaches at least a fifth of the half
  diagonal (a lens whose light ends nearer the centre leaves 96 % of the sensor dark, no use as a camera). It cannot be
  0 because the log grid needs a positive inner radius, and the band's cost grows only as ln(1/0.2)·r_max radial
  samples. Narrow it when the coverage is known roughly, e.g. from a previous run.
- `rim_angles`: the angle bins, `:auto` (chosen as in step 2) or a count ≥ 5.
- `significance`: the edge test's family-wise level per angle: the probability that the floor's noise alone makes an
  edge anywhere in an angle. 0.01 is the conventional 1 % level, stated, not tuned. A false edge is one more candidate,
  and it is not consistent with the coverage, so the RANSAC rejects it as an outlier; the level only trades such
  outliers against missing a faint rim's edge.
- `device`: the KernelAbstractions backend of the warp and the bin medians (`default_device()`: the GPU when one is
  functional).
- `min_hole_area`, `smoothing_weight`, `border_margin`: the hull fit's keywords before 0.9.2, accepted and ignored with a
  deprecation warning (the hull is gone).

# Examples
```julia
using VisualGeometryToolkitFeatures, VisualGeometryToolkitEstimation
using Random: Xoshiro
res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = 1000)
res === nothing || (res.circle, res.axes_scale)          # the coverage, an Ellipse, the frames' pixels, and its σ
raws = (load_raw(f; mode = :green_cells) for f in files)
detect_image_circle(Xoshiro(1), Circle, raws; outlier_halfwidth = 500).circle   # keywords in cell px, circle in raw px
```
"""
detect_image_circle(frames; kwargs...) = detect_image_circle(Random.default_rng(), Ellipse, frames; kwargs...)
detect_image_circle(rng::AbstractRNG, frames; kwargs...) = detect_image_circle(rng, Ellipse, frames; kwargs...)
detect_image_circle(::Type{T}, frames; kwargs...) where {T} = detect_image_circle(Random.default_rng(), T, frames; kwargs...)
function detect_image_circle(rng::AbstractRNG, ::Type{T}, frames; outlier_halfwidth::Real,
                             initial::Union{Nothing, Circle, Ellipse} = nothing, rim_band = (0.8, 0.0),
                             rim_angles::Union{Integer, Symbol} = :auto,
                             significance::Real = 0.01, device = default_device(),
                             min_hole_area = nothing, smoothing_weight = nothing, border_margin = nothing) where {T <: Union{Circle, Ellipse}}
    (isfinite(outlier_halfwidth) && outlier_halfwidth > 0) ||
        throw(ArgumentError("detect_image_circle: outlier_halfwidth must be > 0 px, got $outlier_halfwidth"))
    (rim_angles === :auto || (rim_angles isa Integer && rim_angles >= 5)) ||
        throw(ArgumentError("detect_image_circle: rim_angles must be :auto or ≥ 5 (the fit's minimal sample), got $rim_angles"))
    (0 < rim_band[1] < 1 && rim_band[2] >= 0) ||
        throw(ArgumentError("detect_image_circle: rim_band = (inner, outer) needs 0 < inner < 1 and outer ≥ 0, got $rim_band"))
    0 < significance < 1 || throw(ArgumentError("detect_image_circle: significance must be in (0, 1), got $significance"))
    for (k, v) in ((:min_hole_area, min_hole_area), (:smoothing_weight, smoothing_weight), (:border_margin, border_margin))
        v === nothing || Base.depwarn("detect_image_circle: `$k` was the hull fit's keyword and is ignored since " *
                                      "VisualGeometryToolkitFeatures 0.9.2 (the coverage is measured on the rim's profiles)", :detect_image_circle)
    end
    require_extension(() -> _missing_extension("VisualGeometryToolkitEstimation", "the fit (find_ellipse / find_circle)"),
                      @__MODULE__, :VisualGeometryToolkitFeaturesEstimationExt)
    upper, S, sat, variance, nframes = _frame_maximum(frames)          # 1. the per-pixel maximum
    H, W = size(upper)
    plane_initial = initial === nothing ? Ellipse(Point2(W / 2, H / 2), hypot(W, H) / 2, hypot(W, H) / 2, 0.0) :
                    S === nothing ? initial : inv(S)(initial)
    rim, fit, pts = _rim_measure(rng, T, upper, sat, variance, nframes, plane_initial;   # 2.–4. the rim and the coverage
                                 nangles = rim_angles,
                                 band = (Float64(rim_band[1]), Float64(rim_band[2])),
                                 significance = Float64(significance), outlier_halfwidth = Float64(outlier_halfwidth), device)
    fit === nothing && return nothing
    circle = estimate(fit)
    corners = (Point2(0.0, 0.0), Point2(W, 0.0), Point2(0.0, H), Point2(W, H))
    all(q -> norm(q - circle.center) <= _model_radius(circle, (q - circle.center) / norm(q - circle.center)), corners) &&
        return nothing                                                 # its light covers the sensor: no rim
    contour = [rim.center + rim.edge[a] * Point2(cos(rim.angles[a]), sin(rim.angles[a])) for a in eachindex(rim.angles) if rim.reason[a] === :ok]
    cut = any(φ -> (q = circle.center + _model_radius(circle, SVector(cos(φ), sin(φ))) * Point2(cos(φ), sin(φ));
                   !(0 <= q[1] <= W && 0 <= q[2] <= H)), range(0, 2π; length = 721))
    rim_gap = _rim_gap(circle, contour)
    u = _coverage_uncertainty(circle, contour)
    S === nothing && return (; circle, fit, contour, cut, rim_gap, u..., rim)
    s = scale(S)                                                       # the cells' similarity: its scale
    return (; circle = S(circle), fit = map_estimate(S, fit), contour = [Point2{Float64}(S(SVector{2, Float64}(p[1], p[2]))) for p in contour],
            cut, rim_gap, covariance = u.covariance, center_scale = s .* u.center_scale, axes_scale = s .* u.axes_scale, rim)
end

# The frames' per-pixel maximum, one frame at a time: the maximum (Float32), the map of the measured plane onto the
# result's pixels (`nothing`: its own), the saturation (where any frame is), the maximum's own variance under a sensor
# model (`nothing` without one) and the frame count.
function _frame_maximum(frames)
    upper, S, sat, variance, nframes = nothing, nothing, nothing, nothing, 0
    for f in frames
        nframes += 1
        plane, Sf, satf, varf = _detection_plane(f)
        if upper === nothing
            upper, S, sat = Float32.(gray.(plane)), Sf, BitMatrix(satf)
            variance = varf === nothing ? nothing : Float32.(varf)
        else
            size(plane) == size(upper) || throw(DimensionMismatch("detect_image_circle: the frames differ in size (one camera, one lens)"))
            (Sf === nothing) == (S === nothing) || throw(ArgumentError("detect_image_circle: RAW and gray frames mixed"))
            Sf == S || throw(ArgumentError("detect_image_circle: the RAW frames' raw_frame_map differ (one sensor, one window, one mode)"))
            (varf === nothing) == (variance === nothing) ||
                throw(ArgumentError("detect_image_circle: frames with and without a sensor model's variances mixed"))
            variance === nothing || (variance .= ifelse.(gray.(plane) .> upper, varf, variance))   # the maximum's own variance
            upper .= max.(upper, gray.(plane))
            sat .|= satf
        end
    end
    upper === nothing && throw(ArgumentError("detect_image_circle: no frames"))
    return upper, S, sat, variance, nframes
end

# The plane a frame is measured on, the map of its pixels onto the returned ones (`nothing`: its own), its saturation
# and its values' variances (`nothing` without a sensor model's). A RAW image: the green-cell image,
# `LinearRaw(m; mode = :green_cells)` (the stack's one public path to it), read to the host, its raw_frame_map and its
# saturated cells. A cell image of another mode (`:bin2`, the display `:demosaic`) is refused, never re-read.
_detection_plane(f::AbstractMatrix) = (f, nothing, gray.(f) .>= 1, nothing)   # a gray frame's white: its saturation
_detection_plane(m::RawMosaic) = _detection_plane(LinearRaw(m; mode = :green_cells))
function _detection_plane(L::LinearRaw{:green_cells})
    values, variance = _cell_values(Array(green_plane(L)))
    return (values, unlabel(raw_frame_map(L)), Array(L.saturated), variance)
end
_detection_plane(::LinearRaw{M}) where {M} = throw(ArgumentError(
    "detect_image_circle: a RAW frame is measured on its green cells: give the mosaic or LinearRaw(mosaic; mode = :green_cells), got LinearRaw{:$M}"
))
_detection_plane(f) = throw(ArgumentError("detect_image_circle: a frame is a gray image, a LinearRaw or a RawMosaic, got $(typeof(f))"))

# A cell image's values and, under a sensor model with a variance law (an Uncertain image, ruling D11), each value's
# variance; a plain image has none (the rim's noise then comes from the data).
_cell_values(img::AbstractMatrix{<:Uncertain}) = (value_of.(img), var.(img))
_cell_values(img::AbstractMatrix) = (img, nothing)

# The largest angle between consecutive points around the model's centre, radians (2π for fewer than two points).
function _rim_gap(model, points)
    length(points) < 2 && return 2π
    c = model.center
    φ = sort!([atan(p[2] - c[2], p[1] - c[1]) for p in points])
    return max(maximum(diff(φ)), 2π - (φ[end] - φ[1]))
end

# The extension hooks. VisualGeometryToolkitFeaturesEstimationExt adds `_fit_image_circle` for `Circle` and `Ellipse`
# (the RANSAC on the candidates) and `_coverage_covariance` (its unit-noise covariance). Without it loaded the call stops
# before any work, naming the package to load (VisualGeometryToolkitCore's `require_extension`); the fallbacks name it
# too, or raise the MethodError of arguments the loaded extension does not take.
_missing_extension(pkg, what) = ArgumentError("detect_image_circle: $what needs `using $pkg` (its package extension is not loaded)")
_fit_image_circle(T::Type, contour; outlier_halfwidth, rng) =
    require_extension(() -> _missing_extension("VisualGeometryToolkitEstimation", "the fit (find_ellipse / find_circle)"),
                      _fit_image_circle, (T, contour), @__MODULE__, :VisualGeometryToolkitFeaturesEstimationExt)
_coverage_covariance(m, pts) =
    require_extension(() -> _missing_extension("VisualGeometryToolkitEstimation", "the coverage's covariance"),
                      _coverage_covariance, (m, pts), @__MODULE__, :VisualGeometryToolkitFeaturesEstimationExt)
