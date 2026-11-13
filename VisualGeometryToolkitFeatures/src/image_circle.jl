# =============================================================================
# The image circle of a lens, MEASURED from photographs (moved from RadiometricCamera)
# =============================================================================
#
# The frames' per-pixel maximum quantized to the 8 bits VLFeat's MSER reads (an L0 smoothing first only when asked:
# ImageSmooth, the weak-dependency extension), the dark MSERs (`detect_mser`), every boundary of them off the frame's
# border, their convex hull (DelaunayTriangulation's `convex_hull`) and the RANSAC ellipse or circle through its
# vertices (VisualGeometryToolkitEstimation's `find_ellipse` / `find_circle`: the weak-dependency extension
# VisualGeometryToolkitFeaturesEstimationExt). A RAW frame is measured on its green cells and the result taken to the
# raw's pixels by the cells' `raw_frame_map`.
# =============================================================================

using DelaunayTriangulation: convex_hull, get_vertices

"""
    detect_image_circle([rng,] [T = Ellipse,] frames; smoothing_weight = 0, outlier_halfwidth, min_hole_area, border_margin = 5)
        -> (; circle::T, fit, contour, cut, rim_gap) | nothing

The image circle of a lens MEASURED from photographs `frames` — an iterable of images of one size (a vector, or a
generator that streams them: one camera, one lens, one focus, so the circle is the same in all of them) — in the frames'
COLMAP pixel frame. It is where light falls on the sensor, which no lens model predicts (a model's fold says where the
MODEL is valid, not where the field stop cuts the light). The lit region is not necessarily round: it is fitted as the
model type `T`, Core's `Ellipse` (the default) or `Circle`, on the same contour with the same keywords.

The fit is VisualGeometryToolkitEstimation's `find_ellipse` / `find_circle`, through this package's extension on it:
`using VisualGeometryToolkitEstimation` first (without it the call is an `ArgumentError` naming the package).
`smoothing_weight > 0` needs `using ImageSmooth` the same way.

The frames are gray images (anything `Colors.gray` reads: `Gray`, a real) or camera RAW images, never both:
- a `LinearRaw` of mode `:green_cells` or a `RawMosaic` (read as `LinearRaw(mosaic; mode = :green_cells)`) is
  measured on its GREEN CELLS, the geometry image, and `circle` and `contour` are taken to the raw's pixels by the cells'
  `raw_frame_map` (Core's similarity action on an `Ellipse` or a `Circle`). Only the RESULT is mapped: the keywords
  are in GREEN-CELL pixels, the resolution measured on (half the raw's: a pipeline config's values are at the cell
  resolution), the default `border_margin = 5` included. Every frame must have the same map (one sensor, one window),
  and a cell image on a device is read to the host;
- any other image is measured as it is, and its result is in its own pixels.

The method assumes a dark surround outside the image circle, as a lens's field stop gives. Each frame lights a different
part of the rim; their union of light does the measuring:

1. the per-pixel MAXIMUM over the frames (its sublevel set at a level is the intersection of the frames': the region dark
   in every frame);
2. it stored as `N0f8`, the 8-bit levels VLFeat's MSER reads, after an L0 gradient smoothing (ImageSmooth's `L0Smooth`,
   weight `smoothing_weight`) only when `smoothing_weight > 0`. Off by default: on a soft rim it erases the rim where the
   scene beside it is dark (on eleven Sigma 14 mm panorama frames with a dark floor across the lower third the smoothed
   maximum lost the lower rim and the ellipse came out (2119.7, 1254.4), 1676.6 / 1456.6 px; unsmoothed (2140.4, 1374.4),
   1746.4 / 1704.4 px, the rim seen all round);
3. its DARK maximally stable regions, [`detect_mser`](@ref)`(levels; bright_on_dark = false, max_area = 1,
   min_diversity = 1)`: every region up to the whole image, VLFeat's duplicate removal off. The rim is carried by the
   LARGEST dark regions, the surround and its near-copies, which VLFeat's defaults (`max_area` 0.75, duplicates
   removed) drop: on ten Sigma 14 mm frames the defaults kept 2066 of 4446 regions and moved the ellipse's centre 30 px
   off the rim;
4. the RIM: every boundary of every dark region, its outer boundary and its holes ([`region_mask`](@ref) and
   [`boundary_contours`](@ref)), less the holes smaller than `min_hole_area` (|shoelace area|: a hot pixel punches a
   1–5 px hole in the dark surround, and its corners pulled the hull off the rim) and less the vertices on the frame's
   border, pooled, and their CONVEX HULL. A disc inside the frame leaves the surround one region touching all four
   sides: its outer boundary is the frame and its HOLE is the rim. A dark region inside the disc (a dark floor, an
   object) lies inside the hull and cannot move it, so no region is singled out as the surround. The hull follows the
   surround's copy at the darkest stable level, on the DARK side of the rim's 1–2 px partial-coverage band: the fit
   comes out ≈ 1 px outside the true rim (+0.9 … +1.1 px on synthetic discs, the centre within 0.03 px), a mask that
   errs toward including the rim's last pixels;
5. the contour: the hull's vertices farther than `border_margin` from the frame's border. Where the disc reaches past
   the frame the hull closes the cut with an edge along the border; its vertices are dropped, so only the rim's arcs
   remain (`cut = true`);
6. the model: `find_ellipse(contour; scoring, rng)` for `Ellipse`, `find_circle(…)` for `Circle` — RANSAC with the
   robustified-LM local optimization, scored by the Student-t + uniform marginal (the inlier scale marginalized; the
   outliers uniform over ±`outlier_halfwidth`; its priors the stack's default, VisualGeometryToolkitEstimation's
   `scoring_from_toml` of `student_t_uniform`). The residual is the Sampson distance to the model in the contour's
   own pixels.

A disc CUT by the frame (the lit area reaching past the frame's edges, an APS-C lens on the full frame) is fitted as `T`
like any other; the axis across the cut is then set by the arcs' curvature alone, not by a rim seen. The circle is a
MASK of the lit region, returned whenever the fit finds one — also from a short arc, where its centre and radius are not
determined and only the boundary near the observed rim is measured (`rim_gap`).

Returns `nothing` when the dark regions have no boundary off the frame's border (the lens lights the whole frame), the
contour has fewer points than the model's minimal sample, or the fit finds no model (an `Ellipse` on a short,
near-straight run of hull vertices: every 5-point sample holds 3 collinear points). Otherwise:
- `circle`: the fitted model, an `Ellipse{Float64}` or a `Circle{Float64}` (`T`), COLMAP pixels (a RAW frame's: raw
  pixels);
- `fit`: the RANSAC's `RansacEstimate`; `inlier_mask(fit)` indexes `contour`. For RAW frames the RANSAC ran on the
  green cells: its estimate is mapped to raw pixels (Core's `map_estimate`, `estimate(fit) == circle`), but its `rms`
  and every residual or scale in its metadata stay in GREEN-CELL pixels, the keywords' (raw pixels ÷ 2);
- `contour`: the hull's vertices off the border, `Vector{Point2{Float64}}`, pixel corners, in `circle`'s pixels;
- `cut`: whether the disc is cut by the frame (the hull reached the border);
- `rim_gap`: the largest angle, radians, between consecutive INLIER contour vertices around the fitted centre. Below π
  the rim is seen on all sides of the centre and the centre and radius are measured; above π the observed rim is one
  arc, and the boundary away from it is the model's extrapolation.

# Keywords
- `outlier_halfwidth` (required, pixels > 0): the Student-t + uniform scorer's outlier half-width; a literal of the
  caller's, no computed default. The contour's outliers run off the rim from a few pixels to hundreds (eight Sony FX3 /
  Sigma 8 mm frames: median residual 3 px, 90 % quantile 336 px); 1000 full-resolution px held one basin on every set
  measured, for 750–2000.
- `min_hole_area` (required, square pixels ≥ 0): the smallest hole of a dark region whose boundary is kept. Outer
  boundaries are kept whatever their size. 256 full-resolution px² drops hot-pixel holes and inner-fringe pockets (the
  rim's own hole is 3.5–5.6 Mpx; 1024 starts losing true rim pockets).
- `smoothing_weight`: L0Smooth's λ; 0 (the default) skips the smoothing.
- `border_margin` (pixels; for RAW frames green-cell pixels): a hull vertex within it of the frame's border is dropped. The rim's last points where it
  meets the frame are 2–3 px inside the border, and the hull's edge across the cut runs through them, so the margin must
  exceed that.

# Examples
```julia
using VisualGeometryToolkitFeatures, VisualGeometryToolkitEstimation
using Random: Xoshiro
res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = 1000, min_hole_area = 256)
res === nothing || res.circle                         # an Ellipse, the frames' pixels
raws = (load_raw(f; mode = :green_cells) for f in files)
detect_image_circle(Xoshiro(1), Circle, raws; outlier_halfwidth = 500, min_hole_area = 64).circle   # keywords in cell px, circle in raw px
```
"""
detect_image_circle(frames; kwargs...) = detect_image_circle(Random.default_rng(), Ellipse, frames; kwargs...)
detect_image_circle(rng::AbstractRNG, frames; kwargs...) = detect_image_circle(rng, Ellipse, frames; kwargs...)
detect_image_circle(::Type{T}, frames; kwargs...) where {T} = detect_image_circle(Random.default_rng(), T, frames; kwargs...)
function detect_image_circle(rng::AbstractRNG, ::Type{T}, frames; smoothing_weight::Real = 0,
                             outlier_halfwidth::Real, min_hole_area::Real,
                             border_margin::Real = 5) where {T <: Union{Circle, Ellipse}}
    (isfinite(outlier_halfwidth) && outlier_halfwidth > 0) ||
        throw(ArgumentError("detect_image_circle: outlier_halfwidth must be > 0 px, got $outlier_halfwidth"))
    (isfinite(min_hole_area) && min_hole_area >= 0) ||
        throw(ArgumentError("detect_image_circle: min_hole_area must be ≥ 0 px², got $min_hole_area"))
    smoothing_weight >= 0 || throw(ArgumentError("detect_image_circle: smoothing_weight must be ≥ 0, got $smoothing_weight"))
    require_extension(() -> _missing_extension("VisualGeometryToolkitEstimation", "the fit (find_ellipse / find_circle)"),
                      @__MODULE__, :VisualGeometryToolkitFeaturesEstimationExt)
    upper, S = nothing, nothing                                       # 1. the per-pixel maximum, one frame at a time
    for f in frames
        plane, Sf = _detection_plane(f)
        if upper === nothing
            upper, S = Float32.(gray.(plane)), Sf
        else
            size(plane) == size(upper) || throw(DimensionMismatch("detect_image_circle: the frames differ in size (one camera, one lens)"))
            (Sf === nothing) == (S === nothing) || throw(ArgumentError("detect_image_circle: RAW and gray frames mixed"))
            Sf == S || throw(ArgumentError("detect_image_circle: the RAW frames' raw_frame_map differ (one sensor, one window, one mode)"))
            upper .= max.(upper, gray.(plane))
        end
    end
    upper === nothing && throw(ArgumentError("detect_image_circle: no frames"))
    H, W = size(upper)
    levels = Matrix{Gray{N0f8}}(undef, H, W)                          # 2. VLFeat's 8 bits (after an L0 smoothing only when asked)
    if smoothing_weight > 0
        require_extension(() -> _missing_extension("ImageSmooth", "the L0 smoothing (smoothing_weight > 0)"), @__MODULE__, :VisualGeometryToolkitFeaturesImageSmoothExt)
        _l0_smooth!(levels, upper, Float64(smoothing_weight))
    else
        levels .= Gray{N0f8}.(clamp.(upper, 0, 1))
    end
    upper = nothing
    regions = detect_mser(levels; bright_on_dark = false, max_area = 1.0, min_diversity = 1.0)   # 3. the dark MSERs
    points = Point2{Float64}[]                                        # 4. all their boundaries, outer and holes, off the border; the hull
    for r in regions
        # an MSER is one 8-connected region: its ONE outer boundary first, then its holes (boundary_contours' order, by signed area)
        for (k, c) in enumerate(boundary_contours(region_mask(r, levels)))
            k > 1 && abs(signed_area(c)) < min_hole_area && continue   # a small hole (a hot pixel in the surround): not the rim
            append!(points, _inside_frame(c, W, H))
        end
    end
    points = unique(points)
    length(points) < 3 && return nothing
    hull = points[get_vertices(convex_hull([(p[1], p[2]) for p in points]))[1:end-1]]   # closed: the first vertex repeats
    m = Float64(border_margin)                                         # 5. the hull's vertices off the border: the rim
    contour = [p for p in hull if m < p[1] < W - m && m < p[2] < H - m]
    cut = length(contour) < length(hull)
    fit = _fit_image_circle(T, contour; outlier_halfwidth = Float64(outlier_halfwidth), rng)   # 6. the model the caller asked for
    (fit === nothing || !found_model(fit)) && return nothing
    circle = estimate(fit)
    rim_gap = _rim_gap(circle, contour[inlier_mask(fit)])
    S === nothing && return (; circle, fit, contour, cut, rim_gap)
    return (; circle = S(circle), fit = map_estimate(S, fit), contour = [Point2{Float64}(S(SVector{2, Float64}(p[1], p[2]))) for p in contour],
            cut, rim_gap)
end

# The plane a frame is measured on, and the map of its pixels onto the returned ones (`nothing`: its own). A RAW image:
# the green-cell image, `LinearRaw(m; mode = :green_cells)` (the stack's one public path to it), read to the host, and
# its raw_frame_map. A cell image of another mode (`:bin2`, the display `:demosaic`) is refused, never re-read.
_detection_plane(f::AbstractMatrix) = (f, nothing)
_detection_plane(m::RawMosaic) = _detection_plane(LinearRaw(m; mode = :green_cells))
_detection_plane(L::LinearRaw{:green_cells}) = (Array(view(L.image, :, :, 1)), unlabel(raw_frame_map(L)))
_detection_plane(::LinearRaw{M}) where {M} = throw(ArgumentError(
    "detect_image_circle: a RAW frame is measured on its green cells: give the mosaic or LinearRaw(mosaic; mode = :green_cells), got LinearRaw{:$M}"
))
_detection_plane(f) = throw(ArgumentError("detect_image_circle: a frame is a gray image, a LinearRaw or a RawMosaic, got $(typeof(f))"))

# The largest angle between consecutive points around the model's centre, radians (2π for fewer than two points).
function _rim_gap(model, points)
    length(points) < 2 && return 2π
    c = model.center
    φ = sort!([atan(p[2] - c[2], p[1] - c[1]) for p in points])
    return max(maximum(diff(φ)), 2π - (φ[end] - φ[1]))
end

# The vertices of a boundary polygon not on the border of the `W × H` frame (COLMAP pixel corners: the border is x ∈ {0, W}, y ∈ {0, H}).
_inside_frame(poly, W, H) = [p for p in poly if 0.5 < p[1] < W - 0.5 && 0.5 < p[2] < H - 0.5]

# The extension hooks. VisualGeometryToolkitFeaturesEstimationExt adds `_fit_image_circle` for `Circle` and `Ellipse`
# (the RANSAC on the contour); VisualGeometryToolkitFeaturesImageSmoothExt adds `_l0_smooth!`. Without them loaded the
# call stops before any work, naming the package to load (VisualGeometryToolkitCore's `require_extension`); the fallbacks
# name it too, or raise the MethodError of arguments the loaded extension does not take.
_missing_extension(pkg, what) = ArgumentError("detect_image_circle: $what needs `using $pkg` (its package extension is not loaded)")
_fit_image_circle(T::Type, contour; outlier_halfwidth, rng) =
    require_extension(() -> _missing_extension("VisualGeometryToolkitEstimation", "the fit (find_ellipse / find_circle)"),
                      _fit_image_circle, (T, contour), @__MODULE__, :VisualGeometryToolkitFeaturesEstimationExt)
_l0_smooth!(levels, upper, λ) =
    require_extension(() -> _missing_extension("ImageSmooth", "the L0 smoothing (smoothing_weight > 0)"),
                      _l0_smooth!, (levels, upper, λ), @__MODULE__, :VisualGeometryToolkitFeaturesImageSmoothExt)
