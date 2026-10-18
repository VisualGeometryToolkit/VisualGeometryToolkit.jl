# =============================================================================
# Raster overlays — array plotting with no Makie, for masks and debug frames
# =============================================================================
#
# Stamping shapes straight into a pixel array (a training mask, a debug frame
# written to video) must work under any backend and with no display stack.
# Points are in detector (x = column, y = row) coordinates; `w` is the
# half-thickness in pixels (`w = 0` is a single pixel). Hoisted from
# BlobBoards' training pipeline (2026-09).
# =============================================================================

"""
    put_pixel!(img, x, y, c; w = 0)

Stamp a filled `(2w + 1)²` block of value `c` centred on `(x, y)` (detector
coordinates, rounded to the nearest pixel), in-bounds pixels only.
"""
@inline function put_pixel!(img, x, y, c; w = 0)
    r = round(Int, y); col = round(Int, x)
    for dr in -w:w, dc in -w:w
        rr = r + dr; cc = col + dc
        (1 <= rr <= size(img, 1) && 1 <= cc <= size(img, 2)) && (img[rr, cc] = c)
    end
    return
end

"""
    draw_ellipse!(img, center, S, cutoff, color; n = 180, w = 2)

Draw the `cutoff`·σ boundary of the elliptic frame `(center, S)` (image
coordinates): boundary point = `center + cutoff · S · [cos t, sin t]`, `n`
samples around the ring, each stamped `w` pixels thick.
"""
function draw_ellipse!(img, center, S, cutoff, color; n = 180, w = 2)
    for t in range(0, 2π; length = n)
        v = center .+ cutoff .* (S * SVector(cos(t), sin(t)))
        put_pixel!(img, v[1], v[2], color; w)
    end
    return
end

"""
    draw_projected_ellipse!(img, H, center, S, cutoff, color; n = 180, w = 2)

Draw the `cutoff`·σ boundary of a WORLD-plane elliptic frame after projecting
each boundary point through the world→image homography `H` (the map before the
shape it maps, after the canvas it draws on). The projection of
an ellipse is only approximately an ellipse under perspective, so points are
mapped one by one rather than through a shape matrix; samples beyond the
plane's horizon are skipped (chirality-guarded [`to_euclidean`](@ref)), never
wrapped into view.
"""
function draw_projected_ellipse!(img, H::AbstractMatrix, center, S, cutoff::Real, color; n = 180, w = 2)
    w_ref = homogeneous_w(H, SVector(Float64(center[1]), Float64(center[2])))
    for t in range(0, 2π; length = n)
        v = center .+ cutoff .* (S * SVector(cos(t), sin(t)))
        p = to_euclidean(H * SVector(v[1], v[2], 1.0), w_ref)
        p === nothing && continue
        put_pixel!(img, p[1], p[2], color; w)
    end
    return
end

"""
    draw_polyline!(img, pts, color; n = 300, w = 2)

Draw the closed polyline through `pts` (each an `(x, y)` tuple or vector),
sampling every edge `n` times so no Bresenham is needed.
"""
function draw_polyline!(img, pts, color; n = 300, w = 2)
    m = length(pts)
    for i in 1:m
        a = pts[i]; b = pts[mod1(i + 1, m)]
        for s in range(0, 1; length = n)
            put_pixel!(img, a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2]), color; w)
        end
    end
    return
end
