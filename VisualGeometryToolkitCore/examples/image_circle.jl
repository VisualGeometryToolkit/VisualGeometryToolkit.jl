# The image circle — where a lens lights the sensor, predicted, never segmented.
# A local lens model (a grid of nodes) lives on `Rect(extent) ∩ image_circle`.
#
#   julia --project=. examples/image_circle.jl

using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: Pinhole, KannalaBrandt

# ── 1. A circular fisheye bounded by its physical sensor (2448 × 2048 @ 3.45 μm).
# Kannala–Brandt r = θ(1 + k θ²) folds at θ = 1/√(−3k): k below puts the fold at
# 90° (a 180° lens), where r = (2/3)·θ. The datasheet states a 6 mm circle.
s = sensor("Sony", "IMX250")
k = -4 / (3π^2)
f = ustrip(px, 6.0mm / (2 * s.pitch.width)) / (π / 3)   # f·r(90°) = the circle's radius in px
cam = BoundedCamera(CameraModel(; focal = f, pp = (1224.0, 1024.0), lens = KannalaBrandt(k)), s)

# ── 2. From the fitted lens: an Ellipse about the principal point, in pixels
q = image_circle(cam)
println("from the lens:      centre ", centroid(q), ", semi-axes ", (q.a, q.b), " px")

# ── 3. From the datasheet, through the pitch: the same primitive, the same circle
qd = image_circle(cam, 6.0mm)
println("from the datasheet: centre ", centroid(qd), ", semi-axes ", (qd.a, qd.b), " px")
println("the routes agree to ", abs(q.a - qd.a), " px")

# ── 4. Is the whole sensor lit? The circular fisheye: no.
println("fisheye lights the whole sensor: ", Rect(extent(cam)) ∈ q)
println("fisheye fov: ", rad2deg(fov(cam)), "° horizontal, ", rad2deg(fov(cam; dimension = :diagonal)), "° diagonal")

# ── 5. A rectilinear lens on the same sensor has no image circle: nothing to restrict
rcam = BoundedCamera(CameraModel(; focal = 2000.0, pp = (1224.0, 1024.0), lens = Pinhole()), s)
println("rectilinear image circle: ", image_circle(rcam))
println("rectilinear fov: ", rad2deg(fov(rcam)), "° horizontal, ", rad2deg(fov(rcam; dimension = :diagonal)), "° diagonal")

# ── 6. The consumer: a grid lens keeps only the nodes the lens lights
domain = Rect(extent(cam)) ∩ image_circle(cam)
nodes = [Point2(x, y) for x in range(0.0, 2448.0; length = 25), y in range(0.0, 2048.0; length = 21)]
lit = filter(n -> n ∈ domain, nodes)
println("grid nodes lit: ", length(lit), " of ", length(nodes), " (", length(nodes) - length(lit), " outside)")
println("rectilinear domain is the extent: ", (Rect(extent(rcam)) ∩ image_circle(rcam)) == Rect(extent(rcam)))
