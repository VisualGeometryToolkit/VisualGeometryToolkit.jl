# The valid domain of a lens model, and the image circle of a lens — two regions
# of the sensor. `valid_domain(camera)` is where the lens FORMULA holds (its
# view-angle limit's image); `image_circle(camera, diameter)` is where light
# falls (a datasheet through the pitch, or a measured circle). A local lens
# model (a grid of nodes) lives on `imaged_region(camera, circle)`.
#
#   julia --project=. examples/image_circle.jl

using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: Pinhole, KannalaBrandt, WholePlane

# ── 1. A circular fisheye bounded by its physical sensor (2448 × 2048 @ 3.45 μm).
# Kannala–Brandt r = θ(1 + k θ²) folds at θ = 1/√(−3k): k below puts the fold at
# 90° (a 180° lens), where r = (2/3)·θ. The datasheet states a 6 mm circle.
s = sensor("Sony", "IMX250")
k = -4 / (3π^2)
f = ustrip(px, 6.0mm / (2 * s.pitch.width)) / (π / 3)   # f·r(90°) = the circle's radius in px
cam = BoundedCamera(CameraModel(; focal = f, pp = (1224.0, 1024.0), lens = KannalaBrandt(k)), s)

# ── 2. The lens model's valid domain: an Ellipse about the principal point, in pixels
v = valid_domain(cam)
println("valid domain:       centre ", centroid(v), ", semi-axes ", (v.a, v.b), " px")

# ── 3. The image circle, from the datasheet through the pitch: where light falls
q = image_circle(cam, 6.0mm)
println("image circle:       centre ", centroid(q), ", semi-axes ", (q.a, q.b), " px")
println("this lens was built so they agree, to ", abs(v.a - q.a), " px (a fitted lens need not)")

# ── 4. Is the whole sensor lit? The circular fisheye: no.
println("fisheye lights the whole sensor: ", Rect(extent(cam)) ∈ q)
println("fisheye fov: ", rad2deg(fov(cam)), "° horizontal, ", rad2deg(fov(cam; dimension = :diagonal)), "° diagonal")

# ── 5. A rectilinear lens has no limit of its own: its valid domain is the whole plane
rcam = BoundedCamera(CameraModel(; focal = 2000.0, pp = (1224.0, 1024.0), lens = Pinhole()), s)
println("rectilinear valid domain: ", valid_domain(rcam))
println("rectilinear fov: ", rad2deg(fov(rcam)), "° horizontal, ", rad2deg(fov(rcam; dimension = :diagonal)), "° diagonal")

# ── 6. The consumer: a grid lens keeps only the nodes the lens lights
domain = imaged_region(cam, q)
nodes = [Point2(x, y) for x in range(0.0, 2448.0; length = 25), y in range(0.0, 2048.0; length = 21)]
lit = filter(n -> n ∈ domain, nodes)
println("grid nodes lit: ", length(lit), " of ", length(nodes), " (", length(nodes) - length(lit), " outside)")
println("no image circle known: the extent ", imaged_region(rcam, WholePlane()) == Rect(extent(rcam)))
