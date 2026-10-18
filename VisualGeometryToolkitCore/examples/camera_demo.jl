# Camera models, poses, and projection — the package's headline surface.
#
#   julia --project=. examples/camera_demo.jl

using VisualGeometryToolkitCore
using StaticArrays: SVector, SMatrix
using Rotations: RotXYZ
using LinearAlgebra: norm

# ── A pinhole model from focal length + principal point (pixels) ────────
pinhole = CameraModel(; focal = 800.0, pp = (320.0, 240.0))

# ── The same intrinsics behind OpenCV's radial-tangential lens ────────
distorted = CameraModel(;
    focal = 800.0, pp = (320.0, 240.0),
    lens = Lens(Perspective(), Additive(Radial((-0.28, 0.07, -0.01)), Tangential(1.0e-3, -5.0e-4))),
)

# ── Pose: world → camera as an EuclideanMap (R, t) ──────────────────────
R = SMatrix{3, 3, Float64}(RotXYZ(0.05, -0.10, 0.02))
t = SVector(0.10, -0.05, 2.0)
cam = Camera(pinhole, EuclideanMap(R, t))

X = SVector(0.2, -0.1, 0.5)          # a world point in front of the camera
u = project(cam, X)                   # pixel coordinates (COLMAP convention)
ray = backproject(cam, u)             # the ray through that pixel, world frame
println("project:      ", u)
println("backproject:  ", ray, "  (unit: ", round(norm(ray); digits = 12), ")")

# ── Lens math on normalized coordinates ─────────────────────────────────
x = SVector(0.30, -0.20)
xd = distort_normalized(lens(distorted), x)
xu = undistort_normalized(lens(distorted), xd)
println("distort:      ", x, " → ", xd)
println("undistort:    ", xd, " → ", xu, "  (round-trip error ", norm(xu - x), ")")

# ── A batch: the same per-point map, broadcast (a GPU array runs one kernel) ──
pts = [SVector(u, v) for (u, v) in zip(range(-0.4, 0.4; length = 8), range(-0.3, 0.3; length = 8))]
out = distort_normalized!(similar(pts), lens(distorted), pts)
println("batched distort of ", length(pts), " points: first = ", out[1])
