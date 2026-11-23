# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The folding SplineLens sources of optical_axis_tests.jl's FoldingSources (and of the golden records of
# ci/golden/fixtures/est_camera_fits.jl): a ray field θof(r) about AXIS_PIXEL as a SplineLens camera, its K re-gauged
# to (500, 500), bounded by a 478 px disc about the axis. Data constructors only; include-once safe.
using StaticArrays: SMatrix, SVector
using LinearAlgebra: norm
using VisualGeometryToolkitCore: CameraModel, SplineLens, BoundedCamera, Size2, Ellipse, CameraCalibrationMat, calibration_matrix

const AXIS_PIXEL = SVector(512.0, 494.0)
# a SplineLens source of the ray field θof(r) about AXIS_PIXEL (θ scaled by 1 + ε·Δx/480: an azimuthal misfit, as a real
# lens's decentring), its K re-gauged to the principal point (500, 500), bounded by the disc of 478 px about the axis
function folding_source(θof; ε = 0.0)
    c = AXIS_PIXEL
    ray(u) = (v = SVector(u[1], u[2]) - c; r = norm(v); θ = θof(r) * (1 + ε * v[1] / 480);
              r == 0 ? SVector(0.0, 0.0, 1.0) : SVector(sin(θ) * v[1] / r, sin(θ) * v[2] / r, cos(θ)))
    grid = CameraModel(SplineLens, ray, Size2(1000, 1000); grid_size = (30, 30), init = :quasi_interpolate)
    K0 = calibration_matrix(grid)
    return BoundedCamera(CameraModel(grid, CameraCalibrationMat(SMatrix{3, 3, Float64}(K0[1, 1], 0, 0, 0, K0[2, 2], 0, 500.0, 500.0, 1))),
                         Size2(1000, 1000); lens_coverage = Ellipse(c, 478.0, 478.0, 0.0))
end
# The ray fields: a pinhole (focal f) out to r0, past it turning away by `turn` more at 480 px (a fold); and a fisheye's
# slope collapsing past r0.
theta_fold(r; turn, f = 600.0, r0 = 350.0) = r <= r0 ? atan(r / f) : atan(r0 / f) + (r - r0) / f + turn * ((r - r0) / (480 - r0))^3
theta_collapse(r; f = 300.0, r0 = 420.0) = r <= r0 ? r / f : r / f + 0.2 * ((r - r0) / 60)^3
