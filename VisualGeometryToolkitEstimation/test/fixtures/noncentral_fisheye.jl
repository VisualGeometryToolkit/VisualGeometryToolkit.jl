# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The synthetic non-central fisheye of camera_regression_testsetups.jl's FitCameraNoncentral and of the golden records of
# ci/golden/fixtures/est_camera_fits.jl: the central fisheye's directions on a 53 × 47 lattice and a smooth, known
# origin field. Data constructors only; include-once safe (functions only).
using StaticArrays: SA, SVector
using VisualGeometryToolkitCore: CameraModel, SplineLens, NoncentralSplineModel, ChartSpline, BoundedCamera, Size2, KannalaBrandt, px

# the central fisheye (a KB lens, f = 300 px) as a SplineLens camera on the 53 × 47 lattice
noncentral_central() = CameraModel(SplineLens, CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02)), Size2(1000, 1000);
                                   grid_size = (53, 47))
# the non-central model of `central` (noncentral_central()) with every line translated by `t` (metres): sub-millimetre
# lateral waves and a 3 mm axial bulge growing with the radius, bounded by its 1000 × 1000 image
function noncentral_fisheye(central, t = SVector(0.0, 0.0, 0.0))
    d = central.lens.backward
    gh, gw = size(d.controls)
    points = map(CartesianIndices((gh, gw))) do ij
        i, j = Tuple(ij)
        x, y = (j - 2) * 1000 / (gw - 3), (i - 2) * 1000 / (gh - 3)
        ρ = hypot(x - 500, y - 500) / 500
        SVector{3, Float64}(2.0e-4 * sin(x / 300) + t[1], 1.5e-4 * cos(y / 400) + t[2], 3.0e-3 * ρ^2 + t[3])
    end
    return BoundedCamera(NoncentralSplineModel(central.intrinsics, d, ChartSpline(d.area, points)), Size2(1000, 1000))
end
