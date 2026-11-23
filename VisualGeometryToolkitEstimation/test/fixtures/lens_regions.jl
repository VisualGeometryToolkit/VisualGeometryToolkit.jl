# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The fit regions of fit_camera_testsetups.jl's FitCameraLensCases (and of the golden records of
# ci/golden/fixtures/est_camera_fits.jl): a disc about a camera's principal point, of a radius in pixels or of the radius
# a view angle images at. Data constructors only; include-once safe (functions only).
using StaticArrays: SVector
using VisualGeometryToolkitCore: Ellipse, projected_radius, radial_part, calibration_matrix, lens

disc(m, R) = Ellipse(SVector(calibration_matrix(m)[1, 3], calibration_matrix(m)[2, 3]), R, R, 0.0)
disc_angle(m, θ) = disc(m, Float64(projected_radius(radial_part(lens(m)), θ)) * calibration_matrix(m)[1, 1])
