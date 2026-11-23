# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# Points on an ellipse's boundary (direct_ellipse_tests.jl, VisualGeometryToolkitEstimation's ellipse_fitting_tests.jl,
# and the golden records of ci/golden/fixtures/est_conics_ransac.jl): n points evenly spaced in angle over `arc`, with
# isotropic noise σ drawn from `rng`. Data constructors only; include-once safe (functions only).
using StaticArrays: SVector
using VisualGeometryToolkitCore: boundary_point

ellipse_boundary_points(rng, e, n; σ = 0.0, arc = 2π) =
    [boundary_point(e, φ) + σ * randn(rng, SVector{2}) for φ in range(0, arc, n + 1)[1:(end - 1)]]
