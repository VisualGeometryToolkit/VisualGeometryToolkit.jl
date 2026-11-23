# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The circle and the ellipse of the RANSAC conic tests (scenes_testsetups.jl's ConicScenes, which includes this file, and
# the golden records of ci/golden/fixtures/{core_minimal,est_conics_ransac}.jl): n points on the curve with isotropic
# noise σ, then a fraction `outl` replaced, half by a straight run (a floor edge), half by uniform clutter. Data
# constructors only; include-once safe (functions, and constants of immutable values).
using Random: AbstractRNG, randn, rand, randperm
using StaticArrays: SVector
using VisualGeometryToolkitCore: Point2, Circle, Ellipse, boundary_point

const C₀ = SVector(512.3, 401.7)
const R₀ = 300.0
const E₀ = Ellipse(Point2(512.3, 401.7), 300.0, 180.0, 0.6)   # axes ratio 0.6, rotated

# the points and the indices replaced
function contaminated_boundary(rng::AbstractRNG, curve, n; σ, outl)
    pts = [Point2{Float64}(SVector(boundary_point(curve, t)...) + σ * randn(rng, SVector{2, Float64})) for t in 2π .* rand(rng, n)]
    bad = randperm(rng, n)[1:round(Int, outl * n)]
    for (k, i) in enumerate(bad)
        pts[i] = isodd(k) ? Point2{Float64}(250.0 + 500 * rand(rng), 760.0 + randn(rng)) :
            Point2{Float64}(1024 * rand(rng), 800 * rand(rng))
    end
    return pts, bad
end
circle_scene(rng, n; σ = 0.5, outl = 0.4) = contaminated_boundary(rng, Circle(Point2(C₀), R₀), n; σ, outl)
ellipse_scene(rng, n; σ = 0.5, outl = 0.4) = contaminated_boundary(rng, E₀, n; σ, outl)
