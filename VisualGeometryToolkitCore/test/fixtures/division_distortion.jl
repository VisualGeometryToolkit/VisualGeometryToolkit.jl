# The division model's forward distortion of an undistorted point (homography_5pt_radial_tests.jl, and the golden
# records of ci/golden/fixtures/core_minimal.jl). Data constructor only; include-once safe.
using LinearAlgebra: norm
using StaticArrays: SVector

# the d with d / (1 + k‖d‖²) = u, or `nothing` past the model's fold
function distort_division(u, k)
    r = norm(u)
    (abs(k) < 1.0e-14 || r < 1.0e-14) && return SVector{2, Float64}(u[1], u[2])
    disc = 1 - 4k * r^2
    disc <= 0 && return nothing
    t = (1 - sqrt(disc)) / (2k * r)
    return SVector{2, Float64}(u[1] * t / r, u[2] * t / r)
end
