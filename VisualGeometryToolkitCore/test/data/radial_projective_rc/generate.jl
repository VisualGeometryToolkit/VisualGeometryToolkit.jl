# =============================================================================
# The projective radial chart's equivalence fixture — RadiometricCamera's
# `ProjectiveRadialVignetting` (src/camera/vignetting.jl), the implementation
# VisualGeometryToolkitCore's `RadialVignetting(…; chart = :projective)` ports.
# =============================================================================
#
# Written ONCE from RadiometricCamera's file itself: src/camera/vignetting.jl of
# rc-port-09 44096174 (branch plan3-adopt-0.9), byte-identical to
# RadiometricCamera 254a3a3b (its last change is d611879d; the sha256 is
# recorded), included verbatim into a module of its own (`RCVignetting`) over
# VisualGeometryToolkitCore at hoists-09 f0342189, whose vignetting code is
# Core's before the port (unchanged since b4aa170b). The file is included, not
# the package loaded: RadiometricCamera 44096174 did not precompile in its own
# environment on 2026-11-11 (its Manifest resolved Core 0.8.0 from the
# registry, which lacks the SplineLens other files of the package import),
# and the file reads nothing of the package but Core. (Over that Core 0.8.0
# the same script gave the same numbers.) Run in VisualGeometryToolkitCore's
# test environment (Core, StaticArrays, CoordinateTransformations,
# ForwardDiff, TOML), from the monorepo's root:
#
#   RC_VIGNETTING_FILE=<RadiometricCamera>/src/camera/vignetting.jl \
#       julia --project=VisualGeometryToolkitCore/test <this file> [out.toml]
#
# Kept as the record of how the fixture was made. Never rerun it to make the
# fixture match: a value that moves is a change to stop at and explain.
# The equivalence item: VisualGeometryToolkitCore/test/vignetting_tests.jl,
# "vignetting: the projective chart is RadiometricCamera's".
#
# Per profile (count, degree, centre, radius, control points, all RC's
# constructor's inputs): the knots, and per sensor point the normalized
# radius, the value, the coefficient row (∂V/∂P), the active coefficients,
# ∂V/∂u (ForwardDiff, off the centre), ∂V/∂centre (RC's `center_gradient`),
# the Float32 value, and the value about a moved centre (`with_center`). Also
# `projective_knots(n, p)` for a few (n, p).

const RC_FILE = ENV["RC_VIGNETTING_FILE"]
module RCVignetting   # the bindings RC's module gives the file
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore
    using StaticArrays: SVector, SMatrix, StaticVector
    using LinearAlgebra: I, dot
    using CoordinateTransformations: CoordinateTransformations
    using ForwardDiff: ForwardDiff
end
Base.include(RCVignetting, RC_FILE)
using .RCVignetting: ProjectiveRadialVignetting, projective_knots, with_center, center_gradient
import VisualGeometryToolkitCore as VgtkCore
using StaticArrays: SVector
using ForwardDiff: ForwardDiff
using TOML: TOML
using SHA: SHA

const PROFILES = [
    (n = 17, degree = 3, center = [2126.0, 1554.0], radius = 2560.5, P = [1.0 - 0.04 * (i - 1)^1.2 for i in 1:17]),
    (n = 8, degree = 2, center = [320.3, 241.7], radius = 401.25, P = [0.5 + 0.0371 * i + 0.0193 * i * i for i in 1:8]),
    (n = 6, degree = 3, center = [50.0, 40.0], radius = 60.0, P = [1.0, 1.0, 0.93, 0.81, 0.66, 0.42]),
    (n = 5, degree = 1, center = [0.0, 0.0], radius = 1.0, P = [2.0, 1.5, 1.25, 1.0, 0.5]),
]
points(c, r) = [c, c .+ [0.3, -0.2], c .+ [0.0, r], c .+ [r / 3, r / 4], c .+ [-2r, 0.0], c .+ [1.0e7, 0.0],
                [10.0, 20.0], [4200.0, 10.0], c .+ [r / 7, -r / 11]]
const MOVED = [7.25, -3.5]

function profile_record(pr)
    v = VgtkCore.with_control_points(ProjectiveRadialVignetting(pr.n, SVector{2}(pr.center), pr.radius; degree = pr.degree), pr.P)
    moved = with_center(v, SVector{2}(pr.center .+ MOVED))
    v32 = VgtkCore.convert_eltype(Float32, v)
    rows = map(points(pr.center, pr.radius)) do p
        u = SVector{2}(p)
        idx, w = VgtkCore.active_coefficients(v, u)
        d = hypot((p .- pr.center)...)
        du = d > 0 ? collect(ForwardDiff.gradient(x -> v(SVector{2}(x)), p)) : Float64[]
        Dict("u" => p, "rho" => VgtkCore.radial_coordinate(v, u), "value" => v(u),
             "row" => collect(VgtkCore.coefficient_jacobian(v, u)), "active_index" => collect(idx), "active_weight" => collect(w),
             "grad_u" => du, "grad_center" => collect(center_gradient(v, u)),
             "value_f32" => Float64(v32(SVector{2, Float32}(p))), "value_moved" => moved(u))
    end
    return Dict("n" => pr.n, "degree" => pr.degree, "center" => pr.center, "radius" => pr.radius, "P" => pr.P,
                "knots" => collect(only(VgtkCore.knot_vectors(v))), "points" => rows)
end

out = isempty(ARGS) ? joinpath(@__DIR__, "rc_values.toml") : ARGS[1]
record = Dict(
    "provenance" => Dict("source" => "RadiometricCamera src/camera/vignetting.jl", "rc_commit" => "44096174 (file byte-identical to 254a3a3b)", "rc_file_sha256" => bytes2hex(open(SHA.sha256, RC_FILE)),
                         "core" => "hoists-09 f0342189 (Core's vignetting before the port)", "generator" => "VisualGeometryToolkitCore/test/data/radial_projective_rc/generate.jl"),
    "projective_knots" => [Dict("n" => n, "p" => p, "knots" => collect(projective_knots(n, p))) for (n, p) in ((17, 3), (8, 2), (6, 3), (5, 1), (12, 4))],
    "profiles" => [profile_record(pr) for pr in PROFILES])
open(io -> TOML.print(io, record; sorted = true), out, "w")
