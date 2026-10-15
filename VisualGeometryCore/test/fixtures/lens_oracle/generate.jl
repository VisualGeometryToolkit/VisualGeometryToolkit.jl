# Generates `lens_oracle.toml`: the behaviour of the lens zoo as it was BEFORE the
# lens-family redesign (VisualGeometryCore 1.0.0, 1c4f93f), recorded once so the
# redesigned families are gated against it (`test/lens_oracle_tests.jl`). Runs
# only against that version — it names the retired types. Kept for provenance.
#
#   julia --project=<VGC 1.0.0 checkout> generate.jl
using VisualGeometryCore, StaticArrays, Random, TOML
const VGC = VisualGeometryCore
# the 1.0.0 lens types, by name (they are retired names now: spelled indirectly)
_old(parts...) = getfield(VGC, Symbol(parts...))
const OLD = (BC = _old("Brown", "Conrady"), RR = _old("Rational", "Radial"), EF = _old("Equidistant", "Fisheye"),
    SF = _old("Stereographic", "Fisheye"), QF = _old("Equisolid", "Fisheye"), OF = _old("Orthographic", "Fisheye"),
    TPF = _old("ThinPrism", "Fisheye"), AB = _old("Additive", "Bundle"))

# Each case: (name, old lens, the lens as a v3 record).
bc(; kw...) = OLD.BC(; kw...)
const CASES = Any[
    ("pinhole", Pinhole(), Dict("perspective" => Float64[])),
    ("equidistant", OLD.EF(), Dict("equidistant" => Float64[])),
    ("stereographic", OLD.SF(), Dict("stereographic" => Float64[])),
    ("equisolid", OLD.QF(), Dict("equisolid" => Float64[])),
    ("orthographic", OLD.OF(), Dict("orthographic" => Float64[])),
    ("division1", Division(-0.31), Dict("lift" => Dict("radial" => [Float64[], [-0.31]]))),
    ("division1_pos", Division(0.12), Dict("lift" => Dict("radial" => [Float64[], [0.12]]))),
    ("division2", Division(-0.31, 0.024), Dict("lift" => Dict("radial" => [Float64[], [-0.31, 0.024]]))),
    ("division3", Division(-0.2, 0.03, -0.004), Dict("lift" => Dict("radial" => [Float64[], [-0.2, 0.03, -0.004]]))),
    ("rational12", OLD.RR((0.05,), (-0.25, 0.02)), Dict("lift" => Dict("radial" => [[0.05], [-0.25, 0.02]]))),
    ("rational21", OLD.RR((0.03, -0.002), (-0.2,)), Dict("lift" => Dict("radial" => [[0.03, -0.002], [-0.2]]))),
    ("scaramuzza", Scaramuzza(; a2 = -0.17, a3 = -0.006, a4 = 0.035), Dict("lift" => Dict("radial_full" => [Float64[], [-0.17, -0.006, 0.035]]))),
    ("opencv_k1", bc(; k1 = -0.12), Dict("perspective" => Float64[], "radial" => [[-0.12], Float64[]])),
    ("opencv4", bc(; k1 = -0.28, k2 = 0.07, p1 = 1.0e-3, p2 = -5.0e-4),
        Dict("perspective" => Float64[], "add" => [Dict("radial" => [[-0.28, 0.07], Float64[]]), Dict("tangential" => [1.0e-3, -5.0e-4])])),
    ("opencv5", bc(; k1 = -0.2817, k2 = 0.0712, k3 = -0.0061, p1 = 0.00091, p2 = -0.00042),
        Dict("perspective" => Float64[], "add" => [Dict("radial" => [[-0.2817, 0.0712, -0.0061], Float64[]]), Dict("tangential" => [0.00091, -0.00042])])),
    ("opencv8", bc(; k1 = -0.12, k2 = 0.03, k3 = 0.004, k4 = 0.02, k5 = -0.001, k6 = 0.0003, p1 = 1.0e-3, p2 = -5.0e-4),
        Dict("perspective" => Float64[], "add" => [Dict("radial" => [[-0.12, 0.03, 0.004], [0.02, -0.001, 0.0003]]), Dict("tangential" => [1.0e-3, -5.0e-4])])),
    ("opencv12_tamron", OLD.BC([-2.7067990842834013, 4.003544352914542, 0.0028263106588956688, 0.0017230351557208537,
            1.5827483616019924, -2.5409744152370775, 3.5017575444899127, 2.4059525010963547,
            -3.137923848171712e-5, 7.363153842759605e-5, -0.00322446080396022, 0.00027475714123090727]),
        Dict("perspective" => Float64[], "add" => [
            Dict("radial" => [[-2.7067990842834013, 4.003544352914542, 1.5827483616019924], [-2.5409744152370775, 3.5017575444899127, 2.4059525010963547]]),
            Dict("tangential" => [0.0028263106588956688, 0.0017230351557208537]),
            Dict("thin_prism" => [[-3.137923848171712e-5, 7.363153842759605e-5], [-0.00322446080396022, 0.00027475714123090727]])])),
    ("tangential_only", OLD.AB(Tangential(1.0e-3, -5.0e-4)), Dict("perspective" => Float64[], "tangential" => [1.0e-3, -5.0e-4])),
    ("kannala_brandt", KannalaBrandt(; k1 = -0.043, k2 = 0.011, k3 = -0.0083, k4 = 0.0015),
        Dict("equidistant" => Float64[], "radial" => [[-0.043, 0.011, -0.0083, 0.0015], Float64[]])),
    ("kannala_brandt_fold", KannalaBrandt(-0.3, 0.0, 0.0, 0.0), Dict("equidistant" => Float64[], "radial" => [[-0.3, 0.0, 0.0, 0.0], Float64[]])),
    ("thin_prism_fisheye", OLD.TPF(-0.02, 0.004, -0.0009, 0.0001, 0.0006, -0.0004, 0.0002, -0.0001),
        Dict("equidistant" => Float64[], "add" => [Dict("radial" => [[-0.02, 0.004, -0.0009, 0.0001], Float64[]]),
            Dict("tangential" => [0.0006, -0.0004]), Dict("thin_prism" => [[0.0002], [-0.0001]])])),
    ("double_sphere", DoubleSphere(; xi = -0.2, alpha = 0.59), Dict("double_sphere" => [-0.2, 0.59])),
    ("double_sphere_low", DoubleSphere(; xi = 0.3, alpha = 0.4), Dict("double_sphere" => [0.3, 0.4])),
]

flat(vs) = Float64[x for v in vs for x in v]

function probe(lens, rng)
    θmax = Float64(view_angle_limit(lens))
    θs = vcat(rand(rng, 64) .* 0.999θmax, θmax .* [0.99, 0.995, 0.999, 0.9995, 0.9999], θmax .* [1.001, 1.01, 1.05])
    θs = min.(θs, π - 1.0e-6)
    φs = 2π .* rand(rng, length(θs))
    sc = 0.5 .+ 2 .* rand(rng, length(θs))
    rays = [SVector(s * sin(θ) * cos(φ), s * sin(θ) * sin(φ), s * cos(θ)) for (θ, φ, s) in zip(θs, φs, sc)]
    proj = [project(lens, X) for X in rays]
    fin = [p for p in proj if all(isfinite, p)]
    ρmax = isempty(fin) ? 1.0 : maximum(p -> hypot(p...), fin)
    extra = [(r = ρmax * 1.1 * rand(rng); a = 2π * rand(rng); SVector(r * cos(a), r * sin(a))) for _ in 1:16]
    pts = vcat(fin, extra)
    back = [backproject(lens, x) for x in pts]
    persp = [SVector(X[1] / X[3], X[2] / X[3]) for X in rays if X[3] > 0.05]
    dist = [distort_normalized(lens, x) for x in persp]
    und = [undistort_normalized(lens, x) for x in pts]
    J = [transform_deriv(lens, X) for X in rays[1:8]]
    Jp = [transform_deriv_params(lens, X) for X in rays[1:8]]
    l32 = VGC._precision(Float32, lens)
    p32 = [project(l32, SVector{3, Float32}(X)) for X in rays[1:16]]
    b32 = [backproject(l32, SVector{2, Float32}(x)) for x in pts[1:min(16, end)]]
    return Dict(
        "monotone_radius" => Float64(VGC.monotone_radius(lens)), "view_angle_limit" => θmax,
        "rays" => flat(rays), "project" => flat(proj),
        "points" => flat(pts), "backproject" => flat(back),
        "persp" => flat(persp), "distort" => flat(dist), "undistort" => flat(und),
        "jac" => flat(vec.(J)), "jac_params" => flat(vec.(Jp)), "nparams" => length(lens_parameters(lens)),
        "project32" => flat([Float64.(p) for p in p32]), "backproject32" => flat([Float64.(b) for b in b32]),
    )
end

out = Dict{String, Any}("generator" => "VisualGeometryCore 1.0.0 (1c4f93f), test/fixtures/lens_oracle/generate.jl")
rng = Xoshiro(20261008)
for (name, lens, rec) in CASES
    d = probe(lens, rng)
    d["lens"] = rec
    out[name] = d
end
open(joinpath(@__DIR__, "lens_oracle.toml"), "w") do io
    TOML.print(io, out; sorted = true)
end
