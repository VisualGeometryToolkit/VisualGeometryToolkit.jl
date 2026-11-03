# =============================================================================
# The chain-equivalence goldens — written ONCE, on the code before
# `CalibrationChain` and the lens `Chain` were replaced by
# CoordinateTransformations' `∘` (VisualGeometryToolkitCore 0.4.1, main 5b31549).
# =============================================================================
#
# Kept as the record of how the goldens were made. Never rerun it to make a
# golden match: a golden that moves is a change to stop at and explain. It
# spells the compositions with the two retired types, so it runs only on that
# code. From the repository root:
#
#   JULIA_LOAD_PATH="@:$PWD/VisualGeometryToolkitCore/test:@stdlib" \
#       julia --project=VisualGeometryToolkitIO VisualGeometryToolkitCore/test/data/chain_equivalence/generate.jl [outdir]
#
# (Core's test environment on the load path for StableRNGs.) The values are
# computed by core_values.jl and io_values.jl through the adapter of the
# goldens' code (`GOLDEN_CODE`), the same files the test items read them back
# with. It writes into `outdir` (default: next to itself):
#
#   corpus_params.txt   the corpus's parameters (StableRNG(20261103)): the
#                       inputs every reader rebuilds the corpus from
#   core_values.txt     Core's outputs: per calibration — parameter vector,
#                       names, bounds, gauge names and directions, length
#                       mask, planarity, the map on 200 plain and 50 physical
#                       points, the parameter and point Jacobians at 10, the
#                       regularizer rows, a with_parameters rebuild, the unit
#                       moves, the target manifold's size, action and
#                       retraction, the surface chart's affine; per lens —
#                       parameter vector, names, bounds, record, project,
#                       backproject, distort and undistort on 200 points, the
#                       input and parameter Jacobians at 10, the monotone
#                       radius and view-angle limit, Float32 projection, a
#                       with_parameters rebuild, the lens manifold's
#                       retraction, a camera's project and backproject, its
#                       rescaled lens and its valid domain
#   io_values.txt       VisualGeometryToolkitIO's: OpenCV coefficient vectors,
#                       COLMAP models and parameters, every rig's content_id
#   records/            the files VisualGeometryToolkitIO writes from them:
#                       vgio/target/v2 (plain and Uncertain calibration),
#                       vgio/rig/v5, vgio/cameras/v3, the OpenCV calibration
#                       export and a COLMAP text model

include(joinpath(@__DIR__, "corpus.jl"))
using .ChainEquivalenceCorpus
include(joinpath(@__DIR__, "core_values.jl"))
include(joinpath(@__DIR__, "io_values.jl"))
using StableRNGs: StableRNG
using VisualGeometryToolkitCore: CalibrationChain, Chain, Affine, calibration_jacobian, calibration_tag

# The adapter of the code the goldens were written on (corpus.jl): chains in
# application order, the `Affine` kind, `calibration_jacobian`.
const GOLDEN_CODE = (
    affine = Affine,
    compose_calibrations = members -> CalibrationChain(members),
    compose_distortions = stages -> Chain(stages...),
    jacobian_theta = calibration_jacobian,
    jacobian_x = (c, x) -> VgtkCore.Cameras._point_jacobian(c, x),
    tag = c -> calibration_tag(typeof(c)),
    legacy_lens_record = identity,
    ct_order = false,
    retired = Set{String}(),
)

function draw_params()
    rng = StableRNG(20261103)
    r(n...) = randn(rng, n...)
    P = Dict{String, Any}()
    P["param/affine/A"] = [1.0 0.0; 0.0 1.0] .+ 0.01 .* r(2, 2)
    P["param/affine/b"] = 0.002 .* r(2)
    P["param/thin_plate_spline/fx"] = 2.0e-4 .* r(9)
    P["param/thin_plate_spline/fy"] = 2.0e-4 .* r(9)
    P["param/height_field/h"] = 5.0e-4 .* r(9)
    # OpenCV's (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, τx, τy); a
    # shorter model is a prefix
    cv = [-0.2 + 0.05r(), 0.05 + 0.02r(), 1.0e-3r(), 1.0e-3r(), -0.01 + 0.005r(), 0.02r(), 0.01r(), 0.005r(),
          1.0e-3r(), 1.0e-3r(), 1.0e-3r(), 1.0e-3r(), 0.01r(), 0.01r()]
    for n in (4, 5, 8, 12, 14)
        P["param/opencv$n"] = cv[1:n]
    end
    # COLMAP's RAD_TAN_THIN_PRISM_FISHEYE extras (k0…k5, p0, p1, s0…s3)
    P["param/fisheye624"] = [[-0.02, 0.005, -0.001, 1.0e-4, -1.0e-5, 1.0e-6] .* (1 .+ 0.1 .* r(6)); 1.0e-4 .* r(6)]
    # the perturbation of a with_parameters rebuild, per case
    for (name, c) in target_cases(P, GOLDEN_CODE)
        P["param/target/$name/delta"] = 1.0e-5 .* r(length(parameter_vector(c)))
    end
    for (name, l) in lens_cases(P, GOLDEN_CODE)
        P["param/lens/$name/delta"] = 1.0e-4 .* r(length(parameter_vector(l)))
    end
    return P
end

write_lines(path, pairs) = open(io -> foreach(((k, v),) -> println(io, golden_line(k, v)), pairs), path, "w")

function main(dir::AbstractString = @__DIR__)
    P = draw_params()
    write_lines(joinpath(dir, "corpus_params.txt"), [k => P[k] for k in sort(collect(keys(P)))])
    # the readers rebuild the corpus from the stored parameters: so does this
    P = read_goldens(joinpath(dir, "corpus_params.txt"))
    write_lines(joinpath(dir, "core_values.txt"), [target_values(P, GOLDEN_CODE); lens_values(P, GOLDEN_CODE)])
    write_lines(joinpath(dir, "io_values.txt"), io_values(P, GOLDEN_CODE, joinpath(dir, "records")))
    return nothing
end

main(isempty(ARGS) ? (@__DIR__) : ARGS[1])
