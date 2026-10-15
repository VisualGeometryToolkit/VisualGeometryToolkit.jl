# =============================================================================
# Geometric solver function declarations
# =============================================================================
#
# NOT a submodule entry point, despite the `solvers/solvers.jl` spelling that
# `primitives/primitives.jl` and `cameras/cameras.jl` use for one. Those own
# their directory's includes; this file owns only the forward declarations
# below, and the parent (src/VisualGeometryCore.jl) still includes all eight
# files of this directory itself, in an order that interleaves them with
# `geometry/radial_models.jl` and places `solver_layer.jl` after `manifolds/`
# and `model_validity.jl`. THE DIRECTORY IS FILING, NOT A LAYER: it groups the
# minimal solvers and the solver layer so `src/geometry/` is navigable, and it
# deliberately does not claim they form a loadable unit — they do not, and the
# parent's include block is the single authority on their order.
#
# The files grouped here:
#   solvers.jl                  this file — forward declarations (load-order anchor)
#   p3p_native.jl               native P3P (PoseLib p3p.cc port)
#   essential_5pt_native.jl     native 5-pt essential (PoseLib relpose_5pt.cc port)
#   homography_solvers.jl       homography_4pt, DLT, the normalizations
#   homography_5pt_radial.jl    5-pt one-sided radial homography (HomLib port)
#   fundmat.jl                  F/E matrices, 7-pt, motion()
#   solver_layer.jl             CspondNormalization + the MinimalSolver protocol + registry
#
# Forward declarations shared by the native solver ports (p3p_native.jl,
# essential_5pt_native.jl) and the `PoseLib` extension package (which adds
# `::PoseLibSolver` backend methods for solution-parity testing).  Declaring
# the names here fixes the load order: natives and PoseLib both add methods
# to the SAME function.

"""
    p3p(rays, points_3d) -> FixedModels{4, Extrinsics3}

Perspective-3-Point solver.  The bare form routes to `NativeSolver` (see
`p3p_native.jl`, a port of PoseLib's `p3p.cc`); `p3p(PoseLibSolver(), …)`
dispatches to the compiled PoseLib backend when `PoseLib` is loaded.
"""
function p3p end

"""
    essential_5pt(x1, x2) -> FixedModels{10, EssentialMat{Float64}}

5-point essential-matrix solver (Nistér 2004).  The bare form routes to
`NativeSolver` (see `essential_5pt_native.jl`, a port of PoseLib's
`relpose_5pt.cc`); `essential_5pt(PoseLibSolver(), …)` dispatches to the
compiled PoseLib backend when `PoseLib` is loaded.
"""
function essential_5pt end

"""
    essential_dlt(cs) -> EssentialMat{Float64} or nothing
    essential_dlt(backend, cs)

8-point linear DLT for the essential matrix from N >= 8 correspondences
`cs` (`x1 => x2` bearing pairs, `StaticVector{3}` each side; any
[`sources`](@ref)/[`targets`](@ref) container, a [`Csponds`](@ref)
preferred), the same correspondence currency as [`homography_dlt`](@ref)
and [`fundamental_dlt`](@ref). Only the PoseLib backend exists; load
`using PoseLib` to make this callable (`essential_dlt(PoseLibSolver(), cs)`,
and the bare form routes to it).
"""
function essential_dlt end
