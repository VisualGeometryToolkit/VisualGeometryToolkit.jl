# =============================================================================
# The stack's package boundaries — who may depend on whom, which names cross
# =============================================================================
#
#   * `stack_dependencies`          — the allow-list, configs/stack_dependencies.toml.
#   * `stack_dependency_violations` — a Project.toml checked against its row.
#   * `private_name_violations`     — `_`-prefixed names reached across a package boundary.
#   * `CONTRACT_GENERICS`           — the generic functions of VisualGeometryToolkitCore
#                                     another package may add methods to.
#   * `contract_generic_violations` — a package's methods on other stack
#                                     packages' functions that are not contracts.
#
# Pure functions over files and loaded modules: the quality gates
# (ext/VisualGeometryToolkitCoreQualityExt) turn them into tests, and a paper
# project's own suite can call them directly.
# =============================================================================

const _STACK_DEPENDENCIES_FILE = joinpath(dirname(dirname(@__DIR__)), "configs", "stack_dependencies.toml")

"""
    stack_dependencies(; extra = nothing) -> Dict{String, Vector{String}}

The stack's dependency allow-list (`configs/stack_dependencies.toml` in
VisualGeometryToolkitCore): each project's name → the stack packages its Project.toml
may name. A *stack package* is any key of this table.

The table lists the published packages and paper projects only. A project kept
elsewhere supplies its own row as `extra`: a path to a TOML file of rows in the
same format, or a dictionary name → packages. Extra rows add projects; one that
repeats a row of the table is an `ArgumentError`.
"""
function stack_dependencies(; extra = nothing)
    table = Dict{String, Vector{String}}(k => String.(v) for (k, v) in TOML.parsefile(_STACK_DEPENDENCIES_FILE))
    extra === nothing && return table
    rows = extra isa AbstractString ? TOML.parsefile(extra) : extra
    for (k, v) in rows
        haskey(table, k) &&
            throw(ArgumentError("extra stack dependency row \"$k\" repeats a row of $_STACK_DEPENDENCIES_FILE"))
        table[String(k)] = String.(v)
    end
    return table
end

"""
    stack_dependency_violations(project_toml_path, name; extra = nothing) -> Vector{String}

The stack packages the Project.toml at `project_toml_path` names — in `[deps]`,
`[weakdeps]`, `[extras]` or the `test` target — that are not in `name`'s row
of [`stack_dependencies`](@ref) (with the `extra` rows, as there), one
`"section: Package"` per finding. A project absent from the table is an
`ArgumentError`.
"""
function stack_dependency_violations(project_toml_path::AbstractString, name::AbstractString; extra = nothing)
    table = stack_dependencies(; extra)
    haskey(table, name) ||
        throw(ArgumentError("project \"$name\" is not in the stack dependency allow-list ($_STACK_DEPENDENCIES_FILE, or its extra rows)"))
    allowed = Set(table[name])
    proj = TOML.parsefile(project_toml_path)
    out = String[]
    for section in ("deps", "weakdeps", "extras")
        for pkg in sort!(collect(keys(get(proj, section, Dict{String, Any}()))))
            haskey(table, pkg) && pkg ∉ allowed && push!(out, "$section: $pkg")
        end
    end
    for pkg in get(get(proj, "targets", Dict{String, Any}()), "test", String[])
        haskey(table, pkg) && pkg ∉ allowed && push!(out, "targets.test: $pkg")
    end
    return out
end

"""
    private_name_violations(dir, name) -> Vector{String}

A heuristic source scan of the `.jl` files under `dir` (a package's `src/` and
`ext/`) for names that begin with `_` and are reached from another stack
package (a key of [`stack_dependencies`](@ref) other than `name`): imported
(`using Pkg: _x`, `import Pkg.Sub: _x`) or accessed qualified (`Pkg._x`,
`Pkg.Sub._x`, also through a `const P = Pkg` or `import Pkg as P` alias).
Comments and string literals are not scanned. Each finding is
`"file:line: Pkg._x"`, relative to `dir`.
"""
function private_name_violations(dir::AbstractString, name::AbstractString)
    pkgs = setdiff(keys(stack_dependencies()), (name,))
    out = String[]
    isdir(dir) || return out
    for (root, _, files) in walkdir(dir), f in sort(files)
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        src = _blank_spans(read(path, String); strings = true)
        # aliases of a stack package in this file
        alias = Dict{String, String}(p => p for p in pkgs)
        for m in eachmatch(r"(?:const\s+(\w+)\s*=\s*(\w+)\s*$|import\s+(\w+)\s+as\s+(\w+))"m, src)
            a, b, c, d = m[1], m[2], m[3], m[4]
            a !== nothing && b !== nothing && b in pkgs && (alias[a] = b)
            c !== nothing && d !== nothing && c in pkgs && (alias[d] = c)
        end
        lines = split(src, '\n')
        # `using/import Pkg[.Sub]: a, _b, …` — the list may span lines
        for m in eachmatch(r"\b(?:using|import)[ \t]+(\w+)((?:\.\w+)*)[ \t]*:[ \t]*((?:[^\n]*,[ \t]*\n)*[^\n]*)", src)
            pkg, sub, list = something(m[1]), something(m[2], ""), something(m[3])
            haskey(alias, pkg) || continue
            line = count(==('\n'), src[1:prevind(src, m.offset)]) + 1
            for nm in split(list, ',')
                n = strip(first(split(strip(nm), r"\s+as\s+")))
                startswith(n, '_') && push!(out, "$(relpath(path, dir)):$line: $(alias[pkg])$sub.$n")
            end
        end
        for (k, l) in enumerate(lines), m in eachmatch(r"(?<![\w.])(\w+)((?:\.[A-Za-z]\w*)*)\.(_\w*!?)", l)
            pkg = something(m[1])
            haskey(alias, pkg) && push!(out, "$(relpath(path, dir)):$k: $(alias[pkg])$(something(m[2], "")).$(something(m[3]))")
        end
    end
    return out
end

# The generic functions of VisualGeometryToolkitCore another package may add methods
# to (see the docstring of CONTRACT_GENERICS, below the list).
const CONTRACT_GENERICS = (
    # the block-coordinate-descent loop's hooks and readers (robust/block_coordinate_descent.jl)
    update_model!, update_residuals!, update_nuisance!, update_weights!, post_iteration!,
    stopping_criterion, stop_reason, block_coordinate_result, current_cost, current_s2,
    # the chart LM's row protocol (robust/lm_core.jl)
    residual_rows!, residual_jacobian_rows!,
    # the RANSAC / LO interface (robust/estimation.jl, robust/losses.jl, robust/width_axes.jl)
    solve, model_type, test_model, test_sample, test_consensus, constraint_with_jacobians,
    inlier_weight, robust_weight, is_inlier,
    loss_family, loss_shape, decision_width, to_native, scale_interface,
    # the loss's half-quadratic lift facet (robust/losses.jl), answered by a tabulated lift too
    lift_wmax, lift_conjugate, lift_zblock,
    lo_label, lo_marker, optimization_path,
    # the constraint-problem protocol (robust/problem.jl)
    data_size, codimension, minimal_sample_size, residuals!, residual_context, whitened_residual,
    solver_covariance, provides_solver_covariance, refine!, refine_workspace,
    # the scoring contract (robust/scoring_contract.jl, robust/baseline_scorers.jl)
    scoring_workspace, score!, inlier_count, consensus_order, inlier_threshold, has_inlier_threshold,
    is_predictive, scored_model, scale_prior, outlier_halfwidth, score_bound, check_workspace,
    with_width, width_axis, residual_quality,
    # the local-optimization contract (robust/lo_contract.jl)
    lo_workspace, locally_optimize!, lo_width, deployed_scale_multiple,
    refine_external!, external_workspace, prepare_external!,
    # selection over a benchmark pool (robust/pool_selection.jl)
    best_pool_model, selection_staging, stage_selection, select_on_grid!,
    # the chart-rows contract (robust/rows.jl)
    rows_provider, lift_init, lift_refresh!, lift_jacobian!, lift_retract, lift_model,
    lift_valid, lift_rows, lift_residuals, lift_size,
    # the minimal-solver backends PoseLib.jl implements (geometry/solvers/)
    homography_4pt, fundamental_7pt, essential_5pt, essential_dlt, p3p,
    # the FitResult property forwarding (base/fit_result.jl)
    nested_metadata,
    # the TOML config hooks (base/config_toml.jl)
    toml_key, toml_skip_fields, toml_field,
    # geometry and camera accessors other packages implement for their own types
    centroid, region, moment_matrix, feature_frame, geom, csponds, focal_length,
    intrinsics, lens, model, extent, bounds_trait, is_posed, Pools.pair_scene,
    # the parameter protocol downstream lenses and target calibrations implement (geometry/cameras/lens_chart.jl)
    parameter_vector, parameter_names, parameter_bounds, with_parameters,
    # the target-calibration protocol a downstream calibration kind or target
    # implements (geometry/cameras/target_calibration.jl, targets.jl)
    calibration_jacobian, regularizer, is_planar, gauge_names, gauge_directions, calibration, design,
    # the surface a ray meets (geometry/primitives/surfaces.jl): a downstream
    # surface primitive answers the queries, a downstream calibration kind `surface`
    intersection, normal, chart, bounds, surface_point, surface,
    # the camera maps a downstream camera, lens or intrinsics implements
    # (geometry/cameras/camera_models.jl, lens_rays.jl), and its float-type conversion
    project, backproject, convert_eltype,
    # the Manifold protocol (base/manifold.jl): what a downstream Manifold subtype
    # must implement (tangent_size, retract, recenter) and may override
    # (local_coordinates, retract_jacobian, action_jacobian, zero_tangent,
    # restrict, tangent_part, transform_covariance), and the two a downstream
    # model type answers: minimal_manifold (its manifold) and ambient_coordinates
    # (what the default retract_jacobian differentiates; `model` is above)
    tangent_size, retract, recenter, local_coordinates, retract_jacobian, action_jacobian,
    zero_tangent, restrict, tangent_part, transform_covariance, minimal_manifold,
    ambient_coordinates,
    # the plotting stubs (visualization_stubs.jl) the Makie extensions implement
    imshow!, poly!, matches, matches!, ellipses!,
)

@doc """
    CONTRACT_GENERICS

The generic functions of VisualGeometryToolkitCore that another package may add
methods to across a package boundary — its extension points. Any stack package
may declare its own `CONTRACT_GENERICS` (a tuple or vector of functions); the
quality gate ([`contract_generic_violations`](@ref)) requires every method a
package adds to another stack package's function to land on one of that
owner's contracts. Functions owned by Base, Core, the standard library or any
package outside the stack are exempt.

VisualGeometryToolkitCore's, by group:
- the block-coordinate-descent hooks and readers (`update_model!`, …, `current_s2`);
- the chart-LM row protocol (`residual_rows!`, `residual_jacobian_rows!`);
- the RANSAC / local-optimization interface, the constraint-problem protocol,
  the scoring and local-optimization contracts, pool selection and the
  chart-rows contract;
- the minimal-solver backends PoseLib.jl implements;
- the `FitResult` property forwarding and the TOML config hooks;
- the geometry and camera accessors other packages implement for their own
  types (`centroid`, `intrinsics`, `lens`, `model`, …);
- the parameter protocol (`parameter_vector`, `parameter_names`,
  `parameter_bounds`, `with_parameters`);
- the target-calibration protocol (`calibration_jacobian`, `regularizer`,
  `is_planar`, `gauge_names`, `gauge_directions`) and
  the target accessors `calibration` and `design`;
- the surface protocol (`intersection`, `normal`, `chart`, `surface_point`, `bounds`) and
  `surface`, which builds a posed target's primitive;
- the camera maps `project` and `backproject`, and `convert_eltype`;
- the `Manifold` protocol: `tangent_size`, `retract` and `recenter`
  (required), `local_coordinates`, `retract_jacobian`, `action_jacobian`,
  `zero_tangent`, `restrict`, `tangent_part` and `transform_covariance`
  (optional), and `minimal_manifold` and `ambient_coordinates`, which a
  downstream model type answers (a model that is not a static array or a tuple
  of them adds `ambient_coordinates` rather than overriding `retract_jacobian`);
- the plotting stubs the Makie extensions implement.
""" CONTRACT_GENERICS

"""
    contract_generic_violations(M::Module) -> Vector{String}

Every function owned by another loaded stack package (a key of
[`stack_dependencies`](@ref)) that has a method defined in `M` or one of its
submodules but is not in its owner's `CONTRACT_GENERICS` (a package without
one has no contracts). Constructors of another package's types are not
counted (Aqua's piracy check covers them). Each finding is
`"Owner.f (file:line)"`, located at the first such method.
"""
function contract_generic_violations(M::Module)
    stack = keys(stack_dependencies())
    mine = Set(_module_tree(M))
    out = String[]
    for owner in Base.loaded_modules_array()
        owner === M && continue
        string(nameof(owner)) in stack && Base.moduleroot(owner) === owner || continue
        contracts = isdefined(owner, :CONTRACT_GENERICS) ? getfield(owner, :CONTRACT_GENERICS) : ()
        seen = Set{Any}()
        for m in _module_tree(owner), n in names(m; all = true)
            isdefined(m, n) && !Base.isdeprecated(m, n) || continue
            f = getfield(m, n)
            f isa Function && !(f isa Type) && f ∉ seen || continue
            push!(seen, f)
            Base.moduleroot(parentmodule(f)) === owner || continue
            any(c -> c === f, contracts) && continue
            hit = findfirst(meth -> meth.module in mine, collect(methods(f)))
            hit === nothing && continue
            meth = collect(methods(f))[hit]
            push!(out, "$(parentmodule(f)).$(nameof(f)) ($(meth.file):$(meth.line))")
        end
    end
    return sort!(out)
end
