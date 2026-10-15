# =============================================================================
# quality_tests.jl — static quality gates (Aqua + ExplicitImports + JET +
# calling conventions)
#
# Added in the 2026-07-29 release audit; ambiguity and import-owner gates
# added 2026-09-02. These gates keep the audits' wins from regressing: no
# stale deps, no piracy beyond the documented allowlist, no method
# ambiguities, no implicit or stale imports, and a JET report count that may
# only DECREASE. They check the package and its submodules; the extensions are
# separate root modules and are not checked.
#
# Two items: the shared gate skeleton (test_quality_gates) and the calling
# conventions. A worker runs items in no fixed order, so a gate may run after
# an item that extended a package function for its fixtures — the gate's
# ambiguity filter handles that.
# =============================================================================

@testitem "quality: gates" begin
    # The stack's one gate skeleton (VisualGeometryCore.test_quality_gates, the
    # QualityExt extension): Aqua, ExplicitImports, JET. VGC supplies only its own
    # bounds and allowlists; the history of each is kept here.
    using Test, Aqua, ExplicitImports, JET
    using CoordinateTransformations
    using GeometryBasics

    VisualGeometryCore.test_quality_gates(
        VisualGeometryCore;
        # Piracy allowlist: deliberate glue extending CoordinateTransformations /
        # GeometryBasics constructors over owned semantics (canonical-frame maps,
        # covariant conic transforms). Each entry is a design decision documented
        # at its definition site — additions to this list require the same, and
        # every entry must clear at least one method.
        treat_as_own = [
            # This stack owns circle/rect SEMANTICS even though the types are
            # GeometryBasics'; the CT constructors are the canonical-frame-map
            # idiom documented in transforms/coord_maps.jl. (`Translation` left
            # this list 2026-09-21 with `ustrip(::Translation)`, its last method.)
            CoordinateTransformations.AffineMap,
            # `transform_deriv(::PerspectiveMap, X)` — the chain-rule leaf the
            # camera's imaging transform needs (cameras/projections.jl).
            CoordinateTransformations.PerspectiveMap,
            GeometryBasics.Point,
            GeometryBasics.HyperRectangle,
            # Circles. Aqua matches a type by its base name and `Circle` is only
            # an alias of `HyperSphere{2}`, so the entry must be `HyperSphere`: it
            # covers the Primitives circle operators (translate, scale, the
            # identity constructor) and the CT callables on circles, such as
            # `(lm::LinearMap)(::Circle)`.
            GeometryBasics.HyperSphere,
        ],
        # pnp_pixel_whitening: imported into the top module solely as the binding
        # RobustVisualGeometry imports — unused here by design.
        # epipolar_map / FrameDependence: docstring-scope
        # imports into Cameras (used only by @ref cross-references; see the
        # import sites in cameras.jl).
        # RegistryTools / dimvaltype: the PixelUnits registry module's imports
        # for FlexUnits' `@generate_*` macros, used only in their expansions.
        stale_ignore = (:pnp_pixel_whitening, :epipolar_map, :FrameDependence, :RegistryTools, :dimvaltype),
        # Not run here before the skeleton was shared, and it spawns a process.
        persistent_tasks = false,
        # RATCHET: may only go DOWN. Lower it whenever you fix some.
        # Baseline set 2026-07-29 at 25; lowered to 21 on 2026-09-02 (a real
        # UndefVarError in Synthetic.imc_pose_error fixed, three reports retired
        # with dead code); lowered to 19 on 2026-09-15 (the uncalled
        # zero-allocation path of the generic loop deleted, and its two stub
        # reports with it). RAISED to 21 on 2026-09-19 — the one exception, not a
        # regression: the weight-policy reading interface (`inlier_weight`,
        # `robust_weight`, `is_inlier`, `inlier_mask`) was hoisted into VGC for
        # RobustVisualGeometry and RansacScoringDoneRight, whose VGC defaults
        # each raise one report of the abstract-stub class. Lowered to 20 on
        # 2026-09-21 (the `_remap_sample` union split retired with a real bug).
        # 2026-09-22: KernelAbstractions LAUNCH reports are no longer counted
        # (filtered by signature inside the gate): 9 at the switch — the 4
        # AbstractBlockCoordinateMethod stubs (init_state!/update_model!/
        # is_converged/block_coordinate_result, implemented downstream), the 2
        # weight-policy defaults, the 2 solver-backend fallbacks
        # (homography_4pt/fundamental_7pt on AbstractSolverBackend — PoseLib.jl
        # adds the methods), the FramedImage `window::Any` split. Lowered to 8
        # on 2026-10-06: `is_converged` retired with the stopping criteria
        # (a method's `stopping_criterion` has a default, StopNever).
        # RAISED to 13 on 2026-10-09 with the K+ contracts (net +5, the count
        # of the package ANALYSED ALONE — an item that defines a test problem's
        # protocol methods in the same worker hides three of these): the two
        # weight-policy defaults are gone (−2) and seven reports arrived (+7).
        # Five are the abstract-stub class, with the code that moved here from
        # RansacScoringDoneRight — the constraint-problem protocol
        # (`data_size`, `codimension` on an instance and on a type,
        # `constraint_with_jacobians`, implemented by each problem) and the
        # predictive scorers' optional `score!(…, ::Uncertain, …)` behind
        # `score_model!` — and two come from the new local-optimization contract
        # (`_loss_at`/`_width_s2` on `lo_width(lo) === missing`, the default of
        # an optimizer without a width mode, which has no method there). Lower
        # it as those go.
        jet_bound = 13,
    )
end

@testitem "quality: calling conventions" begin
    # Julia's style-guide argument order and `!` naming, which
    # conventions/signatures.md makes binding for the stack: types
    # positional and out of the data's way, no precision keywords, no public
    # function that mutates an argument without `!`. The mechanical check is
    # package-cleanup's argorder_check.jl, vendored here (downstream packages
    # include it from `pkgdir(VisualGeometryCore)`); keep the two copies equal.
    # A finding is accepted only when .argorder_allow names it with its reason — the
    # allowlist only shrinks. It is empty today, so the file does not exist
    # (`read_allowlist` returns the empty set for a missing file). The mutation rule is a heuristic over direct calls;
    # the review in package-cleanup Phase 4a covers the rest.
    include(joinpath(pkgdir(VisualGeometryCore), "test", "argorder_check.jl"))
    allow = read_allowlist(joinpath(pkgdir(VisualGeometryCore), ".argorder_allow"))
    bad = [f for f in check(VisualGeometryCore) if (f[1], f[2]) ∉ allow]
    @test isempty(bad)
    isempty(bad) || foreach(f -> @warn("argument-order finding", f), bad)
end
