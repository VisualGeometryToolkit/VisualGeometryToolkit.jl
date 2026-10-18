# =============================================================================
# evaluation/engine/operating_point.jl — the refit protocol's fixed operating
# point and the evaluation's LO budget (configs/default.toml [refit],
# [local_optimization]). The scale-multiple → native maps and the native-axis
# conversions are VisualGeometryToolkitCore's (`to_native`, `native_map`,
# `outlier_halfwidth_to_loss_width`); the grid-column lookup is Tuning's
# `nearest_column`; the miscalibration corruption is Tuning's `corrupt_native`.
# =============================================================================

"""
    OPERATING_SCALE_MULTIPLE

The refit protocol's operating scale multiple, 2.5 by default: the scale
multiple at which the outlier-half-width ↔ loss-width crossover is inverted
when a loss-width-consuming refiner rides an outlier-half-width selector grid
([`consumed_native`](@ref)). A definition of the refit protocol, not a tuned
value: configs/default.toml `[refit] operating_scale_multiple`.
"""
const OPERATING_SCALE_MULTIPLE = Float64(_config_default("refit", "operating_scale_multiple"))

"""
    lo_iteration_budget() -> Int

THE equal LO model-update budget of the evaluation: configs/default.toml
`[local_optimization] iteration_budget`, so arms compare at equal cost. A
paper with its own protocol supplies its budget explicitly instead.
"""
lo_iteration_budget() = Int(_config_default("local_optimization", "iteration_budget"))
