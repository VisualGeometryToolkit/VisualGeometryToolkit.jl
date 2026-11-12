# =============================================================================
# The termination record of a FitResult, read (the robust-estimation contract, §14)
# =============================================================================
#
# Every FitResult carries its record in its core (base/fit_result.jl), so each
# accessor reads one field and is inferred for every producer.

"""
    retcode(r::FitResult) -> ReturnCode.T

Why the fit stopped, its [`ReturnCode`](@ref) — the verdict every producer
records. Test success with [`converged`](@ref) (or
[`successful_retcode`](@ref)), never by comparing the code to
`ReturnCode.Success`; the rule that fired is [`stop_rule`](@ref). The legacy
Symbol spelling is [`retcode_symbol`](@ref)`(r)`.
"""
retcode(r::FitResult) = getfield(r, :retcode)

"""
    stop_rule(r::FitResult) -> StopRule

Which rule ended the fit ([`StopRule`](@ref)), the detail behind its
[`retcode`](@ref).
"""
stop_rule(r::FitResult) = getfield(r, :stop_rule)

"""
    retcode_symbol(r::FitResult) -> Symbol

The legacy Symbol spelling of the result's record,
`retcode_symbol(retcode(r), stop_rule(r))` (`:converged`, `:solved`,
`:max_iterations`, `:degenerate`, `:solve_failed`, `:stalled`, `:max_time`,
`:no_model`): for a log line or a persisted record that names the Symbol.
"""
retcode_symbol(r::FitResult) = retcode_symbol(retcode(r), stop_rule(r))

"""
    converged(r::FitResult) -> Bool

Whether the fit succeeded — STRICT: [`successful_retcode`](@ref) of its
[`retcode`](@ref), for every producer. A run that hit its iteration (or RANSAC
trial) cap, its time cap or a stall did not converge, and a closed-form solve
did. For a RANSAC, whether it FOUND a model is [`found_model`](@ref). Also
reachable as the property `r.converged`.
"""
converged(r::FitResult) = successful_retcode(retcode(r))

"""
    found_model(r::FitResult) -> Bool

Whether the fit returned an estimate — a RANSAC's "a model was found"
(`estimate(r) !== nothing`), whatever its [`retcode`](@ref): a search that
drew its whole trial budget without reaching its confidence still found one.
"""
found_model(r::FitResult) = getfield(r, :estimate) !== nothing
