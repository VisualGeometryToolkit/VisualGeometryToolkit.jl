# =============================================================================
# The termination record of a FitResult, read (the robust-estimation contract, §14)
# =============================================================================
#
# A result's verdict and rule are its fields, `r.retcode` and `r.stop_rule`;
# success is SciML's test, `successful_retcode(r)`. The two derived readings
# below are cross-type contracts.

"""
    successful_retcode(r::FitResult) -> Bool

Whether the solve succeeded — SciMLBase's accessor: [`successful_retcode`](@ref)
of its `r.retcode`, for every producer. A run that hit its iteration (or
RANSAC trial) cap, its time cap or a stall did not succeed, and a closed-form
solve did. For a RANSAC, whether it FOUND a model is [`found_model`](@ref).
"""
successful_retcode(r::FitResult) = successful_retcode(r.retcode)

"""
    found_model(r::FitResult) -> Bool

Whether the solve returned an estimate — a RANSAC's "a model was found"
(`r.u !== nothing`), whatever its `retcode`: a search that drew its whole
trial budget without reaching its confidence still found one.
"""
found_model(r::FitResult) = r.u !== nothing
