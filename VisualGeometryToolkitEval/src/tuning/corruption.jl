# =============================================================================
# src/tuning/corruption.jl — THE deployment-corruption map of the
# miscalibration protocol: a tuned native width scaled by the deployment error ζ.
# =============================================================================

"""
    corrupt_native(native::Real, ζ::Real) -> Float64

THE deployment-corruption map (docs/src/protocol.md §3, miscalibration): a
tuned-at-CALIBRATION native width (selection outlier half-width, loss width, or
the oracle supplied scale a native is formed from) scaled by the deployment
error ζ — the model of a mis-estimated inlier scale at deployment time. Every
ζ-scaling of a calibrated native goes through this one function so the
corruption sites are greppable.

PARTITION-ANCHORED quantities never go through it: a tuned scale multiple, a
seed-partition Student-t ŝ², or any quantity re-derived from the deployed
data's own partition is immune to calibration error by construction — that
immunity IS the claim the miscal experiment measures.
"""
corrupt_native(native::Real, ζ::Real) = Float64(ζ) * Float64(native)
