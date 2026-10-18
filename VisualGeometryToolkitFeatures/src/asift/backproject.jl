# =============================================================================
# IMAS feature record + the IMAS-25 tilt schedule.
# =============================================================================
#
# Back-projection itself runs on-device in driver.jl (`_bbackproject!`).
# =============================================================================

struct ImasFeatureKA
    x::Float32; y::Float32; σ::Float32; intensity::Float32
    s_xx::Float32; s_xy::Float32; s_yy::Float32
    warp_idx::Int32
end

function _imas_tilts_25()
    ts = Tuple{Float64, Float64}[(1.0, 0.0)]
    for k in 0:7
        push!(ts, (2.62, π * k / 8))
    end
    for k in 0:15
        push!(ts, (5.18, π * k / 16))
    end
    return ts
end
