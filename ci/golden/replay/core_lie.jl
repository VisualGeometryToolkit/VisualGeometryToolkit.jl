# The SO(3) exponential and logarithm are unchanged in Core 0.14 (exp_so3, log_so3 keep their names: jacobians.md):
# every case replays the same call, the identity mapping.
using VisualGeometryToolkitCore: exp_so3, log_so3

function replay_core_lie(cases)
    for T in (Float64, Float32)
        ω = so3_test_rotation_vectors(T)
        replay!(cases, "exp_so3/log_uniform_1e-12_pi_10000/$T", ω, ws -> exp_so3.(ws))
        replay!(cases, "log_so3/exp_of_log_uniform_1e-12_pi_10000/$T", exp_so3.(ω), Rs -> log_so3.(Rs))
    end
    return cases
end
