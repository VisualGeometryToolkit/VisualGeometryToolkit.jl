# The SO(3) exponential and logarithm of VisualGeometryToolkitCore at 3d8a06da, exp_so3 and log_so3, over lie_tests'
# inputs: the rotation vectors, and the rotations exp_so3 makes of them (log's input, as in the test).
using VisualGeometryToolkitCore: exp_so3, log_so3

function record_core_lie(cases)
    for T in (Float64, Float32)
        ω = so3_test_rotation_vectors(T)
        case!(cases, "exp_so3/log_uniform_1e-12_pi_10000/$T", ω, ws -> exp_so3.(ws))
        case!(cases, "log_so3/exp_of_log_uniform_1e-12_pi_10000/$T", exp_so3.(ω), Rs -> log_so3.(Rs))
    end
    return cases
end
