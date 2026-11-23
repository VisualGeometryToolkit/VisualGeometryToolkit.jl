# Fixtures of Core's SO(3) exponential and logarithm: the inputs of VisualGeometryToolkitCore's lie_tests ("exp_so3 and
# log_so3 bits"), whose exact bits RadiometricCamera's goldens pin. For T in (Float64, Float32), Xoshiro(20261027),
# 10⁴ draws of θ log-uniform on [1e-12, π] times a normalized randn direction, in the test's own draw order.
using Random: Xoshiro
using StaticArrays: SVector
using LinearAlgebra: normalize

function so3_test_rotation_vectors(::Type{T}) where {T}
    rng = Xoshiro(20261027)
    return [(θ = exp(log(1.0e-12) + (log(Float64(π)) - log(1.0e-12)) * rand(rng));
             SVector{3, T}(θ * normalize(randn(rng, SVector{3, Float64})))) for _ in 1:(10^4)]
end
