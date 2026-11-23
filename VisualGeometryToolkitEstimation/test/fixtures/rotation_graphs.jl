# The rotation-averaging tests' error of an estimate (fit_rotation_averaging_tests.jl, view_graph_tests.jl). Include-once
# safe (functions only).
using StaticArrays: SMatrix
using Rotations: RotMatrix, rotation_angle

# the largest angle between the estimated rotation `Rh[k]` and the truth `R[k]`, over the views k in 1:n
max_rotation_error(Rh, R, n) = maximum(k -> rotation_angle(RotMatrix{3}(SMatrix{3, 3}(Rh[k])' * R[k])), 1:n)
