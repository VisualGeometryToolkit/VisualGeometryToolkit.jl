# Fixtures of VisualGeometryToolkitEstimation's other reweighted fits: the robust-estimation contract's five scenes
# (fit_contract_tests' FitContractScenes), fit_rotation_tests, fit_linear_tests, the rigid-alignment walk,
# robot_world_hand_eye_tests' _synth and gauge scenes, refine_robot_world_hand_eye_tests' rw_scene, the
# rotation-averaging graphs (fit_rotation_averaging_tests, view_graph_tests, rotation_covariance_tests), the
# residual-mixture draws and its test-only HalfwidthScoring (residual_mixture_tests), the SIMFIT++ pools
# (simfitpp_tests), and the termination-record, units-boundary and strategy-keyword scenes. Each noise draw takes
# its seed from the caller (StableRNG; the suite's Xoshiro/MersenneTwister draws are replaced by StableRNG(seed) so
# the fixture bits do not depend on Base's RNG streams).
using Test, StableRNGs, StaticArrays, Random, LinearAlgebra
using Rotations: RotMatrix, RotXYZ, RotZ, RotationVec, QuatRotation, AngleAxis
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation

const _EOF_PS = VgtkEstimation.synthetic_scenes().PoseScenes
const _EOF_DB = VgtkEstimation.synthetic_scenes().DivisionBoards

# =================================================================================================================
# The robust-estimation contract's scenes (fit_contract_tests.jl FitContractScenes)
# =================================================================================================================
# `contract_linear`, `contract_rotation`, `contract_averaging`, `contract_points`, `contract_handeye`, each drawn from
# the caller's RNG (here StableRNG(seed)): the suite's fixture file
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "contract_scenes.jl"))

# =================================================================================================================
# fit_rotation (fit_rotation_tests.jl)
# =================================================================================================================
function eof_rotation_clean()
    R = RotMatrix(RotXYZ(0.4, -1.1, 2.3))
    as = [normalize(SVector(sin(3k), cos(5k), sin(7k + 1))) for k in 1:30]
    return VgtkCore.Correspondences(as, [R * a for a in as])
end
eof_rotation_clean_long() = (cs = eof_rotation_clean(); VgtkCore.Correspondences(3 .* VgtkCore.sources(cs), VgtkCore.targets(cs)))
eof_rotation_one_pair() = (cs = eof_rotation_clean(); VgtkCore.Correspondences(VgtkCore.sources(cs)[1:1], VgtkCore.sources(cs)[1:1]))
# 200 directions, 1e-3 rad tangent noise, every fifth a random direction; `out` its outlier indices
function eof_rotation_outliers(seed; n = 200, σ = 1.0e-3)
    rng = StableRNG(seed)
    R = RotMatrix(RotXYZ(0.2, 0.3, -0.5))
    as = [normalize(randn(rng, SVector{3, Float64})) for _ in 1:n]
    function jitter(b)
        t = normalize(cross(b, randn(rng, SVector{3, Float64})))
        return normalize(b + σ * randn(rng) * t + σ * randn(rng) * normalize(cross(b, t)))
    end
    bs = [jitter(R * a) for a in as]
    out = collect(1:5:n)
    for i in out
        bs[i] = normalize(randn(rng, SVector{3, Float64}))
    end
    return VgtkCore.Correspondences(as, bs), out
end
function eof_rotation_outliers_prior(seed)
    cs, out = eof_rotation_outliers(seed)
    prior = ones(length(cs)); prior[out] .= 0
    return VgtkCore.withattrs(cs; weights = prior)
end

# =================================================================================================================
# fit_linear (fit_linear_tests.jl, strategy_keywords_tests.jl, termination_record_tests.jl)
# =================================================================================================================
# the 400 × 5 design and its clean, contaminated (80 rows + 5·N(0,1)), shifted (10 rows + 100, prior 0) and the
# hoisted loop's (0.01 noise, 40 rows + 1) right-hand sides
function eof_linear_scene(seed; m = 400, n = 5)
    rng = StableRNG(seed)
    A = randn(rng, m, n); x0 = randn(rng, n)
    y_out = A * x0 .+ 1.0e-3 .* randn(rng, m)
    bad = randperm(rng, m)[1:80]
    y_out[bad] .+= 5 .* randn(rng, 80)
    y_shift = A * x0; y_shift[1:10] .+= 100
    p_shift = ones(m); p_shift[1:10] .= 0
    y_loop = A * x0 .+ 0.01 .* randn(rng, m); y_loop[1:40] .+= 1
    cnt = Float64.(rand(rng, 0:2, m))
    p_few = zeros(m); p_few[1:3] .= 1
    return (; A, y_clean = A * x0, y_out, y_shift, p_shift, y_loop, cnt, p_few)
end
# the strategy item's 300 × 4 design, 60 rows + 3·N(0,1)
function eof_linear_strategy_scene(seed; m = 300, n = 4)
    rng = StableRNG(seed)
    A = randn(rng, m, n); x0 = randn(rng, n)
    y = A * x0 .+ 1.0e-3 .* randn(rng, m); y[1:60] .+= 3 .* randn(rng, 60)
    return A, y
end
eof_linear_tiny() = (A = [1.0 0.0; 0.0 1.0; 1.0 1.0; 2.0 1.0]; (A, A * [0.5, -1.0]))
eof_linear_unsolvable(seed) = (rng = StableRNG(seed); (randn(rng, 8, 3), randn(rng, 8)))

# =================================================================================================================
# fit_rigid_alignment (fit_contract_tests.jl: a walk of 150 camera centres, five gross)
# =================================================================================================================
function eof_rigid_walk(seed)
    rng = StableRNG(seed)
    R = RotMatrix(RotXYZ(0.05, -0.1, 1.2)); t = SVector(1.0, -2.0, 0.3)
    s = range(0, 6π; length = 150)
    walk = [SVector(2cos(u), 1.5sin(u), 0.3sin(3u)) for u in s]
    gt = [R * a + t + 0.005 .* randn(rng, SVector{3, Float64}) for a in walk]
    bad = [44, 52, 57, 60, 63]
    for (i, d) in zip(bad, (0.9, 1.6, 2.2, 2.2, 2.4))
        gt[i] += d .* normalize(randn(rng, SVector{3, Float64}))
    end
    return walk, gt, bad
end
eof_rigid_cs(seed) = ((walk, gt, _) = eof_rigid_walk(seed); VgtkCore.Correspondences(walk, gt))
function eof_rigid_clean_subset(seed)
    walk, gt, bad = eof_rigid_walk(seed)
    keep = setdiff(1:150, bad)
    return VgtkCore.Correspondences(walk[keep], gt[keep])
end
function eof_rigid_prior(seed)
    walk, gt, bad = eof_rigid_walk(seed)
    p = ones(150); p[bad] .= 0
    return VgtkCore.withattrs(VgtkCore.Correspondences(walk, gt); weights = p)
end
eof_rigid_physical(seed) = ((walk, gt, _) = eof_rigid_walk(seed); VgtkCore.Correspondences([a .* VgtkCore.m for a in walk], [b .* VgtkCore.m for b in gt]))
eof_rigid_two() = ((walk, gt, _) = eof_rigid_walk(1); VgtkCore.Correspondences(walk[1:2], gt[1:2]))
eof_rigid_collinear() = (line = [SVector(Float64(k), 0.0, 0.0) for k in 1:5]; VgtkCore.Correspondences(line, line))

# =================================================================================================================
# fit_robot_world_hand_eye (robot_world_hand_eye_tests.jl, fit_contract_tests.jl, units_boundary_tests.jl)
# =================================================================================================================
# A_i X_m = Y B_i^m: markers[m] holds A_i => B_i^m, one per frame it is seen in
function eof_rwhe_synth(seed; N = 25, M = 4, σ_R = 0.0, σ_t = 0.0, drop = 0.0)
    rng = StableRNG(seed)
    (; Y, X, A) = _EOF_PS.robot_world_hand_eye_scene(rng; N, M, Y = (t_scale = 0.5,), X = (ang_max = 0.5, t_scale = 0.3), A = (t_scale = 1.0,))
    markers = Dict(m => Pair{typeof(Y), typeof(Y)}[] for m in 1:M)
    for i in 1:N, m in 1:M
        rand(rng) < drop && continue
        B = _EOF_PS.marker_pose(Y, A[i], X[m])
        push!(markers[m], A[i] => (σ_R == 0 && σ_t == 0 ? B : _EOF_PS.perturb_pose(rng, B; σ_R, σ_t)))
    end
    return markers
end
# the rig base 10 km from the world origin
eof_rwhe_far(seed) = Dict(m => [VgtkCore.EuclideanMap(VgtkCore.rotation(a), VgtkCore.translation(a) + SVector(1.0e4, -7.0e3, 2.5e3)) => b for (a, b) in cs]
                          for (m, cs) in eof_rwhe_synth(seed; N = 12, M = 3))
# rig rotations all about z, zero translation: the two-parameter gauge family
function eof_rwhe_pure_z(seed)
    rng = StableRNG(seed)
    Y = _EOF_PS.rand_pose(rng; t_scale = 0.5)
    X = [_EOF_PS.rand_pose(rng; ang_max = 0.3, t_scale = 0.3) for _ in 1:2]
    A = [VgtkCore.EuclideanMap(RotMatrix(RotZ(θ)), SVector(0.0, 0.0, 0.0); check = false) for θ in range(0.3, 2.0; length = 8)]
    return Dict(m => [A[i] => _EOF_PS.marker_pose(Y, A[i], X[m]) for i in 1:8] for m in 1:2)
end
eof_rwhe_underdetermined() = (mk = eof_rwhe_synth(7; N = 6, M = 2); Dict(1 => mk[1][1:2], 2 => mk[2][1:1]))
eof_rwhe_empty_marker() = (mk = eof_rwhe_synth(7; N = 6, M = 2); Dict(1 => mk[1], 2 => empty(mk[2])))
eof_rwhe_named() = (mk = eof_rwhe_synth(7; N = 6, M = 2); Dict("b" => mk[1], "a" => mk[2]))
# 30 frames × 2 markers at 2e-3, five gross observations of marker 1 (fit_contract_tests)
function eof_rwhe_gross(seed; N = 30, M = 2)
    rng = StableRNG(seed)
    (; Y, X, A) = _EOF_PS.robot_world_hand_eye_scene(rng; N, M, Y = (t_scale = 0.5,), X = (ang_max = 0.5, t_scale = 0.3), A = (t_scale = 1.0,))
    markers = Dict(m => Pair{typeof(Y), typeof(Y)}[] for m in 1:M)
    for i in 1:N, m in 1:M
        push!(markers[m], A[i] => _EOF_PS.perturb_pose(rng, _EOF_PS.marker_pose(Y, A[i], X[m]); σ_R = 2.0e-3, σ_t = 2.0e-3))
    end
    for i in (4, 9, 15, 22, 27)
        a, b = markers[1][i]
        markers[1][i] = a => _EOF_PS.perturb_pose(rng, b; σ_R = 0.8, σ_t = 1.0)
    end
    return markers
end
# units_boundary_tests: 20 frames × 3 markers at 1e-3, plain and physical (metres)
function eof_rwhe_units(seed)
    rng = StableRNG(seed)
    (; Y, X, A) = _EOF_PS.robot_world_hand_eye_scene(rng; N = 20, M = 3, Y = (t_scale = 0.5,), X = (ang_max = 0.5, t_scale = 0.3), A = (t_scale = 1.0,))
    markers = Dict(k => [a => _EOF_PS.perturb_pose(rng, _EOF_PS.marker_pose(Y, a, X[k]); σ_R = 1.0e-3, σ_t = 1.0e-3) for a in A] for k in 1:3)
    phys(E) = VgtkCore.with_unit(E, VgtkCore.m)
    markers_m = Dict(k => [phys(first(c)) => phys(last(c)) for c in cs] for (k, cs) in markers)
    return markers, markers_m
end

# =================================================================================================================
# refine_robot_world_hand_eye (refine_robot_world_hand_eye_tests.jl, termination_record_tests.jl, units_boundary_tests.jl)
# =================================================================================================================
# `rw_grid`, `rw_marker_origin`, `rw_scene`, `rw_markers`, `rw_seed`: the suite's fixture file
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "robot_world_scenes.jl"))
# (guess, camera, markers): a scene and its perturbed seed
function eof_rw_problem(seed; σ = 0.5, rot = 0.01, tr = 0.01, kw...)
    rng = StableRNG(seed)
    sc = rw_scene(rng; σ, kw...)
    return (rw_seed(rng, sc; rot, tr), sc.camera, rw_markers(sc))
end
# 5 % of the features moved 20–40 px
function eof_rw_moved(seed)
    rng = StableRNG(seed)
    sc = rw_scene(rng; σ = 0.5)
    mk = rw_markers(sc)
    moved = Dict(m => [A => [X => (rand(rng) < 0.05 ? u + (20 + 20rand(rng)) * normalize(randn(rng, SVector{2, Float64})) : u)
                             for (X, u) in cs] for (A, cs) in els] for (m, els) in mk)
    return (rw_seed(rng, sc), sc.camera, moved)
end
# the argument contract's refused inputs, from one 2-marker scene
function eof_rw_contract_inputs()
    sc = rw_scene(StableRNG(2); M = 2)
    seed = rw_seed(StableRNG(3), sc)
    mk = rw_markers(sc)
    rigid = _EOF_PS.rigid
    bad(f) = Dict(1 => [mk[1][1:(end - 1)]; [first(mk[1][end]) => f(last(mk[1][end]))]], 2 => mk[2])
    inputs = [
        "unknown_marker" => Dict(7 => mk[1], 1 => mk[1], 2 => mk[2]),
        "missing_marker" => Dict(1 => mk[1]),
        "underdetermined" => Dict(1 => [first(mk[1][1]) => last(mk[1][1])[1:1]], 2 => [first(mk[2][1]) => last(mk[2][1])[1:1]]),
        "empty_marker" => Dict(1 => mk[1], 2 => empty(mk[2])),
        "empty_features" => Dict(1 => mk[1], 2 => [first(e) => empty(last(e)) for e in mk[2]]),
        "nan_pixel" => bad(cs -> [first(cs[1]) => SVector(NaN, last(cs[1])[2]); cs[2:end]]),
        "nan_point" => bad(cs -> [SVector(NaN, 0.0, 0.0) => last(cs[1]); cs[2:end]]),
        "behind" => bad(cs -> [SVector(0.0, 0.0, -10.0) => last(cs[1]); cs[2:end]]),
        "nan_rig" => Dict(1 => [mk[1][1:(end - 1)]; [rigid(VgtkCore.rotation(first(mk[1][end])), SVector(NaN, 0.0, 0.0)) => last(mk[1][end])]], 2 => mk[2]),
        "untouched" => mk,
    ]
    return seed, sc.camera, inputs
end
# the pure-z gauge family against its features
function eof_rw_pure_z()
    rng = StableRNG(7)
    rigid = _EOF_PS.rigid
    camera = VgtkCore.CameraModel(; focal = 600.0, pp = (400.0, 300.0))
    Y = rigid(RotMatrix{3}(1.0I), SVector(0.0, 0.0, -1.2))
    X = [rigid(AngleAxis(0.3 * rand(rng), normalize(randn(rng, 3))...), SVector(0.08 * (m - 1.5), 0.03, 0.0)) for m in 1:2]
    A = [rigid(RotMatrix(RotZ(θ)), SVector(0.0, 0.0, 0.0)) for θ in range(0.3, 2.0; length = 8)]
    markers = Dict(m => [A[i] => [P => SVector{2, Float64}(VgtkCore.project(camera, _EOF_PS.marker_pose(Y, A[i], X[m])(P))...) for P in rw_grid()] for i in 1:8] for m in 1:2)
    return (VgtkEstimation.RobotWorldHandEyeCalibration(Y, Dict(1 => X[1], 2 => X[2])), camera, markers)
end
# termination_record_tests' set (seed 3) and units_boundary_tests' (seed 13, with its physical twin)
function eof_rw_fixed_scene(seed, nA, ang)
    rng = StableRNG(seed)
    rigid = _EOF_PS.rigid
    camera = VgtkCore.CameraModel(; focal = 600.0, pp = (400.0, 300.0))
    grid = rw_grid()
    Y = rigid(AngleAxis(0.7, normalize(SVector(0.3, -0.2, 0.9))...), SVector(0.2, -0.1, 0.3))
    X = [rigid(AngleAxis(0.2, normalize(SVector(1.0, 0.5, 0.2))...), SVector(0.08 * (m - 1.5), 0.0, 0.0)) for m in 1:2]
    A = [Y ∘ rigid(AngleAxis(ang * rand(rng), normalize(randn(rng, 3))...), SVector(0.1 * randn(rng), 0.1 * randn(rng), 1.2)) for _ in 1:nA]
    markers = Dict(m => [A[i] => [P => SVector{2, Float64}(VgtkCore.project(camera, _EOF_PS.marker_pose(Y, A[i], X[m])(P))) + 0.3 * randn(rng, SVector{2, Float64})
                                  for P in grid] for i in eachindex(A)] for m in 1:2)
    guess = VgtkEstimation.RobotWorldHandEyeCalibration(_EOF_PS.perturb_pose(rng, Y; σ_R = 0.01, σ_t = 0.01),
                                                        Dict(m => _EOF_PS.perturb_pose(rng, X[m]; σ_R = 0.01, σ_t = 0.01) for m in 1:2))
    return (guess, camera, markers)
end
function eof_rw_physical(seed)
    guess, camera, markers = eof_rw_fixed_scene(seed, 8, 0.35)
    m = VgtkCore.m
    phys(E) = VgtkCore.with_unit(E, m)
    markers_m = Dict(k => [phys(first(e)) => [(P .* m) => q for (P, q) in last(e)] for e in els] for (k, els) in markers)
    guess_m = VgtkEstimation.RobotWorldHandEyeCalibration(phys(guess.Y), Dict(k => phys(guess.X[k]) for k in guess.ids))
    return (guess_m, camera, markers_m)
end

# =================================================================================================================
# fit_rotation_averaging, fit_filtered_rotation_averaging
# =================================================================================================================
# n frames (frame 1 the identity), edges to the next three, the truth R and its relatives
function eof_avg_truth(rng; n = 8)
    return Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
end
eof_avg_edges(n) = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 3)]
eof_avg_weighted(d, p) = Dict(e => VgtkCore.Attributed(d[e], (; weight = p[e])) for e in keys(d))
# fit_rotation_averaging_tests: consistent, noisy with one bad edge (2, 4), its zero-weight twin, the bridge
# chains, the hung frame 9
function eof_avg_scene(seed)
    rng = StableRNG(seed)
    n = 8
    R = eof_avg_truth(rng; n)
    rel(i, j) = R[j]' * R[i]
    edges = eof_avg_edges(n)
    consistent = Dict(e => rel(e...) for e in edges)
    noisy = Dict(e => RotMatrix(RotationVec((2.0e-3 .* randn(rng, 3))...)) * rel(e...) for e in edges)
    bad = (2, 4); noisy[bad] = RotMatrix(RotationVec(0.5, -0.3, 0.2)) * noisy[bad]
    dropped = eof_avg_weighted(noisy, Dict(e => (e == bad ? 0.0 : 1.0) for e in keys(noisy)))
    chain = Dict((k, k + 1) => rel(k, k + 1) for k in 1:3)
    zero_bridge = eof_avg_weighted(chain, Dict(e => (e == (2, 3) ? 0.0 : 1.0) for e in keys(chain)))
    all_zero = eof_avg_weighted(chain, Dict(e => 0.0 for e in keys(chain)))
    R[9] = RotMatrix(QuatRotation(normalize(randn(rng, 4))))
    hung = Dict(e => RotMatrix(RotationVec((2.0e-3 .* randn(rng, 3))...)) * rel(e...) for e in edges)
    hung[(7, 9)] = RotMatrix(RotationVec(0.6, 0.0, 0.0)) * rel(7, 9)
    hung[(8, 9)] = RotMatrix(RotationVec(-0.6, 0.0, 0.0)) * rel(8, 9)
    hung = eof_avg_weighted(hung, Dict(e => (9 in e ? 1.0e-3 : 1.0) for e in keys(hung)))
    split = Dict((1, 2) => rel(1, 2), (3, 4) => rel(3, 4))
    single = Dict((1, 2) => rel(1, 2))
    return (; consistent, noisy, dropped, zero_bridge, all_zero, hung, split, single)
end
# the least-squares item: 0.02 rad noise
function eof_avg_ls_scene(seed)
    rng = StableRNG(seed)
    n = 8
    R = eof_avg_truth(rng; n)
    return Dict(e => SMatrix{3, 3}(RotMatrix(RotationVec((0.02 .* randn(rng, 3))...)) * (R[e[2]]' * R[e[1]])) for e in eof_avg_edges(n))
end
# view_graph_tests' spanning-tree item: 1e-3 noise, weights 50 + i; R the truth (an explicit seed reads it)
function eof_avg_tree_scene(seed)
    rng = StableRNG(seed)
    n = 8
    R = eof_avg_truth(rng; n)
    rel(i, j) = R[j]' * R[i]
    edges = eof_avg_edges(n)
    noisy = Dict(e => RotMatrix(RotationVec((1.0e-3 .* randn(rng, 3))...)) * rel(e...) for e in edges)
    weighted = Dict(e => VgtkCore.Attributed(noisy[e], (; weight = 50.0 + e[1])) for e in edges)
    G = RotMatrix(RotationVec(0.3, 0.1, -0.2))
    gauged_truth = Dict(k => G * R[k] for k in 1:n)
    truth = Dict(k => R[k] for k in 1:n)
    partial = Dict(1 => R[1])
    split = Dict((1, 2) => rel(1, 2), (3, 4) => rel(3, 4))
    sub = Dict(e => noisy[e] for e in edges if e[1] >= 2)
    flipped = Dict(((j, i) => noisy[(i, j)]' for (i, j) in edges if i == 1)..., (e => noisy[e] for e in edges if e[1] != 1)...)
    isolated = VgtkEstimation.ViewGraph(vcat(1:n, 99), Dict(e => noisy[e] for e in edges))
    return (; noisy, weighted, gauged_truth, truth, partial, split, sub, flipped, isolated)
end
# view_graph_tests' filtered item: 10 frames, 1e-3 noise, the pair (4, 6) wrong; plain or Uncertain edges
function eof_avg_filter_graph(seed; uncertain = false)
    rng = StableRNG(seed)
    n = 10
    R = eof_avg_truth(rng; n)
    pairs = eof_avg_edges(n)
    rel = Dict(p => RotMatrix(RotationVec((1.0e-3 .* randn(rng, 3))...)) * R[p[2]]' * R[p[1]] for p in pairs)
    rel[(4, 6)] = RotMatrix(RotationVec(0.8, -0.4, 0.3)) * rel[(4, 6)]
    Σ = SMatrix{3, 3, Float64}(1.0e-6I)
    edge(x) = uncertain ? VgtkCore.Attributed(VgtkCore.Uncertain(x, VgtkCore.CovMat(Σ)), (; weight = 100.0)) : VgtkCore.Attributed(x, (; weight = 100.0))
    return VgtkEstimation.ViewGraph(Dict(p => edge(rel[p]) for p in pairs))
end
# termination_record_tests' consistent 5-frame graph
eof_avg_termination_graph() = (Rs = Dict(k => SMatrix{3, 3, Float64, 9}(RotXYZ(0.3k, -0.2k, 0.1k^2)) for k in 1:5);
                               VgtkEstimation.ViewGraph(Dict((i, j) => Rs[j] * Rs[i]' for i in 1:5 for j in (i + 1):5)))
# rotation_covariance_tests' synthetic Uncertain edges: 7 frames, σ = 0.004, the weak edge (3, 4) at 0.04
function eof_avg_uncertain_scene(seed)
    rng = StableRNG(seed)
    n = 7
    edges = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 2)]
    weak = (3, 4); σ, σw = 0.004, 0.04
    s(e) = e == weak ? σw : σ
    R = eof_avg_truth(rng; n)
    rel = Dict(e => RotMatrix(RotationVec((s(e) .* randn(rng, 3))...)) * (R[e[2]]' * R[e[1]]) for e in edges)
    attributed = Dict(e => VgtkCore.Attributed(VgtkCore.Uncertain(rel[e], VgtkCore.CovMat(SMatrix{3, 3}(s(e)^2 * I))), (; weight = 1.0)) for e in edges)
    eye = Dict(e => VgtkCore.Uncertain(rel[e], VgtkCore.CovMat(SMatrix{3, 3}(σ^2 * I))) for e in edges)
    Σω = SMatrix{3, 3}(Diagonal(SVector(1.0, 2.0, 3.0))) * σ^2
    nine = Dict(e => VgtkCore.Uncertain(rel[e], VgtkCore.CovMat(VgtkCore.propagate(VgtkCore.retract_jacobian(VgtkCore.SO3Manifold(SMatrix{3, 3}(rel[e]))), Σω))) for e in edges)
    three = Dict(e => VgtkCore.Uncertain(rel[e], VgtkCore.CovMat(Σω)) for e in edges)
    two = Dict(e => VgtkCore.Uncertain(rel[e], VgtkCore.CovMat(SMatrix{2, 2}(1.0I))) for e in edges)
    return (; rel, attributed, eye, nine, three, two)
end
# rotation_covariance_tests' pixel-edge scene (the pairwise fits are computed by the recorder, not here): per edge
# of 5 frames, 30 directions imaged in both views with anisotropic detection covariances, the weak edge (2, 3)
# 8× noisier
eof_edge_camera() = VgtkCore.CameraModel(800.0VgtkCore.px, SA[320.0VgtkCore.px, 240.0VgtkCore.px])
function eof_edge_pixels(seed)
    rng = StableRNG(seed)
    cam = eof_edge_camera()
    nf = 5
    edges = [(i, j) for i in 1:nf for j in (i + 1):min(nf, i + 2)]
    s(e) = e == (2, 3) ? 4.0 : 0.5
    R = Dict(k => SMatrix{3, 3}(RotMatrix(RotXYZ(0.03k, -0.04k, 0.02k))) for k in 1:nf)
    R = Dict(k => R[1]' * R[k] for k in 1:nf)
    function covs(n, sc)
        map(1:n) do _
            θ = π * rand(rng); a = 1 + 3 * rand(rng)
            Q = SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ))
            sc^2 * Q * SMatrix{2, 2}(a, 0.0, 0.0, 1 / a) * Q'
        end
    end
    jitter(u, Σ) = typeof(u)(u + SMatrix{2, 2}(cholesky(Symmetric(Matrix(Σ))).L) * SVector(randn(rng), randn(rng)))
    unc(us, Σs) = [VgtkCore.Uncertain(u, VgtkCore.CovMat(SMatrix{2, 2}(Σ))) for (u, Σ) in zip(us, Σs)]
    return map(edges) do (i, j)
        Rij = R[j]' * R[i]
        dirs = [normalize(SVector(0.7 * (rand(rng) - 0.5), 0.5 * (rand(rng) - 0.5), 1.0)) for _ in 1:30]
        u1 = [VgtkCore.project(cam, d) for d in dirs]; u2 = [VgtkCore.project(cam, Rij * d) for d in dirs]
        Σ1 = covs(30, s((i, j))); Σ2 = covs(30, s((i, j)))
        (i, j) => VgtkCore.correspondences(unc([jitter(u, Σ) for (u, Σ) in zip(u1, Σ1)], Σ1), unc([jitter(u, Σ) for (u, Σ) in zip(u2, Σ2)], Σ2))
    end
end

# =================================================================================================================
# fit_residual_mixture (residual_mixture_tests.jl, termination_record_tests.jl)
# =================================================================================================================
# The suite's test-only scorer with a uniform outlier model (`HalfwidthScoring`) and its codimension-2 residual draw
# (`reprojection_residuals`: Rayleigh inliers at σ, gross outliers uniform over the ±halfwidth square): the suite's
# fixture file
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "residual_mixture.jl"))
eof_mixture_residuals(seed; n_in, n_out, σ, halfwidth) = first(reprojection_residuals(StableRNG(seed); n_in, n_out, σ, halfwidth))
eof_mixture_signed() = (r = eof_mixture_residuals(11; n_in = 100, n_out = 10, σ = 0.5, halfwidth = 50.0); r[3] = -r[3]; r)
# termination_record_tests' scalar draw
eof_mixture_scalar(seed) = abs.(vcat(0.5 .* randn(StableRNG(seed), 200), 40.0 .* rand(StableRNG(seed + 100), 40)))
# the calibration pipeline's [classification] EM table
eof_mixture_classification_em() = VgtkEstimation.mixture_em_from_toml(Dict{String, Any}(
    "max_iter" => 200, "model_update" => Dict{String, Any}("type" => "keep_model", "tol" => 1.0e-10),
    "nuisance" => Dict{String, Any}("scale" => "estimated", "inlier_prior" => "estimated", "nu" => "consensus_dof"),
); section = "classification.em")

# =================================================================================================================
# SIMFIT++ (simfitpp_tests.jl)
# =================================================================================================================
# one true model (σ·|N(0,1)| on 60 %, uniform on the rest) and 30 decoys with uniform residuals
function eof_simfit_pool(seed; n = 4000, σ = 1.0, inlier_frac = 0.6, n_decoys = 30, outlier_span = 40.0)
    rng = StableRNG(seed)
    n_in = round(Int, inlier_frac * n)
    D = Matrix{Float32}(undef, n, n_decoys + 1)
    D[:, 1] = vcat(σ .* abs.(randn(rng, n_in)), outlier_span .* rand(rng, n - n_in))
    for j in 2:(n_decoys + 1)
        D[:, j] = outlier_span .* rand(rng, n)
    end
    return D
end
# the light record of a pool of identity models over zero correspondences (the pool frame IS the pixel frame)
function eof_identity_pool(D)
    I3 = SMatrix{3, 3, Float64, 9}(I)
    KC = SMatrix{3, 3, Float64, 9}(400.0, 0.0, 0.0, 0.0, 400.0, 0.0, 320.0, 240.0, 1.0)
    return VgtkCore.Pools.PoolPair{typeof(I3)}(;
        correspondences = [SVector(0.0, 0.0) => SVector(0.0, 0.0) for _ in 1:size(D, 1)], K1 = KC, K2 = KC,
        R_gt = I3, t_gt = SVector(1.0, 0.0, 0.0), models = [I3 for _ in 1:size(D, 2)], sample_indices = zeros(Int, 5, size(D, 2)),
        errors_r = fill(NaN, size(D, 2)), errors_t = fill(NaN, size(D, 2)),
        gt_inliers = falses(size(D, 1)), gt_variance = NaN, source_map = I3, destination_map = I3)
end
# a pool of 12 random fundamental models over 300 pool-frame correspondences, its frames similarities of scale s
function eof_fundmat_pool(seed; n = 300, K = 12, s = 1 / 650.0)
    rng = StableRNG(seed)
    sim(s, tx, ty) = SMatrix{3, 3, Float64, 9}(s, 0.0, 0.0, 0.0, s, 0.0, tx, ty, 1.0)
    Kc = SMatrix{3, 3, Float64, 9}(400.0, 0.0, 0.0, 0.0, 400.0, 0.0, 320.0, 240.0, 1.0)
    I3s = SMatrix{3, 3, Float64, 9}(I)
    cs = [SVector(2rand(rng) - 1, 2rand(rng) - 1) => SVector(2rand(rng) - 1, 2rand(rng) - 1) for _ in 1:n]
    models = [VgtkCore.FundMat(Tuple(SMatrix{3, 3, Float64, 9}(randn(rng, 9)...))) for _ in 1:K]
    return VgtkCore.Pools.PoolPair(; correspondences = cs, K1 = Kc, K2 = Kc, R_gt = I3s, t_gt = SVector(1.0, 0.0, 0.0),
                                   models, sample_indices = zeros(Int, 7, K), errors_r = fill(NaN, K), errors_t = fill(NaN, K),
                                   gt_inliers = falses(n), gt_variance = NaN, source_map = sim(s, -0.9, -0.7), destination_map = sim(s, -0.95, -0.65))
end
