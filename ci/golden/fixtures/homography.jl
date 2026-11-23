# Fixtures of the homography family: the scenes of VisualGeometryToolkitEstimation's suite (fitting_pipelines_tests'
# h_scene, ransac_homography_tests' make_homography_data and its board, the prototype's larger sets), built from
# the shared synthetic scenes.
using Test, StableRNGs, StaticArrays, StructArrays, Random
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation

const _HS = VgtkEstimation.synthetic_scenes().HomographyScenes
_hpx_target(rng) = randn(rng, 2) .* 400 .+ 300

function homography_scenes()
    HS = _HS
    board = [SA[0.05 * i / 7, 0.05 * j / 7] for i in 0:7 for j in 0:7]
    warp(b) = (h = HS.H_PX * SA[b[1] * 1.0e4, b[2] * 1.0e4, 1.0]; SA[h[1] / h[3], h[2] / h[3]])
    rng = StableRNG(21)
    pix = [warp(b) + SA[0.2 * randn(rng), 0.2 * randn(rng)] for b in board]
    for k in 1:8
        pix[k] = pix[k] + SA[40.0 * randn(rng), 40.0 * randn(rng)]
    end
    return [
        "h_scene_120+60" => let (s, d) = HS.homography_pairs(StableRNG(11), HS.H_PX; n_in = 120, n_out = 60, σ = 0.3,
                    source = HS.gaussian_source, target = _hpx_target); StructArray(s .=> d) end,
        "htrue_100+30" => let (s, d) = HS.homography_pairs(StableRNG(42), HS.H_TRUE; n_in = 100, n_out = 30, σ = 0.5); StructArray(s .=> d) end,
        "htrue_40+10_corr" => let (s, d) = HS.homography_pairs(StableRNG(3), HS.H_TRUE; n_in = 40, n_out = 10, σ = 0.5); VgtkCore.correspondences(s, d) end,
        "hpx_1000+1000" => let (s, d) = HS.homography_pairs(StableRNG(7), HS.H_PX; n_in = 1000, n_out = 1000, σ = 1.0,
                    source = HS.gaussian_source, target = _hpx_target); StructArray(s .=> d) end,
        "pairs_300+100" => let (s, d) = HS.homography_pairs(StableRNG(3), HS.H_TRUE; n_in = 300, n_out = 100, σ = 0.8); s .=> d end,
        "board_exact_64" => VgtkCore.correspondences(VgtkCore.exact.(board), pix),
        "board_plain_64" => VgtkCore.correspondences(board, pix),
    ]
end

Base.include(@__MODULE__, joinpath(@__DIR__, "common.jl"))   # the shared sets (`homography_too_few`)
