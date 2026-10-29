# =============================================================================
# DoG scale-space numerics — Gaussian taps, pyramid border rule, octave count.
# =============================================================================
#
# Shared by the fused pyramid kernels in driver.jl (`_btile_blur_dog!`,
# `_bdown!`), which build the scale space (Gaussian blur, 2× downsample,
# difference of Gaussians). The kernel numerics:
#
#   • n_levels_gauss = nb_scales + 3   (Gaussian levels per octave)
#   • seed  sep_sigma = √(seed_sigma² − input_blur²)
#   • i≥1   sep_sigma = √((2^(i/S)·seed_sigma)² − (2^((i−1)/S)·seed_sigma)²)
#   • kernel radius r = ceil(4·sep_sigma); k[0]=1, k[i]=exp(−i²/2σ²);
#     normalized by k[0] + 2·Σ_{i≥1} k[i]
#   • separable: horizontal sweep then vertical, MIRRORED_REPEAT borders
#     (`_mirror_idx`)
#   • Downsample2x: Lowe pixel-aligned  dst(i,j) = src(2i+1, 2j+1)  (0-indexed)
#   • DoG sign: DoG[s] = G[s+1] − G[s]
#
# Images are stored (W,H) indexed [x,y] (x-fastest).
# =============================================================================

# ----- Gaussian kernel construction (Float32, matches setupGaussianKernels) --

"""
    sift_sep_sigmas(seed_sigma, input_blur, nb_scales) -> Vector{Float32}

Per-level separable-blur σ for one octave (nb_scales+3 levels). Level 0 lifts
the assumed `input_blur` up to `seed_sigma`; each subsequent level advances by a
factor 2^(1/nb_scales) in absolute scale.
"""
function sift_sep_sigmas(seed_sigma::Real, input_blur::Real, nb_scales::Integer)
    S = Int(nb_scales)
    σ0 = Float32(seed_sigma)
    seps = Vector{Float32}(undef, S + 3)
    seps[1] = sqrt(Float32(seed_sigma)^2 - Float32(input_blur)^2)
    for i in 1:(S + 2)                     # levels 1..S+2 (0-indexed)
        sig_prev = 2.0f0^(Float32(i - 1) / S) * σ0
        sig_tot = 2.0f0^(Float32(i) / S) * σ0
        seps[i + 1] = sqrt(sig_tot^2 - sig_prev^2)
    end
    return seps
end

# One-sided Gaussian taps `[k0, k1, …, kr]` normalized so k0 + 2·Σ kᵢ = 1. The
# taps are the package's one builder's (`_gaussian_profile`, linear_image.jl);
# the pyramid blur (`gaussian_taps`) and the σ_aa pre-blur (`sigma_aa_taps`,
# warp.jl) differ only in the radius rule.
_unit_sum_taps(k::Vector{Float32}) = k ./= (k[1] + 2.0f0 * sum(@view k[2:end]))

"""
    gaussian_taps(sep_sigma) -> Vector{Float32}

One-sided normalized Gaussian taps `[k0, k1, …, kr]` (center + r taps), with
r = max(1, ceil(4·sep_sigma)). Normalized so k0 + 2·Σ kᵢ = 1.
"""
gaussian_taps(sep_sigma::Real) = _unit_sum_taps(_gaussian_profile(Float32(sep_sigma), 4; min_radius = 1))

# ----- pyramid border rule ---------------------------------------------------

# Border rule for the PYRAMID blur: mirrored repeat,
# i.e. reflect-with-edge-duplication: …, 2, 1 | 1, 2, … (1-indexed).
# NOT clamp-to-edge — clamp matches only on constant borders (which is why
# synthetic tests never caught the difference). The σ_aa pre-blur is different:
# the fused path's GaussBlur1DStorage does explicit clamp — keep clamp there.
# Closed-form periodic reflect (period 2N). The previous single-fold form
# (`i < 1 ? 1 - i : i > N ? 2N + 1 - i : i`) returns OUT-OF-RANGE indices
# when the overhang exceeds N — reachable in `_btile_blur_dog!`, whose 58-row
# tile halo spans source coords up to tile_origin + 45: any pyramid level
# with a dimension in (32, 38] (e.g. 1080p octave 5, H = 33 → coord 71 >
# 2H = 66 → index −5) faulted under --check-bounds=yes and read garbage
# otherwise. This form is bit-identical to the single fold on its valid band
# and folds arbitrarily far overhangs correctly.
@inline function _mirror_idx(i, N)
    m = mod(i - 1, 2N)      # 0 .. 2N-1
    return m < N ? m + 1 : 2N - m  # 1 .. N, reflect-with-edge-duplication
end

# ----- octave geometry -------------------------------------------------------

"""
    sift_nb_octaves(W, H; upsample=false) -> Int

Auto octave count: floor(log2(min(W,H))) − 4 (+1 if upsampling).
"""
sift_nb_octaves(W, H; upsample::Bool = false) =
    Int(floor(log2(min(W, H)))) - 4 + (upsample ? 1 : 0)
