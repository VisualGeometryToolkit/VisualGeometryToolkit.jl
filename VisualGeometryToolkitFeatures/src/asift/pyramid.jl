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

# THE map from an octave's level coordinates to the warp canvas's reported ones, the
# one spelling `_extract_at!` emits with and the continuous refinement
# (continuous_refine.jl) reports through: a 0-based level coordinate `x` of the octave
# of scale `sf = 2^octave` sits at canvas pixel `x·sf + (sf − 1)` (0-based; the 2×
# downsamples keep even samples), and a 0-based DoG slice coordinate `s` has the σ
# `seed_sigma·2^((s + ½)/S)·sf` (the geometric mean of the slice's two Gaussians).
# Kernel-lawful, in the coordinates' own type.
@inline _level_to_canvas(x, sf) = x * sf + (sf - one(sf))
@inline _slice_sigma(s, seed_sigma, S, sf) = seed_sigma * exp2((s + oftype(s, 0.5)) / oftype(s, S)) * sf

"""
    sift_nb_octaves(W, H; upsample=false) -> Int

Auto octave count: floor(log2(min(W,H))) − 4 (+1 if upsampling).
"""
sift_nb_octaves(W, H; upsample::Bool = false) =
    Int(floor(log2(min(W, H)))) - 4 + (upsample ? 1 : 0)

# ----- the Gaussian scale space ----------------------------------------------

"""
    GaussianScaleSpace

The octaves' Gaussian levels of an image, built by [`gaussian_scale_space`](@ref)
(and, one per tilted warp, inside every [`AsiftPlan`](@ref): ASIFT's DoG pyramid is
the difference of these levels).

- `levels[o]`: octave `o`'s `S + 3` levels, a `(W_o, H_o, S + 3, slots)` array on
  the device, `(x, y)` order (x along the first axis, as ASIFT holds images);
  level `l` (1-based) has the blur `seed_sigma·2^((l − 1)/S)` in the octave's
  pixels. Level `S + 1` of an octave is the next octave's level 1 before its 2×
  (Lowe's pixel-aligned) downsample.
- `dims[o]`: `(W_o, H_o)`; `first_octave`: `−1` when the first octave is the 2×
  upsampled image, else `0` (octave `o` has pixels of `2^(first_octave + o − 1)`
  input pixels);
- `nb_scales` (`S`), `seed_sigma`, the input's assumed blur `input_blur` (input pixels), and the per-level separable blur `taps`
  (one-sided, [`gaussian_taps`](@ref)) and their `radii`.

[`level_sigma`](@ref) gives a level's blur in input pixels.
"""
struct GaussianScaleSpace{A4, A1}
    levels::Vector{A4}
    dims::Vector{Tuple{Int, Int}}
    first_octave::Int
    nb_scales::Int
    seed_sigma::Float64
    input_blur::Float64
    taps::Vector{A1}
    radii::Vector{Int32}
end

"""
    level_sigma(space::GaussianScaleSpace, o, l) -> Float64

The Gaussian blur of level `l` (1-based, fractional allowed) of octave `o`
(1-based) of `space`, in INPUT pixels: `seed_sigma·2^((l − 1)/S)·2^(first_octave + o − 1)`.
"""
level_sigma(space::GaussianScaleSpace, o::Real, l::Real) =
    space.seed_sigma * exp2((l - 1) / space.nb_scales) * exp2(space.first_octave + o - 1)

# The pyramid's octave sizes: `nb_octaves` octaves of a W×H image, from octave −1 (2W×2H) when upsampling. `>>` with a
# negative shift is a LEFT shift in Julia, so octave −1 comes out at (2W, 2H) from the same expression.
_octave_dims(W, H, first_octave, nb_octaves) = [(W >> o, H >> o) for o in first_octave:(first_octave + nb_octaves - 1)]

# The per-level separable blur taps on `dev` and their radii: level 1 lifts the input's assumed blur (doubled at an
# upsampled first octave: its pixels are half an input pixel) to `seed_sigma`, each next level advances by 2^(1/S).
function _scale_space_taps(dev, seed_sigma, input_blur, S, upsample::Bool)
    seps = sift_sep_sigmas(seed_sigma, upsample ? 2 * input_blur : input_blur, S)
    radii = Int32[Int32(length(gaussian_taps(s)) - 1) for s in seps]
    maximum(radii) <= 13 || throw(
        ArgumentError(
            "seed_sigma = $seed_sigma needs a blur tap radius of $(maximum(radii)), beyond the " *
                "tiled blur's RMAX = 13 (with nb_scales = 3 and upsampling, seed_sigma must be ≲ 1.68)"
        )
    )
    return [ka_upload(Float32, dev, gaussian_taps(s)) for s in seps], radii
end

"""
    gaussian_scale_space(img; device = default_device(), nb_scales = 3, seed_sigma = 1.6,
                         input_blur = 0.5, upsample = true, nb_octaves = sift_nb_octaves(W, H; upsample))
        -> GaussianScaleSpace

The Gaussian scale space of the grey image `img` (H×W, read as
[`asift_detect`](@ref) reads it) on `device`: `nb_octaves` octaves of
`nb_scales + 3` levels each, by ASIFT's own pyramid (the same taps, the same
fused tiled blur kernel, Lowe's pixel-aligned 2× downsample and, with `upsample`,
the package's 2× upsampling of the first octave), level for level what
`asift_detect` builds for the untilted warp. See [`GaussianScaleSpace`](@ref).
"""
function gaussian_scale_space(
        img::AbstractMatrix; device::KA.Backend = default_device(), nb_scales::Integer = 3,
        seed_sigma::Real = 1.6, input_blur::Real = 0.5, upsample::Bool = true, nb_octaves::Union{Nothing, Integer} = nothing
    )
    inp = _asift_input(img)
    W, H = inp isa Matrix{UInt8} ? (size(inp, 2), size(inp, 1)) : size(inp)
    S = Int(nb_scales)
    first_octave = upsample ? -1 : 0
    no = something(nb_octaves, sift_nb_octaves(W, H; upsample))
    1 <= no <= sift_nb_octaves(W, H; upsample) || throw(ArgumentError(
        "gaussian_scale_space: a $(W)×$(H) image holds 1 … $(sift_nb_octaves(W, H; upsample)) octaves, got nb_octaves = $no"))
    dims = _octave_dims(W, H, first_octave, no)
    taps, radii = _scale_space_taps(device, seed_sigma, input_blur, S, upsample)
    levels = [KA.allocate(device, Float32, d[1], d[2], S + 3, 1) for d in dims]
    space = GaussianScaleSpace(levels, dims, first_octave, S, Float64(seed_sigma), Float64(input_blur), taps, radii)
    K = _scheduled(_asift_kernels, device)
    canvas = KA.allocate(device, Float32, W, H, 1)
    if inp isa Matrix{UInt8}
        K.u8xy(view(canvas, :, :, 1), on_device(UInt8, device, inp); ndrange = (W, H))
    else
        copyto!(canvas, on_device(Float32, device, inp))
    end
    # one slot, every tile active (no warp's empty margin to mask)
    lists = [_tile_lists(d..., Int32[d[1]], Int32[d[2]]) for d in dims]
    tiles = [ka_upload(Int32, device, t[1]) for t in lists]
    masked = [ka_upload(Int32, device, t[2]) for t in lists]
    seedtiles = ka_upload(Int32, device, _dilate_tiles(lists[1][1], dims[1]...))
    awd = [ka_upload(Int32, device, Int32[d[1]]) for d in dims]
    ahd = [ka_upload(Int32, device, Int32[d[2]]) for d in dims]
    _fill_scale_space!(K, space, nothing, canvas, tiles, masked, seedtiles, awd, ahd, 1:1)
    KA.synchronize(device)
    return space
end
