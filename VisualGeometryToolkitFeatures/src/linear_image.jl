# =============================================================================
# Linear-radiance image input — a PNG as the numbers it stores
# =============================================================================
#
# A radiometric consumer (a renderer's albedo map, a photograph used as a
# radiance target) needs the samples of a PNG as LINEAR values, never as a
# display re-encoding of them. Two traps, each handled once here:
#
#   gAMA / sRGB chunks — PNGFiles by default re-encodes the samples for a 2.2
#     screen when the file carries one, which brightens every mid-tone of a
#     file saved with `file_gamma = 1` (BlobBoards' boards) relative to a
#     renderer that reads the same file raw. The file is read with
#     `gamma = 1.0`: the stored samples.
#   display-encoded photographs — a phone or camera 8-bit file is sRGB
#     encoded; `srgb = true` decodes the IEC 61966-2-1 transfer curve
#     (`srgb_to_linear`) BEFORE any averaging, so a box downsample averages
#     radiance, not code values.
#
# Hoisted from RadiometricCamera (`load_texture`, `load_photo`,
# `srgb_to_linear`, `downsample`), 2026-10-27. Lives here rather than in
# VisualGeometryToolkitIO: IO's kinds are serializations of VGC types (a PNG is
# not one, and IO carries no image stack), while this package owns the image
# input ladders (image_input.jl) and already depends on ImageCore.
# =============================================================================

"""
    srgb_to_linear(x) -> linear value

The sRGB (IEC 61966-2-1) electro-optical transfer function: an encoded value
`x ∈ [0, 1]` to linear, `x / 12.92` for `x ≤ 0.04045`, otherwise
`((x + 0.055) / 1.055)^2.4`.
"""
@inline srgb_to_linear(x::Real) = x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055)^2.4

"""
    box_downsample(img::AbstractArray{<:Real, 3}, f::Integer) -> Array{Float64, 3}

The `H × W × C` image averaged over `f × f` blocks, per channel: output pixel
`(i, j)` is the mean of input rows `(i − 1)f + 1 : if` and columns
`(j − 1)f + 1 : jf`. The output is `H ÷ f × W ÷ f × C`; rows and columns past
the last whole block are dropped. `f = 1` returns a `Float64` copy.
"""
function box_downsample(img::AbstractArray{<:Real, 3}, f::Integer)
    f >= 1 || throw(ArgumentError("box_downsample: the factor must be a positive integer, got $f"))
    H, W, C = size(img)
    out = zeros(Float64, H ÷ f, W ÷ f, C)
    inv_n = 1 / (f * f)
    for c in 1:C, j in 1:(W ÷ f), i in 1:(H ÷ f)
        s = 0.0
        for jj in ((j - 1) * f + 1):(j * f), ii in ((i - 1) * f + 1):(i * f)
            s += img[ii, jj, c]
        end
        out[i, j, c] = s * inv_n
    end
    return out
end

"""
    load_linear_image(path; srgb = false, downsample = 1) -> Array{Float64, 3}

A PNG file as an `H × W × 3` `Float64` array of linear RGB values in `[0, 1]`:
the samples AS STORED (read with `gamma = 1`, so a `gAMA`/`sRGB` chunk never
re-encodes them; gray files are replicated to three channels, an alpha
channel is dropped, 8- and 16-bit depths both normalized to `[0, 1]`).

- `srgb = true`: the file is display-encoded (a phone or camera photograph);
  [`srgb_to_linear`](@ref) decodes every sample.
- `downsample = f`: the decoded image averaged over `f × f` blocks
  ([`box_downsample`](@ref)), in linear space.

The array is indexed `[row, column, channel]`; pixel `(row, column)` is the
image's `PixelFrame{:julia}` pixel `(column, row)`.
"""
function load_linear_image(path::AbstractString; srgb::Bool = false, downsample::Integer = 1)
    raw = PNGFiles.load(String(path); gamma = 1.0)
    img = permutedims(Float64.(channelview(RGB.(raw))), (2, 3, 1))     # 3 × H × W → H × W × 3
    srgb && (img .= srgb_to_linear.(img))
    return downsample == 1 ? img : box_downsample(img, downsample)
end

"""
    gaussian_blur(img::AbstractArray{<:Real, 3}, σ[, mask]) -> Array{Float64, 3}

The `H × W × C` image convolved per channel with the separable Gaussian of
standard deviation `σ` pixels, truncated at `⌈3σ⌉` pixels and normalized to
unit sum; at the image border the kernel is renormalized over the pixels
inside it (no padding value enters). The taps are the package's one Gaussian
taps builder's, at truncation 3 — not ASIFT's pyramid's 4σ with a mirrored
border: the vignetting chain this smooths for was reproduced with this
truncation and border rule, so they stay as they are.

With a `mask` (an `H × W` array of weights, `0` outside the support — a
fisheye's image circle) it is the NORMALIZED convolution
`blur(img·mask) / blur(mask)`: pixels outside the mask do not bleed in, and a
pixel where the mask is zero is `0`. `σ = 0` returns a `Float64` copy (masked
to zero outside the mask). Edge-safe smoothing of a linear image before it is
sampled (the rooftop vignetting chain's cell samples).
"""
function gaussian_blur(img::AbstractArray{<:Real, 3}, σ::Real, mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing)
    σ >= 0 || throw(ArgumentError("gaussian_blur: σ must be non-negative, got $σ"))
    H, W, C = size(img)
    mask === nothing || size(mask) == (H, W) ||
        throw(DimensionMismatch("gaussian_blur: an $(H)×$(W) image and a $(join(size(mask), '×')) mask"))
    m = mask === nothing ? ones(H, W) : Float64.(mask)
    out = zeros(Float64, H, W, C)
    if iszero(σ)
        for c in 1:C
            out[:, :, c] .= ifelse.(m .> 0, img[:, :, c], 0.0)
        end
        return out
    end
    g = _gaussian_profile(σ, 3)                    # [g(0), …, g(r)], r = ⌈3σ⌉
    k = Vector{Float64}([reverse(@view g[2:end]); g])
    r = length(g) - 1
    den = _separable_blur(m, k, r)
    for c in 1:C
        num = _separable_blur(Float64.(view(img, :, :, c)) .* m, k, r)
        out[:, :, c] .= ifelse.(den .> 0, num ./ den, 0.0)
        mask === nothing || (out[:, :, c] .*= (m .> 0))
    end
    return out
end

# THE Gaussian taps builder of the package: the one-sided profile
# `[g(0), g(1), …, g(r)]`, `g(i) = exp(−i²/(2σ²))`, unnormalized, at the
# radius `r = ⌈truncation·σ⌉` clamped to `min_radius:max_radius`, in `σ`'s own
# arithmetic (a `Float32` σ gives `Float32` taps). `gaussian_blur` takes it at
# truncation 3 and renormalizes per pixel; ASIFT's pyramid (`gaussian_taps`) at
# truncation 4, at least one tap, and its σ_aa pre-blur (`sigma_aa_taps`) at 4
# capped at 16 taps, both normalized to unit sum (`_unit_sum_taps`).
function _gaussian_profile(σ::Real, truncation::Real; min_radius::Integer = 0, max_radius::Integer = typemax(Int))
    r = clamp(ceil(Int, truncation * σ), min_radius, max_radius)
    g = [exp(-i^2 / (2σ^2)) for i in 0:r]
    g[1] = 1                                       # the centre, exactly (σ = 0 included)
    return g
end

# The separable convolution of `a` with the kernel `k` (half-width r) along both
# axes, each axis renormalized by the kernel mass inside the image.
function _separable_blur(a::AbstractMatrix{Float64}, k::Vector{Float64}, r::Int)
    H, W = size(a)
    tmp = similar(a); out = similar(a)
    @inbounds for j in 1:W, i in 1:H
        s = 0.0; w = 0.0
        for t in max(-r, 1 - i):min(r, H - i)
            s += k[t + r + 1] * a[i + t, j]; w += k[t + r + 1]
        end
        tmp[i, j] = s / w
    end
    @inbounds for j in 1:W, i in 1:H
        s = 0.0; w = 0.0
        for t in max(-r, 1 - j):min(r, W - j)
            s += k[t + r + 1] * tmp[i, j + t]; w += k[t + r + 1]
        end
        out[i, j] = s / w
    end
    return out
end
