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
