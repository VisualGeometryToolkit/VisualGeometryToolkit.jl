# =============================================================================
# The detection image of a RAW: variance-stabilized, noise σ = 1 everywhere
# =============================================================================
#
# A green-cell value is the mean of two photosites, each `(raw − black) / (white
# − black) × gain` with `raw − black` = electrons / K (K electrons per code) plus
# Gaussian read noise of σ_r codes. So the value is `α·P + N`, P Poisson and N
# Gaussian, with
#
#     α  = gain / (4K) · (1/d₁ + 1/d₂),     σ² = gain² σ_r² / 4 · (1/d₁² + 1/d₂²),
#
# d_k = white − black[c_k] the two greens' code spans (`L.geometry.green_dn`).
# The generalized Anscombe transform (Starck, Murtagh and Bijaoui 1998; Mäkitalo
# and Foi 2013)
#
#     f(x) = (2/α) √(max(α x + 3α²/8 + σ², 0))
#
# makes the noise σ ≈ 1 at every level. ASIFT on f's image then thresholds its
# DoG at k noise σ: the same contrast in noise units in a dim interior as in a
# bright window, where a tone curve on [0, 1] and a fixed threshold see only the
# window.
# =============================================================================

"""
    StabilizedImage(image, saturated, white)

A detection image whose noise has σ = 1 at every pixel: `image` (a
`Matrix{Float64}`, in noise σ), the `saturated` pixels (a `BitMatrix`, masked:
[`asift_detect`](@ref) returns no detection over them) and `white`, the image's
value at the white level (the scale that maps it into `[0, 1]`). An `image` that
is a `Matrix{Float64}` and a `saturated` that is a `BitMatrix` are held, not
copied. [`variance_stabilize`](@ref) makes one of a RAW's green cells.
"""
struct StabilizedImage
    image::Matrix{Float64}
    saturated::BitMatrix
    white::Float64
    function StabilizedImage(image::AbstractMatrix{<:Real}, saturated::AbstractMatrix{Bool}, white::Real)
        size(image) == size(saturated) ||
            throw(DimensionMismatch("StabilizedImage: the image is $(size(image)), the saturation mask $(size(saturated))"))
        isfinite(white) && white > 0 || throw(ArgumentError("StabilizedImage: the white level must be positive and finite, got $white"))
        return new(convert(Matrix{Float64}, image), convert(BitMatrix, saturated), Float64(white))
    end
end

"""
    variance_stabilize(L::LinearRaw; electrons_per_dn, read_noise) -> StabilizedImage

The generalized Anscombe transform of the green cells of `L` (`:green_cells`,
or `:bin2`'s G plane, the same values) for Poisson–Gaussian noise: after it the
noise has σ ≈ 1 at every level, from a few electrons to the white level.

- `electrons_per_dn` — the sensor's conversion gain at the frame's ISO,
  electrons per raw code (> 0).
- `read_noise` — the read noise in raw codes (≥ 0), the codes' rounding
  included.

Neither is in a RAW file's metadata (LibRaw reports neither): they are the
camera's, at the frame's ISO, measured once (e.g. from the variance of the two
greens' difference against their mean on a flat scene). The code spans are the
photosites' own (`L.geometry.green_dn`, each green's `white − black`), and `L.gain`
scales the photon and the read noise alike. A green cell whose value is
`x = (x₁ + x₂)/2` is `α·P + N` with
`α = gain/(4K)·(1/d₁ + 1/d₂)`, `σ² = gain²σ_r²/4·(1/d₁² + 1/d₂²)`, and its
transform is `(2/α)·√(max(α x + 3α²/8 + σ², 0))`.

`L.saturated` is carried over: [`asift_detect`](@ref) masks those pixels
rather than reading the clipped values. The transform is pixel-wise, so the
detections' positions are the cell grid's, as on any rendering of `L` (a
display rendering for a descriptor among them).
"""
function variance_stabilize(L::LinearRaw; electrons_per_dn::Real, read_noise::Real)
    isfinite(electrons_per_dn) && electrons_per_dn > 0 ||
        throw(ArgumentError("variance_stabilize: electrons_per_dn must be positive and finite, got $electrons_per_dn"))
    isfinite(read_noise) && read_noise >= 0 ||
        throw(ArgumentError("variance_stabilize: the read noise must be nonnegative and finite, got $read_noise"))
    # the green-cell plane: the one whose frame IS the cell grid (Core's raw_frame_map; the display image refused there)
    S = raw_frame_map(L)
    G = view(L.image, :, :, findfirst(c -> bayer_plane_map(L, c) === S, 1:size(L.image, 3)))
    d1, d2 = L.geometry.green_dn
    g = L.gain
    α = g / (4 * electrons_per_dn) * (1 / d1 + 1 / d2)
    σ² = g^2 * read_noise^2 / 4 * (1 / d1^2 + 1 / d2^2)
    f(x) = 2 / α * sqrt(max(α * x + 3α^2 / 8 + σ², 0.0))
    return StabilizedImage(f.(G), L.saturated, f(g))
end

# The `intensity_thr` of `asift_detect` that is one noise σ of unit iid input noise in its noisiest searched DoG slice
# (the finest octave's, at its noisiest sample phase when that octave is upsampled; the slices the extremum search
# reads, the interior ones 2 … NS − 1 — a boundary slice is no candidate's): the DoG is linear, so a slice's noise
# σ at a sample is the root energy of its impulse response there, read off the pyramid of a unit impulse through a
# plan of the same seed blur and upsampling (the untilted view: a tilt's anti-aliasing only lowers it). The driver
# compares the DoG with `intensity_thr / nb_scales`, hence the factor.
const _NOISE_THR = Dict{Tuple{Float64, Bool}, Float64}()
const _NOISE_THR_LOCK = ReentrantLock()

function _noise_intensity_thr(seed_sigma::Real, upsample::Bool)
    return lock(_NOISE_THR_LOCK) do
        get!(_NOISE_THR, (Float64(seed_sigma), upsample)) do
            n = 64
            plan = AsiftPlan(n, n; device = KA.CPU(), tilts = :untilted, seed_sigma, upsample)
            impulse = zeros(Float32, n, n); impulse[n ÷ 2, n ÷ 2] = 1
            asift_detect!(plan, impulse; intensity_thr = Inf)
            dog = Array(plan.dogs[1])
            f = plan.o_first < 0 ? 2 : 1                       # an upsampled octave: one impulse response per sample phase
            σ = maximum(sqrt(sum(abs2, view(dog, px:f:size(dog, 1), py:f:size(dog, 2), sl, 1)))
                        for px in 1:f, py in 1:f, sl in 2:(size(dog, 3) - 1))
            plan.nb_scales * Float64(σ)
        end
    end
end

"""
    asift_detect(s::StabilizedImage; noise_sigmas = 3.0, kwargs...)

ASIFT on a variance-stabilized image ([`variance_stabilize`](@ref)), its DoG
contrast threshold `noise_sigmas` noise σ: on unit iid noise, the DoG's noise
σ in its noisiest searched slice (the finest octave's finest interior slice, the
extremum search's first) is derived from the pyramid's
own impulse response, and an extremum must exceed `noise_sigmas` times it. Every
coarser slice, and every tilted view, has less noise, so is held to more than
`noise_sigmas` σ of its own. The default 3 is Lowe's three-sigma reading of a
contrast: on pure noise, extrema above it are rare.

The saturated pixels are the `exclusion_mask` ([`asift_detect`](@ref)'s rule: a
detection whose centre lies within three of its scales, the DoG's support, of a
masked pixel is dropped): a clipped value is no measurement. A caller's `exclusion_mask`
joins them. The image is scaled by `1 / s.white` into `[0, 1]` for the
detector (the warps' fill is mid-grey), the threshold with it. `kwargs` are
[`asift_detect`](@ref)'s but `intensity_thr` (the threshold is `noise_sigmas`).
"""
function asift_detect(s::StabilizedImage; noise_sigmas::Real = 3.0, seed_sigma::Real = 1.6, upsample::Bool = true,
                      exclusion_mask::Union{Nothing, AbstractMatrix{Bool}} = nothing, kwargs...)
    _refuse_retired_mask(kwargs)
    haskey(kwargs, :intensity_thr) && throw(ArgumentError(
        "asift_detect: a StabilizedImage's threshold is `noise_sigmas` (noise σ), not `intensity_thr`"))
    isfinite(noise_sigmas) && noise_sigmas > 0 ||
        throw(ArgumentError("asift_detect: noise_sigmas must be positive and finite, got $noise_sigmas"))
    _check_mask(exclusion_mask, size(s.image))
    thr = noise_sigmas * _noise_intensity_thr(seed_sigma, upsample) / s.white
    return asift_detect(Float32.(s.image ./ s.white); intensity_thr = thr, seed_sigma, upsample,
                        exclusion_mask = exclusion_mask === nothing ? s.saturated : exclusion_mask .| s.saturated, kwargs...)
end
