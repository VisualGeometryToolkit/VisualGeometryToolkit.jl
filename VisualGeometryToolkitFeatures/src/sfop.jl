# =============================================================================
# SFOP scale selection (Förstner, Dickscheid & Schindler 2009) on the package's
# Gaussian scale space, α fixed at 90°: the circular (dot) model.
#
# Per level of every octave of `gaussian_scale_space`: the gradient at the
# differentiation scale τ (`gaussian_gradient` on the level), the squared
# gradient, its Gaussian average at the integration scale σ = 3τ (the structure
# tensor M̄, λ₂ its smaller eigenvalue) and the circular model's residual Ω̄
# about the window centre; the precision w = (N(σ) − 2)·λ₂/Ω̄, N(σ) = 12σ² + 1.
# Its 3-D maxima (ASIFT's extraction, maxima only) with sub-level interpolation,
# the λ₂ significance test against the image noise, then ASIFT's NMS.
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const

# The paper's ratio of the integration to the differentiation scale, σ = 3τ (its Algorithm 1, k = 3).
const _SFOP_SCALE_RATIO = 3

# One row of `detect_sfop`'s result: a disc (the selected integration scale σ, input pixels) with its precision.
const _SfopDetection = Attributed{Disc{Float64}, @NamedTuple{precision::Float64}}

"""
    detect_sfop(img; device = default_device(), noise_sigma = 2.0, significance_level = 0.999,
                precision_threshold = 0.5, nb_scales = 3, input_blur = 0.5, upsample = true,
                nb_octaves = min(3, sift_nb_octaves(size(img)...; upsample)),
                seed_sigma = (upsample ? 2 : 1) * input_blur, polarity = :none) -> StructArray{<:Detection}
    detect_sfop(L::LinearRaw; sensor_model::SensorModel = SensorModel(), kwargs...)

SFOP scale selection [Foerstner2009](@cite) for CIRCULAR points (the spiral
model at α = 90°): each point where the precision `w` of a circular point's
estimate peaks in position and scale, with the integration scale `σ` it peaks
at. Rows `Detection(Disc(x, σ), (; precision))`, COLMAP pixel centres of the
input (of a RAW's green cells, `L::LinearRaw`: the cell grid, their values) and σ in input pixels; `precision` is `w` (px⁻²). Seed
[`detect_foerstner(img, seeds)`](@ref detect_foerstner) with them to localise
each point at full resolution in its own window.

- **The scale space:** [`gaussian_scale_space`](@ref)`(img; nb_octaves, nb_scales,
  seed_sigma, input_blur, upsample)`, ASIFT's pyramid, 3 octaves by default or as many as the image holds
  ([`gaussian_scale_space`](@ref)'s `sift_nb_octaves`) when that is fewer (a `nb_octaves` beyond them is refused). Its first level is the
  input itself (`seed_sigma` the input's own blur in the first octave's pixels:
  no seed blur), so the finest scale searched is the finest the image holds:
  with the defaults the integration scales run from `3·2^(2/3)·0.5 ≈ 2.4` input
  pixels (the second level of the 2× upsampled octave) up. Level `l` of octave `o`
  (blur `σₗ`, octave pixels) gives the differentiation scale `τ = σₗ₊₁`, the
  next level's blur: [`gaussian_gradient`](@ref) on the level at the pyramid's
  own blur increment `√(σₗ₊₁² − σₗ²)`, so every octave's levels sit on one ladder of
  scales and an octave's last levels are the next one's first; the
  integration scale is `σ = 3τ`.
- **The precision:** `w = (N(σ) − 2)·λ₂(M̄)/Ω̄`, `N(σ) = 12σ² + 1` (σ in the
  octave's pixels, so `w` is in px⁻² of the input at every octave), `M̄` the
  structure tensor (the squared gradient's average under the unit-sum Gaussian
  `G_σ`), `Ω̄ = Σ G_σ(d)·(dᵀ ∇g⊥)²` the circular model's weighted squared distances
  about the window centre (eq. 5 of the paper at α = 90°).
- **Maxima:** of `w` over the 26 neighbours in (x, y, level), levels 2 … S + 2 of
  every octave (adjacent octaves share a level), refined by the quadratic
  interpolation of ASIFT's extraction on `−1/w` (sub-pixel, sub-level; the
  reference point's misfit is quadratic in its offset), or kept on its pixel and
  level where that interpolation fails (where clutter enters the window `w`
  steps); `precision` is `w` at the sample.
- **Significance:** `λ₂ > T`, `T = ½E[tr M̄]·χ²_ν(significance_level)/ν` for
  white noise of `noise_sigma` grey levels of an 8-bit image: `E[tr M̄]` and the
  degrees of freedom `ν = 2E²/Var` (Satterthwaite) of the Gaussian-weighted sum
  of the correlated squared gradients, both from the gradient field's exact
  covariance under the level's blur. No empirical factor: the paper's 1.5 was the
  correlation the independent-pixel count `2N(σ)` ignored. `λ₂ ≤ tr/2`, so the test
  is conservative.
- **Precision:** a maximum is kept when its point standard deviation `1/√w` is at
  most `precision_threshold` (input pixels, [`detect_foerstner`](@ref)'s
  keyword and default): the paper's `T_w`.
- **Accuracy:** seeded [`detect_foerstner`](@ref) at the selected σ equals the best
  fixed window within Monte Carlo error at dot radii 0.9–5 px, except near
  1.8 px, where it is 0.0008 px RMS worse (0.0092 vs 0.0084 px): a measured,
  accepted limit of the window estimator's pixel locking (no window rule tested
  removed it).
- **Suppression:** ASIFT's elliptical NMS ([`cross_warp_nms`](@ref)), the
  paper's eq. (8) with its defaults (fp = 1, fs = ½, T_r = 1) as footprint
  overlap: a footprint of radius `σ/√2` and half-octave scale bands, ± 1 band.
  No further deduplication: one blob found at two scales more than a band apart (the
  NMS keeps both) gives two rows, which the consumer resolves by its own geometry.
- **Polarity** (`polarity = :none | :dark | :light`, ASIFT's vocabulary and sign):
  `:dark` keeps the dark blobs, `:light` the light ones, by the sign of the
  scale-normalised Laplacian of Gaussian `σ²∇²(G_σ ⋆ I)` at each candidate's selected
  σ (positive at a dark blob, ASIFT's DoG sign; [`is_dark`](@ref)), tested one-sided at
  `significance_level` under the λ₂ test's noise: a candidate whose sign the noise could
  have given (a window on flat background between dots) is kept for neither. Applied
  before the NMS (ASIFT's order: a blob of the other sign never suppresses a kept one).
  `:none` (the default) applies no gate: the structure tensor is sign-invariant and
  bright and dark blobs alike are found.
- **Noise:** `noise_sigma` (grey levels of an 8-bit image) is one level for the whole
  image. On a RAW's green cells under a gain-carrying `sensor_model` (the frame's, the one
  `L` was linearised with, so `L` carries each value's variance) the λ₂ and polarity
  tests read each pixel's variance instead (`noise_sigma` unread): the floor is linear in
  the noise variance, and each window's is the variance averaged under `G_σ`. Shot noise
  in a bright region then raises no blob. Without a gain, `noise_sigma` as for a matrix.
"""
detect_sfop(img::AbstractMatrix; kw...) = _detect_sfop(img, nothing; kw...)

# A RAW's green cells: SFOP on their values (Core's `green_plane`), positions on the cell grid; under a gain-carrying
# sensor model (the image linearised with it carries each value's variance) the λ₂ test reads each pixel's variance,
# else `noise_sigma` as for a matrix — the pixel noise `detect_foerstner` propagates on the same image (`_raw_inputs`).
detect_sfop(L::LinearRaw; sensor_model::SensorModel = SensorModel(), kw...) =
    ((img, model) = _raw_inputs(L, sensor_model); _detect_sfop(img, model === nothing ? nothing : first(model); kw...))

# `variance`: each pixel's noise variance (the [0, 1] image, a matrix of its size), or `nothing` for `noise_sigma`'s.
function _detect_sfop(
        img::AbstractMatrix, variance; device::KA.Backend = default_device(),
        noise_sigma::Real = 2.0, significance_level::Real = 0.999, precision_threshold::Real = 0.5,
        nb_scales::Integer = 3, input_blur::Real = 0.5, upsample::Bool = true,
        nb_octaves::Integer = min(3, sift_nb_octaves(size(img)...; upsample)), seed_sigma::Real = (upsample ? 2 : 1) * input_blur,
        polarity::Symbol = :none
    )
    0 < significance_level < 1 ||
        throw(ArgumentError("detect_sfop: significance_level must lie in (0, 1), got $significance_level"))
    isfinite(precision_threshold) && precision_threshold > 0 ||
        throw(ArgumentError("detect_sfop: precision_threshold must be positive and finite, got $precision_threshold"))
    isfinite(noise_sigma) && noise_sigma > 0 ||
        throw(ArgumentError("detect_sfop: noise_sigma must be positive and finite, got $noise_sigma"))
    sign = _polarity_sign(polarity)
    space = gaussian_scale_space(img; device, nb_scales, seed_sigma, input_blur, upsample, nb_octaves)
    noise = variance === nothing ? noise_sigma / 255 : ka_upload(Float32, device, variance)
    # the polarity gate's image (H × W, Float32) and the gate on the candidates' COLMAP centres and σ,
    # its sign test under the λ₂ test's noise and level
    gate = sign == 0 ? nothing : let g = _detection_image(Float32, device, img), v = noise isa Real ? noise^2 : noise
        points -> _polarity_kept(device, g, points, sign, v, significance_level)
    end
    W, H = space.dims[1] .>> (space.first_octave < 0 ? 1 : 0)
    K = _scheduled(_asift_kernels, device)
    S = space.nb_scales
    capacity = 1 << 16
    while true
        pool = KA.allocate(device, Float32, 5, capacity)
        count = KA.zeros(device, Int32, 2)
        ends = Int[]                                        # the pool's last row of each octave
        for o in eachindex(space.levels)
            precision = _sfop_precision(device, space, o, noise, significance_level)
            _sfop_maxima!(pool, count, device, precision, Float32(space.seed_sigma), Int32(o - 1 + space.first_octave), Int32(S); kernels = K)
            push!(ends, Int(Array(count)[1]))
        end
        n = last(ends)
        n > capacity && (capacity = n; continue)           # the pool overflowed: once more, large enough
        return _sfop_detections(device, space, Array(view(pool, :, 1:n)), ends, W, H, 1 / precision_threshold^2; gate)
    end
end

# The maxima of one octave's precision cube `cube` ((W_o, H_o, S + 3, 1), −1/w: `_sfop_precision`) appended to the
# pool `pool` (5 × capacity) by ASIFT's extraction in its precision mode (maxima only, no edge test, an unrefined
# maximum kept), `count[1]` its rows (past the capacity when it overflowed). `kernels`: ASIFT's, scheduled once per call.
function _sfop_maxima!(pool, count, device, cube, seed_sigma::Float32, octave_idx::Int32, S::Int32;
                       kernels = _scheduled(_asift_kernels, device))
    Wo, Ho = size(cube, 1), size(cube, 2)
    t, _ = _tile_lists(Wo, Ho, Int32[Wo], Int32[Ho])
    kernels.ext(
        pool, count, cube, 0.0f0, floatmax(Float32), seed_sigma, octave_idx, S, Int32(size(pool, 2)),   # the edge threshold: unread here
        ka_upload(Int32, device, Int32[Wo]), ka_upload(Int32, device, Int32[Ho]), nothing,
        ka_upload(Int32, device, t), Val(Int(S) + 3), Val(_TILE_NY), Val(true);
        ndrange = (32, _TILE_NY * (length(t) ÷ 3))
    )
    KA.synchronize(device)
    return pool
end

# The extracted maxima (`raw`, ASIFT's 5 × n rows: 0-based input x, y, the slice σ, w, slot; octave o's rows end at
# `ends[o]`) as detections: the level's blur from the slice σ (`_slice_sigma` reports the geometric mean of a DoG
# slice's two levels, so the level is half a slice lower), τ the next level's, σ = 3τ in input pixels, and ASIFT's
# NMS over them (eq. 8 as footprint overlap, see `detect_sfop`).
function _sfop_detections(device, space, raw, ends, W, H, min_precision; gate = nothing)
    S = space.nb_scales
    feats = ImasFeatureKA[]
    # each detection's own w, keyed by its octave too: adjacent octaves share a level, so two octaves can find a peak
    # at one position and scale, with different w (the NMS keeps one of a double; the octave rides in its slot)
    precision = Dict{Tuple{Float32, Float32, Float32, Int32}, Float64}()
    w = -1 ./ view(raw, 4, :)                               # the cube holds −1/w
    wmax = maximum(w; init = 0.0f0)
    for o in eachindex(ends), i in (o == 1 ? 1 : ends[o - 1] + 1):ends[o]
        w[i] >= min_precision || continue
        τ = raw[3, i] * exp2(0.5 / S)                       # slice σ·2^(½/S): the next level's blur, input pixels
        σ = Float32(_SFOP_SCALE_RATIO * τ)
        x, y = raw[1, i], raw[2, i]
        push!(feats, ImasFeatureKA(x, y, σ, 0.5f0 * w[i] / wmax, σ^2, 0.0f0, σ^2, Int32(o)))
        precision[(x, y, σ, Int32(o))] = w[i]
    end
    # the polarity gate BEFORE the NMS (ASIFT's order): a blob of the other sign never suppresses a kept one
    gate === nothing || isempty(feats) || (feats = feats[gate(stack(Float64[f.x + 0.5, f.y + 0.5, f.σ] for f in feats; dims = 2))])
    kept = isempty(feats) ? feats : cross_warp_nms(
        feats, W, H; device, mode = :ellipse, cutoff = 1 / sqrt(2), band_octaves = 1.0, band_reach = 1
    )
    rows = [
        _SfopDetection(Disc(Point2{Float64}(f.x + 0.5, f.y + 0.5), Float64(f.σ)), (; precision = precision[(f.x, f.y, f.σ, f.warp_idx)]))
            for f in kept
    ]
    return StructArray(collect(_SfopDetection, rows); unwrap = T -> T <: NamedTuple)
end

# The precision cube of octave o, (W_o, H_o, S + 3, 1): −1/w at every level (its maxima are w's), −∞ where λ₂ fails
# the significance test or the windows reach past the octave's border. The maxima are searched on levels 2 … S + 2, so each octave's range
# overlaps the next one's by a level: a peak between an octave's last levels is seen on one grid even where the two
# grids order the shared level differently (the NMS keeps one of a double).
function _sfop_precision(device, space::GaussianScaleSpace, o, noise, significance_level)
    S = space.nb_scales
    Wo, Ho = space.dims[o]
    sf = exp2(space.first_octave + o - 1)                   # input pixels per octave pixel
    cube = KA.allocate(device, Float32, Wo, Ho, S + 3, 1)
    noise = _octave_noise(device, noise, sf, Wo, Ho)           # a variance matrix sampled on this octave's grid, once
    for l in 1:(S + 3)
        level = view(space.levels[o], :, :, l, 1)
        σl = space.seed_sigma * exp2((l - 1) / S)           # the level's blur, octave pixels
        σd = σl * sqrt(exp2(2 / S) - 1)                     # the pyramid's blur increment: τ = √(σₗ² + σd²) = σₗ₊₁
        σ = _SFOP_SCALE_RATIO * σl * exp2(1 / S)
        g2, g1 = gaussian_gradient(level, σd)                # ∂/∂(dim 2), ∂/∂(dim 1)
        γ = KA.allocate(device, Float32, Wo, Ho, 3)
        _launch!(_squared_gradient!, device, γ, g1, g2; ndrange = (Wo, Ho))
        G, r = _integration_taps(:gaussian, σ)
        d = collect(-r:r)
        up(m) = ka_upload(Float32, device, m)
        M = _separable_correlate(device, γ, up([G G G]), up([G G G]))
        # Ω̄ = γ₂ ⋆ (d₁²G) − 2 γ₃ ⋆ (d₁G·d₂G) + γ₁ ⋆ (d₂²G), d = q − p: dim 1 the rows' taps, dim 2 the columns'
        Ω = _separable_correlate(device, γ, up([G (d .^ 2 .* G) (d .* G)]), up([(d .^ 2 .* G) G (d .* G)]))
        floor = _sfop_floor(device, noise, sf, σl, σd, space, G, significance_level)
        border = r + ceil(Int, 3.3 * σd) + 1
        _launch!(
            _sfop_precision!, device, view(cube, :, :, l, 1), M, Ω, Float32(12σ^2 + 1 - 2), floor, Int32(border);
            ndrange = (Wo, Ho)
        )
    end
    return cube
end

@kernel function _sfop_precision!(w, @Const(m), @Const(omega), redundancy, floor, border::Int32)
    i, j = @index(Global, NTuple)
    T = eltype(w)
    W, H = size(w)
    @inbounds begin
        _, λ2 = _window_measure(m[i, j, 1], m[i, j, 2], m[i, j, 3], true, zero(T), zero(T))   # its smaller eigenvalue
        Ω = omega[i, j, 1] + omega[i, j, 2] - 2 * omega[i, j, 3]
        inside = (border < i <= W - border) & (border < j <= H - border)
        # −1/w: the reference point's misfit Ω is quadratic in its offset from the point (Ω_min + eᵀNe), so −1/w is
        # what the extraction's quadratic interpolation fits exactly; w itself (∝ 1/|e|² near a sharp point) is not,
        # and its refinement failed on a coarse octave's grid
        w[i, j] = ifelse(inside & (λ2 > _floor_at(floor, i, j)) & (Ω > zero(T)), -Ω / (redundancy * λ2), -T(Inf))
    end
end

# The λ₂ floor of one level (octave pixels): white input noise of σ `s` (the [0, 1] image) seen through the level's
# blur. The noise was blurred by what the pyramid applied, √(σₗ² − b²) (b the assumed input blur, in octave pixels),
# then by the gradient's σ_d: the gradient of a white field of density s² blurred by τₙ (input pixels) has
# E|∇g|² = s²/(4πτₙ⁴) and component covariances C_xx(δ) = s² g(δ)(1/(2τₙ²) − δx²/(4τₙ⁴)), C_xy = −s² g(δ) δx δy/(4τₙ⁴),
# g = G_{√2 τₙ}, for a unit density (in octave pixels the density is (s/sf)²). tr M̄ = Σ_q G(q) |∇g(q)|²: mean E, variance
# Σ_δ R(δ)·2(C_xx² + C_yy² + 2C_xy²)(δ), R the weights' autocorrelation; ν = 2E²/Var; T = ½E·χ²_ν(level)/ν.
function _sfop_lambda2_floor(s, sf, σl, σd, space, G::Vector{Float64}, significance_level)
    τn = sqrt(max(σl^2 - (space.input_blur / sf)^2, 0.0) + σd^2)   # octave pixels
    E = (s / sf)^2 / (4π * τn^4)      # the noise density s² per input pixel is (s/sf)² per octave pixel
    r = length(G) ÷ 2
    ac = [sum(G[i] * G[i + δ] for i in max(1, 1 - δ):min(length(G), length(G) - δ)) for δ in -2r:2r]
    var = 0.0
    for (iy, δy) in enumerate(-2r:2r), (ix, δx) in enumerate(-2r:2r)
        g = exp(-(δx^2 + δy^2) / (4τn^2)) / (4π * τn^2)
        cxx = g * (1 / (2τn^2) - δx^2 / (4τn^4)); cyy = g * (1 / (2τn^2) - δy^2 / (4τn^4)); cxy = -g * δx * δy / (4τn^4)
        var += ac[ix] * ac[iy] * 2 * (cxx^2 + cyy^2 + 2cxy^2)
    end
    var *= (s / sf)^4
    ν = 2E^2 / var
    return E / 2 * quantile(Chisq(ν), significance_level) / ν
end

# The λ₂ floor of one level: `noise` the white noise's σ on the [0, 1] image (one floor), or each input pixel's
# variance (a device matrix, H × W): the floor is linear in the noise variance (E and √Var of tr M̄ both scale with
# it, ν not), so a pixel's floor is the unit-variance floor times the variance its window sees — the variance taken
# on the octave's grid and averaged under the integration Gaussian G_σ, the weights of tr M̄ (each window's noise
# locally white: the approximation).
_sfop_floor(device, noise::Real, sf, σl, σd, space, G, significance_level) =
    Float32(_sfop_lambda2_floor(noise, sf, σl, σd, space, G, significance_level))
function _sfop_floor(device, sampled::AbstractArray, sf, σl, σd, space, G, significance_level)
    unit = _sfop_lambda2_floor(1.0, sf, σl, σd, space, G, significance_level)
    taps = ka_upload(Float32, device, reshape(G .* unit, :, 1))   # the unit floor folded into one axis' taps
    return _separable_correlate(device, sampled, taps, ka_upload(Float32, device, reshape(G, :, 1)))
end

# The octave's noise: one σ as it is, or the input variance (H × W) sampled on the octave's grid (`_octave_variance!`).
_octave_noise(device, noise::Real, sf, Wo, Ho) = noise
function _octave_noise(device, variance::AbstractMatrix, sf, Wo, Ho)
    sampled = KA.allocate(device, Float32, Wo, Ho, 1)
    _launch!(_octave_variance!, device, sampled, variance, Float32(sf); ndrange = (Wo, Ho))
    return sampled
end

@inline _floor_at(floor::Real, i, j) = floor
@inline _floor_at(floor::AbstractArray, i, j) = @inbounds floor[i, j, 1]

# The input variance (H × W, rows y) on an octave's grid (W_o × H_o, x first): at the input pixel the pyramid puts each
# octave pixel at (`_level_to_canvas`, 0-based both; the nearest one on the upsampled octave), `sf` input pixels per
# octave pixel.
@kernel function _octave_variance!(out, @Const(v), sf::Float32)
    i, j = @index(Global, NTuple)
    H, W = size(v)
    x = round(Int, _level_to_canvas(Float32(i - 1), sf)) + 1
    y = round(Int, _level_to_canvas(Float32(j - 1), sf)) + 1
    @inbounds out[i, j, 1] = v[clamp(y, 1, H), clamp(x, 1, W)]
end
