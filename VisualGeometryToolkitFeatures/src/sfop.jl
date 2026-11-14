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
                precision_threshold = 0.5, nb_octaves = 3, nb_scales = 3, input_blur = 0.5,
                upsample = true, seed_sigma = (upsample ? 2 : 1) * input_blur) -> StructArray{<:Detection}
    detect_sfop(L::LinearRaw; kwargs...)

SFOP scale selection [Foerstner2009](@cite) for CIRCULAR points (the spiral
model at α = 90°): each point where the precision `w` of a circular point's
estimate peaks in position and scale, with the integration scale `σ` it peaks
at. Rows `Detection(Disc(x, σ), (; precision))`, COLMAP pixel centres of the
input (of a RAW's green cells, `L::LinearRaw`: the cell grid, their values) and σ in input pixels; `precision` is `w` (px⁻²). Seed
[`detect_foerstner(img, seeds)`](@ref detect_foerstner) with them to localise
each point at full resolution in its own window.

- **The scale space:** [`gaussian_scale_space`](@ref)`(img; nb_octaves, nb_scales,
  seed_sigma, input_blur, upsample)`, ASIFT's pyramid. Its first level is the
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
"""
function detect_sfop(
        img::AbstractMatrix; device::KA.Backend = default_device(),
        noise_sigma::Real = 2.0, significance_level::Real = 0.999, precision_threshold::Real = 0.5,
        nb_octaves::Integer = 3, nb_scales::Integer = 3, input_blur::Real = 0.5, upsample::Bool = true,
        seed_sigma::Real = (upsample ? 2 : 1) * input_blur
    )
    0 < significance_level < 1 ||
        throw(ArgumentError("detect_sfop: significance_level must lie in (0, 1), got $significance_level"))
    isfinite(precision_threshold) && precision_threshold > 0 ||
        throw(ArgumentError("detect_sfop: precision_threshold must be positive and finite, got $precision_threshold"))
    isfinite(noise_sigma) && noise_sigma > 0 ||
        throw(ArgumentError("detect_sfop: noise_sigma must be positive and finite, got $noise_sigma"))
    space = gaussian_scale_space(img; device, nb_scales, seed_sigma, input_blur, upsample, nb_octaves)
    W, H = space.dims[1] .>> (space.first_octave < 0 ? 1 : 0)
    K = _scheduled(_asift_kernels, device)
    S = space.nb_scales
    capacity = 1 << 16
    while true
        pool = KA.allocate(device, Float32, 5, capacity)
        count = KA.zeros(device, Int32, 2)
        ends = Int[]                                        # the pool's last row of each octave
        for o in eachindex(space.levels)
            precision = _sfop_precision(device, space, o, noise_sigma, significance_level)
            Wo, Ho = space.dims[o]
            t, _ = _tile_lists(Wo, Ho, Int32[Wo], Int32[Ho])
            K.ext(
                pool, count, precision, 0.0f0, floatmax(Float32), Float32(space.seed_sigma),
                Int32(o - 1 + space.first_octave), Int32(S), Int32(capacity),
                ka_upload(Int32, device, Int32[Wo]), ka_upload(Int32, device, Int32[Ho]), nothing,
                ka_upload(Int32, device, t), Val(S + 3), Val(_TILE_NY), Val(true);
                ndrange = (32, _TILE_NY * (length(t) ÷ 3))
            )
            KA.synchronize(device)
            push!(ends, Int(Array(count)[1]))
        end
        n = last(ends)
        n > capacity && (capacity = n; continue)           # the pool overflowed: once more, large enough
        return _sfop_detections(device, space, Array(view(pool, :, 1:n)), ends, W, H, 1 / precision_threshold^2)
    end
end

# A RAW's green cells: SFOP on their values (Core's `green_plane`), positions on the cell grid.
detect_sfop(L::LinearRaw; kw...) = detect_sfop(_green_values(green_plane(L)); kw...)

# The extracted maxima (`raw`, ASIFT's 5 × n rows: 0-based input x, y, the slice σ, w, slot; octave o's rows end at
# `ends[o]`) as detections: the level's blur from the slice σ (`_slice_sigma` reports the geometric mean of a DoG
# slice's two levels, so the level is half a slice lower), τ the next level's, σ = 3τ in input pixels, and ASIFT's
# NMS over them (eq. 8 as footprint overlap, see `detect_sfop`).
function _sfop_detections(device, space, raw, ends, W, H, min_precision)
    S = space.nb_scales
    feats = ImasFeatureKA[]
    precision = Dict{Tuple{Float32, Float32, Float32}, Float64}()
    w = -1 ./ view(raw, 4, :)                               # the cube holds −1/w
    wmax = maximum(w; init = 0.0f0)
    for o in eachindex(ends), i in (o == 1 ? 1 : ends[o - 1] + 1):ends[o]
        w[i] >= min_precision || continue
        τ = raw[3, i] * exp2(0.5 / S)                       # slice σ·2^(½/S): the next level's blur, input pixels
        σ = Float32(_SFOP_SCALE_RATIO * τ)
        x, y = raw[1, i], raw[2, i]
        push!(feats, ImasFeatureKA(x, y, σ, 0.5f0 * w[i] / wmax, σ^2, 0.0f0, σ^2, Int32(1)))
        precision[(x, y, σ)] = w[i]
    end
    kept = isempty(feats) ? feats : cross_warp_nms(
        feats, W, H; device, mode = :ellipse, cutoff = 1 / sqrt(2), band_octaves = 1.0, band_reach = 1
    )
    rows = [
        _SfopDetection(Disc(Point2{Float64}(f.x + 0.5, f.y + 0.5), Float64(f.σ)), (; precision = precision[(f.x, f.y, f.σ)]))
            for f in kept
    ]
    return StructArray(collect(_SfopDetection, rows); unwrap = T -> T <: NamedTuple)
end

# The precision cube of octave o, (W_o, H_o, S + 3, 1): −1/w at every level (its maxima are w's), −∞ where λ₂ fails
# the significance test or the windows reach past the octave's border. The maxima are searched on levels 2 … S + 2, so each octave's range
# overlaps the next one's by a level: a peak between an octave's last levels is seen on one grid even where the two
# grids order the shared level differently (the NMS keeps one of a double).
function _sfop_precision(device, space::GaussianScaleSpace, o, noise_sigma, significance_level)
    S = space.nb_scales
    Wo, Ho = space.dims[o]
    sf = exp2(space.first_octave + o - 1)                   # input pixels per octave pixel
    cube = KA.allocate(device, Float32, Wo, Ho, S + 3, 1)
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
        floor = _sfop_lambda2_floor(noise_sigma / 255, sf, σl, σd, space, G, significance_level)
        border = r + ceil(Int, 3.3 * σd) + 1
        _launch!(
            _sfop_precision!, device, view(cube, :, :, l, 1), M, Ω, Float32(12σ^2 + 1 - 2), Float32(floor), Int32(border);
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
        w[i, j] = ifelse(inside & (λ2 > floor) & (Ω > zero(T)), -Ω / (redundancy * λ2), -T(Inf))
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
