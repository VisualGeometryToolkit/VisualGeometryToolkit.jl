# =============================================================================
# The Förstner interest operator (Förstner & Gülch 1987, with Köthe 2003's
# doubled resolution and window criterion), after the reference
# implementation ip_fop.m (M. Luxen, University of Bonn).
#
# Each step follows ip_fop.m; the line numbers below are that file's. The
# image-wide steps are KernelAbstractions kernels (one code path: `KA.CPU()` is
# the CPU path), their per-element math plain generic functions. The
# deviations from the reference are listed in `detect_foerstner`'s docstring
# and on the docs page, with the evidence for each.
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const

# One row of `detect_foerstner`'s result: a point frame with its 2×2
# covariance, labelled, with its window centre and the classification statistic.
const _FoerstnerDetection = Attributed{
    Uncertain{PointFrame{Float64}, CovMat{2, Float64, SMatrix{2, 2, Float64, 4}, UnlabeledFrame{2}}},
    @NamedTuple{label::Symbol, window::Point2{Float64}, F::Float64, covariance_status::Symbol},
}

"""
    detect_foerstner([T = Float32], img; device = default_device(),
                     method = :foerstner, noise = nothing, noise_sigma = something(noise, 2.0),
                     derivative = :gaussian2d, integration = :box,
                     derivative_sigma = 1.0, integration_sigma = 1.41 * derivative_sigma,
                     precision_threshold = 0.5, roundness_threshold = 0.3,
                     significance_level = 0.999) -> StructArray{<:Detection}
    detect_foerstner([T = Float32], L::LinearRaw; sensor_model::SensorModel = SensorModel(), kwargs...)

The Förstner interest operator [Foerstner1987](@cite): the junction (corner)
points and circular points of a grey image at sub-pixel precision, each with
its covariance, after the reference implementation `ip_fop.m`
[Luxen2005](@cite), with Köthe's doubled resolution and window criterion
[Koethe2003](@cite).

Returns one row per optimal window, a `StructArray` of
`Detection(Uncertain(PointFrame(x), Σ), (; label, window, F))` (VisualGeometryToolkitCore's
`PointFrame`: a position, no scale):

- `label`: `:corner` (a junction point), `:circular` (Förstner & Gülch's
  circular point: the centre of a rotationally symmetric pattern, with no size
  estimate) or `:unknown` (the significance test assigned neither);
- the frame: the corner or circular point's position and its 2×2 covariance
  (px², `(x, y)` order); for an `:unknown` row the window centre with a NaN
  covariance (the reference returns no estimate for it);
- `window`: the window centre, a `Point2`;
- `F`: the classification statistic `Ω_corner / Ω_circular`;
- `covariance_status`: how the frame's covariance was obtained, or why there is
  none: the image noise propagated through the detector (below) — `:propagated`
  for a given σₙ, `:propagated_measured` / `:propagated_estimated` /
  `:propagated_specified` for a RAW under its frame's `SensorModel`, by the
  source of its Poisson–Gaussian noise — `:a_posteriori` (the reference's
  `Ω/redundancy · N⁻¹`, when no noise is known), or `:unclassified` on an
  `:unknown` row (no estimate, a NaN covariance).

`filter(d -> d.label === :corner, points)` are the corners; `Ellipse(d.value; sigma = 1)`
is a point's 1σ error ellipse (`ip_errell.m`'s default).

`img` is any grey image the package's detectors take, read through
[`gray_f32`](@ref) as grey values in `[0, 1]` (an 8-bit image as `value/255`,
as MATLAB's `im2double`); it is not stretched to its own range. The image-wide
steps run on `device`, any KernelAbstractions backend, in precision `T`
(the reference computes in `Float64`; supply `Float64` for that).

# Keywords (defaults: the values the reference's CODE runs with)
- `method`: `:foerstner` — windows where the roundness `4 det/tr² > roundness_threshold`
  and the precision `det/tr > (σₙ / precision_threshold)²`, ranked by `det/tr`;
  `:koethe` — windows whose precision test succeeds, ranked by the smaller eigenvalue.
- `noise`: the image noise the covariance is propagated from: the standard
  deviation σₙ of iid pixel noise, in grey levels of an 8-bit image (0–255, as
  `noise_sigma`). The covariance is then `σₙ² J Jᵀ` (`σₙ` on the `[0, 1]` image),
  `J` the exact first-order derivative of the estimate with respect to every input
  pixel, through the detector's own doubling, derivative taps, squared gradient
  and window estimate (step 6). `nothing`: no noise is known, and the covariance
  is the reference's a-posteriori `Ω/redundancy · N⁻¹` (step 4), marked
  `covariance_status = :a_posteriori`; that one is not calibrated (it carries the
  model misfit, which does not scale with the noise).
- `sensor_model` (a RAW's green cells, `L::LinearRaw`, read through
  VisualGeometryToolkitCore's `green_plane`, positions on the cell grid): the
  frame's resolved `SensorModel` (`sensor_model`), the one `L` was linearised
  with. Under a Poisson–Gaussian noise (a gain) `L` carries each value's
  variance (Core's one law), and each pixel's variance weighs its column,
  `Σ = J diag(var) Jᵀ`; read noise alone, a relative noise or none leaves the
  a-posteriori covariance (the owner's ruling: only a gain-carrying model is fed).
- `noise_sigma`: the noise level of the PRECISION TEST (step 3), in grey levels
  of an 8-bit image (0–255), as the reference's for a `uint8` image;
  `σₙ = noise_sigma / 255` on the `[0, 1]` image. It defaults to `noise` when that
  is given, else to the reference's 2. It does not enter the covariance.
- `derivative`: `:gaussian2d` (a Gaussian derivative smoothed across by a
  Gaussian of the same σ) or `:gaussian1d` (the derivative alone).
- `integration`: `:box` (a box whose standard deviation is `integration_sigma`)
  or `:gaussian`.
- `derivative_sigma`, `integration_sigma`: in input pixels.
- `precision_threshold`: the largest accepted point standard deviation, input px.
- `roundness_threshold`: the smallest accepted roundness (`:foerstner` only).
- `significance_level`: of the F-test that labels a point corner or circular.

# Algorithm (ip_fop.m, step by step)
1. The image is doubled bilinearly (the package's one 2× upsampling, ASIFT's)
   and every scale and threshold doubled with it (lines 170–180).
2. The gradient by correlation with the derivative taps (radius `⌈3.3σ⌉`, zero
   padding, as `imfilter`), its squares and product (the squared gradient),
   and their integration by the box (width `2⌈√(12σ² + 1)/2⌉ + 1`) or the
   Gaussian (radius `⌈3.3σ⌉`), zero padded (lines 184–207, 617–678).
3. The window measure (`det/tr` or the smaller eigenvalue) masked by the tests
   above; its regional maxima (8-connected plateaus higher than every
   neighbour, `imregionalmax`), thinned by a second non-maximum suppression
   over a square of radius `2⌈√(12σᵢ² + 1)⌉` doubled pixels, away from the
   border (lines 210–281).
4. In the `(2ρ + 1)²` window around each centre, `ρ = ⌈√(12σᵢ² + 1)/2⌉ + 1`
   (σᵢ doubled), the pixels weighed equally (the reference's choice, also for
   a Gaussian integration): the corner is the least-squares intersection of
   the lines through every pixel ACROSS its gradient (normal matrix
   `Σ ∇g∇gᵀ`), the circular point that of the lines ALONG it (`Σ ∇g⊥∇g⊥ᵀ`).
   Each has a weighted residual sum `Ω`, redundancy `(2ρ + 1)² − 2` and
   a-posteriori covariance `Ω/redundancy · N⁻¹` (lines 314–423, 440, 449).
5. `F = Ω_corner / Ω_circular` against the F distribution of (redundancy,
   redundancy) degrees of freedom, as the reference's
   `(T < F(α)) − (T > F(1 − α))` (line 429): a corner below the
   `1 − significance_level` quantile, a circular point above the
   `significance_level` quantile, unknown in between.
6. With `noise` given, the covariance is propagated instead (not in the
   reference): the estimate `x̂ = N⁻¹ b` moves, by the implicit-function theorem,
   by `δx̂ = N⁻¹ Σₚ A(p) δ∇g(p)` with `A(p) w = (vᵀe) R w + v eᵀ R w`, `v = R∇g`,
   `e = p − x̂` (`R` the identity for a corner, the 90° rotation for a circular
   point), the window centre fixed; the gradient is linear in the input, so each
   input pixel `q` gives the column `J[:, q]`, exact at the image borders, and
   `Σ = Σ_q σₙ² J[:, q] J[:, q]ᵀ`. Monte Carlo against each dot's noise-free
   estimate gives the χ²₂ NEES ≈ 2 and 95 % coverage (Förstner study, Part 2).

# Coordinates
COLMAP pixel centres of the input image, `(x, y) = (column, row)`, converted
once, at the output. The doubled image's pixel `(r′, c′)` (1-based) samples the
input at 1-based `(r′/2, c′/2)`, so a doubled-grid position is at COLMAP
`((c′ − 1)/2, (r′ − 1)/2)` in the input, and its covariance is the doubled one
`/4`, reordered to `(x, y)`.

# Differences from ip_fop.m
1. **Undoing the doubling — fixed.** The reference doubles with MATLAB's
   `imresize(g, 2, 'bilinear')` (line 172; doubled pixel `j` samples input
   `j/2 + 1/4`) but maps positions back with `transf_H`, `transf_v`
   (lines 340–350, applied at 436, 446, 457), the inverse of an align-corners
   grid: its points are off by `((n + 1)/2 − x)/(2n − 1)` (1-based, `n` the
   image size), 0 at the centre, up to 0.25 px toward the centre at the
   borders. Here the doubling is the package's upsampling (input `j/2`, a
   quarter pixel from `imresize`'s grid) undone by its own exact inverse.
2. **The circular point's residual sum — kept as the reference has it.** Its
   line 423 takes two of the three terms from the corner model's residuals;
   the circular model's own residual sum would take all three from its own.
   Monte Carlo coverage favours the reference's form, so it is kept.
3. **Defaults — the code's, not the header's.** The header states
   `SIGMA_N = 2.5` and `SIGMA_DIFF = 0.7` (lines 37, 50); the code runs `2.0`
   and `1.0` (lines 521, 524). A documentation conflict, not a fix.

The docs page gives the evidence for each. The reference warns that with a
Gaussian integration the window estimation still weighs the pixels equally;
so does this.

# Example
```jldoctest
julia> using VisualGeometryToolkitFeatures, VisualGeometryToolkitCore

julia> img = [Float32((c - 20.5)^2 + (r - 24.5)^2 < 6.25) for r in 1:48, c in 1:40];   # a dot at pixel (24.5, 20.5)

julia> points = detect_foerstner(img);

julia> only(points).label
:circular

julia> round.(Tuple(only(points).center); digits = 1)   # COLMAP: (column − ½, row − ½)
(20.0, 24.0)
```
"""
detect_foerstner(img::Union{AbstractMatrix, LinearRaw}; kw...) = detect_foerstner(Float32, img; kw...)

detect_foerstner(::Type{T}, img::AbstractMatrix; noise::Union{Nothing, Real} = nothing, noise_sigma::Real = something(noise, 2.0),
                 kw...) where {T <: AbstractFloat} = _detect_foerstner(T, img, _pixel_noise(noise); noise_sigma, kw...)

"""
    detect_foerstner([T = Float32], img, seeds::AbstractVector{<:Detection}; device = default_device(),
                     noise = nothing, derivative = :gaussian2d, derivative_sigma = 1.0,
                     significance_level = 0.999) -> StructArray{<:Detection}

The Förstner operator's window estimation (steps 4–6 of the method above, its
doubled resolution, its kernels) at GIVEN windows: one window per seed, centred
on the doubled-grid pixel nearest the seed's centre, its integration scale the
seed's characteristic scale σ (input pixels; a [`detect_sfop`](@ref) row's
selected σ). The box window has the standard deviation σ, the estimation window
the half-width `⌈√(12(2σ)² + 1)/2⌉ + 1` doubled pixels, as `integration_sigma = σ`
would give: SFOP's `N(σ) = 12σ² + 1` is the pixel count of the box of the same
energy as its Gaussian `G_σ`. The image-wide window search (step 3) is not run.

Returns the rows of `detect_foerstner(img)` (label, frame and covariance,
window, F; the F-test of each window at its own redundancy), one per seed whose
window lies inside the doubled image, in the order of `seeds`.
"""
detect_foerstner(img::Union{AbstractMatrix, LinearRaw}, seeds::AbstractVector{<:Detection}; kw...) =
    detect_foerstner(Float32, img, seeds; kw...)

detect_foerstner(::Type{T}, img::AbstractMatrix, seeds::AbstractVector{<:Detection}; noise::Union{Nothing, Real} = nothing,
                 kw...) where {T <: AbstractFloat} = _detect_foerstner(T, img, _pixel_noise(noise); seeds, kw...)

detect_foerstner(::Type{T}, L::LinearRaw; sensor_model::SensorModel = SensorModel(), kw...) where {T <: AbstractFloat} =
    _detect_foerstner(T, _green_values(green_plane(L)), _pixel_noise(sensor_model.noise, green_plane(L)); kw...)

# The values of a green plane: the plane itself, or each uncertain element's value (Core's `value_of`), whatever its
# storage (a lazy view or a materialised one).
_green_values(G::AbstractMatrix{<:Real}) = G
_green_values(G::AbstractMatrix{<:Uncertain}) = map(value_of, G)

# The detector on the grey image `img`; `model` the pixel noise it propagates (`_pixel_noise`) or `nothing`.
function _detect_foerstner(
        ::Type{T}, img::AbstractMatrix, model;
        device::KA.Backend = default_device(),
        method::Symbol = :foerstner,
        noise_sigma::Real = 2.0,
        derivative::Symbol = :gaussian2d,
        integration::Symbol = :box,
        derivative_sigma::Real = 1.0,
        integration_sigma::Real = 1.41 * derivative_sigma,
        precision_threshold::Real = 0.5,
        roundness_threshold::Real = 0.3,
        significance_level::Real = 0.999,
        seeds::Union{Nothing, AbstractVector{<:Detection}} = nothing
    ) where {T <: AbstractFloat}
    0 < significance_level < 1 ||
        throw(ArgumentError("detect_foerstner: significance_level must lie in (0, 1), got $significance_level"))
    # a float matrix at its own precision (gray_f32 would round a Float64 one), anything else as gray_f32 reads it
    g = ka_upload(T, device, img isa AbstractMatrix{<:AbstractFloat} ? img : gray.(gray_f32(img)))
    doubled = KA.allocate(device, T, 2 .* size(g)...)
    _launch!(_upsample_doubled!, device, doubled, reshape(g, size(g)..., 1); ndrange = size(doubled))
    found = seeds === nothing ? _foerstner_windows(
        device, doubled; method, noise_sigma, derivative, integration, derivative_sigma,
        integration_sigma, precision_threshold, roundness_threshold
    ) : _foerstner_seeded(device, doubled, seeds; derivative, derivative_sigma)
    classes = _foerstner_classes(found, significance_level)
    model === nothing && return _foerstner_points(found, classes, _undouble)
    variance, covariance_status = model
    v = variance isa Real ? fill!(KA.allocate(device, T, size(g)...), T(variance)) : ka_upload(T, device, variance)
    covariances = _propagated_covariances(device, found, classes, v)
    return _foerstner_points(found, classes, _undouble; covariances, covariance_status)
end

# The pixel noise `detect_foerstner` propagates, `(variance, covariance_status)`: the variance of every input pixel (a
# scalar for iid noise, else a matrix of the image's size). `nothing`: no noise is known, the covariance is the
# a-posteriori one.
_pixel_noise(::Nothing) = nothing
# iid noise of σn grey levels of an 8-bit image, on the [0, 1] image
function _pixel_noise(σn::Real)
    isfinite(σn) && σn > 0 || throw(ArgumentError("detect_foerstner: the noise σ must be positive and finite, got $σn"))
    return (σn / 255)^2, :propagated
end
# A RAW's green cells: fed only under a gain-carrying model (Poisson–Gaussian, the owner's ruling A), each pixel's
# variance from the uncertain image the model made (Core's one law, D11); read noise alone, relative noise or none
# leaves the a-posteriori covariance. A gain whose image carries no variances was not linearised with the model.
_pixel_noise(::NoiseComponent, G::AbstractMatrix) = nothing
_pixel_noise(n::PoissonGaussian, G::AbstractMatrix{<:Uncertain}) = map(var, G), Symbol("propagated_", string(n.source))
_pixel_noise(::PoissonGaussian, G::AbstractMatrix) = throw(ArgumentError(
    "detect_foerstner: the sensor model has a gain but the LinearRaw holds no variances: linearise it with the same model, LinearRaw(mosaic; mode, sensor_model) or load_raw(…; sensor_model)"))

# A kernel launch through the stack's launch policy (VisualGeometryToolkitCore's
# `OnBackend`, the backend's choice of workgroup), completed before the next:
# KernelAbstractions launches are asynchronous (on the CPU backend too).
function _launch!(kernel!, device, args...; ndrange)
    OnBackend(device)(kernel!)(args...; ndrange)
    KA.synchronize(device)
    return nothing
end

# A 1-based doubled-grid (row, column) → COLMAP (x, y) of the input: the
# package's upsampling puts doubled pixel j at input j/2 (1-based).
_undouble(r::Real, c::Real) = Point2{Float64}((c - 1) / 2, (r - 1) / 2)

@kernel function _upsample_doubled!(dst, @Const(src))
    I, J = @index(Global, NTuple)
    @inbounds dst[I, J] = _upsample_sample(src, I, J, 1)
end

# The image-wide steps on the doubled image (ip_fop.m 127, 177–281) and the
# window estimates (314–423), positions on the doubled grid, 1-based (row,
# column). Returns the window centres, the estimates (a 14 × K host matrix, see
# `_window_estimate`), the window half-width ρ, its redundancy and, on the device,
# the squared gradient (`grr, gcc, grc` as a H × W × 3 array) and the gradient.
function _foerstner_windows(
        device, doubled::AbstractMatrix{T}; method::Symbol, noise_sigma::Real, derivative::Symbol,
        integration::Symbol, derivative_sigma::Real, integration_sigma::Real,
        precision_threshold::Real, roundness_threshold::Real
    ) where {T}
    method in (:foerstner, :koethe) ||
        throw(ArgumentError("detect_foerstner: method must be :foerstner or :koethe, got :$method"))
    integration in (:box, :gaussian) ||
        throw(ArgumentError("detect_foerstner: integration must be :box or :gaussian, got :$integration"))
    upscale = 2                                             # 170–180
    nonmax_radius = upscale * ceil(Int, sqrt(12 * integration_sigma^2 + 1))
    σi = upscale * integration_sigma
    precision_floor = T(((noise_sigma / 255) / (upscale * precision_threshold))^2)   # 213
    rows, cols = size(doubled)

    # 184–207: the squared gradient and its integration
    grad = _foerstner_gradient(device, doubled; derivative, derivative_sigma, upscale)
    gamma = grad.gamma
    integration_taps, integration_radius = _integration_taps(integration, σi)
    mean_gamma = _separable_correlate(device, gamma, ka_upload(T, device, repeat(integration_taps, 1, 3)),
                                      ka_upload(T, device, repeat(integration_taps, 1, 3)))

    # 210–248: the window measure and its regional maxima
    candidates = KA.allocate(device, T, rows, cols)
    ranking = similar(candidates)
    _launch!(
        _window_measures!, device,
        candidates, ranking, mean_gamma, method === :koethe, precision_floor, T(roundness_threshold);
        ndrange = (rows, cols)
    )
    is_max = KA.allocate(device, Bool, rows, cols)
    _launch!(_plateau_maxima_init!, device, is_max, candidates; ndrange = (rows, cols))
    changed = KA.allocate(device, Int32, 1)
    while true                                              # a plateau loses where an equal neighbour lost
        fill!(changed, Int32(0))
        _launch!(_plateau_maxima_spread!, device, is_max, changed, candidates; ndrange = (rows, cols))
        KA.synchronize(device)
        Array(changed)[1] == 0 && break
    end

    # 249–281: the second non-maximum suppression, away from the border; the kept centres compacted on the device
    border = max(integration_radius, nonmax_radius) + 1
    keep = KA.allocate(device, Bool, rows, cols)
    _launch!(_second_suppression!, device, keep, is_max, ranking, Int32(nonmax_radius), Int32(border); ndrange = (rows, cols))
    centres = _kept_centres(device, keep, is_max, ranking)

    # 314–423: both point models in each window
    ρ = ceil(Int, sqrt(12 * σi^2 + 1) / 2) + 1
    return (; _foerstner_estimates(device, gamma, centres, ρ)..., upscale, grad...)
end

# 184–207 (the gradient half): the gradient of the doubled image at the doubled derivative σ, by the package's one
# Gaussian gradient, and its squared gradient (`grr, gcc, grc` as an H × W × 3 device array), with the taps it ran.
function _foerstner_gradient(device, doubled::AbstractMatrix{T}; derivative::Symbol, derivative_sigma::Real, upscale::Integer) where {T}
    derivative in (:gaussian2d, :gaussian1d) ||
        throw(ArgumentError("detect_foerstner: derivative must be :gaussian2d or :gaussian1d, got :$derivative"))
    σd = upscale * derivative_sigma
    derivative_taps, across_taps = _derivative_taps(derivative, σd)
    gx, gy = gaussian_gradient(doubled, σd; derivative)    # the package's one Gaussian gradient (∂/∂column, ∂/∂row)
    rows, cols = size(doubled)
    gamma = KA.allocate(device, T, rows, cols, 3)
    _launch!(_squared_gradient!, device, gamma, gy, gx; ndrange = (rows, cols))
    return (; gamma, gradient_rows = gy, gradient_columns = gx, taps = [derivative_taps across_taps])
end

# The windows of `seeds` on the doubled image: each centred on the doubled pixel nearest its centre (COLMAP x is
# doubled column 2x + 1), half-width ρ = ⌈√(12(2σ)² + 1)/2⌉ + 1 for its characteristic scale σ (the box window of
# `integration_sigma = σ`); a window that leaves the image is dropped. Then both point models in each (314–423).
function _foerstner_seeded(device, doubled::AbstractMatrix, seeds; derivative::Symbol, derivative_sigma::Real)
    upscale = 2
    grad = _foerstner_gradient(device, doubled; derivative, derivative_sigma, upscale)
    rows, cols = size(doubled)
    centres, radii = CartesianIndex{2}[], Int[]
    for d in seeds
        disc = Disc(d)
        r, c = round(Int, upscale * disc.center[2] + 1), round(Int, upscale * disc.center[1] + 1)
        ρ = ceil(Int, sqrt(12 * (upscale * disc.σ)^2 + 1) / 2) + 1
        (ρ < r <= rows - ρ) && (ρ < c <= cols - ρ) || continue
        push!(centres, CartesianIndex(r, c)); push!(radii, ρ)
    end
    return (; _foerstner_estimates(device, grad.gamma, centres, radii)..., upscale, grad...)
end

# 314–423: both point models in the window of half-width ρ (one for every window, or one per window) around each
# centre (doubled 1-based CartesianIndex), each window's redundancy `(2ρ + 1)² − 2`.
function _foerstner_estimates(device, gamma::AbstractArray{T, 3}, centres, ρ) where {T}
    centre_list = ka_upload(Int32, device, [Int32(Tuple(p)[k]) for k in 1:2, p in centres])
    radii = ρ isa Integer ? Int32(ρ) : ka_upload(Int32, device, Int32.(ρ))
    device_estimates = KA.allocate(device, T, 14, length(centres))
    isempty(centres) ||
        _launch!(_window_estimates!, device, device_estimates, gamma, centre_list, radii; ndrange = length(centres))
    KA.synchronize(device)
    return (;
        centres, estimates = Float64.(Array(device_estimates)), ρ = radii, redundancy = (2 .* ρ .+ 1) .^ 2 .- 2,
        centre_list, device_estimates,
    )
end

# A window parameter shared by every window (a number) or one per window (a vector): window k's.
@inline _per_window(x::Number, k) = x
@inline _per_window(x::AbstractVector, k) = @inbounds x[k]

# 228, 249–281: the kept window centres (the second suppression's `keep`), compacted on the device — a count, then
# each kept pixel's linear index appended — and sorted column-major as MATLAB's find. None when every pixel is a
# regional maximum of positive rank (255–258: the reference discards a constant image).
function _kept_centres(device, keep, is_max, ranking)
    counts = KA.zeros(device, Int32, 2)
    _launch!(_count_kept!, device, counts, keep, is_max, ranking; ndrange = size(keep))
    n, n_max = Array(counts)
    (n == 0 || n_max == length(keep)) && return CartesianIndex{2}[]
    list = KA.allocate(device, Int32, n)
    fill!(counts, Int32(0))
    _launch!(_append_kept!, device, list, counts, keep; ndrange = size(keep))
    return CartesianIndices(keep)[sort!(Array(list))]
end

@kernel function _count_kept!(counts, @Const(keep), @Const(is_max), @Const(ranking))
    I = @index(Global, Linear)
    @inbounds keep[I] && Atomix.@atomic counts[1] += Int32(1)
    @inbounds (is_max[I] & (ranking[I] > zero(eltype(ranking)))) && Atomix.@atomic counts[2] += Int32(1)
end

@kernel function _append_kept!(list, counts, @Const(keep))
    I = @index(Global, Linear)
    @inbounds if keep[I]
        k = Atomix.@atomic counts[1] += Int32(1)
        list[k] = Int32(I)
    end
end

# 427–429: each window's class from the F-test of `F = Ω_corner / Ω_circular`: 1 a corner, −1 a circular point,
# 0 unknown (the reference's noclass).
function _foerstner_classes(found, significance_level)
    critical(red) = (fdist = FDist(red, red); (quantile(fdist, significance_level), quantile(fdist, 1 - significance_level)))
    shared = found.redundancy isa Number ? critical(found.redundancy) : nothing     # one F distribution for all
    function class(k, e)
        corner_critical, circular_critical = something(shared, critical(_per_window(found.redundancy, k)))
        return (e[7] / e[14] < corner_critical) - (e[7] / e[14] > circular_critical)
    end
    return Int8[class(k, e) for (k, e) in enumerate(eachcol(found.estimates))]
end

# 323–460: one labelled row per window, positions mapped by `undouble`. The covariance of a classified window is
# the reference's a-posteriori `Ω/redundancy · N⁻¹` unless `covariances` (a 4 × K matrix, column-major 2×2 on the
# doubled grid, (row, column)) gives the propagated one; `covariance_status` names which.
function _foerstner_points(found, classes, undouble; covariances = nothing, covariance_status::Symbol = :a_posteriori)
    (; centres, estimates, redundancy, upscale) = found
    unknown_cov = CovMat(SMatrix{2, 2, Float64, 4}(NaN, NaN, NaN, NaN))
    rows = map(enumerate(centres)) do (k, p)
        e = view(estimates, :, k)
        class = classes[k]
        window = undouble(p[1], p[2])
        Σ(Ninv, Ω) = covariances === nothing ? Ninv .* (Ω / _per_window(redundancy, k)) : view(covariances, :, k)
        frame = if class == 1
            _uncertain_point(undouble, e[1], e[2], Σ(e[3:6], e[7]), upscale)
        elseif class == -1
            _uncertain_point(undouble, e[8], e[9], Σ(e[10:13], e[14]), upscale)
        else
            Uncertain(PointFrame(window), unknown_cov)   # the reference's noclass: no estimate (its status :unclassified)
        end
        label = class == 1 ? :corner : class == -1 ? :circular : :unknown
        status = class == 0 ? :unclassified : covariance_status
        _FoerstnerDetection(frame, (; label, window, F = e[7] / e[14], covariance_status = status))
    end
    return StructArray(collect(_FoerstnerDetection, rows); unwrap = T -> T <: NamedTuple)   # never `[rows...]`: a splat per point count
end

# ----- the propagated covariance: the exact first-order propagation of the pixel noise -----------------------
#
# The window estimate is x̂ = N⁻¹ b, N = Σₚ M(p), b = Σₚ M(p) p over the window's pixels p (doubled grid), with
# M = v vᵀ, v = R ∇g (R = I for the corner, the 90° rotation for the circular point: ip_fop.m's Γ and Γ⊥). By the
# implicit-function theorem a perturbation δ∇g of the gradient moves it by
#
#     δx̂ = N⁻¹ Σₚ A(p) δ∇g(p),   A(p) w = (vᵀe) R w + v eᵀ R w,   e = p − x̂;
#
# the window centre is a discrete choice and does not move. The gradient is linear in the input: the doubling
# then the correlation with the derivative taps along one axis and the across taps along the other, i.e. per
# component a product of two one-axis operators (`_pipeline_bands`), exact at the borders (the doubling's clamp,
# the correlations' zero padding). So an input pixel q moves the estimate by J[:, q] = N⁻¹ Σₚ A(p) ∂∇g(p)/∂I(q),
# and independent pixel noise of variance var(q) gives Σ = Σ_q var(q) J[:, q] J[:, q]ᵀ (σn² J Jᵀ for iid noise),
# evaluated at the window's own gradient and estimate.

# The doubled grid's samples along one axis of `n` input pixels as linear maps of the input, after the doubling
# (`_upsample_sample`) and the correlation along that axis with each column of `taps` (the detector's
# `_correlate_rows!` / `_correlate_columns!` taps, radius r, zero padded): column p is the adjoint of both stages
# applied to the doubled sample p (ASIFT's `_correlate_adj`, `_up_adj`, bounded: exact at the borders). `B[k, p, s]` is the
# weight of input pixel `_band_start(p, r) + k − 1` for taps column s: a doubled sample m reads input ⌊m/2⌋ and
# ⌊m/2⌋ + 1 (clamped), so p reads at most r + 2 consecutive input pixels.
function _pipeline_bands(taps::AbstractMatrix{Float64}, n::Integer)
    r = size(taps, 1) ÷ 2
    B = zeros(r + 2, 2n, size(taps, 2))
    for s in axes(taps, 2), p in 1:(2n)
        line = _up_adj(_correlate_adj(_Line(p, [1.0]), view(taps, :, s), 2n), n)
        for (j, w) in enumerate(line.v)
            w == 0 && continue
            B[line.i0 + j - 1 - _band_start(p, r) + 1, p, s] = w
        end
    end
    return B
end

@inline _band_start(p::Integer, r::Integer) = max(1, max(p - r, 1) >> 1)

# The propagated covariance of every classified window (a 4 × K host matrix, column-major 2×2 on the doubled grid,
# (row, column); NaN for an unknown window) for independent input noise of variance `variance[q]` at input pixel q
# (on the device of `found`), through the
# taps `_foerstner_windows` ran: rows correlated with (derivative, across), columns with (across, derivative).
function _propagated_covariances(device, found, classes, variance::AbstractMatrix)
    (; centres, ρ, gradient_rows, gradient_columns, taps, centre_list, device_estimates) = found
    H, W = size(variance)
    K = length(centres)
    K == 0 && return zeros(Float64, 4, 0)
    T = eltype(device_estimates)
    rows = ka_upload(T, device, _pipeline_bands(taps, H))
    columns = ka_upload(T, device, _pipeline_bands(taps[:, [2, 1]], W))
    out = KA.allocate(device, T, 4, K)
    _launch!(
        _propagated_covariances!, device, out, gradient_rows, gradient_columns, variance, centre_list, device_estimates,
        ka_upload(Int8, device, classes), rows, columns, ρ, Int32(size(taps, 1) ÷ 2); ndrange = K
    )
    return Float64.(Array(out))
end

# One window k: Σ = Σ_q variance[q] δ_q δ_qᵀ, δ_q = N⁻¹ Σₚ A(p) ∂∇g(p)/∂I(q). The gradient's component s at the doubled
# pixel (pr, pc) reads the input pixel (q1, q2) with weight rows[·, pr, s] · columns[·, pc, s], nonzero only for
# |p − 2q| ≤ r + 2 along each axis.
@kernel function _propagated_covariances!(
        out, @Const(gradient_rows), @Const(gradient_columns), @Const(variance), @Const(centres), @Const(estimates), @Const(classes),
        @Const(rows), @Const(columns), radii, r::Int32
    )
    k = @index(Global)
    ρ = _per_window(radii, k)
    T = eltype(out)
    @inbounds class = classes[k]
    @inbounds rw, cw = Int(centres[1, k]), Int(centres[2, k])
    H, W = size(variance)
    w = Int(r) + 2
    Σrr = Σrc = Σcc = zero(T)
    if class == 0
        Σrr = Σrc = Σcc = T(NaN)
    else
        o = class == 1 ? 0 : 7                            # the corner's estimate, or the circular point's
        @inbounds xr, xc = estimates[1 + o, k] - rw, estimates[2 + o, k] - cw
        @inbounds Ninv = SMatrix{2, 2, T, 4}(estimates[3 + o, k], estimates[4 + o, k], estimates[5 + o, k], estimates[6 + o, k])
        dot = class == -1
        for q2 in _band_start(cw - Int(ρ), Int(r)):min(W, _band_start(cw + Int(ρ), Int(r)) + w - 1),
                q1 in _band_start(rw - Int(ρ), Int(r)):min(H, _band_start(rw + Int(ρ), Int(r)) + w - 1)
            s1 = s2 = zero(T)
            for pc in max(cw - Int(ρ), 2q2 - w):min(cw + Int(ρ), 2q2 + w),
                    pr in max(rw - Int(ρ), 2q1 - w):min(rw + Int(ρ), 2q1 + w)
                k1 = q1 - _band_start(pr, Int(r)) + 1
                k2 = q2 - _band_start(pc, Int(r)) + 1
                (1 <= k1 <= w) & (1 <= k2 <= w) || continue
                @inbounds δr = rows[k1, pr, 1] * columns[k2, pc, 1]
                @inbounds δc = rows[k1, pr, 2] * columns[k2, pc, 2]
                @inbounds gr, gc = gradient_rows[pr, pc], gradient_columns[pr, pc]
                # v = R∇g and u = Rδ∇g (R the 90° rotation for the circular point), e = p − x̂
                vr, vc = dot ? (-gc, gr) : (gr, gc)
                ur, uc = dot ? (-δc, δr) : (δr, δc)
                er, ec = T(pr - rw) - xr, T(pc - cw) - xc
                ve = vr * er + vc * ec
                eu = er * ur + ec * uc
                s1 += ve * ur + eu * vr
                s2 += ve * uc + eu * vc
            end
            δ = Ninv * SVector(s1, s2)
            @inbounds v = variance[q1, q2]
            Σrr += v * δ[1]^2
            Σrc += v * δ[1] * δ[2]
            Σcc += v * δ[2]^2
        end
    end
    @inbounds out[1, k] = Σrr
    @inbounds out[2, k] = Σrc
    @inbounds out[3, k] = Σrc
    @inbounds out[4, k] = Σcc
end

# A doubled-grid estimate and its (row, column) covariance `Σ` (column-major
# 2×2) → an uncertain COLMAP (x, y) point frame of the input image.
function _uncertain_point(undouble, r, c, Σ, upscale)
    Σxy = SMatrix{2, 2, Float64, 4}(Σ[4], Σ[3], Σ[2], Σ[1]) / upscale^2
    return Uncertain(PointFrame(undouble(r, c)), CovMat(Σxy))
end

# ----- per-element math (plain functions, called inside the kernels) -----------

# 210–245: the window measures at one pixel from the integrated squared
# gradient (mrr, mcc, mrc): the masked candidate value and the value its
# regional maxima are ranked by (`det/tr` for Förstner, the smaller eigenvalue
# for Köthe). `tiny` is MATLAB's `eps` (Float64's, as the reference).
@inline function _window_measure(mrr::T, mcc::T, mrc::T, koethe::Bool, precision_floor::T, roundness_threshold::T) where {T}
    tiny = T(eps(Float64))
    trace = mrr + mcc
    det = mrr * mcc - mrc^2
    w = det / (trace + tiny)
    if koethe
        μ2 = (trace - sqrt((mrr - mcc)^2 + 4 * mrc^2)) / 2
        return ifelse(w > precision_floor, μ2, zero(T)), μ2
    end
    roundness = 4 * det / (trace^2 + tiny)
    return ifelse((roundness > roundness_threshold) & (w > precision_floor), w, zero(T)), w
end

# 352–423 at one window centre (rw, cw), half-width ρ, on the squared gradient
# `gamma` (H × W × 3: grr, gcc, grc), pixels weighed equally. The sums run in
# coordinates relative to the centre (the estimates are shift-equivariant;
# relative sums keep Float32 accurate). Returns the 14 numbers
# (corner r, c, vec(N⁻¹), Ω, dot r, c, vec(N⁻¹), Ω), positions 1-based.
@inline function _window_estimate(gamma, rw::Integer, cw::Integer, ρ::Integer)
    T = eltype(gamma)
    srr = scc = src = zero(T)
    srr_r = srr_c = scc_r = scc_c = src_r = src_c = zero(T)
    for dc in (-ρ):ρ, dr in (-ρ):ρ
        @inbounds a, b, d = gamma[rw + dr, cw + dc, 1], gamma[rw + dr, cw + dc, 2], gamma[rw + dr, cw + dc, 3]
        r, c = T(dr), T(dc)
        srr += a; scc += b; src += d
        srr_r += a * r; srr_c += a * c
        scc_r += b * r; scc_c += b * c
        src_r += d * r; src_c += d * c
    end
    corner_inverse = inv(SMatrix{2, 2, T, 4}(srr, src, src, scc))
    corner = corner_inverse * SVector(srr_r + src_c, src_r + scc_c)
    dot_inverse = inv(SMatrix{2, 2, T, 4}(scc, -src, -src, srr))
    dot = dot_inverse * SVector(scc_r - src_c, -src_r + srr_c)
    corner_residual = dot_residual = zero(T)
    for dc in (-ρ):ρ, dr in (-ρ):ρ
        @inbounds a, b, d = gamma[rw + dr, cw + dc, 1], gamma[rw + dr, cw + dc, 2], gamma[rw + dr, cw + dc, 3]
        er, ec = T(dr) - corner[1], T(dc) - corner[2]
        corner_residual += a * er^2 + 2 * d * er * ec + b * ec^2
        # line 423 as the reference has it: the rr term from the blob model's
        # residual, the rc and cc terms from the corner model's (see the docs)
        dot_residual += b * (T(dr) - dot[1])^2 - 2 * d * er * ec + a * ec^2
    end
    return (
        corner[1] + rw, corner[2] + cw, corner_inverse[1], corner_inverse[2], corner_inverse[3],
        corner_inverse[4], corner_residual, dot[1] + rw, dot[2] + cw, dot_inverse[1], dot_inverse[2],
        dot_inverse[3], dot_inverse[4], dot_residual,
    )
end

# ----- kernels ------------------------------------------------------------------

# MATLAB's imfilter: correlation, 'same' size, zero padding — along the rows
# (index 1) or the columns (index 2) of each slice k, with that slice's taps
# column `taps[:, k]` (odd length, centred).
@kernel function _correlate_rows!(out, @Const(a), @Const(taps))
    i, j, k = @index(Global, NTuple)
    H = size(a, 1)
    r = size(taps, 1) ÷ 2
    s = zero(eltype(out))
    for t in max(-r, 1 - i):min(r, H - i)
        @inbounds s += taps[t + r + 1, k] * a[i + t, j, min(k, size(a, 3))]
    end
    @inbounds out[i, j, k] = s
end

@kernel function _correlate_columns!(out, @Const(a), @Const(taps))
    i, j, k = @index(Global, NTuple)
    W = size(a, 2)
    r = size(taps, 1) ÷ 2
    s = zero(eltype(out))
    for t in max(-r, 1 - j):min(r, W - j)
        @inbounds s += taps[t + r + 1, k] * a[i, j + t, k]
    end
    @inbounds out[i, j, k] = s
end

# 196–199: (∂r g)², (∂c g)², ∂r g ∂c g from the gradient's row and column components.
@kernel function _squared_gradient!(gamma, @Const(grad_r), @Const(grad_c))
    i, j = @index(Global, NTuple)
    @inbounds begin
        gr, gc = grad_r[i, j], grad_c[i, j]
        gamma[i, j, 1] = gr * gr
        gamma[i, j, 2] = gc * gc
        gamma[i, j, 3] = gr * gc
    end
end

@kernel function _window_measures!(candidates, ranking, @Const(m), koethe::Bool, precision_floor, roundness_threshold)
    i, j = @index(Global, NTuple)
    @inbounds candidates[i, j], ranking[i, j] =
        _window_measure(m[i, j, 1], m[i, j, 2], m[i, j, 3], koethe, precision_floor, roundness_threshold)
end

# imregionalmax, 8-connected, for the positive values (a zero plateau is a
# regional maximum only when it is the whole image, which the reference
# discards, lines 255–258): a pixel starts as a maximum when no neighbour is
# greater, and `_plateau_maxima_spread!` takes it off while an equal neighbour
# is off, until nothing changes.
@kernel function _plateau_maxima_init!(is_max, @Const(a))
    i, j = @index(Global, NTuple)
    H, W = size(a)
    @inbounds v = a[i, j]
    m = v > zero(v)
    for dj in -1:1, di in -1:1
        ii, jj = i + di, j + dj
        @inbounds if 1 <= ii <= H && 1 <= jj <= W && a[ii, jj] > v
            m = false
        end
    end
    @inbounds is_max[i, j] = m
end

@kernel function _plateau_maxima_spread!(is_max, changed, @Const(a))
    i, j = @index(Global, NTuple)
    H, W = size(a)
    @inbounds if is_max[i, j]
        v = a[i, j]
        for dj in -1:1, di in -1:1
            ii, jj = i + di, j + dj
            if 1 <= ii <= H && 1 <= jj <= W && a[ii, jj] == v && !is_max[ii, jj]
                is_max[i, j] = false
                changed[1] = Int32(1)
            end
        end
    end
end

# 228, 243, 268–281: a regional maximum of positive rank, inside the border,
# not below any rank of the regional maxima within `radius`.
@kernel function _second_suppression!(keep, @Const(is_max), @Const(ranking), radius::Int32, border::Int32)
    i, j = @index(Global, NTuple)
    H, W = size(ranking)
    @inbounds v = ifelse(is_max[i, j], ranking[i, j], zero(eltype(ranking)))
    k = (v > zero(v)) & (border < i < H - border) & (border < j < W - border)
    if k
        for dj in (-radius):radius, di in (-radius):radius
            @inbounds if is_max[i + di, j + dj] && ranking[i + di, j + dj] > v
                k = false
            end
        end
    end
    @inbounds keep[i, j] = k
end

@kernel function _window_estimates!(out, @Const(gamma), @Const(centres), radii)
    k = @index(Global)
    @inbounds e = _window_estimate(gamma, centres[1, k], centres[2, k], _per_window(radii, k))
    for n in 1:14
        @inbounds out[n, k] = e[n]
    end
end

# ----- the Gaussian gradient ---------------------------------------------------------

"""
    gaussian_gradient(img::AbstractMatrix{<:Real}, σ[, mask]; derivative = :gaussian2d) -> (gx, gy)

The gradient of `img` at scale `σ` (pixels): `gx` the derivative along the
columns (COLMAP x, ∂/∂column), `gy` along the rows (y, ∂/∂row), each the
image's size, in its element type (an integer image's `float`), on its device
(the image's KernelAbstractions backend: a host array on the CPU, a GPU array on
its GPU — one kernel for both). Per pixel, so a step of height 1 gives a peak
of ≈ 1/(√(2π)σ) per pixel for `:gaussian2d`.

The derivative of a Gaussian, correlated along one axis: the taps
`x·exp(−x²/2σ²)/(√(2π)σ³)` on radius `⌈3.3σ⌉`; along the other axis
`derivative = :gaussian2d` smooths with the unit-sum Gaussian of the same σ and
radius, `:gaussian1d` not at all. Zero padding at the image border (MATLAB's
`imfilter`): the Förstner operator's gradient ([`detect_foerstner`](@ref) calls
this one), bit for bit.

With a `mask` (an `H × W` array of weights, `0` outside the support — a
fisheye's image circle) it honours [`gaussian_blur`](@ref)'s semantics: the
gradient of the NORMALIZED convolution `blur(img·mask) / blur(mask)` (the blur
the across-axis Gaussian), `(∂N·D − N·∂D) / D²` with `N` and `∂N` of
`img·mask`, `D` and `∂D` of `mask`; pixels outside the mask do not bleed in, and
the gradient is `0` where the mask is zero.
"""
function gaussian_gradient(img::AbstractMatrix{<:Real}, σ::Real, mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing;
                           derivative::Symbol = :gaussian2d)
    isfinite(σ) && σ > 0 || throw(ArgumentError("gaussian_gradient: σ must be positive and finite, got $σ"))
    derivative in (:gaussian2d, :gaussian1d) ||
        throw(ArgumentError("gaussian_gradient: derivative must be :gaussian2d or :gaussian1d, got :$derivative"))
    H, W = size(img)
    mask === nothing || size(mask) == (H, W) ||
        throw(DimensionMismatch("gaussian_gradient: an $(H)×$(W) image and a $(join(size(mask), '×')) mask"))
    T = float(eltype(img))
    device = KA.get_backend(img)
    a = img isa AbstractMatrix{T} ? img : T.(img)
    d, c = _derivative_taps(derivative, σ)
    if mask === nothing                                     # ∂row then ∂column: the rows' taps (d, c), the columns' (c, d)
        g = _separable_correlate(device, reshape(a, H, W, 1), ka_upload(T, device, [d c]), ka_upload(T, device, [c d]))
        return view(g, :, :, 2), view(g, :, :, 1)
    end
    m = ka_upload(T, device, mask)
    # N, ∂rN, ∂cN of img·mask and D, ∂rD, ∂cD of mask: the rows' taps (c, d, c), the columns' (c, c, d) — the blur across
    # the derivative is the across taps (the Gaussian, or the identity for :gaussian1d), the blur itself the across taps
    rows, cols = ka_upload(T, device, [c d c]), ka_upload(T, device, [c c d])
    am = similar(a, H, W, 1)
    _launch!(_masked_product!, device, am, a, m; ndrange = (H, W))
    nm = _separable_correlate(device, am, rows, cols)
    dm = _separable_correlate(device, reshape(m, H, W, 1), rows, cols)
    g = similar(a, H, W, 2)
    _launch!(_normalized_gradient!, device, g, nm, dm, m; ndrange = (H, W))
    return view(g, :, :, 2), view(g, :, :, 1)
end

# `a` (H × W × 1, or × K) correlated along the rows with each column of `rows` and then along the columns with the same
# column of `cols`: one H × W × K slice per taps column.
function _separable_correlate(device, a::AbstractArray{T, 3}, rows, cols) where {T}
    H, W = size(a, 1), size(a, 2)
    tmp = KA.allocate(device, T, H, W, size(rows, 2))
    out = similar(tmp)
    _launch!(_correlate_rows!, device, tmp, a, rows; ndrange = size(tmp))
    _launch!(_correlate_columns!, device, out, tmp, cols; ndrange = size(out))
    return out
end

@kernel function _masked_product!(out, @Const(a), @Const(m))
    i, j = @index(Global, NTuple)
    @inbounds out[i, j, 1] = a[i, j] * m[i, j]
end

# The gradient of N/D from (N, ∂rN, ∂cN) and (D, ∂rD, ∂cD): (∂N·D − N·∂D)/D², 0 where the mask or D is 0 (gaussian_blur's).
@kernel function _normalized_gradient!(g, @Const(n), @Const(d), @Const(m))
    i, j = @index(Global, NTuple)
    T = eltype(g)
    @inbounds N, Nr, Nc, D, Dr, Dc = n[i, j, 1], n[i, j, 2], n[i, j, 3], d[i, j, 1], d[i, j, 2], d[i, j, 3]
    inside = (D > zero(T)) & (m[i, j] > zero(T))
    @inbounds g[i, j, 1] = ifelse(inside, (Nr * D - N * Dr) / (D * D), zero(T))
    @inbounds g[i, j, 2] = ifelse(inside, (Nc * D - N * Dc) / (D * D), zero(T))
end

# ----- taps -----------------------------------------------------------------------

# 617–650: the derivative taps x·exp(−x²/2σ²)/(√(2π)σ³) on radius ⌈3.3σ⌉ (the
# package's one Gaussian taps builder, at truncation 3.3) and the taps across
# the derivative: for `:gaussian2d` the unit-sum Gaussian of the same σ and
# radius (fspecial's), for `:gaussian1d` the identity (zero padded to the same
# length, so the two share one taps matrix).
function _derivative_taps(derivative::Symbol, σ::Real)
    full = _symmetric_profile(Float64(σ), 3.3)
    r = length(full) ÷ 2
    derivative_taps = [x * full[x + r + 1] / (sqrt(2π) * σ^3) for x in -r:r]
    across = derivative === :gaussian2d ? full ./ sum(full) : [x == 0 ? 1.0 : 0.0 for x in -r:r]
    return derivative_taps, across
end

# 656–678: the integration taps per axis (the 2D filter is their outer product:
# fspecial's 'average' box, or its normalized 'gaussian' on radius ⌈3.3σ⌉) and
# their radius.
function _integration_taps(integration::Symbol, σ::Real)
    if integration === :gaussian
        full = _symmetric_profile(Float64(σ), 3.3)
        return full ./ sum(full), length(full) ÷ 2
    end
    r = ceil(Int, sqrt(12σ^2 + 1) / 2)
    return fill(1 / (2r + 1), 2r + 1), r
end

_symmetric_profile(σ, truncation) = (g = _gaussian_profile(σ, truncation); [reverse(g[2:end]); g])
