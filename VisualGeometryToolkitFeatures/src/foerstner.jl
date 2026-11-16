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
`Detection(Uncertain(PointFrame(x), Σ), (; label, window, F, covariance_status))`
(VisualGeometryToolkitCore's `PointFrame`: a position, no scale):

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
(the reference computes in `Float64`; supply `Float64` for that): `Float32` or
`Float64`, any other refused (an `ArgumentError`): in `Float16` the squared
gradient's window sums would accumulate in it. A Float16-stored image (`LinearRaw(Float16, …)`)
is read in `T`.

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
   padding, as `imfilter`: [`gaussian_gradient`](@ref)), its squares and product (the squared gradient),
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
5. `F = Ω_corner / Ω_circular` against the F distribution of the two residual
   sums' EFFECTIVE degrees of freedom `(ν_corner, ν_circular)`, as the reference's
   `(T < F(α)) − (T > F(1 − α))` (line 429): a corner below the
   `1 − significance_level` quantile, a circular point above the
   `significance_level` quantile, unknown in between. `ν = (tr Q)²/tr Q²`
   (Satterthwaite), `Ω = Iᵀ Q I` the residual sum as a quadratic form of the
   input pixels under white noise, for the window's half-width and the
   derivative taps, about the window centre (difference 4 below).
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
4. **The F-test's degrees of freedom — corrected.** The reference tests `F`
   against `F(redundancy, redundancy)`, the window's doubled-pixel count less 2,
   as if every residual were independent. The doubled samples are not: the
   doubling and the derivative taps correlate them, and each residual sum has
   the effective degrees of freedom `ν` of step 5, about 1/23 of the pixel count
   (ν ≈ 87 against 2023 at ρ = 22, `derivative_sigma` 1). At the pixel count the
   test's level is not the stated one: on white noise about 20 % of the windows
   were labelled circular and 25 % corner at `significance_level` 0.999 (the
   suite's test), where `ν` gives at most the stated rate. On the chrome
   photographs the windows between dots, on line artefacts and on letter strokes
   (F ≈ 1–3) were labelled circular. The a-posteriori covariance keeps the
   reference's `Ω/redundancy · N⁻¹` (its calibration, NEES ≈ 2, is the
   Förstner study's).

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

# A seed of the window estimation: a detection of a frame with a scale, its window's integration scale (a point has
# none). An uncertain one is read through its frame (`Disc` of the detection), as `Disc(d)` reads any uncertain frame.
# Checked per seed (`_check_seeds`), so a vector of an abstract element type (`Detection[]`, mixed frames) is a seed vector.
const _ScaledFrame = Union{Disc, OrientedDisc, EllipticFrame, AffineFrame}
const _SeedDetection = Detection{<:Union{_ScaledFrame, Uncertain{<:_ScaledFrame, <:CovMat}}}

"""
    detect_foerstner([T = Float32], img, seeds::AbstractVector{<:Detection}; device = default_device(),
                     noise = nothing, derivative = :gaussian2d, derivative_sigma = 1.0,
                     significance_level = 0.999, polarity = :none, f_tests = nothing) -> StructArray{<:Detection}
    detect_foerstner([T = Float32], L::LinearRaw, seeds; sensor_model::SensorModel = SensorModel(), kwargs...)

The Förstner operator's window estimation (steps 4–6 of the method above, its
doubled resolution, its kernels) at GIVEN windows: one window per seed, centred
on the doubled-grid pixel nearest the seed's centre, its integration scale the
seed's characteristic scale σ (input pixels; a [`detect_sfop`](@ref) row's
selected σ). The box window has the standard deviation σ, the estimation window
the half-width `⌈√(12(2σ)² + 1)/2⌉ + 1` doubled pixels, as `integration_sigma = σ`
would give: SFOP's `N(σ) = 12σ² + 1` is the pixel count of the box of the same
energy as its Gaussian `G_σ`. The image-wide window search (step 3) is not run.

On a RAW's green cells (`L::LinearRaw`, positions on the cell grid; seeds from
`detect_sfop(L)`) the frame's `sensor_model` gives the covariance as for the
image-wide call on `L`.

The seeds are detections of a frame with a scale (a `Disc`, an `OrientedDisc`, an `EllipticFrame` or
an `AffineFrame`, its characteristic scale σ; each also with its covariance); a point has no
window (an `ArgumentError` naming its frame). The keywords are the ones above: the image-wide search's (`method`, `noise_sigma`,
`integration`, `integration_sigma`, `precision_threshold`, `roundness_threshold`) are not read at given windows and
are refused (`ArgumentError`, naming them).

`polarity = :dark | :light` keeps only the seeds whose blob has that sign, [`detect_sfop`](@ref)'s test: the
scale-normalised Laplacian of Gaussian at the seed's σ, its sign significant at `significance_level` under the pixel
noise (`noise` on a matrix, the gain-carrying `sensor_model` on a `LinearRaw`; with none known the call is refused).
`:none` (the default) keeps every seed.

`f_tests`: a [`FoerstnerTests`](@ref) table the F-test's critical values are read from (each window half-width's
computed once, on its first use): pass one table to every call of a configuration (a set of photographs), else each
call computes them again on the host. The rows are the same either way.

Returns the rows of `detect_foerstner(img)` (label, frame and covariance,
window, F; the F-test of each window at its own half-width's degrees of freedom), one per kept seed whose
window lies inside the doubled image, in the order of `seeds`. Two seeds of one blob (a dot SFOP sees at two
scales) give two rows: duplicates are the consumer's to resolve (its own geometry, e.g. an assignment to a target).
"""
detect_foerstner(img::Union{AbstractMatrix, LinearRaw}, seeds::AbstractVector{<:Detection}; kw...) =
    detect_foerstner(Float32, img, seeds; kw...)

detect_foerstner(::Type{T}, img::AbstractMatrix, seeds::AbstractVector{<:Detection}; noise::Union{Nothing, Real} = nothing,
                 kw...) where {T <: AbstractFloat} = _detect_seeded(T, img, _pixel_noise(noise), seeds; kw...)

detect_foerstner(::Type{T}, L::LinearRaw; sensor_model::SensorModel = SensorModel(), kw...) where {T <: AbstractFloat} =
    _detect_foerstner(T, _raw_inputs(L, sensor_model)...; kw...)

# The seeded windows on a RAW's green cells (positions on the cell grid), under the frame's sensor model as the image-wide call.
detect_foerstner(::Type{T}, L::LinearRaw, seeds::AbstractVector{<:Detection}; sensor_model::SensorModel = SensorModel(),
                 kw...) where {T <: AbstractFloat} = _detect_seeded(T, _raw_inputs(L, sensor_model)..., seeds; kw...)

"""
    FoerstnerTests()

A table of the Förstner F-test's critical values ([`detect_foerstner`](@ref)'s `f_tests` keyword): each window
half-width's pair (the corner's and the circular point's critical `F` at the significance level, from the residual sums'
effective degrees of freedom) is computed on its first use and read by every later call. The values depend on the
configuration only — the gradient taps (`derivative`, `derivative_sigma`), the half-width ρ, the pixel noise's
structure (white input pixels, or the photosites through a mosaic's bilinear green image) and `significance_level` —
and each entry is keyed by all four, so one table serves calls of different settings and the rows are bit-identical to
a call without one. The caller makes one per detect configuration and passes it to each of its calls (a set of
photographs: the host computation was most of a seeded call's time). Safe to share between threads.
"""
struct FoerstnerTests
    critical::Dict{Tuple{Matrix{Float64}, Int, Any, Float64}, NTuple{2, Float64}}   # (taps, ρ, noise, level) => critical F
    lock::ReentrantLock
    FoerstnerTests() = new(Dict(), ReentrantLock())   # converted to the field's type: the key spelled once
end
Base.length(t::FoerstnerTests) = lock(() -> length(t.critical), t.lock)
Base.isempty(t::FoerstnerTests) = length(t) == 0
Base.show(io::IO, t::FoerstnerTests) = print(io, "FoerstnerTests(", length(t), " windows)")

# The F-test's two critical values at half-width ρ (`_foerstner_classes`' corner and circular thresholds): computed, or
# read from (and on a miss added to) the table `f_tests`.
function _f_critical(::Nothing, taps, ρ, noise, significance_level)
    d = _residual_dof(taps, ρ, noise)
    f = FDist(d.corner.ν, d.circular.ν)
    return (quantile(f, significance_level), quantile(f, 1 - significance_level))
end
_f_critical(t::FoerstnerTests, taps, ρ, noise, significance_level) =
    lock(t.lock) do
        get!(t.critical, (Matrix{Float64}(taps), Int(ρ), noise, Float64(significance_level))) do
            _f_critical(nothing, taps, ρ, noise, significance_level)
        end
    end

# The seeded call's keywords, once for both inputs (the pixel noise already resolved): the image-wide search's are not
# read at given windows, and a caller who gives one expects an effect it cannot have, so they are refused by name.
function _detect_seeded(
        ::Type{T}, img, noise, seeds; device::KA.Backend = default_device(), derivative::Symbol = :gaussian2d,
        derivative_sigma::Real = 1.0, significance_level::Real = 0.999, polarity::Symbol = :none,
        f_tests::Union{Nothing, FoerstnerTests} = nothing, unread...
    ) where {T}
    _refuse_unread_seed_keywords(unread)
    _check_seeds(seeds)
    return _detect_foerstner(T, img, noise; seeds, device, derivative, derivative_sigma, significance_level, polarity, f_tests)
end

_refuse_unread_seed_keywords(unread) = isempty(unread) || throw(ArgumentError(
    "detect_foerstner(img, seeds): $(join(string.(keys(unread)), ", ")) not read by the seeded call (each window's " *
    "scale is its seed's σ): it takes device, derivative, derivative_sigma, significance_level, polarity, f_tests and the pixel noise " *
    "(noise on a matrix, sensor_model on a LinearRaw)"))

# Every seed a detection of a frame with a scale (`_SeedDetection`): a point's window has no scale.
function _check_seeds(seeds)
    for d in seeds
        d isa _SeedDetection || throw(ArgumentError(
            "detect_foerstner(img, seeds): a seed of a $(_frame_name(getfield(d, :value))) has no scale, so no window: " *
            "the seeds are detections of a Disc, OrientedDisc, EllipticFrame or AffineFrame (each also uncertain)"))
    end
    return seeds
end
_frame_name(v::Attributed) = _frame_name(getfield(v, :value))   # an uncertain frame: its frame
_frame_name(v) = nameof(typeof(v))

# A RAW's green cells as the detector reads them: their values and the pixel noise its sensor model gives (`_pixel_noise`).
_raw_inputs(L::LinearRaw, sensor_model::SensorModel) = (G = green_plane(L); (_green_values(G), _pixel_noise(sensor_model.noise, G)))

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
        seeds::Union{Nothing, AbstractVector{<:Detection}} = nothing,
        polarity::Symbol = :none,
        f_tests::Union{Nothing, FoerstnerTests} = nothing
    ) where {T <: AbstractFloat}
    _check_precision(T)
    0 < significance_level < 1 ||
        throw(ArgumentError("detect_foerstner: significance_level must lie in (0, 1), got $significance_level"))
    sign = _polarity_sign(polarity)
    seeds === nothing && sign != 0 && throw(ArgumentError(
        "detect_foerstner: polarity gates given windows (detect_foerstner(img, seeds; polarity)), not the image-wide search"))
    g = _detection_image(T, device, img)
    seeds === nothing || sign == 0 || (seeds = _polarity_seeds(device, g, seeds, sign, model, significance_level))
    doubled = KA.allocate(device, T, 2 .* size(g)...)
    _launch!(_upsample_doubled!, device, doubled, reshape(g, size(g)..., 1); ndrange = size(doubled))
    found = seeds === nothing ? _foerstner_windows(
        device, doubled; method, noise_sigma, derivative, integration, derivative_sigma,
        integration_sigma, precision_threshold, roundness_threshold
    ) : _foerstner_seeded(device, doubled, seeds; derivative, derivative_sigma)
    classes = _foerstner_classes(found, significance_level, _noise_terms(model), f_tests)
    _noise_known(model) || return _foerstner_points(found, classes, _undouble)
    covariances = _propagated_covariances(device, found, classes, _device_noise(T, device, model, size(g))...)
    return _foerstner_points(found, classes, _undouble; covariances, covariance_status = _noise_status(model))
end

# The pixel noise of the bilinear green image on a mosaic's raw grid (raw_green.jl), read from its photosites: `support`
# (each pixel's averaged greens, `_bilinear_green!`'s plane) and `photosite` (σ_k², 0 at every site but a usable green), device planes
# of the image's size, `density` its white-equivalent per-pixel variance (the polarity test's), `status` the covariance
# status — `nothing` when the sensor model carries no gain (the a-posteriori covariance, as on a LinearRaw). Its F-test
# reads the photosite noise's structure in any case.
struct _BilinearGreenNoise{C <: AbstractMatrix{UInt8}, P <: AbstractMatrix{Float32}}
    support::C
    photosite::P
    density::P
    status::Union{Nothing, Symbol}
end

# The pixel noise `model` read four ways: whether it is known (else the a-posteriori covariance), its covariance status,
# the noise structure of the F-test's residual sums (`_residual_dof`'s `noise`), and the propagation's inputs on the
# device (`_propagated_covariances`' last two arguments: the variances and, for photosite noise, the pixels' supports).
_noise_known(model) = model !== nothing
_noise_known(n::_BilinearGreenNoise) = n.status !== nothing
_noise_status(model::Tuple) = last(model)
_noise_status(n::_BilinearGreenNoise) = n.status
_noise_terms(model) = _WhiteNoise()
_noise_terms(::_BilinearGreenNoise) = _BilinearGreenStructure()
_device_noise(::Type{T}, device, (variance, _)::Tuple, dims) where {T} =
    (variance isa Real ? fill!(KA.allocate(device, T, dims...), T(variance)) : ka_upload(T, device, variance), nothing)
_device_noise(::Type{T}, device, n::_BilinearGreenNoise, dims) where {T} = (n.photosite, n.support)   # on the device already

# The detector's image on `device`, in `T`: a float matrix at its own values (gray_f32 would round a Float64 one; a
# Float16 one is read in `T`), anything else as gray_f32 reads it. The Förstner doubling's and the polarity gate's.
_detection_image(::Type{T}, device, img::AbstractMatrix) where {T} =
    ka_upload(T, device, img isa AbstractMatrix{<:AbstractFloat} ? img : gray.(gray_f32(img)))

# The seeds whose blob has the requested polarity (`_polarity_kept` on the detection image `g`), in their order: the
# sign of the scale-normalised Laplacian at the seed's own σ (its window's scale; SFOP's selection), significant at
# `significance_level` under the pixel noise `model` (`_pixel_noise`'s; none known is refused).
function _polarity_seeds(device, g, seeds, sign::Real, model, significance_level)
    _noise_known(model) || throw(ArgumentError(
        "detect_foerstner(img, seeds; polarity): the polarity's sign test needs the pixel noise: `noise` on a matrix, " *
        "a gain-carrying `sensor_model` on a LinearRaw or a RawMosaic"))
    isempty(seeds) && return seeds
    variance = _polarity_variance(model)
    points = stack((c = Disc(d); Float64[c.center[1], c.center[2], c.σ]) for d in seeds; dims = 2)
    keep = _polarity_kept(device, g, points, sign, variance isa Real ? variance : ka_upload(Float32, device, variance),
                          significance_level)
    return seeds[keep]
end

# The polarity test's per-pixel noise variance: the input pixels' (independent), or the bilinear image's white-equivalent.
_polarity_variance(model::Tuple) = first(model)
_polarity_variance(n::_BilinearGreenNoise) = n.density

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
_pixel_noise(n::NoiseComponent, G::AbstractMatrix) = nothing
_pixel_noise(n::PoissonGaussian, G::AbstractMatrix{<:Uncertain}) = map(var, G), _propagated_status(n)
# The covariance status of a propagated covariance under the noise slot: only a gain-carrying (Poisson–Gaussian) model is
# fed (the owner's ruling A), named by its source; `nothing` for any other.
_check_precision(::Type{T}) where {T} = (T === Float32 || T === Float64) || throw(ArgumentError(
    "detect_foerstner: the precision T must be Float32 or Float64, got $T (Float16 would accumulate the window sums in it; a Float16 image is read in T)"))
_propagated_status(::NoiseComponent) = nothing
_propagated_status(n::PoissonGaussian) = Symbol("propagated_", string(n.source))
_pixel_noise(::PoissonGaussian, G::AbstractMatrix) = throw(ArgumentError(
    "the sensor model has a gain but the LinearRaw holds no variances: linearise it with the same model, LinearRaw(mosaic; mode, sensor_model) or load_raw(…; sensor_model)"))

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
        centres, estimates = Float64.(Array(device_estimates)), ρ, radii, redundancy = (2 .* ρ .+ 1) .^ 2 .- 2,   # ρ on the host, radii on the device
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
# 0 unknown (the reference's noclass). The F distribution's degrees of freedom are the two residual sums' effective
# ones (`_residual_dof`, one per window half-width ρ), not the reference's pixel count `(2ρ + 1)² − 2`.
function _foerstner_classes(found, significance_level, noise = _WhiteNoise(), f_tests = nothing)
    tests = Dict(ρ => _f_critical(f_tests, found.taps, ρ, noise, significance_level) for ρ in unique(found.ρ))
    function class(k, e)
        corner_critical, circular_critical = tests[_per_window(found.ρ, k)]
        return (e[7] / e[14] < corner_critical) - (e[7] / e[14] > circular_critical)
    end
    return Int8[class(k, e) for (k, e) in enumerate(eachcol(found.estimates))]
end

# The effective degrees of freedom of the window's two residual sums under white input noise (Satterthwaite):
# `Ω = Σₚ (aₚᵀ∇g(p))²` over the (2ρ + 1)² doubled pixels p of the window, `aₚ = e` (the corner model: the distance
# across the gradient) or `R e` (the circular point: along it), `e = p − c` about the window centre c, is the quadratic
# form `Ω = Iᵀ Q I` of the input pixels I, so for iid noise of unit variance `E[Ω] = tr Q`, `Var Ω = 2 tr Q²` and
# `ν = (tr Q)²/tr Q²`. The doubled samples are correlated (the doubling, the derivative and across taps), so ν is far
# below the pixel count. The gradient's components are separable — the rows' and the columns' one-axis operators
# (`_pipeline_bands` of `taps`, `_propagated_covariances`' convention: ∂/∂row reads (derivative, across), ∂/∂column
# (across, derivative)) — so their covariance is `K_ss′(p, p′) = Kr_ss′(r, r′)·Kc_ss′(c, c′)`, and with `aₚ` linear in
# one coordinate each `aₚᵀ K(p, p′) aₚ′` is a sum of four products `f_t(r, r′)·g_t(c, c′)`: `tr Q = Σ_t tr f_t tr g_t`,
# `tr Q² = Σ_tu ⟨f_t, f_u⟩⟨g_t, g_u⟩`, O(ρ²). The sums are about the window centre: the estimate's two parameters, which the
# detector's Ω removes, are neglected (about 2 of ν; Monte Carlo on white noise: Ω_corner's measured ν 89 against 89.8, Ω_circular's
# 77 against 87, the latter line 423's sum — the test is conservative). Interior windows (every window is inside the image); the
# doubling's phase (the centre's parity) changes ν by < 1e-5 relative, so the centre's parity is whichever n gives.
# Returns `(; corner = (; ν, E), circular = (; ν, E))`, `E = tr Q`.
#
# `noise`: the input noise's structure, `I = Σ_j (A_j ⊗ B_j) n` for white unit noise n (A_j acting along the rows, B_j
# along the columns; `_noise_factors(noise, n, ε)` the pairs on an axis of n pixels at the lattice phase ε,
# `_noise_reach(noise)` their reach in pixels), so each gradient covariance is `Σ_jj′ Kr_{sj,s′j′} ⊗ Kc_{sj,s′j′}` with
# `Kr_{sj,s′j′} = Dₛᵀ A_j A_j′ᵀ Dₛ′`, and the moments run over the pairs (j, j′) as over (s, s′): with the terms'
# matrices as the columns of F and G, `tr Q = Σ_t tr f_t tr g_t` and `tr Q² = Σ ((FᵀF) .* (GᵀG))`. White input noise is
# the one pair (I, I) (`_WhiteNoise`); the bilinear green image's photosite noise four (`_BilinearGreenStructure`), at
# each lattice phase, the smaller ν kept (the conservative test).
function _residual_dof(taps::AbstractMatrix, ρ::Integer, noise = _WhiteNoise())
    # each axis' centre parity on its own (a seeded window's doubled centre is rounded per axis), every lattice phase
    d = [_residual_dof(taps, ρ, noise, (cr, cc), ε) for cr in _noise_centres(noise) for cc in _noise_centres(noise)
         for ε in _noise_phases(noise)]
    return (; corner = argmin(c -> c.ν, getproperty.(d, :corner)), circular = argmin(c -> c.ν, getproperty.(d, :circular)))
end
# One window centre (`c = (c_row, c_column)`: each shifts that axis' centre by a doubled pixel, on an input pixel or
# between two) and one lattice phase ε.
function _residual_dof(taps::AbstractMatrix, ρ::Integer, noise, c::NTuple{2, Integer}, ε)
    r = size(taps, 1) ÷ 2
    base = ρ + r + 4 + _noise_reach(noise)                  # input pixels: the window, the taps' and the noise's reach, interior
    nr, nc = base + c[1], base + c[2]
    x = Float64.(collect(-ρ:ρ))                             # e along either axis
    function bands(B, n)                                    # Dₛ[q, i], the weight of input q at the window's i-th sample
        span = (n - ρ):(n + ρ)                              # the window along the axis, doubled 1-based, centre n
        D = zeros(n, length(span), 2)
        for (i, p) in enumerate(span), s in 1:2, k in axes(B, 1)
            D[_band_start(p, r) + k - 1, i, s] += B[k, p, s]
        end
        return D
    end
    Dr, Dc = bands(_pipeline_bands(Float64.(taps), nr), nr), bands(_pipeline_bands(Float64.(taps[:, [2, 1]]), nc), nc)
    function grams(D, factors)                              # K_{sj,s′j′} = (A_jᵀDₛ)ᵀ(A_j′ᵀDₛ′)
        E = [A' * D[:, :, s] for s in 1:2, A in factors]
        return [E[s, j]' * E[s2, j2] for s in 1:2, j in eachindex(factors), s2 in 1:2, j2 in eachindex(factors)]
    end
    on(axis, along) = axis == along ? x : ones(length(x))
    # aₚᵀ K aₚ′ = Σ_ss′jj′ f(r, r′) g(c, c′) for `a` = (sign, axis) of each gradient component (∂/∂row, ∂/∂column), axis 1
    # the row offset, 2 the column offset: the corner a = (r, c), the circular point a = (−c, r)
    function moments(a, Kr, Kc, J)
        idx = [(s, j, s2, j2) for s in 1:2 for s2 in 1:2 for j in J for j2 in J]
        F = reduce(hcat, [vec(a[s][1] * a[s2][1] .* on(a[s][2], 1) .* on(a[s2][2], 1)' .* Kr[s, j, s2, j2]) for (s, j, s2, j2) in idx])
        G = reduce(hcat, [vec(on(a[s][2], 2) .* on(a[s2][2], 2)' .* Kc[s, j, s2, j2]) for (s, j, s2, j2) in idx])
        diag_ = diagind(length(x), length(x))
        E = sum(sum(view(F, diag_, t)) * sum(view(G, diag_, t)) for t in axes(F, 2))
        V = sum((F' * F) .* (G' * G))
        return (; ν = E^2 / V, E)
    end
    rows_, cols_ = _noise_factors(noise, nr, ε), _noise_factors(noise, nc, ε)   # one pair list, each axis at its own n
    Kr, Kc, J = grams(Dr, first.(rows_)), grams(Dc, last.(cols_)), eachindex(rows_)
    return (; corner = moments(((1, 1), (1, 2)), Kr, Kc, J), circular = moments(((-1, 2), (1, 1)), Kr, Kc, J))
end

# White input noise: one pair, the identity along both axes, one phase.
struct _WhiteNoise end
_noise_reach(::_WhiteNoise) = 0
_noise_phases(::_WhiteNoise) = (1,)
_noise_centres(::_WhiteNoise) = (0,)                       # the doubling's centre parity changes ν by < 1e-5 under white noise
_noise_factors(::_WhiteNoise, n, ε) = [(I, I)]

# The bilinear green image's noise (raw_green.jl) from white photosite noise n (unit variance; masked to the green
# sites): `Ī = M_g n + M_rb X M_g n`, M_g = ½(1⊗1 + ε P⊗P) the green sites (P = diag((−1)^i), ε = ±1 the lattice's
# phase), M_rb = 1 − M_g, X = ¼(S⊗1 + 1⊗S) the cross's neighbours (S the ±1 shift); with PSP = −S and SP = −PS it is
# `(½ + ⅛S)⊗1 + ε(½P + ⅛SP)⊗P + 1⊗⅛S + εP⊗⅛SP`: four pairs, reach 1. Both phases (a window centred on a green or on a
# red or blue site) at both doubled centre parities (on an input pixel or between two), the smallest ν kept.
struct _BilinearGreenStructure end
_noise_reach(::_BilinearGreenStructure) = 1
_noise_phases(::_BilinearGreenStructure) = (1, -1)
_noise_centres(::_BilinearGreenStructure) = (0, 1)         # the checkerboard makes ν depend on the centre's parity too
function _noise_factors(::_BilinearGreenStructure, n, ε)
    P = Diagonal([(-1.0)^i for i in 1:n])
    S = diagm(1 => ones(n - 1), -1 => ones(n - 1))
    return [(0.5I + S / 8, I), (ε * (P / 2 + S * P / 8), P), (I, S / 8), (ε * P, S * P / 8)]
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
# (row, column); NaN for an unknown window) through the taps `_foerstner_windows` ran: rows correlated with
# (derivative, across), columns with (across, derivative). `support === nothing`: independent input noise of variance
# `variance[q]` at input pixel q; else photosite noise through the bilinear green image (raw_green.jl), `variance[k]`
# the variance of photosite k (0 but at a usable green), `support[q]` pixel q's averaged greens (on the device of
# `found`).
function _propagated_covariances(device, found, classes, variance::AbstractMatrix, support = nothing)
    (; centres, radii, gradient_rows, gradient_columns, taps, centre_list, device_estimates) = found
    H, W = size(variance)
    K = length(centres)
    K == 0 && return zeros(Float64, 4, 0)
    T = eltype(device_estimates)
    rows = ka_upload(T, device, _pipeline_bands(taps, H))
    columns = ka_upload(T, device, _pipeline_bands(taps[:, [2, 1]], W))
    out = KA.allocate(device, T, 4, K)
    _launch!(
        _propagated_covariances!, device, out, gradient_rows, gradient_columns, variance, support, centre_list, device_estimates,
        ka_upload(Int8, device, classes), rows, columns, radii, Int32(size(taps, 1) ÷ 2); ndrange = K
    )
    return Float64.(Array(out))
end

# One window k: Σ = Σ_q variance[q] δ_q δ_qᵀ (independent input pixels) or Σ = Σ_k variance[k] δ_k δ_kᵀ over the
# usable green photosites k, δ_k = Σ_q δ_q / n_q over the pixels q whose support holds k (the bilinear weight of k at
# q, n_q its count: `_in_support`). The pixels' δ_q = N⁻¹ Σₚ A(p) ∂∇g(p)/∂I(q) (`_pixel_shift`).
@kernel function _propagated_covariances!(
        out, @Const(gradient_rows), @Const(gradient_columns), @Const(variance), @Const(support), @Const(centres), @Const(estimates),
        @Const(classes), @Const(rows), @Const(columns), radii, r::Int32
    )
    k = @index(Global)
    ρ = Int(_per_window(radii, k))
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
        window = (; gradient_rows, gradient_columns, rows, columns, rw, cw, ρ, r = Int(r), w, xr, xc, Ninv, dot = class == -1)
        reach = _noise_reach(support)
        for q2 in max(1, _band_start(cw - ρ, Int(r)) - reach):min(W, _band_start(cw + ρ, Int(r)) + w - 1 + reach),
                q1 in max(1, _band_start(rw - ρ, Int(r)) - reach):min(H, _band_start(rw + ρ, Int(r)) + w - 1 + reach)
            δ, v = _noise_shift(support, variance, window, q1, q2)
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

# The noise source at (q1, q2): its shift of the estimate and its variance. An input pixel (independent pixel noise),
# or a photosite of the bilinear image, whose shift is its weights' sum over the pixels under its cross (a site other
# than a usable green has variance 0 and is skipped).
@inline _noise_reach(::Nothing) = 0
@inline _noise_reach(::AbstractMatrix) = 1
@inline _noise_shift(::Nothing, variance, window, q1, q2) = (_pixel_shift(window, q1, q2), @inbounds variance[q1, q2])
@inline function _noise_shift(support::AbstractMatrix, variance, window, q1, q2)
    T = eltype(window.Ninv)
    H, W = size(support)
    δ = zero(SVector{2, T})
    @inbounds v = T(variance[q1, q2])
    v > zero(T) || return δ, v
    for d2 in -1:1, d1 in -1:1                              # the pixels whose support can hold k: its 3×3
        p1, p2 = q1 + d1, q2 + d2
        ((1 <= p1 <= H) & (1 <= p2 <= W)) || continue
        @inbounds s = support[p1, p2]
        (s != 0x00 && _in_support(s, d1, d2)) || continue
        δ += _pixel_shift(window, p1, p2) / T(_support_count(s))
    end
    return δ, v
end

# δ_q = N⁻¹ Σₚ A(p) ∂∇g(p)/∂I(q) of input pixel q = (q1, q2) in one window: the gradient's component s at the doubled
# pixel (pr, pc) reads q with weight rows[·, pr, s] · columns[·, pc, s], nonzero only for |p − 2q| ≤ r + 2 along each axis.
@inline function _pixel_shift(window, q1, q2)
    (; gradient_rows, gradient_columns, rows, columns, rw, cw, ρ, r, w, xr, xc, Ninv, dot) = window
    T = eltype(Ninv)
    s1 = s2 = zero(T)
    for pc in max(cw - ρ, 2q2 - w):min(cw + ρ, 2q2 + w), pr in max(rw - ρ, 2q1 - w):min(rw + ρ, 2q1 + w)
        k1 = q1 - _band_start(pr, r) + 1
        k2 = q2 - _band_start(pc, r) + 1
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
    return Ninv * SVector(s1, s2)
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
