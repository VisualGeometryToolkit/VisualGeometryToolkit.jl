# Shared fixtures for the keypoint-uncertainty items (uncertainty_tests.jl) and
# the full Monte Carlo study (benchmark/mc/keypoint_uncertainty_mc.jl, which includes
# this file outside ReTestItems).
@testsetup module UncFixtures

using LinearAlgebra: Symmetric, cholesky, inv, det, tr, diag
using Random: Xoshiro, randn!
using Statistics: mean, quantile
import KernelAbstractions as KA
using VisualGeometryFeatures
using VisualGeometryFeatures: AsiftPlan, asift_detect!, _functional_matrix, _unpack_M,
    _ift_cov_reference, asift_detections
using VisualGeometryCore: cov

export detect_on, blob_image, gauss_field, impulse_M, functionals_at, chi2_q, mc_run, mc_summary,
    hessian_at, extremum_of, hess3d_cov, zeisl_cov, zeisl_frobenius_constant, dog_blob_image,
    ZEISL_W5_SIGMA, Symmetric, mean, quantile, Xoshiro, det, tr, diag

# =============================================================================
# Synthetic images
# =============================================================================

"""
    blob_image(H, W, blobs; bg = 0.5) -> Matrix{Float64}

A float image (H×W, row = y) of isotropic Gaussian blobs `(x, y, σ, a)` given in
COLMAP pixel-centre coordinates: pixel `img[r, c]` has its centre at
`(c − 0.5, r − 0.5)`, so a blob at `(x, y)` peaks there. With `A` each blob is
`exp(−|A⁻¹(p − p₀)|²/2σ²)`: isotropic of std σ in the frame an ASIFT warp with
linear part `A` resamples to.
"""
function blob_image(H, W, blobs; bg = 0.5, background = nothing, A = [1.0 0.0; 0.0 1.0])
    img = background === nothing ? fill(Float64(bg), H, W) : copy(background)
    Ai = inv(A)
    for (x, y, s, a) in blobs
        R = ceil(Int, 6s * maximum(abs, A))
        c0 = floor(Int, x + 0.5); r0 = floor(Int, y + 0.5)
        for c in max(1, c0 - R):min(W, c0 + R), r in max(1, r0 - R):min(H, r0 + R)
            u = Ai * [c - 0.5 - x, r - 0.5 - y]
            img[r, c] += a * exp(-(u[1]^2 + u[2]^2) / (2s^2))
        end
    end
    return img
end

"""
    gauss_field(H, W, ℓ, amp; seed) -> Matrix{Float64}

A smooth random texture: white noise blurred by a Gaussian of std `ℓ` px,
scaled to standard deviation `amp`.
"""
function gauss_field(H, W, ℓ, amp; seed = 1)
    rng = Xoshiro(seed)
    n = randn(rng, H, W)
    r = ceil(Int, 3ℓ)
    k = [exp(-i^2 / (2ℓ^2)) for i in -r:r]; k ./= sum(k)
    a = similar(n); b = similar(n)
    for j in 1:W, i in 1:H
        a[i, j] = sum(k[t + r + 1] * n[clamp(i + t, 1, H), j] for t in -r:r)
    end
    for j in 1:W, i in 1:H
        b[i, j] = sum(k[t + r + 1] * a[i, clamp(j + t, 1, W)] for t in -r:r)
    end
    b .*= amp / sqrt(sum(abs2, b) / length(b))
    return b
end

# =============================================================================
# The table's exactness: M from impulse responses of the real kernels
# =============================================================================

# The nine finite-difference functionals at (X, Y, sl) of a host DoG cube
# (W, H, NS), in `_bextract!`'s order (gS, gX, gY, h11, h22, h33, h12, h13, h23).
function functionals_at(d, X, Y, sl)
    c = d[X, Y, sl]
    return (
        0.5 * (d[X, Y, sl + 1] - d[X, Y, sl - 1]),
        0.5 * (d[X + 1, Y, sl] - d[X - 1, Y, sl]),
        0.5 * (d[X, Y + 1, sl] - d[X, Y - 1, sl]),
        d[X, Y, sl + 1] - 2c + d[X, Y, sl - 1],
        d[X + 1, Y, sl] - 2c + d[X - 1, Y, sl],
        d[X, Y + 1, sl] - 2c + d[X, Y - 1, sl],
        0.25 * (d[X + 1, Y, sl + 1] - d[X - 1, Y, sl + 1] - d[X + 1, Y, sl - 1] + d[X - 1, Y, sl - 1]),
        0.25 * (d[X, Y + 1, sl + 1] - d[X, Y - 1, sl + 1] - d[X, Y + 1, sl - 1] + d[X, Y - 1, sl - 1]),
        0.25 * (d[X + 1, Y + 1, sl] - d[X - 1, Y + 1, sl] - d[X + 1, Y - 1, sl] + d[X - 1, Y - 1, sl]),
    )
end

# The Hessian (s, x, y) at an integer location of a host DoG cube.
function hessian_at(d, X, Y, sl)
    f = functionals_at(d, X, Y, sl)
    return [f[4] f[7] f[8]; f[7] f[5] f[9]; f[8] f[9] f[6]]
end

"""
    impulse_M(plan, oi, sl, px, py) -> Matrix{Float64} (9×9)

M = K·Kᵀ of the nine functionals at octave `oi`, slice `sl`, parity (px, py),
measured on the plan's own kernels (identity warp): impulses at every phase of
the octave's input stride, each run through the real pyramid; the functionals'
responses at all level pixels of the parity class, summed. Interior only: the
plan's image must hold the kernel's support.
"""
function impulse_M(plan, oi, sl, px, py)
    W, H = plan.W, plan.H
    o = oi - 1 + plan.o_first
    q = o <= 0 ? 1 : 1 << o                 # input pixels per level pixel
    Wo, Ho = plan.dims[oi]
    M = zeros(9, 9)
    margin = 6                              # level pixels kept clear of the border
    for φy in 0:(q - 1), φx in 0:(q - 1)
        img = zeros(Float32, W, H)
        img[W ÷ 2 + φx, H ÷ 2 + φy] = 1.0f0
        asift_detect!(plan, img; intensity_thr = 1.0e9)    # pyramid only: no extremum survives
        d = Float64.(Array(plan.dogs[oi])[:, :, :, 1])
        for Y in (margin + 1):(Ho - margin), X in (margin + 1):(Wo - margin)
            (X & 1 == px && Y & 1 == py) || o >= 0 || continue
            f = functionals_at(d, X, Y, sl)
            for j in 1:9, i in 1:9
                M[i, j] += f[i] * f[j]
            end
        end
    end
    return M
end

# =============================================================================
# Monte Carlo
# =============================================================================

# χ²₃ and χ²₂ quantiles at 68.27 % and 95 % (Wilson–Hilferty is not needed: exact values).
const _CHI2 = Dict((3, 0.6827) => 3.5267, (3, 0.95) => 7.8147, (2, 0.6827) => 2.2977, (2, 0.95) => 5.9915)
chi2_q(k, p) = _CHI2[(k, p)]

"""
    detect_on(plan, img; kw...) -> Vector{Detection}

`asift_detect(img; uncertainty = FeatureCovariance(), nms = :mxm, kw...)` on an
explicit `plan` (an `AsiftPlan` of the image's size), so that the baselines can
read the DoG that produced the detections without depending on the global plan
cache. `img` is H×W.
"""
function detect_on(plan, img; nms = :mxm, kw...)
    r = asift_detect!(plan, permutedims(Float32.(img)); nms, uncertainty = FeatureCovariance(), kw...)
    return asift_detections(r.features, r.cov)
end

"""
    mc_run(img, truth; σn, ntrials, seed, device, tilts, gate, kw...) -> Vector{NamedTuple}

Detect on `ntrials` noisy copies of the float image `img` (H×W) with iid
Gaussian noise of std `σn` (no clamping, no quantization: the float input path),
associate each detection to the nearest truth `(x, y, σ)` in (x, y, log σ) within
`gate`, and return per truth the estimates, the per-trial predicted covariance
`σn²·Σ̃`, the count of associated detections that were unrefined (NaN
covariance) and, with `baselines = true`, the baselines from the plan's DoG:
`hess` the 3D Hessian-only `σn²·J(∓Ĥ)⁻¹Jᵀ` (unscaled otherwise), `z3`/`z5` Zeisl
et al.'s 2×2 position covariances (raw, before their normalization).
"""
function mc_run(img, truth; σn, ntrials, seed = 1, device = KA.CPU(), tilts = [(1.0, 0.0)],
        gate = (1.5, 0.35), upsample = true, baselines = false, kw...)
    rng = Xoshiro(seed)
    H, W = size(img)
    out = [
        (
            est = Vector{Vector{Float64}}(), cov = Vector{Matrix{Float64}}(),
            hess = Vector{Matrix{Float64}}(), z3 = Vector{Matrix{Float64}}(), z5 = Vector{Matrix{Float64}}(),
            unrefined = Ref(0),
        ) for _ in truth
    ]
    noise = zeros(Float64, H, W)
    plan = AsiftPlan(W, H; device, tilts, upsample)
    for _ in 1:ntrials
        randn!(rng, noise)
        noisy = Float32.(img .+ σn .* noise)
        dets = detect_on(plan, noisy; kw...)
        for (k, (x, y, s)) in enumerate(truth)
            best = 0; bd = Inf
            for (i, d) in enumerate(dets)
                dx = d.center[1] - x; dy = d.center[2] - y; dl = log(d.σ) - log(s)
                (hypot(dx, dy) < gate[1] && abs(dl) < gate[2]) || continue
                dd = dx^2 + dy^2 + (dl / 0.1)^2
                dd < bd && (bd = dd; best = i)
            end
            best == 0 && continue
            d = dets[best]
            Σ = Matrix(cov(d))
            all(isfinite, Σ) || (out[k].unrefined[] += 1; continue)
            push!(out[k].est, [d.center[1], d.center[2], log(d.σ)])
            push!(out[k].cov, σn^2 .* Σ)
            if baselines
                e = extremum_of(plan, d)
                push!(out[k].hess, σn^2 .* hess3d_cov(e, d.response))
                push!(out[k].z3, zeisl_cov(e, d.response; window = 3))
                push!(out[k].z5, zeisl_cov(e, d.response; window = 5))
            end
        end
    end
    return out
end

"""
    extremum_of(plan, d) -> NamedTuple or nothing

The integer DoG extremum behind detection `d`, from the plan's resident DoG (all
warps in one tilt chunk): the octave/pixel/slice whose quadratic refinement
reproduces `d`'s warp-frame position and scale. Returns `oi` (plan octave), `o`
(octave index, −1 for the upsampled one), `X, Y, sl` (1-indexed level pixel and
slice), `w` (warp), `off` (the refinement offset (s, x, y)), `A` (the warp's
back-projection, warp canvas → input), `S`, and `patch`, the DoG on
`X−3:X+3 × Y−3:Y+3 × sl−1:sl+1` (centre (4, 4, 2)). `nothing` within 3 level
pixels of the border.
"""
function extremum_of(plan, d)
    S = plan.nb_scales; σ0 = plan.seed_sigma
    w = d.warp_idx
    wp = plan.wp isa Array ? plan.wp[:, w] : Array(view(plan.wp, :, w))
    A = [wp[1] wp[2]; wp[4] wp[5]]
    xw = A \ ([d.center[1] - 0.5, d.center[2] - 0.5] - [wp[3], wp[6]])
    x = xw[1]; y = xw[2]; σ = d.σ / sqrt(abs(det(A)))     # the warp's σ: d.σ = σ_w·√t
    for oi in 1:plan.nb_octaves
        o = oi - 1 + plan.o_first
        sf = 2.0^o
        sx = (x - (sf - 1)) / sf; sy = (y - (sf - 1)) / sf
        ss = S * log2(σ / (σ0 * sf)) - 0.5
        dog = plan.dogs[oi]
        NS = size(dog, 3)
        0.4 < ss < NS - 1.4 || continue
        for X in (floor(Int, sx) + 1):(floor(Int, sx) + 2), Y in (floor(Int, sy) + 1):(floor(Int, sy) + 2), sl in (floor(Int, ss) + 1):(floor(Int, ss) + 2)
            (4 <= X <= size(dog, 1) - 3 && 4 <= Y <= size(dog, 2) - 3 && 2 <= sl < NS) || continue
            patch = Float64.(Array(view(dog, (X - 3):(X + 3), (Y - 3):(Y + 3), (sl - 1):(sl + 1), w)))
            f = functionals_at(patch, 4, 4, 2)
            Hm = [f[4] f[7] f[8]; f[7] f[5] f[9]; f[8] f[9] f[6]]
            abs(det(Hm)) > 0 || continue
            off = -(Hm \ [f[1], f[2], f[3]])
            r = (X - 1 + off[2], Y - 1 + off[3], sl - 1 + off[1])
            if abs(r[1] - sx) < 1.0e-3 && abs(r[2] - sy) < 1.0e-3 && abs(r[3] - ss) < 1.0e-3
                return (; oi, o, X, Y, sl, w, off = (off[1], off[2], off[3]), A, S, patch)
            end
        end
    end
    return nothing
end

"""
    hess3d_cov(e, response) -> Matrix (3×3, (x, y, log σ))

The 3D Hessian-only baseline `J·(∓Ĥ)⁻¹·Jᵀ` (unscaled) at the extremum `e`
(`extremum_of`): Ĥ the detector's own 3×3 Hessian in (s, x, y), the sign making
it positive definite (a DoG maximum has Ĥ ≺ 0), J to (x, y, log σ), then the
back-warp. All NaN without an extremum.
"""
function hess3d_cov(e, response)
    e === nothing && return fill(NaN, 3, 3)
    f = functionals_at(e.patch, 4, 4, 2)
    Hm = [f[4] f[7] f[8]; f[7] f[5] f[9]; f[8] f[9] f[6]]
    sf = 2.0^e.o; q = log(2) / e.S
    J = [0 sf 0; 0 0 sf; q 0 0]
    Jw = [e.A zeros(2); 0 0 1]
    s = response > 0 ? -1.0 : 1.0
    return Jw * J * inv(s * Hm) * J' * Jw'
end

# Zeisl et al.'s 5×5 Gaussian weights (their eq. 14). The paper does not give the
# weight's σ and the released code (campar.in.tum.de CovarianceEstimator, 2009) is
# no longer obtainable; σ = 1 level pixel puts the window's edge at 2σ, the usual
# truncation for a 5-tap Gaussian. Normalized to sum 1, so a uniform Hessian field
# gives the 3×3 value.
const ZEISL_W5_SIGMA = 1.0
const _ZW5 = let w = [exp(-(i^2 + j^2) / (2 * ZEISL_W5_SIGMA^2)) for i in -2:2, j in -2:2]
    w ./ sum(w)
end

# The 2×2 spatial DoG Hessian [Dxx Dxy; Dxy Dyy] at patch pixel (4+i, 4+j), slice 2
# (the detection layer), by differences of neighbouring samples.
function _spatial_hessian(P, i, j)
    X = 4 + i; Y = 4 + j
    c = P[X, Y, 2]
    dxx = P[X + 1, Y, 2] - 2c + P[X - 1, Y, 2]
    dyy = P[X, Y + 1, 2] - 2c + P[X, Y - 1, 2]
    dxy = 0.25 * (P[X + 1, Y + 1, 2] - P[X - 1, Y + 1, 2] - P[X + 1, Y - 1, 2] + P[X - 1, Y - 1, 2])
    return [dxx dxy; dxy dyy]
end

"""
    zeisl_cov(e, response; window = 3) -> Matrix (2×2, input pixels)

Zeisl, Georgel, Schweiger, Steinbach and Navab (BMVC 2009), raw: `Σ = ∓H⁻¹` with
H the 2×2 SPATIAL Hessian of the DoG at the keypoint on its detection layer
(eq. 6; the sign negative at a maximum), or with `window = 5` the
Gaussian-weighted sum of the Hessians over the 5×5 neighbourhood (eq. 14,
weights `_ZW5`), back-projected to the input image by `(2^(octave + (σ̂ − σ)))²`
— their detection-layer form of eq. 15, σ̂ − σ the refinement's scale offset in
octaves (offS/S) — and through the ASIFT warp's linear part. No noise model and
no normalization: their constant (`zeisl_frobenius_constant`) or a fitted one
multiplies it. All NaN without an extremum or for a non-definite H.
"""
function zeisl_cov(e, response; window::Int = 3)
    e === nothing && return fill(NaN, 2, 2)
    P = e.patch
    H = window == 3 ? _spatial_hessian(P, 0, 0) :
        sum(_ZW5[i + 3, j + 3] .* _spatial_hessian(P, i, j) for i in -2:2, j in -2:2)
    s = response > 0 ? -1.0 : 1.0
    Σ = s .* inv(H)
    (Σ[1, 1] > 0 && det(Σ) > 0) || return fill(NaN, 2, 2)
    k = (2.0^(e.o + e.off[1] / e.S))^2
    return k .* (e.A * Σ * e.A')
end

"""
    zeisl_frobenius_constant(; window, device, amplitude = 0.3) -> Float64

Zeisl et al.'s normalization (§4.3): one global constant such that "a circular
feature detected in the very bottom pyramid layer" has ‖Σ‖_F ≈ 1. Measured on a
noiseless DoG-shaped (matched-filter) blob detected on the bottom refinable
layer — the upsampled octave's slice 1, reported σ ≈ 1.6·2^(1.5/3)/2 ≈ 1.13 px,
which a DoG blob of σ_b = 1.5 produces (detected σ ≈ 0.77·σ_b) — of peak
amplitude `amplitude` (the paper does not state the contrast; the constant, and
so every normalized covariance, scales with it).
"""
function zeisl_frobenius_constant(; window::Int = 3, device = KA.CPU(), amplitude = 0.3)
    img = dog_blob_image(64, 64, [(32.3, 31.6, 1.5, amplitude)])
    plan = AsiftPlan(64, 64; device, tilts = [(1.0, 0.0)])
    dets = detect_on(plan, img)
    d = argmin(d -> hypot(d.center[1] - 32.3, d.center[2] - 31.6), dets)
    e = extremum_of(plan, d)
    (e !== nothing && e.oi == 1 && e.sl == 2) ||
        error("calibration blob not detected on the bottom layer: $(e === nothing ? nothing : (e.oi, e.sl))")
    return 1 / sqrt(sum(abs2, zeisl_cov(e, d.response; window)))
end

"""
    dog_blob_image(H, W, blobs; bg = 0.5, k = 2^(1/3), H2 = nothing) -> Matrix{Float64}

Zeisl et al.'s matched-filter blobs: a DoG `G(kσ) − G(σ)` at each `(x, y, σ, a)`
(COLMAP pixel centres), scaled to peak amplitude `a`. With `H2` (a function
taking an image point to plane coordinates, e.g. an inverse homography) each
blob is rendered on the plane and imaged through it.
"""
function dog_blob_image(H, W, blobs; bg = 0.5, k = 2^(1 / 3), H2 = nothing)
    img = fill(Float64(bg), H, W)
    for (x, y, s, a) in blobs
        g(r2, t) = exp(-r2 / (2t^2)) / t^2
        peak = g(0.0, s) - g(0.0, k * s)
        for c in 1:W, r in 1:H
            p = (c - 0.5, r - 0.5)
            u, v = H2 === nothing ? (p[1] - x, p[2] - y) : H2(p)
            r2 = u^2 + v^2
            r2 < (8k * s)^2 || continue
            img[r, c] += a * (g(r2, s) - g(r2, k * s)) / peak
        end
    end
    return img
end

"""
    mc_summary(est, cov; truth = nothing, Σbase = nothing) -> NamedTuple

Calibration statistics of one feature's estimates `est` against the per-trial
predicted covariances `cov`: the normalized squared errors `e = dᵀΣ⁻¹d` (d =
estimate − mean estimate) in 3D and on the 2×2 position block, their means
(≈ 3 and ≈ 2 when calibrated), coverage at the χ² 68.27 %/95 % quantiles, the
per-axis variance ratio empirical/predicted (mean prediction), and the bias of
the mean against `truth`.
"""
function mc_summary(est, cov; truth = nothing)
    n = length(est)
    μ = sum(est) / n
    e3 = Float64[]; e2 = Float64[]
    for (v, Σ) in zip(est, cov)
        d = v - μ
        push!(e3, d' * (Symmetric(Σ) \ d))
        push!(e2, d[1:2]' * (Symmetric(Σ[1:2, 1:2]) \ d[1:2]))
    end
    # the mean of e for estimates centred on their own mean is (n−1)/n·k
    corr = n / (n - 1)
    Σemp = sum((v - μ) * (v - μ)' for v in est) / (n - 1)
    Σpred = sum(cov) / n
    return (
        n = n,
        mean_e3 = corr * mean(e3), mean_e2 = corr * mean(e2),
        cov68_3 = mean(corr .* e3 .<= chi2_q(3, 0.6827)), cov95_3 = mean(corr .* e3 .<= chi2_q(3, 0.95)),
        cov68_2 = mean(corr .* e2 .<= chi2_q(2, 0.6827)), cov95_2 = mean(corr .* e2 .<= chi2_q(2, 0.95)),
        ratio = [Σemp[i, i] / Σpred[i, i] for i in 1:3],
        bias = truth === nothing ? nothing : μ - [truth[1], truth[2], log(truth[3])],
        sd = sqrt.([Σemp[i, i] for i in 1:3]),
        Σemp = Σemp, Σpred = Σpred,
    )
end

end # module UncFixtures
