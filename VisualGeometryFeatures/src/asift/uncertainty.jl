# =============================================================================
# Keypoint uncertainty — the implicit-function-theorem covariance of a DoG
# keypoint, optional per call (`asift_detect(img; uncertainty = FeatureCovariance())`).
# =============================================================================
#
# A refined DoG keypoint is a stationary point of the detector's local
# quadratic model of the DoG response in v = (s, x, y) (slice, level pixels):
#
#     g(v; I) = ĝ(I) + Ĥ(I)·(v − v₀) = 0,
#
# ĝ, Ĥ the central finite differences at the integer extremum v₀ (the 3 + 6
# values `_bextract!` forms), all LINEAR in the input image I: the warp, the
# 2× upsample, the Gaussian blurs, the octave downsamples and the DoG are
# linear. The implicit function theorem gives δv = −Ĥ⁻¹·(δĝ + δĤ·off), off =
# v − v₀ the refinement offset, so for iid pixel noise of unit variance
#
#     Σ_v = Ĥ⁻¹ · P(off)·M·P(off)ᵀ · Ĥ⁻¹,
#
# with M = K·Kᵀ the 9×9 covariance of the nine finite-difference functionals
# (K: input pixels → [ĝ; Ĥ]) and P(off) = [I₃ | the off-weighted Hessian rows].
# M depends only on the detector configuration — warp, octave, slice and, at
# the upsampled octave, the pixel parity — never on the image, so it is
# tabulated once per plan (`_cov_tables`), exactly for the discrete pipeline
# (`_canvas_kernel` runs the adjoint of every stage); the per-keypoint work is
# `_bcov!` (driver.jl), from the refinement state `_bextract!` stores. The δĤ·off term is the
# implicit function theorem applied to the function the detector actually
# zeroes; dropping it (P = [I₃ | 0]) is the "gradient-only" middle term.
#
# Reported coordinates: x = sf·x_level + (sf − 1) (+ ½ COLMAP), likewise y,
# log σ = log σ₀ + (s + ½)·ln2/S + log sf, so J = diag(sf, sf, ln2/S); the ASIFT
# back-projection moves (x, y) by the warp's linear part A and leaves the
# third coordinate a constant shift (see `_bbackproject!`).
# =============================================================================

"""
    FeatureCovariance()

Ask `asift_detect` for each keypoint's 3×3 covariance over `(x, y, log σ)`, by the implicit function theorem, for iid unit pixel noise.

`asift_detect(img; uncertainty = FeatureCovariance())` returns detections whose
frame is VisualGeometryCore's `Uncertain(frame, Σ̃)`; the default
`uncertainty = nothing` runs the detector unchanged. `cov(d)` is `Σ̃`,
`feature_frame(d)` the frame, and `d.center`, `d.σ` and the metadata read as on
any detection.

**Model.** A refined DoG keypoint is a stationary point of the detector's local
quadratic model of the DoG in (slice, x, y). With `Ĥ` its finite-difference
Hessian and `C` the covariance, under unit iid pixel noise, of that model's
gradient at the keypoint, `Σ̃ = J·Ĥ⁻¹·C·Ĥ⁻¹·Jᵀ`: `C` is tabulated exactly for
the discrete pyramid (warp, upsample, blurs, octave downsamples) once per plan,
and `J` maps to the reported coordinates, then through the ASIFT back-projection.

**Parameterization.** `Σ̃` is a `CovMat{3}` over `(x, y, log σ)` in the
detection's image frame (COLMAP pixel centres, as `d.center`), `σ = d.σ` (the
frame's characteristic scale `√det S`), `log` natural. Position-only consumers
take the top-left 2×2 block (the marginal). A tilted ASIFT warp's frame has the
shape `S = σ_w·M_w` with `M_w` fixed by the warp (`d.warp_idx` into the tilt
schedule), so `(x, y, log σ)` are its only free parameters and `Σ̃` is full rank.

**Units.** Unit noise variance in the detector's intensity units: those of a
float input image, and the `[0, 1]` scale for 8-bit input (`UInt8`, `N0f8`, …;
σ_n = σ_gray / 255). The physical covariance is `σ_n²·Σ̃`.

**Unrefined detections** — on a boundary scale slice, or whose sub-pixel
refinement failed — sit on the octave's pixel grid and have no such covariance:
their `Σ̃` is all `NaN`.

Extension point (not implemented): an intensity-dependent noise model, e.g.
`FeatureCovariance(noise = ShotNoise(a, b))`, makes the middle term
`K·diag(σ²(I))·Kᵀ`, gathered from the pyramid per keypoint instead of tabulated.
"""
struct FeatureCovariance end

# The covariance an uncertain ASIFT frame carries: (x, y, log σ), Float64 like
# the frame. Unlabeled: VGC's image-frame tag is a Frame{2}, and no Frame{3} tag
# names a feature frame's (x, y, log σ) parameters.
const _FrameCov = CovMat{3, Float64, 9, UnlabeledFrame{3}}

# ----- packed symmetric storage ----------------------------------------------

# The nine finite-difference functionals, in `_bextract!`'s axis order
# (1 = slice, 2 = x, 3 = y): ĝ = (gS, gX, gY), Ĥ = (h11, h22, h33, h12, h13, h23).
# M is stored packed, upper triangle by columns: (k, l), k ≤ l → k + l(l−1)/2.
const _NFUN = 9
const _NPACK = _NFUN * (_NFUN + 1) ÷ 2   # 45
@inline _pk(k, l) = k <= l ? k + (l * (l - 1)) >> 1 : l + (k * (k - 1)) >> 1

# Rows of P(off): row i of the quadratic model's gradient is ĝᵢ + Σⱼ Ĥᵢⱼ·offⱼ,
# i.e. functional `_PROW[i][1]` plus `_PROW[i][2:4]` weighted by (oS, oX, oY).
const _PROW = ((1, 4, 7, 8), (2, 7, 5, 9), (3, 8, 9, 6))

# ----- the 1D linear pipeline, adjoint form ----------------------------------
#
# Every pyramid stage is separable, so the impulse response of a Gaussian level
# value G[o][l](X, Y) on the canvas is kx ⊗ ky, each the 1D response of the 1D
# pipeline. `_Line` is a 1D signal on the unbounded integer line (no borders: the
# table describes the interior); the adjoint stages below take a level-pixel delta
# back to canvas pixels.

struct _Line
    i0::Int                 # index of v[1]
    v::Vector{Float64}
end
@inline _line_at(l::_Line, i::Int) = (k = i - l.i0 + 1; 1 <= k <= length(l.v) ? l.v[k] : 0.0)

# The separable blur y[i] = Σₜ k[|t|+1]·x[i+t] is its own adjoint (symmetric taps).
function _blur_adj(l::_Line, taps::AbstractVector)
    r = length(taps) - 1
    out = zeros(length(l.v) + 2r)
    for (j, a) in enumerate(l.v)
        a == 0 && continue
        for t in -r:r
            out[j + r + t] += Float64(taps[abs(t) + 1]) * a
        end
    end
    return _Line(l.i0 - r, out)
end

# Adjoint of `_bdown!`: dst[i] = src[2i].
function _down_adj(l::_Line)
    out = zeros(2length(l.v) - 1)
    for (j, a) in enumerate(l.v)
        out[2j - 1] = a
    end
    return _Line(2l.i0, out)
end

# Adjoint of `_bupsample!` away from its clamped edge: dst[2k] = src[k],
# dst[2k+1] = (src[k] + src[k+1]) / 2.
function _up_adj(l::_Line)
    lo = fld(l.i0 - 1, 2)
    out = zeros(cld(l.i0 + length(l.v), 2) - lo + 2)
    for (j, a) in enumerate(l.v)
        I = l.i0 + j - 1
        if iseven(I)
            out[I ÷ 2 - lo + 1] += a
        else
            k = fld(I - 1, 2)
            out[k - lo + 1] += 0.5a
            out[k - lo + 2] += 0.5a
        end
    end
    return _Line(lo, out)
end

# Canvas-space 1D response of the Gaussian level `l` (1-indexed, 1 … S+3) at
# level pixel `X` of octave `oi` (1 = the plan's finest), for the pyramid with
# per-level taps `taps` (`taps[1]` the seed blur): `_bdown!` takes level S+1 of
# the octave below, `_bupsample!` precedes the seed blur when upsampling.
function _canvas_kernel(taps, S::Int, upsample::Bool, oi::Int, l::Int, X::Int)
    v = _Line(X, [1.0])
    for o in oi:-1:1
        for lev in l:-1:2
            v = _blur_adj(v, taps[lev])
        end
        if o > 1
            v = _down_adj(v)
            l = S + 1
        else
            v = _blur_adj(v, taps[1])
            upsample && (v = _up_adj(v))
        end
    end
    return v
end

# Σᵤ a(u)·b(u + δ)
function _xcorr(a::_Line, b::_Line, δ::Int)
    s = 0.0
    for (j, x) in enumerate(a.v)
        x == 0 && continue
        s += x * _line_at(b, a.i0 + j - 1 + δ)
    end
    return s
end

# ----- functionals over the 36 Gaussian-level basis values -------------------

# Basis value (dx, dy, ℓ): G at level-pixel offset (dx, dy) and level sl + ℓ,
# ℓ ∈ −1:2 — every Gaussian value the nine functionals at slice `sl` read.
@inline _bidx(dx, dy, ℓ) = (dx + 2) + 3 * (dy + 1) + 9 * (ℓ + 1)

# F (9 × 36): functionals as combinations of basis values, from the DoG
# stencils of `_bextract!` with D(dx, dy, ds) = G(sl+ds+1) − G(sl+ds).
function _functional_matrix()
    F = zeros(_NFUN, 36)
    D!(i, c, dx, dy, ds) = (F[i, _bidx(dx, dy, ds + 1)] += c; F[i, _bidx(dx, dy, ds)] -= c)
    # gradient: gS, gX, gY
    D!(1, 0.5, 0, 0, 1); D!(1, -0.5, 0, 0, -1)
    D!(2, 0.5, 1, 0, 0); D!(2, -0.5, -1, 0, 0)
    D!(3, 0.5, 0, 1, 0); D!(3, -0.5, 0, -1, 0)
    # Hessian diagonal: h11 (ss), h22 (xx), h33 (yy)
    D!(4, 1.0, 0, 0, 1); D!(4, -2.0, 0, 0, 0); D!(4, 1.0, 0, 0, -1)
    D!(5, 1.0, 1, 0, 0); D!(5, -2.0, 0, 0, 0); D!(5, 1.0, -1, 0, 0)
    D!(6, 1.0, 0, 1, 0); D!(6, -2.0, 0, 0, 0); D!(6, 1.0, 0, -1, 0)
    # mixed: h12 (sx), h13 (sy), h23 (xy)
    for (i, (a, b)) in ((7, (:s, :x)), (8, (:s, :y)), (9, (:x, :y)))
        for sa in (-1, 1), sb in (-1, 1)
            o = Dict(:s => 0, :x => 0, :y => 0)
            o[a] = sa; o[b] = sb
            D!(i, 0.25 * sa * sb, o[:x], o[:y], o[:s])
        end
    end
    return F
end

# ----- canvas noise covariance -----------------------------------------------

# Weights of the canvas pixel (X, Y) of warp geometry `g` on input pixels, by the
# arithmetic of `_bwarp3!` and `_bfpb!` (interior: no clamps, no fill), appended
# to `ws` as (ix, iy, w).
function _canvas_weights!(ws::Vector{Tuple{Int, Int, Float64}}, g, taps, X::Int, Y::Int)
    empty!(ws)
    ca = g.ca; sa = g.sa
    a13 = ca * Float32(g.xmin) - sa * Float32(g.ymin)
    a23 = sa * Float32(g.xmin) + ca * Float32(g.ymin)
    sy = (Float32(g.t) * Float32(Y - 1) + 0.5f0) - 0.5f0
    yi = floor(Int, sy); uy = Float64(sy - Float32(yi))
    r = length(taps) - 1
    for (row0, wrow) in ((yi + 1, 1.0 - uy), (yi + 2, uy))
        wrow == 0 && continue
        for k in -r:r
            wk = wrow * Float64(taps[abs(k) + 1])
            cx = Float32(X - 1); cy = Float32(row0 + k - 1)
            xin = ca * cx + (-sa) * cy + a13
            yin = sa * cx + ca * cy + a23
            x0 = floor(Int, xin); y0 = floor(Int, yin)
            fx = Float64(xin - Float32(x0)); fy = Float64(yin - Float32(y0))
            for (ix, wx) in ((x0 + 1, 1.0 - fx), (x0 + 2, fx)), (iy, wy) in ((y0 + 1, 1.0 - fy), (y0 + 2, fy))
                w = wk * wx * wy
                w == 0 || push!(ws, (ix, iy, w))
            end
        end
    end
    return ws
end

# Canvas noise autocovariance c(δx, δy) = E_p⟨w_p, w_{p+δ}⟩ for unit iid input
# noise, averaged over canvas positions (a rotated or tilted warp resamples at a
# phase that varies across the canvas; the table uses its average). An identity
# warp is exact: c = δ. Returns (c, Rx, Ry) with c[δx + Rx + 1, δy + Ry + 1].
function _canvas_autocov(g, W::Int, H::Int; nsamp::Int = 1024)
    g.t == 1 && g.θ == 0 && return (ones(1, 1), 0, 0)
    taps = g.t > 1 ? sigma_aa_taps(0.8f0 * sqrt(Float32(g.t)^2 - 1.0f0)) : Float32[1.0f0]
    r = length(taps) - 1
    Rx = 2                                   # bilinear footprints: 2 px apart share nothing
    Ry = ceil(Int, (2r + 4) / g.t)           # σ_aa rows of canvas rows Ry apart overlap
    cw = min(g.W_rot, W)
    ch = g.t > 1 ? min(max(floor(Int, g.H_rot / Float32(g.t)), 1), H) : min(g.H_rot, H)
    c = zeros(2Rx + 1, 2Ry + 1)
    B = ceil(Int, Rx + Ry * g.t + 2r) + 8    # input box half-width around a sample's centroid
    box = zeros(2B + 1, 2B + 1)
    w0 = Tuple{Int, Int, Float64}[]; w1 = Tuple{Int, Int, Float64}[]
    # deterministic low-discrepancy positions over the canvas interior
    φ1 = 0.7548776662466927; φ2 = 0.5698402909980532
    for k in 1:nsamp
        X = 1 + Rx + floor(Int, mod(0.5 + k * φ1, 1.0) * (cw - 2Rx - 1))
        Y = 1 + Ry + floor(Int, mod(0.5 + k * φ2, 1.0) * (ch - 2Ry - 1))
        _canvas_weights!(w0, g, taps, X, Y)
        ox = round(Int, sum(w[1] * w[3] for w in w0) / sum(w[3] for w in w0)) - B - 1
        oy = round(Int, sum(w[2] * w[3] for w in w0) / sum(w[3] for w in w0)) - B - 1
        for (ix, iy, w) in w0
            box[ix - ox, iy - oy] += w
        end
        for dy in -Ry:Ry, dx in -Rx:Rx
            s = 0.0
            for (ix, iy, w) in _canvas_weights!(w1, g, taps, X + dx, Y + dy)
                s += w * box[ix - ox, iy - oy]
            end
            c[dx + Rx + 1, dy + Ry + 1] += s
        end
        for (ix, iy, _) in w0
            box[ix - ox, iy - oy] = 0.0
        end
    end
    c ./= nsamp
    return (c, Rx, Ry)
end

# ----- the tables --------------------------------------------------------------

# M (packed, 45) for every (slice, parity class, octave, warp) of a plan. Parity
# class pc = 1 + (X & 1) + 2(Y & 1) of the integer extremum's level pixel: the
# upsampled octave's pixels alternate between input samples and midpoints; the
# other octaves repeat one table in all four classes.
function _cov_tables(taps::AbstractVector, S::Int, upsample::Bool, noct::Int, geo, W::Int, H::Int)
    NS = S + 2
    nw = length(geo)
    F = _functional_matrix()
    # Threaded over warps and octaves: every task fills its own entries, so the
    # tables do not depend on the scheduling.
    autoc = fetch.([Threads.@spawn(_canvas_autocov(g, W, H)) for g in geo])
    Rmax = maximum(max(a[2], a[3]) for a in autoc)
    tabs = [zeros(Float32, _NPACK, NS, 4, nw) for _ in 1:noct]
    X0 = 1 << 14                                  # far from the line's origin
    @sync for oi in 1:noct
        Threads.@spawn _octave_tables!(tabs[oi], taps, S, upsample, oi, X0, F, autoc, Rmax)
    end
    return tabs
end

# One octave's tables, `tab[:, sl, pc, w]` (see `_cov_tables`).
function _octave_tables!(tab, taps, S, upsample, oi, X0, F, autoc, Rmax)
    NS = S + 2
    nw = length(autoc)
    # 1D kernels for levels 1:S+3 at offsets −1:1, per pixel parity
    kern = [
        [_canvas_kernel(taps, S, upsample, oi, l, X0 + p + d) for d in -1:1, l in 1:(S + 3)]
            for p in 0:1
    ]
    # cross-correlations XC[p][(d₁,l₁), (d₂,l₂), δ]
    XC = map(kern) do K
        n = length(K)
        A = zeros(n, n, 2Rmax + 1)
        for b in 1:n, a in 1:n, δ in -Rmax:Rmax
            A[a, b, δ + Rmax + 1] = _xcorr(K[a], K[b], δ)
        end
        A
    end
    @sync for w in 1:nw
        Threads.@spawn for sl in 2:(NS - 1), py in 0:1, px in 0:1
            c, Rx, Ry = autoc[w]
            Bm = zeros(36, 36)
            for a in 1:36, b in 1:36
                dxa = (a - 1) % 3 - 1; dya = ((a - 1) ÷ 3) % 3 - 1; la = (a - 1) ÷ 9 - 1 + sl
                dxb = (b - 1) % 3 - 1; dyb = ((b - 1) ÷ 3) % 3 - 1; lb = (b - 1) ÷ 9 - 1 + sl
                ia = (dxa + 2) + 3 * (la - 1); ib = (dxb + 2) + 3 * (lb - 1)
                ja = (dya + 2) + 3 * (la - 1); jb = (dyb + 2) + 3 * (lb - 1)
                s = 0.0
                for δy in -Ry:Ry, δx in -Rx:Rx
                    cv = c[δx + Rx + 1, δy + Ry + 1]
                    cv == 0 && continue
                    s += cv * XC[px + 1][ia, ib, δx + Rmax + 1] * XC[py + 1][ja, jb, δy + Rmax + 1]
                end
                Bm[a, b] = s
            end
            M = F * Bm * F'
            pc = 1 + px + 2py
            for l in 1:_NFUN, k in 1:l
                tab[_pk(k, l), sl, pc, w] = Float32(M[k, l])
            end
        end
    end
    return tab
end

# The 9×9 M of a table entry, unpacked (host).
function _unpack_M(tab::AbstractArray, sl, pc, w)
    M = zeros(_NFUN, _NFUN)
    for l in 1:_NFUN, k in 1:_NFUN
        M[k, l] = tab[_pk(k, l), sl, pc, w]
    end
    return M
end

# ----- per keypoint (device and host) -----------------------------------------

# The keypoint covariance in the warp canvas' reported coordinates (x, y, log σ),
# packed (xx, xy, xℓ, yy, yℓ, ℓℓ): Ĥ⁻¹·P(off)·M·P(off)ᵀ·Ĥ⁻¹ (in s, x, y), then
# J = diag(sf, sf, ln2/S). `h` = (h11, h22, h33, h12, h13, h23).
@inline function _ift_cov(tab, sl, pc, w, h, oS::Float32, oX::Float32, oY::Float32, sf::Float32, S::Int32)
    off = (1.0f0, oS, oX, oY)
    # C = P·M·Pᵀ: row i of P has `off` at the functionals `_PROW[i]` (unrolled)
    @inline function crow(ri, rj)
        acc = 0.0f0
        Base.Cartesian.@nexprs 4 a -> Base.Cartesian.@nexprs 4 b -> begin
            acc += off[a] * off[b] * (@inbounds tab[_pk(ri[a], rj[b]), sl, pc, w])
        end
        return acc
    end
    R1, R2, R3 = _PROW
    c11 = crow(R1, R1); c12 = crow(R1, R2); c13 = crow(R1, R3)
    c22 = crow(R2, R2); c23 = crow(R2, R3); c33 = crow(R3, R3)
    h11, h22, h33, h12, h13, h23 = h
    # Ĥ⁻¹ by the adjugate (the refinement's own inverse)
    a11 = h22 * h33 - h23 * h23; a12 = h13 * h23 - h12 * h33; a13 = h12 * h23 - h13 * h22
    a22 = h11 * h33 - h13 * h13; a23 = h12 * h13 - h11 * h23; a33 = h11 * h22 - h12 * h12
    inv = 1.0f0 / (h11 * a11 + h12 * a12 + h13 * a13)
    Hi = SMatrix{3, 3, Float32}(a11, a12, a13, a12, a22, a23, a13, a23, a33) * inv
    C = SMatrix{3, 3, Float32}(c11, c12, c13, c12, c22, c23, c13, c23, c33)
    Σ = Hi * C * Hi          # (s, x, y)
    q = 0.6931472f0 / Float32(S)
    return (
        sf * sf * Σ[2, 2], sf * sf * Σ[2, 3], sf * q * Σ[2, 1],
        sf * sf * Σ[3, 3], sf * q * Σ[3, 1], q * q * Σ[1, 1],
    )
end

"""
    _ift_cov_reference(H, off, M, sf, S) -> SMatrix{3, 3} over (x, y, log σ)

Host reference of the covariance `_bextract!` emits, through VisualGeometryCore's
`implicit_covariance`: the square system `g(v, f) = ĝ + Ĥ·(v − v₀) = 0` with
`A = ∂g/∂v = Ĥ` and `B = ∂g/∂f = P(off)` over the nine functionals `f`, whose
covariance is `M`; then `propagate` by `J = diag(sf, sf, ln2/S)`, which also
reorders (s, x, y) to (x, y, log σ).
"""
function _ift_cov_reference(H::AbstractMatrix, off, M::AbstractMatrix, sf::Real, S::Integer)
    oS, oX, oY = off
    P = zeros(3, _NFUN)
    for i in 1:3
        r = _PROW[i]
        P[i, r[1]] += 1.0
        P[i, r[2]] += oS; P[i, r[3]] += oX; P[i, r[4]] += oY
    end
    Σv = implicit_covariance(Matrix{Float64}(H), P; Σx = [Matrix{Float64}(M)])
    Σv === nothing && return SMatrix{3, 3}(fill(NaN, 3, 3))
    q = log(2) / S
    J = SMatrix{3, 3, Float64}(0, 0, q, sf, 0, 0, 0, sf, 0)   # (s, x, y) → (x, y, log σ)
    return propagate(J, SMatrix{3, 3, Float64}(Σv))
end

# Host reference of `_backproject_cov!`: the back-warp's linear part A on (x, y),
# log σ fixed, by `propagate`.
function _backproject_cov_reference(Σ::AbstractMatrix, A::AbstractMatrix)
    J = SMatrix{3, 3, Float64}(A[1, 1], A[2, 1], 0, A[1, 2], A[2, 2], 0, 0, 0, 1)
    return propagate(J, SMatrix{3, 3, Float64}(Σ))
end

# Host: the packed (xx, xy, xℓ, yy, yℓ, ℓℓ) column as a _FrameCov.
@inline _keypoint_cov(c) = _FrameCov(
    Float64(c[1]), Float64(c[2]), Float64(c[3]),
    Float64(c[2]), Float64(c[4]), Float64(c[5]),
    Float64(c[3]), Float64(c[5]), Float64(c[6]),
)
