# The ASIFT keypoint covariance (`uncertainty = FeatureCovariance()`) against
# Zeisl, Georgel, Schweiger, Steinbach and Navab, "Estimation of Location
# Uncertainty for Scale Invariant Feature Points", BMVC 2009. Not part of the
# test suite.
#
#   julia --project=<env with VisualGeometryToolkitFeatures, CUDA (optional)> benchmark/mc/zeisl_comparison.jl [outdir]
#
# outdir defaults to benchmark/mc/results/zeisl. It receives one CSV per experiment,
# a pgfplots source per figure (.tex: article + preview, so it compiles to a
# tight PDF; the tikzpicture inputs into a paper unchanged), the PDFs when `pdflatex` is on the PATH, and
# summary.md with the tables below. MC_TRIALS (default 400; 300 for the sweeps)
# sets the trials.
#
# Zeisl et al. (the baselines, test/uncertainty_testsetup.jl): Σ = ∓H⁻¹, H the
# 2×2 SPATIAL Hessian of the DoG on the detection layer (eq. 6, "Zeisl-3×3"), or
# the Gaussian-weighted sum of the Hessians over the 5×5 neighbourhood (eq. 14,
# "Zeisl-5×5", weight σ = 1 level pixel: not given in the paper, the released
# code is no longer obtainable), back-projected by (2^(octave + (σ̂ − σ)))²
# (eq. 15, detection-layer form) and the ASIFT warp. Their normalization (§4.3):
# one constant making a circular bottom-layer feature's ‖Σ‖_F ≈ 1
# ("Frobenius"); or, most generous to them, one constant fitted over all rows
# so the pooled mean e₂ is 2 ("fit"). Their model is 2D, so the comparison is on
# position only: our covariance's 2×2 block. "Hess-3D" is the earlier baseline,
# the detector's 3×3 Hessian in (s, x, y) inverted, one fitted constant.
#
# Experiments
#   1. calibration: the configurations of keypoint_uncertainty_mc.jl, per blob
#      mean e₂ = dᵀΣ⁻¹d over the position (target 2) and χ² coverage.
#   2. contrast sweep: blobs of σ 2.5 / 5 / 10 at amplitude a ∈ [0.03, 1]
#      (1.5 decades), σn = 0.005; empirical and predicted position variance
#      against a, log-log slopes with 95 % CIs. Expected: empirical ∝ a⁻²,
#      ∓H⁻¹ ∝ a⁻¹.
#   3. scale sweep: one blob, σ from 1.6 to 40 px (4.6 octaves), a = 0.3,
#      σn = 0.01; slopes against σ, and Zeisl/empirical per scale.
#   4. Zeisl §5: a DoG-shaped (matched-filter) blob on a plane seen under a
#      viewpoint change of 0–60°, plain SIFT (identity warp), pixel noise; their
#      Bhattacharyya distance after normalizing both covariances to unit
#      Frobenius norm (Table 1), and the unnormalized magnitude error.
#
# The detection threshold is lowered (intensity_thr = 0.005) in the contrast
# sweep only, so that the faintest blobs are detected; it gates detections and
# does not enter the covariances.

using Printf, Statistics, LinearAlgebra
using VisualGeometryToolkitFeatures
using VisualGeometryToolkitCore: cov
import KernelAbstractions as KA

const _HERE = @__DIR__
include_string(@__MODULE__, replace(read(joinpath(_HERE, "..", "..", "test", "uncertainty_testsetup.jl"), String),
    "@testsetup module" => "module"))
using .UncFixtures
include(joinpath(_HERE, "mc_configs.jl"))   # configs()

device = try
    @eval using CUDA
    CUDA.functional() ? CUDA.CUDABackend() : KA.CPU()
catch
    KA.CPU()
end

const NTR = parse(Int, get(ENV, "MC_TRIALS", "400"))
const NSW = min(NTR, 300)
const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(_HERE, "results", "zeisl")
mkpath(OUT)

# ---------------------------------------------------------------- helpers
pos(v) = v[1:2]
okΣ(Σ) = all(isfinite, Σ)
meanpos(r) = sum(pos.(r.est)) / length(r.est)
# mean of dᵀΣ⁻¹d with d about the sample mean, unbiased for n−1 (expectation 2)
function mean_e2(r, covs; c = 1.0)
    μ = meanpos(r); n = length(r.est)
    e = [(pos(v) - μ)' * ((c .* Σ) \ (pos(v) - μ)) for (v, Σ) in zip(r.est, covs) if okΣ(Σ)]
    isempty(e) && return (NaN, NaN, NaN)
    k = n / (n - 1)
    return (k * mean(e), mean(k .* e .<= chi2_q(2, 0.6827)), mean(k .* e .<= chi2_q(2, 0.95)))
end
raw_e2(r, covs) = [mean_e2(r, covs)[1]]
posblock(Σ) = Σ[1:2, 1:2]
empvar(r) = (μ = meanpos(r); Σ = sum((pos(v) - μ) * (pos(v) - μ)' for v in r.est) / (length(r.est) - 1); tr(Σ) / 2)
meanvar(covs) = (v = [tr(Σ) / 2 for Σ in covs if okΣ(Σ)]; isempty(v) ? NaN : mean(v))

# Student t 0.975 quantiles (df 1…30), normal beyond
const _T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
    2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042]
t975(df) = df <= 30 ? _T975[df] : 1.96
# OLS slope of log y on log x with its 95 % CI
function loglog_slope(x, y)
    ok = isfinite.(y) .& (y .> 0)
    lx = log.(x[ok]); ly = log.(y[ok]); n = length(lx)
    n >= 3 || return (NaN, NaN, NaN)
    X = [ones(n) lx]
    β = X \ ly
    res = ly - X * β
    s2 = sum(abs2, res) / (n - 2)
    se = sqrt(s2 * inv(X' * X)[2, 2])
    h = t975(n - 2) * se
    return (β[2], β[2] - h, β[2] + h)
end
fmtci(t) = @sprintf("%.2f [%.2f, %.2f]", t...)

function writecsv(name, header, rows)
    open(joinpath(OUT, name), "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join([x isa AbstractFloat ? @sprintf("%.6g", x) : string(x) for x in r], ","))
        end
    end
end

summary = IOBuffer()
println(summary, "# Keypoint covariance vs Zeisl et al. (BMVC 2009)\n")
println(summary, "Device: $(typeof(device)); trials: $NTR (calibration, §5), $NSW (sweeps). Zeisl-5×5 weight σ = $(ZEISL_W5_SIGMA) level px.\n")

cF3 = zeisl_frobenius_constant(; window = 3, device)
cF5 = zeisl_frobenius_constant(; window = 5, device)
println(summary, @sprintf("Frobenius constants (bottom-layer DoG blob, amplitude 0.3): Zeisl-3×3 %.4g, Zeisl-5×5 %.4g.\n", cF3, cF5))

# ---------------------------------------------------------------- 1. calibration
rows = []
for c in configs()
    truth = [(x, y, s) for (x, y, s, _) in c.blobs]
    res = mc_run(c.img, truth; σn = c.σn, ntrials = NTR, seed = 1, device, tilts = c.tilts,
        baselines = true, gate = (2.0, 0.5))
    for (k, r) in enumerate(res)
        length(r.est) >= 0.5NTR || continue
        push!(rows, (cfg = c.name, k, blob = c.blobs[k], σn = c.σn, r))
    end
    @info "calibration" c.name
end
# fitted constants: pooled mean e₂ over all rows = 2
fitc(f) = mean(mean_e2(x.r, f(x.r))[1] for x in rows) / 2
c3 = fitc(r -> r.z3); c5 = fitc(r -> r.z5); ch = fitc(r -> posblock.(r.hess))
meths = [
    ("ours", r -> posblock.(r.cov), 1.0),
    ("Zeisl-3x3 Frobenius", r -> r.z3, cF3),
    ("Zeisl-5x5 Frobenius", r -> r.z5, cF5),
    ("Zeisl-3x3 fit", r -> r.z3, c3),
    ("Zeisl-5x5 fit", r -> r.z5, c5),
    ("Hess-3D fit (position block)", r -> posblock.(r.hess), ch),
]
calib = [(x.cfg, i, x.blob[3], x.blob[4], x.σn, length(x.r.est), [mean_e2(x.r, f(x.r); c)[1] for (_, f, c) in meths]...)
    for (i, x) in enumerate(rows)]
writecsv("calibration.csv", ["config", "row", "sigma", "amplitude", "sigma_n", "n", [replace(m[1], " " => "_") for m in meths]...], calib)
println(summary, "## 1. Calibration (position, mean e₂; target 2)\n")
println(summary, "Fitted constants: Zeisl-3×3 $(round(c3, sigdigits = 3)), Zeisl-5×5 $(round(c5, sigdigits = 3)), Hess-3D $(round(ch, sigdigits = 3)) (pooled mean e₂ = 2). $(length(rows)) blobs.\n")
println(summary, "| method | median e₂ | 10–90 % | blobs with e₂ ∈ [1.6, 2.4] | mean cov68 / cov95 |")
println(summary, "|---|---|---|---|---|")
for (j, (name, f, c)) in enumerate(meths)
    v = [x[6 + j] for x in calib]
    cv = [mean_e2(x.r, f(x.r); c)[2:3] for x in rows]
    @printf(summary, "| %s | %.3g | %.3g – %.3g | %d / %d | %.2f / %.2f |\n", name, median(v), quantile(v, 0.1), quantile(v, 0.9),
        count(e -> 1.6 <= e <= 2.4, v), length(v), mean(first.(cv)), mean(last.(cv)))
end
println(summary)

# ---------------------------------------------------------------- 2. contrast sweep
amps = exp.(range(log(0.03), log(1.0), length = 10))
σs_c = (2.5, 5.0, 10.0)
σn_c = 0.005
crow = []
for a in amps
    blobs = [(60.3 + 120 * (k - 1), 80.6, s, a) for (k, s) in enumerate(σs_c)]
    img = blob_image(170, 120 * length(σs_c) + 40, blobs)
    res = mc_run(img, [(b[1], b[2], b[3]) for b in blobs]; σn = σn_c, ntrials = NSW, seed = 2, device,
        baselines = true, gate = (2.0, 0.5), intensity_thr = 0.005)
    for (k, r) in enumerate(res)
        length(r.est) >= 0.5NSW || (push!(crow, (σs_c[k], a, length(r.est), NaN, NaN, NaN, NaN, NaN)); continue)
        push!(crow, (σs_c[k], a, length(r.est), empvar(r), meanvar(posblock.(r.cov)), meanvar(r.z3), meanvar(r.z5),
            mean_e2(r, posblock.(r.cov))[1]))
    end
    @info "contrast" a
end
writecsv("contrast_sweep.csv", ["sigma", "amplitude", "n", "var_empirical", "var_ours", "var_zeisl3_raw", "var_zeisl5_raw", "e2_ours"], crow)
println(summary, "## 2. Contrast sweep (σn = $σn_c, a ∈ [0.03, 1]): log-log slope of position variance vs a, 95 % CI\n")
println(summary, "| σ | empirical | ours | Zeisl-3×3 | Zeisl-5×5 | ours mean e₂ (range) |")
println(summary, "|---|---|---|---|---|---|")
cslopes = Dict()
for s in σs_c
    R = [r for r in crow if r[1] == s]
    a = [r[2] for r in R]
    sl = [loglog_slope(a, [r[j] for r in R]) for j in 4:7]
    cslopes[s] = sl
    e2 = filter(isfinite, [r[8] for r in R])
    @printf(summary, "| %.1f | %s | %s | %s | %s | %.2f (%.2f–%.2f) |\n", s, fmtci.(sl)..., mean(e2), extrema(e2)...)
end
println(summary)

# ---------------------------------------------------------------- 3. scale sweep
σs = exp.(range(log(1.6), log(40.0), length = 15))
# Five blobs per scale at seeded random sub-pixel positions, the medians over
# them: a single position can sit where noise flips the refinement between two
# start pixels or octaves (a two-mode estimate, see keypoint_uncertainty_mc.jl);
# the medians show the scaling, `switching` counts the blobs with ours e₂ > 4.
srow = []
for (q, s) in enumerate(σs)
    K = 5; L = max(96, ceil(Int, 14s))
    rng = Xoshiro(100 + q)
    blobs = [(L * (k - 0.5) + rand(rng), L / 2 + rand(rng), s, 0.3) for k in 1:K]
    img = blob_image(L, K * L, blobs)
    res = mc_run(img, [(b[1], b[2], b[3]) for b in blobs]; σn = 0.01, ntrials = NSW, seed = 3, device,
        baselines = true, gate = (2.0, 0.5))
    res = [r for r in res if length(r.est) >= 0.5NSW]
    if isempty(res)
        push!(srow, (s, 0, NaN, NaN, NaN, NaN, NaN, 0)); continue
    end
    e2 = [mean_e2(r, posblock.(r.cov))[1] for r in res]
    push!(srow, (s, length(res), median(empvar.(res)), median([meanvar(posblock.(r.cov)) for r in res]),
        median([meanvar(r.z3) for r in res]), median([meanvar(r.z5) for r in res]), median(e2), count(>(4), e2)))
    @info "scale" s
end
writecsv("scale_sweep.csv", ["sigma", "blobs", "var_empirical", "var_ours", "var_zeisl3_raw", "var_zeisl5_raw", "e2_ours", "switching"], srow)
println(summary, "## 3. Scale sweep (a = 0.3, σn = 0.01, σ ∈ [1.6, 40], medians over 5 blobs per σ): log-log slope vs σ, 95 % CI\n")
sx = [r[1] for r in srow]
ssl = [loglog_slope(sx, [r[j] for r in srow]) for j in 3:6]
clean = [r for r in srow if r[8] <= 1]
ssl2 = [loglog_slope([r[1] for r in clean], [r[j] for r in clean]) for j in 3:6]
println(summary, "| scales | empirical | ours | Zeisl-3×3 | Zeisl-5×5 |\n|---|---|---|---|---|")
println(summary, "| all $(length(srow)) | $(join(fmtci.(ssl), " | ")) |")
println(summary, "| $(length(clean)) with ≤ 1 switching blob | $(join(fmtci.(ssl2), " | ")) |\n")
println(summary, "Switching: noise moves the integer extremum the refinement starts from — a neighbouring pixel (a blob near a half level pixel) or, for a σ half-way between two DoG slices (s = 3·log2(σ/(1.6·2^o)) − ½ ≈ k + ½: σ 4.0, 5.05, 6.36, 10.1, 12.7, 40 here), the neighbouring slice and with it often a neighbouring pixel. The quadratic refinements from different starts give positions 0.1–0.3 px apart: a multi-modal estimate that no linearization describes.\n")
println(summary, "Per scale (Zeisl ratios: Frobenius-normalized σn-free variance over the empirical one, and the same with the calibration fit):\n")
println(summary, "| σ | blobs | empirical var (px²) | ours/emp | Zeisl-3×3 Frob/emp | Zeisl-3×3 fit/emp | Zeisl-5×5 fit/emp | ours e₂ (median) | switching |")
println(summary, "|---|---|---|---|---|---|---|---|---|")
for r in srow
    @printf(summary, "| %.2f | %d | %.3g | %.2f | %.3g | %.2f | %.2f | %.2f | %d |\n", r[1], r[2], r[3], r[4] / r[3], cF3 * r[5] / r[3],
        c3 * r[5] / r[3], c5 * r[6] / r[3], r[7], r[8])
end
println(summary, "\nOctave seams (top slice of one octave meets the bottom of the next) at σ ≈ 2.0, 4.0, 8.1, 16.1, 32.3.\n")

# ---------------------------------------------------------------- 4. Zeisl §5
# A DoG blob on the plane Z = 0, camera rotated by θ about the vertical axis
# through the blob at distance Dz = f (unit scale at the centre); the blob images
# at the principal point (cx, cy).
function s5_image(θ; L = 128, cx = 64.3, cy = 63.6, σb = 3.0, a = 0.3)
    f = 500.0
    Hinv(p) = (xp = (p[1] - cx) / f; yp = (p[2] - cy) / f;
        u = xp * f / (cos(θ) + xp * sin(θ)); v = yp * (f - u * sin(θ)); (u, v))
    return dog_blob_image(L, L, [(cx, cy, σb, a)]; H2 = Hinv), (cx, cy)
end
# Bhattacharyya distance of two zero-mean Gaussians
bhatt(A, B) = 0.5 * log(det((A + B) / 2) / sqrt(det(A) * det(B)))
fro1(A) = A ./ sqrt(sum(abs2, A))
s5 = []
θs = 0:10:60
for θd in θs
    img, (cx, cy) = s5_image(deg2rad(θd))
    det0 = asift_detect(Float32.(img); device, tilts = [(1.0, 0.0)])
    d0 = argmin(d -> hypot(d.center[1] - cx, d.center[2] - cy), det0)
    r = only(mc_run(img, [(d0.center[1], d0.center[2], d0.σ)]; σn = 0.02, ntrials = NTR, seed = 4, device,
        baselines = true, gate = (2.0, 0.5)))
    μ = meanpos(r)
    ΣML = sum((pos(v) - μ) * (pos(v) - μ)' for v in r.est) / (length(r.est) - 1)
    P = Dict("ours" => sum(posblock.(r.cov)) / length(r.cov),
        "Zeisl-3x3" => sum(filter(okΣ, r.z3)) / count(okΣ, r.z3),
        "Zeisl-5x5" => sum(filter(okΣ, r.z5)) / count(okΣ, r.z5))
    push!(s5, (θd, length(r.est), ΣML, P, hypot((μ .- (cx, cy))...)))
    @info "§5" θd
end
# the generous fit for §5: one constant per Zeisl variant over all angles (trace ratio 1 on geometric mean)
gfit(name) = exp(mean(log(tr(x[3]) / tr(x[4][name])) for x in s5))
f3 = gfit("Zeisl-3x3"); f5 = gfit("Zeisl-5x5")
s5rows = [(x[1], x[2], x[5], tr(x[3]) / 2,
    bhatt(fro1(x[3]), fro1(x[4]["ours"])), bhatt(fro1(x[3]), fro1(x[4]["Zeisl-3x3"])), bhatt(fro1(x[3]), fro1(x[4]["Zeisl-5x5"])),
    tr(x[4]["ours"]) / tr(x[3]), cF3 * tr(x[4]["Zeisl-3x3"]) / tr(x[3]), cF5 * tr(x[4]["Zeisl-5x5"]) / tr(x[3]),
    f3 * tr(x[4]["Zeisl-3x3"]) / tr(x[3]), f5 * tr(x[4]["Zeisl-5x5"]) / tr(x[3]),
    bhatt(x[3], x[4]["ours"]), bhatt(x[3], cF3 .* x[4]["Zeisl-3x3"]), bhatt(x[3], f3 .* x[4]["Zeisl-3x3"])) for x in s5]
writecsv("zeisl_s5.csv", ["theta_deg", "n", "bias_px", "var_ML", "DB_norm_ours", "DB_norm_zeisl3", "DB_norm_zeisl5",
    "trratio_ours", "trratio_zeisl3_frob", "trratio_zeisl5_frob", "trratio_zeisl3_fit", "trratio_zeisl5_fit",
    "DB_ours", "DB_zeisl3_frob", "DB_zeisl3_fit"], s5rows)
println(summary, "## 4. Zeisl §5 reproduction (DoG blob σ_b = 3, a = 0.3, σn = 0.02, viewpoint 0–60°)\n")
println(summary, "Normalized Bhattacharyya distance ×10³ (both covariances scaled to unit Frobenius norm, as Table 1), trace ratio predicted/ML, and unnormalized Bhattacharyya distance.\n")
println(summary, "| θ | n | ML var (px²) | D_B·10³ ours | D_B·10³ Z3 | D_B·10³ Z5 | tr ratio ours | Z3 Frob | Z5 Frob | Z3 fit | Z5 fit | D_B ours | D_B Z3 Frob | D_B Z3 fit |")
println(summary, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in s5rows
    @printf(summary, "| %d | %d | %.3g | %.2f | %.2f | %.2f | %.2f | %.3g | %.3g | %.2f | %.2f | %.3g | %.3g | %.3g |\n",
        x[1], x[2], x[4], 1e3 * x[5], 1e3 * x[6], 1e3 * x[7], x[8:12]..., x[13:15]...)
end
println(summary)

# ---------------------------------------------------------------- figures (pgfplots)
function figure(name, body; extra = "")
    tex = """
    \\documentclass{article}
    \\usepackage{pgfplots}
    \\usepackage[active,tightpage]{preview}
    \\PreviewEnvironment{tikzpicture}
    \\setlength\\PreviewBorder{2pt}
    \\pgfplotsset{compat=1.17}
    $extra
    \\begin{document}
    $body
    \\end{document}
    """
    write(joinpath(OUT, name * ".tex"), tex)
    if Sys.which("pdflatex") !== nothing
        ok = cd(OUT) do
            success(pipeline(`pdflatex -interaction=batchmode -halt-on-error $(name).tex`; stdout = devnull, stderr = devnull))
        end
        ok || @warn "pdflatex failed on $(name).tex (see $(name).log)"
    end
end

# 2. contrast sweep: one panel per σ, empirical marks, predictions lines
for s in σs_c
    cs = filter(r -> r[1] == s && isfinite(r[4]), crow)
    writecsv(@sprintf("contrast_sweep_s%.1f.csv", s), ["amplitude", "emp", "ours", "zeisl3", "zeisl5"],
        [(r[2], r[4], r[5], c3 * r[6], c5 * r[7]) for r in cs])
end
cpanels = join([
    """
    \\nextgroupplot[title={\$\\sigma = $(s)\$ px}]
    \\addplot[only marks, mark=*, black] table[x=amplitude, y=emp, col sep=comma] {$(@sprintf("contrast_sweep_s%.1f.csv", s))};
    \\addplot[thick, blue] table[x=amplitude, y=ours, col sep=comma] {$(@sprintf("contrast_sweep_s%.1f.csv", s))};
    \\addplot[thick, red, dashed] table[x=amplitude, y=zeisl3, col sep=comma] {$(@sprintf("contrast_sweep_s%.1f.csv", s))};
    \\addplot[thick, orange, dotted] table[x=amplitude, y=zeisl5, col sep=comma] {$(@sprintf("contrast_sweep_s%.1f.csv", s))};
    """ for s in σs_c], "\n")
figure("contrast_sweep", """
\\begin{tikzpicture}
\\begin{groupplot}[group style={group size=3 by 1, horizontal sep=2.1cm}, xmode=log, ymode=log,
  width=5.2cm, height=4.6cm, xlabel={contrast \$a\$}, ylabel={position variance (px\$^2\$)},
  legend to name=cleg, legend columns=4]
$cpanels
\\legend{empirical, ours, Zeisl \$3{\\times}3\$ (fit), Zeisl \$5{\\times}5\$ (fit)}
\\end{groupplot}
\\node at (current bounding box.south) [below] {\\pgfplotslegendfromname{cleg}};
\\end{tikzpicture}"""; extra = "\\usepgfplotslibrary{groupplots}")

# 3. scale sweep
writecsv("scale_sweep_plot.csv", ["sigma", "emp", "ours", "zeisl3", "zeisl5"],
    [(r[1], r[3], r[4], c3 * r[5], c5 * r[6]) for r in srow if isfinite(r[3])])
seams = join(["\\draw[gray, very thin] (axis cs:$s,1e-4) -- (axis cs:$s,10);"
    for s in (2.0, 4.03, 8.06, 16.1, 32.3)], "\n")
figure("scale_sweep", """
\\begin{tikzpicture}
\\begin{axis}[xmode=log, ymode=log, width=8cm, height=5.5cm, xlabel={blob \$\\sigma\$ (px)},
  ylabel={position variance (px\$^2\$)}, legend pos=north west, legend style={font=\\small}, ymin=1e-4, ymax=10]
\\addplot[only marks, mark=*, black] table[x=sigma, y=emp, col sep=comma] {scale_sweep_plot.csv};
\\addplot[thick, blue] table[x=sigma, y=ours, col sep=comma] {scale_sweep_plot.csv};
\\addplot[thick, red, dashed] table[x=sigma, y=zeisl3, col sep=comma] {scale_sweep_plot.csv};
\\addplot[thick, orange, dotted] table[x=sigma, y=zeisl5, col sep=comma] {scale_sweep_plot.csv};
\\legend{empirical, ours, Zeisl \$3{\\times}3\$ (fit), Zeisl \$5{\\times}5\$ (fit)}
$seams
\\end{axis}
\\end{tikzpicture}""")

# 1. calibration: mean e₂ per blob, per method
writecsv("calibration_plot.csv", ["row", "ours", "z3frob", "z3fit", "z5fit", "hess3d"],
    [(x[2], x[7], x[8], x[10], x[11], x[12]) for x in calib])
figure("calibration", """
\\begin{tikzpicture}
\\begin{axis}[ymode=log, width=12cm, height=5.5cm, xlabel={blob (configurations of the calibration study)},
  ylabel={mean \$e_2\$ (target 2)}, legend pos=outer north east, legend style={font=\\small}, ymin=1e-6, ymax=1e3, xmin=0, xmax=60]
\\addplot[only marks, mark=*, blue, mark size=1.5pt] table[x=row, y=ours, col sep=comma] {calibration_plot.csv};
\\addplot[only marks, mark=x, red] table[x=row, y=z3fit, col sep=comma] {calibration_plot.csv};
\\addplot[only marks, mark=+, orange] table[x=row, y=z5fit, col sep=comma] {calibration_plot.csv};
\\addplot[only marks, mark=triangle, gray] table[x=row, y=hess3d, col sep=comma] {calibration_plot.csv};
\\addplot[only marks, mark=square, red!50!black, mark size=1.2pt] table[x=row, y=z3frob, col sep=comma] {calibration_plot.csv};
\\addplot[black, dashed, domain=0:60] {2};
\\legend{ours, Zeisl \$3{\\times}3\$ fit, Zeisl \$5{\\times}5\$ fit, Hess-3D fit, Zeisl \$3{\\times}3\$ Frobenius}
\\end{axis}
\\end{tikzpicture}""")

# 4. §5: normalized D_B and the magnitude ratio
writecsv("zeisl_s5_plot.csv", ["theta", "db_ours", "db_z3", "db_z5", "tr_ours", "tr_z3frob", "tr_z3fit"],
    [(x[1], 1e3 * x[5], 1e3 * x[6], 1e3 * x[7], x[8], x[9], x[11]) for x in s5rows])
figure("zeisl_s5", """
\\begin{tikzpicture}
\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.8cm}, width=6cm, height=5cm, xlabel={viewpoint change (\$^\\circ\$)}]
\\nextgroupplot[ylabel={normalized \$D_B \\times 10^3\$}, legend pos=north west, legend style={font=\\small}]
\\addplot[thick, blue, mark=*] table[x=theta, y=db_ours, col sep=comma] {zeisl_s5_plot.csv};
\\addplot[thick, red, dashed, mark=x] table[x=theta, y=db_z3, col sep=comma] {zeisl_s5_plot.csv};
\\addplot[thick, orange, dotted, mark=+] table[x=theta, y=db_z5, col sep=comma] {zeisl_s5_plot.csv};
\\legend{ours, Zeisl \$3{\\times}3\$, Zeisl \$5{\\times}5\$}
\\nextgroupplot[ymode=log, ylabel={trace ratio predicted / ML}, ymin=0.1, ymax=1e4, legend pos=north east, legend style={font=\\small}]
\\addplot[thick, blue, mark=*] table[x=theta, y=tr_ours, col sep=comma] {zeisl_s5_plot.csv};
\\addplot[thick, red, dashed, mark=x] table[x=theta, y=tr_z3fit, col sep=comma] {zeisl_s5_plot.csv};
\\addplot[thick, red!50!black, mark=square] table[x=theta, y=tr_z3frob, col sep=comma] {zeisl_s5_plot.csv};
\\addplot[black, dashed, domain=0:60] {1};
\\legend{ours, Zeisl \$3{\\times}3\$ fit, Zeisl \$3{\\times}3\$ Frobenius}
\\end{groupplot}
\\end{tikzpicture}"""; extra = "\\usepgfplotslibrary{groupplots}")

for f in readdir(OUT)
    any(endswith(f, e) for e in (".aux",)) && rm(joinpath(OUT, f))
    endswith(f, ".log") && isfile(joinpath(OUT, replace(f, ".log" => ".pdf"))) && rm(joinpath(OUT, f))
end
write(joinpath(OUT, "summary.md"), String(take!(summary)))
println(read(joinpath(OUT, "summary.md"), String))
