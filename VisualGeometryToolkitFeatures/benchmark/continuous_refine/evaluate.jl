# Stage C (continuous DoG-extremum refinement, src/asift/continuous_refine.jl) against today's refinement (A), on
# synthetic Gaussian blobs, noise-free and deterministic, plus a noise MC for the NEES of C's covariance and a timing
# run. One process, every table written to `<out>.md` and the raw rows to `<out>.jls`.
#
#   julia --project=VisualGeometryToolkitFeatures/test evaluate.jl <out-prefix> [smoke|full] [farid_simoncelli_spatial|spline]
#
# The third argument picks the response: `farid_simoncelli_spatial` is the recipe, `ContinuousRefinement()` (C+B: the
# matched first-derivative fields for the spatial gradient, the unfiltered spline for the scale row); `spline` is stage
# C alone (the spline's exact derivatives). Tables label it "C". The earlier ablations (the prefiltered scale row with
# the d₂ or exact Jacobian, and the β² scale correction) are reproduced at commit 00874345.
#
# Ground truth: each blob's known centre, and its scale the continuous DoG extremum of the ANALYTIC scale space at the
# centre (the exact Gaussian convolution of the continuous blob: level σ ↦ blur τ² = σ² − 0.25 on the image's assumed
# input blur 0.5, as the pyramid; D(σ) = G(σ√k) − G(σ/√k), k = 2^(1/S), the reported σ of slice s being σ₀·2^((s+½)/S)).
# The same analytic response is the REFERENCE the product interpolant is measured against (value, gradient, Hessian).
# Detection candidates are FIXED: C starts from A's refined point of the same detection.
using Printf, Statistics, LinearAlgebra, Serialization
using StaticArrays
using StableRNGs: StableRNG
using VisualGeometryToolkitFeatures
import VisualGeometryToolkitFeatures as VgtkFeatures
using VisualGeometryToolkitFeatures: AsiftPlan, asift_detect!, asift_detections
using VisualGeometryToolkitCore: cov
import VisualGeometryToolkitCore as VgtkCore
import KernelAbstractions as KA
include_string(@__MODULE__, replace(read(joinpath(@__DIR__, "..", "..", "test", "uncertainty_testsetup.jl"), String),
    "@testsetup module" => "module"))
using .UncFixtures
using CUDA
const GPU = CUDA.functional()
const device = GPU ? CUDA.CUDABackend() : KA.CPU()

const OUT = ARGS[1]
const SMOKE = length(ARGS) > 1 && ARGS[2] == "smoke"
const MODE = length(ARGS) > 2 ? Symbol(ARGS[3]) : :farid_simoncelli_spatial
const S = 3; const σ0 = 1.6; const kS = 2.0^(1 / S)
const CFG = VgtkFeatures._continuous_refinement(MODE)
md = IOBuffer()
say(s...) = (println(md, s...); println(s...); flush(stdout))

# ----- the analytic scale space ------------------------------------------------
struct Blob
    c::SVector{2, Float64}        # centre, 0-indexed image px
    a::Float64
    Σb::SMatrix{2, 2, Float64, 4}
end
function G(b::Blob, p, σ)
    Σ = b.Σb + (σ^2 - 0.25) * I
    d = p - b.c
    return b.a * sqrt(det(b.Σb) / det(Σ)) * exp(-0.5 * dot(d, Σ \ d))
end
Dimg(b, p, σ) = G(b, p, σ * sqrt(kS)) - G(b, p, σ / sqrt(kS))
# level coordinates v (1-based x, y, slice) of octave o ↦ image point and σ
img_of(v, o) = (sf = 2.0^o; (SVector((v[1] - 1) * sf + sf - 1, (v[2] - 1) * sf + sf - 1), σ0 * 2.0^((v[3] - 1 + 0.5) / S) * sf))
level_of(p, σ, o) = (sf = 2.0^o; SVector((p[1] - (sf - 1)) / sf + 1, (p[2] - (sf - 1)) / sf + 1, S * log2(σ / (σ0 * sf)) - 0.5 + 1))
Dref(b, v, o) = ((p, σ) = img_of(v, o); Dimg(b, p, σ))
function ref_response(b, v, o; h = 1.0e-3)
    f(u) = Dref(b, u, o)
    e(i) = SVector(ntuple(j -> j == i ? h : 0.0, 3))
    D = f(v)
    g = SVector(ntuple(i -> (f(v + e(i)) - f(v - e(i))) / 2h, 3))
    H = SMatrix{3, 3}(ntuple(k -> begin
        i = (k - 1) % 3 + 1; j = (k - 1) ÷ 3 + 1
        i == j ? (f(v + e(i)) - 2D + f(v - e(i))) / h^2 :
            (f(v + e(i) + e(j)) - f(v + e(i) - e(j)) - f(v - e(i) + e(j)) + f(v - e(i) - e(j))) / 4h^2
    end, 9))
    return D, g, H
end
# the truth scale: argmax |D(c, σ)| (golden section in ln σ, then Newton on the analytic profile)
function truth_sigma(b)
    f(l) = -abs(Dimg(b, b.c, exp(l)))
    σg = sqrt(sqrt(det(b.Σb)))
    lo, hi = log(0.4σg), log(3σg)
    φ = (sqrt(5) - 1) / 2
    x1 = hi - φ * (hi - lo); x2 = lo + φ * (hi - lo)
    for _ in 1:200
        f(x1) < f(x2) ? (hi = x2; x2 = x1; x1 = hi - φ * (hi - lo)) : (lo = x1; x1 = x2; x2 = lo + φ * (hi - lo))
    end
    l = (lo + hi) / 2; h = 1.0e-4
    for _ in 1:5
        d1 = (f(l + h) - f(l - h)) / 2h; d2 = (f(l + h) - 2f(l) + f(l - h)) / h^2
        l -= d1 / d2
    end
    return exp(l)
end

# ----- detection, A and C -------------------------------------------------------
function nearest(dets, x, y, ls; gate = (2.0, 0.6))
    best = 0; bd = Inf
    for (i, d) in enumerate(dets)
        dx = d.center[1] - x; dy = d.center[2] - y; dl = log(d.σ) - ls
        (hypot(dx, dy) < gate[1] && abs(dl) < gate[2]) || continue
        dd = dx^2 + dy^2 + (dl / 0.1)^2
        dd < bd && (bd = dd; best = i)
    end
    return best
end

# per blob: the A detection's level start, grouped by octave; C on every group; returns rows
function detect_and_refine(plan, img, blobs::Vector{Blob}; dev = device, with_cov = false)
    dets = detect_on(plan, img)
    found = Vector{Any}(nothing, length(blobs))
    for (k, b) in enumerate(blobs)
        σg = sqrt(sqrt(det(b.Σb)))
        i = nearest(dets, b.c[1] + 0.5, b.c[2] + 0.5, log(σg); gate = (max(2.0, 0.5σg), 0.6))
        i == 0 && continue
        e = extremum_of(plan, dets[i])
        e === nothing && continue
        vA = SVector(e.X + e.off[2], e.Y + e.off[3], e.sl + e.off[1])
        found[k] = (; det = dets[i], e, vA)
    end
    C = Vector{Any}(nothing, length(blobs))
    tim = (; coefficients = 0.0, refinement = 0.0, n = 0)
    for oi in unique(f.e.oi for f in found if f !== nothing)
        ks = [k for k in eachindex(found) if found[k] !== nothing && found[k].e.oi == oi]
        starts = [(found[k].vA..., 1) for k in ks]
        ext, wins, t = VgtkFeatures._refine_dog_extrema(dev, plan.dogs[oi], starts; seed_sigma = plan.space.seed_sigma, S = plan.space.nb_scales, cfg = CFG)
        tim = (; coefficients = tim.coefficients + t.coefficients, refinement = tim.refinement + t.refinement, n = tim.n + length(ks))
        for (j, k) in enumerate(ks)
            Σ = with_cov && VgtkCore.successful_retcode(ext[j].retcode) ? VgtkFeatures._continuous_covariance(plan, oi, 1, wins[j], ext[j], CFG) : nothing
            C[k] = (; ext = ext[j], win = wins[j], Σ)
        end
    end
    return found, C, tim
end

# ----- 1. phase sweeps -----------------------------------------------------------
const PHX = SMOKE ? collect(0:0.25:1.0) : collect(0:0.05:1.0)
const PHY = SMOKE ? [0.0, 0.5] : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
Arot(θ, q) = SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ)) * SMatrix{2, 2}(sqrt(q), 0.0, 0.0, 1 / sqrt(q))

function sweep(o, σb; q = 1.0, θ = 0.0, phx = PHX, phy = PHY)
    sf = 2.0^o
    A = Arot(θ, q)
    ext = σb * sqrt(max(q, 1 / q))
    N = ceil(Int, max(14ext, 40.0) / sf); m = ceil(Int, max(9ext, 30.0) / sf)
    nl = 2m + (length(phx) - 1) * N + 2
    W = 32 * cld(max(512, ceil(Int, sf * nl) + 8), 32); H = 32 * cld(max(512, ceil(Int, sf * (2m + 2)) + 8), 32)
    plan = AsiftPlan(W, H; device, tilts = [(1.0, 0.0)], upsample = true)
    rows = []
    tim = (; coefficients = 0.0, refinement = 0.0, n = 0)
    for φy in phy
        blobs = Blob[]
        for (jx, φx) in enumerate(phx)
            c = SVector(sf * (m + (jx - 1) * N + φx) + sf - 1, sf * (m + φy) + sf - 1)
            push!(blobs, Blob(c, 0.3, σb^2 * A * A'))
        end
        img = blob_image(H, W, [(b.c[1] + 0.5, b.c[2] + 0.5, σb, b.a) for b in blobs]; A = Matrix(A))
        found, C, t = detect_and_refine(plan, img, blobs)
        tim = (; coefficients = tim.coefficients + t.coefficients, refinement = tim.refinement + t.refinement, n = tim.n + t.n)
        for (jx, b) in enumerate(blobs)
            f = found[jx]
            if f === nothing
                push!(rows, (; o, σb, q, θ, φx = phx[jx], φy, ok = false)); continue
            end
            od = f.e.o
            σt = truth_sigma(b)
            vt = level_of(b.c, σt, od)
            r = C[jx].ext
            # interpolant vs reference: at the truth point, at C's point, and at 8 seeded points of the box
            win = C[jx].win
            rng = StableRNG(hash((o, σb, q, θ, jx, φy)) % UInt32)
            pts = vcat([vt, r.v, round.(vt)], [f.vA + SVector(2rand(rng) - 1, 2rand(rng) - 1, 2rand(rng) - 1) .* SVector(CFG.spatial_bound, CFG.spatial_bound, CFG.scale_bound) for _ in 1:8])
            ierr = map(pts) do v
                lo, hi = VgtkFeatures._model_box(win)
                all(lo .<= v .<= hi) || return (NaN, NaN, NaN, NaN)
                Ds, gs, Hs = VgtkFeatures._dog_response(win, v)
                Dr, gr, Hr = ref_response(b, v, od)
                Dt = abs(Dref(b, vt, od))
                δ = Hr \ (gs - gr)            # the extremum displacement the gradient error implies (level px, slice)
                (abs(Ds - Dr) / Dt, norm(gs - gr) / Dt, norm(Hs - Hr) / norm(Hr), norm(δ[1:2]))
            end
            push!(rows, (; o, σb, q, θ, φx = phx[jx], φy, ok = true, od, vt, vA = f.vA, vC = r.v,
                code = r.retcode, rule = r.stop_rule, ierr, Dt = Dref(b, vt, od)))
        end
    end
    return rows, tim
end

configs = SMOKE ? [(0, 2.85), (2, 3.36)] :
    [(o, f * 2.0^o) for o in -1:3 for f in (2.05, 2.42, 2.85, 3.36, 3.95)]
ellipses = SMOKE ? [(1, 2.85, 1.6, π / 6)] : [(o, 2.85 * 2.0^o, 1.6, θ) for o in (0, 2) for θ in (0.0, π / 6, π / 3)]
allrows = []; sweep_time = (; coefficients = 0.0, refinement = 0.0, n = 0)
for (o, σb) in configs
    t = @elapsed (r, tm) = sweep(o, σb)
    append!(allrows, r); global sweep_time = map(+, sweep_time, tm)
    @info "sweep" o σb t count(x -> x.ok, r) length(r)
end
for (o, σb, q, θ) in ellipses
    (r, tm) = sweep(o, σb; q, θ, phy = SMOKE ? [0.0] : [0.0, 0.25, 0.5])
    append!(allrows, r); global sweep_time = map(+, sweep_time, tm)
    @info "ellipse" o σb q θ count(x -> x.ok, r)
end
serialize(OUT * "_sweep.jls", allrows)

# ----- tables ---------------------------------------------------------------------
lσ(v3) = (v3 - 1 + 0.5) * log(2) / S
errs(r, v) = (v[1] - r.vt[1], v[2] - r.vt[2], lσ(v[3]) - lσ(r.vt[3]))
okC(r) = VgtkCore.successful_retcode(r.code)
fm(x) = isnan(x) ? "—" : @sprintf("%+.4f", x)
f4(x) = isnan(x) ? "—" : @sprintf("%.4f", x)
e3(x) = isnan(x) ? "—" : @sprintf("%.1e", x)

say("# ", MODE === :spline ? "Stage C" : "The recipe, C+B (spatial-only Farid–Simoncelli)", " (continuous refinement; labelled C below) vs A (today), noise-free synthetic blobs\n")
say("Device: ", GPU ? "CUDA" : "CPU", ". Errors in LEVEL px of the detecting octave (×2^o for image px); scale error in ln σ. ",
    "Truth: the known centre and the analytic continuous-DoG scale extremum. C starts from A's point of the same detection.\n")
say("## Table 1. Bias over the full phase grid, per configuration (A / C)\n")
say("| case | o | σ_b (img px) | n | C fail | mean b_x A / C | max ‖b‖ A / C | rms ‖b‖ A / C | max |b_lnσ| A / C | rms b_lnσ A / C |")
say("|---|---|---|---|---|---|---|---|---|---|")
groups = unique((r.o, r.σb, r.q, r.θ) for r in allrows)
for gk in groups
    rs = [r for r in allrows if (r.o, r.σb, r.q, r.θ) == gk && r.ok]
    nall = count(r -> (r.o, r.σb, r.q, r.θ) == gk, allrows)
    isempty(rs) && continue
    eA = [errs(r, r.vA) for r in rs]; eC = [errs(r, r.vC) for r in rs if okC(r)]
    nb(e) = hypot(e[1], e[2])
    st(E) = isempty(E) ? (NaN, NaN, NaN, NaN, NaN) :
        (mean(first, E), maximum(nb, E), sqrt(mean(e -> nb(e)^2, E)), maximum(e -> abs(e[3]), E), sqrt(mean(e -> e[3]^2, E)))
    a = st(eA); c = st(eC)
    case = gk[3] == 1 ? "iso" : @sprintf("ell q=%.1f θ=%.0f°", gk[3], rad2deg(gk[4]))
    say(@sprintf("| %s | %d | %.2f | %d/%d | %d | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s |", case, gk[1], gk[2], length(rs), nall,
        count(r -> !okC(r), rs), fm(a[1]), fm(c[1]), f4(a[2]), f4(c[2]), f4(a[3]), f4(c[3]), f4(a[4]), f4(c[4]), f4(a[5]), f4(c[5])))
end

say("\n## Table 2. Signed bias b_x(φx) at φy = 0 (level px), A / C — no averaging over phase\n")
phs = SMOKE ? PHX : [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
say("| case | o | σ_b | ", join([@sprintf("%.2f", p) for p in phs], " | "), " | max step A / C |")
say("|---|---|---|", repeat("---|", length(phs)), "---|")
for gk in groups
    rs = sort([r for r in allrows if (r.o, r.σb, r.q, r.θ) == gk && r.φy == 0.0 && r.ok], by = r -> r.φx)
    isempty(rs) && continue
    cell(p) = (i = findfirst(r -> abs(r.φx - p) < 1.0e-9, rs); i === nothing ? "—" :
        @sprintf("%+.4f / %s", errs(rs[i], rs[i].vA)[1], okC(rs[i]) ? @sprintf("%+.5f", errs(rs[i], rs[i].vC)[1]) : "fail"))
    bA = [errs(r, r.vA)[1] for r in rs]; bC = [okC(r) ? errs(r, r.vC)[1] : NaN for r in rs]
    stp(b) = length(b) < 2 ? NaN : maximum(abs, filter(!isnan, diff(b)); init = 0.0)
    case = gk[3] == 1 ? "iso" : @sprintf("ell θ=%.0f°", rad2deg(gk[4]))
    say("| $case | $(gk[1]) | $(@sprintf("%.2f", gk[2])) | ", join(cell.(phs), " | "), " | ", f4(stp(bA)), " / ", f4(stp(bC)), " |")
end

say("\n## Table 3. The product interpolant's measured error vs the analytic reference\n")
say("Relative to |D| at the truth point: value error, gradient error (level units), Hessian relative error, and the ",
    "position displacement the gradient error implies (‖H⁻¹δg‖ spatial, level px). Max over blobs at the truth point / at C's point / ",
    "at the integer node nearest the truth (for C, where the cardinal spline reproduces the pyramid's own sample: the PYRAMID's error ",
    "vs the continuous scale space, not the interpolant's) / over 8 seeded box points (the implied δx there is ill-conditioned away from the extremum).\n")
say("| o | n | |δD|/|D| truth / C / node / box | ‖δg‖/|D| truth / C / node / box | ‖δH‖/‖H‖ truth / C / node / box | implied δx truth / C / node / box (level px) |")
say("|---|---|---|---|---|---|")
for o in sort(unique(r.od for r in allrows if r.ok))
    rs = [r for r in allrows if r.ok && r.od == o]
    m(i, which) = (v = filter(x -> isfinite(x), [which == :box ? maximum(x -> isnan(x[i]) ? -Inf : x[i], r.ierr[4:end]) : r.ierr[which == :truth ? 1 : which == :C ? 2 : 3][i] for r in rs]); isempty(v) ? NaN : maximum(v))
    say("| $o | $(length(rs)) | ", join([join(e3.([m(i, :truth), m(i, :C), m(i, :node), m(i, :box)]), " / ") for i in 1:4], " | "), " |")
end

say("\n## Table 4. C's termination records (§14)\n")
codes = Dict{Any, Int}()
for r in allrows
    r.ok || continue
    codes[(r.code, r.rule)] = get(codes, (r.code, r.rule), 0) + 1
end
for (k, v) in sort(collect(codes), by = last, rev = true)
    say("- ", k[1], " / ", k[2], ": ", v)
end
say("- blobs without an A detection: ", count(r -> !r.ok, allrows), " of ", length(allrows))
say(@sprintf("\nSweep kernel time (%s): coefficients %.3g s, refinement %.3g s over %d detections (%.2f / %.2f μs per detection; includes per-call launch latency).",
    GPU ? "GPU" : "CPU", sweep_time.coefficients, sweep_time.refinement, sweep_time.n,
    1.0e6 * sweep_time.coefficients / max(sweep_time.n, 1), 1.0e6 * sweep_time.refinement / max(sweep_time.n, 1)))
write(OUT * ".md", String(take!(copy(md))))

# ----- 1b. tilted ASIFT views ----------------------------------------------------------------------------------------
# One tilted warp per plan; truth in the warp canvas: the centre c ↦ A⁻¹(c − t) (A, t the canvas → image map of
# `_bbackproject!`), the blob's canvas covariance A⁻¹Σ_bA⁻ᵀ plus the warp's anti-alias blur along the canvas y
# ((0.8²(t² − 1))/t², `sigma_aa_taps`), its scale truth the analytic DoG extremum of that canvas blob (approximate: the
# warp's bilinear resampling is not modelled; the centre is exact).
function tilt_sweep(o, σb, tl, φt; phx = SMOKE ? [0.0, 0.5] : collect(0:0.1:0.9), phy = SMOKE ? [0.0] : [0.0, 0.5])
    sf = 2.0^o
    N = ceil(Int, max(14σb * tl, 40.0)); m = ceil(Int, max(9σb * tl, 30.0))
    W = 32 * cld(max(512, m * 2 + N * length(phx)), 32); H = 32 * cld(max(512, 2m + N * length(phy)), 32)
    plan = AsiftPlan(W, H; device, tilts = [(tl, φt)], upsample = true)
    wp = Array(plan.wp)[:, 1]
    A = SMatrix{2, 2}(wp[1], wp[4], wp[2], wp[5]); tv = SVector(wp[3], wp[6]); Ai = inv(A)
    aa = tl > 1 ? 0.8^2 * (tl^2 - 1) / tl^2 : 0.0
    blobs = Blob[]; cblobs = Blob[]
    for (jy, φy) in enumerate(phy), (jx, φx) in enumerate(phx)
        c = SVector(m + (jx - 1) * N + φx, m + (jy - 1) * N + φy)
        push!(blobs, Blob(c, 0.3, σb^2 * SMatrix{2, 2}(1.0, 0, 0, 1.0)))
        push!(cblobs, Blob(Ai * (c - tv), 0.3, Ai * (σb^2 * SMatrix{2, 2}(1.0, 0, 0, 1.0)) * Ai' + SMatrix{2, 2}(0.0, 0, 0, aa)))
    end
    img = blob_image(H, W, [(b.c[1] + 0.5, b.c[2] + 0.5, σb, b.a) for b in blobs])
    found, C, _ = detect_and_refine(plan, img, blobs)
    rows = []
    for (k, cb) in enumerate(cblobs)
        f = found[k]
        f === nothing && (push!(rows, (; o, σb, tl, φt, ok = false)); continue)
        od = f.e.o; sfd = 2.0^od
        vt = level_of(cb.c, truth_sigma(cb), od)
        r = C[k].ext
        img_err(v) = (p = A * SVector((v[1] - 1) * sfd + sfd - 1, (v[2] - 1) * sfd + sfd - 1) + tv; norm(p - blobs[k].c))
        push!(rows, (; o, σb, tl, φt, ok = true, od, vt, vA = f.vA, vC = r.v, code = r.retcode, eAi = img_err(f.vA), eCi = img_err(r.v)))
    end
    return rows
end
tilts = SMOKE ? [(0, 2.85, 2.0, π / 4)] : [(o, 2.85 * 2.0^o, tl, φt) for o in (0, 2) for (tl, φt) in ((2.0, 0.0), (2.0, π / 4), (2 * sqrt(2), π / 6))]
tiltrows = []
for (o, σb, tl, φt) in tilts
    append!(tiltrows, tilt_sweep(o, σb, tl, φt))
    @info "tilt" o σb tl φt
end
say("\n## Table 7. Tilted ASIFT views (one warp per plan): bias A / C, canvas level px and image px\n")
say("| o | σ_b | tilt t | φ | n | C fail | max ‖b‖ canvas A / C | rms ‖b‖ canvas A / C | max ‖b‖ image px A / C | max |b_lnσ| A / C |")
say("|---|---|---|---|---|---|---|---|---|---|")
for gk in unique((r.o, r.σb, r.tl, r.φt) for r in tiltrows)
    rs = [r for r in tiltrows if (r.o, r.σb, r.tl, r.φt) == gk && r.ok]
    isempty(rs) && continue
    nb(r, v) = hypot(v[1] - r.vt[1], v[2] - r.vt[2])
    ok = [r for r in rs if okC(r)]
    mx(f, xs) = isempty(xs) ? NaN : maximum(f, xs)
    rm(f, xs) = isempty(xs) ? NaN : sqrt(mean(x -> f(x)^2, xs))
    say(@sprintf("| %d | %.2f | %.2f | %.0f° | %d | %d | %s / %s | %s / %s | %s / %s | %s / %s |", gk[1], gk[2], gk[3], rad2deg(gk[4]), length(rs), length(rs) - length(ok),
        f4(mx(r -> nb(r, r.vA), rs)), f4(mx(r -> nb(r, r.vC), ok)), f4(rm(r -> nb(r, r.vA), rs)), f4(rm(r -> nb(r, r.vC), ok)),
        f4(mx(r -> r.eAi, rs)), f4(mx(r -> r.eCi, ok)),
        f4(mx(r -> abs(lσ(r.vA[3]) - lσ(r.vt[3])), rs)), f4(mx(r -> abs(lσ(r.vC[3]) - lσ(r.vt[3])), ok))))
end
serialize(OUT * "_tilt.jls", tiltrows)
write(OUT * ".md", String(take!(copy(md))))

# ----- 2. noise MC: NEES of C's covariance (about clean C) vs A's FeatureCovariance (about clean A) -----------------
function mc_layout(o)
    rng = StableRNG(100 + o)
    σs = [2.42, 2.85, 3.36] .* 2.0^o
    sp = max(14 * maximum(σs), 40.0); m = max(8 * maximum(σs), 24.0)
    W = 32 * cld(max(512, ceil(Int, 2m + 3sp)), 32); H = 32 * cld(max(512, ceil(Int, 2m + 2sp)), 32)
    bl = Blob[]; k = 0
    for (r, s) in enumerate(σs), c in 1:4
        k += 1
        push!(bl, Blob(SVector(m + (c - 1) * sp + 4 * rand(rng) - 0.5, m + (r - 1) * sp + 4 * rand(rng) - 0.5), isodd(k) ? 0.3 : -0.3, s^2 * SMatrix{2, 2}(1.0, 0, 0, 1.0)))
    end
    return H, W, bl
end
outC(v, o) = ((p, σ) = img_of(v, o); SVector(p[1], p[2], log(σ)))
outA(d) = SVector(d.center[1] - 0.5, d.center[2] - 0.5, log(d.σ))
ntr = SMOKE ? 20 : 300
say("\n## Table 5. NEES under iid pixel noise (error about each blob's clean detection; ", ntr, " trials × 12 blobs per octave)\n")
say("| σn | o | n A / C | NEES₂ A / C | NEES₃ A / C | cov₂ 90 % A / C | emp/pred var x A / C | emp/pred var lnσ A / C |")
say("|---|---|---|---|---|---|---|---|")
mcrows = []
for σn in (0.01, 0.001), o in (SMOKE ? (0, 2) : (-1, 0, 1, 2, 3))
    H, W, bl = mc_layout(o)
    plan = AsiftPlan(W, H; device, tilts = [(1.0, 0.0)], upsample = true)
    img = blob_image(H, W, [(b.c[1] + 0.5, b.c[2] + 0.5, sqrt(b.Σb[1, 1]), b.a) for b in bl])
    f0, C0, _ = detect_and_refine(plan, img, bl)
    rng = StableRNG(7 + o + round(Int, 1.0e4σn))
    sA = Tuple{SVector{3, Float64}, SMatrix{3, 3, Float64, 9}}[]; sC = similar(sA)
    for _ in 1:ntr
        noisy = img .+ σn .* randn(rng, size(img))
        f, C, _ = detect_and_refine(plan, noisy, bl; with_cov = true)
        for k in eachindex(bl)
            (f[k] === nothing || f0[k] === nothing) && continue
            f[k].e.oi == f0[k].e.oi || continue
            push!(sA, (outA(f[k].det) - outA(f0[k].det), σn^2 * SMatrix{3, 3}(cov(f[k].det))))
            c = C[k]; c0 = C0[k]
            if c !== nothing && c0 !== nothing && c.Σ !== nothing && okC((; code = c0.ext.retcode))
                push!(sC, (outC(c.ext.v, f[k].e.o) - outC(c0.ext.v, f0[k].e.o), σn^2 * c.Σ))
            end
        end
    end
    nees(s, d) = mean(e -> (E = e[1][1:d]; dot(E, Symmetric(Matrix(e[2][1:d, 1:d])) \ E)), s)
    cov90(s) = mean(e -> (E = e[1][1:2]; dot(E, Symmetric(Matrix(e[2][1:2, 1:2])) \ E) <= 4.605), s)
    vr(s, i) = var([e[1][i] for e in s]) / mean(e -> e[2][i, i], s)
    q(s, f) = isempty(s) ? NaN : f(s)
    say(@sprintf("| %g | %d | %d / %d | %.3f / %.3f | %.3f / %.3f | %.3f / %.3f | %.3f / %.3f | %.3f / %.3f |", σn, o, length(sA), length(sC),
        q(sA, s -> nees(s, 2)), q(sC, s -> nees(s, 2)), q(sA, s -> nees(s, 3)), q(sC, s -> nees(s, 3)), q(sA, cov90), q(sC, cov90),
        q(sA, s -> vr(s, 1)), q(sC, s -> vr(s, 1)), q(sA, s -> vr(s, 3)), q(sC, s -> vr(s, 3))))
    push!(mcrows, (; σn, o, sA, sC))
    write(OUT * ".md", String(take!(copy(md))))
end
serialize(OUT * "_mc.jls", mcrows)

# ----- 3. timing (dense scene) and CPU = GPU ----------------------------------------------------------------------------
function dense_scene(H, W; seed = 9)
    rng = StableRNG(seed)
    img = fill(0.5, H, W)
    for _ in 1:600
        cx, cy, σ, a = W * rand(rng), H * rand(rng), 1.5 + 14rand(rng), 0.4 * (rand(rng) - 0.5)
        r = ceil(Int, 4σ)
        for j in max(1, floor(Int, cx) - r):min(W, floor(Int, cx) + r), i in max(1, floor(Int, cy) - r):min(H, floor(Int, cy) + r)
            img[i, j] += a * exp(-((j - 0.5 - cx)^2 + (i - 0.5 - cy)^2) / (2σ^2))
        end
    end
    return img .+ 0.005 .* randn(rng, H, W)
end
say("\n## Table 6. Time per detection (dense 1536×1024 scene, identity tilt, upsample; warm, min of 3)\n")
H, W = 1024, 1536
img = dense_scene(H, W)
say("| device | detections | A: asift_detect! (whole detector) per detection | C: coefficient kernel per detection | C: refinement kernel per detection | C total / A |")
say("|---|---|---|---|---|---|")
results = Dict{Symbol, Any}()
for (name, dev) in (GPU ? [(:gpu, device), (:cpu, KA.CPU())] : [(:cpu, KA.CPU())])
    plan = AsiftPlan(W, H; device = dev, tilts = [(1.0, 0.0)], upsample = true)
    xy = permutedims(Float32.(img))
    asift_detect!(plan, xy; intensity_thr = 0.02)
    tA = minimum(@elapsed(asift_detect!(plan, xy; intensity_thr = 0.02)) for _ in 1:3)
    dets = asift_detections(asift_detect!(plan, xy; intensity_thr = 0.02))
    es = [extremum_of(plan, d) for d in dets]
    tc = 0.0; tr = 0.0; n = 0; vs = Dict{Int, Any}()
    for oi in unique(e.oi for e in es if e !== nothing)
        st = [(e.X + e.off[2], e.Y + e.off[3], e.sl + e.off[1], 1) for e in es if e !== nothing && e.oi == oi]
        VgtkFeatures._refine_dog_extrema(dev, plan.dogs[oi], st; seed_sigma = plan.space.seed_sigma, S = plan.space.nb_scales, cfg = CFG)
        best = (Inf, Inf); ext = nothing
        for _ in 1:3
            ext, _, t = VgtkFeatures._refine_dog_extrema(dev, plan.dogs[oi], st; seed_sigma = plan.space.seed_sigma, S = plan.space.nb_scales, cfg = CFG)
            best = (min(best[1], t.coefficients), min(best[2], t.refinement))
        end
        tc += best[1]; tr += best[2]; n += length(st)
        # CPU = GPU: the same kernels on the host, on THIS device's DoG copied over (one pyramid, two backends), and the
        # attribution of their gap: (i) the coefficients' relative difference; (ii) the GPU LM on the CPU's coefficients
        # (one model, two LMs); (iii) both backends again (Core's StaticLmStops: the cost floor is 0, the LM stops only at rounding)
        if name == :gpu
            kw = (; seed_sigma = plan.space.seed_sigma, S = plan.space.nb_scales)
            eg, wg, _ = VgtkFeatures._refine_dog_extrema(dev, plan.dogs[oi], st; kw..., cfg = CFG)
            dogh = Array(plan.dogs[oi])
            ec, wc, _ = VgtkFeatures._refine_dog_extrema(KA.CPU(), dogh, st; kw..., cfg = CFG)
            cg = parent(wg[1].coef); cc = parent(wc[1].coef)
            el, _, _ = VgtkFeatures._refine_dog_extrema(dev, plan.dogs[oi], st; kw..., cfg = CFG, coefficients = cc)
            CF0 = CFG
            eg0, _, _ = VgtkFeatures._refine_dog_extrema(dev, plan.dogs[oi], st; kw..., cfg = CF0)
            ec0, _, _ = VgtkFeatures._refine_dog_extrema(KA.CPU(), dogh, st; kw..., cfg = CF0)
            vs[oi] = (; eg, ec, el, eg0, ec0, dcoef = maximum(abs, cg .- cc) / maximum(abs, cc))
        end
    end
    results[name] = vs
    say(@sprintf("| %s | %d | %.2f μs | %.2f μs | %.2f μs | %.2f |", name, n, 1.0e6tA / max(length(dets), 1), 1.0e6tc / max(n, 1),
        1.0e6tr / max(n, 1), (tc + tr) / tA))
end
if haskey(results, :gpu)
let
    R = results[:gpu]
    dv(f, g) = maximum(maximum(maximum(abs, a.v - b.v) for (a, b) in zip(f(r), g(r))) for r in values(R))
    nrec(f, g) = sum(count(a.retcode != b.retcode || a.stop_rule != b.stop_rule for (a, b) in zip(f(r), g(r))) for r in values(R))
    nbit = sum(count(a.v !== b.v for (a, b) in zip(r.eg, r.ec)) for r in values(R))
    nt = sum(length(r.eg) for r in values(R))
    tol = VgtkFeatures._cr("cpu_gpu_tolerance")
    say(@sprintf("\nCPU = GPU (the same DoG, both backends' kernels), %d detections: records differing %d; max |Δv| = %.3g level units (tolerance %.3g, derived in the TOML); %d of %d not bit-identical.",
        nt, nrec(r -> r.eg, r -> r.ec), dv(r -> r.eg, r -> r.ec), tol, nbit, nt))
    say(@sprintf("Attribution: the coefficients differ by up to %.3g of their range (Float32 contraction); the GPU LM on the CPU's coefficients is %.3g from the CPU (records differing %d); with the cost floor at 0 the two backends are %.3g apart (records differing %d).",
        maximum(r.dcoef for r in values(R)), dv(r -> r.el, r -> r.ec), nrec(r -> r.el, r -> r.ec), dv(r -> r.eg0, r -> r.ec0), nrec(r -> r.eg0, r -> r.ec0)))
end
end
write(OUT * ".md", String(take!(copy(md))))
println("done ", OUT)
