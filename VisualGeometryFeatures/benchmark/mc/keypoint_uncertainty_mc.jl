# Full Monte Carlo study of the ASIFT keypoint covariance
# (`asift_detect(img; uncertainty = FeatureCovariance())`). Not part of the test
# suite: many configurations and trials, on a GPU when one is available.
#
#   julia --project=<an env with VisualGeometryFeatures, CUDA> benchmark/mc/keypoint_uncertainty_mc.jl [out.md]
#
# For each configuration: N noisy trials of a synthetic image with known blobs
# (iid Gaussian noise of std σn on the float input path: no quantization), each
# detection associated to its blob, and per blob the calibration of the
# per-trial predicted σn²·Σ̃ against the empirical spread (see `mc_summary`):
# mean normalized squared error (≈ 3 over (x, y, log σ), ≈ 2 over the position
# block), coverage at the χ² 68.27 % / 95 % quantiles, per-axis variance ratio
# empirical/predicted, and the bias of the mean against the truth. The same
# statistics for the 3D Hessian-only baseline "Hess-3D", Σ ∝ J(∓Ĥ)⁻¹Jᵀ with Ĥ
# the detector's 3×3 Hessian in (s, x, y), its scale fitted once over all
# configurations (Σ = c·σn²·J(∓Ĥ)⁻¹Jᵀ: the best a single scale can do). The
# position-only comparison with Zeisl et al. (2009) is zeisl_comparison.jl.
#
# Writes a markdown table to `out.md` (default: a temp file) and prints it.

using Printf, Statistics, LinearAlgebra
using VisualGeometryFeatures
import KernelAbstractions as KA

const _HERE = @__DIR__
include_string(@__MODULE__, replace(read(joinpath(_HERE, "..", "..", "test", "uncertainty_testsetup.jl"), String),
    "@testsetup module" => "module"))
using .UncFixtures

device = try
    @eval using CUDA
    CUDA.functional() ? CUDA.CUDABackend() : KA.CPU()
catch
    KA.CPU()
end

const NTR = parse(Int, get(ENV, "MC_TRIALS", "400"))

include(joinpath(_HERE, "mc_configs.jl"))   # configs()

# ---------------------------------------------------------------- run
function run_study(; ntrials = NTR)
    rows = []
    for c in configs()
        truth = [(x, y, s) for (x, y, s, _) in c.blobs]
        t = @elapsed res = mc_run(c.img, truth; σn = c.σn, ntrials, seed = 1, device, tilts = c.tilts,
            baselines = true, gate = (2.0, 0.5))
        for (k, r) in enumerate(res)
            length(r.est) < 0.5ntrials && (push!(rows, (cfg = c.name, k, blob = c.blobs[k], n = length(r.est), s = nothing, r)); continue)
            s = mc_summary(r.est, r.cov; truth = truth[k])
            push!(rows, (cfg = c.name, k, blob = c.blobs[k], n = length(r.est), s, r))
        end
        @info "MC config done" c.name t
    end
    return rows
end

# The Hess-3D baseline: one scale for all rows, fitted so the pooled mean e₃ is 3.
function hessian_stats(rows)
    ok = [r for r in rows if r.s !== nothing]
    raw = Float64[]
    for r in ok, (v, Σh) in zip(r.r.est, r.r.hess)
        all(isfinite, Σh) || continue
        μ = sum(r.r.est) / length(r.r.est)
        d = v - μ
        push!(raw, d' * (Symmetric(Σh) \ d))
    end
    c = mean(raw) / 3                     # Σ = c·(∓Ĥ)⁻¹ makes the pooled mean 3
    return c
end

function table(rows, c)
    io = IOBuffer()
    println(io, "| config | blob (x, y, σ, a) | n (unrefined) | mean e₃ | mean e₂ | cov68 / cov95 (3D) | var ratio x / y / logσ | bias x / y / logσ | sd x / y (px) | Hess-3D mean e₃ |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for r in rows
        b = r.blob
        bs = @sprintf("(%.1f, %.1f, %.1f, %+.2f)", b...)
        if r.s === nothing
            println(io, "| $(r.cfg) | $bs | $(r.n) ($(r.r.unrefined[])) | — refined in < 50 % of trials — |||||||")
            continue
        end
        s = r.s
        μ = sum(r.r.est) / length(r.r.est)
        eh = [((v - μ)' * (Symmetric(c .* Σh) \ (v - μ))) for (v, Σh) in zip(r.r.est, r.r.hess) if all(isfinite, Σh)]
        @printf(io, "| %s | %s | %d (%d) | %.2f | %.2f | %.2f / %.2f | %.2f / %.2f / %.2f | %+.3f / %+.3f / %+.4f | %.3f / %.3f | %.2f |\n",
            r.cfg, bs, s.n, r.r.unrefined[], s.mean_e3, s.mean_e2, s.cov68_3, s.cov95_3, s.ratio..., s.bias..., s.sd[1], s.sd[2],
            isempty(eh) ? NaN : mean(eh) * s.n / (s.n - 1))
    end
    return String(take!(io))
end

rows = run_study()
c = hessian_stats(rows)
out = table(rows, c) * "\nHess-3D scale fitted over all rows: Σ = $(round(c, sigdigits = 4))·σn²·J(∓Ĥ)⁻¹Jᵀ (pooled mean e₃ = 3).\nTrials per config: $NTR; device: $(typeof(device)).\n"
path = length(ARGS) >= 1 ? ARGS[1] : joinpath(tempdir(), "keypoint_uncertainty_mc.md")
write(path, out)
println(out)
println("written to ", path)
