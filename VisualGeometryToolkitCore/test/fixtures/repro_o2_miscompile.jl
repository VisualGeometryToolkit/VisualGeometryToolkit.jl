# =============================================================================
# repro_o2_miscompile.jl — a context-dependent result of an inlined LM at -O2
# =============================================================================
#
# A standalone reproducer, frozen: it reads VisualGeometryToolkitCore only through
# public names (the cameras' construction, `spline_value`, `spline_window`,
# `equidistant_chart`, `calibration_matrix`); every internal it runs is its own
# copy below, so a change to the package's internals cannot make it stop (or
# start) reproducing. Run in the Core test environment:
#
#     julia --project=VisualGeometryToolkitCore/test VisualGeometryToolkitCore/test/fixtures/repro_o2_miscompile.jl
#
# What it shows: the spline lens's projection as it stood at plan 1b Task 3, fix
# round 1 (d54a0637), with the bounded two-unknown Levenberg–Marquardt (the
# package's LM as it stood at 02dfb2ed: its residual and Jacobian two functions)
# taking its Jacobian by ForwardDiff of the direction map INLINED into the solve,
# run over a tuple of four cameras (unmasked, masked, and both with a forward
# spline) on the 64 golden rays of the spline-lens fixture. The same code,
# compiled in a fresh module at `@optlevel` 1 and at 2, must image the same rays.
#
# Measured on Julia 1.13.2 (plan 1b Task 3, fix round 2): at -O1 the masked camera
# images 50 of 64 rays (the package's answer, and the batched kernel's); at -O2 it
# images 20, in both the fused and the stored form. The unmasked and forward cameras
# agree (64, 64, 50). Nothing in the code reads out of bounds, truncates a NaN or
# depends on the tag: the review ruled those out, and -O0/-O1 are clean. So it is
# most likely an LLVM -O2 miscompile. (run_tests runs with check_bounds == 0.)
# Also measured on Julia 1.13.3 and 1.12.7 (CI's): -O1 fused/stored
# [64, 50, 64, 50], -O2 [64, 20, 64, 50]. This self-contained copy (plan 1b Task 4b,
# fix round 1: the internals compiled in the fresh module at its level too) shows the
# fault on the unmasked camera instead, on Julia 1.13.2 and 1.13.3 alike: -O1
# [64, 50, 64, 50], -O2 [22, 50, 64, 50].
#
# The package no longer forms this pattern: every LM Jacobian is analytic (the
# central `_spline_direction_differential`, the non-central
# `_noncentral_residual_and_jacobian`), and the package's LM reads its residual and
# Jacobian from one evaluation per point — through which even an inlined ForwardDiff
# Jacobian no longer shows the fault (Julia 1.13.3: -O1 = -O2). So no test of the
# package's own code can be shown red against it; the protection is the owner's
# rule, no automatic differentiation inside an LM. Whether to report this upstream
# is the owner's decision.

using StaticArrays
using LinearAlgebra: norm, normalize
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, px, with_present, Rect2,
    LogicalIntrinsics, CameraCalibrationMat, calibration_matrix, equidistant_chart

# The cameras of the fixture: the unmasked and masked SplineLens cameras, and the two
# with the golden forward spline (in the gauge whose principal point images its axis),
# each with its forward chart and its K.
function _repro_cameras()
    include(joinpath(@__DIR__, "spline_lens_golden.jl"))
    G = Base.invokelatest(() -> GOLDEN)
    src = CameraModel(G.f * px, SA[G.pp[1] * px, G.pp[2] * px]; lens = KannalaBrandt(G.kb...))
    cam = CameraModel(SplineLens, src, (G.image[1], G.image[2]); grid_size = (G.grid_size[1], G.grid_size[2]))
    masked = with_present(cam, G.masked.present)
    K = calibration_matrix(cam)
    S = G.spline
    Kp = SMatrix{3, 3, Float64}(K[1, 1], 0.0, 0.0, 0.0, K[2, 2], 0.0, 499.5, 499.5, 1.0)
    Ki = inv(Kp)
    lo = Ki * SVector(0.0, 0.0, 1.0); hi = Ki * SVector(1000.0, 1000.0, 1.0)
    backward = ChartSpline(Rect2(lo[1], lo[2], hi[1] - lo[1], hi[2] - lo[2]), cam.lens.backward.controls)
    C = [SVector((S.controls[1, i, j] - 499.5) / K[1, 1], (S.controls[2, i, j] - 499.5) / K[2, 2]) for i in axes(S.controls, 2), j in axes(S.controls, 3)]
    cam_fwd = CameraModel(LogicalIntrinsics(CameraCalibrationMat(Kp)), SplineLens(backward; forward = ChartSpline(Rect2(S.area[1], S.area[2], S.area[3], S.area[4]), C)))
    rays = [SVector(S.rays[1, i], S.rays[2, i], S.rays[3, i]) for i in axes(S.rays, 2)]
    chart(c) = (d = c.lens.backward(SVector(0.0, 0.0)); equidistant_chart(SVector{3, Float64}(d / norm(d))))
    cams = (cam, masked, cam_fwd, with_present(cam_fwd, G.masked.present))
    return map(c -> (c, chart(c), SMatrix{3, 3, Float64}(calibration_matrix(c))), cams), rays
end

# FROZEN: a copy of round 1's code (spline_lens.jl at d54a0637) with the LM's
# ForwardDiff Jacobian INLINED (there it sat behind a `@noinline` barrier), and of the
# internals it ran, as at 02dfb2ed. Do not update it to the package's current code:
# it must keep the pattern that miscompiles.
const REPRO_BODY = quote
    const LM_AREA_UNITS = 1000
    @inline function lm_box(area)
        o = area.origin; w = area.widths
        lb = SVector{2, Float64}(o[1], o[2])
        ws = SVector{2, Float64}(w[1], w[2])
        return lb, lb + ws .* (1 - 0.001 / LM_AREA_UNITS), LM_AREA_UNITS ./ ws
    end
    @inline function lm_increment(JtJ, λ, DtD, g)
        x1 = g[1]
        x2 = g[2]
        A11 = muladd(λ, DtD[1], JtJ[1, 1])
        A11 >= zero(A11) || return zero(SVector{2, Float64}), 1
        A21 = JtJ[2, 1]
        A22 = muladd(λ, DtD[2], JtJ[2, 2])
        A22 >= zero(A22) || return zero(SVector{2, Float64}), 2
        A22 = muladd(-A21, A21 / A11, A22)
        x2 = muladd(-x1, A21 / A11, x2)
        x2 /= A22
        x1 = muladd(-x2, A21, x1)
        x1 /= A11
        return SVector(x1, x2), 0
    end
    @inline function value_and_jacobian(f::F, x::SVector{N, T}) where {F, N, T}
        tag = typeof(ForwardDiff.Tag(f, T))
        duals = SVector{N, ForwardDiff.Dual{tag, T, N}}(
            ntuple(i -> ForwardDiff.Dual{tag}(x[i], ForwardDiff.Partials{N, T}(ntuple(j -> T(i == j), Val(N)))), Val(N))
        )
        y = f(duals)
        M = length(y)
        value = SVector{M, T}(ntuple(i -> ForwardDiff.value(tag, y[i]), Val(M)))
        J = SMatrix{M, N, T}(ntuple(k -> ForwardDiff.partials(tag, y[(k - 1) % M + 1], (k - 1) ÷ M + 1), Val(M * N)))
        return value, J
    end
    @inline lattice_coord(x, lo, span, gdim) = (T = float(typeof(x)); 1 + (T(gdim - 3) / T(span)) * (x - T(lo)))
    @inline window_present(::Nothing, jx, jy) = true
    @inline function window_present(present::AbstractMatrix{Bool}, jx, jy)
        for c in -2:1, r in -2:1
            present[jy + r, jx + c] || return false
        end
        return true
    end
    @inline function in_backward(l, x)
        cs = l.backward
        gsz = gh, gw = size(cs.controls)
        all(isfinite, x) || return false
        o = cs.area.origin; w = cs.area.widths
        gx, gy = lattice_coord(x[1], o[1], w[1], gw), lattice_coord(x[2], o[2], w[2], gh)
        sx, sy = (gw - 3) / w[1], (gh - 3) / w[2]
        ε = max(eps(float(typeof(x[1]))), eps(eltype(eltype(cs.controls))))
        τx = 4 * ε * sx * (abs(x[1]) + abs(o[1]))
        τy = 4 * ε * sy * (abs(x[2]) + abs(o[2]))
        (1 - τx <= gx < gw - 2 && 1 - τy <= gy < gh - 2) || return false
        ix, iy, _, _ = spline_window(cs.area, gsz, x)
        return window_present(l.present, ix, iy)
    end
    function in_fitted_chart(cs, c)
        o = cs.area.origin; w = cs.area.widths
        gh, gw = size(cs.controls)
        hx, hy = w[1] / (gw - 3), w[2] / (gh - 3)
        return o[1] - hx <= c[1] <= o[1] + w[1] + hx && o[2] - hy <= c[2] <= o[2] + w[2] + hy
    end
    @inline function lm(
            res::R, jac::J, x₀::SVector{2, Float64}, lb::SVector{2, Float64}, ub::SVector{2, Float64},
            s::SVector{2, Float64}, maxiter::Int, tol2::Float64
        ) where {R, J}
        ϵ = eps(Float64)
        ϵx = 1.0e-10 ./ s
        λ = 1.0e-6
        x = min.(max.(x₀, lb), ub)
        f = res(x, nothing)
        F = sum(abs2, f)
        F < ϵ && return x, F < tol2
        Jx = jac(x, nothing)
        g = Jx' * f
        norm(g, Inf) < ϵ && return x, F < tol2
        JtJ = Jx' * Jx
        DtD = clamp.(diag(JtJ), 1.0e-12 .* s .^ 2, Inf)
        DtD_max = 1.0e16 .* s .^ 2
        iter = 0
        while iter < maxiter
            iter += 1
            accepted = false
            ac = 0.0
            pr = 0.0
            ρ = 0.0
            xk = x
            fk = f
            Fk = F
            p, info = lm_increment(JtJ, λ, DtD, g)
            # a step clipped to zero by either bound is rejected, as a failed solve is
            clipped = false
            if info == 0
                p = min.(x .- lb, p)
                clipped = all(iszero, p)
                if !clipped
                    p = max.(x .- ub, p)
                    clipped = all(iszero, p)
                end
            end
            if info == 0 && !clipped
                xk = x .- p
                fk = res(xk, nothing)
                Fk = sum(abs2, fk)
                if Fk < F
                    ac = F - Fk
                    pr = dot(p, (@. λ * DtD * p + g))
                    ρ = ac / pr
                    accepted = true
                end
                if all(>(0), (@. abs(p) < ϵx * (ϵx + abs(x))))
                    if accepted
                        x, f, F = xk, fk, Fk
                    end
                    break
                end
            end
            if accepted
                x = xk
                f = fk
                Fold = F
                F = Fk
                # the reference routine's ftol test reads its swapped buffers: `Fk` is then the old F
                abs(ac) < ϵ * Fold && pr < ϵ * Fold && ρ < 2 && break
                Jx = jac(x, nothing)
                g = Jx' * f
                norm(g, Inf) < ϵ && break
                JtJ = Jx' * Jx
                DtD = clamp.(diag(JtJ), DtD, DtD_max)
                if ρ > 0.75
                    λ *= 1 / 13
                elseif ρ < 1.0e-3
                    λ *= 3
                end
            else
                λ *= 3
            end
            λ = clamp(λ, 1.0e-28, 1.0e32)
        end
        return x, F < tol2
    end
    @inline function solve(controls, area, t, x₀, lb, ub, s, maxiter, tol2)
        tmax = max(abs(t[1]), abs(t[2]), abs(t[3]))
        (tmax > 0 && isfinite(tmax) && all(isfinite, x₀)) || return SVector(NaN, NaN), false
        tn = normalize(SVector{3, Float64}(t[1] / tmax, t[2] / tmax, t[3] / tmax))
        dir(p) = (S = spline_value(controls, area, p); SVector{3}(S) / norm(S))
        res(p, _) = SVector{3, Float64}(dir(p)) - tn
        jac(p, _) = last(value_and_jacobian(dir, p))           # ForwardDiff, inlined into the solve
        return lm(res, jac, x₀, lb, ub, s, maxiter, tol2)
    end
    @inline function project_lens(l, chart, X, x₀, maxiter, tol2)
        nan = SVector(NaN, NaN)
        if l.forward !== nothing
            c = chart(X)
            in_fitted_chart(l.forward, c) || return nan, false
            y = l.forward(c)
            x = SVector{2, Float64}(y[1], y[2])
            return in_backward(l, x) ? (x, true) : (nan, false)
        end
        b = l.backward
        lb, ub, s = lm_box(b.area)
        x, ok = solve(b.controls, b.area, X, x₀, lb, ub, s, maxiter, tol2)
        return ok && in_backward(l, x) ? (x, true) : (nan, false)
    end
    @inline function project_camera((c, chart, K), X)
        l = c.lens
        o = l.backward.area.origin; w = l.backward.area.widths
        x = first(project_lens(l, chart, X, SVector{2, Float64}(o[1] + w[1] / 2, o[2] + w[2] / 2), 100, 1.0e-12))
        u = K * SVector(x[1], x[2], 1.0)
        return SVector{2, Float64}(u[1], u[2])
    end
    fused(cams, rays) = [[all(isfinite, project_camera(c, X)) for X in rays] for c in cams]
    stored(cams, rays) = [[all(isfinite.(project_camera(c, X))) for X in rays] for c in cams]
end

function repro_o2_miscompile()
    cams, rays = _repro_cameras()
    imaged = Dict{Int, Any}()
    for level in (1, 2)
        M = Module()
        Core.eval(M, :(using StaticArrays, LinearAlgebra, ForwardDiff))
        Core.eval(M, :(using VisualGeometryToolkitCore: spline_value, spline_window))
        Core.eval(M, :(Base.Experimental.@optlevel $level))
        Core.eval(M, REPRO_BODY)
        imaged[level] = (fused = Base.invokelatest(M.fused, cams, rays), stored = Base.invokelatest(M.stored, cams, rays))
    end
    # the same code and data: every count must be the same at both levels
    return (; O1 = map(v -> count.(v), imaged[1]), O2 = map(v -> count.(v), imaged[2]),
        same = imaged[1].fused == imaged[2].fused && imaged[1].stored == imaged[2].stored && imaged[1].fused == imaged[1].stored)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println((VERSION, repro_o2_miscompile()))
end
