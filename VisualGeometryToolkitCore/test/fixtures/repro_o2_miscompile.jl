# =============================================================================
# repro_o2_miscompile.jl — a context-dependent result of an inlined LM at -O2
# =============================================================================
#
# VisualGeometryToolkitCore at plan 1b Task 3, fix round 2. Run in the Core test
# environment:
#
#     julia --project=VisualGeometryToolkitCore/test VisualGeometryToolkitCore/test/fixtures/repro_o2_miscompile.jl
#
# What it shows: the spline lens's projection as it stood at fix round 1 (d54a0637),
# with the bounded two-unknown Levenberg–Marquardt (`_spline_lm`) taking its
# Jacobian by ForwardDiff (`_value_and_jacobian`) of the direction map INLINED into
# the solve, run over a tuple of four cameras (unmasked, masked, and both with a
# forward spline) on the 64 golden rays of the spline-lens fixture. The same code,
# compiled in a fresh module at `@optlevel` 1 and at 2, must image the same rays.
#
# Measured on Julia 1.13.2 (plan 1b Task 3, fix round 2): at -O1 the masked camera
# images 50 of 64 rays (the package's answer, and the batched kernel's); at -O2 it
# images 20, in both the fused and the stored form. The unmasked and forward cameras
# agree (64, 64, 50). In fix round 1 the package form (with `_in_backward` and the
# camera-level `project` inlined around it) also showed the fused/stored split: 42 of
# 64 unmasked rays. Nothing in the code reads out of bounds, truncates a NaN or
# depends on the tag: the review ruled those out, and -O0/-O1 are clean. So it is
# most likely an LLVM -O2 miscompile. (run_tests runs with check_bounds == 0.)
# Also measured (by the controller, a cold run of this file) on Julia 1.13.3 and on
# 1.12.7 (CI's): the same on both — -O1 fused/stored [64, 50, 64, 50], -O2
# [64, 20, 64, 50]. So the AD-Jacobian-inside-the-LM pattern is wrong at -O2 on every
# supported Julia.
#
# The package no longer forms this pattern: every LM Jacobian is analytic (the
# central `_spline_direction_and_jacobian`, the non-central
# `_noncentral_residual_jacobian`). The guard is the test item "SplineLens:
# project's imaged mask is the same at every optimization level". Whether to report
# this upstream is the owner's decision.

using StaticArrays
using LinearAlgebra: norm, normalize
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, px, with_present, Rect2,
    LogicalIntrinsics, CameraCalibrationMat, calibration_matrix
using VisualGeometryToolkitCore.Cameras: _spline_lm, _spline_lm_box, _value_and_jacobian, spline_value, _in_backward,
    _in_fitted_chart, _forward_chart, _normalized_point, _pixel_affine, _spline_camera

# The cameras of the fixture: the unmasked and masked SplineLens cameras, and the two
# with the golden forward spline (in the gauge whose principal point images its axis).
function _repro_cameras()
    include(joinpath(@__DIR__, "spline_lens_golden.jl"))
    G = Base.invokelatest(() -> GOLDEN)
    src = CameraModel(G.f * px, SA[G.pp[1] * px, G.pp[2] * px]; lens = KannalaBrandt(G.kb...))
    cam = CameraModel(SplineLens, src, (G.image[1], G.image[2]); grid_size = (G.grid_size[1], G.grid_size[2]))
    masked = with_present(cam, G.masked.present)
    K = calibration_matrix(cam)
    S = G.spline
    K′ = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0.0, 0.0, 0.0, K[2, 2], 0.0, 499.5, 499.5, 1.0)))
    cam′ = _spline_camera(K′, Rect2(0.0, 0.0, 1000.0, 1000.0), cam.lens.backward.controls)
    C = [SVector((S.controls[1, i, j] - 499.5) / K[1, 1], (S.controls[2, i, j] - 499.5) / K[2, 2]) for i in axes(S.controls, 2), j in axes(S.controls, 3)]
    cam_fwd = CameraModel(K′, SplineLens(cam′.lens.backward; forward = ChartSpline(Rect2(S.area[1], S.area[2], S.area[3], S.area[4]), C)))
    rays = [SVector(S.rays[1, i], S.rays[2, i], S.rays[3, i]) for i in axes(S.rays, 2)]
    return (cam, masked, cam_fwd, with_present(cam_fwd, G.masked.present)), rays
end

# FROZEN: a copy of round 1's code (spline_lens.jl at d54a0637), with the LM's
# ForwardDiff Jacobian INLINED (there it sat behind a `@noinline` barrier). Do not
# update it to the package's current code: it must keep the pattern that miscompiles.
const REPRO_BODY = quote
    @inline function solve(controls, area, t, x₀, lb, ub, s, maxiter, tol2)
        tmax = max(abs(t[1]), abs(t[2]), abs(t[3]))
        (tmax > 0 && isfinite(tmax) && all(isfinite, x₀)) || return SVector(NaN, NaN), false
        tn = normalize(SVector{3, Float64}(t[1] / tmax, t[2] / tmax, t[3] / tmax))
        dir(p) = (S = spline_value(controls, area, p); SVector{3}(S) / norm(S))
        res(p, _) = SVector{3, Float64}(dir(p)) - tn
        jac(p, _) = last(_value_and_jacobian(dir, p))           # ForwardDiff, inlined into the solve
        return _spline_lm(res, jac, x₀, lb, ub, s, maxiter, tol2)
    end
    @inline function project_lens(l, chart, X, x₀, maxiter, tol2)
        nan = SVector(NaN, NaN)
        if l.forward !== nothing
            c = chart(X)
            _in_fitted_chart(l.forward, c) || return nan, false
            y = l.forward(c)
            x = SVector{2, Float64}(y[1], y[2])
            return _in_backward(l, x) ? (x, true) : (nan, false)
        end
        b = l.backward
        lb, ub, s = _spline_lm_box(b.area)
        x, ok = solve(b.controls, b.area, X, x₀, lb, ub, s, maxiter, tol2)
        return ok && _in_backward(l, x) ? (x, true) : (nan, false)
    end
    @inline function project_camera(c, X)
        l = c.lens
        o = l.backward.area.origin; w = l.backward.area.widths
        x = first(project_lens(l, _forward_chart(l), X, SVector{2, Float64}(o[1] + w[1] / 2, o[2] + w[2] / 2), 100, 1.0e-12))
        u = _pixel_affine(c.intrinsics)(x)
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
        Core.eval(M, :(using StaticArrays, LinearAlgebra))
        for name in (:_spline_lm, :_spline_lm_box, :_value_and_jacobian, :spline_value, :_in_backward, :_in_fitted_chart, :_forward_chart, :_pixel_affine)
            Core.eval(M, :(const $name = $(getfield(VisualGeometryToolkitCore.Cameras, name))))
        end
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
