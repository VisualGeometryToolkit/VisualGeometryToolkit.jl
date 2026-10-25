# =============================================================================
# fit_radial_camera — the radial camera of one board, an iterative whitened fit
# (moved from VisualGeometryToolkitCore, estimator taxonomy v2: fit/find/refine live
# in VisualGeometryToolkitEstimation; the minimal 5-point solve stays there as
# `RadialCamera5pt`)
# =============================================================================

"""
    fit_radial_camera(cs::AbstractVector{<:Cspond{<:StaticVector{2}, <:StaticVector{2}}};
                      center, whitening = SampsonWhitening(), iterations = 2)
        -> FitResult{RadialCamera{Float64}, Nothing}

The radial camera of one board from its `plane_point => image_point`
correspondences `cs`, the image points RAW as measured; `center` (pixels,
REQUIRED — the principal point, as `find_radial_camera` and
`RadialCameraProblem` require it) is the pencil's centre and rides on the
returned camera, so nothing is pre-centred. There is no default: a pencil
about the image corner fits raw pixels as well as any other, with no error.
Each pair gives one linear constraint on the six entries of `P`
([`constraint_with_jacobians`](@ref)); the null vector is solved by SVD in the
radial camera's normalization, `CspondNormalization(RadialCamera, cs; center)`
(Hartley at the plane centroid, Hartley about `center` on the image side — the
same strategy `find_radial_camera`'s problem solves in), and composed back into
pixels. Each iteration re-solves; from the second on, every row is
whitened under `whitening` at the previous iteration's solution (`SampsonWhitening`
turns the algebraic error into the exact point–radial-line distance); a
`weights` column on `cs` (a `Csponds` attribute, e.g. `withattrs(cs; weights)`)
multiplies on top, the row weight being √w. This is how the BabelCalib back-end
obtains the radial camera when no homography is available; when one is, its
rows 1–2 are the same object (`RadialCamera(H)`). Five correspondences
determine `P` (five degrees of freedom).

The `FitResult`'s estimate is the camera, `retcode` `:solved` after `iterations`
iterations (its `iterations` field), and `rms` the point–radial-line distance (px) of the `n` pairs at the
fit, whatever the whitening policy. When the constraint rows leave more than
one null direction (a degenerate configuration — the plane points on a line,
or a pixel on the centre) the retcode is `:degenerate`, the camera NaN and the
rms NaN.

Units: the source points are plain numbers. A radial camera maps from the source
plane, so its entries carry that plane's unit, which a matrix cannot record:
physical (unitful) source points are an `ArgumentError` — strip them in the
unit the rest of your plain data uses — metres, as the stack's boundaries strip
(`ustrip.(m, X)`).
"""
function fit_radial_camera(
        cs::AbstractVector{<:Cspond{<:StaticVector{2}, <:StaticVector{2}}}; center,
        whitening::WhiteningPolicy = SampsonWhitening(), iterations::Int = 2
    )
    _refuse_physical_plane(cs, "fit_radial_camera")
    iterations >= 1 || throw(ArgumentError("fit_radial_camera: iterations must be ≥ 1 (0 would return the unfitted seed), got $iterations"))
    # The problem's one construction (canonical data, the radial camera's
    # normalization, the normalized view — where the pencil's centre is the
    # origin) and its candidate and residual; it refuses fewer than 5 pairs.
    p = RadialCameraProblem(cs; center)
    n = data_size(p)
    weights = VgtkCore.check_weights(VgtkCore.weights_column(cs))   # the `weights` column, or nothing
    Xn = sources(p.cs_solve); xn = targets(p.cs_solve)
    o = Point2{Float64}(0.0, 0.0)
    A = Matrix{Float64}(undef, n, 6)
    svd_ws = VgtkCore.SvdWorkspace{Float64}(n, 6)
    P = RadialCamera{Float64}(SMatrix{2, 3, Float64, 6}(1, 0, 0, 1, 0, 0), o)
    degenerate(iter) = FitResult(RadialCamera{Float64}(SMatrix{2, 3, Float64, 6}(ntuple(_ -> NaN, 6)), p.context), :degenerate, iter, NaN, n, nothing)
    for iter in 1:iterations
        for i in 1:n
            g, G_x, G_θ = constraint_with_jacobians(P, Xn[i], xn[i])
            # iteration 1 has no solution to whiten at; from iteration 2 the policy applies
            _, row = whitened_rows(iter == 1 ? NoWhitening() : whitening, g, G_x, G_θ)
            weights === nothing || (row = row * sqrt(Float64(weights[i])))
            A[i, :] .= row
        end
        # The null vector and the null-space gap, VGC's (`svd_nullvec!` thin
        # once there are six rows, full for the minimal five): more than one
        # null direction is a degenerate configuration.
        v = VgtkCore.svd_nullvec!(svd_ws, A, n, Val(6))
        VgtkCore.test_dlt_conditioning(svd_ws; max_cond = 1.0e10) || return degenerate(iter)
        P = RadialCamera{Float64}(SMatrix{2, 3, Float64, 6}(v), o)
    end
    # into pixels by composition with the problem's normalization, the centre
    # the exact datum (the minimal solve's `_candidate`)
    P = _candidate(p, P)
    # the reported residual is the point–radial-line distance in px (the
    # problem's scoring residual), one currency whatever policy weighted the
    # iterations
    ss = sum(abs2, residuals!(Vector{Float64}(undef, n), p, P))
    return FitResult(P, :solved, iterations, sqrt(ss / n), n, nothing)
end

