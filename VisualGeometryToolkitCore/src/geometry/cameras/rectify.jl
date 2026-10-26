# ==============================================================================
# rectify — stereo rectification on the rig types
# ==============================================================================
#
# Fusiello-style epipolar rectification: one common rectified orientation for
# both views — new x-axis along the baseline, new z as close as possible to
# both optical axes — so corresponding points land on the SAME ROW and the
# rectified relative pose is a pure x-translation. The per-view resampling is
# one fused warp each (`distortion_map ∘ division ∘ homography`), so any lens
# model the imaging stack knows rectifies in the same warp that undistorts it.
#
# The output speaks the package's currencies: the rectified pair is a
# `StereoRig` of pinhole cameras whose relative transform IS the baseline
# (`extrinsics(rig_rect).t = (±B, 0, 0)`), and the two canvases are
# `FramedImage`s over ONE shared rectified pixel frame — `imshow` places
# them in directly comparable axes, disparity is a horizontal ruler.

"""
    rectify(rig::StereoRig; K = nothing) -> (; rig, R, K)
    rectify([T = Float64], rig::StereoRig, images::Tuple; K = nothing, out_size = nothing, scale = 1)
        -> (; rig, R, K, images::NTuple{2, FramedImage})

Epipolar rectification of a stereo rig.

Geometry form: computes the rectifying rotations `R = (Q₁, Q₂)`
(`RotMatrix3`s, original view-i orientation → the common rectified
orientation, `Q₂ = Q₁·R_relᵀ`), the shared pinhole calibration `K`
(`SMatrix{3,3}`; default: mean focal of the two views, mean principal
point, zero skew), and the RECTIFIED rig — a [`StereoRig`](@ref) of two
pinhole cameras with the same ids whose relative transform is the pure
baseline translation. Rectified frames are NEW frames; the returned rig's
`ViewFrame{id}`s label those, not the input views.

Image form: additionally resamples the two source images into the shared
rectified pixel frame — for each output pixel, `K⁻¹` → rotate the ray by
`Qᵢᵀ` → project through view i's OWN lens model → sample: distortion and
rectification in ONE fused warp per view, any `AbstractLensModel`. Returns
two [`FramedImage`](@ref)s over the same window `Rect(0, 0, W, H)` of the
rectified pixel frame (`out_size = (W, H)`, default: the elementwise max of the
input sizes; `scale` downsamples like the rect form of
[`undistort`](@ref)). After rectification, corresponding points share a
row and `disparity = fx·B/z`.

KernelAbstractions end to end: the destination is allocated with `similar`,
so it lives on the SOURCE images' backend — supply `CuArray`s and the fused
warp (one bilinear kernel; the isbits chain evaluated per output pixel on
the device, no remap grid) runs on the GPU. `rectify(Float32, rig, images)`
keeps the warp math single-precision for GPU throughput (the type leads, as in
Base's `zeros(Float32, n)`).

The construction assumes a predominantly HORIZONTAL baseline (the new
x-axis is sign-fixed toward +x); a vertical rig wants the transposed
convention. Throws for a near-zero baseline.
"""
function rectify(rig::StereoRig; K = nothing)
    # a physical rig rectifies in metres; the rectified rig is read back in metres
    if has_units(rig)
        r = rectify(ustrip(m, rig); K)
        return (; rig = with_unit(r.rig, m), R = r.R, K = r.K)
    end
    X = extrinsics(rig)                     # x₂ = R_rel x₁ + t (view1 → view2)
    R_rel = SMatrix{3, 3, Float64, 9}(RotMatrix(X.R))
    t = SVector{3, Float64}(X.t)
    B = norm(t)
    B > 0 || throw(ArgumentError("rectify: zero baseline — the two views share a center"))

    b1 = -R_rel' * t                        # cam2 center in view-1 coordinates
    e1 = b1 / B
    e1 = e1[1] < 0 ? -e1 : e1               # sign-fix toward +x (horizontal rigs)
    z_avg = SVector(0.0, 0.0, 1.0) + R_rel' * SVector(0.0, 0.0, 1.0)
    e2 = LinearAlgebra.normalize(LinearAlgebra.cross(z_avg, e1))
    e3 = LinearAlgebra.cross(e1, e2)
    Q1 = RotMatrix{3, Float64}(vcat(e1', e2', e3'))
    Q2 = RotMatrix{3, Float64}(SMatrix{3, 3, Float64, 9}(Q1) * R_rel')

    Kr = K === nothing ?
        begin
            # The default shared pinhole averages the two calibration
            # matrices — which a grid-model sensor does not have: name the
            # rectified K yourself there (same contract as generic undistort).
            (model(source(rig)) isa CameraModel && model(destination(rig)) isa CameraModel) ||
            throw(
                ArgumentError(
                    "rectify: a grid-model sensor has no calibration matrix to " *
                    "derive the shared rectified K from — supply K = ... explicitly"
                )
            )
            K1 = SMatrix{3, 3, Float64, 9}(calibration_matrix(source(rig)))
            K2 = SMatrix{3, 3, Float64, 9}(calibration_matrix(destination(rig)))
            f = (K1[1, 1] + K1[2, 2] + K2[1, 1] + K2[2, 2]) / 4
            cx = (K1[1, 3] + K2[1, 3]) / 2
            cy = (K1[2, 3] + K2[2, 3]) / 2
            _pixel_K(f, f, (cx, cy))
        end : SMatrix{3, 3, Float64, 9}(K)

    # Rectified relative pose: x₂ᵣ = x₁ᵣ + Q₂t, and Q₂t = -Q₁b₁ = ∓B·e₁ᵣ —
    # a pure x-translation by construction.
    t_r = SMatrix{3, 3, Float64, 9}(Q2) * t
    model_r = CameraModel(LogicalIntrinsics(CameraCalibrationMat(Kr)), Pinhole())
    cam1r = PosedCamera(model_r, EuclideanMap(one(RotMatrix{3, Float64}), zero(Vec3{Float64})))
    cam2r = PosedCamera(model_r, EuclideanMap(one(RotMatrix{3, Float64}), Vec3{Float64}(t_r)))
    a, b = ids(rig)
    return (; rig = StereoRig(cam1r, cam2r; ids = (a, b)), R = (Q1, Q2), K = Kr)
end

# Per-view warp math: rect pixel → the view-i RAY (Qᵢᵀ·K_r⁻¹, no division) →
# view i's own lens → K_i → source pixel. The same chain for every lens, a
# pinhole included: a ray the rectifying rotation turns past a lens's field is
# NaN (fill), never a mirrored source pixel. Built as explicit
# ComposedTransformations: `∘` on a homography whose type inference cannot pin
# would admit Base's function composition, which `image_warp` does not take.
function _rectify_math(
        model::CameraModel, Ki::SMatrix{3, 3, Float64, 9},
        Qi::RotMatrix{3, Float64}, Kr_inv::SMatrix{3, 3, Float64, 9}
    )
    ray = HomographyMap(SMatrix{3, 3, Float64, 9}(Qi)' * Kr_inv)
    return CoordinateTransformations.ComposedTransformation(
        AffineMap(Ki), CoordinateTransformations.ComposedTransformation(model.lens, ray)
    )
end

# One rectified view through a PARAMETRIC camera: the fused isbits warp.
# The destination FOLLOWS the source's backend (similar, not zeros): supply
# CuArrays and the homography∘lens∘K math runs per output pixel ON
# the device — one KA bilinear kernel, no remap grid. `precision` (the leading
# type of `rectify`) keeps the per-pixel math single-precision for GPU throughput. The grid-
# model method lives in generic_models.jl (host coarse-inverse remap).
function _rectified_view(
        mi::CameraModel, img::AbstractMatrix, Ki::SMatrix{3, 3, Float64, 9},
        Qi, Kr_inv::SMatrix{3, 3, Float64, 9}, window, Wout::Int, Hout::Int, precision
    )
    wd = GeometryBasics.widths(window)
    place = AffineMap(SMatrix{2, 2, Float64}(wd[1] / Wout, 0, 0, wd[2] / Hout), SVector(0.0, 0.0))
    math = _rectify_math(mi, Ki, Qi, Kr_inv) ∘ place
    bm = image_warp(precision, math, Rect(0, 0, Wout, Hout))
    dst = fill!(similar(img, Hout, Wout), zero(eltype(img)))
    return FramedImage(image_warp!(dst, bm, img), window)
end

rectify(rig::StereoRig, images::Tuple{AbstractMatrix, AbstractMatrix}; kwargs...) =
    rectify(Float64, rig, images; kwargs...)

function rectify(
        precision::Type{<:AbstractFloat}, rig::StereoRig,
        images::Tuple{AbstractMatrix, AbstractMatrix};
        K = nothing, out_size = nothing, scale::Real = 1
    )
    geo = rectify(rig; K)
    Wr, Hr = out_size === nothing ?
        (max(size(images[1], 2), size(images[2], 2)), max(size(images[1], 1), size(images[2], 1))) :
        (Int(out_size[1]), Int(out_size[2]))
    Wout, Hout = ceil(Int, Wr / scale), ceil(Int, Hr / scale)
    Kr_inv = inv(geo.K)
    window = Rect(0.0, 0.0, Float64(Wr), Float64(Hr))
    cis = ntuple(2) do i
        sensor = i == 1 ? source(rig) : destination(rig)
        mi = model(sensor)
        Ki = mi isa CameraModel ? SMatrix{3, 3, Float64, 9}(calibration_matrix(sensor)) : nothing
        _rectified_view(mi, images[i], Ki, geo.R[i], Kr_inv, window, Wout, Hout, precision)
    end
    return (; geo.rig, geo.R, geo.K, images = cis)
end
