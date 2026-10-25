# ==============================================================================
# StereoRig - Stereo camera system with epipolar geometry
# ==============================================================================

"""
    StereoRig{F, U, V} = Rig{F, Tuple{U, V}}

Stereo camera pair — the 2-sensor specialization of [`Rig`](@ref): a stereo
rig IS a rig whose camera container is a 2-tuple, so the pair's length is in
the type and the stereo surface dispatches on it with no runtime check.

Since 0.16 there are NO `source`/`target`/`extrinsics` FIELDS: the rig
stores its two cameras rig-relative (reference frame `F = ViewFrame{id₁}`;
camera 1's stored extrinsics is the identity, camera 2's is the labeled
relative transform `ViewFrame{id₁} → ViewFrame{id₂}`), and the old fields
are method-form accessors:

- [`source`](@ref)`(rig)` / [`target`](@ref)`(rig)`: the two cameras
  (rig-relative — the constructors re-express world-posed inputs).
- `extrinsics(rig)`: the relative rigid transform (monogram `X_{V₂V₁}`),
  DERIVED from the stored per-camera maps as
  `extrinsics(rig, 2) ∘ pose(rig, 1)` — it consumes source-view
  (`ViewFrame{id₁}`) coordinates and produces target-view
  (`ViewFrame{id₂}`) coordinates, i.e. `X₂ = R·X₁ + t`.
- `pose(rig)`: `inv(extrinsics(rig))` (the typed algebra swaps the frames,
  `X_{V₁V₂}`).
- [`ids`](@ref)`(rig)`: the identity symbols, read off the stored labels.

The labels ARE the single source of truth — the twin-typing and
id-duplication policy violations of 0.6–0.8 remain unrepresentable, and the
0.9-era separately-stored relative transform is gone too (one authority:
the per-camera rig-relative maps).

# Constructors
```julia
StereoRig(left_camera, right_camera)   # Computes relative transform
StereoRig(source, target, extrinsics)  # Explicit transform, CHECKED against
                                       # the cameras (check = false to skip)
StereoRig(left, right; ids = (:left, :right))  # Mint per-view identities
```

The `ids` keyword mints the view identities at construction (label-at-
source: the constructor re-expresses the cameras in the rig frame
`ViewFrame{id₁}` and labels the relative transform
`ViewFrame{id₁} → ViewFrame{id₂}`, overriding whatever frames the inputs
carried — the rig's ids are authoritative for its own transforms). The
labeled per-view tags are exposed by [`view_frames`](@ref) (3D
`ViewFrame{id}`) and [`image_frames`](@ref) (2D `ImageFrame{id}`);
[`essential_form`](@ref) / [`EpipolarForm`](@ref)`(rig)` mint the labeled
epipolar forms. Purely additive: nothing in the rig forces identities onto
callers who ignore them.

The 3-arg constructor validates the given transform against
`extrinsics(target) ∘ pose(source)` computed from the GIVEN cameras
(`check = true` by default, `atol` tolerance on the rotation angle and the
relative translation deviation). Supply `check = false` for synthetic rigs
whose cameras are placeholders and do not encode the relative geometry.
`check = false` also admits UNPOSED members (a [`BoundedCamera`](@ref) with no
pose): the rig poses them, so only the given relative transform matters. The
2-argument form computes the transform from the cameras and so requires them posed.
"""
const StereoRig{F <: Frame{3}, U <: AbstractCamera, V <: AbstractCamera} =
    Rig{F, Tuple{U, V}}

# Consistency gate for the 3-arg constructor (mirrors the EuclideanMap
# check/atol idiom): the caller-supplied relative transform must agree with
# the one implied by the GIVEN cameras, `extrinsics(target) ∘ pose(source)`.
# Rotation deviation is the relative rotation angle (radians); translation
# deviation is relative, unit-free: the deviation over the larger of the
# expected baseline and the two camera centres' distances from the world
# origin. The centres floor a near-zero baseline at the scale the composition
# `extrinsics(target) ∘ pose(source)` rounds at, in the data's own unit (a
# plain map is in the caller's unit, so no fixed floor can be right for it —
# frames.md rule 4).
function _assert_rig_consistency(
        source, target, given::EuclideanMap{3};
        atol::Real   # StereoRig's `atol` is the one authority
    )
    expected = extrinsics(target) ∘ pose(source)
    # a physical relative pose against plain cameras (or the reverse) never compares
    has_units(given) == has_units(expected) || throw(DimensionError(translation(given), translation(expected)))
    dev_rot = Rotations.rotation_angle(RotMatrix{3}(SMatrix{3, 3, Float64, 9}(given.R)' * SMatrix{3, 3, Float64, 9}(expected.R)))
    t_given, t_expected = _translation_metres(given), _translation_metres(expected)
    scale = max(norm(t_expected), norm(_translation_metres(pose(source))), norm(_translation_metres(pose(target))))
    Δ = norm(t_given - t_expected)
    dev_t = Δ == 0 ? 0.0 : Δ / scale
    (dev_rot ≤ atol && dev_t ≤ atol) && return nothing
    throw(
        ArgumentError(
            "StereoRig: the given relative transform disagrees with " *
                "extrinsics(target) ∘ pose(source) from the given cameras: " *
                "rotation deviation = $dev_rot rad, relative translation " *
                "deviation = $dev_t (tolerance = $atol). Supply `check=false` " *
                "for synthetic rigs whose cameras are placeholders."
        )
    )
end

function StereoRig(
        source::AbstractCamera, target::AbstractCamera, extrinsics::EuclideanMap{3};
        ids::Tuple{Symbol, Symbol} = (:cam1, :cam2),
        check::Bool = true,
        atol::Real = 1.0e-6
    )
    # the check reads the given cameras' poses; unchecked, only their models are kept
    check && _check_posed("StereoRig", (source, target))
    check && _assert_rig_consistency(source, target, extrinsics; atol = atol)
    # A physical relative pose: the rig is built from the plain one in metres, then
    # read back in metres — the same numbers as the plain construction. (The
    # cameras give only their models here; their poses were the check's.)
    has_units(extrinsics) &&
        return with_unit(StereoRig(source, target, ustrip(m, extrinsics); ids, check = false), m)
    X = relabel(ViewFrame{ids[1]}, ViewFrame{ids[2]}, extrinsics)
    F = ViewFrame{ids[1]}
    T = eltype(X.t)
    ident = relabel(F, F, EuclideanMap(one(RotMatrix{3, T}), zero(Vec3{T})))
    return Rig{F}((_with_extrinsics(source, ident), _with_extrinsics(target, X)))
end

# 2-arg form takes either PosedCamera or BoundedCamera (or any posed AbstractCamera).
# Uses method-form accessors (extrinsics, pose) so MacroTools.@forward on
# BoundedCamera transparently delegates to the inner camera. The relative
# transform is computed FROM the cameras here, so the consistency check
# would compare it against itself — skip it.
function StereoRig(source::AbstractCamera, target::AbstractCamera; ids::Tuple{Symbol, Symbol} = (:cam1, :cam2))
    _check_posed("StereoRig", (source, target))
    # physical cameras compose in metres, plain; a physical camera beside a plain one is refused
    (has_units(source) || has_units(target)) &&
        return with_unit(StereoRig(ustrip(m, source), ustrip(m, target); ids), m)
    return StereoRig(source, target, extrinsics(target) ∘ pose(source); ids, check = false)
end

"""
    source(rig::StereoRig) -> AbstractCamera

The source (left, view 1) camera — stored rig-relative: its extrinsics is
the identity map on the rig frame `ViewFrame{id₁}`.
"""
source(rig::StereoRig) = rig.members[1]

"""
    target(rig::StereoRig) -> AbstractCamera

The target (right, view 2) camera — stored rig-relative: its extrinsics is
the labeled relative transform `ViewFrame{id₁} → ViewFrame{id₂}`.
"""
target(rig::StereoRig) = rig.members[2]

"""
    extrinsics(rig::StereoRig) -> EuclideanMap

The source→target relative rigid transform (monogram `X_{V₂V₁}`), DERIVED
on demand from the stored per-camera rig-relative maps as
`extrinsics(rig, 2) ∘ pose(rig, 1)` — labeled
`ViewFrame{id₁} → ViewFrame{id₂}`, i.e. `X₂ = R·X₁ + t`. With the stereo
convention (reference = view 1, camera 1's stored map the identity) this is
numerically exactly camera 2's stored extrinsics.
"""
extrinsics(rig::StereoRig) = extrinsics(rig, 2) ∘ pose(rig, 1)

"""
    pose(rig::StereoRig) -> EuclideanMap

The target-to-source relative transform, `inv(extrinsics(rig))` computed on
demand (monogram `X_{V₁V₂}`; the typed algebra swaps the frame labels to
`ViewFrame{id₂} → ViewFrame{id₁}`).
"""
pose(rig::StereoRig) = inv(extrinsics(rig))

# =============================================================================
# Labeled epipolar-form mints — rig → EpipolarForm in one call
# =============================================================================

"""
    essential_form(rig::StereoRig)
        -> EpipolarForm{GnomonicFrame{id₁}, GnomonicFrame{id₂}, EssentialMat}

The rig's CALIBRATED epipolar form: `E = [t]×R` from the labeled relative
transform (`x̂₂ᵀ E x̂₁ = 0` for gnomonic coordinates `x̂ = π(K⁻¹ũ)`), with the
slots labeled by the per-view [`GnomonicFrame`](@ref) tags — NOT
[`ImageFrame`](@ref), which is documented as the canonical-`:colmap` PIXEL
frame and is never overloaded for gnomonic coordinates. Slot 1 is the source
view, matching the pinned `x₂ᵀMx₁` convention of [`EpipolarForm`](@ref).

Composing its [`epipolar_map`](@ref) with the calibrated → pixel lifts
`HomographyMap{GnomonicFrame{idᵢ}, ImageFrame{idᵢ}}(Kᵢ)` turns this into the
pixel-frame fundamental form — which is exactly what
[`EpipolarForm`](@ref)`(rig)` does.
"""
function essential_form(rig::StereoRig)
    a, b = ids(rig)
    # Delegates to the EssentialMat(::EuclideanMap) constructor (fixed in
    # 0.10) — the labeled relative transform is consumed directly; its frame
    # tags are irrelevant to the E = [t]×R algebra.
    E = EssentialMat(extrinsics(rig))
    return EpipolarForm{GnomonicFrame{a}, GnomonicFrame{b}}(E)
end

"""
    EpipolarForm(rig::StereoRig)
        -> EpipolarForm{ImageFrame{id₁}, ImageFrame{id₂}, FundMat}

The rig's PIXEL-frame fundamental form, minted in one call: the calibrated
form `E = [t]×R` of [`essential_form`](@ref) is composed with the per-view
calibrated → pixel collineations `Kᵢ = HomographyMap{GnomonicFrame{idᵢ},
ImageFrame{idᵢ}}(Kᵢ)` — `K₂ ∘ epipolar_map(E) ∘ inv(K₁)` — computing
`F = K₂⁻ᵀ E K₁⁻¹` with every junction frame-checked. Slot 1 takes
source-view pixels, slot 2 target-view pixels (`u₂ᵀ F u₁ = 0` — the pinned
convention of [`EpipolarForm`](@ref); an `ImageFrame{id}` point is always
read in the canonical `:colmap` origin convention).

The associated point → line correlation is [`epipolar_map`](@ref); the
kernels are [`epipoles`](@ref).
"""
# WHY THE E FORM STAYED ON THE CHART WHILE EVERYTHING ELSE MOVED TO BEARINGS
#
# Calibrated geometry in this stack acts on unit bearings (unit rays, `StaticVector{3}`), and
# `constraint_with_jacobians(E, f₁, f₂; M₁, M₂)` is the route. `essential_form`
# is the exception, and deliberately so rather than by oversight:
#
# `EpipolarForm(rig)` bridges E to F by composing with
# `HomographyMap{GnomonicFrame, ImageFrame}(K)`. That is a projective map on
# the CHART, and it has NO bearing-domain equivalent — a 3×3 homography cannot
# map the sphere to pixels for a non-pinhole lens, because the pixel of a ray
# is whatever the lens says it is, not a linear function of the ray. So the
# E → F bridge is intrinsically chart-bound, and therefore pinhole-only.
#
# `EpipolarForm` accordingly stays the chart/pixel citizen it is: its slots are
# `Frame{2}` and it evaluates on `StaticVector{2}`. A bearing-domain FORM would
# be a genuinely different object, not a relabelling — but no dispatch site
# consumes one today (E reaches bearings through the constraint directly, and
# RVG's only form-mediated E path is removed when E drops its measurement
# maps), and this repo adds a type when a consumer needs it. If one appears,
# it is a new type beside this one, NOT a widening of this one: making
# `EpipolarForm` mean both is the confusion the bearing/`GnomonicFrame` split
# exists to prevent.
function EpipolarForm(rig::StereoRig)
    a, b = ids(rig)
    # The view ids are run-time values, so the lifts' frame tags are too: the
    # assertions keep `∘` on the collineation × correlation methods.
    K₁ = HomographyMap{GnomonicFrame{a}, ImageFrame{a}}(calibration_matrix(source(rig)))::HomographyMap{3}
    K₂ = HomographyMap{GnomonicFrame{b}, ImageFrame{b}}(calibration_matrix(target(rig)))::HomographyMap{3}
    return EpipolarForm(K₂ ∘ epipolar_map(essential_form(rig)) ∘ inv(K₁))
end

"""
    StereoRig(K1, K2, R_rel, t_rel) -> StereoRig

Construct a stereo rig from two intrinsics matrices plus the relative pose
`(R_rel, t_rel)` from camera 1 to camera 2 (extrinsic convention:
`X_cam2 = R_rel · X_cam1 + t_rel`).  Camera 1 is placed at the world origin
with identity rotation; camera 2 gets the given relative pose.

Convenience constructor for benchmarks where ground-truth is expressed as
two calibration matrices plus a per-pair relative pose (e.g. HEB's
`K_<img>` and `pose_<pair>` keys).
"""
function StereoRig(
        K1::StaticMatrix{3, 3, T},
        K2::StaticMatrix{3, 3, T},
        R_rel::StaticMatrix{3, 3, T},
        t_rel::StaticVector{3, T};
        check::Bool = true,
        ids::Tuple{Symbol, Symbol} = (:cam1, :cam2)
    ) where {T}
    cam1 = PosedCamera(K1, one(RotMatrix{3, T}), zero(Vec3{T}); check = check)
    cam2 = PosedCamera(K2, RotMatrix{3, T}(R_rel), Vec3{T}(t_rel); check = check)
    return StereoRig(cam1, cam2; ids)
end

"""
    depth_transfer(source::PosedCamera, target::PosedCamera, extrinsics)
    depth_transfer(rig::StereoRig)

Depth-ASSUMED point transfer from the source camera to the target camera:
returns a closure mapping a source pixel to a target pixel by backprojecting
to a ray, fixing **unit depth** in the source camera frame, transforming
through `extrinsics`, and projecting into the target image.

WARNING: this is a point-to-point transfer under an arbitrary depth
assumption — it is NOT the epipolar correlation. Without depth, a pixel
backprojects to a ray, not a point; the honest depth-free two-view object is
the point-to-LINE map of an [`EpipolarForm`](@ref) (see
[`epipolar_map`](@ref), which since 0.8 names only that action). For real
transfer with known depth, use [`transfer`](@ref) with a depth-attributed
point; for correspondence search, use triangulation with matched features.

Renamed from `epipolar_map` in 0.8 — the old name promised a correlation
and delivered a transfer.

# Arguments
- `source::PosedCamera`: Source camera
- `target::PosedCamera`: Target camera
- `extrinsics::EuclideanMap`: Relative transform from source to target

# Returns
- Function mapping pixel coordinates from source to target camera
"""
function depth_transfer(source::PosedCamera, target::PosedCamera, extrinsics)
    # This implementation assumes a specific depth for the 3D point (see the
    # docstring): unit depth in the source camera frame.
    function map_pixel(u)
        # Backproject to normalized ray (no depth)
        ray_source = backproject(source.model, u)
        # Assume depth=1 in source camera frame
        X_source = ray_source  # SVector{3} with unit depth
        # Transform to target camera frame (a plain map: unit depth is unitless)
        X_target = extrinsics(X_source)
        # Project to target image
        return project(target.model, X_target)
    end
    return map_pixel
end

# A physical relative pose is read in metres: the unit depth is 1 m.
depth_transfer(source::PosedCamera, target::PosedCamera, extrinsics::UnitfulEuclideanMap{3}) =
    depth_transfer(source, target, ustrip(m, extrinsics))
depth_transfer(rig::StereoRig) = depth_transfer(source(rig), target(rig), extrinsics(rig))

# ============================================================================
# Multi-view Point Transfer (depth-driven)
# ============================================================================

"""
    transfer(src_model::CameraModel, extrinsics::EuclideanMap,
             tgt_model::CameraModel, p::Attributed{<:StaticVector{2},<:Real})
        -> Attributed{Point2, T}

Primitive form: transfer a depth-attributed pixel from one view to another
through an explicit `(src_model, extrinsics, tgt_model)` triple.  Composes
`unproject(src_model, ·, ·)`, `extrinsics(·)`, and `project(tgt_model, ·)`,
binding the resulting target-frame depth as the attribute on the returned
point.

Multi-view analog of `unproject` / `project` (see also `depth_transfer`,
which is the unit-depth-assumed special case).
"""
function transfer(
        src_model::CameraModel,
        extrinsics::EuclideanMap,
        tgt_model::CameraModel,
        p::Attributed{<:StaticVector{2}, <:Real}
    )
    X_src = unproject(src_model, p.value, p.attributes)
    X_tgt = extrinsics(X_src)
    pixel = project(tgt_model, X_tgt)
    return Attributed(pixel, X_tgt[3])
end
# A physical relative pose strips in metres: the depths are plain metres.
transfer(src_model::CameraModel, extrinsics::UnitfulEuclideanMap{3}, tgt_model::CameraModel, p::Attributed{<:StaticVector{2}, <:Real}) =
    transfer(src_model, ustrip(m, extrinsics), tgt_model, p)

"""
    transfer(rig::StereoRig, p::Attributed{<:StaticVector{2},<:Real})
        -> Attributed{Point2, T}

Convenience wrapper: transfer through `rig` in the source→target direction.
For the reverse direction, call the four-arg primitive form directly with
`target(rig).model`, `pose(rig)`, `source(rig).model`.
"""
transfer(rig::StereoRig, p::Attributed{<:StaticVector{2}, <:Real}) =
    transfer(model(source(rig)), extrinsics(rig), model(target(rig)), p)

# ----------------------------------------------------------------------------
# Internal: COLMAP pixel → 1-based array index, with bounds-failure fallback.
# Convention: invalid depths are NaN (stored as such in our depth maps; data-
# specific markers like uint16 0 or float Inf are converted at the loader
# boundary).  Out-of-bounds lookups also return NaN — same "unknown" semantics.
# ----------------------------------------------------------------------------
@inline function _depth_at(depth_map::AbstractMatrix, p)
    # COLMAP convention: 1-based index (i, j) has center at (j-0.5, i-0.5).
    # Bounds are decided in float, before `floor(Int, ·)`: a transfer behind
    # the camera is NaN, and a huge coordinate would not fit an Int.
    x, y = p[1], p[2]
    h, w = size(depth_map)
    (0 <= x < w && 0 <= y < h) || return convert(eltype(depth_map), NaN)
    return depth_map[floor(Int, y) + 1, floor(Int, x) + 1]
end

# ============================================================================
# Inlier mask: validate existing csponds against rig + depth ground truth
# ============================================================================

# The GT-validation defaults, in ONE place: `validate_gt_inliers` forwards to
# `validate_gt_metric` and both expose the same three knobs, so a literal in
# each signature was two authorities for one value.
const _GT_DEFAULTS = (rel_tol = 0.05, gross_outlier_pixel_frac = 0.01, bidirectional_depth = true)

# Extract the 2D pixel from either a plain Point2/SVector or an Attributed{Point2,…}
# Endpoint anchoring goes through `centroid` (#21): identity on points,
# unwrap on Attributed, recurse on Uncertain — one generic, no local helper.

"""
    validate_gt_metric(rig, csponds, depth_src, depth_tgt;
                       rel_tol = 0.05, gross_outlier_pixel_frac = 0.01,
                       bidirectional_depth = true)
        -> (residuals::Vector{Vec3{Float64}}, valid::BitVector)

Geometric pre-validation in metric space, without any pixel-residual
threshold. A cspond is valid when it satisfies, in order:

1. a gross-outlier gate: its GT-pose Sampson distance is at most
   `gross_outlier_pixel_frac` × the target image diagonal (pose only, no depth);
2. `depth_src` is finite at `p_A` and `depth_tgt` is finite at `p_B`;
3. the forward warp of `p_A` lands where `depth_tgt` is finite, and the
   predicted and observed target depths agree within `rel_tol` (relative);
4. with `bidirectional_depth`, warping `p_B` back at its observed depth agrees
   with `depth_src` the same way.

The residual is **3D, in meters**: forward, in the target camera frame,

    residuals[i] = unproject(tgt, p̂_B, d̂_B) − unproject(tgt, p_B, d_obs)

and, with `bidirectional_depth`, the longer of that and the backward residual
(in the source camera frame), so `‖residuals[i]‖` is the worse-direction
metric disagreement at the surface.  Used by
`calibrate_metric_tol` to estimate the noise floor without applying a
tolerance gate.
"""
function validate_gt_metric(
        rig::StereoRig,
        csponds::AbstractVector,
        depth_src::AbstractMatrix,
        depth_tgt::AbstractMatrix;
        rel_tol::Real = _GT_DEFAULTS.rel_tol,
        gross_outlier_pixel_frac::Real = _GT_DEFAULTS.gross_outlier_pixel_frac,
        bidirectional_depth::Bool = _GT_DEFAULTS.bidirectional_depth
    )
    # the depth maps are plain metres: a physical rig is read in metres
    has_units(rig) && return validate_gt_metric(
        ustrip(m, rig), csponds, depth_src, depth_tgt; rel_tol, gross_outlier_pixel_frac, bidirectional_depth
    )
    n = length(csponds)
    residuals = Vector{Vec3{Float64}}(undef, n)
    valid = falses(n)
    src_model = model(source(rig))
    tgt_model = model(target(rig))

    # Stage-1 gross-outlier filter: GT-pose Sampson distance.
    #   F_gt = K₂⁻ᵀ · [t_gt]× · R_gt · K₁⁻¹
    #   reject if sampson_distance(F_gt, p_A, p_B) > frac_diag · diag_target
    # Sampson is depth-free, so this filter works on datasets with sparse or
    # noisy depth (KITTI Velodyne, ScanNet Kinect) where the pixel depth-warp
    # filter would jitter or NaN-out most candidates. The filter is also
    # structurally distinct from the metric 3D residual (stage 3) which uses
    # depth — so the cspond label still comes from a depth-based residual,
    # only gross-outlier pre-filtering uses pose-only Sampson.
    K_src = calibration_matrix(source(rig))
    K_tgt = calibration_matrix(target(rig))
    R_gt = SMatrix{3, 3, Float64}(extrinsics(rig).R)
    t_gt = Vec3{Float64}(extrinsics(rig).t)
    F_gt = FundMat(EssentialMat(R_gt, t_gt), K_src, K_tgt)
    sz_tgt = extent(target(rig))
    diag_px = hypot(Float64(sz_tgt[1]), Float64(sz_tgt[2]))
    cap_samp = gross_outlier_pixel_frac * diag_px

    @inbounds for i in 1:n
        residuals[i] = Vec3{Float64}(NaN, NaN, NaN)
        p_A = centroid(csponds[i].first)
        p_B = centroid(csponds[i].second)

        # Stage 1 — gross outlier filter via GT-pose Sampson (pose-only,
        # no depth dependence).
        sampson_distance(F_gt, Point2{Float64}(p_A[1], p_A[2]), Point2{Float64}(p_B[1], p_B[2])) > cap_samp && continue

        d_A_obs = Float64(_depth_at(depth_src, p_A))
        isnan(d_A_obs) && continue
        d_B_obs = Float64(_depth_at(depth_tgt, p_B))
        isnan(d_B_obs) && continue

        # Forward warp: (p_A, d_src(p_A)) → predicted (p̂_B, d̂_B).
        fwd = transfer(rig, Attributed(p_A, d_A_obs))
        p_B_pred = fwd.value
        d_B_pred = Float64(fwd.attributes)
        d_B_obs_at_pBhat = Float64(_depth_at(depth_tgt, p_B_pred))
        isnan(d_B_obs_at_pBhat) && continue
        abs(d_B_pred - d_B_obs_at_pBhat) / d_B_obs_at_pBhat > rel_tol && continue

        # Forward 3D residual in target frame, via VGC's `unproject`
        # (handles any pinhole/distorted lens model uniformly).
        X_pred_B = unproject(tgt_model, p_B_pred, d_B_pred)
        X_obs_B = unproject(tgt_model, p_B, d_B_obs)
        r_fwd = Vec3{Float64}(X_pred_B - X_obs_B)
        n_fwd = norm(r_fwd)

        if bidirectional_depth
            bwd = transfer(
                tgt_model, pose(rig), src_model,
                Attributed(p_B, d_B_obs)
            )
            p_A_pred = bwd.value
            d_A_pred = Float64(bwd.attributes)
            d_A_obs_at_pAhat = Float64(_depth_at(depth_src, p_A_pred))
            isnan(d_A_obs_at_pAhat) && continue
            abs(d_A_pred - d_A_obs_at_pAhat) / d_A_obs_at_pAhat > rel_tol && continue

            # Backward 3D residual in source frame.
            X_pred_A = unproject(src_model, p_A_pred, d_A_pred)
            X_obs_A = unproject(src_model, p_A, d_A_obs)
            r_bwd = Vec3{Float64}(X_pred_A - X_obs_A)
            n_bwd = norm(r_bwd)

            # Store the worst-direction vector so `norm(residuals[i])` is
            # `max(‖fwd‖, ‖bwd‖)` — what the inlier predicate consumes.
            residuals[i] = n_bwd > n_fwd ? r_bwd : r_fwd
        else
            residuals[i] = r_fwd
        end
        valid[i] = true
    end
    return residuals, valid
end

"""
    validate_gt_inliers(rig::StereoRig, csponds, depth_src, depth_tgt;
               rel_tol                  = 0.05,
               metric_tol::Quantity,
               gross_outlier_pixel_frac = 0.01,
               bidirectional_depth      = true) -> BitVector

Validate csponds against GT rig + per-view depth maps, returning a
BitVector of inliers. The threshold is a physical **metric length at the
surface** (e.g. `5mm`, `50mm`); see `_gt_inliers_metric` for the math.

Pixel-based dispatch was removed in favor of metric-only to keep the GT
truth process apples-to-apples across datasets at different
resolutions/focal lengths.
"""
function validate_gt_inliers(
        rig::StereoRig,
        csponds::AbstractVector,
        depth_src::AbstractMatrix,
        depth_tgt::AbstractMatrix;
        rel_tol::Real = _GT_DEFAULTS.rel_tol,
        metric_tol::Quantity,
        gross_outlier_pixel_frac::Real = _GT_DEFAULTS.gross_outlier_pixel_frac,
        bidirectional_depth::Bool = _GT_DEFAULTS.bidirectional_depth
    )
    return _gt_inliers_metric(
        rig, csponds, depth_src, depth_tgt;
        rel_tol = rel_tol,
        metric_tol = metric_tol,
        gross_outlier_pixel_frac = gross_outlier_pixel_frac,
        bidirectional_depth = bidirectional_depth
    )
end


"""
    _gt_inliers_metric(rig, csponds, depth_src, depth_tgt;
                       rel_tol, metric_tol::Quantity, bidirectional_depth) -> BitVector

Metric-tolerance variant. The 3D-disagreement vector at the surface is

    r₃ᴅ = X̂_B − X_B

where `X̂_B` is the predicted 3D point in the target camera frame (from
forward-warping `(p_A, depth_src(p_A))` through the rig) and `X_B` is the
observed 3D point at `(p_B, depth_tgt(p_B))`. An inlier has `‖r₃ᴅ‖ < metric_tol`.

The metric residual is **resolution- and focal-length-independent** by
construction — perfect for cross-dataset comparisons (laser, Kinect, lidar
all share one tolerance number in mm/cm/m).

`metric_tol` is any physical length quantity; we strip to meters internally.
"""
function _gt_inliers_metric(
        rig::StereoRig,
        csponds::AbstractVector,
        depth_src::AbstractMatrix,
        depth_tgt::AbstractMatrix;
        rel_tol::Real,
        metric_tol::Quantity,
        gross_outlier_pixel_frac::Real,
        bidirectional_depth::Bool
    )
    residuals, valid = validate_gt_metric(
        rig, csponds, depth_src, depth_tgt;
        rel_tol = rel_tol,
        gross_outlier_pixel_frac = gross_outlier_pixel_frac,
        bidirectional_depth = bidirectional_depth
    )
    # Residuals are in metres (ETH3D world scale): strip the tolerance in metres
    # (a ConversionError unless it is a length).
    tol_m = ustrip(m, metric_tol)
    @inbounds for i in eachindex(valid)
        valid[i] || continue
        r = residuals[i]
        if norm(r) >= tol_m
            valid[i] = false
        end
    end
    return valid
end
