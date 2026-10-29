# =============================================================================
# Model-validity interface — sample degeneracy + model feasibility per class
# =============================================================================
#
# These predicates are properties of the model-class GEOMETRY, not of any
# RANSAC harness. They are consumed by VisualGeometryToolkitEstimation's RANSAC problems
# (which delegate to them) and by offline dataset generators — both need the
# SAME gates, so they live here as a single source of truth.
#
# Dispatch:
#   test_sample(::Type{M}, cs, idx)   — pre-solve, dispatch on model CLASS
#   test_model(model::M, cs, idx)     — post-solve feasibility, on INSTANCE
#   test_consensus(model::M, cs, mask)— full-inlier-set gate, on INSTANCE
#
# INNER-LOOP GATES (the frame-in = frame-out rule, conventions/frames.md): they take data
# and model in the enclosing estimator's normalization — the frame
# `CspondNormalization(M, cs)` builds for the model class (Hartley on both
# sides for H/F, the target about the distortion centre for RadialHomography)
# — and never normalize on their own. NOT ALL TOLERANCES LIVE IN THAT FRAME,
# and the difference is the whole subject of this file:
#   * the collinearity ratio (1e-3) is a thin-triangle RATIO tested in
#     normalized coords, so it is scale-free;
#   * the conditioning bound (κ = 100) is in the normalized frame;
#   * the coincidence tolerance (`COINCIDENT_PX`, 0.1 px) is in the CALLER'S
#     DATA frame and is forwarded as `tol_px` — the one gate whose unit is not
#     the estimator's, because a duplicate detection is a pixel fact.
# A caller holding a pixel tolerance for the first two converts it through the
# normalization's scale. Public, not exported: RVG's problems and the pool
# generators call them with their normalized view.
#
# `cs` is a correspondence container satisfying the `sources`/`targets`
# protocol (Csponds, StructArray of Pairs, plain Pair vector); `idx` is the
# minimal-sample index vector; `mask` is a BitVector selecting the consensus
# set.
# =============================================================================

# -----------------------------------------------------------------------------
# Geometric helpers (signed area, triangle thinness, coincidence)
# -----------------------------------------------------------------------------

# Twice the signed area of triangle (A, B, Q) — the turn at B, (B − A)×(Q − B);
# sign encodes orientation (CCW > 0). THE 2-D orientation primitive: the sample
# gate's handedness (`_orientation_preserving`) and the extent quad's convexity
# (`is_convex_quad`) both read it. Written on edge differences, so it is
# translation invariant (no cancellation between large absolute coordinates).
@inline _orient(A::StaticVector{2}, B::StaticVector{2}, Q::StaticVector{2}) =
    (B[1] - A[1]) * (Q[2] - B[2]) - (B[2] - A[2]) * (Q[1] - B[1])

# THE sign rule of the orientation tests: the non-zero values share one sign and
# at least one is non-zero — mixed signs are a bow-tie, all zero is degenerate
# (no evidence), both `false`.
@inline function _one_sign(σ::Tuple)
    pos = any(>(0), σ)
    neg = any(<(0), σ)
    return pos ⊻ neg
end

# Triangle "thinness" in the plane or in space: 2·area / L_max² (h/L_max, with h
# the perpendicular distance from the third vertex to the longest side).
# 0 ⟺ collinear, √3/2 ⟺ equilateral. Scale-invariant. Twice the area is the
# cross product's magnitude — the 2-D one's, or the 3-D one's norm.
@inline _twice_area(u::StaticVector{2}, v::StaticVector{2}) = abs(_cross2d(u, v))
@inline _twice_area(u::StaticVector{3}, v::StaticVector{3}) = norm(cross(u, v))
@inline function thin_ratio(a::StaticVector{N}, b::StaticVector{N}, c::StaticVector{N}) where {N}
    ab = b - a; bc = c - b; ca = a - c
    s = max(sum(abs2, ab), sum(abs2, bc), sum(abs2, ca))
    return iszero(s) ? 0.0 : _twice_area(ab, ca) / s
end

# True when the implicit src→dst correspondence map preserves orientation: all
# four triangles (3 of the 4 points) share a handedness, OR all flip it. Mixed
# signs ⟹ bow-tie (a real degeneracy). A uniform flip (e.g. world y-up board ↔
# y-down image) is a labelling convention, accepted. Degenerate triangles
# (signed area 0) contribute no evidence, and all-degenerate is `false`
# (`_one_sign`) — in `test_sample` that sample is already refused by
# `_not_collinear`, which reads the same four triangles; coincidence is handled
# separately.
@inline function _orientation_preserving(src::NTuple{4}, dst::NTuple{4})
    a, b, c, d = src
    a′, b′, c′, d′ = dst
    return _one_sign((
        _orient(a, b, c) * _orient(a′, b′, c′),
        _orient(a, b, d) * _orient(a′, b′, d′),
        _orient(c, d, a) * _orient(c′, d′, a′),
        _orient(c, d, b) * _orient(c′, d′, b′),
    ))
end

# The thinness (`thin_ratio`) at or below which a minimal sample's triangle is
# (near-)collinear: ≈ 0.1% of its longest side. A ratio, so it is the same in
# every frame. THE one value both point-sample gates read — the homography's
# (`_not_collinear`) and the circle's (`test_sample(::Type{<:Circle}, …)`).
const _COLLINEAR_THIN_RATIO = 1.0e-3

# True when no 3 of the 4 points are (near-)collinear within `tol` (a 4-pt
# homography solve needs no triple collinear, else its 3×3 LU systems are
# rank-deficient and produce a NaN-filled H).
@inline function _not_collinear(pts::NTuple{4, <:StaticVector{2}}, tol::Float64 = _COLLINEAR_THIN_RATIO)
    @inbounds a, b, c, d = pts
    return thin_ratio(a, b, c) > tol &&
        thin_ratio(a, b, d) > tol &&
        thin_ratio(a, c, d) > tol &&
        thin_ratio(b, c, d) > tol
end

# =============================================================================
# Coincidence: one check, two arities, ONE UNIT (pixels)
# =============================================================================
#
# `_are_points_distinct` (fixed-arity tuple, homography's 4) and
# `_points_distinct_idx` (indexed, F's 7 and E's 5) are THE SAME CHECK for
# different sample sizes, in the same unit: calibrate both or neither.
#
# THE TOLERANCE IS IN PIXELS, not in the estimator's Hartley frame. That frame
# normalises by THE SAMPLE'S OWN SPREAD, so a tolerance there rejects at a pixel
# distance that moves with the data — 0.16 px for a full-frame sample, 0.0015 px
# for one spanning 6 px, a 100x swing — and from 200 px spread down it sits
# BELOW the detector's own sub-pixel accuracy, unable to tell a duplicate
# detection from a genuine re-detection.
#
# WHAT THE CHECK IS FOR: deduplication, not numerical safety. Measured on the
# 5-point solver, conditioning rises like 1/θ as two points approach but the
# error stays at 2.5e-9 even at 0.021 arcsec separation, and NO sample at any
# separation failed to return a solution. Only EXACT coincidence fails, where
# the constraint row becomes literally duplicate. So the question is "are these
# two detections the same feature?", which is a statement about the image.
#
# WHY 0.1 px: VisualGeometryToolkitFeatures pins its sub-pixel refinement accuracy in
# its own suite — exact to 0.005 px on a pixel centre, at most 0.03 px at
# sigma = 4, 0.06 px worst case over sigma in [2,8]. A duplicate of the SAME
# feature differs by at most that; 0.1 px clears it with margin and sits far
# below the 1-3 px at which genuinely distinct features appear. The NMS radius
# (window_half = 3) was considered and REJECTED as an anchor: it is
# config-dependent, and `band_octaves > 0` makes close detections deliberate so
# that board-sized extrema stop erasing the blobs inside them.
#
# A BACKSTOP, not the fix. If duplicate detections can reach the sampler at
# all, deduplicating at ingest is cheaper and catches them once rather than
# per sample.
#
# -----------------------------------------------------------------------------
# THE PRECONDITION IS THE CALLER'S, AND NOTHING HERE CHECKS IT — OR CAN
# -----------------------------------------------------------------------------
#
# "In pixels" is a claim about the DATA A CALLER HANDS IN. An `SVector{2}`
# carries no unit; this file sees two numbers and a threshold. It cannot tell a
# pixel from a metre from a Hartley-normalized coordinate, and neither can the
# type system as these gates are typed today — the points arrive as bare static
# vectors, stripped of any frame tag several layers up. So the unit is a
# PRECONDITION ON THE CALLER'S FRAME, not an invariant this code maintains. The
# constant below is correct only for a caller who has already answered a
# question this file cannot ask.
#
# That is not hypothetical. The SAME 0.1 has been read in three different
# frames by three consumers, and NOT ONE OF THEM WRITES THE CONSTANT'S NAME:
# RVG's P3P calls `_are_points_distinct` with a constant of its own, RVG's F/H
# problems go through `test_sample(model_type(p), …)` and take the default,
# VisualGeometryToolkitEval's pools reach it through RVG's pool-pair problem constructors. So grepping for
# `COINCIDENT_PX` finds this definition and one test, and NO call site at all —
# which is why the frame each caller is in has to be documented HERE, where the
# number lives, and cannot be left to be discovered at the call sites. Measured
# by RVG and the evaluation engine; the pixel arithmetic in the middle column reproduced here:
#
#   the caller's frame        0.1 actually means        what it cost
#   ------------------------  ------------------------  ----------------------
#   world units               0.1 m — a 10 cm ball      (see below: averted)
#     (P3P's control points)    around each point
#   Hartley-normalized        14 px on a 640×480        6.4% of CLEAN samples
#     (spread-normalized)       full-frame spread;        rejected (RVG)
#                               17 px on RVG's frame
#   calibrated, K⁻¹u          0.1·f — 80 px at          22.1 POINTS of clean
#                               f = 800, 120 px at        minimal samples lost:
#                               f = 1200                  3069/4000 admitted,
#                                                         from 3954/4000
#                                                         (the HEB pool chart)
#
# The Hartley row is a property of the SAMPLE'S OWN SPREAD, so it has no fixed
# pixel value at all — 14 px is what a full-frame spread gives; a clustered one
# gives less, by the same 100× swing documented above.
#
# The world row is averted, not hypothetical: RVG supplies `_P3P_COINCIDENT_WORLD`
# explicitly and its docstring says "do NOT borrow `COINCIDENT_PX`". But the
# KEYWORD it supplies that world-unit constant through is named `tol_px`. The
# name is a caller's only warning, and at that call site it is wrong.
#
# THE STANDING QUESTION, which is what finds these: **what units is my data in,
# at every problem I construct?** Not "at this call" — by the time a sample
# reaches this gate the frame was fixed several layers above, when the problem
# was built around a container, and the answer is not recoverable from
# anything in scope here. The question has to be asked where the problem is
# CONSTRUCTED, every time, including when the answer seems too obvious to
# state. Each of the three rows above was written by someone for whom the
# answer was obvious.
#
# THE SECOND INSTANCE OF THAT QUESTION: a THRESHOLD and the RESIDUAL it is
# compared against must be in the SAME frame. The three rows above are one
# frame mismatch — data against a constant; this is the other — a bound against
# a residual, and it is the one that appears the moment a problem solves
# through measurement maps. It need not be assumed. The TransferForm ↔
# HomographyMap composition (geometry/forms.jl) pins the conversion: for an AFFINE destination map the
# transfer residual obeys `g′ = A₂ g` EXACTLY, not to first order, because an
# affine map has a constant Jacobian — measured to 3.2e-13 relative over 2000
# conditioned draws. `Kᵢ⁻¹` is affine, so that covers the calibrated path, and
# a pixel bound carried into the model frame is exactly `A₂` applied to it.
#
# Which makes the conversion ANISOTROPIC whenever `fx ≠ fy`: `K⁻¹`'s linear
# part is `diag(1/fx, 1/fy)`, so an isotropic pixel disc of radius τ arrives in
# the model frame as an ELLIPSE with semi-axes `τ/fx` and `τ/fy` — aspect
# exactly `fy/fx`. A scalar bound cannot be both. That is the same shape as the
# anisotropy already handled on this gate's bearing path, where the pixel
# conversion divides by `σ_max(M)` and the resulting over-rejection factor is
# documented at `test_sample(::Type{<:EssentialMat}, …)` below — the same
# question, asked of a different Jacobian.
#
# A GREEN SUITE IS WEAK EVIDENCE HERE. This gate can only REJECT. An
# over-rejecting gate makes RANSAC discard more minimal samples; RANSAC
# responds by drawing more, and the estimate it returns is built from the
# samples that succeeded — which are clean. So the symptom is a quieter suite, not
# a redder one: fewer hypotheses per run, a slower search, at worst a wider
# tail on a hard case, and nothing that trips an assertion. The HEB defect lost
# 22.1 points of clean minimal samples with every suite in the stack green; it
# was found by COUNTING ADMITTED SAMPLES against a pre-change baseline, which
# is the only measurement that sees it. A test that asserts this gate's verdict
# on clean data is therefore not evidence either — it succeeds whether or not the
# tolerance means what it says.
#
# WHAT IT COSTS, MEASURED (BlobBoards). A homography absorbs a
# rescaling of its source side exactly, so geometry, inliers and pixel
# residuals are all invariant to it — which makes the scale a CONTROL, and any
# difference attributable to this gate reading the wrong unit and nothing else.
# Same scene, same 20 seeds, source units changed:
#
#   source units   converged   trials (median)   true inliers   false inliers
#   -------------  ----------  ----------------  -------------  -------------
#   mm (BB's own)     20/20      100 (the floor)     379.9            0
#   cm                20/20      100                 380.0            0
#   m                 12/20     3577.5 (max 8326)    380.0            0
#   km                 0/20        —                   —              —
#
# At metres RANSAC burns 36× the trials and fails 8 times in 20 — and WHEN IT
# CONVERGES IT IS STILL EXACTLY RIGHT: 380 true inliers, zero false. The model
# never degrades. The damage is entirely in ADMISSION, which is the reason no
# suite turns red, stated as a measurement rather than as a worry.
#
# BB's own board sources are millimetres with 7× headroom — closest pair 0.7014
# mm over 50 bundles and 60,361 blobs, and not one pair under 0.1 mm — so the
# model-to-image case is safe BY THE FILE FORMAT, not by luck. Worth knowing
# before anyone generalises this gate on the strength of it working today.
#
# There is ONE tolerance, not one per side: no consumer hands the gate metres,
# and a second way to express a unit is how a unit mismatch arises. This comment
# is where a caller is told what happens if they do.
#
# THE REMEDY IS STRUCTURAL, NOT A BETTER CONSTANT: a problem carries its
# measurement maps (`src_map`/`dst_map`, data → model), so there is no supported
# way to construct an image-to-image problem whose data is not pixels — the
# precondition is not a thing a caller must remember. That invariant covers
# image-to-image only. A board → image homography's SOURCE side is in board
# units by nature, and no measurement map makes it pixels, so 0.1 there is a
# board-unit tolerance.
const COINCIDENT_PX = 0.1

# Pairwise coincidence check over `pts[idx]`, in PIXELS (see the note above
# before changing the unit) — any sample arity (F=7, E=5), no tuple built.
@inline function _points_distinct_idx(pts, idx; tol_px::Real = COINCIDENT_PX)
    tol² = tol_px * tol_px
    K = length(idx)
    @inbounds for a in 1:(K - 1), b in (a + 1):K
        sum(abs2, pts[idx[a]] - pts[idx[b]]) < tol² && return false
    end
    return true
end
# The same check on a whole fixed-arity tuple (the homography's 4, P3P's 3).
@inline _are_points_distinct(pts::NTuple{K, <:StaticVector}; tol_px::Real = COINCIDENT_PX) where {K} =
    _points_distinct_idx(pts, Base.OneTo(K); tol_px)

# -----------------------------------------------------------------------------
# test_sample — pre-solve degeneracy gate (dispatch on model class)
# -----------------------------------------------------------------------------

"""
    test_sample(::Type{<:HomographyMat}, cs, idx) -> Bool

Reject degenerate 4-point homography samples: bow-tie (mixed orientation),
coincident points, or any 3-of-4 (near-)collinear. An inner-loop gate.

UNITS ARE MIXED HERE, AND DELIBERATELY SO. The coincidence part takes `cs` in
PIXELS and its tolerance is `COINCIDENT_PX`. The other two parts are
frame-independent and take whatever frame they are given:
`_orientation_preserving` is a SIGN, and `_not_collinear` is a RATIO
(`thin_ratio`, ≈ 0.1% of the longest side), so neither has a unit to get wrong.

`_not_collinear` STAYS, although coincidence catches duplicate detections and
it can look redundant: two near-identical points make every triangle through
them thin, so it also rejects duplicates, incidentally. Genuine collinearity is
a DIFFERENT degeneracy, the one it is for: a 3-of-4 collinear sample makes the
homography's 3×3 LU systems rank-deficient regardless of how distinct the
points are.
"""
function test_sample(
        ::Type{<:HomographyMat}, cs, idx::AbstractVector{<:Integer};
        tol_px::Real = COINCIDENT_PX
    )
    u = sources(cs); v = targets(cs)
    @inbounds src = (u[idx[1]], u[idx[2]], u[idx[3]], u[idx[4]])
    @inbounds dst = (v[idx[1]], v[idx[2]], v[idx[3]], v[idx[4]])
    return _orientation_preserving(src, dst) &&
        _are_points_distinct(src; tol_px) && _are_points_distinct(dst; tol_px) &&
        _not_collinear(src) && _not_collinear(dst)
end

"""
    test_sample(::Type{<:Circle}, points, idx) -> Bool

Reject a degenerate 3-point circle sample: three (near-)collinear points — a
repeated point among them — whose circle does not exist or runs off to an
unbounded radius. The test is the homography gate's collinearity ratio
(`thin_ratio(p₁, p₂, p₃)` above the same ≈ 0.1% of the longest side), a ratio
and so frame-free. An inner-loop gate, run before [`Circle3pt`](@ref)'s solve.
"""
@inline function test_sample(::Type{<:Circle}, points, idx::AbstractVector{<:Integer})
    @inbounds a, b, c = points[idx[1]], points[idx[2]], points[idx[3]]
    return thin_ratio(SVector{2, Float64}(a), SVector{2, Float64}(b), SVector{2, Float64}(c)) > _COLLINEAR_THIN_RATIO
end

"""
    test_sample(::Type{<:FundMat}, cs, idx) -> Bool

Reject minimal F samples with coincident points — the only degeneracy gated
pre-solve for the epipolar solvers, and unlike homography there is NO other
check here, so this one carries the whole gate. Arity-agnostic (a 7- or
8-point sample alike); E has its own method below.

`cs` is in PIXELS and the tolerance is `COINCIDENT_PX` (see the note above
`COINCIDENT_PX`). With no collinearity check behind it, a duplicate detection
this gate admits goes straight into the solver.
"""
function test_sample(
        ::Type{<:FundMat}, cs, idx::AbstractVector{<:Integer};
        tol_px::Real = COINCIDENT_PX
    )
    return _pixels_distinct(cs, idx, tol_px)
end

# The pixel coincidence gate of an epipolar sample, both sides: the F method
# above and the E method's no-Jacobian branch below share it.
@inline _pixels_distinct(cs, idx, tol_px) =
    _points_distinct_idx(sources(cs), idx; tol_px) &&
    _points_distinct_idx(targets(cs), idx; tol_px)

"""
    test_sample(::Type{<:EssentialMat}, cs, idx; tol_px = COINCIDENT_PX) -> Bool

The same coincidence question for E on UNIT BEARINGS: are any two of the
sample's bearings the same detection? Bearings are directions, so the
comparison cannot be a distance between them — it is the angle, converted to
pixels through each point's backprojection Jacobian.

`cs` carries `M₁`/`M₂` as attribute COLUMNS (`withattrs(cs; M₁, M₂)`), the
3×2 Jacobians from [`backproject_with_jacobian`](@ref); the pair's separation
in pixels is estimated as `angle(fₐ, f_b) / σ_max(M)`. Without those columns
(any container without them, a plain `Pair` vector included) this takes the
shared bearing-free path of the F method, because there is then no way to
convert an angle into the caller's units — and guessing one would be a
threshold with nothing behind it.

# The estimate is CONSERVATIVE, and by how much

`M` is generally anisotropic: a pixel step produces a different angle along
its major and minor singular directions. Dividing by `σ_max` therefore
UNDER-states the separation of a pair displaced along the minor direction, by
exactly the factor `1/aniso` (`aniso = σ_max/σ_min`) — measured to five digits
across a fisheye field. Rejection happens when the estimate falls below
`tol_px`, so the gate actually rejects true separations up to
**`tol_px × aniso`**: the effective threshold is `tol_px` along M's major axis
and `tol_px × aniso` along its minor.

Larger, not smaller — so it never UNDER-rejects, which is the failure a
deduplication backstop must not have. It begins over-rejecting genuinely
DISTINCT features only when `tol_px × aniso > d_min`, i.e.
`aniso > d_min / tol_px` — **aniso > 10** for the default 0.1 px and features
no closer than 1 px. A moderate KannalaBrandt fisheye (f = 300, k1 = −0.02)
reaches aniso 1.48 at field radius 386 px, an effective 0.148 px: about 7×
below where distinct features live. Stated because this stack exists for
strongly distorted lenses, and a field-varying threshold is otherwise the kind
of thing that resurfaces as unexplained behaviour at the periphery.

# The tangency contract

`M` is tangent to its bearing — `fᵀM = 0`, measured to 4e-20 across the field
— which is why no explicit projection onto the tangent plane appears here or
in any consumer. It is a CONTRACT, not an observation: a violation would show
up as slow drift rather than an error, so it is pinned at
`backproject_with_jacobian` and depended on here.
"""
function test_sample(
        ::Type{<:EssentialMat}, cs, idx::AbstractVector{<:Integer};
        tol_px::Real = COINCIDENT_PX
    )
    # `attrs_of` answers for any container (no columns for a plain vector)
    attrs = attrs_of(cs)
    (haskey(attrs, :M₁) && haskey(attrs, :M₂)) || return _pixels_distinct(cs, idx, tol_px)
    return _bearings_distinct_idx(sources(cs), attrs.M₁, idx, tol_px) &&
        _bearings_distinct_idx(targets(cs), attrs.M₂, idx, tol_px)
end

# Pairwise coincidence on bearings: the angle between two unit directions,
# divided by the local pixels→angle scale, against a PIXEL tolerance. σ_max is
# the conservative choice — see the docstring's anisotropy note.
@inline function _bearings_distinct_idx(fs, Ms, idx, tol_px::Real)
    K = length(idx)
    @inbounds for a in 1:(K - 1), b in (a + 1):K
        fa = fs[idx[a]]; fb = fs[idx[b]]
        ang = translation_angle(fa, fb)
        σ = max(sigma_max(Ms[idx[a]]), sigma_max(Ms[idx[b]]))
        σ > 0 || continue                      # no scale to convert with: cannot judge
        ang / σ < tol_px && return false
    end
    return true
end

# Largest singular value of a 3×2 Jacobian, without an SVD: the larger
# eigenvalue of MᵀM, which is 2×2 and closed-form.
@inline function sigma_max(M)
    a = M[1, 1]^2 + M[2, 1]^2 + M[3, 1]^2
    d = M[1, 2]^2 + M[2, 2]^2 + M[3, 2]^2
    b = M[1, 1] * M[1, 2] + M[2, 1] * M[2, 2] + M[3, 1] * M[3, 2]
    t = (a + d) / 2
    r = sqrt(max(zero(t), t * t - (a * d - b * b)))
    return sqrt(max(zero(t), t + r))
end

# -----------------------------------------------------------------------------
# test_model — post-solve feasibility gate (dispatch on instance)
# -----------------------------------------------------------------------------

"""
    test_model(H::HomographyMat, cs, idx) -> Bool

Homography feasibility: SVD conditioning (κ ≤ 100) AND projective cheirality
(uniform-sign λ) over the sample. Composes [`test_conditioning`](@ref) and
[`test_cheirality`](@ref). An inner-loop gate: `H` and `cs` are both in the
estimator's normalization, `CspondNormalization(HomographyMat, cs)` — where the
κ bound is calibrated.

(`test_model(F::FundMat, cs, idx)` — the oriented epipolar check — is defined
in `geometry/solvers/fundmat.jl`.)
"""
test_model(H::HomographyMat, cs, idx::AbstractVector{<:Integer}) =
    test_conditioning(H) && test_cheirality(H, cs, idx)

# -----------------------------------------------------------------------------
# test_consensus — full-inlier-set gate (dispatch on instance)
# -----------------------------------------------------------------------------

"""
    test_consensus(model, cs, mask) -> Bool

Validate a model against its consensus (inlier) set `mask`. For homography this
re-runs cheirality on the full inlier set (catches bow-ties that a sample-only
check misses). F/E have no consensus-set gate today and return `true`. An
inner-loop gate: `model` and `cs` in the same frame (cheirality is sign-based,
so any positive-scale normalization gives the same answer).
"""
test_consensus(H::HomographyMat, cs, mask::AbstractVector{Bool}) =
    test_cheirality(H, cs, mask)
test_consensus(::Union{FundMat, EssentialMat}, cs, mask::AbstractVector{Bool}) = true

# -----------------------------------------------------------------------------
# test_dlt_conditioning — the DESIGN matrix's gate (dispatch on the workspace)
# -----------------------------------------------------------------------------

"""
    test_dlt_conditioning(ws::SvdWorkspace; max_cond = 100.0, null_dim = 1) -> Bool

Is the DLT design matrix of the solve that filled `ws` well enough conditioned
to trust? `dlt_conditioning(ws; null_dim) <= max_cond`. An inner-loop gate:
`true` succeeds, like every other `test_*` here.

NOT `test_conditioning`, and the difference is the point. `test_conditioning(H)`
measures the MODEL matrix; this measures the DESIGN matrix's null-space gap —
whether the solution direction is isolated. A degenerate configuration can hand
back a perfectly well-conditioned `H` from a design matrix whose null space is
not, which is exactly what a collinear correspondence set does. The two
thresholds are both `100.0` by default, on different matrices, so a
wrong-matrix gate looks plausibly calibrated: the name has to carry which
matrix is meant, because the number does not.

The defaults `(max_cond, null_dim) = (100.0, 1)` are the one threshold:
VisualGeometryToolkitEstimation's DLT refit calls this at them for the relative families.
`max_cond = Inf` never refuses. The refusal is a STRICT `>`, so a ratio of
exactly `max_cond` succeeds.

The threshold is only comparable across callers that solve in the same frame —
see [`dlt_conditioning`](@ref).
"""
@inline test_dlt_conditioning(ws::SvdWorkspace; max_cond::Real = 100.0, null_dim::Int = 1) =
    dlt_conditioning(ws; null_dim) <= max_cond
