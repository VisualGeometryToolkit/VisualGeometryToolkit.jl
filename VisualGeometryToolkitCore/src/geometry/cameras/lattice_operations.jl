# ==============================================================================
# Exact lattice operations: a SplineLens lattice extended on its own spacing, and
# a uniform cubic lattice subdivided by integers
# ==============================================================================
#
# Two host-side operations on a uniform cubic lattice (node j at cell j − 2, `size − 3`
# cells per axis over the closed area), which a resample of one SplineLens onto another
# lattice composes (VisualGeometryToolkitEstimation's `fit_camera(SplineLens, spline
# source)`):
#
# - `subdivide_lattice`: the same function on a lattice whose every cell is split into
#   m_x × m_y equal cells. Knot insertion (Boehm 1980) of the m − 1 knots k + i/m into
#   every span of a uniform knot vector has a closed form, the m-ary refinement of the
#   cardinal cubic B-spline M (support [0, 4]):
#
#       M(t) = Σₖ aₖ M(m t − k),   a(z) = m⁻³ (1 + z + … + z^{m−1})⁴,   k = 0 … 4(m − 1),
#
#   so the coarse node j (basis M(t − j + 4)) is Σₖ aₖ times the fine node
#   j′ = m(j − 4) + k + 4 (basis M(m t − j′ + 4), centred at (j′ − 2)/m). The fine
#   controls are d = R c per axis, R the n′ × n matrix of those aₖ, n′ = m(n − 3) + 3,
#   and on the lattice D = R_y C R_xᵀ. The representation is unique, so this is the
#   result of inserting the knots one at a time (the tests compare it with a literal
#   Boehm insertion). The fine nodes 1 … n′ read coarse nodes 1 … n only, so the fine
#   spline is the coarse one on the whole closed area. Each fine edge cell is a piece
#   of the coarse edge cell's cubic, so the continuation past the area is the same
#   cubic too.
#
# - `extend_lattice`: the lattice grown outward by whole cells, with the same origin and
#   phase, until every node whose open support meets a region exists. The source's
#   present nodes keep their controls bit for bit. Every window inside the source's area
#   reads source nodes only, so the spline there is unchanged to the rounding of the
#   lattice coordinate. The new nodes (and the absent source nodes the region needs) are
#   the minimiser of the discrete second-order bending energy
#   Σ Δxx² + 2Δxy² + Δyy² on the lattice's cells (`difference_rows`), the present source
#   nodes held: rotation-invariant, so neither the normal nor the tangential curvature
#   grows outward, and its null space is the affine functions, so an affine lattice is
#   continued exactly. In 1-D (no tangential variation) it is the natural rule
#   c₀ = 2c₁ − c₂ (`natural_extrapolation`). It sets the START values of the
#   extrapolated nodes only: no refinement's objective holds it.
#
# - `folded_pixels`: where a lattice's backward map folds over a region (the orientation
#   d · (∂d/∂u × ∂d/∂v) against the source's sign). `extend_lattice` reports it as
#   `folds` and never refuses: the extension is C² but not guaranteed injective past a
#   rim the source has already folded, and the policy is the caller's.
#
# Neither operation runs on a device: they build a new lattice once on the host. The
# camera they return evaluates on a device like any SplineLens camera.

# ------------------------------------------------------------------------------
# Subdivision
# ------------------------------------------------------------------------------

# The m-ary refinement mask of the cardinal cubic B-spline, (a₀, …, a_{4(m−1)}):
# the coefficients of m⁻³ (1 + z + … + z^{m−1})⁴. Each residue class mod m sums to
# one, so a fine node's coefficients sum to one (a constant is reproduced).
function _subdivision_mask(m::Int)
    a = [1.0]
    for _ in 1:4
        b = zeros(length(a) + m - 1)
        for i in eachindex(a), j in 0:(m - 1)
            b[i + j] += a[i]
        end
        a = b
    end
    return a ./ m^3
end

# The refinement matrix of one axis: the `m(n − 3) + 3` fine nodes as combinations of
# the `n` coarse ones (`d = R c`), fine node j′ = m(j − 4) + k + 4 taking aₖ of coarse j.
function _subdivision_matrix(n::Int, m::Int)
    a = _subdivision_mask(m)
    nf = m * (n - 3) + 3
    I, J, V = Int[], Int[], Float64[]
    for j in 1:n, k in 0:(length(a) - 1)
        jf = m * (j - 4) + k + 4
        1 <= jf <= nf && (push!(I, jf); push!(J, j); push!(V, a[k + 1]))
    end
    return sparse(I, J, V, nf, n)
end

# `(m_x, m_y)` from an integer or a pair, each ≥ 1.
_subdivision_factors(m::Integer) = _subdivision_factors((m, m))
function _subdivision_factors(factors::Tuple{Integer, Integer})
    all(>=(1), factors) || throw(ArgumentError("subdivide_lattice: the factors are integers ≥ 1, got $factors"))
    return Int(factors[1]), Int(factors[2])
end

# `R_y C R_xᵀ` per component, the controls `SVector{K, T}`s as given (computed in Float64,
# rounded once to `T`).
function _subdivided_controls(C::AbstractMatrix{SVector{K, T}}, Ry, Rx) where {K, T}
    D = ntuple(k -> Matrix(Ry * (Float64.(getindex.(C, k)) * transpose(Rx))), K)
    return map((d...) -> SVector{K, T}(d), D...)
end

# The fine nodes that read any coarse node of `mask` (`[gy, gx]`): every mask coefficient
# is positive, so a positive entry of R_y·mask·R_xᵀ is a node read.
_reads_any(Ry, Rx, mask::AbstractMatrix{Bool}) = BitMatrix(Matrix(Ry * (Float64.(mask) * transpose(Rx))) .> 0)

"""
    subdivide_lattice(spline::ChartSpline, factors) -> ChartSpline

The same spline on a lattice whose every cell is split into `m_x × m_y` equal cells:
`factors = (m_x, m_y)`, integers ≥ 1 (along the chart's first and second coordinate,
the `grid_size` order), or one integer for both. The area is the same, and each axis
of `n` controls becomes `m (n − 3) + 3`. It is exact knot insertion (Boehm's, every
span at once): the fine controls are `R_y C R_xᵀ`, with `R` the m-ary refinement of
the cubic B-spline, `m⁻³ (1 + z + … + z^{m−1})⁴`. The value is the same at every
point, the continuation past the area included, to the rounding of the sums.
Host-side.
"""
function subdivide_lattice(cs::ChartSpline, factors)
    mx, my = _subdivision_factors(factors)
    gh, gw = size(cs.controls)
    return ChartSpline(cs.area, _subdivided_controls(cs.controls, _subdivision_matrix(gh, my), _subdivision_matrix(gw, mx)))
end

"""
    subdivide_lattice(camera::CameraModel{<:AbstractIntrinsics, <:SplineLens}, factors; extrapolated = nothing)
        -> (; camera, extrapolated::BitMatrix)

The same [`SplineLens`](@ref) camera on a lattice whose every cell is split into
`m_x × m_y` equal cells: its backward spline subdivided by `factors = (m_x, m_y)`
(integers ≥ 1, or one integer for both; [`subdivide_lattice`](@ref)`(::ChartSpline, factors)`).
The backward spline is exact knot insertion, so it is the same function everywhere.
`K`, the backward area and the forward spline are kept (the backward has not changed).
[`lattice_spacing`](@ref) is divided by `factors`.

- **Presence.** A fine node is present when every coarse node it reads is present.
  The camera then images exactly the pixels the source images, and its
  [`valid_domain`](@ref) is the source's. An absent node holds NaN.
- **Marks.** `extrapolated` (a `Bool` matrix of the source lattice's size, `[gy, gx]`,
  such as [`extend_lattice`](@ref)'s) marks coarse nodes. The returned `extrapolated`
  marks every fine node that reads a marked coarse node (all `false` without marks).
  No fine node that reads an unmarked node only is marked.

Host-side lattice algebra: the result evaluates on a device like any SplineLens camera.
"""
function subdivide_lattice(cam::CameraModel{<:AbstractIntrinsics, <:SplineLens}, factors; extrapolated::Union{Nothing, AbstractMatrix{Bool}} = nothing)
    mx, my = _subdivision_factors(factors)
    l = cam.lens
    b = l.backward
    gsz = gh, gw = size(b.controls)
    extrapolated === nothing || size(extrapolated) == gsz ||
        throw(DimensionMismatch("subdivide_lattice: a $(size(extrapolated)) extrapolated mask for a $gsz lattice"))
    Ry, Rx = _subdivision_matrix(gh, my), _subdivision_matrix(gw, mx)
    C = l.present === nothing ? b.controls : map((v, p) -> p ? v : zero(v), b.controls, l.present)   # an absent slot is never read
    fine = _subdivided_controls(C, Ry, Rx)
    present = l.present === nothing ? nothing : .!_reads_any(Ry, Rx, .!l.present)
    present === nothing || (fine = _absent_nan(fine, present))
    marks = extrapolated === nothing ? falses(size(fine)) : _reads_any(Ry, Rx, extrapolated)
    lens = SplineLens(ChartSpline(b.area, fine); l.forward, present)
    return (; camera = CameraModel(cam.intrinsics, lens), extrapolated = marks)
end

# ------------------------------------------------------------------------------
# Extension
# ------------------------------------------------------------------------------

# The regions a lattice operation takes, in pixels: a rectangle, an ellipse, a circle, an intersection of them.
const _LatticeRegion = Union{GeometryBasics.HyperRectangle{2}, Ellipse, Circle, RegionIntersection}

# The rounding slack of a region edge's cell coordinate: a quotient within it of an
# integer is that integer (a region equal to the source's own area adds no ring).
const _CELL_TOL = 1.0e-9

# The cell coordinates `t` of the nodes whose open support `(t − 2, t + 2)` meets the
# closed interval `[q_lo, q_hi]` (in cells from the area's origin): `q_lo − 2 < t < q_hi + 2`.
_needed_cells(qlo, qhi) = (floor(Int, qlo - 2 + _CELL_TOL) + 1, ceil(Int, qhi + 2 - _CELL_TOL) - 1)

"""
    extend_lattice(camera::CameraModel{<:AbstractIntrinsics, <:SplineLens}, region; fold_spacing = 8.0)
        -> (; camera, extrapolated::BitMatrix, offset::NTuple{2, Int}, folds)

The [`SplineLens`](@ref) camera's lattice extended outward on its own spacing, with the
same origin and phase, until every node whose support meets `region` exists. `region`
is a region in pixels of the camera's pixel frame: a `Rect` (the calibrated area of the
lattice a resample targets), an `Ellipse`, a `Circle` or an intersection of them (a
sensor ∩ its lens coverage). Its bounding box sets the nodes. A node's support here is open: the open rectangle
`node ± 2·spacing`, where its basis is nonzero. So a region equal to the source's
calibrated area adds nothing. (A node's [`support`](@ref) for [`node_mask`](@ref) is the
closed footprint, so a node touching the region only at its edge is free there but not
needed here: its basis is zero on the region.) The lattice never shrinks. The extended camera has the
same `K`, and its backward area grows by the whole cells added, so the region lies
in its [`valid_domain`](@ref).

- **Inside the source's domain** every present source node keeps its control bit for
  bit, so the spline is unchanged there, to the rounding of the lattice coordinate.
- **The extrapolated nodes** are the new nodes, and every absent source node whose
  support meets the region. Their controls minimise the discrete second-order bending
  energy `Σ Δxx² + 2Δxy² + Δyy²` of the extended lattice, each difference divided by
  its cells' widths ([`difference_rows`](@ref)`(·, 2)` and its mixed rows, per
  component), with the present source nodes held fixed. An absent node that stays
  absent is not part of it. The energy is rotation-invariant in pixels (the thin-plate
  discretization), so the curvature, normal, tangential and mixed, does not grow
  outward, corners included. Its null space is the affine functions of the node
  indices, so an affine lattice is continued exactly; without tangential variation it
  is the natural rule `c₀ = 2c₁ − c₂` ([`natural_extrapolation`](@ref)). Across the
  source's domain edge the spline is C², since a uniform cubic B-spline is C² at every
  knot. This penalty gives only the start values of the extrapolated nodes. It is not
  part of any refinement's objective: a later fit moves those nodes by its own terms.
  (A refinement's own rule for the nodes outside an imaged region is
  [`BoundaryRule`](@ref), ring by ring from the interior.)
- **Presence.** Every node is present except an absent source node whose support does
  not meet the region (it stays absent, NaN in its slot). A source with every node
  present gives a camera with every node present.
- **`extrapolated`** marks the extrapolated nodes, a `BitMatrix` of the extended lattice
  (`[gy, gx]`, like its controls), `false` for a source node kept as it was.
  [`subdivide_lattice`](@ref)'s keyword carries it to a subdivided lattice.
- **`offset`** is the 0-based `(row, col)` of the source's first control in the
  extended grid ([`window_offset`](@ref)'s convention).
- **The forward spline** is dropped, since the backward changed outside the domain.
  When nothing is extrapolated, `camera` is the input camera, its forward kept.
- **`folds`** reports where the extended camera's backward map folds over `region`:
  [`folded_pixels`](@ref)`(camera, region; spacing = fold_spacing)`, its `folded`
  samples, `samples` and `unimaged`, the orientation's sign taken in the source's own
  domain. `fold_spacing = nothing` skips the check (`folds = nothing`). The extension
  is C² and its curvature does not grow, but it is not guaranteed injective. A source
  whose rim is already folded or strongly curved continues that into the band. `extend_lattice` never refuses on a
  fold: the caller reads `folds` and decides (a fit can record it in its metadata).
  An empty `folds.folded` with `folds.unimaged == 0` means no fold at that density.

An `ArgumentError` for a region whose bounding box is not a finite rectangle, or when no complete
4 × 4 window of the source is present (its present nodes then need not determine the
extrapolation).
Host-side lattice algebra: the result evaluates on a device like any SplineLens camera.
"""
function extend_lattice(cam::CameraModel{<:AbstractIntrinsics, <:SplineLens}, region::_LatticeRegion;
        fold_spacing::Union{Nothing, Real} = 8.0)
    fold_spacing === nothing || fold_spacing > 0 ||
        throw(ArgumentError("extend_lattice: fold_spacing must be positive (or nothing: no fold report), got $fold_spacing"))
    l = cam.lens
    b = l.backward
    gsz = gh, gw = size(b.controls)
    box = Rect2(region)
    ro = GeometryBasics.origin(box); rw = GeometryBasics.widths(box)
    (all(isfinite, ro) && all(isfinite, rw) && all(>=(0), rw)) ||
        throw(ArgumentError("extend_lattice: the region's bounding box must be a finite rectangle, got $box"))
    rect = _pixel_area(cam.intrinsics, b.area)
    h = _lattice_spacing(rect, gsz)
    o = GeometryBasics.origin(rect)
    # the needed nodes' cell coordinates per axis; the source's nodes sit at −1 … g − 2
    tx = _needed_cells((ro[1] - o[1]) / h[1], (ro[1] + rw[1] - o[1]) / h[1])
    ty = _needed_cells((ro[2] - o[2]) / h[2], (ro[2] + rw[2] - o[2]) / h[2])
    (left, right), (top, bottom) = _added_nodes(tx, gw), _added_nodes(ty, gh)
    ngh, ngw = gh + top + bottom, gw + left + right
    rows, cols = top .+ (1:gh), left .+ (1:gw)
    # every node is present but an absent source node the region does not need
    needed = falses(ngh, ngw)
    needed[(ty[1] + top + 2):(ty[2] + top + 2), (tx[1] + left + 2):(tx[2] + left + 2)] .= true
    known = falses(ngh, ngw); known[rows, cols] .= present_nodes(cam)
    _has_complete_window(known) || throw(ArgumentError(
        "extend_lattice: no complete 4 × 4 window of the source is present, so its present nodes do not determine the extrapolation"))
    present = trues(ngh, ngw); present[rows, cols] .= known[rows, cols] .| needed[rows, cols]
    filled = present .& .!known
    # the folds over the region, the orientation's sign the source's own (a band never sets it)
    folds(c) = fold_spacing === nothing ? nothing : folded_pixels(c, region; spacing = fold_spacing, reference = _reference_pixel(cam))
    any(filled) || return (; camera = cam, extrapolated = falses(gsz), offset = (0, 0), folds = folds(cam))
    C = fill(SVector{3, eltype(eltype(b.controls))}(NaN, NaN, NaN), ngh, ngw)
    C[rows, cols] .= b.controls
    _extrapolate_controls!(C, known, present, h)
    # the backward area grown by the whole cells added: the source lattice's own nodes
    area = _window_area(b.area, gsz, (1 - top):(gh + bottom), (1 - left):(gw + right))
    lens = SplineLens(ChartSpline(area, _absent_nan(C, present)); present)
    extended = CameraModel(cam.intrinsics, lens)
    return (; camera = extended, extrapolated = filled, offset = (top, left), folds = folds(extended))
end

# Whether some 4 × 4 window of the lattice holds `mask` nodes only.
_has_complete_window(mask) = any(_window_present(mask, ix, iy) for iy in 3:(size(mask, 1) - 1), ix in 3:(size(mask, 2) - 1))

# The nodes added before and after an axis of `g` nodes (cell coordinates −1 … g − 2) for
# the needed cell coordinates `t = (t_lo, t_hi)`; the lattice never shrinks.
_added_nodes(t, g) = (max(0, -1 - t[1]), max(0, t[2] - (g - 2)))

# The controls of the nodes `nodes .& .!known` filled in place by the minimiser of the
# bending energy over the lattice of the `nodes` (the others are not in it), the `known`
# nodes held: per component, the axis second differences weighted by the cell size `h`
# (the energy f_xx² + 2f_xy² + f_yy² on cells hx × hy, times hx²hy²: `Δyy·hx/hy`,
# `Δxx·hy/hx`, `√2·Δxy`; controls `[gy, gx]`, so axis 1 is y), `difference_rows(nodes, 2)`.
# The normal equations DᵤᵀDᵤ cᵤ = −DᵤᵀD_k c_k: one sparse Cholesky factorization and one
# solve for every component.
function _extrapolate_controls!(C::AbstractMatrix{SVector{K, T}}, known::BitMatrix, nodes::BitMatrix, h) where {K, T}
    D = vcat(
        (h[1] / h[2]) * difference_rows(nodes, 2; axes = (1,)), (h[2] / h[1]) * difference_rows(nodes, 2; axes = (2,)),
        difference_rows(nodes, 2; axes = (), mixed = sqrt(2.0)),
    )
    k, u = vec(known), vec(nodes .& .!known)
    Du, Dk = D[:, u], D[:, k]
    F = try
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(transpose(Du) * Du))
    catch e
        e isa LinearAlgebra.PosDefException || rethrow()
        throw(ArgumentError("extend_lattice: the bending energy does not determine an extrapolated node (no difference reads it)"))
    end
    ck = [Float64(v[i]) for v in vec(C)[k], i in 1:K]       # the held controls, one column per component
    x = F \ Matrix(-(transpose(Du) * (Dk * ck)))
    C[u] .= SVector{K, T}.(eachrow(x))
    return C
end

# ------------------------------------------------------------------------------
# The orientation check: where the backward map folds
# ------------------------------------------------------------------------------

# The orientation of the backward map at pixel `p`: d · (∂d/∂u × ∂d/∂v), the signed area
# its Jacobian spans on the sphere (chart-free, so a ray past 90° counts). NaN where the
# camera images nothing.
function _orientation(cam, p)
    d, M = backproject_with_jacobian(cam, p)
    return dot(d, cross(SVector{3}(M[:, 1]), SVector{3}(M[:, 2])))
end

# A pixel the camera images inside its own lattice: the centre of its valid domain, or
# where that is not imaged (a presence mask with a hole there) the centre of the first
# complete 4 × 4 window's cell. `nothing` when it has none.
function _reference_pixel(cam::CameraModel{<:AbstractIntrinsics, <:SplineLens})
    l = cam.lens
    gsz = gh, gw = size(l.backward.controls)
    any(_window_present(l.present, ix, iy) for iy in 3:(gh - 1), ix in 3:(gw - 1)) || return nothing
    c = SVector{2, Float64}(centroid(Rect2{Float64}(valid_domain(cam))))
    isfinite(_orientation(cam, c)) && return c
    rect = _pixel_area(cam.intrinsics, l.backward.area)
    for iy in 3:(gh - 1), ix in 3:(gw - 1)
        _window_present(l.present, ix, iy) || continue
        return SVector{2, Float64}(centroid(_window_area(rect, gsz, (iy - 2):(iy + 1), (ix - 2):(ix + 1))))
    end
end

"""
    folded_pixels(camera::CameraModel{<:AbstractIntrinsics, <:SplineLens}, region; spacing = 8.0, reference = nothing)
        -> (; folded, samples, unimaged)
    folded_pixels(camera::Union{PosedCamera, BoundedCamera}, region; spacing, reference)

Where the backward map pixel → direction of the [`SplineLens`](@ref) `camera` folds over
`region` (a `Rect`, an `Ellipse`, a `Circle` or an intersection of them, in pixels of the
camera's pixel frame). The check samples the pixels of `region` on a grid of `spacing`
px, aligned to the corner of its bounding box. At each sample it forms the orientation
`d · (∂d/∂u × ∂d/∂v)` ([`backproject_with_jacobian`](@ref)), the signed area the map's
Jacobian spans on the sphere. Being chart-free, it holds for rays past 90°. A sample
folds when its orientation does not have the sign of the camera's at `reference`, or is
zero. `reference` is a pixel: by default the centre of the camera's
[`valid_domain`](@ref), or the centre of its first complete 4 × 4 window where that
centre is not imaged. [`extend_lattice`](@ref) gives it the source's, so a band never
sets the sign.

Returns the folded samples, `folded::Vector{SVector{2, Float64}}`, the number of
samples in `region` (`samples`), and the number of those the camera does not image
(`unimaged`, NaN directions: no orientation, not counted as folded). The map is locally
injective at that density over the imaged samples when `folded` is empty, and over the
whole region when `unimaged` is zero too. A fold narrower than `spacing` can escape it,
and an empty region has no samples. An `ArgumentError` when the camera does not image
`reference` (or has no complete window), when `spacing` is not positive, or when the
region's bounding box is not finite.
"""
function folded_pixels(cam::CameraModel{<:AbstractIntrinsics, <:SplineLens}, region::_LatticeRegion; spacing::Real = 8.0,
        reference::Union{Nothing, StaticVector{2}} = nothing)
    spacing > 0 || throw(ArgumentError("folded_pixels: spacing must be positive, got $spacing"))
    ref = reference === nothing ? _reference_pixel(cam) : SVector{2, Float64}(reference)
    s0 = ref === nothing ? NaN : _orientation(cam, ref)
    (isfinite(s0) && !iszero(s0)) || throw(ArgumentError("folded_pixels: the camera does not image its reference pixel $ref"))
    folded = SVector{2, Float64}[]
    samples = unimaged = 0
    B = Rect2(region)
    o = GeometryBasics.origin(B); w = GeometryBasics.widths(B)
    any(<(0), w) && return (; folded, samples, unimaged)                # an empty region
    (all(isfinite, o) && all(isfinite, w)) || throw(ArgumentError("folded_pixels: the region's bounding box must be finite, got $B"))
    for y in range(o[2], o[2] + w[2]; step = spacing), x in range(o[1], o[1] + w[1]; step = spacing)
        p = SVector{2, Float64}(x, y)
        p ∈ region || continue
        samples += 1
        s = _orientation(cam, p)
        if !isfinite(s)
            unimaged += 1
        elseif !(sign(s) == sign(s0))
            push!(folded, p)
        end
    end
    return (; folded, samples, unimaged)
end
