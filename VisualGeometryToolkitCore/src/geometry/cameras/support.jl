# =============================================================================
# Supports — which of a model's parameters a camera can see
# =============================================================================
#
# A camera images a region of its sensor (`imaged_region`). Each parameter of a
# model has a SUPPORT on the sensor, the region it influences. A parameter is
# free when its support meets the imaged region and frozen otherwise. A global
# model — a parametric lens, K, a radial vignetting — has parameters whose
# support is the whole plane, so nothing of it is ever frozen; a lattice or
# node model declares its nodes and their supports through two generics.
#
# Not every node model of the stack is a sensor model: the thin-plate target
# calibrations (`ThinPlateSpline`, `HeightField`) have nodes on the TARGET's
# plane, not on the sensor, and their basis functions are global; they stay
# `GlobalSupport` (an imaged region restricts where their nodes are PLACED, not
# which are free).

"""
    SupportTrait
    GlobalSupport()
    LocalSupport()

How a model's parameters are supported on the sensor. `GlobalSupport` (the
default for every type): every parameter influences the whole image, so every
parameter is free whatever the camera images. `LocalSupport`: the model is a
set of nodes, each influencing a region of the sensor — it implements
[`nodes`](@ref)`(m)` and [`support`](@ref)`(m, i)`, and opts in with
`support_trait(::Type{<:MyModel}) = LocalSupport()`.
"""
abstract type SupportTrait end
struct GlobalSupport <: SupportTrait end
struct LocalSupport <: SupportTrait end

"""
    support_trait(m) -> SupportTrait
    support_trait(::Type{M}) -> SupportTrait

Whether `m`'s parameters have global or local support on the sensor
([`SupportTrait`](@ref)). `GlobalSupport()` unless the type opts in; the
instance form forwards to the type form.
"""
support_trait(::Type) = GlobalSupport()
support_trait(m) = support_trait(typeof(m))

"""
    nodes(m) -> indices

The node ids of a `LocalSupport` model — `CartesianIndices` of a lattice, `1:N`
for scattered nodes — in the shape [`node_mask`](@ref) returns. A contract
generic: a node model implements it.
"""
function nodes end

"""
    support(m, i) -> region

The region of the sensor node `i` of a `LocalSupport` model influences, in
pixels in the camera's pixel frame: a `Rect` for a node of a uniform B-spline
lattice ([`cubic_bspline_footprint`](@ref)), `WholePlane()` for a global basis
function. A contract generic: a node model implements it.
"""
function support end

"""
    node_position(m, i) -> Union{Point2, Nothing}

Where node `i` of a model sits on the sensor, in pixels (for placing nodes);
`nothing` when the model does not say. A contract generic with that default.
"""
node_position(m, i) = nothing

"""
    free_nodes(m, region) -> indices

The parameters of `m` free under the imaged `region` (an
[`imaged_region`](@ref), a `Rect`, an `Ellipse`, …): for a `LocalSupport` model
the nodes whose [`support`](@ref) meets the region,
`[i for i in nodes(m) if intersects(region, support(m, i))]`; for a
`GlobalSupport` model every parameter, `Colon()` — an index that selects them
all (`θ[free_nodes(m, region)]`), the same answer [`node_mask`](@ref) gives. A
consumer that needs to iterate nodes asks a `LocalSupport` model
(`support_trait(m) isa LocalSupport`).
"""
free_nodes(m, region) = _free_nodes(support_trait(m), m, region)
_free_nodes(::GlobalSupport, m, region) = Colon()
_free_nodes(::LocalSupport, m, region) = [i for i in nodes(m) if intersects(region, support(m, i))]

"""
    node_mask(m, region) -> BitArray

[`free_nodes`](@ref) as a mask in the shape of [`nodes`](@ref)`(m)` (`true` =
free), for assembling a Levenberg–Marquardt system. For a `GlobalSupport`
model `Colon()`, as `free_nodes` answers: an index meaning "every parameter",
so `θ[node_mask(m, region)]` reads the same for every model.
"""
node_mask(m, region) = _node_mask(support_trait(m), m, region)
_node_mask(::GlobalSupport, m, region) = Colon()
_node_mask(::LocalSupport, m, region) = BitArray(map(i -> intersects(region, support(m, i)), nodes(m)))

"""
    cubic_bspline_footprint(center, spacing) -> Rect2{Float64}

The footprint of one node of a uniform cubic B-spline lattice: the closed
rectangle `center ± 2·spacing` (two cells on each side, the 4×4 cells the
node's basis function is nonzero on), in the lattice's pixel frame. `center`
is the node's position and `spacing` the lattice's cell size, each a pair
`(x, y)`. The one footprint every uniform bicubic lattice model reads for its
[`support`](@ref).
"""
function cubic_bspline_footprint(center, spacing)
    c = SVector{2, Float64}(center[1], center[2])
    h = SVector{2, Float64}(spacing[1], spacing[2])
    return Rect2{Float64}(c - 2h, 4h)
end

# -----------------------------------------------------------------------------
# Core's generic grid models: a uniform cubic B-spline lattice over the
# calibrated area
# -----------------------------------------------------------------------------

support_trait(::Type{<:_GenericModel}) = LocalSupport()
nodes(m::_GenericModel) = CartesianIndices(_grid_size(m))
node_position(m::_GenericModel, i::CartesianIndex{2}) = Point2{Float64}(_node_pixel(m, i[2], i[1]))
# A node's footprint, clipped to the calibrated area the model is defined on.
# (The closed meet, Core's one rect-meet convention: a footprint always overlaps
# the area by at least a cell, so it is never empty.)
support(m::_GenericModel, i::CartesianIndex{2}) =
    _closed_meet(cubic_bspline_footprint(node_position(m, i), _lattice_spacing(m)), _calibrated_rect(m))

# -----------------------------------------------------------------------------
# The boundary rule of a lattice under an imaged region: the nodes outside it are
# DEPENDENT, defined from the interior
# -----------------------------------------------------------------------------
#
# Under an imaged region the parameters of a lattice model are its interior nodes
# (`free_nodes(m, region)`, the nodes whose support meets the region). Every other
# node is neither a parameter nor frozen at a start value: it is defined from the
# interior by the natural boundary condition f'' = 0 of the correction across the
# region's border — BSplineKit's `RecombinedBSplineBasis(B, Natural())`, whose
# recombination on a uniform cubic lattice is the linear extrapolation of the
# correction outward (`natural_extrapolation`), applied ring by ring so it holds
# for a border of any shape. That is the rule the `:quasi_interpolate`
# initialization applies to the source's samples past its domain, made the
# model's rule for every later change.
# Every imaged pixel's window holds interior nodes only (a node whose basis is
# nonzero at an imaged pixel has that pixel in its support), so the rule changes no
# value inside the region and the interior's weights sum to one there; what it fixes
# is the lattice outside, which follows the interior instead of keeping the start.

"""
    BoundaryRule(m, region)

The boundary rule of the lattice model `m` (a `LocalSupport` model whose
[`nodes`](@ref) are the `CartesianIndices` of a 2-D lattice) under the imaged
`region`: its `interior` nodes ([`node_mask`](@ref)`(m, region)`, the
parameters) and, per dependent node, the linear combination of interior nodes
its CORRECTION is — the natural boundary condition `f'' = 0` across the
border, the outermost coefficient of BSplineKit's natural recombination
([`natural_extrapolation`](@ref), `w₁ c₁ + w₂ c₂ = 2 c₁ − c₂`), applied
outward ring by ring (Chebyshev rings around the interior): along each lattice
axis on which the two nodes inward are already defined, averaged over those axes;
with none, the same along the diagonals; with none of those either, the mean
of the defined neighbours (constant extrapolation). The coefficients of each
dependent node sum to one. [`extend_controls`](@ref) applies the rule;
[`boundary_rule_jacobian`](@ref) is its Jacobian. An `ArgumentError` when the
region meets no node's support.
"""
struct BoundaryRule
    interior::BitMatrix
    dependent::Vector{CartesianIndex{2}}
    terms::Vector{Vector{Tuple{CartesianIndex{2}, Float64}}}
end

function BoundaryRule(m, region)
    support_trait(m) isa LocalSupport || throw(ArgumentError("BoundaryRule: $(typeof(m)) is not a lattice model (GlobalSupport)"))
    interior = BitMatrix(node_mask(m, region))
    return BoundaryRule(interior)
end

"""
    BoundaryRule(interior::AbstractMatrix{Bool})

The boundary rule of a lattice whose interior nodes are `interior` (`true`),
the rest dependent ([`BoundaryRule`](@ref)`(m, region)` with the mask given).
"""
function BoundaryRule(interior::AbstractMatrix{Bool})
    any(interior) || throw(ArgumentError("BoundaryRule: no interior node (the region meets no node's support)"))
    interior = BitMatrix(interior)
    I = CartesianIndices(interior)
    coef = Dict{CartesianIndex{2}, Dict{CartesianIndex{2}, Float64}}(n => Dict(n => 1.0) for n in I if interior[n])
    known = copy(interior)
    w₁, w₂ = natural_extrapolation()
    axes_ = (CartesianIndex(1, 0), CartesianIndex(-1, 0), CartesianIndex(0, 1), CartesianIndex(0, -1))
    diags = (CartesianIndex(1, 1), CartesianIndex(1, -1), CartesianIndex(-1, 1), CartesianIndex(-1, -1))
    isknown(n) = checkbounds(Bool, known, n) && known[n]
    function combine!(acc, pairs)
        for (w, n) in pairs, (k, c) in coef[n]
            acc[k] = get(acc, k, 0.0) + w * c
        end
        return acc
    end
    dependent = CartesianIndex{2}[]
    while !all(known)
        ring = [n for n in I if !known[n] && any(isknown(n + s) for s in (axes_..., diags...))]
        isempty(ring) && break
        new = Dict{CartesianIndex{2}, Dict{CartesianIndex{2}, Float64}}()
        for n in ring
            acc = Dict{CartesianIndex{2}, Float64}()
            lin = [s for s in axes_ if isknown(n + s) && isknown(n + s + s)]
            isempty(lin) && (lin = [s for s in diags if isknown(n + s) && isknown(n + s + s)])
            if !isempty(lin)
                w = 1.0 / length(lin)
                for s in lin
                    combine!(acc, ((w₁ * w, n + s), (w₂ * w, n + s + s)))
                end
            else
                nb = [n + s for s in (axes_..., diags...) if isknown(n + s)]
                combine!(acc, ((1.0 / length(nb), k) for k in nb))
            end
            filter!(kv -> kv[2] != 0, acc)
            new[n] = acc
        end
        for (n, acc) in new
            coef[n] = acc; known[n] = true; push!(dependent, n)
        end
    end
    sort!(dependent)
    terms = [sort!([(k, c) for (k, c) in coef[n]]; by = first) for n in dependent]
    return BoundaryRule(interior, dependent, terms)
end

"""
    extend_controls(rule::BoundaryRule, controls, reference) -> matrix

The full lattice of controls under `rule`: the interior nodes of `controls`
as given, each dependent node its `reference` control plus the extrapolated
correction, `reference[j] + Σ cₖ (controls[k] − reference[k])` over its
interior terms. `controls` and `reference` are matrices of the lattice's size
(the dependent entries of `controls` are not read) of any control type with
`+`, `-` and scalar `*` (3-vectors, scalars, duals). With `controls ==
reference` the result is `reference`.
"""
function extend_controls(rule::BoundaryRule, controls::AbstractMatrix, reference::AbstractMatrix)
    size(controls) == size(reference) == size(rule.interior) ||
        throw(DimensionMismatch("extend_controls: controls $(size(controls)), reference $(size(reference)), lattice $(size(rule.interior))"))
    out = Matrix{promote_type(eltype(controls), eltype(reference))}(undef, size(controls))
    for n in eachindex(IndexCartesian(), out)
        out[n] = rule.interior[n] ? controls[n] : reference[n]
    end
    for (j, tm) in zip(rule.dependent, rule.terms)
        acc = reference[j]
        for (k, c) in tm
            acc += c * (controls[k] - reference[k])
        end
        out[j] = acc
    end
    return out
end

"""
    boundary_rule_jacobian(rule::BoundaryRule) -> Matrix{Float64}

The Jacobian of [`extend_controls`](@ref) with respect to the interior
controls, per scalar component: row = a node of the lattice (linear index),
column = an interior node in linear-index order (`findall(rule.interior)`).
An interior node's row is its unit vector, a dependent node's its
coefficients; every row sums to one.
"""
function boundary_rule_jacobian(rule::BoundaryRule)
    L = LinearIndices(rule.interior)
    col = Dict(n => k for (k, n) in enumerate(findall(rule.interior)))
    J = zeros(Float64, length(rule.interior), length(col))
    for (n, k) in col
        J[L[n], k] = 1.0
    end
    for (j, tm) in zip(rule.dependent, rule.terms), (k, c) in tm
        J[L[j], col[k]] += c
    end
    return J
end
