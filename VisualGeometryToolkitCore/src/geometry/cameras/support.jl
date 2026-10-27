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
all (`θ[free_nodes(m, region)]`).
"""
free_nodes(m, region) = _free_nodes(support_trait(m), m, region)
_free_nodes(::GlobalSupport, m, region) = Colon()
_free_nodes(::LocalSupport, m, region) = [i for i in nodes(m) if intersects(region, support(m, i))]

"""
    node_mask(m, region) -> BitArray

[`free_nodes`](@ref) as a mask in the shape of [`nodes`](@ref)`(m)` (`true` =
free), for assembling a Levenberg–Marquardt system. A `GlobalSupport` model is
one support covering everything: a 0-dimensional `trues()`.
"""
node_mask(m, region) = _node_mask(support_trait(m), m, region)
_node_mask(::GlobalSupport, m, region) = trues()
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
function support(m::_GenericModel, i::CartesianIndex{2})
    gh, gw = _grid_size(m)
    spacing = ((m.calib_max_x + 1 - m.calib_min_x) / (gw - 3), (m.calib_max_y + 1 - m.calib_min_y) / (gh - 3))
    area = Rect2{Float64}(m.calib_min_x, m.calib_min_y, m.calib_max_x + 1 - m.calib_min_x, m.calib_max_y + 1 - m.calib_min_y)
    return intersect(cubic_bspline_footprint(node_position(m, i), spacing), area)
end
