# =============================================================================
# MSER — Maximally Stable Extremal Regions, VLFeat's selection on a component tree
# =============================================================================
#
# The component tree is ImageMorphology's `MaxTree` (8-connected, as VLFeat's
# 3ⁿ − 1 neighbourhood of `vl/mser.c` lines 524–632). What follows it is
# VLFeat's `vl_mser_process` (vl/mser.c lines 716–855 at VLFeat 1b9075f): the
# variation over `delta`, the stability maxima and the filters, in VLFeat's
# order and Float32 arithmetic. validation/mser_vlfeat.jl compares the result
# with libvl; test/data/mser_vlfeat.toml freezes VLFeat's output for the suite.
#
# VLFeat's copyright notice and license, retained as its condition 1 requires:
#
#   Copyright (C) 2007-11, Andrea Vedaldi and Brian Fulkerson
#   Copyright (C) 2012-13, The VLFeat Team
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the
#      distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

"""
    MserRegion

One maximally stable extremal region, as [`detect_mser`](@ref) returns it.

# Fields
- `seed::CartesianIndex{2}`: the region's reference pixel, a pixel at the
  region's level (the tree node `MaxTree` chose; VLFeat may pick another pixel
  of the same plateau). A VLFeat seed `s` (0-based linear index into the
  column-major `H × W` image given with `dims = (H, W)`) is the pixel
  `CartesianIndices(img)[s + 1]`, and ours is VLFeat's index
  `LinearIndices(img)[seed] - 1`.
- `level::UInt8`: the region's level in the input image: a `:dark` region is
  the 8-connected component of `img .<= level` holding `seed`, a `:bright` one
  that of `img .>= level`. `img[seed] == level`.
- `area::Int`: its number of pixels.
- `variation::Float32`: VLFeat's (in)stability `(|R₊Δ| − |R|) / |R|`.
- `polarity::Symbol`: `:dark` or `:bright`.
- `pixels::Vector{CartesianIndex{2}}`: the region's pixels (VLFeat's
  `vl_erfill`), in ascending linear order.
- `contours::Vector{Vector{Point2{Float64}}}`: the boundary of the pixel set
  as closed polygons on the pixel edges (the last vertex joins the first), the
  outer boundary first and one per hole after it, in image order — `x` along
  the columns, `y` along the rows — and the COLMAP pixel origin
  (`conventions/pixel-origins.md`): the pixel `img[r, c]` is the square
  `[c − 1, c] × [r − 1, r]`, centred at `(c − 0.5, r − 0.5)`, so every vertex
  is a pixel corner with integer coordinates. See [`boundary_contours`](@ref).

[`is_dark`](@ref) and [`is_light`](@ref) read the polarity, [`region_mask`](@ref)
the pixels as a mask.
"""
struct MserRegion
    seed::CartesianIndex{2}
    level::UInt8
    area::Int
    variation::Float32
    polarity::Symbol
    pixels::Vector{CartesianIndex{2}}
    contours::Vector{Vector{Point2{Float64}}}
end

is_dark(r::MserRegion) = r.polarity === :dark
is_light(r::MserRegion) = r.polarity === :bright

"""
    detect_mser(img; delta = 5, min_area = 3 / length(img), max_area = 0.75,
                max_variation = 0.25, min_diversity = 0.2,
                bright_on_dark = true, dark_on_bright = true) -> Vector{MserRegion}

The maximally stable extremal regions (MSER; Matas, Chum, Urban and Pajdla,
"Robust wide baseline stereo from maximally stable extremal regions", BMVC
2002) of the 8-bit gray image `img`, selected as VLFeat's `vl_mser` selects
them (Andrea Vedaldi and Brian Fulkerson, BSD license; `src/mser.jl` carries
the notice). VLFeat's stability criterion is its own, simpler than the paper's.

The extremal regions are the 8-connected components of the level sets
`img .<= l`, arranged in a tree by ImageMorphology's `MaxTree`. The variation
of a region `R` at level `l` is `(|R₊Δ| − |R|) / |R|`, where `R₊Δ` is the
largest region containing `R` at a level at most `l + delta`. Of a region and
its parent region exactly one level up, the one of larger variation (on a tie,
the child) is not maximally stable; the root never is. The survivors are then
filtered, largest first:

- `max_variation`: a region of variation `>= max_variation` is dropped;
- `max_area`, `min_area`: FRACTIONS of `length(img)`; a region larger than
  `max_area * length(img)` or smaller than `min_area * length(img)` pixels is
  dropped (the default `min_area` is three pixels);
- `min_diversity`: a region whose area differs from that of its nearest
  surviving enclosing region by less than `min_diversity` times the latter's
  area is a duplicate and is dropped (`min_diversity = 1` turns this off).

`dark_on_bright` returns the dark regions (`polarity = :dark`) and
`bright_on_dark` the bright ones (`:bright`), the components of `img .>= l`:
VLFeat runs the same on `255 .- img`, whose `<=` level sets these are. Dark
regions come first; each polarity is ordered as VLFeat orders it, by level
(darkest first, resp. brightest first), then by seed.

# Input
A matrix of `UInt8`, `N0f8` or `Gray{N0f8}`: VLFeat's `vl_mser_pix` is a byte,
and these hold its 256 levels exactly. Any other element type is an
`ArgumentError`: quantize explicitly (e.g. [`gray_u8`](@ref)), so the levels
the regions are cut at are the caller's choice.

```jldoctest
julia> using VisualGeometryToolkitFeatures

julia> img = fill(0xc8, 40, 60); img[11:20, 21:35] .= 0x28;   # a dark rectangle

julia> r = only(detect_mser(img; bright_on_dark = false));

julia> r.polarity, r.area, Tuple.(only(r.contours))
(:dark, 150, [(20.0, 10.0), (35.0, 10.0), (35.0, 20.0), (20.0, 20.0)])
```
"""
function detect_mser(
        img::AbstractMatrix; delta::Integer = 5, min_area::Real = 3 / length(img),
        max_area::Real = 0.75, max_variation::Real = 0.25, min_diversity::Real = 0.2,
        bright_on_dark::Bool = true, dark_on_bright::Bool = true
    )
    0 <= delta <= 255 || throw(ArgumentError("delta must be in 0:255, got $delta"))
    0 <= min_area <= 1 || throw(ArgumentError("min_area is a fraction of the image area in [0, 1], got $min_area"))
    0 <= max_area <= 1 || throw(ArgumentError("max_area is a fraction of the image area in [0, 1], got $max_area"))
    max_variation >= 0 || throw(ArgumentError("max_variation must be non-negative, got $max_variation"))
    0 <= min_diversity <= 1 || throw(ArgumentError("min_diversity must be in [0, 1], got $min_diversity"))
    levels = _mser_levels(img)
    config = (; delta = Int(delta), min_area, max_area, max_variation, min_diversity)
    regions = MserRegion[]
    dark_on_bright && _mser!(regions, levels, :dark, config)
    bright_on_dark && _mser!(regions, levels, :bright, config)
    return regions
end

"""
    region_mask(region::MserRegion, img) -> BitMatrix

The pixels of `region`, an MSER of `img`, as a mask the size of `img`
(VLFeat's `vl_erfill`).
"""
function region_mask(region::MserRegion, img::AbstractMatrix)
    mask = falses(size(img))
    mask[region.pixels] .= true
    return mask
end

# The image as VLFeat's bytes: N0f8 and Gray{N0f8} hold exactly its 256 levels.
_mser_levels(img::AbstractMatrix{UInt8}) = (Base.require_one_based_indexing(img); img)
_mser_levels(img::AbstractMatrix{<:Union{N0f8, Gray{N0f8}}}) =
    (Base.require_one_based_indexing(img); map(p -> reinterpret(gray(p)), img))
_mser_levels(img::AbstractMatrix) = throw(
    ArgumentError(
        "detect_mser takes 8-bit gray levels (UInt8, N0f8 or Gray{N0f8}), got $(eltype(img)); " *
            "quantize explicitly, e.g. with gray_u8(img)"
    )
)

# The MSERs of one polarity, appended to `regions`. `lev` is the level VLFeat
# sees: the byte for dark regions, its inverse 255 − v for bright ones (the
# image vl_mser.c inverts), so every comparison below reads as in vl/mser.c.
function _mser!(regions::Vector{MserRegion}, levels::AbstractMatrix{UInt8}, polarity::Symbol, config)
    dark = polarity === :dark
    tree = MaxTree(levels; connectivity = 2, rev = dark)   # rev: the root is the brightest pixel
    parent = tree.parentindices
    area = areas(tree)
    lev(p) = dark ? Int(levels[p]) : 255 - Int(levels[p])
    # the tree's nodes (VLFeat's extremal regions), root first, each after its parent
    nodes = filter(p -> parent[p] == p || levels[parent[p]] != levels[p], tree.traverse)

    # Variation over +delta (lines 723–750): `top` is the largest enclosing
    # region of level at most lev + delta.
    variation = zeros(Float32, size(levels))
    for i in nodes
        top = i
        while parent[top] != top && lev(parent[top]) <= lev(i) + config.delta
            top = parent[top]
        end
        variation[i] = Float32(area[top] - area[i]) / Float32(area[i])
    end

    # Maximally stable (lines 769–792): of a region and its parent exactly one
    # level up, the more variable loses; the root loses to itself.
    stable = falses(size(levels))
    stable[nodes] .= true
    for i in nodes
        p = parent[i]
        lev(p) > lev(i) + 1 && continue
        stable[variation[i] < variation[p] ? p : i] = false
    end

    # Filters, largest region first (lines 801–855), in VLFeat's Float32.
    n = Float32(length(levels))
    max_area, min_area = Float32(config.max_area) * n, Float32(config.min_area) * n
    max_var, min_div = Float32(config.max_variation), Float32(config.min_diversity)
    for i in nodes
        stable[i] || continue
        if variation[i] >= max_var || area[i] > max_area || area[i] < min_area
            stable[i] = false
        elseif min_div < 1 && parent[i] != i
            p = parent[i]
            while !stable[p] && parent[p] != p
                p = parent[p]
            end
            Float32(area[p] - area[i]) / Float32(area[p]) < min_div && (stable[i] = false)
        end
    end

    # The survivors, in VLFeat's order; each region's pixels are its subtree.
    children = _children(parent)
    cart = CartesianIndices(levels)
    for i in sort!([i for i in nodes if stable[i]]; by = i -> (lev(i), i))
        pixels = sort!(cart[_subtree(children, i)])
        push!(regions, MserRegion(cart[i], levels[i], area[i], variation[i], polarity, pixels, boundary_contours(pixels)))
    end
    return regions
end

# The children of every pixel of the parent-link tree, packed: the children
# of `q` are `kids[first[q]:first[q + 1] - 1]`.
function _children(parent::AbstractArray{Int})
    first = [1; zeros(Int, length(parent))]
    for (p, q) in pairs(vec(parent))
        p != q && (first[q + 1] += 1)
    end
    cumsum!(first, first)
    kids, fill = Vector{Int}(undef, length(parent) - 1), copy(first)
    for (p, q) in pairs(vec(parent))
        p != q && (kids[fill[q]] = p; fill[q] += 1)
    end
    return kids, first
end

function _subtree((kids, first), root::Int)
    members = [root]
    for p in members              # grows as it goes: a breadth-first walk
        append!(members, view(kids, first[p]:(first[p + 1] - 1)))
    end
    return members
end

"""
    boundary_contours(pixels) -> Vector{Vector{Point2{Float64}}}

The boundary of a set of pixels (`CartesianIndex{2}` `(row, column)`, or a
`Bool` mask), traced along the pixel edges: one closed polygon per boundary
curve (the last vertex joins the first), vertices only where the boundary
turns, in image order (`x` = column, `y` = row) and the COLMAP pixel origin,
so the pixel `(r, c)` is the square `[c − 1, c] × [r − 1, r]`. The outer
boundaries come first, with positive shoelace area (clockwise on screen), then
the holes, negative. Diagonal neighbours are connected (8-connectivity, the
connectivity of an MSER), so where two pixels of the set meet at a corner one
curve runs through it; holes are 4-connected.
"""
boundary_contours(mask::AbstractMatrix{Bool}) = boundary_contours(findall(mask))

function boundary_contours(pixels::AbstractVector{CartesianIndex{2}})
    isempty(pixels) && return Vector{Point2{Float64}}[]
    lo, hi = extrema(pixels)
    offset = lo - CartesianIndex(2, 2)
    inside = falses(Tuple(hi - offset) .+ 1)    # one pixel of margin all round
    for p in pixels
        inside[p - offset] = true
    end
    h = size(inside, 1)
    vertex(x, y) = y + x * (h + 1) + 1     # local corner (x, y) in 0:w × 0:h
    # Directed edges with the set on their right (clockwise around each pixel):
    # start and end corner and the pixel they belong to.
    from, to, owner = Int[], Int[], Int[]
    out = Dict{Int, Vector{Int}}()          # corner -> its outgoing edges
    for p in CartesianIndices(inside)
        inside[p] || continue
        r, c = Tuple(p)
        for (dr, dc, x0, y0, x1, y1) in ((-1, 0, c - 1, r - 1, c, r - 1), (0, 1, c, r - 1, c, r),
                                         (1, 0, c, r, c - 1, r), (0, -1, c - 1, r, c - 1, r - 1))
            inside[r + dr, c + dc] && continue
            push!(from, vertex(x0, y0)); push!(to, vertex(x1, y1)); push!(owner, LinearIndices(inside)[p])
            push!(get!(Vector{Int}, out, last(from)), length(from))
        end
    end
    corner(v) = Point2{Float64}(div(v - 1, h + 1) + offset[2], rem(v - 1, h + 1) + offset[1])
    used = falses(length(from))
    contours = Vector{Point2{Float64}}[]
    for e0 in eachindex(from)
        used[e0] && continue
        edges = Int[]
        e = e0
        while true
            used[e] = true
            push!(edges, e)
            next = out[to[e]]
            # two ways on at a corner two diagonal pixels share: leave by the
            # other pixel's edge, which joins them (8-connectivity)
            e = length(next) == 1 ? only(next) : only(filter(f -> owner[f] != owner[e], next))
            e == e0 && break
        end
        heading(f) = to[f] - from[f]
        turns = [from[f] for (k, f) in enumerate(edges) if heading(f) != heading(edges[mod1(k - 1, length(edges))])]
        push!(contours, corner.(turns))
    end
    shoelace(c) = sum(k -> c[k][1] * c[mod1(k + 1, end)][2] - c[mod1(k + 1, end)][1] * c[k][2], eachindex(c)) / 2
    return sort!(contours; by = c -> -shoelace(c))
end
