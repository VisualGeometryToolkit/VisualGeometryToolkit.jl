# detect_mser against VLFeat. test/data/mser_vlfeat.toml is libvl's output
# (VLFeat 1b9075f, vl_mser_process), frozen by validation/mser_vlfeat.jl
# --freeze: the images (column-major bytes), the parameter sets and VLFeat's
# 0-based seeds. A seed's region is VLFeat's vl_erfill of it: the 8-connected
# component of the pixels at most (bright: at least) the seed's level.

@testsetup module MserFixture
using TOML: TOML
export FIXTURE, fixture_image, erfill, edge_set, contour_edges, inside_contours

const FIXTURE = TOML.parsefile(joinpath(@__DIR__, "data", "mser_vlfeat.toml"))

function fixture_image(name)
    im = FIXTURE["images"][name]
    return reshape(hex2bytes(im["bytes"]), im["size"]...)
end

# vl_erfill's set: the 0-based seed `s`, filled over 8 neighbours.
function erfill(levels, s, bright)
    value = levels[s + 1]
    inside(v) = bright ? v >= value : v <= value
    cart, lin = CartesianIndices(levels), LinearIndices(levels)
    seen = falses(size(levels))
    seen[s + 1] = true
    queue = [cart[s + 1]]
    for p in queue, d in CartesianIndices((-1:1, -1:1))
        q = p + d
        checkbounds(Bool, levels, q) && !seen[q] && inside(levels[q]) || continue
        seen[q] = true
        push!(queue, q)
    end
    return sort!(lin[queue] .- 1)
end

# The boundary of a pixel set as directed unit edges between pixel corners
# (x, y), clockwise around each pixel: COLMAP, the pixel (r, c) is [c-1, c] × [r-1, r].
function edge_set(mask)
    edges = Tuple{Int, Int, Int, Int}[]
    out(r, c) = !checkbounds(Bool, mask, r, c) || !mask[r, c]
    for p in findall(mask)
        r, c = Tuple(p)
        out(r - 1, c) && push!(edges, (c - 1, r - 1, c, r - 1))
        out(r, c + 1) && push!(edges, (c, r - 1, c, r))
        out(r + 1, c) && push!(edges, (c, r, c - 1, r))
        out(r, c - 1) && push!(edges, (c - 1, r, c - 1, r - 1))
    end
    return sort!(edges)
end

# The polygons' sides cut into unit edges.
function contour_edges(contours)
    edges = Tuple{Int, Int, Int, Int}[]
    for poly in contours, k in eachindex(poly)
        (x0, y0), (x1, y1) = Int.(Tuple(poly[k])), Int.(Tuple(poly[mod1(k + 1, length(poly))]))
        @assert x0 == x1 || y0 == y1
        n = abs(x1 - x0) + abs(y1 - y0)
        dx, dy = sign(x1 - x0), sign(y1 - y0)
        for t in 0:(n - 1)
            push!(edges, (x0 + t * dx, y0 + t * dy, x0 + (t + 1) * dx, y0 + (t + 1) * dy))
        end
    end
    return sort!(edges)
end

# Even-odd: is (x, y) inside the polygons (holes subtract)?
function inside_contours(contours, x, y)
    crossings = 0
    for poly in contours, k in eachindex(poly)
        (x0, y0), (x1, y1) = Tuple(poly[k]), Tuple(poly[mod1(k + 1, length(poly))])
        x0 == x1 && x0 > x && min(y0, y1) < y < max(y0, y1) && (crossings += 1)
    end
    return isodd(crossings)
end
end

@testitem "detect_mser: VLFeat's regions, frozen" setup = [MserFixture] begin
    params = Dict(p => (; (Symbol(k) => v for (k, v) in nt)...) for (p, nt) in FIXTURE["parameters"])
    ncases = 0
    for case in FIXTURE["cases"]
        levels = fixture_image(case["image"])
        bright = case["polarity"] == "bright"
        mine = detect_mser(levels; params[case["parameters"]]..., dark_on_bright = !bright, bright_on_dark = bright)
        theirs = Set(erfill(levels, s, bright) for s in case["seeds"])
        ours = Set(LinearIndices(levels)[findall(region_mask(r, levels))] .- 1 for r in mine)   # ascending
        @test length(mine) == length(case["seeds"])
        @test ours == theirs
        @test all(r -> levels[r.seed] == r.level && r.area == count(region_mask(r, levels)) &&
                       r.polarity === (bright ? :light : :dark), mine)
        ncases += 1
    end
    @test ncases == 72
end

@testitem "detect_mser: index mapping, 5 × 11 image, one bright pixel at (2, 7)" setup = [MserFixture] begin
    using VisualGeometryToolkitFeatures: boundary_contours, MserRegion
    img = fixture_image("asymmetric_5x11")
    @test size(img) == (5, 11) && findall(==(0xf0), img) == [CartesianIndex(2, 7)]
    case = only(c for c in FIXTURE["cases"] if c["image"] == "asymmetric_5x11" && c["polarity"] == "bright" && c["parameters"] == "all_areas")
    @test case["seeds"] == [31]                      # VLFeat: (7 − 1)·5 + (2 − 1), 0-based column-major
    r = only(detect_mser(img; min_area = 0.0, max_area = 1.0, dark_on_bright = false))
    @test r.seed == CartesianIndex(2, 7)
    @test LinearIndices(img)[r.seed] - 1 == 31
    @test findall(region_mask(r, img)) == [CartesianIndex(2, 7)] && r.level == 0xf0 && is_light(r) && !is_dark(r)
    @test r.polarity === :light
    # COLMAP: the pixel (row 2, column 7) is the square x ∈ [6, 7], y ∈ [1, 2]
    @test Tuple.(only(boundary_contours(r, img))) == [(6.0, 1.0), (7.0, 1.0), (7.0, 2.0), (6.0, 2.0)]
end

@testitem "detect_mser: the dark surround of a bright disc, border-touching" begin
    using VisualGeometryToolkitFeatures: boundary_contours, MserRegion
    # the fisheye image circle's shape: a bright disc, a dark surround, and a
    # dark patch inside the disc that is not part of the surround
    img = [hypot(i - 30.5, j - 40.5) < 25 ? 0xb4 : 0x0a for i in 1:60, j in 1:80]
    img[28:31, 38:42] .= 0x0a
    surround = [hypot(i - 30.5, j - 40.5) >= 25 for i in 1:60, j in 1:80]
    rs = filter(r -> is_dark(r) && r.level == 0x0a && r.area == count(surround), detect_mser(img))
    r = only(rs)
    @test region_mask(r, img) == surround
    @test count(region_mask(r, img)) == r.area
    cs = boundary_contours(r, img)
    @test Tuple.(first(cs)) == [(0.0, 0.0), (80.0, 0.0), (80.0, 60.0), (0.0, 60.0)]
    @test length(cs) == 2                          # the image border, then the disc as a hole
    # the mask is of the image the region came from: another size, another image, refused
    @test_throws ArgumentError region_mask(r, img[1:30, :])
    @test_throws ArgumentError region_mask(r, img[:, end:-1:1] .+ 0x01)
    other = copy(img); other[1, 1] = 0xff                  # same seed level, a different fill
    r.seed == CartesianIndex(1, 1) || @test_throws ArgumentError region_mask(r, other)
end

@testitem "boundary_contours: exactly the boundary of the pixel set" setup = [MserFixture] begin
    using VisualGeometryToolkitFeatures: boundary_contours, MserRegion
    import VisualGeometryToolkitFeatures as VgtkFeatures
    shoelace(c) = sum(k -> c[k][1] * c[mod1(k + 1, end)][2] - c[mod1(k + 1, end)][1] * c[k][2], eachindex(c)) / 2
    function check(mask, contours)
        @test contour_edges(contours) == edge_set(mask)
        @test all(p -> inside_contours(contours, p[2] - 0.5, p[1] - 0.5) == mask[p], CartesianIndices(mask))
        @test shoelace(first(contours)) > 0 && all(c -> shoelace(c) < 0, contours[2:end])
    end
    n = 0
    for name in ("real_crop_64x96", "blocks_24x32", "nested_28x36", "disc_surround_48x64")
        img = fixture_image(name)
        for r in detect_mser(img; min_area = 0.0, max_area = 1.0)
            check(region_mask(r, img), boundary_contours(r, img))
            n += 1
        end
    end
    @test n > 100
    # corner-touching pixels (8-connected: one curve), a hole, a pinched hole
    m = BitMatrix([1 0 0 0; 0 1 1 1; 0 1 0 1; 0 1 1 1; 1 0 0 0])
    cs = VgtkFeatures.boundary_contours(m[1:4, :])
    check(m[1:4, :], cs)
    @test length(cs) == 2
    check(m, VgtkFeatures.boundary_contours(m))
    # the order: outer boundaries largest first, then the holes, smallest first, largest last
    h = trues(8, 9); h[2, 2] = false; h[4:6, 4:6] .= false
    hs = VgtkFeatures.boundary_contours(h)
    @test length(hs) == 3 && shoelace(hs[1]) == 72 && shoelace(hs[2]) == -1 && shoelace(hs[3]) == -9
end

@testitem "detect_mser: inputs" begin
    using VisualGeometryToolkitFeatures: boundary_contours, MserRegion
    using FixedPointNumbers: N0f8
    using Colors: Gray
    img = [UInt8((7i + 13j + i * j) % 256) for i in 1:30, j in 1:40]
    ref = detect_mser(img)
    same(a, b) = a == b
    @test same(detect_mser(reinterpret(N0f8, img)), ref)
    @test same(detect_mser(Gray{N0f8}.(reinterpret(N0f8, img))), ref)
    @test_throws ArgumentError detect_mser(Float32.(img) ./ 255)
    @test_throws ArgumentError detect_mser(img; delta = 300)
    @test_throws ArgumentError detect_mser(img; max_area = 2)
    # fewer than three pixels: no region (VLFeat returns none), not a refusal of the default min_area
    @test detect_mser(zeros(UInt8, 0, 0)) == MserRegion[]
    @test detect_mser(UInt8[0x10 0xf0]) == MserRegion[]
    @test detect_mser(fill(0x80, 1, 1)) == MserRegion[]
    @test_throws ArgumentError detect_mser(zeros(UInt8, 0, 0); delta = 300)
end
