# =============================================================================
# AsiftPlan — batched, preallocated, fused IMAS-25 ASIFT driver.
# =============================================================================
#
# Performance structure (see git history for the incremental derivation):
#   • all warps batched along a slot dimension: the AffineWarp runs as ONE
#     3D launch into a (Wr_max, Hr_max, nw) rotated buffer, and the fused
#     σ_aa-blur+Fproj as ONE 3D launch into the (W,H,nw) canvas — no
#     dependent 50-kernel chain, no GPU drain between small launches
#   • pyramid blur: shared-memory tiled kernel fusing h-sweep + v-sweep + DoG
#     in a single launch per level — the (TX+2R)×(TY+2R) input tile is read
#     once into @localmem, halving global traffic vs two separate sweeps
#   • per-warp valid-region masking (content bounds + 64-px margin): fully
#     masked tiles skip all loads via a group-uniform early exit
#   • every buffer preallocated: zero device allocation once the feature pool
#     fits the frame (`_grow_pool!`), one count sync per tilt chunk, and the
#     features cross the bus once
#   • on-device cross-warp NMS against the plan's feature buffer
#   • fully-parametric plan: type-stable host orchestration
#
# Numerical parity notes:
#   • tile h→v tap order and clamping reproduce the two-sweep FP sequence
#     bit-for-bit in the active region (loads clamp at image borders exactly
#     as the plain kernels did; the Float32 h-intermediate rounds identically)
#   • fused σ_aa+Fproj is bit-exact vs blur-then-fproj because Fproj's
#     x-interpolation weight is exactly 0 (integer x sample positions)
# =============================================================================

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize
import Atomix

# tile geometry for the fused blur (@localmem needs static dims: the tap
# radius R is a type parameter, one specialization per level radius);
# TX=TY=32, R ≤ RMAX = 13 (asserted at plan build) → tile ≤ 58×58 (13.5 KB)
# + hbuf ≤ 32×58 (7.4 KB) ≈ 21 KB

# ----- kernels ---------------------------------------------------------------

# raw H×W UInt8 upload → (W,H)[x,y] Float32 in [0,1] (transpose + scale on
# device; avoids the ~18 ms host-side permutedims/convert at 2 MP)
@kernel function _u8_to_xy!(out, @Const(u8))
    x, y = @index(Global, NTuple)
    @inbounds out[x, y] = Float32(u8[y, x]) * (1.0f0 / 255.0f0)
end

@kernel function _fill3!(a, v::Float32)
    i, j, l = @index(Global, NTuple)
    @inbounds a[i, j, l] = v
end

# batched AffineWarp: all warps in one launch. rp rows: a11,a12,a13,a21,a22,
# a23, Wr, Hr (per warp). Threads beyond the warp's rotated canvas idle.
@kernel function _bwarp3!(rot3, @Const(inp), @Const(rp))
    X, Y, w = @index(Global, NTuple)
    @inbounds if X <= Int(rp[7, w]) && Y <= Int(rp[8, w])
        W = size(inp, 1); H = size(inp, 2)
        cx = Float32(X - 1); cy = Float32(Y - 1)
        xin = rp[1, w] * cx + rp[2, w] * cy + rp[3, w]
        yin = rp[4, w] * cx + rp[5, w] * cy + rp[6, w]
        if xin < -0.5f0 || xin > Float32(W) - 0.5f0 ||
                yin < -0.5f0 || yin > Float32(H) - 0.5f0
            rot3[X, Y, w] = 0.5f0
        else
            x0 = floor(Int, xin); y0 = floor(Int, yin)
            fx = xin - x0; fy = yin - y0
            xa = clamp(x0 + 1, 1, W); xb = clamp(x0 + 2, 1, W)
            ya = clamp(y0 + 1, 1, H); yb = clamp(y0 + 2, 1, H)
            rot3[X, Y, w] = (1.0f0 - fx) * (1.0f0 - fy) * inp[xa, ya] + fx * (1.0f0 - fy) * inp[xb, ya] +
                (1.0f0 - fx) * fy * inp[xa, yb] + fx * fy * inp[xb, yb]
        end
    end
end

# batched fused σ_aa vertical blur + FprojBilinearY, all warps in one launch.
# fp rows: t, cw, ch, Wr, Hr. tapm is (17, nw) one-sided σ_aa taps; rads (nw,).
@kernel function _bfpb!(canvas, @Const(rot3), @Const(tapm), @Const(rads), @Const(fp))
    X, Y, w = @index(Global, NTuple)
    @inbounds if X <= Int(fp[2, w]) && Y <= Int(fp[3, w])
        Wr = Int(fp[4, w]); Hr = Int(fp[5, w]); t = fp[1, w]
        x0 = Float32(X - 1); y0 = Float32(Y - 1)
        xp = x0 + 0.5f0; yp = t * y0 + 0.5f0
        if xp < 0.0f0 || xp > Float32(Wr) || yp < 0.0f0 || yp > Float32(Hr)
            canvas[X, Y, w] = 0.0f0
        else
            sy = yp - 0.5f0
            yi = floor(Int, sy); uy = sy - yi
            xa = clamp(X, 1, Wr)                     # Fproj x-weight ≡ 0
            ya = clamp(yi + 1, 1, Hr); yb = clamp(yi + 2, 1, Hr)
            r = rads[w]
            aa = rot3[xa, ya, w] * tapm[1, w]
            ab = rot3[xa, yb, w] * tapm[1, w]
            for k in 1:r
                wk = tapm[k + 1, w]
                aa += (rot3[xa, max(ya - k, 1), w] + rot3[xa, min(ya + k, Hr), w]) * wk
                ab += (rot3[xa, max(yb - k, 1), w] + rot3[xa, min(yb + k, Hr), w]) * wk
            end
            canvas[X, Y, w] = (1.0f0 - uy) * aa + uy * ab
        end
    end
end

# shared-memory tiled separable blur, fused h+v+DoG in one launch per level,
# over the level's ACTIVE 32×32 tiles: `tiles` packs (bx, by, l) per tile
# (`_tile_lists`; the fully masked tiles are `_tile_fill!`'s). A workgroup is
# (32, NY) threads, each loading and h-sweeping a share of the tile and
# v-sweeping the rows ly, ly + NY, … of its column: fewer threads per tile than
# pixels, so several tiles are resident per SM and their barriers overlap (one
# 1024-thread tile per SM left it idle at every barrier). writedog=0 → seed
# mode (dog/gprev are placeholders, never read/written).
@kernel function _btile_blur_dog!(
        gout, dog, @Const(inp), @Const(gprev),
        @Const(kern), ::Val{R}, ::Val{NY}, @Const(aw), @Const(ah), fillv::Float32,
        writedog::Int32, @Const(tiles)
    ) where {R, NY}
    # NOTE structure: @synchronize sits at top level (uniform — GPU
    # requirement) and KA's CPU backend splits the body into segments at each
    # barrier: only direct `@index` assignments survive the split (and they
    # must not be nested under other macros), so each segment recomputes its
    # derived values from the top-level index bindings.
    #
    # R is the level's tap radius, a compile-time constant (one specialization
    # per radius): the tile is (32+2R)², so a level with short taps loads and
    # sweeps only the halo it reads, the tile indexing divides by a constant,
    # and the tap loops unroll (23 % of this kernel's time on a B200). On a GPU
    # the unrolled sums contract into FMAs, so the pyramid differs in its last
    # bits from the runtime-bound loops this replaced — checked to move no
    # detection beyond Monte Carlo noise (position, scale, calibration), and
    # BlobBoards' boards and poses by < 0.1 mm / 0.04°. On the CPU nothing
    # contracts and the results are unchanged.
    tile = @localmem Float32 (32 + 2R, 32 + 2R)      # (TX+2R, TY+2R)
    hbuf = @localmem Float32 (32, 32 + 2R)           # (TX, TY+2R)
    lx, ly = @index(Local, NTuple)
    g = @index(Group, Linear)

    # segment 1 — cooperative tile load.
    # Border rule is MIRRORED_REPEAT (reflect with edge duplication), the
    # pyramid sampler's — see _mirror_idx in pyramid.jl.
    # A tile whose halo lies inside the image (most of them) indexes directly:
    # `_mirror_idx` is an integer modulo by a runtime size per element, the
    # loader's main cost. The test is uniform over the workgroup.
    let W = size(inp, 1), H = size(inp, 2), T = 32 + 2R
        ox, oy, l = _tile_origin(tiles, g)
        inner = ox >= R && ox + T - R <= W && oy >= R && oy + T - R <= H
        i = (ly - 1) * 32 + lx
        while i <= T * T
            tx = (i - 1) % T + 1
            ty = (i - 1) ÷ T + 1
            sx = inner ? ox + tx - R : _mirror_idx(ox + tx - R, W)
            sy = inner ? oy + ty - R : _mirror_idx(oy + ty - R, H)
            @inbounds tile[tx, ty] = inp[sx, sy, l]
            i += 32 * NY
        end
    end
    @synchronize
    # segment 2 — horizontal sweep into hbuf (32 cols × 32+2R halo rows)
    let T = 32 + 2R
        j = (ly - 1) * 32 + lx
        while j <= 32 * T
            hx = (j - 1) % 32 + 1
            hy = (j - 1) ÷ 32 + 1
            @inbounds begin
                acc = tile[hx + R, hy] * kern[1]
                for k in 1:R
                    acc += (tile[hx + R - k, hy] + tile[hx + R + k, hy]) * kern[k + 1]
                end
                hbuf[hx, hy] = acc
            end
            j += 32 * NY
        end
    end
    @synchronize
    # segment 3 — vertical sweep + DoG, masked per pixel (masked pixels write
    # fill without touching hbuf)
    let W = size(inp, 1), H = size(inp, 2)
        ox, oy, l = _tile_origin(tiles, g)
        gx = ox + lx
        for q in 0:(32 ÷ NY - 1)
            y = ly + NY * q
            gy = oy + y
            if gx <= W && gy <= H
                @inbounds if gx > aw[l] || gy > ah[l]
                    gout[gx, gy, l] = fillv
                    if writedog == Int32(1)
                        dog[gx, gy, l] = 0.0f0
                    end
                else
                    acc = hbuf[lx, y + R] * kern[1]
                    for k in 1:R
                        acc += (hbuf[lx, y + R - k] + hbuf[lx, y + R + k]) * kern[k + 1]
                    end
                    gout[gx, gy, l] = acc
                    if writedog == Int32(1)
                        dog[gx, gy, l] = acc - gprev[gx, gy, l]
                    end
                end
            end
        end
    end
end

# The fully masked tiles of a level (`_tile_lists`): fill, and DoG 0 when
# `writedog == 1` — what `_btile_blur_dog!` writes at a masked pixel. A (32, 8)
# workgroup per 32×32 tile, four rows per thread.
@kernel function _tile_fill!(gout, dog, fillv::Float32, writedog::Int32, @Const(tiles))
    lx, ly = @index(Local, NTuple)
    g = @index(Group, Linear)
    ox, oy, l = _tile_origin(tiles, g)
    gx = ox + lx
    for q in 0:3
        gy = oy + ly + 8q
        @inbounds if gx <= size(gout, 1) && gy <= size(gout, 2)
            gout[gx, gy, l] = fillv
            if writedog == Int32(1)
                dog[gx, gy, l] = 0.0f0
            end
        end
    end
end

# The 0-based pixel origin and the warp slot of tile g of a packed list.
@inline function _tile_origin(tiles, g)
    @inbounds return 32 * Int(tiles[3g - 2]), 32 * Int(tiles[3g - 1]), Int(tiles[3g])
end

# Threads per tile column of `_btile_blur_dog!` and `_bextract!`: (32, 4)
# workgroups, measured fastest on a B200 (NY = 8 is 4–7 % slower, one thread
# per pixel 1.8× slower).
const _TILE_NY = 4

@kernel function _bdown!(dst, @Const(src), seed::Int32)
    i, j, l = @index(Global, NTuple)
    @inbounds dst[i, j, l] = src[2i, 2j, seed, l]
end

# Octave −1 seed: the exact inverse of `_bdown!`. Since the downsample
# resamples `src[2i,2j]`, the upsampled grid must satisfy `dst[2i,2j] =
# src[i,j]` — original samples land on EVEN Julia indices, odd slots are the
# bilinear midpoints, and the first row/column replicate the edge (VLFeat
# clamps at the far edge instead; the parity is mirrored because its
# downsample is origin-aligned, `dst[i]=src[2i]`, where ours is
# Lowe-aligned, `dst[i]=src[2i+1]` 0-indexed).
#
# This alignment is what lets `_bextract!`'s existing coordinate map
# `x₀ = sx·2^o + (2^o − 1)` stay correct at `o = −1` (giving `x₀ = sx/2 − 1/2`)
# with no bias term: it is precisely the inverse of `x₋₁ = 2·x₀ + 1`.
#
# The upsample over a packed tile list (`_dilate_tiles`): the tiles the seed blur
# reads, a (32, 8) workgroup per 32×32 tile, four rows per thread. The rest of
# the upsampled slot is scratch the next level overwrites unread.
@kernel function _bupsample!(dst, @Const(src), @Const(tiles))
    lx, ly = @index(Local, NTuple)
    g = @index(Group, Linear)
    ox, oy, l = _tile_origin(tiles, g)
    I = ox + lx
    for q in 0:3
        J = oy + ly + 8q
        @inbounds if I <= size(dst, 1) && J <= size(dst, 2)
            dst[I, J, l] = _upsample_sample(src, I, J, l)
        end
    end
end

# THE 2× upsampling of the package (the ASIFT seed above, `detect_foerstner`'s
# doubling): the doubled pixel (I, J), 1-based, is the bilinear sample of slice
# `l` of `src` at input position (I/2, J/2), in `src`'s element type. Clamping
# the CONTINUOUS coordinate (not the index) is what makes the edge replicate
# exactly: I=1 → u=1 → fu=0 → src[1]. In COLMAP pixel centres the doubled
# position x′ is input position (x′ − 1/2)/2.
@inline function _upsample_sample(src, I, J, l)
    T = eltype(src)
    W = size(src, 1); H = size(src, 2)
    u = clamp(T(I) / 2, one(T), T(W))
    v = clamp(T(J) / 2, one(T), T(H))
    i0 = unsafe_trunc(Int32, u); j0 = unsafe_trunc(Int32, v)
    fu = u - T(i0); fv = v - T(j0)
    i1 = min(i0 + Int32(1), Int32(W)); j1 = min(j0 + Int32(1), Int32(H))
    @inbounds return (one(T) - fv) * ((one(T) - fu) * src[i0, j0, l] + fu * src[i1, j0, l]) +
        fv * ((one(T) - fu) * src[i0, j1, l] + fu * src[i1, j1, l])
end

# `a` beats `b` in the extremum test: strictly for a neighbour that precedes
# the centre in (ds, dy, dx) order, or ties it for one that follows (`after`).
@inline _beats(a, b, after::Bool) = after ? a >= b : a > b

# ExtractKeypoints over a (W,H,NS,nw) DoG cube, one candidate: `_bextract!`
# calls it at (X, Y) on slice Sl of warp Lw. Scale+warp folded, masked per
# warp. Emits (x, y, sigma, intensity, warp) columns, [x,y,s] 1-indexed
# (s0 = s-1 in a 0-indexed shader's terms):
#   • 26-neighbour extremum (center excluded on its own slice), strict against
#     the neighbours before the centre in (ds, dy, dx) order and non-strict
#     against those after it — the one deliberate departure from the shader,
#     which is strict against all 26 and so rejects both pixels of an exact
#     tie; the boundary slices (s0 == 0, s0 == last) use the 17-neighbour test
#   • early reject |center| ≤ 0.8·dog_threshold
#   • up to 5 Newton iterations; axis map h11↔scale, h22↔x, h33↔y; symmetric 3×3
#     inverse; converge < 0.6; round-walk within the interior; accept |off| < 1.5;
#     a failed refine keeps the integer extremum; boundary slices emit integer
#   • validity gate |new_val| > dog_threshold
#   • edge test tr²/det < (r+1)²/r on the spatial Hessian, r = edge_thr, det > 0
#   • emission (0-indexed subpixel): sf = 2^octave_idx,
#       sigma = seed_sigma·2^((s + 0.5)/S)·sf,  x = x·sf + (sf − 1)
#   • keypoint covariance (`uncertainty = FeatureCovariance()`, uncertainty.jl):
#     `rawcov` a (6, capacity) buffer, or `nothing` — a separate specialization,
#     so the default compiles to the kernel without it. Only the refinement's
#     state is stored here, `(xi, yi, code, offS, offX, offY)` with code =
#     sl + 8·(octave_idx + 2) (0: unrefined); `_bcov!` computes the covariance
#     from it, keeping this kernel's registers (and occupancy) the default's
@inline function _extract_at!(
        out, count, dog, X, Y, Sl, Lw, W, H, ::Val{NS},
        dog_thr, edge_thr, seed_sigma, octave_idx, nb_scales, capacity, rawcov
    ) where {NS}
    @inbounds begin
        c = dog[X, Y, Sl, Lw]
        if abs(c) > 0.8f0 * dog_thr
            is_low = (Sl == 1); is_high = (Sl == NS); at_bnd = is_low || is_high
            # 26-neighbour extremum test with a lexicographic tie-break over
            # (ds, dy, dx): strict against the neighbours before the centre,
            # non-strict against those after it, so of two equal DoG values the
            # first is the extremum. A test strict against all 26
            # rejects BOTH pixels of an exact tie — a blob centred exactly
            # between two pixels then lost its refined detection and a coarser,
            # unrefined one was returned 0.5 px off.
            ismax = true; ismin = true
            for dy in -1:1, dx in -1:1
                (dx == 0 && dy == 0) && continue
                v = dog[X + dx, Y + dy, Sl, Lw]
                after = dy > 0 || (dy == 0 && dx > 0)
                ismax &= _beats(c, v, after); ismin &= _beats(v, c, after)
            end
            if ismax || ismin
                for ds in (-1, 1)
                    (ds == -1 && is_low) && continue
                    (ds == 1 && is_high) && continue
                    for dy in -1:1, dx in -1:1
                        v = dog[X + dx, Y + dy, Sl + ds, Lw]
                        ismax &= _beats(c, v, ds > 0); ismin &= _beats(v, c, ds > 0)
                    end
                end
            end
            if ismax || ismin
                xi = X; yi = Y; sl = Sl
                offX = 0.0f0; offY = 0.0f0; offS = 0.0f0
                new_val = c; valid = true; good = false
                if !at_bnd
                    for iter in 1:5
                        cc = dog[xi, yi, sl, Lw]
                        gX = 0.5f0 * (dog[xi + 1, yi, sl, Lw] - dog[xi - 1, yi, sl, Lw])
                        gY = 0.5f0 * (dog[xi, yi + 1, sl, Lw] - dog[xi, yi - 1, sl, Lw])
                        gS = 0.5f0 * (dog[xi, yi, sl + 1, Lw] - dog[xi, yi, sl - 1, Lw])
                        h22 = dog[xi + 1, yi, sl, Lw] - 2.0f0 * cc + dog[xi - 1, yi, sl, Lw]
                        h33 = dog[xi, yi + 1, sl, Lw] - 2.0f0 * cc + dog[xi, yi - 1, sl, Lw]
                        h11 = dog[xi, yi, sl + 1, Lw] - 2.0f0 * cc + dog[xi, yi, sl - 1, Lw]
                        h12 = 0.25f0 * (dog[xi + 1, yi, sl + 1, Lw] - dog[xi - 1, yi, sl + 1, Lw] - dog[xi + 1, yi, sl - 1, Lw] + dog[xi - 1, yi, sl - 1, Lw])
                        h13 = 0.25f0 * (dog[xi, yi + 1, sl + 1, Lw] - dog[xi, yi - 1, sl + 1, Lw] - dog[xi, yi + 1, sl - 1, Lw] + dog[xi, yi - 1, sl - 1, Lw])
                        h23 = 0.25f0 * (dog[xi + 1, yi + 1, sl, Lw] - dog[xi - 1, yi + 1, sl, Lw] - dog[xi + 1, yi - 1, sl, Lw] + dog[xi - 1, yi - 1, sl, Lw])
                        a11 = h22 * h33 - h23 * h23; a12 = h13 * h23 - h12 * h33; a13 = h12 * h23 - h13 * h22
                        det = h11 * a11 + h12 * a12 + h13 * a13
                        if det == 0.0f0
                            valid = false; break
                        end
                        a22 = h11 * h33 - h13 * h13; a23 = h12 * h13 - h11 * h23; a33 = h11 * h22 - h12 * h12
                        inv = 1.0f0 / det
                        offS = -inv * (a11 * gS + a12 * gX + a13 * gY)
                        offX = -inv * (a12 * gS + a22 * gX + a23 * gY)
                        offY = -inv * (a13 * gS + a23 * gX + a33 * gY)
                        (abs(offX) < 0.6f0 && abs(offY) < 0.6f0 && abs(offS) < 0.6f0) && break
                        # `det == 0` above is NOT sufficient: a denormal-but-nonzero
                        # det makes 1f0/det overflow to Inf, so the offsets come out
                        # Inf (or NaN, via Inf*0) and `round(Int, ...)` below throws
                        # InexactError INSIDE the kernel — a KernelException that
                        # takes down the whole detect call. A non-finite refinement
                        # offset means the quadratic interpolation failed, so the
                        # candidate is simply not a keypoint.
                        if !(isfinite(offS) && isfinite(offX) && isfinite(offY))
                            valid = false; break
                        end
                        if iter < 5
                            xi += round(Int, offX); yi += round(Int, offY); sl += round(Int, offS)
                            if !(2 <= xi <= W - 1 && 2 <= yi <= H - 1 && 2 <= sl <= NS - 1)
                                valid = false; break
                            end
                        end
                    end
                    good = valid && abs(offX) < 1.5f0 && abs(offY) < 1.5f0 && abs(offS) < 1.5f0
                end
                local sx::Float32, sy::Float32, ssv::Float32
                if good
                    sx = Float32(xi - 1) + offX; sy = Float32(yi - 1) + offY; ssv = Float32(sl - 1) + offS
                    gX = 0.5f0 * (dog[xi + 1, yi, sl, Lw] - dog[xi - 1, yi, sl, Lw])
                    gY = 0.5f0 * (dog[xi, yi + 1, sl, Lw] - dog[xi, yi - 1, sl, Lw])
                    gS = 0.5f0 * (dog[xi, yi, sl + 1, Lw] - dog[xi, yi, sl - 1, Lw])
                    new_val = dog[xi, yi, sl, Lw] + 0.5f0 * (gS * offS + gX * offX + gY * offY)
                else
                    sx = Float32(X - 1); sy = Float32(Y - 1); ssv = Float32(Sl - 1)
                    new_val = c; xi = X; yi = Y; sl = Sl
                end
                keep = abs(new_val) > dog_thr &&
                    0.0f0 <= sx <= Float32(W - 1) && 0.0f0 <= sy <= Float32(H - 1) &&
                    0.0f0 <= ssv <= Float32(NS - 1)
                if keep
                    exx = dog[xi + 1, yi, sl, Lw] - 2.0f0 * dog[xi, yi, sl, Lw] + dog[xi - 1, yi, sl, Lw]
                    eyy = dog[xi, yi + 1, sl, Lw] - 2.0f0 * dog[xi, yi, sl, Lw] + dog[xi, yi - 1, sl, Lw]
                    exy = 0.25f0 * (dog[xi + 1, yi + 1, sl, Lw] - dog[xi - 1, yi + 1, sl, Lw] - dog[xi + 1, yi - 1, sl, Lw] + dog[xi - 1, yi - 1, sl, Lw])
                    edet = exx * eyy - exy * exy
                    etr = exx + eyy
                    lim = (edge_thr + 1.0f0)^2 / edge_thr
                    keep = edet > 0.0f0 && (etr * etr / edet) < lim
                end
                if keep
                    idx = Atomix.@atomic count[1] += Int32(1)
                    if idx <= capacity
                        sf = exp2(Float32(octave_idx))
                        out[1, idx] = sx * sf + (sf - 1.0f0)
                        out[2, idx] = sy * sf + (sf - 1.0f0)
                        out[3, idx] = seed_sigma * exp2((ssv + 0.5f0) / Float32(nb_scales)) * sf
                        out[4, idx] = new_val
                        out[5, idx] = Float32(Lw)
                        if rawcov !== nothing
                            rawcov[1, idx] = Float32(xi); rawcov[2, idx] = Float32(yi)
                            rawcov[3, idx] = good ? Float32(sl + 8 * (octave_idx + 2)) : 0.0f0
                            rawcov[4, idx] = offS; rawcov[5, idx] = offX; rawcov[6, idx] = offY
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# `_bextract!` over a list of 32×32 tiles (`_tile_lists`), (32, NY) threads per
# tile: each thread visits its column's rows ly, ly + NY, … on every slice.
@kernel function _bextract!(
        out, count, @Const(dog),
        dog_thr::Float32, edge_thr::Float32, seed_sigma::Float32,
        octave_idx::Int32, nb_scales::Int32, capacity::Int32,
        @Const(aw), @Const(ah), rawcov, @Const(tiles), ::Val{NS}, ::Val{NY}
    ) where {NS, NY}
    lx, ly = @index(Local, NTuple)
    gi = @index(Group, Linear)
    W = size(dog, 1); H = size(dog, 2)
    ox, oy, Lw = _tile_origin(tiles, gi)
    X = ox + lx
    @inbounds if 2 <= X <= W - 1 && X <= aw[Lw]
        for Sl in 1:NS, q in 0:(32 ÷ NY - 1)
            Y = oy + ly + NY * q
            if 2 <= Y <= H - 1 && Y <= ah[Lw]
                _extract_at!(
                    out, count, dog, X, Y, Sl, Lw, W, H, Val(NS), dog_thr, edge_thr,
                    seed_sigma, octave_idx, nb_scales, capacity, rawcov
                )
            end
        end
    end
end

# GPU back-projection + boundary rejection over the pooled features, matching
# BackProjectFeatures.comp. `wp` is the per-warp table the plan builds, from
# sift_detector.c::fillFusedWarpState:
#   position (asymmetric R(+φ)·T(t)):
#     a11=cosφ, a12=-t·sinφ, a21=sinφ, a22=t·cosφ
#     a13=cosφ·xmin - sinφ·ymin,  a23=sinφ·xmin + cosφ·ymin
#   shape S=σ·A_inv (symmetric R(-φ)·diag(1,t)·R(φ)):
#     s_xx=σ(cos²φ+t·sin²φ), s_xy=σ(t-1)cosφsinφ, s_yy=σ(sin²φ+t·cos²φ)
#   boundary reject (skipped for identity): pad=K·σ·max(1,t), K=3;
#     keep iff xi,yi are pad-clear of [0,W-1]×[0,H-1].
#   covariance (`rawcov`, `featcov` both arrays or both `nothing`): (x, y) by the
#     position map's linear part A, log σ unchanged — the frame's scale is σ
#     times a per-warp constant (√det of its shape, √t).
@kernel function _bbackproject!(
        out, @Const(inp), @Const(wp),
        n::Int32, Wm1::Float32, Hm1::Float32, K::Float32, rawcov, featcov
    )
    i = @index(Global)
    @inbounds if i <= n
        w = Int(inp[5, i])
        x = inp[1, i]; y = inp[2, i]; σ = inp[3, i]
        xi = wp[1, w] * x + wp[2, w] * y + wp[3, w]
        yi = wp[4, w] * x + wp[5, w] * y + wp[6, w]
        keep = true
        if wp[11, w] == 0.0f0
            pad = K * σ * wp[10, w]
            keep = pad < xi < Wm1 - pad && pad < yi < Hm1 - pad
        end
        out[1, i] = xi; out[2, i] = yi; out[3, i] = σ
        out[4, i] = inp[4, i]
        out[5, i] = σ * wp[7, w]; out[6, i] = σ * wp[8, w]; out[7, i] = σ * wp[9, w]
        out[8, i] = Float32(w)
        out[9, i] = keep ? 1.0f0 : 0.0f0
        if featcov !== nothing
            _backproject_cov!(featcov, rawcov, i, wp[1, w], wp[2, w], wp[4, w], wp[5, w])
        end
    end
end

# The covariances of a tilt chunk's keypoints, the pooled entries `n0+1 …
# n0+n`, into `cov` from the refinement state `_bextract!` stored in `rawcov`
# (see there): the IFT covariance (uncertainty.jl) when the refinement
# converged, NaN for an unrefined keypoint. One launch for all octaves:
# `dogs`/`ctabs` are tuples of the octaves' chunk views (one array type), the
# keypoint's octave decoded from its code. `raw[5, i]` is the chunk-local warp.
@kernel function _bcov!(
        cov, @Const(rawcov), @Const(raw), dogs, ctabs, n0::Int32, n::Int32,
        o_first::Int32, nb_scales::Int32
    )
    j = @index(Global)
    @inbounds if j <= n
        i = Int(n0) + j
        code = unsafe_trunc(Int32, rawcov[3, i])
        if code == Int32(0)
            for k in 1:6
                cov[k, i] = NaN32
            end
        else
            octave_idx = (code >> 3) - Int32(2)
            oi = Int(octave_idx - o_first) + 1
            xi = unsafe_trunc(Int, rawcov[1, i]); yi = unsafe_trunc(Int, rawcov[2, i])
            sl = Int(code & Int32(7)); Lw = unsafe_trunc(Int, raw[5, i])
            sf = exp2(Float32(octave_idx))
            c = _ift_cov_at(dogs[oi], xi, yi, sl, Lw, ctabs[oi], rawcov[4, i], rawcov[5, i], rawcov[6, i], sf, nb_scales)
            for k in 1:6
                cov[k, i] = c[k]
            end
        end
    end
end

# The IFT covariance of a refined keypoint at the integer extremum (xi, yi, sl)
# of warp Lw, from the DoG Hessian there and the refinement offset.
@inline function _ift_cov_at(dog, xi, yi, sl, Lw, ctab, oS, oX, oY, sf, S)
    @inbounds begin
        cc = dog[xi, yi, sl, Lw]
        h22 = dog[xi + 1, yi, sl, Lw] - 2.0f0 * cc + dog[xi - 1, yi, sl, Lw]
        h33 = dog[xi, yi + 1, sl, Lw] - 2.0f0 * cc + dog[xi, yi - 1, sl, Lw]
        h11 = dog[xi, yi, sl + 1, Lw] - 2.0f0 * cc + dog[xi, yi, sl - 1, Lw]
        h12 = 0.25f0 * (dog[xi + 1, yi, sl + 1, Lw] - dog[xi - 1, yi, sl + 1, Lw] - dog[xi + 1, yi, sl - 1, Lw] + dog[xi - 1, yi, sl - 1, Lw])
        h13 = 0.25f0 * (dog[xi, yi + 1, sl + 1, Lw] - dog[xi, yi - 1, sl + 1, Lw] - dog[xi, yi + 1, sl - 1, Lw] + dog[xi, yi - 1, sl - 1, Lw])
        h23 = 0.25f0 * (dog[xi + 1, yi + 1, sl, Lw] - dog[xi - 1, yi + 1, sl, Lw] - dog[xi + 1, yi - 1, sl, Lw] + dog[xi - 1, yi - 1, sl, Lw])
    end
    pc = 1 + (xi & 1) + 2 * (yi & 1)
    return _ift_cov(ctab, sl, pc, Lw, (h11, h22, h33, h12, h13, h23), oS, oX, oY, sf, S)
end

# Σ ↦ blockdiag(A, 1)·Σ·blockdiag(A, 1)ᵀ on a packed (xx, xy, xℓ, yy, yℓ, ℓℓ) column.
@inline function _backproject_cov!(featcov, rawcov, i, a11, a12, a21, a22)
    @inbounds begin
        xx = rawcov[1, i]; xy = rawcov[2, i]; xl = rawcov[3, i]
        yy = rawcov[4, i]; yl = rawcov[5, i]; ll = rawcov[6, i]
        # A·Σxy
        b11 = a11 * xx + a12 * xy; b12 = a11 * xy + a12 * yy
        b21 = a21 * xx + a22 * xy; b22 = a21 * xy + a22 * yy
        featcov[1, i] = b11 * a11 + b12 * a12
        featcov[2, i] = b11 * a21 + b12 * a22
        featcov[3, i] = a11 * xl + a12 * yl
        featcov[4, i] = b21 * a21 + b22 * a22
        featcov[5, i] = a21 * xl + a22 * yl
        featcov[6, i] = ll
    end
    return nothing
end

# Polarity gate over the pooled features, run BEFORE the cross-warp splat: a
# feature whose DoG sign disagrees with the requested polarity has its validity
# flag (row 9) cleared, so it neither splats into the suppression image nor
# survives. Filtering AFTER NMS — as the port did through 0.6.0 — let light blobs
# (the white gaps between a board's dark blobs, |response| up to 1.5× the blobs')
# suppress the dark blobs beside them and then be discarded: a pure loss of ~20%
# of true blobs on dense boards. The pool is not compacted, so index-based tie
# breaking in the splat is unchanged.
@kernel function _polarity_pool!(feat, n::Int32, sign::Float32)
    i = @index(Global)
    @inbounds if i <= n && feat[9, i] == 1.0f0 && feat[4, i] * sign <= 0.0f0
        feat[9, i] = 0.0f0
    end
end

function _asift_kernels(dev)
    WG = (256, 1, 1)
    return (
        u8xy = _u8_to_xy!(dev, (256, 1)), fill3 = _fill3!(dev, WG),
        bwarp = _bwarp3!(dev, WG), bfpb = _bfpb!(dev, WG),
        tile = _btile_blur_dog!(dev, (32, _TILE_NY)), tfill = _tile_fill!(dev, (32, 8)),
        down = _bdown!(dev, WG), up = _bupsample!(dev, (32, 8)),
        ext = _bextract!(dev, (32, _TILE_NY)),
        bp = _bbackproject!(dev, (256,)), pol = _polarity_pool!(dev, (256,)),
        cov = _bcov!(dev, (256,)),
        _nms_kernels(dev)...,
    )
end

# ----- CPU launch scheduling ---------------------------------------------------
#
# KernelAbstractions' CPU backend launches a kernel either by spawning one task
# per thread and syncing on them (`CPU()`, dynamic) or through `Threads.@threads
# :static` (`CPU(static = true)`). The pipeline is a few hundred launches per
# frame, and the dynamic launch's task spawning dominated the CPU detector:
# static launches took a 12 MP frame from 7.8 s to 3.1 s, 24 MP from 10.7 s to
# 6.1 s (medians, 224 threads), the same features. So every CPU
# device's kernels launch statically — the scheduling is not a choice a caller
# makes through `device`, it only decides how a launch is spread over threads.
#
# `@threads :static` throws when another `@threads` loop is running anywhere in
# the process — always when the detector is called from inside one (a threaded
# loop over frames), and racily when one runs on another task. A launch is
# therefore static only outside a threaded region; inside one, or if the region
# check races with a loop that starts after it, the launch runs dynamically.
# The error is raised before any work is scheduled, so retrying the launch is
# safe. Both paths compute every work item identically: the results do not
# depend on the scheduling.
struct _CPUKernel{S, D}
    static::S
    dynamic::D
end

function (k::_CPUKernel)(args...; kw...)
    if !_in_threaded_region()
        try
            return k.static(args...; kw...)
        catch e
            _static_conflict(e) || rethrow()
        end
    end
    return k.dynamic(args...; kw...)
end

# Base's own test before a `@threads :static` loop.
_in_threaded_region() = ccall(:jl_in_threaded_region, Cint, ()) != 0
_static_conflict(e) = e isa ErrorException && occursin("@threads :static", e.msg)

# The kernels `build(dev)` returns, launched as above on a CPU device.
_scheduled(build, dev::KA.Backend) = build(dev)
_scheduled(build, ::KA.CPU) =
    map(_CPUKernel, build(KA.CPU(static = true)), build(KA.CPU(static = false)))

# ----- plan ------------------------------------------------------------------

"""
    AsiftPlan(W, H; device = default_device(), tilts = <IMAS-25>, seed_sigma = 1.6,
              input_blur = 0.5, nb_scales = 3, capacity = 2^20, chunk = 0,
              upsample = true)

The preallocated buffers of the ASIFT detector for `W`×`H` images on `device`
(any KernelAbstractions backend): the tilted warps, the scale-space pyramids,
the feature pool and the compiled kernels. [`asift_detect!`](@ref) runs the
detector in them without allocating on the device.

[`asift_detect`](@ref) builds and caches plans itself (see
[`set_max_cached_plans!`](@ref) and [`clear_asift_plans!`](@ref)), so most
callers never construct one. Build one to own its lifetime, or to detect in a
loop over same-size frames without the cache's lock.

- `tilts`: the tilt schedule, a vector of `(t, φ)` pairs (tilt and longitude
  in radians). The default is the 25 tilts of IMAS-25; `[(1.0, 0.0)]` alone is
  plain DoG-SIFT.
- `seed_sigma`, `input_blur`, `nb_scales`, `upsample`: the SIFT pyramid's seed
  blur, the input's assumed blur, the scales per octave and whether the first
  octave is the 2× upsampled image.
- `capacity`: the initial size of the feature pool; it grows when a frame
  produces more extrema.
- `chunk`: how many warps stay resident at once; `0` sizes the chunk from the
  device's free memory.

A plan holds device memory (several GiB for a large image under the full
schedule). It is mutable only because its feature pool grows.

```jldoctest
julia> using VisualGeometryToolkitFeatures

julia> plan = AsiftPlan(64, 48; tilts = [(1.0, 0.0)]);

julia> img = zeros(Float32, 64, 48);   # (W, H): x along the first axis

julia> asift_detect!(plan, img)
ImasFeatureKA[]
```
"""
mutable struct AsiftPlan{D, A1, A2, A3, A4, AI, AU, AU8, GT, KN}
    const dev::D
    const W::Int
    const H::Int
    const tilts::Vector{Tuple{Float64, Float64}}
    const nw::Int
    const nwc::Int                       # warps resident per chunk (≤ nw)
    const nb_octaves::Int
    const nb_scales::Int
    const o_first::Int                   # index of the finest octave: −1 when
    # upsampling (2× seed), else 0
    const inp::A2
    const inp_u8::AU8                    # (H,W) raw byte upload staging
    const rot::A3                        # (Wr_max, Hr_max, nw) batched rotate
    const canvas::A3                     # (W,H,nw)
    const gauss::Vector{A4}              # [o] :: (Wo,Ho,S+3,nw)
    const dogs::Vector{A4}               # [o] :: (Wo,Ho,S+2,nw)
    const dims::Vector{Tuple{Int, Int}}
    const pyr_taps::Vector{A1}
    const pyr_r::Vector{Int32}
    # batched warp-stage parameter tables (device)
    const rp::A2                         # (8,nw)  affine warp params + dims
    const fpt::A2                        # (5,nw)  fproj params (t,cw,ch,Wr,Hr)
    const tapm::A2                       # (17,nw) σ_aa taps per warp
    const rads::AI                       # (nw,)   σ_aa radii
    const geo::Vector{GT}
    const wp::A2                         # (11,nw) backprojection params
    const awd::Vector{AI}
    const ahd::Vector{AI}
    # per tilt chunk and octave, the level's active and fully masked 32×32
    # tiles (`_tile_lists`)
    const tiles::Vector{Vector{AI}}
    const masked::Vector{Vector{AI}}
    const seedtiles::Vector{AI}          # per chunk: octave 1's active tiles and their neighbours
    # the non-const fields are the feature pool, replaced by `_grow_pool!`
    raw::A2
    feat::A2
    const count::AI
    const splat::AU
    nmskeep::AI
    band::AI                             # (capacity,) scale band per pooled feature
    feath::Vector{Float32}
    keeph::Vector{Int32}
    bandh::Vector{Int32}
    const counth::Vector{Int32}
    const kern::KN
    const seed_sigma::Float64            # the pyramid's seed blur, and the σ the extractor reports
    # keypoint covariance (uncertainty.jl), built on the first call that asks for
    # it: per octave the (45, NS, 4, nw) IFT tables, and two (6, capacity) pool
    # buffers — or (6, 0) — that `_grow_pool!` grows with the pool
    const ctab::Vector{A4}
    rawcov::A2
    featcov::A2
    covh::Vector{Float32}
end

"""
    device_free_bytes(dev) -> Int

Free memory on `dev`, in bytes, used to size the tilt chunk and the matcher's
score blocks. The package that defines the device's backend type answers: for
CUDA that is `free_memory()` of `CUDA` (5.x) or `CUDACore` (6.x). A backend with
no known query gets `typemax(Int)`, which disables chunking, with a one-time
warning.
"""
function device_free_bytes(dev)
    dev isa KA.CPU && return Int(Sys.free_memory())
    pkg = Base.moduleroot(parentmodule(typeof(dev)))
    if nameof(pkg) in (:CUDA, :CUDACore) && isdefined(pkg, :free_memory)
        # This used `available_memory` — deprecated in CUDA 5, gone in 6 — inside
        # a bare `catch` that returned typemax: chunking switched off silently,
        # every warp stayed resident, and a 24 MP frame ran out of device memory.
        return Int(Base.invokelatest(getfield(pkg, :free_memory)))
    end
    @warn "no free-memory query for $(typeof(dev)): tilt and score chunking are disabled" maxlog = 1
    return typemax(Int)
end

# Fraction of free device memory the resident pyramids may claim. Note the
# descriptor session's arena is already allocated before the plan is built, so
# it is excluded from "free" — the reserve here only has to cover what comes
# after: the adaptation buffers and the descriptor-matching matmul (~1 GiB at
# 12 MP). Hence a fairly aggressive fraction; fewer chunks is strictly faster.
const _PYRAMID_BUDGET = 0.8

"""
    _chunk_warps(dev, nw, warp_bytes, chunk) -> Int

Warps to keep resident per chunk. `chunk > 0` is honoured verbatim (capped at
`nw`); `chunk <= 0` auto-sizes from `device_free_bytes` so the pyramids fit in
`_PYRAMID_BUDGET` of what is free right now.
"""
function _chunk_warps(dev, nw::Int, warp_bytes::Int, chunk::Integer)
    chunk > 0 && return min(Int(chunk), nw)
    free = device_free_bytes(dev)
    free == typemax(Int) && return nw
    budget = floor(Int, _PYRAMID_BUDGET * free)
    return clamp(fld(budget, max(warp_bytes, 1)), 1, nw)
end

function AsiftPlan(
        W::Integer, H::Integer;
        device::KA.Backend = default_device(),
        tilts = _imas_tilts_25(),
        seed_sigma::Real = 1.6, input_blur::Real = 0.5,
        nb_scales::Integer = 3, capacity::Integer = 1 << 20,
        chunk::Integer = 0, upsample::Bool = true
    )
    dev = device
    nw = length(tilts)
    S = Int(nb_scales)
    no = sift_nb_octaves(W, H; upsample)
    no >= 1 || throw(
        ArgumentError(
            "a $(W)×$(H) image is too small for the SIFT pyramid: min(W, H) must be " *
                "≥ 16 with upsample = true, ≥ 32 without"
        )
    )
    o_first = upsample ? -1 : 0

    geo = [(; imas_rotated_canvas(W, H, θ)..., t = t, θ = θ) for (t, θ) in tilts]
    Wr_max = maximum(g.W_rot for g in geo)
    Hr_max = maximum(g.H_rot for g in geo)

    # `>>` with a negative shift is a LEFT shift in Julia, so octave −1 comes
    # out at (2W, 2H) from the same expression.
    dims = [(W >> o, H >> o) for o in o_first:(o_first + no - 1)]

    # Resident-pyramid footprint per warp. Chunking the tilt schedule scales
    # this linearly, which is what lets the full 25-tilt IMAS schedule run on a
    # card that cannot hold all 25 pyramids at once.
    warp_bytes = 4 * (
        Wr_max * Hr_max + W * H +
            sum(d[1] * d[2] * (S + 3) for d in dims) +
            sum(d[1] * d[2] * (S + 2) for d in dims)
    )
    nwc = _chunk_warps(dev, nw, warp_bytes, chunk)

    inp = KA.allocate(dev, Float32, W, H)
    inp_u8 = KA.allocate(dev, UInt8, H, W)
    rot = KA.allocate(dev, Float32, Wr_max, Hr_max, nwc)
    canvas = KA.allocate(dev, Float32, W, H, nwc)

    gauss = [KA.allocate(dev, Float32, d[1], d[2], S + 3, nwc) for d in dims]
    dogs = [KA.allocate(dev, Float32, d[1], d[2], S + 2, nwc) for d in dims]

    # Blur bookkeeping for the 2× seed: the input's nominal blur is stated in
    # NATIVE pixels, so at octave −1 — whose pixels are half a native pixel —
    # it doubles. Only `seps[1]` depends on `input_blur`, so this is the whole
    # correction: √(1.6²−1.0²)=1.249 upsampled px instead of √(1.6²−0.5²)=1.520.
    # Leaving it at 0.5 would over-blur by ~22% and throw away much of the
    # resolution the upsample just bought. (VLFeat reaches the same number the
    # other way: it keeps σ in input units and divides Δσ by step=2^o.)
    seps = sift_sep_sigmas(seed_sigma, upsample ? 2 * input_blur : input_blur, S)
    _upload(v) = (d = KA.allocate(dev, Float32, length(v)); copyto!(d, v); d)
    pyr_taps = [_upload(gaussian_taps(s)) for s in seps]
    pyr_r = Int32[Int32(length(gaussian_taps(s)) - 1) for s in seps]
    maximum(pyr_r) <= 13 || throw(
        ArgumentError(
            "seed_sigma = $seed_sigma needs a blur tap radius of $(maximum(pyr_r)), beyond the " *
                "tiled blur's RMAX = 13 (with nb_scales = 3 and upsampling, seed_sigma must be ≲ 1.68)"
        )
    )

    # batched warp-stage tables
    rph = zeros(Float32, 8, nw)
    fpth = zeros(Float32, 5, nw)
    tapmh = zeros(Float32, 17, nw)
    radh = zeros(Int32, nw)
    for (w, g) in enumerate(geo)
        rph[1, w] = g.ca; rph[2, w] = -g.sa
        rph[3, w] = g.ca * Float32(g.xmin) - g.sa * Float32(g.ymin)
        rph[4, w] = g.sa; rph[5, w] = g.ca
        rph[6, w] = g.sa * Float32(g.xmin) + g.ca * Float32(g.ymin)
        rph[7, w] = Float32(g.W_rot); rph[8, w] = Float32(g.H_rot)
        Hs = g.t > 1 ? max(floor(Int, g.H_rot / Float32(g.t)), 1) : g.H_rot
        fpth[1, w] = Float32(g.t)
        fpth[2, w] = Float32(min(g.W_rot, W)); fpth[3, w] = Float32(min(Hs, H))
        fpth[4, w] = Float32(g.W_rot); fpth[5, w] = Float32(g.H_rot)
        tp = g.t > 1 ? sigma_aa_taps(0.8f0 * sqrt(Float32(g.t)^2 - 1.0f0)) : Float32[1.0f0]
        tapmh[1:length(tp), w] .= tp
        radh[w] = Int32(length(tp) - 1)
    end
    rp = KA.allocate(dev, Float32, 8, nw); copyto!(rp, rph)
    fpt = KA.allocate(dev, Float32, 5, nw); copyto!(fpt, fpth)
    tapm = KA.allocate(dev, Float32, 17, nw); copyto!(tapm, tapmh)
    rads = KA.allocate(dev, Int32, nw); copyto!(rads, radh)

    wph = zeros(Float32, 11, nw)
    for (w, g) in enumerate(geo)
        cf = cos(Float32(g.θ)); sf = sin(Float32(g.θ)); tf = Float32(g.t)
        wph[1, w] = cf; wph[2, w] = -tf * sf
        wph[3, w] = cf * Float32(g.xmin) - sf * Float32(g.ymin)
        wph[4, w] = sf; wph[5, w] = tf * cf
        wph[6, w] = sf * Float32(g.xmin) + cf * Float32(g.ymin)
        wph[7, w] = cf * cf + tf * sf * sf
        wph[8, w] = (tf - 1.0f0) * cf * sf
        wph[9, w] = sf * sf + tf * cf * cf
        wph[10, w] = max(1.0f0, tf)
        wph[11, w] = (g.t == 1.0 && g.θ == 0.0) ? 1.0f0 : 0.0f0
    end
    wp = KA.allocate(dev, Float32, 11, nw); copyto!(wp, wph)

    MARGIN = 64
    _iupload(v) = (d = KA.allocate(dev, Int32, length(v)); copyto!(d, v); d)
    _content_h(g) = g.t > 1 ? max(floor(Int, g.H_rot / Float32(g.t)), 1) : g.H_rot
    # o ≥ 0: content shrinks by 2^o. o = −1: it doubles (both the bound and the
    # cap), since `1 << o` / `lim >> o` are not meaningful for a negative o.
    _bound(c, lim, o) = o >= 0 ?
        Int32(min(cld(min(c, lim), 1 << o) + MARGIN, lim >> o)) :
        Int32(min((min(c, lim) << -o) + MARGIN, lim << -o))
    orange = o_first:(o_first + no - 1)
    awh = [[_bound(g.W_rot, W, o) for g in geo] for o in orange]
    ahh = [[_bound(_content_h(g), H, o) for g in geo] for o in orange]
    awd = [_iupload(v) for v in awh]
    ahd = [_iupload(v) for v in ahh]
    chunks = [(w0 + 1):min(w0 + nwc, nw) for w0 in 0:nwc:(nw - 1)]
    lists = [
        [_tile_lists(dims[o]..., view(awh[o], r), view(ahh[o], r)) for o in 1:no]
            for r in chunks
    ]
    tiles = [[_iupload(t[1]) for t in c] for c in lists]
    masked = [[_iupload(t[2]) for t in c] for c in lists]
    seedtiles = [_iupload(_dilate_tiles(c[1][1], dims[1]...)) for c in lists]

    pool = _pool_buffers(dev, Int(capacity))
    count = KA.allocate(dev, Int32, 1)
    splat = KA.allocate(dev, UInt64, W, H)

    kern = _scheduled(_asift_kernels, dev)

    return AsiftPlan(
        dev, Int(W), Int(H), collect(tilts), nw, nwc, no, S, o_first,
        inp, inp_u8, rot, canvas, gauss, dogs, dims,
        pyr_taps, pyr_r, rp, fpt, tapm, rads, geo, wp, awd, ahd, tiles, masked, seedtiles,
        pool.raw, pool.feat, count, splat, pool.nmskeep, pool.band,
        pool.feath, pool.keeph, pool.bandh, Int32[0], kern, Float64(seed_sigma),
        empty(gauss), KA.allocate(dev, Float32, 6, 0), KA.allocate(dev, Float32, 6, 0),
        Float32[]
    )
end

# The 32×32 tiles of a (Wo, Ho) level for the warps of a tilt chunk, whose
# content bounds are `aw`, `ah` (slot l = the chunk's l-th warp): those with
# their first pixel inside the bounds (`_btile_blur_dog!`, `_bextract!`) and the
# rest (`_tile_fill!`), each packed as (bx, by, l) Int32 triples, bx and by
# 0-based. The active tiles are the ones the padded-grid launch did not skip.
function _tile_lists(Wo::Integer, Ho::Integer, aw, ah)
    act = Int32[]; msk = Int32[]
    for l in eachindex(aw), by in 0:(cld(Ho, 32) - 1), bx in 0:(cld(Wo, 32) - 1)
        push!(32bx + 1 <= aw[l] && 32by + 1 <= ah[l] ? act : msk, bx, by, l)
    end
    return act, msk
end

# A packed tile list and every tile next to one of its tiles (8-neighbourhood,
# same warp slot): the tiles a blur of radius ≤ 32 reads from.
function _dilate_tiles(act::Vector{Int32}, Wo::Integer, Ho::Integer)
    nx = cld(Wo, 32); ny = cld(Ho, 32)
    seen = Set{NTuple{3, Int32}}()
    for i in 1:3:length(act), dy in -1:1, dx in -1:1
        bx = act[i] + dx; by = act[i + 1] + dy
        (0 <= bx < nx && 0 <= by < ny) && push!(seen, (Int32(bx), Int32(by), act[i + 2]))
    end
    out = Int32[]
    for t in sort!(collect(seen); by = t -> (t[3], t[2], t[1]))
        push!(out, t...)
    end
    return out
end

# Every buffer with one slot per pooled feature, `cap` slots each. The plan and
# `_grow_pool!` both build them here, so a buffer added for one grows in the other.
# `_grow_pool!` supplies `feat`, already allocated and filled.
_pool_buffers(dev, cap::Int; feat = KA.allocate(dev, Float32, 9, cap)) =
    (
    raw = KA.allocate(dev, Float32, 5, cap), feat = feat,
    nmskeep = KA.allocate(dev, Int32, cap), band = KA.allocate(dev, Int32, cap),
    feath = Vector{Float32}(undef, 9cap), keeph = Vector{Int32}(undef, cap),
    bandh = Vector{Int32}(undef, cap),
)

_capacity(plan::AsiftPlan) = size(plan.feat, 2)

# The pooled-feature buffers, regrown to hold `n` features. The capacity doubles
# until it does, so the frames that follow rarely grow it again. The first
# `keep` columns of `feat` are the earlier chunks' back-projected features and
# are carried over; nothing else is.
#
# This runs mid-frame beside the resident pyramids, so the old buffers are
# released EAGERLY (as in `_free_plan!`) and never held beside a full new pool:
# the plan first swaps in an empty pool, freeing everything but `feat`, whose old
# copy goes as soon as its columns are copied. 1 << 20 → 1 << 21 slots peaks at
# 128 MiB of device memory, not 192 MiB. A failed allocation leaves the plan with
# that empty pool (valid, and grown again by the next call), never with a
# buffer already released.
function _grow_pool!(plan::AsiftPlan, n::Integer, keep::Integer)
    dev = plan.dev
    old = _capacity(plan)
    cap = max(2old, 1)                   # 1: the empty pool a failed growth leaves
    while cap < n
        cap *= 2
    end
    KA.synchronize(dev)
    feat_old = plan.feat
    for (name, buf) in Base.pairs(_pool_buffers(dev, 0))
        name === :feat || _free_plan_buffers!(getfield(plan, name))
        setfield!(plan, name, buf)
    end
    feat = try
        f = KA.allocate(dev, Float32, 9, cap)
        keep > 0 && copyto!(f, 1, feat_old, 1, 9 * keep)
        KA.synchronize(dev)              # the copy reads `feat_old`, freed below
        f
    finally
        _free_plan_buffers!(feat_old)
    end
    for (name, buf) in Base.pairs(_pool_buffers(dev, cap; feat))
        setfield!(plan, name, buf)
    end
    # the keypoint covariances, when in use, grow alongside (the first `keep`
    # columns are the earlier chunks' back-projected covariances)
    if size(plan.featcov, 2) > 0
        fc_old = plan.featcov
        _free_plan_buffers!(plan.rawcov)
        plan.rawcov = KA.allocate(dev, Float32, 6, 0)
        plan.featcov = KA.allocate(dev, Float32, 6, 0)
        plan.covh = Float32[]
        fc = try
            f = KA.allocate(dev, Float32, 6, cap)
            keep > 0 && copyto!(f, 1, fc_old, 1, 6 * keep)
            KA.synchronize(dev)
            f
        finally
            _free_plan_buffers!(fc_old)
        end
        plan.featcov = fc
        plan.rawcov = KA.allocate(dev, Float32, 6, cap)
        plan.covh = Vector{Float32}(undef, 6cap)
    end
    @debug "asift_detect!: raw-extrema pool grown" from = old to = cap
    return plan
end

# ----- execute ---------------------------------------------------------------

"""
    asift_detect!(plan, img_xy; nms=:none, window_half=3, cutoff=3.0,
                  polarity=:none, band_octaves=0.0, band_reach=1,
                  intensity_thr=0.04, edge_thr=10.0,
                  boundary_K=3.0, uncertainty=nothing) -> Vector{ImasFeatureKA}

Full batched IMAS-25 detection in the plan's preallocated buffers. With
`nms = :mxm` or `:ellipse` the cross-warp NMS runs on-device and only the
survivors are returned; `:none` returns the full boundary-filtered pool.
`polarity` (`:dark` → response > 0, `:light` → response < 0) gates the pool
BEFORE the NMS splat, so a blob of the other sign can never suppress a kept one.
`band_octaves > 0` restricts suppression to features of comparable scale: a
feature in scale band `b = ⌊log2(√det S) / band_octaves⌋` can only be
suppressed by features in bands `b ± band_reach` (see `_band_pool!`); `0`
is the whole-pool rule. The seed blur `σ₀` — the pyramid's, and the one the
reported σ is scaled by — is the plan's (`AsiftPlan(W, H; seed_sigma)`).

`uncertainty = FeatureCovariance()` also returns each feature's covariance:
the result is then `(; features, cov)`, `cov[i]` the [`FeatureCovariance`](@ref)
covariance of `features[i]`, a `CovMat{3}` over `(x, y, log σ)`.
"""
function asift_detect!(
        plan::AsiftPlan, img_xy::AbstractMatrix{<:Real};
        nms::Symbol = :none, window_half::Integer = 3,
        cutoff::Real = 3.0, polarity::Symbol = :none,
        band_octaves::Real = 0.0, band_reach::Integer = 1,
        intensity_thr::Real = 0.04, edge_thr::Real = 10.0,
        boundary_K::Real = 3.0, uncertainty::Union{Nothing, FeatureCovariance} = nothing
    )
    polarity_sign = polarity === :none ? 0.0f0 : polarity === :dark ? 1.0f0 :
        polarity === :light ? -1.0f0 :
        throw(ArgumentError("polarity must be :none, :dark or :light"))
    nms in (:none, :mxm, :ellipse) ||
        throw(ArgumentError("nms must be :none, :mxm or :ellipse"))
    _check_nms_params(window_half, cutoff, band_octaves, band_reach)
    # No `seed_sigma` keyword: the pyramid was blurred for the plan's value and
    # the reported σ must use the same one. The plan and this call each had
    # their own default, and a non-default value on either side silently
    # disagreed with the other; then a keyword that could only repeat the
    # plan's value, or throw.
    dev = plan.dev
    W, H, nw, S = plan.W, plan.H, plan.nw, plan.nb_scales
    K = plan.kern
    if img_xy isa AbstractMatrix{UInt8}
        # H×W raw bytes: upload (4× smaller than f32) + transpose/scale on device
        size(img_xy) == (H, W) || throw(
            DimensionMismatch(
                "UInt8 input must be H×W = ($(H), $(W)), got $(size(img_xy))"
            )
        )
        copyto!(plan.inp_u8, img_xy)
        K.u8xy(plan.inp, plan.inp_u8; ndrange = (W, H))
    else
        # W×H [x,y] floats. `copyto!` copies element by element, so without this
        # check a transposed (H×W) frame of the same length was silently scrambled
        # and a smaller one left the previous frame's pixels in the buffer.
        size(img_xy) == (W, H) || throw(
            DimensionMismatch(
                "float input must be W×H = ($(W), $(H)) [x,y], got $(size(img_xy))"
            )
        )
        copyto!(plan.inp, img_xy isa AbstractMatrix{Float32} ? img_xy : Float32.(img_xy))
    end
    uncertainty === nothing || _ensure_cov!(plan)
    fill!(plan.count, Int32(0))
    npooled = 0                              # `count` as read after each chunk

    dog_thr = Float32(intensity_thr) / S
    NS = Int32(S + 2)
    nwc = plan.nwc

    # Tilts are processed `nwc` at a time; the pyramid buffers are sized for one
    # chunk and reused. Detections accumulate into the shared pool (`raw`/`feat`,
    # atomic `count`), so cross-warp NMS below still sees every warp at once and
    # the result is identical to holding all `nw` pyramids resident.
    for w0 in 0:nwc:(nw - 1)
        r = (w0 + 1):min(w0 + nwc, nw)
        c = w0 ÷ nwc + 1                     # the chunk's tile lists
        m = length(r)                        # warps in this chunk (last may be short)
        cv = view(plan.canvas, :, :, 1:m)
        rt = view(plan.rot, :, :, 1:m)

        # ---- warp stage: 3 batched launches over the chunk ----
        K.fill3(cv, 0.5f0; ndrange = (W, H, m))
        K.bwarp(
            rt, plan.inp, view(plan.rp, :, r);
            ndrange = (size(plan.rot, 1), size(plan.rot, 2), m)
        )
        K.bfpb(
            cv, rt, view(plan.tapm, :, r), view(plan.rads, r),
            view(plan.fpt, :, r); ndrange = (W, H, m)
        )

        # ---- pyramid: tiled fused h+v+DoG per level ----
        for o in 1:plan.nb_octaves
            Wo, Ho = plan.dims[o]
            g4 = view(plan.gauss[o], :, :, :, 1:m); d4 = view(plan.dogs[o], :, :, :, 1:m)
            aw = view(plan.awd[o], r); ah = view(plan.ahd[o], r)
            if o == 1   # seed blur of the canvas (writedog = 0)
                if plan.o_first < 0
                    # Octave −1: upsample the canvas 2× first. Slot 2 is free
                    # scratch here — the s=1 blur below overwrites it in full
                    # after reading only slot 1 — so this needs no extra buffer.
                    st = plan.seedtiles[c]
                    K.up(view(g4, :, :, 2, :), cv, st; ndrange = (32, 8 * (length(st) ÷ 3)))
                    _blur_level!(
                        K, plan, c, o, 1, view(g4, :, :, 1, :), view(g4, :, :, 1, :),
                        view(g4, :, :, 2, :), view(g4, :, :, 2, :), aw, ah, Int32(0)
                    )
                else
                    _blur_level!(
                        K, plan, c, o, 1, view(g4, :, :, 1, :), view(g4, :, :, 1, :), cv, cv,
                        aw, ah, Int32(0)
                    )
                end
            end
            for s in 1:(S + 2)
                _blur_level!(
                    K, plan, c, o, s + 1, view(g4, :, :, s + 1, :), view(d4, :, :, s, :),
                    view(g4, :, :, s, :), view(g4, :, :, s, :), aw, ah, Int32(1)
                )
            end
            if o < plan.nb_octaves
                Wn, Hn = plan.dims[o + 1]
                K.down(
                    view(plan.gauss[o + 1], :, :, 1, 1:m), view(g4, :, :, :, :),
                    Int32(S + 1); ndrange = (Wn, Hn, m)
                )
            end
        end

        # ---- extract: every octave's extrema into the pool ----
        # `ext` counts past the capacity but stores nothing beyond it, so an
        # overflow drops whatever is extracted last: the COARSE octaves (a dense
        # 24 MP frame had 1.3 M extrema for 262 144 slots and lost every blob
        # above σ ≈ 6 px). The DoG stacks are still resident, so on overflow the
        # pool grows and only `ext` reruns, not the blurs.
        nbefore = npooled
        for attempt in 1:2
            for o in 1:plan.nb_octaves
                Wo, Ho = plan.dims[o]
                t = plan.tiles[c][o]
                length(t) > 0 && K.ext(
                    plan.raw, plan.count, view(plan.dogs[o], :, :, :, 1:m), dog_thr,
                    Float32(edge_thr), Float32(plan.seed_sigma), Int32(o - 1 + plan.o_first),
                    Int32(S), Int32(_capacity(plan)), view(plan.awd[o], r),
                    view(plan.ahd[o], r), _ext_cov_arg(plan, uncertainty), t, Val(Int(NS)),
                    Val(_TILE_NY); ndrange = (32, _TILE_NY * (length(t) ÷ 3))
                )
            end
            copyto!(plan.counth, plan.count)
            npooled = Int(plan.counth[1])
            npooled <= _capacity(plan) && break
            attempt == 2 && error(
                "asift_detect!: the extrema pool overflowed after growth ($npooled > $(_capacity(plan)))"
            )
            _grow_pool!(plan, npooled, nbefore)
            fill!(plan.count, Int32(nbefore))
        end
        nnew = npooled - nbefore
        if nnew > 0
            _chunk_covs!(plan, uncertainty, r, nbefore, nnew)
            sl = (nbefore + 1):npooled
            K.bp(
                view(plan.feat, :, sl), view(plan.raw, :, sl), view(plan.wp, :, r),
                Int32(nnew), Float32(W - 1), Float32(H - 1), Float32(boundary_K),
                _bp_cov_args(plan, uncertainty, sl)...; ndrange = nnew
            )
            # `ext` records a CHUNK-LOCAL warp index (1:m). Downstream consumers
            # read feat[8,:] as the global warp id (it selects the tilt a feature
            # was back-projected through), so rebase it onto the global schedule.
            if w0 > 0
                wrow = view(plan.feat, 8, sl)
                wrow .+= Float32(w0)
            end
        end
    end

    # ---- pooled results: NMS below runs across every chunk's detections ----
    n = npooled
    n == 0 && return _no_features(uncertainty)
    if polarity_sign != 0.0f0
        K.pol(plan.feat, Int32(n), polarity_sign; ndrange = n)
    end

    if nms !== :none
        _pool_nms!(
            dev, K, plan.nmskeep, plan.splat, plan.feat, plan.band, plan.bandh, n,
            nms, window_half, cutoff, band_octaves, band_reach
        )
        copyto!(plan.keeph, 1, plan.nmskeep, 1, n)
    end
    KA.synchronize(dev)
    copyto!(plan.feath, 1, reshape(plan.feat, :), 1, 9n)
    fh = reshape(view(plan.feath, 1:9n), 9, n)
    # `:none` returns the valid pool (row 9), an NMS the survivors
    feats = nms === :none ? _pool_features(fh, n, view(fh, 9, :), 1.0f0) :
        _pool_features(fh, n, plan.keeph, Int32(1))
    uncertainty === nothing && return feats
    copyto!(plan.covh, 1, reshape(plan.featcov, :), 1, 6n)
    ch = reshape(view(plan.covh, 1:6n), 6, n)
    cov = nms === :none ? _pool_covs(ch, n, view(fh, 9, :), 1.0f0) :
        _pool_covs(ch, n, plan.keeph, Int32(1))
    return (features = feats, cov = cov)
end

# Level `lev` of octave `o` (tilt chunk `c`): the active tiles blurred, the
# masked ones filled.
function _blur_level!(K, plan, c, o, lev, gout, dog, inp, gprev, aw, ah, writedog)
    t = plan.tiles[c][o]; mk = plan.masked[c][o]
    length(t) > 0 && K.tile(
        gout, dog, inp, gprev, plan.pyr_taps[lev], Val(Int(plan.pyr_r[lev])), Val(_TILE_NY),
        aw, ah, 0.5f0, writedog, t; ndrange = (32, _TILE_NY * (length(t) ÷ 3))
    )
    length(mk) > 0 && K.tfill(gout, dog, 0.5f0, writedog, mk; ndrange = (32, 8 * (length(mk) ÷ 3)))
    return nothing
end

# The covariance arguments of `_bextract!` and `_bbackproject!`: `nothing` for
# the default kernels, the plan's tables and buffers for `FeatureCovariance`.
_ext_cov_arg(plan, ::Nothing) = nothing
_ext_cov_arg(plan, ::FeatureCovariance) = plan.rawcov

# The covariances of a tilt chunk's `nnew` new keypoints (after `nbefore`),
# one `_bcov!` launch while the chunk's DoG stacks are resident.
_chunk_covs!(plan, ::Nothing, r, nbefore, nnew) = nothing
function _chunk_covs!(plan, ::FeatureCovariance, r, nbefore, nnew)
    m = length(r)
    dogs = Tuple(view(d, :, :, :, 1:m) for d in plan.dogs)
    ctabs = Tuple(view(t, :, :, :, r) for t in plan.ctab)
    plan.kern.cov(
        plan.featcov, plan.rawcov, plan.raw, dogs, ctabs, Int32(nbefore), Int32(nnew),
        Int32(plan.o_first), Int32(plan.nb_scales); ndrange = nnew
    )
    return nothing
end
_bp_cov_args(plan, ::Nothing, sl) = (nothing, nothing)
# `_bcov!` left the canvas-frame covariances in `featcov`; the back-projection
# maps them in place
_bp_cov_args(plan, ::FeatureCovariance, sl) = (view(plan.featcov, :, sl), view(plan.featcov, :, sl))

_no_features(::Nothing) = ImasFeatureKA[]
_no_features(::FeatureCovariance) = (features = ImasFeatureKA[], cov = _FrameCov[])

# The IFT tables (once per plan) and the pool's covariance buffers at its capacity.
function _ensure_cov!(plan::AsiftPlan)
    if isempty(plan.ctab)
        taps = [Array(t) for t in plan.pyr_taps]
        tabs = _cov_tables(
            taps, plan.nb_scales, plan.o_first < 0, plan.nb_octaves, plan.geo, plan.W, plan.H
        )
        for t in tabs
            d = KA.allocate(plan.dev, Float32, size(t)...)
            copyto!(d, t)
            push!(plan.ctab, d)
        end
    end
    cap = _capacity(plan)
    if size(plan.featcov, 2) != cap
        _free_plan_buffers!(plan.rawcov)
        _free_plan_buffers!(plan.featcov)
        plan.rawcov = KA.allocate(plan.dev, Float32, 6, cap)
        plan.featcov = KA.allocate(plan.dev, Float32, 6, cap)
        plan.covh = Vector{Float32}(undef, 6cap)
    end
    return plan
end
