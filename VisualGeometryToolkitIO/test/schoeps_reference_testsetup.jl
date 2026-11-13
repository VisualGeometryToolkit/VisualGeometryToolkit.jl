# SchoepsReference: the reference implementation's grid conventions (Schöps et al.), shared by
# schoeps_tests.jl and spline_lens_record_tests.jl.

@testsetup module SchoepsReference
    export schoeps_raw_grid, schoeps_surface, schoeps_grid_coordinate, FIXTURES

    const FIXTURES = joinpath(@__DIR__, "fixtures", "schoeps")

    # the `gw × gh` grid under `tag`, as the reference stores it: entry y·gw + x
    function schoeps_raw_grid(path, tag, gw, gh)
        txt = read(path, String)
        body = match(Regex("^$(tag)\\s*:\\s*\\[(.*?)\\]", "ms"), txt).captures[1]
        nums = parse.(Float64, split(body, r"[,\s]+"; keepempty = false))
        return [nums[(3 * (y * gw + x) + 1):(3 * (y * gw + x) + 3)] for x in 0:(gw - 1), y in 0:(gh - 1)]   # [x+1, y+1]
    end
    # util.h's uniform cubic weights at the local coordinate x ∈ [3, 4)
    bfac(x) = (-1 / 6 * (x - 4)^3, 1 / 2 * x^3 - 11 / 2 * x^2 + 39 / 2 * x - 131 / 6, -1 / 2 * x^3 + 5 * x^2 - 16 * x + 50 / 3, 1 / 6 * (x - 3)^3)
    # EvalUniformCubicBSplineSurface at the grid coordinate (gx, gy) ∈ [1, gw − 2] × [1, gh − 2]
    function schoeps_surface(cp, gx, gy)
        x = gx + 2; y = gy + 2
        ix = min(floor(Int, x), size(cp, 1) - 1); iy = min(floor(Int, y), size(cp, 2) - 1)
        ax, bx, cx, dx = bfac(x - (ix - 3))
        at(i, j) = cp[i + 1, j + 1]
        rows = [ax * at(ix - 3, iy - 4 + k) + bx * at(ix - 2, iy - 4 + k) + cx * at(ix - 1, iy - 4 + k) + dx * at(ix, iy - 4 + k) for k in 1:4]
        ay, by, cy, dy = bfac(y - (iy - 3))
        return ay * rows[1] + by * rows[2] + cy * rows[3] + dy * rows[4]
    end
    # pixel (corner convention) → grid coordinate, the reference's PixelCornerConvToGridPoint
    schoeps_grid_coordinate(px, py; gw, gh, cminx, cminy, cmaxx, cmaxy) =
        (1 + (gw - 3) * (px - cminx) / (cmaxx + 1 - cminx), 1 + (gh - 3) * (py - cminy) / (cmaxy + 1 - cminy))
end
