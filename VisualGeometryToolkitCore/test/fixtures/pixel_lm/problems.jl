# The 1000 seeded problems of the pixel-LM reference (generate.jl writes FLM's answers,
# pixel_lm_reference_tests.jl reads them): three nonlinear residuals in two bounded
# unknowns, starts inside and outside the box, maxit ∈ {3, 10, 50, 200}. Generated from a
# fixed Xoshiro seed; `pixel_lm_problems_checksum` guards the random stream.
pixel_res(a, b, c, d, e) = (p, _) -> SVector(p[1]^2 * 0.01 + a * p[2] - b, sin(0.3p[1]) * c + 0.5p[2] - d, 0.1p[1] * p[2] - e)
pixel_jac(a, b, c, d, e) = (p, _) -> SMatrix{3, 2}(0.02p[1], 0.3c * cos(0.3p[1]), 0.1p[2], a, 0.5, 0.1p[1])
function pixel_lm_problems(n = 1000)
    rng = Xoshiro(20261027)
    return map(1:n) do _
        a, b, c, d, e = randn(rng, 5) .* (1, 5, 2, 5, 3)
        minx, miny = 10rand(rng), 10rand(rng)
        maxx, maxy = minx + 1 + 19rand(rng), miny + 1 + 19rand(rng)
        x0, y0 = minx - 3 + (maxx - minx + 6) * rand(rng), miny - 3 + (maxy - miny + 6) * rand(rng)
        maxit = rand(rng, (3, 10, 50, 200))
        (; coef = (a, b, c, d, e), minx, miny, maxx, maxy, x0, y0, maxit)
    end
end
pixel_lm_problems_checksum(ps) = sum(p -> sum(p.coef) + p.minx + p.miny + p.maxx + p.maxy + p.x0 + p.y0 + p.maxit, ps)
