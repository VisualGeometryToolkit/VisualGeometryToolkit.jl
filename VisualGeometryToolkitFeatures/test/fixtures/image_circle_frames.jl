# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The synthetic frames of the image-circle detection (image_circle_tests.jl's ImageCircleFixtures and RimFixtures, each
# of which includes this file and exports its own names, and the golden records of
# ci/golden/fixtures/features_image_circle.jl): a bright disc of known centre and radius on a dark surround, and rims of
# known outer edge and knee, every noise draw from the caller's RNG. Data constructors only; include-once safe
# (functions only).
using Random: AbstractRNG, randn, rand
using Statistics: mean

# the fraction of pixel (x, y) (1-based column, row) inside the disc of centre (cx, cy) and radius ρ, COLMAP pixels,
# on a 4 × 4 subpixel grid
cover(x, y, cx, cy, ρ) = mean((x - 1 + (i - 0.5) / 4 - cx)^2 + (y - 1 + (j - 0.5) / 4 - cy)^2 <= ρ^2 for i in 1:4, j in 1:4)
# six frames of the disc, each darkened over one part of the rim
function rim_frames(rng::AbstractRNG, W, H, c, r)
    disc = [cover(u, v, c..., r) for v in 1:H, u in 1:W]
    return map(1:6) do k
        φ = 2π * (k - 1) / 6
        scene = [cover(u, v, c[1] + r * cos(φ), c[2] + r * sin(φ), 45.0) for v in 1:H, u in 1:W]   # dark over one part of the rim
        Float32.(clamp.(0.01 .+ 0.75 .* disc .* (1 .- scene) .+ 0.01 .* randn(rng, H, W), 0, 1))
    end
end
# n frames of the whole disc, with noise
function disc_frames(rng::AbstractRNG, W, H, c, r, n)
    disc = [cover(u, v, c..., r) for v in 1:H, u in 1:W]
    return [Float32.(clamp.(0.01 .+ 0.75 .* disc .+ 0.01 .* randn(rng, H, W), 0, 1)) for _ in 1:n]
end

# A rim of known outer edge `re` and inner knee `rk` for the rim profile (`detect_image_circle`'s `rim`, `rim_margin`): the
# interior a quadratic vignetting, a linear ramp from the knee to the edge (its ends rounded over `foot` px when given), a
# dark floor beyond (`re == rk`: a sharp stop, its 1-px partial band by 4 × 4 supersampling); noise σ; scene clutter (six
# dark discs inside the knee, a different set per frame); and in the first frame a saturated patch of radius
# `sat_radius` px, `sat_at` px right of the centre.
function rim_synth(rng::AbstractRNG, W, H, c, re, rk; nframes = 3, fl = 0.01, level = 0.8, vig = 0.15, σ = 0.004, sat_at = 0.6rk,
                   sat_radius = 12, foot = 0.0)
    # `foot` > 0: the ramp's knee and foot rounded over `foot` px (the ramp's clamps softened, x ↦ foot·log(1 + eˣ/ᶠᵒᵒᵗ)),
    # a soft tail into the floor like the 14 mm's
    soft(x) = foot * log1p(exp(x / foot))
    prof(r) = foot > 0 ? fl + (level - fl) * (1 - vig * (min(r, rk) / re)^2) * (soft(re - r) - soft(rk - r)) / (re - rk) :
              r >= re ? fl : r <= rk ? fl + (level - fl) * (1 - vig * (r / re)^2) :
              fl + (level - fl) * (1 - vig * (rk / re)^2) * (re - r) / (re - rk)
    base = re == rk ?
        [mean(prof(hypot(u - 1 + (i - 0.5) / 4 - c[1], v - 1 + (j - 0.5) / 4 - c[2])) for i in 1:4, j in 1:4) for v in 1:H, u in 1:W] :
        [prof(hypot(u - 0.5 - c[1], v - 0.5 - c[2])) for v in 1:H, u in 1:W]
    return map(1:nframes) do k
        img = copy(base)
        for _ in 1:6
            φ, ρ, s = 2π * rand(rng), (rk - 25) * sqrt(rand(rng)), 8 + 10rand(rng)
            cx, cy = c[1] + ρ * cos(φ), c[2] + ρ * sin(φ)
            for v in 1:H, u in 1:W
                hypot(u - 0.5 - cx, v - 0.5 - cy) < s && (img[v, u] *= 0.3)
            end
        end
        img = clamp.(img .+ σ .* randn(rng, H, W), 0, 1)
        if k == 1
            for v in 1:H, u in 1:W
                hypot(u - 0.5 - (c[1] + sat_at), v - 0.5 - c[2]) < sat_radius && (img[v, u] = 1.0)
            end
        end
        Float32.(img)
    end
end
