# =============================================================================
# Homography augmentation — perspective strengthening + tilt jitter
# =============================================================================
#
# Training-data augmentation for a planar target seen through a known
# world→image homography: strengthen its perspective (scale the vanishing
# line) and add a small tilt in a random in-plane direction. Magnitudes are
# applied in a board-normalised frame (world extent → [-1, 1]²) so they are
# unit-invariant. Hoisted from BlobBoards' track builder (2026-09); the RNG is
# explicit so a caller can make the draw reproducible.
# =============================================================================

using Random: AbstractRNG
using LinearAlgebra: I

"""
    normalize_map(cx, cy, hx, hy) -> SMatrix{3, 3, Float64}

The world→normalised affine map sending centre `(cx, cy)` to the origin and
half-extents `(hx, hy)` to 1 (guarded against a zero extent).
"""
function normalize_map(cx, cy, hx, hy)
    hx = max(abs(hx), eps()); hy = max(abs(hy), eps())
    return SMatrix{3, 3, Float64}(
        1 / hx, 0.0, 0.0,
        0.0, 1 / hy, 0.0,
        -cx / hx, -cy / hy, 1.0
    )
end

"""
    perspective_augmentation(rng, H_wi::SMatrix{3, 3}, extent;
                             strengthen_max = 1.0, jitter_max = 0.1) -> SMatrix{3, 3, Float64}

An image→image homography that strengthens `H_wi`'s perspective and adds a
random-direction tilt. `extent = (cx, cy, hx, hy)` is the target's world
bounding box (centre + per-axis half-extent). Strengthening scales the
projective row's in-plane entries (the vanishing line) by `1 + γ`,
`γ ~ U(0, strengthen_max)`, leaving the centroid fixed; the jitter is a tilt of
magnitude `~ U(0, jitter_max)` in a uniformly random direction, in the
normalised frame. Returns the identity on any degeneracy (a singular `H_wi`),
i.e. no augmentation.
"""
function perspective_augmentation(
        rng::AbstractRNG, H_wi::SMatrix{3, 3}, extent;
        strengthen_max::Real = 1.0, jitter_max::Real = 0.1
    )
    cx, cy, hx, hy = extent
    S = normalize_map(cx, cy, hx, hy)
    # StaticArrays' `inv` of a singular 3×3 does not throw — it returns a
    # matrix of NaN/Inf — so the degeneracy check is on finiteness, not on an
    # exception (the original try/catch never fired and let NaNs through).
    Hn = H_wi * inv(S)          # normalised target → image
    Hn = Hn ./ Hn[3, 3]
    Hn_inv = inv(Hn)
    (all(isfinite, Hn) && all(isfinite, Hn_inv)) || return SMatrix{3, 3, Float64}(I)
    γ = strengthen_max * rand(rng)
    Hn_s = SMatrix{3, 3, Float64}(
        Hn[1, 1], Hn[2, 1], (1 + γ) * Hn[3, 1],
        Hn[1, 2], Hn[2, 2], (1 + γ) * Hn[3, 2],
        Hn[1, 3], Hn[2, 3], Hn[3, 3]
    )
    φ = 2π * rand(rng); m = jitter_max * rand(rng)
    a = m * cos(φ); b = m * sin(φ)
    Tj = SMatrix{3, 3, Float64}(
        1.0, 0.0, a,
        0.0, 1.0, b,
        0.0, 0.0, 1.0
    )
    H_aug = (Hn_s * Tj) * Hn_inv     # image → image (the normalising S cancels)
    return H_aug ./ H_aug[3, 3]
end
