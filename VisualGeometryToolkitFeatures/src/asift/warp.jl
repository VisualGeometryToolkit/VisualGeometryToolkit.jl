# =============================================================================
# IMAS warp geometry — rotated canvas + σ_aa taps.
# =============================================================================
#
# Shared by the batched warp stage in driver.jl (`_bwarp3!`, `_bfpb!`), which
# ports the fused IMAS forward chain (per warp):
#   input --AffineWarp(R(-φ))--> rotated  --σ_aa vertical blur--> rotated'
#         --FprojBilinearY(t)--> tilted (W_rot × ⌊H_rot/t⌋)
#
# Exact geometry (all Float32):
#   corners (nx-1,0),(0,ny-1),(nx-1,ny-1) plus (0,0); rx=floor(ca·cx+sa·cy),
#   ry=floor(-sa·cx+ca·cy); xmin/xmax/… include origin (init 0);
#   W_rot=xmax-xmin+1, H_rot=ymax-ymin+1, H_sub=⌊H_rot/t⌋.
#   AffineWarp inverse (out→in): a11=ca,a12=-sa,a13=ca·xmin-sa·ymin,
#     a21=sa,a22=ca,a23=sa·xmin+ca·ymin, fill=0.5.
#   σ_aa=0.8·√(t²-1) (vertical); Fproj samples (x+0.5, t·y+0.5), bg=0.0.
#
# Images are (W,H)[x,y] Float32, matching the oracle's readback layout.
# =============================================================================

"Rotated-canvas geometry for a θ rotation of an nx×ny image (Float32, matches C)."
function imas_rotated_canvas(nx::Integer, ny::Integer, θ::Real)
    ca = cos(Float32(θ)); sa = sin(Float32(θ))
    xn = 0; xm = 0; yn = 0; ym = 0
    cx = (Int(nx) - 1, 0, Int(nx) - 1)
    cy = (0, Int(ny) - 1, Int(ny) - 1)
    for i in 1:3
        rx = floor(Int, ca * Float32(cx[i]) + sa * Float32(cy[i]))
        ry = floor(Int, -sa * Float32(cx[i]) + ca * Float32(cy[i]))
        xn = min(xn, rx); xm = max(xm, rx); yn = min(yn, ry); ym = max(ym, ry)
    end
    W_rot = xm - xn + 1; H_rot = ym - yn + 1
    return (; ca, sa, xmin = xn, ymin = yn, W_rot, H_rot)
end

# ----- σ_aa tap builder (PreBlur1D / GaussBlur1DStorage: cap radius at 16) ----
sigma_aa_taps(σ::Real) = _unit_sum_taps(_gaussian_profile(Float32(σ), 4; max_radius = 16))
