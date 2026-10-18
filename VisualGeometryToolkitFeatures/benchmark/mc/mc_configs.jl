# The calibration configurations shared by the Monte Carlo scripts in benchmark/mc
# (keypoint_uncertainty_mc.jl, zeisl_comparison.jl). Needs the UncFixtures
# helpers (blob_image, gauss_field) and Printf in scope.

function configs()
    cs = []
    # 1. reference: σ 2.4–6.8 inside octaves −1…1 (away from the octave seams,
    #    σ ≈ 4.03·2^o, see 8.), light and dark, contrast 0.15–0.3
    b = [(40.3, 30.6, 2.5, 0.25), (90.7, 34.2, 2.9, -0.25), (140.45, 36.9, 3.3, 0.2),
        (36.2, 92.55, 5.0, -0.3), (98.9, 96.1, 5.7, 0.3), (160.6, 100.4, 6.5, -0.3),
        (52.35, 160.8, 2.7, -0.15), (120.15, 158.3, 6.0, 0.15), (178.5, 170.5, 3.1, 0.3),
        (190.1, 40.2, 2.4, 0.3)]
    push!(cs, (name = "blobs σ2.4–6.5, σn .01", img = blob_image(210, 230, b), blobs = b, σn = 0.01, tilts = [(1.0, 0.0)], A = nothing))
    # 2. same, higher noise
    push!(cs, (name = "blobs σ2.4–6.5, σn .03", img = blob_image(210, 230, b), blobs = b, σn = 0.03, tilts = [(1.0, 0.0)], A = nothing))
    # 3. low contrast: amplitude 0.10–0.12 at σn .03 (SNR 3–4, near the DoG threshold)
    bl = [(x, y, s, sign(a) * (0.10 + 0.02 * (k % 2))) for (k, (x, y, s, a)) in enumerate(b)]
    push!(cs, (name = "low contrast, σn .03", img = blob_image(210, 230, bl), blobs = bl, σn = 0.03, tilts = [(1.0, 0.0)], A = nothing))
    # 4. large scales, octaves 1…3
    bb = [(80.4, 80.7, 10.0, 0.3), (240.6, 82.2, 14.0, -0.3), (90.3, 250.45, 20.0, 0.3), (260.2, 255.6, 28.0, -0.3)]
    push!(cs, (name = "large σ10–28, σn .01", img = blob_image(340, 340, bb), blobs = bb, σn = 0.01, tilts = [(1.0, 0.0)], A = nothing))
    # 5. textured background (correlated clutter, ℓ = 5 px, std 0.05) + blobs
    bg = 0.5 .+ gauss_field(210, 230, 5.0, 0.05; seed = 11)
    push!(cs, (name = "textured bg, σn .01", img = blob_image(210, 230, b; background = bg), blobs = b, σn = 0.01, tilts = [(1.0, 0.0)], A = nothing))
    # 6./7. ASIFT warps: blobs isotropic in the warp's canvas, detected in that warp only
    for (t, φ) in ((2.62, π / 8), (5.18, 5π / 16))
        A = [cos(φ) -t * sin(φ); sin(φ) t * cos(φ)]
        bw = [(120.3 + 140 * ((k - 1) % 3), 120.6 + 150 * ((k - 1) ÷ 3), s, a)
            for (k, (s, a)) in enumerate(((2.2, 0.3), (3.0, -0.3), (4.0, 0.3), (2.6, -0.3), (5.0, 0.3), (3.5, -0.3)))]
        # the detection's σ is σ_w·√t (characteristic scale of S = σ_w·M_w)
        push!(cs, (name = @sprintf("warp t=%.2f φ=%.0f°, σn .01", t, rad2deg(φ)), img = blob_image(400, 520, bw; A),
            blobs = [(x, y, s * sqrt(t), a) for (x, y, s, a) in bw], σn = 0.01, tilts = [(t, φ)], A = A))
    end
    # 8. octave seams: σ where one octave's top slice meets the next one's bottom
    #    (σ = σ₀·2^(4/3)·2^o ≈ 2.0, 4.0, 8.1, 16.1), each at two sub-pixel positions
    bs = [(60.3 + 110 * (k - 1), y, s, (-1)^k * 0.3) for (k, s) in enumerate((2.0, 4.0, 8.1, 16.1)) for y in (100.6, 300.2)]
    push!(cs, (name = "octave seams σ2/4/8/16, σn .01", img = blob_image(420, 480, bs), blobs = bs, σn = 0.01, tilts = [(1.0, 0.0)], A = nothing))
    return cs
end

