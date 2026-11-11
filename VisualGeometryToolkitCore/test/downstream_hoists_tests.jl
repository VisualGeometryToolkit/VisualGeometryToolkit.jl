# =============================================================================
# The 0.7 wave's upstream additions for the 2026-11-06 downstream duplication
# audit ("(b)" items): each API, and where a downstream copy exists, the copy's
# body reproduced here and pinned against the Core function (same outputs on
# the same inputs), so phase 4 deletes the copies against a test.
# =============================================================================

@testitem "downstream_hoists: SCORING_TAGS, decision_threshold, FrozenNuisance, sha256_hex" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: SCORING_TAGS, RansacScoring, MsacScoring, TagRegistry, tag_of, decision_threshold,
        decision_width, WelschShape, GemanMcclureShape, CauchyShape, SmoothTLSShape, TLSShape, BarronShape, FrozenNuisance,
        update_nuisance!, update_weights!, AbstractWeightPolicy, sha256_hex
    @test SCORING_TAGS["ransac"] === RansacScoring && SCORING_TAGS["msac"] === MsacScoring
    @test tag_of(SCORING_TAGS, MsacScoring) == "msac"
    # a downstream registry of the same family merges, retired tags included (RSDR's `gau` → `mlesac`)
    struct _MlesacScoring end
    down = TagRegistry(:scoring, "Scoring", "mlesac" => _MlesacScoring; retired = ("gau" => "mlesac",))
    all_tags = merge(SCORING_TAGS, down)
    @test all_tags["mlesac"] === _MlesacScoring && all_tags["msac"] === MsacScoring
    @test_throws ArgumentError all_tags["gau"]
    @test isempty(VgtkCore.tag_rule_violations(SCORING_TAGS))

    # the inverse of decision_width, shape by shape; LODR's Welsch arm spelled √ln2 by hand
    for s in (WelschShape(), GemanMcclureShape(), CauchyShape(), SmoothTLSShape(), TLSShape(), BarronShape(0.5), BarronShape(-1.0))
        for t in (0.3, 1.0, 7.5)
            @test decision_threshold(s, decision_width(s, t)) ≈ t rtol = 1.0e-14
        end
    end
    m = 1.7
    @test decision_threshold(WelschShape(), m) == m * sqrt(log(2.0))     # accv2026's welsch_half_weight_depth, bit for bit

    p = FrozenNuisance()
    @test p isa AbstractWeightPolicy && update_nuisance!(p) === p && update_weights!(p) === p

    @test sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    @test sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    @test sha256_hex(codeunits("abc")) == sha256_hex(collect(codeunits("abc"))) == sha256_hex(IOBuffer("abc")) == sha256_hex("abc")
    mktempdir() do dir
        f = joinpath(dir, "f.bin")
        write(f, UInt8[0x00, 0xff, 0x61])
        @test open(sha256_hex, f) == sha256_hex(read(f)) == "da8509555ad27dac0593c6952d947eff96058cb6e81376f852f54ef9caf7e652"
        @test sha256_hex(f) != open(sha256_hex, f)                     # a string is hashed as its bytes, never opened
    end
end

@testitem "downstream_hoists: pool argmax and residual-matrix contract" begin
    using VisualGeometryToolkitCore.Pools: best_model_index
    # RSDR's `_best_model` (scoring_eval_common.jl), the copy Core's replaces
    rsdr_best(score, M) = (best_j = 0; best = -Inf; for j in 1:M; s = score(j); s > best && (best = s; best_j = j); end; best_j)
    for sc in ([1.0, 3.0, 3.0, 2.0], [NaN, 1.0, 2.0], [-Inf, -Inf], Float64[], [0.5, NaN, 0.7, 0.7])
        @test best_model_index(j -> sc[j], length(sc)) == rsdr_best(j -> sc[j], length(sc))
    end
    @test best_model_index(j -> [1.0, 3.0, 3.0][j], 3) == 2               # the first strictly best
    @test best_model_index(j -> -Inf, 4) == 0                              # none
    # the one difference: +Inf never wins in Core (non-finite), it does in RSDR's copy
    @test best_model_index(j -> [1.0, Inf][j], 2) == 1 && rsdr_best(j -> [1.0, Inf][j], 2) == 2
end

@testitem "downstream_hoists: difference_rows" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: difference_rows, difference_stencil
    using SparseArrays: spdiagm, sparse, SparseMatrixCSC
    using LinearAlgebra: I
    # Estimation's former `_third_difference` + kron penalty: the same DᵀD, entry for entry
    td(n) = (s = difference_stencil(3); spdiagm(n - 3, n, (k - 1 => fill(s[k], n - 3) for k in 1:4)...))
    for (ny, nx) in ((7, 9), (4, 4), (5, 12))
        Dx = kron(td(nx), sparse(I, ny, ny)); Dy = kron(sparse(I, nx, nx), td(ny))
        D = difference_rows((ny, nx), 3)
        @test D isa SparseMatrixCSC{Float64, Int}
        @test transpose(D) * D == transpose(Dx) * Dx + transpose(Dy) * Dy
    end
    # RadiometricCamera's masked `third_difference_rows` (term lists, the stencil negated): the same rows, in order
    function rc_rows(mask)
        L = LinearIndices(mask); n1, n2 = size(mask); coef = .-difference_stencil(3); rows = Vector{Tuple{Int, Float64}}[]
        for j in 1:n2, i in 1:(n1 - 3)
            all(k -> mask[i + k, j], 0:3) && push!(rows, [(L[i + k, j], coef[k + 1]) for k in 0:3])
        end
        for j in 1:(n2 - 3), i in 1:n1
            all(k -> mask[i, j + k], 0:3) && push!(rows, [(L[i, j + k], coef[k + 1]) for k in 0:3])
        end
        return rows
    end
    mask = trues(7, 9); mask[1:2, 1:3] .= false; mask[6, 8] = false
    R = rc_rows(mask)
    Rm = zeros(length(R), length(mask))
    for (r, ts) in enumerate(R), (j, c) in ts
        Rm[r, j] = c
    end
    @test Matrix(difference_rows(mask, 3)) == -Rm
    # a profile: rows along its one axis; a quadratic is annihilated at order 3
    P = difference_rows((6, 1), 3)
    @test size(P) == (3, 6) && P * [k^2 for k in 1.0:6] == zeros(3)
    @test size(difference_rows(trues(3, 3), 3), 1) == 0
    @test_throws ArgumentError difference_rows((4, 4), 0)
end

@testitem "downstream_hoists: StudentTMixture{T} and the ν update in a kernel's terms" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: StudentTMixture, inlier_posterior, posterior_weight, nu_point_terms, next_nu_from_sums,
        next_nu, lgamma_lanczos
    using SpecialFunctions: loggamma, digamma, trigamma
    # BlobBoards' adapt_ka.jl copies (d_g = 1, π = ½)
    bb_logC(ν, s2, lg) = lg((ν + 1.0) / 2.0) - lg(ν / 2.0) - 0.5 * log(ν * 3.141592653589793 * s2)
    function bb_pw(r2, ν, s2, half, log_C, log_pout)
        t = r2 / (ν * s2); log_pin = log_C - half * log1p(t)
        p = 1.0 / (1.0 + exp(log_pout - log_pin)); u = (ν + 1.0) / (ν * s2 + r2)
        return p, p * u
    end
    function bb_terms(τ, ν, ψp, ψ)
        inv_ν = 1 / ν; inv_ν2 = inv_ν * inv_ν; inv_1pτ = 1 / (1 + τ)
        dq = 0.5ψp - 0.5ψ - 0.5inv_ν - 0.5log1p(τ) + (ν + 1) * τ * inv_1pτ * 0.5 * inv_ν
        d2 = 0.5 * τ * inv_1pτ * inv_ν - 0.5 * τ * (ν + 2.0 + τ) * inv_1pτ * inv_1pτ * inv_ν2
        return dq, d2
    end
    ν, s2, l2h = 7.3, 0.81, log(2 * 12.0)
    m = StudentTMixture(ν, s2, l2h, 1)
    @test m.log_C == bb_logC(ν, s2, loggamma)
    for r2 in (0.0, 0.3, 1.7, 5.0, 40.0)
        @test (inlier_posterior(m, r2), posterior_weight(m, r2)) == bb_pw(r2, ν, s2, (ν + 1) / 2, m.log_C, -l2h)   # bit for bit
    end
    # built in T with a supplied lgamma: isbits, allocation-free, Float32 throughout
    m32 = StudentTMixture{Float32}(Float32(ν), Float32(s2), Float32(l2h), 1; lgamma = lgamma_lanczos)
    @test isbits(m32) && posterior_weight(m32, 1.0f0) isa Float32
    @test posterior_weight(m32, 1.0f0) ≈ posterior_weight(m, 1.0) rtol = 1.0e-6
    build(ν) = StudentTMixture{Float32}(ν, 1.0f0, 2.0f0, 1; lgamma = lgamma_lanczos)
    build(3.0f0); @test (@allocated build(3.0f0)) == 0
    @test StudentTMixture{Float64}(ν, s2, l2h, 2, 0.3).log_odds_out == StudentTMixture(ν, s2, l2h, 2, 0.3).log_odds_out
    # the per-point terms: BlobBoards' copy to rounding (its `ν + 2 + τ` rounds apart from `ν + d + d + d·z`)
    ψp, ψ = digamma((ν + 1) / 2), digamma(ν / 2)
    for τ in (0.01, 0.5, 3.0, 100.0)
        a, b = nu_point_terms(τ, ν, ψp, ψ), bb_terms(τ, ν, ψp, ψ)
        @test a[1] == b[1] && isapprox(a[2], b[2]; atol = 1.0e-17)
    end
    @test nu_point_terms(0.5f0, 4.0f0, 1.0f0, 0.5f0) isa NTuple{2, Float32}
    # the kernel's two-stage update reproduces `next_nu`'s one Newton update exactly
    sq = [0.2, 1.3, 0.05, 4.0, 0.9, 2.2, 30.0]; w = [0.9, 0.8, 1.0, 0.4, 0.95, 0.7, 0.01]
    ν0 = 6.0; dg = 1
    ψa, ψb = digamma((ν0 + dg) / 2), digamma(ν0 / 2)
    dQ = 0.0; d2 = 0.0; sw = 0.0
    for i in eachindex(sq)
        t = nu_point_terms(sq[i] / (ν0 * 1.1), ν0, ψa, ψb)
        dQ += w[i] * t[1]; d2 += w[i] * t[2]; sw += w[i]
    end
    ν1, _ = next_nu_from_sums(ν0, dQ, d2, sw, trigamma((ν0 + dg) / 2), trigamma(ν0 / 2))
    @test ν1 === next_nu(ν0, sq, 1.1, w, length(sq); max_newton = 1)
    # the ascent safeguard (BlobBoards' copy steps ν − dQ/d2Q whatever the sign): a convex point steps geometrically
    @test next_nu_from_sums(6.0, 1.0, 50.0, 1.0, 0.0, 0.0)[1] == 12.0
    @test next_nu_from_sums(6.0, -1.0, 50.0, 1.0, 0.0, 0.0)[1] == 3.0
    @test next_nu_from_sums(6.0, 1.0, 0.0, 1.0e-12, 0.0, 0.0) == (6.0, true)   # vanished weights: unchanged, done
end

@testitem "downstream_hoists: closed-form peak and shape gates" begin
    using VisualGeometryToolkitCore: is_peak, is_valid_shape, EllipticFrame, Point2, aspect_ratio
    using StaticArrays: SMatrix
    using LinearAlgebra: svdvals
    using Random: Xoshiro
    # BlobBoards' `_shape_ok(θ₄, θ₅, θ₆, aspect²)`: concave paraboloid with C's eigen ratio ≤ aspect²
    function bb_shape_ok(θ4, θ5, θ6, max_aspect_sq)
        tr = θ4 + θ6; dt = θ4 * θ6 - θ5 * θ5
        (tr < 0 && dt > 0) || return false
        disc = max(tr * tr - 4dt, 0.0); sd = sqrt(disc); den = tr + sd
        abs(den) > 1.0e-30 && (tr - sd) / den > max_aspect_sq && return false
        return true
    end
    rng = Xoshiro(5); mism = 0; oldmism = 0
    for _ in 1:50_000
        A = randn(rng, 2, 2); C = SMatrix{2, 2}(-(A * A') .* exp(2randn(rng)) .+ 0.3randn(rng) .* [0 1; 1 0])
        k = 1 + 9rand(rng)
        mism += (is_peak(C) && is_valid_shape(-C; max_aspect_ratio = k)) != bb_shape_ok(C[1, 1], C[1, 2], C[2, 2], k^2)
        S = -C
        is_peak(C) && (oldmism += is_valid_shape(S; max_aspect_ratio = k) !=
                                   (aspect_ratio(EllipticFrame(Point2(0.0, 0.0), S)) ≤ k))
    end
    @test mism == 0 && oldmism == 0
    f(C) = is_peak(C) && is_valid_shape(-C; max_aspect_ratio = 10.0)
    C = SMatrix{2, 2}(-2.0, 0.3, 0.3, -1.0)
    f(C); @test (@allocated f(C)) == 0
    @test is_valid_shape(SMatrix{2, 2, Float32}(1, 0, 0, 2); max_aspect_ratio = 10) && !is_peak(SMatrix{2, 2}(1.0, 0, 0, 1))
end

@testitem "downstream_hoists: spline models (accessors, rebuilders, NaN backproject)" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, SplineLens, NoncentralSplineModel, ChartSpline, px,
        backproject, lattice_spacing, with_present, with_directions, with_points, central_model, present_nodes, window, Ray3,
        directions, grid_size, window_offset
    using StaticArrays: SA, SVector, StaticVector
    model = CameraModel(300.0px, SA[320.0px, 240.0px]; lens = KannalaBrandt(-0.02))
    g = CameraModel(SplineLens, model, Size2(640, 480); grid_size = (12, 10))
    G = g.lens.backward.controls
    # RadiometricCamera's private-name copies: `_dirs`, `_full_size` / `grid_dims`, `_first_node`, `_lattice_spacing_px`
    @test directions(g) === G && size(G) == (10, 12)
    @test grid_size(g) == (12, 10) == reverse(size(G))
    @test window_offset(g, g) == (0, 0) && window_offset(window(g, 3:6, 2:5), g) == (2, 1)
    @test grid_size(window(g, 3:6, 2:5)) == (4, 4)
    @test all(lattice_spacing(g) .≈ (640 / (12 - 3), 480 / (10 - 3)))
    @test all(lattice_spacing(window(g, 3:6, 2:5)) .≈ lattice_spacing(g))     # a window keeps the lattice's spacing
    # the rebuilders keep every other part (RadiometricCamera's nine field-by-field rebuilds)
    P = trues(size(G)); P[1:2, :] .= false
    gp = with_present(g, P)
    @test present_nodes(gp) == P && all(isnan, gp.lens.backward.controls[1, 1]) && gp.lens.backward.controls[5, 5] == G[5, 5]
    @test gp.intrinsics === g.intrinsics && gp.lens.backward.area == g.lens.backward.area
    @test_throws DimensionMismatch with_present(g, trues(3, 3))
    D = map(v -> 2v, G)
    b = g.lens.backward
    nc = NoncentralSplineModel(g.intrinsics, b, ChartSpline(b.area, zero(G)); present = P)
    Pts = fill(SVector(0.001, 0.0, 0.0), size(G))
    @test with_points(nc, Pts).points.controls == Pts && with_points(nc, Pts).directions == nc.directions
    @test with_directions(nc, D).directions.controls == D && with_directions(nc, D).points == nc.points   # kept as given, not normalized
    cm = central_model(nc)
    @test cm isa CameraModel{<:Any, <:SplineLens} && cm.lens.backward == nc.directions && present_nodes(cm) == P
    ncp = with_present(nc, P)
    @test all(isnan, ncp.points.controls[1, 1]) && all(isnan, ncp.directions.controls[1, 1])
    # the lens contract's backproject: NaN, one concrete type, never `nothing`
    out = SVector(-1.0, 10.0)
    @test backproject(g, out) isa StaticVector{3, Float64} && all(isnan, backproject(g, out))
    @test isfinite(backproject(g, SVector(320.0, 240.0))[3])
    r = backproject(nc, out)
    @test r isa Ray3{Float64} && all(isnan, r.origin) && all(isnan, r.direction)
    probe = SVector(320.0, 10.0)                                             # a pixel whose window reads an absent node
    @test all(isnan, backproject(gp, probe)) && all(isfinite, backproject(g, probe))
    f(m, u) = backproject(m, u)
    f(g, out); @test (@allocated f(g, out)) == 0
    @test VgtkCore.primal_float(VgtkCore.ForwardDiff.Dual{Nothing, VgtkCore.ForwardDiff.Dual{Nothing, Float32, 1}, 2}) === Float32
    @test VgtkCore.primal_float(Float64) === Float64
end

@testitem "downstream_hoists: convert_eltype to any Real (Duals)" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, EuclideanMap, BoundedCamera, convert_eltype, project, px,
        parameter_vector, CameraFrame, WorldFrame, relabel
    using ForwardDiff: Dual, value
    using StaticArrays: SA, SVector, SMatrix
    using Rotations: RotMatrix, RotXYZ, QuatRotation
    D = Dual{Nothing, Float64, 2}
    cam = CameraModel(300.0px, SA[500.0px, 400.0px]; lens = KannalaBrandt(-0.02, 0.003, 0.0, 0.0))
    cd = convert_eltype(D, cam)
    @test eltype(cd.intrinsics.K) === D && eltype(parameter_vector(cd.lens)) === D
    X = SVector(0.1, 0.2, 1.0)
    @test value.(project(cd, D.(X))) ≈ project(cam, X)
    E = relabel(CameraFrame, WorldFrame, EuclideanMap(RotMatrix(RotXYZ(0.1, 0.2, 0.3)), SVector(1.0, 2.0, 3.0)))
    Ed = convert_eltype(D, E)
    # RadiometricCamera's `_pose_eltype`: the same numbers, entry by entry; frames and representation kept
    rc = (SMatrix{3, 3, D, 9}(E.R), SVector{3, D}(E.t))
    @test SMatrix{3, 3}(Ed.R) == rc[1] && Ed.t == rc[2] && isbits(Ed)
    @test typeof(Ed).parameters[5:6] == typeof(E).parameters[5:6]
    @test convert_eltype(Float32, EuclideanMap(QuatRotation(RotXYZ(0.1, 0.2, 0.3)), SVector(1.0, 2.0, 3.0))).R isa QuatRotation{Float32}
    bc = BoundedCamera(cam, VgtkCore.Size2(1000, 800))
    @test eltype(convert_eltype(D, bc).camera.intrinsics.K) === D
end

@testitem "downstream_hoists: equidistant_chart, pixel_centre, distortion_map(RadialHomography), lens_label, shape_matrix(Disc)" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: equidistant_chart, pixel_centre, distortion_map, RadialHomography, HomographyMat, project,
        lens_label, KannalaBrandt, Division, Lens, DoubleSphere, Pinhole, Disc, OrientedDisc, Point2, shape_matrix, Equidistant,
        Perspective, Tilt, Tangential, Radial, Even, Additive
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: normalize, norm, I, dot
    # the chart: a ray at angle θ from the axis lands at radius θ; the axis at the origin; right-handed
    for a in (SVector(0.0, 0.0, 1.0), normalize(SVector(0.3, -0.2, 1.0)), SVector(0.0, 1.0, 0.0))
        ch = equidistant_chart(a)
        @test norm(ch(a)) < 1.0e-12
        d = normalize(a + 0.4 * normalize(SVector(1.0, 0.5, -0.2) - dot(SVector(1.0, 0.5, -0.2), a) * a))
        @test norm(ch(d)) ≈ acos(dot(d, a)) rtol = 1.0e-12
        R = ch.t2.linear
        @test R * R' ≈ I && R[3, :] ≈ a
    end
    @test equidistant_chart(SVector(0.0, 0.0, 1.0))(SVector(0.1, 0.2, 1.0)) ≈ Lens(Equidistant())(SVector(0.1, 0.2, 1.0))
    # the index → COLMAP centre (PanoramaStitching's three copies: (I[2] − ½, I[1] − ½))
    @test pixel_centre(CartesianIndex(3, 7)) === SVector(6.5, 2.5)
    @test pixel_centre(Float32, CartesianIndex(1, 1)) === SVector(0.5f0, 0.5f0)
    # BlobBoards' `_radial_distort` (project of the centred lift) is the map, bit for bit in the field
    rh = RadialHomography(HomographyMat(SMatrix{3, 3}(1.0I)), -2.0e-7, Point2(640.0, 480.0))
    Dm = distortion_map(rh)
    bb(u) = project(rh, SVector(u[1] - rh.center[1], u[2] - rh.center[2], 1.0))
    for u in (SVector(100.0, 50.0), SVector(640.0, 480.0), SVector(1200.0, 900.0))
        @test SVector{2}(Dm(u)) == SVector{2}(bb(u))
        @test inv(Dm)(Dm(u)) ≈ u rtol = 1.0e-12
    end
    # display names by the tag rule (RadiometricCamera's table spelled Division{2} "rational")
    @test lens_label(Pinhole()) == "pinhole" && lens_label(KannalaBrandt(0.1, 0.0, 0.0, 0.0)) == "kannala_brandt"
    @test lens_label(Division(0.1, 0.2)) == "division" && lens_label(Lens(DoubleSphere(0.1, 0.5))) == "double_sphere"
    @test lens_label(Lens(Perspective(), Tilt(0.01, 0.0) ∘ Additive(Radial{Even}((0.1,), ()), Tangential(0.0, 0.0)))) ==
        "perspective_tilt_radial_tangential"
    @test shape_matrix(Disc(Point2(1.0, 2.0), 1.5)) == SMatrix{2, 2}(2.25, 0.0, 0.0, 2.25)
    @test shape_matrix(OrientedDisc(Point2(1.0, 2.0), 1.5, 0.3)) == shape_matrix(VgtkCore.EllipticFrame(Disc(Point2(1.0, 2.0), 1.5)))
end

@testitem "downstream_hoists: convert_field with the key, intersects" begin
    using VisualGeometryToolkitCore: thrown_message
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: convert_field, intersects, Circle, Point2, Rect2
    using LinearAlgebra: dot, norm
    @test convert_field(NTuple{2, Float64}, [1, 2], "a.b") === (1.0, 2.0)
    @test convert_field(Vector{Int}, [1, 2.0], "a.v") == [1, 2]
    @test occursin("`a.t` needs 2 values, got 3", thrown_message(() -> convert_field(NTuple{2, Float64}, [1, 2, 3], "a.t")))
    @test occursin("`a.v` = \"x\" is not a Vector{Float64} (an array)", thrown_message(() -> convert_field(Vector{Float64}, "x", "a.v")))
    @test occursin("`a.f` = \"x\" is not a Float64", thrown_message(() -> convert_field(Float64, "x", "a.f")))
    @test occursin("`a.d` is a value, not a table", thrown_message(() -> convert_field(Float64, Dict("k" => 1), "a.d")))
    @test occursin("`a.i`", thrown_message(() -> convert_field(Int, 1.5, "a.i")))             # an InexactError, named
    # a static-vector field converts by its constructor, as the unkeyed form does
    using StaticArrays: SVector
    @test convert_field(SVector{2, Float64}, [1, 2], "a.s") === SVector(1.0, 2.0) === convert_field(SVector{2, Float64}, [1, 2])
    # (the type prints module-qualified where StaticArrays is not loaded in Main: match its two parts)
    m3 = thrown_message(() -> convert_field(SVector{2, Float64}, [1, 2, 3], "a.s"))
    @test startswith(m3, "`a.s` = [1, 2, 3] is not a ") && endswith(m3, "SVector{2, Float64}")
    mab = thrown_message(() -> convert_field(SVector{2, Float64}, "ab", "a.s"))
    @test startswith(mab, "`a.s` = \"ab\" is not a ") && endswith(mab, "SVector{2, Float64} (an array)")
    # a Vararg tuple takes any length, element by element; its errors name the key
    @test convert_field(Tuple{Vararg{Float64}}, [1, 2.5, 3], "a.u") === (1.0, 2.5, 3.0)
    @test convert_field(Tuple{Vararg{Float64}}, [], "a.u") === ()
    @test convert_field(Tuple{Int, Vararg{Float64}}, [1, 2], "a.u") === (1, 2.0)
    @test occursin("`a.u` = \"x\" is not a Float64", thrown_message(() -> convert_field(Tuple{Vararg{Float64}}, [1, "x"], "a.u")))
    @test occursin("`a.u` = \"x\" is not a Tuple{Vararg{Float64}} (an array)", thrown_message(() -> convert_field(Tuple{Vararg{Float64}}, "x", "a.u")))
    @test occursin("`a.u`", thrown_message(() -> convert_field(Tuple{Int, Vararg{Float64}}, [], "a.u")))  # too short, named
    # read_table's default converter on a struct with both
    struct _HoistFields
        s::SVector{2, Float64}
        u::Tuple{Vararg{Float64}}
    end
    kw = VgtkCore.read_table(_HoistFields, Dict("s" => [1, 2], "u" => [3, 4, 5]))
    @test Dict(kw) == Dict(:s => SVector(1.0, 2.0), :u => (3.0, 4.0, 5.0))
    # BlobBoards' `_overlaps` (squares, strict): the same predicate
    bb_cc(a, b) = (d = a.center - b.center; R = a.r + b.r; dot(d, d) < R * R)
    bb_cr(c, r) = (lo = VgtkCore.GeometryBasics.origin(r); hi = lo + VgtkCore.GeometryBasics.widths(r);
        dx = c.center[1] - clamp(c.center[1], lo[1], hi[1]); dy = c.center[2] - clamp(c.center[2], lo[2], hi[2]); dx * dx + dy * dy < c.r * c.r)
    c1 = Circle(Point2(0.0, 0.0), 1.0)
    for c2 in (Circle(Point2(2.0, 0.0), 1.0), Circle(Point2(1.5, 0.5), 1.0), Circle(Point2(0.1, 0.1), 0.2), Circle(Point2(0.3, 0.4), 4.5))
        @test intersects(c1, c2) == bb_cc(c1, c2)
    end
    # circle–circle compares distances, Core's form before 0.7: at an ulp from
    # tangency it keeps its decision where the squared form flips
    ca = Circle(Point2(1.5526026771305754, 0.8274757264628023), 0.9667711192602672)
    cb = Circle(Point2(2.772163837984363, -0.49276547206988286), 0.8305508461555176)
    @test intersects(ca, cb) == (norm(ca.center - cb.center) < ca.r + cb.r) == false
    @test bb_cc(ca, cb)
    # PanoramaStitching's phase-4 pair: the 2×2 Sylvester test is public, and a similarity maps the whole plane to itself
    @test :is_positive_definite in names(VgtkCore) && Base.ispublic(VgtkCore, :is_positive_definite)
    using StaticArrays: SMatrix
    using Rotations: RotMatrix
    @test VgtkCore.is_positive_definite(SMatrix{2, 2}(2.0, 0.5, 0.5, 1.0)) &&
        !VgtkCore.is_positive_definite(SMatrix{2, 2}(1.0, 2.0, 2.0, 1.0))
    S = VgtkCore.SimilarityMap(2.0, RotMatrix{2}(0.3), SVector(1.0, -2.0))
    @test S(VgtkCore.WholePlane()) === VgtkCore.WholePlane()
    for r in (Rect2(1.0, -0.5, 1.0, 1.0), Rect2(0.9, -0.5, 1.0, 1.0), Rect2(-0.2, -0.2, 0.4, 0.4), Rect2(0.8, 0.8, 1.0, 1.0))
        @test intersects(c1, r) == bb_cr(c1, r) == intersects(r, c1)
    end
    @test !intersects(c1, Rect2(1.0, -0.5, 1.0, 1.0))                            # tangent: no overlap
end

@testitem "downstream_hoists: the parameter protocol of vignettings, light and blur; restrict(ch, Z)" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: TensorProductVignetting, RadialVignetting, LogVignetting, EnvironmentLight,
        SphericalHarmonicMap, GaussianBlur, parameter_vector, parameter_names, parameter_bounds, with_parameters,
        ParameterManifold, retract, recenter, retract_jacobian, local_coordinates, restrict, tangent_size,
        control_points, coefficients, pole_regular_basis, Rect2, cov
    using StaticArrays: SVector
    using LinearAlgebra: Diagonal, qr
    v = TensorProductVignetting((5, 4), Rect2(0.0, 0.0, 100.0, 80.0))
    θ = parameter_vector(v)
    @test length(θ) == 20 && θ == vec(control_points(v)) && parameter_names(v)[[1, 20]] == (:c1, :c20)
    θ2 = θ .+ (1:20) ./ 100
    @test parameter_vector(with_parameters(v, θ2)) == θ2
    @test_throws ArgumentError with_parameters(v, θ[1:3])
    ch = ParameterManifold(v)
    @test tangent_size(ch) == 20 && parameter_vector(retract(ch, θ2 - θ)) == θ2
    @test local_coordinates(ch, with_parameters(v, θ2)) ≈ θ2 - θ
    lv = LogVignetting(v)
    @test parameter_vector(with_parameters(lv, θ2)) == θ2
    # a radial profile on its pole-regular subspace: vec(P) = vec(P₀) + Z y
    rv = RadialVignetting(6, SVector(50.0, 40.0), 60.0)
    Z = pole_regular_basis(rv)
    rch = restrict(ParameterManifold(rv), Z)
    y = SVector{size(Z, 2)}(range(0.1, 0.5; length = size(Z, 2)))
    @test tangent_size(rch) == size(Z, 2)
    @test parameter_vector(retract(rch, y)) ≈ parameter_vector(rv) + Z * y
    @test retract_jacobian(rch) ≈ Z
    @test local_coordinates(rch, retract(rch, y)) ≈ y
    @test parameter_vector(VgtkCore.model(recenter(rch, y))) ≈ parameter_vector(rv) + Z * y
    @test_throws ArgumentError restrict(ParameterManifold(rv), zeros(3, 2))
    @test_throws ArgumentError restrict(ParameterManifold(rv), hcat(Z[:, 1], Z[:, 1]))  # dependent columns
    # the cached factorization gives the normal-equation values, on a large basis too
    v_off = parameter_vector(rv) .+ (1:length(parameter_vector(rv))) ./ 50
    vfull = local_coordinates(ParameterManifold(rv), with_parameters(rv, v_off))
    @test local_coordinates(rch, with_parameters(rv, v_off)) ≈ (Z' * Z) \ (Z' * vfull)
    big = TensorProductVignetting((12, 12), Rect2(0.0, 0.0, 100.0, 80.0))
    Zb = Matrix(qr(reshape(sin.(1:144 * 30), 144, 30)).Q)[:, 1:30] .+ 0.01 .* reshape(cos.(1:144 * 30), 144, 30)
    tch = restrict(ParameterManifold(big), Zb)
    @test tch.basis isa Matrix{Float64} && tangent_size(tch) == 30
    yb = collect(range(-0.2, 0.3; length = 30))
    @test parameter_vector(retract(tch, yb)) ≈ parameter_vector(big) + Zb * yb
    @test local_coordinates(tch, retract(tch, yb)) ≈ yb
    θb = parameter_vector(big) .+ cos.(1:144)
    vb = local_coordinates(ParameterManifold(big), with_parameters(big, θb))
    @test local_coordinates(tch, with_parameters(big, θb)) ≈ (Zb' * Zb) \ (Zb' * vb)
    @test local_coordinates(recenter(tch, yb), retract(tch, yb)) ≈ zeros(30) atol = 1e-12
    # the light: its SH coefficients, column-major, centre and radius fixed
    L = EnvironmentLight(SphericalHarmonicMap(1.0, 1; channels = 2), SVector(0.0, 0.0, 0.0), 5.0)
    a = parameter_vector(L)
    @test length(a) == 8 && parameter_names(L)[1:3] == (:a1_1, :a2_1, :a1_2)
    L2 = with_parameters(L, a .+ 1)
    @test vec(coefficients(L2)) == a .+ 1 && L2.radius == 5.0
    @test local_coordinates(ParameterManifold(L), L2) ≈ ones(8)
    # the blur: the free entries of its storage, variances bounded below by 0 (a log chart)
    b = GaussianBlur(0.7, 1.2)
    @test parameter_names(b) == (:variance_x, :variance_y) && parameter_vector(b) ≈ [0.49, 1.44]
    @test parameter_bounds(b)[1] == [0.0, 0.0]
    bch = ParameterManifold(b)
    @test parameter_vector(retract(bch, SVector(log(2.0), 0.0))) ≈ [0.98, 1.44]
    @test parameter_names(GaussianBlur(0.5)) == (:variance,)
    full = GaussianBlur(SVector(2.0, 0.3, 0.3, 1.0) |> x -> VgtkCore.SMatrix{2, 2}(x))
    @test parameter_names(full) == (:variance_x, :covariance_xy, :variance_y) && parameter_bounds(full)[1][2] == -Inf
    @test cov(with_parameters(full, [1.0, 0.1, 2.0])) == VgtkCore.CovMat(VgtkCore.SMatrix{2, 2}(1.0, 0.1, 0.1, 2.0))
    @test_throws ArgumentError with_parameters(full, [1.0, 3.0, 1.0])            # not PSD: the constructor's refusal
    @test_throws ArgumentError parameter_vector(GaussianBlur(0.02 * VgtkCore.mm))  # a print's blur: strip first
end

@testitem "downstream_hoists: Eliminable" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Eliminable, SphereManifold, nlls_features
    import NLLSsolver
    using StaticArrays: SVector
    p = Eliminable{true}(SVector(1.0, 2.0, 3.0))
    @test NLLSsolver.nvars(p) == NLLSsolver.static(3)
    q = NLLSsolver.update(p, [0.0, 0.1, 0.2, 0.3], 2)
    @test q isa Eliminable{true} && q.value == SVector(1.1, 2.2, 3.3)
    s = Eliminable{false}(SphereManifold(SVector(0.0, 0.0, 1.0)))
    @test NLLSsolver.nvars(s) == NLLSsolver.static(2) && NLLSsolver.update(s, [0.1, 0.0]) isa Eliminable{false}
    if nlls_features().schur
        @test NLLSsolver.iseliminable(typeof(p)) == NLLSsolver.static(true)
    end
end
