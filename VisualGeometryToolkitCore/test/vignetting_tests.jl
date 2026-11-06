# The vignetting fields (cameras/vignetting.jl): evaluation against a brute-force Cox–de Boor recursion, the coefficient
# Jacobian, the knots against BSplineKit, the pole's regular subspace, the support protocol, eltype conversion, the NLLSsolver
# variable, and the device against the host.
@testitem "vignetting" begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: dot
    using VisualGeometryToolkitCore: TensorProductVignetting, RadialVignetting, coefficient_jacobian, control_points, with_control_points,
        n_control_points, knot_vectors, spline_degree, center, field_radius, radial_coordinate, pole_regular_basis,
        clamped_knots, convert_eltype, pointwise!, Rect, BoundaryRule, boundary_rule_jacobian, free_nodes, Circle, Point2, LogVignetting
    import BSplineKit
    import ForwardDiff

    # The textbook recursion (Piegl & Tiller, eq. 2.5), written independently of `bspline_basis`: 0/0 := 0, the last non-empty
    # span closed on the right, x clamped onto the knots.
    function N_ref(k, i, p, x)
        if p == 0
            last = findlast(j -> k[j] < k[j + 1], 1:(length(k) - 1))
            return (k[i] <= x < k[i + 1] || (x == k[end] && i == last)) ? 1.0 : 0.0
        end
        a = k[i + p] > k[i] ? (x - k[i]) / (k[i + p] - k[i]) * N_ref(k, i, p - 1, x) : 0.0
        b = k[i + p + 1] > k[i + 1] ? (k[i + p + 1] - x) / (k[i + p + 1] - k[i + 1]) * N_ref(k, i + 1, p - 1, x) : 0.0
        return a + b
    end
    basis_ref(k, p, x) = (xc = clamp(x, k[1], k[end]); [N_ref(k, i, p, xc) for i in 1:(length(k) - p - 1)])

    @testset "clamped_knots: BSplineKit's" begin
        k = clamped_knots(7, 3, 0, 4000)
        @test length(k) == 11 && all(k[1:4] .== 0) && all(k[8:11] .== 4000)
        @test collect(k) == collect(BSplineKit.knots(BSplineKit.BSplineBasis(BSplineKit.BSplineOrder(4), range(0.0, 4000.0; length = 5))))
        @test_throws ArgumentError clamped_knots(3, 3, 0, 1)
    end

    W, H = 640.0, 480.0
    P2 = [1.0 + 0.1 * sin(i) * cos(2j) for i in 1:7, j in 1:5]
    t = with_control_points(TensorProductVignetting((7, 5), Rect(0, 0, W, H); degree = 2), P2)
    r = with_control_points(RadialVignetting(9, SVector(300.0, 250.0), 400.0), [1.0 - 0.05 * (i - 1)^1.3 for i in 1:9])
    pts = [SVector(x, y) for x in range(-20.0, W + 20; length = 23) for y in range(-15.0, H + 15; length = 17)]

    @testset "evaluation equals the brute-force B-spline" begin
        kx, ky = knot_vectors(t)
        for u in pts
            Nx = basis_ref(kx, 2, u[1]); My = basis_ref(ky, 2, u[2])
            @test t(u) ≈ sum(Nx[i] * My[j] * P2[i, j] for i in 1:7, j in 1:5) atol = 1e-13
            ρ = hypot(u[1] - 300, u[2] - 250) / 400
            @test r(u) ≈ dot(basis_ref(knot_vectors(r)[1], 3, ρ), vec(control_points(r))) atol = 1e-13
        end
        @test spline_degree(t) == (2, 2) && spline_degree(r) == (3,)
        @test center(t) == SVector(W / 2, H / 2) && center(r) == SVector(300.0, 250.0) && field_radius(r) == 400.0
        @test radial_coordinate(r, SVector(300.0, 650.0)) == 1.0
        @test TensorProductVignetting((6, 6), Rect(10, 20, 100, 50))(SVector(30.0, 40.0)) ≈ 1.0      # the partition of unity
    end

    @testset "linear in the control points: the coefficient Jacobian" begin
        for v in (t, r), u in pts[1:40:end]
            J = coefficient_jacobian(v, u)
            @test length(J) == n_control_points(v)
            @test dot(J, vec(control_points(v))) ≈ v(u) atol = 1e-14
            @test sum(J) ≈ 1.0                                                          # a partition of unity
        end
        for v in (t, r), u in pts                                                       # the sparse form: every nonzero, nothing else
            J = coefficient_jacobian(v, u); idx, w = VisualGeometryToolkitCore.active_coefficients(v, u)
            @test all(J[collect(idx)] .== collect(w))
            @test all(iszero, J[setdiff(1:length(J), collect(idx))])
        end
        u = SVector(123.0, 321.0); J = coefficient_jacobian(t, u)
        e = zeros(7, 5); e[3, 2] = 1.0                                                  # column-major: i + (j − 1)·nx
        @test with_control_points(t, e)(u) ≈ J[3 + 7] atol = 1e-15
        @test with_control_points(t, vec(P2)) == t
    end

    @testset "the pole's regular subspace: V′(0) = 0" begin
        Z = pole_regular_basis(r)
        @test size(Z) == (9, 8) && Z[1, 1] == Z[2, 1] == 1.0
        y = [0.3 * cos(j) for j in 1:8]
        v = with_control_points(r, Z * y)                                               # a profile in the regular subspace
        c = center(v); δ = SVector(1e-4 * 400, 0.0)
        growth(v) = (v(c + 2δ) - v(c)) / (v(c + δ) - v(c))                              # 4 for V − V(c) = O(ρ²), 2 for a cone O(ρ)
        @test abs(growth(v) - 4) < 1e-2
        @test abs(growth(with_control_points(r, [1.0; 0.8; ones(7)])) - 2) < 1e-2       # untied: a cone at the centre
    end

    @testset "the support protocol: a BoundaryRule over an imaged region" begin
        region = Rect(200.0, 150.0, 240.0, 180.0)
        @test 0 < length(free_nodes(t, region)) < n_control_points(t)
        rule = BoundaryRule(t, region)
        @test size(boundary_rule_jacobian(rule), 1) == n_control_points(t)
    end

    @testset "convert_eltype and isbits" begin
        for v in (t, r)
            @test isbitstype(typeof(v))
            v32 = convert_eltype(Float32, v)
            @test eltype(control_points(v32)) === Float32
            @test v32(SVector(200.0f0, 100.0f0)) ≈ v(SVector(200.0, 100.0)) rtol = 1e-6
        end
    end

    @testset "LogVignetting: exp of the spline, its Jacobian by the chain rule" begin
        lr = LogVignetting(with_control_points(r, [-0.02 * (i - 1)^1.5 for i in 1:9]))
        lt = LogVignetting(with_control_points(t, 0.1 .* (P2 .- 1)))
        for (lv, s) in ((lr, VisualGeometryToolkitCore.log_spline(lr)), (lt, VisualGeometryToolkitCore.log_spline(lt))), u in pts[1:25:end]
            @test lv(u) == exp(s(u)) && lv(u, nothing) == lv(u)
            J = coefficient_jacobian(lv, u); P0 = vec(collect(control_points(lv))); h = 1e-6
            fd = [(lv |> x -> with_control_points(x, P0 .+ h .* (1:length(P0) .== q)))(u) - lv(u) for q in eachindex(P0)] ./ h
            @test maximum(abs, J .- fd) < 1e-5                              # the Jacobian against finite differences
            idx, w = VisualGeometryToolkitCore.active_coefficients(lv, u)
            @test all(J[collect(idx)] .≈ collect(w))
        end
        @test pole_regular_basis(lr) == pole_regular_basis(r) && center(lr) == center(r)
        # one basis evaluation per row, bit for bit the two-evaluation chain rule
        for (lv, s) in ((lr, VisualGeometryToolkitCore.log_spline(lr)), (lt, VisualGeometryToolkitCore.log_spline(lt))), u in pts[1:7:end]
            @test coefficient_jacobian(lv, u) == exp(s(u)) .* coefficient_jacobian(s, u)
            idx, w = VisualGeometryToolkitCore.active_coefficients(s, u); V = exp(s(u))
            @test VisualGeometryToolkitCore.active_coefficients(lv, u) == (idx, map(x -> V * x, w))
        end
        @test knot_vectors(lt) == knot_vectors(t) && n_control_points(lt) == n_control_points(t)
        @test isbitstype(typeof(lr))
        @test convert_eltype(Float32, lr)(SVector(200.0f0, 100.0f0)) ≈ lr(SVector(200.0, 100.0)) rtol = 1e-6
        @test 0 < length(free_nodes(lt, Rect(200.0, 150.0, 240.0, 180.0))) < n_control_points(lt)
    end

    @testset "with_control_points promotes, never narrows the knots, centre or radius" begin
        ti = with_control_points(t, ones(Int, 7, 5))                    # Int control points: no InexactError on the knots
        @test eltype(control_points(ti)) === Float64 && knot_vectors(ti) == knot_vectors(t)
        t32 = with_control_points(t, Float32.(P2))                      # Float32: the knots keep their Float64 values
        @test knot_vectors(t32) == knot_vectors(t) && eltype(knot_vectors(t32)[1]) === Float64
        r32 = with_control_points(RadialVignetting(9, SVector(300.1, 250.3), 400.7), ones(Float32, 9))
        @test center(r32) == SVector(300.1, 250.3) && field_radius(r32) == 400.7
        ri = with_control_points(r, ones(Int, 9))
        @test eltype(control_points(ri)) === Float64 && center(ri) == center(r)
    end

    @testset "support of a Dual field: the knots' values" begin
        D = ForwardDiff.Dual{Nothing, Float64, 1}
        td = with_control_points(t, D.(P2))                              # Dual control points: a Dual field, Dual knots
        @test eltype(knot_vectors(td)[1]) === D
        region = Rect(200.0, 150.0, 240.0, 180.0)
        @test free_nodes(td, region) == free_nodes(t, region)
        @test VisualGeometryToolkitCore.support(td, CartesianIndex(3, 2)) == VisualGeometryToolkitCore.support(t, CartesianIndex(3, 2))
    end

    @testset "RadialVignetting: pixel quantities, a positive radius" begin
        px, mm = VisualGeometryToolkitCore.px, VisualGeometryToolkitCore.mm
        rq = RadialVignetting(9, SVector(300.0px, 250.0px), 400.0px)
        @test center(rq) == SVector(300.0, 250.0) && field_radius(rq) == 400.0
        @test rq(SVector(310.0, 260.0)) == RadialVignetting(9, SVector(300.0, 250.0), 400.0)(SVector(310.0, 260.0))
        @test_throws VisualGeometryToolkitCore.FlexUnits.DimensionError RadialVignetting(9, SVector(300.0, 250.0), 4.0mm)
        @test_throws ArgumentError RadialVignetting(9, SVector(300.0, 250.0), 0.0)
        @test_throws ArgumentError RadialVignetting(9, SVector(300.0, 250.0), -1.0)
        @test_throws ArgumentError RadialVignetting(9, SVector(300.0, 250.0), Inf)
    end

    @testset "GPU: device == host (availability-gated)" begin
        # StaticArrays' products are `muladd`s, which the compiler may contract to FMAs differently per call site and backend:
        # the batch agrees with the scalar calls to rounding, not bit for bit
        host_t = [t(u) for u in pts]; host_r = [r(u) for u in pts]
        lg = LogVignetting(with_control_points(r, [-0.02 * (i - 1)^1.5 for i in 1:9])); host_l = [lg(u) for u in pts]
        out = similar(host_t); pointwise!(t, out, pts); @test out ≈ host_t rtol = 1e-14
        cuda = try
            mod = Base.require(Main, :CUDA)
            (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
        catch
            nothing
        end
        if cuda === nothing
            @test_skip cuda !== nothing      # no functional GPU here: recorded as Broken, not as a success
        else
            CuArray = getproperty(cuda, :CuArray)
            dpts = Base.invokelatest(CuArray, pts)
            for (v, host) in ((t, host_t), (r, host_r), (lg, host_l))
                d = Base.invokelatest(similar, dpts, Float64)
                Base.invokelatest(pointwise!, v, d, dpts)
                @test Base.invokelatest(Array, d) ≈ host rtol = 1e-14
            end
        end
    end
end

# The one tag registry of the vignetting fields: the tag rule, the reverse lookup on a value, the retired tag.
@testitem "vignetting: VIGNETTING_TAGS" begin
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: VIGNETTING_TAGS, TensorProductVignetting, RadialVignetting, LogVignetting, Rect, tag_of,
        tag_rule_violations

    @test collect(keys(VIGNETTING_TAGS)) == ["tensor_product", "radial", "log"]
    @test isempty(tag_rule_violations(VIGNETTING_TAGS))
    t = TensorProductVignetting((6, 6), Rect(10, 20, 100, 50))
    r = RadialVignetting(8, SVector(300.0, 250.0), 400.0)
    @test VIGNETTING_TAGS["tensor_product"] === TensorProductVignetting && tag_of(VIGNETTING_TAGS, t) == "tensor_product"
    @test tag_of(VIGNETTING_TAGS, r) == "radial" && tag_of(VIGNETTING_TAGS, LogVignetting(r)) == "log"
    err = try VIGNETTING_TAGS["tensor"]; nothing catch e; e end
    @test err isa ArgumentError && occursin("tensor_product", err.msg)
end
