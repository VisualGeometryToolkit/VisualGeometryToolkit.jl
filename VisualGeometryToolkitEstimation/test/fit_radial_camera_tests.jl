# fit_radial_camera — the radial camera of one board (an iterative whitened fit)
# and the division lift's constraint, moved from VisualGeometryToolkitCore's
# lens_contract tests with the functions (estimator taxonomy v2). The minimal
# solve (`radial_camera_5pt` / `RadialCamera5pt`) and the RadialCamera model's
# own assertions stay in VGC.
@testitem "fit_radial_camera" setup = [DivisionBoards] begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra: norm, dot
    using StaticArrays: SVector, SMatrix, @SMatrix
    using Rotations: RotXYZ
    import ForwardDiff
    using VisualGeometryToolkitCore: RadialCamera, radial_rows, center, HomographyMat, NoWhitening, SampsonWhitening,
        whitened_rows, radial_camera_5pt, constraint_with_jacobians, estimate, retcode, iterations, rms
    import VisualGeometryToolkitCore
    using VisualGeometryToolkitCore.Synthetic: projective_distance
    using VisualGeometryToolkitCore: Csponds
    using VisualGeometryToolkitEstimation: fit_radial_camera, RadialLift

    @testset "fit_radial_camera recovers the radial camera" begin
        cam = board_camera(RotXYZ(BOARD_ANGLES[1]...), BOARD_TS[1])     # centred pixels: principal point at the origin
        pts = board_pixels(cam)
        P_true = RadialCamera(cam)                                          # rows 1–2 of K·[r₁ r₂ t], lens-free
        o = SVector(0.0, 0.0)                                              # these pixels are centred already
        rc = fit_radial_camera(pts; center = o)
        Pc = estimate(rc)
        @test Pc isa RadialCamera
        @test retcode(rc) === :solved && iterations(rc) == 2 && rc.n == length(pts)
        @test rms(rc) < 1.0e-6                                               # noise-free: every pixel ON its radial line
        a = vec(radial_rows(Pc))
        @test projective_distance(Pc |> radial_rows, radial_rows(P_true)) < sqrt(2.0e-9)   # up to scale and sign
        @test SMatrix{3, 3, Float64}(HomographyMat(Pc))[3, :] == SVector(0.0, 0.0, 1.0)
        @test_throws ArgumentError fit_radial_camera(pts[1:4]; center = o)
        @test_throws UndefKeywordError fit_radial_camera(pts)                # the centre is required: no silent corner pencil
        # the exactly determined case: the least-squares null vector of five rows
        # agrees with VGC's minimal solve of the same five
        P5 = radial_camera_5pt(pts[[1, 5, 8, 21, 25]])
        Pn = estimate(fit_radial_camera(pts[[1, 5, 8, 21, 25]]; center = o, whitening = NoWhitening(), iterations = 1))
        @test projective_distance(radial_rows(P5), radial_rows(Pn)) < sqrt(2.0e-9)
        # the centre rides on the camera: RAW pixels about c give the SAME P, every
        # line goes through c, and the homography carries the translation
        c = SVector(931.0, -472.0)
        raw = [X => (x + c) for (X, x) in pts]
        Pr = estimate(fit_radial_camera(raw; center = c))
        @test Tuple(center(Pr)) == (c[1], c[2])
        @test projective_distance(radial_rows(Pr), a) < sqrt(2.0e-9)
        for (X, x) in raw
            line = VgtkCore.project(Pr, X)
            @test abs(VgtkCore.distance(line, x)) < 1.0e-6
            @test abs(VgtkCore.distance(line, c)) < 1.0e-9
        end
        @test SMatrix{3, 3, Float64}(HomographyMat(Pr))[3, :] == SVector(0.0, 0.0, 1.0)
        # P is fixed only up to sign (the SVD null vector's): read Pc in Pr's sign
        @test SMatrix{3, 3, Float64}(HomographyMat(Pr))[1:2, 3] ≈ sign(dot(vec(radial_rows(Pr)), a)) * radial_rows(Pc)[:, 3] + c
        # degenerate: collinear plane points leave more than one null direction
        col = [SVector(t, 2t) => pts[1][2] .* (1 + 0.1t) for t in 1.0:5.0]
        rd = fit_radial_camera(col; center = o, whitening = NoWhitening(), iterations = 1)
        @test retcode(rd) === :degenerate && all(isnan, radial_rows(estimate(rd))) && isnan(rms(rd))
        # the whitening policy is a parameter; on noise-free data all currencies agree
        Pu = estimate(fit_radial_camera(pts; center = o, whitening = NoWhitening()))
        @test projective_distance(radial_rows(Pu), a) < sqrt(2.0e-9)
        # per-correspondence weights are the `weights` column: a gross outlier
        # weighted to zero leaves the noise-free camera exact
        bad = copy(pts); bad[3] = bad[3][1] => bad[3][2] + SVector(40.0, -25.0)
        w = ones(length(bad)); w[3] = 0.0
        Pw = estimate(fit_radial_camera(Csponds(first.(bad), last.(bad); weights = w); center = o))
        @test projective_distance(radial_rows(Pw), a) < sqrt(2.0e-9)
        Pb = estimate(fit_radial_camera(bad; center = o))
        @test projective_distance(radial_rows(Pb), a) > sqrt(2.0e-6)   # unweighted, it moves
        # a unit column is the unweighted fit
        P1 = estimate(fit_radial_camera(Csponds(first.(pts), last.(pts); weights = ones(length(pts))); center = o))
        @test radial_rows(P1) ≈ radial_rows(Pc) rtol = 1.0e-12
    end

    @testset "RadialLift: constraint derivatives vs ForwardDiff" begin
        m = RadialLift{2, Float64}(SMatrix{3, 3, Float64}(RotXYZ(0.3, -0.2, 0.1)), SVector(0.4, -0.3), 1.2, SVector(-0.3, 0.05), 2.5)
        Xl = SVector(0.2, -0.35); xl = SVector(0.31, 0.22)
        g2, Gx2, Gθ2 = constraint_with_jacobians(m, Xl, xl)
        gxl(v) = constraint_with_jacobians(m, Xl, v)[1]
        gθl(θ) = constraint_with_jacobians(RadialLift{2, eltype(θ)}(m.R, m.t12, θ[1], SVector(θ[2], θ[3]), θ[4]), Xl, xl)[1]
        @test ForwardDiff.jacobian(gxl, xl) ≈ Gx2 atol = 1.0e-9
        @test ForwardDiff.jacobian(gθl, SVector(m.f, m.λ[1], m.λ[2], m.t3)) ≈ Gθ2 atol = 1.0e-9
        # linear in θ: g = G_θ θ + c₀
        θ0 = SVector(m.f, m.λ[1], m.λ[2], m.t3)
        @test gθl(2θ0) ≈ Gθ2 * (2θ0) .+ (g2 .- Gθ2 * θ0) atol = 1.0e-9
        # (What the whitening policies do to these rows is VGC's `whitened_rows`,
        # tested there on this two-row form: lens_contract_tests.jl.)
    end
end
