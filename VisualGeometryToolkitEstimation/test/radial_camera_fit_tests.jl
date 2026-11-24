# RadialCameraFit — the radial camera of one board (an iterative whitened fit)
# and the division lift's constraint, moved from VisualGeometryToolkitCore's
# lens_contract tests with the functions (estimator taxonomy v2). The minimal
# solve (`radial_camera_5pt` / `RadialCamera5pt`) and the RadialCamera model's
# own assertions stay in VGC.
@testitem "RadialCameraFit" setup = [DivisionBoards] begin
    using VisualGeometryToolkitCore: retcode_symbol
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra: norm, dot
    using StaticArrays: SVector, SMatrix, @SMatrix
    using Rotations: RotXYZ
    import ForwardDiff
    using VisualGeometryToolkitCore: RadialCamera, radial_rows, center, HomographyMat, radial_camera_5pt, constraint_with_jacobians
    import VisualGeometryToolkitCore
    using VisualGeometryToolkitCore.Synthetic: projective_distance
    using VisualGeometryToolkitCore: Correspondences
    using VisualGeometryToolkitEstimation: RadialCameraFit, RadialLift
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel

    @testset "RadialCameraFit recovers the radial camera" begin
        cam = board_camera(RotXYZ(BOARD_ANGLES[1]...), BOARD_TS[1])     # centred pixels: principal point at the origin
        pts = board_pixels(cam)
        P_true = RadialCamera(cam)                                          # rows 1–2 of K·[r₁ r₂ t], lens-free
        o = SVector(0.0, 0.0)                                              # these pixels are centred already
        rc = solve(EstimationProblem(RadialCamera, pts, (; center = o)), RadialCameraFit())
        Pc = rc.u
        @test Pc isa RadialCamera
        @test retcode_symbol(rc) === :solved && rc.stats.nsteps == 2 && rc.n == length(pts)
        @test rc.rms < 1.0e-6                                               # noise-free: every pixel ON its radial line
        a = vec(radial_rows(Pc))
        @test projective_distance(Pc |> radial_rows, radial_rows(P_true)) < sqrt(2.0e-9)   # up to scale and sign
        @test SMatrix{3, 3, Float64}(HomographyMat(Pc))[3, :] == SVector(0.0, 0.0, 1.0)
        @test_throws ArgumentError solve(EstimationProblem(RadialCamera, pts[1:4], (; center = o)), RadialCameraFit())
        @test_throws ArgumentError solve(EstimationProblem(RadialCamera, pts), RadialCameraFit())                # the centre is required: no silent corner pencil
        # the exactly determined case: the least-squares null vector of five rows
        # agrees with VGC's minimal solve of the same five
        P5 = radial_camera_5pt(pts[[1, 5, 8, 21, 25]])
        Pn = solve(EstimationProblem(RadialCamera, pts[[1, 5, 8, 21, 25]], (; center = o)), RadialCameraFit(; iterations = 1)).u
        @test projective_distance(radial_rows(P5), radial_rows(Pn)) < sqrt(2.0e-9)
        # the centre rides on the camera: RAW pixels about c give the SAME P, every
        # line goes through c, and the homography carries the translation
        c = SVector(931.0, -472.0)
        raw = [X => (x + c) for (X, x) in pts]
        Pr = solve(EstimationProblem(RadialCamera, raw, (; center = c)), RadialCameraFit()).u
        @test Tuple(center(Pr)) == (c[1], c[2])
        @test projective_distance(radial_rows(Pr), a) < sqrt(2.0e-9)
        for (X, x) in raw
            line = VgtkCore.project(Pr, X)
            @test abs(VgtkCore.signed_distance(line, x)) < 1.0e-6
            @test abs(VgtkCore.signed_distance(line, c)) < 1.0e-9
        end
        @test SMatrix{3, 3, Float64}(HomographyMat(Pr))[3, :] == SVector(0.0, 0.0, 1.0)
        # P is fixed only up to sign (the SVD null vector's): read Pc in Pr's sign
        @test SMatrix{3, 3, Float64}(HomographyMat(Pr))[1:2, 3] ≈ sign(dot(vec(radial_rows(Pr)), a)) * radial_rows(Pc)[:, 3] + c
        # degenerate: collinear plane points leave more than one null direction
        col = [SVector(t, 2t) => pts[1][2] .* (1 + 0.1t) for t in 1.0:5.0]
        rd = solve(EstimationProblem(RadialCamera, col, (; center = o)), RadialCameraFit(; iterations = 1))
        @test retcode_symbol(rd) === :degenerate && all(isnan, radial_rows(rd.u)) && isnan(rd.rms)
        # on noise-free data the algebraic (one-iteration) and whitened fits agree
        Pu = solve(EstimationProblem(RadialCamera, pts, (; center = o)), RadialCameraFit(; iterations = 1)).u
        @test projective_distance(radial_rows(Pu), a) < sqrt(2.0e-9)
        # per-correspondence weights are the `weights` column: a gross outlier
        # weighted to zero leaves the noise-free camera exact
        bad = copy(pts); bad[3] = bad[3][1] => bad[3][2] + SVector(40.0, -25.0)
        w = ones(length(bad)); w[3] = 0.0
        Pw = solve(EstimationProblem(RadialCamera, Correspondences(first.(bad), last.(bad); weights = w), (; center = o)), RadialCameraFit()).u
        @test projective_distance(radial_rows(Pw), a) < sqrt(2.0e-9)
        Pb = solve(EstimationProblem(RadialCamera, bad, (; center = o)), RadialCameraFit()).u
        @test projective_distance(radial_rows(Pb), a) > sqrt(2.0e-6)   # unweighted, it moves
        # a unit column is the unweighted fit
        P1 = solve(EstimationProblem(RadialCamera, Correspondences(first.(pts), last.(pts); weights = ones(length(pts))), (; center = o)), RadialCameraFit()).u
        @test radial_rows(P1) ≈ radial_rows(Pc) rtol = 1.0e-12
    end

    @testset "RadialLift: constraint derivatives vs ForwardDiff" begin
        m = RadialLift{2, Float64}(SMatrix{3, 3, Float64}(RotXYZ(0.3, -0.2, 0.1)), SVector(0.4, -0.3), 1.2, SVector(-0.3, 0.05), 2.5)
        Xl = SVector(0.2, -0.35); xl = SVector(0.31, 0.22)
        g2, Gx2, Gθ2 = constraint_with_jacobians(m, Xl, xl)
        gxl(v) = VgtkCore.constraint(m, Xl, v)
        gθl(θ) = VgtkCore.constraint(RadialLift{2, eltype(θ)}(m.R, m.t12, θ[1], SVector(θ[2], θ[3]), θ[4]), Xl, xl)
        @test ForwardDiff.jacobian(gxl, xl) ≈ Gx2 atol = 1.0e-9
        @test ForwardDiff.jacobian(gθl, SVector(m.f, m.λ[1], m.λ[2], m.t3)) ≈ Gθ2 atol = 1.0e-9
        # linear in θ: g = G_θ θ + c₀
        θ0 = SVector(m.f, m.λ[1], m.λ[2], m.t3)
        @test gθl(2θ0) ≈ Gθ2 * (2θ0) .+ (g2 .- Gθ2 * θ0) atol = 1.0e-9
        # (What the frozen whitening does to these rows is VGC's `whitening_factor`,
        # tested there on this two-row form: lens_contract_tests.jl.)
    end
end
