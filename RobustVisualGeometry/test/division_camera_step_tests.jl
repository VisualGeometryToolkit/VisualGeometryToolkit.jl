# division_camera_step — one iteration of the stacked linear focal solve via the
# division-model lift (BabelCalib's back-end), its state and its failure. Moved
# from VisualGeometryCore's zhang_intrinsics/accessors tests with the step
# (estimator taxonomy v2: the step lives beside fit_division_camera).
@testitem "division_camera_step" setup = [DivisionBoards] begin
    using Test
    using StaticArrays: SVector, SMatrix, @SMatrix
    using Rotations: RotXYZ
    using VisualGeometryCore: HomographyMat, LogicalIntrinsics, CameraModel, intrinsics, lens, focal_length,
        RadialCamera, radial_rows, Point2, calibration_matrix
    using VisualGeometryCore: Csponds, CspondNormalization, center
    using RobustVisualGeometry: division_camera_step, DivisionCameraError, DivisionCameraState, division_coeffs,
        board_sq_residuals
    import RobustVisualGeometry

    # Keyed inputs (the views' identity is the key; here the board index): the
    # radial cameras, and the views with an optional per-board vote folded into
    # each correspondence's `weights` column (the row weight √w is the same).
    _rc(Hs) = Dict(enumerate(Hs))
    _vw(boards, weights = nothing) = Dict(
        i => (weights === nothing ? b : Csponds(first.(b), last.(b); weights = fill(Float64(weights[i]), length(b))))
            for (i, b) in enumerate(boards)
    )

    # The step takes the estimator's normalization: the one strategy, about the
    # cameras' shared centre (a bare homography's is the origin).
    _center(H::HomographyMat) = SVector(0.0, 0.0)
    _center(P::RadialCamera) = SVector(center(P)[1], center(P)[2])
    _N(Hs, vw) = CspondNormalization(DivisionCameraState, vw; center = _center(first(Hs)))
    _division_state(Hs, vw; kw...) = division_camera_step(_N(Hs, vw), _rc(Hs), vw; kw...)

    # A board's radial camera and its noise-free pixels under the division lens
    # with pixel-radius coefficients `ks` (DivisionBoards).
    _rcam(R, t) = RadialCamera(board_camera(R, t))
    _board(R, t, ks = (K0,)) = board_pixels(board_camera(R, t; ks))
    # The four near boards these tests stack.
    _Rs = [RotXYZ(a...) for a in BOARD_ANGLES[1:4]]
    _ts = [
        SVector(30.0, -20.0, 400.0), SVector(-40.0, 10.0, 500.0),
        SVector(20.0, 30.0, 350.0), SVector(-10.0, -30.0, 450.0),
    ]

    # ------------------------------------------------------------------------------
    # division_camera_step — BabelCalib radial-camera focal solve under a known division k
    # ------------------------------------------------------------------------------

    # The camera one unwhitened iteration determines.
    _division_camera(Hs, boards, k = nothing; ncoef = 1, weights = nothing, N = _N(Hs, _vw(boards, weights))) =
        CameraModel(division_camera_step(N, _rc(Hs), _vw(boards, weights); k, ncoef))

    @testset "division camera (radial camera + division lift)" begin
        f0, k0 = F0, K0
        Rs, ts = _Rs, _ts
        Hs = [_rcam(R, t) for (R, t) in zip(Rs, ts)]
        boards = [_board(R, t) for (R, t) in zip(Rs, ts)]

        @testset "recovers f from one board" begin
            intr = _division_camera(Hs[1:1], boards[1:1], k0)
            @test intr isa CameraModel
            @test focal_length(intrinsics(intr))[1] ≈ f0 rtol = 1.0e-9
            @test lens(intr).radial.d[1] ≈ k0 * f0^2 rtol = 1.0e-8   # the given pixel-frame k, re-expressed in K's frame
        end

        @testset "shared f over all boards" begin
            intr = _division_camera(Hs, boards, k0; weights = [3.0, 1.0, 2.0, 1.0])
            f̂x, f̂y = focal_length(intrinsics(intr))
            @test f̂x ≈ f0 rtol = 1.0e-9
            @test f̂y == f̂x                    # square pixels by construction
        end

        @testset "rows 1–2 of H are all that matter (radial camera)" begin
            # perturb row 3 arbitrarily — the radial camera ignores it
            Hp = [
                HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(vcat(radial_rows(P), [0.3 -0.7 2.0]))))
                    for P in Hs
            ]
            intr = _division_camera(Hp, boards, k0)
            @test focal_length(intrinsics(intr))[1] ≈ f0 rtol = 1.0e-9
        end

        @testset "the state carries the pencil's centre into the camera" begin
            c = Point2(640.0, 480.0)
            Pc = [RadialCamera{Float64}(radial_rows(P), c) for P in Hs]      # the same pencils, about c
            bc = [[pd[1] => pd[2] + c for pd in b] for b in boards]             # raw pixels about c
            st = _division_state(Pc, _vw(bc); k = k0)
            @test st.center == c
            cam = CameraModel(st)
            @test calibration_matrix(cam)[1, 3] == c[1] && calibration_matrix(cam)[2, 3] == c[2]
            @test focal_length(intrinsics(cam))[1] ≈ f0 rtol = 1.0e-9
            # one focal length implies one pencil: cameras that disagree on the centre are rejected
            Pm = [Pc[1], RadialCamera{Float64}(radial_rows(Pc[2]), Point2(0.0, 0.0)), Pc[3], Pc[4]]
            @test_throws ArgumentError _division_state(Pm, _vw(bc); k = k0)
        end

        @testset "frontal boards are degenerate" begin
            Rf = [RotXYZ(0.0, 0.0, ψ) for ψ in (0.0, 0.5)]
            Hf = [_rcam(R, t) for (R, t) in zip(Rf, ts[1:2])]
            bf = [_board(R, t) for (R, t) in zip(Rf, ts[1:2])]
            @test_throws DivisionCameraError _division_camera(Hf, bf, k0)
        end

        @testset "input validation" begin
            @test_throws ArgumentError _division_camera(Hs, boards[1:2], k0)
        end
    end

    @testset "division camera joint (f, k…) — k omitted" begin
        f0 = F0
        Rs, ts = _Rs, _ts
        Hs = [_rcam(R, t) for (R, t) in zip(Rs, ts)]
        board(R, t, ks) = _board(R, t, ks)
        bs_all(k) = [board(R, t, (k,)) for (R, t) in zip(Rs, ts)]

        @testset "one coefficient" begin
            k0 = -6.4e-7
            res = _division_camera(Hs, [board(R, t, (k0,)) for (R, t) in zip(Rs, ts)])
            @test focal_length(intrinsics(res))[1] ≈ f0 rtol = 1.0e-8
            @test length(lens(res).radial.d) == 1
            @test lens(res).radial.d[1] / focal_length(intrinsics(res))[1]^2 ≈ k0 rtol = 1.0e-6
        end

        @testset "a board weighted to zero leaves the iteration without a pose, not without a solve" begin
            k0 = -6.4e-7
            bs = [board(R, t, (k0,)) for (R, t) in zip(Rs, ts)]
            weighted(ws) = Dict(i => Csponds(first.(b), last.(b); weights = w) for (i, (b, w)) in enumerate(zip(bs, ws)))
            w = [fill(1.0, length(b)) for b in bs]
            w[2] .= 0.0                                                  # the robust re-weighting rejected board 2 outright
            st = _division_state(Hs, weighted(w); ncoef = 1)
            @test st.f ≈ f0 rtol = 1.0e-6
            @test !haskey(st.poses, 2)                                   # no pose this iteration: absent, not a `nothing` slot
            @test all(haskey(st.poses, i) for i in (1, 3, 4))
            @test st.ids == [1, 2, 3, 4] && all(isnan, board_sq_residuals(st)[2])
            @test_throws DivisionCameraError division_camera_step(_N(Hs, _vw(bs)), _rc(Hs), weighted([zeros(length(b)) for b in bs]); ncoef = 1)
            # a fixed `k` leaves no extra coefficients to estimate: a larger ncoef is refused, not ignored
            @test_throws ArgumentError _division_state(Hs, _vw(bs); k = k0, ncoef = 2)
            # a SEED focal whitens the iteration (weights only): noise-free, f and k are
            # still solved exactly, from a seed well off the truth either way
            for fs in (0.5 * f0, 2.5 * f0)
                st = _division_state(Hs, _vw(bs); ncoef = 1, whiten_at = fs)
                @test st.f ≈ f0 rtol = 1.0e-6
                @test division_coeffs(st)[1] ≈ k0 rtol = 1.0e-5
                @test st.ids == [1, 2, 3, 4] && all(haskey(st.poses, i) for i in 1:4)
            end
            @test_throws ArgumentError _division_state(Hs, _vw(bs); whiten_at = 0.0)
            @test_throws ArgumentError _division_state(Hs, _vw(bs); whiten_at = NaN)
            # the same with k KNOWN: the folded f column is exact only at λ₁ = k·f,
            # so the seed camera carries the given k (at λ = 0 iteration 0 read
            # f +4.3 % at the true focal, and an extreme seed threw)
            for r in (0.5, 1.0, 2.0, 3.0)
                st = _division_state(Hs, _vw(bs); k = k0, whiten_at = r * f0)
                @test st.f ≈ f0 rtol = 1.0e-8
            end
            # a whitening state must carry the model this iteration solves: its
            # coefficient count is checked, not read out of bounds or truncated
            st1 = _division_state(Hs, _vw(bs); ncoef = 1, whiten_at = f0)
            @test_throws ArgumentError _division_state(Hs, _vw(bs); ncoef = 2, whiten_at = st1)
            @test_throws ArgumentError _division_state(Hs, _vw(bs); k = k0, whiten_at = _division_state(Hs, _vw(bs); ncoef = 2))
            # negative row weights are an argument error, not a DomainError
            @test_throws ArgumentError division_camera_step(_N(Hs, _vw(bs)), _rc(Hs), weighted([fill(-1.0, length(b)) for b in bs]); ncoef = 1)
        end

        @testset "the branch audit re-judges a view's tilt at the shared solution" begin
            k0 = -6.4e-7
            RVG = RobustVisualGeometry
            b1 = board(Rs[1], ts[1], (k0,))
            cands = RVG._radial_pose_candidates(Hs[1])
            adm = [c for c in eachindex(cands) if RVG._same_side_fraction(cands[c]..., b1) > 0.5]   # the tilt pair
            itrue = only(filter(c -> isapprox(SMatrix{3, 3}(cands[c][1]), SMatrix{3, 3}(Rs[1]); atol = 1.0e-9), adm))
            iwrong = only(filter(!=(itrue), adm))
            λs = (k0 * f0,)                                               # at the shared (f, k), in these units
            @test RVG._audit_branch(cands, iwrong, b1, f0, λs) == itrue  # the wrong tilt is moved
            @test RVG._audit_branch(cands, itrue, b1, f0, λs) == itrue   # the right one is kept
            # the audit judges at a previous iteration, so it needs one
            @test_throws ArgumentError RVG._division_camera_step(_N(Hs, _vw(bs_all(k0))), Hs, bs_all(k0); audit = true)
        end

        @testset "two coefficients (r², r⁴)" begin
            ks = (-6.4e-7, 2.0e-13)
            res = _division_camera(Hs, [board(R, t, ks) for (R, t) in zip(Rs, ts)]; ncoef = 2)
            @test focal_length(intrinsics(res))[1] ≈ f0 rtol = 1.0e-7
            @test lens(res).radial.d[1] / focal_length(intrinsics(res))[1]^2 ≈ ks[1] rtol = 1.0e-5
            @test lens(res).radial.d[2] / focal_length(intrinsics(res))[1]^4 ≈ ks[2] rtol = 1.0e-4
        end

        @test_throws ArgumentError _division_camera(Hs, [board(R, t, (-6.4e-7,)) for (R, t) in zip(Rs, ts)]; ncoef = 0)
    end

    # ------------------------------------------------------------------------------
    # the normalization — the one strategy, pooled: plane centroid, image about the centre
    # ------------------------------------------------------------------------------
    using VisualGeometryCore: HartleyNormalization, AffineMap, target_map

    @testset "the stack's normalization: required, about the centre" begin
        f0, k0 = F0, K0
        Rs, ts = _Rs[1:3], _ts[1:3]
        Hs = [_rcam(R, t) for (R, t) in zip(Rs, ts)]
        boards = [_board(R, t) for (R, t) in zip(Rs, ts)]

        N = CspondNormalization(DivisionCameraState, _vw(boards); center = SVector(0.0, 0.0))
        @test N isa CspondNormalization
        @test target_map(N)(SVector(0.0, 0.0)) ≈ SVector(0.0, 0.0) atol = 1.0e-15   # about the centre
        @test focal_length(intrinsics(_division_camera(Hs, boards; N)))[1] ≈ f0 rtol = 1.0e-8
        @test_throws MethodError division_camera_step(_rc(Hs), _vw(boards))            # no fallback: N is required

        # a centroid (translated) map on the image side would move the principal point → rejected
        plane = [pd[1] for b in boards for pd in b]; image = [pd[2] for b in boards for pd in b]
        bad = CspondNormalization(AffineMap(HartleyNormalization(), plane), AffineMap(HartleyNormalization(), image))
        @test_throws ArgumentError _division_camera(Hs, boards; N = bad)
        # any isotropic scale about the centre gives the same pixel answer
        o = SVector(0.0, 0.0)
        other = CspondNormalization(AffineMap(HartleyNormalization(; center = o), plane), AffineMap(HartleyNormalization(; center = o), image .* 3.0))
        @test focal_length(intrinsics(_division_camera(Hs, boards; N = other)))[1] ≈ f0 rtol = 1.0e-8
        # ... and so does a WHITENED iteration: a previous state whitens the rows
        # under any normalization (its poses matched to rounding, not bit for bit —
        # an exact match silently dropped every view's whitening, f 5 % off).
        # Noisy pixels, so the whitening shows in f.
        noisy = [[pd[1] => pd[2] + 0.4 * SVector(sin(7j + b), cos(5j - b)) for (j, pd) in enumerate(bd)] for (b, bd) in enumerate(boards)]
        nplane = [pd[1] for bd in noisy for pd in bd]; nimage = [pd[2] for bd in noisy for pd in bd]
        Nn = CspondNormalization(DivisionCameraState, _vw(noisy); center = o)
        No = CspondNormalization(
            AffineMap(HartleyNormalization(; center = SVector(40.0, -25.0)), nplane .* 2.0),
            AffineMap(HartleyNormalization(; center = o), nimage .* 3.0)
        )
        s1 = division_camera_step(Nn, _rc(Hs), _vw(noisy); whiten_at = 1.5f0)
        fa = division_camera_step(Nn, _rc(Hs), _vw(noisy); whiten_at = s1).f
        fb = division_camera_step(No, _rc(Hs), _vw(noisy); whiten_at = s1).f
        fplain = division_camera_step(Nn, _rc(Hs), _vw(noisy)).f
        @test fb ≈ fa rtol = 1.0e-8
        @test abs(fa - fplain) > 1.0e-6 * f0                              # the whitening does move f here
    end

    @testset "DivisionCameraState: focal_length, division_coeffs, board_sq_residuals" begin
        st = DivisionCameraState{Int}(800.0, [-0.1, 0.01], Point2(1.0, 2.0), Dict{Int, RobustVisualGeometry._BoardPose}(), Dict(1 => [1.0, 2.0]), [1])
        @test focal_length(st) == (800.0, 800.0)
        @test division_coeffs(st) == [-0.1, 0.01]
        @test board_sq_residuals(st) == Dict(1 => [1.0, 2.0])
    end
end
