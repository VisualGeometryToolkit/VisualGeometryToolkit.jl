# =============================================================================
# Targets, posed targets, rigs, cameras and captures carry units the way their poses
# do: a physical value holds FlexUnits' base-form lengths (metres), its plain
# form is `ustrip(u, x)`, and `with_unit(x, u)` reads a plain value as being in
# `u`. A unitful value meeting plain numbers is FlexUnits' `DimensionError`.
#
# The unitful values are built as a file reader builds them,
# `parse(Float64, "…e-3") * m` — the decimal shift of a millimetre text.
# FlexUnits' `ustrip(mm, ·)` divides by the unit's scale, so the plain form it
# returns is within one ulp of the millimetre value (the decimal-exact
# conversion of a file's text is the IO layer's `read_length`/`write_length`).
# =============================================================================

@testitem "units carriers" setup = [UnitHelpers] begin
    using Test
    using StaticArrays, Rotations, LinearAlgebra, Statistics
    using FlexUnits: Quantity
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: EuclideanMap, m, mm, ustrip, DimensionError, translation, rotation,
        relabel, WorldFrame, CameraFrame, ViewFrame, TargetFrame, BodyFrame, Uncertain, CovMat, geom,
        has_units, with_unit, PointTarget, PlanarTarget, PosedTarget, Rig, PosedCamera, BoundedCamera, CameraModel,
        CameraCalibrationMat, Trajectory, TargetPoses, TargetPoints3D, Size2, project, pose, extrinsics, points

    maps_close(A, B) = rotation(A) == rotation(B) && within_ulp(translation(A), translation(B))
    pose_close(A::EuclideanMap, B::EuclideanMap) = maps_close(A, B)
    pose_close(A::Uncertain, B::Uncertain) =
        maps_close(geom(A), geom(B)) && within_ulp(SMatrix{6, 6}(cov(A)), SMatrix{6, 6}(cov(B)))
    point_close(a::StaticVector, b::StaticVector) = within_ulp(a, b)
    point_close(a::Uncertain, b::Uncertain) = within_ulp(geom(a), geom(b)) && within_ulp(cov(a), cov(b))

    R = RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.3))
    t_mm = SVector(62.8, 8029.3853, 62.746506)
    E_mm = EuclideanMap(R, t_mm)
    E_u = EuclideanMap(R, from_mm(t_mm))
    Σ6 = CovMat(SMatrix{6, 6, Float64}(Diagonal(SVector(1.0e-4, 2.0e-4, 3.0e-4, 0.5, 0.7, 0.9))))
    pts2_mm = [SVector(62.8, 13.9), SVector(-4.2, 120.5), SVector(0.0, 7.3)]
    pts3_mm = [SVector(62.8, 13.9, 1.1), SVector(-4.2, 120.5, 0.0), SVector(0.0, 7.3, -3.3)]

    @testset "PointTarget" begin
        T_mm = PointTarget(pts3_mm)
        T_u = PointTarget(from_mm.(pts3_mm))
        @test has_units(T_u) && !has_units(T_mm)
        @test eltype(eltype(points(T_u))) <: Quantity
        @test all(point_close.(points(ustrip(mm, with_unit(T_mm, mm))), points(T_mm)))
        @test all(point_close.(points(ustrip(mm, T_u)), points(T_mm)))
        @test_throws ArgumentError ustrip(mm, T_mm)
        @test_throws ArgumentError with_unit(T_u, mm)
        # an estimated constellation: the covariance moves with the unit (mm² ↔ m²)
        U_mm = PointTarget([Uncertain(p, CovMat{3}(σ = 0.1)) for p in pts3_mm])
        U_u = with_unit(U_mm, mm)
        @test has_units(U_u)
        @test cov(points(U_u)[1]) ≈ CovMat{3}(σ = 1.0e-4)
        @test all(point_close.(points(ustrip(mm, U_u)), points(U_mm)))
        # all plain or all lengths
        @test_throws DimensionError PointTarget(Any[pts3_mm[1], from_mm(pts3_mm[2])])
    end

    @testset "PlanarTarget" begin
        T_mm = PlanarTarget(pts2_mm)
        T_u = PlanarTarget(from_mm.(pts2_mm))
        @test has_units(T_u) && !has_units(T_mm)
        @test eltype(eltype(points(T_u))) <: Quantity
        @test all(point_close.(points(ustrip(mm, with_unit(T_mm, mm))), points(T_mm)))
        @test all(point_close.(points(ustrip(mm, T_u)), points(T_mm)))
        @test_throws ArgumentError ustrip(mm, T_mm)
        @test_throws ArgumentError with_unit(T_u, mm)
        # an unresolved target holds no lengths: neither form, returned unchanged
        D = PlanarTarget(:board)
        @test !has_units(D)
        @test ustrip(mm, D) == D && with_unit(D, mm) == D
    end

    @testset "PosedTarget and target rigs: world points carry units" begin
        M_mm = PosedTarget(PlanarTarget(pts2_mm), E_mm)
        M_u = PosedTarget(PlanarTarget(from_mm.(pts2_mm)), E_u)
        @test has_units(M_u) && !has_units(M_mm)
        M_back = ustrip(mm, with_unit(M_mm, mm))
        @test maps_close(pose(M_back), pose(M_mm))
        @test all(point_close.(points(M_back), points(M_mm)))
        @test maps_close(pose(ustrip(mm, M_u)), E_mm)
        # units move a posed target as an AbstractTarget: it stays posed, its model moves with it
        @test M_u isa VgtkCore.AbstractTarget && VgtkCore.is_posed(ustrip(mm, M_u)) && VgtkCore.is_posed(with_unit(M_mm, mm))
        @test VgtkCore.model(ustrip(mm, M_u)) == ustrip(mm, VgtkCore.model(M_u)) && has_units(VgtkCore.model(with_unit(M_mm, mm)))
        # a unitful pose on plain points, or the reverse, never promotes
        @test_throws DimensionError PosedTarget(PlanarTarget(pts2_mm), E_u)
        @test_throws DimensionError PosedTarget(PlanarTarget(from_mm.(pts2_mm)), E_mm)
        # an UNRESOLVED target is held to the form of its calibration's and
        # board's lengths, without applying the calibration: a plain pose on a
        # physical calibration or board is refused as on physical points
        R0 = rotation(E_mm)
        cal_u = VgtkCore.Affine(SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0), SVector(1mm, 2mm))
        board_u = VgtkCore.Board(Size2(width = 100mm, height = 100mm))
        for tgt in (PlanarTarget(:x; calibration = cal_u, board = board_u), PlanarTarget(:x; calibration = cal_u),
                PlanarTarget(:x; board = board_u))
            @test has_units(tgt) || tgt.board !== nothing
            @test_throws DimensionError PosedTarget(tgt, EuclideanMap(R0, SVector(0, 0, 500.0)))
            @test has_units(PosedTarget(tgt, EuclideanMap(R0, SVector(0, 0, 500.0) .* mm)))
        end
        @test !has_units(PosedTarget(PlanarTarget(:x), EuclideanMap(R0, SVector(0, 0, 500.0))))   # no lengths: either form
        @test has_units(PosedTarget(PlanarTarget(:x), EuclideanMap(R0, SVector(0, 0, 500.0) .* mm)))
        # an uncertain pose: the covariance's translation block moves too
        MU_mm = PosedTarget(PointTarget(pts3_mm), Uncertain(E_mm, Σ6))
        MU_back = ustrip(mm, with_unit(MU_mm, mm))
        @test pose_close(pose(MU_back), pose(MU_mm))

        rig_u = Rig{WorldFrame}(Dict(:a => M_u, :b => PosedTarget(PlanarTarget(:b), E_u)))
        rig_mm = Rig{WorldFrame}(Dict(:a => M_mm, :b => PosedTarget(PlanarTarget(:b), E_mm)))
        @test has_units(rig_u) && !has_units(rig_mm)
        W = points(rig_u, :a)
        @test eltype(eltype(W)) <: Quantity
        @test all(isapprox.(map(p -> ustrip.(mm, p), W), points(rig_mm, :a); rtol = 1.0e-12))
        @test eltype(valtype(points(rig_u))) == eltype(W)
        back = ustrip(mm, with_unit(rig_mm, mm))
        @test maps_close(geom(pose(back, :a)), geom(pose(rig_mm, :a)))
        @test all(isapprox.(points(back, :a), points(rig_mm, :a); rtol = 1.0e-14))
        @test maps_close(geom(pose(ustrip(mm, rig_u), :b)), geom(pose(rig_mm, :b)))
        # the gauge check reads a unitful identity
        ident_u = EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 0.0) .* m)
        @test Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(PlanarTarget(from_mm.(pts2_mm)), ident_u))) isa Rig
        @test_throws ArgumentError Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(PlanarTarget(from_mm.(pts2_mm)), E_u)))
        # a rig holds one form: a physical member beside a plain one is refused
        @test_throws DimensionError Rig{WorldFrame}(Dict(:a => M_u, :b => M_mm))
    end

    model = CameraModel(CameraCalibrationMat(SMatrix{3, 3, Float64}(800, 0, 0, 0, 800, 0, 320, 240, 1)))

    @testset "PosedCamera, BoundedCamera and sensor rigs" begin
        C_mm = PosedCamera(model, E_mm)
        C_u = PosedCamera(model, E_u)
        @test has_units(C_u) && !has_units(C_mm)
        @test maps_close(extrinsics(ustrip(mm, with_unit(C_mm, mm))), extrinsics(C_mm))
        @test maps_close(extrinsics(ustrip(mm, C_u)), extrinsics(C_mm))
        @test_throws ArgumentError ustrip(mm, C_mm)
        @test_throws ArgumentError with_unit(C_u, mm)
        B_u = BoundedCamera(C_u, Size2(640, 480))
        @test has_units(B_u) && !has_units(BoundedCamera(C_mm, Size2(640, 480)))
        @test maps_close(extrinsics(ustrip(mm, B_u)), extrinsics(C_mm))
        @test with_unit(ustrip(mm, B_u), mm) isa BoundedCamera
        @test !has_units(BoundedCamera(model, Size2(640, 480)))   # unposed: no lengths

        # projection: the unitful call is the plain call in metres, bit for bit
        X = SVector(0.3, -0.1, 4.0)
        @test project(PosedCamera(model, E_u), X .* m) === project(PosedCamera(model, ustrip(m, E_u)), X)
        @test project(B_u, X .* m) === project(PosedCamera(model, ustrip(m, E_u)), X)
        @test_throws DimensionError project(C_u, X)

        # a sensor rig of unitful cameras, gauge identity included
        ident_u = EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 0.0) .* m)
        rig_u = Rig{ViewFrame{:l}}((
            PosedCamera(model, relabel(ViewFrame{:l}, ViewFrame{:l}, ident_u)),
            PosedCamera(model, relabel(ViewFrame{:l}, ViewFrame{:r}, E_u)),
        ))
        @test has_units(rig_u)
        rig_mm = ustrip(mm, rig_u)
        @test !has_units(rig_mm)
        @test maps_close(extrinsics(rig_mm, 2), relabel(ViewFrame{:l}, ViewFrame{:r}, E_mm))
        @test maps_close(extrinsics(ustrip(mm, with_unit(rig_mm, mm)), 2), extrinsics(rig_mm, 2))
        @test_throws DimensionError Rig{ViewFrame{:l}}((
            PosedCamera(model, relabel(ViewFrame{:l}, ViewFrame{:l}, ident_u)),
            PosedCamera(model, relabel(ViewFrame{:l}, ViewFrame{:r}, E_mm)),
        ))
        @test_throws DimensionError Rig{BodyFrame{:b}}(
            (PosedCamera(model, relabel(BodyFrame{:b}, ViewFrame{:r}, E_u)),),
            Dict(:t => PosedTarget(PlanarTarget(pts2_mm), E_mm)),
        )
    end

    @testset "Trajectory, TargetPoses, TargetPoints3D" begin
        tr_mm = Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => E_mm, 3 => Uncertain(E_mm, Σ6)); recovered = [3])
        tr_u = with_unit(tr_mm, mm)
        @test has_units(tr_u) && !has_units(tr_mm)
        @test tr_u[1] isa EuclideanMap{3, Float64, <:Any, <:Any, BodyFrame{:rig}, WorldFrame}
        back = ustrip(mm, tr_u)
        @test pose_close(back[1], tr_mm[1]) && pose_close(back[3], tr_mm[3])
        @test back.recovered == tr_mm.recovered
        @test maps_close(ustrip(mm, Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => E_u)))[1], tr_mm[1])
        @test_throws ArgumentError ustrip(mm, tr_mm)
        @test_throws DimensionError Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => E_mm, 2 => E_u))
        @test_throws DimensionError Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => Uncertain(E_mm, Σ6), 2 => E_u))

        tp_mm = TargetPoses{CameraFrame}(Dict(1 => Dict(:b => E_mm, :c => Uncertain(E_mm, Σ6))); sensors = "cam")
        tp_u = with_unit(tp_mm, mm)
        @test has_units(tp_u) && !has_units(tp_mm)
        back = ustrip(mm, tp_u)
        @test pose_close(back[1][:b], tp_mm[1][:b]) && pose_close(back[1][:c], tp_mm[1][:c])
        @test back.sensors == tp_mm.sensors
        @test maps_close(geom(pose(tp_u, 1, :b)), geom(pose(with_unit(tp_mm, mm), 1, :b)))
        @test_throws ArgumentError with_unit(tp_u, mm)
        @test_throws DimensionError TargetPoses{CameraFrame}(Dict(1 => Dict(:b => E_mm), 2 => Dict(:c => E_u)))

        obs = [(target = :body, index = 3, xyz = pts3_mm[1]), (target = :body, index = 4, xyz = Uncertain(pts3_mm[2], CovMat{3}(σ = 0.2)))]
        p3_mm = TargetPoints3D{WorldFrame}(Dict(7 => obs); sensors = "mocap")
        p3_u = with_unit(p3_mm, mm)
        @test has_units(p3_u) && !has_units(p3_mm)
        @test eltype(p3_u[7][1].xyz) <: Quantity
        back = ustrip(mm, p3_u)
        @test point_close(back[7][1].xyz, p3_mm[7][1].xyz) && point_close(back[7][2].xyz, p3_mm[7][2].xyz)
        @test back[7][2].index == 4 && back.sensors == p3_mm.sensors
        p3_read = TargetPoints3D{WorldFrame}(Dict(7 => [(target = :body, index = 3, xyz = from_mm(pts3_mm[1]))]))
        @test point_close(ustrip(mm, p3_read)[7][1].xyz, pts3_mm[1])
        @test_throws ArgumentError ustrip(mm, p3_mm)
    end
end
