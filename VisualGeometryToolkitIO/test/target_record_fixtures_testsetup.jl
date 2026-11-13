# The `TargetRecordFixtures` test setup (moved out of target_record_tests.jl: ReTestItems reads setups from *_testsetup.jl files,
# so an item of another file finds it when that file runs alone).

@testsetup module TargetRecordFixtures
    using VisualGeometryToolkitCore: Rigid, ThinPlateSpline, HeightField, Scale, Aspect, Skew, Uncertain, CovMat,
        grid_nodes, parameter_vector, geom, reference, calibration
    using Statistics: cov
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: Diagonal
    using Rotations: QuatRotation, RotMatrix, Angle2d
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: mm, m

    export kinds, uncertain, design_points, mmq, identity_rig, fileborn

    # one millimetre number as a reader returns it (lengths.jl: decimal-exact)
    mmq(x) = VgtkIO.read_length(x, mm)
    design_points() = [SVector(mmq(0.0), mmq(0.0)), SVector(mmq(62.8), mmq(0.0)), SVector(mmq(-13.9), mmq(40.25)), SVector(mmq(30.5), mmq(77.0))]
    # a plain metre value as a millimetre file holds it and reads it back
    fileborn(x::Real) = VgtkIO.read_length(VgtkIO.write_length(x * m, mm), mm)

    # every kind, physical, its numbers file-born (each length the metre value
    # nearest a millimetre decimal, so the record writes it back exactly). The
    # thin-plate kinds are pure bending (VGC projects their values: a spline's
    # affine displacement, a field's plane, dropped), so theirs are LITERALS:
    # pure-bending values, which the constructor keeps bit for bit. They are
    # never computed here. The projection is a BLAS product whose last ulp
    # depends on the OpenBLAS kernel the CPU selects (Haswell/Zen vs
    # Sandybridge, Prescott or GitHub's AVX-512 runners), so a computed value,
    # and the content_id pinned on it, would differ by machine. (The literals
    # are the file-born projections, on the Haswell kernel, of
    # fx = [0.01, 0, -0.02, 0.03, 0, 0, 0.01, -0.01, 0] mm,
    # fy = [-0.02, 0.01, 0, 0, 0.04, 0, 0, 0.02, -0.03] mm and
    # h = [0.1, 0, -0.1, 0.2, 0, 0, 0, 0.3, 0] mm.)
    function kinds()
        nodes = [SVector(mmq(p[1]), mmq(p[2])) for p in grid_nodes(3; lo = 0.0, hi = 100.0)]
        E = VgtkCore.EuclideanMap(Angle2d{Float64}(0.01), SVector(mmq(0.5), mmq(-0.25)); check = false)   # the member stores its angle
        S, A, U = Scale(0.001), Aspect(0.002), Skew(-0.003)
        T = ThinPlateSpline(nodes,
            [1.5764702259321875e-5, 1.5158418539816482e-5, 4.552134820311087e-6, 2.9552134820311098e-5, 8.945851100805704e-6,
                1.833956738130031e-5, 3.33956738130032e-6, -7.266716338205075e-6, 1.2126999942289532e-5] .* m,
            [6.897225551491389e-5, 0.00010177539737466663, 9.457853923441935e-5, 9.238168109417197e-5, 0.0001351848229539247,
                9.798796481367743e-5, 9.579110667343005e-5, 0.00011859424853318279, 7.139739039293552e-5] .* m)
        H = HeightField(nodes, [0.0002615206031300322, 0.00020545776593497856, 0.00014939492873992493, 0.00037970911471519355,
            0.0002236462775201399, 0.0002675834403250863, 0.0001978976263003549, 0.0005418347891053013, 0.0002857719519102476] .* m)
        right(cs...) = foldr(∘, cs)   # right-associated, as a target stores a composition
        return (rigid = Rigid(), euclidean = E, scale = S, aspect = A, skew = U, thin_plate_spline = T, height_field = H,
            print_error = A ∘ U, affine_factors = right(E, S, A, U), calibration_chain = H ∘ T, chain_affine = right(H, A, U),
            canonical = right(H, T, E, S, A, U))
    end

    # A rig of targets of every kind — rigid, affine, thin-plate, height field,
    # a composition, Uncertain calibrations, a point target with covariances, an
    # Uncertain pose — its numbers file-born: content_id's pin.
    # rotation = :quat builds the poses as the reader returns them (QuatRotation);
    # :matrix as RotMatrix; :quat_of_matrix as QuatRotation(RotMatrix(...)).
    function identity_rig(; boards::Bool = false, rotation::Symbol = :quat)
        k = kinds()
        bd = boards ? VgtkCore.Board(VgtkCore.Size2(width = mmq(150.0), height = mmq(77.25)), SVector(mmq(-13.9), mmq(0.0))) : nothing
        rot(q) = rotation === :quat ? q : rotation === :matrix ? RotMatrix{3, Float64}(q) : QuatRotation(RotMatrix{3, Float64}(q))
        X(t) = VgtkCore.EuclideanMap(rot(QuatRotation(0.9128709291752769, 0.18257418583505536, 0.36514837167011072, 0.0, false)), SVector(mmq(t[1]), mmq(t[2]), mmq(t[3])))
        I6 = VgtkCore.CovMat(SMatrix{6, 6, Float64}(Diagonal([1.0e-6, 1.0e-6, 1.0e-6, 2.5e-7, 2.5e-7, 4.0e-7])))
        ident = VgtkCore.EuclideanMap(rot(QuatRotation(1.0, 0.0, 0.0, 0.0, false)), SVector(mmq(0.0), mmq(0.0), mmq(0.0)))
        pt = VgtkCore.PointTarget([VgtkCore.Uncertain(SVector(mmq(0.0), mmq(1.5), mmq(-2.0)), VgtkCore.CovMat(SMatrix{3, 3, Float64}(Diagonal([1.0e-8, 1.0e-8, 4.0e-8]))))])
        members = Dict(
            :a_rigid => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(nothing, design_points(), Rigid(), bd), ident),
            :b_affine => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(design_points(); calibration = uncertain(k.affine_factors)), X((10.0, 0.0, 900.0))),
            :c_tps => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(:c_tps; calibration = k.thin_plate_spline), X((-200.0, 15.5, 850.0))),
            :d_height => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(design_points(); calibration = k.height_field), VgtkCore.Uncertain(X((0.0, 120.0, 910.0)), I6)),
            :e_chain => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(nothing, design_points(), uncertain(k.calibration_chain), bd), X((300.0, -40.25, 1000.0))),
            :f_points => VgtkCore.PosedTarget(pt, X((5.0, 5.0, 700.0))),
        )
        return VgtkCore.Rig{VgtkCore.TargetFrame{:a_rigid}}(members)
    end

    # an Uncertain one: a covariance over its parameters (lengths² in m²), its
    # entries file-born decimals (i·10⁻⁶ on the diagonal, a correlation in the
    # first row and column) so the record writes them back exactly
    function uncertain(c)
        n = length(parameter_vector(c))
        Σ = Matrix(Diagonal([parse(Float64, "$(i)e-6") for i in 1:n]))
        if n >= 2
            Σ[1, 2] = Σ[2, 1] = 2.5e-7
            Σ[1, n] = Σ[n, 1] = -1.5e-7
        end
        return Uncertain(c, CovMat(SMatrix{n, n, Float64}(Σ)))
    end
end
