import VisualGeometryToolkitCore as VgtkCore
using Test
using LinearAlgebra: norm, normalize
using PoseLib
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: HomographyMat, EssentialMat, FundMat, Extrinsics3, FixedModels,
                          PoseLibSolver, Vec3, Point3, Essential5pt
using StaticArrays: SVector, SMatrix
using Rotations: RotMatrix

@testset "PoseLib" begin

    include("agent_docs.jl")

    # Every julia block of the README runs as written.
    include("readme_tests.jl")

    @testset "the library comes from PoseLib_jll" begin
        @test PoseLib.is_available() && Base.ispublic(PoseLib, :is_available)
    end

    @testset "smoke" begin
        # Identity pose with 3 known 3D points → P3P should recover it.
        X = [Point3(1.0, 0.5, 5.0), Point3(-0.5, 1.0, 4.0), Point3(0.0, -0.5, 6.0)]
        rays = [Vec3(x ./ norm(x)) for x in X]
        result = VgtkCore.p3p(PoseLibSolver(), rays, X)
        @test result isa FixedModels{4, Extrinsics3}
        @test length(result) >= 1

        # essential_5pt returns FixedModels{10, EssentialMat}
        x1 = ntuple(_ -> SVector{3,Float64}(normalize([randn(), randn(), 1.0])), Val(5))
        x2 = ntuple(_ -> SVector{3,Float64}(normalize([randn(), randn(), 1.0])), Val(5))
        @test VgtkCore.essential_5pt(PoseLibSolver(), x1, x2) isa FixedModels{10, EssentialMat{Float64}}

        # homography_4pt PoseLib backend
        s = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)]
        d = [SVector(0.1, 0.1), SVector(1.1, 0.0), SVector(1.0, 1.1), SVector(0.0, 1.0)]
        H = VgtkCore.homography_4pt(PoseLibSolver(), s[1], s[2], s[3], s[4],
                                                                d[1], d[2], d[3], d[4])
        @test H isa HomographyMat{Float64}

    end

    # P3P solver tests (moved from VisualGeometryToolkitCore.jl when p3p became PoseLib-only).
    include("test_p3p.jl")
    include("test_p3p_poselib.jl")

    # Truncated-loss refinement wrappers + essential_dlt: Cspond input.
    include("test_refine.jl")

    # The stack's naming rules (VisualGeometryToolkitCore.check_naming).
    include("test_quality_naming.jl")
end
