using Test
using LinearAlgebra: norm, normalize
using PoseLib
using VisualGeometryCore
using VisualGeometryCore: HomographyMat, EssentialMat, FundMat, Extrinsics3, FixedModels,
                          PoseLibSolver, Vec3, Point3, Essential5pt
using StaticArrays: SVector, SMatrix
using Rotations: RotMatrix
using Artifacts: load_artifacts_toml

@testset "PoseLib" begin

    include("agent_docs.jl")

    @testset "smoke" begin
        # Identity pose with 3 known 3D points → P3P should recover it.
        X = [Point3(1.0, 0.5, 5.0), Point3(-0.5, 1.0, 4.0), Point3(0.0, -0.5, 6.0)]
        rays = [Vec3(x ./ norm(x)) for x in X]
        result = VisualGeometryCore.p3p(PoseLibSolver(), rays, X)
        @test result isa FixedModels{4, Extrinsics3}
        @test length(result) >= 1

        # essential_5pt returns FixedModels{10, EssentialMat}
        x1 = ntuple(_ -> SVector{3,Float64}(normalize([randn(), randn(), 1.0])), Val(5))
        x2 = ntuple(_ -> SVector{3,Float64}(normalize([randn(), randn(), 1.0])), Val(5))
        @test VisualGeometryCore.essential_5pt(PoseLibSolver(), x1, x2) isa FixedModels{10, EssentialMat{Float64}}

        # homography_4pt PoseLib backend
        s = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)]
        d = [SVector(0.1, 0.1), SVector(1.1, 0.0), SVector(1.0, 1.1), SVector(0.0, 1.0)]
        H = VisualGeometryCore.homography_4pt(PoseLibSolver(), s[1], s[2], s[3], s[4],
                                                                d[1], d[2], d[3], d[4])
        @test H isa HomographyMat{Float64}

    end

    # The recipe (which builds this package's artifact and is submitted to
    # Yggdrasil) carries the wrapper and LICENSE as bundled copies until it can
    # take them from a PoseLib.jl tag; they must not drift.
    @testset "recipe bundle matches the wrapper" begin
        root = pkgdir(PoseLib)
        bundled = joinpath(root, "yggdrasil", "P", "PoseLib", "bundled")
        for f in ("wrapper/julia_wrapper.cpp", "LICENSE")
            @test read(joinpath(bundled, f)) == read(joinpath(root, f))
        end
    end

    # Artifacts.toml binds the tarballs of the recipe's version, one per
    # platform, at that version's release (scripts/build_binaries.jl writes it).
    # Until the PoseLib_jll release it binds a Linux subset of the recipe's
    # platforms (the script's default); the recipe keeps Yggdrasil's full list.
    @testset "artifact matches the recipe" begin
        root = pkgdir(PoseLib)
        recipe = read(joinpath(root, "yggdrasil", "P", "PoseLib", "build_tarballs.jl"), String)
        v = match(r"^version = v\"([^\"]+)\""m, recipe)[1]
        entries = load_artifacts_toml(joinpath(root, "Artifacts.toml"))["PoseLib"]
        @test !isempty(entries)
        for e in entries
            @test e["os"] == "linux"
            url = only(e["download"])["url"]
            @test occursin("/releases/download/PoseLib-binaries-v$v/PoseLib.v$v.", url)
            @test occursin(r"-linux-[a-z]+-cxx11\.tar\.gz$", url)
            @test occursin(Regex("PoseLib\\.v$v\\.$(e["arch"])-"), url)
        end
        @test isfile(PoseLib.libposelib_jl)
    end

    # P3P solver tests (moved from VisualGeometryCore.jl when p3p became PoseLib-only).
    include("test_p3p.jl")
    include("test_p3p_poselib.jl")

    # Truncated-loss refinement wrappers + essential_dlt: Cspond input.
    include("test_refine.jl")

    # The stack's naming rules (VisualGeometryCore.check_naming).
    include("test_quality_naming.jl")
end
