# The lens families against the oracle recorded from the pre-redesign lens zoo
# (test/fixtures/lens_oracle: generate.jl wrote lens_oracle.toml from
# VisualGeometryToolkitCore 1.0.0, 1c4f93f). Every recorded lens is rebuilt from its
# lens record and asked the same questions.
#
# Forward maps and their Jacobians agree to rounding: the polynomials are now
# Horner (`evalpoly`) and the additive terms sum in one documented order, so a
# few units in the last place differ. The inverses agree where the old ones
# converged fully; where they now run a different solver (a lone radial factor
# by the 1-D ratio solve instead of OpenCV's fixed point; an additive radial
# lens by the fixed point instead of the peel) the new root is at least as
# accurate — its residual, in BigFloat, is never worse. `view_angle_limit` is
# now the exact fold of the radial factor read through the frame, not a
# 1024-step scan: within one scan step of the old answer.
@testitem "lens_oracle" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore
    using TOML
    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "lens_oracle", "compare.jl"))

    path = joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "lens_oracle", "lens_oracle.toml")
    oracle = TOML.parsefile(path)
    res = compare_oracle(path)
    forward = ("project", "distort", "jac", "jac_params", "monotone_radius")
    inverse = ("backproject", "undistort")

    # an inverse may move only toward the true root
    function no_worse(name)
        d = oracle[name]
        lens = lens_from_record(d["lens"])
        lb = VgtkCore._precision(BigFloat, lens)
        for (x, o) in zip(_pairs(d["points"], 2), _pairs(d["undistort"], 2))
            n = undistort_normalized(lens, x)
            isnan(n[1]) == isnan(o[1]) || return false
            isnan(n[1]) && continue
            rn = Float64(norm(distort_normalized(lb, big.(n)) - big.(x)))
            ro = Float64(norm(distort_normalized(lb, big.(o)) - big.(x)))
            rn <= max(ro * (1 + 1.0e-6), 1.0e-14 * max(1, norm(x))) || return false   # at a fold both stall at one residual
        end
        return true
    end

    # The thin-prism fisheye's radial factor now bounds what it images, as
    # Kannala–Brandt's always did (1.0.0 composed it with no fold facts and
    # reported `Inf`): its monotone radius is the factor's fold, the same as the
    # Kannala–Brandt lens of that factor.
    tpf = lens_from_record(oracle["thin_prism_fisheye"]["lens"])
    @test oracle["thin_prism_fisheye"]["monotone_radius"] == Inf
    @test VgtkCore.monotone_radius(tpf) == VgtkCore.monotone_radius(Lens(Equidistant(), radial(tpf)))
    @test isfinite(VgtkCore.monotone_radius(tpf))

    for (name, r) in sort(collect(res); by = first)
        @testset "$name" begin
            for q in forward
                haskey(r, q) || continue
                q == "monotone_radius" && name == "thin_prism_fisheye" && continue     # below
                @test r[q].nan == 0
                @test r[q].ulps <= 8 || r[q].abs <= 1.0e-13
            end
            for q in ("project32",)
                @test r[q].nan == 0
                @test r[q].ulps <= 8
            end
            @test r["view_angle_limit"].abs <= π / 1024
            @test r["backproject"].nan == 0 && r["undistort"].nan == 0
            @test no_worse(name)
            # Float32 inverses: within the Float32 solve tolerance of the old ones;
            # one old non-converged point (opencv8, ρ ≈ 80) now converges
            @test r["backproject32"].abs <= 5.0e-6
            @test r["backproject32"].nan <= (name == "opencv8" ? 3 : 0)       # three components of one point
        end
    end
end
