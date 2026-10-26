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
# accurate — its residual, in BigFloat, is never worse; where the old ones did
# not converge, the continuation from the axis may now reach a root (12 far
# points of two lenses). `view_angle_limit` is
# now the exact fold of the radial factor read through the frame, not a
# 1024-step scan: within one scan step of the old answer.
@testitem "lens_oracle" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm, det
    using ForwardDiff
    using VisualGeometryToolkitCore
    using TOML
    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "lens_oracle", "compare.jl"))

    path = joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "lens_oracle", "lens_oracle.toml")
    oracle = TOML.parsefile(path)
    res = compare_oracle(path)
    forward = ("project", "distort", "jac", "jac_params", "monotone_radius")
    inverse = ("backproject", "undistort")

    # An inverse may move only toward the true root, and loses no point the old
    # one imaged. It may gain one: the continuation from the axis reaches roots
    # the old solve did not (far out, ρ_d up to 2·10⁴), each a root to rounding
    # where the map keeps orientation. Returns (ok, number gained).
    function no_worse(name)
        d = oracle[name]
        lens = lens_from_record(d["lens"])
        lb = VgtkCore._precision(BigFloat, lens)
        gained = 0
        for (x, o) in zip(_pairs(d["points"], 2), _pairs(d["undistort"], 2))
            n = undistort_normalized(lens, x)
            isnan(n[1]) && !isnan(o[1]) && return false, gained
            isnan(n[1]) && continue
            rn = Float64(norm(distort_normalized(lb, big.(n)) - big.(x)))
            if isnan(o[1])
                J = ForwardDiff.jacobian(z -> SVector{2}(Tuple(distort_normalized(lens, z))), SVector{2}(n[1], n[2]))
                (rn <= 1.0e-14 * max(1, norm(x)) && det(J) > 0) || return false, gained
                gained += 1
                continue
            end
            ro = Float64(norm(distort_normalized(lb, big.(o)) - big.(x)))
            rn <= max(ro * (1 + 1.0e-6), 1.0e-14 * max(1, norm(x))) || return false, gained   # at a fold both stall at one residual
        end
        return true, gained
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
            ok, gained = no_worse(name)
            @test ok
            # every NaN mismatch is a gained point (2 components undistorted, 3 back-projected)
            @test r["undistort"].nan == 2 * gained && r["backproject"].nan == 3 * gained
            @test gained == get(Dict("opencv8" => 8, "tangential_only" => 4), name, 0)
            # Float32 inverses: within the Float32 solve tolerance of the old ones;
            # one old non-converged point (opencv8, ρ ≈ 80) now converges
            @test r["backproject32"].abs <= 5.0e-6
            @test r["backproject32"].nan <= (name == "opencv8" ? 3 : 0)       # three components of one point
        end
    end
end
