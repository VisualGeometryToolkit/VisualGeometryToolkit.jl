# The 2-D inverse of a folding distortion returns the preimage on the imaged
# branch — the one connected to the axis — and finds it wherever it exists.
# Strong OpenCV chains (rational radial + tangential + thin prism, then the
# sensor tilt) from RadiometricCamera's port comparison, where the inverse once
# returned a second preimage past the fold, or NaN near the branch's edge.
@testitem "lens_branch" begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: det, norm
    using ForwardDiff
    using VisualGeometryToolkitCore

    # (f_x, f_y, c_x, c_y, k₁, k₂, p₁, p₂, k₃, k₄, k₅, k₆, s₁, s₂, s₃, s₄, τ_x, τ_y)
    thetas = (
        (0.51352, 0.537785, 0.0154108, 0.0105699, -0.150994, -0.0501823, 0.00244469, 0.00575199, 0.023916, 0.187097, 0.0255086, -0.00128661, -0.00157781, -0.000969799, -0.000762879, 0.00210571, -0.00992587, -0.0062454),
        (0.616308, 0.664222, 0.0153228, 0.0131739, 0.0168802, 0.13711, -0.0314255, 0.00843137, -0.0469514, -0.213397, -0.133988, 0.0338001, 0.000157232, -0.00241694, 0.0032617, -0.00350022, 0.0043038, 0.0557206),
        (0.583578, 0.569513, -0.0143208, -0.00990872, 0.247528, 0.0355112, 0.0263991, 0.0256147, 0.00127868, 0.859257, -0.173361, -0.0114911, -0.0118242, 0.00959124, -0.00728219, 0.00239398, -0.00657118, 0.00513063),
        (1.2395, 1.23279, -0.0212201, 0.0346951, 0.44066, -0.123327, -0.00869389, 0.0622737, -0.025476, -0.341814, 0.735874, -0.0966712, 0.0105684, -0.00407888, -0.034989, -0.00520582, -0.111483, -0.0853656),
        (1.29322, 1.18042, 0.0310774, -0.0347893, 1.40468, -0.00298835, 0.103695, -0.0803676, 0.0664061, -2.03318, 0.387737, 0.0918448, -0.0276789, -0.00434197, 0.00567138, -0.00185455, -0.0326923, -0.0250503),
    )
    opencv(θ) = Lens(Perspective(), Chain(Additive(Radial((θ[5], θ[6], θ[9]), (θ[10], θ[11], θ[12])), Tangential(θ[7], θ[8]),
        ThinPrism((θ[13], θ[14]), (θ[15], θ[16]))), Tilt(θ[17], θ[18])))
    fwd(lens, x) = project(lens, SVector(x[1], x[2], 1.0))
    jac(lens, x) = ForwardDiff.jacobian(z -> SVector{2}(Tuple(fwd(lens, z))), x)
    # Independent of the solver: the radial profile ρ·N/D increases (and D stays
    # positive) from the axis to the root's radius, and the map keeps orientation.
    function on_branch(θ, lens, x)
        N(s) = 1 + s * (θ[5] + s * (θ[6] + s * θ[9]))
        D(s) = 1 + s * (θ[10] + s * (θ[11] + s * θ[12]))
        ρ = norm(x)
        prof(r) = r * N(r^2) / D(r^2)
        rs = range(0, ρ, length = 400)
        mono = all(i -> D(rs[i + 1]^2) > 0 && prof(rs[i + 1]) > prof(rs[i]), 1:(length(rs) - 1))
        return mono && det(jac(lens, x)) > 0
    end
    # The reference root: a fine continuation from the axis, plain Newton.
    function reference(lens, m)
        x = SVector(0.0, 0.0)
        for t in range(0, 1, length = 65)[2:end], _ in 1:30
            J = jac(lens, x)
            det(J) > 0 || return SVector(NaN, NaN)
            x = x - J \ (fwd(lens, x) - t * m)
        end
        return norm(fwd(lens, x) - m) < 1.0e-10 ? x : SVector(NaN, NaN)
    end

    for θ in thetas
        lens = opencv(θ)
        n_found = 0; n_ref = 0
        edge = SVector(0.0, 0.0)
        for px in range(-1, 1, length = 41), py in range(-1, 1, length = 41)
            m = SVector((px - θ[3]) / θ[1], (py - θ[4]) / θ[2])
            ray = backproject(lens, m)
            x = SVector(ray[1] / ray[3], ray[2] / ray[3])
            xr = reference(lens, m)
            if all(isfinite, x)
                n_found += 1
                @test norm(fwd(lens, x) - m) < 1.0e-9
                @test on_branch(θ, lens, x)                     # never a preimage past the fold
                norm(x) > norm(edge) && (edge = m)
            end
            if all(isfinite, xr) && on_branch(θ, lens, xr)
                n_ref += 1
                @test all(isfinite, x) && norm(x - xr) < 1.0e-8  # found wherever it exists
            end
        end
        @test n_found >= n_ref > 0
        # derivatives through the implicit-function step at the outermost root
        Jad = ForwardDiff.jacobian(z -> backproject(lens, z), edge)
        h = 1.0e-6
        Jfd = hcat((backproject(lens, edge + h * SVector(1.0, 0.0)) - backproject(lens, edge - h * SVector(1.0, 0.0))) / 2h,
                   (backproject(lens, edge + h * SVector(0.0, 1.0)) - backproject(lens, edge - h * SVector(0.0, 1.0))) / 2h)
        @test maximum(abs.(Jad - Jfd)) < 1.0e-5
    end
end
