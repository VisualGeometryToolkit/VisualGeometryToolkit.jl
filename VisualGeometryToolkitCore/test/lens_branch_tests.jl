# The 2-D inverse of a folding distortion returns a preimage on the imaged
# branch — the region connected to the axis — and finds one wherever it exists.
# Strong OpenCV chains (rational radial + tangential + thin prism, then the
# sensor tilt) from RadiometricCamera's port comparison, where the inverse once
# returned a second preimage past the fold, or NaN near the branch's edge. The
# reference is not a homotopy: the axis component of {det J > 0, radial factor
# on its branch} by flood fill (test/fixtures/branch_component.jl), and the
# images of its interior points.
@testitem "lens_branch" begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: det, norm
    using ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: _precision
    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "branch_component.jl"))

    # (f_x, f_y, c_x, c_y, k₁, k₂, p₁, p₂, k₃, k₄, k₅, k₆, s₁, s₂, s₃, s₄, τ_x, τ_y), and the
    # half-width of the lattice of sensor points: five from the set the inverse was
    # first measured on (MersenneTwister(7)); the third also over a lattice of
    # half-width 3; and lens 12 of a fresh set (MersenneTwister(11)), whose image of
    # the branch is not star-shaped about the axis.
    cases = (
        ((0.51352, 0.537785, 0.0154108, 0.0105699, -0.150994, -0.0501823, 0.00244469, 0.00575199, 0.023916, 0.187097, 0.0255086, -0.00128661, -0.00157781, -0.000969799, -0.000762879, 0.00210571, -0.00992587, -0.0062454), 1.0),
        ((0.616308, 0.664222, 0.0153228, 0.0131739, 0.0168802, 0.13711, -0.0314255, 0.00843137, -0.0469514, -0.213397, -0.133988, 0.0338001, 0.000157232, -0.00241694, 0.0032617, -0.00350022, 0.0043038, 0.0557206), 1.0),
        ((0.583578, 0.569513, -0.0143208, -0.00990872, 0.247528, 0.0355112, 0.0263991, 0.0256147, 0.00127868, 0.859257, -0.173361, -0.0114911, -0.0118242, 0.00959124, -0.00728219, 0.00239398, -0.00657118, 0.00513063), 1.0),
        ((0.583578, 0.569513, -0.0143208, -0.00990872, 0.247528, 0.0355112, 0.0263991, 0.0256147, 0.00127868, 0.859257, -0.173361, -0.0114911, -0.0118242, 0.00959124, -0.00728219, 0.00239398, -0.00657118, 0.00513063), 3.0),
        ((1.2395, 1.23279, -0.0212201, 0.0346951, 0.44066, -0.123327, -0.00869389, 0.0622737, -0.025476, -0.341814, 0.735874, -0.0966712, 0.0105684, -0.00407888, -0.034989, -0.00520582, -0.111483, -0.0853656), 1.0),
        ((1.29322, 1.18042, 0.0310774, -0.0347893, 1.40468, -0.00298835, 0.103695, -0.0803676, 0.0664061, -2.03318, 0.387737, 0.0918448, -0.0276789, -0.00434197, 0.00567138, -0.00185455, -0.0326923, -0.0250503), 1.0),
        ((1.0814372829625307, 1.0078432705375948, 0.05367441926664141, 0.020059434003556524, -0.36405747836680913, -0.003326126819997479, 0.0037657811228031042, -0.0057581915148212925, 0.042994416425077696, -0.21037524661653864, 0.1316433835242977, -0.003957229171810205, -0.003325159914997237, 0.0025195594077938002, -0.009930701796065578, 0.0019373073031102954, 0.009812759296076286, -0.022888098663893065), 1.0),
    )
    opencv(θ) = Lens(Perspective(), Chain(Additive(Radial((θ[5], θ[6], θ[9]), (θ[10], θ[11], θ[12])), Tangential(θ[7], θ[8]),
        ThinPrism((θ[13], θ[14]), (θ[15], θ[16]))), Tilt(θ[17], θ[18])))
    fwd(lens, x) = project(lens, SVector(x[1], x[2], one(x[1])))
    undist(lens, m) = (r = backproject(lens, m); SVector(r[1] / r[3], r[2] / r[3]))

    for (θ, R) in cases
        lens = opencv(θ)
        normalised(px, py) = SVector((px - θ[3]) / θ[1], (py - θ[4]) / θ[2])
        ms = [normalised(px, py) for px in range(-R, R, length = 41), py in range(-R, R, length = 41)]
        roots = [undist(lens, m) for m in ms]
        found = filter(x -> all(isfinite, x), roots)
        @test !isempty(found)
        Rx = 1.1 * maximum(norm, found)
        ac = axis_component(x -> fwd(lens, x), radial_branch_radius((θ[5], θ[6], θ[9]), (θ[10], θ[11], θ[12]), Rx), Rx, 601)
        # every root is one: a preimage, on the axis component
        for (m, x) in zip(ms, roots)
            all(isfinite, x) || continue
            @test norm(fwd(lens, x) - m) < 1.0e-9 * max(1, norm(m))
            @test in_component(ac, x)
        end
        # found wherever it exists: the image of every interior point of the component
        # is inverted to a preimage on it (the map need not be injective there: two
        # preimages on the component are both the branch's). Away from the fold's
        # edge only: where det J < 0.1 the solve is near-singular and converges
        # linearly (measured on the last lens: 4 points with det J ≈ 0.02 need 24 to
        # 178 trust-region trips), past every trip budget.
        jac(x) = ForwardDiff.jacobian(z -> SVector{2}(Tuple(fwd(lens, z))), x)
        for x in interior_points(ac; stride = 6)
            det(jac(x)) > 0.1 || continue
            m = fwd(lens, x)
            xb = undist(lens, m)
            @test all(isfinite, xb) && norm(fwd(lens, xb) - m) < 1.0e-9 * max(1, norm(m)) && in_component(ac, xb)
        end
        # derivatives through the implicit-function step at the outermost root
        k = argmax(i -> all(isfinite, roots[i]) ? norm(roots[i]) : -1.0, eachindex(roots))
        edge = ms[k]
        Jad = ForwardDiff.jacobian(z -> backproject(lens, z), edge)
        h = 1.0e-6
        Jfd = hcat((backproject(lens, edge + h * SVector(1.0, 0.0)) - backproject(lens, edge - h * SVector(1.0, 0.0))) / 2h,
                   (backproject(lens, edge + h * SVector(0.0, 1.0)) - backproject(lens, edge - h * SVector(0.0, 1.0))) / 2h)
        @test maximum(abs.(Jad - Jfd)) < 1.0e-5 * max(1, maximum(abs.(Jad)))
    end

    # A radial factor that is not the chain's first stage is tested at its own
    # input: here ρ(1 − 0.3ρ² + 0.04ρ⁴) folds at ρ = √2 after the tilt and rises
    # again past ρ = √2.5, where det J > 0 once more (the trust-region Newton from
    # x_d lands there for 233 of these points). Every root lies before the fold;
    # every point well inside the branch's image (radius ≈ 0.79) is imaged.
    tilted = Lens(Perspective(), Chain(Tilt(0.02, -0.01), Additive(Radial((-0.3, 0.04)), Tangential(1.0e-3, -5.0e-4))))
    ρfold = sqrt(2.0)
    for u in range(-1.0, 1.0, length = 41), v in range(-1.0, 1.0, length = 41)
        m = SVector(u, v)
        x = undist(tilted, m)
        if all(isfinite, x)
            @test norm(fwd(tilted, x) - m) < 1.0e-9
            @test norm(distort_normalized(Lens(Perspective(), Tilt(0.02, -0.01)), x)) < ρfold
        else
            @test norm(m) > 0.75
        end
    end
    # a Full-basis radial factor has no forward map in a lens's frame, anywhere
    full = VisualGeometryToolkitCore.Cameras.Radial{VisualGeometryToolkitCore.Cameras.Full}((0.1,), ())
    @test_throws ArgumentError Lens(Perspective(), Chain(Additive(full, Tangential(0.0, 1.0e-3)), Tilt(0.01, 0.0)))

    # Float32 on the CPU, the continuation included: the same points are imaged as in
    # Float64, at the Float32 solve tolerance
    θ, R = cases[4]
    lens = opencv(θ)
    lens32 = _precision(Float32, lens)
    for px in range(-R, R, length = 41), py in range(-R, R, length = 41)
        m = SVector((px - θ[3]) / θ[1], (py - θ[4]) / θ[2])
        x = undist(lens, m)
        x32 = undist(lens32, SVector{2, Float32}(m))
        @test all(isfinite, x) == all(isfinite, x32)
        all(isfinite, x) && @test norm(x32 - x) <= 1.0e-4 * max(1, norm(x))
    end
end
