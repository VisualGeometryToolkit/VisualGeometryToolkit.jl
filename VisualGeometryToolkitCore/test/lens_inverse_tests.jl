# The parametric lens inverse runs on Core's one static LM (`Cameras.static_lm`)
# with the distortion's analytic Jacobian — every term's `transform_deriv` in
# closed form — COLMAP's step clipping as the LM's step bound, and no Dual in the
# solve loop (JuliaLang/julia#64827's pattern): a Dual argument or lens is
# differentiated once at the root (the implicit function theorem).
@testitem "lens inverse: Core's one LM with an analytic Jacobian, no Dual in its loop" begin
    using Test, LinearAlgebra
    using StaticArrays: SVector, SMatrix, StaticVector
    using ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: transform_deriv, undistort_normalized, distort_normalized, backproject
    const C = VisualGeometryToolkitCore.Cameras

    terms = (
        Radial((-0.28, 0.07, 0.01)),
        Radial((0.6, -0.4, 0.05), (0.9, -0.2, 0.03)),
        Tangential(1.5e-3, -8.0e-4),
        ThinPrism((2.0e-3, -1.0e-4), (-1.5e-3, 2.0e-4)),
        Tilt(0.02, -0.015),
        FieldOfView(0.9),
        Additive(Radial((-0.3, 0.09)), Tangential(1.0e-3, 5.0e-4), ThinPrism((1.0e-3,), (-6.0e-4,))),
        Additive(Tangential(4.0e-4, -2.0e-4), ThinPrism((1.0e-3,), (-6.0e-4,))) ∘ Radial((0.03, -0.01, 0.002)),
        Tilt(0.03, 0.01) ∘ FieldOfView(0.5),
    )
    points = (SVector(0.0, 0.0), SVector(1.0e-9, -2.0e-9), SVector(1.0e-4, 3.0e-5), SVector(0.3, -0.2), SVector(-0.7, 0.55))
    @testset "a term's transform_deriv is its Jacobian, in closed form ($(nameof(typeof(t))))" for t in terms
        for x in points
            J = transform_deriv(t, x)
            @test J isa SMatrix{2, 2, Float64, 4}
            @test isapprox(J, ForwardDiff.jacobian(z -> SVector{2}(Tuple(t(z))), x); rtol = 1.0e-13, atol = 1.0e-15)
        end
    end

    lenses = (
        Lens(Perspective(), Tilt(0.02, -0.015) ∘ Additive(Radial((-0.3, 0.09, -0.01)), Tangential(1.0e-3, 5.0e-4))),
        Lens(Equidistant(), Additive(Tangential(4.0e-4, -2.0e-4), ThinPrism((1.0e-3,), (-6.0e-4,))) ∘ Radial((0.03, -0.01, 0.002, -2.0e-4))),
        Lens(Perspective(), Additive(Radial((0.6, -0.4, 0.05), (0.9, -0.2, 0.03)), Tangential(1.5e-3, -8.0e-4))),
    )

    # Whether the compiled call reaches a ForwardDiff type anywhere in its call tree:
    # the inferred types of every statement, through every invoked method.
    function reaches_forwarddiff(@nospecialize(tt); depth = 10)
        seen = Set{Any}()
        stack = Any[(tt, 0)]
        while !isempty(stack)
            t, k = pop!(stack)
            t in seen && continue
            push!(seen, t)
            occursin("ForwardDiff", string(t)) && return true
            k >= depth && continue
            for (ci, _) in Base.code_typed_by_type(t; optimize = true)
                any(S -> occursin("ForwardDiff", string(S)), ci.ssavaluetypes) && return true
                for st in ci.code
                    st isa Expr && st.head === :invoke || continue
                    mi = st.args[1]
                    mi isa Core.CodeInstance && (mi = mi.def)
                    mi isa Core.MethodInstance && push!(stack, (mi.specTypes, k + 1))
                end
            end
        end
        return false
    end
    @testset "no Dual in the inverse: its call tree reaches no ForwardDiff type" begin
        for lens in lenses, T in (Float64, Float32)
            l = VisualGeometryToolkitCore.convert_eltype(T, lens)
            @test !reaches_forwarddiff(Tuple{typeof(backproject), typeof(l), SVector{2, T}})
        end
    end

    @testset "the solve is inferred and allocation-free (a boxed Jacobian is a heap object per trip)" begin
        allocs(f, args...) = (f(args...); @allocated f(args...))
        for lens in lenses, T in (Float64, Float32)
            l = VisualGeometryToolkitCore.convert_eltype(T, lens)
            xd = SVector{2, T}(0.45, 0.3)
            @test @inferred(backproject(l, xd)) isa StaticVector{3, T}
            @test allocs(backproject, l, xd) == 0
            @test only(Base.return_types(C._inverse_lm, (typeof(l.distortion), SVector{2, T}, SVector{2, T}))) ==
                Tuple{SVector{2, T}, Bool, T}
        end
    end

    @testset "a seed within √eps of the root is not taken for it (the LM's cost floor 0)" begin
        d = lenses[2].distortion
        for xd in (SVector(1.5e-3, 1.5e-3), SVector(-2.0e-4, 1.0e-4))
            x, ok, dJ = C._inverse_lm(d, xd)
            @test ok
            @test norm(C._distort(d, x) - xd) <= 4 * eps()
            @test x != xd
            @test dJ ≈ det(transform_deriv(d, x))
        end
    end

    @testset "the increment bound: COLMAP's relative radius, `nothing` the reference's" begin
        p, x = SVector(3.0, 4.0), SVector(0.0, 0.0)
        @test C._bounded_increment(nothing, p, x) === p
        @test C._bounded_increment(0.1, p, x) ≈ SVector(0.06, 0.08)
        @test C._bounded_increment(0.1, p, SVector(60.0, 80.0)) === p          # within 0.1·‖x‖
        @test C._bounded_increment(0.1f0, SVector(3.0f0, 4.0f0), SVector(0.0f0, 0.0f0)) isa SVector{2, Float32}
    end

    @testset "a Dual argument is differentiated at the root: J⁻¹ of the forward map" begin
        for lens in lenses, xd in (SVector(0.1, -0.05), SVector(0.45, 0.3))
            xu = undistort_normalized(lens, xd)
            Ju = ForwardDiff.jacobian(z -> undistort_normalized(lens, z), xd)
            Jf = ForwardDiff.jacobian(z -> distort_normalized(lens, z), xu)
            @test isapprox(Ju * Jf, I; atol = 1.0e-10)
        end
        # a Dual lens parameter: the tilt angle, against central differences
        xd = SVector(0.4, -0.3)
        f(τ) = undistort_normalized(Lens(Perspective(), Tilt(τ, -0.015) ∘ Additive(Radial((-0.3, 0.09, -0.01)), Tangential(1.0e-3, 5.0e-4))), xd)
        h = 1.0e-6
        @test isapprox(ForwardDiff.derivative(f, 0.02), (f(0.02 + h) - f(0.02 - h)) / (2h); rtol = 1.0e-6)
    end
end
