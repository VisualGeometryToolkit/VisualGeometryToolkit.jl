# The axis component of a distortion's imaged region, on a grid and independent
# of the inverse under test: the cells where the forward map keeps orientation
# (det J > 0) and the radial factor is still on its increasing branch (the
# radius is below the profile's first fold or pole, found by a dense scan),
# flood-filled from the axis. A root the inverse returns must lie in it — the
# pointwise acceptance test cannot see connection; this checks it — and every
# interior cell's image must be inverted to a root in it. Shared by
# test/lens_branch_tests.jl and test/lens_oracle_tests.jl.
using StaticArrays: SVector
using LinearAlgebra: det, norm
using ForwardDiff

# The first radius where ρ·N(ρ²)/D(ρ²) stops increasing or D(ρ²) stops being
# positive, scanned in steps of `h` up to `R` (`R` itself when it never does).
function radial_branch_radius(k::Tuple, d::Tuple, R; h = R / 20000)
    N(s) = 1 + sum(k[i] * s^i for i in eachindex(k); init = 0.0)
    D(s) = 1 + sum(d[i] * s^i for i in eachindex(d); init = 0.0)
    prof(ρ) = ρ * N(ρ^2) / D(ρ^2)
    ρ, p = 0.0, 0.0
    while ρ < R
        ρn = ρ + h
        (D(ρn^2) > 0 && prof(ρn) > p) || return ρ
        ρ, p = ρn, prof(ρn)
    end
    return R
end

struct AxisComponent
    xs::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int}
    comp::BitMatrix
end

function axis_component(fwd, ρbranch, R, n)
    xs = range(-R, R, length = n)
    jac(x) = ForwardDiff.jacobian(z -> SVector{2}(Tuple(fwd(z))), x)
    mask = [(x = SVector(a, b); norm(x) < ρbranch && det(jac(x)) > 0) for a in xs, b in xs]
    comp = falses(n, n)
    c = (n + 1) ÷ 2
    stack = [(c, c)]
    comp[c, c] = mask[c, c]
    while !isempty(stack)
        i, j = pop!(stack)
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            a, b = i + di, j + dj
            (1 <= a <= n && 1 <= b <= n && mask[a, b] && !comp[a, b]) || continue
            comp[a, b] = true
            push!(stack, (a, b))
        end
    end
    return AxisComponent(xs, comp)
end

_cell(ac::AxisComponent, v) = clamp(round(Int, (v - first(ac.xs)) / step(ac.xs)) + 1, 1, length(ac.xs))
# In the component, to within `pad` cells (a root on the component's edge sits
# in a cell the grid may have just missed). Outside the grid is not inside.
function in_component(ac::AxisComponent, x; pad = 2)
    n = length(ac.xs)
    (first(ac.xs) <= x[1] <= last(ac.xs) && first(ac.xs) <= x[2] <= last(ac.xs)) || return false
    i, j = _cell(ac, x[1]), _cell(ac, x[2])
    return any(ac.comp[a, b] for a in max(1, i - pad):min(n, i + pad), b in max(1, j - pad):min(n, j + pad))
end

# The component's interior cells (all eight neighbours in it), every `stride`-th.
interior_points(ac::AxisComponent; stride = 4) =
    [SVector(ac.xs[i], ac.xs[j]) for i in 2:stride:(length(ac.xs) - 1), j in 2:stride:(length(ac.xs) - 1)
     if all(ac.comp[a, b] for a in (i - 1):(i + 1), b in (j - 1):(j + 1))]
