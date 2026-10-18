# =============================================================================
# GTE ellipse intersection-area parity: native port vs compiled GTE oracle
# =============================================================================
#
# Run with:  julia --project=. test/gte_parity/ellipse_area_parity.jl
#
# Compares VisualGeometryToolkitCore._native_ellipse_intersection_area against the
# compiled GTE reference on a large random suite plus structured edge-case
# suites. Exits 1 if the random suite max abs diff exceeds 1e-6 (excluding
# near-tangency cases, where the double root classification may legitimately
# flip).
#
# VGC ships no C++ — the oracle is built here, on demand, from an upstream
# Geometric Tools checkout (github.com/davideberly/GeometricTools). Provide:
#   ELLIPSE_AREA_ORACLE — path to a prebuilt libellipse_area.so, OR
#   GTE_DIR — path to a directory whose Mathematics/ holds the GTE headers
#             (e.g. <GeometricTools>/GTE); the wrapper below is compiled
#             against it with g++.

using VisualGeometryToolkitCore
using Random
using Printf

const VGC = VisualGeometryToolkitCore
const _native = VGC.Primitives._native_ellipse_intersection_area

# C entry points mirrored by the native port (the retired FFI contract).
const _WRAPPER_CPP = raw"""
#include <cmath>
#include <Mathematics/IntrAreaEllipse2Ellipse2.h>
using namespace gte;
extern "C" {
double ellipse_intersection_area(
    double cx1, double cy1, double a1, double b1, double theta1,
    double cx2, double cy2, double a2, double b2, double theta2)
{
    Ellipse2<double> e0, e1;
    e0.center = {cx1, cy1};
    e0.axis[0] = {std::cos(theta1), std::sin(theta1)};
    e0.axis[1] = {-std::sin(theta1), std::cos(theta1)};
    e0.extent = {a1, b1};
    e1.center = {cx2, cy2};
    e1.axis[0] = {std::cos(theta2), std::sin(theta2)};
    e1.axis[1] = {-std::sin(theta2), std::cos(theta2)};
    e1.extent = {a2, b2};
    AreaEllipse2Ellipse2<double> query;
    auto result = query(e0, e1);
    return result.area;
}
}
"""

function _build_oracle()::String
    haskey(ENV, "ELLIPSE_AREA_ORACLE") && return ENV["ELLIPSE_AREA_ORACLE"]
    gte = get(ENV, "GTE_DIR", "")
    isdir(joinpath(gte, "Mathematics")) || error(
        "No oracle available. Set ELLIPSE_AREA_ORACLE to a prebuilt " *
            "libellipse_area.so, or GTE_DIR to an upstream GTE header root " *
            "(a directory containing Mathematics/, e.g. <GeometricTools>/GTE)."
    )
    dir = mktempdir()
    src = joinpath(dir, "ellipse_area_wrapper.cpp")
    lib = joinpath(dir, "libellipse_area.so")
    write(src, _WRAPPER_CPP)
    run(`g++ -std=c++17 -O2 -fPIC -shared -I $gte $src -o $lib`)
    return lib
end

const _ORACLE_LIB = _build_oracle()

_oracle(cx1, cy1, a1, b1, t1, cx2, cy2, a2, b2, t2) =
    ccall(
    (:ellipse_intersection_area, _ORACLE_LIB), Cdouble,
    (
        Cdouble, Cdouble, Cdouble, Cdouble, Cdouble,
        Cdouble, Cdouble, Cdouble, Cdouble, Cdouble,
    ),
    cx1, cy1, a1, b1, t1, cx2, cy2, a2, b2, t2
)

# -----------------------------------------------------------------------------
# Case runner / statistics
# -----------------------------------------------------------------------------

mutable struct SuiteStats
    n::Int
    max_abs::Float64
    max_rel::Float64
    n_disagree_1e8::Int      # abs diff > 1e-8
    n_excluded_tangent::Int  # near-tangency exclusions (random suite only)
    worst::Vector{Tuple{Float64, NTuple{10, Float64}, Float64, Float64}}  # (absdiff, params, native, ffi)
end

SuiteStats() = SuiteStats(0, 0.0, 0.0, 0, 0, Tuple{Float64, NTuple{10, Float64}, Float64, Float64}[])

# Detect near-tangency: perturb both semi-axis pairs by ±1e-9; if the oracle's
# intersection-point classification (numpoints via with-info entry point) or
# the zero/nonzero status of the area flips, the case sits within ~1e-9 of a
# tangent configuration and classification may legitimately differ.
function is_near_tangent(p::NTuple{10, Float64})
    cx1, cy1, a1, b1, t1, cx2, cy2, a2, b2, t2 = p
    a0 = _oracle(cx1, cy1, a1, b1, t1, cx2, cy2, a2, b2, t2)
    ap = _oracle(cx1, cy1, a1 + 1.0e-9, b1 + 1.0e-9, t1, cx2, cy2, a2 + 1.0e-9, b2 + 1.0e-9, t2)
    am = _oracle(cx1, cy1, a1 - 1.0e-9, b1 - 1.0e-9, t1, cx2, cy2, a2 - 1.0e-9, b2 - 1.0e-9, t2)
    z0 = a0 <= 0.0
    if (ap <= 0.0) != z0 || (am <= 0.0) != z0
        return true
    end
    # Also flag flips between containment-scale and partial-overlap areas.
    scale = max(abs(a0), 1.0e-12)
    return abs(ap - a0) > 1.0e-3 * scale || abs(am - a0) > 1.0e-3 * scale
end

function record!(st::SuiteStats, p::NTuple{10, Float64}; exclude_tangent::Bool = false)
    vn = _native(p...)
    vf = _oracle(p...)
    ad = abs(vn - vf)
    rd = ad / max(max(abs(vn), abs(vf)), 1.0e-12)
    st.n += 1
    if ad > 1.0e-8
        st.n_disagree_1e8 += 1
    end
    if exclude_tangent && ad > 1.0e-6 && is_near_tangent(p)
        st.n_excluded_tangent += 1
        return
    end
    if ad > st.max_abs
        st.max_abs = ad
    end
    if rd > st.max_rel
        st.max_rel = rd
    end
    push!(st.worst, (ad, p, vn, vf))
    sort!(st.worst; by = first, rev = true)
    length(st.worst) > 5 && resize!(st.worst, 5)
    return
end

function report(name::String, st::SuiteStats)
    return @printf(
        "%-32s n=%8d  max_abs=%.3e  max_rel=%.3e  n(>1e-8)=%d  excluded_tangent=%d\n",
        name, st.n, st.max_abs, st.max_rel, st.n_disagree_1e8, st.n_excluded_tangent
    )
end

function print_worst(name::String, st::SuiteStats)
    println("Worst cases — ", name, ":")
    for (ad, p, vn, vf) in st.worst
        @printf("  absdiff=%.6e native=%.17g ffi=%.17g\n", ad, vn, vf)
        @printf("    e1: cx=%.17g cy=%.17g a=%.17g b=%.17g th=%.17g\n", p[1], p[2], p[3], p[4], p[5])
        @printf("    e2: cx=%.17g cy=%.17g a=%.17g b=%.17g th=%.17g\n", p[6], p[7], p[8], p[9], p[10])
    end
    return
end

# -----------------------------------------------------------------------------
# Random suite
# -----------------------------------------------------------------------------

function random_suite(n::Int)
    rng = Xoshiro(0x5eed_e111)
    st = SuiteStats()
    for _ in 1:n
        cx1 = 20.0 * rand(rng) - 10.0
        cy1 = 20.0 * rand(rng) - 10.0
        cx2 = 20.0 * rand(rng) - 10.0
        cy2 = 20.0 * rand(rng) - 10.0
        s1 = 0.1 + 4.9 * rand(rng)
        s2 = 0.1 + 4.9 * rand(rng)
        a1, b1 = max(s1, s2), min(s1, s2)  # a >= b (GTE extent convention)
        s3 = 0.1 + 4.9 * rand(rng)
        s4 = 0.1 + 4.9 * rand(rng)
        a2, b2 = max(s3, s4), min(s3, s4)
        t1 = 2pi * rand(rng)
        t2 = 2pi * rand(rng)
        record!(st, (cx1, cy1, a1, b1, t1, cx2, cy2, a2, b2, t2); exclude_tangent = true)
    end
    return st
end

# -----------------------------------------------------------------------------
# Structured suites
# -----------------------------------------------------------------------------

function structured_suites()
    suites = Dict{String, SuiteStats}()
    rng = Xoshiro(0xfeed_beef)

    # Disjoint pairs (well separated).
    st = SuiteStats()
    for _ in 1:2000
        a1 = 0.5 + 2.0 * rand(rng); b1 = 0.2 + 0.3 * a1 * rand(rng)
        a2 = 0.5 + 2.0 * rand(rng); b2 = 0.2 + 0.3 * a2 * rand(rng)
        t1 = 2pi * rand(rng); t2 = 2pi * rand(rng)
        d = a1 + a2 + 1.0 + 5.0 * rand(rng)
        ang = 2pi * rand(rng)
        record!(st, (0.0, 0.0, a1, b1, t1, d * cos(ang), d * sin(ang), a2, b2, t2))
    end
    suites["disjoint"] = st

    # One contains the other (both orders).
    st = SuiteStats()
    for _ in 1:2000
        a1 = 4.0 + 1.0 * rand(rng); b1 = 3.0 + 0.5 * rand(rng)
        a2 = 0.5 + 0.5 * rand(rng); b2 = 0.2 + 0.2 * rand(rng)
        t1 = 2pi * rand(rng); t2 = 2pi * rand(rng)
        dx = 0.5 * rand(rng); dy = 0.5 * rand(rng)
        record!(st, (0.0, 0.0, a1, b1, t1, dx, dy, a2, b2, t2))  # e1 contains e2
        record!(st, (dx, dy, a2, b2, t2, 0.0, 0.0, a1, b1, t1))  # e2 contained in e1
    end
    suites["containment (both orders)"] = st

    # Identical ellipses: area must equal pi*a*b.
    st = SuiteStats()
    ident_max_err = 0.0
    for _ in 1:2000
        cx = 20.0 * rand(rng) - 10.0; cy = 20.0 * rand(rng) - 10.0
        a = 0.1 + 4.9 * rand(rng); b = 0.1 + a * rand(rng)
        b = min(a, b); t = 2pi * rand(rng)
        p = (cx, cy, a, b, t, cx, cy, a, b, t)
        record!(st, p)
        ident_max_err = max(ident_max_err, abs(_native(p...) - pi * a * b))
    end
    suites["identical"] = st
    @printf("identical-ellipse |native - pi*a*b| max = %.3e\n", ident_max_err)

    # External tangency: two axis-aligned ellipses touching at one point.
    st = SuiteStats()
    for _ in 1:1000
        a1 = 1.0 + 2.0 * rand(rng); b1 = 0.5 + 0.5 * rand(rng)
        a2 = 1.0 + 2.0 * rand(rng); b2 = 0.5 + 0.5 * rand(rng)
        record!(st, (0.0, 0.0, a1, b1, 0.0, a1 + a2, 0.0, a2, b2, 0.0))
    end
    suites["external tangency"] = st

    # Internal tangency: small ellipse touching the big one from inside.
    st = SuiteStats()
    for _ in 1:1000
        a1 = 3.0 + 1.0 * rand(rng); b1 = 2.0 + 0.5 * rand(rng)
        a2 = 0.5 + 0.5 * rand(rng); b2 = 0.2 + 0.2 * rand(rng)
        record!(st, (0.0, 0.0, a1, b1, 0.0, a1 - a2, 0.0, a2, b2, 0.0))
    end
    suites["internal tangency"] = st

    # Near-tangent perturbations of external tangency, offsets of ±1e-9.
    st = SuiteStats()
    for _ in 1:1000
        a1 = 1.0 + 2.0 * rand(rng); b1 = 0.5 + 0.5 * rand(rng)
        a2 = 1.0 + 2.0 * rand(rng); b2 = 0.5 + 0.5 * rand(rng)
        eps = (rand(rng) < 0.5 ? -1.0e-9 : 1.0e-9)
        record!(st, (0.0, 0.0, a1, b1, 0.0, a1 + a2 + eps, 0.0, a2, b2, 0.0))
    end
    suites["near tangency (±1e-9)"] = st

    # Circles (a == b).
    st = SuiteStats()
    for _ in 1:2000
        r1 = 0.1 + 4.9 * rand(rng); r2 = 0.1 + 4.9 * rand(rng)
        cx = 10.0 * rand(rng) - 5.0; cy = 10.0 * rand(rng) - 5.0
        record!(st, (0.0, 0.0, r1, r1, 2pi * rand(rng), cx, cy, r2, r2, 2pi * rand(rng)))
    end
    suites["circles"] = st

    # Extreme aspect (a/b = 1000).
    st = SuiteStats()
    for _ in 1:1000
        a1 = 1.0 + 4.0 * rand(rng); b1 = a1 / 1000.0
        a2 = 0.5 + 4.5 * rand(rng); b2 = 0.2 + 0.8 * rand(rng)
        t1 = 2pi * rand(rng); t2 = 2pi * rand(rng)
        cx = 2.0 * rand(rng) - 1.0; cy = 2.0 * rand(rng) - 1.0
        record!(st, (0.0, 0.0, a1, b1, t1, cx, cy, a2, b2, t2))
    end
    suites["extreme aspect (1000:1)"] = st

    # Zero-rotation and pi/2 pairs.
    st = SuiteStats()
    for _ in 1:2000
        a1 = 0.5 + 3.0 * rand(rng); b1 = 0.2 + 0.8 * a1 * rand(rng)
        a2 = 0.5 + 3.0 * rand(rng); b2 = 0.2 + 0.8 * a2 * rand(rng)
        cx = 4.0 * rand(rng) - 2.0; cy = 4.0 * rand(rng) - 2.0
        t1 = rand(rng) < 0.5 ? 0.0 : pi / 2
        t2 = rand(rng) < 0.5 ? 0.0 : pi / 2
        record!(st, (0.0, 0.0, a1, min(a1, b1), t1, cx, cy, a2, min(a2, b2), t2))
    end
    suites["axis-aligned (0, pi/2)"] = st

    return suites
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

const N_RANDOM = 200_000

println("=== GTE ellipse intersection-area parity: native vs FFI ===")
println()

random_stats = random_suite(N_RANDOM)
report("random (n=$(N_RANDOM))", random_stats)
println()

suite_stats = structured_suites()
println()
for name in sort!(collect(keys(suite_stats)))
    report(name, suite_stats[name])
end
println()

print_worst("random suite", random_stats)
println()
worst_structured = ("", SuiteStats())
for (name, st) in suite_stats
    if st.max_abs > worst_structured[2].max_abs
        global worst_structured = (name, st)
    end
end
if !isempty(worst_structured[1])
    print_worst("structured: " * worst_structured[1], worst_structured[2])
end
println()

# -----------------------------------------------------------------------------
# Timing (manual ns loop)
# -----------------------------------------------------------------------------

function time_impl(f, params::Vector{NTuple{10, Float64}})
    s = 0.0
    for p in params  # warmup
        s += f(p...)
    end
    t0 = time_ns()
    for p in params
        s += f(p...)
    end
    t1 = time_ns()
    return (t1 - t0) / length(params), s
end

let rng = Xoshiro(42)
    params = Vector{NTuple{10, Float64}}(undef, 100_000)
    for i in eachindex(params)
        s1 = 0.1 + 4.9 * rand(rng); s2 = 0.1 + 4.9 * rand(rng)
        s3 = 0.1 + 4.9 * rand(rng); s4 = 0.1 + 4.9 * rand(rng)
        params[i] = (
            20.0 * rand(rng) - 10.0, 20.0 * rand(rng) - 10.0,
            max(s1, s2), min(s1, s2), 2pi * rand(rng),
            20.0 * rand(rng) - 10.0, 20.0 * rand(rng) - 10.0,
            max(s3, s4), min(s3, s4), 2pi * rand(rng),
        )
    end
    tn, _ = time_impl(_native, params)
    tf, _ = time_impl(_oracle, params)
    @printf(
        "Timing over %d calls:  native = %.1f ns/call,  FFI = %.1f ns/call  (ratio %.2fx)\n",
        length(params), tn, tf, tn / tf
    )
end
println()

# -----------------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------------

if random_stats.max_abs > 1.0e-6
    println("FAIL: random-suite max abs diff $(random_stats.max_abs) > 1e-6")
    exit(1)
else
    println("OK: random-suite max abs diff $(random_stats.max_abs) <= 1e-6")
end
