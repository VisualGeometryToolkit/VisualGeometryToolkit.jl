# =============================================================================
# Pointwise maps over arrays — one kernel, every backend
# =============================================================================
#
# A batch of points goes through a per-point function in ONE KernelAbstractions
# kernel on the backend that owns the output, the CPU's included (the owner's
# one-kernel policy, 2026-11-11: the host's broadcast and its threaded split,
# up to 3.4× faster at 1024 points, are gone; the slower CPU launch is
# accepted). The per-point function — a lens's `project`, a camera, a framed
# stage — is the kernel body, with no per-family kernel code.
#
# Failure semantics are the per-point function's own: a point it does not map
# comes back NaN (`all(isfinite, out[i])` is its validity), never a throw.

"""
    OnBackend(backend; workgroupsize = nothing)
    (exec::OnBackend)(kernel[, ndrange]) -> KernelAbstractions.Kernel

The stack's ONE kernel-launch policy. Called on a KernelAbstractions `@kernel`,
it instantiates it on `backend` itself: the backend value, so its compiler
options hold (`CUDABackend(; always_inline = true)`), with the workgroup
`workgroupsize` names, per backend:

- `nothing`: the backend's choice (on the CPU, KernelAbstractions' default:
  up to 1024 items as one workgroup on the calling task, a larger range in
  groups of 1024 items over the threads).
- one work item (`1`, `(1, 1)`, `(1, 1, 1)`): one item per workgroup on the
  CPU, so a kernel whose work item is a long loop (a pool's (pair, model[, grid
  point]) item) runs its items on every thread where the default's 1024-item
  groups leave most threads idle; on a device, one warp (32 items) per
  workgroup: one item would leave 31 of 32 lanes idle, and the backend's own
  choice puts these small ranges of long items on few multiprocessors
  (RANSACScoringDoneRight's grids on a B200: the Student-t grid 30.6 ms at one
  item, 1.55 at the backend's choice, 1.35 at one warp; its mixture marginal
  62.1, 90.4 and 17.7 ms).
- any other size (a kernel's tile, a measured device size): that size, on
  every backend.

A static `ndrange` instantiates a statically sized kernel at `workgroupsize`
as given (the CPU's allocation-free one-item launch). Every stack kernel is
instantiated here.

As [`pointwise!`](@ref)'s `exec`, it is that one kernel's launch; without it the
kernel runs on `get_backend(out)` with the defaults. A functor whose fields are
each one of its type parameters or an isbits value reaches the kernel as its
fields, rebuilt on the device: one large argument costs local memory there.
"""
struct OnBackend{B <: KernelAbstractions.Backend, W}
    backend::B
    workgroupsize::W
end
OnBackend(backend::KernelAbstractions.Backend; workgroupsize = nothing) = OnBackend(backend, workgroupsize)
# A one-item kernel's workgroup on a device is one warp, 32 items. Measured on a B200 (RANSACScoringDoneRight's five grids,
# 128 pairs × 800 points × 48 models, min of 7): Student-t 1.55 / 30.6 / 1.35 ms, MSAC 10.9 / 19.6 / 6.5 ms,
# MAGSAC++ 18.6 / 26.9 / 15.6 ms, Welsch marginal 86.6 / 67.7 / 30.6 ms, mixture marginal 90.4 / 62.1 / 17.7 ms at the
# backend's choice / one item / 32. The backend's choice fills few multiprocessors with large workgroups over these
# small ndranges of long items; one item per workgroup leaves 31 of 32 lanes idle.
@inline function (p::OnBackend)(kernel)
    w = p.workgroupsize
    w === nothing && return kernel(p.backend)
    !(p.backend isa KernelAbstractions.CPU) && all(isone, w) && return kernel(p.backend, 32)
    return kernel(p.backend, w)
end
@inline (p::OnBackend)(kernel, ndrange) = kernel(p.backend, p.workgroupsize, ndrange)

"""
    pointwise!(f, out, xs...; exec = OnBackend(get_backend(out))) -> out

`out[i] = f(xs[1][i], xs[2][i], …)` for every index, as ONE KernelAbstractions
kernel on the backend that owns `out` (every backend, the CPU's included), or
on the backend [`OnBackend`](@ref) `exec` names. `out` and `xs` share one shape
(no broadcasting: a `DimensionMismatch` otherwise). `f` is any per-point
function; for a GPU array it must be isbits and kernel-lawful (no allocation,
no throw, no dynamic dispatch), as every lens and camera map is. On the CPU the
launch has finished on return; on a device it is asynchronous, as a device
broadcast is: `KernelAbstractions.synchronize(backend)` before `out` is read on
the host.
"""
function pointwise!(f::F, out::AbstractArray, xs::AbstractArray...; exec::OnBackend = OnBackend(KernelAbstractions.get_backend(out))) where {F}
    kernel, args, ndrange = pointwise_kernel(f, out, xs...; exec)
    isempty(out) || kernel(args...; ndrange)
    return out
end

"""
    pointwise_kernel(f, out, xs...; exec = OnBackend(get_backend(out))) -> (kernel, args, ndrange)

The one launch [`pointwise!`](@ref)`(f, out, xs...; exec)` makes, unlaunched: the KernelAbstractions kernel instance,
its arguments and its `ndrange`, so `kernel(args...; ndrange)` is that call. `pointwise!` is exactly this followed by
the launch (none for an empty `out`). Its use is to ask about the launch without restating how it is built:
`device_resident_threads(pointwise_kernel(f, out, xs...)...)` counts the threads of the very kernel `pointwise!` runs
(the functor adapted to the backend and given to the kernel as its fields or whole, the kernel compiled for those arguments). The
arguments are checked as `pointwise!` checks them (`DimensionMismatch`).
"""
function pointwise_kernel(f::F, out::AbstractArray, xs::AbstractArray...; exec::OnBackend = OnBackend(KernelAbstractions.get_backend(out))) where {F}
    all(x -> axes(x) == axes(out), xs) ||
        throw(DimensionMismatch("pointwise!: the inputs' axes $(map(axes, xs)) are not the output's $(axes(out))"))
    return _pointwise_launch(exec, Adapt.adapt(exec.backend, f), out, xs)   # the functor's own adapt rules, before it is taken apart
end

# OnBackend's kernels. A per-point functor reaches the kernel as its FIELDS, separate kernel arguments, and is rebuilt on the device
# (`_device_functor`): one aggregate argument holding every field makes the device compiler copy it to local memory once any of its
# parts is used by reference — RadiometricCamera's per-pixel flux functor (2.3 KB of fields) needed 2800 bytes of local memory as
# one argument against 2456 as its fields, and ran 5 % slower. A functor whose type cannot be rebuilt from its fields' device types
# (`_splattable`), or with more fields than the kernels take, goes as one argument. The functor is first adapted to the backend as
# a whole (`Adapt.adapt(backend, f)`, its own rules), and the launch then converts each field as it converts any argument.
const _MAX_FIELDS = 16

# The type variables of `F`'s declaration, in order, and the declared type of each field (a TypeVar or a type).
function _declaration(::Type{F}) where {F}
    vars = TypeVar[]; w = F.name.wrapper
    while w isa UnionAll
        push!(vars, w.var); w = w.body
    end
    return vars, [fieldtype(w, i) for i in 1:fieldcount(F)]
end
_occurs(v::TypeVar, T) = T === v || (T isa DataType && any(p -> _occurs(v, p), T.parameters)) ||
                         (T isa UnionAll && (_occurs(v, T.var.ub) || _occurs(v, T.body))) || (T isa Union && (_occurs(v, T.a) || _occurs(v, T.b)))
# Every field is declared as one of `F`'s type variables, used by no other field's declaration, or holds an isbits value (which
# the device conversion keeps): then the device type is `F` with those variables bound to the fields' device types.
Base.@assume_effects :foldable function _splattable(::Type{F}) where {F}   # a property of the type: folded at compile time
    (isconcretetype(F) && !ismutabletype(F) && fieldcount(F) <= _MAX_FIELDS) || return false
    vars, decl = _declaration(F)
    for (i, d) in enumerate(decl)
        if d isa TypeVar && d in vars
            any(k -> k != i && decl[k] !== d && _occurs(d, decl[k]), eachindex(decl)) && return false
        else
            isbitstype(fieldtype(F, i)) || return false
        end
    end
    return true
end
# The functor of host type `F` from its fields' device values: each field-declared type variable bound to its field's device type,
# the others kept.
@generated function _device_functor(::Type{F}, fields...) where {F}
    vars, decl = _declaration(F)
    params = collect(Any, F.parameters)
    for (i, d) in enumerate(decl)
        d isa TypeVar && (j = findfirst(v -> v === d, vars); j === nothing || (params[j] = fields[i]))
    end
    T = isempty(params) ? F : F.name.wrapper{params...}
    return Expr(:new, T, (:(fields[$i]) for i in 1:fieldcount(F))...)
end
# Each field argument is annotated with its own type parameter (`a1::A1`): a field that is a function (`Fix1(project, lens)`'s
# `project`) is otherwise only handed to `_device_functor`, never called, and Julia compiles such an argument for `Function`;
# KernelAbstractions' CPU backend calls the kernel through that widened instance, so every point paid a dynamic
# `_device_functor` call and a boxed functor (≈380 bytes a point: OpenCV-5 undistort at 1024 points 1.8× the serial
# broadcast in one task, at 10⁶ points over 224 threads 15× the threaded broadcast). A device compiler specializes on the
# concrete types anyway: the device kernel is unchanged.
for n in 0:_MAX_FIELDS
    as = [Symbol(:a, i) for i in 1:n]
    Ts = [Symbol(:A, i) for i in 1:n]
    @eval @kernel function $(Symbol(:_pointwise_fields_kernel_, n, :!))(out, xs, ::Type{F}, $((:($a::$T) for (a, T) in zip(as, Ts))...)) where {F, $(Ts...)}
        I = @index(Global, Cartesian)
        f = _device_functor(F, $(as...))
        @inbounds out[I] = f(map(x -> @inbounds(x[I]), xs)...)
    end
end
const _POINTWISE_FIELDS_KERNELS = Tuple(getfield(@__MODULE__, Symbol(:_pointwise_fields_kernel_, n, :!)) for n in 0:_MAX_FIELDS)
# The functor as one argument.
@kernel function _pointwise_kernel!(out, xs, f)
    I = @index(Global, Cartesian)
    @inbounds out[I] = f(map(x -> @inbounds(x[I]), xs)...)
end
# A generated function's body for the tuple of the first `n` fields of its argument `x`
# (a lens leaf's parameter fields, cameras/lenses.jl; a functor's fields, below).
_field_tuple_expr(n::Integer) = Expr(:tuple, (:(getfield(x, $i)) for i in 1:n)...)
# The fields of a functor, as a tuple of exactly its field count.
@generated _fields(x) = _field_tuple_expr(fieldcount(x))
# The kernel, arguments and ndrange of a pointwise launch: the functor as its fields (`_splattable`), else whole.
_pointwise_launch(p::OnBackend, f::G, out, xs) where {G} = _pointwise_launch(Val(_splattable(G)), p, f, out, xs)
_pointwise_launch(::Val{true}, p::OnBackend, f::G, out, xs) where {G} =
    (fs = _fields(f); (p(_POINTWISE_FIELDS_KERNELS[length(fs) + 1]), (out, xs, G, fs...), size(out)))
_pointwise_launch(::Val{false}, p::OnBackend, f, out, xs) = (p(_pointwise_kernel!), (out, xs, f), size(out))
