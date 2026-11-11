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

The launch of [`pointwise!`](@ref)'s one kernel on `backend` itself: the
backend value, so its compiler options hold (`CUDABackend(; always_inline =
true)`), and `workgroupsize` fixes the launch's workgroup (`nothing`: the
backend's choice). Without it the kernel runs on `get_backend(out)` with the
defaults. A functor whose fields are each one of its type parameters or an
isbits value reaches the kernel as its fields, rebuilt on the device: one large
argument costs local memory there.
"""
struct OnBackend{B <: KernelAbstractions.Backend, W}
    backend::B
    workgroupsize::W
end
OnBackend(backend::KernelAbstractions.Backend; workgroupsize = nothing) = OnBackend(backend, workgroupsize)

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
pointwise!(f::F, out::AbstractArray, xs::AbstractArray...; exec::OnBackend = OnBackend(KernelAbstractions.get_backend(out))) where {F} =
    _pointwise!(exec, f, out, xs...)

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
function _splattable(::Type{F}) where {F}
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
_launch(p::OnBackend, k) = p.workgroupsize === nothing ? k(p.backend) : k(p.backend, p.workgroupsize)
function _pointwise!(p::OnBackend, f::F, out, xs...) where {F}
    all(x -> axes(x) == axes(out), xs) ||
        throw(DimensionMismatch("pointwise!: the inputs' axes $(map(axes, xs)) are not the output's $(axes(out))"))
    isempty(out) && return out
    _pointwise_on!(p, Adapt.adapt(p.backend, f), out, xs)   # the functor's own adapt rules, before it is taken apart
    return out
end
# A generated function's body for the tuple of the first `n` fields of its argument `x`
# (a lens leaf's parameter fields, cameras/lenses.jl; a functor's fields, below).
_field_tuple_expr(n::Integer) = Expr(:tuple, (:(getfield(x, $i)) for i in 1:n)...)
# The fields of a functor, as a tuple of exactly its field count.
@generated _fields(x) = _field_tuple_expr(fieldcount(x))
function _pointwise_on!(p::OnBackend, f::G, out, xs) where {G}
    if _splattable(G)
        fs = _fields(f)
        _launch(p, _POINTWISE_FIELDS_KERNELS[length(fs) + 1])(out, xs, G, fs...; ndrange = size(out))
    else
        _launch(p, _pointwise_kernel!)(out, xs, f; ndrange = size(out))
    end
    return out
end
