# Ray Maps and Frame Transitions

```@meta
CurrentModule = VisualGeometryCore
```

A central lens does not "distort 2D coordinates" — it **re-orients rays**.
The familiar 2D distortion polynomial of a lens model is the conjugation

```
section ∘ D ∘ lift
```

of a self-map `D: S² → S²` of the view's spherical retina, expressed in
the model's defining frame (`lift` takes a point of that 2D frame onto the
sphere, `section` takes a ray back into it). Since 0.11,
[`ray_map`](@ref) exposes `D` itself.

## `ray_map` — the sphere self-map

```julia
D = ray_map(lens)          # RayMap{ViewFrame{:_}, C, typeof(lens)}
v′ = D(v)                  # any ray representative in; unit bearing vector out
inv(D)                     # InverseRayMap — the undistort conjugation
```

[`RayMap`](@ref) is the framed stage `{ViewFrame{id} → ViewFrame{id}}`:
both endpoints are the view's 3D frame, so `D` needs no 2D frame tag — the
type parameter `C` only records which defining frame the conjugation runs
through (`section` then the in-frame distortion then `lift`). Input norm is
irrelevant (the frame projections are scale-invariant on ray
representatives); output is always the unit bearing vector.

**Why `D` has an honest `inv` when the projections do not.** Restricted to
the sphere, the retinal-frame projections are *sections* — invertible —
unlike the fiber-collapsing full 3D → 2D projections (`FramedPerspective`
and friends), which deliberately have no `inv`. So
[`InverseRayMap`](@ref) = `lift ∘ undistort ∘ section` is a genuine
inverse, minted by `inv` and inverting back.

For identity-in-frame lenses (`Pinhole`, the pure fisheye projections), `D`
is the identity — still framed, still returning the unit representative.
`D` exists only for **central** lenses (every VisualGeometryCore lens model):
a finite or thick pupil has no sphere self-map, because its rays do not
share a center — that regime is ray tracing, downstream.

## Honest domains

The conjugation is defined exactly on the defining frame's domain. Rays
outside it throw a `DomainError` — not a sentinel, not a silently aliased
value:

- gnomonic ([`GnomonicFrame`](@ref)): forward hemisphere `Z > 0` only;
- equidistant / stereographic: every ray but the exact backward direction;
- equisolid: the whole sphere;
- orthographic: `θ ∈ [0, π/2]` **only**.

### The orthographic fold — the cautionary example

The orthographic frame radius `r = sin θ` rises to 1 at the equator and
then *decreases*: past 90° every radius value is taken twice. The frame
**folds** — a ray with `Z < 0` is not "a large radius", it is *outside the
frame's domain*, and the section/lift machinery throws rather than aliasing
it onto a forward ray (the lift always selects the forward sheet
`cos θ = +√(1 − r²)`). This is the poster child for the honest-domain
discipline: the alternative is a silently wrong forward ray.

## `frame_transition` — first-class frame hops

The transition between any two retinal frames of a view is a pure radial
reparameterization of the sphere. [`frame_transition`](@ref)`(C1, C2)`
mints it as the framed stage [`FrameTransition`](@ref)`{C1, C2}`,
implemented once through the sphere (`section(Dst) ∘ lift(Src)`) — every
pair among the five frames is served by one implementation, with no
per-pair closed forms to keep consistent:

```julia
t = frame_transition(EquidistantFrame{:_}, GnomonicFrame{:_})
t(Vec2(θ, 0.0))            # == Vec2(tan(θ), 0.0)
inv(t)                     # the reverse transition
```

This names what production code usually hides inline: OpenCV's
`cv::fisheye::undistortPoints` ends with the line `scale = tan(θ)/θ_d` —
that line *is* `frame_transition(EquidistantFrame{id}, GnomonicFrame{id})`,
the equidistant → gnomonic frame change applied after the polynomial
θ-inverse, unnamed and unchecked in the C++. Here it is a framed stage with
checked junctions (the view ids must match — a transition between frames of
different views is a `MethodError`) and honest domain edges. Being isbits,
a `FrameTransition` composes into a per-point map that [`pointwise!`](@ref)
runs over a batch, on any backend.

## See also

- [The Framed Imaging Chain](imaging-chain.md) — the frame table and stage family
- [Back-projection and Undistortion](undistortion.md) — the lens contract and its inverse maps
- [Imaging Chain and Stereo API](../api/imaging.md)
