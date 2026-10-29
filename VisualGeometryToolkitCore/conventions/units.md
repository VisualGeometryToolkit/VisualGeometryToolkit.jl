## Units

Units are FlexUnits.jl: no stack quantity type or dimension set. The stack's
unit code lives in VisualGeometryToolkitCore's `src/base/units.jl`: its own
unit registry, `StackUnits`, built with FlexUnits' `RegistryTools` exactly as
FlexUnits documents for a registry of another transform type — a
`PermanentDict` of `Units{Dimensions{FixRat32}, AffineTransform{Rational{Int}}}`
filled by `register_unit!` and exposed by `@generate_registry_exports` — the
unit constants, `base_quantity`, and the stack's one guarded parse. Parsing
(`ud"…"`, `uparse`, `qparse`), printing, conversion and dimension errors are
FlexUnits'.

**Scales are exact rationals.** Every registered scale is written as an exact
ratio of base units and handed to `register_unit!` as data: `mm = 1//1000`,
`cm = 1//100`, `μm = 1//1000000`, `km = 1000`, `inch = 127//5000` (25.4 mm),
`ft = 12 inch`, `pt = inch/72`, `px` and `pd` dimensionless with scale 1,
`dpi = px/inch`, and the SI base units (m, kg, s, A, K, cd, mol) at 1. The
registry holds those and nothing else. FlexUnits' `registry_defaults!` is not
called: in a Rational registry it converts its Float64 scales to their binary
values (`Rational(0.001)` = 1152921504606847//1152921504606846976). So a length
built from an integer is exact — `142mm` is 71//500 m, whose Float64 is the
decimal-nearest 0.142 (FlexUnits' Float64 `mm` gave 0.14200000000000002) —
and a float literal is FlexUnits' promotion, `142.0mm` = `142.0 * 0.001`,
correctly rounded since FlexUnits 0.6.16 (Deduction42/FlexUnits.jl#131: a float
times a rational scale as `x * num / den`) — which rounds the BINARY value of the float, so a decimal read from text can still land one ulp from
the decimal's nearest double; VisualGeometryToolkitIO's `read_length`/`write_length` (decimal-exact,
by exponent shift) remain the stack's one documented exception for that case. An exact quantity
compares exactly: `ustrip(m, 142mm) == 0.142` is false (the double 0.142 is
not 71/500), while `ustrip(m, base_quantity(142mm)) === 0.142`.

### The layering: units at the boundary, plain numbers inside

Units live at the public boundary and plain numbers live inside. A value moves
from one to the other where the frames rule already conditions data: once at
an estimator's entry and once at its exit.

| Layer | Holds |
|---|---|
| Public values (held by users, handed between packages, persisted): `EuclideanMap` translations, `PlanarTarget` and `PointTarget` points, `Rig` target poses, `Trajectory`, `TargetPoses`, `TargetPoints3D`, sensor and board sizes | FlexUnits quantities for every physical length, in metres; a dimensionless quantity where a public value has no unit |
| IO | a file declares its `length_unit`; a reader returns quantities, and a writer writes numbers in the declared unit |
| Estimator, renderer and solver entry and exit (`fit_*`, `find_*`, `refine_*`, a one-step algorithm's public form) | one strip into the private working unit at entry, one reattachment at exit |
| Inner loop (minimal kernels, `*_step` updates, `!` workspace forms, RANSAC internals, LM rows and charts, GPU kernels) | plain numbers of eltype `T` (`Float64`, `Float32`, a ForwardDiff `Dual`) |

Pixel coordinates keep their frame tags, because `PixelFrame` is their unit.
`px` quantities stay for sizes and densities.

A physical value has two forms, and both are bits types:

| Form | Element type | Where |
|---|---|---|
| base form, in metres | `Quantity{T, Dimensions{FixRat32}}` | public values: poses, points, sizes |
| plain | `T` (`Float64`, `Float32`, `Dual`) | inner loops and GPU kernels |

The plain form of a value is the whole value's `ustrip(u, X)`, for a pose, a
target, a rig or a capture. It is never a value rebuilt from stripped
components. A unit is attached by construction, `EuclideanMap(R, t .* m)`, or
by `with_unit(X, u)`, which reads a plain value — a map included, its rotation
kept exactly — as being in `u`. One exception
is forced by its type: [`NoncentralGenericModel`](@ref) holds its line points
as plain metres, because its grid of `Vec3{T}` cannot hold quantities, and its
docstring says so. Because a plain extrinsics is "in the caller's unit", a
`PosedCamera` of that model refuses one (`ArgumentError`, naming `t .* m`); its
plain form is `ustrip(m, camera)` only.

Units are allowed in a type but not required by it, so the layering is held at
four points:

1. VisualGeometryToolkitIO's readers always return quantities, through
   `read_length`.
2. Its writers refuse a plain translation or point for a physical kind (`rig`,
   `trajectory`, `target_poses`, `transforms`, `target`, `target_points3d`,
   and the object points of `correspondences`). The `ArgumentError` names the
   fix, `t .* m`.
3. A public estimator that takes physical values strips them in metres at
   entry (`ustrip(m, x)`) and reattaches the unit at exit, so unitful input
   gives unitful output. Plain input gives plain output, in the caller's own
   unit, as before units: estimators are not where plain data is refused, the
   IO writers of point 2 are. A model that cannot record a unit — a plane
   homography, a radial or division camera, the up-to-scale relative pose of
   `bundle_refine_relpose` — refuses unitful input instead, with an
   `ArgumentError` naming the fix (strip the plane points in metres, the unit
   the stack's boundaries use). This supersedes the plan's earlier "a plain
   physical input is refused" (ruling of 2026-10-25, Task 9): refusing plain
   input would break every existing caller, so a later audit must not "fix"
   the code back to it.
4. `check_naming` rule R11 flags a `ustrip(…)` fed into an `EuclideanMap` or a
   target constructor, and a power-of-ten scale in a definition that names a
   file's `length_unit`. Unit names are read only by `parse_unit`.

**A pixel is dimensionless.** FlexUnits' dimensions are the seven SI ones, so
`px` cannot be told from `rad` or a ratio, and `3px + 1.0` is allowed. Pixel
*coordinates* keep their safety through `PixelFrame` tags; only pixel-sized
quantities lose it. A function that takes a pixel-sized quantity accepts a
plain number or a dimensionless quantity, and refuses a length with
`DimensionError`.

**Unit names from outside** (a file, a QR code, Python) are read with
`parse_unit(s)` — one registered symbol, or with `compound = true` a product,
quotient or power of them (`"mm/px"`) — and written with `unit_string(u)`,
FlexUnits' printing of a registered unit. `uparse` evaluates what it parses,
so nothing else calls it. A dimension check is `dimension(q) == dimension(u)`.
`check_naming` enforces this everywhere (rule R11): no package outside the
units module calls `uparse`/`qparse`/`register_unit`, builds a unit or a
registry (`Units`, `AffineTransform`, `PermanentDict`, `register_unit!`,
`registry_defaults!`), reaches into the registry, writes a table of unit
names, converts with `convert(Quantity{…}, q)` or a bare `ubase(q)` in source
(`base_quantity(q)`), branches `x isa Quantity ? ustrip(…)`
(`strip_or_bare(u, x)`) or writes a conversion factor (25.4, 0.0254, 304.8,
0.3048, 127//5000); what FlexUnits lacks, or arithmetic that must stay
bit-identical to stored values, is allowed by file with its reason
(`_UNIT_RULE_ALLOW`). The one hand-rolled conversion is VisualGeometryToolkitIO's
`read_length` (below), until FlexUnits rounds a rational scale once.

**Values are stored in base units.** A length is held in metres, a pixel count
as a plain dimensionless value, so a bare `ustrip(q)` of a length is in metres. Code that wants a unit
names it: `ustrip(mm, q)`, `ustrip(px, q)`, `ustrip(dpi, q)`. A bare `ustrip`
is correct only on a pixel quantity or a dimensionless ratio. Feature
geometry (`Disc`, `Circle`, …) strips through one boundary that names its
units: pixels in px, lengths in mm.

**Dynamic quantities in structs.** The exported constants (`px`, `mm`, `cm`,
`μm`, `m`, `inch`, `ft`, `pt`, `dpi`) are the registry's dynamic units, so
`10.0mm`, `3.0px` and `300.0dpi` share FlexUnits' base-form type
`Quantity{Float64, Dimensions{FixRat32}}`, a bits type; `10mm` (an integer
times an exact scale) is a `Quantity{Rational{Int}, …}` until it is stored. A
struct that holds quantities stores the Float64 form, named once in VisualGeometryToolkitCore as the
computed constant `BaseQuantity = typeof(ubase(1.0mm))` — FlexUnits' stored
form, not a stack quantity type: no signature takes it, every boundary takes
any `Quantity` — and needs no unit type parameter, and a mixed-dimension
container is concrete. Static
units (`u"…"`) are not used in struct fields.

**Boundaries accept any quantity and convert.** A quantity combined with a
unit keeps that unit (`5.86μm / px` is a `Quantity{Float64, Units{…}}`), and
`uconvert` returns that form too. Public signatures therefore take
FlexUnits' `Quantity`, and a constructor that stores one converts it with
`base_quantity(q)` = `float(ubase(q))` (`base_quantity.(s)` for a `Size2`,
which FlexUnits broadcasts over as it does `ustrip.(u, s)`), or by converting
into a `BaseQuantity` field. A `Rational` never reaches a struct field or a
kernel: the stack's strips (`strip_metres`, `strip_or_bare`) return floats. A pixel–length crossing is a product with
a density (`q * 300dpi` is in px); there is no density-taking conversion
verb. A dimension mismatch raises `DimensionError` (arithmetic) or
`ConversionError` (`ustrip`/`uconvert` into a unit of the wrong dimension).
Angles are plain radians.

### Files keep human units

Memory holds metres. A file holds the unit that reads naturally for its
quantity: focal lengths and board sizes in mm, pixel pitch in μm, room-scale
poses in m. The file declares its unit, once per file (`length_unit`) or per
field. VisualGeometryToolkitIO's default file unit is mm.

**Never convert a file length by multiplying by a scale.** The rounding comes
from the multiplication, not from metres. `62.8 * 0.001` is `0.0628`, but
`0.0628 / 0.001` is `62.79999999999999`, and the same happens for a few
percent of typical millimetre values. VisualGeometryToolkitIO converts every
file length with one pair, `read_length(x, u)` and `write_length(q, u)`. For a
power-of-ten unit (mm, cm, μm, km) the pair shifts the decimal exponent, which
is one correctly rounded parse. For another unit (inch, ft, pt) it reads by
the unit's scale and writes the shortest decimal that reads back to the
identical value. A side benefit is that memory values are correctly rounded:
`13.9` mm reads as `0.0139` m.

**What is exact.** A file that VisualGeometryToolkitIO wrote is rewritten byte
for byte when its numbers have at most 15 significant digits. A full-precision
number (16 or 17 digits) may move by at most 2 ulp on its first rewrite, and it
stays fixed after that. This is inherent to metres in memory and millimetres in
files: the two grids differ, so some 17-digit millimetre numbers share their
nearest metre value with a neighbour, and one of the two cannot come back. An identity follows the value, not the bytes:
VisualGeometryToolkitIO's `content_id` of a rig hashes a canonical tree of
the loaded value in metres, so a file, the rig loaded from it and the same
rig written in another unit name the same targets.

**One quantity form.** A JSON record holds each quantity as a number in the
unit it declares (`"units": {"world": "mm", "density": "dpi"}`), through the
pair above; a record whose lengths need more than one unit declares each
further one as a length role (`"lengths": {"pattern": "pt"}`), each number
decimal-exact in its own. The `{"value": v, "unit": u}` objects that
VisualGeometryToolkitFeatures once wrote, and that BlobBoards' boards hold,
are not read: their owner migrates such files once (`vgio_migrate`).

**A float length built in code has already rounded.** `[62.8, 1.3] .* mm`
multiplies by the Float64 of the scale when it is stored in metres. `1.3mm` is
held as `0.0013000000000000002` m, and a file written from it holds the
17-digit text `1.3000000000000003`. An integer length is exact (`142mm`,
`210mm`). To build an exact length from a decimal in code, read it the way a
file is read: `VisualGeometryToolkitIO.read_length(1.3, mm)`.

**Covariances follow the same split.** In memory a covariance is in base units:
for a pose, rad² in the rotation block, m² in the translation block and m·rad
in the cross block. On disk it is in the file's declared `length_unit`,
converted by the same decimal shift raised to each entry's power of length
(mm² for a translation entry, mm·rad for a cross entry in a millimetre file).
