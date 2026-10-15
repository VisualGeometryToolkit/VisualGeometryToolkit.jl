## Units

Units are FlexUnits.jl with a pixel dimension. The stack's unit code lives in
VisualGeometryCore's `src/base/units.jl`: FlexUnits' documented
custom-dimension recipe — the dimension struct `PixelDimensions` (the seven SI
base dimensions plus `px`) and the registry module `PixelUnits`
(`registry_defaults!`, plus `dpi = px/inch`, `pt = inch/72` and `pd = px`) —
and the stack's one guarded parse. Parsing (`u"…"`, `ud"…"`, `uparse`,
`qparse`), printing, conversion and dimension errors are FlexUnits'.

**Unit names from outside** (a file, a QR code, Python) are read with
`parse_unit(s)` — one registered symbol, or with `compound = true` a product,
quotient or power of them (`"mm/px"`) — and written with `unit_string(u)`,
FlexUnits' printing of a registered unit. `uparse` evaluates what it parses,
so nothing else calls it. A dimension check is `dimension(q) == dimension(u)`.
`check_naming` enforces this everywhere (rule R11): no package outside the
units module calls `uparse`/`qparse`, reaches into the registry, writes a
table of unit names, converts with `convert(VgcQuantity…, q)` (FlexUnits'
`ubase`), branches `x isa Quantity ? ustrip(…)` (`strip_or_bare(u, x)`) or
writes a conversion factor such as 25.4; what FlexUnits lacks, or arithmetic
that must stay bit-identical to stored values, is allowed by file with its
reason (`_UNIT_RULE_ALLOW`).

**Values are stored in base units.** A length is held in metres, a pixel count
in px, so a bare `ustrip(q)` of a length is in metres. Code that wants a unit
names it: `ustrip(mm, q)`, `ustrip(px, q)`, `ustrip(dpi, q)`. A bare `ustrip`
is correct only on a pixel quantity or a dimensionless ratio. Feature
geometry (`Disc`, `Circle`, …) strips through one boundary that names its
units: pixels in px, lengths in mm.

**Dynamic quantities in structs.** The exported constants (`px`, `mm`, `cm`,
`μm`, `m`, `inch`, `ft`, `pt`, `dpi`) are the registry's dynamic units, so
`10mm`, `3px` and `300dpi` share the concrete type `VgcQuantity{Float64}`
(`Quantity{Float64, PixelDimensions{FixRat32}}`). A struct that holds
quantities stores that type and needs no unit type parameter, and a
mixed-dimension container is concrete. Static units (`PixelUnits.@u_str`)
are not used in struct fields.

**Boundaries accept any quantity and convert.** A quantity combined with a
unit keeps that unit (`5.86μm / px` is a `Quantity{Float64, Units{…}}`), and
`uconvert` returns that form too. Public signatures therefore take
FlexUnits' `Quantity`, and a constructor that stores one converts it to
base units with FlexUnits' `ubase(q)` (`ubase.(s)` for a `Size2`, which
FlexUnits broadcasts over as it does `ustrip.(u, s)`). A pixel–length crossing is a product with
a density (`q * 300dpi` is in px); there is no density-taking conversion
verb. A dimension mismatch raises `DimensionError` (arithmetic) or
`ConversionError` (`ustrip`/`uconvert` into a unit of the wrong dimension).
Angles are plain radians.
