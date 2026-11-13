# Content Identity

An observations file (`vgio/observations`, `vgio/target_homographies`) names
the target definitions it indexes by their *content identity*: the
`content_id` of a rig or of a single unposed target. This page specifies it
completely, so any language can recompute it from a stored `vgio/rig` or
`vgio/target` file; the package's test suite does so in Python
(`test/python/content_id.py`, an independent implementation of this page).

```
content_id = lowercase hex of SHA-256( UTF-8( JCS( identity tree ) ) )
```

- **JCS** is RFC 8785, the JSON Canonicalization Scheme: object members sorted
  by their names' UTF-16 code units, no whitespace, strings escaped minimally
  (`\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`, any other character below U+0020
  as `\u00xx` in lowercase hex, everything else as it is) and every number in
  ECMAScript's `Number.prototype.toString` form of its IEEE-754 double: the
  shortest digits that round-trip, without an exponent for a decimal exponent
  in −6 < n ≤ 21 (`1.0` is `1`, `1e-5` is `0.00001`, `-0.0` is `0`) and as
  `d.ddde±x` outside it (`1e+21`, `5e-7`). Julia's writer is
  VisualGeometryToolkitCore's `canonical_json` (re-exported here).
- **The identity tree** is the record's BODY — what follows the envelope — in
  base units, under the record's current key names. No `schema`, `generator`,
  `units`, `conventions` or `provenance` enters it, so the identity does not
  change when the file format does, nor with the unit or pixel frame a file is
  written in.

## The identity tree of a rig

```
{"frame": <the rig's frame name>,
 "sensors": [{"id", "camera", "extrinsics"}, …],          in the rig's order (absent: no sensors)
 "targets": [{"id", "target", "pose"}, …]}                sorted by id, UTF-8 bytes (absent: no targets)
```

- `frame` and each `id` are the strings the record holds.
- `camera` is the sensor's camera body (below), `extrinsics` and `pose` are
  poses (below).

A single target's identity tree is its target body alone (below); it has no
`frame`, so it differs from every rig's. A target held with a
`calibration_record` (who calibrated it) has the identity of the target
without it.

## Numbers in base units

Every number is converted from the file's declared units to base units:

| Quantity | In the tree |
|---|---|
| a length (a target point, a translation, a node, a board, canvas or texture size or origin, a placement, a pixel pitch, a non-central camera's `point` control) | metres, then **normalised** (below) |
| a covariance entry | the unit of its entry: `units.length` to the power of the entry's length dimension (the sum of its row's and column's), in metres to that power, then **normalised** — every entry, a dimensionless one (a pose's rotation block) included |
| a pixel coordinate (a principal point `cx`, `cy`; a lens coverage's centre or origin) | the canonical `colmap` pixel frame: the file's number as a double, plus the frame's offset in double arithmetic (`opencv` +0.5, `julia` −0.5, `colmap` 0) |
| anything else (a rotation, an angle, a focal length in pixels, a lens coefficient, a dimensionless calibration parameter, an image size, a lattice's `area`, `grid_size`, `present` and `direction` controls) | the double of the file's number, as it is |

A length in a power-of-ten unit (`m`, `cm`, `mm`, `μm`/`um`, `km`) is the
file's decimal with its exponent shifted (`62.8` mm is the decimal `0.0628`),
rounded once to the nearest double: never the double of the file number
multiplied by a scale, which rounds twice. A length in another unit (`inch`,
`pt`) is the unit's exact rational scale applied, rounded once.

**Normalisation.** A length or a covariance entry enters as
`normalise(x) = the double nearest to x rounded to 15 significant decimal
digits` (C's `strtod(sprintf("%.14e", x))`; 0 and ±0 stay as they are). Every
decimal number of at most 15 significant digits in a file, in any decimal unit,
reads back as exactly such a value, so a rig, the file written from it in any
decimal unit, the rig loaded back from that file and a value built in code from
the same decimals (`14.2mm`, one ulp off the nearest double of 0.0142 m) have
one identity. Normalisation is of the metre value itself: no unit is chosen,
and it moves a value by at most one rounding at the 15th digit.

## Poses

```
{"rotation": [w, x, y, z], "translation": [x, y, z], "covariance": …}
```

The rotation is the unit quaternion the record stores (Hamilton, w ≥ 0), as it
is: an identity is minted from what is saved — a rotation held as a matrix
enters as the quaternion the writer stores for it. The translation is in
metres, normalised. The `covariance`, present iff the pose is uncertain, is in
the storage the record holds (an array of rows, `{"diagonal": […]}` or
`{"isotropic": λ}`) over [δω; δt], rotation first: entry (i, j) has length
dimension dᵢ + dⱼ with d = (0, 0, 0, 1, 1, 1).

## Target bodies

A `point_target`:

```
{"type": "point_target", "points": [{"xy": [x, y]} | {"xyz": [x, y, z]}, "covariance"?: …], …]}
```

each coordinate in metres (normalised), each point covariance of length
dimension 2 in every entry.

A `planar_target`:

```
{"type": "planar_target",
 "reference": <the design's name>  |  "nominal": [[x, y], …],     its design
 "calibration": {…},
 "board"?: {"size": {"width", "height"}, "origin": [x, y]},
 "canvas"?: {"size", "origin", "texture"?: {"path", "sha256", "size", "origin"}},
 "placement"?: {"origin": [x, y], "side": s}}
```

- The design enters by its `reference` (a design whose points are its owner's)
  or by its nominal points, in metres. A target built from a **layout** (an
  AprilGrid, a Schöps pattern) enters by the nominal points VGC derives from
  the layout (`design(t)`), not by the layout record: its identity is
  geometric, the same as a nominal target's of the same points. (Recomputing
  it outside Julia needs the layout's geometry.)
- Board, canvas, texture and placement lengths are in metres, normalised; the
  texture's `path` and `sha256` are strings.

## Target calibrations

```
{"type": <tag>, "nodes"?: [[x, y], …], "parameters"?: […], "covariance"?: …}
{"type": "composition", "compose": [<member>, …], "covariance"?: …}       members outer first
```

A calibration kind's parameters are lengths or not by
VisualGeometryToolkitCore's `length_mask` of the kind:

| `type` | parameters | lengths |
|---|---|---|
| `rigid` | none | — |
| `euclidean` | `[θ, tx, ty]` | `tx`, `ty` |
| `scale` | `[s]` | none |
| `aspect` | `[a]` | none |
| `skew` | `[k]` | none |
| `thin_plate_spline` | the K nodes' x displacements, then their y displacements | all; its `nodes` too |
| `height_field` | the K nodes' heights | all; its `nodes` too |

A length parameter and every node coordinate enter in metres, normalised; a
dimensionless parameter enters as it is. The `covariance` is over the
parameters in their order (a composition's: its members' parameters
concatenated, outer member first), entry (i, j) of length dimension
`mask[i] + mask[j]`, every entry normalised.

A **placed** target (one with a `placement`) holds its calibration in the
placement's dimensionless chart: its nodes, parameters and covariance enter as
the record's numbers, unconverted and unnormalised.

## Camera bodies

```
{"image_size": {"width", "height"},
 "K": {"fx", "fy", "cx", "cy", "skew"},          cx, cy in the colmap pixel frame
 "lens": <VGC's lens record>                     as it is
 | "type": "noncentral_spline", "spline": {"area", "grid_size", "present"?, "controls": {"direction", "point"}},
 "pitch"?: {"width", "height"},                   metres per pixel, normalised
 "lens_coverage"?: {"circle": [cx, cy, r]} | {"ellipse": [cx, cy, a, b, θ]} | {"rect": [x, y, w, h]}}
```

A lens coverage's centre (or a rect's origin) is moved into the colmap frame;
its radius, semi-axes, angle and extent are as they are. A non-central
camera's `point` controls are lengths (metres, normalised); its `direction`
controls and its lattice are as they are.

## Changing the tree

A change to anything on this page re-ids every rig and target, and the
observation files that name them must be repointed: `vgio_migrate_tree` does
that, keeping the map of old to new identities. IO 0.10 made one such
deliberate re-id (from the 0.9 tree, which kept the rig v5 names and Julia's
float printing): JCS numbers, the current record names, and this
specification in metres.
