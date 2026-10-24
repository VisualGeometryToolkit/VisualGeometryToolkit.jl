# Getting Started

Everything is read and written through two functions, `load` and `save`,
dispatched on the VisualGeometryToolkitCore type. They are public but not
exported, because FileIO and other packages export functions with the same
names. Call them through a module alias:

```jldoctest quickstart
julia> using VisualGeometryToolkitCore, VisualGeometryToolkitIO

julia> import VisualGeometryToolkitIO as VgtkIO
```

## A pose

A pose is a VGC `EuclideanMap`. Its source and destination frames are type
parameters, and a file names them by those types. Here the pose maps the
camera rig's body frame into the camera's optical frame:

```jldoctest quickstart
julia> X = relabel(BodyFrame{:camera_rig}, CameraFrame,
           lookat(Vec3(0.0, 0.0, 500.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, -1.0, 0.0)));

julia> path = joinpath(mktempdir(), "hand_eye.json");

julia> VgtkIO.save(path, [X]; provenance = (method = "robot-world hand-eye",))

julia> print(read(path, String))
{
    "schema": "vgio/transforms/v3",
    "length_unit": "mm",
    "rotation_encoding": "quaternion_wxyz_hamilton",
    "transform_convention": "x_to = R * x_from + t",
    "transforms": [
        {
            "from": "body:camera_rig",
            "to": "camera_optical",
            "rotation": [
                0.0,
                1.0,
                0.0,
                0.0
            ],
            "translation": [
                0.0,
                0.0,
                500.0
            ]
        }
    ],
    "provenance": {
        "method": "robot-world hand-eye"
    }
}
```

The file states everything a reader needs: the schema and its version, the
length unit, how the rotation is encoded and what the transform does to a
point. The two frames are named by the frame grammar (`body:camera_rig`,
`camera_optical`).

Loading takes the type you expect. Asking for the wrong kind is an error, not
a guess:

```jldoctest quickstart
julia> X2 = only(VgtkIO.load(Vector{EuclideanMap}, path));

julia> X2 isa EuclideanMap{3, Float64, <:Any, <:Any, BodyFrame{:camera_rig}, CameraFrame}
true

julia> X2.R ≈ X.R && X2.t == X.t
true
```

The rotation comes back as the stored quaternion. A pose that was read from a
file and saved again produces the same bytes. The `provenance` keyword is
written to the file but is not part of the value, so `load` does not return
it.

## A camera store

A camera store maps names to `BoundedCamera`s: a `CameraModel` (intrinsics
and lens) with its image size.

```jldoctest quickstart
julia> model = CameraModel(2400.0px, [2120.0, 1416.0] .* px;
           lens = KannalaBrandt(0.1, -0.02, 0.003, -0.0004));

julia> camera = BoundedCamera(model, Size2(4240, 2832));

julia> store = joinpath(mktempdir(), "cameras.json");

julia> VgtkIO.save(store, Dict(:left => camera))

julia> VgtkIO.load(BoundedCamera, store; name = :left) == camera
true

julia> VgtkIO.validate(store)
"cameras"
```

`validate` checks any vgio file against its kind's table and returns the
kind. It runs the same checks the reader and the writer run.

## Next steps

- [Kinds and the I/O Surface](kinds.md) lists every kind and the calls for it.
- [Schema Rules](schemas.md) explains what every file guarantees.
- [Third-Party Calibrations](third-party.md) converts OpenCV and MATLAB
  exports into native files.
