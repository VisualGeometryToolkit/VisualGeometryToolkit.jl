# =============================================================================
# experiments/paper_support.jl — what the BlobBoards bmvc2026 and CaliBlobs
# icra2026 paper projects each restated: the active-project assertion every
# driver opens with, and a hand-eye calibration's two records (bmvc2026 writes
# them, icra2026 reads them).
# =============================================================================
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: EuclideanMap, BodyFrame, CameraFrame, ViewFrame, TargetFrame, Rig, PosedTarget, PlanarTarget,
    relabel, unlabel, with_unit, geom, mm, ustrip

"""
    assert_active_project(marker) -> nothing

That DrWatson's `projectdir()` is the paper project the driver belongs to: the
file `marker` (relative to the project, e.g. `"src/BlobBoardsEval.jl"`) exists
there. `projectdir()` resolves from the ACTIVE project, which survives any file
move but is right only when the paper project is the active one; a driver run
under the parent package's project fails here, naming both, instead of
resolving its paths into the wrong tree.
"""
function assert_active_project(marker::AbstractString)
    isfile(projectdir(marker)) && return nothing
    error("""
          DrWatson's projectdir() resolves to $(projectdir()), which holds no $(marker): not the paper project.
          Run the driver with `--project=<the paper project>`; the active project is $(Base.active_project()).
          """)
end

"""
    hand_eye_mount_file(file) -> String

The marker-mount record beside a hand-eye record `file`: `<stem>_board_rig<ext>`.
"""
hand_eye_mount_file(file::AbstractString) = string(splitext(file)[1], "_board_rig", splitext(file)[2])

const _CAMERA_RIG = BodyFrame{:camera_rig}
const _BOARD_RIG = BodyFrame{:board_rig}

"""
    save_hand_eye(file, X, Y; provenance = (;)) -> (file, mount_file)

A hand-eye calibration as its two records: `file`, a `vgio/transforms` record
of `X` (`body:camera_rig → camera_optical`), and [`hand_eye_mount_file`](@ref)`(file)`,
a `vgio/rig` record of the marker mount `Y` (target id → its pose `target:<id>
→ body:board_rig`, each target a pose-only `PlanarTarget` named by its id).
`X` and `Y`'s poses are plain millimetres (an `Uncertain` pose keeps its
covariance); `provenance` goes on the transforms record.
"""
function save_hand_eye(file::AbstractString, X, Y::AbstractDict; provenance::NamedTuple = (;))
    mount = hand_eye_mount_file(file)
    VgtkIO.save(file, [with_unit(relabel(_CAMERA_RIG, CameraFrame, unlabel(X)), mm)]; provenance)
    rig = Rig{_BOARD_RIG}(Dict(Symbol(id) => PosedTarget(PlanarTarget(Symbol(id)), with_unit(T, mm)) for (id, T) in Y))
    VgtkIO.save(mount, rig; provenance = (hand_eye = basename(file),))
    return file, mount
end

"""
    load_hand_eye(file) -> (X, mount)

The hand-eye record pair [`save_hand_eye`](@ref) writes: `X`, the camera-mount
pose `body:camera_rig → ViewFrame{:optical}`, and `mount`, the marker mount as a
`Rig{BodyFrame{:board_rig}}`, both in plain millimetres, each pose `Uncertain`
where the record states its covariance.

A record without the mount file — the 2026-09 hand-eye record, one
`vgio/transforms/v2` file that held the marker mount too, which
VisualGeometryToolkitIO reads as the current transforms record — takes its mount
from the file's `target:<id> → body:board_rig` poses. [`save_hand_eye`](@ref)
writes the pair.
"""
function load_hand_eye(file::AbstractString)
    poses = VgtkIO.load(Vector{EuclideanMap}, file)
    ix = findfirst(p -> geom(p) isa EuclideanMap{3, <:Any, <:Any, <:Any, _CAMERA_RIG, CameraFrame}, poses)
    ix === nothing && error("no body:camera_rig -> camera_optical record in $file")
    X = relabel(_CAMERA_RIG, ViewFrame{:optical}, ustrip(mm, poses[ix]))
    mount = isfile(hand_eye_mount_file(file)) ? ustrip(mm, VgtkIO.load(Rig, hand_eye_mount_file(file))) : _mount_of(poses, file)
    mount isa Rig{_BOARD_RIG} || error("$(hand_eye_mount_file(file)): the marker mount is posed in body:board_rig, got $(typeof(mount))")
    return X, mount
end

# The marker mount a one-file record states among its transforms: every `target:<id> → body:board_rig` pose,
# each target a pose-only `PlanarTarget` named by its id (as `save_hand_eye` writes the mount file).
function _mount_of(poses, file)
    targets = Dict{Symbol, Any}()
    for p in poses
        g = geom(p)
        g isa EuclideanMap{3, <:Any, <:Any, <:Any, <:TargetFrame, _BOARD_RIG} || continue
        id = _target_id(g)
        targets[id] = PosedTarget(PlanarTarget(id), with_unit(unlabel(ustrip(mm, p)), mm))
    end
    isempty(targets) && error("no $(hand_eye_mount_file(file)) and no target:<id> -> body:board_rig record in $file: the record has no marker mount")
    return ustrip(mm, Rig{_BOARD_RIG}(targets))
end
_target_id(::EuclideanMap{3, <:Any, <:Any, <:Any, TargetFrame{id}}) where {id} = id
