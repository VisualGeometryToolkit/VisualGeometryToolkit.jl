# =============================================================================
# COLMAP binary model I/O (cameras.bin / images.bin / points3D.bin)
# =============================================================================
#
# Byte layout per COLMAP's src/colmap/scene/reconstruction_io.cc. All
# little-endian; COLMAP only ships little-endian builds, so no swapping.

function _write_cameras_bin(path::AbstractString, cameras::Vector{ColmapCamera})
    open(path, "w") do io
        write(io, UInt64(length(cameras)))
        for c in cameras
            write(
                io, Int32(c.camera_id), Int32(model_id(c.model)),
                UInt64(c.width), UInt64(c.height)
            )
            write(io, c.params)
        end
    end
    return nothing
end

function _read_cameras_bin(path::AbstractString)
    return open(path, "r") do io
        n = read(io, UInt64)
        cameras = Vector{ColmapCamera}(undef, n)
        for i in 1:n
            id = read(io, Int32)
            model = model_name(read(io, Int32))
            width = Int64(read(io, UInt64))
            height = Int64(read(io, UInt64))
            params = [read(io, Float64) for _ in 1:model_nparams(model)]
            cameras[i] = ColmapCamera(id, model, width, height, params)
        end
        return cameras
    end
end

function _write_images_bin(path::AbstractString, images::Vector{ColmapImage})
    open(path, "w") do io
        write(io, UInt64(length(images)))
        for im in images
            write(io, Int32(im.image_id))
            foreach(v -> write(io, v), im.qvec)
            foreach(v -> write(io, v), im.tvec)
            write(io, Int32(im.camera_id))
            write(io, im.name, '\0')
            write(io, UInt64(length(im.points2D)))
            for (x, y, id) in im.points2D
                write(io, x, y, Int64(id))
            end
        end
    end
    return nothing
end

function _read_images_bin(path::AbstractString)
    return open(path, "r") do io
        n = read(io, UInt64)
        images = Vector{ColmapImage}(undef, n)
        for i in 1:n
            id = read(io, Int32)
            qvec = canonical_qvec(SVector{4, Float64}(ntuple(_ -> read(io, Float64), 4)))
            tvec = SVector{3, Float64}(ntuple(_ -> read(io, Float64), 3))
            camera_id = read(io, Int32)
            name = readuntil(io, '\0')
            npts = read(io, UInt64)
            pts = Vector{Tuple{Float64, Float64, Int64}}(undef, npts)
            for j in 1:npts
                pts[j] = (read(io, Float64), read(io, Float64), read(io, Int64))
            end
            images[i] = ColmapImage(id, qvec, tvec, camera_id, name, pts)
        end
        return images
    end
end

function _write_points3D_bin(path::AbstractString, points::Vector{ColmapPoint3D})
    open(path, "w") do io
        write(io, UInt64(length(points)))
        for p in points
            write(io, Int64(p.point3D_id))
            foreach(v -> write(io, v), p.xyz)
            foreach(v -> write(io, v), p.rgb)
            write(io, p.error)
            write(io, UInt64(length(p.track)))
            for (img, idx) in p.track
                write(io, Int32(img), Int32(idx))
            end
        end
    end
    return nothing
end

function _read_points3D_bin(path::AbstractString)
    return open(path, "r") do io
        n = read(io, UInt64)
        points = Vector{ColmapPoint3D}(undef, n)
        for i in 1:n
            id = read(io, Int64)
            xyz = SVector{3, Float64}(ntuple(_ -> read(io, Float64), 3))
            rgb = SVector{3, UInt8}(ntuple(_ -> read(io, UInt8), 3))
            err = read(io, Float64)
            ntrack = read(io, UInt64)
            track = Vector{Tuple{Int32, Int32}}(undef, ntrack)
            for j in 1:ntrack
                track[j] = (read(io, Int32), read(io, Int32))
            end
            points[i] = ColmapPoint3D(id, xyz, rgb, err, track)
        end
        return points
    end
end

# =============================================================================
# Model-directory entry points
# =============================================================================

"""
    save(dir, model::ColmapModel; format = :binary)

Write a COLMAP model directory (`cameras`/`images`/`points3D` in `.bin` or
`.txt`) plus the `vgio_units.json` sidecar declaring the unit contract.
"""
function save(dir::AbstractString, model::ColmapModel; format::Symbol = :binary)
    mkpath(dir)
    if format === :binary
        _write_cameras_bin(joinpath(dir, "cameras.bin"), model.cameras)
        _write_images_bin(joinpath(dir, "images.bin"), model.images)
        _write_points3D_bin(joinpath(dir, "points3D.bin"), model.points3D)
    elseif format === :text
        _write_cameras_txt(joinpath(dir, "cameras.txt"), model.cameras)
        _write_images_txt(joinpath(dir, "images.txt"), model.images)
        _write_points3D_txt(joinpath(dir, "points3D.txt"), model.points3D)
    else
        throw(ArgumentError("format must be :binary or :text, got $format"))
    end
    save_units_sidecar(dir, model.contract)
    return nothing
end

"""
    load(ColmapModel, dir; contract = nothing) -> ColmapModel

Read a COLMAP model directory (binary preferred, text fallback). The unit
contract comes from the `vgio_units.json` sidecar when present; for models
produced by other tools supply `contract` explicitly — refusing to guess units
is the point of the sidecar. A structure-from-motion model with no metric
scale (PhotoTourism) is `contract = UnknownScale()`: its translations stay
unitless, and metric assembly refuses (see [`UnknownScale`](@ref)). A caller
contract that contradicts the sidecar is refused.
"""
function load(
        ::Type{ColmapModel}, dir::AbstractString;
        contract::Union{UnitContract, UnknownScale, Nothing} = nothing
    )
    sidecar = load_units_sidecar(dir)
    if contract === nothing
        contract = sidecar
        contract === nothing && throw(
            ArgumentError(
                "$dir has no $(UNITS_SIDECAR) sidecar — supply `contract` to declare " *
                    "the units this model's magnitudes are expressed in"
            )
        )
    elseif sidecar !== nothing && sidecar != contract
        throw(ArgumentError("caller contract $(contract) disagrees with $(sidecar) declared by $(UNITS_SIDECAR) in $dir"))
    end
    if isfile(joinpath(dir, "cameras.bin"))
        return ColmapModel(
            _read_cameras_bin(joinpath(dir, "cameras.bin")),
            _read_images_bin(joinpath(dir, "images.bin")),
            _read_points3D_bin(joinpath(dir, "points3D.bin")),
            contract
        )
    elseif isfile(joinpath(dir, "cameras.txt"))
        return ColmapModel(
            _read_cameras_txt(joinpath(dir, "cameras.txt")),
            _read_images_txt(joinpath(dir, "images.txt")),
            _read_points3D_txt(joinpath(dir, "points3D.txt")),
            contract
        )
    end
    throw(ArgumentError("no COLMAP model found in $dir"))
end
