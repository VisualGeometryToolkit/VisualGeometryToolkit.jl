# =============================================================================
# COLMAP text model I/O (cameras.txt / images.txt / points3D.txt)
# =============================================================================

function _write_cameras_txt(path::AbstractString, cameras::Vector{ColmapCamera})
    open(path, "w") do io
        println(io, "# Camera list with one line of data per camera:")
        println(io, "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]")
        println(io, "# Number of cameras: $(length(cameras))")
        for c in cameras
            print(io, c.camera_id, ' ', c.model, ' ', c.width, ' ', c.height)
            foreach(p -> print(io, ' ', p), c.params)
            println(io)
        end
    end
    return nothing
end

function _read_cameras_txt(path::AbstractString)
    cameras = ColmapCamera[]
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, '#')) && continue
        f = split(line)
        push!(
            cameras, ColmapCamera(
                parse(Int32, f[1]), Symbol(f[2]),
                parse(Int64, f[3]), parse(Int64, f[4]),
                parse.(Float64, f[5:end])
            )
        )
    end
    return cameras
end

function _write_images_txt(path::AbstractString, images::Vector{ColmapImage})
    open(path, "w") do io
        println(io, "# Image list with two lines of data per image:")
        println(io, "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME")
        println(io, "#   POINTS2D[] as (X, Y, POINT3D_ID)")
        println(io, "# Number of images: $(length(images))")
        for im in images
            print(io, im.image_id)
            foreach(v -> print(io, ' ', v), im.qvec)
            foreach(v -> print(io, ' ', v), im.tvec)
            println(io, ' ', im.camera_id, ' ', im.name)
            println(io, join((string(x, ' ', y, ' ', id) for (x, y, id) in im.points2D), ' '))
        end
    end
    return nothing
end

function _read_images_txt(path::AbstractString)
    images = ColmapImage[]
    lines = [l for l in eachline(path) if !startswith(strip(l), '#')]
    i = 1
    while i <= length(lines)
        l = strip(lines[i])
        if isempty(l)
            i += 1
            continue
        end
        # NAME is the REMAINDER of the line, not one token: COLMAP image names
        # are paths and may contain spaces (COLMAP's own reader getlines the
        # tail for exactly this reason). `limit = 10` keeps that tail intact —
        # splitting it would silently truncate the name at the first space.
        f = split(l; limit = 10)
        qvec = canonical_qvec(SVector{4, Float64}(parse.(Float64, f[2:5])))
        tvec = SVector{3, Float64}(parse.(Float64, f[6:8]))
        pts = Tuple{Float64, Float64, Int64}[]
        if i + 1 <= length(lines)
            pf = split(strip(lines[i + 1]))
            for j in 1:3:(length(pf) - 2)
                push!(
                    pts, (
                        parse(Float64, pf[j]), parse(Float64, pf[j + 1]),
                        parse(Int64, pf[j + 2]),
                    )
                )
            end
        end
        push!(
            images, ColmapImage(
                parse(Int32, f[1]), qvec, tvec,
                parse(Int32, f[9]), String(strip(f[10])), pts
            )
        )
        i += 2
    end
    return images
end

function _write_points3D_txt(path::AbstractString, points::Vector{ColmapPoint3D})
    open(path, "w") do io
        println(io, "# 3D point list with one line of data per point:")
        println(io, "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)")
        println(io, "# Number of points: $(length(points))")
        for p in points
            print(io, p.point3D_id)
            foreach(v -> print(io, ' ', v), p.xyz)
            foreach(v -> print(io, ' ', Int(v)), p.rgb)
            print(io, ' ', p.error)
            foreach(t -> print(io, ' ', t[1], ' ', t[2]), p.track)
            println(io)
        end
    end
    return nothing
end

function _read_points3D_txt(path::AbstractString)
    points = ColmapPoint3D[]
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, '#')) && continue
        f = split(line)
        track = [(parse(Int32, f[j]), parse(Int32, f[j + 1])) for j in 9:2:(length(f) - 1)]
        push!(
            points, ColmapPoint3D(
                parse(Int64, f[1]),
                SVector{3, Float64}(parse.(Float64, f[2:4])),
                SVector{3, UInt8}(parse.(UInt8, f[5:7])),
                parse(Float64, f[8]), track
            )
        )
    end
    return points
end
