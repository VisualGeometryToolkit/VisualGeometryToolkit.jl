# SALAD: global place-recognition descriptors of three images, compared by
# cosine similarity.
#
#   SALAD_MODEL_DIR=path/to/onnx julia --project=VisualGeometryToolkitFeatures/examples VisualGeometryToolkitFeatures/examples/salad.jl
#
# No ONNX exports are published: SALAD_MODEL_DIR is the directory holding the
# exports of https://github.com/the-sauer/salad's export_onnx.py, needed once;
# the models are then cached in VisualGeometryToolkitFeatures' scratch space,
# as are the images. SALAD_VARIANT picks the model: dino_salad (default),
# dino_salad_2048_64 or dino_salad_512_32.
#
# Runs on the GPU when CUDA is functional and on the CPU otherwise; set
# SALAD_PROVIDER=cpu to force the CPU. Two views of one place (LoMa's toronto
# pair) should score far above a view against an unrelated image (SALAD's
# method figure).

using ImageIO
using FileIO
using ImageTransformations
using ImageCore
using Downloads: Downloads
using Scratch: get_scratch!
using SHA: sha256
import CUDA

using VisualGeometryToolkitFeatures

const execution_provider =
    get(ENV, "SALAD_PROVIDER", "") != "cpu" && CUDA.functional() ? :cuda : :cpu
# cuDNN completes ONNXRunTime's CUDA extension, whose __init__ throws without a
# GPU (it queries the CUDA runtime version), so load it only for :cuda.
execution_provider === :cuda && @eval import cuDNN

const variant = Symbol(get(ENV, "SALAD_VARIANT", "dino_salad"))
@info "SALAD" execution_provider variant

descriptor = SaladDescriptor(variant; execution_provider)

# pinned to commits and by sha256
const IMAGES = [
    ("toronto_A.jpg", "https://github.com/davnords/LoMa/raw/48cfbec8e56853cdcfd307f97a6b5061c49c4728/assets/toronto_A.jpg",
        "40270c227df93f0f31b55e0f2ff38eb24f47940c4800c83758a74a5dfd7346ec"),
    ("toronto_B.jpg", "https://github.com/davnords/LoMa/raw/48cfbec8e56853cdcfd307f97a6b5061c49c4728/assets/toronto_B.jpg",
        "a2c07550ed87e40fca8c38076eb3a81395d760a88bf0b8615167704107deff2f"),
    ("salad_method.jpg", "https://github.com/serizba/salad/raw/6aede13a3f6c25750bf7fde10209c06cb73060bb/assets/method.jpg",
        "db4d2175d4a108ff743b61be78e8cee06156956d32fbcd92c80a9bc1cfd86782"),
]

function cached_image(name, url, digest)
    path = joinpath(get_scratch!(VisualGeometryToolkitFeatures, "salad_example_images"), name)
    isfile(path) && return path
    tmp = Downloads.download(url, tempname(dirname(path)))
    bytes2hex(open(sha256, tmp)) == digest || (rm(tmp); error("$name: sha256 mismatch"))
    return mv(tmp, path; force = true)
end

chw(path) = Float32.(channelview(imresize(RGB.(load(path)), (322, 322))))

names = first.(IMAGES)
D = descriptor([chw(cached_image(img...)) for img in IMAGES])   # descriptor_dim × 3, unit columns
@info "SALAD" descriptor_dim = size(D, 1)

similarity = D' * D
println(rpad("", 18), join(rpad.(names, 18)))
for (i, name) in enumerate(names)
    println(rpad(name, 18), join(rpad.(round.(similarity[i, :]; digits = 3), 18)))
end
