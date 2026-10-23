# =============================================================================
# DINOv2 SALAD (global place-recognition descriptors), ONNX Runtime on the CPU or CUDA
# =============================================================================
#
# The network and trained weights are the SALAD authors' (Izquierdo and Civera,
# CVPR 2024; https://github.com/serizba/salad). This file runs ONNX exports of
# them; the README's Citing section has the BibTeX entry.
#
# SALAD describes a whole image: one L2-normalized vector per image, compared
# by dot product (cosine similarity). No coordinates go in or come out.
#
# ONNXRunTime.jl reverses dims at the boundary, so a Julia array's size IS the
# ONNX shape: the batch goes in as (N, 3, 322, 322), comes back as (N, D).
#
# The exported graph takes ImageNet-normalized input and does not normalize
# itself; the normalization is done here and nowhere else.
#
# `execution_provider` is ONNXRunTime's (`:cpu` or `:cuda`). `:cuda` loads
# through ONNXRunTime's own CUDA extension, so CUDA is not a dependency here.
# =============================================================================

# =============================================================================
# Models: ONNX exports of the authors' checkpoints, pinned by sha256
# =============================================================================
#
# The authors publish PyTorch checkpoints only. The exports are made with
# `export_onnx.py` of https://github.com/the-sauer/salad (fixed 322×322 input,
# dynamic batch) and read from a local directory: the `model_dir` keyword or
# ENV["SALAD_MODEL_DIR"]. A file is checked against its digest once, when it
# is copied into this package's scratch space (removed by Pkg.gc() with the
# package); a cached file is trusted from then on. Once the exports are
# published, the copy becomes a download, as in loma.jl's `_loma_model_path`.

# variant => (file, sha256, descriptor dimension: clusters × cluster dim + token dim)
const _SALAD_MODELS = Dict(
    :dino_salad => ("dino_salad.onnx", "c92a98d35d03898d8ca3ab73498ff297ba591d6708a0c2b0f5efda6996402aaf", 8448),
    :dino_salad_2048_64 => ("dino_salad_2048_64.onnx", "2c9171d222a8e85ebf38342f1ef5fe449db08bead004d84a18f357ea7f42fb5d", 2112),
    :dino_salad_512_32 => ("dino_salad_512_32.onnx", "9e280ccef0d27d27e80ad770905c93afe44bbef64bb900ce731b28a88a7bd160", 544),
)

function _salad_model(variant)
    haskey(_SALAD_MODELS, variant) && return _SALAD_MODELS[variant]
    known = join(sort!(repr.(collect(keys(_SALAD_MODELS)))), ", ")
    throw(ArgumentError("no SALAD model $(repr(variant)); known: $known"))
end

# The cached path of an export, copied from `model_dir` and verified on first use.
function _salad_model_path(file, digest, model_dir)
    path = joinpath(@get_scratch!("salad_models"), file)
    isfile(path) && return path
    model_dir === nothing && throw(ArgumentError(
        "SALAD model $file is not cached and no ONNX exports are published: supply `model_dir` " *
        "(or set ENV[\"SALAD_MODEL_DIR\"]) to a directory holding the exports of " *
        "https://github.com/the-sauer/salad's export_onnx.py"))
    src = joinpath(model_dir, file)
    isfile(src) || throw(ArgumentError("SALAD model $file not found in $model_dir"))
    got = bytes2hex(open(sha256, src))
    got == digest || error("SALAD model $src: sha256 $got, expected $digest")
    @info "Caching SALAD model $file in this package's scratch space"
    tmp = cp(src, tempname(dirname(path)))
    mv(tmp, path; force = true)   # same directory: atomic, so no partial file is ever cached
    return path
end

# =============================================================================
# Descriptor
# =============================================================================

const _SALAD_IMAGE_SIZE = (3, 322, 322)   # the exported graph's fixed input shape
const _IMAGENET_MEAN = Float32[0.485, 0.456, 0.406]
const _IMAGENET_STD = Float32[0.229, 0.224, 0.225]

"""
    SaladDescriptor(variant = :dino_salad; model_dir = get(ENV, "SALAD_MODEL_DIR", nothing), execution_provider = :cpu)

A DINOv2 SALAD global place-recognition descriptor: `:dino_salad`
(8448-dimensional, 8192 + 256), `:dino_salad_2048_64` (2112, 2048 + 64) or
`:dino_salad_512_32` (544, 512 + 32), each about 350 MB. No ONNX exports are
published: on first use the model is read from `model_dir`, which holds the
exports of [the-sauer/salad](https://github.com/the-sauer/salad)'s
`export_onnx.py`, checked by sha256 and cached; later calls need no
`model_dir`. `execution_provider = :cuda` needs `import CUDA, cuDNN` and a
functional device.

`(descriptor)(img)` returns the image's unit-norm `Vector{Float32}`;
`(descriptor)(imgs)` on a vector of images returns a `D × N` `Matrix{Float32}`,
one column per image, in one network call. Two images are compared by the dot
product of their descriptors (cosine similarity; higher is more alike).

`img` is a `(3, 322, 322)` Float32 RGB channelview in [0, 1], as
`Float32.(channelview(imresize(rgb, (322, 322))))` gives it: the exported
graph's input shape is fixed. It is ImageNet-normalized here; do not normalize
it yourself. The authors resize with torchvision's antialiased bilinear
`Resize`, so another resampler moves the descriptor slightly.

The network is [SALAD](https://github.com/serizba/salad)'s (Izquierdo and
Civera, "Optimal Transport Aggregation for Visual Place Recognition", CVPR
2024); cite it (BibTeX in the package README).
"""
struct SaladDescriptor
    variant::Symbol
    descriptor_dim::Int
    model::ORT.InferenceSession
end

function SaladDescriptor(
        variant::Symbol = :dino_salad; model_dir = get(ENV, "SALAD_MODEL_DIR", nothing),
        execution_provider::Symbol = :cpu
    )
    file, digest, dim = _salad_model(variant)
    path = _salad_model_path(file, digest, model_dir)
    return SaladDescriptor(variant, dim, ORT.load_inference(path; execution_provider))
end

# N×3×322×322 Float32, ImageNet-normalized
function _salad_input(imgs)
    x = Array{Float32}(undef, length(imgs), _SALAD_IMAGE_SIZE...)
    for (b, img) in enumerate(imgs)
        size(img) == _SALAD_IMAGE_SIZE || throw(ArgumentError(
            "SaladDescriptor takes a $(join(_SALAD_IMAGE_SIZE, '×')) RGB channelview, got size $(size(img))"))
        for c in 1:3
            x[b, c, :, :] .= (view(img, c, :, :) .- _IMAGENET_MEAN[c]) ./ _IMAGENET_STD[c]
        end
    end
    return x
end

function (s::SaladDescriptor)(imgs::AbstractVector{<:AbstractArray})
    out = s.model(Dict(s.model.input_names[1] => _salad_input(imgs)))["descriptor"]   # N × D
    return permutedims(out)
end

(s::SaladDescriptor)(img::AbstractArray{<:Real, 3}) = vec(s([img]))
