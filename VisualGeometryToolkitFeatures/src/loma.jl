# =============================================================================
# LoMa (DeDoDe descriptors + LoMa matchers), ONNX Runtime on the CPU or CUDA
# =============================================================================
#
# The descriptor returns a StructArray{DescribedFeature}; the matcher returns
# the stack's match type, Vector{NNMatch}, so `csponds(fa, fb, matches)` and
# everything downstream of `match` take LoMa output unchanged.
#
# ONNXRunTime.jl reverses dims at the boundary, so a Julia array's size IS the
# ONNX shape: the image goes in as (1, 3, H, W), keypoints as (1, N, 2).
#
# Keypoints cross into the networks' [-1, 1] chart here and nowhere else: the
# detectors' COLMAP pixel centres (image corners at 0 and (w, h)) map to
# 2x/w - 1, DeDoDe's `to_normalized_coords`. Nothing normalized leaves this file.
#
# `execution_provider` is ONNXRunTime's (`:cpu` or `:cuda`). `:cuda` loads
# through ONNXRunTime's own CUDA extension, so CUDA is not a dependency here.
# =============================================================================

import ONNXRunTime as ORT
using Downloads: Downloads
using Scratch: @get_scratch!
using SHA: sha256

# =============================================================================
# Models: the ONNX files of the `loma` release, pinned by sha256
# =============================================================================
#
# Release assets are bare .onnx files, not tarballs, so they cannot be Pkg
# artifacts. They are cached in this package's scratch space (removed by
# Pkg.gc() with the package) and checked against GitHub's published digest
# once, after the download; a cached file is trusted from then on.
#
# Every bf16 variant keeps Float32 inputs and outputs (bf16 inside only). The
# descriptor dimension is each graph's declared `descriptions` / `desc0` width,
# read from the files; it decides which descriptor a matcher takes.

const _LOMA_RELEASE = "https://github.com/davnords/storage/releases/download/loma"

# (variant, bf16) => (file, sha256, descriptor dimension)
const _LOMA_DESCRIPTORS = Dict(
    (:b, false) => ("loma_descriptor_dedode_b.onnx", "82660a364299013618fe649092ebc4f617559f6a77e1ab5a3412be62a47ddc2d", 128),
    (:g, false) => ("loma_descriptor_dedode_g.onnx", "5a7b9eaf7425d4513c5d7feae86080bae7ed3aceae7fb1b9f059d0752e2ad564", 256),
    (:g, true) => ("loma_descriptor_dedode_g_bf16.onnx", "679b3ead5385b2bf0424548ed027edd48f96eb8b7cddcb42344e899cf0d97e7b", 256),
)

const _LOMA_MATCHERS = Dict(
    (:b128, false) => ("loma_matcher_B128.onnx", "e71ad490d13713374433a7ef99a7b4f4877d09338e40f347b7e64cc90150ee16", 128),
    (:b128, true) => ("loma_matcher_B128_bf16.onnx", "7dec0563c0955400a7bef18ba0c62a70772c9c5c3bf2734372c32de5f1876c32", 128),
    (:b, false) => ("loma_matcher_B.onnx", "ba5a2773b29cace19f1240e14e5a080cca3eaf9f69a7adb829a1d470557001c7", 256),
    (:b, true) => ("loma_matcher_B_bf16.onnx", "e61751cdbac6f305efa0d5b2ddc8f8c8e095b1b7196bfac3ed8d4e29b007f625", 256),
    (:l, false) => ("loma_matcher_L.onnx", "918da5427f37c3d9e05264a58a4d05729f87d459c3d5d69add7ce72eeacea66d", 256),
    (:l, true) => ("loma_matcher_L_bf16.onnx", "bab859300bbec88af30705021356f526a374d48ac7917b135d979686ade0ed23", 256),
    (:g, false) => ("loma_matcher_G.onnx", "62134cde779ed1031982c13c2c1952ed14856d715e97072364821a91f6276cf9", 256),
    (:g, true) => ("loma_matcher_G_bf16.onnx", "2d21420e1e0e3bcd085cd9a7d33be1f990e820d6b13ffbbddc73d9d04713b6c6", 256),
    (:r, false) => ("loma_matcher_R.onnx", "6c55247068568861d983b6005d9c401a4c62b8f8ea75d1a6925f13b0211407b5", 256),
    (:r, true) => ("loma_matcher_R_bf16.onnx", "172d214a91e5c12126acc7c7c66804c22ec47b9a8fc572b10ba9711714386e43", 256),
)

function _loma_model(table, kind, variant, bf16)
    haskey(table, (variant, bf16)) && return table[(variant, bf16)]
    known = join(sort!([string(repr(v), b ? " (bf16)" : "") for (v, b) in keys(table)]), ", ")
    throw(ArgumentError("no LoMa $kind $(repr(variant))$(bf16 ? " in bf16" : ""); known: $known"))
end

# The cached path of a release file, downloaded and verified on first use.
function _loma_model_path(file, digest)
    path = joinpath(@get_scratch!("loma_models"), file)
    isfile(path) && return path
    @info "Downloading LoMa model $file (cached in this package's scratch space)"
    tmp = Downloads.download("$_LOMA_RELEASE/$file", tempname(dirname(path)))
    got = bytes2hex(open(sha256, tmp))
    if got != digest
        rm(tmp; force = true)
        error("LoMa model $file: sha256 $got, expected $digest")
    end
    mv(tmp, path; force = true)   # same directory: atomic, so no partial file is ever cached
    return path
end

_loma_session(file, digest, execution_provider) =
    ORT.load_inference(_loma_model_path(file, digest); execution_provider)

# Checked before the download: ONNX Runtime's CPU provider lacks bf16 kernels,
# so a bf16 export fails to load there ("Could not find an implementation for
# MatMul(13)" with loma_matcher_B128_bf16.onnx, "Mul(14)" with
# loma_descriptor_dedode_g_bf16.onnx; measured).
function _check_bf16_provider(bf16, execution_provider)
    bf16 && execution_provider === :cpu && throw(ArgumentError(
        "the bf16 LoMa exports need execution_provider = :cuda: ONNX Runtime's CPU provider lacks bf16 kernels"))
    return nothing
end

# =============================================================================
# Keypoints and detections
# =============================================================================

# (h, w) from an image size: (H, W) or a channelview's (C, H, W)
_image_hw(sz) = (sz[end - 1], sz[end])

# 1×N×2 Float32 keypoints in the networks' [-1, 1] chart
function _loma_keypoints(features, sz)
    h, w = _image_hw(sz)
    kpts = Array{Float32}(undef, 1, length(features), 2)
    for (i, f) in enumerate(features)
        x, y = centroid(f)
        kpts[1, i, 1] = 2x / w - 1
        kpts[1, i, 2] = 2y / h - 1
    end
    return kpts
end

# The n strongest detections (largest |response|, the DoG response is signed),
# in detector order. Not `first(detections, n)`: ASIFT emits in scan order, so
# a prefix covers only the top rows of the image.
function _strongest(detections, n)
    length(detections) <= n && return detections
    keep = partialsortperm(abs.(getproperty.(detections, :response)), 1:n; rev = true)
    return detections[sort!(collect(keep))]
end

# =============================================================================
# Descriptor
# =============================================================================

"""
    LoMaDescriptor(variant = :b; max_detections = 2048, bf16 = false, execution_provider = :cpu)

A DeDoDe descriptor of LoMa: `:b` (DeDoDe-B, 128-dimensional, 54 MB) or `:g`
(DeDoDe-G, 256-dimensional, 1.3 GB; 690 MB with `bf16 = true`, which needs
`execution_provider = :cuda`). The model is
downloaded on first use and cached. `(descriptor)(img, detections)` describes
the `max_detections` strongest detections (largest `|response|`) of `img` and
returns a `StructArray{DescribedFeature}`. `execution_provider = :cuda` needs
`import CUDA, cuDNN` and a functional device.

`img` is a `(3, 784, 784)` Float32 RGB channelview in [0, 1], as
`Float32.(channelview(rgb))` gives it: the exported graph's input shape is
fixed, and it feeds the first convolution directly, with no normalization.
Do not ImageNet-normalize it: on LoMa's toronto pair that drops the matches
from 166 to 14 (and none above probability 0.5).

A descriptor pairs with the [`LoMaMatcher`](@ref)s of its dimension: `:b` with
`:b128`, `:g` with `:b`, `:l`, `:g` and `:r`.
"""
struct LoMaDescriptor
    variant::Symbol
    max_detections::Int
    descriptor_dim::Int
    descriptor_model::ORT.InferenceSession
end

function LoMaDescriptor(
        variant::Symbol = :b; max_detections::Integer = 2048, bf16::Bool = false,
        execution_provider::Symbol = :cpu
    )
    file, digest, dim = _loma_model(_LOMA_DESCRIPTORS, "descriptor", variant, bf16)
    _check_bf16_provider(bf16, execution_provider)
    return LoMaDescriptor(variant, max_detections, dim, _loma_session(file, digest, execution_provider))
end

const _LOMA_IMAGE_SIZE = (3, 784, 784)   # the descriptor graph's fixed input shape

function (l::LoMaDescriptor)(img, detections)
    size(img) == _LOMA_IMAGE_SIZE || throw(ArgumentError(
        "LoMaDescriptor takes a $(join(_LOMA_IMAGE_SIZE, '×')) RGB channelview, got size $(size(img))"))
    detections = _strongest(detections, l.max_detections)
    descriptor_inputs = l.descriptor_model.input_names

    descriptors = l.descriptor_model(Dict(
        descriptor_inputs[1] => insertdims(img, dims=1),
        descriptor_inputs[2] => _loma_keypoints(detections, size(img))
    ))["descriptions"]
    described_features(detections, dropdims(descriptors, dims=1))
end

# =============================================================================
# Matcher
# =============================================================================

"""
    LoMaMatcher(variant = :b128; bf16 = false, execution_provider = :cpu)

A LoMa matcher: `:b128` (48 MB) on 128-dimensional DeDoDe-B descriptors, or
`:b` (48 MB), `:l` (184 MB), `:g` (726 MB) and `:r` (48 MB) on 256-dimensional
DeDoDe-G ones ([`LoMaDescriptor`](@ref)`(:g)`). `bf16 = true` takes the
half-size bf16 export of any of them, which needs `execution_provider = :cuda`
(ONNX Runtime's CPU provider lacks bf16 kernels). The model is downloaded on first use and
cached; see the call method for matching. `execution_provider = :cuda` needs
`import CUDA, cuDNN` and a functional device.
"""
struct LoMaMatcher
    variant::Symbol
    descriptor_dim::Int
    matcher_model::ORT.InferenceSession
end

function LoMaMatcher(variant::Symbol = :b128; bf16::Bool = false, execution_provider::Symbol = :cpu)
    file, digest, dim = _loma_model(_LOMA_MATCHERS, "matcher", variant, bf16)
    _check_bf16_provider(bf16, execution_provider)
    return LoMaMatcher(variant, dim, _loma_session(file, digest, execution_provider))
end

"""
    (matcher::LoMaMatcher)(features_a, features_b, size_a, size_b; min_score = 0) -> Vector{NNMatch}

Match two sets of [`LoMaDescriptor`](@ref) features, whose descriptor dimension
must be the matcher's (`ArgumentError` otherwise). `size_a`/`size_b` are the
image sizes, `(H, W)` or a channelview's `(C, H, W)`.

Returns the same type as [`match`](@ref): `m.first` indexes `features_a`,
`m.second` indexes `features_b`, sorted best-first. `m.score` is the network's
match probability (higher is better, a similarity, so the default
`csponds(features_a, features_b, matches)` reads it correctly); `m.ratio` is
`Inf32`, since the network has no second neighbour. Matches scoring below
`min_score` are dropped.
"""
function (l::LoMaMatcher)(features_a, features_b, size_a, size_b; min_score::Real = 0)
    for features in (features_a, features_b)
        n = isempty(features) ? l.descriptor_dim : length(first(features).descriptor)
        n == l.descriptor_dim || throw(ArgumentError(
            "LoMa matcher $(repr(l.variant)) takes $(l.descriptor_dim)-dimensional descriptors, got $n " *
            "(a LoMaDescriptor(:$(l.descriptor_dim == 128 ? "b" : "g")) pairs with it)"))
    end
    matcher_inputs = l.matcher_model.input_names

    desc_a = insertdims(Float32.(stack(map(feat -> feat.descriptor, features_a), dims=1)), dims=1)
    desc_b = insertdims(Float32.(stack(map(feat -> feat.descriptor, features_b), dims=1)), dims=1)

    out = l.matcher_model(Dict(
        matcher_inputs[1] => _loma_keypoints(features_a, size_a),
        matcher_inputs[2] => _loma_keypoints(features_b, size_b),
        matcher_inputs[3] => desc_a,
        matcher_inputs[4] => desc_b
    ))
    return _loma_matches(vec(out["m0"]), vec(out["mscores0"]), min_score)
end

# m0[i] is the 0-based index into b matched to a's i-th feature, -1 for none
function _loma_matches(m0, mscores0, min_score)
    matches = NNMatch[]
    for (i, (j, s)) in enumerate(zip(m0, mscores0))   # ORT outputs index Cartesian
        (j < 0 || s < min_score) && continue
        push!(matches, NNMatch(i, Int(j) + 1, Float32(s), Inf32))
    end
    return sort!(matches; by = m -> m.score, rev = true)
end
