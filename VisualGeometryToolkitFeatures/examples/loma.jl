# LoMa: DeDoDe descriptors on ASIFT detections, matched by a LoMa matcher.
#
#   julia --project=VisualGeometryToolkitFeatures/examples VisualGeometryToolkitFeatures/examples/loma.jl
#
# LOMA_MATCHER picks the matcher, and with it the descriptor: b128 (default;
# DeDoDe-B) or b, l, g, r (DeDoDe-G, a 1.3 GB download); LOMA_BF16=1 takes the
# bf16 exports (CUDA only). Models and images are downloaded once and cached in
# VisualGeometryToolkitFeatures' scratch space.
#
# Runs on the GPU when CUDA is functional and on the CPU otherwise; set
# LOMA_PROVIDER=cpu to force the CPU. Shows the matches as lines between the two
# images, side by side, in a GLMakie window.

using ImageIO
using FileIO
using ImageTransformations
using ImageCore
using Downloads: Downloads
using Scratch: get_scratch!
using SHA: sha256
import CUDA
import GLMakie

using VisualGeometryToolkitFeatures
using VisualGeometryToolkitCore: Point2, Rect, centroid, imshow, imshow!, matches!

const execution_provider =
    get(ENV, "LOMA_PROVIDER", "") != "cpu" && CUDA.functional() ? :cuda : :cpu
# cuDNN completes ONNXRunTime's CUDA extension, whose __init__ throws without a
# GPU (it queries the CUDA runtime version), so load it only for :cuda.
execution_provider === :cuda && @eval import cuDNN

const matcher_variant = Symbol(get(ENV, "LOMA_MATCHER", "b128"))
const descriptor_variant = matcher_variant === :b128 ? :b : :g
const bf16 = get(ENV, "LOMA_BF16", "0") == "1" && (execution_provider === :cuda ||
    (@warn "LOMA_BF16 needs CUDA (ONNX Runtime's CPU provider lacks bf16 kernels); using the full-precision models"; false))
@info "LoMa" execution_provider descriptor_variant matcher_variant bf16

descriptor = LoMaDescriptor(descriptor_variant; bf16 = bf16 && descriptor_variant === :g, execution_provider)
matcher = LoMaMatcher(matcher_variant; bf16, execution_provider)

# LoMa's demo pair, pinned to a commit and by sha256
const ASSETS = "https://github.com/davnords/LoMa/raw/48cfbec8e56853cdcfd307f97a6b5061c49c4728/assets"
const IMAGES = Dict(
    "toronto_A.jpg" => "40270c227df93f0f31b55e0f2ff38eb24f47940c4800c83758a74a5dfd7346ec",
    "toronto_B.jpg" => "a2c07550ed87e40fca8c38076eb3a81395d760a88bf0b8615167704107deff2f",
)

function cached_image(name)
    path = joinpath(get_scratch!(VisualGeometryToolkitFeatures, "loma_example_images"), name)
    isfile(path) && return path
    tmp = Downloads.download("$ASSETS/$name", tempname(dirname(path)))
    bytes2hex(open(sha256, tmp)) == IMAGES[name] || (rm(tmp); error("$name: sha256 mismatch"))
    return mv(tmp, path; force = true)
end

function describe(name)
    img = imresize(load(cached_image(name)), (784, 784))
    detections = asift_detect(img)
    chw = Float32.(channelview(img))
    return img, descriptor(chw, detections), size(chw)
end

img_a, desc_a, size_a = describe("toronto_A.jpg")
img_b, desc_b, size_b = describe("toronto_B.jpg")

matches = matcher(desc_a, desc_b, size_a, size_b; min_score = 0.0)

cs = csponds(desc_a, desc_b, matches)   # matched features in pixels, with score/ratio columns
@info "LoMa" n_a = length(desc_a) n_b = length(desc_b) n_matches = length(cs)

# B sits to the right of A in one plotting frame (COLMAP pixels, as the
# detections are), so its points shift by A's width.
h_a, w_a = size(img_a)
h_b, w_b = size(img_b)
fig = GLMakie.Figure(size = (w_a + w_b, max(h_a, h_b)))
ax, _ = imshow(fig[1, 1], img_a, Rect(0, 0, w_a, h_a))
imshow!(ax, img_b, Rect(w_a, 0, w_b, h_b))
GLMakie.hidedecorations!(ax)
src = [Point2{Float64}(centroid(p.first)) for p in cs]
tgt = [Point2{Float64}(centroid(p.second)) + Point2(w_a, 0) for p in cs]
colors = getindex.(Ref(GLMakie.cgrad(:viridis)), cs.score)   # match probability, dark to bright
matches!(ax, src, tgt; colors, src_markersize = 6, tgt_markersize = 6, linewidth = 1)
GLMakie.Colorbar(fig[1, 2]; colormap = :viridis, limits = (0, 1), label = "match probability")

isinteractive() ? display(fig) : wait(display(fig))   # a script keeps the window open
