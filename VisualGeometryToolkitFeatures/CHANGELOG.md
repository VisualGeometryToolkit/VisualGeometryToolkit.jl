# VisualGeometryToolkitFeatures changelog

## 0.8.0 (unreleased)

Not breaking. `[compat]` VisualGeometryToolkitCore 0.9.2 (its dependency allow-list names VisualGeometryToolkitEstimation
for this package).

- **New: `detect_image_circle([rng,] [T = Ellipse,] frames; smoothing_weight = 0, outlier_halfwidth, min_hole_area,
  border_margin = 5)`** (exported; moved from RadiometricCamera, the same signature and, on its tests, the same results
  bit for bit): the image circle of a lens measured from photographs — the frames' per-pixel maximum as `N0f8`, its
  dark MSERs (`detect_mser`, `max_area = 1`, duplicates kept), every region boundary less the holes smaller than
  `min_hole_area` and the border's vertices, their convex hull, and a RANSAC `Ellipse` or `Circle` through the hull's
  vertices off the border, scored by the Student-t + uniform marginal at `outlier_halfwidth`. Returns `(; circle, fit,
  contour, cut, rim_gap)` or `nothing`. A RAW frame (a `RawMosaic`, or `LinearRaw(mosaic; mode = :green_cells)`) is
  measured on its green cells, with the keywords in green-cell pixels (a pipeline config's resolution), and only the
  result is mapped: its circle and contour come back in the raw's pixels through the cells' `raw_frame_map`, and `fit`'s
  estimate with them, while its `rms` and residuals stay in green-cell pixels. A cell image on a device is read to the host (the result bit for bit the host's, a GPU test).
- **New dependencies:** DelaunayTriangulation (the convex hull) and Random. **New weak dependencies and extensions:**
  VisualGeometryToolkitEstimation (`VisualGeometryToolkitFeaturesEstimationExt`: the fit, `find_ellipse` /
  `find_circle`; without it `detect_image_circle` is an `ArgumentError` naming the package) and ImageSmooth
  (`VisualGeometryToolkitFeaturesImageSmoothExt`: the optional L0 smoothing, `smoothing_weight > 0`).

## 0.7.0 (unreleased)

- **New: `propagate(m, d::Detection{<:Uncertain})`**: the uncertain centre of a detection through a map (its centre and
  the position block of its frame covariance, carried by VisualGeometryToolkitCore's `propagate`; moved from
  PanoramaStitching).
- **Device buffers are released through VisualGeometryToolkitCore's `free_device_buffers!`** (an evicted ASIFT plan's,
  the matcher's score blocks), no longer KernelAbstractions' `unsafe_free!`, which CUDA.jl does not extend: on a GPU
  they were not released until the collector ran. Now they are, at once.

- **ASIFT with `FeatureCovariance` drops a detection whose covariance is singular or NaN** (the owner's ruling,
  2026-11-12), exactly as it drops a failed sub-pixel refinement, before the cross-warp NMS, and counts it in
  `plan.refine_dropped`. Every returned keypoint's covariance is finite and positive definite. Without
  `uncertainty` nothing changes.

- **Internal: every kernel is instantiated through VisualGeometryToolkitCore's one launch policy**, `OnBackend(device;
  workgroupsize)(kernel)` (the owner's ruling, 2026-11-12): the ASIFT, cross-warp NMS, matching, Förstner, SSC and RAW
  cell kernels, each at the workgroup it had (`_launch!` keeps only the synchronize). Results and timings unchanged.
- **Breaking: `pairwise_matrix` and `nn_match` default to `default_device()`** (the stack's GPU-first rule):
  `pairwise_matrix(metric, d1, d2, device = default_device())` and `nn_match(…; device = default_device())` run on an
  accelerator when one is functional, else on `KA.CPU()` as before. `nn_match(…; device = nothing)` is a
  `TypeError`: give the backend, or leave the keyword out. `match` stays the host call.
- **Breaking: `RawMosaic{A} <: AbstractMatrix{UInt16}` is its photosite matrix** (the owner's raw-image decision;
  BayeredImages' idea, no dependency on it). `size`, indexing, `IndexStyle` and `parent` are the held array's (`m.raw`,
  any `AbstractMatrix{UInt16}`, a device array included), so an array or image function reads a mosaic unchanged;
  what one makes (`similar`, `copy`, a broadcast, a reduction) is a plain array of the held array's kind. A mosaic is
  now `==` to its photosite matrix, iterates over its photosites and prints one summary line. The type parameters are
  `RawMosaic{A, M}` (was `RawMosaic{M}`). `copy`, `==`, `isequal` and `isapprox` are the held array's own (a device
  mosaic's `copy(m)` is a device array, its comparison no scalar read). Equality is the photosite matrix's, by design:
  two mosaics (or windows) with the same photosites and another colour phase, window origin or metadata are `==`
  (and hash alike); compare `photosite_colors`, `window_origin` and `metadata` for the rest.
- **`load_raw`'s cache key includes the decoder library's version** (`libraw_version()` for `LibRawDecoder`, the
  `_decoder_version(decoder)` hook; `nothing` for a decoder that states none), so after a LibRaw_jll upgrade a cached
  decode (its black levels, white level, crop) is a miss, not the old library's values.
- **New: windows of a mosaic.** `m[rows, columns]` (unit ranges, a copy) and `view(m, rows, columns)` (no copy) are
  mosaics of the same sensor: the new field `window_origin` (the constructor's keyword, `(0, 0)` as decoded) is the
  window's offset into the visible area, so its colours are the parent's at those photosites for either parity of
  its start, and `raw_frame_map(m::RawMosaic)` (new) is that offset. A `LinearRaw` made from a window maps onto the
  parent's photosites exactly (its `raw_frame_map` and `bayer_plane_map` include the offset).
- **New: `raw_coordinates(img, x)`, THE accessor of an image's sensor-pixel coordinates**: `raw_frame_map(img)(x)`, in
  the visible raw's COLMAP frame — the identity on a decoded mosaic, `x ↦ 2x` on the cell grid of a `LinearRaw`; the
  display demosaic is refused, a plain matrix has no method. A detection on the green cells reports through it (the
  half-scale bug of writing cell positions as raw pixels). The design note: the guide's "Raw images" page.
- **New: a mosaic on the device.** `adapt(CuArray, m)` or `adapt(backend, m)` (Adapt, a new dependency; a KernelAbstractions
  backend) move the photosites and keep the mosaic; `LinearRaw(m; mode)` runs its cell kernel on `parent(m)`'s device and
  leaves the image and the saturation flags there. **Breaking:** `LinearRaw{Mode, I, S, G, M}` holds
  `image::AbstractArray{Float64, 3}` and `saturated::AbstractMatrix{Bool}` (an `Array` and a `BitMatrix` on the host,
  as before); `CACHE_FORMAT` is 3, so decodes cached under the earlier layout are a miss.

- **Internal: one launch path for the ASIFT kernels on every backend** (the owner's one-kernel policy; Core's
  `check_duplication` flagged the `_scheduled` method pair): every backend's kernels launch through `_ScheduledKernel`
  (formerly `_CPUKernel`), built on the device's static form (KernelAbstractions' `CPU(static = true)` on the CPU, the
  device itself elsewhere) and on the device; the CPU's static/dynamic launches are unchanged. `_chunk_warps`' backend
  test is a free-memory reserve (an allocation, the same kernels on every backend): a DUPLICATION.toml `[[backend]]`
  entry with that reason.
- **New: `onnx_session(path; execution_provider = :auto, threads, gpu_mem_limit)`, public**: the package's ONNX Runtime
  session builder (formerly the internal `_onnx_session`), for another package's network: the provider resolved as
  LoMa's is (`resolve_execution_provider`; on CUDA, conda-forge's ONNX Runtime build, which runs on the B200's sm_100),
  ONNX Runtime's intra-op pool the caller's `threads`, and on CUDA the session's arena capped at the keyword
  `gpu_mem_limit` bytes (default 8 GiB, LoMa's). The detector's plans now leave free the room the live CUDA sessions'
  arenas may still take, each session's cap less what its arena already holds (measured at its creation and around
  LoMa's runs; the free memory the plans start from already excludes it), rather than 8 GiB per session ever created.
  A session released or collected reserves nothing, and the CUDA probe's one-op session (built by `onnx_cuda_status`,
  so by any `:auto` or `:cuda` resolution) reserves nothing and is released at once: before, the reserve only grew,
  the probe alone added 8 GiB, and a descriptor and a matcher on a 24 GB device left the detector one warp per chunk
  (slower ASIFT, the same detections). BlobBoards' descriptor network moves onto it.
- **Breaking: `asift_detect`'s detection keyword `mask` is `exclusion_mask`** (the image and the `StabilizedImage`
  methods; the semantics are unchanged: a marked pixel excludes every detection whose centre lies within three of its
  scales, and a `StabilizedImage`'s saturation joins it). A call with `mask` is an `ArgumentError` naming
  `exclusion_mask`. `gaussian_blur(img, σ, mask)`'s positional weight map is another thing and keeps its name.

- **Faster: `asift_detect(…; exclusion_mask)`** decides each detection's clearance from a summed-area table of the mask, read
  over the detection's disc, instead of a full-frame Euclidean distance transform per call: the same decisions (the
  same distance rule), ~6× faster and ~6.6× less memory on a 24 MP frame (also every `asift_detect(::StabilizedImage)`,
  whose saturation is its mask).

- **Changed: `StabilizedImage(image, saturated, white)` holds a `Matrix{Float64}` image and a `BitMatrix` mask as
  given** (another element type is converted, once): `variance_stabilize` no longer copies the transformed frame and
  the saturation mask a second time.

- **Breaking: a `LinearRaw`'s mode is its type, `LinearRaw{:green_cells}`, `LinearRaw{:bin2}` or `LinearRaw{:demosaic}`,
  and its sample geometry one field.** The fields `plane_maps`, `green_offsets` and `green_dn` are
  `L.geometry = (; plane_maps, green_offsets, green_dn)`, `nothing` for the display demosaic (and only for it); the
  `mode` field is gone (`L isa LinearRaw{:bin2}`, or `raw_provenance(L).mode`). The geometric accessors
  (`raw_frame_map`, `bayer_plane_map`, `green_cell_sites`, `variance_stabilize`) are methods of the cell images, the
  display image's their refusals (the same `ArgumentError`), with no run-time mode test; the green-cell plane is
  chosen by dispatch.

- **New: `load_raw(path; …, cache)`, the decoded-RAW cache**: with `cache` a `FileCache` (`open_cache`, the JLD2
  extension), each RAW is decoded once per content and settings — the entry is keyed by the file's SHA-256, the mode,
  the gain, the white balance and the decoder's type (`cache_key`) — and a hit returns the `LinearRaw` as decoded,
  value for value. One cache for the stack's one RAW path: PanoramaStitching's example kept its own (code-sum PNGs
  and a JSON index, `decoded_frames`), which goes for it. `cache = nothing` (the default) decodes every call.

- **Fixed: the stabilized image's noise threshold** (`asift_detect(::StabilizedImage; noise_sigmas)`) read the noise σ of
  every DoG slice, the two boundary slices included, while the extremum search reads only the interior ones (2 … NS − 1,
  since the interior-slice search): the threshold was set by a slice no extremum comes from, so too high, and faint
  extrema were dropped. It is the noisiest searched slice's σ now: a lower threshold, more detections at the same
  `noise_sigmas`.
- **Changed: LoMa on ONNX Runtime's CUDA provider no longer releases the ASIFT detector's plans at every call** (a full GC,
  the pool reclaimed and the plan rebuilt at the next detection). Each CUDA session's arena is capped
  (`gpu_mem_limit` 8 GiB, grown by request), and the detector's plans size their resident warps from the free device
  memory less that cap for every CUDA session of the process; the plans cached before a session is created are released
  once, then.
- **Internal: Core's device queries.** The chunk sizing's `device_free_bytes` and the plan release's pool reclaim are
  Core's (`device_free_bytes`, `reclaim_device!`, answered by its CUDA extension); the CUDA 12 cuDNN probe reads its
  version with the module's one `_library_version`.
- **Faster: `load_raw(...; mode = :demosaic)` through LibRaw** copies the visible photosites once, into the display image:
  its checks of the mosaic's layout no longer copy the mosaic too.
- **New: the stack's one image viewer, `compare_images(entries; titles, default, size)`** (exported; the Makie
  extension): a menu over named entries, each a tuple of images (or a function returning one, loaded when chosen),
  one axis per image, the axes linked, each image drawn by Core's orientation rule (`imshow!`). PanoramaStitching's
  `compare_makie.jl` keeps only which renders form a pair.

- **New: the one display encoding, `srgb_image([T = N0f8], img; exposure, white_balance, background, tone)`**
  (exported), and **`linear_to_srgb`** (public), the IEC 61966-2-1 OETF beside `srgb_to_linear`. A linear
  `H × W × 3` image (or one plane) to `RGB{T}` (`Gray{T}`): exposure, a white-balance factor per plane, an optional
  tone curve (default none), clamp, sRGB, `N0f8` or `N0f16`; a NaN value is the `background`. A broadcast, so a
  device image is encoded on the device. The copies of the encoder in PanoramaStitching's `save_srgb`, RadiometricCamera's
  `write_pngs` and `write_decoded`, and BlobBoards' hero script become calls to it.

- **Breaking: each plane of a Bayer image is a frame-labelled map onto the raw, `bayer_plane_map(L, c)`; the bare
  offsets are gone.** `BayerCellFrame` is now `BayerCellFrame{C}`, the frame of plane `C` (`:R`, `:G`, `:B`): the
  planes are indexed alike but sampled at different points of the cell. `bayer_plane_map(L, c)` (exported) is plane
  `c`'s Core `SimilarityMap{…, BayerCellFrame{C}, PixelFrame{:colmap}}`, its translation onto the cell grid composed
  with `raw_frame_map(L)`, so the ×2 is stated once; it sends every sample onto the centre of its photosites exactly,
  for all four phases and both origin parities. The green plane's map IS `raw_frame_map(L)`, whose source is now
  `BayerCellFrame{:G}` (the cell grid). A warp reads plane `c` through `inv(bayer_plane_map(L, c)) ∘ math`, like any
  other map, all planes in one call through Core's `image_warp!(dst, warps, src)`; a composition that joins two
  planes' frames is refused (`MethodError`). Removed: the `LinearRaw` fields `plane_offsets` (a plane's offset is
  `translation(inv(raw_frame_map(L)) ∘ bayer_plane_map(L, c))`) and `frame_map` (`raw_frame_map(L)`); the new field
  `plane_maps` holds the maps. `raw_provenance(L).frame_map.from` reads `"BayerCellFrame{:G}"` (was
  `"BayerCellFrame"`). `plane_offsets` and PanoramaStitching's `raw_scale` are retired names.

- **ASIFT's result no longer depends on the threads: the same image gives the same keypoints, in the same order, run
  to run.** The extrema pool fills through an atomic counter, in the order the threads (or a GPU's warps) reach it,
  and the cross-warp NMS breaks ties in its quantized response by pool index, so on many threads both the order of the
  detections and, at a tie, which of two survived varied between runs (a panorama's frames detected twice reproduced
  on 12 of 38 frames with `spread_strongest` after, 36 of 38 with a top-n cut, in a 32-thread session). The pool is
  now sorted into a canonical order before the NMS (by position, scale and response, then every other field, the
  covariance's included: a total order), on the host, and the detections come back in it. `cross_warp_nms` sorts its
  input the same way, so its survivors no longer depend on the order of the pool handed in, and on the pool
  `asift_detect(…; nms = :none)` returns it keeps exactly what the in-pipeline NMS keeps (no longer "up to ties").
  `spread_strongest` was already a function of its input's order; given the canonical detections it is reproducible
  too. **Breaking (order):** a caller that relied on the pool's warp-by-warp order now gets the canonical order.
- **New: an exclusion mask, `asift_detect(img; exclusion_mask)`** — an H×W `Bool` matrix, `true` where detection is not allowed
  (clipped highlights, a vignetted rim): a detection whose centre lies within three of its scales (the DoG's support)
  of a masked pixel is dropped, on Detections, `raw` features and covariances alike. There was no way to exclude
  clipped regions of an ordinary (tone-mapped) image, so keypoints landed on clipped window panes. The rule is the
  one `asift_detect(::StabilizedImage)` applied to sensor saturation, now in one place: that method masks
  `s.saturated` through `exclusion_mask` (joined with a caller's `exclusion_mask`), and takes `raw = true`, which it refused while the
  masking read Detections.
- **New: `tilt_schedule(name)`, ASIFT's tilts by name; `asift_detect(img; tilts = :untilted)` restricts ASIFT to the
  untilted view (t = 1).** `:imas25` (the default of `asift_detect` and `AsiftPlan`) is IMAS-25; `:untilted` is the
  identity warp alone, plain DoG-SIFT, exactly the identity warp's detections under `:imas25`. On real panorama
  frames a tilted warp's keypoints localize at about 0.95 px rms, the untilted view's at about 0.3 px, and the
  untilted schedule gave the best seams. `tilts` still takes a vector of `(t, φ)` pairs. **Breaking:** `tilts =
  nothing` (the old spelling of the default) is gone: give `:imas25` or omit it. The private IMAS-25 helper is retired
  (`tilt_schedule(:imas25)`).

- **Every ASIFT detection is refined; an extremum whose refinement fails is dropped, never returned on the octave's
  grid.** The extrema were searched on all S + 2 DoG slices of an octave, the two boundary slices included, whose
  extrema have no slice above or below and were emitted unrefined (integer positions on the octave's grid, 2^octave
  input pixels; `NaN` covariance), and an extremum whose sub-pixel refinement failed (a singular or non-finite step, a
  walk off the interior) was emitted unrefined too. On a Tamron panorama frame 1451 of the 3077 detections (47 %) were
  unrefined: 1382 from boundary slices, 69 failed refinements. Now only the interior slices are searched, as Lowe's
  SIFT does (the boundary slices' scales are the adjacent octaves' interior slices), and a failed refinement is
  dropped and counted (`plan.refine_dropped`). The same frame gives 2099 detections, all refined. On a synthetic field
  of blobs at known sub-pixel centres (IMAS-25, every pooled detection): rms error 0.52 → 0.17 px (octave −1 0.46 →
  0.08, octave 0 0.60 → 0.28, octave 1 0.65 → 0.39); two renders 1.4° apart, the matched detections' rms 0.30 →
  0.16 px. The tilted views' back-projection carries no bias (every warp's mean error below 0.04 px).
- **`clear_asift_plans!()` returns the memory to the device** (CUDA.jl's pool is reclaimed), and a `LoMaDescriptor`
  or `LoMaMatcher` on ONNX Runtime's CUDA provider calls it before it runs: the cached plan (most of the device, sized
  from what was free) stayed resident after detection, and ONNX Runtime, which allocates outside CUDA.jl's pool, then
  failed to allocate in a long-lived process. The next detection rebuilds its plan (+0.04 s on a 2128 × 1424 frame on
  the RTX 4090; the release itself 0.3 s: a full collection, since `unsafe_free!` frees a buffer only once the arrays
  derived from it during a detection are collected, and they held a 10.6 GiB plan in use until then). A plan whose
  pyramid buffers no longer fit (another allocator grew after the chunk was sized from the free memory) halves its
  resident warps and retries instead of failing; an explicit `chunk` is never shrunk.

- **New: `variance_stabilize(L::LinearRaw; electrons_per_dn, read_noise) -> StabilizedImage`, the detection image of a
  RAW.** The generalized Anscombe transform of the green cells (`:green_cells`, or `:bin2`'s G plane) for
  Poisson–Gaussian noise, from the photosites' own code spans (`white − black` of each green), the gain, the conversion
  gain (electrons per code) and the read noise (codes): after it the noise has σ ≈ 1 from a few electrons to the white
  level. Neither noise parameter is in a RAW's metadata (LibRaw reports neither): they are stated per camera and ISO.
  `LinearRaw` gains the field `green_dn` (the two greens' `white − black`; `nothing` for `:demosaic`).
- **New: `asift_detect(s::StabilizedImage; noise_sigmas = 3.0, kwargs...)`**: the DoG contrast threshold in noise σ.
  The DoG's noise σ for unit iid input is derived from the pyramid's own impulse response (the root energy per sample
  phase) in its noisiest slice, the finest octave's finest (0.137 for the default upsampled pyramid, seed blur 1.6):
  every coarser slice and tilted view is held to more than `noise_sigmas` of its own σ. A detection whose centre lies
  within three of its scales of a saturated pixel is dropped (masked, not clamped). `intensity_thr` stays the threshold
  of `asift_detect` on an arbitrary image, which has no noise model.

- **New: `spread_strongest(detections, image_size, n; tolerance = 0.1, device)`, about `n` of the strongest
  detections spread over the image** — Suppression via Square Covering (SSC) of Bailo et al. (Pattern Recognition
  Letters 106, 2018), ported from the authors' code (https://github.com/BAILOOL/ANMS-Codes, MIT License, Copyright (c)
  2018 Oleksandr Bailo; credited in the docstring and the README's Citing section) and parallelised on
  KernelAbstractions: one implementation on the CPU and CUDA. The sequential covering is computed as per-cell maxima
  (two atomic reductions, no sort) and synchronous rounds of the greedy independent set of the cells in score order, and
  returns the authors' sequential result exactly, ties included (test/data/ssc_bailo.toml is their C++'s output). Where
  the authors' integer width search misses the tolerance band, the width is bisected between the bracketing integers,
  so the count lands in the band. It **replaces `strongest` (removed)**, the global top `n` by `|response|`, which on
  a frame with a bright window and a dim interior kept nearly every detection in the window; `LoMaDescriptor` now
  describes `spread_strongest(detections, size(img), max_detections)`. A caller of `strongest(d, n)` calls
  `spread_strongest(d, image_size, n)`.

- **LoMa on CUDA runs on the B200 (sm_100) too.** Microsoft's GPU builds of ONNX Runtime (1.30 and 1.31, CUDA 12 and
  13, and the CUDA plugin EP) carry device code for sm_60 to sm_90 and sm_120 only, no sm_100 and no PTX, so on a B200
  every kernel failed (`cudaErrorNoKernelImageForDevice`) and `:auto` fell back to the CPU. The CUDA path now runs
  conda-forge's build of ONNX Runtime 1.31.1 (`onnxruntime-cpp`, `cuda129`; sm_70 to sm_120 with sm_100), merged with
  the libraries it links (libprotobuf, libabseil, libre2) into one artifact, `onnxruntime_cuda` in the new
  `Artifacts.toml`, pinned by its tree hash and installed by the first CUDA probe from the four conda-forge files it pins
  by sha256 (Pkg cannot unpack a `.conda`; Julia's own 7z does), 0.34 GB once per depot; ONNXRunTime's own GPU
  download is no longer used. The CUDA 12 runtime and cuDNN stay the JLLs' (the runtime now at least 12.9, the build's
  floor; `libcusparse.so.12` is loaded too). The CPU path is unchanged (ONNXRunTime's Microsoft 1.30 build).
  `loma_provenance` records `onnxruntime_build` on CUDA. New dependencies: the stdlibs `Tar` and `p7zip_jll`.
- **LoMa runs on ONNX Runtime's CUDA provider wherever it can, with no CUDA setup.** `execution_provider = :auto` is the
  new default of `LoMaDescriptor` and `LoMaMatcher` (was `:cpu`): CUDA when a CUDA device is visible and the active
  environment holds `CUDA_Runtime_jll` and `CUDNN_jll`, otherwise the CPU with a warning naming why. `:cpu` and `:cuda`
  stay explicit; an explicit `:cuda` that cannot run is an error naming why. ONNX Runtime's GPU build needs CUDA's
  12.x runtime and cuDNN 9, which CUDA.jl 5.11 (runtime 13 on a CUDA 13 driver) and cuDNN.jl (1.4 does not resolve
  beside CUDA_Runtime_Discovery 2) could not give it: the package now takes the CUDA 12 builds of CUDA_Runtime_jll's and
  CUDNN_jll's artifacts (lazy: downloaded on the first CUDA session), selected by their `cuda` tag and loaded for ONNX
  Runtime alone, beside whatever runtime CUDA.jl uses. The two JLLs are weak dependencies pinned by `[compat]`
  (`CUDA_Runtime_jll` 0.21 - 0.26, `CUDNN_jll` 9.20 - 9) and never loaded as modules (CUDNN_jll would load the host
  CUDA's build under the same soname); an environment without them never resolves or downloads them. Neither CUDA.jl
  nor cuDNN.jl is needed for LoMa on the GPU any more; a cuDNN or CUDA 12 library already loaded from another file is
  refused by name. Needs a driver for CUDA ≥ 12 (≥ 525.60; Blackwell needs ≥ 570).
- **New (public): `onnx_cuda_status()`** (an `OnnxCudaStatus`: available, or the reason why not; the driver, CUDA
  runtime and cuDNN versions) and **`resolve_execution_provider(execution_provider)`**, `:auto`'s answer, for a caller
  that keys a cache on the provider.
- **ONNX Runtime's threads are the process's, not the machine's.** A new `threads` keyword of both constructors
  (default `Threads.nthreads()`, the stack's measure of the allotment: a Slurm job starts Julia with
  `--threads=$SLURM_CPUS_PER_TASK`) sizes ONNX Runtime's intra-op pool; the inter-op pool is 1 (the graphs run
  sequentially). ONNX Runtime sized its pool from all cores and pinned each thread to one, which inside a Slurm cgroup
  oversubscribed the allotment (six 28-CPU jobs drove a 112-thread machine to a load of 673) and failed to pin
  (`pthread_setaffinity_np failed`); an explicit size disables the pinning.
- **New (public): `loma_provenance(l)`** of a descriptor or matcher: the model file and its sha256, the provider used,
  the threads, ONNX Runtime's version and, on CUDA, the driver, runtime and cuDNN versions.
- `ONNXRunTime` compat is `~2.1`: the sessions are built through its C API (its `load_inference` takes no session
  options), so a new minor version is checked before it is admitted.
- Measured 2026-11-10 (RTX 4090, 32 threads, two 2128 × 1424 panorama frames, 2048 detections each, `:b` + `:b128`):
  both descriptions and the match in 0.11 s on CUDA against 3.46 s on the CPU. CUDA against the CPU on six frames:
  descriptors within 3.2e-3 of their largest component, 1103 of the 1109 matches of 15 frame pairs identical.

- **The one RAW path.** Every RAW → image step of the stack lives in this module (`load_raw`, `load_raw_mosaic`,
  `LinearRaw`; the LibRaw_jll extension decodes), and VisualGeometryToolkitCore's `check_raw_decoding` flags any other
  decoding path (rawpy, LibRaw C calls, dcraw, a demosaic or a CFA interpolation, a decoded RAW re-read as a PNG); the
  quality item runs it on this package.
- **New: `mode = :green_cells`**, THE image for geometry and radiometry: each 2×2 Bayer cell's two green photosites'
  mean, half resolution, one plane: `:bin2`'s green plane, bit for bit, from the same kernel. A cell is saturated when
  either green is.
- **New: the exact frame map of the cell grid.** `BayerCellFrame` (the cell images' COLMAP frame) and
  `raw_frame_map(L)`, a Core `SimilarityMap{2, …, BayerCellFrame, PixelFrame{:colmap}}`, `x ↦ 2x` onto the visible
  raw for every Bayer phase and visible-origin parity (an odd margin changes only the colour phase). `inv` is its
  inverse. A calibration matrix moves with it through Core's `reframed_calibration(raw_frame_map(L), K)`.
- **New: `green_cell_sites(L, row, column)`**, a green-cell value's footprint: its two diagonal green photosites' raw
  COLMAP centres, at `(±½, ∓½)` from the cell centre in an RGGB or BGGR cell and `(±½, ±½)` in GRBG or GBRG
  (`LinearRaw.green_offsets`).
- **New: `mode = :demosaic`, for display figures only**: LibRaw's AHD demosaic (`libraw_dcraw_process`, linear: gamma 1,
  no auto-brightening, no maximum adjustment, 16 bits, raw colour, white balance 1) of the visible area, with an
  optional `white_balance` and the gain applied in Julia. It carries no frame map: `raw_frame_map` and
  `green_cell_sites` refuse it, and its `plane_offsets` is empty (no pixel is a sample). Through
  `load_raw(path; mode = :demosaic)` only.
- **The EXIF flip is never applied, in any mode** (the demosaic zeroes LibRaw's `flip` before it copies the image
  out); a demosaic whose output is not the visible area's size is refused.
- **New: provenance.** `LinearRaw` carries its `frame_map`; `raw_provenance(L)` (public) gives the decoder, LibRaw's
  version, the mode, the gain and the frame map as plain data.
- **Breaking: the raw-grid `:green` mode is removed** (the green photosites at full resolution, zeros at red and blue):
  read `:green_cells`, or the mosaic itself. With it `LinearRaw.sampled` is removed (every pixel of every remaining
  mode carries its sample).
- **Breaking: `RawpyDecoder` and its PythonCall extension are removed**, with the package's `CondaPkg.toml` (rawpy,
  exifread) and the PythonCall weak dependency: `LibRawDecoder` is the one decoder.

- **The two-axis `matches!(ax_A, ax_B, cs)` and `matches(img_A, img_B, cs)` take weighted csponds**: a cspond is bare
  (opaque) or `Attributed(src => tgt, (; weight))`, its weight (Core's `weight_of`, in [0, 1]) the alpha of its markers.

## 0.6.1 (2026-11-09)

- On VisualGeometryToolkitCore 0.8 (compat `"0.8"`); nothing else changes.

## 0.6.0 (2026-11-07)

- The LoMa model digest and `cache_key` are VisualGeometryToolkitCore's `sha256_hex` (the same values); the SHA
  dependency is dropped.
- **Breaking, with VisualGeometryToolkitCore 0.7:** a `Detection`'s and a `DescribedFeature`'s blob shape is
  `shape_matrix(d)` (Core's renamed generic; `moment_matrix` is retired). `d.Σ` is unchanged.
- **`LoMaDescriptor` takes an image of any size**: `(descriptor)(img, detections)` with `img` a `3 × H × W` RGB
  channelview of any real element type and the detections in its COLMAP pixels. An image that is not the graph's
  fixed `LOMA_IMAGE_SIZE`² (public, 784) is resampled to it per channel by Core's bilinear `image_warp!` (the COLMAP
  scaling, Float64 math, Float32 taps: PanoramaStitching's hand resampling, bit for bit); a `3 × 784 × 784` image is
  taken as before. An image smaller than 784 on either side (an upscale) is resampled with its border replicated:
  the network's outermost pixel centres map within half a pixel past the image's, and their taps clamp to the
  border pixel instead of blending in the fill. The keypoint chart reads the image's own size, so the detections are never rescaled and the
  described features carry them as given (PanoramaStitching rescaled its frames to 784² by hand, moment matrices
  included, which the descriptor never read).
- **`strongest(detections, n)` (public)**, the selection `LoMaDescriptor` applies (formerly the private
  `_strongest`): the `n` largest `|response|`, in detector order. Ties at the cut now keep the earlier detections
  (a stable partial selection, O(N) as the former `partialsortperm`, which left that choice unspecified). BlobBoards' cap keeps the same set
  but re-sorts it by response.
- **`box_downsample` takes a 2-D image and any real or colour element type**, averaging in the element type's
  float (`Float32` for a `Float32`, `Gray{Float32}` or 8-bit image: BlobBoards' supersampling filter, bit for bit;
  `Float64` for a `Float64` or integer image, as before). A `Float32` 3-D image now gives a `Float32` result
  (formerly `Float64`).

## 0.5.0 (2026-11-06)

VisualGeometryToolkitCore 0.6 (compat `"0.6"`). Breaking with Core's
`CovMat`, which now holds a static matrix (full, diagonal or isotropic).

- Detections' covariances are Core 0.6's `CovMat{2, T, M}`; the spelled-out
  `CovMat{N, T, N², F}` types are gone.
- `CACHE_FORMAT` (2) is part of every cache key, so a JLD2 cache entry of the
  earlier `CovMat` layout is a miss, not a failed load.

## 0.4.2 (2026-11-04)

Additive: `RawMosaic.metadata` keeps its type and every field it had.

- **The RAW's EXIF as Core's typed values**: `exposure_settings`,
  `lens_settings`, `body_identity`, `lens_identity` and `capture_time`
  (public) of a `RawMosaic` or its metadata, and `extract_capture_settings(frames;
  lens = nothing)`, a capture's `CaptureSettings` (one body; a stated lens
  overrides EXIF's claim, with a warning when they disagree: an adapted lens
  is misnamed, the FX3 with Sigma EF lenses claims a Sony "DT" lens).
- **`load_raw_metadata(path; decoder)`**: the EXIF fields alone; LibRaw reads
  the headers without unpacking the photosites.
- `metadata` gains `make`, `body_serial`, `lens_make`, `lens_model`,
  `lens_serial` and `focus_distance_m` (`nothing` where absent; a non-finite
  or non-positive focus distance too, never a distance). LibRaw: the lens from
  `libraw_get_lensinfo` (its leading fields mirrored, the offset checked);
  no body serial or focus distance. rawpy: the EXIF tags.
- Compat: VisualGeometryToolkitCore 0.5.4.

## 0.4.1 (2026-11-04)

- **New `load_raw(path; mode, gain = 1, decoder = LibRawDecoder())`**: a camera
  RAW file (any Bayer format LibRaw reads) as exact linear values,
  `(raw − black[c]) / (white − black[c]) × gain`, camera raw colour, white
  balance 1, in the sensor's orientation (no EXIF flip; the raw's COLMAP frame).
  Ported from RadiometricCamera's `convert_raw.py`. Modes `:bin2` (2×2 Bayer
  quads: R, the mean of the two greens, B; half size) and `:green` (the green
  photosites alone on the raw grid). Its `fullres` demosaic is not provided.
  Returns a `LinearRaw`: `image`, `sampled`, `saturated` (a quad is saturated
  when ANY of its photosites is at the white level, `raw ≥ white`, no margin),
  `plane_offsets` (each plane's true sample position: the green at the quad
  centre, R and B `(±¼, ±¼)` binned pixels from it) and the EXIF `metadata`.
  Rounded to 16 bits, `:bin2` equals `convert_raw.py --mode bin2`'s linear PNG
  value for value.
- **New `load_raw_mosaic`, `RawMosaic`, `LinearRaw(mosaic; mode, gain)`**: the
  decode and the processing apart; the processing is all Julia, one
  KernelAbstractions kernel per mode. A `RawMosaic` holds the colour
  `pattern` (square, at most 4 × 4, validated) and the visible area's origin;
  each photosite's colour index is `photosite_colors(mosaic)`, the pattern
  tiled from the raw origin (rawpy's `raw_colors_visible`, which the rawpy
  decoder checks against it).
- **Black levels: one rule.** LibRaw's per-channel black levels with a 2×2
  spatial pattern folded in; a spatial pattern with no per-channel form is
  refused, naming the file, never dropped. rawpy exposes only the per-channel
  levels (its `black_level_per_channel` drops such a pattern), so
  `RawpyDecoder()` cannot apply the rule; its docstring says so.
- **Two decoders, both weak dependencies**: `LibRawDecoder()` (new extension
  `VisualGeometryToolkitFeaturesLibRawExt` on LibRaw_jll 0.22.2: a `ccall`
  layer whose mirrored `libraw_data_t` layout is checked against LibRaw's
  accessors on every file, refusing a LibRaw other than 0.22) and
  `RawpyDecoder()` (new extension `VisualGeometryToolkitFeaturesPythonExt` on
  PythonCall; rawpy and exifread declared in the new `CondaPkg.toml`, reading
  one open file object). Without
  the extension loaded, `load_raw` names the package to load.
