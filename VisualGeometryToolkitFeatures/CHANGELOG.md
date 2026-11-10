# VisualGeometryToolkitFeatures changelog

## Unreleased

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
