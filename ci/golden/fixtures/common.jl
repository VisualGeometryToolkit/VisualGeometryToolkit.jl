# The fixtures more than one golden group builds: VisualGeometryToolkitEstimation's find_* scenes and too-few sets
# (`h_scene`, `stereo_scene`, `pnp_scene`, `radial_homography_scene`, `hand_eye_scene`, `radial_camera_pairs`,
# `few_points`, `few_correspondences`, `PCAM`, `R_PNP`, `T_PNP`, `RW`, `RH`, `M_RH`, `RC_CENTER`) — the suite's own
# fixture file, test/fixtures/find_scenes.jl, the one copy — and its pixel-pose scenes (`true_pose`, `perturbed`,
# `scene_points`, `pixel_pairs`, test/fixtures/pixel_pose_scenes.jl), its conic scenes (`circle_scene`, `ellipse_scene`,
# `contaminated_boundary`, `C₀`, `R₀`, `E₀`, test/fixtures/conic_scenes.jl), and the golden groups' own shared sets.
# Each group runs in its own module (record.jl), and its fixture file includes this file once.
const _GOLDEN_EST_FIXTURES = joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures")
Base.include(@__MODULE__, joinpath(_GOLDEN_EST_FIXTURES, "find_scenes.jl"))
Base.include(@__MODULE__, joinpath(_GOLDEN_EST_FIXTURES, "pixel_pose_scenes.jl"))
Base.include(@__MODULE__, joinpath(_GOLDEN_EST_FIXTURES, "conic_scenes.jl"))

# Too few data for a homography's minimal sample (the suite's ransac_no_model cases; the homography and noise_scales groups).
homography_too_few() = let HS = VgtkEstimation.synthetic_scenes().HomographyScenes
    (s, d) = HS.homography_pairs(StableRNG(42), HS.H_TRUE; n_in = 3, n_out = 0, σ = 0.5); StructArray(s .=> d)
end
