# The old minimal-solver protocol of VisualGeometryToolkitCore at 3d8a06da, `solve(solver, cs, idx) -> FixedModels`, for
# every MinimalSolver but the native Homography4pt (the homography pilot's): Circle3pt, Ellipse5pt, Rotation2pt,
# Fundamental7pt, Essential5pt, P3P (native and PoseLib backends), Homography5ptRadial (with and without its
# extra check), RadialCamera5pt, and Homography4pt's PoseLib backend.
using PoseLib: PoseLib
using VisualGeometryToolkitCore: Circle3pt, Ellipse5pt, Rotation2pt, Fundamental7pt, Essential5pt, P3P, Homography5ptRadial,
    RadialCamera5pt, Homography4pt, NativeSolver, PoseLibSolver

const _SOLVE_ALL = ((s, cs, idxs),) -> [solve(s, cs, idx) for idx in idxs]
const _SOLVE_SETS = ((s, sets, k),) -> [solve(s, cs, collect(1:k)) for cs in sets]

# One case per (fixture, solver): the solver, the data and the samples are the fixture (a solver's settings, like
# RadialCamera5pt's centre or Homography5ptRadial's extra check, are the problem's geometry, so they travel with it).
function _scene_cases!(cases, family, name, solvers, cs, k, seed; n = 300)
    idxs = index_samples(StableRNG(100 + seed), length(cs), k, n)
    for (lbl, s) in solvers
        case!(cases, "$family/$name/$(lbl)_$(n)samples/seed$seed", (s, cs, idxs), _SOLVE_ALL)
    end
end

function record_core_minimal(cases)
    native_poselib(T) = ("native" => T(NativeSolver()), "poselib" => T(PoseLibSolver()))
    for seed in SEEDS
        rng = StableRNG(seed)
        # --- Circle3pt ------------------------------------------------------------------------------------------
        _scene_cases!(cases, "circle3pt", "conic_scene_200", ("default" => Circle3pt(),), first(circle_scene(StableRNG(seed), 200)), 3, seed)
        case!(cases, "circle3pt/exact_triples_200/default/seed$seed", (Circle3pt(), circle_triples(StableRNG(seed)), 3), _SOLVE_SETS)
        # --- Ellipse5pt -----------------------------------------------------------------------------------------
        _scene_cases!(cases, "ellipse5pt", "conic_scene_200", ("default" => Ellipse5pt(),), first(ellipse_scene(StableRNG(seed), 200)), 5, seed)
        case!(cases, "ellipse5pt/exact_quintuples_200/default/seed$seed", (Ellipse5pt(), ellipse_quintuples(StableRNG(seed)), 5), _SOLVE_SETS)
        # --- Rotation2pt ----------------------------------------------------------------------------------------
        R = RotMatrix(QuatRotation(normalize(randn(StableRNG(50 + seed), 4))))
        _scene_cases!(cases, "rotation2pt", "fisheye_scene_100", ("default" => Rotation2pt(),), rotation_scene(StableRNG(seed), R, 100), 2, seed)
        case!(cases, "rotation2pt/exact_pairs_200/default/seed$seed", (Rotation2pt(), rotation_pairs(StableRNG(seed)), 2), _SOLVE_SETS)
        # --- Fundamental7pt -------------------------------------------------------------------------------------
        ep = minimal_epipolar_scene(StableRNG(seed))
        _scene_cases!(cases, "fundamental7pt", "fundmat_scene_130", native_poselib(Fundamental7pt), minimal_fundmat_pairs(StableRNG(seed)), 7, seed)
        _scene_cases!(cases, "fundamental7pt", "layer_scene_20", native_poselib(Fundamental7pt), Correspondences(ep.u1, ep.u2), 7, seed; n = 100)
        # --- Essential5pt ---------------------------------------------------------------------------------------
        _scene_cases!(cases, "essential5pt", "refine_scene_200", native_poselib(Essential5pt), minimal_essential_scene(StableRNG(seed)), 5, seed)
        _scene_cases!(cases, "essential5pt", "layer_scene_20", native_poselib(Essential5pt), Correspondences(ep.x1b, ep.x2b), 5, seed; n = 100)
        eq = essential_quintuples(StableRNG(seed))
        for (lbl, s) in native_poselib(Essential5pt)
            case!(cases, "essential5pt/native_ensemble/$lbl/seed$seed", (s, eq, 5), _SOLVE_SETS)
        end
        case!(cases, "essential5pt/layer_scene_20/refused_plane_points/seed$seed", (Essential5pt(), Correspondences(ep.u1, ep.u2)),
            ((s, cs),) -> solve(s, cs, 1:5))
        # --- P3P ------------------------------------------------------------------------------------------------
        _scene_cases!(cases, "p3p", "pnp_scene_180", native_poselib(P3P), pnp_bearing_scene(StableRNG(seed)), 3, seed)
        pt = p3p_triples(StableRNG(seed))
        for (lbl, s) in native_poselib(P3P)
            case!(cases, "p3p/native_ensemble/$lbl/seed$seed", (s, pt, 3), _SOLVE_SETS)
        end
        # --- Homography5ptRadial --------------------------------------------------------------------------------
        h5 = ("default" => Homography5ptRadial(), "extra_check" => Homography5ptRadial(extra_check = true))
        rh = radial_homography_normalized_scene(StableRNG(seed))
        _scene_cases!(cases, "homography5pt_radial", "find_scene_100_raw", h5, rh.raw, 5, seed)
        _scene_cases!(cases, "homography5pt_radial", "find_scene_100_normalized", h5, rh.normalized, 5, seed)
        rq = radial_quintuples(StableRNG(seed))
        for (lbl, s) in h5
            case!(cases, "homography5pt_radial/synthetic_ensemble/$lbl/seed$seed", (s, rq, 5), _SOLVE_SETS)
        end
        # --- RadialCamera5pt ------------------------------------------------------------------------------------
        _scene_cases!(cases, "radial_camera5pt", "board_scene_101", ("center_rc" => RadialCamera5pt(; center = RC_CENTER),),
            radial_camera_scene(StableRNG(seed)), 5, seed)
        # --- Homography4pt, PoseLib backend ---------------------------------------------------------------------
        _scene_cases!(cases, "homography4pt", "htrue_100+30", ("poselib" => Homography4pt(PoseLibSolver()),), homography_scene(StableRNG(seed)), 4, seed)
        _scene_cases!(cases, "homography4pt", "layer_scene_20", ("poselib" => Homography4pt(PoseLibSolver()),), homography_layer_scene(StableRNG(seed)), 4, seed; n = 100)
    end
    # --- fixed samples: degenerate, golden, the tests' explicit ones ----------------------------------------------
    for (key, pts) in pairs(circle_degenerate())
        case!(cases, "circle3pt/degenerate/$key", (Circle3pt(), pts), ((s, p),) -> solve(s, p, [1, 2, 3]))
    end
    for (key, pts) in pairs(ellipse_degenerate())
        case!(cases, "ellipse5pt/degenerate/$key", (Ellipse5pt(), pts), ((s, p),) -> solve(s, p, 1:5))
    end
    for (key, cs) in pairs(rotation_degenerate())
        case!(cases, "rotation2pt/degenerate/$key", (Rotation2pt(), cs), ((s, cs),) -> solve(s, cs, [1, 2]))
    end
    ep = minimal_epipolar_scene(Random.MersenneTwister(2026))
    for (lbl, s) in (("native", Fundamental7pt()), ("poselib", Fundamental7pt(PoseLibSolver())))
        case!(cases, "fundamental7pt/layer_scene_mt2026/first7/$lbl", (s, Correspondences(ep.u1, ep.u2)), ((s, cs),) -> solve(s, cs, 1:7))
        case!(cases, "fundamental7pt/degenerate/identical/$lbl", (s, Correspondences(fill(ep.u1[1], 7), fill(ep.u2[1], 7))), ((s, cs),) -> solve(s, cs, 1:7))
    end
    for (lbl, s) in (("native", Essential5pt()), ("poselib", Essential5pt(PoseLibSolver())))
        case!(cases, "essential5pt/layer_scene_mt2026/first5/$lbl", (s, Correspondences(ep.x1b, ep.x2b)), ((s, cs),) -> solve(s, cs, 1:5))
    end
    for (key, cs) in pairs(p3p_degenerate()), (lbl, s) in (("native", P3P()), ("poselib", P3P(PoseLibSolver())))
        case!(cases, "p3p/fixed/$key/$lbl", (s, cs), ((s, cs),) -> solve(s, cs, 1:3))
    end
    rg = radial_golden()
    for (lbl, s) in (("default", Homography5ptRadial()), ("extra_check", Homography5ptRadial(extra_check = true)))
        case!(cases, "homography5pt_radial/homlib_golden/$lbl", (s, rg.golden), ((s, cs),) -> solve(s, cs, 1:5))
        case!(cases, "homography5pt_radial/collinear_sources/$lbl", (s, rg.collinear), ((s, cs),) -> solve(s, cs, 1:5))
    end
    rb = radial_board()
    for (name, cs) in (("raw", rb.raw), ("normalized", rb.normalized))
        idxs = index_samples(Random.Xoshiro(3), length(cs), 5, 100)
        case!(cases, "homography5pt_radial/centre_board_$name/default_100samples", (Homography5ptRadial(), cs, idxs), _SOLVE_ALL)
    end
    case!(cases, "homography5pt_radial/refused_backend", nothing, _ -> Homography5ptRadial(backend = PoseLibSolver()))
    b = radial_camera_board()
    for (name, cs, c) in (("centred", b.centred, SVector(0.0, 0.0)), ("raw", b.raw, b.center))
        s = RadialCamera5pt(; center = c)
        case!(cases, "radial_camera5pt/contract_board_$name/corners_and_interior", (s, cs), ((s, cs),) -> solve(s, cs, [1, 5, 8, 21, 25]))
        case!(cases, "radial_camera5pt/contract_board_$name/first_column", (s, cs), ((s, cs),) -> solve(s, cs, 1:5))
        idxs = index_samples(StableRNG(7), length(cs), 5, 300)
        case!(cases, "radial_camera5pt/contract_board_$name/300samples", (s, cs, idxs), _SOLVE_ALL)
    end
    case!(cases, "radial_camera5pt/collinear/default", (RadialCamera5pt(), b.collinear), ((s, cs),) -> solve(s, cs, 1:5))
    return cases
end
