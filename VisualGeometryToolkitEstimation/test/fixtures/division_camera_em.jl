# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# DivisionCameraFit's EM table at another iteration cap (division_camera_fit_tests.jl, and the golden records of
# ci/golden/record/est_camera_fits.jl). Include-once safe (functions only).
using TOML: TOML
import VisualGeometryToolkitEstimation as VgtkEstimation

# the fit's EM table (configs/default.toml [DivisionCameraFit] mixture_em) at another iteration cap
em_at(max_iter) = VgtkEstimation.mixture_em_from_toml(merge(TOML.parsefile(VgtkEstimation.default_config_path())["division_camera_fit"]["mixture_em"],
                                                            Dict("max_iter" => max_iter)); section = "division_camera_fit.mixture_em")
