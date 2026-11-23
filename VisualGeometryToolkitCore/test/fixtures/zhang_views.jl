# The plane homographies of the Zhang calibration's synthetic views (zhang_intrinsics_tests.jl, and the golden
# records of ci/golden/fixtures/core_closed_forms.jl): a raw-pixel K about the principal point ZHANG_CENTER and six
# fixed board poses. Data constructors only; include-once safe (functions and an isbits constant).
using StaticArrays: SVector, SMatrix, @SMatrix
using Rotations: RotXYZ
using VisualGeometryToolkitCore: HomographyMat

const ZHANG_CENTER = SVector(640.0, 480.0)
# the homography K [r₁ r₂ t] of the board plane z = 0
zhang_homography(K, R, t) = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(K * hcat(R[:, 1], R[:, 2], t))))
# six views of the board through the raw-pixel K of focal lengths (fx, fy), principal point ZHANG_CENTER
function zhang_views(fx, fy)
    c = ZHANG_CENTER
    K = @SMatrix [fx 0.0 c[1]; 0.0 fy c[2]; 0.0 0.0 1.0]
    angles = [(0.4, -0.3, 0.2), (-0.5, 0.2, -0.1), (0.2, 0.5, 0.3), (-0.3, -0.4, 0.1), (0.6, 0.1, -0.2), (0.1, -0.6, 0.4)]
    ts = [SVector(0.3, -0.2, 2.5), SVector(-0.4, 0.1, 3.0), SVector(0.2, 0.3, 2.2), SVector(-0.1, -0.3, 2.8), SVector(0.5, 0.2, 3.5),
          SVector(0.0, 0.4, 2.0)]
    return [zhang_homography(K, RotXYZ(a...), t) for (a, t) in zip(angles, ts)]
end
