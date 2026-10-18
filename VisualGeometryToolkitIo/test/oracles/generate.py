"""Frozen oracles for the lens conventions: the pixels OpenCV (cv2) and COLMAP
(pycolmap) compute for each model, on fixed rays, so test/lens_conventions_tests.jl
checks VisualGeometryToolkitIo's adapters against the reference implementations without
Python at test time.

    pixi run -e py python test/oracles/generate.py   # writes test/oracles/lens_conventions.json
"""
import json
import pathlib

import cv2
import numpy as np
import pycolmap

rng = np.random.default_rng(20261008)


def rays(n, max_deg):
    th = np.deg2rad(max_deg) * np.sqrt(rng.random(n))
    ps = 2 * np.pi * rng.random(n)
    s = 0.5 + 2 * rng.random(n)
    return np.stack([s * np.sin(th) * np.cos(ps), s * np.sin(th) * np.sin(ps), s * np.cos(th)], axis=1)


K = dict(fx=812.5, fy=809.25, cx=640.75, cy=480.25)
Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1.0]])
cases = []

# --- OpenCV projectPoints: 4, 5, 8, 12, 14 coefficients ---------------------------
opencv = {
    4: [-0.28, 0.07, 1.0e-3, -5.0e-4],
    5: [-0.2817, 0.0712, 0.00091, -0.00042, -0.0061],
    8: [-0.12, 0.03, 1.0e-3, -5.0e-4, 0.004, 0.02, -0.001, 0.0003],
    12: [-2.7067990842834013, 4.003544352914542, 0.0028263106588956688, 0.0017230351557208537,
         1.5827483616019924, -2.5409744152370775, 3.5017575444899127, 2.4059525010963547,
         -3.137923848171712e-5, 7.363153842759605e-5, -0.00322446080396022, 0.00027475714123090727],
    14: [-0.12, 0.03, 1.0e-3, -5.0e-4, 0.004, 0.02, -0.001, 0.0003, -5.5e-3, 7.0e-4, 2.2e-3, -1.2e-3, 0.01, -0.02],
}
for n, d in opencv.items():
    X = rays(64, 40)
    px, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Km, np.array(d))
    cases.append(dict(convention="opencv", model=f"projectPoints{n}", K=K, coeffs=d, rays=X.tolist(), pixels=px.reshape(-1, 2).tolist()))

# --- OpenCV fisheye ------------------------------------------------------------------
d = [-0.043, 0.011, -0.0083, 0.0015]
X = rays(64, 80)
px, _ = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Km, np.array(d))
cases.append(dict(convention="opencv", model="fisheye", K=K, coeffs=d, rays=X.tolist(), pixels=px.reshape(-1, 2).tolist()))

# --- COLMAP: every model -----------------------------------------------------------
f, f2, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
colmap = {
    "SIMPLE_PINHOLE": ([f, cx, cy], 60),
    "PINHOLE": ([f, f2, cx, cy], 60),
    "SIMPLE_RADIAL": ([f, cx, cy, -0.12], 40),
    "RADIAL": ([f, cx, cy, -0.12, 0.03], 40),
    "OPENCV": ([f, f2, cx, cy, -0.28, 0.07, 1.0e-3, -5.0e-4], 40),
    "OPENCV_FISHEYE": ([f, f2, cx, cy, -0.043, 0.011, -0.0083, 0.0015], 80),
    "FULL_OPENCV": ([f, f2, cx, cy, -0.12, 0.03, 1.0e-3, -5.0e-4, 0.004, 0.02, -0.001, 0.0003], 40),
    "FOV": ([f, f2, cx, cy, 0.9], 60),
    "SIMPLE_RADIAL_FISHEYE": ([f, cx, cy, -0.04], 80),
    "RADIAL_FISHEYE": ([f, cx, cy, -0.04, 0.008], 80),
    "THIN_PRISM_FISHEYE": ([f, f2, cx, cy, 0.21047, 0.21102, -5.36e-6, 0.000515, -0.158, 0.407, -8.46e-5, 0.000861], 60),
    "RAD_TAN_THIN_PRISM_FISHEYE": ([f, f2, cx, cy, 0.1, -0.02, 0.003, 0.0, 0.0, 1.0e-4, 3.0e-4, -2.0e-4, 1.0e-3, 0.0, -5.0e-4, 2.0e-5], 70),
    "SIMPLE_DIVISION": ([f, cx, cy, -0.2], 60),
    "DIVISION": ([f, f2, cx, cy, -0.2], 60),
    "SIMPLE_FISHEYE": ([f, cx, cy], 85),
    "FISHEYE": ([f, f2, cx, cy], 85),
}
for model, (params, deg) in colmap.items():
    cam = pycolmap.Camera(model=model, width=1280, height=960, params=params)
    X = rays(64, deg)
    px = cam.img_from_cam(X)
    cases.append(dict(convention="colmap", model=model, params=params, rays=X.tolist(), pixels=np.asarray(px).tolist()))

out = pathlib.Path(__file__).with_name("lens_conventions.json")
out.write_text(json.dumps(dict(
    generator="test/oracles/generate.py",
    opencv=cv2.__version__, pycolmap=pycolmap.__version__, cases=cases), indent=1))
print(f"{len(cases)} cases -> {out}")
