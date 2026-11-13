"""Recompute a vgio `content_id` from a stored rig or target file, outside Julia.

    python content_id.py <file.json>       prints the content id (64 lowercase hex digits)

An independent implementation of the "Content identity" specification
(VisualGeometryToolkitIO docs, guide/content-identity.md): the identity tree of
the record (its body in base units — metres, the colmap pixel frame — each
length normalised to 15 significant digits), serialized by RFC 8785 (JCS),
hashed by SHA-256. It shares no code with the Julia implementation: the JCS
writer below formats numbers by ECMAScript's Number::toString, and the file's
numbers are read as decimals (a length in a power-of-ten unit is the decimal
shifted, then rounded once to a double). Only power-of-ten length units are
read (m, cm, mm, um, μm, km); a layout target (whose points are its layout's
geometry) is refused.
"""

import hashlib
import json
import math
import re
import sys
from decimal import Decimal

# --- RFC 8785 (JCS) ----------------------------------------------------------


def jcs_number(x):
    """ECMAScript Number::toString of a finite double (RFC 8785 §3.2.2.3)."""
    if isinstance(x, bool):
        raise TypeError("a boolean is not a number")
    if isinstance(x, int):
        if abs(x) > 2**53:
            raise ValueError(f"integer {x} is not exactly a double")
        x = float(x)
    if not math.isfinite(x):
        raise ValueError("JCS has no NaN or Infinity")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    m = re.fullmatch(r"(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?", repr(abs(x)))   # Python's shortest round-trip digits
    ip, fp, e = m.group(1), m.group(2) or "", int(m.group(3) or 0)
    alldigits = ip + fp
    stripped = alldigits.lstrip("0")
    n = len(ip) + e - (len(alldigits) - len(stripped))   # the value is 0.d1d2…dk × 10^n
    digits = stripped.rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        s = digits + "0" * (n - k)
    elif 0 < n <= 21:
        s = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        s = "0." + "0" * (-n) + digits
    else:
        exp = n - 1
        s = digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + ("+" if exp >= 0 else "-") + str(abs(exp))
    return sign + s


_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def jcs_string(s):
    out = ['"']
    for c in s:
        if c in _ESCAPES:
            out.append(_ESCAPES[c])
        elif ord(c) < 0x20:
            out.append("\\u%04x" % ord(c))
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


def jcs(x):
    """The RFC 8785 serialization of a JSON value (dict, list, str, int, float, bool, None)."""
    if x is None:
        return "null"
    if x is True:
        return "true"
    if x is False:
        return "false"
    if isinstance(x, (int, float)):
        return jcs_number(x)
    if isinstance(x, str):
        return jcs_string(x)
    if isinstance(x, list):
        return "[" + ",".join(jcs(v) for v in x) + "]"
    if isinstance(x, dict):
        keys = sorted(x, key=lambda k: k.encode("utf-16-be"))   # by UTF-16 code units
        return "{" + ",".join(jcs_string(k) + ":" + jcs(x[k]) for k in keys) + "}"
    raise TypeError(f"not a JSON value: {type(x)}")


# --- the identity tree -------------------------------------------------------

# The power-of-ten length units: the decimal exponent of one unit in metres.
DECIMAL_UNITS = {"m": 0, "cm": -2, "mm": -3, "um": -6, "μm": -6, "km": 3}
# The pixel frames: the position of the image's top-left corner in each (conventions/pixel-origins.md);
# the identity's frame is colmap's (corner at 0).
PIXEL_CORNER = {"opencv": -0.5, "colmap": 0.0, "julia": 0.5}
# Which parameters of each target calibration kind are lengths (VisualGeometryToolkitCore's `length_mask`);
# a thin-plate kind's parameters are all lengths.
LENGTH_MASK = {
    "rigid": [],
    "euclidean": [False, True, True],   # [θ, tx, ty]
    "scale": [False],
    "aspect": [False],
    "skew": [False],
}
POSE_DIMS = [0, 0, 0, 1, 1, 1]          # a pose covariance over [δω; δt]


def persisted(x):
    """The float nearest to x rounded to 15 significant decimal digits (0 and non-finite as they are)."""
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return x
    return float("%.14e" % x)


class Units:
    def __init__(self, units):
        if units["length"] not in DECIMAL_UNITS:
            raise ValueError(f"only power-of-ten length units are read here, not {units['length']!r}")
        if units["length_roles"]:
            raise ValueError("a rig or target record names no length role")
        self.exp = DECIMAL_UNITS[units["length"]]
        self.corner = PIXEL_CORNER[units["pixel"]]

    def power(self, x, n):
        """A file number of length dimension n (0: none) in base units, normalised."""
        return persisted(Decimal(x).scaleb(self.exp * n))

    def length(self, x):
        return self.power(x, 1)

    def lengths(self, xs):
        return [self.length(x) for x in xs]

    def pixel(self, x):
        """A pixel coordinate in the colmap frame: the file's double, less the frame's corner offset (in doubles)."""
        return float(x) - self.corner


def number(x):
    return float(x) if isinstance(x, Decimal) else x


def covariance(c, dims, convert):
    """A covariance in its storage, each entry (i, j) through convert(entry, dims[i] + dims[j])."""
    if isinstance(c, list):
        return [[convert(c[i][j], dims[i] + dims[j]) for j in range(len(c))] for i in range(len(c))]
    if "diagonal" in c:
        return {"diagonal": [convert(v, 2 * dims[i]) for i, v in enumerate(c["diagonal"])]}
    return {"isotropic": convert(c["isotropic"], 2 * dims[0])}


def pose(p, u):
    out = {"rotation": [number(q) for q in p["rotation"]], "translation": u.lengths(p["translation"])}
    if "covariance" in p:
        out["covariance"] = covariance(p["covariance"], POSE_DIMS, u.power)
    return out


def rect(r, u):
    return {"size": {"width": u.length(r["size"]["width"]), "height": u.length(r["size"]["height"])},
            "origin": u.lengths(r["origin"])}


def calibration_mask(c):
    t = c["type"]
    if t == "composition":
        return [b for m in c["compose"] for b in calibration_mask(m)]
    if t in ("thin_plate_spline", "height_field"):
        return [True] * len(c["parameters"])
    return LENGTH_MASK[t]


def calibration(c, convert):
    """A target calibration record, its lengths through convert(x, power)."""
    out = {"type": c["type"]}
    if c["type"] == "composition":
        out["compose"] = [calibration(m, convert) for m in c["compose"]]
    if "nodes" in c:
        out["nodes"] = [[convert(x, 1), convert(y, 1)] for x, y in c["nodes"]]
    if "parameters" in c:
        mask = calibration_mask(c)
        out["parameters"] = [convert(x, 1) if mask[i] else number(x) for i, x in enumerate(c["parameters"])]
    if "covariance" in c:
        dims = [int(b) for b in calibration_mask(c)]
        out["covariance"] = covariance(c["covariance"], dims, convert)
    return out


def target(t, u):
    if t["type"] == "point_target":
        pts = []
        for p in t["points"]:
            key = "xy" if "xy" in p else "xyz"
            q = {key: u.lengths(p[key])}
            if "covariance" in p:
                q["covariance"] = covariance(p["covariance"], [1] * len(p[key]), u.power)
            pts.append(q)
        return {"type": t["type"], "points": pts}
    if "layout" in t:
        raise ValueError("a layout target's identity is of its layout's points, which this script does not compute")
    out = {"type": t["type"]}
    if "reference" in t:
        out["reference"] = t["reference"]
    else:
        out["nominal"] = [u.lengths(p) for p in t["nominal"]]
    # a placed target's calibration is in the placement's dimensionless chart: its numbers as they are
    placed = "placement" in t
    out["calibration"] = calibration(t["calibration"], (lambda x, n: number(x)) if placed else u.power)
    if "board" in t:
        out["board"] = rect(t["board"], u)
    if "canvas" in t:
        cv = rect(t["canvas"], u)
        if "texture" in t["canvas"]:
            tx = t["canvas"]["texture"]
            cv["texture"] = {"path": tx["path"], "sha256": tx["sha256"], **rect(tx, u)}
        out["canvas"] = cv
    if placed:
        out["placement"] = {"origin": u.lengths(t["placement"]["origin"]), "side": u.length(t["placement"]["side"])}
    return out


def lattice(sp, convert_controls):
    out = {"area": [number(x) for x in sp["area"]], "grid_size": dict(sp["grid_size"])}
    if "present" in sp:
        out["present"] = sp["present"]
    out["controls"] = {k: [[convert_controls(k)(x) for x in c] for c in v] for k, v in sp["controls"].items()}
    return out


def plain(x):
    """A record of plain numbers (a lens record: unitless coefficients), as it is."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [plain(v) for v in x]
    return number(x)


def camera(c, u):
    k = c["K"]
    out = {"image_size": dict(c["image_size"]),
           "K": {"fx": number(k["fx"]), "fy": number(k["fy"]), "cx": u.pixel(k["cx"]), "cy": u.pixel(k["cy"]),
                 "skew": number(k["skew"])}}
    if "lens" in c:
        out["lens"] = plain(c["lens"])
    else:
        out["type"] = c["type"]
        out["spline"] = lattice(c["spline"], lambda key: u.length if key == "point" else number)
    if "pitch" in c:
        out["pitch"] = {"width": u.length(c["pitch"]["width"]), "height": u.length(c["pitch"]["height"])}
    if "lens_coverage" in c:
        (shape, v), = c["lens_coverage"].items()
        moved = [u.pixel(v[0]), u.pixel(v[1])] + [number(x) for x in v[2:]]
        out["lens_coverage"] = {shape: moved}
    return out


def identity_tree(d):
    kind = d["schema"].split("/")[1]
    u = Units(d["units"])
    if kind == "target":
        return target(d["target"], u)
    if kind != "rig":
        raise ValueError(f"a content id is a rig's or a target's, not a {kind}'s")
    tree = {"frame": d["frame"]}
    if "sensors" in d:
        tree["sensors"] = [{"id": s["id"], "camera": camera(s["camera"], u), "extrinsics": pose(s["extrinsics"], u)}
                           for s in d["sensors"]]
    if "targets" in d:
        ts = sorted(d["targets"], key=lambda t: t["id"].encode("utf-8"))
        tree["targets"] = [{"id": t["id"], "target": target(t["target"], u), "pose": pose(t["pose"], u)} for t in ts]
    return tree


def content_id(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f, parse_float=Decimal)
    return hashlib.sha256(jcs(identity_tree(d)).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(content_id(p))
