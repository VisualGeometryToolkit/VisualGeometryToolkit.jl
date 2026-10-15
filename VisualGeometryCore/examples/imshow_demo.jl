# Image display through the Makie extension: the Axis-based `imshow`, and
# the SpecApi `CampixelImshow` scene with primitive overlays.
#
#   julia --project=. examples/imshow_demo.jl     (GLMakie must be loadable)

using GLMakie
using VisualGeometryCore
using VisualGeometryCore: CampixelImshow
import VisualGeometryCore as VGC   # VGC.poly! — Makie exports its own poly!
using Colors: Gray
import GLMakie.Makie.SpecApi as S

# A synthetic 640×480 image (no test-image dependency)
img = [Gray(0.5 + 0.45 * sin(i / 25) * cos(j / 40)) for i in 1:480, j in 1:640]

# ── 1. Axis-based display; primitives overlay on the Axis ───────────────
fig, ax, plt = imshow(img)
scatter!(ax, [160.0, 480.0], [120.0, 360.0]; color = :red, markersize = 12)
e = Ellipse(Point2(320.0, 240.0), 90.0, 45.0, π / 6)
VGC.poly!(ax, e; strokecolor = :cyan, strokewidth = 2)          # Ellipse → polygon recipe
d = Disc(Point2(160.0, 120.0), 20.0)
VGC.poly!(ax, region(d, 3.0); strokecolor = :yellow, strokewidth = 2)   # 3σ circle
display(fig)

# ── 2. SpecApi scene with a campixel camera; overlays are PlotSpecs ─────
scene = CampixelImshow(img)
push!(scene.plots, S.Poly(region(d, 3.0); color = :transparent, strokecolor = :yellow, strokewidth = 2))
fig2 = plot(S.GridLayout([scene]); figure = (; size = (640, 480)))
display(fig2)

println("imshow demo done")
