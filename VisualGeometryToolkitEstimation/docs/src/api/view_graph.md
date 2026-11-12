# View and Observation Graphs

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

Two graphs carry a multi-view problem's topology, as in Theia, GLOMAP and
OpenSfM. A [`ViewGraph`](@ref) has one vertex per frame and one edge per
verified pair `(i, j)`, `i < j`; an [`ObservationGraph`](@ref) has frames on one
side, the things they observe (targets, boards, tracks) on the other, and one
edge per observation. The graph algorithms are Graphs.jl's; what is defined here
is the label bookkeeping and the geometry they cannot know. Each graph's edge
dictionary is its source of truth: [`edgelist`](@ref) returns it and the graph
is rebuilt from it.

## ViewGraph

```@docs
ViewGraph
ViewGraph(::AbstractVector, ::AbstractDict)
views
adjacent_views
induced
```

## ObservationGraph

```@docs
ObservationGraph
ObservationGraph(::AbstractDict)
observers
observed
observed_in
observers_of
covisibility
```

## Structure and connectivity policies

A method picks its policy: one common gauge ([`require_connected`](@ref)), the
largest part ([`keep_largest_component`](@ref), the SfM default), or one gauge
per component ([`label_components`](@ref), what a calibration does).

```@docs
edgelist
nedges
components
label_components
require_connected
keep_largest_component
articulation
bridges
```

## Spanning trees and edge filters

[`chain`](@ref) composes relative transforms along the maximum
[`spanning_tree`](@ref), the initialization Theia and OpenCV use;
[`fit_rotation_averaging`](@ref) takes it as `seed = :spanning_tree`.
[`filter_cycles`](@ref) drops edges no consistent loop supports before
averaging, [`filter_by_orientation`](@ref) those that disagree with the averaged
rotations after it.
[`fit_filtered_rotation_averaging`](@ref) runs both around the averaging.

```@docs
spanning_tree
chain
triangles
filter_cycles
rotation_loop_angle
filter_by_orientation
fit_filtered_rotation_averaging
```
