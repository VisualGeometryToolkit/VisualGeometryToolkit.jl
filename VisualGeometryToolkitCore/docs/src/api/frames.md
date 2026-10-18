# Frames API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Coordinate-frame tags: the open [`Frame`](@ref)`{N}` hierarchy carried by
`EuclideanMap`/`HomographyMap` (`Src`/`Dst`), by `CovMat`/`PrecMat`
(the trailing `F`), and by the framed imaging-chain and batched stages. See the
[Coordinate Frames](../guide/frames.md) guide.

## Frame Tags and Image Frames

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/frames.jl", "base/image_frames.jl"]
```

## Index

```@index
Pages = ["frames.md"]
```
