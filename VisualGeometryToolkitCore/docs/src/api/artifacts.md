# The Artifact Store

Published artifact trees — a paper project's DataArtifacts tiers, a package's
model weights — are installed into the **artifact store**, a directory of
`<git-tree-sha1>/` trees in Pkg's layout on storage that every job of a
machine shares, and never into a cluster's Julia depot (node-local and
evicted, or a home with a quota). [`artifact_store`](@ref) resolves it, the
first rung that names one:

| Rung | Set by |
|---|---|
| `VGTK_ARTIFACT_STORE` | the environment (a reproduction's jobs get it from their site) |
| the site profile `VGTK_SITE` names | its `[storage] artifact_store` (VisualGeometryToolkitEval's site profiles, registered with [`set_site_artifact_store!`](@ref)) |
| the preference `artifact_store` | `Preferences.set_preferences!(VisualGeometryToolkitCore, "artifact_store" => dir)` |
| the depot | Pkg's default; **refused on the cluster** (a job there sets `NODE_CACHE`) |

[`install_artifact`](@ref) installs by name from an Artifacts.toml into the
store, inside [`with_artifact_store`](@ref) (Pkg's `with_artifacts_directory`
under one lock), so Pkg's sha256 and git-tree-sha1 checks and its atomic move
into place hold; a Pidfile lock per tree makes the jobs that start together
download a tree once. A model published as a single file goes in with
[`store_file`](@ref) (verified by [`download_verified`](@ref)). A machine whose
depot already holds trees moves them with [`migrate_artifacts`](@ref), and
[`store_name_clashes`](@ref) lists a name the store holds under two trees.

```julia
import VisualGeometryToolkitCore as VgtkCore
VgtkCore.artifact_store()                       # ArtifactStore(path, source)
tree = VgtkCore.install_artifact("tier_a", "Artifacts.toml")
VgtkCore.migrate_artifacts("Artifacts.toml")   # the depot's copies into the store, hash-checked
```

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["base/artifacts.jl"]
```
