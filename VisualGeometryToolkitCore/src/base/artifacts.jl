# =============================================================================
# artifacts.jl — the one install of a published Julia artifact: by name from an
# Artifacts.toml, lazily, verified against its pinned sha256 and git-tree-sha1,
# into the shared depot, a private Hugging Face dataset with the user's token.
# Hoisted 2026-11-12 from VisualGeometryToolkitEval's experiments/data.jl (its
# `install_artifact`, which Experiments re-exports) so a package without an
# Eval dependency (BlobBoards' descriptor networks and example data) installs
# through the same code.
# =============================================================================
using Pkg.Artifacts: ensure_artifact_installed, download_artifact
using Downloads: Downloads
using Artifacts: artifact_exists, artifact_path
using Base: SHA1

# A record not yet published or uploaded binds its placeholder, an UPPER_CASE token, where the Zenodo id
# or the Hugging Face revision goes (DataArtifacts).
_published(url) = !occursin(r"/[A-Z][A-Z0-9]*(_[A-Z0-9]+)+/", url)

"""
    install_artifact(name, toml) -> String

Install artifact `name` of `toml` (download, verify sha256 and git-tree-sha1,
unpack into the depot) unless present, and return its tree. A Hugging Face
tarball is downloaded with the user's Hugging Face token when there is one
([`huggingface_token`](@ref)), so a private dataset installs for whoever has
access to it. A placeholder URL (an UPPER_CASE token in place of the Zenodo id or
the Hugging Face revision: the record is not published or uploaded) and a failed
download (an unpublished Zenodo draft, a private dataset without a token) are
refused with the way out.
"""
function install_artifact(name::AbstractString, toml::AbstractString)
    meta = TOML.parsefile(toml)
    haskey(meta, name) || error("unknown artifact '$name' (see $toml)")
    hash = SHA1(meta[name]["git-tree-sha1"])
    artifact_exists(hash) && return artifact_path(hash)
    url = meta[name]["download"][1]["url"]
    _published(url) || error(
        "artifact '$name' is not published yet ($toml carries the placeholder URL $url) and no data " *
        "root is bound: set VGTK_DATA_ROOT or link data/exp_raw at the data tree.")
    hf = startswith(url, "https://huggingface.co/")
    token = hf ? huggingface_token() : nothing
    try
        if token === nothing
            ensure_artifact_installed(name, toml)
        else
            mktempdir() do dir
                tarball = Downloads.download(url, joinpath(dir, "$name.tar.gz"); headers = ["Authorization" => "Bearer $token"])
                r = download_artifact(hash, "file://" * tarball, meta[name]["download"][1]["sha256"])
                r === true || throw(r)            # Pkg returns the verification or unpack error
            end
        end
    catch err
        why = hf && token === nothing ?
            "A private Hugging Face dataset needs your token: log in with `hf auth login` (or set HF_TOKEN) and retry." :
            "If its record is still an UNPUBLISHED Zenodo draft its file URLs are not live yet."
        error("downloading artifact '$name' ($toml) failed. $why Otherwise bind a data root (VGTK_DATA_ROOT or " *
              "data/exp_raw). Underlying error: $(sprint(showerror, err))")
    end
    return artifact_path(hash)
end

"""
    huggingface_token() -> Union{String, Nothing}

The user's Hugging Face token: `ENV["HF_TOKEN"]`, else the file `hf auth login`
writes (`\$HF_HOME/token`, `HF_HOME` defaulting to `~/.cache/huggingface`), else
`nothing`. [`install_artifact`](@ref) sends it to huggingface.co only.
"""
function huggingface_token()
    t = strip(get(ENV, "HF_TOKEN", ""))
    isempty(t) || return String(t)
    f = joinpath(get(ENV, "HF_HOME", joinpath(homedir(), ".cache", "huggingface")), "token")
    isfile(f) || return nothing
    t = strip(read(f, String))
    return isempty(t) ? nothing : String(t)
end
