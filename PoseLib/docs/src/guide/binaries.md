# Platforms and binaries

PoseLib.jl calls a shared library, `libposelib_jl`, through `ccall`. The
library links upstream PoseLib, the Eigen linear-algebra headers and a small C
wrapper that this package maintains. It ships prebuilt as the package's own
artifact, named `PoseLib` in its `Artifacts.toml`, and `Pkg` downloads the
tarball for your platform when the package is installed. No compiler is
needed.

## Supported platforms

For now the artifact covers Linux only:

- x86_64, aarch64, i686, armv6l and armv7l, with glibc or musl;
- powerpc64le and riscv64, with glibc.

On macOS and Windows the package installs but refuses to load, with an error
that points to this page.

The binaries are built with
[BinaryBuilder](https://github.com/JuliaPackaging/BinaryBuilder.jl), the
tool behind Julia's binary packages. The same build recipe is being submitted
to [Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil), the community
build tree, as the binary package `PoseLib_jll`, which Yggdrasil builds for
every platform. Once `PoseLib_jll` is registered, a patch release of
PoseLib.jl will depend on it instead of its own artifact. That brings macOS,
Windows and FreeBSD, with no change to the API.

## Licenses

PoseLib.jl itself is released under the MIT license. The prebuilt library
also contains upstream PoseLib, released under the BSD-3-Clause license, and
Eigen, released under the Mozilla Public License 2.0. The three license texts
ship inside the artifact, under `share/licenses/PoseLib/`.

## Running against your own build

A library built from source can replace the downloaded one through Julia's
standard artifact override; see [Developer notes](../developer.md).
