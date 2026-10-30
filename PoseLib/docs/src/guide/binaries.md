# Platforms and binaries

PoseLib.jl calls a shared library, `libposelib_jl`, through `ccall`. The
library links upstream PoseLib, the Eigen linear-algebra headers and a small C
wrapper that this package maintains. It ships prebuilt as the binary package
[`PoseLib_jll`](https://github.com/JuliaBinaryWrappers/PoseLib_jll.jl), built
by [Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil) (recipe
`P/PoseLib`), and `Pkg` downloads the build for your platform when the
package is installed. No compiler is needed.

## Supported platforms

Every platform `PoseLib_jll` is built for: Linux (glibc and musl), macOS,
Windows and FreeBSD, on their common architectures. On a platform without a
build the package still installs and loads — so does every package that
depends on it — but `PoseLib.is_available()` is `false`, and the first call
that needs the library raises an error that points to this page.

## Licenses

PoseLib.jl itself is released under the MIT license. The prebuilt library
also contains upstream PoseLib, released under the BSD-3-Clause license, and
Eigen, released under the Mozilla Public License 2.0. The three license texts
ship inside `PoseLib_jll`'s artifact, under `share/licenses/PoseLib/`.

## Running against your own build

A library built from source can replace the downloaded one through Julia's
standard artifact override; see [Developer notes](../developer.md).
