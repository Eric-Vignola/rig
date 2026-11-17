#!/usr/bin/env bash
# Builds rig.bundle, rig's plug-in for macOS (the rigExpression node rig.fuse() compiles into),
# into <this folder>/<Maya version>/rig.bundle: rig loads it from there when neither RIG_PLUGIN
# nor MAYA_PLUG_IN_PATH gives one (rig._internal.fuse). build.bat is the Windows twin.
#
# Usage: ./build.sh [maya version ...]      e.g. ./build.sh 2025 2026
#   No version: every Maya in /Applications/Autodesk with its headers (include/maya).
#   MAYA=/path/to/maya2025 ./build.sh   builds for that install alone (its version from its
#                                       folder's name, else from include/maya/MTypes.h).
#   MAYA_INSTALLS=<folder>              looks for the installs there instead.
#   ARCH=x86_64                         for an Intel Mac's Maya (default arm64; a list such
#                                       as "arm64 x86_64" makes a universal bundle).
#
# The recipe follows MPyNode's build.sh:
#   * clang++ from Xcode's command line tools, -bundle, against the install's include/ and
#     the libraries in Maya.app/Contents/MacOS;
#   * -ffp-contract=off: no fused multiply-add, so the node rounds like the Maya nodes it
#     replaces (build.bat's /fp:precise);
#   * the bundle is linked in a temporary folder and renamed into place: only rig.bundle
#     lands here, and a Maya that has the old one loaded keeps running it until it loads
#     the new one.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
INSTALLS="${MAYA_INSTALLS:-/Applications/Autodesk}"
ARCH_FLAGS=""
for a in ${ARCH:-arm64}; do ARCH_FLAGS="$ARCH_FLAGS -arch $a"; done
TMP_BASE="${TMPDIR:-/tmp}"
TMP_BASE="${TMP_BASE%/}"

# The Maya version of an install: the year in its folder's name (maya2025), else the first
# four digits of MAYA_API_VERSION in its MTypes.h.
maya_version() {
  local name
  name="$(basename "$1")"
  case "$name" in
    [Mm][Aa][Yy][Aa][0-9][0-9][0-9][0-9]*) echo "${name:4:4}"; return 0 ;;
  esac
  sed -n 's/^#define[[:space:]]\{1,\}MAYA_API_VERSION[[:space:]]\{1,\}\([0-9]\{4\}\).*/\1/p' \
    "$1/include/maya/MTypes.h" 2>/dev/null | head -n 1
}

# The install of a version: maya2025 (Maya's own name for it), else Maya2025.
install_of() {
  if [ -d "$INSTALLS/Maya$1" ] && [ ! -d "$INSTALLS/maya$1" ]; then
    echo "$INSTALLS/Maya$1"
  else
    echo "$INSTALLS/maya$1"
  fi
}

# build <install>: <version>/rig.bundle here. Every step is checked: a function called
# with || runs without set -e.
build() {
  local maya="$1" ver tmp out
  if [ ! -f "$maya/include/maya/MPxNode.h" ]; then
    echo "build.sh: no Maya headers at $maya/include" >&2
    return 1
  fi
  ver="$(maya_version "$maya")"
  if [ -z "$ver" ]; then
    echo "build.sh: cannot tell which Maya version $maya is: no year in its folder name" \
         "and no MAYA_API_VERSION in include/maya/MTypes.h" >&2
    return 1
  fi
  tmp="$(mktemp -d "$TMP_BASE/rig_bundle_build.XXXXXX")" || return 1
  # shellcheck disable=SC2086  # ARCH_FLAGS is a list of words
  if ! clang++ -std=c++17 -O3 -ffp-contract=off $ARCH_FLAGS -bundle \
      -D OSMac_ -D REQUIRE_IOSTREAM -D _BOOL \
      -Wno-unknown-warning-option -Wno-nontrivial-memcall \
      -I"$maya/include" -L"$maya/Maya.app/Contents/MacOS" -lOpenMaya -lFoundation \
      -o "$tmp/rig.bundle" "$HERE/src/rig.cpp"; then
    rm -rf "$tmp"
    echo "build.sh: the build for Maya $ver failed" >&2
    return 1
  fi
  out="$HERE/$ver"
  # a new file renamed over the old one: a Maya running the old one keeps it
  if ! { mkdir -p "$out" && cp "$tmp/rig.bundle" "$out/rig.bundle.new" &&
         mv -f "$out/rig.bundle.new" "$out/rig.bundle"; }; then
    rm -rf "$tmp"
    rm -f "$out/rig.bundle.new"
    echo "build.sh: $out/rig.bundle could not be written" >&2
    return 1
  fi
  rm -rf "$tmp"
  echo "built $out/rig.bundle (Maya $ver, $(lipo -archs "$out/rig.bundle" 2>/dev/null || echo "${ARCH:-arm64}"))"
}

failed=0
if [ -n "${MAYA:-}" ]; then
  build "$MAYA" || failed=1
elif [ "$#" -gt 0 ]; then
  for v in "$@"; do
    build "$(install_of "$v")" || failed=1
  done
else
  any=0
  for d in "$INSTALLS"/[Mm]aya[0-9][0-9][0-9][0-9]; do
    [ -f "$d/include/maya/MPxNode.h" ] || continue
    any=1
    build "$d" || failed=1
  done
  if [ "$any" -eq 0 ]; then
    echo "build.sh: no Maya with its headers in $INSTALLS" >&2
    echo "  pass a version:  ./build.sh 2025" >&2
    echo "  or an install:   MAYA=/path/to/maya2025 ./build.sh" >&2
    exit 1
  fi
fi
exit "$failed"
