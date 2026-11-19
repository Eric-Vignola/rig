# rig's plug-ins

rig's two Maya plug-ins, and the code that loads them.

| File | What |
|---|---|
| `__init__.py` | `load_plugin()`, a context manager (a plug-in shipped here loads by full path, any other by name), `bundled_plugin_path()`, and the route of rig's API edits onto the undo queue |
| `undoable_api_command.py` | the undo plug-in: `rigUndoableAPICommand`, which puts API edits on Maya's undo queue |
| `src/rig.cpp` | rig.mll's node, `rigExpression`: its attributes, the program's format and validation, the instruction loop, the plug-in (`"rig"`, version 1.2) |
| `src/rig_math.h` | rig.mll's math by hand: matrices, quaternions, slerp, aim, rotate orders (Maya's nodes' rules), and rig.random's hash |
| `build.bat` | Windows: builds `<version>/rig.mll` here, one folder per Maya version |
| `build.sh` | macOS: builds `<version>/rig.bundle` here, one folder per Maya version |

The `<version>` folders are builds, ignored by git.

## The undo plug-in

`undoable_api_command.py` is Python: there is nothing to build. rig loads it at the first
API edit of a session, by full path from this folder (by name from `MAYA_PLUG_IN_PATH` if the
file is not here). With this folder on `MAYA_PLUG_IN_PATH` (a module file's
`MAYA_PLUG_IN_PATH+:=` line), that first load costs about 8 ms instead of about 165 ms. What
goes through it, and the rules of an edit of your own, are in
[nodetypes' CHEATSHEET section 21](../nodetypes/CHEATSHEET.md#21-plugins--load_plugin-and-undo)
and [the root CHEATSHEET's section 38](../CHEATSHEET.md#38-undo).

## rig.mll

The `rigExpression` node that `rig.fuse()` compiles a block's math into: one node runs a
small program of rig's operators and functions, in doubles. `rig.unfuse()` reads the node's
program back into rig's network.

### Building

**Windows:** `build.bat 2025` builds for Maya 2025, `build.bat 2025 2027` for both; a
double-click (no version) builds for every Maya installed with its headers
(`C:\Program Files\Autodesk\Maya<version>\include`), then waits for a key. It needs MSVC 2022
with the C++ tools. The source is compiled and linked in `%TEMP%` and only `rig.mll` is
copied back, so the folder can be on a streamed drive (Google Drive). A Maya that has the
plug-in loaded keeps the file locked: close it before building again.

**macOS:** `./build.sh 2025` builds `2025/rig.bundle`, `./build.sh 2025 2026` both; with no
version it builds for every Maya in `/Applications/Autodesk` with its headers
(`include/maya`). It needs clang++ from Xcode's command line tools (`xcode-select
--install`), and builds for Apple silicon (`ARCH=x86_64 ./build.sh` for an Intel Mac's Maya).
`MAYA=/path/to/maya2025 ./build.sh` builds for an install elsewhere. The bundle is linked in
a temporary folder and renamed into place: a Maya that has the old one loaded keeps running
it until it loads the new one.

Both compile with no fused multiply-add (`/fp:precise`, `-ffp-contract=off`), so the node
rounds like the Maya nodes it replaces. The matrix product is SSE2 on x64, bit for bit
`MMatrix`'s; elsewhere (Apple silicon) it is the same products added in the same order, in
plain doubles.

### How rig finds it

The first time a session fuses (a `rig.fuse()` block, or `set_options(fuse=True)`), rig
uses the plug-in already loaded, else loads the first of these that loads:

1. `RIG_PLUGIN`, a path to a plug-in file, when set;
2. the name `rig`, which Maya looks for on `MAYA_PLUG_IN_PATH` (a deployed one: a module's
   `plug-ins` folder, say);
3. its build for the running Maya here, `plugins/<version>/rig.mll` (`rig.bundle` on macOS,
   `rig.so` on Linux), when the file is there: a checkout nothing points at needs no setup.

One that is not there or does not load (a build for another Maya, say) is skipped. When none
loads, or the one loaded is older than `rig._internal.fuse.MIN_VERSION`, rig warns once per
session and builds its own nodes instead. `cmds.pluginInfo("rig", query=True, path=True)`
says which one is loaded: where `rig.mll` is also on `MAYA_PLUG_IN_PATH`, a build here is
used only through `RIG_PLUGIN`.

`MIN_VERSION` is the lowest version its programs run on and give the documented results
with (1.2: `Q_ARC`, angleBetween's turn as Maya's `MQuaternion(a, b)`; `TIME`, the scene's
time; `RAND`, rig.random; eulers through Maya's own decompositions; 1.3: the three guards
on numpy: `asin` / `acos` clamp their input, `normalize` of a zero vector is the zero vector,
no output is nan or inf; and the outputs' raw twins, `scalarRaw[]` ..., which fused nodes read
each other through, so that the third guard alters nothing between them). The program format
and every op are described at the top of
`src/rig.cpp`. Each op is checked against the Maya node it stands for by
`rig_sequence_prototype\check_math_nodes.py`, and the whole of fusing by
`_tests/test_rig/test_fuse.py`.
