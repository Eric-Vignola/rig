# rig.mll

rig's plug-in: the `rigExpression` node that `rig.fuse()` compiles a block's math into (one
node runs a small program of rig's operators and functions, in doubles). `rig.unfuse()` reads
the node's program back into rig's network.

| File | What |
|---|---|
| `src/rig.cpp` | the node: its attributes, the program's format and validation, the instruction loop, the plug-in (`"rig"`, version 1.2) |
| `src/rig_math.h` | the math by hand: matrices, quaternions, slerp, aim, rotate orders (Maya's nodes' rules), and rig.random's hash |
| `build.bat` | builds `<version>/rig.mll` here, one folder per Maya version (ignored by git) |

## Building

`build.bat 2025` builds for Maya 2025, `build.bat 2025 2027` for both; a double-click (no
version) builds for every Maya installed with its headers
(`C:\Program Files\Autodesk\Maya<version>\include`), then waits for a key. It needs MSVC 2022
with the C++ tools. The source is compiled and linked in `%TEMP%` and only `rig.mll` is
copied back, so the folder can be on a streamed drive (Google Drive). A Maya that has the
plug-in loaded keeps the file locked: close it before building again.

## How rig finds it

The first time a session fuses (a `rig.fuse()` block, or `set_options(fuse=True)`), rig
uses the plug-in already loaded, else loads the first of these that loads:

1. `RIG_PLUGIN`, a path to a plug-in file, when set;
2. the name `rig`, which Maya looks for on `MAYA_PLUG_IN_PATH` (a deployed one: a module's
   `plug-ins` folder, say);
3. its build for the running Maya here, `plugin/<version>/rig.mll` (`rig.so` on Linux,
   `rig.bundle` on macOS), when the file is there: a checkout nothing points at needs no
   setup.

One that is not there or does not load (a build for another Maya, say) is skipped. When none
loads, or the one loaded is older than `rig._internal.fuse.MIN_VERSION`, rig warns once per
session and builds its own nodes instead. `cmds.pluginInfo("rig", query=True, path=True)`
says which one is loaded: where `rig.mll` is also on `MAYA_PLUG_IN_PATH`, a build here is
used only through `RIG_PLUGIN`.

`MIN_VERSION` is the lowest version its programs run on (1.2: `Q_ARC`, angleBetween's turn as
Maya's `MQuaternion(a, b)`; `TIME`, the scene's time; `RAND`, rig.random; eulers through
Maya's own decompositions). The program format and every op are described at the top of
`src/rig.cpp`. Each op is checked against the Maya node it stands for by
`rig_sequence_prototype\check_math_nodes.py`, and the whole of fusing by
`_tests/test_rig/test_fuse.py`.
