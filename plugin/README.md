# rig.mll

rig's plug-in: the `rigExpression` node that `rig.fuse()` compiles a block's math into (one
node runs a small program of rig's operators and functions, in doubles). `rig.unfuse()` reads
the node's program back into rig's network.

| File | What |
|---|---|
| `rig.cpp` | the node: its attributes, the program's format and validation, the instruction loop, the plug-in (`"rig"`, version 1.2) |
| `rig_math.h` | the math by hand: matrices, quaternions, slerp, aim, rotate orders (Maya's nodes' rules), and rig.random's hash |
| `build.bat` | `build.bat 2025` / `build.bat 2027`: MSVC 2022, built in `%TEMP%`, the `.mll` copied to `build\<version>\rig.mll` (ignored by git) |

rig finds the plug-in already loaded, at `RIG_PLUGIN` (its path), or by name on
`MAYA_PLUG_IN_PATH`. `rig._internal.fuse.MIN_VERSION` is the lowest version its programs run on
(1.2: `Q_ARC`, angleBetween's turn as Maya's `MQuaternion(a, b)`; `TIME`, the scene's time;
`RAND`, rig.random; eulers through Maya's own decompositions). The program format and every op
are described at the top of `rig.cpp`. Each op is checked against the Maya node it stands for
by `rig_sequence_prototype\check_math_nodes.py`, and the whole of fusing by
`_tests/test_rig/test_fuse.py`.
