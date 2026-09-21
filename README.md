# `rig` — a Pythonic node-network DSL for Maya

Rigging in a node-based DCC comes down to three verbs: create nodes, set
attributes, connect them. `rig` turns those verbs into Python operators
on wrapped Maya objects, so a network reads like the maths it computes
and a build script reads like a description of the rig instead of a
transcript of `createNode` / `setAttr` / `connectAttr` calls. It uses
Maya's own nodes (no plug-in, nothing custom saved in the scene), vectorizes 
over lists, and memoizes so the same expression twice costs one network.

![](https://github.com/Eric-Vignola/rig/blob/main/examples/ye_olde_lerp.gif)

---

## Quick taste


Three cubes, a lerp driven by a new attribute, a container around the
network, a component tag and a material.

```python
from maya import cmds
from rig import Node, PlugList, container, Tag
from rig.spec import Float, lock
from rig.bridges import commands as rc
from rig.shade import Blinn, Material

obj1, obj2, obj3 = [rc.polyCube(name=n)[0] for n in ("cube1", "cube2", "cube3")]
obj2.t << [10, 0, 0]                             # setAttr
obj3   << Float("weight", min=0, max=1) << 0.25  # addAttr, then setAttr on the new plug

with container("ye_olde_lerp"):                        # nodes built inside join the container
    obj3.t << (obj2.t - obj1.t) * obj3.weight + obj1.t

print(obj3.t >> None)                                         # [2.5 0.  0. ]
print(cmds.container("ye_olde_lerp", q=True, nodeList=True))  # ['sub1', 'mul1', 'add1']

obj1.f[:3] << Tag("lid")                               # a face component tag on cube1Shape
print(obj1 >> Tag("lid"))                              # [0 1 2]

obj1 << Blinn("red", color=(1, 0, 0))                  # builds red + redSG, assigns cube1
print(Material.of(obj1))                               # [Blinn('red')]
```


---

## The language in one table

Two operators carry the language:
- `<<` points the way data flows in. (aka: injection)
- `>>` points the way it flows out. (aka: introspection)

| Spelling | Meaning | Returns |
|---|---|---|
| `a << b` | **inject**: `b` flows into `a`. A value is `setAttr`, a plug is `connectAttr`, `None` disconnects | `a`, so it chains |
| `a >> None` | **introspect**: read the value (`getAttr`) | a float, a NumPy array, a string... |
| `a >> node` | clone `a`'s attribute definition onto `node` | the new plug |
| `node << Float("x")` | add an attribute: any `rig.spec` type, then modifiers such as `<< lock` / `<< hide` | the new plug, so its value goes next |
| `node >> Float("x")` | add an output-only (non-writable) attribute | the new plug |
| `node.tx = 5` | sugar for `node.tx << 5` | |
| `a + b`, `a - b`, `a * b`, `a / b`, `a ** b`, `a // b`, `a % b` | arithmetic builds nodes. Matrices and quaternions are detected and routed to `multMatrix`, `quatProd`...; `[x, y, z] * m` is point-by-matrix | the output plug |
| `-a`, `~a` | negate; logical NOT | the output plug |
| `a == b`, `!=`, `<`, `<=`, `>`, `>=` | comparisons build condition nodes | the output plug, **never a bool** |
| `a & b`, `a \| b`, `a ^ b` | logical AND / OR / XOR networks | the output plug |


---

## Conventions

- **Injection is right-to-left and returns the left-hand side.**
  `obj.t << [1, 2, 3] << lock` reads "t receives 1,2,3, then a lock". The
  one exception is an attribute spec: `node << Float("w")` returns the
  **new plug**, because the next thing you inject is its value.
  Collection specs return the LHS again, so `cube.f[:3] << Tag("a") << Tag("b")`.
- **`PlugList` vectorizes, under two broadcast rules.** Attribute access
  maps over the list (`cubes.t` is a `PlugList` of plugs). The
  **operators** (`<<`, `+`, `==`...) pair elements and cap the shorter
  operand to its last element: `cubes.ty << [1, 2]` sets 1, 2, 2. The
  **function libraries** (`rig.functions`, `lerp`, `dist`...) are
  NumPy-strict: every list must be the same length, or length 1, or a
  scalar, and a mismatch is a `ValueError`.
- **Components live on the shape.** `cube.f` and `cube.e` are
  `Components` (faces and edges have no plug to wrap); `cube.vtx` and
  `cube.uv` are plugs (`controlPoints`, `uvpt`), as is `surface.cv`, and
  `cube.vtx[3]` is one element plug. A transform with one shape resolves
  to it; two shapes are an `AttributeError` naming both. A real attribute
  always wins: `curve.f` is `curveShape.form`.
- **Nodes join the active container.** Inside `with container("x"):`
  every node the DSL creates is added to `x`; nested blocks flatten into
  the outermost, prefixing their nodes with the inner block's name
  (`inner_add1`); `preserve=True` makes a real sub-container instead.
  Only nodes a call *creates* join: a query, a `parent` or a `rename`
  never moves a node in. Geometry, display layers and materials never
  join (`container=True` on a material spec opts a per-asset look in);
  `container=False` opts any `rc` / `rn` call out.
- **Memoization.** A function or operator called twice with the same
  plugs and the same literals returns the same output plug; the cache
  forgets nodes that were deleted. All-literal math never touches the
  scene (`functions.abs(-5)` is `5`); `with force_nodes():` builds it anyway.
- **Maya-version dispatch.** Each operation is registered per Maya
  release; the highest implementation not newer than the target runs.
  The target is the live Maya, or `set_options(maya_version=2023)` to
  build networks an older release can open.
- **Comparisons are nodes.** `a == b` builds a node and returns its
  output plug; `Plug.__hash__` is overridden so plugs still work as dict
  keys and set members.
- **Sibling fallback.** `plug.foo` looks for a child attribute first,
  then a sibling on the same node, so `(a.tx + 5).operation` reaches the
  math node behind the output.

```python
from rig import set_options, lerp, functions as f

cmds.file(new=True, force=True)
cubes = PlugList([rc.polyCube(name=n)[0] for n in ("a", "b", "c")])
cubes.ty << [1, 2, 3]  # one value per element
cubes.tz << 7          # a scalar broadcasts
print(cubes.ty >> None, cubes.tz >> None)            # [1. 2. 3.] [7. 7. 7.]
cubes.ty << [1, 2]                                   # an operator caps: the last element repeats
print(cubes.ty >> None)                              # [1. 2. 2.]

a, b, c = cubes
try:
    lerp(cubes.tx, PlugList([a.ty, b.ty]), 0.5)      # a function is strict: 3 against 2
except ValueError:
    print("strict")                                  # strict

print(str(a.tx + b.tx) == str(a.tx + b.tx))  # True   memoized: one sum node
print(f.abs(-5), f.abs(a.tx))                # 5 abs1.output   literals fold, plugs build

set_options(maya_version=2023)
print(a.ty + b.ty)                           # add2.output1D   a plusMinusAverage
set_options(maya_version=None)               # back to the live Maya
```

---

## What's inside

`rig/__init__.py` re-exports the core types, the `rig.spec` names and
the cross-type math verbs, so most scripts import from `rig` directly.

```
rig/
├── __init__.py          Node, Plug, PlugList, Container, container, set_options / get_options,
│                        force_nodes, cleanup, memoize / vectorize, lift, condition, constant
├── spec/                attribute specs: Float Int Bool Angle Time, Vector Color Euler Quat, Enum,
│                        Matrix Mesh Message NurbsCurve NurbsSurface String; modifiers lock unlock
│                        hide unhide skip destroy Note
├── bridges/
│   ├── commands.py      maya.cmds returning Node / PlugList instead of strings   (rc)
│   └── nodes.py         one factory per Maya node type, kwargs are injections     (rn)
│
├── functions.py         abs clamp min max sum avg floor ceil round choice searchsorted ... on plugs
├── trigonometry.py      sin cos tan asin acos atan atan2, radians and degrees variants
├── matrix.py            decompose compose aim multiply determinant translation rotation ...
├── vector.py            dot cross length triple_product rotate, the X Y Z axes
├── quaternion.py        add multiply conjugate from_axis_angle to_axis_angle ...
├── euler.py             reorder
├── _dispatch.py         dist lerp slerp blend elerp normalize inverse angle to_euler to_matrix ...
│                        the cross-type verbs, re-exported flat from rig
├── interpolate.py       sequence smoothstep smootherstep inverse_lerp
├── tween.py             Penner easing curves: in_quad, out_bounce, in_out_elastic ...
├── random.py            LCG networks that live inside the DG
│
├── membership.py        Tag (component tags), Layer (display layers), Components
├── shade.py             Blinn Lambert Phong PhongE SurfaceShader StandardSurface OpenPBRSurface,
│                        Material, Default; convert / repair / tidy / materials / bindings
│
├── maya/                the object-model layer under the DSL: nodetypes/ (typed PyNode wrappers),
│                        attribute.py, node_name.py, constants.py, pycmds.py, plugins/
├── examples/            rail_spine.py, rail_spine_simple.py, image_loop.py,
│                        perspective_image_planes.py, ye_olde_lerp.gif
├── utils.py             run_tests()
├── _defaults.toml       team-wide container option defaults
├── _internal/           private: node.py plug.py list.py container.py members.py memoize.py
│                        node_ops.py shorthand.py maya_version.py ...
└── _tests/              the suite, run with rig.utils.run_tests()
```

The names most scripts start from:

```python
from rig import Node, Plug, PlugList, Container, container, Components, Tag, Layer
from rig import Float, Vector, Enum, lock, hide                      # rig.spec, re-exported
from rig import dist, lerp, slerp, blend, normalize, to_euler, to_matrix
from rig import functions, trigonometry, matrix, vector, quaternion, euler, interpolate, tween
from rig.shade import Blinn, Lambert, Phong, Material, Default
from rig.bridges import commands as rc
from rig.bridges import nodes as rn
```

`rig.functions` shadows Python builtins (`abs`, `min`, `max`, `sum`,
`int`, `round`...): import the module, never `from rig.functions import *`.
Put the **parent** of `rig/` on `sys.path`, not `rig/` itself, or
`rig.maya` and `rig.random` shadow Autodesk's `maya` and the stdlib.

---

## Where to go next

Every subpackage has a **README** (concepts, conventions, when to reach
for it) and a **CHEATSHEET** (every public name, with a runnable example).

| Page | What it holds | |
|---|---|---|
| **this page** | the operator table, conventions, the map | [CHEATSHEET](https://github.com/Eric-Vignola/rig/blob/main/CHEATSHEET.md) — every operator and top-level name, runnable |
| `rig.spec` | attribute specs (`Float`, `Vector`, `Enum`...) and the modifiers (`lock`, `hide`, `destroy`...) | [README](https://github.com/Eric-Vignola/rig/blob/main/spec/README.md) · [CHEATSHEET](https://github.com/Eric-Vignola/rig/blob/main/spec/CHEATSHEET.md) |
| `rig.bridges` | `commands` (`maya.cmds` returning nodes) and `nodes` (a factory per node type) | [README](https://github.com/Eric-Vignola/rig/blob/main/bridges/README.md) · [CHEATSHEET](https://github.com/Eric-Vignola/rig/blob/main/bridges/CHEATSHEET.md) |
| `rig.maya` | the typed node layer the DSL stands on: `PyNode`, `Attribute`, node names, constants | [README](https://github.com/Eric-Vignola/rig/blob/main/maya/README.md) · [CHEATSHEET](https://github.com/Eric-Vignola/rig/blob/main/maya/CHEATSHEET.md) |
| `examples/` | complete builds: a rail spine, an image loop, perspective image planes | [README](https://github.com/Eric-Vignola/rig/blob/main/examples/README.md) |

---

## Real behaviour, verified

Not bugs to work around blindly; things a rigger meets in the first hour.

- **`cube.f` is not a plug.** Faces and edges have no attribute behind
  them, so `cube.f` / `cube.e` are `Components`; `cube.vtx` is
  `Plug("cubeShape.controlPoints")` and `cube.uv` is `uvpt`. `cube.cv` on
  a mesh is an `AttributeError` (CVs belong to NURBS).
- **A comparison is always truthy.** `cube.tx == 3` returns the output
  plug of an `equal` node, and a `Plug` is a truthy object, so
  `if cube.tx == 3:` always enters the branch and leaves a node behind.
  Read values first: `(cube.tx >> None) == 3`. The same holds for
  `a == b` between two results: compare `str(a) == str(b)`.
- **`polyCube`'s six face tags are procedural.** With construction
  history on, `top`, `bottom`, `front`, `back`, `left`, `right` are owned
  by `polyCube1`: `cube.f[:3] << Tag("top")` is a `TypeError` naming the
  owner, and `Tag("top", at=cube)` shadows it with an editable tag on the
  shape. `polyCube(ch=False)` bakes them into the shape instead.
- **One tag, one category.** A tag holds vertices, edges or faces, never
  a mix: faces into a vertex tag is a `TypeError` with nothing written;
  `Tag("x").set(cube.f[:2])` replaces and may flip the category. A tag
  name needs at least two characters (Maya stores a one-character name
  as an empty string).
- **A missing collection is a `ValueError`, never an empty answer.**
  `cube >> Tag("nope")` raises; a tag that exists but holds none of the
  LHS answers with an empty array. Same for materials and layers.
- **Materials are exclusive, and removal leaves faces green.**
  `cube.f[:3] << Blinn("decal")` carves those faces out of `redSG`;
  `cube.f[:3] << -Blinn("decal")` puts them in **no** engine (Maya draws
  them green). `cube << Default()` is the way back to
  `initialShadingGroup`; `shade.repair()` re-homes every green shape.
- **Conversion parks, Maya's Type dropdown deletes.** `Phong(red)`
  retypes `red` in place and keeps the name, the engine, the container
  and the locks; a value or an incoming wire the new type cannot hold is
  parked on the node as a hidden `__attr__` and comes back the next time
  the material becomes a type that has it. Maya's own Attribute Editor
  dropdown makes a brand-new node and cascade-deletes any texture or
  animCurve whose only link was the shader; it also wipes parked state.
- **Unit conversions stay outside containers.** `xf.rx << cube.tx`
  inserts Maya's `unitConversion` (linear to angle) and leaves it outside
  the active container by default (`absorb_unit_conversions=False`),
  because the Node Editor's "Hide Unit Conversion Nodes" view would
  otherwise draw the container as leaking.
- **`//` is floor division**, not PyMEL's disconnect. `plug << None`
  disconnects.

```python
cmds.file(new=True, force=True)
cube = rc.polyCube(name="pCube1")[0]
print(type(cube.f).__name__, cube.vtx, cube.uv)  # Components pCube1Shape.controlPoints pCube1Shape.uvpt
print(cube >> Tag())                             # [Tag('back'), Tag('bottom'), Tag('front'), Tag('left'), Tag('right'), Tag('top')]
try:
    cube.f[:3] << Tag("top")
except TypeError as e:
    print(str(e)[:34])                               # 'top' on pCube1Shape is PROCEDURAL
cube.f[:3] << Tag("top", at=cube)                    # an editable shadow on the shape
print(cube >> Tag("top"))                            # [0 1 2]

cond = cube.tx == 3
print(cond, bool(cond))                              # equal1.output True

xf = rc.createNode("transform", name="xf")
with container("c1"):
    xf.rx << cube.tx
print(cmds.ls(type="unitConversion"), cmds.container("c1", q=True, nodeList=True))   # ['unitConversion1'] None
```

---

## Requirements

Autodesk Maya (the suite runs on 2025; version-keyed dispatch builds the
legacy networks on older releases), with numpy and scipy installed for
mayapy.

## Author

* **Eric Vignola** (eric.vignola@gmail.com)

## License

BSD 3-Clause License:
Copyright (c)  2026, Eric Vignola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:


1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
