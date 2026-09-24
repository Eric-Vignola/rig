# `rig` Cheatsheet

Every public name of the language, one small runnable block each: the
core (`Node`, `Plug`, `PlugList`, the operators, containers), the
function libraries, and the membership grammar (tags, layers,
materials). The blocks run top to bottom as one script and share a
namespace, notebook style; every section that needs a clean scene
starts with one.

Concepts and conventions live in [`README.md`](README.md). The
subpackages have their own cheatsheets: [`spec/`](spec/CHEATSHEET.md)
for attribute specs and modifiers, [`bridges/`](bridges/CHEATSHEET.md)
for `maya.cmds` and the node factories, [`nodetypes/`](nodetypes/CHEATSHEET.md)
for the typed node layer underneath.

---

## Contents

| # | Section | Covers |
|---|---|---|
| — | [Setup](#setup) | `mayapy` bootstrap, an empty scene |
| 1 | [Node](#1-node) | create, wrap, `str` / `repr`, hash and equality, `Node(plug)`, `>> None`, `Node.wrap`, `lift` |
| 2 | [Plug — read, set, connect](#2-plug--read-set-connect) | attribute access, sibling fallback, `>> None`, `<< value`, `<< plug`, `<< None`, chaining, `node.tx = 5` |
| 3 | [Plug — compounds, multis, aliases](#3-plug--compounds-multis-aliases) | `[a, b, c]` fan-out, `skip` / `lock` / `None` slots, `[:]` slicing, multi attrs, blendShape targets |
| 4 | [Plug — connections, hashing, equality](#4-plug--connections-hashing-equality) | `get_inputs` / `get_outputs`, `==` builds a node, `equals`, dict and set keys |
| 5 | [Plug — `>>` clones and publishes](#5-plug---clones-and-publishes) | `plug >> Node`, `plug >> "newName"`, `plug >> container` |
| 6 | [PlugList](#6-pluglist) | construction, broadcast, asymmetric lists, slicing, fancy indexing, `>> None` |
| 7 | [Arithmetic](#7-arithmetic) | `+ - * / ** // %`, reflected forms, `-x`, what each builds |
| 8 | [Matrices and quaternions](#8-matrices-and-quaternions) | `wm * wim`, point-matrix, `**`, quaternion routing, auto-decompose, `node << matrix` |
| 9 | [Logic, comparisons, `condition`, `constant`](#9-logic-comparisons-condition-constant) | `& \| ^ ~`, `== != < <= > >=`, per-channel fan-out, `condition(...)`, `constant(...)` |
| 10 | [Components](#10-components) | `vtx` / `cv` / `pt` handles, `f` / `e`, `Components(...)`, `.indices` / `.count` / `.names`, `>> None` |
| 11 | [Containers](#11-containers) | `with container(...)`, nesting and flattening, `container.add`, `container=False`, publishing, `Container` |
| 12 | [Options, `force_nodes`, `cleanup`](#12-options-force_nodes-cleanup) | `set_options` / `get_options` / `ContainerOptions`, constant folding, garbage collection |
| 13 | [`memoize`, `vectorize`, `prune_memoize_caches`](#13-memoize-vectorize-prune_memoize_caches) | the two decorators and the cache sweep |
| 14 | [`sequences` and `arguments`](#14-sequences-and-arguments) | the asymmetric generators behind every broadcast |
| 15 | [`NodeOp` and the target Maya version](#15-nodeop-and-the-target-maya-version) | version-keyed impls, `set_options(maya_version=...)`, `set_target_version` |
| 16 | [`InjectionError`](#16-injectionerror) | the one thing `<<` refuses |
| 17 | [Function libraries: the map](#17-function-libraries-the-map) | the nine modules and twelve verbs, folding, memoisation, `PlugList` broadcast, the five rules |
| 18 | [`functions` — scalar math](#18-functions--scalar-math) | `abs` `int` `trunc` `floor` `ceil` `sign` `round` `clamp` `sqrt` `pow` `exp` `log` `rev` |
| 19 | [`functions` — lists: reduce, pick, search](#19-functions--lists-reduce-pick-search) | `sum` `avg` `max` `min` `argmax` `argmin` `all` `any` `diff` `cumsum` `choice` `searchsorted` |
| 20 | [`functions` — time, constants, comparison and logic](#20-functions--time-constants-comparison-and-logic) | `frame` `pi` `inf`, `equal` and the comparison wrappers, `logical_and` / `or` / `xor` / `not` |
| 21 | [`trigonometry`](#21-trigonometry) | `sind` .. `atan2d` in degrees, `sin` .. `atan2` in radians, `degrees` / `radians` |
| 22 | [`matrix` — read a matrix](#22-matrix--read-a-matrix) | `decompose`, `to_euler` / `to_quaternion`, `translation` / `rotation` / `scale_of`, `axis` / `row` / `column`, `determinant`, `dist` |
| 23 | [`matrix` — build and transform](#23-matrix--build-and-transform) | `compose`, `fourbyfour`, `aim`, `multiply` / `add` / `inverse` / `transpose` / `normalize`, `transform_point` / `transform_vector` |
| 24 | [`matrix` — blend](#24-matrix--blend) | `lerp` vs `blend` vs `slerp` vs `pow` |
| 25 | [`vector`](#25-vector) | `X` `Y` `Z`, `dot` / `cross` / `triple_product`, `length` / `normalize` / `dist`, `angle`, `rotate`, `lerp` / `slerp` / `elerp` |
| 26 | [`quaternion`](#26-quaternion) | Hamilton arithmetic, `angle`, conversions, `slerp` / `pow`, axis-angle both ways |
| 27 | [`euler`](#27-euler) | `reorder`, `to_matrix` / `to_quaternion`, `slerp`, the rotate-order table |
| 28 | [`interpolate`](#28-interpolate) | `sequence` and its `method=`, `smoothstep` / `smootherstep`, `inverse_lerp` |
| 29 | [`tween`](#29-tween) | the 42 easing curves by family, extrapolation |
| 30 | [`random`](#30-random) | `value` / `uniform` / `randint` and the `3D` variants, `seed`, `trigger` |
| 31 | [The cross-type verbs at `rig.*`](#31-the-cross-type-verbs-at-rig) | `dist` `lerp` `slerp` `blend` `elerp` `normalize` `inverse` `angle` `to_euler` `to_quaternion` `to_matrix`, the routing table |
| 32 | [The membership grammar](#32-the-membership-grammar) | `<< Spec('x')` / `<< -Spec('x')` / `<< Spec()` / `>> Spec('x')` / `>> Spec()` for `Tag`, materials and `Layer`; the return-value rule; the rejected spellings |
| 33 | [Components as members](#33-components-as-members) | the bare handle is the whole kind, `sph.vtx` is all points, no `__len__`, `PlugList` pairing |
| 34 | [Tag](#34-tag) | one category per tag, `.set` / `.clear` / `.rename` / `.delete`, queries as arrays, `componentTagExpression`, procedural `polyCube` tags and `at=` |
| 35 | [Layer](#35-layer) | exclusive display layers, `Layer.of`, kwargs and `update=`, `defaultLayer` as no layer |
| 36 | [Materials](#36-materials) | `Blinn` / `Lambert` / `Phong` / ... / `Default`, kwargs, `unique=`, per-face carving, green faces, `shade.repair` / `tidy` |
| 37 | [Shader conversion](#37-shader-conversion) | `Phong(mat)`, `.astype`, `shade.convert` and the `Conversion` report, parked attributes, `strict=` / `park=`, undo |

---

## Setup

Runs under `mayapy` from a terminal and, harmlessly, inside a Maya session.
Nothing ships as a scene file: every block builds what it needs.

```python
from maya import standalone
try:
    standalone.initialize()          # running from mayapy; inside Maya this raises and is skipped
except Exception:
    pass
from maya import cmds
cmds.file(new=True, force=True)

import numpy as np

from rig import Node, Plug, PlugList
print(cmds.about(version=True))      # the Maya this page was vetted on
```

The Maya version matters: on 2024+ the arithmetic operators build the native
`sum` / `multiply` / `equal` / `and` nodes; on older releases, or when you
target one (section 15), they build the classic `plusMinusAverage` /
`multiplyDivide` / `condition` networks. The result plugs below are the
2024+ spellings.

---

## 1. Node

A `Node` wraps any Maya node. Attribute access returns a `Plug`; method
access delegates to the typed node underneath.

```python
a = Node.create("transform", name="a")  # createNode, registered with the active container scope
b = Node("a")                           # wrap by name
print(str(a), repr(a))                          # a Node("a")
print(a == b, hash(a) == hash(b), len({a, b}))  # True True 1  -- identity is the MObject, not the string
print(cmds.objExists(a), cmds.nodeType(a))      # True transform  -- a Node passes straight into cmds
print(a.list_attr()[:2])                        # [Attribute("a.message"), Attribute("a.caching")]  -- a DGNode method, delegated
```

`Node(plug)` and `Node("node.attr")` strip the attribute and give the owning
node; `node >> None` leaves the DSL and hands back the typed node
(`Transform`, `Mesh`, ...).

```python
print(repr(Node(a.tx)), repr(Node("a.translate")))   # Node("a") Node("a")
typed = a >> None
print(type(typed).__name__, isinstance(typed, Node))  # Transform False
```

`Node.wrap` turns a `maya.cmds` result back into the DSL and `lift` casts a
string, `Attribute` or `DGNode` to the right wrapper.

```python
from rig import lift

print(repr(Node.wrap(cmds.createNode("transform", name="c"))))   # Node("c")
cmds.select("a", "c")
print(repr(Node.wrap(cmds.ls(selection=True))))                  # PlugList([Node("a"), Node("c")])
print(Node.wrap(5.0), Node.wrap(None), Node.wrap("not a node"))  # 5.0 None not a node  -- passthrough
print(repr(lift("a")), repr(lift("a.tx")), lift(a) is a)         # Node("a") Plug("a.translateX") True
```

`node << spec` adds an attribute and `node >> spec` adds an output-only
one (`writable=False`); `node << matrix` decomposes onto the transform
(section 8); a collection spec (`Tag`, a material, `Layer`) is membership
(section 32). A bare value on a bare `Node` is a `TypeError`.

```python
from rig.spec import Float

print(repr(a << Float("blend")))                               # Plug("a.blend")
print(repr(a >> Float("result")))                              # Plug("a.result")
print(cmds.attributeQuery("result", node="a", writable=True))  # False
try:
    a << 5
except TypeError as err:
    print(type(err).__name__)                             # TypeError
```

---

## 2. Plug — read, set, connect

`<<` is *inject*: the left side receives from the right. It sets a value,
connects a plug, or disconnects on `None`, and always returns the left
side so it chains. `>> None` reads.

```python
cmds.file(new=True, force=True)
a = Node.create("transform", name="a")
b = Node.create("transform", name="b")

a.tx << 5                                  # setAttr
print(a.tx >> None)                        # 5.0
b.tx << a.tx                               # connectAttr
print(cmds.listConnections("b.tx", source=True, plugs=True))   # ['a.translateX']
b.tx << None                               # disconnect
print(cmds.listConnections("b.tx", source=True))               # None
```

Short and long names both work, and `node.tx = 5` is sugar for `node.tx << 5`.

```python
a.translateY = 2
print(str(a.ty), a.ty >> None, str(a.translate))   # a.translateY 2.0 a.translate
```

`>> None` is numpy-aware: scalars stay Python numbers, compounds come back
as arrays, matrices as `(4, 4)`, strings as `str`. `plug.get()` is the same
read as a method.

```python
from rig.spec import String

a.t << [1, 2, 3]
a   << String("label") << "hero"
print(a.t >> None, type(a.t >> None).__name__)  # [1. 2. 3.] ndarray
print((a.matrix >> None).shape)                 # (4, 4)
print(a.label >> None, a.tx.get())              # hero 1.0
```

Chaining: an attribute spec returns the new plug, so the value and the
modifier go next.

```python
from rig.spec import Float, lock

a << Float("weight", min=0, max=1) << 0.5 << lock
print(a.weight >> None, cmds.getAttr("a.weight", lock=True))   # 0.5 True
```

A plug looks up a child attribute first, then falls back to a *sibling* on
the same node, so a result plug still reaches its node's other attributes.

```python
total = a.tx + b.tx                          # a result plug on a sum node
print(str(total), str(total.input))  # add1.output add1.input
print(repr(a.matrix.ro))             # Plug("a.rotateOrder")  -- sibling of a typed-atomic plug
total.node.input[1] << 10                    # .node is the owning Node
print(total >> None)                         # 11.0
```

Plug introspection you will reach for:

```python
print(a.tx.alias, a.tx.data_type, a.t.num_children)  # translateX doubleLinear 3
print(a.wm.is_multi, a.wm.data_type)                 # True matrix
print(a.tx.full_name, repr(a.tx.node))               # a.translateX Node("a")
```

---

## 3. Plug — compounds, multis, aliases

`plug << [a, b, c]` fans one slot per channel. Each slot speaks the whole
vocabulary: a number sets, a plug connects, `None` disconnects, a modifier
applies, `skip` leaves the channel alone.

```python
from rig.spec import lock, skip, unlock

src = Node.create("transform", name="src")
dst = Node.create("transform", name="dst")
src.t << [11, 22, 33]

dst.t << [src.tx, 0, src.tz]                 # connect, set, connect
print(dst.t >> None)                         # [11.  0. 33.]
dst.t << [None, 4.0, None]                   # disconnect x and z, set y
print(dst.t >> None)                         # [11.  4. 33.]
dst.t << [lock, skip, lock]
print([cmds.getAttr(f"dst.t{ax}", lock=True) for ax in "xyz"])   # [True, False, True]
dst.t << [unlock, skip, unlock]
dst.t << 7                                   # a scalar broadcasts across the compound
print(dst.t >> None)                         # [7. 7. 7.]
```

A compound indexes and slices like a sequence; the slice is a `PlugList`.

```python
print(str(dst.t[0]), str(dst.t[-1]))  # dst.translateX dst.translateZ
print(repr(dst.t[::-1]))              # PlugList([Plug("dst.translateZ"), Plug("dst.translateY"), Plug("dst.translateX")])
dst.t[:] << [1, 2, 3]
print(dst.t >> None)                         # [1. 2. 3.]
```

Shape mismatches are refused, numpy style: the source must match the
channel count or be a single value.

```python
try:
    dst.t << [1, 2]
except ValueError as err:
    print(str(err)[:56])                     # Cannot inject sequence of size 2 into 3-channel compound
```

Multi attributes: a bare multi `<< scalar` appends at the next free index,
`<< sequence` writes index by index, `[:]` slices the populated elements,
and a bounded slice or `[i]` addresses indices that are created on write.

```python
from rig.spec import Float, Vector

net = Node.create("network", name="net")
net   << Float("w", multi=True)
net.w << 1.0
net.w << 2.0                                 # appended at [1]
print(net.w.next_index, repr(net.w[:]))      # 2 PlugList([Plug("net.w[0]"), Plug("net.w[1]")])
net.w[:4] << [10, 20, 30, 40]                # indices 0..3, the missing two created on write
print(net.w[:] >> None)                      # [10. 20. 30. 40.]
net.w << [100, 200]                          # the bare multi with a sequence: index by index from 0
print(net.w[:] >> None)                                # [100. 200.  30.  40.]
print(str(net.w.append(50.0)), net.w[[1, 4]] >> None)  # net.w[4] [200.  50.]  -- fancy indexing on a multi

net         << Vector("offsets", multi=True)
net.offsets << [[1, 2, 3], [4, 5, 6]]
print(net.offsets[:] >> None)                # [[1. 2. 3.] [4. 5. 6.]]
```

A blendShape aliases every `weight[i]` to its target name, and the DSL
resolves the alias.

```python
base    = cmds.polyCube(name="base", constructionHistory=False)[0]
targets = [cmds.polyCube(name=f"t{i}", constructionHistory=False)[0] for i in range(4)]
bs      = Node(cmds.blendShape(*targets, base, name="bs1")[0])

bs.weight[:4] << [0.1, 0.2, 0.3, 0.4]
print(np.round(bs.weight[:] >> None, 2))    # [0.1 0.2 0.3 0.4]
print(str(bs.t2), round(bs.t2 >> None, 2))  # bs1.t2 0.3  -- alias of weight[2]
bs.t3 << 1.0
print(bs.weight[3] >> None)                  # 1.0
```

---

## 4. Plug — connections, hashing, equality

Connection queries are methods, never operators, and always answer with a
`PlugList` (empty when nothing is wired). Direct connections only: a
compound whose children are driven reports nothing, so slice it.

```python
cmds.file(new=True, force=True)
drv, ctrl, spare = (Node.create("transform", name=n) for n in ("drv", "ctrl", "spare"))
ctrl.tx  << drv.tx
spare.tx << ctrl.tx
spare.ty << ctrl.tx

print(repr(ctrl.tx.get_inputs()))                                # PlugList([Plug("drv.translateX")])
print(repr(ctrl.tx.get_outputs()))                               # PlugList([Plug("spare.translateX"), Plug("spare.translateY")])
print(repr(ctrl.ty.get_inputs()))                                # PlugList([])
print(len(ctrl.t.get_inputs()), len(ctrl.t[:].get_inputs()[0]))  # 0 1
```

`==` on a plug builds a comparison node and returns *its output plug*, not
a bool. Use `equals` for identity. Hashing is by MObject handle plus
attribute name, so plugs still work as dict and set keys.

```python
test = ctrl.tx == 5
print(repr(test), cmds.nodeType(test.node))                      # Plug("equal1.output") equal
print(ctrl.tx.equals(ctrl.tx), ctrl.tx.equals(drv.tx))           # True False
print(len({ctrl.tx, ctrl.tx}), {ctrl.tx: "x"}[Node("ctrl").tx])  # 1 x
print(ctrl.tx in PlugList([drv.tx, ctrl.tx]))                    # True  -- containment compares names, builds nothing
```

A `Plug` is a `str` subclass, so `cmds` accepts it directly and `in` is
substring containment. Every public `str` method name is a legal Maya
attribute name, so those names resolve as attributes; cast when you want
the string method.

```python
print(cmds.getAttr(ctrl.tx), "translate" in ctrl.tx)  # 0.0 True
print(str(ctrl.tx).upper())                           # CTRL.TRANSLATEX
```

---

## 5. Plug — `>>` clones and publishes

`plug >> Node` clones the attribute's spec onto another node (no value, no
connection; a same-named dynamic attribute on the target is replaced).
`plug >> "name"` clones onto the *same* node under a new name and copies
the value; `plug >> "other.name"` onto another node; the named forms refuse
a name the target already has. What travels and what does not is in
[`spec/CHEATSHEET.md`](spec/CHEATSHEET.md), section 12.

```python
from rig.spec import Color, Float

src = Node.create("transform", name="src")
dst = Node.create("transform", name="dst")
src << Float("blend", min=0, max=1) << 0.25
src << Color("tint") << [0.1, 0.2, 0.3]

print(repr(src.blend >> dst),       dst.blend >> None)      # Plug("dst.blend") 0.0
print(repr(src.blend >> "blend2"),  src.blend2 >> None)     # Plug("src.blend2") 0.25
print(repr(src.tint >> "dst.tint"), dst.tint.num_children)  # Plug("dst.tint") 3
try:
    src.blend >> "tint"
except TypeError as err:
    print(str(err)[:37])                                  # 'src' already has an attribute 'tint'
```

`plug >> container` publishes onto the active container (or an explicit
`Container`): an input if the plug is writable, an output if it is not.
The published name is a native `bindAttr` alias, so the return value is
the real inner plug.

```python
from rig import container

with container("box") as ctn:
    knob = Node.create("transform", name="knob")
    knob << Float("blend") << 0.5
    mul = Node.create("multiplyDivide", name="mul")
    mul.input1X << knob.blend
    mul.input2X << 2
    print(repr(knob.blend >> container))  # Plug("knob.blend")
    print(repr(mul.outputX >> ctn))       # Plug("mul.outputX")

print(cmds.container("box", query=True, publishName=True))     # ['blend', 'outputX']
print(cmds.getAttr("box.blend"), cmds.getAttr("box.outputX"))  # 0.5 1.0
print(repr(ctn.blend))                                         # Plug("knob.blend")  -- Container resolves the alias
```

Publishing is a no-op that returns the source plug when the scope is
flattened (nested under the default `flatten_containers=True`), when
`create_containers=False`, or when `publish_attributes=False`.

---

## 6. PlugList

A `PlugList` is a `list` whose attribute access and operators broadcast
over its elements. Strings lift to `Node` / `Plug`; numbers and `None` pass
through.

```python
cmds.file(new=True, force=True)
for name in ("c0", "c1", "c2"):
    Node.create("transform", name=name)

nodes = PlugList(["c0", "c1", "c2"])
print(repr(nodes.tx))                               # PlugList([Plug("c0.translateX"), Plug("c1.translateX"), Plug("c2.translateX")])
print(repr(PlugList([Node("c0"), 3.14, None]).ty))  # PlugList([Plug("c0.translateY"), 3.14, None])
```

`<<` broadcasts a scalar to every element and pairs a sequence element by
element; a shorter right side is capped to its last entry.

```python
nodes.t  << 0          # every translate to the origin
nodes.tx << [1, 2, 3]  # pairwise
nodes.ty << [10, 20]   # asymmetric: c2.ty gets 20
print(nodes.t >> None)                       # [[ 1. 10.  0.] [ 2. 20.  0.] [ 3. 20.  0.]]
nodes.tz = 5                                 # assignment sugar broadcasts too
print(nodes.tz >> None)                      # [5. 5. 5.]
```

Slicing keeps the type; a list, tuple or array key picks slots (fancy
indexing); an `int` returns the element.

```python
print(repr(nodes[1:]))                        # PlugList([Node("c1"), Node("c2")])
print(repr(nodes.tx[[0, 2]]))                 # PlugList([Plug("c0.translateX"), Plug("c2.translateX")])
print(repr(nodes[-1]), repr(nodes["ty"][0]))  # Node("c2") Plug("c0.translateY")
```

`>> None` (or `.get()`) stacks the values into one array when the shapes
agree and falls back to a plain list otherwise. Arithmetic and comparisons
map element by element.

```python
print(nodes.tx >> None, (nodes.t >> None).shape, (nodes.matrix >> None).shape)  # [1. 2. 3.] (3, 3) (3, 4, 4)
print(type(nodes >> None).__name__, type((nodes >> None)[0]).__name__)          # list Transform
print(repr(nodes.tx + nodes.ty))                                                # PlugList([Plug("add1.output"), Plug("add2.output"), Plug("add3.output")])
print((nodes.tx * 2) >> None)                                                   # [2. 4. 6.]
print(Node("c1") in nodes, nodes.index(Node("c2")))                             # True 2
```

Connection queries are N-aligned: one `PlugList` per element, so a slot is
never a `None` that `<<` would read as "disconnect".

```python
Node("c1").tx << Node("c0").tx
print(repr(nodes.tx.get_inputs()))  # PlugList([PlugList([]), PlugList([Plug("c0.translateX")]), PlugList([])])
print(nodes.tx.get_inputs().get())  # [[], array([1.]), []]  -- a query result reads as values
```

---

## 7. Arithmetic

Every operator builds a node and returns its output plug. Numbers on
either side are fine (the reflected forms work), and the result reads back
through `>> None`.

```python
cmds.file(new=True, force=True)
a = Node.create("transform", name="a")
b = Node.create("transform", name="b")
a.tx << 6
b.tx << 4

print(repr(a.tx + b.tx),     (a.tx + b.tx) >> None)        # Plug("add1.output") 10.0
print((a.tx - b.tx) >> None, (10 - a.tx) >> None)          # 2.0 4.0
print((a.tx * b.tx) >> None, (a.tx / b.tx) >> None)        # 24.0 1.5
print((a.tx ** 2) >> None,   (2 ** b.tx) >> None)          # 36.0 16.0
print((a.tx // 4) >> None,   (a.tx % 4) >> None)           # 1 2.0
print((-a.tx) >> None,       cmds.nodeType((-a.tx).node))  # -6.0 negate
```

Repeated expressions dedupe: the same operands give the same node, not a
second one (section 13).

```python
first  = a.tx + b.tx
second = a.tx + b.tx
print(first.equals(second))                  # True
```

Compounds go through the same operators, channel by channel. A literal
vector is an operand too.

```python
a.t << [1, 2, 3]
b.t << [10, 20, 30]
print(repr(a.t + b.t), (a.t + b.t) >> None)     # Plug("add2.output3D") [11. 22. 33.]
print((a.t * 2) >> None, (a.t + b.tx) >> None)  # [2. 4. 6.] [11. 12. 13.]  -- a scalar broadcasts
print(([100, 100, 100] - a.t) >> None)          # [99. 98. 97.]
```

| Operator | Scalar node (2024+ / legacy) | Compound |
|---|---|---|
| `+` | `sum` / `plusMinusAverage` | `plusMinusAverage.output3D` |
| `-` | `subtract` / `plusMinusAverage` | `plusMinusAverage.output3D` |
| `*` `/` `**` | `multiply` `divide` `power` / `multiplyDivide` | `multiplyDivide.output` |
| `//` | a `floordiv` network: divide, subtract, cast to `long` | the same network, compound-wise |
| `%` | `modulo` / a divide-floor-multiply network | one `modulo` per channel / the network compound-wise |
| `-x` | `negate` / `multiplyDivide` by `-1` | `multiplyDivide` by `-1` |

`//` and `%` (and the legacy paths) are multi-node expressions, so they
are wrapped in their own container (`floordiv1`, `modulo1`) with published
`input1` / `input2` / `output`.

---

## 8. Matrices and quaternions

`*` on matrix plugs chains a `multMatrix`; a vector against a matrix is a
point transform. `-` between matrices multiplies by the inverse. Matrix
times scalar is undefined; `**` is the fractional transform toward
identity.

```python
a.t << [10, 0, 0]
b.t << [0, 5, 0]
print(repr(a.wm * b.wim))                                  # Plug("multMatrix1.matrixSum")
print(np.round((a.wm * b.wim) >> None, 3)[3])              # [10. -5.  0.  1.]  -- a in b's space
print(repr([1, 0, 0] * a.wm), ([1, 0, 0] * a.wm) >> None)  # Plug("multiplyPointByMatrix1.output") [11.  0.  0.]
print(np.round((a.wm - b.wm) >> None, 3)[3])               # [10. -5.  0.  1.]  -- a.wm * inverse(b.wm)
print(np.round((a.wm ** 0.5) >> None, 3)[3])               # [5. 0. 0. 1.]  -- halfway to identity
try:
    a.wm * 0.5
except TypeError as err:
    print(str(err)[:30])                     # matrix * scalar is undefined;
```

A 4-channel compound is a quaternion: `*` is `quatProd`, `+` / `-` are
`quatAdd` / `quatSub`, `**` is the fractional rotation.

```python
from rig.spec import Quat

a << Quat("qa") << [0, 0, 0, 1]
b << Quat("qb") << [0, 0.7071, 0, 0.7071]
print(repr(a.qa * b.qb), repr(a.qa + b.qb))  # Plug("quatProd1.outputQuat") Plug("quatAdd1.outputQuat")
print(np.round((a.qa * b.qb) >> None, 4))    # [0.     0.7071 0.     0.7071]
```

Type-aware shorthand: connecting a matrix into a transform channel inserts
a `decomposeMatrix`; a bare `node << matrix_plug` wires translate, rotate,
scale and shear at once. A static numpy matrix decomposes in Python and
sets the channels.

```python
c = Node.create("transform", name="c")
c.t << a.wm                                  # decomposeMatrix.outputTranslate -> c.t
print(cmds.listConnections("c.t", source=True, plugs=True))   # ['decomposeMatrix1.outputTranslate']
c.r << a.wm                                  # same decomposeMatrix, reused
print(len(cmds.ls(type="decomposeMatrix")))  # 1

d = Node.create("transform", name="d")
d << a.matrix                                # t / r / s / shear all driven
print(set(cmds.listConnections("d", source=True, type="decomposeMatrix")))   # {'decomposeMatrix2'}  -- a.matrix, not a.wm: a second node

m        = np.eye(4)
m[3, :3] = [1, 2, 3]
d << m                                       # static: decomposed here, channels set, drivers removed
print(d.t >> None, cmds.listConnections("d.t", source=True))   # [1. 2. 3.] None
d.s << np.diag([2.0, 3.0, 4.0, 1.0])         # a single channel takes just its component
print(d.s >> None, d.t >> None)              # [2. 3. 4.] [1. 2. 3.]
```

---

## 9. Logic, comparisons, `condition`, `constant`

`& | ^` build AND / OR / XOR networks and `~` is logical NOT; every
comparison builds a comparison node. On a compound they fan out per
channel inside a published container and return its assembled output.

```python
cmds.file(new=True, force=True)
a = Node.create("transform", name="a")
b = Node.create("transform", name="b")
a.tx << 6
b.tx << 4

print(repr(a.tx > b.tx), (a.tx > b.tx) >> None)  # Plug("greater1.output") True
print(repr(a.tx == 6),   repr(a.tx >= 6))        # Plug("equal1.output") Plug("greater_or_equal1.outColorR")
print(repr(a.tx & b.tx), (a.tx & 0) >> None)     # Plug("logical_and1.output") False
print(repr(a.tx | b.tx), repr(a.tx ^ b.tx))      # Plug("logical_or1.output") Plug("equal2.output")  -- xor is a network in a logical_xor1 container
print((~a.tx) >> None,   (~(a.tx * 0)) >> None)  # False True

a.t << [0, 1, 0]
zero = a.t == 0                                  # three equal nodes in a condition1 container
print(repr(zero), zero >> None)  # Plug("output_plug1.value") [1. 0. 1.]
print(repr(zero.input1))         # Plug("condition1_host.input1")  -- the published interface
```

`condition(test, if_true, if_false)` is the value picker (a `condition`
node); a numeric test short-circuits in Python. `constant(values)` holds a
literal in a `network` node (a `holdMatrix` for matrix shapes) and dedupes.

```python
from rig import condition, constant

a.tx << 6                                        # the a.t write above zeroed it
pick = condition(a.tx > b.tx, a.tx, b.tx)
print(repr(pick), pick >> None)                              # Plug("condition2.outColorR") 6.0
print(repr(condition(a.tx > b.tx, a.t, [0, 0, 0])))          # Plug("condition3.outColor")
print(condition(1, "yes", "no"), condition(0, "yes", "no"))  # yes no

k = constant(5.0)
print(repr(k), k >> None, constant(5.0).equals(k))             # Plug("constant1.value") 5.0 True
print(repr(constant([1, 2, 3])), constant([1, 2, 3]) >> None)  # Plug("constant2.value") [1. 2. 3.]
print(repr(constant(np.eye(4))))                               # Plug("constant3.outMatrix")
```

---

## 10. Components

Point components are plugs: `vtx` / `cv` / `pt` / `map` resolve to the
shape's `controlPoints` / `uvpt`, through a transform with one geometry
shape. A slice is a `PlugList`, a list key is fancy indexing.

```python
cmds.file(new=True, force=True)
cube  = Node(cmds.polyCube(name="pCube1", constructionHistory=False)[0])
shape = Node("pCube1Shape")

print(repr(cube.vtx), len(cube.vtx[:]))  # Plug("pCube1Shape.controlPoints") 8
print(repr(shape.vtx[[0, 7]]))           # PlugList([Plug("pCube1Shape.controlPoints[0]"), Plug("pCube1Shape.controlPoints[7]")])

cube.vtx[::2] << [0, 0, 0]                   # one point value, broadcast to the four even vertices
print(cmds.pointPosition("pCube1.vtx[2]", local=True))   # [0.0, 0.0, 0.0]
cube.vtx[::2] << [[1, 0, 0], [0, 2, 0], [0, 0, 3], [4, 4, 4]]   # one value per element
print(cmds.pointPosition("pCube1.vtx[6]", local=True))   # [4.0, 4.0, 4.0]
```

Faces and edges have no plug, so `cube.f` / `cube.e` are a `Components`:
an opaque, sorted, unique selection that costs no plugs. `>> None` reads
the native ids.

```python
from rig import Components

print(repr(cube.f), repr(cube.e[[4, 0]]))    # Components("|pCube1|pCube1Shape.f[*]") Components("|pCube1|pCube1Shape.e[0] e[4]")
faces = cube.f[:3]
print(faces.indices, faces.count, faces.names)                                    # [0 1 2] 3 ['|pCube1|pCube1Shape.f[0:2]']
print(faces.kind, faces.geometry, str(faces.shape), faces.is_all, cube.f.is_all)  # f mesh pCube1Shape False True
print(cube.e[[4, 0]] >> None, cube.f >> None)                                     # [0 4] [0 1 2 3 4 5]
print(cube.f[[5, 2]] == cube.f[[2, 5]], bool(cube.f[6:]))                         # True False
```

`Components(node, kind, ids)` is the explicit carrier for every point kind
(a numpy array in, no `PlugList` built), and `Components("pCube1.f[0:3]")`
reads a Maya component string.

```python
verts = Components(cube, "vtx", np.array([4, 0, 2, 2]))
print(repr(verts), verts.count)                                          # Components("|pCube1|pCube1Shape.vtx[0] vtx[2] vtx[4]") 3
print(repr(Components("pCube1.f[0:3]")), Components(shape, "uv").count)  # Components("|pCube1|pCube1Shape.f[0:3]") 14
cmds.select(cube.f[[0, 1, 2, 5]].names)                                  # names are selectable, compact tokens
print(cmds.ls(selection=True))                                           # ['pCube1.f[0:2]', 'pCube1.f[5]']
```

A NURBS surface `cv` and a lattice `pt` index per axis, numpy style.

```python
surface = Node(cmds.nurbsPlane(name="plane1")[0])
print(repr(surface.cv[1, 2]), len(surface.cv[:, 0]), len(surface.cv[:]))  # ComponentPlug("plane1Shape.cv[1][2]") 4 16
print(Components(surface, "cv")[:2].names)                                # ['|plane1|plane1Shape.cv[0][0:1]']
```

Real attributes always win over the fallback: `curveShape.f` is `form`.
A transform with two geometry shapes raises an `AttributeError` naming
them.

---

## 11. Containers

`with container("name"):` creates a Maya container and every node made
inside joins it. Nested blocks flatten by default: no inner container, the
nodes go into the outer one with a `name_` prefix. `preserve=True` forces a
real sub-container, `enabled=False` makes a scope-only block.

```python
from rig import container, Container

cmds.file(new=True, force=True)
outside = Node.create("transform", name="outside")

with container("arm") as arm:
    ctrl = Node.create("transform", name="ctrl")
    with container("fk") as fk:
        fk_ctrl = Node.create("transform", name="ctrl")
    with container("ik", preserve=True) as ik:
        ik_ctrl = Node.create("transform", name="ctrl")
    with container("scratch", enabled=False) as scratch:
        tmp = Node.create("transform", name="tmp")
    container.add(outside)                          # adopt a node made elsewhere
    loose = Node.create("multiplyDivide", name="loose", container=False)   # create, but do not add

print(repr(arm), fk, repr(ik), scratch)                          # Container("arm") None Container("ik") None
print(str(fk_ctrl), str(ik_ctrl), str(tmp))                      # fk_ctrl ctrl1 scratch_tmp
print(sorted(cmds.container("arm", query=True, nodeList=True)))  # ['ctrl', 'fk_ctrl', 'ik', 'outside', 'scratch_tmp']
print(cmds.container(query=True, findContainer="loose"))         # None
print(container.is_active, container.stack)                      # False []
```

Every scope prefixes the names it creates, the real sub-container `ik` is
itself a member of `arm`, and a disabled scope makes no container of its
own but still hands its nodes to the enclosing one. Containers are not
DAG parents, so the second `ctrl` is `ctrl1`.

A `Container` is a `Node`: it takes attribute specs, and a published name
resolves to the real plug behind it (section 5).

```python
from rig.spec import Float

arm << Float("stretch") << 1.5
print(repr(arm.stretch), Container("arm").stretch >> None)   # Plug("arm.stretch") 1.5
with container("lerp1") as lerp:
    w   = container.publish_input(0.25, "weight", min=0, max=1)          # a knob with no inner home
    out = container.publish_output((ctrl.tx - outside.tx) * w, "output")
print(repr(w), repr(lerp.weight), cmds.container("lerp1", query=True, publishName=True))   # Plug("lerp1_host.weight") Plug("lerp1_host.weight") ['weight', 'output']
```

The math operators register their nodes with the active scope too, so a
whole expression lands in one container:

```python
with container("offset1") as box:
    outside.ty << ctrl.tx * 2 + 1
print(sorted(cmds.nodeType(n) for n in cmds.container("offset1", query=True, nodeList=True)))   # ['multiply', 'sum']
```

---

## 12. Options, `force_nodes`, `cleanup`

`set_options` / `get_options` mirror numpy's print options: a setter that
returns `None` and a getter that returns a snapshot. `ContainerOptions` is
the class the flags live on.

```python
from rig import ContainerOptions, get_options, set_options

print(get_options())
# {'create_containers': True, 'use_shorthand': True, 'skip_selection': True, 'cleanup_on_exit': False,
#  'maya_version': None, 'flatten_containers': True, 'publish_attributes': True,
#  'absorb_unit_conversions': False, 'native_multi_publish': False, 'constant_folding': True}
set_options(create_containers=False)          # every with container(): becomes scope-only
with container("ghost") as ghost:
    Node.create("transform", name="free")
print(ghost, cmds.ls(type="container"), ContainerOptions.create_containers)   # None ['arm', 'ik', 'lerp1', 'offset1'] False
set_options(create_containers=True)
```

| Option | Default | Effect |
|---|---|---|
| `create_containers` | `True` | `False`: no Maya containers at all; scopes stay logical |
| `use_shorthand` | `True` | type-aware translation (`t << wm` inserts a `decomposeMatrix`) |
| `skip_selection` | `True` | `createNode(skipSelect=True)` |
| `cleanup_on_exit` | `False` | run `cleanup()` when a `with container():` block exits |
| `maya_version` | `None` | target an older Maya's node set (section 15) |
| `flatten_containers` | `True` | `False`: nested blocks create real sub-containers and publish |
| `publish_attributes` | `True` | `False`: every publish is a passthrough |
| `absorb_unit_conversions` | `False` | pull auto-inserted `unitConversion` nodes into the container |
| `native_multi_publish` | `False` | publish multi attrs natively (Maya renders them badly today) |
| `constant_folding` | `True` | all-literal math returns a Python value instead of a node |

Constant folding: a function given only numbers returns a number.
`force_nodes()` turns that off for a block so you can inspect the network
a literal call would build; dedupe still applies inside.

```python
from rig import force_nodes, functions as f

print(f.abs(-5), type(f.abs(-5)).__name__)   # 5 int
with force_nodes():
    plug = f.abs(-5)
    print(repr(plug), plug.equals(f.abs(-5)), get_options()["constant_folding"])   # Plug("abs1.output") True False
print(get_options()["constant_folding"])     # True
```

`cleanup()` garbage-collects the utility nodes the DSL created that have no
downstream consumer, iterating until a fixpoint, and deletes containers it
has emptied. Three filters keep it safe: only nodes carrying the hidden
`__rig__` tag that `Node.create` / the operators stamp, only DG utility
types (never a transform, joint, shape, light, set, layer or shading node),
and only when every consumer is also going. User-made nodes, locked nodes
and referenced nodes are never touched.

```python
from rig import cleanup

cmds.file(new=True, force=True)
cmds.createNode("multiplyDivide", name="user_mul")     # made outside the DSL: no tag
with container("gc") as gc:
    keep = Node.create("transform",      name="keep")
    live = Node.create("multiplyDivide", name="live")
    dead = Node.create("multiplyDivide", name="dead")
    keep.tx << live.outputX
    orphan_chain = dead.outputX * 2                     # dead -> multiply, nothing reads the product

print(cleanup(container="gc", dry_run=True))
# {'by_owner': {'gc': ['dead', 'mul1']}, 'deleted_containers': [], 'memoize_entries_pruned': 0, 'dry_run': True}
report = cleanup(container="gc")
print(sorted(report["by_owner"]["gc"]), report["memoize_entries_pruned"] > 0)   # ['dead', 'mul1'] True
print([cmds.objExists(n) for n in ("keep", "live", "dead", "user_mul", "gc")])  # [True, True, False, True, True]
print(repr(gc.cleanup()["by_owner"]))                                           # {}  -- the Container method, same call scoped to itself
```

`cleanup_on_exit=True` (per block or globally) runs it as the block exits,
and `cleanup()` with no `container=` sweeps the whole scene, including
nodes made with `create_containers=False`.

---

## 13. `memoize`, `vectorize`, `prune_memoize_caches`

`@memoize` caches a function's return keyed on the identity of every
`Plug` / `Node` / `PlugList` argument and the value of every number. The
key survives renames; an entry is dropped when any node it returned has
been deleted. The operators (through `NodeOp`, section 15) and the library
functions are memoized this way, which is why repeated expressions dedupe.

```python
from rig import memoize, prune_memoize_caches

cmds.file(new=True, force=True)
a     = Node.create("transform", name="a")
calls = []

@memoize
def double(plug):
    calls.append(str(plug))
    node = Node.create("multiplyDivide", name="double1")
    node.input1X << plug
    node.input2X << 2
    return node.outputX

print(repr(double(a.tx)), repr(double(a.tx)), len(calls))  # Plug("double1.outputX") Plug("double1.outputX") 1
cmds.rename("a", "renamed")
print(repr(double(Node("renamed").tx)), len(calls))        # Plug("double1.outputX") 1  -- identity, not name
cmds.delete("double1")
print(repr(double(Node("renamed").tx)), len(calls))        # Plug("double1.outputX") 2  -- stale entry recomputed: a new node, the freed name reused
```

`@memoize(foldable="scalar")` (or `"reduce"`, or a predicate) declares that
the function returns a plain value when every input is a literal; such
calls fold before the cache and are never cached.

```python
@memoize(foldable="scalar")
def half(x):
    if isinstance(x, (int, float)):
        return x / 2
    return x / 2.0                              # a Plug: builds a divide node

print(half(8), repr(half(Node("renamed").tx)), half._foldable)   # 4.0 Plug("div1.output") scalar
```

`@vectorize` broadcasts a call across `PlugList` arguments with numpy's
strict rule: every list must be the same length, or length one, or a
scalar. One row returns the bare result; more return a `PlugList`.

```python
from rig import vectorize

for name in ("p", "q", "r"):
    Node.create("transform", name=name)
ctrls = PlugList(["p", "q", "r"])

@vectorize
def offset(plug, amount):
    return plug + amount

print(repr(offset(ctrls.tx, 1)))                 # PlugList([Plug("add1.output"), Plug("add2.output"), Plug("add3.output")])
print(repr(offset(ctrls.tx, [10, 20, 30])))      # PlugList([Plug("add4.output"), Plug("add5.output"), Plug("add6.output")])
print(repr(offset(PlugList([ctrls.tx[0]]), 5)))  # Plug("add7.output")  -- one row, unwrapped
try:
    offset(ctrls.tx, [1, 2])
except ValueError as err:
    print(str(err)[:53])                      # vectorize('offset'): cannot broadcast lengths [2, 3]
```

`prune_memoize_caches()` walks every `@memoize` and `NodeOp` cache and
drops entries whose nodes are gone; `cleanup()` calls it for you.

```python
cmds.delete("add1", "add2", "add3")
print(prune_memoize_caches() >= 3)            # True
```

---

## 14. `sequences` and `arguments`

The asymmetric generators behind every broadcast in the DSL. Pure Python,
no Maya. Scalars and strings repeat; a shorter sequence caps to its last
entry.

```python
from rig import arguments, sequences

print(list(sequences([1, 2, 3], 5)))                      # [[1, 5], [2, 5], [3, 5]]
print(list(sequences([1, 2, 3, 4], ["a", "b"], "xy")))    # [[1, 'a', 'xy'], [2, 'b', 'xy'], [3, 'b', 'xy'], [4, 'b', 'xy']]
print(list(arguments([1, 2], scale=[10, 20], name="n")))  # [([1], {'scale': 10, 'name': 'n'}), ([2], {'scale': 20, 'name': 'n'})]
print(list(sequences()), list(arguments()))               # [] []
```

`sequences` is the permissive rule (`<<` and `PlugList` use it);
`@vectorize` checks lengths first and raises instead of capping.

---

## 15. `NodeOp` and the target Maya version

A `NodeOp` is one operation with several version-keyed implementations.
At call time it picks the highest `since` that is at or below the *target*
Maya version, falls through on `NotImplementedError`, folds all-numeric
calls through `scalar_fn`, memoizes like `@memoize`, and fans a
`scope="scalar"` impl out per channel for compound input inside a
published container.

```python
from rig import NodeOp

cmds.file(new=True, force=True)
a     = Node.create("transform", name="a")
twice = NodeOp("twice", scalar_fn=lambda x: x * 2)

@twice.impl(since=2024, scope="scalar")
def _twice_native(x):
    node = Node.create("multiply", name="twice1")
    node.input[0] << x
    node.input[1] << 2
    return node.output

@twice.impl(since=0, scope="scalar")
def _twice_legacy(x):
    node = Node.create("multiplyDivide", name="twice1")
    node.input1X << x
    node.input2X << 2
    return node.outputX

a.tx << 3
print(twice(3), repr(twice(a.tx)), twice(a.tx) >> None)  # 6 Plug("twice1.output") 6.0
print(repr(twice(a.t)), cmds.nodeType(twice(a.t).node))  # Plug("output_plug1.value") network  -- one assembled output
print(len(cmds.ls(type="multiply")))                     # 4  -- twice1 plus one per channel of a.t
```

`set_options(maya_version=N)` retargets the whole DSL: every `NodeOp` and
every `is_at_least` check reads the target, and the caches are cleared so
the next call re-dispatches. `None` reverts to the running Maya.

```python
from rig import set_options
from rig._internal.maya_version import get_maya_version, get_target_version, is_at_least

set_options(maya_version=2022)
print(get_target_version(), is_at_least(2024))                        # 2022 False
print(cmds.nodeType(twice(a.tx).node), repr(a.tx + a.ty))             # multiplyDivide Plug("add1.output1D")
set_options(maya_version=None)
print(get_target_version() == get_maya_version(), repr(a.tx + a.ty))  # True Plug("add2.output")
```

`rig._internal.maya_version.set_target_version(N)` is the raw switch
underneath: it moves the target without clearing the `@memoize` caches, so
prefer `set_options`. A few operations have no legacy network at all and
raise `RuntimeError` when the target is below 2024.

---

## 16. `InjectionError`

`<<` asserts the desired state of a plug and breaks an incoming connection
that is in the way. The one barrier it respects is a lock: setting or
connecting into a locked attribute raises `InjectionError` (a
`RuntimeError`), and a compound with a single locked child mutates nothing.
Disconnecting is still allowed.

```python
from rig import InjectionError
from rig.spec import lock, unlock

cmds.file(new=True, force=True)
a = Node.create("transform", name="a")
b = Node.create("transform", name="b")
b.tx << a.tx
b.ty << 1
b.ty << lock

try:
    b.ty << 5
except InjectionError as err:
    print(type(err).__name__, isinstance(err, RuntimeError))   # InjectionError True
try:
    b.t << [7, 7, 7]
except InjectionError:
    print(b.t >> None)                       # [0. 1. 0.]  -- all or nothing
b.ty << None                                 # a disconnect is fine on a locked plug
b.ty << unlock << 5
print(b.ty >> None)                          # 5.0
```

---

## 17. Function libraries: the map

Nine submodules and twelve top-level verbs. Every function takes plugs
or literals, builds a small network, and returns the output `Plug` —
ready for `<<` or the next call. The node names in the comments below
come from a fresh scene on Maya 2025; the suffix numbers move, the node
types do not.

```python
cmds.file(new=True, force=True)

import math

from rig import Node, Plug, PlugList
from rig import functions as f, trigonometry as trig
from rig import matrix as m, vector as v, quaternion as q, euler as e
from rig import interpolate as ip, tween as tw, random as r
from rig import (dist, lerp, slerp, blend, elerp, normalize, inverse,
                 angle, angle_degrees, to_euler, to_quaternion, to_matrix)


def kind(plug):
    """The node type behind a plug -- what the call built."""
    return cmds.nodeType(str(plug).split(".")[0])


src = Node.create("transform", name="src")
src.t << (1, 2, 3)

out = f.abs(src.tx)
print(repr(out), kind(out))                     # Plug("abs1.output") absolute
print(out >> None)                              # 1.0 -- the live value
print(f.abs(-2.5))                              # 2.5 -- all-literal input folds to a float, no node
print(str(f.abs(src.tx)) == str(out))           # True -- same call, same plug (memoised)
print(repr(f.abs(PlugList([src.tx, src.ty]))))  # PlugList([Plug("abs1.output"), Plug("abs2.output")]) -- broadcast; abs1 reused
```

| Module | Import as | Holds |
|---|---|---|
| `rig.functions` | `f` | scalar math, reducers over lists, `frame` / `pi` / `inf`, comparison and logic wrappers |
| `rig.trigonometry` | `trig` | `sin` .. `atan2` in radians, `sind` .. `atan2d` in degrees, `degrees` / `radians` |
| `rig.matrix` | `m` | decompose / compose / aim, multiply / add / inverse, `lerp` / `blend` / `slerp` / `pow`, extraction helpers |
| `rig.vector` | `v` | `dot` / `cross` / `length` / `normalize` / `dist` / `angle` / `rotate`, `lerp` / `slerp` / `elerp`, the `X` `Y` `Z` constants |
| `rig.quaternion` | `q` | Hamilton arithmetic, `slerp` / `pow`, axis-angle, conversions |
| `rig.euler` | `e` | `reorder`, `to_matrix` / `to_quaternion` / `slerp` |
| `rig.interpolate` | `ip` | `sequence`, `smoothstep` / `smootherstep`, `inverse_lerp` |
| `rig.tween` | `tw` | 42 Penner easing curves |
| `rig.random` | `r` | LCG pseudo-random networks, scalar and 3D |
| `rig.*` verbs | flat | `dist` `lerp` `slerp` `blend` `elerp` `normalize` `inverse` `angle` `angle_degrees` `to_euler` `to_quaternion` `to_matrix` — type-dispatched |

Five rules that hold everywhere:

| Rule | What it means |
|---|---|
| Literals fold | all-number input returns a Python number and builds nothing (`f.clamp(5, 0, 1)` is `1`). `tween` and `random` are the exceptions: they always build |
| Calls are memoised | the same function with the same arguments returns the same plug, so a value used twice costs one network |
| `PlugList` broadcasts, a plain list is a value | `f.abs(PlugList([a, b]))` is two networks; `f.abs([a, b])` tries to inject a 2-vector |
| Maya 2024+ gets native nodes | `absolute`, `clampRange`, `sin`, `dotProduct`, `lerp`, `smoothStep` … Older Maya gets the equivalent legacy network — same value, more nodes. Each section's table says which |
| `functions` shadows builtins | `abs`, `int`, `round`, `min`, `max`, `sum`, `pow`, `all`, `any` … Always `from rig import functions as f`, never `import *` |

With the default options every composite function's container is
flattened, so you get the raw node plugs shown here.
`rig.set_options(flatten_containers=False)` turns each one into a real
container with a published `input` / `output` (or `input1` / `input2` /
`weight`) interface instead.

---

## 18. `functions` — scalar math

Rounding, truncation and sign. `int` is a long cast; `trunc` / `floor`
/ `ceil` are the 2024+ nodes of the same name.

```python
x = Node.create("transform", name="x")
x.tx << -3.7

for name in ("abs", "int", "trunc", "floor", "ceil", "sign"):
    out = getattr(f, name)(x.tx)
    print(f"{name:6s} {kind(out):10s} {out >> None}")
# abs    absolute   3.7
# int    network    -3
# trunc  truncate   -3.0
# floor  floor      -4.0
# ceil   ceil       -3.0
# sign   condition  -1.0
```

`round` takes a digit count; `clamp` is `clampRange` when everything
is scalar.

```python
x.tx << 3.14159
print(repr(f.round(x.tx, 2)), kind(f.round(x.tx, 2)), f.round(x.tx, 2) >> None)           # Plug("div1.output") divide 3.14
print(repr(f.round(x.tx)), kind(f.round(x.tx)))                                           # Plug("round3.output") round
print(repr(f.clamp(x.tx, 0, 1)), kind(f.clamp(x.tx, 0, 1)), f.clamp(x.tx, 0, 1) >> None)  # Plug("clamp1.output") clampRange 1.0
```

Powers and logs all land on the 2024+ `power` node; `log` has no
pre-2024 fallback and raises `RuntimeError` there.

```python
x.tx << 16.0
print(kind(f.sqrt(x.tx)),       f.sqrt(x.tx) >> None)         # power 4.0
print(kind(f.pow(x.tx, 0.25)),  f.pow(x.tx, 0.25) >> None)    # power 2.0
print(kind(f.exp(x.tx)))                                      # power
print(kind(f.log(x.tx)),        f.log(x.tx, base=2) >> None)  # log 4.0
print(kind(f.rev(x.tx)),        f.rev(x.tx) >> None)          # reverse -15.0  (1 - x)
print(f.sqrt(16), f.pow(2, 8), f.log(100, base=10))           # 4.0 256 2.0 -- literals fold
```

| Function | Maya 2024+ | Older Maya |
|---|---|---|
| `abs` | `absolute` | `condition(x < 0, -x, x)` |
| `clamp` | `clampRange` | two nested `condition` |
| `round` | `round` (`digits=0`), or scale → `round` → divide | long cast trick |
| `floor` / `ceil` / `trunc` | `floor` / `ceil` / `truncate` | long cast / `condition(ceil, floor)` |
| `sqrt` / `pow` / `exp` | `power` | `multiplyDivide` op 3 |
| `log` | `log` | raises `RuntimeError` |
| `int` / `sign` / `rev` | long-cast `network` / `condition` / `reverse` | same |

---

## 19. `functions` — lists: reduce, pick, search

The reducers take **one list** as their argument, not `*args`. `max`
and `min` want at least two entries.

```python
p = Node.create("transform", name="p")
p.t << (1, 5, 3)
chans = [p.tx, p.ty, p.tz]

for name in ("sum", "avg", "max", "min", "argmax", "argmin", "all", "any"):
    out = getattr(f, name)(chans)
    print(f"{name:7s} {kind(out):10s} {out >> None}")
# sum     sum        9.0
# avg     average    3.0
# max     max        5.0
# min     min        1.0
# argmax  condition  1.0
# argmin  condition  0.0
# all     condition  1.0
# any     condition  1.0
```

`diff` and `cumsum` return a `PlugList`, one plug per output:

```python
print(repr(f.diff(chans)))                   # PlugList([Plug("sub8.output"), Plug("sub9.output")])
print([d >> None for d in f.diff(chans)])    # [4.0, -2.0]
print([c >> None for c in f.cumsum(chans)])  # [1.0, 6.0, 9.0]
```

`choice` wraps Maya's `choice` node; `searchsorted` is
`numpy.searchsorted` as a `condition` ladder:

```python
pick = f.choice(chans, selector=2)
print(repr(pick), kind(pick), pick >> None)        # Plug("choice1.output") choice 3.0

p.tx << 1.5
idx = f.searchsorted([0.0, 1.0, 2.0, 3.0], p.tx)
print(kind(idx), idx >> None)                                                  # condition 1.0 -- segment index
print(f.searchsorted([0.0, 1.0, 2.0, 3.0], p.tx, return_index=False) >> None)  # 1.0 -- the knot value instead
print(f.sum([1, 2, 3]), f.max([1, 5, 3]), f.argmin([5.0, 1.0, 3.0]))           # 6 5 1 -- literals fold
```

| Function | Maya 2024+ | Older Maya |
|---|---|---|
| `sum` / `avg` | `sum` / `average` | `plusMinusAverage` (op 1 / 3) — also used for compound inputs on any version |
| `max` / `min` | `max` / `min` | nested `condition` |
| `all` / `any` / `argmin` / `argmax` / `searchsorted` | `condition` networks | same |
| `diff` / `cumsum` / `choice` | `subtract` / `sum` / `choice` | `plusMinusAverage` / `choice` |

---

## 20. `functions` — time, constants, comparison and logic

`frame()` is an `animCurveTL` with two linear keys and linear infinity:
it ticks one unit per frame.

```python
fr = f.frame()
print(repr(fr), kind(fr))  # Plug("frame1.output") animCurveTL
cmds.currentTime(10)
print(fr >> None)          # 10.0
print(f.inf())             # inf -- a plain float, no node
```

`pi()` is the native `pi` node on 2024+, and its output is a
`doubleAngle`: pi radians internally, which reads back as `180.0` in a
degrees scene and lands as `180.0` on a unitless plug such as `tx`.
Wire it into angle plugs; for plain scalar math use `math.pi`.

```python
pi = f.pi()
print(repr(pi), kind(pi), pi >> None)    # Plug("pi1.output") pi 180.0
```

`equal` is the fuzzy one (an `epsilon`); the other five comparison
wrappers are exactly the `==` `!=` `<` `>` `<=` `>=` operators on
`Plug`, for people who prefer names.

```python
c = Node.create("transform", name="c")
c.tx << 1.0
c.ty << 1.0001
print(kind(f.equal(c.tx, c.ty, eps=1e-3)), f.equal(c.tx, c.ty, eps=1e-3) >> None)                      # equal True
print(f.equal(c.tx, c.ty) >> None)                                                                     # False -- default eps is 1e-6
print(kind(f.not_equal(c.tx, c.ty)), kind(f.greater_than(c.tx, c.ty)), kind(f.less_than(c.tx, c.ty)))  # condition greaterThan lessThan
print(kind(f.greater_or_equal(c.tx, c.ty)), kind(f.less_or_equal(c.tx, c.ty)))                         # condition condition
print(f.greater_than(c.tx, c.ty) >> None, f.less_or_equal(c.tx, c.ty) >> None)                         # False 1.0
```

The logical wrappers are `&` `|` `^` `~`:

```python
c.tz << 0.0
print(kind(f.logical_and(c.tx, c.tz)), f.logical_and(c.tx, c.tz) >> None)  # and False
print(kind(f.logical_or(c.tx, c.tz)),  f.logical_or(c.tx, c.tz) >> None)   # or True
print(kind(f.logical_xor(c.tx, c.tz)), f.logical_xor(c.tx, c.tz) >> None)  # equal True  (built as (a!=0)+(b!=0) == 1)
print(kind(f.logical_not(c.tz)),       f.logical_not(c.tz) >> None)        # not True
```

| Function | Maya 2024+ | Older Maya |
|---|---|---|
| `pi` | `pi` (`doubleAngle` out) | a `network` constant holding `math.pi` |
| `equal` | `equal` | `abs(a - b) <= eps` |
| `greater_than` / `less_than` | `greaterThan` / `lessThan` | `condition` |
| `not_equal` / `greater_or_equal` / `less_or_equal` | `condition` | same |
| `logical_and` / `logical_or` / `logical_not` | `and` / `or` / `not` | `condition` networks |
| `logical_xor` | `(a != 0) + (b != 0) == 1` | same |

---

## 21. `trigonometry`

Two spellings of everything: `sind` / `cosd` / … take and return
**degrees**, `sin` / `cos` / … take and return **radians**. The degree
variants are the native 2024+ nodes; the radian ones wrap them in a
conversion. `sin(x)` is `sind(degrees(x))`, so you still get the `sin`
node's plug; `asin(x)` is `radians(asind(x))`, so you get the
conversion's `multiply`.

```python
h = Node.create("transform", name="h")
h.tx << 45.0
for name in ("sind", "cosd", "tand"):
    out = getattr(trig, name)(h.tx)
    print(name, repr(out), kind(out), round(out >> None, 6))
# sind Plug("sin1.output") sin 0.707107
# cosd Plug("cos1.output") cos 0.707107
# tand Plug("tan1.output") tan 1.0

h.tx << math.pi / 4
for name in ("sin", "cos", "tan"):
    out = getattr(trig, name)(h.tx)
    print(name, repr(out), kind(out), round(out >> None, 6))
# sin Plug("sin3.output") sin 0.707107
# cos Plug("cos3.output") cos 0.707107
# tan Plug("tan3.output") tan 1.0
```

The inverse functions return angles: `asind` / `acosd` / `atand` /
`atan2d` hand back the native node's `doubleAngle` output (reads as
degrees, connects to a rotate channel with no conversion); `asin` /
`acos` / `atan` / `atan2` convert to radians.

```python
h.tx << 1.0
h.ty << 0.0
print(kind(trig.asind(h.tx)),        trig.asind(h.tx) >> None)                  # asin 90.0
print(kind(trig.acosd(h.tx)),        trig.acosd(h.tx) >> None)                  # acos 0.0
print(kind(trig.atand(h.tx)),        trig.atand(h.tx) >> None)                  # atan 45.0
print(kind(trig.atan2d(h.tx, h.ty)), trig.atan2d(h.tx, h.ty) >> None)           # atan2 90.0
print(kind(trig.asin(h.tx)),         round(trig.asin(h.tx) >> None, 6))         # multiply 1.570796
print(kind(trig.acos(h.tx)),         trig.acos(h.tx) >> None)                   # multiply 0.0
print(kind(trig.atan(h.tx)),         round(trig.atan(h.tx) >> None, 6))         # multiply 0.785398
print(kind(trig.atan2(h.tx, h.ty)),  round(trig.atan2(h.tx, h.ty) >> None, 6))  # multiply 1.570796
```

`degrees` and `radians` are a multiply by `180/pi` or `pi/180`;
literals fold everywhere.

```python
print(kind(trig.degrees(h.tx)), round(trig.degrees(h.tx) >> None, 4))  # multiply 57.2958
print(kind(trig.radians(h.tx)), round(trig.radians(h.tx) >> None, 6))  # multiply 0.017453
print(trig.sind(90), trig.atan2(1, 1), trig.degrees(math.pi))          # 1.0 0.7853981633974483 180.0
```

| Function | Maya 2024+ | Older Maya |
|---|---|---|
| `sind` / `cosd` | `sin` / `cos` | `eulerToQuat` (`outputQuatX` = sin(x/2), `outputQuatW` = cos(x/2), fed `2x`) |
| `tand` | `tan` | `sind / cosd` with divide-by-zero quieting |
| `asind` / `acosd` / `atand` | `asin` / `acos` / `atan` | an `angleBetween` triangle trick |
| `atan2d` / `atan2` | `atan2` | `atan(y / x)` plus `condition` quadrant fix-ups |
| `sin` / `cos` / `tan` / `asin` / `acos` / `atan` | the degree variant wrapped in `degrees` / `radians` | same |
| `degrees` / `radians` | `multiply` | `multiplyDivide` |

The legacy `sind` / `cosd` / `asind` / `acosd` paths need the
`quatNodes` plugin; it is loaded for you.

---

## 22. `matrix` — read a matrix

`decompose` returns the `decomposeMatrix` node's `.outputTranslate`;
its siblings (`.outputRotate`, `.outputScale`, `.outputShear`,
`.outputQuat`) are one attribute lookup away, because a `Plug` falls
back to a sibling attribute when it has no child of that name.

```python
drv = Node.create("transform", name="drv")
drv.t << (10, 0, 0)
drv.r << (0, 90, 0)
drv.s << (3, 1, 1)

dec = m.decompose(drv.matrix)
print(repr(dec), kind(dec))                               # Plug("decomposeMatrix1.outputTranslate") decomposeMatrix
print(repr(dec.outputRotate), dec.outputRotate >> None)   # Plug("decomposeMatrix1.outputRotate") [-0. 90.  0.]
print(repr(m.to_euler(drv.matrix, rotate_order=drv.ro)))  # Plug("decomposeMatrix2.outputRotate")
print(m.to_quaternion(drv.matrix) >> None)                # [0.         0.70710678 0.         0.70710678]
```

On 2024+ the one-channel extractors are single native nodes:

```python
for name in ("translation", "rotation", "scale_of"):
    out = getattr(m, name)(drv.matrix)
    print(f"{name:12s} {repr(out):32s} {kind(out):22s} {out >> None}")
# translation  Plug("translation1.output")      translationFromMatrix  [10.  0.  0.]
# rotation     Plug("rotation1.output")         rotationFromMatrix     [-0. 90.  0.]
# scale_of     Plug("scale_of1.output")         scaleFromMatrix        [3. 1. 1.]
```

`scale_of`, not `scale` — the name dodges `compose(scale=...)`.

Maya matrices are row-major: rows 0-2 are the basis vectors, row 3 is
the translation. `axis` gives a basis vector with its scale still in;
`row` / `column` give the raw 4-vectors.

```python
print(m.axis(drv.matrix, "x") >> None)                                                         # [ 0.  0. -3.] -- +X after a 90 about Y, times scale 3
print(m.axis(drv.matrix, 2) >> None)                                                           # [1. 0. 0.]
print(m.row(drv.matrix, 3) >> None)                                                            # [10.  0.  0.  1.] -- translation lives in row 3
print(m.column(drv.matrix, 3) >> None)                                                         # [0. 0. 0. 1.]    -- column 3 is the homogeneous column
print(kind(m.axis(drv.matrix, 0)), kind(m.row(drv.matrix, 0)), kind(m.column(drv.matrix, 0)))  # axisFromMatrix rowFromMatrix columnFromMatrix
```

`determinant` and `dist` (translation to translation) round it off:

```python
origin = Node.create("transform", name="origin")
print(kind(m.determinant(drv.matrix)), m.determinant(drv.matrix) >> None)                  # determinant 3.0
print(kind(m.dist(origin.matrix, drv.matrix)), m.dist(origin.matrix, drv.matrix) >> None)  # distanceBetween 10.0
```

| Function | Maya 2024+ | Older Maya |
|---|---|---|
| `decompose` / `to_euler` / `to_quaternion` | `decomposeMatrix` | same |
| `translation` / `rotation` / `scale_of` | `translationFromMatrix` / `rotationFromMatrix` / `scaleFromMatrix` | `decomposeMatrix` outputs |
| `axis` | `axisFromMatrix` | `vectorProduct` op 3 |
| `row` | `rowFromMatrix` | `axis` (rows 0-2) or `translation` (row 3), literal index only |
| `column` | `columnFromMatrix` | raises `RuntimeError` |
| `determinant` | `determinant` | expanded 3x3 from three `pointMatrixMult` |
| `dist` | `distanceBetween` on `inMatrix1` / `inMatrix2` | same |

---

## 23. `matrix` — build and transform

`compose` is a `composeMatrix`; `rotate=` takes a 3-channel euler or a
4-channel quaternion and routes it to the right input.

```python
built = m.compose(translate=drv.t, rotate=drv.r, scale=drv.s, rotate_order=drv.ro)
print(repr(built), kind(built), m.rotation(built) >> None)  # Plug("composeMatrix1.outputMatrix") composeMatrix [-0. 90.  0.]
print(kind(m.compose(rotate=to_quaternion(drv.r))))         # composeMatrix -- four channels go to inputQuat
```

`fourbyfour` fills rows from vectors (identity for anything omitted);
`aim` is an `aimMatrix` — an aim constraint as a matrix, and the only
vector-to-orientation primitive. Convert its output with `to_euler` /
`to_quaternion`.

```python
basis = m.fourbyfour(x=[0, 0, -1], z=[1, 0, 0], position=[5, 5, 5])
print(repr(basis), kind(basis), m.translation(basis) >> None)   # Plug("fourbyfour2.output") fourByFourMatrix [5. 5. 5.]

fwd = Node.create("transform", name="fwd")
fwd.t << (0, 0, -1)
orient = m.aim(fwd.t, [0, 1, 0])                  # aim_axis=X, up_axis=Y by default
print(repr(orient), kind(orient), to_euler(orient) >> None)     # Plug("matrix_aim2.outputMatrix") aimMatrix [-0. 90.  0.]
```

Arithmetic:

```python
print(kind(m.multiply(drv.matrix, drv.matrix)), m.translation(m.multiply(drv.matrix, drv.matrix)) >> None)  # multMatrix [ 10.   0. -30.]
print(kind(m.add(drv.matrix, drv.matrix)))                                                                  # addMatrix
print(kind(m.add(drv.matrix, drv.matrix, weights=[0.25, 0.75])))                                            # wtAddMatrix
inv = m.inverse(drv.matrix)
print(repr(inv), kind(inv), m.translation(inv) >> None)                             # Plug("inverseMatrix1.outputMatrix") inverseMatrix [ -0.  -0. -10.]
print(kind(m.transpose(drv.matrix)), m.column(m.transpose(drv.matrix), 3) >> None)  # transposeMatrix [10.  0.  0.  1.]
rigid = m.normalize(drv.matrix)
print(kind(rigid), m.scale_of(rigid) >> None, m.translation(rigid) >> None)   # composeMatrix [1. 1. 1.] [10.  0.  0.] -- scale and shear dropped
```

Moving a point through a matrix, with and without its translation:

```python
pnt = Node.create("transform", name="pnt")
pnt.t << (1, 0, 0)
print(kind(m.transform_point(pnt.t, drv.matrix)),  m.transform_point(pnt.t, drv.matrix) >> None)   # multiplyPointByMatrix  [10.  0. -3.]
print(kind(m.transform_vector(pnt.t, drv.matrix)), m.transform_vector(pnt.t, drv.matrix) >> None)  # multiplyVectorByMatrix [ 0.  0. -3.]
print(kind(m.multiply(pnt.t, drv.matrix)))                                                         # multiplyPointByMatrix -- two args, one a vector: a point transform
```

| Function | Maya 2024+ | Older Maya |
|---|---|---|
| `compose` / `fourbyfour` / `aim` | `composeMatrix` / `fourByFourMatrix` / `aimMatrix` | same |
| `multiply` | `multMatrix`; vector x matrix → `multiplyPointByMatrix` | `multMatrix` / `pointMatrixMult` |
| `add` | `addMatrix`, or `wtAddMatrix` with `weights=` | same |
| `inverse` / `transpose` | `inverseMatrix` / `transposeMatrix` | same |
| `normalize` | `decomposeMatrix` → `composeMatrix` (rotate + translate only) | same |
| `transform_point` | `multiplyPointByMatrix` | `pointMatrixMult` |
| `transform_vector` | `multiplyVectorByMatrix` | `vectorProduct` op 3 |

---

## 24. `matrix` — blend

Four ways to get from one matrix to another. Pick by what you want
interpolated.

```python
rest = Node.create("transform", name="rest")      # identity

def trs(mtx):
    """(translate, rotate, scale) of a matrix plug, rounded for reading."""
    return tuple((fn(mtx) >> None).round(3) for fn in (m.translation, m.rotation, m.scale_of))


for name in ("lerp", "blend", "slerp"):
    out = getattr(m, name)(rest.matrix, drv.matrix, 0.5)
    print(f"{name:6s} {kind(out):13s}", *trs(out))
half = m.pow(drv.matrix, 0.5)
print(f"pow    {kind(half):13s}", *trs(half))
# lerp   wtAddMatrix   [5. 0. 0.] [ 0.    71.565  0.   ] [1.581 1.    0.632]   -- raw elements, not a transform
# blend  blendMatrix   [5. 0. 0.] [ 0. 45.  0.] [2. 1. 1.]
# slerp  composeMatrix [0. 0. 0.] [ 0. 45.  0.] [1. 1. 1.]
# pow    blendMatrix   [5. 0. 0.] [ 0. 45.  0.] [2. 1. 1.]
```

| Function | Interpolates | Builds | Use it for |
|---|---|---|---|
| `lerp(m0, m1, w)` | the 16 elements, `[1-w, w]` weights | `wtAddMatrix` | additive blends of matrices you know are compatible |
| `blend(m0, m1, w)` | translate and scale linearly, rotation by quaternion slerp, shear natively | `blendMatrix` | a full transform blend — the usual answer |
| `slerp(m0, m1, w)` | rotation **only**; returns a pure rotation matrix | `decomposeMatrix` → `quatSlerp` → `composeMatrix` | orientation blends |
| `pow(m, w)` | `blend(identity, m, w)` | `blendMatrix` with its `inputMatrix` left at identity | a fraction of a transform |

None of them clamp: `w < 0` and `w > 1` extrapolate.

---

## 25. `vector`

`X`, `Y`, `Z` are the unit axes as tuples. The products are the 2024+
`dotProduct` / `crossProduct` nodes unless you ask for `normalize=True`,
which only the legacy `vectorProduct` offers.

```python
ex = Node.create("transform", name="ex")
ey = Node.create("transform", name="ey")
ex.t << v.X
ey.t << v.Y
print(v.X, v.Y, v.Z)                                            # (1, 0, 0) (0, 1, 0) (0, 0, 1)

d = v.dot(ex.t, ey.t);   print(repr(d), kind(d), d >> None)      # Plug("dot1.output") dotProduct 0.0
c = v.cross(ex.t, ey.t); print(repr(c), kind(c), c >> None)      # Plug("cross1.output") crossProduct [0. 0. 1.]
print(repr(v.dot(ex.t, ey.t, normalize=True)), kind(v.dot(ex.t, ey.t, normalize=True)))    # Plug("dot2.outputX") vectorProduct
print(kind(v.triple_product(ex.t, ey.t, v.Z)), v.triple_product(ex.t, ey.t, v.Z) >> None)  # determinant 1.0
```

Length, unit and distance. A matrix given to `length` or `dist` means
its translation.

```python
leg = Node.create("transform", name="leg")
leg.t << (3, 4, 0)
print(kind(v.length(leg.t)),      v.length(leg.t) >> None)                # length 5.0
print(kind(v.normalize(leg.t)),   v.normalize(leg.t) >> None)             # normalize [0.6 0.8 0. ]
print(kind(v.dist(ex.t, leg.t)),  round(v.dist(ex.t, leg.t) >> None, 4))  # distanceBetween 4.4721
print(kind(v.length(drv.matrix)), v.length(drv.matrix) >> None)           # distanceBetween 10.0
print(v.length([3, 4, 0]))                                                # 5.0 -- literal folds
```

`angle` is radians, `angle_degrees` is the `atan2` node's own
`doubleAngle` output; `rotate` spins a vector by an euler.

```python
print(kind(v.angle(ex.t, ey.t)),         round(v.angle(ex.t, ey.t) >> None, 6))  # multiply 1.570796
print(kind(v.angle_degrees(ex.t, ey.t)), v.angle_degrees(ex.t, ey.t) >> None)    # atan2 90.0

spin = Node.create("transform", name="spin")
spin.r << (0, 0, 90)
print(kind(v.rotate(ex.t, spin.r)), v.rotate(ex.t, spin.r) >> None)   # rotateVector [0. 1. 0.]
```

`lerp` is linear, `slerp` walks the arc between the two vectors (and
blends their magnitudes), `elerp` is the geometric blend for scale.

```python
p0 = Node.create("transform", name="p0")
p1 = Node.create("transform", name="p1")
p1.t << (10, 20, 30)
out = v.lerp(p0.t, p1.t, 0.25)
print(repr(out), kind(out), out >> None)                                     # Plug("lerp_out1.value") network [2.5 5.  7.5]
print(kind(v.lerp(p0.tx, p1.tx, 0.25)), v.lerp(p0.tx, p1.tx, 0.25) >> None)  # lerp 2.5 -- scalars get the native node
print(v.slerp([0, 0, 1], [1, 0, 0], 0.5) >> None)                            # [0.70710677 0.         0.70710677] -- on the arc, not the chord

p0.tx << 2.0
p1.tx << 8.0
print(kind(v.elerp(p0.tx, p1.tx, 0.5)), round(v.elerp(p0.tx, p1.tx, 0.5) >> None, 6))  # multiply 4.0
print(v.lerp(0.0, 10.0, 0.5), round(v.elerp(2.0, 8.0, 0.5), 6))                        # 5.0 4.0 -- literals fold
```

`slerp` never takes the antipodal shortcut a quaternion would past 90
degrees, and falls back to `lerp` when the inputs are parallel or
opposite, so the result is always finite.

| Function | Maya 2024+ | Older Maya |
|---|---|---|
| `dot` / `cross` | `dotProduct` / `crossProduct` | `vectorProduct` op 1 / 2 (also for `normalize=True`) |
| `length` | `length` | `distanceBetween` (always, for a matrix) |
| `normalize` | `normalize` | `v / length(v)` with divide-by-zero quieting |
| `dist` | `distanceBetween` | same |
| `angle` / `angle_degrees` | `atan2(|a x b|, a . b)` through the nodes above | same |
| `rotate` | `rotateVector` | `composeMatrix` + `vectorProduct` op 3 |
| `triple_product` | `fourByFourMatrix` + `determinant` | expanded 3x3 |
| `lerp` | `lerp` for scalars; per-channel `blendWeighted` + a `network` gather for vectors | `blendWeighted` |
| `slerp` | `angleBetween` + `sin` + `condition` | same |
| `elerp` | `a ** (1-w) * b ** w` | same |

---

## 26. `quaternion`

The `quat*` nodes come from the `quatNodes` plugin on every Maya
version; the module loads it for you. Quaternions are `(x, y, z, w)`,
scalar last.

```python
qa = to_quaternion(p0.r)    # identity -- p0 is unrotated
qb = to_quaternion(spin.r)  # 90 degrees about Z
print(repr(qb), kind(qb), (qb >> None).round(4))    # Plug("eulerToQuat3.outputQuat") eulerToQuat [0.     0.     0.7071 0.7071]

for name in ("add", "multiply", "subtract"):
    out = getattr(q, name)(qb, qb)
    print(f"{name:9s} {repr(out):30s} {kind(out)}")
# add       Plug("quatAdd1.outputQuat")    quatAdd
# multiply  Plug("quatProd1.outputQuat")   quatProd
# subtract  Plug("quatSub1.outputQuat")    quatSub
print(to_euler(q.multiply(qb, qb)) >> None)     # [  0.  -0. 180.] -- two 90s

for name in ("negate", "normalize", "inverse", "conjugate"):
    out = getattr(q, name)(qb)
    print(f"{name:10s} {kind(out):14s} {(out >> None).round(4)}")
# negate     quatNegate     [-0.     -0.     -0.7071 -0.7071]
# normalize  quatNormalize  [0.     0.     0.7071 0.7071]
# inverse    quatInvert     [-0.     -0.     -0.7071  0.7071]
# conjugate  quatConjugate  [-0.     -0.     -0.7071  0.7071]
```

The shortest-arc angle between two quaternions, and the conversions
out:

```python
print(kind(q.angle(qa, qb)), round(q.angle(qa, qb) >> None, 6))                      # multiply 1.570796
print(kind(q.angle_degrees(qa, qb)), q.angle_degrees(qa, qb) >> None)                # multiply 90.0
print(repr(q.to_euler(qb, rotate_order=3)), q.to_euler(qb, rotate_order=3) >> None)  # Plug("quatToEuler2.outputRotate") [ 0.  0. 90.]
print(kind(q.to_matrix(qb)), m.rotation(q.to_matrix(qb)) >> None)                    # composeMatrix [ 0. -0. 90.]
print(kind(q.to_vector(qb)), (q.to_vector(qb) >> None).round(4))                     # network [0.     0.     0.7071]
```

`slerp` is a `quatSlerp`; `pow` is the same node slerping from identity,
so it is the true power of a unit quaternion and extrapolates.

```python
print(kind(q.slerp(qa, qb, 0.5)), to_euler(q.slerp(qa, qb, 0.5)) >> None)  # quatSlerp [ 0. -0. 45.]
print(kind(q.pow(qb, 0.5)), to_euler(q.pow(qb, 0.5)) >> None)              # quatSlerp [ 0. -0. 45.]
print(to_euler(q.pow(qb, 2.0)) >> None)                                    # [  0.  -0. 180.]
```

Axis-angle both ways. The angle plug is a `doubleAngle`, so a literal
or a unitless plug is read in the scene's angle unit — degrees by
default: pass `90`, not `math.pi / 2`, or feed it an angle plug and the
units take care of themselves.

```python
ax = q.from_axis_angle([0, 0, 1], 90)
print(repr(ax), kind(ax), (ax >> None).round(4))   # Plug("from_axis_angle2.outputQuat") axisAngleToQuat [0.     0.     0.7071 0.7071]

axis_plug, angle_plug = q.to_axis_angle(qb)     # a 2-tuple of plugs
print(repr(axis_plug), repr(angle_plug))      # Plug("to_axis_angle2.outputAxis") Plug("to_axis_angle2.outputAngle")
print(axis_plug >> None, angle_plug >> None)  # [0. 0. 1.] 90.0
```

---

## 27. `euler`

Rotate orders are Maya's: `XYZ=0 YZX=1 ZXY=2 XZY=3 YXZ=4 ZYX=5`.
`reorder` needs **both** orders, positionally; it goes through
quaternion space, so what comes back is a `quatToEuler`.

```python
tilt = Node.create("transform", name="tilt")
tilt.r << (30, 45, 60)
zyx = e.reorder(tilt.r, 0, 5)                  # XYZ -> ZYX
print(repr(zyx), kind(zyx), (zyx >> None).round(3))                                              # Plug("quatToEuler6.outputRotate") quatToEuler [-24.597  47.663  58.334]
print(e.reorder(zyx, 5, 0) >> None)                                                              # [30. 45. 60.] -- round trip

print(kind(e.to_matrix(tilt.r, rotate_order=tilt.ro)), m.rotation(e.to_matrix(tilt.r)) >> None)  # composeMatrix [30. 45. 60.]
print(kind(e.to_quaternion(spin.r)), (e.to_quaternion(spin.r) >> None).round(4))                 # eulerToQuat [0.     0.     0.7071 0.7071]
print(kind(e.slerp(p0.r, spin.r, 0.5)), e.slerp(p0.r, spin.r, 0.5) >> None)                      # quatToEuler [ 0. -0. 45.] -- shortest arc, XYZ out
```

There is no `euler.to_euler`: the module converts *from* eulers. To
land somewhere as an euler, call `to_euler` on a quaternion or a matrix.

---

## 28. `interpolate`

`sequence(x, xp, yp)` samples a piecewise curve: `searchsorted` picks
the segment, `choice` nodes pull its ends, and `method` (default
`rig.lerp`) blends them. `xp` and `yp` are lists of scalars. Past
either end the end segment keeps going — clamp `x` first if you want a
hold.

```python
xs = Node.create("transform", name="xs")
xs.tx << 2.5
knots = [0.0, 1.0, 2.0, 3.0]
seq   = ip.sequence(xs.tx, knots, [10.0, 20.0, 30.0, 40.0])
print(repr(seq), kind(seq), seq >> None)                                                    # Plug("lerp6.output") lerp 35.0
print(round(ip.sequence(xs.tx, knots, [10.0, 20.0, 30.0, 40.0], method=elerp) >> None, 4))  # 34.641 -- geometric mean of 30 and 40

eased = ip.sequence(xs.tx, knots, [10.0, 20.0, 30.0, 40.0],
                    method=lambda y0, y1, weight: lerp(y0, y1, tw.in_out_quad(weight)))
xs.tx << 2.25
print(eased >> None)                          # 31.25 -- in_out_quad(0.25) is 0.125
xs.tx << 5.0
print(seq >> None)                            # 60.0 -- extrapolated past the last knot
```

`method` is any callable with the shape `method(y0, y1, weight=w)`:
`rig.slerp` for rotations, `rig.elerp` for scale, or a lambda that
runs the weight through a `tween` first.

`smoothstep` / `smootherstep` take the two edges and the sample
position as `weight`; `inverse_lerp` is the `t` that `lerp` would need.

```python
wt = Node.create("transform", name="wt")
wt.tx << 0.25
print(kind(ip.smoothstep(0.0, 1.0, weight=wt.tx)),   ip.smoothstep(0.0, 1.0, weight=wt.tx) >> None)    # smoothStep 0.15625
print(kind(ip.smootherstep(0.0, 1.0, weight=wt.tx)), ip.smootherstep(0.0, 1.0, weight=wt.tx) >> None)  # sum 0.103515625
print(ip.smoothstep(0.0, 1.0, weight=0.25),          ip.smootherstep(0.0, 1.0, weight=0.25))           # 0.15625 0.103515625 -- literals fold

wt.tx << 2.5
print(kind(ip.inverse_lerp(0.0, 10.0, wt.tx)), ip.inverse_lerp(0.0, 10.0, wt.tx) >> None)   # inverseLerp 0.25
```

| Function | Maya 2024+ | Older Maya |
|---|---|---|
| `sequence` | `condition` ladder + `choice` + `method` | same |
| `smoothstep` | `smoothStep` when all inputs are scalar | `clamp` + arithmetic (also for vector inputs on any version) |
| `smootherstep` | `clamp` + arithmetic | same |
| `inverse_lerp` | `inverseLerp` when all inputs are scalar | `(x - a) / (b - a)` |

---

## 29. `tween`

42 easing curves. Each takes one normalised `t` and returns the eased
plug.

| Family | Curves |
|---|---|
| linear | `in_linear` `out_linear` |
| quadratic `t^2` | `in_quad` `out_quad` `in_out_quad` `out_in_quad` |
| cubic `t^3` | `in_cubic` `out_cubic` `in_out_cubic` `out_in_cubic` |
| quartic `t^4` | `in_quart` `out_quart` `in_out_quart` `out_in_quart` |
| quintic `t^5` | `in_quint` `out_quint` `in_out_quint` `out_in_quint` |
| sinusoidal | `in_sine` `out_sine` `in_out_sine` `out_in_sine` |
| exponential `2^t` | `in_expo` `out_expo` `in_out_expo` `out_in_expo` |
| circular | `in_circ` `out_circ` `in_out_circ` `out_in_circ` |
| elastic (decaying sine) | `in_elastic` `out_elastic` `in_out_elastic` `out_in_elastic` |
| back (overshooting cubic) | `in_back` `out_back` `in_out_back` `out_in_back` |
| bounce | `in_bounce` `out_bounce` `in_out_bounce` `out_in_bounce` |

`in_*` accelerates from rest, `out_*` decelerates to rest, `in_out_*`
does both around the midpoint and `out_in_*` is its mirror.

```python
tt = Node.create("transform", name="tt")
tt.tx << 0.5

built = {name: getattr(tw, name)(tt.tx) for name in tw.__all__}
print(len(built), all(isinstance(plug, Plug) for plug in built.values()))   # 42 True

for name in ("in_quad", "out_quad", "in_out_quad", "out_in_quad"):
    print(f"{name:12s} {repr(built[name]):34s} {kind(built[name]):10s} {built[name] >> None}")
# in_quad      Plug("pow4.output")                power      0.25
# out_quad     Plug("mul17.output")               multiply   0.75
# in_out_quad  Plug("condition17.outColorR")      condition  0.5
# out_in_quad  Plug("in_out_quad1_condition2.outColorR") condition  0.5

families = ("linear", "quad", "cubic", "quart", "quint", "sine", "expo", "circ", "elastic", "back", "bounce")
print({fam: round(built["in_" + fam] >> None, 4) for fam in families})
# {'linear': 0.5, 'quad': 0.25, 'cubic': 0.125, 'quart': 0.0625, 'quint': 0.0312, 'sine': 0.2929,
#  'expo': 0.0312, 'circ': 0.134, 'elastic': -0.0156, 'back': -0.0877, 'bounce': 0.2344}
```

Nothing is clamped: `t` outside `[0, 1]` extrapolates, which is what
anticipation and follow-through want from `back` and `elastic`. Clamp
the input yourself when you want a hold. Tweens are also the one
library that builds nodes for a literal.

```python
tt.tx << 1.2
print(round(built["in_back"] >> None, 4), built["in_quad"] >> None)  # 2.2181 1.44
print(tw.in_quad(f.clamp(tt.tx, 0, 1)) >> None)                      # 1.0
print(repr(tw.in_quad(0.5)))                                         # Plug("pow17.output") -- a node, not 0.25
```

Every tween is built from `functions` and `trigonometry`, so the node
mix is `power` / `multiply` / `sum` / `sin` / `cos` / `condition` on
2024+ and the `multiplyDivide` / `plusMinusAverage` equivalents before.

---

## 30. `random`

A linear congruential generator as a self-feeding cycle in the DG.
`value()` is a scalar in `[0, 1)` that re-rolls every frame (its
default trigger is `frame()`) or whenever the `trigger` plug you pass
changes. An explicit `seed` memoises the call; no seed means a fresh,
independent stream every call.

```python
cmds.currentTime(1)
rnd = r.value(seed=42)
print(repr(rnd), kind(rnd))               # Plug("div9.output") divide
print(0.0 <= (rnd >> None) < 1.0)         # True
print(str(r.value(seed=42)) == str(rnd))  # True  -- explicit seed: one network
print(str(r.value()) == str(r.value()))   # False -- no seed: a new stream each call

samples = []
for frame in (2, 3, 4):
    cmds.currentTime(frame)
    samples.append(rnd >> None)
print(len(set(samples)))                          # 3 -- a new value per frame

trg    = Node.create("transform", name="trg")
seeded = r.value(trigger=trg.tx, seed=7)
before = seeded >> None
trg.tx << 1.0
print(before != (seeded >> None))                 # True -- re-rolls when the trigger changes
```

`uniform` / `randint` remap it; the `3D` variants run three independent
LCGs, one per channel, and take a **three-element** `seed`.

```python
cmds.currentTime(1)
u  = r.uniform(5, 15, seed=42)
i  = r.randint(1, 10, seed=42)
v3 = r.value3D(seed=[1, 2, 3])
u3 = r.uniform3D([0, 0, 0], [10, 10, 10], seed=[1, 2, 3])
i3 = r.randint3D([0, 0, 0], [10, 10, 10], seed=[1, 2, 3])

print(repr(u),  kind(u),  5 <= (u >> None) < 15)          # Plug("add34.output") sum True
print(repr(i),  kind(i),  float(i >> None).is_integer())  # Plug("constant7.value") network True
print(repr(v3), kind(v3), (v3 >> None).shape)             # Plug("constant8.value") network (3,)
print(repr(u3), kind(u3), (u3 >> None).shape)             # Plug("add35.output3D") plusMinusAverage (3,)
print(repr(i3), kind(i3), (i3 >> None).dtype)             # Plug("constant9.value") network int32
```

The cycle is intentional and evaluates correctly, but Maya will print
its usual cycle warning when a scene holding one loads;
`cmds.cycleCheck(e=False)` silences it. The stdlib `random` is
untouched — this module is only ever `rig.random`.

---

## 31. The cross-type verbs at `rig.*`

Twelve verbs classify their first argument (scalar, vector, matrix,
quaternion, euler) and call the matching per-type function from the
sections above. Use them when the operand type is not fixed at the call
site; call `matrix.slerp` / `vector.lerp` / … directly when it is.

| Verb | scalar | vector | matrix | quaternion | euler |
|---|---|---|---|---|---|
| `dist` | — | `distanceBetween` | `distanceBetween` | — | — |
| `lerp` | `lerp` | `blendWeighted` + gather | `wtAddMatrix` | rejected | rejected |
| `slerp` | — | arc network | `composeMatrix` (rotation only) | `quatSlerp` | `quatToEuler` |
| `blend` | — | — | `blendMatrix` | — | — |
| `elerp` | `power` / `multiply` | `power` / `multiply` | — | rejected | rejected |
| `normalize` | — | `normalize` | `composeMatrix` (orthonormalise) | `quatNormalize` | — |
| `inverse` | — | — | `inverseMatrix` | `quatInvert` | — |
| `angle` / `angle_degrees` | — | radians / degrees | — | radians / degrees | — |
| `to_euler` | — | — | `decomposeMatrix` | `quatToEuler` | — |
| `to_quaternion` | — | — | `decomposeMatrix` | — | `eulerToQuat` |
| `to_matrix` | — | rejected, use `matrix.aim` | — | `composeMatrix` | `composeMatrix` |

A dash is a `TypeError`; the rejected cells raise one with a hint.
`dist(vector, matrix)` mixes types inside one `distanceBetween`.

```python
print(kind(dist(p0.t, p1.t)), kind(dist(p0.matrix, p1.matrix)), kind(dist(p0.t, p1.matrix)))                          # distanceBetween distanceBetween distanceBetween
print(kind(lerp(p0.tx, p1.tx)), kind(lerp(p0.t, p1.t)), kind(lerp(p0.matrix, p1.matrix)))                             # lerp network wtAddMatrix
print(kind(slerp(p0.t, p1.t)), kind(slerp(qa, qb)), kind(slerp(p0.r, spin.r)), kind(slerp(rest.matrix, drv.matrix)))  # condition quatSlerp quatToEuler composeMatrix
print(kind(blend(rest.matrix, drv.matrix)), kind(elerp(p0.tx, p1.tx)))                                                # blendMatrix multiply
print(kind(normalize(leg.t)), kind(normalize(qb)), kind(normalize(drv.matrix)))                                       # normalize quatNormalize composeMatrix
print(kind(inverse(drv.matrix)), kind(inverse(qb)))                                                                   # inverseMatrix quatInvert
print(kind(angle(ex.t, ey.t)), kind(angle_degrees(ex.t, ey.t)), kind(angle(qa, qb)), kind(angle_degrees(qa, qb)))     # multiply atan2 multiply multiply
print(repr(to_euler(qb)),          repr(to_euler(drv.matrix)))                                                        # Plug("quatToEuler10.outputRotate") Plug("decomposeMatrix1.outputRotate")
print(repr(to_quaternion(spin.r)), repr(to_quaternion(drv.matrix)))                                                   # Plug("eulerToQuat3.outputQuat") Plug("decomposeMatrix1.outputQuat")
print(repr(to_matrix(spin.r)),     repr(to_matrix(qb)))                                                               # Plug("composeMatrix9.outputMatrix") Plug("composeMatrix5.outputMatrix")
```

`slerp` on a matrix is orientation only; `blend` is the full
transform. `lerp` and `elerp` refuse rotations on purpose:

```python
for call in (lambda: lerp(qa, qb), lambda: to_matrix(p0.t), lambda: dist(1.0, 2.0), lambda: blend(p0.t, p1.t)):
    try:
        call()
    except TypeError as err:
        print(err)
# lerp() does not support quaternion operands; use slerp() for rotations
# to_matrix() does not support vector operands; build an orientation from aim/up vectors with matrix.aim()
# dist() does not support scalar operands
# blend() does not support vector operands
```

A `PlugList` operand is classified by its first element and broadcast:

```python
out = lerp(PlugList([p0.t, p1.t]), PlugList([p1.t, p0.t]), weight=0.5)
print(repr(out), len(out))     # PlugList([Plug("lerp_out2.value"), Plug("lerp_out3.value")]) 2
```

---

## 32. The membership grammar

Everything below builds its own geometry; `make()` wraps a `maya.cmds`
creator's transform as a `Node`, `members()` reads a shading engine's
membership the way `cmds.sets` prints it.

```python
cmds.file(new=True, force=True)

from rig import Components, Tag, Layer, container, lock, shade
from rig.shade import Blinn, Lambert, Phong, Material, Default
from rig.bridges import nodes as rn


def make(command, name, **kwargs):
    """Run a maya.cmds creator and wrap its transform as a Node."""
    return Node(command(name=name, **kwargs)[0])


def members(engine):
    """A shading engine's membership, as cmds.sets prints it."""
    return cmds.sets(str(engine), query=True) or []


sph = make(cmds.polySphere, "sph")      # a polySphere ships no procedural tags: safe for every example
print(sph, sph.f.count, sph.vtx)        # sph 400 sphShape.controlPoints
```

Three collection kinds ship: `Tag` (component tags, per geometry
node), `Layer` (display layers) and the materials of `rig.shade`
(`Blinn`, `Lambert`, `Phong`, …). Every kind uses the same four operator
spellings; everything else is a method.

| Verb | Spelling | `Tag` | Material (`Blinn` / `Lambert` / …) | `Layer` |
|---|---|---|---|---|
| add | `lhs << Spec('x')` | members into tag `x` (created with them when missing); a **node** on the left creates the tag itself, empty | move into `x`'s shading engine; material + engine built on first use (exclusive) | move the node into layer `x` (exclusive) |
| remove these | `lhs << -Spec('x')` | members out of `x`; the tag survives; a node on the left **deletes** the tag | leave `x`'s engine: those faces are in **no** engine (green) | back to `defaultLayer`; a no-op when the node is in another layer |
| purge | `lhs << Spec()` (== `Spec(None)`) | out of every editable tag of that category; a node on the left deletes every editable tag | out of every engine (green); `Default()` reverts | `defaultLayer` |
| query | `lhs >> Spec('x')` | ndarray of the LHS ids in the tag; a node reads the whole tag, in its own category | ndarray of the LHS face ids wearing `x`; all faces when object-level | `bool` |
| enumerate | `lhs >> Spec()` == `Spec.of(lhs)` | `[Tag('a'), …]` holding the LHS | `[Blinn('red'), …]` typed by the live node type | `Layer('x')` or `None` (`Layer.of` gives `[]` / `[Layer('x')]`) |
| methods | | `.set(members)` `.clear(node)` `.rename(node, new)` `.delete(node)` | `.rename(new)` `.delete()` `.astype(type)` | `.clear()` `.rename(new)` `.delete()` |

Two rules hold for every kind. `<<` returns what the next `<<` should
target: an attribute spec returns the new plug (a value goes next), a
collection spec returns the **left-hand side** unchanged, so kinds chain.
And an **attribute plug on the left stands for its node** (`sph.tx <<
Tag('x')` is `sph << Tag('x')`); component plugs keep their own meaning.

```python
faces  = sph.f[:3]
result = faces << Tag("lid") << Tag("rim")      # every collection << returns the LHS
print(result is faces)                                          # True
print(sph >> Tag("lid"))                                        # [0 1 2]   a query is a plain value, like plug >> None
print(sph.tx >> Tag())                                          # [Tag('lid'), Tag('rim')]   an attribute plug stands for its node

print(sph << Blinn("red") << Layer("geometry") is sph)          # True   a node on the left: its shapes wear red, it sits in 'geometry'
print((sph >> Blinn("red")).size, sph.ty >> Layer("geometry"))  # 400 True
```

A **missing** collection is a `ValueError` on `>>`, `-Spec` and the
methods, never an empty answer; a present collection holding nothing of
the LHS is an empty result. The rejected spellings are `TypeError`s that
write nothing: members never ride on the spec, `-Spec()` is a double
negative, `~Spec('x')` is unassigned, and `<< None` is never a clear (on a
plug it disconnects).

```python
try:
    sph >> Tag("nope")
except ValueError as err:
    print(err)                                  # no component tag 'nope' on sphShape

for bad in (
    lambda: Tag("xx", [0, 1, 2]),               # members belong on the left
    lambda: -Tag(),                             # double negative
    lambda: ~Tag("xx"),                         # unassigned
    lambda: sph.f[:2] << Tag("zz") << None,     # the first << ran; '<< None' refuses
):
    try:
        bad()
    except TypeError as err:
        print(type(err).__name__)
```

---

## 33. Components as members

Section 10 covers the `Components` carrier itself (`.indices`, `.count`,
`.names`, `>> None`, the explicit constructor). On the left of a membership
operator the bare handle means the whole kind: `sph.f` is every face, and
`sph.vtx`, the bare `controlPoints` multi, means **all points** at no cost,
while `sph.vtx[:3]` is a `PlugList` of element plugs. Surfaces index as
`srf.cv[u, v]`, lattices as `lat.pt[s, t, u]`.

```python
print(sph.f, sph.f.is_all, sph.f[:3])   # Components("|sph|sphShape.f[*]") True Components("|sph|sphShape.f[0:2]")
print(repr(sph.vtx), len(sph.vtx[:3]))  # Plug("sphShape.controlPoints") 3
```

`Components` has no `__len__` and no `__iter__` on purpose: a `PlugList`
keeps it as one opaque element and broadcasts it as a scalar, so a list of
selections pairs with a list of specs.

```python
PlugList([sph.f[:2], sph.f[2:4]]) << [Tag("aa"), Tag("bb")]
print(sph >> Tag("aa"), sph >> Tag("bb"))       # [0 1] [2 3]
```

---

## 34. Tag

A tag lives on one geometry node — its injection node: the shape, or the
Orig once a deformer exists — and holds **one category** (vertices / CVs /
lattice points, edges, or faces). Deformers read tags by name through
`input[i].componentTagExpression`; Maya 2025 creates no deformer sets.

```python
cmds.file(new=True, force=True)
sph = make(cmds.polySphere, "sph")

sph.vtx[:8]   << Tag("cap")  # created WITH vtx[0:7] at the injection node (the shape here)
sph.vtx[3:10] << Tag("cap")  # add is a union
print(sph >> Tag("cap"))                        # [0 1 2 3 4 5 6 7 8 9]
sph << Tag("empty")                             # node on the left: the tag itself, created empty
print(sph >> Tag("empty"))                      # []
sph.f[:3]    << Tag("lid")                      # a face tag; sph.e[[0, 4]] << Tag('ed') makes an edge tag
print(sph >> Tag("lid"))                        # [0 1 2]   ids in the tag's own category
sph.vtx      << Tag("allv")                     # the bare handle is every vertex
print(len(sph >> Tag("allv")))                  # 382
```

One category per tag: faces into a vertex tag is a `TypeError` (raw Maya
returns `False` silently). `.set()` replaces the contents and may flip
the category.

```python
try:
    sph.f[:2] << Tag("cap")
except TypeError as err:
    print("vertex tag" in str(err))             # True
Tag("cap").set(sph.f[:2])                       # replace: 'cap' is a face tag now
print(sph >> Tag("cap"))                        # [0 1]
Tag("cap").set(sph.vtx[:8])                     # and back
```

Remove members, empty, delete, purge. An empty tag is a legal state (it
deforms nothing and silences a deformer's "Missing componentTags").

```python
sph.vtx[:2] << -Tag("cap")                      # remove these; the tag survives
print(sph >> Tag("cap"))                        # [2 3 4 5 6 7]
sph.vtx     << -Tag("cap")                      # empty it
print((sph >> Tag("cap")).shape)                # (0,)
sph << -Tag("empty")                    # node on the left: delete the tag

sph.vtx[:4]  << Tag("aa")
sph.vtx[2:6] << Tag("bb")
sph.vtx[:3]  << Tag()                           # out of every editable VERTEX tag; the face tag is untouched
print(sph >> Tag("aa"), sph >> Tag("bb"), sph >> Tag("lid"))   # [3] [3 4 5] [0 1 2]
sph          << Tag()                           # delete every editable tag on the node
print(Tag.of(sph))                              # []
```

Queries return plain arrays: the ids of the LHS that are in the tag, in
the tag's own category. `>> Tag()` is `Tag.of`, the tags holding the LHS.
The ids go back in through the handle by fancy indexing.

```python
sph.vtx[:8] << Tag("cap")
sph.f[:3]   << Tag("lid")
print(sph.vtx[4:12] >> Tag("cap"))                                 # [4 5 6 7]
print(sph.vtx[[7, 0, 9]] >> Tag("cap"))                            # [0 7]
print(sph >> Tag())                                                # [Tag('cap'), Tag('lid')]   == Tag.of(sph)
print(sph.vtx[3] >> Tag(), Tag.of(sph.f[1]), Tag.of(sph.vtx[20]))  # [Tag('cap')] [Tag('lid')] []
sph.vtx[sph >> Tag("cap")] << Tag("copy")       # round trip through the handle
print((sph >> Tag("copy")).size)                # 8

srf = make(cmds.sphere, "srf")                  # NURBS: cv[u, v]
srf.cv[1:3, 2:4] << Tag("rim")
print(srf >> Tag("rim"))                        # [[1 2] [1 3] [2 2] [2 3]]   (N, 2) on a surface, (N, 3) on a lattice
```

The methods take the node. A deformer's `componentTagExpression` plug is
the one string plug that receives a `Tag`'s **name** (the expression
sugar); expressions stay plain strings. Deleting or renaming a tag a
deformer references refuses without `force=True`; a forced rename rewrites
exact-token references.

```python
sph     = make(cmds.polySphere, "clustered", ch=False)
cluster = Node(cmds.cluster(str(sph))[0])       # the injection node is now the Orig shape
expr    = cluster.input[0].componentTagExpression
sph.vtx[:4] << Tag("cap")
expr        << Tag("cap")                              # writes 'cap'; warns when the tag is absent on the deformer's input
print(expr >> None)                             # cap
expr << "cap + lid"                             # expressions stay plain strings
try:
    Tag("cap").delete(sph)
except RuntimeError as err:
    print("force=True" in str(err))             # True   referenced by cluster1.input[0].componentTagExpression
Tag("cap").rename(sph, "crown", force=True)     # rewrites the reference
print(expr >> None)                             # crown + lid
Tag("crown").clear(sph)                         # empty, name survives
print((sph >> Tag("crown")).shape)              # (0,)
Tag("crown").delete(sph, force=True)            # sph << -Tag('crown', force=True) is the operator form
print(Tag.of(sph))                              # []
```

The `polyCube` trap: with history, a polyCube ships six **procedural**
face tags owned by `polyCube1`. They read fine and refuse edits; `at=` is
the consent to shadow one with an editable tag on the shape (raw Maya
would silently create a vertex shadow). A `polyCube(ch=False)` bakes them
into the shape, where they edit normally.

```python
cube = make(cmds.polyCube, "cube")              # WITH history
print(Tag.of(cube))        # [Tag('back'), Tag('bottom'), Tag('front'), Tag('left'), Tag('right'), Tag('top')]
print(cube >> Tag("top"))  # [1]
try:
    cube.f[:3] << Tag("top")
except TypeError as err:
    print("PROCEDURAL" in str(err))             # True   ... shadow it on purpose with Tag('top', at='cubeShape')
cube.f[:3] << Tag("top", at=cube)               # an editable shadow on cubeShape; later edits find it without at=
cube.f[3]  << Tag("top")
print(cube >> Tag("top"))                       # [0 1 2 3]
cube << -Tag("top")                             # delete the shadow: the procedural tag shows again
print(cube >> Tag("top"))                       # [1]

baked = make(cmds.polyCube, "baked", ch=False)
baked.f[:3] << Tag("top")                       # baked tags edit through the shape's own componentTags multi
print(baked >> Tag("top"))                      # [0 1 2]
```

---

## 35. Layer

Display layers hold **objects** — the node itself, never its subtree
(children draw with the parent's override through the DAG without
joining) — and a node is in exactly one. A component on the left is a
`TypeError` (Maya would silently store the shape). `defaultLayer` reads
as "no layer".

```python
cmds.file(new=True, force=True)
cube  = make(cmds.polyCube, "cube", ch=False)
other = make(cmds.polyCube, "other", ch=False)

cube << Layer("geometry")                              # find-or-create; returns cube
cube << Layer("ref", displayType=2, visibility=False)  # exclusive: cube leaves 'geometry'; kwargs are the layer's attributes on create
print(cube >> Layer("geometry"), cube >> Layer("ref"))  # False True
print(cube >> Layer())                                  # ref   the one Layer('ref'), or None
print(Layer.of(cube), Layer.of(other))                  # [Layer('ref')] []   defaultLayer is no layer
other << Layer("ref", visibility=True)          # found: kwargs skipped
print(Layer("ref").visibility >> None)          # False
other << Layer("ref", visibility=True, update=True)     # update=True re-asserts them
print(Layer("ref").visibility >> None)          # True
PlugList([cube, other]) << Layer("rig")         # one editDisplayLayerMembers call
```

```python
grp = Node(cmds.group(str(cube), name="grp"))
grp << Layer("bg", visibility=False)
print(cube >> Layer(), cube >> Layer("bg"))     # rig False   the child stays where it was, and draws hidden through grp
try:
    cube.f[:2] << Layer("bg")
except TypeError as err:
    print("layers hold objects" in str(err))    # True
cube.tx << Layer("bg")                          # an attribute plug stands for its node
print(cube >> Layer())                          # bg
```

`-Layer('x')` and `Layer()` both land in `defaultLayer`. The spec is the
find-only handle (`.node`, any attribute); `.delete()` sends the members
to `defaultLayer`, `.rename()` is followed by the spec, `.clear()` keeps
the layer. A new layer never joins the active rig container.

```python
cube  << -Layer("bg")                           # back to defaultLayer
print(cube >> Layer())                          # None
other << Layer()                                # the purge is defaultLayer too
print(Layer.of(other))                          # []

bg = Layer("bg")
print(bg.node, bg.visibility << False)  # bg bg.visibility   a Node and the Plug, for chaining
bg.rename("background")                 # the spec follows
print(bg, cmds.objExists("bg"))         # background False
Layer("rig").clear()                            # members to defaultLayer, the layer survives
Layer("background").delete()                    # members to defaultLayer, the layer is gone
print(cmds.ls(type="displayLayer"))             # ['defaultLayer', 'geometry', 'ref', 'rig']
```

---

## 36. Materials

A material spec is a lazy handle on a surface shader **and** its shading
engine. Zero Maya calls until it meets `<<`: then the name is found (exact,
then in the current namespace) or the whole network is built —
`red`, `redSG`, its materialInfo and the render-partition / shader-list
wiring — the kwargs are applied and the LHS is moved into the engine in one
`cmds.sets`. The classes are `Blinn`, `Lambert`, `Phong`, `PhongE`,
`SurfaceShader`, `StandardSurface`, `OpenPBRSurface`, the generic
`Material(name, type=...)`, and `Default()` for `initialShadingGroup`.

```python
cmds.file(new=True, force=True)
cube  = make(cmds.polyCube, "cube", ch=False)
other = make(cmds.polyCube, "other", ch=False)

red   = Blinn("red", color=(1, 0, 0))             # inert: zero Maya calls
print(cube << red)                             # cube   returns the LHS; builds red + redSG, sets color, assigns
print(cmds.nodeType("red"), members("redSG"))  # blinn ['cubeShape']
other << Blinn("red", color=(0, 0, 1))          # 'red' exists: a plain assignment, kwargs skipped
print(cmds.getAttr("red.color")[0])             # (1.0, 0.0, 0.0)
other << Blinn("red", color=(0, 0, 1), update=True)     # update=True re-asserts them
print(cmds.getAttr("red.color")[0])             # (0.0, 0.0, 1.0)
```

Kwargs are attribute injections, exactly like the `rig.bridges.nodes`
factories: a value sets, a plug connects, a spec (`lock`) applies. The
spec is the find-only handle afterwards.

```python
print(red.node, red.engine)    # red redSG   Nodes; a ValueError until built
print(red.color << (0, 1, 0))  # red.color   the Plug: forwards to the node; a typo raises and creates nothing
red.diffuse = 0.5                               # the same injection as sugar
print(red.diffuse >> None)                      # 0.5

tex = rn.file(name="tex")
cube << Blinn("skin", color=tex.outColor, diffuse=0.25, reflectivity=lock)
print(cmds.listConnections("skin.color", plugs=True), cmds.getAttr("skin.reflectivity", lock=True))   # ['tex.outColor'] True
```

A `PlugList` on the left is one material, one engine, one `cmds.sets`.
`unique=True` builds a fresh network per `<<`. A material is a shared,
scene-level asset: inside `with container():` a new network stays out of
the scope unless `container=True` (a per-asset look); the geometry never
joins.

```python
PlugList([cube, other.f[:2]]) << Lambert("both")
print(sorted(members("bothSG")))                # ['cubeShape', 'other.f[0:1]']   one engine holds both entries

spec = Lambert("plane", unique=True)
cube  << spec
other << spec
print(sorted(cmds.ls("plane*", type="lambert")))   # ['plane', 'plane1']

with container("look"):
    cube << Blinn("free")                    # stays out: a material is a scene asset
    cube << Blinn("inside", container=True)  # opts in: the network joins 'look'
print(cmds.container(query=True, findContainer=["free"]), cmds.container(query=True, findContainer=["inside"]))   # None look
```

Membership is exclusive and materials bind faces or whole objects (a
vertex, edge or UV on the left is a `TypeError`). Faces into an engine
that owns the whole shape carve it; faces into the engine that **already
owns their object** is the one permitted no-op of the grammar.

```python
cube       << red
cube.f[:3] << Lambert("decal")                  # faces 0-2 leave redSG for decalSG; rig carves the whole-shape membership
print(members("redSG"), members("decalSG"))     # ['cube.f[3:5]'] ['cube.f[0:2]']
other       << red
other.f[:2] << red                              # the one no-op: red already owns the whole object
print(sorted(members("redSG")))                 # ['cube.f[3:5]', 'otherShape']
```

`-Material('x')` leaves faces in **no** engine (green); `Material()` (any
subclass, same spec as `Material(None)`) does it for every engine;
`Default()` reverts to `initialShadingGroup` and `-Default()` leaves it.

```python
cube.f[[0]] << -Lambert("decal")                # face 0 is green
print(members("decalSG"), Material.of(cube.f[0]))    # ['cube.f[1:2]'] []
cube << Material()                              # green everywhere
print(Material.of(cube), shade.bindings(cube))  # [] []
cube << Default()                               # initialShadingGroup again
print(Material.of(cube))                        # [Default()]
cube << -Default()
print(Material.of(cube))                        # []
```

Queries read face ids; `>> Material()` / `>> Blinn()` enumerate like
`Material.of` / `Blinn.of`, typed by the live node type.

```python
cube       << Default()
cube.f[:3] << red
print(cube >> red, cube.f[1:5] >> red)                        # [0 1 2] [1 2]
print(cube >> Default(), other >> red)                        # [3 4 5] [0 1 2 3 4 5]   every face when object-level
print(cube >> Material(), cube >> Blinn(), Lambert.of(cube))  # [Default(), Blinn('red')] [Blinn('red')] []
print(cube.f[4] >> Material())                                # [Default()]
cube.f[cube >> red] << Lambert("decal")         # the ids go back through the handle
print(members("decalSG"))                       # ['cube.f[0:2]']
```

The module readers, `.delete()` / `.rename()`, and the two sweepers:
`shade.repair()` re-homes anything green to `initialShadingGroup`,
`shade.tidy()` collapses all-faces memberships to object level and
deletes orphan groupIds. Both are off the operator path.

```python
print(shade.materials(cube))                    # PlugList([Node("standardSurface1"), Node("decal")])
for material, faces in shade.bindings(cube):
    print(material, faces.indices)              # standardSurface1 [3 4 5] / decal [0 1 2]

Material("decal").delete()                      # material + engine + materialInfo; members go green
print(Material.of(cube.f[0]))  # []
print(shade.repair())          # PlugList([Node("cubeShape")])   the shapes re-homed
print(cube >> Default())       # [0 1 2 3 4 5]

cube.f[:3] << red
cube.f[3:] << red                               # every face, as face components
print(sorted(members("redSG")))  # ['cube.f[0:5]', 'otherShape']
shade.tidy()
print(sorted(members("redSG")))  # ['cubeShape', 'otherShape']

red.rename("crimson")            # the engine follows the <mat>SG convention, the spec follows the name
print(red, red.engine)           # crimson crimsonSG
for bad in (lambda: cube.vtx[:3] << red, lambda: cube << Lambert("crimson")):
    try:
        bad()
    except TypeError as err:
        print(str(err).split(";")[0])           # materials bind faces or whole objects / 'crimson' exists as a blinn
```

`Material.build()` creates the network with no target (a library look
built before any geometry exists); `Material('lambert1')` wraps any
existing surface shader, `Material('redSG')` the shader feeding an engine.
A typed class asserts the type: `Lambert('crimson')` on a blinn is a
`TypeError` pointing at `Material('crimson')`.

---

## 37. Shader conversion

`Phong(mat)` and `Material(mat, type='phong')` retype a **lazy** spec for
free and convert a **realised** material in the scene — the one
constructor in rig with a scene side effect. `mat` is retyped in place
and the call returns a fresh, equal handle. `mat.astype(...)` is the verb,
`shade.convert(...)` the engine that returns the `Conversion` report.

```python
cmds.file(new=True, force=True)
cube = make(cmds.polyCube, "cube", ch=False)

mat  = Blinn("red", color=(1, 0, 0))  # lazy
p    = Phong(mat)                     # a free retype: no node yet, nothing to lose
print(type(mat).__name__, p == mat, p is not mat)   # Phong True True
cube << mat
print(cmds.nodeType("red"))                                                    # phong
Material(mat, type="blinn")                                                    # realised: a scene conversion in one undo chunk; nothing set beyond color -> no warning
print(cmds.nodeType("red"), type(mat).__name__, mat.engine, members("redSG"))  # blinn Blinn redSG ['cubeShape']   name, engine, members kept

k = Phong(Blinn("k", eccentricity=0.6))         # lazy kwargs the target lacks are pruned with one warning
print(k.attrs)                                  # {}
```

The verb takes the options; the constructor carries attribute injections
only (`Phong(mat, cosinePower=40)` writes inside the chunk; `Phong(mat,
strict=True)` is an `AttributeError`).

```python
print(mat.astype("phong") is mat)                                   # True   in place, chainable
mat.astype(Blinn, diffuse=0.4)                                      # a class works; kwargs validated against the target before any write
print(cmds.nodeType("red"), round(cmds.getAttr("red.diffuse"), 2))  # blinn 0.4
report = shade.convert(mat, "phong", dry_run=True)         # the engine: a Conversion, nothing written, no warning
print(bool(report), repr(str(report)))          # False ''   nothing parked or lost: nothing said
```

The loss policy: transfer what the target can hold, **park** what it
cannot. Every non-default value, wire or animCurve on an attribute the
target lacks is cloned onto the same node as a hidden `__attr__` of the
same type, rides through the conversion, and is restored the next time
the material becomes a type that has it. One `cmds.warning` names what
was parked, before anything moves; `str(report)` is that text.

```python
lossy = Blinn("lossy")
cube                  << lossy
lossy.eccentricity    << 0.66 << lock                    # a blinn-only value
lossy.specularRollOff << rn.ramp(name="ramp1").outAlpha  # a wire into a blinn-only attribute
lossy.diffuse         << 0.33                            # shared: carried

report = shade.convert(lossy, "phong", dry_run=True)
print(str(report))
# rig.shade: 'lossy' blinn -> phong parks:
#    wire        ramp1.outAlpha -> specularRollOff
#    value       eccentricity = 0.66 [locked]
print(report.parked, "diffuse" in report.carried)          # ('eccentricity', 'specularRollOff') True

Phong(lossy)                                               # the same text once, as a warning
print(cmds.listAttr("lossy", userDefined=True))            # ['__eccentricity__', '__specularRollOff__']   hidden, typed like the originals
print(cmds.listConnections("ramp1.outAlpha", plugs=True))  # ['lossy.__specularRollOff__']   the ramp survives, re-homed
```

The round trip restores everything and destroys the parked attributes.
`park=False` gives the drop-and-disconnect behaviour (sources kept in the
scene, values lost); `strict=True` refuses anything lossy with the same
text and writes nothing.

```python
back = shade.convert(lossy, "blinn")            # no warning
print(back.restored, round(cmds.getAttr("lossy.eccentricity"), 2))                                          # ('eccentricity', 'specularRollOff') 0.66
print(cmds.listConnections("lossy.specularRollOff", plugs=True), cmds.listAttr("lossy", userDefined=True))  # ['ramp1.outAlpha'] None

try:
    lossy.astype("phong", strict=True)
except ValueError as err:
    print(str(err).splitlines()[0], cmds.nodeType("lossy"))   # rig.shade: 'lossy' blinn -> phong parks: blinn

drop = Blinn("drop")
cube              << drop
drop.eccentricity << 0.5
drop.astype("phong", park=False)
# Warning: rig.shade: 'drop' blinn -> phong loses:
#    value       eccentricity = 0.5
print(cmds.listAttr("drop", userDefined=True))  # None
```

Refused before any write, none forceable: an unregistered or non-surface
type, a Maya default, referenced or `lockNode`'d material, `Default()`,
a `Node` (wrap it first), and a **string** with the wrong type (a name
asserts; `Phong(Material('lossy'))` converts).

```python
for bad in (
    lambda: lossy.astype("ramp"),                    # TypeError: 'ramp' is a texture, not a surface shader
    lambda: lossy.astype("aiStandardSurface"),       # ValueError: not a registered surface shader (load its plugin)
    lambda: Material("lambert1").astype("phong"),    # RuntimeError: a Maya default node
    lambda: Default().astype("phong"),               # TypeError: never converted
    lambda: Phong(Node("lossy")),                    # TypeError: wrap it first: Phong(Material('lossy'))
    lambda: Phong("lossy").node,                     # TypeError: 'lossy' exists as a blinn
):
    try:
        bad()
    except (TypeError, ValueError, RuntimeError) as err:
        print(type(err).__name__)
print(cmds.nodeType("lossy"), cmds.ls(type="unknown"))   # blinn []
```

One `cmds.undo()` restores the old node (its uuid included). Live `Node`
/ `Plug` wrappers of the old node die; the spec, resolved by name, is the
surviving handle — after an undo its class is stale until a same-type
constructor re-syncs it for free.

```python
cmds.undoInfo(state=True, infinity=True)
stale = Node("lossy")
Phong(lossy)                                     # warns: parks again
print(cmds.undoInfo(query=True, undoName=True))  # rig.shade.convert
try:
    stale.name
except RuntimeError as err:
    print("already deleted" in str(err))        # True
print(Node("lossy").name, round(lossy.diffuse >> None, 2))  # lossy 0.33   fresh wrappers and the spec are fine

cmds.undo()
print(cmds.nodeType("lossy"), type(lossy).__name__)         # blinn Phong   the spec still says phong
Blinn(lossy)                                                # same type as the scene: a free re-sync
print(repr(lossy.node))                                     # Node("lossy")
```

---

## Where to go next

| Read | For |
|---|---|
| [`README.md`](README.md) | the operator table, the conventions, the map, the verified behaviour |
| [`spec/CHEATSHEET.md`](spec/CHEATSHEET.md) | every attribute spec, kwarg and modifier |
| [`bridges/CHEATSHEET.md`](bridges/CHEATSHEET.md) | `maya.cmds` returning nodes, and the node factories |
| [`nodetypes/CHEATSHEET.md`](nodetypes/CHEATSHEET.md) | the typed node layer underneath |
| [`examples/README.md`](examples/README.md) | four complete builds |
