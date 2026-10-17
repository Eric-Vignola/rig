# `rig.nodetypes` — cheatsheet

Copy-paste recipes for the node classes the `rig` DSL is made of: every
class in `rig.nodetypes`, the `Attribute` wrapper, the name helpers,
`Axis` and the bundled undo plug-in. The blocks run top to
bottom as one script and share a namespace — the **Setup** block comes first,
and each section that wants a clean scene starts with `cmds.file(new=True, force=True)`.

Concepts, the resolution rules and the verified behaviour live in [`README.md`](README.md).

## Contents

| # | Section | Covers |
|---|---|---|
| — | [Setup](#setup) | mayapy / Maya bootstrap, the imports |
| 1 | [One node, two attribute spellings](#1-one-node-two-attribute-spellings) | `Node(x)` is the typed node, `node.<attr>` (`Plug`) vs `find_attr` (`Attribute`), `get_node()`, keys |
| 2 | [`Node` — resolution and registration](#2-node--resolution-and-registration) | what a name resolves to, `create`, `find_all`, the three verbs (refer / define / create), `exists`, the errors, a custom node type |
| 3 | [`DGNode`](#3-dgnode) | identity, rename, namespace, attributes, connections, lifecycle, `lock()` / `unlock()` / `is_locked()`, equality |
| 4 | [`DAGNode`](#4-dagnode) | parents, children, shapes, bounding box, deformers |
| 5 | [`Attribute`](#5-attribute) | get / set, `lock()` / `unlock()`, `>>` and `//` on `find_attr` results, typed arrays, multis, slicing with list keys, components |
| 6 | [`Transform`](#6-transform) | shapes, matrices, pivots, `duplicate_geometry`, `serialize`, hierarchies, `save_hierarchy` / `load_hierarchy` (files), `serialize_hierarchy(bindpose=True)` |
| 7 | [`Joint`](#7-joint) | skeleton traversal, duplicate / rename a chain, orients, skinclusters |
| 8 | [`Geometry` — component tags](#8-geometry--component-tags) | `get_injection_node()`, add / set / query / serialize, `tag_references` |
| 9 | [`Mesh`](#9-mesh) | points, UV sets, `serialize` / `create` (the hard edges and locked normals back; a `cgmath` `Object`), `save` / `load` (files: the class creates, a mesh updates), paintable maps, colour sets, materials, transfer |
| 10 | [`NurbsCurve` and `NurbsSurface`](#10-nurbscurve-and-nurbssurface) | CV counts, points, `serialize`, 2-D tags, `NurbsCurve.create`, `NurbsSurface.create`, `save` / `load` (files) |
| 11 | [`Deformer` and `SkinCluster`](#11-deformer-and-skincluster) | geometries, influences, weights as `(V, I)` arrays, `SkinData` round trips, `save` / `load` (files) |
| 12 | [`BlendShape`](#12-blendshape) | targets by name or index, `MorphData` in and out, rebuild, `save` / `load` (files) |
| 13 | [`ObjectSet`](#13-objectset) | `define`, members and components |
| 14 | [`ShadingEngine` and the shaders](#14-shadingengine-and-the-shaders) | `for_material`, `assign`, `get_face_members`; `Blinn` / `Lambert` / ... / `Material`, the network verbs |
| 15 | [`DisplayLayer`](#15-displaylayer) | `define`, `for_node`, exclusive membership, `Layer is DisplayLayer` |
| 16 | [`Choice`](#16-choice) | data types that follow the selector; inputs that take any plug whole, and a number |
| 17 | [`Follicle`](#17-follicle) | rivet a transform to a mesh |
| 18 | [`Reference`](#18-reference) | file references and their namespaces, locks on reference and referenced nodes |
| 19 | [Name helpers](#19-name-helpers) | short / clean names, suffixes, component range strings |
| 20 | [`cmds` results as typed nodes, and `Axis`](#20-cmds-results-as-typed-nodes-and-axis) | casting `maya.cmds` results with `Node`; the mirror axis |
| 21 | [`plugins` — `load_plugin` and undo](#21-plugins--load_plugin-and-undo) | `rigUndoableAPICommand`: your own API edit as one undo step, and its rules |
| 22 | [`AnimCurve` and the curve classes](#22-animcurve-and-the-curve-classes) | refer, the cast, `create` / `define` with keys, `get_keys` / `set_keys` / `get_num_keys` / `evaluate`, `AnimCurve.of`, keys from the class call |

---

## Setup

Runs from `mayapy` or inside Maya. Under `mayapy`, `standalone.initialize()`
boots the scene; inside Maya it raises and is skipped.

```python
from maya import standalone
try:
    standalone.initialize()          # running from mayapy; inside Maya this raises and is skipped
except Exception:
    pass
from maya import cmds
cmds.file(new=True, force=True)

import numpy as np

from rig import Node
from rig.bridges import commands as rc
from rig.nodetypes import (
    Attribute, Axis, BlendShape, Choice, DAGNode, DGNode, DisplayLayer, Follicle,
    Geometry, Joint, Mesh, NurbsCurve, NurbsSurface, ObjectSet, Reference,
    ShadingEngine, SkinCluster, Transform,
)
from rig.nodetypes.deformer import Deformer, tag_references
from rig.nodetypes.dg_node import get_clean_name, get_short_name
from rig.nodetypes.geometry import iter_component_ranges, iter_component_tokens
from rig.nodetypes.joint import replace_suffix
from rig.nodetypes.mesh import ColorSet
from rig.nodetypes.plugins import bundled_plugin_path, load_plugin

print(cmds.about(version=True))
```

`rig.nodetypes` re-exports the classes, `Attribute` and `Axis`.
`Deformer`, `tag_references`, `ColorSet` and the name helpers come from
their own modules (`deformer`, `mesh`, `dg_node`, `geometry`, `joint`),
and `load_plugin` from `rig.nodetypes.plugins`.

---

## 1. One node, two attribute spellings

The DSL node *is* the typed node: `Node(x)` returns the `Transform`, the
`Mesh`, ... itself, so the typed methods are right there, and `node >> None`
is the node.

```python
cube = rc.polyCube(name="cube", ch=False)[0]  # a Transform
xf   = cube >> None                           # the same node
print(repr(cube), xf is cube, Node(cube) is cube)  # Transform("cube") True True
print(type(Node("cubeShape")).__name__)            # Mesh

print(cube.get_shapes())                          # [Mesh("cubeShape")]
print(cube.get_shape().get_materials())           # [StandardSurface("standardSurface1")]
print(Node("cubeShape").get_material_bindings())  # standardSurface1 -- one material, the node itself
```

Attributes have two spellings. `node.<attr>` is a DSL `Plug`: an
`Attribute` whose operators build networks (`+`, `//` and `<` make nodes,
`<<` sets or connects) and whose `get()` is numpy-shaped.
`node.find_attr("<attr>")` is the plain `Attribute`: `get()` mirrors
`cmds.getAttr`, `>>` force-connects, `//` disconnects. Either one's `node`
is the object it was read from, and the two spellings of one plug are equal
and one dict or set key.

```python
print(repr(cube.t), repr(cube.find_attr("t")))  # Plug("cube.translate") Attribute("cube.translate")
print(cube.t.get(), cube.find_attr("t").get())  # [0. 0. 0.] [(0.0, 0.0, 0.0)]
print(Transform("cube") == cube, isinstance(cube.tx, Attribute))  # True True
print(cube.tx.get_node() is cube, cube.find_attr("tx").get_node() is cube, cube.tx == cube.find_attr("tx"), {cube.tx: 1}.get(cube.find_attr("tx")))  # True True True 1
```

---

## 2. `Node` — resolution and registration

`Node(name)` is a factory: it walks `cmds.nodeType(name, inherited=True)`
from the most derived type down and returns the first registered class
(a shader class takes only its exact type: a blinn is a `Blinn`, never a
`Lambert`). Anything unregistered is a `DAGNode` or a `DGNode` (a type Maya
classifies a surface shader is a `Material`); a string with a `.` is
the node before it (the attribute is `Attribute("node.attr")`); a uuid
string works too.

```python
cmds.file(new=True, force=True)
rc.polyCube(name="cube", ch=False)
rc.sphere(name="ball")
rc.curve(point=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], name="crv")
cmds.spaceLocator(name="loc")
cmds.lattice("cube", name="ffd")
cmds.joint(name="jnt")
cmds.shadingNode("blinn", asShader=True, name="shiny")
cmds.shadingNode("anisotropic", asShader=True, name="aniso")

for name in ("cube", "cubeShape", "ballShape", "curveShape1", "locShape", "ffdLatticeShape",
             "ffd", "jnt", "perspShape", "time1", "lambert1", "shiny", "aniso",
             "initialShadingGroup", "defaultObjectSet", "defaultLayer"):
    print(f"{name:20s} {cmds.nodeType(name):14s} -> {type(Node(name)).__name__}")
# cube                 transform      -> Transform
# cubeShape            mesh           -> Mesh
# ballShape            nurbsSurface   -> NurbsSurface
# curveShape1          nurbsCurve     -> NurbsCurve
# locShape             locator        -> Geometry
# ffdLatticeShape      lattice        -> Geometry
# ffd                  ffd            -> Deformer
# jnt                  joint          -> Joint
# perspShape           camera         -> DAGNode
# time1                time           -> DGNode
# lambert1             lambert        -> Lambert
# shiny                blinn          -> Blinn
# aniso                anisotropic    -> Material
# initialShadingGroup  shadingEngine  -> ShadingEngine
# defaultObjectSet     objectSet      -> ObjectSet
# defaultLayer         displayLayer   -> DisplayLayer

print(repr(Node("cube.tx")), repr(Attribute("cube.tx")))  # Transform("cube") Attribute("cube.translateX")
print(Node(Node("cube").get_uuid()))   # cube -- a uuid resolves too
```

`Node.create(type, ...)` routes to the registered class's `create`
(so `"skinCluster"` takes a geometry and influences, `"shadingEngine"` is
built wired, a `"transform"` takes keyword arguments only), and falls back
to `createNode` for the rest. `Node.find_all` lists any type.

```python
grp = Node.create("transform", name="grp")
jnt = Node.create("joint", name="jnt2", parent=grp)
md  = Node.create("multiplyDivide", name="md")
print(repr(grp), repr(jnt), repr(md))                                                           # Transform("grp") Joint("jnt2") DGNode("md")
print(jnt.get_parent())                                                                         # grp

print(Node.find_all("transform")[:2])                                                           # [Transform("ball"), Transform("crv")]
print(len(Node.find_all("transform")) < len(Node.find_all("transform", exact_type=False)))      # True -- joints join in
print(Node.find_all("multiplyDivide"))                                                          # [DGNode("md")] -- no class needed
try:
    Node.find_all("noSuchType")
except ValueError as e:
    print(e)                                       # 'noSuchType' is not a Maya node type

print(Transform.exists("jnt"), Joint.exists("jnt"))                          # True True -- a joint is a transform
print(Transform.is_type("jnt"), Transform.is_type("jnt", exact_type=False))  # False True
```

A class names a node three ways (the
[construction rule](../README.md#the-construction-rule)). `Cls("x")`
**refers**: it never writes the scene, returns the node as its most derived
class, and raises when `x` is missing, ambiguous or of another type.
`Cls.exists("x")` is whether `Cls("x")` would return a node. The errors
are one family, each also a `TypeError` and a `ValueError`:
`NodeNotFoundError` and `AmbiguousNodeError` (both `NodeLookupError`),
and `NodeTypeError`.

```python
from rig import AmbiguousNodeError, NodeNotFoundError, NodeTypeError

print(repr(Transform("jnt")), repr(ObjectSet("initialShadingGroup")))   # Joint("jnt") ShadingEngine("initialShadingGroup")
print(Transform("jnt") == Node("jnt"), Joint.exists("grp"), Transform.exists("nope"))   # True False False
try:
    Joint("jnnt")
except NodeNotFoundError as e:
    print(e)                                       # no joint named 'jnnt' (did you mean 'jnt' or 'jnt2'?); Joint.define('jnnt') finds or makes it
try:
    Joint("grp")
except NodeTypeError as e:
    print(e)                                       # 'grp' is a transform, not a joint; Node('grp') is Transform("grp")
```

`Cls.define("x", ...)` **finds** the node at its key or **makes** it
there, never a second node under a name `Cls("x")` already refers to; the
attributes given are set either way, so the script is the source of truth
(`update=False` leaves a found node as it is).
`Cls.create(name="x", ...)` always makes a new node; `name=` and
`parent=` are keywords. The key is the node `create` would make: the
namespace (the name's own, else the current one), the flattened scope's
prefix, and `parent=` (the world without it). Every refusal writes
nothing: a name that exists elsewhere (pass `parent=`), a namespace that
does not exist or belongs to a file reference, another type at the key.
`Node.define(type, name, ...)` does the same for a type by name.

```python
hub = Transform.define("hub", tx=1)                                   # missing: made at the world, tx set
print(repr(hub), Transform.define("hub", tx=5) == hub, hub.tx.get())  # Transform("hub") True 5.0 -- found, and tx set
Transform.define("hub", tx=7, update=False)                            # update=False: a found node is left as it is
print(hub.tx.get(), repr(Transform.create(name="hub")))               # 5.0 Transform("hub1")
print(repr(Joint.define("jnt2", parent="grp")), repr(Node.define("multiplyDivide", "md")))   # Joint("jnt2") DGNode("md")
try:
    Joint.define("jnt2")
except AmbiguousNodeError as e:
    print(e)                                       # Joint.define('jnt2'): 'jnt2' exists at |grp|jnt2; Joint('jnt2') refers to it, and Joint.define('jnt2', parent='grp') keys it there
try:
    Transform.define("chr:hub")
except ValueError as e:
    print(e)                                       # Transform.define('chr:hub'): there is no namespace 'chr'; define never creates one (cmds.namespace(add='chr') does)
```

A subclass with a `CUSTOM_NODE_TYPE` registers itself (that is the
`NodeMeta` metaclass at work). The type is stamped on the node as a locked
`__custom_node_type__` string, and `Node` reads it before anything else.
`CUSTOM_NODE_TYPE`, `NATIVE_NODE_TYPE`, `FN_SET` and any UPPER_CASE
declaration are class data: `Control.CUSTOM_NODE_TYPE` and, in a method,
`type(self).CUSTOM_NODE_TYPE` read it; a node never does (`hand.DEFAULT` is
its Maya attribute). Python state goes under a `_` name the class body
declares (`_side = None`); a lowercase public `side = None` is refused.

```python
class Control(Transform):
    CUSTOM_NODE_TYPE = "control"

hand = Control.create(name="hand_ctl")
print(repr(hand), hand.get_node_type(), cmds.getAttr("hand_ctl.__custom_node_type__"))  # Control("hand_ctl") control control
print(repr(Node("hand_ctl")), repr(Node.create("control", name="foot_ctl")))           # Control("hand_ctl") Control("foot_ctl")
print(Control.find_all())                                                              # [Control("foot_ctl"), Control("hand_ctl")]
print(Transform.is_type("hand_ctl"), Transform.is_type("hand_ctl", exact_type=False))  # False True
print(Control.CUSTOM_NODE_TYPE, Control.NATIVE_NODE_TYPE, hasattr(hand, "CUSTOM_NODE_TYPE"))  # control transform False
try:
    Control("persp")
except ValueError as e:
    print(e)                                                                       # 'persp' is a transform, not a control; Node('persp') is Transform("persp")
```

---

## 3. `DGNode`

The base of everything. It holds an `MObject`, so it survives renames; it
prints as its name, so it drops straight into `maya.cmds`. Every public
member is a method (a bare name is the node's Maya attribute of that name if
it has one, else the method): `str(node)` is the name, `get_name()` the same
as a method, a predicate is called (`node.is_valid()`).

```python
cmds.file(new=True, force=True)
box = Transform.create(name="cube")
print(box.get_name(), box.get_long_name(), box.get_short_name(), box.get_clean_name(), box.get_node_type())  # cube |cube cube cube transform
print(box.is_valid(), len(box.get_uuid()), box.has_base_type("dagNode"))       # True 36 True

box.rename("box")
box.set_namespace("asset")                 # creates the namespace when missing: one undo step
print(box, box.get_namespace(), box.get_clean_name())  # asset:box asset box
print(str(box), repr(box), cmds.getAttr(f"{box}.tx"))  # asset:box Transform("asset:box") 0.0
```

Attributes: `find_attr` returns the typed `Attribute` (`node.<name>` is the
DSL `Plug` of the same plug); `add_attr` / `rename_attr` / `delete_attr` wrap
`cmds` and hand back `Attribute`s.

```python
print(repr(box.find_attr("tx")), box.find_attr("nope", quiet=True), box.has_attr("tx"))   # Attribute("asset:box.translateX") None True
w = box.add_attr("weight", attributeType="double", keyable=True, min=0, max=1)
print(repr(w), w.get_data_type(), w.is_dynamic(), w.get_default_value())         # Attribute("asset:box.weight") double True 0.0
print(repr(box.rename_attr("weight", "blend")), box.list_attr(userDefined=True))  # Attribute("asset:box.blend") [Attribute("asset:box.blend")]
print(box.delete_attr("blend"), box.has_attr("blend"))                            # True False

box.set_attrs(tx=1, ty=2, tz=3)
box.set_attrs(skip_missing=True, jointOrient=[1, 2, 3], rx=45)                    # a transform has no jointOrient: skipped
box.set_attrs(notes="built by rig")                                               # notes is added on first write
print(box.t.get(), box.rx.get(), box.notes.get())                                 # [1. 2. 3.] 45.0 built by rig
```

Connections and lifecycle:

```python
md = Node.create("multiplyDivide", name="md")
md.input1X << box.tx                       # `<<` connects (the typed spelling: box.find_attr("tx") >> ...)
box.ty     << md.outputX
print(box.list_connections(source=False, destination=True, plugs=True))            # [Attribute("md.input1X")]
print(box.find_connected_nodes(), md.find_connected_nodes(node_type="transform"))  # [DGNode("md")] [Transform("asset:box")]

print(box.duplicate())                                                             # [Transform("box")]
copy = box.duplicate(name="copy", returnRootsOnly=True)[0]
copy.delete()
print(copy.is_valid())                                    # False
try:
    str(copy)
except RuntimeError as e:
    print(e)                                              # copy already deleted!

s = ObjectSet.create(name="s1")
s.add_members(box)
box.remove_from_all_sets()
print(s.get_members())                                      # []
print(sorted([md, box]), {box: 1}[Transform("asset:box")])  # [Transform("asset:box"), DGNode("md")] 1
```

Locks: `lock()` / `unlock()` run `lockNode` and return the node (the DSL's
`node << lock` is the same call); `is_locked()` reads it, and is True for
a member of a locked container too. A node already in that state
is left as it is, with no undo step. A locked node takes values and
connections; `rename`, `delete`, `set_parent`, `add_attr`, `delete_attr`,
`rename_attr`, a new attribute spec and `destroy` refuse it with one
`InjectionError` (a `RuntimeError`), before Maya's own error. Maya locks no
plug of a locked node: lock the plugs first. A referenced node is refused
either way (Maya saves no `lockNode` edit for it), and so is a container
locked inside its own `with` block.

```python
box.lock()
print(box.is_locked(), box.lock() is box)     # True True -- locked already: nothing written
try:
    box.rename("crate")
except RuntimeError as e:
    print(e)                                  # cannot rename 'asset:box': it is locked; Node('asset:box').unlock() unlocks it. Nothing was changed
box.tx.set(4)                                 # values still change
print(box.unlock().is_locked(), box.tx.get())  # False 4.0
```

Equality is by name between node objects of the node's class, hashing by
long name. Every spelling of one node gives that class — `Node("x")`,
`Transform("x")` and `DGNode("x")` all return the most derived one — so
two node objects of one node are equal and one dict key. A node is never
equal to a plug.

---

## 4. `DAGNode`

```python
cmds.file(new=True, force=True)
grp  = Transform.create(name="grp")
cube = Node(cmds.polyCube(name="cube", ch=False)[0])    # Node casts the name to its typed node
cube.set_parent(grp)
shape = cube.get_children(shapes=True)[0]

print(cube.get_parent(), cube.get_parent(1), cube.get_parents(), grp.get_parent())  # grp None [Transform("grp")] None
print(grp.get_children(), grp.get_children(allDescendents=True))                    # [Transform("cube")] [Mesh("cubeShape"), Transform("cube")]
print(shape.is_shape(), cube.is_shape(), shape.get_mdagpath().fullPathName())     # True False |grp|cube|cubeShape
print(cube, cube.get_long_name())                                                   # cube |grp|cube

cube.set_parent(None)                                                               # to world
print(cube.get_parent(), cube.get_long_name())                                      # None |cube
try:
    shape.set_parent(None)
except RuntimeError as e:
    print(e)                                             # Cannot parent shapes to world.

bbox = cube.get_bounding_box()
print(bbox.min, bbox.max)                                # (-0.5, -0.5, -0.5, 1) (0.5, 0.5, 0.5, 1)
```

`get_children` takes every `cmds.listRelatives` flag. `get_deformers` gives
the deformers `findRelatedDeformer` finds as typed nodes, nearest the shape
first (tweak nodes skipped), read through the API, so it leaves no undo step:

```python
cmds.cluster("cube")
print(cube.get_deformers(), shape.get_deformers(node_type="cluster", first_only=True))  # [Deformer("cluster1")] cluster1
print(repr(shape.get_injection_node()), shape.get_injection_node().intermediateObject.get())   # Mesh("cubeShapeOrig") True
```

---

## 5. `Attribute`

An `Attribute` wraps an `MPlug` and subclasses `str`, so `cmds.setAttr(attr, ...)`
works unchanged. `find_attr` returns one; `node.<name>` returns the DSL
`Plug` of the same plug, which has every method below but builds networks
with its operators. On an `Attribute`, `>>` connects (force) and `//`
disconnects; on a `Plug` they are the DSL's `>>` (read, clone, publish) and
floor division, so the operator examples use `find_attr`.

```python
cmds.file(new=True, force=True)
cube = Node(cmds.polyCube(name="cube", ch=False)[0])
mesh = cube.get_shape()

a = cube.find_attr("tx")
print(a, repr(a), a.get_name(), a.get_full_name(), a.get_node(), isinstance(a, str))              # cube.translateX Attribute("cube.translateX") translateX cube.translateX cube True
print(repr(cube.tx), isinstance(cube.tx, Attribute))                                              # Plug("cube.translateX") True -- the DSL spelling
print(a.get_attribute_type(), a.get_data_type(), cube.t.get_data_type(), cube.matrix.get_data_type())  # doubleLinear doubleLinear double3 matrix

cube.t.set(1, 2, 3)
cube.t.set([4, 5, 6])                                                                             # a single sequence is unpacked for double3 & co
cube.tx.set(7)
t = cube.find_attr("t")
print(t.get(), cube.t.get(), cube.tx.get())                                                       # [(7.0, 5.0, 6.0)] [7. 5. 6.] 7.0 -- the cmds.getAttr shape, the Plug's numpy
print(t.get_num_children(), repr(t.translateX), repr(t.child(1)), a.get_parent())                 # 3 Attribute("cube.translateX") Attribute("cube.translateY") cube.translate
print(cube.tx.get_default_value(), cube.t.get_default_value())                                    # 0.0 [0.0, 0.0, 0.0]

print(cube.tx.is_locked(), cube.tx.is_keyable(), cube.tx.is_channel_box(), cube.tx.is_dynamic())  # False True False False
cube.tx.lock()                             # setAttr -lock; returns the attribute, as plug << lock
print(cube.tx.is_locked())                 # True
cube.tx.unlock()

cmds.setAttr(cube.ty, 9)                                                                   # it is a str
ty, tz = cube.find_attr("ty"), cube.find_attr("tz")
print(cmds.getAttr(ty), ty == Attribute("cube.ty"), sorted([tz, a]))                       # 9.0 True [Attribute("cube.translateX"), Attribute("cube.translateZ")]
print(cube.ty == ty, cube.ty.equals("cube.translateY"))                                    # True True -- one plug; sorted() of Plugs raises (an ordering has no truth value)
```

Connections:

```python
a >> ty                                    # force-connect (the Plug spelling: cube.ty << cube.tx)
print(ty.get(), ty.is_connected(), ty.is_free_to_change())                                      # 7.0 True False
print(ty.get_connected_attrs(src=True, dst=False), a.get_connected_attrs(src=False, dst=True))  # [Attribute("cube.translateX")] [Attribute("cube.translateY")]
print(a.list_connections(plugs=True), a.find_connected_nodes())                                 # [Attribute("cube.translateY")] [Transform("cube")]

try:
    tz.connect(ty)                         # connect() without force refuses an occupied input
except RuntimeError as e:
    print(str(e)[:52])                     # 'cube.translateY' already has an incoming connection
tz.connect(ty, force=True)
tz // ty                                   # disconnect (the Plug spelling: cube.tz.disconnect(cube.ty))
print(ty.is_connected())                   # False

a >> ty
a >> tz
print(a.break_connections(), a.is_connected())             # [Attribute("cube.translateZ"), Attribute("cube.translateY")] False
```

Typed data: `set()` fills in `type=` for you, and a `stringArray` /
`vectorArray` / `pointArray` takes one list instead of `(count, *items)`.

```python
weights = cube.add_attr("weights", dataType="doubleArray")
weights.set([0, 1, 0, 0, 4])
print(weights.get(), weights.is_typed(), weights.get_data_type())  # [0.0, 1.0, 0.0, 0.0, 4.0] True doubleArray
print(weights.filter_array_values(0.0))                    # ([1, 4], [1.0, 4.0]) -- sparse ids and values

pts = cube.add_attr("pts", dataType="pointArray")
pts.set([[1, 2, 3, 1], [4, 5, 6, 1]])
print(pts.get())                                           # [(1.0, 2.0, 3.0, 1.0), (4.0, 5.0, 6.0, 1.0)]
names = cube.add_attr("names", dataType="stringArray")
names.set(["a", "b"])
print(names.get())                                         # ['a', 'b']

mode = cube.add_attr("mode", attributeType="enum", enumName="off:on:auto")
mode.set("auto")                                           # or 2: an enum takes a field name too
mode.set_default_value(1)                                  # dynamic attrs only
mode.add_category("rigging")
print(mode.get(), mode.get_enums(), mode.get_default_value(), mode.get_categories(), mode.has_category("rigging"))  # 2 ['off', 'on', 'auto'] 1.0 ['rigging'] True
print(cube.message.get_data_type(), cube.worldMatrix.is_multi(), cube.worldMatrix[0].get_data_type())             # message True matrix
```

Multis: `[i]` is a logical index (created on access), a slice or a list of
ints gives a list of elements, `del attr[i]` removes one.

```python
multi = cube.add_attr("multi", attributeType="double", multi=True)
multi[0].set(1)
multi[3].set(3)
print(multi.is_multi(), multi.get_num_elements(), list(multi.get_logical_indices()), multi.get_next_available_index())  # True 2 [0, 3] 1
print(multi[[0, 3]])                                                                                            # [Attribute("cube.multi[0]"), Attribute("cube.multi[3]")]
print(len(multi[:]), multi[3].logical_index(), multi.element_by_physical_index(1))                              # 4 3 cube.multi[3] -- [:] creates the gaps
print([x.get() for x in multi[:]])                                                                              # [1.0, 0.0, 0.0, 3.0]
del multi[0]
print(list(multi.get_logical_indices()))                   # [1, 2, 3]
```

Geometry components resolve through the `vtx` / `cv` / `pt` aliases to the
live plug, and are range-checked against the point count; list keys accept
negative ids.

```python
print(mesh.vtx, mesh.vtx._component_type, repr(mesh.vtx[0]))     # cubeShape.controlPoints kMeshVertComponent Plug("cubeShape.controlPoints[0]")
print(mesh.vtx[[0, 3, -1]])                                      # List([Plug("cubeShape.controlPoints[0]"), Plug("cubeShape.controlPoints[3]"), Plug("cubeShape.controlPoints[7]")])
print(len(mesh.vtx[:]), len(mesh.vtx[::2]), len(mesh.vtx[2:4]))  # 8 4 2
print(mesh.pnts == mesh.vtx, mesh.pt == mesh.vtx)                # True True -- one plug, three spellings
try:
    mesh.vtx[[0, 8]]
except IndexError as e:
    print(e)                                                   # cubeShape.controlPoints index out of range for 8 points
try:
    mesh.cv                                                    # a curve alias; cmds.listAttr rejects it on a mesh
except AttributeError as e:
    print(e)                                                   # Attribute not found: cubeShape.cv
```

`get()` on a geometry plug never calls `cmds.getAttr` (which errors on
mesh data). A shape's own geometry plug gives back the shape serialised,
a consumer's plug gives the consumer, a dangling output gives an
OpenMaya fn set.

```python
data, uv_list = mesh.outMesh.get()                             # a Mesh's outMesh -> (MeshData, UVList)
print(type(data).__name__, data.point_count, mesh.outMesh.get_data_type())  # MeshData 8 mesh
print(type(mesh.inMesh.get_data_fn_set()).__name__)                   # MFnMeshData
```

---

## 6. `Transform`

```python
cmds.file(new=True, force=True)
grp  = Transform.create(name="grp")
cube = Node(cmds.polyCube(name="cube", ch=False)[0])
cube.set_parent(grp)
grp.t.set(1, 2, 3)
cube.t.set(1, 0, 0)
cube.r.set(0, 90, 0)

print([a.get_name() for a in cube.iter_xform_attrs(axis_only=True)])  # ['translateX', 'translateY', 'translateZ', 'rotateX', 'rotateY', 'rotateZ', 'scaleX', 'scaleY', 'scaleZ']
cube.set_xfrom_attrs_locked(True)                               # sic -- the method is spelled xfrom
print(cube.tx.is_locked())                                      # True
cube.set_xfrom_attrs_locked(False)

world = cube.get_matrix()                                        # MMatrix, world space by default
print(type(world).__name__, np.array(world).reshape(4, 4)[3, :3])         # MMatrix [2. 2. 3.]
print(np.array(cube.get_matrix(world_space=False)).reshape(4, 4)[3, :3])  # [1. 0. 0.]
print(type(cube.get_matrix(as_transform_matrix=True)).__name__)           # MTransformationMatrix

other = Transform.create(name="other")
other.match_matrix(cube, world_space=True)
print(other.t.get(), np.round(other.r.get(), 3))                         # [2. 2. 3.] [ 0. 90.  0.]
other.set_matrix(np.eye(4).ravel().tolist())
print(other.t.get())                                                     # [0. 0. 0.]

print(cube.get_rotate_pivot(), cube.get_scale_pivot(world_space=False))  # (2, 2, 3, 1) (0, 0, 0, 1)
cube.set_pivots((0.5, 0.5, 0.5), world_space=False)
cube.freeze(t=True, r=True, s=True)                                      # cmds.makeIdentity(apply=True, ...)
print(cube.t.get(), cube.r.get())                                        # [0. 0. 0.] [0. 0. 0.]
```

Shapes under a transform, and a clean duplicate of its geometry:

```python
print(cube.get_shapes(), cube.get_shape(), cube.get_shapes(no_interm=False))  # [Mesh("cubeShape")] cubeShape [Mesh("cubeShape")]
print(list(grp.iter_shapes("mesh")), grp.find_shape("mesh", "cu"))            # [Transform("cube")] cube -- transforms of matching shapes
print(list(grp.iter_shapes("mesh", as_transform=False)))                      # [Mesh("cubeShape")]

DisplayLayer.define("geo").add_members(cube)
clean = cube.duplicate_geometry(name="cube_clean")           # no history, no sets or layers, zeroed xform, pivots at origin
print(repr(clean), clean.get_parent(), clean.get_shapes(), clean.t.get())           # Transform("cube_clean") None [Mesh("cube_cleanShape")] [0. 0. 0.]
print(DisplayLayer.for_node(clean), cube.duplicate_geometry(parent=grp).get_long_name())  # None |grp|cube2
```

Serialisation goes through `cgmath.hierarchy`: one node to a
`TransformData`, a whole branch to a `HierarchyData`. Only `transform` and
`joint` nodes (and a transform holding a locator) are captured; constraints
and the like are skipped with a warning.

```python
cube.add_attr("weight", attributeType="double").set(0.5)
data = cube.serialize()
print(type(data).__name__, data.name, data.node_type, data.user_defined_attributes)   # TransformData cube transform {'weight': {'attributeType': 'double', 'value': 0.5, 'keyable': False, 'channel_box': False}}

from rig.spec import Typed                     # an attribute that holds any data: only its numbers are written
slotted = Transform.create(name="slotted")
slots   = slotted << Typed("slots", multi=True)
slots[0] << 2.0
slots[1] << cube.worldMatrix
print(slotted.serialize().user_defined_attributes)   # {'slots': {'attributeType': 'typed', 'multi': True, 'value': [[0, 2.0]], 'keyable': False, 'channel_box': False}}

root  = Joint.create(name="root")
child = Joint.create(name="child", parent=root)
child.t.set(0, 5, 0)
Node(cmds.spaceLocator(name="probe")[0]).set_parent(child)
skeleton = root.serialize_hierarchy()
print(type(skeleton).__name__, skeleton.name, [x.node_type for x in skeleton])   # HierarchyData ['root', 'child', 'probe'] ['joint', 'joint', 'locator']
```

`Transform.create(data)` builds one node from its `TransformData`, as
`Mesh.create(mesh_data)` builds a mesh: the type the data names (a joint's
data makes a `Joint`, a locator's a transform with its locator shape), its
name (`name=` overrides it; Maya renames a taken one), its values as local
values, its user-defined attributes and a joint's orient, segment scale
compensate, radius and draw style. `parent=` and the attribute keywords
are `create`'s (a keyword may also name one of the data's own attributes);
the data's parent and uuid are not used. `Joint.create(joint_data)` is the
same, and refuses a transform's data. Every refusal comes before anything
is made; one undo step; the selection is kept.

```python
twin = Transform.create(data, name="twin", tx=2)                # cube's data, without its mesh: a transform
print(repr(twin), twin.tx.get(), twin.weight.get())             # Transform("twin") 2.0 0.5
bone = Joint.create(child.serialize(), name="bone")             # a joint's orient, radius ... come too
print(repr(bone), bone.ty.get())                                # Joint("bone") 5.0
try:
    Joint.create(data)                                          # a transform's data
except TypeError as err:
    print(err)                                                  # Joint.create(): the data describes a transform, which Joint does not make; Transform.create(data) makes it
cmds.delete("twin", "bone")
```

`create_hierarchy` rebuilds it (needs `cgmath` 1.0.1 or later, for
`SUPPORTED_NODE_TYPES`) and returns the new nodes as a `List`, typed as
`Transform.create` types its node (`Joint`, `Transform`; a locator's
transform and a space_transform box are `Transform`s), parents first.
One undo step. (`List(skeleton)`, without a create, is the nodes the data
was serialized from, found by their uuids; in another scene, by their names,
a name that names several nodes a `TypeError`.)

<!-- notest -->
```python
cmds.file(new=True, force=True)
made = Transform.create_hierarchy(skeleton)
print(repr(made), made[1].ty.get(), cmds.listRelatives("probe", shapes=True))   # List([Joint("root"), Joint("child"), Transform("probe")]) 5.0 ['probeShape']
print(repr(Transform.create_hierarchy(skeleton, parent=Transform.create(name="rig"))))  # List([Joint("rig|root"), Joint("rig|root|child"), Transform("rig|root|child|probe")])
```

`save_hierarchy` / `load_hierarchy` put that data in a `cgmath` file
(`.npz`, `.json` or `.pkl`, by its extension; never load a `.pkl` you did not
write). `Joint.load_hierarchy(path)` (or `Transform.load_hierarchy`) creates,
as `create_hierarchy` does, and returns the new nodes as a typed `List`,
parents first; it is one undo step and keeps the selection. A hierarchy is
never updated in place: from a node, `load_hierarchy` is a `TypeError`.

```python
import os
import tempfile

folder = tempfile.mkdtemp()                                   # the files of this page go here
path   = root.save_hierarchy(os.path.join(folder, "skeleton.npz"))
print(os.path.basename(path))                                 # skeleton.npz
cmds.select("grp")
made = Joint.load_hierarchy(path)                             # creates: root is taken, so root1
print(made[0], [x.get_short_name() for x in made], cmds.ls(selection=True))  # root1 ['root1', 'child', 'probe'] ['grp']
try:
    root.load_hierarchy(path)                                 # from a node: never an update in place
except TypeError as e:
    print(str(e).split(";")[1].strip())                       # Joint.load_hierarchy(path, parent='root') makes a new one under it
print(Transform.load_hierarchy(path, parent="grp")[0].get_long_name())  # |grp|root
try:
    Joint.load_hierarchy(cube.get_shape().save(os.path.join(folder, "cube.npz")))   # a mesh's file
except TypeError as e:
    print(str(e).split(" holds ")[1])                         # a MeshData; the hierarchy is a HierarchyData
```

At its bind pose: `serialize_hierarchy(bindpose=True)` is what a serialize
would give after Maya's Go to Bind Pose (`dagPose -restore -global`), and
nothing moves. A pose member takes scale, rotate, translate, rotate axis and
joint orient from the pose; a non-member (here `tip`, bound with `tsb=True`
and no influence) keeps the scene's values:

```python
cmds.file(new=True, force=True)
root = Joint.create(name="root")
arm  = Joint.create(name="arm", parent=root)
tip  = Joint.create(name="tip", parent=arm)
arm.t.set(0, 5, 0)
tip.t.set(0, 3, 0)
body = Node(cmds.polyCube(name="body", ch=False)[0])
skin = SkinCluster.create(body, [root, arm])   # bindPose1 holds root and arm, not tip
root.r.set(0, 0, 45)                           # posed away
arm.r.set(30, 0, 0)
tip.t.set(0, 4, 0)

bind = root.serialize_hierarchy(bindpose=True)
print([x.name for x in bind], bind[0].rotate, bind[1].rotate)  # ['root', 'arm', 'tip'] [0. 0. 0.] [0. 0. 0.]
print(bind[2].translate, root.r.get())                         # [0. 4. 0.] [ 0.  0. 45.]
print(root.serialize_hierarchy(bindpose="bindPose1")[0].rotate)  # [0. 0. 0.]

cape = Node(cmds.polyCube(name="cape", ch=False)[0])
SkinCluster.create(cape, [root, arm])          # bound while posed: a second bind pose, which disagrees
try:
    root.serialize_hierarchy(bindpose=True)
except ValueError as e:
    print(e)  # 'root' is in bind poses that disagree (bindPose1, bindPose2): 'root' differs between 'bindPose1' and 'bindPose2'; name one: bindpose='bindPose1'

saved = root.save_hierarchy(os.path.join(folder, "bind.npz"), bindpose="bindPose1")   # bindpose= as serialize_hierarchy takes it
print(Joint.load_hierarchy(saved)[1].r.get(), arm.r.get())  # [0. 0. 0.] [30.  0.  0.] -- a new arm at the bind rotation
```

Agreeing bind poses (another influence set bound at the same pose) are
merged. A name or a node reads one dagPose, any dagPose (a saved `tPose`
too); a member reparented since the pose was saved raises:

<!-- notest -->
```python
bind = root.serialize_hierarchy(bindpose="bindPose5")             # that dagPose
bind = root.serialize_hierarchy(bindpose=Node("char:bindPose1"))  # a node, a namespace with it
root.serialize_hierarchy(bindpose="grp")   # NodeTypeError: 'grp' is a transform, not a dagPose
root.serialize_hierarchy(bindpose=None)    # TypeError: bindpose= takes False, True or a dagPose name, not None
# a member moved: ValueError: 'bindPose1' holds 'arm' (posed under 'root', now under 'grp'): a pose member
# is stored relative to the parent it had when the pose was saved, restore the parent or save a new pose
```

---

## 7. `Joint`

```python
cmds.file(new=True, force=True)
root  = Joint.create(name="root")
hip   = Joint.create(name="hip_jnt",   parent=root)
knee  = Joint.create(name="knee_jnt",  parent=hip)
ankle = Joint.create(name="ankle_jnt", parent=knee)
hip.t.set(2, 0, 0)
knee.t.set(2, 1, 0)
ankle.t.set(2, 0, 0)

print(ankle.get_root_joint(), ankle.get_parent_joint(), root.get_parent_joint())  # root knee_jnt None
print(list(root.iter_joints()))                                                   # [Joint("hip_jnt"), Joint("knee_jnt"), Joint("ankle_jnt")]
print(list(root.iter_joints(match_name="knee")), root.find_joint("ankle"))        # [Joint("knee_jnt")] ankle_jnt
```

Orientation: `orient_chain` aims each joint at its first child and leaves
the result in `rotate`; `hierarchy_to_orients` bakes it into
`jointOrient` (and `hierarchy_to_rotations` goes the other way). These
address joints by short name, so run them before the scene holds a second
`hip_jnt`.

```python
root.orient_chain(aim_axis="x", up_axis="y")                                            # prints once for the end joint
print(np.round(hip.jo.get(), 3), np.round(hip.r.get(), 3))                              # [0. 0. 0.] [ 0.    -0.    26.565]
root.hierarchy_to_orients()
print(np.round(hip.jo.get(), 3), np.round(hip.r.get(), 3), np.round(knee.jo.get(), 3))  # [ 0.    -0.    26.565] [0. 0. 0.] [  0.      0.    -26.565]
hip.convert_orients_to_rotation()
print(np.round(hip.jo.get(), 3), np.round(hip.r.get(), 3))                              # [0. 0. 0.] [ 0.    -0.    26.565]
hip.convert_rotation_to_orients()
print(np.round(hip.r.get(), 3), np.round(hip.jo.get(), 3))                              # [0. 0. 0.] [ 0.    -0.    26.565]
```

Duplicate a chain with a new suffix or prefix (non-joint children are
dropped), rename one in place, and match one chain onto another:

```python
ctl = root.duplicate_skeleton(suffix="ctl")
print(repr(ctl), list(ctl.iter_joints()))                        # Joint("root_ctl") [Joint("hip_ctl"), Joint("knee_ctl"), Joint("ankle_ctl")]
ik = root.duplicate_skeleton(prefix="ik", clean_rotations=False)
print(repr(ik), list(ik.iter_joints()))                          # Joint("ik_root") [Joint("ik_hip_jnt"), Joint("ik_knee_jnt"), Joint("ik_ankle_jnt")]
part = root.duplicate_skeleton(include_list=[hip, knee])         # same short names under root1
print(repr(part), list(part.iter_joints()))  # Joint("root1") [Joint("root1|hip_jnt"), Joint("root1|hip_jnt|knee_jnt")]

ctl.rename_skeleton("fk")
print(repr(ctl), list(ctl.iter_joints()))    # Joint("root_fk") [Joint("hip_fk"), Joint("knee_fk"), Joint("ankle_fk")]

hip.t.set(3, 0, 0)
ctl.get_children(type="joint")[0].match_hierarchy(hip)           # from a child joint: the roots are found through it
print(np.round(ctl.get_children(type="joint")[0].t.get(), 3))    # [3. 0. 0.]
```

Which skinclusters a joint drives:

```python
geo  = Node(cmds.polyCube(name="geo", ch=False)[0])
skin = SkinCluster.create(geo, [root, hip])
print(hip.find_skinclusters(), knee.find_skinclusters(), root.find_skinclusters(recursive=True))  # [SkinCluster("geo_skincluster")] [] [SkinCluster("geo_skincluster")]
```

---

## 8. `Geometry` — component tags

`Geometry` is the shape base (`Mesh`, `NurbsCurve`, `NurbsSurface`, and the
`DAGNode` fallback for lattices and locators). Its job is component tags.
Every editor writes to `get_injection_node()`: the shape itself until a
deformer exists, the intermediate `Orig` shape from then on.

```python
cmds.file(new=True, force=True)
ball = Node(cmds.polySphere(name="ball", ch=False)[0]).get_shape()
print(ball.get_injection_node() == ball, ball.get_component_tags())                                                              # True []

cmds.cluster("ball")
print(repr(ball.get_injection_node()))                                                                                             # Mesh("ballShapeOrig")

ball.add_component_tag("cap")                                                                                                      # returns the new multi index
ball.set_component_tag_contents("cap", [0, 1, 2, 3])                                                                               # ids, category "v" by default
print(ball.get_component_tags(), ball.has_component_tag("cap"), ball.get_component_tag_index("cap"))                               # ['cap'] True 0
print(ball.get_component_tag_contents("cap"), ball.get_component_tag_indices("cap"))                                               # ['vtx[0:3]'] [0 1 2 3]
print(ball.get_component_tag_category("cap"), ball.get_component_tag_name(0))                                                      # v cap
print(cmds.getAttr("ballShapeOrig.componentTags", multiIndices=True), cmds.getAttr("ballShape.componentTags", multiIndices=True))  # [0] None
print(ball.get_component_tag_contents("cap", full_name=True))                                                                      # ['ballShape.vtx[0:3]']
```

A name Maya refuses is a `ValueError` naming Maya's rule, before anything
is made. Maya takes at least two characters, letters, digits and `_`,
starting with a letter or `_`, with single `.` or `:` between characters.
It trims the whitespace at the ends and makes a space inside `_`. A name
that is not a `str` is a `TypeError` (`add_component_tag(5)` is not the
tag at index 5):

```python
try:
    ball.add_component_tag("b")
except ValueError as e:
    print(str(e)[:59])                                        # 'b' is not a component tag name: Maya takes a component tag
try:
    ball.add_component_tag(5)
except TypeError as e:
    print(str(e)[:46])                                        # 5 is not a component tag name: a name is a str
print(ball.get_component_tag_name(ball.add_component_tag("lid rim")))  # lid_rim
ball.remove_component_tag("lid_rim")
```

Contents can be ids with a category, component strings, or a
`GeomSubsetData`; `set_component_tag_contents` renders ranges for you.

```python
ball.set_component_tag_contents("cap", [4, 6, 5], category="f")
print(ball.get_component_tag_contents("cap"), ball.get_component_tag_category("cap"))  # ['f[4:6]'] f
ball.set_component_tag_contents("cap", ["e[3]", "e[5]", "e[6]"])
print(ball.get_component_tag_contents("cap"))                                          # ['e[3]', 'e[5:6]']

subset = ball.get_component_tag_data("cap")                      # a cgmath GeomSubsetData
print(subset.name, subset.component_type, subset.indices)                         # cap e [3 5 6]
ball.set_component_tag_contents("cap", subset)
print([s.name for s in ball.serialize_component_tags()])                          # ['cap']

ball.set_component_tag_contents("cap", [])
print(ball.is_component_tag_empty("cap"), ball.get_component_tag_indices("cap"))  # True []
```

Expressions and history, the way deformers see them:

```python
ball.set_component_tag_contents("cap", [0, 1, 2, 3])
cmds.setAttr("cluster1.input[0].componentTagExpression", "cap", type="string")
print(tag_references("cap"), tag_references("lid"))                                                        # [('cluster1', 0)] []
print(ball.component_tag_expression_subset_state("cap"), ball.component_tag_expression_subset_state("*"))  # 1 2 -- some / all points
entry = ball.get_component_tag_history()[0]
print(entry["key"], entry["node"], entry["affectCount"], entry["procedural"], entry["editable"])   # cap ballShapeOrig 4 False True
```

Rename and remove address the injection node too. A tag that only exists
on the output plug (procedural, such as a history `polyCube`'s six face
tags) is readable but refuses every edit:

```python
ball.rename_component_tag("cap", "lid")
print(ball.has_component_tag("cap"), ball.has_component_tag("lid"))  # False True
ball.remove_component_tag("lid")
print(ball.get_component_tags())                                     # []

hist = Node(cmds.polyCube(name="hist")[0]).get_shape()    # ch=True: the tags are polyCube1's outputs
print(hist.get_component_tags(), hist.get_component_tag_index("top"), hist.get_component_tag_contents("top"))   # ['back', 'bottom', 'front', 'left', 'right', 'top'] -1 ['f[1]']
try:
    hist.set_component_tag_contents("top", [0], category="f")
except RuntimeError as e:
    print(str(e)[:64])                                           # Component tag 'top' on histShape is procedural (owned by polyCub
hist.add_component_tag("lid")   # a new name is editable next to them
print(hist.get_component_tags()[:2])  # ['lid', 'back']
```

The plugs Maya evaluates tags on, and the component `MObject`s the
OpenMaya calls want:

```python
print(ball.get_local_shape_attr(), ball.get_world_shape_attr())                                                    # ballShape.outMesh ballShape.worldMesh
print(ball.get_component_mobject().apiTypeStr, ball.get_component_mobject("f", np.array([0, 1])).apiTypeStr)  # kMeshVertComponent kMeshPolygonComponent
```

---

## 9. `Mesh`

```python
cmds.file(new=True, force=True)
cube = Node(cmds.polyCube(name="cube", ch=False)[0])
mesh = cube.get_shape()
print(repr(Mesh("cube")), Mesh("cubeShape") == mesh)                 # Mesh("cubeShape") True -- a transform name finds its mesh
print(mesh.get_num_vertices(), mesh.get_num_polygons(), mesh.get_num_weight_points())  # 8 6 8

points = mesh.get_points()                                       # MPointArray, world space by default
print(type(points).__name__, np.array(points).shape, np.array(points)[0])   # MPointArray (8, 4) [-0.5 -0.5  0.5  1. ]
lifted = np.array(points)[:, :3]
lifted[:, 1] += 1
mesh.set_points(lifted)                                                             # undoable
print(np.array(mesh.get_points())[0, :3])                                           # [-0.5  0.5  0.5]
cmds.undo()
print(np.array(mesh.get_points())[0, :3])                                           # [-0.5 -0.5  0.5]

print(mesh.get_closest_point([0, 0, 5]))                                            # (maya.api.OpenMaya.MPoint(0, 0, 0.5, 1), 0) -- (point, face id)
print(mesh.get_vertex_normals()[0], len(mesh.get_per_face_vertex_normals()))        # (-0.57735, -0.57735, 0.57735) 24
print(mesh.get_vertices_above_plane(np.array([0.0, 0, 0]), np.array([0.0, 1, 0])))  # [2 3 4 5]
print(mesh.get_faces_above_plane(np.array([0.0, 0, 0]), np.array([0.0, 1, 0])))     # [1]
```

UV sets:

```python
print(mesh.get_uv_sets(), mesh.get_current_uv_set(), mesh.get_uv_coords().shape)   # ('map1',) map1 (14, 2)
counts, ids = mesh.get_assigned_uvs()
print(len(counts), len(ids), mesh.get_uv_at_point([0, 0, 0.5], uv_set=None)[:2])   # 6 24 (0.49666666984558105, 0.125)

uv = mesh.serialize_uv()                                         # a cgmath UVData
print(type(uv).__name__, uv.name, uv.points.shape)               # UVData map1 (14, 2)
uv.name = "map2"                                                 # set_uv_data renames the set to the data's name
mesh.add_uv_set("map2")
mesh.set_uv_data(uv, "map2")
mesh.rename_uv_set("uv2", "map2")
print(mesh.get_uv_sets())  # ('map1', 'uv2')
mesh.delete_uv_set("uv2")
print(mesh.get_uv_sets())  # ('map1',)
```

To and from `cgmath.geometry.MeshData`; `Mesh.create` is undoable and puts
the new shape in `initialShadingGroup`:

```python
data, uv_list = mesh.serialize()                                 # world space, with every UV set
print(type(data).__name__, data.point_count, data.face_count, type(uv_list).__name__, uv_list[0].name)  # MeshData 8 6 UVList map1
print(mesh.serialize(include_uvs=False, world_space=False).matrix.shape)                                # (4, 4) -- the parent's world matrix

rebuilt = Mesh.create(data, uv_data=uv_list, name="rebuilt")
print(repr(rebuilt), rebuilt.get_num_vertices(), rebuilt.get_parent(), rebuilt.get_uv_sets())  # Mesh("rebuiltShape") 8 rebuilt ('map1',)
print(rebuilt.get_shading_engines())                                               # [ShadingEngine("initialShadingGroup")]
print(sum(not rebuilt.get_fn_set().isEdgeSmooth(e) for e in range(12)))                  # 12 -- the data's normals bring back the cube's hard edges
```

`create` also takes a `cgmath.render.scene.Object`, for the mesh and its UVs:
its `mesh`, its `uv` (a `UVData` is one set, a `UVList` every set), its name and
its transform; its render settings are not used.

```python
from cgmath.render.scene import Object

hero = Mesh.create(Object(name="hero", mesh=data, uv=uv_list[0], translate=(5, 0, 0)))
print(repr(hero), hero.get_parent().t.get(), hero.get_uv_sets())   # Mesh("heroShape") [5. 0. 0.] ('map1',)
```

To and from files: `save` writes the `MeshData` to one `cgmath` file and, with
`uvs=`, the UV sets to a second (`.npz` recommended, `.json`, `.pkl`; by the
extension). `Mesh.load(...)` from the class creates, as `create` does;
`Mesh("x").load(...)` from a mesh updates it in place, the inverse of its
`save`: the same topology, then its points, its hard edges and locked normals,
and the file's UV sets by name. Each is one undo step and keeps the selection;
every file is read, and checked, before anything changes. `normals=False` (a
*restore flag*, the same on `create` and both `load` forms; True by default)
skips the hard edges and locked normals, for speed: a new mesh is all smooth,
an updated one keeps its own. Restoring them costs within 1.5x of a create
without normals for a mostly smooth mesh, 1.75x to 2.6x for a faceted or
heavily locked one.

```python
import os
import tempfile

folder = tempfile.mkdtemp()
paths  = mesh.save(os.path.join(folder, "cube.npz"), uvs=os.path.join(folder, "cube_uv.npz"))
print([os.path.basename(p) for p in paths])                  # ['cube.npz', 'cube_uv.npz']
copy = Mesh.load(*paths, name="copy")                        # creates
print(repr(copy), copy.get_uv_sets())                        # Mesh("copyShape") ('map1',)
cmds.polySoftEdge("copy", angle=180, constructionHistory=False)   # softened ...
copy.load(*paths)                                            # ... and updated in place from the files
print(sum(not copy.get_fn_set().isEdgeSmooth(e) for e in range(12)))    # 12
fast = Mesh.load(*paths, name="fast", normals=False)         # creates, the shading skipped
print(sum(not fast.get_fn_set().isEdgeSmooth(e) for e in range(12)))    # 0
try:
    Mesh.load(paths[1], paths[0])                            # the two files swapped
except TypeError as e:
    print(str(e).split(" holds ")[1])                        # a UVList; the mesh is a MeshData or an Object (the UVs go in uvs=)
```

Paintable maps are `doubleArray` attributes in the `PaintableMap`
category, one value per vertex, padded or trimmed to the vertex count:

```python
mask = mesh.add_map("mask", values=0.5, category="weights")
print(repr(mask), mask.get_data_type(), mask.get_categories(), mesh.get_paintable_maps())  # Attribute("cubeShape.mask") doubleArray ['weights', 'PaintableMap'] ['mask']
mesh.set_map_values("mask", [1, 0, 1])                                         # short lists are padded with 0.0
print(mesh.get_map_values("mask"))                                             # [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mesh.set_map_values("mask", np.linspace(0, 1, 8))
print(np.round(mesh.get_map_values("mask"), 2))                                # [0.   0.14 0.29 0.43 0.57 0.71 0.86 1.  ]

map_data = mesh.get_map_data("mask")                             # a cgmath MapData, sparse: zeros dropped
print(map_data.name, map_data.categories, map_data.indices.tolist())                       # mask ['weights'] [1, 2, 3, 4, 5, 6, 7]
print(repr(mesh.duplicate_map("mask", "mask2")), [m.name for m in mesh.serialize_maps()])  # Attribute("cubeShape.mask2") ['mask', 'mask2']
print(mesh.get_map_attrs(category="weights"), mesh.find_map_attr("visibility"))            # [Attribute("cubeShape.mask")] None
mesh.mirror_map("mask", Axis.X)                                                            # +X values onto their -X partners
print(np.round(mesh.get_map_values("mask"), 2))                                            # [0.14 0.14 0.43 0.43 0.71 0.71 1.   1.  ]
from cgmath.geometry.mesh import Axis as CgmathAxis
mesh.mirror_map("mask2", CgmathAxis.X)                                                     # cgmath's Axis names the same planes
print(mesh.get_map_values("mask2") == mesh.get_map_values("mask"))                         # True
mesh.delete_map("mask2")
print(mesh.get_paintable_maps(), mesh.sanitize_map_values([1, 2]))                               # ['mask'] [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

Colour sets are `ColorSet` handles on the scene, so every member is a
method: `get_name()`, `get_mesh()`, `is_valid()`, `get_representation()`,
and `get_data()` / `set_data(colours)`, an `(V, 4)` array. `get_data()` is a
read (the current colour set stays, the scene is not marked modified);
`set_data` is one undo step.

```python
print(mesh.get_color_sets())                                     # []
cset = mesh.add_color_set("tint", ColorSet.Representation.RGBA)   # Alpha, RGB or RGBA
print(cset, cset.get_name(), cset.get_mesh(), cset.is_valid(), cset.get_representation())   # ColorSet<cubeShape.tint> tint cubeShape True Representation.RGBA
cset.set_data(np.tile([1.0, 0.0, 0.0, 1.0], (8, 1)))
print(cset.get_data().shape, cset.get_data()[0])   # (8, 4) [1. 0. 0. 1.]
cset.delete()
print(cset.is_valid(), mesh.get_color_sets())  # False []
```

Materials, read from the shading engines the shape belongs to.
`get_material_bindings` returns the one material when it covers the whole
mesh, otherwise `(material, face ids)` pairs; `verbose=True` always gives
the pairs.

```python
print(mesh.get_materials(), mesh.get_shading_engines())  # [StandardSurface("standardSurface1")] [ShadingEngine("initialShadingGroup")]
print(mesh.get_material_bindings())                      # standardSurface1
print(mesh.get_material_bindings(verbose=True))          # [(StandardSurface("standardSurface1"), array([0, 1, 2, 3, 4, 5]))]

red = cmds.shadingNode("blinn", asShader=True, name="red")
ShadingEngine.for_material(red).assign(["cube.f[0:1]"], touched=[mesh.get_long_name()])
print(sorted(mesh.get_materials()), sorted(mesh.get_shading_engines()))                # [Blinn("red"), StandardSurface("standardSurface1")] [ShadingEngine("initialShadingGroup"), ShadingEngine("redSG")]
print(mesh.get_material_bindings())                                                    # [(Blinn("red"), array([0, 1])), (StandardSurface("standardSurface1"), array([2, 3, 4, 5]))]
print(mesh.get_materials(as_pairs=True)[0][0].get_material() in mesh.get_materials())  # True -- (engine, material) pairs
```

Transfer maps, tags and skin weights onto another topology (a
`cgmath` `MeshDataResampler`; `mode=ResampleMode.UV` samples through the
first UV sets). Each target vertex gets the blend of the source values its
closest point reads, and a map keeps its categories. A tag holds what lands
on it: a vertex whose sample reads a tagged vertex, a face whose centre (in
space, or its UV centre by UV) is closest to a tagged face, an edge both of
whose ends are in the vertex set the tag's vertices transfer to (so a rung
between two tagged loops is in). A flat concave or holed face lands by its
first triangle's centre, and a centre on a shared corner or edge by the
lowest-numbered face there. Onto the same topology (by UV: same UVs, not
overlapping) a tag comes back as it was, but for those rungs.

```python
dst = Node(cmds.polyCube(name="dst", ch=False)[0]).get_shape()
mesh.transfer_maps("mask", dst)
print(dst.get_paintable_maps(), np.round(dst.get_map_values("mask"), 2))   # ['mask'] [0.14 0.14 0.43 0.43 0.71 0.71 1.   1.  ]
print(dst.get_map_data("mask").categories)                           # ['weights']

mesh.add_component_tag("cap")
mesh.set_component_tag_contents("cap", [0, 1, 2, 3])
mesh.transfer_component_tags("cap", dst)                             # written at dst's injection node
print(dst.get_component_tags()[-1], dst.get_component_tag_indices("cap"))  # cap [0 1 2 3]
mesh.add_component_tag("lid")
mesh.set_component_tag_contents("lid", [1], "f")
mesh.transfer_component_tags("lid", dst)                             # where dst's face centres land
print(dst.get_component_tag_indices("lid"))                          # [1]

jnt  = Joint.create(name="jnt")
skin = SkinCluster.create(cube, jnt)
print(repr(dst.apply_skin_data(skin.serialize())))                   # SkinCluster("dstShape_skincluster") -- creates or updates
```

---

## 10. `NurbsCurve` and `NurbsSurface`

Both are `Geometry`, so the tag methods above apply; their class data
`POINT_COMP_TYPE` is `cv`, and a surface's ids are `(N, 2)` `(u, v)` pairs.

```python
cmds.file(new=True, force=True)
crv = Node(cmds.curve(point=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)], name="crv")).get_shape()
print(repr(crv), crv.get_num_cvs(), crv.get_num_weight_points(), np.array(crv.get_points())[1, :3])   # NurbsCurve("curveShape1") 5 5 [1. 0. 0.]
spline = crv.serialize()                                          # cgmath BSplineData
print(type(spline).__name__, spline.points.shape, spline.degree, spline.periodic)  # BSplineData (5, 3) 3 False
print(crv.cv, len(crv.cv[:]), crv.cv[[0, 2]])                                      # curveShape1.controlPoints 5 List([Plug("curveShape1.controlPoints[0]"), Plug("curveShape1.controlPoints[2]")])

crv.add_component_tag("root")
crv.set_component_tag_contents("root", [0, 1])
print(crv.get_component_tag_contents("root"))                                      # ['cv[0:1]']

ball = Node(cmds.sphere(name="ball")[0]).get_shape()
print(repr(ball), ball.get_num_cvs(), ball.get_num_weight_points())   # NurbsSurface("ballShape") 77 56 -- periodic in V: 3 wrapped rows per column
patch = ball.serialize()                                          # cgmath BSplinePatchData, wrapped CVs trimmed
print(type(patch).__name__, patch.points.shape, patch.periodic_u, patch.periodic_v)            # BSplinePatchData (7, 8, 3) False True

ball.add_component_tag("rim")
ball.set_component_tag_contents("rim", np.array([[1, 0], [1, 1]]))
print(ball.get_component_tag_contents("rim"), ball.get_component_tag_indices("rim").tolist())  # ['cv[1][0:1]'] [[1, 0], [1, 1]]
print(ball.get_component_tag_indices("rim").shape)                                             # (2, 2)
```

`NurbsCurve.create` builds a curve from its control points, with a degree
(3 unless given) and knots in Maya's layout (the ones `cmds.curve` gives
when left out), or from a `cgmath` `BSplineData`, whose points, degree,
knots and form it takes (`uniform` and `registered` have no Maya
equivalent). Like `Mesh.create` it is one undo step and leaves the
selection alone; the points are stored as given, in Maya's internal unit
(`cmds.curve` reads them in the scene's linear unit). `serialize` is its
inverse: a periodic curve's own points (Maya's repeated CVs left out), and
its knots when they are not Maya's default ones and the installed `cgmath`'s
`BSplineData` takes them (it keeps their spacing, so the shape, but not
their range):

```python
points = [(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (4, 0, 0)]
arc = NurbsCurve.create(points, name="arc")                                       # a cubic with cmds.curve's knots
print(repr(arc), arc.get_parent(), arc.get_fn_set().degree, list(arc.get_fn_set().knots()))   # NurbsCurve("arcShape") arc 3 [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
zig = NurbsCurve.create(points, degree=1, kv=[0, 1, 2, 4, 8], name="zig")         # a degree and knots of your own
print(zig.get_fn_set().degree, list(zig.get_fn_set().knots()))                          # 1 [0.0, 1.0, 2.0, 4.0, 8.0]
twin = NurbsCurve.create(crv.serialize(), name="twin")                            # from a BSplineData
print(repr(twin), twin.get_num_cvs(), twin.get_fn_set().degree, np.array(twin.get_points())[1, :3])   # NurbsCurve("twinShape") 5 3 [1. 0. 0.]
```

`NurbsSurface.create` builds a surface from a `BSplinePatchData`: its points,
degrees and forms (a `kClosed` direction serializes open, keeping its CVs). A
`BSplinePatchData` holds no knots, so they come back evenly spaced, over
`0..spans`: the same surface when they were evenly spaced, and `serialize`
warns when they were not. Both classes save and load `cgmath` files: from the
class `load` creates, from a shape it updates the CVs in place (the same CV
layout; a surface keeps its own knots). A `BSplineData` holds no name and no
matrix: save in world space (the default) to keep where the shape is. The
update takes `world_space=` with no default, the space the file was saved in
(the wrong one would move the CVs), by keyword: without it, or given by
position (`load(path, True)`), a `TypeError` naming it.

```python
import os
import tempfile

sheet = NurbsSurface.create(ball.serialize(), name="sheet")
print(repr(sheet), sheet.get_num_cvs(), sheet.get_fn_set().knotDomainInU)    # NurbsSurface("sheetShape") 77 (0.0, 4.0)

folder = tempfile.mkdtemp()
saved  = crv.save(os.path.join(folder, "crv.json"))
again = NurbsCurve.load(saved, name="again")                     # creates
cmds.xform("again.cv[0]", translation=(0, 3, 0))
again.load(saved, world_space=True)                              # the CVs back from the file
print(repr(again), np.array(again.get_points())[0, :3])          # NurbsCurve("againShape") [0. 0. 0.]
try:
    again.load(saved)                                            # no world_space=
except TypeError as e:
    print(str(e).split(":")[0])                                  # NurbsCurve('againShape').load needs world_space=
patch = NurbsSurface.load(ball.save(os.path.join(folder, "ball.npz")), name="patch")
print(repr(patch), patch.get_num_cvs())                          # NurbsSurface("patchShape") 77
try:
    NurbsSurface.load(saved)                                     # a curve's file
except TypeError as e:
    print(str(e).split(" holds ")[1])                            # a BSplineData; the surface is a BSplinePatchData
```

---

## 11. `Deformer` and `SkinCluster`

`Deformer` is the `geometryFilter` base every deformer resolves to;
`SkinCluster` and `BlendShape` add to it. Weights are dense `(V, I)` numpy
arrays, one column per influence, in `get_influence_objects()` order.

```python
cmds.file(new=True, force=True)
j1 = Joint.create(name="j1")
j2 = Joint.create(name="j2", parent=j1)
j2.t.set(0, 1, 0)
cube = Node(cmds.polyCube(name="cube", ch=False, height=2)[0])

skin = SkinCluster.create(cube, [j1, j2])                          # cmds.skinCluster(toSelectedBones=True), replaces any existing one
print(repr(skin), skin.get_influence_objects(), skin.get_mesh())           # SkinCluster("cube_skincluster") [Joint("j1"), Joint("j2")] cubeShape
print(skin.get_geometries(), skin.get_original_geometries())               # [Mesh("cubeShape")] [Mesh("cubeShapeOrig")]
print(repr(Node("cube_skincluster")), cube.get_shape().get_deformers())    # SkinCluster("cube_skincluster") [SkinCluster("cube_skincluster")]

weights = skin.get_weights()
print(weights.shape, weights.sum(axis=1))                         # (8, 2) [1. 1. 1. 1. 1. 1. 1. 1.]
split        = np.zeros_like(weights)
split[:4, 0] = 1
split[4:, 1] = 1
skin.set_weights(split)                                          # undoable; Maya may warn about normalisation
print(skin.get_weights()[:, 0])                                  # [1. 1. 1. 1. 0. 0. 0. 0.]
skin.set_weights(np.array([[0.5, 0.5]]), indices=np.array([0]))  # a subset of vertices
print(skin.get_weights()[0])                                     # [0.5 0.5]
```

`serialize()` gives a `cgmath` `SkinData`; `set_weights(SkinData)` conforms
influences by name (adding what is missing, dropping extras unless
`additive=True`), and `create(geo, SkinData)` rebuilds from one.

```python
data = skin.serialize()
print(type(data).__name__, data.influences, data.weights.shape, data.valid)  # SkinData ['j1', 'j2'] (8, 2) True
print(skin.serialize(full_path=True).influences)                             # ['|j1', '|j1|j2']

j3 = Joint.create(name="j3", parent=j2)
skin.add_influence_objects(j3)                                 # keeps your selection
print(skin.get_influence_objects(), skin.get_weights().shape)  # [Joint("j1"), Joint("j2"), Joint("j3")] (8, 3)
skin.set_weights(data)                                         # j3 is not in the data: removed again
print(skin.get_influence_objects())                            # [Joint("j1"), Joint("j2")]
skin.set_influence_objects([j1, j2, j3])
skin.set_weights(data, additive=True)
print(skin.get_influence_objects())                            # [Joint("j1"), Joint("j2"), Joint("j3")]
skin.remove_influence_objects(j3)

skin.set_max_influences(1)
skin.prune_small_weights(0.01)
skin.normalize_weights()
print(skin.get_weights()[0])                                   # [1. 0.]

skin.delete()
skin = SkinCluster.create(cube, data)                             # influences and weights from the data
print(skin.get_influence_objects(), skin.serialize() == data)                                 # [Joint("j1"), Joint("j2")] True
print(repr(Node.create("skinCluster", Node(cmds.polyCube(name="c2", ch=False)[0]), j1)))      # SkinCluster("c2_skincluster")
```

To and from a `cgmath` file: `save` writes the `SkinData` (the weights and the
influences, by name; not the skin's settings nor its bind pose).
`SkinCluster.load(path, geo, ...)` from the class creates, as `create` does
(the geometry's own skin replaced, bound at the joints' current pose); a
skin's own `load(path)` sets its weights in place, as `set_weights(SkinData)`
does. The file comes first, then the geometry:

```python
import os
import tempfile

saved = skin.save(os.path.join(tempfile.mkdtemp(), "cube_skin.npz"))
skin.set_weights(np.full((8, 2), 0.5))
skin.load(saved)                                                # updated in place
print(skin.get_weights()[7])                                    # [0. 1.]
skin = SkinCluster.load(saved, cube)                            # creates: the cube's own skin replaced
print(repr(skin), skin.get_weights()[7])                        # SkinCluster("cube_skincluster") [0. 1.]
try:
    SkinCluster.load(cube, saved)                               # the geometry first
except TypeError as e:
    print(str(e).split(":")[0])                                 # SkinCluster.load(skin, geometry) takes the file first, then the geometry
```

Transfer to another mesh (pass the shape or a name, not a `Transform`).
`copy_skincluster` copies from one or more skinned meshes onto vertices
(or whole meshes, by name), by closest point; the joints the target's
skin lacks are added, and its other vertices keep their weights. The
copied rows are the sampled weights as they are: the target's locked
influences and its `maxInfluences` are not applied to them. On a mesh with
several skinClusters, `copy_skincluster`, `apply_skin_data` and
`bake_deformation` use the one nearest the shape, normally
`get_deformers(node_type="skinCluster")[0]` (the skin of a blendShape's
intermediate target, which that lists too, is not the mesh's own):

```python
other = Node(cmds.polyCube(name="other", ch=False, height=2)[0]).get_shape()
moved = skin.transfer_to_mesh(other)
print(repr(moved), moved.get_influence_objects(), moved.get_weights().shape)   # SkinCluster("otherShape_skincluster") [Joint("j1"), Joint("j2")] (8, 2)

third = Node(cmds.polyCube(name="third", ch=False, height=2)[0])
SkinCluster.create(third, j1)
SkinCluster.copy_skincluster(str(cube), ["third.vtx[4:7]"])
copied = third.get_shape().get_deformers()[0]
print(copied.get_influence_objects(), copied.get_weights()[:, 1])              # [Joint("j1"), Joint("j2")] [0. 0. 0. 0. 1. 1. 1. 1.]
```

---

## 12. `BlendShape`

Targets are addressed by alias or logical index (index is faster); offsets
go in and out as `cgmath` `MorphData` (sparse: `indices` + `offsets`).

```python
cmds.file(new=True, force=True)
base  = Node(cmds.polyCube(name="base", ch=False)[0])
smile = Node(cmds.polyCube(name="smile", ch=False)[0])
cmds.xform("smile.vtx[3]", ws=True, t=(10, 2, 3))

morph = BlendShape.create(smile, base, name="morph")              # cmds.blendShape(frontOfChain=True)
print(repr(morph), morph.get_num_targets(), morph.get_targets(), morph.get_target_indices())                   # BlendShape("morph") 1 ['smile'] [0]
print(morph.get_target_index("smile"), morph.get_target_name(0), repr(morph.get_target_weight_attr(0)))  # 0 smile Attribute("morph.smile")
morph.set_target_weight("smile", 1.0)
print(morph.get_target_weight(0), morph.smile.get())                                                     # 1.0 1.0 -- the alias resolves as an attribute too

target = morph.get_target_data("smile")                           # read through the live target mesh
print(type(target).__name__, target.name, target.indices, np.round(target.offsets, 2))  # MorphData smile [3] [[9.5 1.5 2.5]]
print(morph.get_geometries(), morph.get_original_geometries())                          # [Mesh("baseShape")] [Mesh("baseShapeOrig")]
```

Add, fill, rename and serialise targets:

```python
i = morph.add_empty_target("frown")
morph.set_target_data("frown", (np.array([0, 1]), np.array([[0, -1, 0], [0, -1, 0]])))  # (indices, offsets) or a MorphData
print(i, morph.get_targets(), morph.get_target_data(i).indices)                         # 1 ['smile', 'frown'] [0 1]
morph.set_target_name("frown", "sad")
print(morph.get_targets(), repr(morph.find_alias("sad")))                               # ['smile', 'sad'] Attribute("morph.sad")
print([repr(a) for a in morph.target_data_attrs(1)][1])                                 # Attribute("morph.inputTarget[0].inputTargetGroup[1].inputTargetItem[6000].inputPointsTarget")

morphs = morph.serialize()                                        # a cgmath MorphList
print(type(morphs).__name__, [m.name for m in morphs], [m.name for m in morph.serialize(match_name="sm.*")])   # MorphList ['smile', 'sad'] ['smile']

geo = morph.rebuild_target("sad", name="sad_geo")                 # a clean duplicate of the base with the offsets applied
print(repr(geo), np.round(np.array(geo.get_shape().get_points())[0, :3], 2))   # Transform("sad_geo") [-0.5 -1.5  0.5]

morph.delete()
morph = BlendShape.create(base, morphs, name="morph2")            # targets built from the data
print(morph.get_targets(), morph.serialize() == morphs)  # ['smile', 'sad'] True
print(morph.find_attr("weight")[:], morph.weight[:])     # [Attribute("morph2.smile"), Attribute("morph2.sad")] List([Plug("morph2.smile"), Plug("morph2.sad")])
```

To and from a `cgmath` file: `save` writes the `MorphList` (each target's
sparse offsets and its name; not its weight nor its in-betweens).
`BlendShape.load(path, base, ...)` from the class creates, as `create` does (a
second load stacks a second blendShape); a blendShape's own `load(path)` sets
the file's targets by name in place (a missing one added, its others kept).
The file comes first, then the geometry:

```python
import os
import tempfile

saved = morph.save(os.path.join(tempfile.mkdtemp(), "base_morph.npz"))
morph.set_target_data("sad", (np.array([0]), np.array([[0, 5, 0]])))
morph.load(saved)                                                 # the file's targets, in place
print(morph.get_target_data("sad").indices)                       # [0 1]
again = BlendShape.load(saved, base, name="morph3")               # creates: a second blendShape on base
print(again.get_targets(), sorted(cmds.ls(type="blendShape")))    # ['smile', 'sad'] ['morph2', 'morph3']
try:
    BlendShape.load(base, saved)                                  # the geometry first
except TypeError as e:
    print(str(e).split(":")[0])                                   # BlendShape.load(targets, geometry) takes the file first, then the geometry
```

---

## 13. `ObjectSet`

```python
cmds.file(new=True, force=True)
cube  = Node(cmds.polyCube(name="cube", ch=False)[0])
mesh  = cube.get_shape()

group = ObjectSet.define("mySet")                                 # found, or made in the current namespace
print(repr(group), ObjectSet.define("mySet") == group)                                       # ObjectSet("mySet") True
group.add_members([cube, "cube.vtx[0:2]"])
print(group.get_members())                                                                   # [Transform("cube")] -- DAG nodes only
print(sorted((node, comp is None) for node, comp in group.get_members(as_components=True)))  # [(Transform("cube"), True), (Mesh("cubeShape"), False)]
print(sorted(cmds.sets("mySet", query=True)))                                                # ['cube', 'cube.vtx[0:2]'] -- set order is not stable, so sort

group.remove_members("cube.vtx[0:2]")
other = ObjectSet.create(name="other")
other.force_elements(cube)                                             # plain sets are not exclusive: cube is now in both
print(cmds.sets("mySet", query=True), cmds.sets("other", query=True))  # ['cube'] ['cube']
other.clear()
print(other.get_members())                                             # []

try:
    ObjectSet.define("cube")
except TypeError as e:
    print(e)                                                      # 'cube' is a transform, not an objectSet; Node('cube') is Transform("cube")
print(repr(ObjectSet.define("initialShadingGroup")))              # ShadingEngine("initialShadingGroup") -- a subtype passes, as the most derived class
```

---

## 14. `ShadingEngine` and the shaders

A shading engine is an `ObjectSet` with render wiring, and only
`cmds.sets(renderable=True)` builds one that accepts members — so
`ShadingEngine.create` does that. `for_material` finds the engine a
material feeds (or builds `<material>SG`), `assign` moves members in
(`forceElement`), and `get_face_members` reads per-face entries that the
inherited `get_members` drops.

```python
red = cmds.shadingNode("blinn", asShader=True, name="red")
print(ShadingEngine.for_material("red", create=False))            # None
sg = ShadingEngine.for_material("red")
print(repr(sg), sg.get_material(), sg.get_material_info(), sg.get_face_members())                             # ShadingEngine("redSG") red ['materialInfo1'] []
print(ShadingEngine.for_material("red") == sg, ShadingEngine.for_material("standardSurface1", create=False))  # True initialShadingGroup

sg.assign(["cube.f[0:1]"], touched=[mesh.get_long_name()])
print(sg.get_face_members())                                                                                  # [(Mesh("cubeShape"), array([0, 1]))]
print(Node("initialShadingGroup").get_face_members())                                                         # [(Mesh("cubeShape"), array([2, 3, 4, 5]))] -- Maya carved the rest
print(cmds.sets("redSG", query=True), sg.get_members())                                                       # ['cube.f[0:1]'] []

sg.assign([mesh.get_long_name()], touched=[mesh.get_long_name()])                                                        # whole object; touched= lets it delete the orphan groupIds
print(sg.get_face_members(), cmds.sets("redSG", query=True), cmds.ls(type="groupId"))                         # [(Mesh("cubeShape"), None)] ['cubeShape'] []
```

`normalise=True` collapses "every face" entries back to object level and
deletes the orphan `groupId` nodes Maya leaves behind:

```python
blue = ShadingEngine.create(name="blueSG")                        # wired, but feeds from nothing yet
print(blue.get_material())                                      # None
blue.set_material("lambert1")                                   # a node (its outColor) or a plug string
print(blue.get_material())                                      # lambert1

blue.assign(["cube.f[0:1]"], touched=[mesh.get_long_name()], normalise=True)
print(sg.get_face_members(), blue.get_face_members())           # [(Mesh("cubeShape"), array([2, 3, 4, 5]))] [(Mesh("cubeShape"), array([0, 1]))]
sg.assign(["cube.f[0:1]"], touched=[mesh.get_long_name()], normalise=True)
print(cmds.sets("redSG", query=True), cmds.ls(type="groupId"))  # ['cubeShape'] []
print(mesh.get_material_bindings())                             # red
```

The surface shaders are node classes, one per exact Maya type: `Lambert`,
`Blinn`, `Phong`, `PhongE`, `SurfaceShader`, `StandardSurface`,
`OpenPBRSurface`. They are siblings, so a blinn is a `Blinn` and never a
`Lambert`; `Material` is the reference that takes any surface shader, and
the class of a surface type with no class of its own (`anisotropic`, a
plug-in's shader). `define` and `create` build the network — the shader,
`<shader>SG` and its materialInfo — outside an active `with container()`
unless `container=True`; `.delete()` and `.rename()` act on the network,
and `cmds.delete` on one node. `rn.blinn()` makes the bare shader, with no
engine. In the DSL a shader node is a membership collection (`cube <<
gold`, `cube in gold`, `cube >> gold`: the root cheatsheet, section 36).

```python
from rig.bridges import nodes as rn
from rig.nodetypes import Blinn, Lambert, Material

gold = Blinn.define("gold", color=(1, 0.8, 0))                                 # made: gold, goldSG, its materialInfo
print(repr(gold), repr(gold.get_engine()), Blinn.define("gold") == gold)      # Blinn("gold") ShadingEngine("goldSG") True
print(repr(Blinn.create(name="gold")), repr(Node.create("blinn", name="b2").get_engine()))   # Blinn("gold1") ShadingEngine("b2SG")
print(repr(Node("red")), repr(Material("gold")), repr(Material.create(type="anisotropic", name="ani")))   # Blinn("red") Blinn("gold") Material("ani")
bare = rn.blinn(name="bare")
print(repr(bare), ShadingEngine.for_material("bare", create=False))           # Blinn("bare") None
try:
    Lambert("gold")
except NodeTypeError as e:
    print(e)                                       # 'gold' is a blinn, not a lambert; Node('gold') is Blinn("gold"); Material('gold') takes any surface shader; Blinn('gold').astype(Lambert) converts it

gold.rename("amber")
print(gold, gold.get_engine())                     # amber amberSG
gold.delete()                                      # amber, amberSG and its materialInfo
print(cmds.ls("amber*"), gold.is_valid())          # [] False
```

---

## 15. `DisplayLayer`

Exclusive, objects only, read from the node's own `drawOverride` input.
`defaultLayer` means "no layer": `for_node` answers `None` there.
`DisplayLayer` is also the DSL's `Layer` (`rig.Layer is DisplayLayer`): a
layer node on the right of `<<` or `in` is the membership collection
(`cube << layer`, `cube in layer`, `layer.remove(cube)`, whose shorthand is
`cube << -layer`: the root cheatsheet, section 35).

```python
layer = DisplayLayer.define("geometry")                           # empty; never becomes the current layer
print(repr(layer), layer.is_default(), DisplayLayer("defaultLayer").is_default())     # DisplayLayer("geometry") False True

layer.add_members(cube)                                                               # the node itself, never the subtree
print(layer.get_members(), DisplayLayer.for_node(cube), DisplayLayer.for_node(mesh))  # [Transform("cube")] geometry None

grp = Transform.create(name="grp")
cube.set_parent(grp)
layer.add_members([grp])
print(layer.get_members(), layer.get_members(no_recurse=False))   # [Transform("grp"), Transform("cube")] [Transform("grp"), Transform("cube")]

ref = DisplayLayer.define("ref")
ref.add_members(cube)                                                                            # exclusive: cube leaves geometry
print(DisplayLayer.for_node(cube), layer.get_members())                                          # ref [Transform("grp")]
layer.remove_members(cube)                                                                       # not ours: left alone
print(DisplayLayer.for_node(cube))                                                               # ref

ref.rename("reference")
ref.visibility.set(False)
print(DisplayLayer.find_all(), Node("cube").v >> None, cmds.getAttr("cube.overrideVisibility"))  # [DisplayLayer("defaultLayer"), DisplayLayer("geometry"), DisplayLayer("reference")] True False
ref.clear()
ref.delete()                                                                                     # members go back to defaultLayer
print(DisplayLayer.for_node(cube), cmds.objExists("reference"))                                  # None False
from rig import Layer
print(Layer is DisplayLayer, cube in layer, DisplayLayer.of(grp))                                # True False [DisplayLayer("geometry")]
try:
    DisplayLayer("defaultLayer").delete()
except TypeError as e:
    print(e)                                                      # 'defaultLayer' cannot be deleted: it is the layer of no layer
```

---

## 16. `Choice`

A `choice` node's plugs have no fixed type; `Choice` resolves
`Attribute.get_data_type()` from whatever is wired to the selected input.

```python
cmds.file(new=True, force=True)
cube = Node(cmds.polyCube(name="cube", ch=False)[0])
pick = Node.create("choice", name="pick")
print(type(pick).__name__, pick.output.get_data_type(), pick.input[0].get_data_type())   # Choice Tdata Tdata

pick.input[0] << cube.tx
pick.input[1] << cube.matrix
pick.input[2] << cube.get_shape().outMesh
print(pick.input[0].get_data_type(), pick.input[1].get_data_type(), pick.output.get_data_type())  # doubleLinear matrix doubleLinear
pick.selector.set(1)
print(pick.output.get_data_type(), len(pick.find_attr("output").get()))         # matrix 16
pick.selector.set(2)
print(pick.output.get_data_type(), type(pick.output.get()[0]).__name__)         # mesh MeshData -- traced back to the source shape
```

Its inputs are typed attributes that hold any data, as a `rig.Typed` one:
an input takes a plug whole, whatever it held (no shorthand node, no split
into channels), and a number as a value.

```python
before = set(cmds.ls())
pick.input[3] << cube.t                                                         # a double3, connected whole
pick.input[3] << cube.worldMatrix                                               # rewired across kinds
pick.input[4] << 2.0
print(set(cmds.ls()) - before, cmds.listConnections("pick.input[3]", plugs=True), pick.input[4].get())   # set() ['cube.worldMatrix'] 2.0
```

---

## 17. `Follicle`

`create_on_mesh` builds a follicle on a mesh at the UV under a reference
transform and wires its transform; `constrain` point / orient constrains
something to that transform.

```python
cmds.file(new=True, force=True)
cube  = Node(cmds.polyCube(name="cube", ch=False)[0])
probe = Transform.create(name="probe")
probe.t.set(0.25, 0.5, 0.25)

rivet = Follicle.create_on_mesh(cube, probe, name="rivet")
print(repr(rivet), rivet.get_parent(), np.round(rivet.get_parent().t.get(), 3))  # Follicle("rivetShape") rivet [0.25 0.5  0.25]
print(rivet.inputMesh.get_connected_attrs(src=True, dst=False))                  # [Attribute("cubeShape.outMesh")]

target = Transform.create(name="target")
rivet.constrain(target)
print(cmds.listRelatives("target", type="constraint"))  # ['target_pointConstraint1', 'target_orientConstraint1']
rivet.set_uv_values([0.1, 0.2])
print(rivet.parameterU.get(), rivet.parameterV.get())   # 0.1 0.2
```

---

## 18. `Reference`

`Reference.create(path, namespace)` references a file and returns the
reference node; `get_namespace()` and `get_file_path()` read
`cmds.referenceQuery`. Needs a scene on disk, so this block is not run by
the vetter. Maya keeps a reference node locked (`ref.unlock()` is refused:
unlocked, it could be deleted under
its reference). A referenced node's `lock()` / `unlock()` is refused too:
Maya records no reference edit for a `lockNode`, so the change would be
gone at the next open; a plug lock on it is a reference edit, and works.

<!-- notest -->
```python
import os, tempfile

cmds.file(new=True, force=True)
cmds.polyCube(ch=False, name="cube")
path = os.path.join(tempfile.mkdtemp(), "asset.ma").replace("\\", "/")
cmds.file(rename=path)
cmds.file(save=True, type="mayaAscii", force=True)
cmds.file(new=True, force=True)

ref = Reference.create(path, namespace="hero")
print(repr(ref), ref.get_namespace(), ref.get_file_path() == path)    # Reference("heroRN") hero True
print(ref.get_nodes()[:2], Reference.find_by_path(path))  # [Transform("hero:cube"), Mesh("hero:cubeShape")] [Reference("heroRN")]

again = Reference.create(path, namespace="hero")                  # Maya makes it hero1, path {1}
print(again.get_namespace(), again.get_file_path(copy_number=True).endswith("{1}"), again.get_file_path() == path)  # hero1 True True

print(ref.is_locked(), ref.lock() is ref)                         # True True -- a reference node stays locked
try:
    Node("hero:cube").lock()
except RuntimeError as e:
    print(str(e).split(" Maya")[0])                               # 'hero:cube' is referenced:
Node("hero:cube").tx.lock()                                       # a plug lock is a reference edit

ref.delete()                                                                                       # cmds.file(removeReference=True)
print(cmds.objExists("hero:cube"), Reference.find_by_path(path))                                   # False [Reference("heroRN1")]
```

---

## 19. Name helpers

Pure string functions; nothing here touches the scene. Each lives beside the
class that uses it: `dg_node` (short and clean names), `joint`
(`replace_suffix`) and `geometry` (component range strings).

```python
print(get_short_name("|grp|ns:cube"), get_clean_name("|grp|ns:cube"))    # ns:cube cube
print(replace_suffix("arm_L_jnt", "ctl"), replace_suffix("arm", "ctl"))  # arm_L_ctl arm_ctl
```

Component ids to the range strings Maya stores. `iter_component_ranges`
takes a flat, ordered list; `iter_component_tokens` sorts, dedupes and
handles the `(N, 2)` and `(N, 3)` forms of surfaces and lattices.

```python
print(list(iter_component_ranges("vtx", [1, 2, 3, 5, 7, 8])))                # ['vtx[1:3]', 'vtx[5]', 'vtx[7:8]']
print(list(iter_component_tokens("f", np.array([4, 6, 5, 4]))))              # ['f[4:6]']
print(list(iter_component_tokens("cv", [[1, 0], [1, 1], [2, 5]])))           # ['cv[1][0:1]', 'cv[2][5]']
print(list(iter_component_tokens("pt", [[0, 0, 0], [0, 0, 1], [1, 2, 3]])))  # ['pt[0][0][0:1]', 'pt[1][2][3]']
```

---

## 20. `cmds` results as typed nodes, and `Axis`

`maya.cmds` hands back names; `Node` turns one into its typed node. The
DSL's `rig.bridges.commands` does this for every command, returning nodes
and `List`s.

```python
cmds.file(new=True, force=True)
made = Node(cmds.polyCube(name="pc", ch=False)[0])
print(repr(made), type(made).__name__)                             # Transform("pc") Transform
print([Node(x) for x in cmds.listRelatives("pc", shapes=True)])    # [Mesh("pcShape")]
```

`Axis` names the mirror plane for `Mesh.mirror_map`, which takes cgmath's
`cgmath.geometry.mesh.Axis` too (the same `X` / `Y` / `Z`); anything else
is a `TypeError` before the map is read.

```python
print(Axis.X, Axis.Z.value)  # Axis.X 2
```

---

## 21. `plugins` — `load_plugin` and undo

`load_plugin` is a context manager: a plug-in bundled in `rig/nodetypes/plugins`
loads by full path (once per session: while its command is in `maya.cmds`,
nothing is queried), any other name by name from `MAYA_PLUG_IN_PATH`. With
`unload_on_exit=True` it unloads, when the block ends, only what it loaded;
Maya refusing (the plug-in still in use) is a warning.

```python
print(bundled_plugin_path("undoable_api_command").endswith("undoable_api_command.py"), bundled_plugin_path("nope"))   # True None
with load_plugin("undoable_api_command"):
    print(cmds.pluginInfo("undoable_api_command", query=True, loaded=True))   # True
try:
    with load_plugin("noSuchPlugin"):
        pass
except RuntimeError as e:
    print(str(e)[:44])                                            # Plug-in, "noSuchPlugin", was not found on MA
```

Maya puts an API edit (`MFnMesh.setPoints`, `MPlug.setDouble`, ...) on
the undo queue only through a registered command. The bundled plug-in
registers one, `rigUndoableAPICommand` (a name of rig's own), for any
object with `doIt` / `undoIt` / `redoIt`: `cmds.rigUndoableAPICommand(obj)`
runs `obj.doIt()` at once as one undo step, and
`cmds.rigUndoableAPICommand.run(obj)` is the same step without the
command's own chunk (inside a `rig.undo_chunk`, both join that chunk's
step). The object's rules:

* **Apply the change first, in `doIt`**: the call runs it. `doIt` checks
  the edit and keeps what `undoIt` needs before it changes anything; an
  exception out of `doIt` queues nothing.
* **`undoIt` puts the scene back, `redoIt` applies the change again.**
* **No `cmds` calls inside `undoIt` / `redoIt`**, and no `cmds` edit in
  `doIt`: API only. A `cmds` call in `undoIt` / `redoIt` is not recorded,
  and Maya replays the `cmds` edits `doIt` made on top of `redoIt` (a
  relative move lands twice).
* The object is handed over for that one call, and released when its step
  leaves the queue (a flush, a new scene).

rig's own API edits keep these rules and go through `.run(obj)`, one step
each: `Mesh.set_points`, the UV set edits (`add_uv_set`, `rename_uv_set`,
`delete_uv_set`, `set_uv_data`), the colour set edits (`add_color_set`,
`ColorSet.set_data`, `ColorSet.delete`) and `SkinCluster.set_weights`.
`Mesh.create` and `NurbsCurve.create` need no command: their shape rides a
recorded `createNode` transform. The DSL's connects and value sets use the
command too, one step per statement ([root CHEATSHEET section 38](../CHEATSHEET.md#38-undo)),
and rig loads the plug-in itself at its first edit of a session. Without
it, the edits above raise a `RuntimeError` that names the missing plug-in,
where rig looked and Maya's reason; the connects and sets take `cmds`.
While those steps are on the undo queue, Maya refuses to unload the
plug-in (a `load_plugin(..., unload_on_exit=True)` that loaded it warns) until
`cmds.flushUndo()`.

```python
from maya.api import OpenMaya as om

class SetX:
    """translateX set through the API, as one undo step."""

    def __init__(self, node, value):
        self.plug  = om.MSelectionList().add(f"{node}.translateX").getPlug(0)
        self.value = float(value)
        self.old   = None
        with load_plugin("undoable_api_command"):
            cmds.rigUndoableAPICommand(self)       # runs doIt now

    def doIt(self):
        self.old = self.plug.asDouble()            # keep what undoIt needs,
        self.plug.setDouble(self.value)            # then apply the change

    def undoIt(self):
        self.plug.setDouble(self.old)

    def redoIt(self):
        self.plug.setDouble(self.value)

cmds.undoInfo(state=True)
loc = cmds.spaceLocator(name="loc")[0]
SetX(loc, 5.0)
print(cmds.getAttr("loc.tx"))  # 5.0
cmds.undo()
print(cmds.getAttr("loc.tx"))  # 0.0
cmds.redo()
print(cmds.getAttr("loc.tx"))  # 5.0
cmds.undo()
cmds.redo()
print(cmds.getAttr("loc.tx"))  # 5.0
```

## 22. `AnimCurve` and the curve classes

`AnimCurve` is the class of every animation curve (`animCurve`, abstract)
and of the types with no class of their own (a driven key's
`animCurveU*`); `AnimCurveTL` / `TA` / `TU` / `TT` are one timed type each.
`AnimCurve("x")` refers to any curve, a typed class to its type only.
Called with values, `AnimCurve([...])` returns **keys**, the value
`plug << AnimCurve(...)` keys a plug with (the root
[CHEATSHEET, section 39](../CHEATSHEET.md#39-animation-curves)); a typed
class called with values is a `TypeError`, and `Cls.create(keys)` is the
spelling for "a curve of this type with these keys". Values and times are
in the UI's units, the curve's Maya attributes are keywords set after the
keys (so `postInfinity=` is set on every run of a `define`; the keys'
`post_infinity=` is compared, and keys a curve holds already write
nothing), and the selection is kept.

```python
from rig import AnimCurve, AnimCurveTL, AnimCurveTU, NodeTypeError

cmds.file(new=True, force=True)
cmds.undoInfo(state=True, infinity=True)
blink = AnimCurveTU.create([0, 1, 0], times=[0, 2, 4], name="blink")   # new, unconnected
print(repr(blink), blink.get_num_keys())                 # AnimCurveTU("blink") 3
print(repr(Node("blink")), repr(AnimCurve("blink")))     # AnimCurveTU("blink") AnimCurveTU("blink")
try:
    AnimCurveTL("blink")
except NodeTypeError as err:
    print(str(err).split("; ")[-1])                      # AnimCurve('blink') takes any curve
try:
    AnimCurveTU([0, 1])
except TypeError as err:
    print(str(err)[:44])                                 # AnimCurveTU([...]) refers to a curve by name
print(type(AnimCurve([0, 1])).__name__)                  # _Keys

lid = AnimCurveTU.define("lid", [0, 1], tangents="step", post_infinity="cycle_relative")   # made
print(cmds.keyTangent("lid", query=True, outTangentType=True), cmds.getAttr("lid.postInfinity"))   # ['step', 'step'] 4
cmds.flushUndo()
again = AnimCurveTU.define("lid", [0, 1], tangents="step", post_infinity="cycle_relative")   # found, the same keys
print(again == lid, cmds.undoInfo(query=True, undoQueueEmpty=True))   # True True
AnimCurveTU.define("lid", [0, 1, 0], tangents="step")    # found: its keys replaced in place
print(lid.get_num_keys(), cmds.undoInfo(query=True, undoName=True))   # 3 rig.define
```

```python
spin = AnimCurveTL.create([0, 10], times=[0, 24], name="spin")
keys = spin.get_keys()                                   # the keys as a value, UI units
print(keys.values, keys.times, keys.out_tangents)        # [ 0. 10.] [ 0. 24.] ['linear' 'linear']
print(spin.evaluate([0, 12, 24]), spin.evaluate(6))      # [ 0.  5. 10.] 2.5
spin.set_keys([0, 3], times=[0, 10], out_slopes=[0.5, np.nan])   # one step
print(cmds.undoInfo(query=True, undoName=True), round(spin.get_keys().out_slopes[0], 4))   # rig.AnimCurve.set_keys 0.5

cube = rc.polyCube(name="cube", constructionHistory=False)[0]
cube.t << AnimCurve([0, 1])                              # three animCurveTL
cube.v << blink.output                                   # share one curve: a plain connect
print(AnimCurve.of(cube.t))                              # List([AnimCurveTL("cube_translateX"), AnimCurveTL("cube_translateY"), AnimCurveTL("cube_translateZ")])
print(len(AnimCurve.of(cube)), AnimCurveTU.of(cube))     # 4 List([AnimCurveTU("blink")])
```

`set_keys` and a `define` that updates follow the rules of `plug <<
AnimCurve(...)` for the plugs the curve keys: a curve that keys a locked
plug refuses with `InjectionError` (even with the keys it holds; unlock
first), and the default out tangent is a step when every plug it keys is a
bool or an enum, so keys it holds write nothing.

```python
from rig import InjectionError
from rig.spec import lock

cube.tx << lock
try:
    AnimCurve.of(cube.tx)[0].set_keys([5, 6])
except InjectionError as err:
    print(str(err).split("Unlock first: ")[1].split(" (")[0])   # Plug('cube.translateX').unlock()
cube.tx.unlock()
blink.set_keys([1, 0, 1])                                # it keys cube.v, a bool: a step
print(cmds.keyTangent("blink", query=True, outTangentType=True))   # ['step', 'step', 'step']
```

---

## Where to go next

| Read | For |
|---|---|
| [`README.md`](README.md) | the map of classes, how `Node` resolves, when to reach for the typed methods, the verified behaviour |
| [`../README.md`](../README.md) | the `rig` DSL itself: `Node`, `Plug`, `<<` and `>>`, `List`, `rig.bridges` |
| [`../CHEATSHEET.md`](../CHEATSHEET.md) | every operator and top-level name of the DSL, runnable |
