# `rig.maya` — cheatsheet

Copy-paste recipes for the typed node layer under the `rig` DSL: every
class in `rig.maya.nodetypes`, the `Attribute` wrapper, the name helpers,
`pycmds`, the constants and the bundled undo plug-in. The blocks run top to
bottom as one script and share a namespace — the **Setup** block comes first,
and each section that wants a clean scene starts with `cmds.file(new=True, force=True)`.

Concepts, the resolution rules and the verified behaviour live in [`README.md`](README.md).

## Contents

| # | Section | Covers |
|---|---|---|
| — | [Setup](#setup) | mayapy / Maya bootstrap, the imports |
| 1 | [From the DSL to the typed layer](#1-from-the-dsl-to-the-typed-layer) | `node >> None`, method delegation, `Plug` vs `Attribute` |
| 2 | [`PyNode` — resolution and registration](#2-pynode--resolution-and-registration) | what a name resolves to, `create`, `find_all`, a custom node type |
| 3 | [`DGNode`](#3-dgnode) | identity, rename, namespace, attributes, connections, lifecycle |
| 4 | [`DAGNode`](#4-dagnode) | parents, children, shapes, bounding box, deformers |
| 5 | [`Attribute`](#5-attribute) | get / set, `>>` and `//`, typed arrays, multis, slicing with list keys, components |
| 6 | [`Transform`](#6-transform) | shapes, matrices, pivots, `duplicate_geometry`, `serialize`, hierarchies |
| 7 | [`Joint`](#7-joint) | skeleton traversal, duplicate / rename a chain, orients, skinclusters |
| 8 | [`Geometry` — component tags](#8-geometry--component-tags) | `injection_node`, add / set / query / serialize, `tag_references` |
| 9 | [`Mesh`](#9-mesh) | points, UV sets, `serialize` / `create`, paintable maps, colour sets, materials, transfer |
| 10 | [`NurbsCurve` and `NurbsSurface`](#10-nurbscurve-and-nurbssurface) | CV counts, points, `serialize`, 2-D tags |
| 11 | [`Deformer` and `SkinCluster`](#11-deformer-and-skincluster) | geometries, influences, weights as `(V, I)` arrays, `SkinData` round trips |
| 12 | [`BlendShape`](#12-blendshape) | targets by name or index, `MorphData` in and out, rebuild |
| 13 | [`ObjectSet`](#13-objectset) | `get_or_create`, members and components |
| 14 | [`ShadingEngine`](#14-shadingengine) | `for_material`, `assign`, `get_face_members` |
| 15 | [`DisplayLayer`](#15-displaylayer) | `get_or_create`, `for_node`, exclusive membership |
| 16 | [`Choice`](#16-choice) | data types that follow the selector |
| 17 | [`Follicle`](#17-follicle) | rivet a transform to a mesh |
| 18 | [`Reference`](#18-reference) | file references and their namespaces |
| 19 | [`SkeletonDeltaBlend` and `AnimReaderNode`](#19-skeletondeltablend-and-animreadernode) | plug-in node types |
| 20 | [`node_name` helpers](#20-node_name-helpers) | short / clean names, suffixes, component range strings |
| 21 | [`pycmds` and `constants`](#21-pycmds-and-constants) | `maya.cmds` returning typed nodes; the enums |
| 22 | [`plugins` — `load_plugin` and undo](#22-plugins--load_plugin-and-undo) | an undoable API command in six lines |

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
from rig.maya import constants, node_name, pycmds
from rig.maya.attribute import Attribute
from rig.maya.nodetypes import (
    BlendShape, Choice, DAGNode, DGNode, DisplayLayer, Follicle, Geometry, Joint,
    Mesh, NurbsCurve, NurbsSurface, ObjectSet, PyNode, Reference, ShadingEngine,
    SkinCluster, Transform,
)
from rig.maya.nodetypes.deformer import Deformer, tag_references
from rig.maya.nodetypes.mesh import ColorSet
from rig.maya.plugins import bundled_plugin_path, load_plugin

print(cmds.about(version=True))
```

`rig.maya` itself re-exports nothing: import the submodule you need.
`Deformer`, `tag_references` and `ColorSet` come from their own modules
(`rig.maya.nodetypes.deformer`, `rig.maya.nodetypes.mesh`); everything else
above comes from `rig.maya.nodetypes`.

---

## 1. From the DSL to the typed layer

A DSL `Node` wraps one typed node. `node >> None` hands it back, and any
method the DSL does not define is looked up on it, so most of the time you
never need the `>>`.

```python
cube = rc.polyCube(name="cube", ch=False)[0]  # a DSL Node
xf   = cube >> None                           # the typed node under it
print(repr(cube), repr(xf))                       # Node("cube") Transform("cube")
print(type(Node("cubeShape") >> None).__name__)   # Mesh

print(cube.get_shapes())                          # [Mesh("cubeShape")] -- delegated, typed result
print(cube.get_shape().get_materials())           # [DGNode("standardSurface1")]
print(Node("cubeShape").get_material_bindings())  # standardSurface1 -- one material, the node itself
```

Attribute access is where the two layers differ: the DSL returns a `Plug`
(operators build networks), the typed node returns an `Attribute`
(`get` / `set` / `connect`).

```python
print(repr(cube.t),     repr(xf.t))            # Plug("cube.translate") Attribute("cube.translate")
print(cube.tx >> None,  xf.tx.get())           # 0.0 0.0
print(Node(xf) == cube, xf == PyNode("cube"))  # True True
```

---

## 2. `PyNode` — resolution and registration

`PyNode(name)` is a factory: it walks `cmds.nodeType(name, inherited=True)`
from the most derived type down and returns the first registered class.
Anything unregistered is a `DAGNode` or a `DGNode`; a string with a `.` is
an `Attribute`; a uuid string works too.

```python
cmds.file(new=True, force=True)
rc.polyCube(name="cube", ch=False)
rc.sphere(name="ball")
rc.curve(point=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], name="crv")
cmds.spaceLocator(name="loc")
cmds.lattice("cube", name="ffd")
cmds.joint(name="jnt")

for name in ("cube", "cubeShape", "ballShape", "curveShape1", "locShape", "ffdLatticeShape",
             "ffd", "jnt", "perspShape", "time1", "lambert1", "initialShadingGroup",
             "defaultObjectSet", "defaultLayer"):
    print(f"{name:20s} {cmds.nodeType(name):14s} -> {type(PyNode(name)).__name__}")
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
# lambert1             lambert        -> DGNode
# initialShadingGroup  shadingEngine  -> ShadingEngine
# defaultObjectSet     objectSet      -> ObjectSet
# defaultLayer         displayLayer   -> DisplayLayer

print(repr(PyNode("cube.tx")))      # Attribute("cube.translateX")
print(PyNode(PyNode("cube").uuid))  # cube -- a uuid resolves too
```

`PyNode.create(type, ...)` routes to the registered class's `create`
(so `"skinCluster"` takes a geometry and influences, `"shadingEngine"` is
built wired), and falls back to `cmds.createNode` for the rest.
`find_all` only knows registered types.

```python
grp = PyNode.create("transform", name="grp")
jnt = PyNode.create("joint", name="jnt2", parent=grp)
md  = PyNode.create("multiplyDivide", name="md")
print(repr(grp), repr(jnt), repr(md))                                                           # Transform("grp") Joint("jnt2") DGNode("md")
print(jnt.get_parent())                                                                         # grp

print(PyNode.find_all("transform")[:2])                                                         # [Transform("ball"), Transform("crv")]
print(len(PyNode.find_all("transform")) < len(PyNode.find_all("transform", exact_type=False)))  # True -- joints join in
try:
    PyNode.find_all("multiplyDivide")
except NotImplementedError as e:
    print(e)                                       # Node type multiplyDivide not implemented

print(Transform.exists("jnt"), Joint.exists("jnt"))                          # False True
print(Transform.is_type("jnt"), Transform.is_type("jnt", exact_type=False))  # False True
```

A subclass with a `CUSTOM_NODE_TYPE` registers itself (that is the
`NodeMeta` metaclass at work). The type is stamped on the node as a locked
`__custom_node_type__` string, and `PyNode` reads it before anything else.

```python
class Control(Transform):
    CUSTOM_NODE_TYPE = "control"

hand = Control.create(name="hand_ctl")
print(repr(hand), hand.node_type, cmds.getAttr("hand_ctl.__custom_node_type__"))       # Control("hand_ctl") control control
print(repr(PyNode("hand_ctl")), repr(PyNode.create("control", name="foot_ctl")))       # Control("hand_ctl") Control("foot_ctl")
print(Control.find_all())                                                              # [Control("foot_ctl"), Control("hand_ctl")]
print(Transform.is_type("hand_ctl"), Transform.is_type("hand_ctl", exact_type=False))  # False True
try:
    Control("persp")
except ValueError as e:
    print(e)                                                                       # persp is not a control
```

---

## 3. `DGNode`

The base of everything. It holds an `MObject`, so it survives renames; it
prints as its name, so it drops straight into `maya.cmds`.

```python
cmds.file(new=True, force=True)
box = Transform.create(name="cube")
print(box.name, box.long_name, box.short_name, box.clean_name, box.node_type)  # cube |cube cube cube transform
print(box.is_valid, len(box.uuid), box.has_base_type("dagNode"))               # True 36 True

box.rename("box")
box.namespace = "asset"                    # creates the namespace when missing
print(box.name, box.namespace, box.clean_name)         # asset:box asset box
print(str(box), repr(box), cmds.getAttr(f"{box}.tx"))  # asset:box Transform("asset:box") 0.0
```

Attributes: `find_attr` is what `node.<name>` calls; `add_attr` /
`rename_attr` / `delete_attr` wrap `cmds` and hand back `Attribute`s.

```python
print(repr(box.find_attr("tx")), box.find_attr("nope", quiet=True), box.has_attr("tx"))   # Attribute("asset:box.translateX") None True
w = box.add_attr("weight", attributeType="double", keyable=True, min=0, max=1)
print(repr(w), w.data_type, w.is_dynamic, w.default_value)                        # Attribute("asset:box.weight") double True 0.0
print(repr(box.rename_attr("weight", "blend")), box.list_attr(userDefined=True))  # Attribute("asset:box.blend") [Attribute("asset:box.blend")]
print(box.delete_attr("blend"), box.has_attr("blend"))                            # True False

box.set_attrs(tx=1, ty=2, tz=3)
box.set_attrs(skip_missing=True, jointOrient=[1, 2, 3], rx=45)                    # a transform has no jointOrient: skipped
box.set_attrs(notes="built by rig")                                               # notes is added on first write
print(box.t.get(), box.rx.get(), box.notes.get())                                 # [(1.0, 2.0, 3.0)] 45.0 built by rig
```

Connections and lifecycle:

```python
md = PyNode.create("multiplyDivide", name="md")
box.tx     >> md.input1X
md.outputX >> box.ty
print(box.list_connections(source=False, destination=True, plugs=True))            # [Attribute("md.input1X")]
print(box.find_connected_nodes(), md.find_connected_nodes(node_type="transform"))  # [DGNode("md")] [Transform("asset:box")]

print(box.duplicate())                                                             # [Transform("box")]
copy = box.duplicate(name="copy", returnRootsOnly=True)[0]
copy.delete()
print(copy.is_valid)                                      # False
try:
    copy.name
except RuntimeError as e:
    print(e)                                              # copy already deleted!

s = ObjectSet.create(name="s1")
s.add_members(box)
box.remove_from_all_sets()
print(s.get_members())                                      # []
print(sorted([md, box]), {box: 1}[Transform("asset:box")])  # [Transform("asset:box"), DGNode("md")] 1
```

Equality is by class and name, hashing by long name — `Transform("x")`,
`PyNode("x")` and `DGNode("x")` compare equal only when they are the same
class, so a `ShadingEngine` never equals the `ObjectSet` wrapping the same node.

---

## 4. `DAGNode`

```python
cmds.file(new=True, force=True)
grp  = Transform.create(name="grp")
cube = pycmds.polyCube(name="cube", ch=False)[0]     # pycmds returns typed nodes directly
cube.set_parent(grp)
shape = cube.get_children(shapes=True)[0]

print(cube.get_parent(), cube.get_parent(1), cube.get_parents(), grp.get_parent())  # grp None [Transform("grp")] None
print(grp.get_children(), grp.get_children(allDescendents=True))                    # [Transform("cube")] [Mesh("cubeShape"), Transform("cube")]
print(shape.is_shape, cube.is_shape, shape.mdagpath.fullPathName())                 # True False |grp|cube|cubeShape
print(cube.name, cube.long_name)                                                    # cube |grp|cube

cube.set_parent(None)                                                               # to world
print(cube.get_parent(), cube.long_name)                                            # None |cube
try:
    shape.set_parent(None)
except RuntimeError as e:
    print(e)                                             # Cannot parent shapes to world.

bbox = cube.get_bounding_box()
print(bbox.min, bbox.max)                                # (-0.5, -0.5, -0.5, 1) (0.5, 0.5, 0.5, 1)
```

`get_children` takes every `cmds.listRelatives` flag. `get_deformers` is
`findRelatedDeformer` with typed results (tweak nodes skipped):

```python
cmds.cluster("cube")
print(cube.get_deformers(), shape.get_deformers(node_type="cluster", first_only=True))  # [Deformer("cluster1")] cluster1
print(repr(shape.injection_node), shape.injection_node.intermediateObject.get())        # Mesh("cubeShapeOrig") True
```

---

## 5. `Attribute`

An `Attribute` wraps an `MPlug` and subclasses `str`, so `cmds.setAttr(attr, ...)`
works unchanged. `>>` connects (force), `//` disconnects.

```python
cmds.file(new=True, force=True)
cube = pycmds.polyCube(name="cube", ch=False)[0]
mesh = cube.get_shape()

a = cube.tx
print(a, repr(a), a.name, a.full_name, a.node, isinstance(a, str))                                # cube.translateX Attribute("cube.translateX") translateX cube.translateX cube True
print(a.attribute_type, a.data_type, cube.t.data_type, cube.matrix.data_type)                     # doubleLinear doubleLinear double3 matrix

cube.t.set(1, 2, 3)
cube.t.set([4, 5, 6])                                                                             # a single sequence is unpacked for double3 & co
cube.tx.set(7)
print(cube.t.get(), cube.tx.get())                                                                # [(7.0, 5.0, 6.0)] 7.0
print(cube.t.num_children, repr(cube.t.translateX), repr(cube.t.child(1)), cube.tx.get_parent())  # 3 Attribute("cube.translateX") Attribute("cube.translateY") cube.translate
print(cube.tx.default_value, cube.t.default_value)                                                # 0.0 [0.0, 0.0, 0.0]

print(cube.tx.is_locked, cube.tx.is_keyable, cube.tx.is_channel_box, cube.tx.is_dynamic)          # False True False False
cube.tx.is_locked = True
print(cube.tx.is_locked)                   # True
cube.tx.is_locked = False

cmds.setAttr(cube.ty, 9)                                                                   # it is a str
print(cmds.getAttr(cube.ty), cube.ty == Attribute("cube.ty"), sorted([cube.tz, cube.tx]))  # 9.0 True [Attribute("cube.translateX"), Attribute("cube.translateZ")]
```

Connections:

```python
cube.tx >> cube.ty
print(cube.ty.get(), cube.ty.is_connected, cube.ty.is_free_to_change)                                      # 7.0 True False
print(cube.ty.get_connected_attrs(src=True, dst=False), cube.tx.get_connected_attrs(src=False, dst=True))  # [Attribute("cube.translateX")] [Attribute("cube.translateY")]
print(cube.tx.list_connections(plugs=True), cube.tx.find_connected_nodes())                                # [Attribute("cube.translateY")] [Transform("cube")]

try:
    cube.tz.connect(cube.ty)               # connect() without force refuses an occupied input
except RuntimeError as e:
    print(str(e)[:52])                     # 'cube.translateY' already has an incoming connection
cube.tz.connect(cube.ty, force=True)
cube.tz // cube.ty
print(cube.ty.is_connected)                # False

cube.tx >> cube.ty
cube.tx >> cube.tz
print(cube.tx.break_connections(), cube.tx.is_connected)   # [Attribute("cube.translateZ"), Attribute("cube.translateY")] False
```

Typed data: `set()` fills in `type=` for you, and a `stringArray` /
`vectorArray` / `pointArray` takes one list instead of `(count, *items)`.

```python
weights = cube.add_attr("weights", dataType="doubleArray")
weights.set([0, 1, 0, 0, 4])
print(weights.get(), weights.is_typed, weights.data_type)  # [0.0, 1.0, 0.0, 0.0, 4.0] True doubleArray
print(weights.filter_array_values(0.0))                    # ([1, 4], [1.0, 4.0]) -- sparse ids and values

pts = cube.add_attr("pts", dataType="pointArray")
pts.set([[1, 2, 3, 1], [4, 5, 6, 1]])
print(pts.get())                                           # [(1.0, 2.0, 3.0, 1.0), (4.0, 5.0, 6.0, 1.0)]
names = cube.add_attr("names", dataType="stringArray")
names.set(["a", "b"])
print(names.get())                                         # ['a', 'b']

mode = cube.add_attr("mode", attributeType="enum", enumName="off:on:auto")
mode.set(2)
mode.default_value = 1                                     # dynamic attrs only
mode.add_category("rigging")
print(mode.get(), mode.enums, mode.default_value, mode.get_categories(), mode.has_category("rigging"))  # 2 ['off', 'on', 'auto'] 1.0 ['rigging'] True
print(cube.message.data_type, cube.worldMatrix.is_multi, cube.worldMatrix[0].data_type)                 # message True matrix
```

Multis: `[i]` is a logical index (created on access), a slice or a list of
ints gives a list of elements, `del attr[i]` removes one.

```python
multi = cube.add_attr("multi", attributeType="double", multi=True)
multi[0].set(1)
multi[3].set(3)
print(multi.is_multi, multi.num_elements, list(multi.get_logical_indices()), multi.get_next_available_index())  # True 2 [0, 3] 1
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
print(mesh.vtx, mesh.vtx._component_type, repr(mesh.vtx[0]))     # cubeShape.controlPoints kMeshVertComponent Attribute("cubeShape.controlPoints[0]")
print(mesh.vtx[[0, 3, -1]])                                      # [Attribute("cubeShape.controlPoints[0]"), Attribute("cubeShape.controlPoints[3]"), Attribute("cubeShape.controlPoints[7]")]
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
print(type(data).__name__, data.point_count, mesh.outMesh.data_type)  # MeshData 8 mesh
print(type(mesh.inMesh.get_data_fn_set()).__name__)                   # MFnMeshData
```

---

## 6. `Transform`

```python
cmds.file(new=True, force=True)
grp  = Transform.create(name="grp")
cube = pycmds.polyCube(name="cube", ch=False)[0]
cube.set_parent(grp)
grp.t.set(1, 2, 3)
cube.t.set(1, 0, 0)
cube.r.set(0, 90, 0)

print([a.name for a in cube.iter_xform_attrs(axis_only=True)])  # ['translateX', 'translateY', 'translateZ', 'rotateX', 'rotateY', 'rotateZ', 'scaleX', 'scaleY', 'scaleZ']
cube.set_xfrom_attrs_locked(True)                               # sic -- the method is spelled xfrom
print(cube.tx.is_locked)                                        # True
cube.set_xfrom_attrs_locked(False)

world = cube.get_matrix()                                        # MMatrix, world space by default
print(type(world).__name__, np.array(world).reshape(4, 4)[3, :3])         # MMatrix [2. 2. 3.]
print(np.array(cube.get_matrix(world_space=False)).reshape(4, 4)[3, :3])  # [1. 0. 0.]
print(type(cube.get_matrix(as_transform_matrix=True)).__name__)           # MTransformationMatrix

other = Transform.create(name="other")
other.match_matrix(cube, world_space=True)
print(other.t.get(), np.round(other.r.get(), 3))                         # [(2.0, 2.0, 3.0)] [[ 0. 90.  0.]]
other.set_matrix(np.eye(4).ravel().tolist())
print(other.t.get())                                                     # [(0.0, 0.0, 0.0)]

print(cube.get_rotate_pivot(), cube.get_scale_pivot(world_space=False))  # (2, 2, 3, 1) (0, 0, 0, 1)
cube.set_pivots((0.5, 0.5, 0.5), world_space=False)
cube.freeze(t=True, r=True, s=True)                                      # cmds.makeIdentity(apply=True, ...)
print(cube.t.get(), cube.r.get())                                        # [(0.0, 0.0, 0.0)] [(0.0, 0.0, 0.0)]
```

Shapes under a transform, and a clean duplicate of its geometry:

```python
print(cube.get_shapes(), cube.get_shape(), cube.get_shapes(no_interm=False))  # [Mesh("cubeShape")] cubeShape [Mesh("cubeShape")]
print(list(grp.iter_shapes("mesh")), grp.find_shape("mesh", "cu"))            # [Transform("cube")] cube -- transforms of matching shapes
print(list(grp.iter_shapes("mesh", as_transform=False)))                      # [Mesh("cubeShape")]

DisplayLayer.get_or_create("geo").add_members(cube)
clean = cube.duplicate_geometry(name="cube_clean")           # no history, no sets or layers, zeroed xform, pivots at origin
print(repr(clean), clean.get_parent(), clean.get_shapes(), clean.t.get())           # Transform("cube_clean") None [Mesh("cube_cleanShape")] [(0.0, 0.0, 0.0)]
print(DisplayLayer.for_node(clean), cube.duplicate_geometry(parent=grp).long_name)  # None |grp|cube2
```

Serialisation goes through `cgmath.hierarchy`: one node to a
`TransformData`, a whole branch to a `HierarchyData`. Only `transform` and
`joint` nodes (and a transform holding a locator) are captured; constraints
and the like are skipped with a warning.

```python
cube.add_attr("weight", attributeType="double").set(0.5)
data = cube.serialize()
print(type(data).__name__, data.name, data.node_type, data.user_defined_attributes)   # TransformData cube transform {'weight': {'attributeType': 'double', 'value': 0.5, 'keyable': False, 'channel_box': False}}

root  = Joint.create(name="root")
child = Joint.create(name="child", parent=root)
child.t.set(0, 5, 0)
pycmds.spaceLocator(name="probe")[0].set_parent(child)
skeleton = root.serialize_hierarchy()
print(type(skeleton).__name__, skeleton.name, [x.node_type for x in skeleton])   # HierarchyData ['root', 'child', 'probe'] ['joint', 'joint', 'locator']
```

`create_hierarchy` rebuilds it (needs `cgmath` 1.0.1 or later, for
`SUPPORTED_NODE_TYPES`) and returns the created names, parents first:

<!-- notest -->
```python
cmds.file(new=True, force=True)
made = Transform.create_hierarchy(skeleton)
print(made, cmds.getAttr("child.ty"), cmds.listRelatives("probe", shapes=True))   # ['root', 'child', 'probe'] 5.0 ['probeShape']
print(Transform.create_hierarchy(skeleton, parent=Transform.create(name="rig")))  # ['root1', 'root1|child', 'root1|child|probe']
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
print(np.round(hip.jo.get(), 3), np.round(hip.r.get(), 3))                              # [[0. 0. 0.]] [[ 0.    -0.    26.565]]
root.hierarchy_to_orients()
print(np.round(hip.jo.get(), 3), np.round(hip.r.get(), 3), np.round(knee.jo.get(), 3))  # [[ 0.    -0.    26.565]] [[0. 0. 0.]] [[  0.      0.    -26.565]]
hip.convert_orients_to_rotation()
print(np.round(hip.jo.get(), 3), np.round(hip.r.get(), 3))                              # [[0. 0. 0.]] [[ 0.    -0.    26.565]]
hip.convert_rotation_to_orients()
print(np.round(hip.r.get(), 3), np.round(hip.jo.get(), 3))                              # [[0. 0. 0.]] [[ 0.    -0.    26.565]]
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
print(np.round(ctl.get_children(type="joint")[0].t.get(), 3))    # [[3. 0. 0.]]
```

Which skinclusters a joint drives:

```python
geo  = pycmds.polyCube(name="geo", ch=False)[0]
skin = SkinCluster.create(geo, [root, hip])
print(hip.find_skinclusters(), knee.find_skinclusters(), root.find_skinclusters(recursive=True))  # [SkinCluster("geo_skincluster")] [] [SkinCluster("geo_skincluster")]
print(root.find_skel_blend())                                                                     # None -- no skeletonDeltaBlend upstream
```

---

## 8. `Geometry` — component tags

`Geometry` is the shape base (`Mesh`, `NurbsCurve`, `NurbsSurface`, and the
`DAGNode` fallback for lattices and locators). Its job is component tags.
Every editor writes to `injection_node`: the shape itself until a deformer
exists, the intermediate `Orig` shape from then on.

```python
cmds.file(new=True, force=True)
ball = pycmds.polySphere(name="ball", ch=False)[0].get_shape()
print(ball.injection_node == ball, ball.component_tags)                                                                            # True []

cmds.cluster("ball")
print(repr(ball.injection_node))                                                                                                   # Mesh("ballShapeOrig")

ball.add_component_tag("cap")                                                                                                      # returns the new multi index
ball.set_component_tag_contents("cap", [0, 1, 2, 3])                                                                               # ids, category "v" by default
print(ball.component_tags, ball.has_component_tag("cap"), ball.get_component_tag_index("cap"))                                     # ['cap'] True 0
print(ball.get_component_tag_contents("cap"), ball.get_component_tag_indices("cap"))                                               # ['vtx[0:3]'] [0 1 2 3]
print(ball.get_component_tag_category("cap"), ball.get_component_tag_name(0))                                                      # v cap
print(cmds.getAttr("ballShapeOrig.componentTags", multiIndices=True), cmds.getAttr("ballShape.componentTags", multiIndices=True))  # [0] None
print(ball.get_component_tag_contents("cap", full_name=True))                                                                      # ['ballShape.vtx[0:3]']
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
print(ball.component_tags)                                           # []

hist = pycmds.polyCube(name="hist")[0].get_shape()              # ch=True: the tags are polyCube1's outputs
print(hist.component_tags, hist.get_component_tag_index("top"), hist.get_component_tag_contents("top"))   # ['back', 'bottom', 'front', 'left', 'right', 'top'] -1 ['f[1]']
try:
    hist.set_component_tag_contents("top", [0], category="f")
except RuntimeError as e:
    print(str(e)[:64])                                           # Component tag 'top' on histShape is procedural (owned by polyCu
hist.add_component_tag("lid")   # a new name is editable next to them
print(hist.component_tags[:2])  # ['lid', 'back']
```

The plugs Maya evaluates tags on, and the component `MObject`s the
OpenMaya calls want:

```python
print(ball.local_shape_attr, ball.world_shape_attr)                                                           # ballShape.outMesh ballShape.worldMesh
print(ball.get_component_mobject().apiTypeStr, ball.get_component_mobject("f", np.array([0, 1])).apiTypeStr)  # kMeshVertComponent kMeshPolygonComponent
```

---

## 9. `Mesh`

```python
cmds.file(new=True, force=True)
cube = pycmds.polyCube(name="cube", ch=False)[0]
mesh = cube.get_shape()
print(repr(Mesh("cube")), Mesh("cubeShape") == mesh)                 # Mesh("cubeShape") True -- a transform name finds its mesh
print(mesh.num_vertices, mesh.num_polygons, mesh.num_weight_points)  # 8 6 8

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
print(mesh.uv_sets, mesh.current_uv_set, mesh.get_uv_coords().shape)   # ('map1',) map1 (14, 2)
counts, ids = mesh.get_assigned_uvs()
print(len(counts), len(ids), mesh.get_uv_at_point([0, 0, 0.5], uv_set=None)[:2])   # 6 24 (0.49666666984558105, 0.125)

uv = mesh.serialize_uv()                                         # a cgmath UVData
print(type(uv).__name__, uv.name, uv.points.shape)               # UVData map1 (14, 2)
uv.name = "map2"                                                 # set_uv_data renames the set to the data's name
mesh.add_uv_set("map2")
mesh.set_uv_data(uv, "map2")
mesh.rename_uv_set("uv2", "map2")
print(mesh.uv_sets)  # ('map1', 'uv2')
mesh.delete_uv_set("uv2")
print(mesh.uv_sets)  # ('map1',)
```

To and from `cgmath.geometry.MeshData`; `Mesh.create` is undoable and puts
the new shape in `initialShadingGroup`:

```python
data, uv_list = mesh.serialize()                                 # world space, with every UV set
print(type(data).__name__, data.point_count, data.face_count, type(uv_list).__name__, uv_list[0].name)  # MeshData 8 6 UVList map1
print(mesh.serialize(include_uvs=False, world_space=False).matrix.shape)                                # (4, 4) -- the parent's world matrix

rebuilt = Mesh.create(data, uv_data=uv_list, name="rebuilt")
print(repr(rebuilt), rebuilt.num_vertices, rebuilt.get_parent(), rebuilt.uv_sets)  # Mesh("rebuiltShape") 8 rebuilt ('map1',)
print(rebuilt.get_shading_engines())                                               # [ShadingEngine("initialShadingGroup")]
```

Paintable maps are `doubleArray` attributes in the `PaintableMap`
category, one value per vertex, padded or trimmed to the vertex count:

```python
mask = mesh.add_map("mask", values=0.5, category="weights")
print(repr(mask), mask.data_type, mask.get_categories(), mesh.paintable_maps)  # Attribute("cubeShape.mask") doubleArray ['weights', 'PaintableMap'] ['mask']
mesh.set_map_values("mask", [1, 0, 1])                                         # short lists are padded with 0.0
print(mesh.get_map_values("mask"))                                             # [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mesh.set_map_values("mask", np.linspace(0, 1, 8))
print(np.round(mesh.get_map_values("mask"), 2))                                # [0.   0.14 0.29 0.43 0.57 0.71 0.86 1.  ]

map_data = mesh.get_map_data("mask")                             # a cgmath MapData, sparse: zeros dropped
print(map_data.name, map_data.categories, map_data.indices)                                # mask ['weights'] [1, 2, 3, 4, 5, 6, 7]
print(repr(mesh.duplicate_map("mask", "mask2")), [m.name for m in mesh.serialize_maps()])  # Attribute("cubeShape.mask2") ['mask', 'mask2']
print(mesh.get_map_attrs(category="weights"), mesh.find_map_attr("visibility"))            # [Attribute("cubeShape.mask")] None
mesh.mirror_map("mask", constants.Axis.X)                                                  # +X values onto their -X partners
print(np.round(mesh.get_map_values("mask"), 2))                                            # [0.14 0.14 0.43 0.43 0.71 0.71 1.   1.  ]
mesh.delete_map("mask2")
print(mesh.paintable_maps, mesh.sanitize_map_values([1, 2]))                               # ['mask'] [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

Colour sets are `ColorSet` handles; `data` is an `(V, 4)` array:

```python
print(mesh.get_color_sets())                                     # []
cset = mesh.add_color_set("tint", ColorSet.Representation.RGBA)   # Alpha, RGB or RGBA
print(cset, cset.name, cset.mesh, cset.is_valid, cset.representation)   # ColorSet<cubeShape.tint>) tint cubeShape True Representation.RGBA
cset.data = np.tile([1.0, 0.0, 0.0, 1.0], (8, 1))
print(cset.data.shape, cset.data[0])         # (8, 4) [1. 0. 0. 1.]
cset.delete()
print(cset.is_valid, mesh.get_color_sets())  # False []
```

Materials, read from the shading engines the shape belongs to.
`get_material_bindings` returns the one material when it covers the whole
mesh, otherwise `(material, face ids)` pairs; `verbose=True` always gives
the pairs.

```python
print(mesh.get_materials(), mesh.get_shading_engines())  # [DGNode("standardSurface1")] [ShadingEngine("initialShadingGroup")]
print(mesh.get_material_bindings())                      # standardSurface1
print(mesh.get_material_bindings(verbose=True))          # [(DGNode("standardSurface1"), array([0, 1, 2, 3, 4, 5]))]

red = cmds.shadingNode("blinn", asShader=True, name="red")
ShadingEngine.for_material(red).assign(["cube.f[0:1]"], touched=[mesh.long_name])
print(sorted(mesh.get_materials()), sorted(mesh.get_shading_engines()))                # [DGNode("red"), DGNode("standardSurface1")] [ShadingEngine("initialShadingGroup"), ShadingEngine("redSG")]
print(mesh.get_material_bindings())                                                    # [(DGNode("red"), array([0, 1])), (DGNode("standardSurface1"), array([2, 3, 4, 5]))]
print(mesh.get_materials(as_pairs=True)[0][0].get_material() in mesh.get_materials())  # True -- (engine, material) pairs
```

Transfer maps, tags and skin weights onto another topology (a
`cgmath` `MeshDataResampler` through the first UV set). The map and tag
transfers land on the target but currently carry every vertex at `1.0`
— the map rides through a one-influence `SkinData` whose rows are
normalised — see *Real behaviour, verified* in [`README.md`](README.md).

```python
dst = pycmds.polyCube(name="dst", ch=False)[0].get_shape()
mesh.transfer_maps("mask", dst)
print(dst.paintable_maps, np.round(dst.get_map_values("mask"), 2))   # ['mask'] [1. 1. 1. 1. 1. 1. 1. 1.] -- every vertex, not the gradient

mesh.add_component_tag("cap")
mesh.set_component_tag_contents("cap", [0, 1, 2, 3])
mesh.transfer_component_tags("cap", dst)                             # written at dst's injection node
print(dst.component_tags[-1], dst.get_component_tag_indices("cap"))  # cap [0 1 2 3 4 5 6 7] -- every vertex again

jnt  = Joint.create(name="jnt")
skin = SkinCluster.create(cube, jnt)
print(repr(dst.apply_skin_data(skin.serialize())))                   # SkinCluster("dstShape_skincluster") -- creates or updates
```

---

## 10. `NurbsCurve` and `NurbsSurface`

Both are `Geometry`, so the tag methods above apply; `POINT_COMP_TYPE` is
`cv`, and a surface's ids are `(N, 2)` `(u, v)` pairs.

```python
cmds.file(new=True, force=True)
crv = pycmds.curve(point=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)], name="crv").get_shape()
print(repr(crv), crv.num_cvs, crv.num_weight_points, np.array(crv.get_points())[1, :3])   # NurbsCurve("curveShape1") 5 5 [1. 0. 0.]
spline = crv.serialize()                                          # cgmath BSplineData
print(type(spline).__name__, spline.points.shape, spline.degree, spline.periodic)  # BSplineData (5, 3) 3 False
print(crv.cv, len(crv.cv[:]), crv.cv[[0, 2]])                                      # curveShape1.controlPoints 5 [Attribute("curveShape1.controlPoints[0]"), Attribute("curveShape1.controlPoints[2]")]

crv.add_component_tag("root")
crv.set_component_tag_contents("root", [0, 1])
print(crv.get_component_tag_contents("root"))                                      # ['cv[0:1]']

ball = pycmds.sphere(name="ball")[0].get_shape()
print(repr(ball), ball.num_cvs, ball.num_weight_points)           # NurbsSurface("ballShape") 77 56 -- periodic in V: 3 wrapped rows per column
patch = ball.serialize()                                          # cgmath BSplinePatchData, wrapped CVs trimmed
print(type(patch).__name__, patch.points.shape, patch.periodic_u, patch.periodic_v)            # BSplinePatchData (7, 8, 3) False True

ball.add_component_tag("rim")
ball.set_component_tag_contents("rim", np.array([[1, 0], [1, 1]]))
print(ball.get_component_tag_contents("rim"), ball.get_component_tag_indices("rim").tolist())  # ['cv[1][0:1]'] [[1, 0], [1, 1]]
print(ball.get_component_tag_indices("rim").shape)                                             # (2, 2)
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
cube = pycmds.polyCube(name="cube", ch=False, height=2)[0]

skin = SkinCluster.create(cube, [j1, j2])                          # cmds.skinCluster(toSelectedBones=True), replaces any existing one
print(repr(skin), skin.get_influence_objects(), skin.get_mesh())           # SkinCluster("cube_skincluster") [Joint("j1"), Joint("j2")] cubeShape
print(skin.get_geometries(), skin.get_original_geometries())               # [Mesh("cubeShape")] [Mesh("cubeShapeOrig")]
print(repr(PyNode("cube_skincluster")), cube.get_shape().get_deformers())  # SkinCluster("cube_skincluster") [SkinCluster("cube_skincluster")]

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
print(skin.get_influence_objects(), skin.serialize() == data)                           # [Joint("j1"), Joint("j2")] True
print(repr(PyNode.create("skinCluster", pycmds.polyCube(name="c2", ch=False)[0], j1)))  # SkinCluster("c2_skincluster")
```

Transfer to another mesh (pass the shape or a name, not a `Transform`):

```python
other = pycmds.polyCube(name="other", ch=False, height=2)[0].get_shape()
moved = skin.transfer_to_mesh(other)
print(repr(moved), moved.get_influence_objects(), moved.get_weights().shape)   # SkinCluster("otherShape_skincluster") [Joint("j1"), Joint("j2")] (8, 2)
```

---

## 12. `BlendShape`

Targets are addressed by alias or logical index (index is faster); offsets
go in and out as `cgmath` `MorphData` (sparse: `indices` + `offsets`).

```python
cmds.file(new=True, force=True)
base  = pycmds.polyCube(name="base", ch=False)[0]
smile = pycmds.polyCube(name="smile", ch=False)[0]
cmds.xform("smile.vtx[3]", ws=True, t=(10, 2, 3))

morph = BlendShape.create(smile, base, name="morph")              # cmds.blendShape(frontOfChain=True)
print(repr(morph), morph.num_targets, morph.get_targets(), morph.get_target_indices())                   # BlendShape("morph") 1 ['smile'] [0]
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
print(morph.weight[:], Node("morph2").weight[:])         # [Attribute("morph2.smile"), Attribute("morph2.sad")] PlugList([Plug("morph2.smile"), Plug("morph2.sad")])
```

---

## 13. `ObjectSet`

```python
cmds.file(new=True, force=True)
cube  = pycmds.polyCube(name="cube", ch=False)[0]
mesh  = cube.get_shape()

group = ObjectSet.get_or_create("mySet")                          # finds it, in the current namespace too
print(repr(group), ObjectSet.get_or_create("mySet") == group)                                # ObjectSet("mySet") True
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
    ObjectSet.get_or_create("cube")
except TypeError as e:
    print(e)                                                      # 'cube' exists and is a transform, not a objectSet
print(repr(ObjectSet.get_or_create("initialShadingGroup")))       # ShadingEngine("initialShadingGroup") -- PyNode picks the most derived class
```

---

## 14. `ShadingEngine`

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

sg.assign(["cube.f[0:1]"], touched=[mesh.long_name])
print(sg.get_face_members())                                                                                  # [(Mesh("cubeShape"), array([0, 1]))]
print(PyNode("initialShadingGroup").get_face_members())                                                       # [(Mesh("cubeShape"), array([2, 3, 4, 5]))] -- Maya carved the rest
print(cmds.sets("redSG", query=True), sg.get_members())                                                       # ['cube.f[0:1]'] []

sg.assign([mesh.long_name], touched=[mesh.long_name])                                                         # whole object; touched= lets it delete the orphan groupIds
print(sg.get_face_members(), cmds.sets("redSG", query=True), cmds.ls(type="groupId"))                         # [(Mesh("cubeShape"), None)] ['cubeShape'] []
```

`normalise=True` collapses "every face" entries back to object level and
deletes the orphan `groupId` nodes Maya leaves behind:

```python
blue = ShadingEngine.create(name="blueSG")                        # wired, but feeds from nothing yet
print(blue.get_material())                                      # None
blue.set_material("lambert1")                                   # a node (its outColor) or a plug string
print(blue.get_material())                                      # lambert1

blue.assign(["cube.f[0:1]"], touched=[mesh.long_name], normalise=True)
print(sg.get_face_members(), blue.get_face_members())           # [(Mesh("cubeShape"), array([2, 3, 4, 5]))] [(Mesh("cubeShape"), array([0, 1]))]
sg.assign(["cube.f[0:1]"], touched=[mesh.long_name], normalise=True)
print(cmds.sets("redSG", query=True), cmds.ls(type="groupId"))  # ['cubeShape'] []
print(mesh.get_material_bindings())                             # red
```

---

## 15. `DisplayLayer`

Exclusive, objects only, read from the node's own `drawOverride` input.
`defaultLayer` means "no layer": `for_node` answers `None` there.

```python
layer = DisplayLayer.get_or_create("geometry")                    # empty; never becomes the current layer
print(repr(layer), layer.is_default, DisplayLayer("defaultLayer").is_default)         # DisplayLayer("geometry") False True

layer.add_members(cube)                                                               # the node itself, never the subtree
print(layer.get_members(), DisplayLayer.for_node(cube), DisplayLayer.for_node(mesh))  # [Transform("cube")] geometry None

grp = Transform.create(name="grp")
cube.set_parent(grp)
layer.add_members([grp])
print(layer.get_members(), layer.get_members(no_recurse=False))   # [Transform("grp"), Transform("cube")] [Transform("grp"), Transform("cube")]

ref = DisplayLayer.get_or_create("ref")
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
try:
    DisplayLayer("defaultLayer").delete()
except TypeError as e:
    print(e)                                                      # 'defaultLayer' cannot be deleted: it is the layer of no layer
```

---

## 16. `Choice`

A `choice` node's plugs have no fixed type; `Choice` resolves
`Attribute.data_type` from whatever is wired to the selected input.

```python
cmds.file(new=True, force=True)
cube = pycmds.polyCube(name="cube", ch=False)[0]
pick = PyNode.create("choice", name="pick")
print(type(pick).__name__, pick.output.data_type, pick.input[0].data_type)   # Choice Tdata Tdata

cube.tx                  >> pick.input[0]
cube.matrix              >> pick.input[1]
cube.get_shape().outMesh >> pick.input[2]
print(pick.input[0].data_type, pick.input[1].data_type, pick.output.data_type)  # doubleLinear matrix doubleLinear
pick.selector.set(1)
print(pick.output.data_type, len(pick.output.get()))                            # matrix 16
pick.selector.set(2)
print(pick.output.data_type, type(pick.output.get()[0]).__name__)               # mesh MeshData -- traced back to the source shape
```

---

## 17. `Follicle`

`create_on_mesh` builds a follicle on a mesh at the UV under a reference
transform and wires its transform; `constrain` point / orient constrains
something to that transform.

```python
cmds.file(new=True, force=True)
cube  = pycmds.polyCube(name="cube", ch=False)[0]
probe = Transform.create(name="probe")
probe.t.set(0.25, 0.5, 0.25)

rivet = Follicle.create_on_mesh(cube, probe, name="rivet")
print(repr(rivet), rivet.get_parent(), np.round(rivet.get_parent().t.get(), 3))  # Follicle("rivetShape") rivet [[0.25 0.5  0.25]]
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
reference node; the properties read `cmds.referenceQuery`. Needs a scene
on disk, so this block is not run by the vetter.

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
print(repr(ref), ref.namespace, ref.file_path == path)    # Reference("heroRN") hero True
print(ref.get_nodes()[:2], Reference.find_by_path(path))  # [Transform("hero:cube"), Mesh("hero:cubeShape")] [Reference("heroRN")]

again = Reference.create(path, namespace="hero")                  # Maya makes it hero1, path {1}
print(again.namespace, again.file_path_with_copy_number.endswith("{1}"), again.file_path == path)  # hero1 True True

ref.delete()                                                                                       # cmds.file(removeReference=True)
print(cmds.objExists("hero:cube"), Reference.find_by_path(path))                                   # False [Reference("heroRN1")]
```

---

## 19. `SkeletonDeltaBlend` and `AnimReaderNode`

Two plug-in node types. `SkeletonDeltaBlend` sets `PLUGIN_NAME`, so
`create()` loads the `SkeletonDeltaBlend` plug-in first (from
`MAYA_PLUG_IN_PATH`); `AnimReaderNode` (`rig.maya.nodetypes.anim_reader`)
is a plain wrapper, not a `DGNode`, for the `AnimReader` plug-in. Neither
plug-in ships with `rig`, so these blocks are not run by the vetter.

<!-- notest -->
```python
from rig.maya.nodetypes import SkeletonDeltaBlend

cmds.file(new=True, force=True)
root  = Joint.create(name="root_joint")
child = Joint.create(name="child_joint", parent=root)
ref   = root.duplicate_skeleton(suffix="ref")
anim  = root.duplicate_skeleton(suffix="anim")

blend = SkeletonDeltaBlend.create(root, ref_root=ref, anim_root=anim)
print(blend.num_targets, blend.get_reference_root(), blend.get_output_skel_root())   # 0 root_ref root_joint

pose = root.duplicate_skeleton(suffix="pose")
pose.get_children(type="joint")[0].t.set(1, 2, 3)
blend.add_target(pose, "lean")
print(blend.get_targets(), blend.get_target_index("lean"))                      # ['lean'] 0
print(list(blend.get_target_matrices("lean", world_space=True))[:1])            # [Joint("child_joint")]
blend.set_target_weight("lean", 0.5)
blend.empty_target_from_reference("rest")
blend.set_target_name("rest", "idle")
blend.remove_target("lean")
print(blend.get_targets(), SkeletonDeltaBlend.from_output_skel(root) == blend)  # ['idle'] True
```

<!-- notest -->
```python
from rig.maya.nodetypes.anim_reader import AnimReaderNode

reader = AnimReaderNode.create(name="clipReader")               # loads the AnimReader plug-in
reader.set_animation_file_paths(["/clips/walk.anm"])
reader.connect_time_node()
print(reader.get_animation_info())
```

---

## 20. `node_name` helpers

Pure string functions; nothing here touches the scene.

```python
print(node_name.get_short_name("|grp|ns:cube"),     node_name.get_clean_name("|grp|ns:cube"))  # ns:cube cube
print(node_name.get_suffix("arm_L_jnt"),            node_name.strip_prefix("arm_L_jnt"))       # jnt L_jnt
print(node_name.replace_suffix("arm_L_jnt", "ctl"), node_name.replace_suffix("arm", "ctl"))    # arm_L_ctl arm_ctl
print(node_name.has_pattern("shot_abc00010_v1"),    node_name.has_pattern("cube"))             # True False -- default pattern [a-z]{3}\d{5}
```

Component ids to the range strings Maya stores. `iter_component_ranges`
takes a flat, ordered list; `iter_component_tokens` sorts, dedupes and
handles the `(N, 2)` and `(N, 3)` forms of surfaces and lattices.

```python
print(list(node_name.iter_component_ranges("vtx", [1, 2, 3, 5, 7, 8])))                # ['vtx[1:3]', 'vtx[5]', 'vtx[7:8]']
print(list(node_name.iter_component_tokens("f", np.array([4, 6, 5, 4]))))              # ['f[4:6]']
print(list(node_name.iter_component_tokens("cv", [[1, 0], [1, 1], [2, 5]])))           # ['cv[1][0:1]', 'cv[2][5]']
print(list(node_name.iter_component_tokens("pt", [[0, 0, 0], [0, 0, 1], [1, 2, 3]])))  # ['pt[0][0][0:1]', 'pt[1][2][3]']
```

---

## 21. `pycmds` and `constants`

`pycmds` wraps every `maya.cmds` function so that any node name in the
result comes back as a typed node (the DSL's `rig.bridges.commands` is the
same idea returning `Node` / `PlugList`).

```python
cmds.file(new=True, force=True)
made = pycmds.polyCube(name="pc", ch=False)
print(made, [type(x).__name__ for x in made])                                                    # [Transform("pc")] ['Transform']
print(pycmds.listRelatives("pc", shapes=True), pycmds.ls(type="mesh"), pycmds.getAttr("pc.tx"))  # [Mesh("pcShape")] [Mesh("pcShape")] 0.0
```

```python
print(constants.Axis.X, constants.Axis.Z.value)                                                                       # Axis.X 2
print(constants.Renderers.ARNOLD, constants.HardwareRenderingModes.SHADED)                                            # arnold 1
print(str(constants.ImageFormats.PNG), constants.ImageFormats.PNG.format_name, constants.ImageFormats.EXR.extension)  # png PNG exr
print(constants.EvaluationManagerModes.PARALLEL)                                                                      # paralell -- sic, the value carries a typo
```

---

## 22. `plugins` — `load_plugin` and undo

`load_plugin` is a context manager: it loads by name from
`MAYA_PLUG_IN_PATH`, and falls back to the copy bundled in `rig/maya/plugins`.

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

The bundled plug-in registers `runUndoableAPICommand`: hand it any object
with `doIt` / `undoIt` / `redoIt` and it runs inside one undo chunk. This
is how `Mesh.set_points`, `Mesh.create` and `SkinCluster.set_weights` get
their undo.

```python
class MoveX:
    def __init__(self, node, value):
        self.node, self.value, self.old = node, value, None
        with load_plugin("undoable_api_command"):
            cmds.runUndoableAPICommand(self)

    def doIt(self):
        self.old = cmds.getAttr(f"{self.node}.tx")
        self.redoIt()

    def redoIt(self):
        cmds.setAttr(f"{self.node}.tx", self.value)

    def undoIt(self):
        cmds.setAttr(f"{self.node}.tx", self.old)

cmds.undoInfo(state=True)
loc = cmds.spaceLocator(name="loc")[0]
MoveX(loc, 5.0)
print(cmds.getAttr("loc.tx"))  # 5.0
cmds.undo()
print(cmds.getAttr("loc.tx"))  # 0.0
cmds.redo()
print(cmds.getAttr("loc.tx"))  # 5.0
```

---

## Where to go next

| Read | For |
|---|---|
| [`README.md`](README.md) | the map of classes, how `PyNode` resolves, when to drop below the DSL, the verified behaviour |
| [`../README.md`](../README.md) | the `rig` DSL itself: `Node`, `Plug`, `<<` and `>>`, `PlugList`, `rig.bridges` |
| [`../CHEATSHEET.md`](../CHEATSHEET.md) | every operator and top-level name of the DSL, runnable |
