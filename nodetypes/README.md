# `rig.nodetypes` — the node classes

The object model the `rig` DSL is made of. Every node object is an instance
of one class from here: a `Transform`, a `Mesh`, a `SkinCluster`, a
`Blinn`, a `ShadingEngine`, or the plain `DGNode` / `DAGNode` fallback, all
under the root class `Node`, which is also the factory that picks the
class. Each class names its nodes three ways: `Cls("x")` refers to one that
exists, `Cls.define("x", ...)` finds or makes it, `Cls.create(name="x",
...)` makes a new one. A node
holds an OpenMaya handle (rename-safe, deletion-aware), prints as its name so
it drops into `maya.cmds` unchanged, and carries the DSL (`node.tx` plugs,
`<<`, `>>`) and the methods the operators do not cover: skin weights as
arrays, component tags, material bindings, hierarchy serialisation to
`cgmath`.

```python
from maya import standalone
try:
    standalone.initialize()          # running from mayapy; inside Maya this raises and is skipped
except Exception:
    pass
from maya import cmds
cmds.file(new=True, force=True)

from rig.bridges import commands as rc

cube = rc.polyCube(name="cube", ch=False)[0]  # a Transform
mesh = cube.get_shape()                       # a typed method, on the same object
print(repr(cube), (cube >> None) is cube, repr(mesh))  # Transform("cube") True Mesh("cubeShape")
print(mesh.get_materials(), mesh.get_num_vertices())   # [StandardSurface("standardSurface1")] 8
```

You rarely import from `rig.nodetypes` at all: `Node(x)` and the `rc` / `rn`
bridges already hand back these classes. You do when you want a class to
name a node of its type (`Mesh("cubeShape")`, which refuses anything but a
mesh), to define or create one (`Transform.define("rig")`,
`SkinCluster.create(...)`), a classmethod (`ShadingEngine.for_material`,
`DisplayLayer.for_node`), or a subclass of your own.

---

## Where to go next

| You want to... | Read |
|---|---|
| Copy-paste an example of every class and method here | [`CHEATSHEET.md`](CHEATSHEET.md) — runnable top to bottom |
| The DSL: `Plug`, `<<` / `>>`, the operators, `List`, `rig.bridges` | [`../README.md`](../README.md) · [`../CHEATSHEET.md`](../CHEATSHEET.md) |

---

## What's inside

```
rig/nodetypes/
├── __init__.py        re-exports Node, the classes below, Attribute and Axis (not Deformer, tag_references, ColorSet)
├── _base.py           Node (the root class and the factory), NodeMeta, the cast, the lookup rule,
│                      the custom-type stamp; Attribute -- the MPlug wrapper: get / set / connect,
│                      slicing, components
├── errors.py          NodeLookupError, NodeNotFoundError, AmbiguousNodeError, NodeTypeError
├── dg_node.py         DGNode                 entity          refer / define / create; get_short_name(), get_clean_name()
├── dag_node.py        DAGNode                dagNode
├── transform.py       Transform              transform
├── _bindpose.py       (private) the dagPose read of serialize_hierarchy(bindpose=...)
├── _files.py          (private) the file side of save / load: read and check a cgmath file, the
│                      two-form load (from the class it creates, from a node it updates)
├── joint.py           Joint                  joint           replace_suffix()
├── geometry.py        Geometry               geometryShape   (component tags); iter_component_ranges() / _tokens()
├── mesh.py            Mesh, ColorSet, Axis   mesh
├── nurbs.py           NurbsCurve, NurbsSurface
├── deformer.py        Deformer, tag_references()   geometryFilter
├── skincluster.py     SkinCluster
├── blendshape.py      BlendShape
├── object_set.py      ObjectSet              objectSet
├── shading_engine.py  ShadingEngine          shadingEngine
├── material_node.py   Material (any surface shader), Lambert, Blinn, Phong, PhongE, SurfaceShader,
│                      StandardSurface, OpenPBRSurface
├── display_layer.py   DisplayLayer           displayLayer
├── reference.py       Reference              reference
├── choice.py          Choice                 choice
├── anim_curve.py      AnimCurve (any curve; its call with values makes keys), AnimCurveTL,
│                      AnimCurveTA, AnimCurveTU, AnimCurveTT     animCurve, animCurveTL ...
├── follicle.py        Follicle               follicle
└── plugins/
    ├── __init__.py    load_plugin() context manager, bundled_plugin_path()
    └── undoable_api_command.py   the rigUndoableAPICommand plug-in (undo for API edits)
```

The public surface:

```python
from rig.nodetypes import (
    Node, Attribute, DGNode, DAGNode, Transform, Joint, Geometry, Mesh, NurbsCurve,
    NurbsSurface, SkinCluster, BlendShape, ObjectSet, ShadingEngine, DisplayLayer,
    Reference, Choice, Follicle, Axis,
    Material, Lambert, Blinn, Phong, PhongE, SurfaceShader, StandardSurface, OpenPBRSurface,
    AnimCurve, AnimCurveTL, AnimCurveTA, AnimCurveTU, AnimCurveTT,
    NodeLookupError, NodeNotFoundError, AmbiguousNodeError, NodeTypeError,
)
from rig.nodetypes.deformer import Deformer, tag_references
from rig.nodetypes.dg_node import get_clean_name, get_short_name
from rig.nodetypes.geometry import iter_component_ranges, iter_component_tokens
from rig.nodetypes.joint import replace_suffix
from rig.nodetypes.mesh import ColorSet
from rig.nodetypes.plugins import load_plugin
```

---

## The class map

| Class | Wraps | What it adds |
|---|---|---|
| `Node` | every node (the root) | the factory: `Node(x)`, `Node.create(type, ...)`, `Node.define(type, name, ...)`, `Node.find_all(type)`, `Node.wrap`; `isinstance(x, Node)` holds for every node object |
| `DGNode` | any dependency node | the three verbs every class has: `Cls("x")` (refer), `Cls.define("x", ...)`, `Cls.create(name="x", ...)`, and `Cls.exists`, `Cls.find_all`; the DSL (`node.<attr>` plugs, `<<`, `>>`, `node.tx = 5`); `str(node)` / `get_name`, `get_long_name` / `get_short_name` / `get_clean_name`, `get_uuid`, `rename`, `get_namespace` / `set_namespace`, `find_attr` / `add_attr` / `delete_attr` / `rename_attr` / `list_attr`, `set_attrs`, `list_connections`, `find_connected_nodes`, `duplicate`, `delete`, `is_valid`, `remove_from_all_sets`, `lock` / `unlock` / `is_locked` |
| `DAGNode` | any DAG node | `get_parent(s)` / `iter_parents`, `get_children`, `set_parent`, `is_shape`, `get_mdagpath`, `get_bounding_box`, `get_deformers` |
| `Transform` | `transform` | `get_shape(s)`, `iter_shapes` / `find_shape`, `get_matrix` / `set_matrix` / `match_matrix`, pivots, `freeze`, `iter_xform_attrs`, `set_xfrom_attrs_locked`, `duplicate_geometry`, `serialize` / `Transform.create(transform_data)` (one node; `Joint.create(joint_data)`), `serialize_hierarchy` / `create_hierarchy`, `save_hierarchy` / `Transform.load_hierarchy` (files) |
| `Joint` | `joint` | `get_root_joint`, `get_parent_joint`, `iter_joints` / `find_joint`, `duplicate_skeleton`, `rename_skeleton`, `match_hierarchy`, `orient_joint` / `orient_chain`, orient <-> rotation conversions, `find_skinclusters` |
| `Geometry` | `geometryShape` | component tags: `get_injection_node`, `get_component_tags`, `add` / `remove` / `rename_component_tag`, `set_` / `get_component_tag_contents`, `get_component_tag_indices` / `_category` / `_data` / `_history`, `serialize_component_tags`, `get_local_shape_attr` / `get_world_shape_attr`, `get_component_mobject`, paintable attrs |
| `Mesh` | `mesh` | counts, `get_points` / `set_points`, closest point, normals, UV sets, `serialize` / `Mesh.create` (`MeshData` + `UVList`, or a `cgmath` `Object`; the data's hard edges and locked normals come back; `transform=`: the shape under an existing transform), `save` / `load` (files: `Mesh.load` creates, `Mesh('x').load` updates), paintable maps (`add_map`, `get_map_values`, `mirror_map`, `MapData`), colour sets (`ColorSet`), `get_materials` / `get_shading_engines` / `get_material_bindings`, `transfer_maps` / `transfer_component_tags`, `apply_skin_data` |
| `NurbsCurve`, `NurbsSurface` | `nurbsCurve`, `nurbsSurface` | `get_num_cvs`, `get_num_weight_points`, `get_points`, `serialize` (`BSplineData` / `BSplinePatchData`), `NurbsCurve.create` (points, or a `BSplineData`), `NurbsSurface.create` (a `BSplinePatchData`; both take `transform=`, as `Mesh.create`), `save` / `load` (files: the class creates, a node updates its CVs) |
| `Deformer` | `geometryFilter` | `get_geometries`, `get_original_geometries`; module function `tag_references(name)` |
| `SkinCluster` | `skinCluster` | `create(geo, influences | SkinData)`, influence add / remove / set, `get_weights` / `set_weights` as `(V, I)` arrays, `serialize` (`SkinData`), `save` / `load` (files: `SkinCluster.load(path, geo)` creates, `skin.load(path)` sets the weights), normalise / prune / max influences, `transfer_to_mesh`, `copy_skincluster`, `connect_bind_pre_matrices` |
| `BlendShape` | `blendShape` | `create(*geos_or_morphs)`, targets by name or index, weights, `get_target_data` / `set_target_data` (`MorphData`), `serialize` (`MorphList`), `save` / `load` (files: `BlendShape.load(path, base)` creates, `bs.load(path)` sets its targets by name), `add_empty_target`, `rebuild_target` |
| `ObjectSet` | `objectSet` | `define`, `get_members(as_components=)`, `add_members` / `remove_members` / `force_elements` / `clear` |
| `ShadingEngine` | `shadingEngine` | `create` (wired through `cmds.sets(renderable=True)`), `for_material`, `get_material` / `set_material`, `get_face_members`, `x << sg`, `remove(x)` (the DSL's `x << -sg`); the raw `add_members` / `remove_members` / `force_elements` and `assign` raise, naming `x << sg` / `sg.remove(x)` |
| `Material` | any surface shader (`shader/surface`), and the types with no class below | `create` / `define` build the network (the shader, `<shader>SG`, its materialInfo; `type=` for a type with no class), `get_engine`, `of`, `remove(x)` (the DSL's `x << -mat`), `astype` (converts the object in place), the network `rename` / `delete`, `find_all(exact_type=False)`; the DSL's materials |
| `Lambert`, `Blinn`, `Phong`, `PhongE`, `SurfaceShader`, `StandardSurface`, `OpenPBRSurface` | exactly `lambert`, `blinn`, `phong`, `phongE`, `surfaceShader`, `standardSurface`, `openPBRSurface` | everything `Material` has, for their one type |
| `DisplayLayer` | `displayLayer` | `define`, `for_node`, `of`, `is_default`, `get_members` / `clear` / `delete`, `x << layer` and `remove(x)` (the DSL's `x << -layer`), the one way in and out (`add_members` / `remove_members` raise, naming them); the DSL's `Layer` |
| `Reference` | `reference` | `create(path, namespace)`, `find_by_path`, `get_namespace`, `get_file_path(copy_number=False)`, `get_nodes`, `delete` |
| `Choice` | `choice` | `input[i]` takes any plug whole and a number (a typed attribute that holds any data, as a `rig.Typed` one: no shorthand node, no split into channels, whatever it held); `output` reads as what is selected; `get_data_type()` of both follows the wiring and the selector |
| `Follicle` | `follicle` | `create_on_mesh`, `constrain`, `set_uv_values` |
| `AnimCurve` | any animation curve (the abstract `animCurve`), and the types with no class below (a driven key's `animCurveU*`) | `AnimCurve.of(x)` (the curves driving a plug, a node, a `List`), `get_keys` / `set_keys`, `get_num_keys`, `evaluate`; called with values, `AnimCurve([...])` makes the keys `plug << ...` takes (a value, not a node); `create` / `define` are refused (abstract) |
| `AnimCurveTL`, `AnimCurveTA`, `AnimCurveTU`, `AnimCurveTT` | exactly `animCurveTL`, `animCurveTA`, `animCurveTU`, `animCurveTT` | everything `AnimCurve` has; `create(keys, ...)` / `define(name, keys, ...)`; called with values a `TypeError` naming `AnimCurve([...])` |

Each class inherits everything above it in its column, so a `Mesh` has
every `Geometry`, `DAGNode` and `DGNode` method, and a `ShadingEngine` is an
`ObjectSet`. The shader classes are siblings under `Material`: Maya derives
blinn from lambert, rig does not, so a blinn is a `Blinn` and never a
`Lambert`. The lookup errors of `errors.py` are one family, each also a
`TypeError` and a `ValueError`: `NodeNotFoundError` and
`AmbiguousNodeError` (both `NodeLookupError`), and `NodeTypeError`.

---

## Concepts

### One node, two attribute spellings

`Node` is the root of every class here and the DSL's factory: `Node("name")`
returns the typed node, `Node(node)` is `node` itself, and the DSL and the
typed API live on the same object. What differs is how you reach an
attribute.

| Spelling | Gives | Notes |
|---|---|---|
| `node >> None` | the node itself | |
| `node.get_shape()`, `node.serialize()`, ... | whatever the typed method returns | typed results |
| `node.tx` | a `Plug` owned by `node` | the DSL: `<<` sets or connects, the operators build networks, `get()` is numpy-shaped (`[1. 2. 3.]`) |
| `node.find_attr("tx")` | an `Attribute` owned by `node` | the typed plug: `get()` mirrors `cmds.getAttr` (`[(1.0, 2.0, 3.0)]`), `>>` connects, `//` disconnects |

A `Plug` is an `Attribute` subclass, so it has every method below; only its
operators and its `get()` differ. The two spellings of one plug are equal
and one dict or set key, and the typed methods hand back `Attribute`s
(`list_attr`, `list_connections(plugs=True)`). `Node(plug)` is the node the
plug was read from.

Both spellings follow one rule for a name: **a bare name is the Maya
attribute if one exists, otherwise the method; with brackets it is the
method.** Every public member of a node or plug class is a method, so
`node.get_name()` (or `str(node)`), `plug.get_node()` and `node.is_valid()`
are how you ask, and a Maya attribute is never hidden. The objects these
classes hand back that point into the scene follow it too: `Components`
(`faces.get_indices()`) and a `ColorSet` (`cset.get_data()`) read through
methods. The cgmath data that `serialize()` returns is a value: its fields
are attributes (`data.name`, `data.node_type`).

### Three verbs: refer, define, create

Every class names its nodes the same three ways, the
[construction rule](../README.md#the-construction-rule): naming never
creates, `define` makes sure, `create` makes new.

- **Refer, `Cls(x)`.** `x` is anything `Node(x)` takes. The result is the
  node `Node(x)` gives, when it is an instance of `Cls`: always the most
  derived class, so `Transform("j1")` is `Joint("j1")` and
  `ObjectSet("initialShadingGroup")` is the `ShadingEngine`, while
  `Lambert("b")` on a blinn is refused (the shader classes are siblings;
  `Material("b")` takes any shader). A reference never writes the scene. A
  missing name is `NodeNotFoundError` with a did-you-mean hint (the closest
  names of the class's node type; of any node for `DGNode`, the generic
  `Material` and a geometry class, which takes a transform too; and the
  class's `define`, when that makes the node from the name), a name two
  nodes have is `AmbiguousNodeError` (use a path, or spell the namespace),
  another type is `NodeTypeError`. `Cls(x, tx=1)` is a `TypeError` that
  names `define` and `create`; `Cls()` and `Cls(None)` are `TypeError`s
  too, except that a membership class called with no name is its kind
  (`Material()`, `DisplayLayer()`: every material, every layer).
  `Cls.exists(x)` is whether `Cls(x)` would return a node; it raises the
  `AmbiguousNodeError` a reference would.
- **Define, `Cls.define(name, *, parent=None, update=True, container=None,
  **attrs)`.** The name is a key: the node `create(name=name,
  parent=parent)` would make, spelled in full — the name's namespace or
  else the current one, the flattened `with container()` scope's prefix,
  and for a DAG class the path of `parent=` (the world without it). Found
  there, the node is returned (a subtype most derived, another type a
  `NodeTypeError`), never moved or reparented, and the attributes given are
  set on it: the script is the source of truth (`update=False` leaves a
  found node as it is). Missing there, it is made with the attributes,
  in one undo step (`rig.define`). A typo in an attribute name raises on a
  hit and on a miss.
- **Create, `Cls.create(*inputs, name=None, parent=None, container=None,
  **kwargs)`.** Always a new node; Maya picks the final name. A shape class
  built from data (`Mesh`, `NurbsCurve`, `NurbsSurface`) takes `transform=`,
  not `parent=`: the new shape under an existing transform.

`define` never makes a second node under a name you can already refer to,
and every refusal writes nothing:

| `define` is refused when | because | instead |
|---|---|---|
| no `parent=` is given and `Cls(name)` already finds the name elsewhere (under a group, or in another namespace) | a node at the world key would be a second one, and `Cls(name)` ambiguous | `parent=`, the namespace, or `Cls(name)` |
| the namespace does not exist | `define` never creates a namespace | `cmds.namespace(add=...)` |
| the namespace belongs to a file reference, loaded or not | Maya would make an unreferenced node inside the reference | define it outside the reference's namespace |
| the key is a namespace's name, or a DG node holds a DAG key's name | Maya would rename the new node | another name |
| inside `with container()`, the node found belongs to a container that is not on the scope's stack | a re-run in the same scene would split the rig ("'arm_root' belongs to container 'arm' from an earlier run; this scope is 'arm1' ...") | delete the old container, build in a new scene, or refer with `Cls(name)` |

If Maya still renames the node it made, `define` deletes exactly what the
call made (never `cmds.undo()`) and raises. A class built from data or
inputs (`Mesh`, `NurbsCurve`, `NurbsSurface`, `SkinCluster`, `BlendShape`,
`Reference`, `Follicle`) refuses `define`, naming its creator.
`Node.define(type, name, ...)` defines a node of any type by name
(`Node.define("multiplyDivide", "knee_md", operation="divide")`).

### `Node` is the factory

`Node(x)` never returns a bare `Node`. It resolves `x` (a name, uuid,
`MObject` or `MDagPath`; an attribute, an `MPlug` or a `"node.attr"` string
gives its node) and returns the most specific registered class:

1. a locked `__custom_node_type__` string attribute, when the node has one;
2. otherwise `cmds.nodeType(x, inherited=True)` walked from the most derived
   type down, the first registered type wins (a shader class takes only its
   exact type: a blinn is no `Lambert`);
3. otherwise `DAGNode` for DAG nodes, `Material` for a type Maya classifies
   a surface shader, `DGNode` for the rest.

So a `cluster` comes back as a `Deformer` (registered as `geometryFilter`),
a lattice shape or a locator as a `Geometry` (`geometryShape`), a camera as
a `DAGNode`, a blinn as a `Blinn` (an anisotropic as a `Material`). A node object is returned as it is
(`Node(x) is x`); anything that names no node raises. The attribute a
`"node.attr"` string names is `Attribute("node.attr")` (typed) or
`Plug("node.attr")` (DSL).

One lookup rule serves `Node(x)`, `Cls(x)`, `define` and the membership
operators: a path or a qualified `ns:name` is looked up exactly (from the
root namespace); a bare name at the root namespace and, while another
namespace is current, in that namespace too, both spelled absolutely, so
`namespace -relativeNames` changes nothing. Found in both is
`AmbiguousNodeError` ("spell the namespace"). A pattern (`"red*"`) is
refused: it is a search (`cmds.ls`, `Cls.find_all`).

Registration is the `NodeMeta` metaclass: any class that sets
`NATIVE_NODE_TYPE` (a Maya type) or `CUSTOM_NODE_TYPE` (a name of your
choosing) registers with the factory when the class statement runs. A
custom type stamps the string onto every node it creates, so `Node`
recognises those nodes later. `Transform.is_type(custom_node)` is `False`
at exact type and `True` with `exact_type=False`.

The metaclass also holds a node class to the one rule (a bare name on a node
is its Maya attribute if one exists, otherwise the method). A class body, and
a later `Cls.x = ...`, declares only methods under a public name; a
property, a nested class or lowercase data is refused when the class is
made. UPPER_CASE declarations (`NATIVE_NODE_TYPE`, `CUSTOM_NODE_TYPE`,
`FN_SET`, `POINT_COMP_TYPE`, `DEFAULT`, your own) are class data: read on
the class (`type(self).FN_SET` in a method), never on a node. Python state
is a `_` name the class body declares (`_side = None`); an undeclared `_`
name is the node's Maya attribute of that name, else an error.

`Node.create(type, ...)` runs the registered class's `create` with the
arguments when there is one — which is how `Node.create("skinCluster", geo,
joints)` ends up in `SkinCluster._create`, `"shadingEngine"` is built
wired and `"blinn"` builds the network (a surface type with no class of its
own is `Material.create(type=...)`) — and `createNode` for any other type.
A keyword that is no flag of the create is an attribute of the new node,
checked on the type before anything is made:
`Node.create("multiplyDivide", operation="divide", input1X=3)`. A class
that makes just its node (`transform`, `joint`, `choice`, ...) takes
keyword arguments only; one built from inputs (`skinCluster`, `blendShape`,
`mesh`, `nurbsCurve`, `reference`) raises a `TypeError` without them, before
anything is made; a display layer is empty unless objects are given.
`shared=` is refused: create always makes a new node. `Node.find_all(type)`
is the registered class's `find_all`, or the nodes `cmds.ls` lists for any
other type.

### Handles, not names

A node object holds an `MObject` (a DAG node also an `MDagPath`). Its name
(`str(node)`, `get_name()`) is read back from the handle each time, so the
object survives renames and reparenting; `is_valid()` says whether the node
still exists and any method
after deletion raises `RuntimeError("... already deleted!")`. A handle never
falls back to its name: a new node that takes the name is not picked up, and
an undo of the delete brings the old one back. A new scene, a file open or a
reference unload frees the node for good; the node object and its
attributes then raise `RuntimeError("Transform node (freed by a new
scene, a file open or a reference unload) already deleted!")`, naming the
class only, since the freed handle can no longer be read. An attribute built from a string or an
MPlug (`Attribute("held.ty")`, `Plug("held.ty")`) that never cast its node
keeps a handle of that node: deleted, it raises `"held already deleted!"`,
and freed, `"held node (freed by a new scene, a file open or a reference
unload) already deleted!"`, named by the name it was built with.
Every attribute read checks the handle first, `get_name()`, `get_mobject()`
and `get_fn_set()` included: the attribute of a dynamic attr is freed with its node.
An attribute built from the MPlug of a node already deleted raises
`"held already deleted!"` when it is built: that node has no name to take
a handle by.
The attribute of a dynamic attr can be freed while its node lives on, once
a delete of the attr leaves the undo queue (a flush, or ten more commands
at mayapy's default queue length): an attribute of a dynamic attr keeps a
handle of its attribute too, and then raises `"held.dyne already
deleted!"`, named by the name it was built with. So does an extension attr
found through its node, which `deleteExtension` frees at once.
`get_mplug()` raises the same once the node was freed: the raw MPlug then
points at freed memory, and reading it (`Plug(old.get_mplug())`) could crash
Maya.

The name is the one thing a deleted or freed node still answers: its `str`
is the last name it was known by, marked (`"held (deleted)"`, a node a
shader conversion replaced by the name it had), and its repr
`Transform("held", deleted)`; an attribute of one is marked the same way
(`"held.translateY (deleted)"`, `Attribute("held.translateY", deleted)`),
for everyone: rig reads names through a private reader, never through `str()`.
The marked name names no Maya node, so `maya.cmds` given the node, a plain
list holding it, or a `List` raises Maya's own error instead of crashing. A
`List` holding such an element raises on every rig operation (the List rule,
in the root cheatsheet's List section). rig's errors for a freed node name its class
(`Transform node (freed by ...) already deleted!`): the freed handle can no
longer be read, while the str is the name rig last read. A plug's str is the
text it was built with, so a raw `cmds` call given a held plug reaches
whichever node has that name now (rig's verbs check the node first).

A DAG node keeps the path it was taken through, and the attributes it finds
are named through that path: with `S` instanced under `T1` and `T2`,
`Node("|T2|S").find_attr("v")` is `T2|S.visibility`, while
`Attribute("|T2|S.v")`, built from a string, is named as Maya names it,
`T1|S.visibility`. When that instance is removed, the node answers through
another path until an undo brings its own back.

Equality is by name between node objects of the node's class, hashing is
the long name. Every spelling of one node gives that class —
`Transform("j1")`, `DGNode("j1")` and `Node("j1")` are all `Joint("j1")`,
and `ObjectSet("initialShadingGroup")` is the `ShadingEngine` — so two node
objects of one node are equal both ways and one dict key. A node is never
equal to a plug. A class called on a node object of another class takes
its node, so `Mesh(cube)` finds the transform's mesh shape as
`Mesh("cube")` does, and `copy.copy(node)` is an equal node object that
shares the handle.

### `create()` is final, `_create()` is the hook

`DGNode.create(*inputs, name=None, parent=None, container=None, **kwargs)`
calls the class's `_create` (which must return a node name), stamps a
custom type when there is one, casts the result and sets the attribute
keywords. `name=` and `parent=` are keywords; the positional arguments are
the inputs a class is built from. A shape class built from data (`Mesh`,
`NurbsCurve`, `NurbsSurface`) takes `transform=`, not `parent=`: its new
shape under an existing transform. A keyword in the class's
`_CREATE_FLAGS` goes to its command (a name that is both a flag and an
attribute, such as a skinCluster's `normalizeWeights`, is the flag); every
other keyword is an attribute, checked on the node type before anything is
made and set with `<<` once the node exists. `shared=` is refused.
Subclasses override `_create`, and that is where the signatures diverge:

| Call | Builds with |
|---|---|
| `DGNode` subclasses (`Choice.create(name=)`, ...) | `cmds.createNode` with `name` / `n` and `skipSelect` / `ss`; `parent=` or a positional argument is a `TypeError` |
| `Transform.create(name=, parent=)`, `Joint.create(...)` | `cmds.createNode(parent=...)`: in the parent's space, at identity; `parent=` is read by the reference rule; `Transform.create("x")` is a `TypeError` that names `Transform.create(name='x')` and `Transform.define('x')` |
| `Mesh.create(mesh_data, uv_data=None, *, name=, normals=True, transform=None, world_space=False)` | `MFnMesh.create` under a recorded `cmds.createNode` transform, one undo step; the data's points are the shape's own and its matrix goes on the transform (the source's world matrix, as `serialize` writes it by default); the data's normals bring back its hard edges and locked normals (`normals=False` skips that, for speed); a `cgmath.render.scene.Object` stands for the mesh, its UVs, its name and its transform. With `transform=`, the shape alone, under that transform; with `world_space=True` and no `transform=`, frozen at its world positions (both below) |
| `NurbsCurve.create(points_or_BSplineData, degree=, kv=, *, name=, transform=None, world_space=False)` | `MFnNurbsCurve.create` under a recorded `cmds.createNode` transform, one undo step; `degree` (3) and `kv` (Maya's default knots) go with points, a `BSplineData` brings its own, and its name and matrix as a `MeshData` does (on the `cgmath` that adds them). `transform=` and `world_space=` as `Mesh.create` takes them |
| `NurbsSurface.create(data, *, name=, transform=None, world_space=False)` | `MFnNurbsSurface.create` under a recorded `cmds.createNode` transform, in `initialShadingGroup`, one undo step; a `BSplinePatchData` (its points, degrees, forms and knots, `knots_u` / `knots_v`: evenly spaced when it holds none; its name and matrix as a curve's). `transform=` and `world_space=` as `Mesh.create` takes them |
| `SkinCluster.create(geo, influences_or_SkinData, **flags)` | `cmds.skinCluster(toSelectedBones=True)` with its create flags, the geometry's own existing skin deleted first (not a driving mesh's), one undo step; a `SkinData` whose weights do not fit the geometry is a `ValueError` before the delete |
| `BlendShape.create(*geometries_or_morphs, **flags)` | `cmds.blendShape(frontOfChain=True)` with its create flags |
| `ShadingEngine.create(name=)` | `cmds.sets(renderable=True, noSurfaceShader=True, empty=True)` |
| `Blinn.create(name=, ...)`, `Material.create(type=, name=, ...)` | `cmds.shadingNode(asShader=True)`, its `<shader>SG` engine and materialInfo, out of a scope unless `container=True` |
| `DisplayLayer.create(*objects, name=, empty= / noRecurse= / number= / makeCurrent=)` | `cmds.createDisplayLayer`, empty unless objects are given, which join as themselves (`noRecurse`) |
| `Reference.create(file_path, namespace)` | `cmds.file(reference=True)`; no `name=` |
| `AnimCurveTL.create(keys=None, times=None, *, <key options>, name=, **attrs)`, `TA` / `TU` / `TT` | `cmds.createNode` (the selection kept), then the keys through the API (`MFnAnimCurve`), then the attributes, one `rig.create` step; `keys` is values or an `AnimCurve(...)` value (then no times or options); `define(name, keys, ...)` replaces a found curve's keys when they differ (`update=True`) |

**A shape under an existing transform: `transform=`.** `Mesh.create`,
`NurbsCurve.create` and `NurbsSurface.create` (and their class-form `load`)
given `transform=` (by name or node) make the shape alone, under it: no
transform is made. `transform=` takes any node derived from transform: a
transform, a joint, an `ikHandle`, a constraint, a `dagContainer`, a plug-in's
`MPxTransform` control.

The shape is named `<transform>Shape` in the target's namespace, whatever the
current one (`transform='char:ctl'` makes `|char:ctl|char:ctlShape`; Maya
numbers it when a sibling has it: `ctlShape1`), or `name=` as given (Maya
numbers only a sibling's name: `name='ctl'` under `ctl` makes `|ctl|ctl`, and
`'ctl'` then names two nodes). Under a referenced target it is a local node in
the reference's namespace, which keeps its name through a save, a reopen, an
unload and a reload of the reference (should the referenced file come to hold
that name, Maya numbers the local one at the next open, and nothing is lost).
The shape is returned by its path under the target as the call named it
(`Mesh('ctl')` would read the transform's first mesh; a joint is no transform
to that lookup, so a shape under one is read by its own name). `world_space=`,
a keyword, says how the data's points go under it:

| `world_space=` | The points (a mesh's locked normals with them) |
|---|---|
| `False` (the default) | the shape's own, as the data holds them; its matrix (an `Object`'s transform) is not used |
| `True` | where the data puts them in the world, through its matrix (an `Object`'s transform; points, or B-spline data of a `cgmath` without the field, are world positions), taken into the transform's space: `data.matrix @ inverse(its world matrix)`, a mesh's locked normals through the inverse transpose (below) |

The shape rides the same recorded transform as without it, then goes under
the target (as `parent -shape -relative`, through the API: the selection and
the hilite list are never touched, whatever is selected), and the ride is
deleted and the shape named, in the create's one undo step: `cmds.undo()`
removes the shape and leaves the target as it was, `cmds.redo()` brings back
the same shape (its UUID), and the selection is kept through all three.
Under an instanced target, or without rig's undo plug-in, the move is Maya's
recorded `parent` (only the command puts the shape's other instances in its
shading group): the selection is put back, the hilite list is cleared. What
the ride gives comes too: `initialShadingGroup`, every UV set, the hard edges
and locked normals, holed faces, a curve's form and knots, a surface's knots.
The target is read before anything is made, by the reference rule: a missing
name is a `NodeNotFoundError`, a node that is no transform (a shape, a DG
node) a `NodeTypeError` (a `TypeError`) naming `transform=`. A locked target
is taken, as Maya's `parent` takes it, and so is a referenced one (the
`parent` is recorded as a reference edit). An instanced target gets the
shape under every instance: with `world_space=True` that is a `ValueError`
before anything is made, since one set of points cannot sit where the data
puts them under each. So is a singular target with `world_space=True`, naming
it and its world scale, X, Y, Z (a scale of 0 on it or on a parent, which Maya
keeps as 1e-12: its world 3x3 scales some direction by less than 1e-9, and the
shape's object points would come out about 1e12 times out; a rotated child of
a flat parent has no short axis, and the message adds that direction's scale);
`world_space=False` takes the data's points there, and a scale of 1e-6 is
taken either way. Under a mirrored target (a negative scale),
`world_space=True` puts the points where the data does, but leaves a surface
inside out against the source (its normals reversed), as
`NurbsSurface('x').load(path, world_space=True)` does into a mirrored
transform; every normal of a mesh locks (below). An error once the shape is
under the target (a value an attribute keyword refuses) deletes it.

**Frozen at its world positions: `world_space=True` without `transform=`.**
By default a new transform takes the data's matrix and the points stay the
shape's own. `world_space=True` puts the matrix in the points instead (a
mesh's locked normals through its inverse transpose, as under `transform=`) and
the new transform sits at the origin with identity channels: the same world
positions, frozen. So, for a `polyCube` named `box` moved to `(0, 10, 0)`, with
`data, uvs = Mesh('box').serialize()`:

| Call | The new transform | `vtx[0]`, object | `vtx[0]`, world |
|---|---|---|---|
| `Mesh.create(data, uvs, name='copyA')` | `t = (0, 10, 0)` | `(-0.5, -0.5, 0.5)` | `(-0.5, 9.5, 0.5)` |
| `Mesh.create(data, uvs, name='copyB', world_space=True)` | `t = (0, 0, 0)` | `(-0.5, 9.5, 0.5)` | `(-0.5, 9.5, 0.5)` |

The class-form `load` takes it the same way (`Mesh.load(path,
world_space=True)` is `copyB` from a file), and so do `NurbsCurve` and
`NurbsSurface`, their knots and forms kept. Data without a matrix (points,
B-spline data of a `cgmath` without the field) or at the identity (world-space
data) has nothing to freeze: the points stay as they are. An `Object`'s
transform goes in the points the same way; its rotate order, and a lone
`Object`'s visibility, which move nothing, still go on the new transform. From
a mirrored matrix (a negative scale) a surface comes out inside out, as under
a mirrored target. A matrix that is exactly singular (only data made by hand
holds one: Maya keeps 1e-12 for a scale of 0) has no inverse for a mesh's
normals: a `ValueError` before anything is made, and `normals=False` takes it.
A matrix that is not finite is a `ValueError` before anything is made, for
every class, under `transform=` too, and in `Mesh('x').load(path,
world_space=True)`.

**`world_space=True` locks a normal that differs on both sets of points**, under
a target, frozen, and in `Mesh('x').load(path, world_space=True)`. A `MeshData`
holds no lock flag, so a normal counts as locked when Maya computes it more
than 1e-4 from the data's on the data's own points, as a plain create decides
it, and where it lands too (what Maya computes there needs no lock). Only those
go through the matrix (its inverse transpose) and are set, which locks them;
every other normal is Maya's own for the points where they land, as
`makeIdentity` leaves them, and follows a deformer. The hard edges come through.
What the matrix changes:

| The matrix | Locked | Every other normal |
|---|---|---|
| The identity | as in a plain create | |
| A move, a rotation, a uniform scale | the source's locked normals | Maya's own, within float noise of the source's: at most 0.6 degrees on a 300 x 300 `polySphere` at `(100, 50, 30)`, none of its 89,702 normals locked |
| A non-uniform scale, a shear | the source's locked normals | a flat face's where the source's is; a soft one Maya's own, off the inverse transpose: at most 0.04 degrees on a 32 x 32 `polySphere` at a scale of `(1, 2, 1)`, 0.5 at `(1, 2, 3)`, 13 at a soft corner of a 2 x 2 x 2 `polyCube` under a shear |
| A mirror (a negative scale) | every normal (54 of 54 on a 2 x 2 x 2 `polyCube`): Maya's own would face the other way, and the world normals stay the source's | |

That is data saved in object space (`save()`, `serialize()`). World-space data
(`save(world_space=True)`) holds normals through the source's matrix, which
Maya does not compute from its points, so only the points they were computed
on tell them from locks. Landing back there (on the mesh it was saved from, or
under a target at its matrix) only the source's locks lock. Anywhere else,
frozen or under another target, the soft ones lock too where a non-uniform
scale, a shear or float noise moved them: 988 of 994 on a 32 x 32 `polySphere`
at a scale of `(1, 2, 3)`, about 54,000 of 89,702 on a 300 x 300 one at
`(100, 50, 30)`, none on a rigid move near the origin. So do normals computed
another way: a source on another
`vertexNormalMethod` than a new mesh's 3 (960 of those 994 in a plain create;
an update into the mesh itself computes as it does, and locks none).

A cluster leaves a locked normal as it is, and a skinCluster turns it with the
skin (`deformUserNormals`, on by default); neither recomputes it.
`normals=False` locks none, and loses the hard edges.

A class built from a `cgmath` file has a `load` beside its `create`:
`Mesh.load(path, ...)`, `NurbsCurve.load`, `NurbsSurface.load`,
`SkinCluster.load(path, geo, ...)`, `BlendShape.load(path, base, ...)` and
`Transform.load_hierarchy(path, ...)`, from the class, are the class's
`create` from a file: a new node on every call
([Serialisation and files](#serialisation-and-files)).

Inside `with container()` a typed create joins the scope as `Node.create`
does: an explicit `name=` takes the flattened scope's prefix, every node the
call made for itself is registered (`Mesh.create`'s transform and shape, with
`transform=` the shape alone, never the transform it goes under;
`SkinCluster.create`'s skin and bind pose, not the `Orig` shape it puts
under your mesh), `skipSelect` defaults to the `skip_selection` option, and
the call is one undo step. `container=False` leaves the nodes unregistered.
Display layers, sets (so shading engines) and references are scene
registries, found again by name: in a scope they stay out, unprefixed,
unless `container=True`. Outside a scope a direct typed create is plain
Maya: no prefix, and the new node is selected (through `Node.create` it is not, per `skip_selection`).

`ObjectSet.define(name)` and `DisplayLayer.define(name)` find the node at
its key (in the current namespace, where `create` puts a new one), refuse
with a `NodeTypeError` when the name belongs to another type, and create it
otherwise, as every class's `define` does.

### `Attribute` wraps an `MPlug` and subclasses `str`

`node.find_attr("tx")` is an `Attribute`: `get()` mirrors `cmds.getAttr`,
`set()` mirrors `cmds.setAttr` and fills in `type=` for typed data (a single
list also stands for the `(count, *items)` form of `stringArray` /
`vectorArray` / `pointArray`), and on an enum takes a field name as well
as the int (`find_attr("ro").set("zyx")`). Like `cmds.setAttr`, `set()` leaves
the input alone: a wired plug refuses it, and a keyed or constrained one
takes it until the next evaluation. The DSL's `plug << value` replaces
whatever drives the plug, then sets. `a >> b` connects with force, `a // b`
disconnects, `connect(other, force=False)` refuses an occupied input. The
DSL spelling, `node.tx`, is a `Plug`: the same methods, the DSL's
operators, a numpy-shaped `get()`.

`lock()` / `unlock()` run `setAttr -lock` and return the attribute (the
DSL's `plug << lock` is the same call); `is_locked()` is True for the plug's
own lock and for a locked parent's (a compound's for its children, a multi
root's for its elements). An attribute already in that state is left as it
is, with no undo step. Both refuse a plug of a locked node (Maya locks no
plug there: lock the plugs first, then the node), and `unlock()` a child
whose locked parent holds the lock. An attribute has no properties: every
member is a method (`attr.get_node()`, `attr.get_full_name()` or
`str(attr)`, `attr.is_multi()`, `attr.set_alias(x)`, `attr.set_keyable(b)`),
and `attr.is_locked = x` raises, naming `lock()` / `unlock()`; a predicate
named without brackets (`if attr.is_locked:`) raises TypeError, as it would
always pass. `DGNode.lock()` / `unlock()` /
`is_locked()` are the node's (`lockNode`), and a
locked node refuses `rename`, `delete`, `set_parent`, `add_attr`,
`delete_attr`, `rename_attr` and a new attribute spec with one
`InjectionError`: `cannot rename 'x': it is locked; Node('x').unlock()
unlocks it. Nothing was changed`. A Maya attribute named like a method
follows the rule of every node and plug: a bare name is the Maya attribute if
one exists, otherwise the method; with brackets it is the method. So
`Node("animLayer1").lock` is the plug of its `lock` attribute and
`Node("animLayer1").lock()` locks the node.

An `Attribute` holds its `MPlug`. After a disconnect, Maya can free the
`MPlug` of an array element (one of a multi that is not keyable, connected
and holding no value of its own), so `disconnect()`, `break_connections()`
and the DSL's `<<` find the element's `MPlug` again for the objects they
were handed (and the ones `break_connections()` returns). Another object
for the same element taken while it was connected, or one held across a
disconnect made outside rig, can dangle: take the attribute again.

Because it is a `str`, `cmds.setAttr(attr, 1)` just works, and `f"{attr}"`
is the full name. Equality and hashing are not the string's, though: two
attributes are equal, and hash alike, when they are the same Maya plug
(node, attribute, logical indices), whatever instance path each is named
through, and a rename or an alias keeps the hash. An attribute never equals
a plain string; compare `str(attr)` for names, or `plug.equals("a.tx")`. A
DSL `Plug` of the same plug is equal too (its `==` folds to `True` and
builds nothing) and hashes alike: the two spellings are one dict or set key.

An attribute a node finds (`node.find_attr("tx")`, and the `Plug`
`node.tx`) is owned by that node object: its `node` is `node`, and its
children, elements and parent share it.

Indexing is the multi / component surface: `attr[i]` is the element at a
logical index (created on access, Maya's own semantics), `attr[a:b]` and
`attr[[i, j, k]]` give lists of elements (a `List` of `Plug`s on a `Plug`),
`del attr[i]` removes one.
Geometry components (`mesh.vtx`, `curve.cv`, `surface.cv`) are
range-checked against the point count and accept negative ids in a list
key; `mesh.vtx`, `mesh.pnts` and `mesh.pt` are the same `controlPoints`
plug, and `mesh.cv` is an `AttributeError` because `cmds.listAttr` rejects
the alias on a mesh.

`get()` on a geometry-typed plug (`mesh`, `nurbsCurve`, `nurbsSurface`)
skips `cmds.getAttr`, which cannot serialise those. A shape's own plug
gives back the shape's `serialize()`, a plug with a downstream consumer
gives the consumer's, a `choice` output is traced through its selector,
and a dangling output is wrapped in `MFnMesh` / `MFnNurbsCurve`. A shape
serializes in the space of the plug its data comes from: an element of
`worldMesh` / `worldSpace` in world space, every other plug (`outMesh`,
`local`, a deformer's output, the plug a `choice` routes) in object space,
with its transform's world matrix, as `serialize()` writes it by default;
a consumer that is no shape (a `skinCluster`) serializes as it does.

### Component tags live at the injection node

Maya writes tags where `cmds.deformableShape(tagInjectionNode=True)` says:
the shape itself until a deformer exists, its intermediate `Orig` shape
from then on. `Geometry.get_injection_node()` answers that (uncached, since a
deformer moves it) and every editor writes there, so the shape you hold
and the tag editor never disagree. Readers go through the evaluated
geometry (`get_local_shape_attr()`), so procedural tags — a history `polyCube`'s
six face tags are outputs of `polyCube1` — are visible and serialisable,
but any edit to one raises.

Ids are native: `(N,)` on meshes and curves, `(N, 2)` `(u, v)` on
surfaces, `(N, 3)` on lattices, and a face tag gives face ids, not the
vertex cast `geometryAttrInfo -pointIndices` applies. Categories are the
strings `"v"`, `"e"`, `"f"`. `tag_references(name)` lists the deformer
inputs whose `componentTagExpression` names a tag, which is what makes a
rename or delete safe to refuse.

The DSL's `Tag` is built on exactly these calls.

### Shading engines are sets with render wiring

`cmds.createNode("shadingEngine")` makes a set with no `renderPartition`
slot, no `materialInfo` and no light-linker entries, and such a set refuses
every member. `ShadingEngine.create` therefore goes through `cmds.sets`.
`for_material(material)` finds the engine a material feeds through
`surfaceShader` (or builds `<material>SG` and lists the material in
`defaultShaderList1`, as the Hypershade does). `x << sg` (or `x << material`)
is `forceElement`, so a member leaves whatever engine held it; Maya carves a
whole-object membership down to the remaining faces but never collapses
per-face membership back (the faces moved back one by one stay
`cube.f[0:5]`); `cube << sg` of the whole object collapses it. `sg.assign` raises a TypeError naming `x << sg` (one way
only). `get_face_members`
reads through `MFnSet` so per-face entries survive (`get_members` drops them).

`Mesh.get_material_bindings()` is the mesh-side view: the one material when
it covers the whole mesh, otherwise `(material, face ids)` pairs. The DSL's
membership grammar (`x << red`, `Material.of(x)`) sits on `for_material` and a private
assign; `ShadingEngine.repair()` puts the shapes no engine holds (green) back in
`initialShadingGroup`, and `ShadingEngine.tidy()` collapses all-faces memberships to
object level. Both are scene-level class methods: called on an engine
(`sg.repair()`) they ignore it, so call them on the class.

### Shaders are node classes, one per exact type

`Lambert`, `Blinn`, `Phong`, `PhongE`, `SurfaceShader`, `StandardSurface`
and `OpenPBRSurface` each take exactly their Maya type, and `Material` takes
any type Maya classifies `shader/surface`: it is the class of `anisotropic`,
`rampShader` or a plug-in's shader, which have no class of their own, and
the reference that takes any shader (`Material("red")` is `Blinn("red")`).
`create` and `define` build the network as the Hypershade does — the
shader (`cmds.shadingNode(asShader=True)`, listed in `defaultShaderList1`),
`<shader>SG` and its materialInfo — and, a material being a shared,
scene-level asset, keep it out of an active `with container()` scope
unless `container=True`. `Material.create(type="aiStandardSurface")` builds
a type with no class. `rn.blinn()` makes only the shader, in the scope; the
first assignment builds its engine. `get_engine()` is the engine the shader
feeds (find-only), `delete()` / `rename()` act on the network (the shader,
its engines and their materialInfo; `cmds.delete` is the one-node escape),
and `astype(Phong)` converts the object in place (its node and its class,
one undo step), while another object of the old node raises from then on. `find_all(exact_type=False)` lists the
shaders Maya derives from the type, each typed by its own class.

### Display layers are exclusive and hold objects only

Membership is a connection from the layer's `drawInfo` into the member's
`drawOverride`, so `DisplayLayer.for_node(node)` reads it from the node
side without enumerating layers. A node is in one layer; `defaultLayer`
(undeletable) is where it sits when in none, and `for_node` answers `None`
there. Children draw with a member's override through the DAG without
being members, and `x << layer` never recurses. The DSL's `Layer` is this
class (`rig.Layer is DisplayLayer`): a layer node on the right of `<<`,
`in` or `-` is the membership collection (`layer.remove(x)` is the method
of `x << -layer`), and `DisplayLayer()` (`Layer()`) the kind, which means
`defaultLayer` on `<<`. That is the one way in and out (user decision
2026-10-16): the raw `add_members` / `remove_members` an object set has, and
a shading engine's `force_elements`, raise a TypeError naming the rig form
(`cube << layer`, `layer.remove(cube)`, `cube.f[0:2] << red`,
`red.remove(...)`); a plain `ObjectSet` keeps them.

### Serialisation and files

Every `serialize()` returns a `cgmath` data object and every `create` /
`set_*` that takes one is its inverse; `cgmath` is imported lazily, so
`rig` loads without it and only these paths need it. `save` writes that
data to a `cgmath` file and `load` reads it back.

| Node | `serialize()` | Back in | Save | Load |
|---|---|---|---|---|
| `Transform` / `Joint` | `TransformData`; `serialize_hierarchy()` a `HierarchyData`, `serialize_hierarchy(bindpose=True \| "bindPose5")` at its bind pose | `Transform.create(transform_data)` / `Joint.create(joint_data)` (one node), `Transform.create_hierarchy` | `node.save_hierarchy(path, *, bindpose=False)` | `Transform.load_hierarchy(path, *, parent=None, world_space=False, container=None)` creates, a typed `List`, parents first |
| `Mesh` | `(MeshData, UVList)`, or a `MeshData` with `include_uvs=False`: object space and the transform's world matrix (`world_space=True`: world space, the identity); `serialize_uv()` a `UVData`; `serialize_maps()` `MapData`s | `Mesh.create`, `set_points`, `set_uv_data`, `set_map_values` | `mesh.save(path, uvs=None, *, world_space=False)`: two files, the mesh and the UVs | `Mesh.load(path, uvs=None, *, name=, container=, normals=True, transform=None, world_space=False, **attrs)` creates (with `transform=`, the shape under it); `Mesh('x').load(path, uvs=None, *, world_space=False, normals=True)` updates |
| `Geometry` | `serialize_component_tags()` -> `GeomSubsetData`s | `set_component_tag_contents` | | |
| `NurbsCurve` / `NurbsSurface` | `BSplineData` (a periodic curve's own points; its knots when not Maya's default ones and the installed `cgmath` takes them) / `BSplinePatchData` (its knots likewise, `knots_u` / `knots_v`); each with the shape's name and the transform's world matrix as a `MeshData`, on the `cgmath` that adds those fields | `NurbsCurve.create`, `NurbsSurface.create` | `shape.save(path, *, world_space=False)` | `NurbsCurve.load(path, *, name=, container=, transform=None, world_space=False, **attrs)` creates (with `transform=`, the shape under it); `NurbsCurve('x').load(path, *, world_space=False)` updates its CVs (the same for `NurbsSurface`) |
| `SkinCluster` | `SkinData` (dense `(V, I)`, influences by short or full path) | `set_weights`, `SkinCluster.create`, `Mesh.apply_skin_data` | `skin.save(path, *, full_path=False, optimize=False)` | `SkinCluster.load(path, geo, **flags)` creates; `SkinCluster('x').load(path)` sets its weights |
| `BlendShape` | `MorphList`; `get_target_data()` a `MorphData` | `set_target_data`, `BlendShape.create` | `bs.save(path, *, match_name=None)` | `BlendShape.load(path, base, **flags)` creates; `BlendShape('x').load(path)` sets its targets by name |

**One file per data object, its format by its extension.** `.npz`
(compressed, recommended), `.json` (to read or diff; the slowest) or `.pkl`.
Never load a `.pkl` you did not write: a pickle runs code when it is read. A
mesh is two files, its `MeshData` and, with `uvs=`, its `UVList` (`save`
without `uvs=` writes no UVs: a `MeshData` holds none). `load` reads a
`cgmath` `.npz` or `.json` file whatever its extension (by its first bytes)
and a pickle by its `.pkl` or `.pickle` extension (cgmath 3.0.0, which
refuses a pickle under another name with its own `ValueError`; 1.0.4 and
2.0.0 read any other file as a pickle), and for a mesh the
assets `cgmath`'s `MeshData` reads (`.obj`, `.fbx`, `.glb`, `.usd`); in place
of a path (a `str` or an `os.PathLike`, `~` expanded) it takes the data
object itself, which it never changes. A file holding a
`cgmath.render.scene.Object` is a mesh and its UVs, as `Mesh.create` takes
one (its `uvs=` then a `TypeError`).

**`load` has two forms, told apart by where it is called.** From the class
it creates, always, as `create` does: `Mesh.load(...)` a second time makes
`body1` (`Mesh.load(path, transform='ctl')` one more shape under `ctl`),
`BlendShape.load(...)` stacks a second blendShape, and
`SkinCluster.load(...)` replaces the geometry's own skin (`create`'s rule).
From a node it updates that node in place, the inverse of its `save`:

- a `Mesh` of the same topology (vertex count, face counts, every face's
  vertex list, the holes) and no history gets every point (by default the
  file's points as its own, object space, the inverse of the default `save`;
  `world_space=True`: where the file puts them in the world, through the
  data's matrix or an `Object`'s transform; the transform is not moved), the
  file's hard edges and locked normals (`normals=False`: its own stay), and
  each UV set of the file by name (made when the mesh has none of that name;
  its other sets stay);
- a `NurbsCurve` of the same CV count, degree, form and knots, a
  `NurbsSurface` of the same CV counts, degrees and forms (it keeps its own
  knots: data that holds knots must hold these, and data that holds none in a
  direction, a `cgmath` without the fields or Maya's default ones, says even
  knots there: the surface's must be, or of degree 1, whose uneven knots only
  reparametrize; of degree 2 or more they would make another shape, the same
  `ValueError` as other knots), with no history, gets its CVs, in the same
  spaces: the file's CVs as its own by default, `world_space=True` where the
  data's matrix puts them in the world (a `BSplineData` of a `cgmath` without
  the field holds no matrix: its CVs are taken as world positions);
- a `SkinCluster` gets the file's weights as `set_weights(SkinData)` sets
  them (influences by name, the missing ones added, the ones the file does
  not name removed); the influences and the weights it holds already (within
  1e-9) write nothing and make no undo step;
- a `BlendShape` gets each target of the file by name (added when missing;
  its other targets stay); a target the file replaces gives up its live
  target mesh (its `inputGeomTarget` input, logged at INFO), whose offsets
  would replace the file's at the next evaluation.

The space of a `Mesh`, `NurbsCurve` or `NurbsSurface` update is a bool, by
keyword only: `world_space=` anything but a bool is a `TypeError` before the
file is read, and `load(path, True)` is a `TypeError` too (a mesh takes the
`True` for `uvs`), nothing changed. A file saved in world space,
`save(world_space=True)`, loads with `world_space=True`: the default would
take its world positions as the shape's own, moved by the transform, silently.
`Transform.load_hierarchy` takes its `world_space=` the same way: a bool, before
the file is read.

Another topology or CV layout, or a shape with history, is a `ValueError`
naming `Cls.load(path)` for a new node. A deformed shape (a skin, a
blendShape, a cluster) names the shape under its deformers instead, its
intermediate `...Orig` shape: `Mesh('bodyShapeOrig').load(path)` updates the
mesh a skin deforms, and the deformed shape follows (deleting the history
would delete the deformers); a deformed curve or surface names
`NurbsCurve('crvShapeOrig').load(path)` the same way. The class form takes `world_space=` as
`create` does: under `transform=` (read before the file is), or without it frozen at the world
positions. A hierarchy is never updated in place
(its transforms are what the rig hangs off: constraints, skins, children):
`Transform('root').load_hierarchy(...)` is a `TypeError` before anything is
read, naming `Transform.load_hierarchy(path, parent='root')`.

**Every file is read, and what it holds checked, before anything is made or
changed.** A file of another class (the UVs' file as the mesh, two files
swapped, a curve's file as a surface) is a `TypeError` naming the file, what
it holds and what the slot takes, with a hint when they look swapped; a
missing file a `FileNotFoundError`; a file of an older `cgmath` `cgmath`'s own
`LegacyFileError`. A deformer's load takes the file first, then the geometry:
the other order is a `TypeError` naming it. **Each load is one undo step**
(`rig.Mesh.load` ... `rig.Joint.load_hierarchy`), redo brings it back, and the
selection is left as it was. **`save` is a read of the scene**: no undo step,
the selection unchanged, and its path checked before anything is serialized
(an extension `cgmath` does not save, or `Mesh.save`'s two paths one file: a
`ValueError`; a missing folder: a `FileNotFoundError`). An existing file is
replaced whole.

**`Mesh.create` brings the data's shading back.** An edge is hard where its
two faces' normals in the data differ at one of its ends, and a normal Maya
then computes away from the data's was locked in the source, so it is set and
locked; data without normals (an `.obj`, a `MeshData` built by hand) is all
smooth, as before. Its cost over a create without normals (a 20k-face sphere,
best of 7, against 2.0.0a5's create of the same data, about 25 ms): all
smooth 1.3x, 800 hard edges 1.45x, 30% of the edges hard 1.75x, every edge
hard (faceted) 2.6x, 2000 vertices of locked normals 2.1x. Maya's normals and
each locked normal cross from API 2.0's arrays one value at a time, which
bounds a faceted or heavily locked mesh.

**`normals=False` skips it, for speed: a restore flag.** A part of the data
that a create or a load brings back at a cost of its own has one keyword,
named after the part, on every door that builds or updates a mesh from data:
`Mesh.create(data, normals=False)`, `Mesh.load(path, normals=False)` and
`Mesh('x').load(path, normals=False)`. True, the default, brings the part
back; False skips it: the data's normals are not read, a new mesh is all
smooth (2.0.0a5's create, its cost) and an updated mesh keeps its own hard
edges and locked normals. A flag is a `bool` (anything else is a `TypeError`
before anything is read or made), and it wins over a mesh attribute of the
same name (the mesh's internal `normals` array is never set through
`create`). `normals` is the first restore flag; a vertex-colour flag would be
the next, `colors=`, on the same three doors (`mesh._RESTORE_FLAGS`).

A `cgmath.render.scene.Object` stands for a mesh and its
UVs: `Mesh.create(obj)` builds its `mesh` with its `uv` (a `UVData` is one set,
a `UVList` every set), takes its name when `name=` is not given and puts the
new transform where it is (scale, rotate, translate, rotate order, rotate axis,
visibility; an `Object` under a parent: its world matrix); the mesh's own
matrix, the render settings and the skin are not used.

`serialize_hierarchy` captures `transform` and `joint` nodes (a transform
holding a locator becomes a `locator`) and skips constraints and other
transform-derived types with a warning; `create_hierarchy` does the same
on the way back and creates parents first. A user attribute that holds any
data (a `rig.Typed`) is written as `attributeType: "typed"`, its value
only the numbers it holds (`[[index, value], ...]` for a multi): what a
connection fed it is not saved, as no connection is. A user attribute that
holds geometry (typed `mesh`, `nurbsCurve`, `nurbsSurface`, `lattice` ...
data, empty or not) is skipped, with a warning: its value never loads back
(a compound keeps its other children, and goes when it has none left).

`serialize_hierarchy(bindpose=True)` reads the hierarchy at its bind pose:
the result is what `serialize_hierarchy()` would return after Maya's own
Go to Bind Pose (`dagPose -restore -global`), and nothing is restored (the
scene, the selection and the undo queue are unchanged). A pose member takes
its scale, rotate, translate, rotate axis and joint orient from the pose, its
rotation re-expressed in the node's current rotate order, as the restore
does; everything a pose does not hold (segment scale compensate, radius,
visibility, user attributes, names, parents) comes from the scene. Which
nodes are members is Maya's: a skinCluster's bind pose holds its influences
and every ancestor up to the world (a group above the root joint too), not
the end joints below them when the bind used `tsb=True`, nor a helper added
after the bind; those non-members keep the scene's values. At the bind pose
the result is bit-identical to `bindpose=False`. `True` takes every dagPose
with `bindPose` on that holds one of its nodes and merges them (binding
another influence set makes another bind pose, even at the same pose); poses
that disagree on a node raise a `ValueError` naming them, and naming one
reads that pose alone: `bindpose="bindPose1"`, or a node,
`Node("bindPose1")`. Any dagPose can be named (a saved `tPose` too, a
namespace with it: `"char:bindPose1"`); a name that is no dagPose is a
`NodeTypeError`. A member is stored relative to the parent it had when the
pose was saved, so a member reparented since (a group inserted above it,
its pose parent deleted) is a `ValueError` naming the member, its parent
then and its parent now: restore the parent or save a new pose. Every check
runs before anything is read into the result.

### Undo for API edits

`MFnMesh.setPoints`, `MFnSkinCluster.setWeights`, `MFnMesh.create` and
`MFnNurbsCurve.create` are not undoable by themselves. The bundled
`undoable_api_command` plug-in registers `cmds.rigUndoableAPICommand(obj)`
(a name of rig's own): give it any object with `doIt` / `undoIt` / `redoIt`
and its edit is one undo step.
rig's mesh edits (points, UV sets and their UVs, colour sets and their
colours) and skin weights go through it, one step each, API only: a UV or
colour set is made, renamed and deleted through `MFnMesh` too, never
`polyUVSet` / `polyColorSet`, whose undo beside API data edits and later
vertex edits restores broken sets. `Mesh.create`, `NurbsCurve.create` and
`NurbsSurface.create` need no command (the shape rides a recorded `createNode`
transform, and the mesh's shading is set on it before any recorded command
touches it; with `transform=` the shape then goes under the target, an API
edit through the command when it is loaded, else a recorded `parent -shape
-relative`, and the ride is deleted, so undo and redo follow the same
records); they, `SkinCluster.create` and `set_weights(SkinData)` are one
step each, named `rig.Mesh.create`, `rig.NurbsCurve.create`,
`rig.NurbsSurface.create`, `rig.SkinCluster.create`,
`rig.SkinCluster.set_weights`. A node's `load` puts its points, shading or
CVs through the command too; every `load` is one step, `rig.Mesh.load` ...
`rig.Joint.load_hierarchy`. The transfers (`Mesh.transfer_maps`, `transfer_component_tags`,
`SkinCluster.transfer_to_mesh`, `copy_skincluster`) are one step each too,
named after the method (`rig.Mesh.transfer_maps` ...).

Your own API edits take the same command: [CHEATSHEET section 21](CHEATSHEET.md#21-plugins--load_plugin-and-undo)
has the recipe and the object's rules (apply the change first, in `doIt`;
API only; no `cmds` in `undoIt` / `redoIt`), and `rig.undo_chunk` makes
several edits one named step. `load_plugin` loads the plug-in by full path
from `rig/nodetypes/plugins` (by name from `MAYA_PLUG_IN_PATH` if the file
is not there), once per session, so nothing needs configuring; rig loads
it itself at its first edit that needs it. The DSL's connects and value
sets use it too: a statement's are one step together ([root CHEATSHEET
section 38](../CHEATSHEET.md#38-undo)). With `rig/nodetypes/plugins` on
`MAYA_PLUG_IN_PATH`, the first load costs about 8 ms instead of about
165 ms. If it cannot be loaded, the edits above (and an attribute's
default or hidden flag re-declared) raise a `RuntimeError` that says rig
needs its undo plug-in, where it looked and why Maya did not load it,
while connects and value sets take the `cmds` route with one warning per
session. While a build's steps are on the undo queue, Maya refuses
`unloadPlugin("undoable_api_command")` until `cmds.flushUndo()`; an
unload forced anyway leaves those steps inert (undoing them does nothing).

---

## Conventions

- **A node prints as its name** and an `Attribute` is a `str`, so both go
  straight into `maya.cmds`. `repr` shows the class: `Mesh("cubeShape")`.
- **Local / object space is the default**, as in Maya's API: points,
  matrices, pivots, bounding boxes, hierarchies, shape data and its files.
  World space is always asked for, with `world_space=True` by keyword (a
  bool given by position is a `TypeError`). A local matrix is
  `.matrix`, in the parent's space without the offset parent matrix; a
  pivot is a point in the node's object space; a bounding box is in the
  parent's space; a shape's data holds its object-space points and its
  transform's world matrix, which `create` puts on its new transform.
- **`get_*` returns, verbs mutate.** `set_points`, `set_weights`,
  `set_component_tag_contents`, `freeze` edit the scene;
  `serialize`, `get_weights`, `duplicate_geometry` hand back something new.
- **Points come back as `MPointArray`**; `np.array(points)` is `(N, 4)`,
  so slice `[:, :3]`. Matrices are `MMatrix`; `np.array(m).reshape(4, 4)`
  is row-major with translation in row 3.
- **Weights are `(V, I)` numpy arrays**, rows are points in
  `get_num_weight_points()` order, columns follow `get_influence_objects()`.
- **Component ids are native**: `(N,)`, `(N, 2)` or `(N, 3)` integer
  arrays; strings are the shape-local `vtx[a:b]` / `cv[u][v0:v1]` /
  `f[a:b]` tokens `geometry.iter_component_tokens` renders.
- **Categories are `"v"`, `"e"`, `"f"`**; `"v"` maps to the geometry's
  class's `POINT_COMP_TYPE` (`vtx` on meshes, `cv` on curves and surfaces).
- **`cmds` flags pass through.** `get_children(**listRelatives_kwargs)`,
  `duplicate(**duplicate_kwargs)`, `find_all(**ls_kwargs)`,
  `list_attr(**listAttr_kwargs)`; a `create` passes its command's flags
  (`SkinCluster.create(..., normalizeWeights=...)`) and sets every other
  keyword as an attribute.
- **One lookup rule.** A bare name is looked up at the root namespace and,
  while another is current, there too; a path or `ns:name` exactly; a
  pattern is refused. `define` keys a bare name in the current namespace,
  because that is where `create` puts a new node.
- **`match_name` arguments are regexes** (`iter_joints`, `iter_shapes`,
  `serialize_maps`, `serialize_component_tags`, `BlendShape.serialize`),
  anchored with `exact_match=True`.

---

## When to reach for the typed methods

- You need a **result as data**: skin weights as an array, a mesh as a
  `MeshData`, a hierarchy as a `HierarchyData`, a target as a `MorphData`.
- You need a **classmethod**: `ShadingEngine.for_material`,
  `DisplayLayer.for_node`, `Transform.define`, `Reference.find_by_path`,
  `Follicle.create_on_mesh`, `Transform.create_hierarchy`.
- You are **writing a tool**, not a network: a duplicate-clean-geometry
  step, a skeleton duplicate with a suffix, an orient pass, a map mirror.
- You want **your own node type**: subclass `Transform` (or any class here)
  with `CUSTOM_NODE_TYPE`, and `Node` will hand your class back for the
  nodes it created. Keep its Python state under declared `_` names
  (`_side = None`) and its constants UPPER_CASE (read as `type(self).X`).
- You need `cmds.*` results as **typed nodes**: cast them,
  `Node(cmds.polyCube(ch=False)[0])`, or `Node.wrap(result)` for a whole
  result.

Stay with the DSL for setting, connecting and building math networks: that
is what `<<`, `>>` and the operators are for, and every method on this page
is on the same node object.

---

## Real behaviour, verified

Verified on Maya 2025; not bugs to work around blindly.

- **`Mesh("transformName")` and `Mesh(transform)` find the transform's
  first mesh shape**; a transform *without* one is a `NodeTypeError`
  (`'x' is a transform, not a mesh; ..., which has no mesh shape`).
- **Iterating an `Attribute` or a `Plug` raises** (`__iter__` takes a
  stray argument). Use `attr[:]` or `attr.get_logical_indices()`.
- **`attr[:]` on a multi creates the gaps**: the slice runs `0..max+1`
  through `elementByLogicalIndex`, so `[0, 3]` becomes `[0, 1, 2, 3]`.
  `attr[[0, 3]]` and `get_logical_indices()` do not.
- **A single negative index is not wrapped**: `mesh.vtx[-1]` is the bogus
  `controlPoints[-1]`; `mesh.vtx[[-1]]` is the last point.
- **`Attribute("cubeShape.vtx[0]")` raises** (`item is not a plug`): the
  string constructor takes real plugs only. Reach components through the
  node, `mesh.vtx[0]`, or build the DSL `Plug("cubeShape.vtx[0]")`; both
  resolve the alias.
- **`Joint.get_root_joint()` walks parents only**, so on a root joint it
  returns `None`. `match_hierarchy` takes a root as its own root, so it
  works from any joint of either chain, the root included.
- **A transform's bounding box leaves out offset parent matrices**: Maya
  builds it from channels, so neither the node's own offset nor those of
  the transforms below it are in it, and `get_bounding_box()` of a group
  whose children sit at their offsets is where their channels put them.
  `get_bounding_box(world_space=True)` counts every offset.
- **`duplicate_skeleton(prefix=..., include_list=[...])` deletes every joint**:
  children are renamed with the prefix before the include check compares
  their short names against the un-prefixed list. Use one or the other.
- **`orient_chain` leaves the result in `rotate`** (each `orient_joint` ends
  in `convert_orients_to_rotation`); call `hierarchy_to_orients()` after it
  to bake into `jointOrient`.
- **`orient_joint` / `orient_chain` address joints by short name**
  (`cmds.listRelatives` without `fullPath`), so a second joint of the same
  short name anywhere in the scene — a `duplicate_skeleton(include_list=...)`
  copy, say — makes them raise `More than one object matches name`. Orient
  first, duplicate after.
- **`Mesh.transfer_maps` and `transfer_component_tags` sample the data
  themselves.** `cgmath`'s `MeshDataResampler.resample_map` (1.0.4 and
  2.0.0) makes the map a one-influence `SkinData`, and the weight
  resampler normalises every row, so every map comes back `1.0` and every
  tag as the whole mesh. rig samples that data with the resampler's own
  sample data, unnormalised: the values are what `resample_map` gives
  without the normalisation. They take `mode` (`SPATIAL`, `UV`) and
  `method`; `ROBUST_BILINEAR` and the skin `options` are not map options.
  They, and `SkinCluster.transfer_to_mesh`, read UVs only for `UV` (the
  first UV sets); a mesh whose first set holds no UVs is then cgmath's
  `RuntimeError: UV data not supplied.`
  A tag holds where the target's components land on the source, in the
  transfer's mode. A vertex is in when its sample reads a tagged vertex
  with a nonzero weight (`GeomSubsetData.from_skin_data`'s rule), so a
  vertex tag onto a finer mesh also gets the vertices on the faces around
  its own. A face is in when its centre (`SPATIAL`: the mean of its
  corners; `UV`: the mean of its UV corners) is closest to a tagged face;
  a flat face whose centre is off it (concave, or holed: the centre is in
  the hole) uses the centre of its first triangle in Maya's triangulation.
  A centre on a corner or edge several source faces share (each face
  centre of a mesh is on a vertex of its subdivision) lands on one of
  them, the lowest-numbered with cgmath 2.0.0.
  An edge is in when both its ends are in the vertex set the tag's
  vertices transfer to, so an edge joining two tagged vertices is in even
  if the tag did not hold it (a rung between two tagged parallel loops).
  Onto the same topology (by `UV`: with the same UVs, not overlapping) a
  vertex or face tag comes back exactly, and an edge tag too but for such
  rungs. Where UVs overlap (stacked or mirrored shells), what lands there
  lands on one of the overlapping faces. (cgmath's own rule sent a face or edge
  tag through its vertices and back, one ring wider, and its `MeshData`
  numbers edges differently from Maya.)
  `SkinCluster.transfer_to_mesh` (many influences) is fine.
- **Several skinClusters on one mesh: the one nearest the shape.**
  `Mesh.apply_skin_data`, `Mesh.bake_deformation` and
  `SkinCluster.copy_skincluster` (each source and the target) act on the
  first skinCluster `get_deformers` lists, which is normally the last one
  applied. The mesh's other skinClusters are left alone. A skinCluster
  that reaches the mesh only from the side is not the mesh's: that of a
  blendShape's intermediate target (which `get_deformers` lists) or of a
  skinned mesh that drives it through a blendShape or a wrap. A mesh with
  none of its own gets a new one, `SkinCluster.create` deletes only the
  geometry's own skinClusters (not every one `listHistory` reaches), and
  `SkinCluster.rebind` rebinds only its own skinCluster (a mesh with none is
  left alone; one with several, layered, is a `RuntimeError` before anything
  changes: Maya re-binds in place only the skin nearest the shape).
- **`Mesh.add_color_set(name, "RGBA")` is a `ValueError`**: the
  representation is looked up by enum *value* (Maya's `kRGBA` integer), so
  pass `ColorSet.Representation.RGBA` (or `["RGBA"]` on the enum).
- **`Mesh.set_uv_data(uv, uv_set)` renames the set to `uv.name`** when they
  differ, and raises a `RuntimeError` before any edit when that name
  already exists on the mesh. Set `uv.name` first.
- **On a mesh with history, UVs written into a set the history made are
  lost when it evaluates again.** `add_uv_set` there adds a `createUVSet`
  node, and the UVs `set_uv_data` writes into that set are dropped by
  Maya on the next evaluation of the history (a `polyCube1.width` change,
  a joint move, an undo through it); UVs in `map1` stay. There too,
  `set_uv_data` with fewer UVs than the set holds is a `RuntimeError`
  before any edit.
- **`Mesh.get_closest_point` returns `(MPoint, face id)`**, the raw
  `MFnMesh.getClosestPoint` tuple, not the point its docstring describes.
- **A point is a value, never a plug, a node or components.**
  `get_closest_point`, `get_uv_at_point` and `set_points` take an `MPoint` or
  numbers (rows of them); a `Plug`, a node, a `Components` selection
  (`mesh.f[0]`, `mesh.e[:3]`), or a list holding one is a `TypeError` naming
  the fix, `plug.get()` or a vertex position (OpenMaya reads a Plug or a
  Components as a sequence, which crashed Maya).
- **`get_data_fn_set()` reads one plug's data.** A compound or a multi
  (array) plug has no data handle of its own: a `TypeError` names a child
  or an element to ask instead. A plug of a numeric, unit, enum or message
  attribute holds no data: `None` (only a typed, matrix or generic
  attribute holds data). A matrix attribute (`offsetParentMatrix`,
  `multMatrix.matrixSum`, an `addAttr(at="matrix")`) gives its
  `MFnMatrixData`, as a typed matrix (`worldMatrix[0]`) does: Maya's data
  handle of one holds no data, so it is read from the plug's value.
- **`Mesh.outMesh.get()` returns `(MeshData, UVList)`**: the shape's
  `serialize()` with its default `include_uvs=True`, in object space
  (`mesh.worldMesh[0].get()`: in world space).
- **`set_component_tag_category` converts nothing on a deformed mesh**: it
  hands the visible shape to Maya's converter while the tag lives on the
  `Orig`. Use `set_component_tag_contents(tag, ids, category=...)`, which
  rewrites the contents in the new category.
- **Maya refuses some component tag names**, with a bare "Maya command
  error". Maya trims the whitespace at the ends of a name and makes a space
  inside `_`. It then takes at least two characters, letters, digits and
  `_`, starting with a letter or `_`, with single `.` or `:` between
  characters. So `b`, `1a`, `a-b` and `a..b` are refused, and `ab`, `_1`
  and `a.b` are taken; `a b` is stored as `a_b`. `add_component_tag` and
  `rename_component_tag` raise a `ValueError` naming this rule before
  anything is made, and `add_component_tag` looks an existing tag up by the
  name Maya stores (`" ab"` finds `ab`). `rig.Tag` checks the same rule and
  means the same stored name (`Tag(" ab")` is `Tag("ab")`). A name with a
  non-ASCII character is left to Maya, which first maps it to the system
  code page; `rig.Tag` refuses it (a `ValueError`), as the name Maya
  stores is not the one it holds.
- **`Follicle.create_on_mesh` with a vertex list fails** (`numpy.AxisError`):
  the positions are collected in a generator that `np.array` cannot shape.
  Pass a transform as `ref_object`.
- **`Transform.create_hierarchy` needs `cgmath` 1.0.1 or later**
  (`SUPPORTED_NODE_TYPES`); `serialize_hierarchy` works with older ones.
  Watch which `cgmath` your `mayapy` imports: a site-packages copy wins
  over a `PYTHONPATH` checkout.
- **An animation curve's keys: what Maya's own tools do.** `cutKey
  -clear` deletes a curve it empties, so an update that clears the keys
  first loses the node: rig replaces keys through `MFnAnimCurve.remove`, last
  key first. `MFnAnimCurve.addKeys` merges (it never replaces, even with
  `keepExistingKeys=False`), sorts unsorted times and silently merges
  duplicate ones, takes internal units (radians, centimetres) and does
  not take a time curve (`animCurveTT`: one `addKey` per key); rig refuses
  times that are not strictly increasing and converts from the UI's units.
  Maya's `setAttr` takes a value on a keyed or constrained plug and the DG
  overrides it at the next evaluation; `plug << value` and `plug <<
  AnimCurve(...)` replace the input first. A tangent slope reads back in
  single precision (`5.0` set, `4.99999994` read).
- **What a `cgmath` file does not carry** (its data class has no field for
  it), so a `save` / `load` round trip loses it: a mesh's current UV set (the
  first set comes back current), its colour sets, paintable maps and component
  tags (a `polyCube(ch=False)`'s six face tags too), its materials, history and
  deformers, and its transform's pivots and parent; on a `cgmath` whose
  `BSplineData` / `BSplinePatchData` have no `name`, `matrix` and `knots_u` /
  `knots_v` fields (3.0.1), a curve's or surface's name and transform (save
  in world space there, `world_space=True`, to keep where it is: the default,
  object space, loads it at the origin) and a surface's uneven knots (they
  come back evenly spaced, over `0..spans`: a direction of degree 1 keeps its
  shape, only reparametrized, one of degree 2 or more comes back another
  shape; `save` warns); a
  skinCluster's settings (`skinningMethod`, `maximumInfluences`,
  `normalizeWeights`, `bindPreMatrix` connections: pass them to `load` again)
  and its bind pose (a load binds at the joints' current pose, as
  `cmds.skinCluster` does); a blendShape's weights (they come back 0), its
  in-betweens and its live target meshes (they come back as offsets); a
  hierarchy's constraints and user attributes that hold geometry (both skipped
  with a warning), connections, and shapes other than locators.
- **A mesh's shading comes back as normals, and an edge's smoothing flag where
  it shades nothing may not.** A border edge reads hard in Maya and comes back
  smooth; a hard edge whose two faces' normals agree anyway comes back smooth.
  The face-vertex normals are the source's (to 1e-5) either way.
  `setFaceVertexNormals` hardens every edge at the vertices it sets: after it
  rig makes the data's smooth edges there smooth again, which keeps the locked
  normals.
- **Undoing a target a blendShape load added keeps its weight element**: the
  blendShape keeps an unnamed `weight[i]` at 0 (as the undo of
  `set_target_data` on a new target always has).
- **Undoing `Mesh('x').load` brings back the points, the hard edges and the
  locked normals, and the mesh's stored normal array may come back longer**:
  a `.ma` saved after the undo lists more `.n` entries (unset ones), with the
  same shading.
- **`create_hierarchy` (so `load_hierarchy`) does not connect a joint's
  `inverseScale`** to its parent's `scale`: a joint with segment scale
  compensate under a scaled parent comes back at another world matrix.
- **Method names are spelled as they are**: `set_xfrom_attrs_locked`
  (sic), and `BlendShape.set_targets_data` beside `set_target_data`.
- `Mesh.bake_deformation` hard-codes `root_joint` as the skeleton. Its
  `max_influences` is the influence cap (below 1 is a `ValueError`), and
  `None` (the default) is no cap. It is one undo step. `merge_maps` / `merge_maps_list` raise `NotImplementedError`.
