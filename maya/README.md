# `rig.maya` — the typed node layer under the DSL

The object model the `rig` DSL stands on. Every DSL `Node` wraps one typed
node from here: a `Transform`, a `Mesh`, a `SkinCluster`, a `ShadingEngine`,
or the plain `DGNode` / `DAGNode` fallback. The typed node holds an
OpenMaya handle (rename-safe, deletion-aware), prints as its name so it
drops into `maya.cmds` unchanged, and carries the methods the DSL does not:
skin weights as arrays, component tags, material bindings, hierarchy
serialisation to `cgmath`.

```python
from maya import standalone
try:
    standalone.initialize()          # running from mayapy; inside Maya this raises and is skipped
except Exception:
    pass
from maya import cmds
cmds.file(new=True, force=True)

from rig.bridges import commands as rc

cube = rc.polyCube(name="cube", ch=False)[0]  # a DSL Node
mesh = cube.get_shape()                       # a method the DSL does not have: delegated, typed
print(repr(cube), repr(cube >> None), repr(mesh))  # Node("cube") Transform("cube") Mesh("cubeShape")
print(mesh.get_materials(), mesh.num_vertices)     # [DGNode("standardSurface1")] 8
```

You rarely import from `rig.maya` at all: the DSL reaches through. You do
when you want a class as a constructor (`Mesh("cubeShape")`,
`SkinCluster.create(...)`), a classmethod (`ShadingEngine.for_material`,
`DisplayLayer.for_node`), or a subclass of your own.

---

## Where to go next

| You want to... | Read |
|---|---|
| Copy-paste an example of every class and method here | [`CHEATSHEET.md`](CHEATSHEET.md) — runnable top to bottom |
| The DSL on top: `Node`, `Plug`, `<<` / `>>`, `PlugList`, `rig.bridges` | [`../README.md`](../README.md) · [`../CHEATSHEET.md`](../CHEATSHEET.md) |

---

## What's inside

```
rig/maya/
├── __init__.py            a docstring; re-exports nothing (import the submodule)
├── attribute.py           Attribute -- the MPlug wrapper: get / set / connect, slicing, components
├── node_name.py           get_short_name, get_clean_name, replace_suffix, iter_component_ranges / _tokens
├── pycmds.py              every maya.cmds function, results wrapped as typed nodes
├── constants.py           Axis, Renderers, ImageFormats, HardwareRenderingModes, EvaluationManagerModes
├── plugins/
│   ├── __init__.py        load_plugin() context manager, bundled_plugin_path()
│   └── undoable_api_command.py   the runUndoableAPICommand plug-in (undo for API edits)
└── nodetypes/
    ├── __init__.py        re-exports the classes below (not Deformer, tag_references, ColorSet)
    ├── _base.py           PyNode factory, NodeMeta, the custom-type stamp
    ├── dg_node.py         DGNode                 entity
    ├── dag_node.py        DAGNode                dagNode
    ├── transform.py       Transform              transform
    ├── joint.py           Joint                  joint
    ├── geometry.py        Geometry               geometryShape   (component tags)
    ├── mesh.py            Mesh, ColorSet         mesh
    ├── nurbs.py           NurbsCurve, NurbsSurface
    ├── deformer.py        Deformer, tag_references()   geometryFilter
    ├── skincluster.py     SkinCluster
    ├── blendshape.py      BlendShape
    ├── object_set.py      ObjectSet              objectSet
    ├── shading_engine.py  ShadingEngine          shadingEngine
    ├── display_layer.py   DisplayLayer           displayLayer
    ├── reference.py       Reference              reference
    ├── choice.py          Choice                 choice
    ├── follicle.py        Follicle               follicle
    ├── skel_delta_blend.py   SkeletonDeltaBlend  (plug-in: SkeletonDeltaBlend)
    └── anim_reader.py     AnimReaderNode         (plug-in: AnimReader; a plain wrapper, not a DGNode)
```

The public surface:

```python
from rig.maya.attribute import Attribute
from rig.maya.nodetypes import (
    PyNode, DGNode, DAGNode, Transform, Joint, Geometry, Mesh, NurbsCurve, NurbsSurface,
    SkinCluster, BlendShape, ObjectSet, ShadingEngine, DisplayLayer, Reference, Choice,
    Follicle, SkeletonDeltaBlend,
)
from rig.maya.nodetypes.deformer import Deformer, tag_references
from rig.maya.nodetypes.mesh import ColorSet
from rig.maya.plugins import load_plugin
from rig.maya import pycmds, node_name, constants
```

`rig.maya` stays inert on purpose: `nodetypes` imports several of its peers
at module scope, and an eager `__init__` would cycle. One more reason:
`import maya.cmds` inside this package still means Autodesk's `maya`, but
only while the *parent* of `rig/` is on `sys.path`. Never put `rig/` itself
on the path, or this `maya` shadows the real one.

---

## The class map

| Class | Wraps | What it adds |
|---|---|---|
| `DGNode` | any dependency node | name / long / short / clean name, uuid, `rename`, `namespace`, `find_attr` / `add_attr` / `delete_attr` / `rename_attr` / `list_attr`, `set_attrs`, `list_connections`, `find_connected_nodes`, `duplicate`, `delete`, `is_valid`, `remove_from_all_sets` |
| `DAGNode` | any DAG node | `get_parent(s)` / `iter_parents`, `get_children`, `set_parent`, `is_shape`, `mdagpath`, `get_bounding_box`, `get_deformers` |
| `Transform` | `transform` | `get_shape(s)`, `iter_shapes` / `find_shape`, `get_matrix` / `set_matrix` / `match_matrix`, pivots, `freeze`, `iter_xform_attrs`, `set_xfrom_attrs_locked`, `duplicate_geometry`, `serialize` / `serialize_hierarchy` / `create_hierarchy` |
| `Joint` | `joint` | `get_root_joint`, `get_parent_joint`, `iter_joints` / `find_joint`, `duplicate_skeleton`, `rename_skeleton`, `match_hierarchy`, `orient_joint` / `orient_chain`, orient <-> rotation conversions, `find_skinclusters`, `find_skel_blend` |
| `Geometry` | `geometryShape` | component tags: `injection_node`, `component_tags`, `add` / `remove` / `rename_component_tag`, `set_` / `get_component_tag_contents`, `get_component_tag_indices` / `_category` / `_data` / `_history`, `serialize_component_tags`, `local_shape_attr` / `world_shape_attr`, `get_component_mobject`, paintable attrs |
| `Mesh` | `mesh` | counts, `get_points` / `set_points`, closest point, normals, UV sets, `serialize` / `Mesh.create` (`MeshData` + `UVList`), paintable maps (`add_map`, `get_map_values`, `mirror_map`, `MapData`), colour sets (`ColorSet`), `get_materials` / `get_shading_engines` / `get_material_bindings`, `transfer_maps` / `transfer_component_tags`, `apply_skin_data` |
| `NurbsCurve`, `NurbsSurface` | `nurbsCurve`, `nurbsSurface` | `num_cvs`, `num_weight_points`, `get_points`, `serialize` (`BSplineData` / `BSplinePatchData`) |
| `Deformer` | `geometryFilter` | `get_geometries`, `get_original_geometries`; module function `tag_references(name)` |
| `SkinCluster` | `skinCluster` | `create(geo, influences | SkinData)`, influence add / remove / set, `get_weights` / `set_weights` as `(V, I)` arrays, `serialize` (`SkinData`), normalise / prune / max influences, `transfer_to_mesh`, `connect_bind_pre_matrices` |
| `BlendShape` | `blendShape` | `create(*geos_or_morphs)`, targets by name or index, weights, `get_target_data` / `set_target_data` (`MorphData`), `serialize` (`MorphList`), `add_empty_target`, `rebuild_target` |
| `ObjectSet` | `objectSet` | `get_or_create`, `get_members(as_components=)`, `add_members` / `remove_members` / `force_elements` / `clear` |
| `ShadingEngine` | `shadingEngine` | `create` (wired through `cmds.sets(renderable=True)`), `for_material`, `get_material` / `set_material`, `assign`, `get_face_members` |
| `DisplayLayer` | `displayLayer` | `get_or_create`, `for_node`, `is_default`, `get_members` / `add_members` / `remove_members` / `clear` / `delete` |
| `Reference` | `reference` | `create(path, namespace)`, `find_by_path`, `namespace`, `file_path`, `get_nodes`, `delete` |
| `Choice` | `choice` | `data_type` of `input[i]` / `output` follows the wiring and the selector |
| `Follicle` | `follicle` | `create_on_mesh`, `constrain`, `set_uv_values` |
| `SkeletonDeltaBlend` | `skeletonDeltaBlend` | plug-in node: targets, weights, matrices per target |
| `AnimReaderNode` | `animReader` | plug-in node: `.anm` clips onto an RT rig (not a `DGNode`) |

Each class inherits everything above it in its column, so a `Mesh` has
every `Geometry`, `DAGNode` and `DGNode` method, and a `ShadingEngine` is an
`ObjectSet`.

---

## Concepts

### Two layers, one node

The DSL's `Node` is a composition wrapper, not a subclass: it holds a typed
node in `_dg_node` and delegates through `__getattr__`. Three things follow.

| Spelling | Gives | Notes |
|---|---|---|
| `node >> None` | the typed node (`Transform`, `Mesh`, ...) | the only `>>` on a `Node` that reads |
| `node.get_shape()`, `node.serialize()`, ... | whatever the typed method returns | typed results, not `Node`s |
| `node.tx` | a `Plug` | the typed node's `Attribute`, re-wrapped for the operators |
| `typed.tx` | an `Attribute` | `get()` / `set()` / `connect()`, no network building |

Going the other way is `Node(typed)`, or `Node("name")`; `Node` and typed
node compare equal by the node they hold.

### `PyNode` is a factory

`PyNode(x)` never returns a `PyNode`. It resolves `x` (a name, uuid,
`MObject`, `MDagPath` or `MPlug`) and returns the most specific registered
class:

1. a locked `__custom_node_type__` string attribute, when the node has one;
2. otherwise `cmds.nodeType(x, inherited=True)` walked from the most derived
   type down, the first registered type wins;
3. otherwise `DAGNode` for DAG nodes, `DGNode` for the rest.

A string with a `.` in it resolves to an `Attribute`. So a `cluster` comes
back as a `Deformer` (registered as `geometryFilter`), a lattice shape or a
locator as a `Geometry` (`geometryShape`), a camera as a `DAGNode`, a
material as a `DGNode`.

Registration is the `NodeMeta` metaclass: any class that sets
`NATIVE_NODE_TYPE` (a Maya type) or `CUSTOM_NODE_TYPE` (a name of your
choosing) lands in `PyNode._NODE_CLASS_DICT` when the class statement
runs. A custom type stamps the string onto every node it creates, so
`PyNode` recognises those nodes later. `Transform.is_type(custom_node)` is
`False` at exact type and `True` with `exact_type=False`.

`PyNode.create(type, ...)` forwards to the registered class's `create`
when there is one — which is how `PyNode.create("skinCluster", geo, joints)`
ends up in `SkinCluster._create` (the DSL's `Node.create` takes keyword
arguments only and forwards the same way).

### Handles, not names

A typed node holds an `MObject` (a DAG node also an `MDagPath`). `name` is
read back from the handle each time, so a wrapper survives renames and
reparenting; `is_valid` says whether the node still exists and any method
after deletion raises `RuntimeError("... already deleted!")`.

Equality is class plus name, hashing is the long name. Two consequences:
`Transform("x") == PyNode("x")` because `PyNode("x")` *is* a `Transform`,
but `ObjectSet("initialShadingGroup") != PyNode("initialShadingGroup")`
because the second is a `ShadingEngine`. Wrap with `PyNode` when you want
the canonical class.

### `create()` is final, `_create()` is the hook

`DGNode.create(*args, **kwargs)` loads `PLUGIN_NAME` when set, calls the
class's `_create` (which must return a node name), stamps a custom type
when there is one and wraps the result. Subclasses override `_create`, and
that is where the constructor signatures diverge:

| Call | Builds with |
|---|---|
| `Transform.create(name=, parent=)`, `Joint.create(...)` | `cmds.createNode`, then `cmds.parent` |
| `Mesh.create(mesh_data, uv_data=, name=)` | `MFnMesh.create` inside an undoable command |
| `SkinCluster.create(geo, influences_or_SkinData, **skinCluster_kwargs)` | `cmds.skinCluster(toSelectedBones=True)`, existing skin deleted first |
| `BlendShape.create(*geometries_or_morphs, **blendShape_kwargs)` | `cmds.blendShape(frontOfChain=True)` |
| `ShadingEngine.create(name=)` | `cmds.sets(renderable=True, noSurfaceShader=True, empty=True)` |
| `DisplayLayer.create(**createDisplayLayer_kwargs)` | `cmds.createDisplayLayer` |
| `Reference.create(file_path, namespace)` | `cmds.file(reference=True)` |
| `SkeletonDeltaBlend.create(out_root, ref_root=, anim_root=)` | the plug-in node plus its wiring |

`ObjectSet.get_or_create(name)` and `DisplayLayer.get_or_create(name)` look
the name up as given and in the current namespace, refuse with a
`TypeError` when the name belongs to something else, and create otherwise.

### `Attribute` wraps an `MPlug` and subclasses `str`

`typed.tx` is an `Attribute`: `get()` mirrors `cmds.getAttr`, `set()`
mirrors `cmds.setAttr` and fills in `type=` for typed data (a single list
also stands for the `(count, *items)` form of `stringArray` /
`vectorArray` / `pointArray`). `a >> b` connects with force, `a // b`
disconnects, `connect(other, force=False)` refuses an occupied input.

Because it is a `str`, `cmds.setAttr(attr, 1)` just works, and `f"{attr}"`
is the full name.

Indexing is the multi / component surface: `attr[i]` is the element at a
logical index (created on access, Maya's own semantics), `attr[a:b]` and
`attr[[i, j, k]]` give lists of elements, `del attr[i]` removes one.
Geometry components (`mesh.vtx`, `curve.cv`, `surface.cv`) are
range-checked against the point count and accept negative ids in a list
key; `mesh.vtx`, `mesh.pnts` and `mesh.pt` are the same `controlPoints`
plug, and `mesh.cv` is an `AttributeError` because `cmds.listAttr` rejects
the alias on a mesh.

`get()` on a geometry-typed plug (`mesh`, `nurbsCurve`, `nurbsSurface`)
skips `cmds.getAttr`, which cannot serialise those. A shape's own plug
gives back the shape's `serialize()`, a plug with a downstream consumer
gives the consumer's, a `choice` output is traced through its selector,
and a dangling output is wrapped in `MFnMesh` / `MFnNurbsCurve`.

### Component tags live at the injection node

Maya writes tags where `cmds.deformableShape(tagInjectionNode=True)` says:
the shape itself until a deformer exists, its intermediate `Orig` shape
from then on. `Geometry.injection_node` answers that (uncached, since a
deformer moves it) and every editor writes there, so the shape you hold
and the tag editor never disagree. Readers go through the evaluated
geometry (`local_shape_attr`), so procedural tags — a history `polyCube`'s
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
`defaultShaderList1`, as the Hypershade does). `assign(members)` is
`forceElement`, so a member leaves whatever engine held it; Maya carves a
whole-object membership down to the remaining faces but never collapses
per-face membership back, and `assign(..., touched=[shapes], normalise=True)`
does that collapse and deletes the orphan `groupId` nodes. `get_face_members`
reads through `MFnSet` so per-face entries survive (`get_members` drops them).

`Mesh.get_material_bindings()` is the mesh-side view: the one material when
it covers the whole mesh, otherwise `(material, face ids)` pairs. The DSL's
materials (`rig.shade`) sit on `for_material` and `assign`.

### Display layers are exclusive and hold objects only

Membership is a connection from the layer's `drawInfo` into the member's
`drawOverride`, so `DisplayLayer.for_node(node)` reads it from the node
side without enumerating layers. A node is in one layer; `defaultLayer`
(undeletable) is where it sits when in none, and `for_node` answers `None`
there. Children draw with a member's override through the DAG without
being members, and `add_members` never recurses. The DSL's `Layer` is this
class.

### Serialisation goes to `cgmath`

Every `serialize()` returns a `cgmath` data object and every `create` /
`set_*` that takes one is its inverse; `cgmath` is imported lazily, so
`rig` loads without it and only these paths need it.

| Node | `serialize()` | Back in |
|---|---|---|
| `Transform` / `Joint` | `TransformData`; `serialize_hierarchy()` a `HierarchyData` | `Transform.create_hierarchy` |
| `Mesh` | `(MeshData, UVList)`, or a `MeshData` with `include_uvs=False`; `serialize_uv()` a `UVData`; `serialize_maps()` `MapData`s | `Mesh.create`, `set_points`, `set_uv_data`, `set_map_values` |
| `Geometry` | `serialize_component_tags()` -> `GeomSubsetData`s | `set_component_tag_contents` |
| `NurbsCurve` / `NurbsSurface` | `BSplineData` / `BSplinePatchData` | — |
| `SkinCluster` | `SkinData` (dense `(V, I)`, influences by short or full path) | `set_weights`, `SkinCluster.create`, `Mesh.apply_skin_data` |
| `BlendShape` | `MorphList`; `get_target_data()` a `MorphData` | `set_target_data`, `BlendShape.create` |

`serialize_hierarchy` captures `transform` and `joint` nodes (a transform
holding a locator becomes a `locator`) and skips constraints and other
transform-derived types with a warning; `create_hierarchy` does the same
on the way back and creates parents first.

### Undo for API edits

`MFnMesh.setPoints`, `MFnSkinCluster.setWeights` and `MFnMesh.create` are
not undoable by themselves. The bundled `undoable_api_command` plug-in
registers `cmds.runUndoableAPICommand(obj)`: give it any object with
`doIt` / `undoIt` / `redoIt` and it runs in one undo chunk. `load_plugin`
finds the plug-in on `MAYA_PLUG_IN_PATH` first and in `rig/maya/plugins`
second, so nothing needs configuring.

---

## Conventions

- **A node prints as its name** and an `Attribute` is a `str`, so both go
  straight into `maya.cmds`. `repr` shows the class: `Mesh("cubeShape")`.
- **`world_space=True` is the default** for points, matrices, pivots and
  bounding boxes; pass `world_space=False` for object space.
- **`get_*` returns, verbs mutate.** `set_points`, `set_weights`,
  `set_component_tag_contents`, `assign`, `freeze` edit the scene;
  `serialize`, `get_weights`, `duplicate_geometry` hand back something new.
- **Points come back as `MPointArray`**; `np.array(points)` is `(N, 4)`,
  so slice `[:, :3]`. Matrices are `MMatrix`; `np.array(m).reshape(4, 4)`
  is row-major with translation in row 3.
- **Weights are `(V, I)` numpy arrays**, rows are points in
  `num_weight_points` order, columns follow `get_influence_objects()`.
- **Component ids are native**: `(N,)`, `(N, 2)` or `(N, 3)` integer
  arrays; strings are the shape-local `vtx[a:b]` / `cv[u][v0:v1]` /
  `f[a:b]` tokens `node_name.iter_component_tokens` renders.
- **Categories are `"v"`, `"e"`, `"f"`**; `"v"` maps to the geometry's
  `POINT_COMP_TYPE` (`vtx` on meshes, `cv` on curves and surfaces).
- **`cmds` flags pass through.** `get_children(**listRelatives_kwargs)`,
  `duplicate(**duplicate_kwargs)`, `SkinCluster.create(..., **skinCluster_kwargs)`,
  `find_all(**ls_kwargs)`, `list_attr(**listAttr_kwargs)`.
- **Names are looked up in the current namespace too** by
  `ObjectSet.get_or_create` and `DisplayLayer.get_or_create`, because that
  is where `create` puts a new node.
- **`match_name` arguments are regexes** (`iter_joints`, `iter_shapes`,
  `serialize_maps`, `serialize_component_tags`, `BlendShape.serialize`),
  anchored with `exact_match=True`.

---

## When to drop below the DSL

- You need a **result as data**: skin weights as an array, a mesh as a
  `MeshData`, a hierarchy as a `HierarchyData`, a target as a `MorphData`.
- You need a **classmethod**: `ShadingEngine.for_material`,
  `DisplayLayer.for_node`, `ObjectSet.get_or_create`, `Reference.find_by_path`,
  `Follicle.create_on_mesh`, `Transform.create_hierarchy`.
- You are **writing a tool**, not a network: a duplicate-clean-geometry
  step, a skeleton duplicate with a suffix, an orient pass, a map mirror.
- You want **your own node type**: subclass `Transform` (or any class here)
  with `CUSTOM_NODE_TYPE`, and `PyNode` will hand your class back for the
  nodes it created.
- You need `cmds.*` results as **typed nodes** without the DSL: `rig.maya.pycmds`.

Stay in the DSL for setting, connecting and building math networks: that
is what `<<`, `>>` and the operators are for, and `Node.__getattr__` already
gives you every method on this page.

---

## Real behaviour, verified

Verified on Maya 2025; not bugs to work around blindly.

- **`Mesh("transformName")` finds the transform's first mesh shape**, but a
  transform *without* one raises `TypeError` (`object of type 'NoneType' has
  no len()`), not the `ValueError` the message in the source promises. And
  `Mesh(a_transform_instance)` is a `ValueError` (`... is not a mesh`): the
  transform shortcut only reads a name string. `SkinCluster.transfer_to_mesh`
  and `Mesh.transfer_maps` go through `Mesh(other)`, so pass them a shape or
  a name.
- **Iterating an `Attribute` raises** (`__iter__` takes a stray argument).
  Use `attr[:]` or `attr.get_logical_indices()`.
- **`attr[:]` on a multi creates the gaps**: the slice runs `0..max+1`
  through `elementByLogicalIndex`, so `[0, 3]` becomes `[0, 1, 2, 3]`.
  `attr[[0, 3]]` and `get_logical_indices()` do not.
- **A single negative index is not wrapped**: `mesh.vtx[-1]` is the bogus
  `controlPoints[-1]`; `mesh.vtx[[-1]]` is the last point.
- **`Attribute("cubeShape.vtx[0]")` raises** (`item is not a plug`): the
  string constructor takes real plugs only. Reach components through the
  node, `mesh.vtx[0]`, which resolves the alias.
- **`Joint.get_root_joint()` walks parents only**, so on a root joint it
  returns `None`, and `match_hierarchy` — which calls it on both sides —
  must be called from a child joint of each chain.
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
- **`Mesh.transfer_maps` and `transfer_component_tags` come back all-ones**:
  the map becomes a one-influence `SkinData` for `resample_map`, and the
  weight resampler normalises every row to `1.0`, so the target gets the
  map at `1.0` on every vertex and the tag on every vertex. Verified with
  `cgmath` 1.0.0 and 1.0.1. `SkinCluster.transfer_to_mesh` (many
  influences) is fine.
- **`Mesh.add_color_set(name, "RGBA")` is a `ValueError`**: the
  representation is looked up by enum *value* (Maya's `kRGBA` integer), so
  pass `ColorSet.Representation.RGBA` (or `["RGBA"]` on the enum).
- **`Mesh.set_uv_data(uv, uv_set)` renames the set to `uv.name`** when they
  differ, and fails when that name already exists on the mesh. Set
  `uv.name` first.
- **`Mesh.get_closest_point` returns `(MPoint, face id)`**, the raw
  `MFnMesh.getClosestPoint` tuple, not the point its docstring describes.
- **`Mesh.outMesh.get()` returns `(MeshData, UVList)`**: the shape's
  `serialize()` with its default `include_uvs=True`.
- **`set_component_tag_category` converts nothing on a deformed mesh**: it
  hands the visible shape to Maya's converter while the tag lives on the
  `Orig`. Use `set_component_tag_contents(tag, ids, category=...)`, which
  rewrites the contents in the new category.
- **`Follicle.create_on_mesh` with a vertex list fails** (`numpy.AxisError`):
  the positions are collected in a generator that `np.array` cannot shape.
  Pass a transform as `ref_object`.
- **`Transform.create_hierarchy` needs `cgmath` 1.0.1 or later**
  (`SUPPORTED_NODE_TYPES`); `serialize_hierarchy` works with older ones.
  Watch which `cgmath` your `mayapy` imports: a site-packages copy wins
  over a `PYTHONPATH` checkout.
- **Method names are spelled as they are**: `set_xfrom_attrs_locked`
  (sic), `BlendShape.set_targets_data` beside `set_target_data`, and
  `constants.EvaluationManagerModes.PARALLEL == "paralell"`.
- `Mesh.bake_deformation` hard-codes `root_joint` as the skeleton and `4`
  as the influence cap; `merge_maps` / `merge_maps_list` raise
  `NotImplementedError`.
