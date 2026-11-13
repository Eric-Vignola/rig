"""
Transform node class
"""

from __future__ import annotations

import logging
import os
import re
from types import MappingProxyType as _MappingProxyType
from typing import Any, Union

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig._internal.journal import barrier as _barrier, cmds_edit as _cmds_edit, statement as _statement
from rig._internal.undo import _undo_chunk
from rig.nodetypes import _files
from rig.nodetypes._base import _NODE_CLASS_DICT, _cast, _is_generic_slot, Attribute, _name_of
from rig.nodetypes.dag_node import DAGNode, _parent_path
from rig.nodetypes.dg_node import DGNode, _attribute_keywords, _got, _positional_name, _typed_creator

LOGGER = logging.getLogger(__name__)


_HIERARCHY_LOAD_DOC = """Loads a hierarchy from a cgmath file (a HierarchyData): it CREATES the nodes.

    ``Transform.load_hierarchy(hierarchy, *, parent=None, world_space=False,
    container=None)`` (and ``Joint.load_hierarchy``, the same), from the class,
    makes the file's transforms, joints and locators, as `create_hierarchy`
    does, always (a second load of one file makes ``root1``)::

        skel = Joint.load_hierarchy("skeleton.npz")        # List: the new nodes, typed, parents first
        arm  = Transform.load_hierarchy("arm.npz", parent="rig_grp")

    Args:
        hierarchy: The file (a str or os.PathLike, ``~`` expanded; a cgmath
            ``.npz`` / ``.json`` / ``.pkl``), or the HierarchyData itself. A
            ClipData, the object or a file that holds one, is built and keyed,
            no keyword, as `create_hierarchy` says; a plain HierarchyData
            (``clip.rest_pose``, when the clip has one) makes no keys. An asset
            (``.fbx``, ``.glb``) reads as one pose, no keys and no warning: a
            keyed one is ``ClipData.load_fbx`` or ``ClipData.load_glb``, then
            `create_hierarchy`.
        parent: An existing node to make the roots under (`create_hierarchy`'s
            ``parent``). None: the world.
        world_space: A bool (anything else is a TypeError, before the file is
            read). With `parent` (without, it changes nothing), as
            `create_hierarchy` takes it: False (the default), the roots' values
            from the file as local values under `parent` (the file holds local
            values: under the node it was saved under, it lands where it was,
            as far as the channels it holds place it); True, the roots' world
            matrices in the file (their values read as world values), on every
            frame of a clip.
        container: Inside ``with container()``, whether the new nodes join the
            scope (None: yes), as for every typed create.
        uuids: A bool (anything else is a TypeError, before the file is read),
            as `create_hierarchy` takes it: False (the default), the nodes
            keep the uuids Maya gives them, so a file loads any number of
            times; True, every node made takes its data's uuid, so a file
            loaded twice is a ValueError on the second load, before anything
            is made (give the data fresh uuids first,
            ``hierarchy.renew_uuids()``).

    Returns:
        A List of the new nodes, typed (``Joint`` / ``Transform``), parents first:
        ``nodes[0]`` is the first root. Maya renames a node whose name is taken.

    The file is read, and what it holds checked, before anything is made: a
    missing file is a FileNotFoundError, a file of another class a TypeError
    naming what it holds, a NaN or an infinity in a column of a clip that is
    keyed a ValueError, a missing undo plug-in, for a clip with keys, a
    RuntimeError.
    Constraints and other types derived from transform are skipped with a
    warning (`create_hierarchy`'s).

    From a node, ``Transform('root').load_hierarchy(...)`` is a TypeError: rig
    does not update a hierarchy in place (its transforms hold the rig's
    constraints, skins and children); ``Transform.load_hierarchy(path,
    parent='root')`` makes one under it.

    Each call is ONE undo step named ``rig.Transform.load_hierarchy`` (or
    ``rig.Joint.load_hierarchy``), a clip's curves and keys in it, and leaves
    the selection as it was.
    """

# what ``create_hierarchy``'s and ``load_hierarchy``'s world_space= says (`_files.space_flag`)
_HIERARCHY_SPACE = "how the roots' values read under parent: local, or world"

# what a difference across a clip's frames may be, per 1 of the value, and still be the round-off
# of a root's solve under a parent (`_clip_rows`): 2000 frames of a turning root leave 7e-16
_SOLVE_GRAIN = 1e-12

# the keywords ``Transform.create(transform_data)`` reads itself (`_create_from_data`):
# every other keyword is an attribute of the new node
_DATA_CREATE_FLAGS = frozenset({"name", "n", "parent", "p", "uuids"})

# how a hierarchy's data takes fresh uuids (`_files.free_uuids`' advice)
_RENEW = "hierarchy.renew_uuids()"

# the exact Maya node types serialize_hierarchy() captures,
# a locator is a "transform" holding a locator shape.
SERIALIZED_NODE_TYPES = ("transform", "joint")

# cmds.addAttr(dataType=...) names of the typed attribute data kinds (read-only,
# user decision 2026-10-17)
TYPED_DATA_NAMES = _MappingProxyType({
    OpenMaya.MFnData.kString:      "string",
    OpenMaya.MFnData.kStringArray: "stringArray",
    OpenMaya.MFnData.kDoubleArray: "doubleArray",
    OpenMaya.MFnData.kFloatArray:  "floatArray",
    OpenMaya.MFnData.kIntArray:    "Int32Array",
    OpenMaya.MFnData.kVectorArray: "vectorArray",
    OpenMaya.MFnData.kPointArray:  "pointArray",
    OpenMaya.MFnData.kMatrix:      "matrix",
})

# the typed data that is geometry: a user attribute of these reads back an MFn set (None while it
# is empty), which no file takes and no cmds.setAttr() sets, so serialize() skips it
_GEOMETRY_DATA = frozenset({
    OpenMaya.MFnData.kMesh,
    OpenMaya.MFnData.kNurbsCurve,
    OpenMaya.MFnData.kNurbsSurface,
    OpenMaya.MFnData.kLattice,
    OpenMaya.MFnData.kSubdSurface,
    OpenMaya.MFnData.kSphere,
    OpenMaya.MFnData.kDynSweptGeometry,
    OpenMaya.MFnData.kPluginGeometry,
})

# array data types older serialized hierarchies stored under "attributeType",
# cmds.addAttr() only accepts them as "dataType".
# "matrix" is left out, it is valid as both.
LEGACY_DATA_TYPES = (
    "stringArray",
    "doubleArray",
    "floatArray",
    "Int32Array",
    "vectorArray",
    "pointArray",
)


def _holds_geometry(attribute: OpenMaya.MObject) -> bool:
    """True for an attribute (its MObject) of a `_GEOMETRY_DATA` type, or a compound whose
    children all are (nothing of it loads back). A generic slot, of any data, is not: only the
    numbers it holds are written."""
    if attribute.hasFn(OpenMaya.MFn.kTypedAttribute):
        return OpenMaya.MFnTypedAttribute(attribute).attrType() in _GEOMETRY_DATA
    if not attribute.hasFn(OpenMaya.MFn.kCompoundAttribute):
        return False
    compound = OpenMaya.MFnCompoundAttribute(attribute)
    children = [compound.child(i) for i in range(compound.numChildren())]
    return bool(children) and all(_holds_geometry(child) for child in children)


def _create_nested(node_type: str, **kwargs) -> DAGNode:
    """A node of ``node_type`` made by `Transform.create_hierarchy`, a typed
    create, so the create is nested: a registered type runs its class's typed
    create, any other type is ``cmds.createNode``, cast. It takes neither
    ``Node.create``'s scope rules (the outer create registers every node the
    call made, under the hierarchy's own names) nor its ``skipSelect`` default."""
    node_cls = _NODE_CLASS_DICT.get(node_type)
    if node_cls is not None:
        return node_cls.create(**kwargs)
    return _cast(cmds.createNode(node_type, **kwargs))


def _make_from_data(data: TransformData, name: Any, parent: str | None) -> DAGNode:
    """The node one TransformData describes, made at identity, named `name`,
    under `parent` (a long name; None: the world): a joint or a transform by
    its class's typed create (nested, see `_create_nested`), a locator's
    transform by ``spaceLocator``. `create_hierarchy`'s per-node create, which
    ``Transform.create(data)`` shares (the values are `_set_values`'s)."""
    if data.node_type == "locator":
        obj = _cast(_cmds_edit.spaceLocator(name=name)[0])
    else:
        return _create_nested(data.node_type, name=name, parent=parent)
    if parent is not None:
        # relative: its values go on as local values, and a parent that kept its world
        # place would leave a shear under a non-uniformly scaled node, which no value undoes
        _cmds_edit.parent(type(obj).get_long_name(obj), parent, relative=True)
    return obj


def _set_values(node: DAGNode, data: TransformData) -> None:
    """Sets the values `data` holds for `node` (scale, rotate, translate,
    rotate order and axis, visibility; on a joint its orient, segment scale
    compensate, radius and draw style): what `create_hierarchy` and
    ``Transform.create(data)`` set once the node exists."""
    node.set_attrs(skip_missing=True, **data.to_attributes())


def _changed_values(node: DAGNode, values: dict) -> dict:
    """The values of `values` (a TransformData's ``to_attributes()``) `node` does not hold
    already (to 1e-12), each a plug it has: what `key_hierarchy` sets, so a channel it leaves
    as it is may be locked (a skinCluster locks its mesh's transform)."""
    changed = {}
    for attr, value in values.items():
        plug = node.find_attr(attr, quiet=True)
        if plug is None:
            continue
        try:
            held = np.asarray(plug.get(), dtype=float).ravel()
            want = np.asarray(value, dtype=float).ravel()
            same = held.shape == want.shape and np.allclose(held, want, rtol=0.0, atol=1e-12)
        except (TypeError, ValueError):
            same = False
        if not same:
            changed[attr] = value
    return changed


def _free_values(node: DAGNode, changed: dict, rows: list[tuple]) -> tuple[dict, list, list]:
    """What the locks of `node` leave of `changed` (`_changed_values`) and of its key `rows`
    (`_clip_rows`): ``(values, rows left out, locked names)``. A locked scalar or compound is
    left out whole. A compound with a locked axis gives its free axes one by one (it takes no
    value while a child is locked), and the locked axis counts only where the data changes it.
    A key on a locked axis is left out too."""
    long_name = type(node).get_long_name(node)
    held      = set(cmds.listAttr(long_name, locked=True) or [])
    if not held:
        return changed, [], []
    free, locked = {}, []
    for attr, value in changed.items():
        if attr in held:  # a scalar, or the compound itself (its children are listed with it)
            locked.append(attr)
            continue
        children = cmds.attributeQuery(attr, node=long_name, listChildren=True) or []
        if held.isdisjoint(children):
            free[attr] = value
            continue
        axes = dict(zip(children, np.asarray(value, dtype=float).ravel().tolist()))
        for child, axis in _changed_values(node, axes).items():
            if child in held:
                locked.append(child)
            else:
                free[child] = axis
    left = [row for row in rows if row[1] in held]
    for _, attr, _ in left:
        above = cmds.attributeQuery(attr, node=long_name, listParent=True) or []
        if attr not in locked and not set(above).intersection(locked):
            locked.append(attr)
    return free, left, locked


def _key_plan(
    hierarchy: HierarchyData,
    keyed:     set           | None = None,
    strict:    bool          | None = None,
    method:    str                  = "key_hierarchy",
) -> tuple[list, list, list]:
    """What `key_hierarchy` writes, checked before the first edit: ``(rows, changes, left)``.
    `hierarchy` is a copy whose nodes' names are the Maya nodes they key; with `keyed` (the
    ids of the nodes it paired) only those are read, the rest of the data keys nothing.
    ``rows`` are the keys (`_clip_rows`: a NaN, a missing plug-in) and ``changes`` the
    ``(node, values)`` to set, each a value its node does not hold already.

    `strict` says what a channel that cannot be written is. True: an attribute a joint's data
    sets and its node lacks (the node is no joint), and a locked channel the data changes or
    keys, are each a ValueError naming every one. False: they are left out, the node's other
    channels written (`_free_values`), and ``left`` describes them for the caller's one
    warning. None (``rig.create_scene``, whose nodes were made from the data): a lock is the
    ValueError, and an attribute a node lacks is left out, as `_set_values` leaves it."""
    rows = _clip_rows(hierarchy, {}, only=keyed)
    keys = {}
    for row in rows:
        keys.setdefault(id(row[0]), []).append(row)

    lacking, locked, changes, dropped = [], [], [], set()
    for data in hierarchy:
        if keyed is not None and id(data) not in keyed:
            continue
        node   = data.name
        values = data.to_attributes()
        if strict is not None and data.node_type == "joint":  # a transform's data sets nothing a joint lacks
            absent = [attr for attr in values if node.find_attr(attr, quiet=True) is None]
            if absent:
                lacking.append(f"{node} ({', '.join(absent)})")
        free, left, held = _free_values(node, _changed_values(node, values), keys.get(id(data), []))
        changes.append((node, free))
        dropped.update(id(row) for row in left)
        locked.extend(f"{node}.{attr}" for attr in held)

    if strict is not False:
        hint = "strict=False skips them, " if strict else ""
        if lacking:
            raise ValueError(
                f"{method}: {len(lacking)} node(s) lack attributes their data sets (a joint's data, a node that "
                f"is no joint): {', '.join(lacking)}; {hint}nothing was keyed"
            )
        if locked:
            raise ValueError(
                f"{method}: the data changes {len(locked)} locked channel(s): {', '.join(locked)} "
                f"(a skinCluster locks its mesh's transform); {hint}nothing was keyed"
            )
    left = []
    if lacking:
        left.append(f"{len(lacking)} node(s) lacking attributes a joint's data sets: {', '.join(lacking)}")
    if locked:
        left.append(f"{len(locked)} locked channel(s): {', '.join(locked)}")
    return [row for row in rows if id(row) not in dropped], changes, left


def _clip_rows(hierarchy: HierarchyData, solved: dict, only: set | None = None) -> list[tuple]:
    """The keys a ClipData needs, ``(node data, attribute, keys)`` per column of its (F, N, 3)
    blocks with any difference across the frames, each with a key on every frame. A flat
    column has no row: `_set_values` sets it as a plain value (so a one-frame clip has
    none, and a plain HierarchyData none at all). `solved` holds the roots solved under a
    parent (`_build_hierarchy`), whose round-off is no difference: a turning root's scale, 1
    in the clip, comes back 1 give or take 1e-15 (`_SOLVE_GRAIN`). A joint that has to take
    out a scale no plug gives it (`_held_scale`) has its own rows, on ``inverseScale``.
    `only` (the ids of the nodes `key_hierarchy` paired) leaves every other node without a
    row, so a NaN in a node that takes no key is no error.

    Key k lands on frame ``start_frame + k`` whatever the scene's rate: another rate logs
    one warning, keys or none, and the scene's time unit and playback range are not touched.
    The tangents are Maya's global default, read once and given to every curve, as
    ``AnimCurve(...)`` alone is linear. Read before any node is made, so a NaN (no key takes
    one) or a missing undo plug-in refuses with nothing made."""
    from cgmath.hierarchy import MAYA_ATTRIBUTE_MAP, ClipData
    from rig.nodetypes import plugins
    from rig.nodetypes.anim_curve import AnimCurve

    if not isinstance(hierarchy, ClipData) or not hierarchy.frame_count:
        return []
    frames = hierarchy.frames
    times  = hierarchy.start_frame + np.arange(hierarchy.frame_count)
    tangents = {
        "in_tangents":  cmds.keyTangent(query=True, g=True, inTangentType=True)[0],
        "out_tangents": cmds.keyTangent(query=True, g=True, outTangentType=True)[0],
    }
    grain = np.array([_SOLVE_GRAIN if id(node) in solved else 0.0 for node in hierarchy])[:, None]
    rows  = []
    for channel in ClipData.FRAMED_CHANNELS:
        attr  = MAYA_ATTRIBUTE_MAP[channel]
        block = getattr(frames, channel.lstrip("_"))
        moved = ~(np.abs(block - block[0]) <= grain * (1.0 + np.abs(block[0])))
        for node, axis in np.argwhere(moved.any(axis=0)).tolist():
            if only is not None and id(hierarchy[node]) not in only:
                continue
            keys = AnimCurve(block[:, node, axis], times=times, **tangents)
            rows.append((hierarchy[node], f"{attr}{'XYZ'[axis]}", keys))
    for node in hierarchy:
        if only is not None and id(node) not in only:
            continue
        held = _held_scale(node, solved)
        for axis in () if held is None else np.flatnonzero(np.ptp(held, axis=0)).tolist():
            keys = AnimCurve(held[:, axis], times=times, **tangents)
            rows.append((node, f"inverseScale{'XYZ'[axis]}", keys))
    if rows and not plugins._fast_path_ready():
        raise RuntimeError(
            f"rig needs its undo plug-in 'undoable_api_command' to key a clip, and it could "
            f"not be loaded ({plugins._plugin_whereabouts()}); nothing was made"
        )
    scene = OpenMaya.MTime(1.0, OpenMaya.MTime.kSeconds).asUnits(OpenMaya.MTime.uiUnit())
    if abs(hierarchy.fps - scene) > 1e-3:  # a clip's 23.976 is the scene's 24000 / 1001
        LOGGER.warning(
            f"The clip is at {hierarchy.fps:g} fps and the scene at {scene:g} fps: its frames "
            f"land frame for frame, on the clip's frame numbers, and play at the scene's rate"
        )
    return rows


def _key_rows(rows: list[tuple]) -> None:
    """Keys the channels of `rows` (`_clip_rows`, its nodes made) in one statement: every row
    planned, then every curve written. A channel the node has not (a transform's joint
    orient) is left out, as `_set_values` leaves out its value."""
    from rig._internal.list import List

    found = [(data.name.find_attr(attr, quiet=True), keys) for data, attr, keys in rows]
    found = [row for row in found if row[0] is not None]
    if found:
        plugs, keys = zip(*found)
        List(plugs) << list(keys)


def _held_scale(node: Any, solved: dict) -> np.ndarray | None:
    """The (F, 3) scale in the clip of the parent of `node`, for a joint that compensates and
    whose parent's plug no longer holds it: a joint root that does not compensate, solved
    under a scaled node, takes that node's scale into its own (`_build_hierarchy`). None for
    any other node."""
    parent = node.get_parent()
    if id(parent) not in solved or parent.node_type != "joint":
        return None
    return solved[id(parent)][1] if node.node_type == "joint" and node.segment_scale_compensate else None


def _clip_inverse_scales(hierarchy: HierarchyData, solved: dict, only: set | None = None) -> None:
    """Makes the joints of a ClipData evaluate what its world matrices hold (its nodes made):
    cgmath takes the parent's scale out of a joint whose segment scale compensate is on, on
    every frame, and Maya only out of a joint whose ``inverseScale`` holds it. A joint under
    a joint of the clip is wired, ``parent.scale >> joint.inverseScale``, as ``cmds.parent``
    wires one, or takes `_held_scale` as its plain value (its keys are `_clip_rows`'). A root
    solved under a node outside the clip takes that node's world scale: the value its solve
    took out, which no plug of that node holds when the scale comes from above it. A joint
    under a transform of the clip is left as Maya leaves it, unwired.

    `only` (the ids of the nodes `key_hierarchy` paired, on nodes it did not make) leaves out a
    joint that is not keyed or whose node has no ``inverseScale`` (it is no joint), and wires
    one only to a keyed parent whose node is its parent in Maya."""
    for node in hierarchy:
        if node.node_type != "joint" or not node.segment_scale_compensate:
            continue
        if only is not None and id(node) not in only:
            continue
        inverse = node.name.find_attr("inverseScale", quiet=only is not None)
        if inverse is None:
            continue
        parent = node.get_parent()
        held   = _held_scale(node, solved)
        if id(node) in solved:
            inverse.set(*solved[id(node)][0].tolist())
        elif held is not None:
            inverse.set(*held[0].tolist())
        elif getattr(parent, "node_type", None) == "joint":
            if only is not None and (id(parent) not in only or not _is_parent(parent.name, node.name)):
                continue
            scale = parent.name.find_attr("scale")
            if not cmds.isConnected(str(scale), str(inverse)):  # a node key_hierarchy keys may be wired
                scale.connect(inverse)


def _is_parent(parent: DAGNode, child: DAGNode) -> bool:
    """True when `parent` is the node `child` sits under in Maya."""
    above = cmds.listRelatives(type(child).get_long_name(child), parent=True, fullPath=True) or []
    return above == [type(parent).get_long_name(parent)]


def _walk_hierarchy(root: DAGNode) -> tuple[list, dict]:
    """`root` and the transforms and joints below it, parents first, typed: the nodes
    `serialize_hierarchy` reads and `key_hierarchy` pairs by order, in the one order. With
    them, the count per type of the nodes left out: ``type="transform"`` also lists every
    type derived from it (constraints, ikHandles...), which no TransformData describes, so
    each is skipped with what is under it."""
    nodes   = [root]
    skipped = {}

    def _recurse(leaf):
        # a hierarchy the way it should be, unlike listRelatives(ad=True), whose order is its own
        for child in leaf.get_children(c=True, noIntermediate=True, type="transform") or []:
            if cmds.nodeType(_name_of(child)) not in SERIALIZED_NODE_TYPES:
                below = cmds.listRelatives(_name_of(child), ad=True, type="transform", f=True)
                for name in [_name_of(child)] + (below or []):
                    node_type          = cmds.nodeType(name)
                    skipped[node_type] = skipped.get(node_type, 0) + 1
                continue
            nodes.append(child)
            _recurse(child)

    _recurse(root)
    return nodes, skipped


# what `key_hierarchy`'s flags say (`_key_flag`)
_KEY_FLAGS = _MappingProxyType({
    "uuids":  "True (a node of the data keys the node holding its uuid) or False (the default: by index)",
    "index":  "True (the default: node i of the data keys node i, in order) or False (the one node of its name)",
    "strict": "True (the default: what cannot be keyed is a ValueError) or False (it is skipped, with a warning)",
})


def _key_flag(method: str, flag: str, value: Any) -> bool:
    """A flag of `key_hierarchy` (`method`, ``"Joint.key_hierarchy"``) as a bool: anything but a
    bool (None, ``1``) is a TypeError naming it and what it says, before anything is read."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{method}: {flag}= is {_KEY_FLAGS[flag]}, not {value!r}")
    return bool(value)


def _own_rows(hierarchy: HierarchyData, own: Any) -> list:
    """The nodes of `hierarchy` a node keys by uuid or by name: `own`, the node of the data
    that is the node itself, with the ones under it in the data, in the data's order; every
    one when the data does not hold the node (`own` None: a group above the skeleton)."""
    if own is None:
        return list(hierarchy)
    below = {id(data) for data in own.get_branch()} | {id(own)}
    return [data for data in hierarchy if id(data) in below]


def _pair_by_uuid(top: DAGNode, hierarchy: HierarchyData) -> tuple[list, list]:
    """``key_hierarchy(uuids=True)``: ``(pairs, missed)``, a ``(node data, Maya node)`` per
    expected node of `hierarchy` (`_own_rows`) whose uuid one node at or below `top` holds,
    and the others described."""
    long_top = type(top).get_long_name(top)
    held     = type(top)._raw_get_uuid(top).upper()
    own      = next((data for data in hierarchy if data.uuid and str(data.uuid).upper() == held), None)
    pairs, missed = [], []
    for data in _own_rows(hierarchy, own):
        found = cmds.ls(str(data.uuid).upper(), long=True) if data.uuid else []
        found = [path for path in found if path == long_top or path.startswith(long_top + "|")]
        if len(found) == 1:
            pairs.append((data, _cast(found[0])))
        else:
            missed.append(f"{data.name!r} ({', '.join(sorted(found)) or 'none'})")
    return pairs, missed


def _pair_by_name(top: DAGNode, hierarchy: HierarchyData) -> tuple[list, list]:
    """``key_hierarchy(index=False)``: ``(pairs, missed)``, a ``(node data, Maya node)`` per
    expected node of `hierarchy` (`_own_rows`: `top` is the one node of the data of its name,
    or none of them) whose name one node at or below `top` has, and the others described: no
    node, several, or a name several expected nodes of the data share."""
    long_top = type(top).get_long_name(top)
    named    = {}
    for path in [long_top] + (cmds.listRelatives(long_top, allDescendents=True, type="transform", fullPath=True) or []):
        named.setdefault(path.rsplit("|", 1)[-1], []).append(path)
    mine = [data for data in hierarchy if data.name == long_top.rsplit("|", 1)[-1]]
    rows = _own_rows(hierarchy, mine[0] if len(mine) == 1 else None)
    held = {}
    for data in rows:
        held[data.name] = held.get(data.name, 0) + 1
    pairs, missed = [], []
    for data in rows:
        found = named.get(data.name, [])
        if held[data.name] > 1:
            missed.append(f"{data.name!r} ({held[data.name]} nodes of the data have the name)")
        elif len(found) == 1:
            pairs.append((data, _cast(found[0])))
        else:
            missed.append(f"{data.name!r} ({', '.join(sorted(found)) or 'none'})")
    return pairs, missed


def _put_back_selection(before: list[str]) -> None:
    """Selects `before` again, in its order (``cmds.ls(orderedSelection=True,
    long=True)`` read before a create), or clears a selection a create made
    when it was empty."""
    if before:
        _cmds_edit.select(before[:-1], replace=True)
        _cmds_edit.select(before[-1], add=True)
    elif cmds.ls(selection=True):
        _cmds_edit.select(clear=True)


def _typed_nodes(names: list[str]) -> List:
    """The List of the typed nodes `names` name (`_build_hierarchy`'s, parents
    first): `create_hierarchy`'s and `load_hierarchy`'s result."""
    from rig._internal.list import List

    return List([_cast(name) for name in names])


def _set_parent_uuid(node: TransformData, uuid: str | None) -> None:
    """Sets the uuid of the parent of the TransformData `node` (None: a root):
    its ``parent_uuid`` since cgmath 2.0.0, its ``parent_node`` on 1.0.4
    (``parent_node`` is the 1.0.4 name: a deprecated alias that takes a uuid
    only on 2.0.0, gone on 3.0.0)."""
    field = "parent_uuid" if hasattr(type(node), "parent_uuid") else "parent_node"
    setattr(node, field, uuid)


def _compensated_inverse_scale(node: DAGNode) -> np.ndarray | None:
    """The inverse scale `Transform.set_matrix` takes out of the matrix of a
    joint whose segment scale compensate is on and whose inverse scale is not
    uniform and positive (a joint under a non-uniformly or negatively scaled
    parent joint): ``xform -m`` misplaces the first, and flips the scale and
    rotate of the second (which moves its compensated children, whose inverse
    scale is its scale). None for every other node (a uniform positive
    inverse scale, (2, 2, 2) say, ``xform -m`` sets exactly, its channels
    kept)."""
    if not type(node)._raw_get_mobject(node).hasFn(OpenMaya.MFn.kJoint):
        return None
    if not node.find_attr("segmentScaleCompensate").get():
        return None
    inverse_scale = node.find_attr("inverseScale").get()[0]
    if inverse_scale[0] == inverse_scale[1] == inverse_scale[2] > 0:
        return None
    return np.array(inverse_scale, dtype=float)


class Transform(DAGNode):
    """
    Transform node class
    """

    NATIVE_NODE_TYPE = "transform"
    FN_SET           = OpenMaya.MFnTransform

    # --- creation

    def duplicate_geometry(
        self,
        name:        str  | None             = None,
        parent:      str  | Transform | None = None,
        clean_sets:  bool                    = True,
        clean_xform: bool                    = True,
    ) -> Transform:
        """Duplicates the geometry under this node.

        Args:
            name: Name for the duplicated geometry.
                If None, use the input geometry name.
            parent: Parent of the duplicated geometry.
                If None, parent to world.
            clean_sets: If True, remove the duplicate from display layers and sets.
            clean_xform: If True, freezes the duplicate's xform channels and
                moves its pivots to its own origin (`set_pivots`' object
                space: under a moved parent, the parent's origin).

        Returns:
            The duplicated geometry name.
        """
        dup = self.duplicate(parentOnly=False)[0]
        dup.set_parent(parent or None)
        dup.rename(name or type(self).get_short_name(self))
        dup.delete(ch=True)
        dup.set_xform_attrs_locked(False)

        # clean shapes
        nodes = [dup]
        for each in dup.get_children(children=True) or []:
            if each.has_base_type("transform"):
                _cmds_edit.delete(each)
            elif each.find_attr("intermediateObject").get():
                _cmds_edit.delete(each)
            else:
                nodes.append(each)

        if clean_sets:
            # remove geom from all display layers
            _cmds_edit.editDisplayLayerMembers("defaultLayer", *nodes, noRecurse=True)
            # remove geom from all object sets
            for node in nodes:
                attrs = cmds.listConnections(
                    node,
                    source      = False,
                    destination = True,
                    connections = True,
                    plugs       = True,
                    type        = "objectSet",
                    exactType   = True,
                )
                if attrs:
                    for src, dst in zip(*[iter(attrs)] * 2):
                        _cmds_edit.disconnectAttr(src, dst)

        # clean xform
        if clean_xform:
            dup.freeze(t=True, r=True, s=True, a=True)
            dup.set_pivots((0, 0, 0))

        return dup

    # --- hierarchy

    def _search_regex(self, match_name, exact_match, _regex):
        # construct search regex, reused in recursions
        if _regex:
            return _regex
        elif match_name:
            names = [match_name] if isinstance(match_name, str) else match_name
            if exact_match:
                return re.compile(r"|".join([rf"^{_name_of(x)}$" for x in names])).match
            else:
                return re.compile(r"|".join([rf"{_name_of(x)}" for x in names])).search

    def iter_shapes(
        self,
        shape_type:   str       | list[str],
        match_name:   list[str] | str | None = None,
        exact_match:  bool                   = False,
        as_transform: bool                   = True,
        _regex:       str       | None       = None,
    ) -> "DAGNode":
        """A generator that traverse the hierarchy of this node and yields shapes
        nodes.

        Args:
            shape_type: The shape type.
            match_name: One or more of names to match, can be regex.
            exact_match: If True, match exact names, otherwise do partial match.
            as_transform: If True, returns the transform node instead of shape node.

        Yields:
            Transform nodes that meet criteria.
        """
        regex = self._search_regex(match_name, exact_match, _regex)
        found = set()
        for node in reversed(
            self.get_children(allDescendents=True, noIntermediate=True, type=shape_type)
        ):
            if as_transform:
                node = node.get_parent()
            if node not in found and (not regex or regex(type(node).get_short_name(node))):
                yield node

    def find_shape(self, shape_type: str, match_name: str, **kwargs) -> DAGNode:
        """Returns the first shape transform matching given criteria,
        or None if no match.

        Args:
            shape_type: The shape type.
            match_name: One or more of names to match, can be regex.
            kwargs: Keyword arguments passed to `iter_shape_xforms` method.
        """
        kwargs["match_name"] = match_name
        it                   = self.iter_shapes(shape_type, **kwargs)
        return next(it, None)

    def get_shapes(self, no_interm: bool = True) -> list[DAGNode]:
        """Returns the shape nodes under this transform.

        Args:
            no_interm: If True, skip intermediate objects.
        """
        return self.get_children(children=False, shapes=True, noIntermediate=no_interm)

    def get_shape(self) -> DAGNode | None:
        """Returns the first non-intermediate shape node under this transform."""
        shapes = self.get_shapes(no_interm=True)
        if shapes:
            return shapes[0]

    # --- transformation

    def iter_xform_attrs(self, axis_only: bool = False) -> Attribute:
        """Iterates over all transform attrs on this node.

        Args:
            axis_only: If True, only return attrs per axis.
                Otherwise also return their parent attrs: "t", "r", "s".

        Yields:
            Transform attributes.
        """
        for at in "trs":
            if not axis_only:
                yield self.find_attr(at)
            for attr in (at + ax for ax in "xyz"):
                yield self.find_attr(attr)

    def set_xform_attrs_locked(self, locked: bool) -> None:
        """Locks or Unlocks all transform attributes on this node."""
        for attr in self.iter_xform_attrs():
            if locked:
                type(attr).lock(attr)
            else:
                type(attr).unlock(attr)

    def freeze(self, *args, **kwargs) -> None:
        """Thin wrapper around cmds.makeIdentity() function."""
        kwargs["apply"] = kwargs.get("apply", kwargs.get("a", True))
        _cmds_edit.makeIdentity(_name_of(self), *args, **kwargs)

    def get_matrix(
        self, *, world_space: bool = False, as_transform_matrix: bool = False
    ) -> OpenMaya.MMatrix | OpenMaya.MTransformationMatrix:
        """Returns the matrix of this node: its local matrix by default, its
        world matrix with ``world_space=True``.

        The local matrix is ``.matrix`` (``xform -q -os -m``): the node's own
        channels (a joint's orient, rotate axis and inverse scale too), in its
        parent's space. It EXCLUDES ``offsetParentMatrix``: on a node with an
        offset it is not relative to the parent (``.dagLocalMatrix`` is
        ``.matrix`` times the offset). The world matrix is ``worldMatrix[0]``
        (``.matrix`` times ``.parentMatrix``, the offset included).

        Args:
            world_space: By keyword only. If True return the world matrix,
                otherwise the local one.
            as_transform_matrix: By keyword only. If True return an
                MTransformationMatrix.
        """
        if world_space:
            attr = self.find_attr("worldMatrix")[0]
        else:
            attr = self.find_attr("matrix")
        mat = OpenMaya.MFnMatrixData(attr._mplug.asMObject()).matrix()
        if as_transform_matrix:
            return OpenMaya.MTransformationMatrix(mat)
        return mat

    def set_matrix(
        self,
        matrix:      OpenMaya.MMatrix | OpenMaya.MTransformationMatrix | list[float],
        *,
        world_space: bool                                                            = False,
    ) -> None:
        """Sets the matrix of this node, the inverse of `get_matrix`: its local
        matrix by default (``xform -os -m``, so ``.matrix``, the offset parent
        matrix left as it is), its world matrix with ``world_space=True``,
        taken into the parent's space first (the offset parent matrix
        included) and set as the local one: placed as ``xform -ws -m`` places
        it, and redone exactly (Maya's redo of ``xform -ws -m`` takes the
        parent's matrix out twice).

        The matrix is exact, not every channel: a transform with pivots gets
        its translate and pivot translates rewritten, and a negative scale can
        come back on another axis, its rotate solved again (the node is placed
        exactly, but a joint's segment scale compensated children, whose
        inverse scale is its scale, move). A joint keeps its orient and rotate
        axis: its rotate, scale and translate are solved. A joint holds no
        shear, so a matrix that needs one (a world matrix under a non-uniformly
        scaled parent, say) lands near, not on. A joint whose segment scale
        compensate is on, under a non-uniformly or negatively scaled parent
        joint (its inverse scale wired, as ``cmds.parent`` and the joint tools
        leave it), is set exactly as well: ``xform -m`` alone misplaces it
        (2.24 world units for its own matrix) or flips its scale and rotate;
        its segment scale compensate is turned off for the set and back on,
        also when the set fails, in one undo step (locked or driven, it is
        Maya's RuntimeError, nothing set).

        Args:
            matrix: The matrix to set (16 numbers, an MMatrix or an
                MTransformationMatrix).
            world_space: By keyword only. If True set the world matrix,
                otherwise the local one.
        """
        if isinstance(matrix, OpenMaya.MTransformationMatrix):
            matrix = matrix.asMatrix()
        if world_space:
            # into the parent's space, its offset parent matrix included: the local matrix
            to_local = np.array(type(self).get_mdagpath(self).exclusiveMatrixInverse()).reshape(4, 4)
            matrix   = (np.array(matrix, dtype=float).reshape(4, 4) @ to_local).ravel().tolist()

        inverse_scale = _compensated_inverse_scale(self)
        if inverse_scale is None:
            _cmds_edit.xform(_name_of(self), objectSpace=True, matrix=matrix)
            return

        # the segment scale compensated joint: the local matrix, its 3x3 times
        # diag(inverse scale), set with the compensation off, gives the channels
        # that hold the matrix once it is back on
        local = np.array(matrix, dtype=float).reshape(4, 4)
        local[:3, :3] *= inverse_scale
        name = _name_of(self)
        with _undo_chunk(f"rig.{type(self).__name__}.set_matrix"):
            _cmds_edit.setAttr(f"{name}.segmentScaleCompensate", False)
            try:
                _cmds_edit.xform(name, objectSpace=True, matrix=local.ravel().tolist())
            finally:
                _cmds_edit.setAttr(f"{name}.segmentScaleCompensate", True)

    def get_rotate_pivot(self, *, world_space: bool = False) -> OpenMaya.MPoint:
        """Returns the rotate pivot, a point in this node's object space (its
        own frame, as a point of a shape under it: ``xform -q -os -rp``), or
        in world space with ``world_space=True`` (``xform -q -ws -rp``). The
        object-space point times the world matrix is the world one. It is not
        always the ``rotatePivot`` attribute's value (``xform -q -rp``).

        Args:
            world_space: By keyword only. If True, in world space.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return type(self)._raw_get_fn_set(self).rotatePivot(space)

    def get_scale_pivot(self, *, world_space: bool = False) -> OpenMaya.MPoint:
        """Returns the scale pivot, a point in this node's object space
        (``xform -q -os -sp``), or in world space with ``world_space=True``
        (``xform -q -ws -sp``), as `get_rotate_pivot`.

        Args:
            world_space: By keyword only. If True, in world space.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return type(self)._raw_get_fn_set(self).scalePivot(space)

    def set_pivots(
        self, pivot: OpenMaya.MPoint | OpenMaya.MVector | list[float], *, world_space: bool = False
    ) -> None:
        """Sets both the scale and rotate pivots to one point, in this node's
        object space (``xform -os -piv``) or, with ``world_space=True``, in
        world space (``xform -ws -piv``): the spaces `get_rotate_pivot` and
        `get_scale_pivot` read, so ``set_pivots(node.get_rotate_pivot())``
        leaves the rotate pivot where it is, in either space. The node does
        not move (Maya writes the pivot translates that keep it). A joint's
        pivots are not set: Maya ignores ``xform -piv`` on a joint.

        Args:
            pivot: The point: an MPoint (as the getters return it), an
                MVector or 3 numbers.
            world_space: By keyword only. If True, in world space.
        """
        if isinstance(pivot, (OpenMaya.MPoint, OpenMaya.MFloatPoint)):
            # 4 values, which xform refuses: the point's x, y, z (anything else goes to
            # xform as given: OpenMaya would read a plug as a sequence and crash)
            point = OpenMaya.MPoint(pivot).cartesianize()
            pivot = (point.x, point.y, point.z)
        _cmds_edit.xform(
            _name_of(self),
            pivots      = pivot,
            worldSpace  = world_space,
            objectSpace = not world_space,
        )

    def match_matrix(self, target_transform: Transform, *, world_space: bool = False) -> None:
        """Matches the matrix of another transform node: its local matrix,
        set as this node's (`get_matrix` / `set_matrix`), by default; its world
        matrix with ``world_space=True``.

        The local match is a pose copy: it puts this node on the target only
        when both have the same parent space (siblings, say), as the values
        are the same. ``world_space=True`` snaps this node onto the target
        wherever their parents are (a joint holds no shear: under a
        non-uniformly scaled parent it lands near).

        Args:
            target_transform: The transform node to match.
            world_space: By keyword only. If True, match the world matrix.
        """
        matrix = target_transform.get_matrix(world_space=world_space)
        self.set_matrix(matrix, world_space=world_space)

    # --- serialization

    def serialize(self) -> TransformData:
        """Serialize this node to a TransformData, the inverse of
        ``Transform.create(transform_data)``."""
        from cgmath.hierarchy import TransformData

        node   = TransformData(name=type(self).get_short_name(self), uuid=type(self)._raw_get_uuid(self))
        parent = self.get_parent()
        if parent is not None:
            _set_parent_uuid(node, type(parent)._raw_get_uuid(parent))

        # if there is a locatorShape, set this type as "locator"
        node.node_type = cmds.nodeType(_name_of(self))

        if node.node_type == "joint":
            node.joint_orient             = np.array(cmds.getAttr(f"{_name_of(node)}.jo")[0])
            node.segment_scale_compensate = bool(cmds.getAttr(f"{_name_of(node)}.ssc"))
            node.rotate_axis              = np.array(self.find_attr("ra").get()[0])
            node.radius                   = self.find_attr("radius").get()

        else:
            locators = cmds.listRelatives(
                _name_of(self), c=True, ni=True, f=False, type="locator"
            )

            # the shape's localScale/localPosition are not carried, only the kind
            if locators:
                node.node_type = "locator"

        node.scale        = np.array(self.find_attr("s").get()[0])
        node.rotate       = np.array(self.find_attr("r").get()[0])
        node.translate    = np.array(self.find_attr("t").get()[0])
        node.rotate_order = self.find_attr("ro").get()
        node.rotate_axis  = np.array(self.find_attr("ra").get()[0])
        node.visibility   = self.find_attr("v").get()

        # capture user defined attrs
        attrs = self.list_attr(ud=True)
        if attrs:
            node.user_defined_attributes = {}
            for att in attrs:
                try:
                    node.user_defined_attributes[type(att)._raw_get_name(att)] = self._serialize_user_attr(att)

                # gracfully pass on this one
                except Exception as e:
                    LOGGER.warning(f"Skipped user defined attr {att}: {e}")

        return node

    @staticmethod
    def _serialize_user_attr(att: Attribute) -> dict:
        """Returns the cmds.addAttr() spec of a user defined attr, with its value,
        keyable and channel box states. Compound parents carry no value, their
        children do. Multi values are [logical index, value] pairs. An attr that
        holds geometry (`_holds_geometry`) raises, whatever it holds right now:
        `serialize` skips it, with a warning (a compound's spec counts the
        children that load back).
        """
        mobject = type(att)._raw_get_mobject(att)
        if _holds_geometry(mobject):
            raise ValueError("it holds geometry, whose value never loads back")

        data = {}

        # a typed attr that holds any data (a Typed spec's): addAttr(attributeType=
        # "typed"), and as its value only the numbers it holds (what a connection
        # fed it is not saved: serialization carries no connections)
        if _is_generic_slot(att):
            data["attributeType"] = "typed"
            if type(att)._raw_is_multi(att):
                data["multi"] = True
                data["value"] = [
                    [int(i), att[i].get()]
                    for i in att.get_logical_indices()
                    if cmds.getAttr(_name_of(att[i]), type=True) == "typed"
                ]
            elif cmds.getAttr(_name_of(att), type=True) == "typed":
                data["value"] = att.get()
            parent = att.get_parent()
            if parent is not None:
                data["parent"] = type(parent)._raw_get_name(parent)
            data["keyable"]     = type(att).is_keyable(att)
            data["channel_box"] = type(att).is_channel_box(att)
            return data

        # a multi's own plug reports "compound", addAttr() wants the element
        # type. it is read off the attribute, a getAttr on an element would
        # create it.
        if type(att)._raw_is_multi(att):
            data["multi"] = True
            if type(att).is_typed(att):
                data_type = OpenMaya.MFnTypedAttribute(type(att)._raw_get_mobject(att)).attrType()
                if data_type not in TYPED_DATA_NAMES:
                    raise ValueError(f"Unsupported multi data type: {data_type}")
                data["dataType"] = TYPED_DATA_NAMES[data_type]
            else:
                data["attributeType"] = type(att).get_attribute_type(att)
            data["value"] = [[int(i), att[i].get()] for i in att.get_logical_indices()]

        else:
            # typed attrs (string, stringArray, doubleArray, etc.)
            # are created with cmds.addAttr(dataType=...)
            if type(att).is_typed(att):
                data["dataType"] = type(att)._raw_get_data_type(att)
            else:
                data["attributeType"] = type(att)._raw_get_data_type(att)

            if att._mplug.isCompound:
                compound = OpenMaya.MFnCompoundAttribute(mobject)
                data["numberOfChildren"] = sum(
                    not _holds_geometry(compound.child(i)) for i in range(compound.numChildren())
                )
            else:
                data["value"] = att.get()

        if data.get("attributeType") == "enum":
            data["enumName"] = ":".join(type(att).get_enums(att))

        parent = att.get_parent()
        if parent is not None:
            data["parent"] = type(parent)._raw_get_name(parent)

        data["keyable"]     = type(att).is_keyable(att)
        data["channel_box"] = type(att).is_channel_box(att)
        return data

    def serialize_hierarchy(self, bindpose: bool | str | DGNode = False) -> HierarchyData:
        """Serialize a whole hierarchy: this node and the transforms and joints
        below it, parents first.

        Args:
            bindpose: ``False`` (the default) reads the scene as it is.
                ``True`` reads the hierarchy at its bind pose: every dagPose
                with ``bindPose`` on that holds one of its nodes (a skinCluster
                makes one; binding another influence set makes another). Poses
                that hold the same node must agree on it.
                A dagPose name (``"bindPose5"``, ``"char:bindPose1"``) or node
                (``Node("bindPose5")``) reads it at that one pose; any dagPose
                qualifies, a saved ``tPose`` too.

        The result is what ``serialize_hierarchy()`` would return after Maya's
        own Go to Bind Pose (``dagPose -restore -global``), and nothing is
        restored: the scene, the selection and the undo queue are unchanged.
        A pose member takes its scale, rotate, translate, rotate axis and joint
        orient from the pose, its rotation re-expressed in the node's current
        rotate order (as the restore does); everything else (segment scale
        compensate, radius, visibility, user attributes, names, parents) and
        every node the pose does not hold (an end joint that is no influence,
        a helper added after the bind) keep the scene's values. At the bind
        pose the result is bit-identical to ``bindpose=False``.

        A user attribute that holds geometry (typed ``mesh``, ``nurbsCurve``,
        ``nurbsSurface``, ``lattice`` ... data) is skipped, with a warning: its
        value never loads back.

        Raises (every check runs before anything is read into the result):
            TypeError: ``bindpose`` is not False, True, a str or a node.
            NodeNotFoundError / AmbiguousNodeError: the name names no node,
                or several.
            NodeTypeError: the name is not a dagPose.
            ValueError: ``True`` and no bind pose holds a node of the
                hierarchy; the named pose holds none of its nodes; ``True``
                and the bind poses disagree on a node (name one); a member's
                DAG parent is not the parent it was posed under (a member is
                stored relative to it: restore the parent or save a new pose).

        Returns:
            A HierarchyData object.
        """
        from cgmath.hierarchy import HierarchyData

        # the argument is checked before the walk; bool before int (True is an int)
        if bindpose is not False:
            if isinstance(bindpose, bool):
                pass
            elif isinstance(bindpose, DGNode):
                bindpose = _name_of(bindpose)
            elif not isinstance(bindpose, str) or isinstance(bindpose, Attribute):
                raise TypeError(
                    f"bindpose= takes False, True or a dagPose name, not {bindpose!r}"
                )

        # serialize the root
        root = self.serialize()
        _set_parent_uuid(root, None)

        # then the nodes below it, in the walk's order
        nodes, skipped = _walk_hierarchy(self)
        tree = [root] + [node.serialize() for node in nodes[1:]]
        objs = [type(node)._raw_get_mobject(node) for node in nodes]  # their MObjects, for the bind pose

        if bindpose is not False:
            from rig.nodetypes import _bindpose

            for index, entry in _bindpose.resolve(type(self).get_short_name(self), objs, bindpose).items():
                _bindpose.apply(tree[index], entry, objs[index])

        if skipped:
            counts = ", ".join(f"{v} {k}" for k, v in sorted(skipped.items()))
            LOGGER.warning(f"Skipped nodes that can't be serialized: {counts}")

        return HierarchyData(tree)

    @classmethod
    @_typed_creator
    def create_hierarchy(
        cls,
        hierarchy:   HierarchyData,
        parent:      Union[str, None] = None,
        *,
        world_space: bool             = False,
        uuids:       bool             = False,
    ) -> List:
        """Creates a hierarchy from a HierarchyData object, and keys a ClipData.

        Args:
            hierarchy: A HierarchyData object (`serialize_hierarchy`), or a
                ClipData, built and keyed (below); it is left as it was.
            parent: An existing node to make the roots under. None: the world.
            world_space: By keyword only, a bool (anything else is a TypeError,
                before anything is made); it changes nothing without `parent`.
                The data is local by nature: it holds each node's channels, no
                matrix, and not where its root's parent was. False (the
                default): the roots take their values as local values under
                `parent`, so a hierarchy made again under the node it was
                serialized under lands where it was, as far as the channels
                place it (the data holds no offset parent matrix, pivot or
                shear, and a joint made from it has no inverse scale wired; a
                ClipData's has, below).
                True: each root keeps its world matrix in the data, its values
                read as world values (its place in the scene it came from only
                if its root sat at the world). A joint root lands off under a
                scaled parent when its segment scale compensate is on (its
                values take the parent's scale out as an inverse scale would,
                and none is wired), and under a non-uniformly scaled one in any
                case (it holds no shear); a ClipData's compensating joints hold
                both, below.
            uuids: By keyword only, a bool (anything else is a TypeError,
                before anything is made). False (the default): every node
                keeps the uuid Maya gives it, so one hierarchy builds any
                number of times, a twin of a live hierarchy too (below). True:
                every node made takes its data's uuid (a recorded
                ``rename -uuid``, in the call's undo step), so the nodes are
                found by it (``Node(uuid)``, ``cmds.ls(uuid)``, a shape's
                ``parent_uuid``, ``key_hierarchy(clip, uuids=True)``) and
                `serialize_hierarchy` of them gives the same uuids back. A
                uuid a scene node already holds, or one the data holds twice,
                is then a ValueError before anything is made, naming both: a
                second copy of a hierarchy that is in the scene takes fresh
                uuids first (``hierarchy.renew_uuids()``; a Scene's
                ``renew_uuids()``). A node whose data holds no uuid keeps
                Maya's.

        Returns:
            A List of the new nodes, typed as ``Transform.create`` types its
            node (``Joint`` for a joint, ``Transform`` for a transform and a
            locator's transform), parents first:
            ``nodes[0]`` is the first root. Maya renames a node whose name is
            taken (``root1``), as for a twin of a hierarchy that is in the
            scene::

                skel = Transform.create_hierarchy(Node("root").serialize_hierarchy())
                skel[1].tx.get()

        A ClipData is built and keyed, no keyword: the type decides. Only what
        changes gets keys: a column (x, y or z) of a node's scale, rotate,
        translate, rotate axis or joint orient with any difference across the
        frames gets one curve, with a key on every frame; a flat one is a plain
        value, with no curve, so a one-frame clip makes no curve. Frame for
        frame: key k lands on frame ``start_frame + k`` whatever the scene's
        rate; a clip at another fps logs one warning naming both, keys or none,
        and the scene's time unit and playback range are not touched. A clip
        defines no tangents: the keys take Maya's global default (``keyTangent
        -query -global``). With `parent` and ``world_space=True`` the roots are
        solved on every frame, against the parent's matrix as it is now, so
        each frame's world matrix is the clip's, and their solved channels are
        keyed where they change; a parent with shear, and a non-uniformly
        scaled one over a root that is a transform, or a joint that does not
        compensate, stay off (the channels hold no shear). A NaN or an infinity
        in a column that is keyed is a ValueError, and a missing undo plug-in,
        with keys to write, a RuntimeError, before anything is made.

        A clip's joints evaluate what its world matrices hold, scale included:
        a joint with segment scale compensate on takes its parent's scale out,
        on every frame. Under a joint of the clip it is wired,
        ``parent.scale >> joint.inverseScale``, as ``cmds.parent`` wires one;
        a root solved under a scaled parent holds that parent's world scale in
        ``inverseScale``, a plain value; under a root that does not compensate
        and gave its scale to that solve, it holds that root's scale in the
        clip, plain, or keyed where it moves. Under a transform of the clip it
        is left unwired, as Maya leaves it (it lands where Maya puts it, not
        where cgmath's ``world_matrix`` says when that transform is scaled). A
        plain HierarchyData wires nothing.

        To build a clip without keys, pass a plain HierarchyData:
        ``clip.rest_pose``, when the clip has one::

            clip = ClipData.load_fbx("walk.fbx")
            walk = Transform.create_hierarchy(clip)            # nodes and keys
            pose = Transform.create_hierarchy(clip.rest_pose)  # nodes only

        Inside ``with container()`` every node the call made joins the scope,
        a clip's curves too (``container=False`` leaves them all unregistered);
        the nodes keep the hierarchy's names (the creates inside this one are
        nested, so they add no prefix).

        One undo step, a clip's curves and keys in it; it selects the nodes it
        made. `load_hierarchy` (from a file or the data, the selection kept)
        returns the same List. A path here is a TypeError naming it.
        """
        if isinstance(hierarchy, (str, os.PathLike)):
            raise TypeError(
                f"{cls.__name__}.create_hierarchy() takes the data, not a file: "
                f"{cls.__name__}.load_hierarchy({os.fspath(hierarchy)!r}) reads one"
            )
        world_space = _files.space_flag(f"{cls.__name__}.create_hierarchy", world_space, _HIERARCHY_SPACE)
        uuids       = _files.uuid_flag(f"{cls.__name__}.create_hierarchy", uuids)
        with _undo_chunk(f"rig.{cls.__name__}.create_hierarchy"):
            return _typed_nodes(cls._build_hierarchy(hierarchy, parent, world_space, select=True, uuids=uuids))

    def key_hierarchy(
        self,
        clip:   HierarchyData,
        *,
        uuids:  bool          = False,
        index:  bool          = True,
        strict: bool          = True,
    ) -> List:
        """Keys a ClipData onto this node and the nodes below it, the inverse of
        `create_hierarchy` building them. Called on a node, as
        `serialize_hierarchy` is::

            root  = Transform("root")
            keyed = root.key_hierarchy(clip)                  # by order: node i takes the data's node i
            keyed = root.key_hierarchy(clip, index=False)     # by name
            keyed = root.key_hierarchy(clip, uuids=True)      # by uuid
            keyed = root.key_hierarchy(clip, strict=False)    # what cannot be keyed is skipped

        Args:
            clip: A ClipData, keyed as `create_hierarchy` keys one (the same
                rows: a column that changes gets a curve with a key on every
                frame, ``start_frame + k``; a flat one a plain value, the
                tangents Maya's global default), or a plain HierarchyData, which
                poses the nodes, no key. It is left as it was.
            uuids: By keyword only, a bool. False (the default): the nodes are
                paired as `index` says. True: each node of the data keys the
                node holding its uuid, at or below this one (what a
                `create_hierarchy` with ``uuids=True`` made, whatever its name
                and its place are now); `index` is not read.
            index: By keyword only, a bool, read when `uuids` is False. True
                (the default): by order, one to one. Node i of the data keys
                node i of this node and the transforms and joints below it,
                in `serialize_hierarchy`'s order (parents first), so the
                data's first node lands on this one; no name and no uuid is
                read. The counts need not agree: the keys stop where the
                nodes or the data run out, with one warning naming both
                counts. False: each node of the data keys the one node of its
                name, at or below this one.
            strict: By keyword only, a bool. True (the default): a node of the
                data with no node to key, an attribute its node lacks and a
                locked channel are each a ValueError naming every one, before
                any edit. False: each is skipped, the rest is keyed, and one
                warning names every skip.

        By uuid and by name the nodes are looked for at or below this node
        only, so two copies of a skeleton each take the clip by name, and a
        branch is keyed alone: when this node is itself a node of the data
        (it holds its uuid; by name, it alone in the data has its name), only
        that node and the ones under it in the data are keyed::

            Transform("left_clavicle").key_hierarchy(clip, uuids=True)   # the clavicle and below

        On a node the data does not hold (a group above the skeleton) every
        node of the data is expected. By order there is no such rule: the
        data's first node lands on this one whatever it is, so a branch takes
        ``uuids=True``, ``index=False`` or a clip of its own, and a transform
        or a joint more or less below this node shifts every later one, with
        no error.

        What `strict` refuses, or skips, is of two kinds. A node: by uuid,
        no node at or below this one holds it; by name, none or several have
        it, or several expected nodes of the data share it. A channel: an
        attribute the node lacks, which is a joint's data on a node that is
        no joint (its joint orient, segment scale compensate, radius and draw
        style; a transform's data sets nothing a joint lacks), and a locked
        channel the data changes or keys. A locked axis is skipped alone: the
        node's other axes and channels are set and keyed. Whatever `strict`
        is, a NaN in a column that is keyed is a ValueError and a missing
        undo plug-in a RuntimeError (`create_hierarchy`'s checks), a flag
        that is no bool, a path and anything that is no HierarchyData a
        TypeError, all before any edit.

        A joint that compensates takes its parent's scale
        (``parent.scale >> joint.inverseScale``, left as it is when wired)
        when its parent in the data is a joint keyed onto its parent in Maya.
        Keys a channel holds already are replaced on the clip's frames.

        Returns:
            A List of the nodes keyed, typed, in the data's order (parents first).

        One undo step named ``rig.<class>.key_hierarchy``, the class this
        node's (``rig.Joint.key_hierarchy`` on a joint); the selection is
        left as it was.
        """
        from cgmath.hierarchy import HierarchyData
        from rig._internal.list import List as _List

        method = f"{type(self).__name__}.key_hierarchy"
        if isinstance(clip, (str, os.PathLike)):
            raise TypeError(f"{method}() takes the data, not a file: ClipData.load(...) reads one")
        if not isinstance(clip, HierarchyData):
            raise TypeError(f"{method}() takes a ClipData or a HierarchyData, not a {type(clip).__name__}")
        uuids  = _key_flag(method, "uuids",  uuids)
        index  = _key_flag(method, "index",  index)
        strict = _key_flag(method, "strict", strict)

        hierarchy = clip.copy()
        missed    = []
        notes     = []
        skips     = []
        if uuids:
            pairs, missed = _pair_by_uuid(self, hierarchy)
            sought = "holding their uuid"
        elif index:
            nodes = _walk_hierarchy(self)[0]
            pairs = list(zip(hierarchy, nodes))
            if len(nodes) != len(hierarchy):
                notes.append(
                    f"{method}: the data holds {len(hierarchy)} node(s), {self} and what is below it "
                    f"{len(nodes)}: the first {len(pairs)} were keyed, in order"
                )
        else:
            pairs, missed = _pair_by_name(self, hierarchy)
            sought = "of their name"
        if missed and strict:
            raise ValueError(
                f"{method}: {len(missed)} node(s) of the data have no one node {sought} at or below {self}: "
                f"{', '.join(missed)}; strict=False skips them, nothing was keyed"
            )
        if missed:
            skips.append(
                f"{len(missed)} node(s) of the data with no one node {sought} at or below {self}: {', '.join(missed)}"
            )

        for data, node in pairs:
            data.name = node
        keyed = {id(data) for data, _ in pairs}
        plan  = _key_plan(hierarchy, keyed, strict, method)
        with _undo_chunk(f"rig.{method}"):
            type(self)._key_found(hierarchy, plan, keyed)

        skips.extend(plan[2])
        if skips:
            notes.append(f"{method}: strict=False skipped {'; '.join(skips)}")
        for note in notes:
            LOGGER.warning(note)
        return _List([node for _, node in pairs])

    @staticmethod
    def _key_found(hierarchy: HierarchyData, plan: tuple | None = None, keyed: set | None = None) -> None:
        """[Internal] `key_hierarchy`'s edits, its nodes found: `hierarchy` a copy whose nodes'
        names are the Maya nodes they key (``rig.scene.create_scene`` gives the nodes it made,
        every one keyed). `plan` is `_key_plan`'s, made here when not given: the keys and the
        values are planned before the first edit (a NaN, a missing plug-in, a locked channel).
        `keyed` holds the ids of the nodes of `hierarchy` that are keyed, None for all."""
        from cgmath.hierarchy import ClipData

        rows, changes, _ = _key_plan(hierarchy) if plan is None else plan
        for node, values in changes:
            if values:
                node.set_attrs(skip_missing=True, **values)
        if isinstance(hierarchy, ClipData) and hierarchy.frame_count:
            _clip_inverse_scales(hierarchy, {}, only=keyed)
        _key_rows(rows)

    @classmethod
    @_typed_creator
    def _create_from_data(
        cls,
        inputs:   tuple,
        name:     Any   = None,
        parent:   Any   = None,
        uuids:    Any   = False,
        **kwargs: Any,
    ) -> DAGNode:
        """[Internal] ``Transform.create(transform_data)`` / ``Joint.create(
        joint_data)`` (user decision 2026-10-16): `DGNode.create` hands it the
        positional ``inputs`` of a create on a transform class.

        The node the data describes (its ``node_type``: a transform, a joint, a
        locator's transform), named the data's name
        (``name=`` / ``n=`` overrides it), made at identity under ``parent=`` /
        ``p=`` (None: the world; the data's own parent uuid is not used), given
        the data's uuid with ``uuids=True`` (a checked bool, as
        `create_hierarchy` takes it: a uuid a scene node holds is a ValueError
        before anything is made; the default, ``uuids=False``, or data without
        one, leaves the node the uuid Maya gives it), then given the data's
        values (as local values: scale, rotate, translate, rotate order and
        axis, visibility; a joint's orient, segment scale compensate, radius
        and draw style), its user-defined attributes, and last the attribute
        keywords, as ``create`` sets them (a keyword names an attribute of the
        node type, or one of the data's user-defined attributes).
        `create_hierarchy`'s per-node code builds it.

        Refused before anything is made: anything but one TransformData
        (TypeError, naming the keyword form: a name given there,
        ``Transform.create("ctl")``, names ``Transform.create(name='ctl')`` and
        ``Transform.define('ctl')``), a node type the data cannot
        rebuild, data of a type the class does not make (``Joint.create`` of a
        transform's data: TypeError naming ``Transform.create``; a joint's
        data on ``Transform`` makes a ``Joint``), a missing ``parent=``
        (NodeNotFoundError) and an attribute keyword the node type does not
        have (AttributeError).

        ONE undo step (``rig.<Class>.create``); the selection is left as it was
        (as ``Mesh.create``); a value that fails once the node exists deletes
        it before the error propagates. Inside ``with container()`` the nodes
        it made join the scope, as for every typed create: an explicit
        ``name=`` takes the flattened scope's prefix, the data's name is kept
        (as ``Mesh.create`` keeps its data's and ``create_hierarchy`` its
        hierarchy's names)."""
        from cgmath.hierarchy import SUPPORTED_NODE_TYPES, TransformData

        call = f"{cls.__name__}.create()"
        named = _positional_name(cls, inputs, built_from="TransformData")
        if named is not None:
            raise named
        if len(inputs) != 1 or not isinstance(inputs[0], TransformData):
            raise TypeError(
                f"{call} takes name= and parent= as keywords, or one TransformData "
                f"(got {_got(inputs)})"
            )
        data  = inputs[0]
        uuids = _files.uuid_flag(call, uuids)
        for short, value in (("n", name), ("p", parent)):
            given = kwargs.pop(short, None)
            if given is not None:
                if value is not None:
                    long_form = "name" if short == "n" else "parent"
                    raise TypeError(f"{call} takes {long_form}= or {short}=, not both")
                if short == "n":
                    name = given
                else:
                    parent = given
        node_type = data.node_type
        if node_type not in SUPPORTED_NODE_TYPES:
            raise TypeError(
                f"{call}: a TransformData of a {node_type!r} cannot be built; it "
                f"describes a {', a '.join(SUPPORTED_NODE_TYPES)}"
            )
        made_cls = _NODE_CLASS_DICT.get("joint") if node_type == "joint" else Transform
        if not issubclass(made_cls, cls):
            raise TypeError(
                f"{call}: the data describes a {node_type}, which "
                f"{cls.__name__} does not make; {made_cls.__name__}.create(data) makes it"
            )
        # the keywords: attributes of the node type (checked on it), or of the data's
        # own user-defined attributes, which the node has once it is built
        own   = {key: kwargs.pop(key) for key in list(kwargs) if key in (data.user_defined_attributes or {})}
        attrs = {}
        if kwargs:
            attrs = _attribute_keywords(made_cls.NATIVE_NODE_TYPE, kwargs, _DATA_CREATE_FLAGS, call)
        attrs.update(own)
        parent = None if parent is None else _parent_path(parent)
        name   = data.name if name is None else name
        if uuids:
            _files.free_uuids(call, [(f"the TransformData {data.name!r}", data.uuid)], _files.FRESH_UUID)

        before = cmds.ls(orderedSelection=True, long=True) or []
        with _undo_chunk(f"rig.{cls.__name__}.create"):
            node = _make_from_data(data, name, parent)
            try:
                if uuids:
                    _files.adopt_uuid(type(node).get_long_name(node), data.uuid)
                _set_values(node, data)
                if data.user_defined_attributes:
                    cls._create_user_attrs(node, data.user_defined_attributes)
                for attr, value in attrs.items():
                    getattr(node, attr) << value
                _barrier()  # the values journaled: a registration that fails deletes the node too
            except BaseException:
                try:
                    _cmds_edit.delete(type(node).get_long_name(node))
                except (RuntimeError, ValueError):
                    pass
                raise
            _put_back_selection(before)
        return node

    @classmethod
    @_typed_creator
    def _load_built(
        cls,
        hierarchy:   HierarchyData,
        parent:      Union[str, None] = None,
        *,
        world_space: bool             = False,
        uuids:       bool             = False,
    ) -> list:
        """[Internal] `load_hierarchy`'s create, a typed creator (so it joins the
        container scope as `create_hierarchy` does): the new nodes, typed,
        parents first; the selection is put back as it was."""
        return _typed_nodes(cls._build_hierarchy(hierarchy, parent, world_space, select=False, uuids=uuids))

    @classmethod
    def _build_hierarchy(
        cls,
        hierarchy:   HierarchyData,
        parent:      Union[str, None],
        world_space: bool,
        select:      bool,
        uuids:       bool             = False,
    ) -> list[str]:
        """[Internal] The body of `create_hierarchy` and `load_hierarchy`: makes
        the nodes of `hierarchy` (a ClipData keyed, `_clip_rows`) and returns their
        names, parents first. With `select` they end selected; without, the
        selection is put back as it was before (in its order). With `uuids`
        every node takes its data's uuid (`_files.adopt_uuid`), the uuids
        checked free before any node is made (`_files.free_uuids`)."""
        from cgmath.hierarchy import SUPPORTED_NODE_TYPES, ClipData, HierarchyData, generate_uuid

        before = None if select else (cmds.ls(orderedSelection=True, long=True) or [])

        # make a copy of the hierarchy to handle new names
        # without affecting source data and set the desired parent.
        hierarchy = hierarchy.copy()

        # drop the nodes that can't be rebuilt from this data (constraints, ikHandles,
        # etc. found in older files) along with what is under them.
        # parents come first, same as the creation loop below expects.
        skipped = {}
        dropped = set()
        for node in hierarchy:
            if node.node_type in SUPPORTED_NODE_TYPES:
                if id(node.get_parent()) not in dropped:
                    continue
            dropped.add(id(node))
            skipped[node.node_type] = skipped.get(node.node_type, 0) + 1

        if skipped:
            hierarchy.list[:] = [x for x in hierarchy.list if id(x) not in dropped]
            counts            = ", ".join(f"{v} {k}" for k, v in sorted(skipped.items()))
            LOGGER.warning(f"Skipped nodes that can't be created: {counts}")

        # the data's uuids, free before any node is made
        if uuids:
            method = f"{cls.__name__}.{'create_hierarchy' if select else 'load_hierarchy'}"
            _files.free_uuids(method, ((f"the node {node.name!r}", node.uuid) for node in hierarchy), _RENEW)

        # if a parent is specified, make set the transforms relative to it
        under  = {}  # id of a root -> the long name of the node outside the data it goes under
        solved = {}  # id of a root solved under it -> its world scale, and the root's `_held_scale` or None
        if parent is not None:
            if parent in hierarchy:
                for node in hierarchy.get_roots():
                    node.set_parent(parent, world_space=world_space)

            # if the parent is not in the hierarchy, add it to compute matrices
            else:
                # (the data keeps the stand-in's uuid: a parent is a uuid since
                # cgmath 2.0.0, so the name it is created under is kept here)
                # its own parent is not in the data, so it stands in as a root that
                # carries its world matrix: the roots are solved against that. It takes
                # a new uuid: a clip's columns go by uuid, and a clip taken from the
                # outside node's own branch holds a column, animated, under the old one
                outside  = _cast(parent)
                obj      = outside.serialize()
                obj.uuid = generate_uuid()
                scales   = {}
                _set_parent_uuid(obj, None)
                hierarchy.list.insert(0, obj)
                obj.matrix = cmds.xform(type(outside).get_long_name(outside), q=True, matrix=True, worldSpace=True)
                if isinstance(hierarchy, ClipData) and world_space and hierarchy.frame_count:
                    # each node's scale, before a solve writes it. This read also lines the new node
                    # up in the blocks: a solve that did would take its scale from before the line-up
                    scales = dict(zip(map(id, hierarchy), np.array(hierarchy.frames.scale).swapaxes(0, 1)))
                for node in hierarchy.get_roots()[1:]:
                    node.set_parent(obj, world_space=world_space)
                    under[id(node)] = type(outside).get_long_name(outside)
                    if world_space:
                        # a root keeps its scale when it compensates or the parent has none to
                        # give it; any other takes the parent's in, and its joints hold its own
                        kept             = node.segment_scale_compensate or np.allclose(obj.scale, 1.0)
                        solved[id(node)] = (np.array(obj.scale, dtype=float), None if kept else scales.get(id(node)))
                hierarchy.list.pop(0)

        # a clip's keys, of the channels as solved: checked before any node is made
        rows = _clip_rows(hierarchy, solved)

        # create the hierarchy at identity before setting any attributes,
        # this is to avoid any default attrs being set by Maya.
        for node in hierarchy:
            # use the parent's long name, the new node is created in world
            # and may shadow the parent's short name.
            parent = node.get_parent()
            if id(node) in under:
                parent = under[id(node)]
            elif parent:
                # hierarchy data by its name, which holds its node once made (a
                # node's own name would be a Maya attr lookup); a node by its
                # long name
                if not isinstance(parent, DAGNode):
                    parent = getattr(parent, "name", parent)
                parent = type(parent).get_long_name(parent) if isinstance(parent, DAGNode) else _name_of(parent)

            # create node
            obj = _make_from_data(node, node.name, parent)
            if uuids:
                _files.adopt_uuid(type(obj).get_long_name(obj), node.uuid)

            # rename the node's string name with the node object
            # in case of scene duplicate during parenting.
            node.name = obj

        # set attributes after hierarchy creation.
        for i, node in enumerate(hierarchy):
            _set_values(node.name, hierarchy[i])

        # add user defined attrs once every transform is set,
        # so a bad attr can never leave the hierarchy at identity.
        for i, node in enumerate(hierarchy):
            node  = node.name
            specs = hierarchy[i].user_defined_attributes
            if specs:
                cls._create_user_attrs(node, specs)

        if isinstance(hierarchy, ClipData) and hierarchy.frame_count:
            _clip_inverse_scales(hierarchy, solved)
        _key_rows(rows)

        # return the created nodes
        created = [str(x.name) for x in hierarchy]
        if select:
            _cmds_edit.select(created)
        else:
            _put_back_selection(before)
        return created

    # --- files

    def save_hierarchy(self, path: str | os.PathLike, *, bindpose: bool | str | DGNode = False) -> str:
        """Saves this node's hierarchy to a cgmath file: its HierarchyData
        (`serialize_hierarchy`), the inverse of `load_hierarchy`.

        Args:
            path: A str or os.PathLike (``~`` expanded). The extension sets the
                format: ``.npz`` (recommended), ``.json`` or ``.pkl`` (never
                load a ``.pkl`` you did not write).
            bindpose: as `serialize_hierarchy` takes it: False (the scene as it
                is), True (at its bind pose(s)), or a dagPose's name or node.

        Returns:
            `path`, as written (``~`` expanded).

        A read of the scene: no undo step, the selection unchanged. The path
        and `bindpose` are checked before anything is serialized (an extension
        cgmath does not save: ValueError; a missing folder: FileNotFoundError;
        a wrong `bindpose`: `serialize_hierarchy`'s TypeError).
        """
        path = _files.check_save_path(path, f"{type(self).__name__}.save_hierarchy")
        self.serialize_hierarchy(bindpose=bindpose).save(path)
        return path

    @_statement
    def _load_hierarchy_new(
        cls,
        hierarchy,
        *,
        parent:      Union[str, DAGNode, None] = None,
        world_space: bool                      = False,
        container:   bool | None               = None,
        uuids:       bool                      = False,
    ):
        from cgmath.hierarchy import ClipData, HierarchyData

        method      = f"{cls.__name__}.load_hierarchy"
        world_space = _files.space_flag(method, world_space, _HIERARCHY_SPACE)
        uuids       = _files.uuid_flag(method, uuids)
        data        = _files.read(hierarchy, (ClipData, HierarchyData), method, "the hierarchy")
        if type(data) is ClipData and not data.frame_count and _files.is_path(hierarchy):
            # a file names no class: a rig's file builds as a clip of no frames, so it is the rig
            data = HierarchyData.load(os.path.expanduser(os.fspath(hierarchy)))
        kwargs = {} if container is None else {"container": container}
        if isinstance(parent, DAGNode):
            parent = type(parent).get_long_name(parent)
        with _undo_chunk(f"rig.{cls.__name__}.load_hierarchy"):
            return cls._load_built(data, parent, world_space=world_space, uuids=uuids, **kwargs)

    def _load_hierarchy_refused(self, *args, **kwargs):
        name = type(self).__name__
        raise TypeError(
            f"{name}({_name_of(self)!r}).load_hierarchy(...) would update the hierarchy at {_name_of(self)!r} "
            f"in place, which rig does not do (its transforms hold the rig's constraints, skins "
            f"and children); {name}.load_hierarchy(path, parent={_name_of(self)!r}) makes a new one "
            f"under it"
        )

    load_hierarchy = _files.creates_or_updates(_load_hierarchy_new, _load_hierarchy_refused, _HIERARCHY_LOAD_DOC)
    del _load_hierarchy_new, _load_hierarchy_refused

    @staticmethod
    def _create_user_attrs(node: DAGNode, specs: dict) -> None:
        """Adds the user defined attrs of a serialized node and sets their values.

        Every attr is added first, in file order: a compound only exists once
        its children, which follow it, are added. Values are set second, on
        the attrs that exist. A failure is logged and never stops the rest.
        """
        # pass 1: add
        failed = set()
        for name, attr in specs.items():
            attr = {k: v for k, v in attr.items() if k not in ("value", "channel_box")}

            # older files stored array data types as "attributeType"
            if attr.get("attributeType") in LEGACY_DATA_TYPES:
                attr["dataType"] = attr.pop("attributeType")

            try:
                node.add_attr(name, **attr)
            except Exception as e:
                failed.add(name)
                LOGGER.warning(f"Skipped user defined attr {_name_of(node)}.{name}: {e}")

        # pass 2: channel box and values
        for name, attr in specs.items():
            if name in failed:
                continue

            att = node.find_attr(name, quiet=True)
            if att is None:
                LOGGER.warning(
                    f"User defined attr {_name_of(node)}.{name} was not created, "
                    "a compound parent is missing children"
                )
                continue

            try:
                if attr.get("channel_box"):
                    type(att).set_channel_box(att, True)

                value = attr.get("value")
                if value is None:
                    continue

                # a multi's value is [logical index, value] pairs
                if attr.get("multi"):
                    for index, element in value:
                        att[index].set(element)
                else:
                    att.set(value)

            except Exception as e:
                LOGGER.warning(f"Skipped user defined attr {_name_of(node)}.{name} value: {e}")


# the public names (user decision 2026-10-17: import leaks hidden)
__all__ = [
    "LEGACY_DATA_TYPES",
    "SERIALIZED_NODE_TYPES",
    "Transform",
    "TYPED_DATA_NAMES",
]
