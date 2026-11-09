"""
Base geometry node class

Usage::

    from rig import Node

    mesh = Node("pSphereShape1")
    mesh.get_injection_node()                    # the node holding editable tags
    mesh.add_tag("cap")
    mesh.set_tag_contents("cap", [0, 1, 2])
    mesh.get_tag_contents("cap")                 # ['vtx[0:2]']
    mesh.get_tag_component_indices("cap")        # array([0, 1, 2])
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Iterator, TYPE_CHECKING

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from numpy.typing import ArrayLike
from rig._internal.api_edits import api_reparent as _api_reparent
from rig._internal.journal import cmds_edit as _cmds_edit
from rig.nodetypes import _files
from rig.nodetypes._base import _cast, Attribute, _name_of
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.dg_node import _ensure_valid
from rig.nodetypes.errors import NodeTypeError

if TYPE_CHECKING:
    from cgmath.geometry import GeomSubsetData


# a bare component tag name, as opposed to an expression ('top + side', '!top');
# '.' is a name character, as ':' is: Maya reads 'ab.cd' as one name, also inside
# an expression (runs/r5/REBIND/probe_dot.txt)
_TAG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:.]*$")

# The names Maya stores as a component tag name (Maya 2025.3, runs/r5/SKIN/probe_tags*.py).
# Maya trims the whitespace at the ends of the name and makes a space inside, with
# any whitespace after it, one '_'. What is left must match this rule.
_TAG_NAME_STORED_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[.:][A-Za-z0-9_]+)*")
_TAG_NAME_RULE = (
    "Maya takes a component tag name of at least two characters, letters, digits "
    "and '_', starting with a letter or '_', with single '.' or ':' between "
    "characters (whitespace at its ends is trimmed and a space inside becomes '_')"
)
_ASCII_WHITESPACE = " \t\n\v\f\r"


def _check_tag_name(key: Any) -> Any:
    """The name Maya stores for `key` as a component tag name ('ab' for
    ' ab', 'a_b' for 'a b'). Raises a ValueError naming Maya's rule for a
    name Maya refuses (it refuses ``setAttr`` of one with a bare "Maya
    command error"), and a TypeError naming it for a name that is not a str
    (an int would be read as a tag index). A name with a non-ASCII character
    is left to Maya, and returned as given: Maya first maps it to the system
    code page by best fit (on code page 1252, 'Ā' becomes 'A' and is taken,
    'é' stays non-ASCII and is refused)."""
    if not isinstance(key, str):
        raise TypeError(
            f"{key!r} is not a component tag name: a name is a str, not a "
            f"{type(key).__name__}; {_TAG_NAME_RULE}"
        )
    if not key.isascii():
        return key
    stored = re.sub(r" [ \t\n\v\f\r]*", "_", key.strip(_ASCII_WHITESPACE))
    if len(stored) < 2 or not _TAG_NAME_STORED_RE.fullmatch(stored):
        raise ValueError(f"{key!r} is not a component tag name: {_TAG_NAME_RULE}")
    return stored

# The component an EMPTY tag resolves to, by geometry data type then tag
# category: Maya hands back a null MObject for an empty tag, so the reader
# builds an empty component of the native (fn set, type) instead.
_EMPTY_COMPONENT_TYPES = {
    OpenMaya.MFn.kMeshData: {
        "v": (OpenMaya.MFnSingleIndexedComponent, OpenMaya.MFn.kMeshVertComponent),
        "e": (OpenMaya.MFnSingleIndexedComponent, OpenMaya.MFn.kMeshEdgeComponent),
        "f": (OpenMaya.MFnSingleIndexedComponent, OpenMaya.MFn.kMeshPolygonComponent),
    },
    OpenMaya.MFn.kNurbsCurveData: {
        "v": (OpenMaya.MFnSingleIndexedComponent, OpenMaya.MFn.kCurveCVComponent),
    },
    OpenMaya.MFn.kNurbsSurfaceData: {
        "v": (OpenMaya.MFnDoubleIndexedComponent, OpenMaya.MFn.kSurfaceCVComponent),
    },
    OpenMaya.MFn.kLatticeData: {
        "v": (OpenMaya.MFnTripleIndexedComponent, OpenMaya.MFn.kLatticeComponent),
    },
}


def iter_component_ranges(prefix: str, ids: list[int]) -> Iterator[str]:
    """A generator that converts a list of indices to a list of range strings.

    e.g.
    ```python
    iter_component_ranges("vtx", [1, 2, 3, 5, 7, 8])
    # ["vtx[1:3]", "vtx[5]", "vtx[7:8]"]
    ```
    """
    gen = itertools.groupby(enumerate(ids), lambda pair: pair[1] - pair[0])
    for _, bit in gen:
        b     = list(bit)
        start = b[0][1]
        end   = b[-1][1]
        if start == end:
            yield f"{prefix}[{start}]"
        else:
            yield f"{prefix}[{start}:{end}]"


def iter_component_tokens(prefix: str, ids: Any) -> Iterator[str]:
    """Range strings for component ids of any dimension, sorted and unique.

    (N,) ids are :func:`iter_component_ranges`; (N, 2) and (N, 3)
    coordinates are grouped on their leading axes and ranged on the last,
    the forms surfaces and lattices store.

    e.g.
    ```python
    list(iter_component_tokens("cv", [[1, 0], [1, 1], [2, 5]]))
    # ["cv[1][0:1]", "cv[2][5]"]
    ```
    """
    ids = np.unique(np.asarray(ids, dtype=int), axis=0)
    if ids.size == 0:
        return
    if ids.ndim == 1:
        yield from iter_component_ranges(prefix, ids.tolist())
        return
    leads = ids[:, :-1]
    for lead in np.unique(leads, axis=0):
        head = prefix + "".join(f"[{k}]" for k in lead)
        tail = ids[np.all(leads == lead, axis=1), -1]
        yield from iter_component_ranges(head, tail.tolist())


def _shape_create_name(name: str | None) -> str | None:
    """The name a shape's typed create (`Mesh.create`, `NurbsCurve.create`)
    gives the transform: `name` without a trailing ``Shape<digits>``
    (``"mShape2"`` is ``"m2"``, whose shape Maya names ``mShape2``); None when
    there is no name."""
    if name and re.search("Shape[0-9]*$", name):
        name = "".join(name.rpartition("Shape")[::2])
    return name or None


def _data_ids(shape, world_space: bool) -> dict:
    """The ``uuid`` and ``parent_uuid`` a shape's `serialize` writes (a MeshData's, a
    NurbsCurveData's, a NurbsSurfaceData's: every cgmath shape is a ``ShapeData``), as keywords:
    the shape node's own UUID, and the UUID of the transform the object-space points are relative
    to, read through the shape's path, as the points are (the transform above the shape; for an
    instanced shape, the path's own instance), or None for world-space points, which sit at
    identity. cgmath places a shape by the transform its ``parent_uuid`` names
    (`_data_placement`); the data itself holds no matrix."""
    _files.require_pinned_cgmath(f"{type(shape).__name__}.serialize")
    path = OpenMaya.MDagPath(type(shape).get_mdagpath(shape))
    ids  = {"uuid": OpenMaya.MFnDependencyNode(path.node()).uuid().asString(), "parent_uuid": None}
    if not world_space:
        path.pop()
        ids["parent_uuid"] = OpenMaya.MFnDependencyNode(path.node()).uuid().asString()
    return ids


def _data_placement(method: str, data: Any, hierarchy: Any = None) -> np.ndarray:
    """Where `data` (a shape's, with a ``parent_uuid``; points have none) puts its points in the
    world, ``(4, 4)`` with row vectors as Maya's: cgmath's rule, the world matrix of the transform
    its ``parent_uuid`` names, the identity when it names none. With `hierarchy` (a HierarchyData,
    or a ClipData at the frame it is on) the transform is looked for there, and cgmath's own
    ``world_matrix`` answers (the identity for one the hierarchy lacks); without, in the scene,
    the node of that UUID, under which data serialized and created in one scene sits as its
    source did, and the identity when the scene has none. A scene node that is instanced is a
    ValueError naming `method`: the points would have no one place in the world."""
    _files.require_pinned_cgmath(method)
    parent = getattr(data, "parent_uuid", None)
    if parent is None:
        return np.eye(4)
    if hierarchy is not None:
        return np.asarray(data.world_matrix(hierarchy), dtype=np.float64).reshape(4, 4)
    sel = OpenMaya.MSelectionList()
    try:
        sel.add(OpenMaya.MUuid(parent))
    except RuntimeError:                                     # no node of that UUID in the scene
        return np.eye(4)
    node = sel.getDependNode(0)
    if not node.hasFn(OpenMaya.MFn.kDagNode):
        return np.eye(4)
    paths = OpenMaya.MDagPath.getAllPathsTo(node)
    if len(paths) > 1:
        raise ValueError(
            f"{method}: the data sits under {paths[0].partialPathName()!r}, which is instanced "
            f"({len(paths)} paths), so its points have no one place in the world; give hierarchy=, "
            f"or serialize the shape in world space"
        )
    return np.array(paths[0].inclusiveMatrix()).reshape(4, 4)


# --- a new shape under an existing transform: ``transform=`` of `Mesh.create`,
# `NurbsCurve.create`, `NurbsSurface.create` and their class-form ``load``. The shape is built
# on the usual ride (a recorded ``createNode`` transform, the shape made under it through the
# API), then moved under the target (a journaled ``MDagModifier``, as ``parent -shape
# -relative``), the ride deleted and the shape named, inside the typed create, so in its one
# undo step and its container frame

# the least a target's world 3x3 may scale along any direction with ``world_space=True``: Maya
# keeps 1e-12 for a scale of 0 (under it a shape's object points would come out 1e12 times out)
_SINGULAR_SCALE = 1e-9


def _shape_target(method: str, transform: Any, world_space: bool) -> OpenMaya.MDagPath:
    """The transform ``transform=`` names for a new shape (`method`: ``"Mesh.create()"``), read
    before anything is made: its DAG path. By the reference rule, ``Transform(x)``: a missing
    name is a NodeNotFoundError, a name several nodes have an AmbiguousNodeError, a node that is
    no transform (a shape, a DG node) a NodeTypeError (a TypeError) naming ``transform=``; a
    joint, or another node type derived from transform, is one. A locked or a referenced target
    is taken, as Maya's ``parent`` takes it. With `world_space`, an instanced target (itself or
    a parent) is a ValueError: the shape goes under every instance, and one set of points
    cannot sit where the data puts them under each. So is a singular one (`_SINGULAR_SCALE`, a
    scale of 0 on it or on a parent, read off its world 3x3's singular values: a rotated child
    of a flat parent has no short axis, and is as flat): no object points land there. The
    message gives its axes' lengths, X, Y, Z, and the flat direction's when no axis is short."""
    from rig.nodetypes.transform import Transform

    try:
        node = Transform(transform)
    except NodeTypeError as error:
        raise NodeTypeError(
            error.name, "transform", error.node_type,
            message=f"{method}: transform= is the transform (or a node derived from one) the new "
                    f"shape goes under: {error}",
        ) from None
    path = OpenMaya.MDagPath(type(node).get_mdagpath(node))
    if world_space and OpenMaya.MFnDagNode(path).isInstanced(True):
        count = len(OpenMaya.MDagPath.getAllPathsTo(path.node()))
        raise ValueError(
            f"{method}: {path.partialPathName()!r} is instanced ({count} paths) and the new shape "
            f"goes under every instance, so world_space=True has no one place to put the points; "
            f"world_space=False (the default) takes the data's points as the shape's own"
        )
    if world_space:
        world = np.array(path.inclusiveMatrix()).reshape(4, 4)[:3, :3]
        least = np.linalg.svd(world, compute_uv=False)[-1]
        if least < _SINGULAR_SCALE:
            axes = np.linalg.norm(world, axis=1)
            flat = "" if axes.min() < _SINGULAR_SCALE else f", but it scales one direction by {least:.3g}"
            raise ValueError(
                f"{method}: {path.partialPathName()!r} is singular (its world scale is "
                f"({', '.join(f'{axis:.3g}' for axis in axes)}){flat}), so world_space=True has no "
                f"object points that land where the data puts them; world_space=False (the default) "
                f"takes the data's points as the shape's own"
            )
    return path


def _target_space(method: str, matrix: Any, target: OpenMaya.MDagPath | None) -> np.ndarray:
    """``(4, 4)``, row vectors as Maya's: from where data puts its points in the world, `matrix`
    (`_data_placement`'s; None: the identity, points), into the space of `target`, a transform
    or the mesh an update lands on, ``matrix @ inverse(its world matrix)``; with no `target`, a
    new transform that stays at the origin, the world itself (``world_space=True``, either way).
    For a create `_shape_target` refused a target whose world matrix has no inverse to speak of
    (an update refuses no mesh); a `matrix` that is not finite is a ValueError naming `method`,
    before anything is made."""
    matrix = np.eye(4) if matrix is None else np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{method}: the data's matrix must be finite (got nan or inf)")
    if target is None:
        return matrix
    return matrix @ np.linalg.inv(np.array(target.inclusiveMatrix()).reshape(4, 4))


def _shape_name(target: OpenMaya.MDagPath, name: str | None) -> str:
    """The name of a new shape under `target`: `name` as given (no transform is made, so it
    names the shape), else the target's own absolute name, its namespace with it, then ``Shape``
    (``:char:ctlShape``, whatever the current namespace; Maya numbers it when a sibling has it:
    ``ctlShape1``). Under a referenced target that is a local node in the reference's namespace,
    which comes back with the scene."""
    if name:
        return name
    return f"{OpenMaya.MFnDagNode(target).absoluteName()}Shape"


def _put_under(shape: OpenMaya.MObject, ride: OpenMaya.MObject, target: OpenMaya.MDagPath, name: str) -> str:
    """The new `shape` moved from its `ride` transform under `target` (as ``parent -shape
    -relative``: its points stay its own; under every instance of an instanced target), the
    ride deleted and the shape renamed `name`, in the create's undo step after the ride's own
    records. The move is a journaled ``MDagModifier`` (`_api_reparent`): the selection and the
    hilite list are never touched, at no cost per selected item. Under an instanced target, or
    without rig's undo command, it is the recorded ``parent`` (only the command puts the
    shape's other instances in its shading group), which selects the shape and clears the
    hilite list: the selection is put back (`_files.kept_selection`), the hilite list is not.
    The delete and the rename are recorded ``cmds`` (Maya's own name rules: ``#`` numbers, a
    sibling's name is numbered). Returns the shape's full path under `target`, the instance
    the caller named."""
    if OpenMaya.MFnDagNode(target).isInstanced(True) or not _api_reparent(shape, target.node()):
        with _files.kept_selection():
            _cmds_edit.parent(
                OpenMaya.MFnDagNode(shape).fullPathName(), target.fullPathName(), shape=True, relative=True
            )
    _cmds_edit.delete(OpenMaya.MFnDagNode(ride).fullPathName())
    _cmds_edit.rename(OpenMaya.MFnDagNode(shape).fullPathName(), name)
    path = OpenMaya.MDagPath(target)
    path.push(shape)
    return path.fullPathName()


def _discard(*handles: OpenMaya.MObjectHandle | None) -> None:
    """After an error in a shape's create, what it made that is still in the scene deleted, in
    the order of `handles`: the ride transform first (its shape goes with it while under it),
    then a shape already put under its target (None: not made yet)."""
    for handle in handles:
        if handle is not None and handle.isValid():
            _cmds_edit.delete(OpenMaya.MFnDagNode(handle.object()).fullPathName())


class Geometry(DAGNode):
    """
    Base geometry node class

    All sub classes must implement the following:
        - NATIVE_NODE_TYPE
        - POINT_COMP_TYPE
        - get_weight_point_count()
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "geometryShape"

    # point component type string
    POINT_COMP_TYPE = None

    def __init__(self, node: str | OpenMaya.MObject | OpenMaya.MDagPath) -> None:
        """Initialize an instance from a node name or a MObject.

        Supports initializing from transform nodes, by name or as node objects
        (a node object of another class is taken by its name).
        """
        if isinstance(node, DAGNode) and not isinstance(node, type(self)):
            node = _name_of(node)
        if isinstance(node, str) and cmds.nodeType(node) == "transform":
            shapes = cmds.listRelatives(
                node,
                shapes         = True,
                type           = type(self).NATIVE_NODE_TYPE,
                noIntermediate = True,
                fullPath       = True,
            )
            if len(shapes) == 0:
                raise ValueError(
                    f"Transform {_name_of(node)} has no {type(self).NATIVE_NODE_TYPE} shape."
                )
            node = shapes[0]

        super().__init__(node)
        d                                 = self.__dict__
        d["_Geometry__local_output_attr"] = None
        d["_Geometry__world_output_attr"] = None

    @classmethod
    def _coerce(cls, node: Any) -> Any:
        """The reference's hook (see ``DGNode._coerce``): a transform stands
        for its first non-intermediate shape of this class's type
        (``Mesh("body")`` gives ``Mesh("bodyShape")``), the rule the
        constructor has; a transform with none is refused."""
        if cmds.nodeType(type(node).get_long_name(node)) == "transform":
            shapes = cmds.listRelatives(
                type(node).get_long_name(node),
                shapes         = True,
                type           = cls.NATIVE_NODE_TYPE,
                noIntermediate = True,
                fullPath       = True,
            )
            if not shapes:
                return None
            node = _cast(shapes[0])
            if isinstance(node, cls):
                return node
        return super()._coerce(node)

    @classmethod
    def _mismatch_hint(cls, node: Any) -> str:
        if cmds.nodeType(type(node).get_long_name(node)) == "transform":
            return f", which has no {cls.NATIVE_NODE_TYPE} shape"
        return ""

    @classmethod
    def _hint_type(cls) -> str | None:
        """See ``DGNode._hint_type``: None, every node, since the reference
        takes a transform too (``Mesh("bdoy")`` may mean the transform
        ``body``)."""
        return None

    # --- attr helpers

    def get_output_attr(self, *, world_space: bool = False) -> Attribute:
        """Returns the shape out attribute.

        Args:
            world_space: By keyword only. False (the default): the local shape
                out attribute (a mesh's ``outMesh``); True: the world one (a
                mesh's ``worldMesh[0]``).
        """
        _ensure_valid(self)
        if world_space:
            if self.__world_output_attr is None:
                attr_name                = cmds.deformableShape(_name_of(self), worldShapeOutAttr=True)[0]
                self.__world_output_attr = self.find_attr(attr_name)[0]
            return self.__world_output_attr
        if self.__local_output_attr is None:
            attr_name                = cmds.deformableShape(_name_of(self), localShapeOutAttr=True)[0]
            self.__local_output_attr = self.find_attr(attr_name)
        return self.__local_output_attr

    def set_attr_paintable(self, attr: Attribute | str, **kwargs) -> None:
        """Set the attribute paintable states. Thin wrapper of cmds.makePaintable()

        Args:
            attr: An attribute to operate on.
            kwargs: kwargs supported by cmds.makePaintable().
        """
        attr               = self.find_attr(attr)
        args               = (type(self).get_node_type(self), type(attr)._raw_get_name(attr))
        kwargs["attrType"] = type(attr)._raw_get_data_type(attr)
        _cmds_edit.makePaintable(*args, **kwargs)

    @staticmethod
    def clear_all_paintable_attrs() -> None:
        """Clears all paintable attrs."""
        _cmds_edit.makePaintable(clearAll=True)

    # --- components

    def get_component_mobject(
        self, component_type: str | None = None, indices: None | np.ndarray = None
    ) -> OpenMaya.MObject:
        """Returns a mobject hosting all components in this geom.

        Args:
            component_type: The component type. e.g. "v", "e", "f", etc
        """
        comp_type = component_type or type(self).POINT_COMP_TYPE
        comp_type = "vtx" if comp_type == "v" else comp_type
        sel       = OpenMaya.MSelectionList()
        if indices is None:
            sel.add(f"{_name_of(self)}.{comp_type}[*]")
        else:
            token = f"{_name_of(self)}.{comp_type}"  # node_name.component_type
            for item in iter_component_ranges(token, indices):
                sel.add(_name_of(item))

        _, mobject = sel.getComponent(0)
        return mobject

    # --- component tags location

    def get_injection_node(self) -> Geometry:
        """Returns the node holding this geometry's editable component tags.

        Maya writes tags to the current tag injection node
        (`cmds.deformableShape(shape, tagInjectionNode=True)`): the shape
        itself until a deformer exists, its intermediate "Orig" shape from then
        on. Every tag editor here addresses that node's componentTags multi so
        that editors and readers agree. Not cached: adding a deformer moves it.

        Raises:
            RuntimeError: for a referenced shape with no local deformer: Maya
                answers no node (``['']``), as it stores no tag on a referenced
                node (runs/r5/TAGLEFT/fix/probe_injnode.txt).
        """
        _ensure_valid(self)
        nodes = cmds.deformableShape(type(self).get_long_name(self), tagInjectionNode=True)
        if nodes and not nodes[0]:
            raise RuntimeError(
                f"{_name_of(self)} has no node Maya can store component tags on: it is "
                f"referenced, and so is its geometry chain; a deformer added to it "
                f"makes a local Orig that holds them"
            )
        if not nodes or cmds.ls(nodes[0], long=True)[0] == type(self).get_long_name(self):
            return self
        return _cast(nodes[0])

    def _tag_owner(self, key: str) -> tuple[str, bool] | None:
        """Returns (node, procedural) for the entry of a tag that resolves, or None."""
        for entry in self.get_tag_history() or []:
            if entry["key"] == key and entry["final"]:
                return entry["node"], entry["procedural"]
        return None

    def _editable_tag(self, tag: str | int) -> Attribute | None:
        """Returns the injection node's componentTags element holding a tag.

        A tag that resolves on this geometry but has no element at the
        injection node cannot be edited here: it is procedural (polyCube's six
        face tags are output of polyCube1) or injected on another node in the
        history, and only its owner can change it. That raises; a name that
        does not exist at all warns and returns None.

        Args:
            tag: Component tag key or index.
        """
        injection = type(self).get_injection_node(self)
        if isinstance(tag, int):
            tags = injection.find_attr("componentTags").get_logical_indices()
            i    = tag if tag in tags else -1
        else:
            i = self.get_tag_index(tag)
            if i == -1 and self.has_tag(tag):
                owner = self._tag_owner(tag)
                kind  = "procedural" if owner and owner[1] else "injected"
                node  = owner[0] if owner else "another node"
                cmds.error(
                    f"Component tag '{tag}' on {_name_of(self)} is {kind} (owned by {_name_of(node)}) "
                    f"and cannot be edited at {injection}."
                )

        if i == -1:
            cmds.warning(f"Component tag not found: {tag}")
            return None
        return injection.find_attr("componentTags")[i]

    # --- component tags editing

    def add_tag(self, key: str, force: bool = False) -> int:
        """Adds a component tag with the given key to the object.

        The tag is created on the injection node, where Maya's own
        componentTag command puts it.

        Args:
            key: Component tag key.
            force: If True, remove the existing component tag.

        Returns:
            The index of the new tag added.

        Raises:
            ValueError: before anything is made, for a name Maya refuses: Maya
                takes at least two characters, letters, digits and '_',
                starting with a letter or '_', with single '.' or ':' between
                characters, once the whitespace at the ends is trimmed and a
                space inside is made '_' ('b', '1a', 'a-b', 'a..b' and '' are
                refused; 'ab', '_1', 'a.b' and 'a b', stored 'a_b', are
                taken). A name with a non-ASCII character is left to Maya.
            TypeError: before anything else, for a name that is not a str
                (``add_tag(5)`` is not the tag at index 5).
            RuntimeError: when a tag of the name Maya stores exists ('ab'
                for ' ab') and `force` is False.
        """
        stored = _check_tag_name(key)
        if self.has_tag(stored):
            if force:
                self.delete_tag(stored)
            else:
                owner = self._tag_owner(stored)
                if owner and owner[1]:
                    cmds.error(
                        f"Component tag '{stored}' already exists (procedural, owned by {_name_of(owner[0])})."
                    )
                cmds.error(f"Component tag '{stored}' already exists.")

        # add to first empty component tag name index.
        injection = type(self).get_injection_node(self)
        i         = injection.find_attr("componentTags").get_next_available_index()
        name_attr = injection.find_attr("componentTags")[i].componentTagName
        name_attr.set(key, type="string")
        return i

    def delete_tag(self, tag: str | int) -> None:
        """Deletes a component tag from the object. Skips the check of
        ``Tag.delete``, which refuses a tag a deformer expression references.

        Args:
            tag: Component tag key or index.
        """
        element = self._editable_tag(tag)
        if element is not None:
            _cmds_edit.removeMultiInstance(_name_of(element), b=True)

    def rename_tag(self, tag: str | int, new_key: str) -> None:
        """Renames a component tag with the given key the object.

        Args:
            tag: Component tag key or index.
            new_key: New component tag key name.

        Raises:
            ValueError: before anything is renamed, for a name Maya refuses
                (the rule ``add_tag`` names).
            TypeError: before anything is renamed, for a name that is not a
                str.
        """
        _check_tag_name(new_key)
        element = self._editable_tag(tag)
        if element is not None:
            element.componentTagName.set(new_key, type="string")

    def set_tag_contents(
        self,
        tag:      str       | int,
        contents: ArrayLike | list[str] | list[int] | GeomSubsetData,
        category: str                                                = "v",
    ) -> None:
        """Sets the component tag with the given key to the given contents.

        Args:
            tag: Component tag key or index.
            contents: One of the following:
              - An array of indices: (N,) native ids, or (N, 2) / (N, 3)
                coordinates for surface CVs and lattice points.
              - A list of component strings or integer indices.
              - A GeomSubsetData object.
            category: The component tag category. "v", "f", "e", etc.
                Only used when contents is a list of index.
        """
        element = self._editable_tag(tag)
        if element is None:
            return

        contents  = self._tag_tokens(contents, category)
        comp_attr = element.componentTagContents
        comp_attr.set(len(contents), *contents, type="componentList")

    def _tag_tokens(
        self,
        contents: ArrayLike | list[str] | list[int] | GeomSubsetData,
        category: str,
    ) -> list[str]:
        """Converts tag contents into the component strings the contents plug takes."""
        # a GeomSubsetData, duck-typed so cgmath is never imported here
        if hasattr(contents, "indices") and hasattr(contents, "component_type"):
            return self._component_strings(contents.component_type, contents.indices)

        if isinstance(contents, (list, tuple)):
            if len(contents) == 0:
                return []
            if isinstance(contents[0], str):
                # the node prefix is rejected by setAttr
                return [x.rsplit(".", 1)[-1] for x in contents]
            contents = np.asarray(contents)

        if isinstance(contents, np.ndarray):
            if contents.size == 0:
                return []
            if not np.issubdtype(contents.dtype, np.integer):
                cmds.error("Component tag indices must be integers.")
            return self._component_strings(category, contents)

        cmds.error(
            "Component tag contents must be either GeomSubsetData, a list of component strings, or an integer np.ndarray"
        )

    def _component_strings(self, category: str, indices: ArrayLike) -> list[str]:
        """Renders indices as this geometry's native component strings.

        (N,) indices become ``vtx[a:b]`` / ``cv[a:b]`` / ``f[a:b]`` ranges;
        (N, 2) and (N, 3) coordinates become the ``cv[u][v0:v1]`` and
        ``pt[s][t][u0:u1]`` forms surfaces and lattices store. Category "v"
        maps to POINT_COMP_TYPE, any other category is the token itself.
        """
        if category == "v":
            prefix = type(self).POINT_COMP_TYPE
            if prefix is None:
                cmds.error(
                    f"{type(self).get_node_type(self)} has no point component type; pass component strings instead."
                )
        else:
            prefix = category

        return list(iter_component_tokens(prefix, indices))

    def select_tag_contents(self, key: str) -> None:
        """Selects component members of the component tag with the given key.

        Args:
            key: Component tag key.
        """
        contents = self.get_tag_contents(key, full_name=True)
        _cmds_edit.select(contents, replace=True)

    # --- component tags query

    def get_tag_names(self) -> list[str]:
        """Returns the component tag keys this geometry resolves.

        Editable tags at the injection node come first, in index order; names
        only visible on the output plug (procedural tags such as polyCube's six
        face tags, or tags injected elsewhere in the history) follow in Maya's
        sorted order.
        """
        injection = type(self).get_injection_node(self)
        tags      = []
        for i in injection.find_attr("componentTags").get_logical_indices():
            name = injection.find_attr("componentTags")[i].componentTagName.get()
            if name:
                tags.append(name)

        visible = cmds.geometryAttrInfo(
            f"{_name_of(type(self).get_output_attr(self))}",
            componentTagNames=True,
        )
        tags.extend(name for name in visible or [] if name not in tags)
        return tags

    def has_tag(self, tag: str | int) -> bool:
        """Returns True if a component tag exists.

        A key is looked up on the evaluated geometry (so procedural and
        upstream tags count); an index is looked up in the injection node's
        componentTags multi.

        Args:
            tag: Component tag key or index.
        """
        if isinstance(tag, int):
            tags = type(self).get_injection_node(self).find_attr("componentTags")
            return tag in tags.get_logical_indices()
        else:
            data_fn = type(self).get_output_attr(self).get_data_fn_set()
            return data_fn.hasComponentTag(tag)

    def get_tag_index(self, key: str) -> int:
        """Returns the logical index of a component tag in the injection node's
        componentTags multi, or -1 if not found (or not editable there)."""
        injection = type(self).get_injection_node(self)
        for i in injection.find_attr("componentTags").get_logical_indices():
            name_attr = injection.find_attr("componentTags")[i].componentTagName
            if name_attr.get() == key:
                return i
        return -1

    def get_tag_name(self, i: int) -> str | None:
        """Returns the name of a component tag, or None if not found."""
        injection = type(self).get_injection_node(self)
        if i not in injection.find_attr("componentTags").get_logical_indices():
            return None
        name_attr = injection.find_attr("componentTags")[i].componentTagName
        return name_attr.get() or None

    def get_tag_subset_state(self, expr: str) -> int:
        """Returns the subset state for the component tag expression.

        [0: no geometry points, 1: some geometry points, 2: all geometry points]

        Args:
            expr: Component tag expression. (E.g. '*', 'top', 'top+bottom', etc.)
        """
        return cmds.geometryAttrInfo(
            f"{_name_of(type(self).get_output_attr(self))}",
            componentTagExpression = expr,
            subsetState            = True,
        )

    def get_tag_history(self) -> list[dict[str, Any]]:
        """Returns verbose descriptions of all historical component tags.

        Example:
        ```
        [{'key': 'top',
          'node': 'pSphereShape1Orig',
          'affectCount': 40,
          'fullCount': 400,
          'modified': False,
          'procedural': False,
          'editable': True,
          'category': 1},
          {...}]
        ```
        """
        return cmds.geometryAttrInfo(
            f"{_name_of(type(self).get_output_attr(self))}",
            componentTagHistory=True,
        )

    def get_tag_category(self, tag: str | int) -> str:
        """Returns the category of a given component tag.

        [v: verts, e: edges, f: faces]

        Args:
            tag: Component tag key or index.
        """
        tag_name = self.get_tag_name(tag) if isinstance(tag, int) else tag
        return cmds.geometryAttrInfo(
            f"{_name_of(type(self).get_output_attr(self))}",
            componentTagExpression = tag_name,
            componentTagCategory   = True,
        )

    def set_tag_category(self, tag: str | int, category: str) -> None:
        """Sets the category of a given component tag and convert its content.

        [v: verts, e: edges, f: faces]

        Args:
            tag: Component tag key or index.
            category: The component tag category. "v", "f", "e", etc.
        """
        import maya.internal.common.utils.componenttag as ctag_utils

        cat = self.get_tag_category(tag)
        if cat != category:
            if category == "f":
                c = OpenMaya.MFnGeometryData.kFaces
            elif category == "v":
                c = OpenMaya.MFnGeometryData.kVerts
            else:
                c = OpenMaya.MFnGeometryData.kEdges
            tag_name = self.get_tag_name(tag) if isinstance(tag, int) else tag
            ctag_utils.convertTagCategory(_name_of(self), tag_name, c)

    def get_tag_contents(
        self, tag: str | int, full_name: bool = False
    ) -> list[str]:
        """Returns a component list store in a given tag.

        Args:
            tag: Component tag key or index.
            full_name: If True, returns list with shape name e.g.
                (['pSphereShape.f[1:9]']).
                Otherwise return a component list (e.g. ['f[1:9]']).
        """
        tag_name = self.get_tag_name(tag) if isinstance(tag, int) else tag
        components = cmds.geometryAttrInfo(
            f"{_name_of(type(self).get_output_attr(self))}",
            componentTagExpression = tag_name,
            components             = True,
        )

        if full_name:
            return [f"{_name_of(self)}.{_name_of(component)}" for component in components]
        return components

    def get_tag_component_indices(self, tag: str | int) -> np.ndarray:
        """Returns the native ids of the components in a given tag.

        Face tags give face ids and edge tags edge ids, never the vertex cast
        `geometryAttrInfo -pointIndices` applies. Shape (N,) on meshes and
        curves, (N, 2) ``(u, v)`` on surfaces, (N, 3) ``(s, t, u)`` on
        lattices; an empty tag gives an empty array of the same shape.

        Args:
            tag: Component tag key or index.
        """
        tag_name = self.get_tag_name(tag) if isinstance(tag, int) else tag
        if tag_name is None:
            cmds.error(f"Component tag {tag} does not exist.")

        comp_fn  = self._resolve_tag_expression(tag_name)
        elements = np.array(list(comp_fn.getElements()), dtype=int)
        if isinstance(comp_fn, OpenMaya.MFnTripleIndexedComponent):
            return elements.reshape(-1, 3)
        if isinstance(comp_fn, OpenMaya.MFnDoubleIndexedComponent):
            return elements.reshape(-1, 2)
        return elements

    def is_tag_empty(self, tag: str | int) -> bool:
        """Returns True if a tag is empty.

        Args:
            key: Component tag key or index.
        """
        if not self.has_tag(tag):
            cmds.error(f"Component tag {tag} does not exist.")
        return not bool(self.get_tag_contents(tag))

    def _resolve_tag_expression(
        self, expr: str
    ) -> (
        OpenMaya.MFnSingleIndexedComponent
        | OpenMaya.MFnDoubleIndexedComponent
        | OpenMaya.MFnTripleIndexedComponent
    ):
        """Returns the component function set a component tag expression resolves to.

        The component comes back in the geometry's native form: single-indexed
        on meshes and curves, double-indexed (u, v) on surfaces, triple-indexed
        (s, t, u) on lattices. An empty tag yields an empty component of that
        form instead of the null MObject Maya returns.

        Args:
            expr: Component tag key, or an expression ('top + side', '!top').
                A bare key that does not exist raises, a dotted one too ('ab.cd'
                is one name); unknown names inside an expression are left to
                Maya, which ignores them.
        """
        data_fn = type(self).get_output_attr(self).get_data_fn_set()
        if _TAG_NAME_RE.match(expr) and not data_fn.hasComponentTag(expr):
            cmds.error(f"Component tag {expr} does not exist.")

        component = data_fn.resolveComponentTagExpression(expr)
        if component.isNull():
            types             = _EMPTY_COMPONENT_TYPES[data_fn.object().apiType()]
            fn_cls, comp_type = types.get(self.get_tag_category(expr), types["v"])
            comp_fn           = fn_cls()
            comp_fn.create(comp_type)
            return comp_fn

        if component.hasFn(OpenMaya.MFn.kTripleIndexedComponent):
            return OpenMaya.MFnTripleIndexedComponent(component)
        if component.hasFn(OpenMaya.MFn.kDoubleIndexedComponent):
            return OpenMaya.MFnDoubleIndexedComponent(component)
        return OpenMaya.MFnSingleIndexedComponent(component)

    # --- component tags serialization

    def get_tag_data(
        self, tag: str | int, category: str | None = None
    ) -> GeomSubsetData:
        """Serialize a given tag.

        Reads the evaluated geometry, so procedural and upstream tags
        serialize like editable ones.

        Args:
            tag: Component tag key or index.
        """
        from cgmath.geometry import GeomSubsetData

        name = self.get_tag_name(tag) if isinstance(tag, int) else tag
        if name is None or not self.has_tag(name):
            cmds.error(f"Component tag {tag} does not exist.")

        return GeomSubsetData(
            name           = name,
            indices        = self.get_tag_component_indices(name),
            component_type = self.get_tag_category(name),
        )

    def serialize_tags(
        self,
        match_name: list[str] | str | None = None,
        category:   str       | None       = None,
    ) -> list[GeomSubsetData]:
        """Serialize component tags into a standard dict.

        Args:
            match_name: One or more of tag names or regex to match.
                If None, serialize all tags.

        Returns:
            A list of GeomSubsetData objects.
        """
        data      = []
        names     = [match_name] if isinstance(match_name, str) else match_name
        tag_regex = re.compile(r"|".join([rf"^{_name_of(x)}$" for x in names])) if names else None
        for name in type(self).get_tag_names(self):
            if not tag_regex or tag_regex.match(name):
                data.append(self.get_tag_data(name, category=category))

        return data


def deformed_refusal(
    method: str, name: str, node: OpenMaya.MObject, kind: int, cls_name: str,
    connected: str, what: str, door: str,
) -> str:
    """Why a node's ``load`` refuses shape ``name`` (MObject ``node``) with history,
    which would compute over the ``what`` it sets. A deformed shape (a skin, a
    blendShape, a cluster ... upstream) names the shape under its deformers, the
    intermediate ``...Orig`` shape of the kind ``kind`` (``cls_name('x').load``:
    the deformed shape follows its load), since deleting the history would delete
    the deformers; construction history names the delete. `door` names the
    creating form. Read only."""
    upstream = OpenMaya.MItDependencyGraph(
        node,
        OpenMaya.MFn.kInvalid,
        OpenMaya.MItDependencyGraph.kUpstream,
        OpenMaya.MItDependencyGraph.kDepthFirst,
        OpenMaya.MItDependencyGraph.kNodeLevel,
    )
    deformers, orig = set(), None
    while not upstream.isDone():
        found = upstream.currentNode()
        if found.hasFn(OpenMaya.MFn.kGeometryFilt):
            deformers.add(OpenMaya.MFnDependencyNode(found).name())
        elif orig is None and found.hasFn(kind) and found != node:
            dag = OpenMaya.MFnDagNode(found)
            if dag.isIntermediateObject:
                orig = dag.partialPathName()
        upstream.next()
    if deformers:
        under = (
            f"{cls_name}({orig!r}).load(path) updates the shape under them (the deformed one "
            f"follows)"
            if orig is not None
            else "load the shape under them"
        )
        return (
            f"{method}: {name!r} is deformed ({', '.join(sorted(deformers))}), which would "
            f"compute over the {what} it sets: {under}. Deleting its history would delete the "
            f"deformers; {door}"
        )
    return (
        f"{method}: {name!r} has history ({connected}: construction history), which would "
        f"compute over the {what} it sets; delete the history, or {door}"
    )


# the public names (user decision 2026-10-17: import leaks hidden)
__all__ = [
    "deformed_refusal",
    "Geometry",
    "iter_component_ranges",
    "iter_component_tokens",
]
