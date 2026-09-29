"""
Materials through the membership grammar.

A surface shader is a node class like ``Transform`` (``rig.nodetypes``; the
same classes are exported here): ``Blinn("red")`` refers to a blinn that
exists and never writes the scene, ``Blinn.define("red", color=(1, 0, 0))``
finds it or makes its network -- the shader (``cmds.shadingNode``, listed in
``defaultShaderList1``), ``redSG`` (``cmds.sets(renderable=True)`` with its
materialInfo, renderPartition and lightLinker wiring) -- and
``Blinn.create(name="red")`` always makes a new one. ``Material`` is the
generic class: ``Material("red")`` takes any surface shader, and
``Material.define(name, type=...)`` / ``Material.create(type=...)`` make a
type without a class of its own (a plug-in's shader).

A shader node on the right of ``<<`` assigns: the left-hand side moves into
the shader's shading engine in ONE ``cmds.sets(forceElement)`` (a shader
feeding no engine yet -- ``lambert1``, a bare ``rn.blinn()`` -- gets
``<shader>SG`` beside it first; a shader feeding several engines goes to
``<shader>SG``, and names none: a ``ValueError`` before any write). A
``ShadingEngine`` node names exactly that engine (``faces <<
ShadingEngine("altSG")``), and ``Default()`` is ``initialShadingGroup``'s.
Shading membership is exclusive: a member leaves whatever engine held it.
``-node`` is the removal token and ``Material()`` / ``Blinn()`` (no name) the
kind token: on ``<<`` it takes the left-hand side out of every engine
(green), on ``>>`` it enumerates the materials holding it. ``Blinn(None)`` is
a ``TypeError`` (a failed lookup must not mean every material).

The left of a material ``<<`` is a node -- its OWN non-intermediate
shadeable shapes (mesh / nurbsSurface / subdiv), never the transform, which
Maya would recurse through the whole subtree -- or mesh faces
(``cube.f[:3]``). Vertices, edges, UVs and attributes are ``TypeError``s:
materials bind faces or whole objects (an attribute plug is refused on
``<<`` / ``>>`` / ``of``: the node is the member; in ``in`` it stands for its
node).

Usage::

    from rig import Node, List, shade
    from rig.nodetypes import ShadingEngine
    from rig.shade import Blinn, Lambert, Material, Phong, Default

    cube = Node("pCube1")
    red  = Blinn.define("red", color=(1, 0, 0))   # Blinn("red"): found, or made now (red, redSG, materialInfo)
    cube << red                              # assigns; returns cube
    cube.f[:3] << Lambert("decal")           # an existing lambert: faces 0-2 leave redSG for decalSG
    red.color << (0, 1, 0)                   # a plain plug: red is the node
    red.engine                               # ShadingEngine("redSG")
    cube >> red                              # array([3, 4, 5])   faces wearing red (all: object-level)
    cube in red ; cube.f[3:] in red          # False ; True   every face of the left
    Material.of(cube)                        # [Blinn("red"), Lambert("decal")]   live nodes

    cube.f[:3] << -Lambert("decal")          # those faces are in NO shading engine (green)
    cube << Material()                       # green everywhere (Blinn() purges the same way)
    cube >> Material() ; cube >> Blinn()     # the operator spelling of Material.of / Blinn.of
    cube << Default()                        # initialShadingGroup again
    cube.f[:2] << ShadingEngine("altSG")     # exactly that engine
    List([cube, sph]) << red                 # one material, one engine, one cmds.sets
    cube << Blinn("nope")                    # NodeNotFoundError before any write: a reference never creates

    red.delete()                             # red, redSG and its materialInfo; members go green
    shade.repair()                           # List of the shapes re-homed to initialShadingGroup
    shade.tidy()                             # all-faces memberships collapsed to object level

    red = red.astype(Phong)                  # Phong("red"): the new node, in one undo chunk
    shade.convert("red", "lambert", dry_run=True)   # the Conversion report, nothing written

Conversion (``mat.astype(Phong)``, ``shade.convert(mat, "phong")``) makes a
new node of the target type and returns it; ``Phong(mat)`` is a reference,
never a conversion (a blinn there is a ``NodeTypeError`` naming ``astype``).
It keeps the name, the engines, the materialInfo, the ``defaultShaderList1``
slot, the container, the dynamic attributes and the locks, moves every wire
the target can hold, and PARKS what it cannot: each non-default value, wire
or animCurve on an attribute the target lacks is cloned onto the same node as
a hidden ``__attr__`` of the same type and given back the next time the
material becomes a type that has it. One ``cmds.warning`` names what was
parked (``park=False`` drops the values and disconnects the wires instead,
sources kept; ``strict=True`` refuses anything lossy; ``dry_run=True`` writes
nothing and returns the :class:`Conversion` report). The old node object, and
its plugs, then raise "'red' was converted to a phong; use the node astype()
returned"; after an undo it is the live node again.

Exclusive membership means faces into the engine that already owns their
whole object is the one no-op of the grammar: the state already holds.
``-mat`` on faces of an engine that owns the whole shape carves it
(the engine keeps the complementary faces), so the removed faces are in no
engine. rig does the carving itself: before a per-face write or a removal
on a non-instanced mesh, a whole-object membership becomes an explicit
group naming every face. Left to Maya, forcing faces into another engine
carves the owner through a ``compInstObjGroups`` link under which the
owner reports every face no other engine lists -- faces removed anywhere
fall back to it and it cannot be released; rig disconnects such a link
when it meets one and re-adds explicitly the faces no other engine holds
(computed from the other engines' face groups: the link's own report is
not trusted). Instanced shapes keep Maya's own behaviour, and faces of an
instance whose engine holds the whole object there cannot be released at
all: that removal is refused. Membership is always read from the shape's
own plugs, so every operation costs one shape, never the scene. A material
is a shared, scene-level asset, so a network made inside ``with
container():`` stays OUT of the scope by default; ``create`` / ``define``
with ``container=True`` enrol the shader, its engine and materialInfo (a
per-asset look), in which case deleting that container later destroys the
network and leaves the geometry green -- ``repair()`` fixes it. The engine an
assignment builds for a shader without one joins the shader's container when
it has one, never the active scope; the geometry is never captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes import material_node as _material_node
from rig.nodetypes.material_node import (
    _own_type,
    Blinn,
    Lambert,
    Material,
    OpenPBRSurface,
    Phong,
    PhongE,
    StandardSurface,
    SurfaceShader,
)
from rig.nodetypes.shading_engine import ShadingEngine
from rig._internal.list import List
from rig._internal.members import (
    _GEOMETRY_TYPES,
    _MemberSpec,
    _NeverHolds,
    _never_held,
    _render_tokens,
    _Selection,
    Components,
    normalise,
)
from rig._internal.node import Node
from rig._internal.plug import Plug
from rig._internal.shade_convert import Conversion
from rig._internal import shade_convert
from rig._internal.undo import _undo_chunk


__all__ = [
    "Material",
    "Lambert",
    "Blinn",
    "Phong",
    "PhongE",
    "SurfaceShader",
    "StandardSurface",
    "OpenPBRSurface",
    "Default",
    "Conversion",
    "convert",
    "materials",
    "bindings",
    "repair",
    "tidy",
]


# ---------- Tables -------------------------------------------------------- #

# Shape types a shading engine binds at object level.
_SHADEABLE = frozenset({"mesh", "nurbsSurface", "subdiv"})

# The engine every new particle shape is a member of; never a material.
_PARTICLE_ENGINE = "initialParticleSE"

# What a shape holds in an engine: the whole object (None), face ids, or
# nothing at all.
_NOT_MEMBER = object()

_KIND_WORD = {
    "vtx": "vertices",
    "e":   "edges",
    "uv":  "UVs",
    "cv":  "CVs",
    "pt":  "lattice points",
}


def _short(path: str) -> str:
    return path.rsplit("|", 1)[-1]


def _leaf(name: str) -> str:
    """The name without its path and namespace."""
    return _short(name).rsplit(":", 1)[-1]


def _dedupe(names: list[str] | None) -> list[str]:
    seen: list[str] = []
    for name in names or []:
        if name not in seen:
            seen.append(name)
    return seen


# ---------- Name resolution ----------------------------------------------- #


@dataclass(frozen=True)
class _Found:
    """A material as a membership token reads it: the shader (None for an
    engine fed by none), the engine it was named through (a ``ShadingEngine``
    node token) and the shader's node type."""

    material:  str | None
    engine:    str | None
    node_type: str | None


def _engine_found(name: str) -> _Found:
    """A ``ShadingEngine`` node token: exactly that engine, whatever feeds it
    (a surface shader, a ramp, nothing); only the particle engine is refused
    (it holds particles, not surfaces)."""
    if _leaf(name) == _PARTICLE_ENGINE:
        raise TypeError(
            f"'{name}' is the particle engine, not a material; Default() is "
            f"initialShadingGroup"
        )
    shaders  = cmds.listConnections(f"{name}.surfaceShader", source=True, destination=False)
    material = shaders[0] if shaders else None
    return _Found(material, name, cmds.nodeType(material) if material else None)


def _engines_of(path: str) -> list[str]:
    """The shading engines a shape is connected to, in connection order."""
    return _dedupe(cmds.listConnections(path, type="shadingEngine"))


def _dag_path(path: str) -> OpenMaya.MDagPath:
    selection = OpenMaya.MSelectionList()
    selection.add(path)
    return selection.getDagPath(0)


def _num_faces(path: str) -> int:
    return OpenMaya.MFnMesh(_dag_path(path)).numPolygons


def _face_names(path: str, ids: np.ndarray) -> list[str]:
    return [f"{path}.{token}" for token in _render_tokens("f", ids)]


@dataclass
class _Groups:
    """The shading membership of the shape at one DAG path, read from the
    shape's own plugs (the cost of one shape, never of an engine's whole
    membership). ``owner`` is the engine holding the whole path through
    ``instObjGroups[i]`` (``i`` the instance number); ``explicit`` the
    ``(engine, face ids)`` of every ``instObjGroups[i].objectGroups[k]`` an
    engine holds, read from its ``objectGrpCompList``, which is exact;
    ``remainders`` the ``(engine, source plug, destination plug)`` of every
    ``compInstObjGroups[i].compObjectGroups[k]`` link Maya's own carve of a
    whole-object membership leaves behind; ``num_faces`` the face count of
    a mesh (``None`` for another shape)."""

    owner:      str | None
    explicit:   list[tuple[str, np.ndarray]]
    remainders: list[tuple[str, str, str]]
    instanced:  bool
    num_faces:  int | None


def _engine_at(plug: OpenMaya.MPlug) -> str | None:
    """The shading engine a plug belongs to, or ``None`` (a plain set holds
    DAG members through the same plugs)."""
    fn = OpenMaya.MFnDependencyNode(plug.node())
    return fn.name() if fn.typeName == "shadingEngine" else None


def _polygon_ids(plug: OpenMaya.MPlug) -> np.ndarray | None:
    """The polygon ids an ``objectGrpCompList`` plug carries, or ``None``
    when it carries no polygon component."""
    data = plug.asMObject()
    if data.isNull():
        return None
    fn    = OpenMaya.MFnComponentListData(data)
    parts = []
    for i in range(fn.length()):
        component = fn.get(i)
        if component.hasFn(OpenMaya.MFn.kMeshPolygonComponent):
            elements = OpenMaya.MFnSingleIndexedComponent(component).getElements()
            parts.append(np.asarray(elements, dtype=np.int64))
    return np.unique(np.concatenate(parts)) if parts else None


def _groups(path: str) -> _Groups:
    dag      = _dag_path(path)
    fn       = OpenMaya.MFnDagNode(dag)
    index    = dag.instanceNumber()
    instance = fn.findPlug("instObjGroups", False).elementByLogicalIndex(index)
    owner    = next(
        (name for name in map(_engine_at, instance.destinations()) if name is not None),
        None,
    )
    explicit  = []
    comp_list = fn.attribute("objectGrpCompList")
    groups    = instance.child(fn.attribute("objectGroups"))
    for k in groups.getExistingArrayAttributeIndices():
        element = groups.elementByLogicalIndex(k)
        for destination in element.destinations():
            name = _engine_at(destination)
            if name is None:
                continue
            ids = _polygon_ids(element.child(comp_list))
            if ids is not None:
                explicit.append((name, ids))
    remainders = []
    links      = fn.findPlug("compInstObjGroups", False)
    if index in links.getExistingArrayAttributeIndices():
        groups = links.elementByLogicalIndex(index).child(fn.attribute("compObjectGroups"))
        for k in groups.getExistingArrayAttributeIndices():
            element = groups.elementByLogicalIndex(k)
            for destination in element.destinations():
                name = _engine_at(destination)
                if name is not None:
                    source = f"{path}.compInstObjGroups[{index}].compObjectGroups[{k}]"
                    remainders.append((name, source, destination.name()))
    num_faces = OpenMaya.MFnMesh(dag).numPolygons if dag.hasFn(OpenMaya.MFn.kMesh) else None
    return _Groups(owner, explicit, remainders, dag.isInstanced(), num_faces)


def _remainder(groups: _Groups, engine: str) -> np.ndarray:
    """The faces a remainder link of ``engine`` stands for: every face no
    other engine holds explicitly. Computed from the other engines' face
    groups, never taken from the link's own report, which claims every
    face for good once the engine has been read while some faces were in
    no engine."""
    others = [ids for holder, ids in groups.explicit if holder != engine]
    taken  = np.concatenate(others) if others else np.empty(0, dtype=np.int64)
    return np.setdiff1d(np.arange(groups.num_faces), taken)


def _held(engine: ShadingEngine, path: str) -> Any:
    """What the engine holds of the shape at ``path``: ``None`` for the whole
    object (or every face of a mesh, a state Maya never collapses), the
    face ids, or :data:`_NOT_MEMBER`. On a non-instanced mesh a remainder
    link counts for every face no other engine lists, which is what Maya
    reports through it; on an instance Maya releases faces despite the
    link, so it counts for nothing."""
    groups = _groups(path)
    name   = engine.name
    if groups.owner == name:
        return None
    parts = [ids for holder, ids in groups.explicit if holder == name]
    if (
        groups.num_faces is not None
        and not groups.instanced
        and any(holder == name for holder, _, _ in groups.remainders)
    ):
        parts.append(_remainder(groups, name))
    if not parts:
        return _NOT_MEMBER
    ids = np.unique(np.concatenate(parts))
    if not ids.size:
        return _NOT_MEMBER
    if ids.size == groups.num_faces:
        return None
    return ids


def _plain_owner(path: str) -> str | None:
    """The engine holding the whole shape at ``path`` through the shape's
    own ``instObjGroups[i]`` plug (the form a whole-object assignment
    takes), or ``None``."""
    return _groups(path).owner


def _is_orphan_group_id(group_id: str) -> bool:
    """A groupId nothing consumes: no set holds it and no groupParts
    carries it (a dead, engine-less group entry on a mesh may still point
    at it; deleting the node clears that entry too)."""
    if cmds.referenceQuery(group_id, isNodeReferenced=True):
        return False
    for node in set(cmds.listConnections(group_id) or []):
        kinds = cmds.nodeType(node, inherited=True) or []
        if "objectSet" in kinds or "groupParts" in kinds:
            return False
    return True


def _make_explicit(path: str) -> None:
    """Make every membership of a non-instanced mesh an explicit face group
    before a per-face write or a removal.

    A whole-object membership is re-homed as a group naming every face, so
    that forcing some faces into another engine never makes Maya carve it:
    a Maya carve leaves a remainder link (``compInstObjGroups``) under which
    faces removed from any other engine fall back to the owner and the
    owner cannot be released. An existing remainder link is disconnected
    and the faces it stands for -- every face no other engine holds
    explicitly, :func:`_remainder` -- are re-added explicitly. What every
    engine holds does not change; only how Maya stores it. Instanced shapes
    are left as Maya keeps them.
    """
    if cmds.nodeType(path) != "mesh" or _dag_path(path).isInstanced():
        return
    groups = _groups(path)
    if groups.owner is not None:
        cmds.sets(path, edit=True, remove=groups.owner)
        cmds.sets(
            _face_names(path, np.arange(groups.num_faces)), edit=True,
            forceElement=groups.owner,
        )
        groups = _groups(path)
    for name, source, destination in groups.remainders:
        remainder = _remainder(_groups(path), name)
        cmds.disconnectAttr(source, destination)
        if remainder.size:
            cmds.sets(_face_names(path, remainder), edit=True, forceElement=name)


# ---------- The left-hand side --------------------------------------------- #


@dataclass
class _Target:
    """One shape on the left: the whole object, or some of its faces."""

    path:      str
    node_type: str
    whole:     bool
    faces:     list[_Selection]

    def face_ids(self) -> np.ndarray:
        """The union of the requested face ids (the whole object is every
        face)."""
        if self.whole:
            return np.arange(_num_faces(self.path))
        return np.unique(np.concatenate([s.indices for s in self.faces]))

    def names(self) -> list[str]:
        """The member strings ``cmds.sets`` takes."""
        if self.whole:
            return [self.path]
        return [name for s in self.faces for name in s.names]


def _kind_message(selection: _Selection) -> str:
    word = _KIND_WORD.get(selection.kind, selection.kind)
    return (
        f"materials bind faces or whole objects; {selection.names[0]} are {word}. "
        f"Use node.f[...] for faces or the node itself"
    )


def _check_kinds(selections: list[_Selection]) -> None:
    for selection in selections:
        if selection.kind not in _MaterialMember.ACCEPTS:
            raise _NeverHolds(_kind_message(selection))


def _shadeable(target: _Target) -> list[_Target]:
    """``target`` when its node wears materials; the own shadeable shapes of
    a transform the normaliser could not expand; a ``TypeError`` naming what
    is missing otherwise."""
    if target.node_type in _SHADEABLE:
        return [target]
    path = target.path
    if target.node_type in _GEOMETRY_TYPES:
        raise _NeverHolds(
            f"'{_short(path)}' is a {target.node_type}, not a shadeable surface "
            f"(mesh / nurbsSurface / subdiv)"
        )
    if not path.startswith("|"):
        raise _NeverHolds(
            f"'{path}' is a {target.node_type}, not geometry; materials bind "
            f"mesh / nurbsSurface / subdiv shapes or mesh faces"
        )
    shapes = cmds.listRelatives(
        path, shapes=True, noIntermediate=True, fullPath=True, type=sorted(_SHADEABLE)
    ) or []
    if shapes:
        return [_Target(shape, cmds.nodeType(shape), True, []) for shape in shapes]
    meshes = cmds.listRelatives(
        path, allDescendents=True, noIntermediate=True, fullPath=True, type="mesh"
    ) or []
    hint = (
        f"; its descendants {[_short(m) for m in meshes]} are meshes: assign "
        f"them (Maya would recurse the whole subtree, rig does not)"
        if meshes
        else ""
    )
    raise _NeverHolds(
        f"'{_short(path)}' ({target.node_type}) has no shadeable shape of its own{hint}"
    )


def _skipped_siblings(selections: list[_Selection]) -> set[str]:
    """The paths of the non-shadeable shapes (curves, lattices) a transform
    on the left was expanded into beside a shadeable one: the control curve
    under a mesh transform is not what the material is for. A shape named
    itself keeps raising, and so does a transform with nothing shadeable."""
    def owner(source: Any) -> Node | None:
        # An attribute plug on the left stands for its node, so the plug's
        # node is the transform that was expanded, exactly like a Node source.
        if isinstance(source, Plug):
            source = Node(source.node)
        if isinstance(source, Node) and source.long_name != selection.path:
            return source
        return None

    shaded: dict[Node, bool] = {}
    for selection in selections:
        if selection.kind != "whole" or selection.node_type not in _GEOMETRY_TYPES:
            continue
        for source in selection.source:
            parent = owner(source)
            if parent is not None:
                shaded[parent] = shaded.get(parent, False) or selection.node_type in _SHADEABLE
    skipped = set()
    for selection in selections:
        if (
            selection.kind != "whole"
            or selection.node_type not in _GEOMETRY_TYPES
            or selection.node_type in _SHADEABLE
        ):
            continue
        parents = [p for p in map(owner, selection.source) if p is not None]
        if len(parents) == len(selection.source) and all(shaded[p] for p in parents):
            skipped.add(selection.path)
    return skipped


def _targets(selections: list[_Selection]) -> list[_Target]:
    """Fold selections into one target per shape; a bare ``cube.f`` handle
    is the whole object; a transform's non-shadeable shapes are skipped
    beside a shadeable one."""
    skipped = _skipped_siblings(selections)
    by_path: dict[str, _Target] = {}
    for selection in selections:
        if selection.path in skipped:
            continue
        target = by_path.get(selection.path)
        if target is None:
            target = by_path[selection.path] = _Target(
                selection.path, selection.node_type, False, []
            )
        if selection.kind == "whole" or selection.is_all:
            target.whole = True
        else:
            target.faces.append(selection)
    result = []
    for target in by_path.values():
        result.extend(_shadeable(target))
    return result


def _single(selections: list[_Selection], verb: str) -> _Target:
    _check_kinds(selections)
    targets = _targets(selections)
    if len(targets) > 1:
        raise TypeError(
            f"{verb} one node at a time; got "
            f"{', '.join(_short(t.path) for t in targets)}"
        )
    target = targets[0]
    if target.whole and target.faces:
        raise TypeError(
            f"{_short(target.path)} and its faces in one {verb}: the node asks about "
            f"the whole object, faces about themselves. Split them"
        )
    return target


def _check_instanced(engine: str, target: _Target) -> None:
    """Refuse, before anything is written, a per-face removal from the
    engine holding the whole object at an instanced path: ``cmds.sets
    -remove`` releases none of those faces (the object-level entry stays
    and every face is still reported)."""
    if target.whole:
        return
    groups = _groups(target.path)
    if not groups.instanced or groups.owner != engine:
        return
    transform = _short(target.path.rsplit("|", 1)[0])
    raise TypeError(
        f"'{target.path}' is an instance and {engine} holds the whole object "
        f"there: Maya releases no faces of an instanced object-level membership. "
        f"Assign the faces to keep explicitly instead ({transform}.f[keep] << "
        f"Material('x')) or de-instance the shape first"
    )


def _remove_members(engine: ShadingEngine, target: _Target) -> None:
    """Take the target out of the engine, leaving the removed faces in no
    engine. The memberships of the shape are made explicit first, so faces
    of an engine that owns the whole shape are carved by rig (the engine
    keeps the complementary faces) and never by Maya."""
    if _held(engine, target.path) is _NOT_MEMBER:
        return
    _make_explicit(target.path)
    held = _held(engine, target.path)
    if target.whole:
        cmds.sets(target.path, edit=True, remove=engine.name)
    elif held is not _NOT_MEMBER:
        requested = target.face_ids()
        inside    = requested if held is None else np.intersect1d(held, requested)
        if inside.size:
            cmds.sets(_face_names(target.path, inside), edit=True, remove=engine.name)
    _verify(engine, target, present=False)


def _verify(engine: ShadingEngine, target: _Target, present: bool) -> None:
    """Read the membership back; the command's return values lie."""
    held = _held(engine, target.path)
    if target.whole:
        ok = held is None if present else held is _NOT_MEMBER
    elif present:
        ok = held is None or (
            held is not _NOT_MEMBER and bool(np.isin(target.face_ids(), held).all())
        )
    else:
        ok = held is _NOT_MEMBER or (
            held is not None and not np.isin(target.face_ids(), held).any()
        )
    if not ok:
        state = "hold" if present else "release"
        raise RuntimeError(
            f"{engine.name} did not {state} {target.names()[0]}; it holds "
            f"{cmds.sets(engine.name, query=True) or []}"
        )


# ---------- The membership of materials ----------------------------------- #


def _adopt(material: str) -> ShadingEngine:
    """Build the engine of a shader that feeds none (``lambert1``, a bare
    ``rn.blinn(name=...)``) next to the shader: in its container when it has
    one, never in the active scope (a found node is never moved)."""
    engine = ShadingEngine.for_material(material, create=True)
    owner  = cmds.container(query=True, findContainer=[material])
    if owner:
        cmds.container(
            owner, edit=True, force=True,
            addNode=[engine.name, *engine.get_material_info()],
        )
    shader = engine.get_material()
    if shader is None or shader.name != material:
        raise RuntimeError(
            f"{engine.name}.surfaceShader does not read '{material}' after the "
            f"build (it reads {shader})"
        )
    return engine


def _engine_of(found: _Found, create: bool) -> ShadingEngine | None:
    """The engine a found material assigns to (``<<``): the one it was named
    through (a ``ShadingEngine`` node), else the one its shader feeds
    (``<shader>SG`` among several; a ``ValueError`` when several and none is
    named so), else None."""
    if found.engine is not None:
        return ShadingEngine._wrap(found.engine)
    return ShadingEngine.for_material(found.material, create=create)


def _engines_read(found: _Found) -> list[ShadingEngine]:
    """The engines a query or a removal reads (``in``, ``>>``, ``-mat``): the
    one a ``ShadingEngine`` token names, else every engine the shader feeds
    (the particle engine aside), so no read raises where a shader feeds
    several engines and none is named ``<shader>SG``."""
    if found.engine is not None:
        return [ShadingEngine._wrap(found.engine)]
    return [
        ShadingEngine._wrap(name) for name in _material_node._engines_fed_by(found.material)
        if _leaf(name) != _PARTICLE_ENGINE
    ]


def _held_by(engines: list[ShadingEngine], path: str) -> Any:
    """What the engines hold of the shape at ``path`` together, as `_held`
    answers for one: ``None`` for the whole object (or every face), the face
    ids, or :data:`_NOT_MEMBER`."""
    parts = []
    for engine in engines:
        held = _held(engine, path)
        if held is None:
            return None
        if held is not _NOT_MEMBER:
            parts.append(held)
    if not parts:
        return _NOT_MEMBER
    ids = np.unique(np.concatenate(parts))
    if cmds.nodeType(path) == "mesh" and ids.size == _num_faces(path):
        return None
    return ids


def _is_default(engine: ShadingEngine) -> bool:
    """Whether ``engine`` is ``initialShadingGroup``, by its leaf (the name
    reads ``:initialShadingGroup`` with ``namespace -relativeNames`` on)."""
    return _leaf(engine.name) == ShadingEngine.DEFAULT


def _answer(kind: type, engine: ShadingEngine) -> Any:
    """What ``kind.of(x)`` lists for an engine holding ``x``: its shader when
    the shader is of ``kind`` (flat: ``Lambert`` never lists a blinn), or None.
    ``Material`` lists every engine's shader, and the engine itself where the
    shader would not go back to it through ``<<`` (so every answer does):
    ``initialShadingGroup`` (``Default()``), an engine fed by nothing or by
    something that is no surface shader, an engine that is not the one
    ``x << shader`` picks (the shader feeds several)."""
    shader = engine.get_material()
    if kind is not Material:
        return shader if isinstance(shader, kind) else None
    if shader is None or _is_default(engine) or not isinstance(shader, Material):
        return engine
    try:
        pick = ShadingEngine.for_material(shader, create=False)
    except ValueError:
        pick = None
    return shader if pick is not None and pick.name == engine.name else engine


def _of(kind: type, selections: list[_Selection]) -> list:
    """The materials of ``kind`` holding the one shape (or faces) of
    ``selections``: every face of the left-hand side, in connection order,
    each once."""
    target = _single(selections, f"{kind.__name__}.of takes")
    found  = []
    for name in _engines_of(target.path):
        engine = ShadingEngine._wrap(name)
        held   = _held(engine, target.path)
        if held is _NOT_MEMBER:
            continue
        if not target.whole and held is not None:
            if not np.isin(target.face_ids(), held).all():
                continue
        answer = _answer(kind, engine)
        if answer is not None and answer not in found:
            found.append(answer)
    return found


class _MaterialMember(_MemberSpec):
    """The membership of materials (private): what a shader node, a
    ``ShadingEngine`` node, their removal tokens and a class's kind token run
    for ``<<`` / ``>>`` / ``in`` / ``of``. Built by the node classes
    (``red._member()``, ``-red``, ``Blinn()``), never by a user.

    ``members << red`` moves the members into red's engine (one
    ``cmds.sets(forceElement)``; a shader feeding no engine gets
    ``<shader>SG`` beside it first, the adopt path; among several engines
    ``<shader>SG``, a ``ValueError`` before any write when none is named so),
    ``members << ShadingEngine("xSG")`` into exactly that engine, ``members
    << -red`` takes them out (faces of an engine that owns the whole shape
    carve it), ``members << Material()`` / ``Blinn()`` out of every engine
    (green); ``<<`` returns the left-hand side and every write of one ``<<``
    is one undo chunk ``rig.material``. ``lhs >> red`` reads the face ids of
    the left-hand side that wear the material (every face when object-level,
    empty when none), ``lhs in red`` whether every face of every shape of the
    left-hand side (or the whole object) wears it, and ``lhs >> Material()`` /
    ``Material.of(lhs)`` the materials holding it (``Blinn.of`` keeps the
    blinns). The particle engine and an engine fed by no surface shader are
    refused before any write.
    """

    KIND        = "material"
    ACCEPTS     = frozenset({"whole", "f"})
    WANT_SHAPES = True

    def __init__(self, node: Any = None, remove: bool = False, kind: type = Material) -> None:
        # the shader or engine node (None: the kind token of ``kind``); its
        # name is read at each use, so a renamed node is followed and a
        # deleted one raises before any write
        self._node    = node
        self._kind    = kind
        self._remove  = remove
        self._options = {}

    @property
    def _name(self) -> str | None:
        return None if self._node is None else self._node.name

    @property
    def purges(self) -> bool:
        return self._node is None

    def __neg__(self) -> "_MaterialMember":
        if self._node is None:
            raise TypeError(
                f"-{self!r} is a double negative: {self!r} already removes the members "
                f"from every shading engine"
            )
        if self._remove:
            raise TypeError(f"-{self!r}: a removal cannot be negated again")
        return _MaterialMember(self._node, remove=True)

    def __invert__(self) -> Any:
        raise TypeError(
            f"~{self!r} is unassigned; -mat removes members and Material() removes "
            f"them from every shading engine"
        )

    def __repr__(self) -> str:
        if self._node is None:
            return f"{self._kind.__name__}()"
        return f"{'-' if self._remove else ''}{self._node!r}"

    def __getattr__(self, name: str) -> Any:
        # a token is no material: its methods and plugs are the node's
        if name.startswith("_"):
            raise AttributeError(name)
        what = "the kind token (every material)" if self._node is None else "a removal token"
        raise AttributeError(
            f"{self!r} is {what}, not a material; {name} is the node's: "
            f"Material('x').{name}"
        )

    def _found(self) -> _Found:
        """The material as it is now: a shader node is itself, an engine
        node stands for the shader feeding it (the particle engine and an
        engine fed by no surface shader raise). A deleted node raises its
        ``already deleted!`` here, before any write."""
        name = self._node.name
        if isinstance(self._node, ShadingEngine):
            return _engine_found(name)
        return _Found(name, None, cmds.nodeType(name))

    def _clone_target(self) -> str | None:
        return None if self._node is None or self._remove else self._node.name

    # -- refusals -- #

    def _kind_error(self, selection: _Selection) -> str:
        return _kind_message(selection)

    # -- '<<' -- #

    def _plan(self, selections: list[_Selection]) -> list[Callable[[], None]]:
        targets = _targets(selections)
        if self._node is None:
            return self._plan_purge(targets)
        if self._remove:
            return self._plan_remove(targets)
        return self._plan_add(targets)

    def _apply(self, plan: list[Callable[[], None]]) -> None:
        for step in plan:
            step()

    @staticmethod
    def _plan_purge(targets: list[_Target]) -> list:
        steps = []
        for target in targets:
            for name in _engines_of(target.path):
                _check_instanced(name, target)
                engine = ShadingEngine._wrap(name)
                steps.append(lambda e=engine, t=target: _remove_members(e, t))
        return steps

    def _plan_remove(self, targets: list[_Target]) -> list:
        steps = []
        for engine in _engines_read(self._found()):
            for target in targets:
                _check_instanced(engine.name, target)
                steps.append(lambda e=engine, t=target: _remove_members(e, t))
        return steps

    def _plan_add(self, targets: list[_Target]) -> list:
        found = self._found()
        # a shader feeding several engines with none named <shader>SG raises
        # here, before any write
        _engine_of(found, create=False)
        return [lambda: self._assign(found, targets)]

    @staticmethod
    def _assign(found: _Found, targets: list[_Target]) -> None:
        engine = _engine_of(found, create=False)
        if engine is None:
            engine = _adopt(found.material)
        for target in targets:
            # Faces into the engine that already owns the whole shape is the
            # one permitted no-op: the state holds and nothing is rewritten.
            if not target.whole and _plain_owner(target.path) != engine.name:
                _make_explicit(target.path)
        engine.assign([name for target in targets for name in target.names()])
        for target in targets:
            _verify(engine, target, present=True)

    # -- '>>' and 'in' -- #

    def _query(self, selections: list[_Selection]) -> np.ndarray:
        target = _single(selections, "'>>' queries")
        if target.node_type != "mesh":
            raise TypeError(
                f"'>>' reads face ids and '{_short(target.path)}' is a "
                f"{target.node_type} without faces; Material.of({Node(target.path)!r}) "
                f"lists its materials"
            )
        engines = _engines_read(self._found())
        empty   = np.empty(0, dtype=np.int64)
        held    = _held_by(engines, target.path)
        if held is _NOT_MEMBER:
            return empty
        if held is None:
            held = np.arange(_num_faces(target.path))
        if target.whole:
            return held
        return np.intersect1d(target.face_ids(), held)

    def _contains(self, selections: list[_Selection]) -> bool:
        targets = _targets(selections)
        engines = _engines_read(self._found())
        if not engines:
            return False
        for target in targets:
            held = _held_by(engines, target.path)
            if held is _NOT_MEMBER:
                return False
            if held is None:
                continue   # the whole object (or every face of it)
            if target.whole or not np.isin(target.face_ids(), held).all():
                return False
        return True

    def _of(self, selections: list[_Selection]) -> list:
        return _of(self._kind, selections)

    def _enumerate(self, x: Any) -> list:
        """``Cls.of(x)``: the materials of this token's kind holding ``x``
        (an attribute plug is a ``TypeError``: the node is the member; what a
        material never holds, a vertex or a curve, wears none: ``[]``)."""
        kind       = self._kind.__name__
        selections = normalise(
            x,
            want_shapes  = True,
            refuse_plugs = lambda plug: TypeError(
                f"'{plug}' is a plug; membership takes the node: {kind}.of({plug.node})"
            ),
        )
        try:
            self._check_kinds(selections)
            return _of(self._kind, selections)
        except TypeError as error:
            if not _never_held(error):
                raise
            return []   # what a material never holds wears none


def Default(*args: Any, **kwargs: Any) -> ShadingEngine:
    """``initialShadingGroup``, the default shading engine, as its node
    (``ShadingEngine("initialShadingGroup")``): ``cube << Default()`` reverts to
    it, ``cube << -Default()`` leaves it (green), ``cube.f[:6] >> Default()``
    reads its faces, ``Material.of(x)`` lists it for the faces it holds. It
    takes no name: ``Default("x")`` is a ``TypeError``."""
    if args or kwargs:
        raise TypeError(
            "Default() takes no name: it is initialShadingGroup. Material() removes "
            "the members from every engine; Blinn('x') refers to a blinn"
        )
    return ShadingEngine._wrap(ShadingEngine.DEFAULT)


# ---------- Conversion ---------------------------------------------------- #


def _target_type(to: Any) -> str:
    """The node type a conversion target names: a shader class of one type
    (``Phong``) or a type name (``'phong'``)."""
    if isinstance(to, type) and issubclass(to, Material):
        node_type = _own_type(to)
        if node_type is None:
            raise TypeError(
                f"{to.__name__} names no node type; convert to a class of one type "
                f"(Phong) or a type name ('phong')"
            )
        return node_type
    if isinstance(to, str) and to:
        return to
    raise TypeError(
        f"a conversion target is a shader class (Phong) or a node type name "
        f"('phong'), not {to!r}"
    )


def convert(
    x:       Any,
    to:      Any,
    *,
    strict:  bool = False,
    dry_run: bool = False,
    park:    bool = True,
    update:  bool = False,
    **attrs: Any,
) -> Any:
    """Make the material ``x`` -- a shader node or its name, read by
    ``Material(x)`` (a missing name is a NodeNotFoundError, another type a
    NodeTypeError) -- the node type ``to`` (a class such as ``Phong`` or a
    type name) and return THE NEW NODE (``Phong("red")``); the node object
    ``x`` was then raises, naming the conversion. See
    :meth:`rig.nodetypes.Material.astype` for the options; ``dry_run=True``
    returns the :class:`Conversion` report instead (``bool(report)`` is
    "something would be parked or lost", ``str(report)`` the warning text) and
    writes nothing."""
    if isinstance(x, _MemberSpec):
        raise TypeError(
            f"{x!r} is a membership token, not a material; convert the material "
            f"node: mat.astype(...) or shade.convert(mat, ...)"
        )
    return shade_convert.convert(
        Material(x), _target_type(to), strict=strict, dry_run=dry_run, park=park,
        attrs=attrs, update=update,
    )


# the membership a material node runs (D31: ``rig.nodetypes`` imports none of
# the DSL modules; this module binds it when it loads)
_material_node._MATERIAL_MEMBER = _MaterialMember


# ---------- Module functions ---------------------------------------------- #


def _entries(target: _Target) -> list[tuple[ShadingEngine, Any]]:
    """``(engine, held)`` for every engine holding something of the target:
    ``held`` is ``None`` for a whole-object target the engine owns entirely,
    else the requested face ids the engine holds."""
    result = []
    for name in _engines_of(target.path):
        engine = ShadingEngine._wrap(name)
        held   = _held(engine, target.path)
        if held is _NOT_MEMBER:
            continue
        if not target.whole:
            held = target.face_ids() if held is None else np.intersect1d(target.face_ids(), held)
            if not held.size:
                continue
        result.append((engine, held))
    return result


def _scene_targets(targets: Any) -> list[_Target]:
    """The shapes a module function works on: every non-intermediate
    shadeable shape in the scene, or the normalised ``targets``."""
    if targets is None:
        paths = cmds.ls(
            type=sorted(_SHADEABLE), long=True, noIntermediate=True, allPaths=True
        ) or []
        return [_Target(path, cmds.nodeType(path), True, []) for path in paths]
    selections = normalise(targets, want_shapes=True)
    _check_kinds(selections)
    return _targets(selections)


def materials(target: Any) -> List:
    """The materials of ``target`` (a node, faces, or a list of them) as
    material ``Node``s, in connection order, each once: the shaders
    themselves (``initialShadingGroup``'s is ``standardSurface1``), where
    ``Material.of(x)`` answers the handles ``<<`` takes back (``Default()``
    for the default engine, the engine for one ``x << shader`` would not
    pick)."""
    found: list[Node] = []
    for item in _scene_targets(target):
        for engine, _ in _entries(item):
            shader = engine.get_material()
            if shader is None:
                continue
            node = Node(shader)
            if node not in found:
                found.append(node)
    return List(found)


def bindings(target: Any) -> list[tuple[Node, Components]]:
    """``(material, faces)`` per engine holding faces of ``target``; an
    object-level membership is reported as the whole-kind ``f[*]``
    :class:`Components`. The material is the engine's shader, as
    :func:`materials` answers it. Read through ``MFnSet``, never through the
    strings ``cmds.sets(q=True)`` prints. Mesh only."""
    result = []
    for item in _scene_targets(target):
        if item.node_type != "mesh":
            raise TypeError(
                f"bindings are per face and '{_short(item.path)}' is a "
                f"{item.node_type}; shade.materials({Node(item.path)!r}) lists its materials"
            )
        shape = Node(item.path)
        for engine, held in _entries(item):
            shader = engine.get_material()
            if shader is None:
                continue
            faces = (
                Components(shape, "f") if held is None
                else Components._from_ids(shape, "f", held)
            )
            result.append((Node(shader), faces))
    return result


def repair(targets: Any = None) -> List:
    """Re-home to ``initialShadingGroup`` every shape (or face) of
    ``targets`` -- every DAG path of the scene by default -- that no shading
    engine holds, as a deleted network or ``Material()`` leaves them.
    Each instance path is judged on its own: an instance whose other path
    wears a material is still re-homed at object level. Returns the shapes
    touched."""
    fixed: list[Node] = []
    with _undo_chunk("rig.material"):
        for item in _scene_targets(targets):
            entries = _entries(item)
            if item.whole and not entries:
                cmds.sets(item.path, edit=True, forceElement=ShadingEngine.DEFAULT)
                fixed.append(Node(item.path))
                continue
            if item.node_type != "mesh":
                continue
            covered = np.empty(0, dtype=np.int64)
            for _, held in entries:
                if held is None:
                    covered = None
                    break
                covered = np.union1d(covered, held)
            if covered is None:
                continue
            missing = np.setdiff1d(item.face_ids(), covered)
            if missing.size:
                cmds.sets(
                    _face_names(item.path, missing), edit=True,
                    forceElement=ShadingEngine.DEFAULT,
                )
                fixed.append(Node(item.path))
    return List(fixed)


def tidy(targets: Any = None) -> None:
    """Collapse to object level every mesh of ``targets`` (the whole scene
    by default) whose faces all sit in one engine as a face component (Maya
    never collapses that state itself), and delete the groupId nodes
    per-face memberships leave behind once no set holds them and no
    groupParts carries them. Off the operator path: a tidy-up is its own
    undo step."""
    with _undo_chunk("rig.material"):
        for item in _scene_targets(targets):
            if item.node_type != "mesh":
                continue
            for engine, held in _entries(_Target(item.path, "mesh", True, [])):
                if held is None:
                    engine.assign([item.path], touched=[item.path], normalise=True)
        for group_id in cmds.ls(type="groupId") or []:
            if _is_orphan_group_id(group_id):
                cmds.delete(group_id)
