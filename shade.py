"""
Materials through the membership grammar.

A material spec -- ``Blinn("red")``, ``Lambert("skin", diffuse=0.8)``,
``Material("look:x", type="phong")`` -- is a lazy handle on a surface shader
and its shading engine. It makes zero Maya calls until it meets ``<<``: then
``red`` is found by name (exact, then in the current namespace) or built --
``cmds.shadingNode`` for the shader, ``cmds.sets(renderable=True)`` for
``redSG`` with its materialInfo, renderPartition, lightLinker and
defaultShaderList1 wiring -- the kwargs are applied as attribute injections
(a value sets, a plug connects, a spec applies; skipped on a FOUND material
unless ``update=True``), and the left-hand side is moved into the engine in
ONE ``cmds.sets(forceElement)``. Shading membership is exclusive: a member
leaves whatever engine held it.

The left of a material ``<<`` is a node -- its OWN non-intermediate
shadeable shapes (mesh / nurbsSurface / subdiv), never the transform, which
Maya would recurse through the whole subtree -- or mesh faces
(``cube.f[:3]``). Vertices, edges, UVs and attributes are ``TypeError``s:
materials bind faces or whole objects.

Usage::

    from rig import Node, PlugList, shade
    from rig.shade import Blinn, Lambert, Material, Default

    cube = Node("pCube1")
    red  = Blinn("red", color=(1, 0, 0))     # inert: zero Maya calls
    cube << red                              # builds red + redSG, sets color, assigns; returns cube
    cube.f[:3] << Lambert("decal")           # faces 0-2 leave redSG for decalSG; returns cube.f[:3]
    red.color << (0, 1, 0)                   # Plug("red.color"): the spec is the handle (find-only)
    red.node ; red.engine                    # Node("red") ; Node("redSG")   ValueError until built
    cube >> Blinn("red")                     # array([3, 4, 5])   faces wearing red (all: object-level)
    Material.of(cube)                        # [Blinn('red'), Lambert('decal')]   by live nodeType

    cube.f[:3] << -Lambert("decal")          # those faces are in NO shading engine (green)
    cube << Material()                       # green everywhere (Material(None) is the same spec)
    cube >> Material() ; cube >> Blinn()     # the operator spelling of Material.of / Blinn.of
    cube.tx << Blinn("red")                  # an attribute plug stands for its node
    cube << Default()                        # initialShadingGroup again
    PlugList([cube, sph]) << Blinn("m")      # one material, one engine, one cmds.sets

    Material("red").delete()                 # red, redSG and its materialInfo; members go green
    shade.repair()                           # PlugList of the shapes re-homed to initialShadingGroup
    shade.tidy()                             # all-faces memberships collapsed to object level

    Phong(red)                               # the type switch: a free retype while red is lazy, a scene
                                             #   conversion once it exists (one undo chunk; the one
                                             #   constructor with a scene side effect); red is retyped
                                             #   in place and the call returns an equal fresh handle
    red.astype("blinn")                      # the verb: in place, returns red (strict=, dry_run=, park=)
    shade.convert(red, "lambert", dry_run=True)   # the engine: a Conversion report, nothing written

Conversion keeps the name, the engines, the materialInfo, the
``defaultShaderList1`` slot, the container, the dynamic attributes and the
locks, moves every wire the target can hold, and PARKS what it cannot:
each non-default value, wire or animCurve on an attribute the target lacks
is cloned onto the same node as a hidden ``__attr__`` of the same type and
given back the next time the material becomes a type that has it. One
``cmds.warning`` names what was parked (``park=False`` drops the values and
disconnects the wires instead, sources kept; ``strict=True`` refuses
anything lossy). Live ``Node`` / ``Plug`` wrappers of the old node die
(``already deleted!``); the spec is the surviving handle.

Exclusive membership means faces into the engine that already owns their
whole object is the one no-op of the grammar: the state already holds.
``-Material("x")`` on faces of an engine that owns the whole shape carves it
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
own plugs, so every operation costs one shape, never the scene. ``.rename``
follows the ``<mat>SG`` convention; ``.delete`` refuses Maya default and
referenced nodes. A material is a shared, scene-level asset, so a network
built inside ``with container():`` stays OUT of the scope by default;
``container=True`` enrols the shader, its engine and materialInfo (a
per-asset look), in which case deleting that container later destroys the
network and leaves the geometry green -- ``repair()`` fixes it. The created
nodes keep their names (no flatten prefix) and the geometry is never
captured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig.maya.nodetypes.shading_engine import ShadingEngine
from rig._internal.list import PlugList
from rig._internal.members import (
    _check_attrs,
    _find_node,
    _GEOMETRY_TYPES,
    _MemberSpec,
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

# The classification every surface shader satisfies.
_SURFACE = "shader/surface"

# The engine every new particle shape is a member of; never a material.
_PARTICLE_ENGINE = "initialParticleSE"

# A node name Maya keeps: identifiers joined by namespace or path separators.
_NAME_RE = re.compile(r"^\|?[A-Za-z_][A-Za-z0-9_]*(?:[:|][A-Za-z_][A-Za-z0-9_]*)*$")

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
    """A material that exists: its node, the engine it was named through
    (when the spec wrapped a shading engine) and its node type."""

    material:  str
    engine:    str | None
    node_type: str


def _is_surface_shader(node_type: str) -> bool:
    return bool(cmds.getClassification(node_type, satisfies=_SURFACE))


def _classify(name: str) -> _Found:
    """What an existing node means as a material: a surface shader is
    itself; a shading engine stands for the shader feeding it."""
    node_type = cmds.nodeType(name)
    if node_type == "shadingEngine":
        if _leaf(name) == _PARTICLE_ENGINE:
            raise TypeError(
                f"'{name}' is the particle engine, not a material; Default() is "
                f"initialShadingGroup"
            )
        shaders = cmds.listConnections(f"{name}.surfaceShader", source=True, destination=False)
        if not shaders:
            raise ValueError(
                f"shading engine '{name}' has no surface shader; connect one or "
                f"name a material"
            )
        material    = shaders[0]
        shader_type = cmds.nodeType(material)
        if not _is_surface_shader(shader_type):
            raise TypeError(
                f"shading engine '{name}' is fed by '{material}', a {shader_type}, "
                f"not a surface shader"
            )
        return _Found(material, name, shader_type)
    if _is_surface_shader(node_type):
        return _Found(name, None, node_type)
    raise TypeError(
        f"'{name}' exists and is a {node_type}, not a surface shader or a "
        f"shading engine"
    )


def _gate_type(node_type: str) -> None:
    """Refuse a type Maya cannot build a surface shader from (``createNode``
    of an unregistered type silently makes an ``unknown`` node)."""
    legal = cmds.listNodeTypes(_SURFACE) or []
    if node_type in legal and _is_surface_shader(node_type):
        return
    classification = [c for c in cmds.getClassification(node_type) or [] if c]
    if classification:
        raise TypeError(
            f"'{node_type}' is not a surface shader ({classification[0]}); "
            f"legal types: {legal}"
        )
    raise ValueError(
        f"'{node_type}' is not a registered surface shader (load its plugin "
        f"first); legal types: {legal}"
    )


def _engines_of(path: str) -> list[str]:
    """The shading engines a shape is connected to, in connection order."""
    return _dedupe(cmds.listConnections(path, type="shadingEngine"))


def _engines_fed_by(material: str) -> list[str]:
    """The shading engines whose ``surfaceShader`` a material feeds."""
    plugs = cmds.listConnections(
        material, type="shadingEngine", source=False, destination=True, plugs=True
    )
    engines = []
    for plug in plugs or []:
        node, _, attr = plug.rpartition(".")
        if attr == "surfaceShader" and node not in engines:
            engines.append(node)
    return engines


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
        if selection.kind not in Material.ACCEPTS:
            raise TypeError(_kind_message(selection))


def _shadeable(target: _Target) -> list[_Target]:
    """``target`` when its node wears materials; the own shadeable shapes of
    a transform the normaliser could not expand; a ``TypeError`` naming what
    is missing otherwise."""
    if target.node_type in _SHADEABLE:
        return [target]
    path = target.path
    if target.node_type in _GEOMETRY_TYPES:
        raise TypeError(
            f"'{_short(path)}' is a {target.node_type}, not a shadeable surface "
            f"(mesh / nurbsSurface / subdiv)"
        )
    if not path.startswith("|"):
        raise TypeError(
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
    raise TypeError(
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
        if isinstance(source, Node) and source._dg_node.long_name != selection.path:
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


# ---------- Material ------------------------------------------------------ #


class Material(_MemberSpec):
    """A material spec: ``Material("x")``, ``Blinn("x")``, ``-Blinn("x")``,
    ``Material()``.

    ``Material(name=None, *, type=None, unique=False, update=False,
    container=None, **attrs)`` makes zero Maya calls. ``name`` is the
    find-or-create key: a surface shader of that name is wrapped
    (``Material("lambert1")``), a shading engine's name or ``Node`` stands
    for the shader feeding it (``Material("redSG")``), and an absent name is
    built on ``<<`` when a type is known -- ``Blinn("x")`` or
    ``Material("x", type="blinn")``; ``Material("x")`` alone is a
    ``ValueError`` at that point. A typed spec asserts the type: ``Blinn("x")``
    on a phong is a ``TypeError`` pointing at ``Material("x")``. ``attrs`` are
    attribute injections applied on create (``color=(1, 0, 0)`` sets,
    ``normalCamera=bump.outNormal`` connects) and skipped on a found material
    unless ``update=True``; ``unique=True`` builds a fresh network on every
    ``<<`` (the name is then not an idempotent key; ``.node`` follows the
    last network this spec built); ``container=True`` puts a new network
    into the active ``with container():`` (a material is a shared,
    scene-level asset and stays out by default).

    ``members << spec`` moves the members into the material's engine (one
    ``cmds.sets(forceElement)``), ``members << -spec`` takes them out (faces
    of an engine that owns the whole shape carve it), ``members <<
    Material()`` takes them out of every engine (green); ``<<`` returns
    the left-hand side and every write of one ``<<`` is one undo chunk.
    ``lhs >> spec`` reads the face ids of the left-hand side that wear the
    material (every face when object-level, empty when none; a missing
    material is a ``ValueError``). ``Material.of(x)`` lists the materials
    of ``x`` typed by their live node type, ``Default()`` for
    initialShadingGroup; ``Blinn.of(x)`` keeps the blinns; ``x >>
    Material()`` / ``x >> Blinn()`` are the operator spellings of the two.
    An attribute plug on the left stands for its node: ``cube.tx <<
    Blinn("x")`` dresses cube's shapes.

    The spec is the find-only handle: ``.node`` / ``.engine`` are the
    material and its engine (``ValueError`` until built), any other name
    forwards to the material node (``red.color << v`` returns the plug for
    chaining; ``red.color = v`` is the same injection as sugar) and a typo
    raises without creating anything. ``.build()`` creates the network with
    no target. ``.delete()`` removes the material, its engines and their
    materialInfos (members go green; :func:`repair` re-homes them);
    ``.rename(new)`` renames the material and a ``<mat>SG`` engine and the
    spec follows the new name. Both refuse Maya default and referenced nodes.

    A spec as the name converts: ``Phong(mat)`` / ``Material(mat,
    type="phong")`` retype a lazy ``mat`` for free (pending kwargs the
    target lacks are pruned with one warning) and convert a realised one in
    the scene (:meth:`astype`) -- the one constructor with a scene side
    effect. ``mat`` is retyped in place and the call returns a fresh, equal
    handle (``p == mat``, ``p is not mat``; specs compare by name).
    ``Material(mat)`` is a plain re-wrap. A string still asserts the type
    and a ``Node`` must be wrapped first: ``Phong(Material(node))``.
    """

    KIND        = "material"
    ACCEPTS     = frozenset({"whole", "f"})
    WANT_SHAPES = True

    # The node type a subclass builds and asserts; the generic spec takes it
    # from ``type=``.
    NODE_TYPE: str | None = None

    def __init__(
        self,
        name:      Any         = None,
        *,
        type:      str  | None = None,
        unique:    bool        = False,
        update:    bool        = False,
        container: bool | None = None,
        **attrs:   Any,
    ) -> None:
        cls = self.__class__.__name__
        if type is not None and self.NODE_TYPE is not None and type != self.NODE_TYPE:
            raise TypeError(
                f"{cls}({name!r}, type={type!r}) is contradictory: {cls} builds a "
                f"{self.NODE_TYPE}; write Material({name!r}, type={type!r})"
            )
        source = None
        if isinstance(name, Material):
            source = name
            name, type, unique, update, container, attrs = self._retype_args(
                source, type, unique, update, container, attrs
            )
        elif isinstance(name, Node):
            if type is not None or self.NODE_TYPE is not None:
                raise TypeError(
                    f"{cls}({name!r}): a Node is not converted directly; wrap it "
                    f"first: {cls}(Material({str(name)!r}))"
                )
            name = str(name)
        super().__init__(name)
        if name is not None and not _NAME_RE.match(name):
            raise ValueError(
                f"{name!r} is not a node name Maya keeps: identifiers of "
                f"[A-Za-z0-9_] not starting with a digit, joined by ':' or '|'"
            )
        self._type      = self.NODE_TYPE if type is None else type
        self._unique    = bool(unique)
        self._update    = bool(update)
        self._container = container
        self._attrs     = dict(attrs)
        self._built     = None
        if source is not None:
            self._built = source._built
            if self._type is not None:
                self.__class__ = _BY_TYPE.get(self._type, Material)

    def _retype_args(
        self,
        source:    "Material",
        type:      str        | None,
        unique:    bool,
        update:    bool,
        container: bool       | None,
        attrs:     dict,
    ) -> tuple:
        """``Cls(spec, ...)``: convert ``spec`` to this class's type (or the
        given one) and return the arguments of the fresh handle."""
        cls    = self.__class__.__name__
        target = self.NODE_TYPE if type is None else type
        if target is None:
            return (
                source._name,
                None,
                unique or source._unique,
                update or source._update,
                source._container if container is None else container,
                {**source._attrs, **attrs},
            )
        if isinstance(source, Default):
            raise TypeError(
                f"{cls}(Default()): Default() is initialShadingGroup and is never "
                f"converted; convert a material of your own"
            )
        if source._name is None:
            raise TypeError(
                f"{cls}(Material(None)): Material(None) names no material to convert"
            )
        if source._remove:
            raise TypeError(
                f"{cls}({source!r}): a removal is not converted; write "
                f"-{cls}({source.__class__.__name__}({source._name!r}))"
            )
        source._verb("astype")
        shade_convert.convert(
            source, target, strict=False, dry_run=False, park=True, attrs=attrs,
            update=bool(update),
        )
        return (
            source._name,
            target,
            unique or source._unique,
            update or source._update,
            source._container if container is None else container,
            dict(source._attrs),
        )

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Material)
            and self._name == other._name
            and self._remove == other._remove
        )

    def __hash__(self) -> int:
        return hash((self._name, self._remove))

    # -- spec-as-handle -- #

    def __getattr__(self, name: str) -> Any:
        # Python probes private and dunder names (``__deepcopy__``,
        # ``__setstate__``); the material's own attributes never start with
        # an underscore, so those stay on the spec.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.node, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        getattr(self.node, name) << value

    @property
    def type(self) -> str | None:
        """The node type the spec builds or asserts (``None``: any surface
        shader)."""
        return self._type

    @property
    def attrs(self) -> dict:
        """The pending attribute injections (a copy)."""
        return dict(self._attrs)

    @property
    def node(self) -> Node:
        """The material node; ``ValueError`` until it exists."""
        return Node(self._require().material)

    @property
    def engine(self) -> Node:
        """The material's shading engine; ``ValueError`` until it exists."""
        found  = self._require()
        engine = self._engine_of(found, create=False)
        if engine is None:
            raise ValueError(
                f"'{found.material}' has no shading engine yet; assign it "
                f"(geometry << {self!r}) or build it ({self!r}.build())"
            )
        return Node(engine.name)

    # -- resolution (reads only) -- #

    def _verb(self, verb: str) -> None:
        cls = type(self).__name__
        if self._remove:
            raise TypeError(
                f"-{cls}({self._name!r}).{verb}() is unassigned; call the method on "
                f"{cls}({self._name!r})"
            )
        if self._name is None:
            raise TypeError(f"{cls}(None).{verb}(): a method names one material")

    def _locate(self) -> _Found | None:
        """The material this spec names as it exists now, whatever its
        type, or ``None``."""
        cls = type(self).__name__
        if self._name is None:
            raise TypeError(f"{cls}(None) names no material; it removes from every engine")
        name = None
        if self._unique and self._built is not None:
            hit  = cmds.ls(self._built) or []
            name = hit[0] if hit else None
        if name is None:
            name = _find_node(self._name)
        if name is None:
            return None
        return _classify(name)

    def _resolve(self) -> _Found | None:
        """The material this spec names as it exists now, or ``None``; a
        typed spec asserts the type."""
        cls   = type(self).__name__
        found = self._locate()
        if found is not None and self._type is not None and found.node_type != self._type:
            raise TypeError(
                f"'{found.material}' exists as a {found.node_type}; {cls}({self._name!r}) "
                f"asserts a {self._type}. Wrap it as it is with Material({self._name!r}), "
                f"or convert it: {cls}(Material({self._name!r}))"
            )
        return found

    def _require(self) -> _Found:
        found = self._resolve()
        if found is None:
            raise ValueError(
                f"no surface shader named '{self._name}'; {self!r} builds it when "
                f"it meets '<<' (or with .build())"
            )
        return found

    @staticmethod
    def _engine_of(found: _Found, create: bool) -> ShadingEngine | None:
        if found.engine is not None:
            return ShadingEngine(found.engine)
        return ShadingEngine.for_material(found.material, create=create)

    # -- refusals -- #

    def _kind_error(self, selection: _Selection) -> str:
        return _kind_message(selection)

    # -- '<<' -- #

    def _plan(self, selections: list[_Selection]) -> list[Callable[[], None]]:
        targets = _targets(selections)
        if self._name is None:
            return self._plan_purge(targets)
        if self._remove:
            return self._plan_remove(targets)
        return self._plan_add(targets)

    def _apply(self, plan: list[Callable[[], None]]) -> None:
        for step in plan:
            step()

    def _plan_purge(self, targets: list[_Target]) -> list:
        steps = []
        for target in targets:
            for name in _engines_of(target.path):
                _check_instanced(name, target)
                engine = ShadingEngine(name)
                steps.append(lambda e=engine, t=target: _remove_members(e, t))
        return steps

    def _plan_remove(self, targets: list[_Target]) -> list:
        found  = self._require()
        engine = self._engine_of(found, create=False)
        if engine is None:
            return []
        for target in targets:
            _check_instanced(engine.name, target)
        return [lambda t=target: _remove_members(engine, t) for target in targets]

    def _plan_add(self, targets: list[_Target]) -> list:
        found = None if self._unique else self._resolve()
        if found is None:
            if self._type is None:
                raise ValueError(
                    f"no surface shader named '{self._name}' and Material({self._name!r}) "
                    f"has no type to build one; write Blinn({self._name!r}) or "
                    f"Material({self._name!r}, type='blinn')"
                )
            _gate_type(self._type)
            _check_attrs(self._attrs, node_type=self._type)
        else:
            # A material feeding several engines with none named <mat>SG
            # raises here, before any write.
            self._engine_of(found, create=False)
            if self._update:
                _check_attrs(self._attrs, node=found.material)
        return [lambda: self._assign(found, targets)]

    def _assign(self, found: _Found | None, targets: list[_Target]) -> None:
        _, engine = self._realise(found)
        for target in targets:
            # Faces into the engine that already owns the whole shape is the
            # one permitted no-op: the state holds and nothing is rewritten.
            if not target.whole and _plain_owner(target.path) != engine.name:
                _make_explicit(target.path)
        engine.assign([name for target in targets for name in target.names()])
        for target in targets:
            _verify(engine, target, present=True)

    # -- the writes -- #

    def _realise(self, found: _Found | None) -> tuple[str, ShadingEngine]:
        """Find or create the network (inside the caller's undo chunk) and
        return ``(material, engine)``."""
        if found is None:
            return self._create()
        material = found.material
        engine   = self._engine_of(found, create=False)
        if engine is None:
            engine = self._adopt(material)
        if self._update:
            self._inject_attrs(material)
        return material, engine

    def _create(self) -> tuple[str, ShadingEngine]:
        requested = self._name
        material  = cmds.shadingNode(
            self._type, asShader=True, name=requested, skipSelect=True
        )
        if _leaf(material) != _leaf(requested) and not self._unique:
            cmds.warning(
                f"rig.shade: Maya named the new {self._type} '{material}', not "
                f"'{requested}', so {self!r} cannot find it again: pick a name "
                f"Maya keeps, or pass unique=True to build a fresh network on purpose"
            )
        engine = ShadingEngine.for_material(material, create=True)
        if self._container is True:
            # A material is a scene-level, shared asset: it stays out of the
            # active rig container unless asked (deleting the container
            # would otherwise take the look with it).
            from rig._internal.container import container

            container.add([material, engine.name, *engine.get_material_info()])
        self._inject_attrs(material)
        shader = engine.get_material()
        if shader is None or shader.name != material:
            raise RuntimeError(
                f"{engine.name}.surfaceShader does not read '{material}' after the "
                f"build (it reads {shader})"
            )
        if self._unique:
            self._built = cmds.ls(material, uuid=True)[0]
        return material, engine

    def _adopt(self, material: str) -> ShadingEngine:
        """Build the engine of a bare material (``rn.blinn(name=...)``) next
        to the material: in its container when it has one, never in the
        active scope (a found node is never moved)."""
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

    def _inject_attrs(self, material: str) -> None:
        node = Node(material)
        for attr, value in self._attrs.items():
            getattr(node, attr) << value

    # -- '>>' -- #

    def _query(self, selections: list[_Selection]) -> np.ndarray:
        target = _single(selections, "'>>' queries")
        if target.node_type != "mesh":
            raise TypeError(
                f"'>>' reads face ids and '{_short(target.path)}' is a "
                f"{target.node_type} without faces; Material.of({_short(target.path)!r}) "
                f"lists its materials"
            )
        found  = self._require()
        engine = self._engine_of(found, create=False)
        empty  = np.empty(0, dtype=np.int64)
        if engine is None:
            return empty
        held = _held(engine, target.path)
        if held is _NOT_MEMBER:
            return empty
        if held is None:
            held = np.arange(_num_faces(target.path))
        if target.whole:
            return held
        return np.intersect1d(target.face_ids(), held)

    @classmethod
    def _of(cls, selections: list[_Selection]) -> list["Material"]:
        target = _single(selections, "Material.of takes")
        found  = []
        for name in _engines_of(target.path):
            engine = ShadingEngine(name)
            held   = _held(engine, target.path)
            if held is _NOT_MEMBER:
                continue
            if not target.whole and held is not None:
                if not np.isin(target.face_ids(), held).all():
                    continue
            spec = cls._spec_for(engine)
            if spec is not None:
                found.append(spec)
        return found

    @classmethod
    def _spec_for(cls, engine: ShadingEngine) -> "Material | None":
        """The spec of this class that names the engine's material, or
        ``None`` when the engine has none or the class does not cover it."""
        shader = engine.get_material()
        if shader is None:
            return None
        if engine.name == ShadingEngine.DEFAULT and cls in (Material, Default):
            return Default()
        if cls is Default:
            return None
        spec_cls = _BY_TYPE.get(shader.node_type, Material)
        if cls is Material:
            return spec_cls(shader.name)
        if cls.NODE_TYPE == shader.node_type:
            return cls(shader.name)
        return None

    # -- methods -- #

    def build(self) -> Node:
        """Find or create the network without assigning anything (the
        escape hatch for a look built before any geometry exists). Returns
        the material node."""
        self._verb("build")
        found = None if self._unique else self._resolve()
        if found is None:
            if self._type is None:
                raise ValueError(
                    f"no surface shader named '{self._name}' and Material({self._name!r}) "
                    f"has no type to build one; write Blinn({self._name!r}) or "
                    f"Material({self._name!r}, type='blinn')"
                )
            _gate_type(self._type)
            _check_attrs(self._attrs, node_type=self._type)
        else:
            self._engine_of(found, create=False)
            if self._update:
                _check_attrs(self._attrs, node=found.material)
        with _undo_chunk(f"rig.{self.KIND}"):
            material, _ = self._realise(found)
        return Node(material)

    def _guard_owned(self, material: str, verb: str) -> None:
        if cmds.ls(material, defaultNodes=True):
            raise RuntimeError(
                f"'{material}' is a Maya default node and cannot be {verb}"
            )
        if cmds.referenceQuery(material, isNodeReferenced=True):
            raise RuntimeError(
                f"'{material}' is referenced and cannot be {verb}; edit the source file"
            )

    def delete(self) -> None:
        """Delete the material, the engines it feeds and their materialInfo
        nodes. Members are left in no engine (green); :func:`repair`
        re-homes them. Refuses a Maya default or referenced node."""
        self._verb("delete")
        found    = self._require()
        material = found.material
        self._guard_owned(material, "deleted")
        engines = [
            name for name in _engines_fed_by(material)
            if not cmds.ls(name, defaultNodes=True)
        ]
        infos = [
            info for name in engines for info in ShadingEngine(name).get_material_info()
        ]
        with _undo_chunk(f"rig.{self.KIND}"):
            for name in [*infos, *engines, material]:
                if cmds.objExists(name):
                    cmds.delete(name)
        if cmds.objExists(material) or any(cmds.objExists(name) for name in engines):
            raise RuntimeError(f"'{material}' survived its deletion")

    def rename(self, new: str) -> None:
        """Rename the material -- and its engine when the engine follows the
        ``<mat>SG`` convention -- and point this spec at the new name."""
        self._verb("rename")
        Material(new)
        found    = self._require()
        material = found.material
        self._guard_owned(material, "renamed")
        if _find_node(new) is not None:
            raise ValueError(f"'{new}' already exists")
        engine  = self._engine_of(found, create=False)
        old     = _leaf(material)
        follows = engine is not None and _leaf(engine.name) == f"{old}SG"
        if follows and _find_node(f"{new}SG") is not None:
            raise ValueError(
                f"'{new}SG' already exists, so the engine of '{new}' could not follow "
                f"the <mat>SG convention; rename or delete '{new}SG' first"
            )
        with _undo_chunk(f"rig.{self.KIND}"):
            got = cmds.rename(material, new)
            if _leaf(got) != _leaf(new):
                raise RuntimeError(f"Maya renamed '{material}' to '{got}', not '{new}'")
            if follows:
                got_engine = cmds.rename(engine.name, f"{new}SG")
                if _leaf(got_engine) != f"{_leaf(new)}SG":
                    raise RuntimeError(
                        f"Maya renamed '{engine.name}' to '{got_engine}', not '{new}SG'"
                    )
        self._name  = got
        self._built = None

    def astype(
        self,
        to:      Any,
        *,
        strict:  bool = False,
        dry_run: bool = False,
        park:    bool = True,
        **attrs: Any,
    ) -> "Material":
        """Make this material the node type ``to`` (a type name or a typed
        class) in place and return ``self``: a free retype while the
        material does not exist yet, otherwise a scene conversion in one
        undo chunk (``rig.shade.convert``) that keeps the name, engines,
        materialInfo, ``defaultShaderList1`` slot, container, dynamic
        attributes and locks, moves the wires the target can hold and parks
        the rest on the node (``park=False`` drops and disconnects instead).
        One ``cmds.warning`` names whatever was parked or lost; ``strict``
        turns it into a ``ValueError``; ``dry_run`` writes nothing.
        ``attrs`` are validated against the target before any write and
        applied inside the chunk (on a material already of that type they
        follow the found-material rule). Refused before any write: an
        unregistered or non-surface type, a Maya default, referenced or
        ``lockNode``'d material."""
        self._verb("astype")
        convert(self, to, strict=strict, dry_run=dry_run, park=park, **attrs)
        return self


class Lambert(Material):
    NODE_TYPE = "lambert"


class Blinn(Material):
    NODE_TYPE = "blinn"


class Phong(Material):
    NODE_TYPE = "phong"


class PhongE(Material):
    NODE_TYPE = "phongE"


class SurfaceShader(Material):
    NODE_TYPE = "surfaceShader"


class StandardSurface(Material):
    NODE_TYPE = "standardSurface"


class OpenPBRSurface(Material):
    NODE_TYPE = "openPBRSurface"


class Default(Material):
    """The spec of ``initialShadingGroup``: ``cube << Default()`` reverts to
    it, ``cube << -Default()`` leaves it (green), ``cube.f[:6] >> Default()``
    queries it. It takes no name (``Default("x")`` / ``Default(None)`` are
    ``TypeError``s) and never creates, deletes or renames anything."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            raise TypeError(
                "Default() takes no name: it is initialShadingGroup. Material(None) "
                "removes from every engine; Blinn('x') names a material"
            )
        super().__init__(ShadingEngine.DEFAULT)

    def __repr__(self) -> str:
        return f"{'-' if self._remove else ''}Default()"

    def build(self) -> Node:
        return self.node

    def delete(self) -> None:
        raise TypeError("Default() is initialShadingGroup and cannot be deleted")

    def rename(self, new: str) -> None:
        raise TypeError("Default() is initialShadingGroup and cannot be renamed")

    def astype(self, to: Any, **options: Any) -> "Material":
        raise TypeError("Default() is initialShadingGroup and is never converted")


# Node type -> the spec class that builds and asserts it.
_BY_TYPE = {
    cls.NODE_TYPE: cls
    for cls in (
        Lambert,
        Blinn,
        Phong,
        PhongE,
        SurfaceShader,
        StandardSurface,
        OpenPBRSurface,
    )
}


# ---------- Conversion ---------------------------------------------------- #


def _target_type(to: Any) -> str:
    """The node type a conversion target names: a type string or a typed
    Material class."""
    if isinstance(to, type) and issubclass(to, Material):
        if to.NODE_TYPE is None:
            raise TypeError(
                f"{to.__name__} names no node type; convert to a typed class (Phong) "
                f"or a type name ('phong')"
            )
        return to.NODE_TYPE
    if isinstance(to, str) and to:
        return to
    raise TypeError(
        f"a conversion target is a node type name or a typed Material class, not "
        f"{to!r}"
    )


def convert(
    x:       Any,
    to:      Any,
    *,
    strict:  bool = False,
    dry_run: bool = False,
    park:    bool = True,
    **attrs: Any,
) -> Conversion:
    """The conversion engine: make the material ``x`` -- a spec or a name --
    the node type ``to`` and return the :class:`Conversion` report
    (``bool(report)`` is "something was parked or lost", ``str(report)`` the
    warning text). See :meth:`Material.astype` for the options; a ``Node``
    is not taken (wrap it: ``Material(node)``). A spec passed in is retyped
    in place."""
    if isinstance(x, str):
        spec = Material(x)
    elif isinstance(x, Material):
        spec = x
    else:
        raise TypeError(
            f"convert() takes a material spec or a name, not {type(x).__name__}; "
            f"wrap it: Material(node)"
        )
    spec._verb("astype")
    return shade_convert.convert(
        spec, _target_type(to), strict=strict, dry_run=dry_run, park=park, attrs=attrs
    )


# ---------- Module functions ---------------------------------------------- #


def _entries(target: _Target) -> list[tuple[ShadingEngine, Any]]:
    """``(engine, held)`` for every engine holding something of the target:
    ``held`` is ``None`` for a whole-object target the engine owns entirely,
    else the requested face ids the engine holds."""
    result = []
    for name in _engines_of(target.path):
        engine = ShadingEngine(name)
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


def materials(target: Any) -> PlugList:
    """The materials of ``target`` (a node, faces, or a list of them) as
    material ``Node``s, in connection order, each once."""
    found: list[Node] = []
    for item in _scene_targets(target):
        for engine, _ in _entries(item):
            shader = engine.get_material()
            if shader is None:
                continue
            node = Node(shader)
            if node not in found:
                found.append(node)
    return PlugList(found)


def bindings(target: Any) -> list[tuple[Node, Components]]:
    """``(material, faces)`` per engine holding faces of ``target``; an
    object-level membership is reported as the whole-kind ``f[*]``
    :class:`Components`. Read through ``MFnSet``, never through the strings
    ``cmds.sets(q=True)`` prints. Mesh only."""
    result = []
    for item in _scene_targets(target):
        if item.node_type != "mesh":
            raise TypeError(
                f"bindings are per face and '{_short(item.path)}' is a "
                f"{item.node_type}; materials({_short(item.path)!r}) lists its materials"
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


def repair(targets: Any = None) -> PlugList:
    """Re-home to ``initialShadingGroup`` every shape (or face) of
    ``targets`` -- every DAG path of the scene by default -- that no shading
    engine holds, as a deleted network or ``Material(None)`` leaves them.
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
    return PlugList(fixed)


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
