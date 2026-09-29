"""
Membership: the collections geometry components and objects belong to.

:class:`Tag` is the component-tag kind, the Maya 2025 pivot for deformers,
which read tags by name through ``input[i].componentTagExpression``. A tag
lives on one geometry node (its injection node: the shape, or the Orig once
a deformer exists), holds ONE category (vertices / CVs / points, edges or
faces: Maya reads a tag as one component type, the first one stored) and is
edited here through ``cmds.componentTag`` with an explicit
``injectionLocation`` so every write lands where Maya reads it. The six
face tags a ``polyCube`` ships are procedural (owned by ``polyCube1``) and
refuse edits; shadowing one on purpose is the ``at=`` escape hatch. A
``polyCube(ch=False)`` bakes them into the shape's own ``componentTags``
multi where the command refuses them but the plug path edits them.

``Layer`` is the display layer node class itself (``Layer is
DisplayLayer``): objects only, exclusive (a node is in one layer), the node
itself and never its subtree (children follow through the DAG), with
``defaultLayer`` read as "no layer". ``Layer('x')`` refers to a layer that
exists (never creates it), ``Layer.define('x', ...)`` finds or makes it and
``Layer.create(name='x')`` makes a new one. Membership is read from the node
side (its ``drawOverride`` input), never by enumerating layers, and a layer
never joins the active rig container.

The membership verbs are one grammar for every kind: ``<<`` adds (``-x``
removes; the kind token ``Tag()`` / ``Layer()`` purges), ``in`` / ``not in``
asks yes or no with all-members semantics (every node, every component, every
element of a list), ``>>`` reads ids (``>> Tag()`` / ``>> Layer()``
enumerates). An attribute plug on the left of ``<<`` / ``>>`` / ``of`` is
refused (the node is the member: ``sph << Tag("x")``), and stands for its
node in ``in``; component plugs are members. A component tag is untyped for
queries: faces in a vertex tag are simply not in it.

Usage::

    from rig import Node, List, Tag, Layer

    sph = Node("pSphere1")
    sph.vtx[:8] << Tag("cap")             # create 'cap' WITH vtx[0:7] at the injection node
    sph.f[:3]   << Tag("lid")             # a face tag; returns sph.f[:3]
    sph         << Tag("empty")           # node on the left: the tag itself, created empty
    sph.vtx[:3] << -Tag("cap")            # remove those vertices; the tag survives
    sph.vtx     << -Tag("cap")            # empty it (an empty tag deforms nothing)
    sph         << -Tag("cap")            # delete it (RuntimeError while a deformer references it)
    sph.vtx[:8] << Tag()                  # out of every editable vertex tag on the node
    sph         << Tag()                  # delete every editable tag on the node
    sph.tx      << Tag("cap")             # TypeError: a plug; the node is the member

    sph >> Tag("lid")                     # array([0, 1, 2])   native ids, the tag's own category
    sph.vtx[4:12] >> Tag("cap")           # array([4, 5, 6, 7])
    sph.f[:2] >> Tag("cap")               # array([], ...)     faces are not in a vertex tag
    sph.vtx[3] >> Tag() ; Tag.of(sph.vtx[3])   # [Tag('cap')]   the tags holding that vertex
    sph.vtx[:3] in Tag("cap")             # True: every one is in it
    sph in Tag("cap") ; sph in Tag("ghost")    # True ; False (no such tag)

    Tag("cap").set(sph.f[:2])             # replace; may flip the category
    Tag("cap").clear(sph)                 # empty, name survives
    Tag("cap").rename(sph, "crown", force=True)   # rewrites exact-token deformer references
    Tag("cap").delete(sph, force=True)

    cube = Node("pCube1")                 # polyCube WITH history
    cube.f[:3] << Tag("top")              # TypeError: 'top' is procedural (owned by polyCube1)
    cube.f[:3] << Tag("top", at=cube)     # an editable shadow on pCubeShape1; later edits find it

    cluster.input[0].componentTagExpression << Tag("cap")   # writes 'cap'; warns when absent upstream

    geo  = Node("arm_geo")
    geom = Layer.define("geometry")       # DisplayLayer("geometry"): found, or made (empty)
    geo << geom                           # moves geo (exclusive); returns geo
    ref  = Layer.define("ref", displayType=2, visibility=False)   # attributes set when it is made
    List([geo, Node("ctl")]) << ref       # one editDisplayLayerMembers call
    geo << -ref ; geo << Layer()          # both: back to defaultLayer
    geo in ref ; geo.tx in ref            # True / False (a plug stands for its node)
    geo >> Layer() ; Layer.of(geo)        # DisplayLayer("ref") or None ; [DisplayLayer("ref")] or []
    Layer("rig")                          # NodeNotFoundError when it does not exist: never created
    ref.visibility << False ; ref.rename("anim") ; ref.clear() ; ref.delete()
    geo.f[:3] << ref                      # TypeError: layers hold objects (Maya would store the shape)
    geo >> ref                            # TypeError: a layer has no ids; ask with geo in ref
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes import display_layer as _display_layer
from rig.nodetypes._base import _cast
from rig.nodetypes.deformer import _GLOB_TOKEN_RE, tag_references
from rig.nodetypes.display_layer import DisplayLayer
from rig.nodetypes.geometry import _TAG_NAME_RE, Geometry
from rig._internal import types as _types
from rig._internal.members import (
    _GEOMETRY_TYPES,
    _MemberSpec,
    _ndims,
    _Selection,
    _single_geometry_shape,
    Components,
    normalise,
)
from rig._internal.node import Node
from rig._internal.plug import Plug
from rig._internal.undo import _undo_chunk


__all__ = ["Components", "Layer", "Tag"]


# ---------- Tables -------------------------------------------------------- #

# Component kind of a selection -> the tag category it lives in.
_CATEGORY_OF_KIND = {"vtx": "v", "cv": "v", "pt": "v", "e": "e", "f": "f"}
_CATEGORY_WORD    = {"v": "vertex", "e": "edge", "f": "face"}
_CATEGORY_PLURAL  = {"v": "vertices", "e": "edges", "f": "faces"}

# componentTagHistory reports the category as an MFnGeometryData enum.
_CATEGORY_OF_CODE = {
    OpenMaya.MFnGeometryData.kVerts: "v",
    OpenMaya.MFnGeometryData.kEdges: "e",
    OpenMaya.MFnGeometryData.kFaces: "f",
}

# ``componentTag -queryEdit`` only answers ``modReplace`` truthfully when it
# is handed a component (any category); the whole object reads False.
_PROBE_TOKEN = {
    "mesh":         "vtx[0]",
    "nurbsCurve":   "cv[0]",
    "nurbsSurface": "cv[0][0]",
    "lattice":      "pt[0][0][0]",
}

_MISSING    = "missing"     # no tag of that name resolves on the node
_EDITABLE   = "editable"    # cmds.componentTag edits it at its home node
_BAKED      = "baked"       # the command refuses it; it lives in the home's own multi
_PROCEDURAL = "procedural"  # output of a history node; nothing edits it


def _exact_token(name: str) -> re.Pattern:
    """The regex matching ``name`` as a whole token of a tag expression."""
    return re.compile(rf"(?<![A-Za-z0-9_:*]){re.escape(name)}(?![A-Za-z0-9_:*])")


def _short(path: str) -> str:
    return path.rsplit("|", 1)[-1]


def _full_path(name: str, near: str) -> str:
    """The full DAG path of ``name`` (as componentTagHistory spells it),
    preferring the one in the history of ``near`` when the short name is
    ambiguous."""
    paths = cmds.ls(name, long=True) or []
    if len(paths) <= 1:
        return paths[0] if paths else name
    history = set(cmds.ls(cmds.listHistory(near) or [], long=True))
    for path in paths:
        if path in history:
            return path
    return paths[0]


def _rows_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A bool mask over ``a``: which of its rows (or ids) appear in ``b``."""
    if a.ndim == 1:
        return np.isin(a, b)
    keys = {tuple(row) for row in b.tolist()}
    return np.fromiter((tuple(row) in keys for row in a.tolist()), dtype=bool, count=len(a))


def _rows_in(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rows (or ids) of ``a`` that also appear in ``b``."""
    return a[_rows_mask(a, b)]


def _cth_dump(geo: Geometry) -> str:
    return "; ".join(
        f"{e['key']} @ {e['node']} (editable={e['editable']}, "
        f"procedural={e['procedural']}, final={e['final']})"
        for e in geo.get_component_tag_history() or []
    ) or "no component tags"


def _references(name: str, node: str) -> list[str]:
    """The ``componentTagExpression`` plugs whose deformer input actually
    sees the tag ``name`` held by ``node`` (a same-named tag on another
    mesh is not a reference; an input whose geometry cannot be read is,
    to be safe)."""
    found = []
    for deformer, i in tag_references(name):
        plug = f"{deformer}.input[{i}]"
        try:
            history = cmds.geometryAttrInfo(
                f"{plug}.inputGeometry", componentTagHistory=True
            ) or []
        except RuntimeError:
            history = None
        if history is None or any(
            e["key"] == name and node in (cmds.ls(e["node"], long=True) or [])
            for e in history
        ):
            found.append(f"{plug}.componentTagExpression")
    return found


# ---------- Per-node state -------------------------------------------- #


@dataclass
class _Home:
    """Where one tag name lives for one left-hand-side shape.

    ``path`` is the shape on the left, ``node`` the full path of the node
    holding the tag (or about to, when ``state`` is missing; the owner when
    procedural). ``category`` / ``empty`` describe that node's entry.
    """

    path:     str
    geo:      Geometry
    name:     str
    state:    str
    node:     str
    category: str
    empty:    bool

    @property
    def holder(self) -> Geometry:
        """The geometry node holding the tag (never a procedural owner)."""
        return _cast(self.node)

    def ids(self) -> np.ndarray:
        """What the holder's own output resolves for the tag."""
        return self.holder.get_component_tag_indices(self.name)

    def describe(self) -> str:
        return f"'{self.name}' on {_short(self.node)}"


def _final(entries: list[dict]) -> dict | None:
    for entry in entries:
        if entry["final"]:
            return entry
    return entries[-1] if entries else None


def _state_of(geo: Geometry, path: str, name: str, entry: dict, node: str) -> _Home:
    category = _CATEGORY_OF_CODE.get(entry["category"], "v")
    empty    = entry["affectCount"] == 0
    if entry["procedural"] or not entry["editable"]:
        return _Home(path, geo, name, _PROCEDURAL, node, category, empty)
    answer = cmds.componentTag(
        f"{path}.{_PROBE_TOKEN[geo.node_type]}",
        queryEdit         = True,
        tagName           = name,
        injectionLocation = node,
    )
    if answer["modReplace"]:
        state = _EDITABLE
    elif _cast(node).get_component_tag_index(name) != -1:
        state = _BAKED
    else:
        state = _PROCEDURAL
    return _Home(path, geo, name, state, node, category, empty)


def _resolve(geo: Geometry, path: str, name: str, at: str | None) -> _Home:
    """Resolve the home of ``name`` for the shape at ``path``: the entry
    that resolves (``final``), or the entry at ``at`` when given (``at`` is
    also where a new tag goes, the consent to shadow a procedural one)."""
    entries = [e for e in geo.get_component_tag_history() or [] if e["key"] == name]
    if at is not None:
        mine = _final([e for e in entries if _full_path(e["node"], path) == at])
        if mine is not None:
            return _state_of(geo, path, name, mine, at)
        final = _final(entries)
        if final is not None and final["editable"] and not final["procedural"]:
            raise TypeError(
                f"'{name}' already lives on {final['node']} and is editable there; "
                f"drop at= to edit it, or pick another name"
            )
        return _Home(path, geo, name, _MISSING, at, "", True)
    final = _final(entries)
    if final is None:
        return _Home(path, geo, name, _MISSING, geo.injection_node.long_name, "", True)
    return _state_of(geo, path, name, final, _full_path(final["node"], path))


def _resolve_at(at: Any, path: str) -> str | None:
    """The full path of the ``at=`` node: a geometry node that is the shape
    itself or upstream of it (a transform stands for its single shape)."""
    if at is None:
        return None
    node = at if isinstance(at, Node) else Node(at)
    if node.node_type not in _GEOMETRY_TYPES:
        shape = _single_geometry_shape(node)
        if shape is None:
            raise TypeError(f"at={at!r}: '{node}' is not a geometry node")
        node = shape
    full    = node.long_name
    allowed = {path} | set(cmds.ls(cmds.listHistory(path) or [], long=True))
    if full not in allowed:
        raise TypeError(
            f"at={at!r}: '{full}' is neither {path} nor in its history; a tag is "
            f"injected on the shape it is read from or a node upstream of it"
        )
    return full


class _TagGroup:
    """The selections of one shape, split into the node itself and its
    components."""

    __slots__ = ("path", "node_type", "whole", "comps")

    def __init__(self, path: str, node_type: str) -> None:
        self.path      = path
        self.node_type = node_type
        self.whole     = False
        self.comps     = []


def _group(selections: list[_Selection]) -> list[_TagGroup]:
    groups: dict[str, _TagGroup] = {}
    for selection in selections:
        group = groups.get(selection.path)
        if group is None:
            group = groups[selection.path] = _TagGroup(selection.path, selection.node_type)
        if selection.kind == "whole":
            group.whole = True
        else:
            group.comps.append(selection)
    return list(groups.values())


# ---------- Tag ----------------------------------------------------------- #


class Tag(_MemberSpec):
    """A component tag: ``Tag('cap')``, ``-Tag('cap')``, ``Tag()``.

    ``members << Tag('x')`` adds members (create-with-components when the
    tag is missing, ``modify replace`` when it is empty -- ``modify add`` on
    an empty tag is a silent no-op -- and ``modify add`` otherwise). Maya
    reads a tag as one component type, the first one stored, so faces into a
    non-empty vertex tag are a ``TypeError`` naming ``Tag('x').set(...)``
    (Maya would store them and never read them); removing faces from a vertex
    tag is a no-op (none are in it). ``node << Tag('x')`` means the tag itself
    (created empty; no-op when present) and ``node << -Tag('x')`` deletes it,
    refusing while a deformer references it unless ``force=True``.
    ``members << Tag()`` removes them from every editable tag of their
    category on that node; ``node << Tag()`` deletes every editable tag.
    ``<<`` returns the left-hand side; every write of one ``<<`` is one undo
    chunk. An attribute plug on the left of ``<<`` / ``>>`` / ``Tag.of`` is
    a ``TypeError`` (the node is the member: ``sph << Tag('x')``); only a
    deformer's ``componentTagExpression`` plug keeps its own meaning and
    receives the name.

    ``lhs >> Tag('x')`` reads the native ids of the left-hand side that are
    in the tag (``(N, 2)`` on a surface, ``(N, 3)`` on a lattice): none for
    components of another kind or a tag the node does not have (an empty
    array); a whole node reads the tag's contents in its own category (a
    missing tag is a ``ValueError`` there); ``lhs >> Tag()`` is
    ``Tag.of(lhs)``, the tags holding the left-hand side. ``lhs in
    Tag('x')`` is True when every component of the left is in the tag (a
    node on the left: the node has the tag); a tag the node does not have,
    or components of another kind, answer False; an attribute plug stands
    for its node there. ``Tag()`` is the kind token. Names are
    validated at construction: at least two characters of ``[A-Za-z0-9_:]``
    not starting with a digit (Maya stores a one-character name as ``''``).
    ``at=`` (a Node or name) is where a NEW tag is injected and the consent
    to shadow a procedural one; ``force=True`` deletes / purges through
    deformer references.

    A deformer input counts as a reference only when its input geometry can
    see the tag on this node: a tag injected downstream of a deformer (the
    visible shape after a polySmooth) is invisible to it, so the expression
    sugar warns and a delete does not refuse. Plain ``controlPoints`` plugs
    of a periodic surface write the flat ids Maya normalises, so the wrap CVs
    (``cv[0][8]`` on a 7 x 8 sphere) come back as coordinates outside the
    ``Components`` grid; read such tags through ``srf.cv[u, v]``.
    """

    KIND        = "tag"
    ACCEPTS     = frozenset({"whole", "vtx", "e", "f", "cv", "pt"})
    WANT_SHAPES = True

    def __init__(
        self,
        name:     Any  = None,
        *members: Any,
        at:       Any  = None,
        force:    bool = False,
    ) -> None:
        if members:
            raise TypeError(
                "members belong on the left: cube.vtx[[0, 1, 2]] << Tag('x') adds "
                "them, cube.vtx[[0, 1, 2]] << -Tag('x') removes them and "
                "Tag('x').set(cube.vtx[[0, 1, 2]]) replaces the tag's contents"
            )
        super().__init__(name, at=at, force=force)

    def __repr__(self) -> str:
        return "Tag()" if self._name is None else super().__repr__()

    def _validate_name(self, name: Any) -> None:
        super()._validate_name(name)
        if len(name) < 2 or not _TAG_NAME_RE.match(name):
            raise ValueError(
                f"{name!r} is not a component tag name: at least two characters of "
                f"[A-Za-z0-9_:] not starting with a digit (Maya stores a "
                f"one-character name as '' and rejects spaces and dashes)"
            )

    # -- refusals -- #

    def _kind_error(self, selection: _Selection) -> str:
        if selection.kind == "uv":
            return (
                f"UVs cannot be tagged ({selection.names[0]}): Maya returns '' and "
                f"creates nothing. Tags hold vertices / CVs / points, edges or faces"
            )
        return super()._kind_error(selection)

    def _verb(self, verb: str) -> None:
        if self._remove:
            raise TypeError(
                f"-Tag({self._name!r}).{verb}() is unassigned; call the method on "
                f"Tag({self._name!r})"
            )
        if self._name is None:
            raise TypeError(f"Tag().{verb}(): a method names one tag")

    @staticmethod
    def _check_geometry(group: _TagGroup) -> None:
        if group.node_type not in _GEOMETRY_TYPES:
            raise TypeError(
                f"'{group.path}' is a {group.node_type}, not geometry; component "
                f"tags live on mesh / nurbsSurface / nurbsCurve / lattice shapes"
            )

    @staticmethod
    def _category(group: _TagGroup) -> str | None:
        categories = sorted({_CATEGORY_OF_KIND[s.kind] for s in group.comps})
        if len(categories) > 1:
            words = " and ".join(_CATEGORY_PLURAL[c] for c in categories)
            raise TypeError(
                f"{_short(group.path)}: {words} in one '<<'; a tag holds one "
                f"category, so nothing was written. Tag them separately"
            )
        return categories[0] if categories else None

    @staticmethod
    def _refuse_procedural(home: _Home, verb: str, shadow: bool) -> TypeError:
        hint = (
            f" Choose another name, or shadow it on purpose with "
            f"Tag({home.name!r}, at={_short(home.path)!r})"
            if shadow
            else ""
        )
        return TypeError(
            f"'{home.name}' on {_short(home.path)} is PROCEDURAL (owned by "
            f"{_short(home.node)}) and cannot be {verb}.{hint}"
        )

    @staticmethod
    def _refuse_missing(home: _Home) -> ValueError:
        return ValueError(f"no component tag '{home.name}' on {_short(home.path)}")

    def _home(self, geo: Geometry, path: str) -> _Home:
        return _resolve(geo, path, self._name, _resolve_at(self._options.get("at"), path))

    # -- the string-plug sugar -- #

    def _plan_plug(self, plug: Plug) -> Any:
        # Decided by attribute name BEFORE the attribute-plug fallback: a
        # deformer's expression plug takes the tag's name, any other plug
        # stands for its node.
        mplug = plug._mplug
        if OpenMaya.MFnAttribute(mplug.attribute()).name != "componentTagExpression":
            return None
        deformer = OpenMaya.MFnDependencyNode(mplug.node()).name()
        if (
            not mplug.isChild
            or not mplug.parent().isElement
            or "geometryFilter" not in (cmds.nodeType(deformer, inherited=True) or [])
        ):
            return None
        if self._remove:
            raise TypeError(
                f"only an add-op tag can be written to {plug}; the complement of "
                f"'{self._name}' is the expression string '!{self._name}'"
            )
        if self._name is None:
            raise TypeError(
                f"Tag() has no name to write to {plug}; write Tag('x') or a "
                f"plain expression string ('*' means every point)"
            )
        index = mplug.parent().logicalIndex()
        name  = self._name

        def step() -> None:
            plug << name
            seen = cmds.geometryAttrInfo(
                f"{deformer}.input[{index}].inputGeometry", componentTagNames=True
            ) or []
            if name not in seen:
                cmds.warning(
                    f"rig.Tag: '{name}' is not a component tag on the input geometry "
                    f"of {deformer}.input[{index}] (it sees {seen}); the deformer "
                    f"moves nothing until the tag exists upstream of it"
                )

        return [step]

    # -- '<<' -- #

    def _plan(self, selections: list[_Selection]) -> list[Callable[[], None]]:
        steps = []
        for group in _group(selections):
            self._check_geometry(group)
            if group.whole and group.comps:
                raise TypeError(
                    f"{_short(group.path)} and its components in one '<<': the node "
                    f"means the tag itself and components mean members. Split them"
                )
            category = self._category(group)
            geo      = _cast(group.path)
            if self._name is None:
                steps.extend(self._plan_purge(group, geo, category))
            elif group.whole:
                steps.extend(self._plan_collection(group, geo))
            else:
                steps.extend(self._plan_members(group, geo, category))
        return steps

    def _apply(self, plan: list[Callable[[], None]]) -> None:
        for step in plan:
            step()

    def _plan_collection(self, group: _TagGroup, geo: Geometry) -> list:
        home = self._home(geo, group.path)
        if self._remove:
            return [self._delete_step(home, self._options["force"])]
        if home.state == _PROCEDURAL:
            # 'already present' would be a lie: nothing here can hold members.
            raise self._refuse_procedural(home, "created", shadow=True)
        if home.state != _MISSING:
            return []
        return [lambda: self._create(home, [group.path])]

    def _plan_members(self, group: _TagGroup, geo: Geometry, category: str) -> list:
        names = [name for s in group.comps for name in s.names]
        home  = self._home(geo, group.path)
        words = _CATEGORY_PLURAL[category]
        if self._remove:
            if home.state == _MISSING:
                raise self._refuse_missing(home)
            if home.state == _PROCEDURAL:
                raise self._refuse_procedural(home, "edited", shadow=False)
            if home.empty or home.category != category:
                # none of them is in it: an empty tag, or another kind (Maya
                # reads a tag as one component type), so nothing to remove
                return []
            return [lambda: self._remove_members(home, group.comps, names)]
        if home.state == _PROCEDURAL:
            raise self._refuse_procedural(home, "edited", shadow=True)
        if home.state == _MISSING:
            return [lambda: self._create(home, names, group.comps)]
        if home.empty:
            return [lambda: self._replace_members(home, group.comps, names, category)]
        if home.category != category:
            raise TypeError(
                f"{home.describe()} is a {_CATEGORY_WORD[home.category]} tag, and Maya "
                f"reads a tag as one component type, the first one stored: it would "
                f"store {names[0]} and never read them, so nothing was written. "
                f"Tag({home.name!r}).set(<{words}>) replaces its contents"
            )
        return [lambda: self._add_members(home, group.comps, names)]

    def _plan_purge(self, group: _TagGroup, geo: Geometry, category: str | None) -> list:
        # Only names that still resolve: a deleted baked tag leaves a dead,
        # unresolvable history entry behind that nothing can edit.
        live   = set(geo.component_tags)
        finals = {}
        for entry in geo.get_component_tag_history() or []:
            if entry["final"] and entry["key"] in live:
                finals[entry["key"]] = entry
        steps, blocked = [], []
        for key in sorted(finals):
            entry = finals[key]
            if not group.whole and (
                _CATEGORY_OF_CODE.get(entry["category"]) != category
                or entry["affectCount"] == 0
            ):
                continue
            home = _state_of(geo, group.path, key, entry, _full_path(entry["node"], group.path))
            if home.state == _PROCEDURAL:
                blocked.append(f"{key} (owned by {_short(home.node)})")
                continue
            if group.whole:
                steps.append(self._delete_step(home, self._options["force"]))
            else:
                names = [name for s in group.comps for name in s.names]
                steps.append(
                    lambda home=home, names=names: self._remove_members(
                        home, group.comps, names
                    )
                )
        if blocked:
            what = "tags" if group.whole else f"{_CATEGORY_WORD[category]} tags"
            what = f"{what} the node does not own"
            steps.append(
                lambda: cmds.warning(
                    f"rig.Tag(): {_short(group.path)}: not editable, left "
                    f"alone: {', '.join(blocked)} ({what})"
                )
            )
        return steps

    def _delete_step(self, home: _Home, force: bool) -> Callable[[], None]:
        if home.state == _MISSING:
            raise self._refuse_missing(home)
        if home.state == _PROCEDURAL:
            raise self._refuse_procedural(home, "deleted", shadow=False)
        references = _references(home.name, home.node)
        if references and not force:
            raise RuntimeError(
                f"{home.describe()} is referenced by {', '.join(references)}; a "
                f"deleted tag makes the deformer move nothing. Pass force=True "
                f"(Tag(..., force=True) or .delete(node, force=True))"
            )
        return lambda: self._delete(home)

    # -- the writes (each verifies; the command's return values lie) -- #

    def _create(
        self, home: _Home, names: list[str], comps: list[_Selection] = ()
    ) -> None:
        result = cmds.componentTag(
            *names, create=True, newTagName=home.name, injectionLocation=home.node
        )
        if result != home.name or not home.holder.has_component_tag(home.name):
            raise RuntimeError(
                f"creating {home.describe()} failed (componentTag returned "
                f"{result!r}); history: {_cth_dump(home.geo)}"
            )
        self._verify_members(home, comps, present=True)

    def _modify(self, home: _Home, mode: str, names: list[str]) -> None:
        if not cmds.componentTag(
            *names, modify=mode, tagName=home.name, injectionLocation=home.node
        ):
            raise RuntimeError(
                f"componentTag -modify {mode} on {home.describe()} returned False; "
                f"history: {_cth_dump(home.geo)}"
            )

    def _add_members(self, home: _Home, comps: list[_Selection], names: list[str]) -> None:
        if home.state == _BAKED:
            self._write_baked(home, comps, present=True)
        else:
            self._modify(home, "add", names)
        self._verify_members(home, comps, present=True)

    def _remove_members(
        self, home: _Home, comps: list[_Selection], names: list[str]
    ) -> None:
        if home.state == _BAKED:
            self._write_baked(home, comps, present=False)
        else:
            self._modify(home, "remove", names)
        self._verify_members(home, comps, present=False)

    def _replace_members(
        self, home: _Home, comps: list[_Selection], names: list[str], category: str
    ) -> None:
        if home.state == _BAKED:
            home.holder.set_component_tag_contents(
                home.name, self._requested(home, comps), category
            )
        else:
            self._modify(home, "replace", names)
        self._verify_members(home, comps, present=True, exact=True)

    def _delete(self, home: _Home) -> None:
        if home.state == _BAKED:
            home.holder.remove_component_tag(home.name)
        elif not cmds.componentTag(
            home.path, delete=True, tagName=home.name, injectionLocation=home.node
        ):
            raise RuntimeError(
                f"componentTag -delete on {home.describe()} returned False; "
                f"history: {_cth_dump(home.geo)}"
            )
        if home.holder.get_component_tag_index(home.name) != -1:
            raise RuntimeError(
                f"{home.describe()} survived its deletion; history: {_cth_dump(home.geo)}"
            )

    def _write_baked(self, home: _Home, comps: list[_Selection], present: bool) -> None:
        """Union or difference through the plug path, for a tag the command
        refuses (a baked polyCube face tag)."""
        current   = home.ids()
        requested = self._requested(home, comps)
        if present:
            merged = np.concatenate([current, requested])
            merged = np.unique(merged, axis=0) if merged.ndim > 1 else np.unique(merged)
        else:
            merged = current[~_rows_mask(current, requested)]
        home.holder.set_component_tag_contents(home.name, merged, home.category)

    @staticmethod
    def _requested(home: _Home, comps: list[_Selection]) -> np.ndarray:
        parts = []
        for selection in comps:
            if selection.flat:
                raise TypeError(
                    f"{selection.names[0]}: plain controlPoints cannot be written "
                    f"through the plug path; use {_short(home.path)}.cv[u, v]"
                )
            if selection.is_all:
                parts.append(Components(home.node, selection.kind).indices)
            else:
                parts.append(selection.indices)
        ids = np.concatenate(parts)
        return np.unique(ids, axis=0) if ids.ndim > 1 else np.unique(ids)

    def _verify_members(
        self,
        home:    _Home,
        comps:   list[_Selection],
        present: bool,
        exact:   bool             = False,
    ) -> None:
        if not comps:
            return
        ids   = home.ids()
        total = 0
        flat  = False
        for selection in comps:
            if selection.flat:
                # Flat controlPoints ids do not map onto (u, v) coordinates;
                # the command's True is the only check available.
                flat = True
                continue
            if selection.is_all:
                # The holder's own count: with at= naming a node upstream
                # of a topology change, ``vtx[*]`` tagged every point of THAT
                # node, not of the shape the selection was read from.
                count  = Components(home.node, selection.kind).count
                total += count
                ok     = len(ids) == count if present else len(ids) == 0
            else:
                inside = _rows_in(selection.indices, ids)
                total += len(selection.indices)
                ok     = len(inside) == (len(selection.indices) if present else 0)
            if not ok:
                raise RuntimeError(
                    f"{home.describe()} did not read back what was written "
                    f"({selection.names[0]}); history: {_cth_dump(home.geo)}"
                )
        if exact and not flat and len(ids) != total:
            raise RuntimeError(
                f"{home.describe()} holds {len(ids)} components after a replace of "
                f"{total}; history: {_cth_dump(home.geo)}"
            )

    # -- '>>' -- #

    @classmethod
    def _single(cls, selections: list[_Selection], verb: str) -> tuple[_TagGroup, Geometry]:
        groups = _group(selections)
        if len(groups) > 1:
            raise TypeError(
                f"{verb} one node at a time; got {', '.join(_short(g.path) for g in groups)}"
            )
        group = groups[0]
        cls._check_geometry(group)
        if group.whole and group.comps:
            raise TypeError(
                f"{_short(group.path)} and its components in one {verb}: the node "
                f"asks about the tag, components about themselves. Split them"
            )
        cls._category(group)
        return group, _cast(group.path)

    def _query(self, selections: list[_Selection]) -> np.ndarray:
        if self._options.get("at") is not None:
            raise TypeError(
                "at= places a tag; a query reads the tag the geometry resolves. "
                "Drop at= from the '>>' spec"
            )
        group, geo = self._single(selections, "'>>' queries")
        name       = self._name
        if not geo.has_component_tag(name):
            if group.whole:
                raise ValueError(f"no component tag '{name}' on {_short(group.path)}")
            return _no_ids(group)   # components: none of them is in it
        ids = geo.get_component_tag_indices(name)
        if group.whole:
            return ids
        return _rows_of(group, geo, name, ids)

    def _contains(self, selections: list[_Selection]) -> bool:
        if self._options.get("at") is not None:
            raise TypeError(
                "at= places a tag; 'in' reads the tag the geometry resolves. "
                "Drop at= from the 'in' spec"
            )
        name = self._name
        for group in _group(selections):
            self._check_geometry(group)
            geo = _cast(group.path)
            if not geo.has_component_tag(name):
                return False
            if not group.comps:
                continue   # a node on the left: it has the tag
            ids = geo.get_component_tag_indices(name)
            if not ids.size:
                return False
            # untyped for queries: components of another kind are not in it
            category = geo.get_component_tag_category(name)
            if any(_CATEGORY_OF_KIND[s.kind] != category for s in group.comps):
                return False
            if not _holds_all(group, name, ids):
                return False
        return True

    @classmethod
    def _of(cls, selections: list[_Selection]) -> list["Tag"]:
        group, geo = cls._single(selections, "Tag.of takes")
        names      = geo.component_tags
        if group.whole:
            return [cls._unchecked(name) for name in names]
        category = _CATEGORY_OF_KIND[group.comps[0].kind]
        found    = []
        for name in names:
            if geo.get_component_tag_category(name) != category:
                continue
            ids = geo.get_component_tag_indices(name)
            if ids.size and _holds_all(group, name, ids):
                found.append(cls._unchecked(name))
        return found

    # -- methods -- #

    def set(self, members: Any) -> None:
        """Replace the tag's contents with ``members`` on their node
        (create-with-components when missing); the category may flip."""
        self._verb("set")
        selections = normalise(members, want_shapes=True)
        self._check_kinds(selections)
        steps = []
        for group in _group(selections):
            self._check_geometry(group)
            if group.whole:
                raise TypeError(
                    f"Tag({self._name!r}).set() takes members ({_short(group.path)}.vtx "
                    f"/ .f[...] / .e[...]); Tag({self._name!r}).clear(node) empties it"
                )
            category = self._category(group)
            geo      = _cast(group.path)
            home     = self._home(geo, group.path)
            names    = [name for s in group.comps for name in s.names]
            if home.state == _PROCEDURAL:
                raise self._refuse_procedural(home, "edited", shadow=True)
            if home.state == _MISSING:
                steps.append(lambda h=home, n=names, c=group.comps: self._create(h, n, c))
            else:
                steps.append(
                    lambda h=home, n=names, c=group.comps, k=category:
                        self._replace_members(h, c, n, k)
                )
        with _undo_chunk(f"rig.{self.KIND}"):
            for step in steps:
                step()

    def clear(self, node: Any) -> None:
        """Empty the tag on ``node``; the name survives (an empty tag deforms
        nothing and silences a deformer's 'Missing componentTags')."""
        self._verb("clear")
        steps = []
        for home in self._homes(node, "clear"):
            if home.state == _MISSING:
                raise self._refuse_missing(home)
            if home.state == _PROCEDURAL:
                raise self._refuse_procedural(home, "cleared", shadow=False)
            steps.append(lambda h=home: self._clear(h))
        with _undo_chunk(f"rig.{self.KIND}"):
            for step in steps:
                step()

    def delete(self, node: Any, force: bool = False) -> None:
        """Delete the tag from ``node``; refuses while a deformer expression
        references it unless ``force``."""
        self._verb("delete")
        force = force or self._options["force"]
        steps = [self._delete_step(home, force) for home in self._homes(node, "delete")]
        with _undo_chunk(f"rig.{self.KIND}"):
            for step in steps:
                step()

    def rename(self, node: Any, new: str, force: bool = False) -> None:
        """Rename the tag on ``node``. Refuses while a deformer expression
        references it unless ``force``, which rewrites exact-token references
        (``cap``, ``!cap``, ``cap + lid``); a glob reference (``ca*``) refuses
        even then."""
        self._verb("rename")
        Tag(new)
        force = force or self._options["force"]
        steps = []
        for home in self._homes(node, "rename"):
            if home.state == _MISSING:
                raise self._refuse_missing(home)
            if home.state == _PROCEDURAL:
                raise self._refuse_procedural(home, "renamed", shadow=False)
            if home.geo.has_component_tag(new):
                raise ValueError(f"'{new}' already exists on {_short(home.path)}")
            references = _references(home.name, home.node)
            if references and not force:
                raise RuntimeError(
                    f"{home.describe()} is referenced by {', '.join(references)}; "
                    f"pass force=True to rewrite the references along with the name"
                )
            rewrites = []
            for plug in references:
                expression = cmds.getAttr(plug) or ""
                rewritten  = _exact_token(home.name).sub(new, expression)
                if any(
                    token != "*" and fnmatch.fnmatchcase(home.name, token)
                    for token in _GLOB_TOKEN_RE.findall(rewritten)
                ):
                    raise RuntimeError(
                        f"{plug} references '{home.name}' through a glob "
                        f"({expression!r}); rewrite that expression by hand first"
                    )
                rewrites.append((plug, rewritten))
            steps.append(lambda h=home, r=rewrites: self._rename(h, new, r))
        with _undo_chunk(f"rig.{self.KIND}"):
            for step in steps:
                step()

    def _homes(self, node: Any, verb: str) -> list[_Home]:
        selections = normalise(node, want_shapes=True)
        homes      = []
        for group in _group(selections):
            self._check_geometry(group)
            if group.comps:
                raise TypeError(
                    f"Tag({self._name!r}).{verb}() takes the node, not components "
                    f"({group.comps[0].names[0]}); members << -Tag({self._name!r}) "
                    f"removes some"
                )
            homes.append(self._home(_cast(group.path), group.path))
        return homes

    def _clear(self, home: _Home) -> None:
        if home.state == _BAKED:
            home.holder.set_component_tag_contents(home.name, [], home.category)
        elif not cmds.componentTag(
            home.path, modify="clear", tagName=home.name, injectionLocation=home.node
        ):
            raise RuntimeError(
                f"componentTag -modify clear on {home.describe()} returned False; "
                f"history: {_cth_dump(home.geo)}"
            )
        if home.ids().size:
            raise RuntimeError(
                f"{home.describe()} did not read back empty; history: {_cth_dump(home.geo)}"
            )

    def _rename(self, home: _Home, new: str, rewrites: list[tuple[str, str]]) -> None:
        if home.state == _BAKED:
            home.holder.rename_component_tag(home.name, new)
        elif not cmds.componentTag(
            home.path, rename=True, tagName=home.name, newTagName=new,
            injectionLocation=home.node,
        ):
            raise RuntimeError(
                f"componentTag -rename on {home.describe()} returned False; "
                f"history: {_cth_dump(home.geo)}"
            )
        if not home.holder.has_component_tag(new):
            raise RuntimeError(
                f"'{new}' did not appear on {_short(home.node)} after renaming "
                f"{home.describe()}; history: {_cth_dump(home.geo)}"
            )
        for plug, expression in rewrites:
            cmds.setAttr(plug, expression, type="string")


def _no_ids(group: _TagGroup) -> np.ndarray:
    """The empty id array of the group's components (their dtype and shape:
    ``(0,)``, or ``(0, 2)`` / ``(0, 3)`` for surface / lattice coordinates)."""
    selection = group.comps[0]
    if selection.indices is not None:
        return selection.indices[:0]
    ndims = _ndims(selection.kind, group.node_type)
    return np.empty((0,) if ndims == 1 else (0, ndims), dtype=int)


def _rows_of(group: _TagGroup, geo: Geometry, name: str, ids: np.ndarray) -> np.ndarray:
    """The ids of the group's components that are in the tag ``name``: none
    when the tag is empty or of another kind (Maya reads a tag as one
    component type, so faces are never in a vertex tag)."""
    category = _CATEGORY_OF_KIND[group.comps[0].kind]
    if ids.size == 0:
        return ids
    if geo.get_component_tag_category(name) != category:
        return _no_ids(group)
    parts = []
    for selection in group.comps:
        if selection.flat:
            raise TypeError(
                f"{selection.names[0]}: plain controlPoints do not index a surface / "
                f"lattice tag; use {_short(group.path)}.cv[u, v] / .pt[s, t, u]"
            )
        if selection.is_all:
            return ids
        parts.append(_rows_in(selection.indices, ids))
    found = np.concatenate(parts)
    return np.unique(found, axis=0) if found.ndim > 1 else np.unique(found)


def _holds_all(group: _TagGroup, name: str, ids: np.ndarray) -> bool:
    """Whether the tag holding ``ids`` holds every component of the group."""
    for selection in group.comps:
        if selection.flat:
            raise TypeError(
                f"{selection.names[0]}: plain controlPoints do not index a surface / "
                f"lattice tag; use {_short(group.path)}.cv[u, v] / .pt[s, t, u]"
            )
        if selection.is_all:
            if len(ids) != Components(group.path, selection.kind).count:
                return False
        elif len(_rows_in(selection.indices, ids)) != len(selection.indices):
            return False
    return True


# ---------- Layer --------------------------------------------------------- #


def _holding_layer(path: str) -> DisplayLayer | None:
    """The display layer holding the node at ``path``, read from the node's
    own ``drawOverride`` input; ``None`` in ``defaultLayer``."""
    return DisplayLayer.for_node(path)


def _component_message(selection: _Selection) -> str:
    return (
        f"layers hold objects, not components ({selection.names[0]}): Maya would "
        f"silently store the shape. Use the node itself"
    )


def _layer_targets(selections: list[_Selection]) -> list[str]:
    """The DAG paths of a layer left-hand side: every selection is a whole
    DAG node (a component is refused before Maya stores its shape, a DG
    node before Maya errors on it)."""
    paths = []
    for selection in selections:
        if selection.kind != "whole":
            raise TypeError(_component_message(selection))
        if not selection.path.startswith("|"):
            raise TypeError(
                f"'{selection.path}' is a {selection.node_type}, not a DAG object; "
                f"layers hold DAG objects (transforms, shapes, joints, ...)"
            )
        paths.append(selection.path)
    return paths


class _LayerMember(_MemberSpec):
    """The membership of display layers (private): what a layer node, its
    removal token and the kind token run for ``<<`` / ``>>`` / ``in`` /
    ``of``. Built by :class:`DisplayLayer` (``layer._member()``, ``-layer``,
    ``DisplayLayer()`` / ``Layer()``), never by a user.

    Layers hold objects, exclusively: ``node << layer`` moves the node itself
    into the layer (never its subtree: children draw with the parent's
    override through the DAG without joining), ``node << -layer`` moves it
    back to ``defaultLayer`` (a no-op when it is in another layer), ``node <<
    Layer()`` is ``defaultLayer`` too; a component on the left is a
    ``TypeError`` (Maya would silently store the shape), a DG node a
    ``TypeError`` (layers hold DAG objects), an attribute plug a
    ``TypeError`` (the node is the member). A ``List`` of nodes is one
    ``editDisplayLayerMembers`` call; every write of one ``<<`` is one undo
    chunk.

    ``defaultLayer`` reads as "no layer": ``node in layer`` is True when the
    node is in it (every node of a list: all-members; ``node in
    Layer('defaultLayer')`` when it is in no layer), ``node >> Layer()`` the
    layer holding the node or ``None``, ``Layer.of(node)`` ``[layer]`` or
    ``[]``. A layer holds whole objects and has no ids: ``node >> layer`` is
    a ``TypeError`` naming ``in``.
    """

    KIND        = "layer"
    ACCEPTS     = frozenset({"whole"})
    WANT_SHAPES = False
    EXCLUSIVE   = True
    HAS_IDS     = False

    def __init__(self, layer: DisplayLayer | None = None, remove: bool = False) -> None:
        # the layer node (None: the kind token); its name is read at each use,
        # so a renamed layer is followed and a deleted one raises
        self._layer   = layer
        self._remove  = remove
        self._options = {}

    @property
    def _name(self) -> str | None:
        return None if self._layer is None else self._layer.name

    @property
    def purges(self) -> bool:
        return self._layer is None

    @classmethod
    def _kind_name(cls) -> str:
        return "Layer"

    def __neg__(self) -> "_LayerMember":
        if self._layer is None:
            raise TypeError(
                "-DisplayLayer() is a double negative: DisplayLayer() (Layer()) already "
                "moves the members to defaultLayer, out of every layer"
            )
        if self._remove:
            raise TypeError(f"-{self!r}: a removal cannot be negated again")
        return _LayerMember(self._layer, remove=True)

    def __invert__(self) -> Any:
        raise TypeError(
            f"~{self!r} is unassigned; -layer removes members and Layer() removes "
            f"them from every layer"
        )

    def __repr__(self) -> str:
        if self._layer is None:
            return "DisplayLayer()"
        return f"{'-' if self._remove else ''}{self._layer!r}"

    def __getattr__(self, name: str) -> Any:
        # a token is no layer: its methods and plugs are the layer node's
        if name.startswith("_"):
            raise AttributeError(name)
        what = "the kind token (every layer)" if self._layer is None else "a removal token"
        raise AttributeError(
            f"{self!r} is {what}, not a layer; {name} is the layer node's: "
            f"Layer('x').{name}"
        )

    @staticmethod
    def _single(selections: list[_Selection], verb: str) -> str:
        paths = _layer_targets(selections)
        if len(paths) > 1:
            raise TypeError(
                f"{verb} one node at a time; got {', '.join(_short(p) for p in paths)}"
            )
        return paths[0]

    # -- refusals -- #

    def _kind_error(self, selection: _Selection) -> str:
        return _component_message(selection)

    # -- '<<' -- #

    def _plan(self, selections: list[_Selection]) -> list[Callable[[], None]]:
        paths = _layer_targets(selections)
        if self._layer is None:
            return [lambda: self._move(None, paths)]
        # a deleted layer raises here, before any write; defaultLayer is no layer
        layer = None if self._layer.is_default else self._layer
        if not self._remove:
            return [lambda: self._move(layer, paths)]
        if layer is None:
            raise TypeError(
                "-DisplayLayer(\"defaultLayer\") is contradictory: defaultLayer is the "
                "layer of no layer, so nothing leaves it; node << layer moves the "
                "node into a layer"
            )
        mine = [path for path in paths if _holding_layer(path) == layer]
        return [lambda: self._move(None, mine)] if mine else []

    def _apply(self, plan: list[Callable[[], None]]) -> None:
        for step in plan:
            step()

    def _move(self, layer: DisplayLayer | None, paths: list[str]) -> None:
        """One ``editDisplayLayerMembers`` for every path into ``layer``
        (None: ``defaultLayer``, by its absolute name), then read each
        membership back from the node side."""
        if not paths:
            return
        target = _display_layer._DEFAULT_ABSOLUTE if layer is None else layer.name
        cmds.editDisplayLayerMembers(target, *paths, noRecurse=True)
        for path in paths:
            if _holding_layer(path) != layer:
                raise RuntimeError(
                    f"'{path}' did not land in {target}; its drawOverride reads "
                    f"{_holding_layer(path)}"
                )

    # -- '>>' and 'in' -- #

    def _query(self, selections: list[_Selection]) -> Any:
        path = _layer_targets(selections)[0]
        raise TypeError(
            f"a layer holds whole objects and has no ids; ask with "
            f"{_short(path)} in {self!r}"
        )

    def _contains(self, selections: list[_Selection]) -> bool:
        paths    = _layer_targets(selections)
        expected = None if self._layer.is_default else self._layer
        return all(_holding_layer(path) == expected for path in paths)

    @classmethod
    def _of(cls, selections: list[_Selection]) -> list[DisplayLayer]:
        layer = _holding_layer(cls._single(selections, "Layer.of takes"))
        return [layer] if layer is not None else []


# ``rig.Layer``: the display layer node class itself (``Layer is DisplayLayer``)
Layer = DisplayLayer

# the membership right-hand sides of ``<<`` / ``>>`` (D31: ``_internal.types``
# and ``nodetypes`` cannot import this module; it binds them when it loads)
_types._MEMBERSHIP           = (_MemberSpec, DisplayLayer)
_display_layer._LAYER_MEMBER = _LayerMember
