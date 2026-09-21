"""
Component selections and the left-hand-side normaliser of the membership
grammar.

:class:`Components` is the opaque carrier for geometry components. It is the
type of ``cube.f`` / ``cube.e`` (faces and edges have no plug to wrap), the
explicit index carrier ``Components(node, "vtx", ids)`` for every point kind
(a numpy array in, no ``PlugList`` built) and the reading of a Maya component
string, ``Components("pCube1.f[0:3]")``. It is NOT a ``str`` subclass and has
NO ``__len__`` / ``__iter__``: ``PlugList`` passes it through untouched and
the broadcast engine treats it as a scalar. A selection is a set -- indices
are stored sorted and unique -- and ``indices`` / ``count`` resolve live from
the shape while the selection is the whole kind (``is_all``).

:func:`normalise` folds any left-hand side of a membership operator -- a
``Node``, a component ``Plug`` (``vtx[i]``), a ``ComponentPlug`` (``cv[u, v]``),
a bare ``controlPoints`` / ``uvpt`` handle, an attribute ``Plug`` (which
stands for its node: ``cube.tx`` is ``cube``), a :class:`Components`, or any
nesting of those in a ``PlugList`` / list / tuple -- into one
:class:`_Selection` per ``(path, kind)`` carrying the compact shape-scoped
tokens ``maya.cmds`` accepts. Nothing is ever dropped: a number, ``None`` or
a raw string on the left is a ``TypeError`` naming the element.

:class:`_MemberSpec` is the protocol base of the collection specs (``Tag``,
the materials, ``Layer``): the constructor contract, ``-spec``, the ``~spec``
refusal, and the ``inject`` / ``query`` / ``of`` verbs that normalise the
left-hand side, validate it BEFORE any scene write and run the kind's writes
inside one undo chunk. ``<<`` with a collection spec returns the left-hand
side unchanged (the next collection goes next); ``>>`` returns a plain value,
and ``>> Spec()`` enumerates the collections holding the left-hand side.

Usage::

    from rig import Node, PlugList
    from rig._internal.members import Components, normalise

    cube = Node("pCube1")                      # transform with ONE mesh shape
    cube.f                                     # Components("|pCube1|pCubeShape1.f[*]")
    cube.f[:3].names                           # ['|pCube1|pCubeShape1.f[0:2]']
    cube.e[[0, 4]] >> None                     # array([0, 4])
    Components(cube, "vtx", [0, 1, 2])         # explicit carrier, zero plugs built
    Components("pCube1.f[0:3]")                # from a Maya component string

    normalise(PlugList([cube.vtx[:8], cube.f[:3]]), want_shapes=True)
    # [_Selection(kind='vtx', tokens=('vtx[0:7]',), ...),
    #  _Selection(kind='f',   tokens=('f[0:2]',),   ...)]

``Node`` / ``Plug`` are imported at module top. That is legal only because
node.py / plug.py / list.py reach this module lazily through
:mod:`rig._internal.types`; it must never be exported from ``rig.spec``,
which plug.py imports at module load.
"""

from __future__ import annotations

import copy
import itertools
import numbers
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig.maya.node_name import iter_component_tokens
from rig.maya.nodetypes.dag_node import DAGNode
from rig._internal.node import Node
from rig._internal.plug import ComponentPlug, Plug
from rig._internal.undo import _undo_chunk


# ---------- Geometry / kind tables -------------------------------------- #

# Shape node types that carry components.
_GEOMETRY_TYPES = frozenset({"mesh", "nurbsSurface", "nurbsCurve", "lattice"})

# Component kind -> the geometry types it lives on. ``f`` / ``e`` are the
# kinds ``Node.__getattr__`` builds; the point kinds come from the explicit
# constructor, the string form and the normaliser.
_KIND_GEOMETRY = {
    "f":   frozenset({"mesh"}),
    "e":   frozenset({"mesh"}),
    "vtx": frozenset({"mesh"}),
    "uv":  frozenset({"mesh"}),
    "cv":  frozenset({"nurbsCurve", "nurbsSurface"}),
    "pt":  frozenset({"lattice"}),
}

# Alternative spellings the constructor and the string parser accept.
_KIND_SYNONYMS = {"map": "uv", "pnts": "vtx"}

# The point kind of each geometry type (what a bare ``controlPoints`` handle
# means) and the token prefix a kind renders with (``uv`` selects ``map[i]``).
_POINT_KIND   = {
    "mesh":         "vtx",
    "nurbsCurve":   "cv",
    "nurbsSurface": "cv",
    "lattice":      "pt",
}
_TOKEN_PREFIX = {"uv": "map"}

# (kind, geometry) pairs whose indices are coordinates, not flat ids.
_NDIMS = {("cv", "nurbsSurface"): 2, ("pt", "lattice"): 3}

# How many compact tokens ``repr`` shows before eliding.
_REPR_TOKENS = 8

_COMPONENT_STRING = re.compile(r"^([A-Za-z]+)((?:\[(?:\*|-?\d+(?::-?\d+)?)\])+)$")
_BRACKET          = re.compile(r"\[([^\]]*)\]")


def _ndims(kind: str, node_type: str) -> int:
    return _NDIMS.get((kind, node_type), 1)


def _join(node_types: Iterable[str]) -> str:
    return "/".join(sorted(node_types))


def _own_shapes(dg: DAGNode, node_types: Iterable[str]) -> list[str]:
    """Full paths of the OWN non-intermediate shapes of ``dg`` whose type is
    in ``node_types`` (never the subtree)."""
    return (
        cmds.listRelatives(
            dg.long_name,
            shapes         = True,
            noIntermediate = True,
            fullPath       = True,
            type           = sorted(node_types),
        )
        or []
    )


def _axis_sizes(shape: DAGNode, kind: str) -> tuple[int, ...]:
    """Per-axis count of DISTINCT components of ``kind`` on ``shape``, read
    live from the API (never ``cmds.polyEvaluate``).

    One entry for the 1-D kinds, ``(u, v)`` for surface CVs with the periodic
    wraps excluded (the same rule as :meth:`ComponentPlug._axis_sizes`) and
    ``(s, t, u)`` for lattice points.
    """
    node_type = shape.node_type
    if kind == "f":
        return (shape.num_polygons,)
    if kind == "e":
        return (shape.fn_set.numEdges,)
    if kind == "uv":
        return (shape.fn_set.numUVs(),)
    if node_type == "nurbsSurface":
        fn       = shape.fn_set
        periodic = OpenMaya.MFnNurbsSurface.kPeriodic
        u        = fn.numCVsInU - (fn.degreeInU if fn.formInU == periodic else 0)
        v        = fn.numCVsInV - (fn.degreeInV if fn.formInV == periodic else 0)
        return (u, v)
    if node_type == "lattice":
        name = shape.long_name
        return tuple(cmds.getAttr(f"{name}.{axis}Divisions") for axis in "stu")
    return (shape.num_weight_points,)


def _grid(sizes: tuple[int, ...]) -> np.ndarray:
    """Every coordinate of an N-D component grid, ``(prod(sizes), N)``,
    in row-major (lexicographic) order."""
    return np.indices(sizes).reshape(len(sizes), -1).T


def _as_ids(indices: Any, sizes: tuple[int, ...], what: str) -> np.ndarray:
    """Validate ``indices`` against ``sizes`` and return them as a sorted,
    unique int array: ``(N,)`` for a 1-D kind, ``(N, ndims)`` otherwise.
    Negative indices wrap; anything else out of range is an ``IndexError``;
    bools and non-integers are a ``TypeError``."""
    ndims = len(sizes)
    ids   = np.asarray(indices)
    if ids.dtype == bool:
        raise TypeError(f"{what}: a bool is not a component index")
    if ids.size == 0:
        return np.empty((0,) if ndims == 1 else (0, ndims), dtype=np.int64)
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"{what}: component indices must be integers, got {ids.dtype}")
    if ndims == 1:
        if ids.ndim != 1:
            raise TypeError(
                f"{what}: expected a flat sequence of ids, got shape {ids.shape}"
            )
    elif ids.ndim != 2 or ids.shape[1] != ndims:
        raise TypeError(
            f"{what}: expected an (N, {ndims}) array of coordinates, "
            f"got shape {ids.shape}"
        )
    bounds = np.asarray(sizes)
    ids    = np.where(ids < 0, ids + bounds, ids)
    if (ids < 0).any() or (ids >= bounds).any():
        raise IndexError(f"{what}: index out of range for {sizes}")
    return np.unique(ids, axis=0) if ndims > 1 else np.unique(ids)


def _render_tokens(kind: str, ids: np.ndarray, flat: bool = False) -> tuple[str, ...]:
    """Shape-scoped compact tokens for sorted-unique ``ids``: ``f[0:2]`` /
    ``f[5]`` for a 1-D kind, ``cv[u][v0:v1]`` / ``pt[s][t][u0:u1]`` for
    coordinates (grouped on the leading axes, ranged on the last) and
    ``controlPoints[a:b]`` when ``flat``."""
    prefix = "controlPoints" if flat else _TOKEN_PREFIX.get(kind, kind)
    return tuple(iter_component_tokens(prefix, ids))


def _resolve_shape(obj: Any, kind: str) -> Node:
    """The geometry shape ``Node`` that ``kind`` components of ``obj`` live on:
    ``obj`` itself when it is such a shape, its single own non-intermediate
    shape of the right type when it is a transform, a ``TypeError`` otherwise."""
    node = obj if isinstance(obj, Node) else Node(obj)
    dg   = node._dg_node
    if not isinstance(dg, DAGNode):
        raise TypeError(
            f"'{node}' is not a DAG node; components live on geometry shapes"
        )
    node_type = dg.node_type
    wanted    = _KIND_GEOMETRY[kind]
    if node_type in wanted:
        return node
    if node_type in _GEOMETRY_TYPES:
        raise TypeError(
            f"'{kind}' components live on {_join(wanted)} shapes; "
            f"'{node}' is a {node_type}"
        )
    shapes = _own_shapes(dg, wanted)
    if len(shapes) == 1:
        return Node(shapes[0])
    if not shapes:
        raise TypeError(
            f"'{node}' has no {_join(wanted)} shape to take '{kind}' components from"
        )
    raise TypeError(
        f"'{node}' has {len(shapes)} {_join(wanted)} shapes ({', '.join(shapes)}); "
        f"use Node('<shape>')"
    )


def _parse_component_string(text: str) -> tuple[str, list[tuple[int, int] | None]]:
    """Split ``'f[0:3]'`` / ``'cv[1][2:4]'`` / ``'vtx[*]'`` into its canonical
    kind and one ``(lo, hi)`` range per bracket (``None`` for ``*``)."""
    match = _COMPONENT_STRING.match(text)
    if not match:
        raise TypeError(
            f"{text!r} is not a component string; expected '<kind>[i]', "
            f"'<kind>[a:b]' or '<kind>[*]' with kind in {sorted(_KIND_GEOMETRY)}"
        )
    spelled = match.group(1)
    kind    = _KIND_SYNONYMS.get(spelled, spelled)
    if kind not in _KIND_GEOMETRY:
        raise TypeError(
            f"{spelled!r} is not a component kind; "
            f"expected one of {sorted(_KIND_GEOMETRY)}"
        )
    ranges = []
    for body in _BRACKET.findall(match.group(2)):
        if body == "*":
            ranges.append(None)
        else:
            lo, _, hi = body.partition(":")
            ranges.append((int(lo), int(hi) if hi else int(lo)))
    return kind, ranges


# ---------- Components ---------------------------------------------------- #


class Components:
    """A set of components of one kind on one geometry shape.

    Built three ways::

        cube.f / cube.e / cube.f[:3] / cube.e[[0, 4]]   # Node.__getattr__ fallback
        Components(node, "vtx", ids)                    # explicit index carrier
        Components("pCube1.f[0:3]")                     # Maya component string

    ``node`` may be the shape or a transform with exactly one shape of the
    right type. Kinds are ``f`` / ``e`` / ``vtx`` / ``uv`` (mesh), ``cv``
    (curve: flat ids; surface: ``(N, 2)`` coordinates) and ``pt`` (lattice:
    ``(N, 3)`` coordinates). ``indices=None`` means the whole kind
    (:attr:`is_all`): nothing is materialised until :attr:`indices` /
    :attr:`count` are read, which resolve live from the shape.

    Indexing is positional and numpy-like -- ``c[2]``, ``c[:8]``, ``c[[0, 4]]``,
    ``c[np.array(...)]`` -- and returns a new :class:`Components`; slices
    clamp, out-of-range ints and sequences raise ``IndexError``, a bool key
    is a ``TypeError``. On the whole-kind handle positions ARE the native ids.

    ``c >> None`` reads the native ids (or coordinates) as an ndarray; ``<<``
    and ``>>`` with a collection spec are the membership verbs (a later
    step). There is deliberately no ``__len__`` / ``__iter__`` and no
    arithmetic: :class:`PlugList` keeps a ``Components`` element opaque and
    broadcasts it as a scalar.
    """

    __slots__ = ("_shape", "_kind", "_indices")

    # ``__getitem__`` alone would let Python's legacy protocol iterate a
    # selection one component at a time; ``None`` opts out, so ``iter(c)``
    # and ``PlugList(c)`` refuse instead of silently expanding it.
    __iter__ = None

    def __init__(
        self,
        node_or_string: Any,
        kind:           str | None = None,
        indices:        Any        = None,
    ) -> None:
        sizes = None
        if isinstance(node_or_string, str) and "." in node_or_string:
            if kind is not None or indices is not None:
                raise TypeError(
                    "Components('<node>.<kind>[...]') carries its kind and indices "
                    "in the string; pass a node to give them separately"
                )
            node_part, comp_part = node_or_string.split(".", 1)
            kind, ranges         = _parse_component_string(comp_part)
            shape = _resolve_shape(node_part, kind)
            sizes = _axis_sizes(shape._dg_node, kind)
            if len(ranges) != len(sizes):
                raise TypeError(
                    f"{comp_part!r}: '{kind}' on a {shape._dg_node.node_type} "
                    f"takes {len(sizes)} indices, got {len(ranges)}"
                )
            if any(r is not None for r in ranges):
                axes = [
                    range(size) if r is None else range(r[0], r[1] + 1)
                    for r, size in zip(ranges, sizes)
                ]
                if len(sizes) == 1:
                    indices = np.arange(axes[0].start, axes[0].stop)
                else:
                    indices = np.array(list(itertools.product(*axes)), dtype=np.int64)
        else:
            if kind is None:
                raise TypeError(
                    "Components(node) needs a kind: 'f', 'e', 'vtx', 'cv', 'pt' or 'uv'"
                )
            kind = _KIND_SYNONYMS.get(kind, kind)
            if kind not in _KIND_GEOMETRY:
                raise TypeError(
                    f"{kind!r} is not a component kind; "
                    f"expected one of {sorted(_KIND_GEOMETRY)}"
                )
            shape = _resolve_shape(node_or_string, kind)
        if indices is not None:
            if sizes is None:
                sizes = _axis_sizes(shape._dg_node, kind)
            indices                 = _as_ids(indices, sizes, f"Components({shape}, {kind!r})")
            indices.flags.writeable = False
        self._shape   = shape
        self._kind    = kind
        self._indices = indices

    @classmethod
    def _from_ids(cls, shape: Node, kind: str, ids: np.ndarray) -> "Components":
        """A selection on the same shape from already-validated ids (sorted
        and deduplicated here; no range check)."""
        ids                 = np.unique(ids, axis=0) if ids.ndim > 1 else np.unique(ids)
        ids.flags.writeable = False
        obj                 = object.__new__(cls)
        obj._shape          = shape
        obj._kind           = kind
        obj._indices        = ids
        return obj

    # -- identity -- #

    @property
    def shape(self) -> Node:
        """The geometry shape ``Node`` (DAG-path backed)."""
        return self._shape

    @property
    def kind(self) -> str:
        """``'f'`` / ``'e'`` / ``'vtx'`` / ``'cv'`` / ``'pt'`` / ``'uv'``."""
        return self._kind

    @property
    def geometry(self) -> str:
        """The shape's node type (``'mesh'``, ``'nurbsSurface'``, ...)."""
        return self._shape._dg_node.node_type

    @property
    def is_all(self) -> bool:
        """``True`` for the whole-kind handle (``cube.f``, ``cube.f[:]``)."""
        return self._indices is None

    @property
    def _path(self) -> str:
        return self._shape._dg_node.long_name

    @property
    def _sizes(self) -> tuple[int, ...]:
        return _axis_sizes(self._shape._dg_node, self._kind)

    # -- contents -- #

    @property
    def indices(self) -> np.ndarray:
        """The native ids, ``(N,)`` -- or coordinates, ``(N, 2)`` on a surface
        and ``(N, 3)`` on a lattice -- sorted and unique. Resolved live from
        the shape when :attr:`is_all`. Read-only."""
        if self._indices is not None:
            return self._indices
        sizes               = self._sizes
        ids                 = np.arange(sizes[0]) if len(sizes) == 1 else _grid(sizes)
        ids.flags.writeable = False
        return ids

    @property
    def count(self) -> int:
        """How many components the selection holds."""
        if self._indices is not None:
            return len(self._indices)
        return int(np.prod(self._sizes))

    @property
    def names(self) -> list[str]:
        """Full-DAG-path compact component strings ``maya.cmds`` accepts:
        ``['|pCube1|pCubeShape1.f[0:2]', '|pCube1|pCubeShape1.f[5]']``, or
        ``['|pCube1|pCubeShape1.f[*]']`` for the whole kind."""
        path = self._path
        if self._indices is None:
            return [f"{path}.{_TOKEN_PREFIX.get(self._kind, self._kind)}[*]"]
        tokens = _render_tokens(self._kind, self._indices)
        return [f"{path}.{token}" for token in tokens]

    # -- indexing -- #

    def __getitem__(self, key: Any) -> "Components":
        if isinstance(key, bool):
            raise TypeError("a bool is not a component index")
        if isinstance(key, numbers.Integral):
            count = self.count
            index = key + count if key < 0 else key
            if not 0 <= index < count:
                raise IndexError(
                    f"{self!r}: index {key} out of range for {count} components"
                )
            return self._select(np.array([index]))
        if isinstance(key, slice):
            count             = self.count
            start, stop, step = key.indices(count)
            if self._indices is None and (start, stop, step) == (0, count, 1):
                return self
            return self._select(np.arange(start, stop, step))
        if isinstance(key, (list, tuple, np.ndarray)):
            positions = np.asarray(key)
            if positions.dtype == bool:
                raise TypeError("a bool is not a component index")
            if positions.size and not np.issubdtype(positions.dtype, np.integer):
                raise TypeError(
                    f"component indices must be integers, got {positions.dtype}"
                )
            if positions.ndim != 1:
                raise TypeError(
                    f"a Components takes a flat sequence of positions, got shape "
                    f"{positions.shape}"
                )
            count     = self.count
            positions = np.where(positions < 0, positions + count, positions)
            if positions.size and (positions.min() < 0 or positions.max() >= count):
                raise IndexError(f"{self!r}: index out of range for {count} components")
            return self._select(positions)
        raise TypeError(
            f"component indices must be an int, a slice or a sequence of ints, "
            f"not {type(key).__name__}"
        )

    def _select(self, positions: np.ndarray) -> "Components":
        """The sub-selection at ``positions`` (already validated)."""
        if self._indices is None:
            sizes = self._sizes
            ids   = positions if len(sizes) == 1 else _grid(sizes)[positions]
        else:
            ids = self._indices[positions]
        return Components._from_ids(self._shape, self._kind, ids)

    # -- operators -- #

    def __lshift__(self, other: Any) -> Any:
        if isinstance(other, _MemberSpec):
            return other.inject(self)
        if other is None:
            raise TypeError(
                "Components << None is not a clear; use members << -Spec('x') to "
                "remove them from one collection or members << Spec(None) to remove "
                "them from every collection of that kind"
            )
        raise TypeError(
            f"the right-hand side of a membership '<<' is a collection spec "
            f"(Tag('x'), Blinn('x'), ...), not {type(other).__name__}"
        )

    def __rshift__(self, other: Any) -> Any:
        if other is None:
            return self.indices
        if isinstance(other, _MemberSpec):
            return other.query(self)
        raise TypeError(
            f"'>>' on Components supports '>> None' (the native ids) or a "
            f"collection spec query, not {type(other).__name__}"
        )

    # -- equality / hashing / display -- #

    def __bool__(self) -> bool:
        return self.count > 0

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Components):
            return NotImplemented
        if self._kind != other._kind or self._path != other._path:
            return False
        if self._indices is None and other._indices is None:
            return True
        return np.array_equal(self.indices, other.indices)

    def __hash__(self) -> int:
        return hash((self._path, self._kind))

    def __repr__(self) -> str:
        prefix = _TOKEN_PREFIX.get(self._kind, self._kind)
        if self._indices is None:
            return f'Components("{self._path}.{prefix}[*]")'
        tokens = _render_tokens(self._kind, self._indices)
        if not tokens:
            return f'Components("{self._path}.{prefix}[]")'
        shown = " ".join(tokens[:_REPR_TOKENS])
        more  = len(tokens) - _REPR_TOKENS
        if more > 0:
            shown = f"{shown} ... (+{more} more)"
        return f'Components("{self._path}.{shown}")'


# ---------- Node.__getattr__ fallbacks ------------------------------------ #


def _maybe_components(node: Node, kind: str) -> Components | None:
    """``Components(shape, kind)`` for a mesh shape or a transform with exactly
    one own non-intermediate mesh shape; ``None`` when ``node`` has no mesh to
    take ``kind`` from (the caller re-raises its own ``AttributeError``);
    ``AttributeError`` naming the shapes when there are several."""
    dg = node._dg_node
    if not isinstance(dg, DAGNode):
        return None
    node_type = dg.node_type
    if node_type == "mesh":
        return Components(node, kind)
    if node_type in _GEOMETRY_TYPES:
        return None
    shapes = _own_shapes(dg, ("mesh",))
    if not shapes:
        return None
    if len(shapes) > 1:
        raise AttributeError(
            f"'{node}' has {len(shapes)} mesh shapes ({', '.join(shapes)}); "
            f"use Node('<shape>').{kind}"
        )
    return Components(Node(shapes[0]), kind)


def _single_geometry_shape(node: Node) -> Node | None:
    """The one own non-intermediate geometry shape under a transform ``node``
    (so its point aliases resolve through the transform), ``None`` when there
    is none or ``node`` is not a transform, ``AttributeError`` naming the
    shapes when there are several."""
    dg = node._dg_node
    if not isinstance(dg, DAGNode) or dg.node_type in _GEOMETRY_TYPES:
        return None
    shapes = _own_shapes(dg, _GEOMETRY_TYPES)
    if not shapes:
        return None
    if len(shapes) > 1:
        raise AttributeError(
            f"'{node}' has {len(shapes)} geometry shapes ({', '.join(shapes)}); "
            f"use Node('<shape>')"
        )
    return Node(shapes[0])


# ---------- The normaliser ------------------------------------------------ #


@dataclass(frozen=True, eq=False)
class _Selection:
    """One ``(path, kind)`` of a normalised left-hand side.

    ``path`` is the full DAG path (the shape for components, the node as
    given for ``'whole'``) or the DG name. ``tokens`` are the shape-scoped
    compact strings ``maya.cmds`` accepts -- ``'f[a:b]'``, ``'vtx[a:b]'``,
    ``'cv[u][v0:v1]'``, ``'map[a:b]'``, ``'controlPoints[a:b]'`` for plain
    surface / lattice plugs (``flat``), ``'<kind>[*]'`` for a bare handle
    (``is_all``) and ``()`` for ``'whole'``. ``indices`` are the native ids /
    coordinates (flat ``controlPoints`` ids when ``flat``), ``None`` for
    ``'whole'`` and ``is_all``. ``source`` is the tuple of the left-hand
    objects that contributed. ``eq=False``: an ndarray field has no
    element-wise-free equality, so selections compare by identity.
    """

    path:      str
    node_type: str
    kind:      str
    tokens:    tuple[str, ...]
    indices:   np.ndarray | None
    is_all:    bool
    flat:      bool
    source:    tuple

    @property
    def names(self) -> list[str]:
        """The tokens qualified with :attr:`path` (``[path]`` for ``'whole'``)."""
        if self.kind == "whole":
            return [self.path]
        return [f"{self.path}.{token}" for token in self.tokens]


class _NodeContext:
    """What the plug loop needs to know about one node, computed once per
    node: its path, type, point kind and the ``controlPoints`` / ``uvpt``
    attribute MObjects that mark a plug as a component."""

    __slots__ = (
        "path",
        "node_type",
        "point_kind",
        "ndims",
        "cp_attr",
        "uv_attr",
        "groups",
    )

    def __init__(self, mobject: OpenMaya.MObject) -> None:
        fn             = OpenMaya.MFnDependencyNode(mobject)
        node_type      = fn.typeName
        self.node_type = node_type
        self.groups    = {}
        if node_type in _GEOMETRY_TYPES:
            self.path       = OpenMaya.MDagPath.getAPathTo(mobject).fullPathName()
            self.point_kind = _POINT_KIND[node_type]
            self.ndims      = _ndims(self.point_kind, node_type)
            self.cp_attr    = fn.attribute("controlPoints")
            self.uv_attr    = fn.attribute("uvpt") if node_type == "mesh" else None
        else:
            self.path       = fn.name()
            self.point_kind = ""
            self.ndims      = 1
            self.cp_attr    = None
            self.uv_attr    = None


class _Group:
    """The accumulator behind one :class:`_Selection`."""

    __slots__ = (
        "path",
        "node_type",
        "kind",
        "flat",
        "is_all",
        "ids",
        "arrays",
        "sources",
    )

    def __init__(self, path: str, node_type: str, kind: str, flat: bool) -> None:
        self.path      = path
        self.node_type = node_type
        self.kind      = kind
        self.flat      = flat
        self.is_all    = False
        self.ids       = []
        self.arrays    = []
        self.sources   = []

    def finish(self) -> _Selection | None:
        """The finished selection, or ``None`` for an empty component group."""
        sources = tuple(self.sources)
        if self.kind == "whole":
            return _Selection(
                self.path, self.node_type, "whole", (), None, False, False, sources
            )
        if self.is_all:
            tokens = (f"{_TOKEN_PREFIX.get(self.kind, self.kind)}[*]",)
            return _Selection(
                self.path, self.node_type, self.kind, tokens, None, True, False, sources
            )
        parts = list(self.arrays)
        if self.ids:
            parts.append(np.asarray(self.ids, dtype=np.int64))
        if not parts:
            return None
        ids = np.concatenate(parts)
        ids = np.unique(ids, axis=0) if ids.ndim > 1 else np.unique(ids)
        if not len(ids):
            return None
        ids.flags.writeable = False
        tokens              = _render_tokens(self.kind, ids, self.flat)
        return _Selection(
            self.path, self.node_type, self.kind, tokens, ids, False, self.flat, sources
        )


def _at(where: str) -> str:
    return f"element {where}" if where else "the left-hand side"


class _Normaliser:
    """Folds a left-hand side into :class:`_Group` accumulators keyed by
    ``(path, kind, flat)`` in first-appearance order."""

    def __init__(self, want_shapes: bool) -> None:
        self.want_shapes = want_shapes
        self.groups:   dict[tuple[str, str, bool], _Group] = {}
        self.contexts: dict[int, _NodeContext] = {}

    def group(self, path: str, node_type: str, kind: str, flat: bool = False) -> _Group:
        key   = (path, kind, flat)
        group = self.groups.get(key)
        if group is None:
            group = self.groups[key] = _Group(path, node_type, kind, flat)
        return group

    def add(self, item: Any, where: str) -> None:
        if isinstance(item, Plug):
            self.add_sequence((item,), where, nested=False)
        elif isinstance(item, Node):
            self.add_node(item)
        elif isinstance(item, Components):
            self.add_components(item)
        elif isinstance(item, (list, tuple)):
            self.add_sequence(item, where, nested=True)
        elif isinstance(item, str):
            raise TypeError(
                f"{_at(where)}: raw component strings are not accepted on the left; "
                f"use Components({item!r})"
            )
        else:
            raise TypeError(
                f"{_at(where)} ({item!r}) is not a Node, a component Plug "
                f"or a Components"
            )

    def add_node(self, node: Node, source: Any = None) -> None:
        """The whole node; ``source`` is what the left-hand side actually
        held (the attribute plug that stands for the node)."""
        source = node if source is None else source
        dg     = node._dg_node
        if not isinstance(dg, DAGNode):
            self.group(dg.name, dg.node_type, "whole").sources.append(source)
            return
        node_type = dg.node_type
        if node_type not in _GEOMETRY_TYPES and self.want_shapes:
            shapes = _own_shapes(dg, _GEOMETRY_TYPES)
            if shapes:
                for path in shapes:
                    self.group(path, cmds.nodeType(path), "whole").sources.append(source)
                return
        self.group(dg.long_name, node_type, "whole").sources.append(source)

    def add_components(self, components: Components) -> None:
        group = self.group(components._path, components.geometry, components._kind)
        group.sources.append(components)
        if components._indices is None:
            group.is_all = True
        else:
            group.arrays.append(components._indices)

    def add_sequence(self, items: Any, where: str, nested: bool) -> None:
        # The hot loop: a 100k-element ``vtx[:]`` slice passes through here.
        # Plugs are classified inline from their MPlug -- node, attribute,
        # logical index -- and never rendered to a string. ``last_node`` /
        # ``last_ctx`` skip the context lookup while consecutive plugs share
        # a node, which is the common case.
        contexts  = self.contexts
        last_node = None
        last_ctx  = None
        for i, item in enumerate(items):
            if not isinstance(item, Plug):
                self.add(item, f"{where}[{i}]" if nested else where)
                continue
            mplug = item._mplug
            node  = mplug.node()
            if last_node is None or node != last_node:
                key = OpenMaya.MObjectHandle(node).hashCode()
                ctx = contexts.get(key)
                if ctx is None:
                    ctx = contexts[key] = _NodeContext(node)
                last_node = node
                last_ctx  = ctx
            ctx  = last_ctx
            attr = mplug.attribute()
            if ctx.cp_attr is not None and attr == ctx.cp_attr:
                if isinstance(item, ComponentPlug) and item._comp_coords is not None:
                    group = self._ctx_group(ctx, item._comp_alias, False)
                    group.ids.append(item._comp_coords)
                else:
                    try:
                        index = mplug.logicalIndex()
                    except TypeError:
                        # The bare ``controlPoints`` multi: every point.
                        group        = self._ctx_group(ctx, ctx.point_kind, False)
                        group.is_all = True
                    else:
                        group = self._ctx_group(ctx, ctx.point_kind, ctx.ndims > 1)
                        group.ids.append(index)
            elif ctx.uv_attr is not None and attr == ctx.uv_attr:
                try:
                    index = mplug.logicalIndex()
                except TypeError:
                    group        = self._ctx_group(ctx, "uv", False)
                    group.is_all = True
                else:
                    group = self._ctx_group(ctx, "uv", False)
                    group.ids.append(index)
            else:
                # An attribute plug stands for its node: ``cube.tx << Layer("x")``
                # is ``cube << Layer("x")``.
                self.add_node(item.node, source=item)
                continue
            group.sources.append(item)

    def _ctx_group(self, ctx: _NodeContext, kind: str, flat: bool) -> _Group:
        group = ctx.groups.get((kind, flat))
        if group is None:
            group                    = self.group(ctx.path, ctx.node_type, kind, flat)
            ctx.groups[(kind, flat)] = group
        return group

    def selections(self) -> list[_Selection]:
        result = [group.finish() for group in self.groups.values()]
        return [selection for selection in result if selection is not None]


def normalise(lhs: Any, *, want_shapes: bool) -> list[_Selection]:
    """Fold a membership left-hand side into one :class:`_Selection` per
    ``(path, kind)``.

    - ``Node``: a geometry shape is ``('whole', shape)``; a transform is its
      own non-intermediate geometry shapes when ``want_shapes`` (itself when
      it has none, so the spec can name it) and ``('whole', transform)``
      otherwise; a DG node is ``('whole', name)``.
    - ``Plug``: the bare ``controlPoints`` / ``uvpt`` multi is ``is_all`` with
      one ``'<kind>[*]'`` token and zero element plugs; an element plug
      contributes ``logicalIndex()``; a resolved ``ComponentPlug`` its
      coordinates; a plain ``controlPoints[k]`` on a surface / lattice passes
      through as ``'controlPoints[a:b]'`` with ``flat=True``; any other plug
      stands for its node, exactly as the ``Node`` would (``cube.tx`` is
      ``cube``), with the plug as the selection's source. Component plugs
      are grouped by node MObject and never rendered to strings.
    - ``Components``: its own selection. ``PlugList`` / list / tuple:
      flattened recursively. ``str``: ``TypeError`` pointing at
      ``Components(...)``. A number / ``None``: ``TypeError`` naming the
      element.
    - Equal ``(path, kind)`` merge; a whole-object entry and component
      entries on one path both survive. An empty result is
      ``ValueError('nothing to inject')``.
    """
    normaliser = _Normaliser(want_shapes)
    normaliser.add(lhs, "")
    selections = normaliser.selections()
    if not selections:
        raise ValueError("nothing to inject")
    return selections


# ---------- Shared helpers of the named kinds ----------------------------- #


def _find_node(name: str) -> str | None:
    """The node called ``name``: exact, then ``<currentNamespace>:name``.
    ``None`` when absent; ``ValueError`` when the name is ambiguous."""
    found = cmds.ls(name) or []
    if not found:
        namespace = cmds.namespaceInfo(currentNamespace=True)
        if namespace != ":":
            found = cmds.ls(f"{namespace}:{name}") or []
    if len(found) > 1:
        raise ValueError(f"'{name}' is ambiguous: {found}; use a full path")
    return found[0] if found else None


def _check_attrs(attrs: dict, *, node_type: str | None = None, node: str | None = None) -> None:
    """Every kwarg must name an attribute of the collection's node (by type
    before a create, on the node before an update), so a typo raises before
    any write."""
    for attr in attrs:
        query = {"type": node_type} if node is None else {"node": node}
        if not cmds.attributeQuery(attr, exists=True, **query):
            where = f"a {node_type}" if node is None else f"'{node}'"
            raise AttributeError(f"{where} has no attribute '{attr}'")


# ---------- Member-spec protocol base ------------------------------------ #


class _MemberSpec:
    """Protocol base of the collection specs (``Tag``, the materials,
    ``Layer``).

    ``Spec('x')`` names a collection; ``Spec(None)`` -- and ``Spec()``, the
    same object -- means no particular one: on ``<<`` it is the purge (every
    collection of this kind, mirroring ``plug << None``) and on ``>>`` it
    enumerates, answering what :meth:`of` answers. ``-Spec('x')`` is a copy
    that removes members instead of adding them; ``-Spec()`` (a double
    negative), ``--Spec('x')`` and ``~Spec(...)`` are ``TypeError``s. The
    constructor makes zero Maya calls; :meth:`_validate_name` is the per-kind
    validation hook.

    The verbs are template methods. :meth:`inject` (``lhs << spec``)
    normalises the left-hand side, applies :attr:`ACCEPTS`, hands the
    selections to :meth:`_plan` -- which validates everything and raises
    BEFORE any scene write -- then runs :meth:`_apply` inside one undo chunk
    named ``rig.<KIND>`` and returns the left-hand side unchanged.
    :meth:`query` (``lhs >> spec``) returns whatever :meth:`_query` reads: a
    plain value, never a :class:`Components`; a missing collection is a
    ``ValueError``. :meth:`of` enumerates the collections holding ``x``; an
    :attr:`EXCLUSIVE` kind (a node is in at most one collection) answers
    ``lhs >> Spec()`` with that one spec or ``None``. A kind that gives a
    bare ``Plug`` a meaning of its own (a tag name written into a deformer's
    expression) claims it in :meth:`_plan_plug`; any other attribute plug
    stands for its node.
    """

    KIND:        str = ""
    ACCEPTS:     frozenset[str] = frozenset()
    WANT_SHAPES: bool = False
    EXCLUSIVE:   bool = False

    def __init__(self, name: Any = None, **kw: Any) -> None:
        if name is not None:
            self._validate_name(name)
        self._name    = name
        self._remove  = False
        self._options = dict(kw)

    @classmethod
    def _unchecked(cls, name: str) -> "_MemberSpec":
        """A spec for a name read back from the scene, skipping validation
        (Maya stores names the constructor would refuse)."""
        spec          = object.__new__(cls)
        spec._name    = name
        spec._remove  = False
        spec._options = {}
        return spec

    def _validate_name(self, name: Any) -> None:
        """Reject anything but a non-empty ``str``; subclasses tighten this."""
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{type(self).__name__} name must be a non-empty str, got {name!r}"
            )

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def removes(self) -> bool:
        """``True`` on a ``-Spec('x')`` copy."""
        return self._remove

    @property
    def purges(self) -> bool:
        """``True`` for ``Spec()`` / ``Spec(None)``: no particular collection
        (every one of this kind on ``<<``, the enumeration on ``>>``)."""
        return self._name is None

    def __neg__(self) -> "_MemberSpec":
        cls = type(self).__name__
        if self._name is None:
            raise TypeError(
                f"-{cls}() is a double negative: {cls}() already removes the "
                f"members from every collection of this kind"
            )
        if self._remove:
            raise TypeError(
                f"--{cls}({self._name!r}): a removal cannot be negated again"
            )
        clone          = copy.copy(self)
        clone._remove  = True
        clone._options = dict(self._options)
        return clone

    def __invert__(self) -> "_MemberSpec":
        cls = type(self).__name__
        raise TypeError(
            f"~{cls}(...) is unassigned; -{cls}('x') removes members and "
            f"{cls}() removes them from every collection of this kind"
        )

    def __str__(self) -> str:
        return self._name if self._name is not None else repr(self)

    def __repr__(self) -> str:
        sign = "-" if self._remove else ""
        return f"{sign}{type(self).__name__}({self._name!r})"

    # -- the verbs -- #

    def inject(self, lhs: Any) -> Any:
        """``lhs << spec``: make the left-hand side a member (or, for a
        removal / purge copy, not a member). Everything is validated before
        the first scene write; the writes run in one undo chunk; the
        left-hand side comes back unchanged so the next ``<<`` targets it."""
        plan = self._plan_plug(lhs) if isinstance(lhs, Plug) else None
        if plan is None:
            selections = normalise(lhs, want_shapes=self.WANT_SHAPES)
            self._check_kinds(selections)
            plan = self._plan(selections)
        with _undo_chunk(f"rig.{self.KIND}"):
            self._apply(plan)
        return lhs

    def query(self, lhs: Any) -> Any:
        """``lhs >> spec``: a plain value read out of the left-hand side --
        the native ids of its members that are in the collection, or a bool
        for whole-object membership. A missing collection is a ``ValueError``,
        never an empty answer. ``lhs >> Spec()`` enumerates instead: the
        specs :meth:`of` lists, or, for an :attr:`EXCLUSIVE` kind, the one
        spec holding the left-hand side or ``None``."""
        cls = type(self).__name__
        if self._remove:
            raise TypeError(
                f"'>>' queries membership; -{cls}({self._name!r}) is a removal. "
                f"Use lhs >> {cls}({self._name!r})"
            )
        selections = normalise(lhs, want_shapes=self.WANT_SHAPES)
        self._check_kinds(selections)
        if self._name is not None:
            return self._query(selections)
        found = self._of(selections)
        if self.EXCLUSIVE:
            return found[0] if found else None
        return found

    @classmethod
    def of(cls, x: Any) -> list["_MemberSpec"]:
        """The collections of this kind holding ``x``, as re-injectable
        specs."""
        selections = normalise(x, want_shapes=cls.WANT_SHAPES)
        cls()._check_kinds(selections)   # the same gate as ``x >> Spec()``
        return cls._of(selections)

    # -- per-kind hooks -- #

    def _check_kinds(self, selections: list[_Selection]) -> None:
        """Apply :attr:`ACCEPTS`; :meth:`_kind_error` words the refusal."""
        for selection in selections:
            if selection.kind not in self.ACCEPTS:
                raise TypeError(self._kind_error(selection))

    def _kind_error(self, selection: _Selection) -> str:
        return (
            f"{type(self).__name__} does not take {selection.kind!r} components "
            f"({selection.names[0]}); it accepts {sorted(self.ACCEPTS)}"
        )

    def _plan_plug(self, plug: Plug) -> Any:
        """A plan for a bare ``Plug`` left-hand side the kind gives its own
        meaning to, or ``None`` to normalise it like any other."""
        return None

    def _plan(self, selections: list[_Selection]) -> Any:
        """Validate the normalised left-hand side and return the writes to
        run. Raises before anything is written."""
        raise NotImplementedError(f"{type(self).__name__} does not implement '<<'")

    def _apply(self, plan: Any) -> None:
        """Run the writes :meth:`_plan` prepared (inside the undo chunk)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement '<<'")

    def _query(self, selections: list[_Selection]) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement '>>'")

    @classmethod
    def _of(cls, selections: list[_Selection]) -> list["_MemberSpec"]:
        raise NotImplementedError(f"{cls.__name__} does not implement 'of'")
