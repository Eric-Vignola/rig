from __future__ import annotations

import functools
import inspect
import re
from functools import total_ordering
from types import FunctionType
from typing import Any, Sequence

from maya import cmds, OpenMaya as OpenMaya1
from maya.api import OpenMaya
from rig.nodetypes._base import (
    _MISSING,
    _attr_handle,
    _attr_mobject,
    _cast,
    _class_attr,
    _deleted_error,
    _ensure_owner_alive,
    _full_name_buffer,
    _handle_valid,
    _lookup,
    _named_through_owner,
    _new_attr,
    _queried_data_type,
    _type_label,
    Attribute,
    get_custom_type,
    Node,
    set_custom_type,
)
from rig.nodetypes.errors import NodeNotFoundError, NodeTypeError


# Maya recognises these short names in cmds and MSelectionList parsing,
# but ``MFnDependencyNode.findPlug()`` does not -- they need explicit
# translation to the underlying plug name. Used as a fast filter in
# :meth:`DGNode.find_attr` to gate the alias-resolution fallback;
# the actual canonical name is looked up per-node-type via
# ``cmds.listAttr(f"{node}.{alias}[0]")`` so that aliases which are
# invalid for a given node type (e.g. ``mesh.cv``, ``curve.vtx``,
# ``lattice.pnts``) are correctly rejected.
#
# Edge / face component types (``e``, ``f``) intentionally have no entry
# here -- they're component types, not plugs, and have no settable /
# connectable underlying attribute.
_COMPONENT_ALIASES = frozenset(
    {
        "vtx",  # mesh vertices
        "cv",  # nurbsCurve / nurbsSurface CVs
        "pt",  # lattice points / generic point alias
        "pnts",  # alternative spelling
        "map",  # mesh UV components
        "uv",  # mesh UV components (alternative spelling)
    }
)


# Maya 2026 changed the canonical plug name that ``cmds.listAttr`` reports for
# some mesh component aliases:
#   * ``vtx`` / ``pt``  -> ``"pnts"``               (was ``"controlPoints"``)
#   * ``map`` / ``uv``  -> ``"uvSet.uvSetPoints"``  (was ``"uvpt"``)
# Normalize those back to the stable, cross-version canonical the rest of the
# DSL uses (mirrors ``rig._internal.plug._COMPONENT_TYPE_TO_PLUG_ATTR``,
# which the ``Plug("mesh.vtx[0]")`` construction path already uses) so that
# Node attribute access (``node.vtx``) and Plug construction resolve a
# component to the SAME plug regardless of Maya version. Both targets are
# valid, LIVE attributes on 2026: ``controlPoints`` mirrors ``pnts`` for
# vertex position, and ``uvpt`` is the attribute that actually drives the
# ``map`` UV component (the same-named ``uvSet.uvSetPoints`` array is
# independent of the live UVs that ``cmds.polyEditUV`` reflects).
_CANONICAL_COMPONENT_PLUG = {
    "pnts": "controlPoints",
    "uvSetPoints": "uvpt",
    "uvSet.uvSetPoints": "uvpt",
}


def _is_unresolved_multi_child(plug: OpenMaya.MPlug) -> bool:
    """True if ``plug`` (or an ancestor) is a multi element with an unresolved
    logical index (-1).

    ``MFnDependencyNode.findPlug`` returns such a plug for a bare child-of-multi
    name -- e.g. a blendShape's ``weights`` (child of the ``weightList`` multi)
    resolves to ``weightList[-1].weights``. That plug cannot be set/connected,
    and slicing it calls ``MPlug.getExistingArrayAttributeIndices()`` which
    aborts Maya with a ``TDEbadMultiIndex`` C++ exception.
    """
    cur = plug
    while True:
        if cur.isElement and cur.logicalIndex() < 0:
            return True
        if cur.isChild:
            cur = cur.parent()
        elif cur.isElement:
            cur = cur.array()
        else:
            return False


_NORMAL_ATTR = OpenMaya.MFnDependencyNode.kNormalAttr
_EXTENSION_ATTR = OpenMaya.MFnDependencyNode.kExtensionAttr

# The rig DSL layer: ``rig._internal.plug`` sets these when it loads, since it
# imports this module (it always loads with ``import rig``). Until then
# ``DGNode.__getattr__`` returns Attributes, a state that only exists mid-import.
_PLUG_CLASS     = None  # rig._internal.plug.Plug
_COMPONENT_PLUG = None  # rig._internal.plug._maybe_component_plug

# The container scope's typed-create hook (D13; the D31 pattern keeps nodetypes
# free of ``rig._internal`` imports). ``rig._internal.container`` sets it when it
# loads, to ``_typed_create(cls, run, args, kwargs, name_index=None)``, which
# calls ``run(cls, args, kwargs)``. Until then (mid-import only) a typed create
# runs plain.
_TYPED_CREATE_HOOK = None


def _create_template(cls, args, kwargs):
    """`DGNode.create`'s body: ``cls._create``, then ``cls.post_create``, given
    the call's ``args`` tuple and ``kwargs`` dict."""
    new_node = cls._create(*args, **kwargs)
    return cls.post_create(new_node, *args, **kwargs)


def _typed_creator(fn):
    """Decorate the body ``fn(cls, ...)`` of a typed creator classmethod that
    does not go through `DGNode.create` (put it under ``@classmethod``).

    Inside ``with container()`` the call joins the scope as `DGNode.create`
    does: ``container=`` is consumed (False: nothing is registered), an
    explicit ``name`` (by keyword, or at its position in ``fn``'s signature)
    takes the flattened scope's prefix, and every node the call made for
    itself is registered. ``skipSelect`` is never added (the body takes no
    such flag). A class that makes nodes before calling
    ``super().create()`` has them tracked only through this decorator."""
    params     = list(inspect.signature(fn).parameters)[1:]  # after cls
    name_index = params.index("name") if "name" in params else None

    def run(cls, args, kwargs):
        return fn(cls, *args, **kwargs)

    @functools.wraps(fn)
    def creator(cls, *args, **kwargs):
        hook = _TYPED_CREATE_HOOK
        if hook is None:
            kwargs.pop("container", None)
            return fn(cls, *args, **kwargs)
        return hook(cls, run, args, kwargs, name_index)

    return creator


# NURBS-surface ``cv`` / lattice ``pt``: the names a ComponentPlug may answer
_MULTIDIM = frozenset({"cv", "pt"})

# Names that ``DGNode.__getattr__`` resolves as geometry components AFTER the
# real attribute lookup fails: faces / edges become a ``Components`` (they have
# no plug), and the point aliases reach through a transform to its one shape.
_COMPONENT_TOKENS = frozenset({"f", "e"}) | _COMPONENT_ALIASES

# The instance state the node constructors, caches and the plug hash store,
# which the ``=`` sugar stores directly (never a Maya attr of that name)
_NODE_STATE = frozenset(
    {
        "_mobject",
        "_mdagpath",
        "_fn_set",
        "_fn_set1",  # NW6: API 1.0 handle store (a name, for the sugar)
        "_objhandle1",  # NW6: API 1.0 handle store (a name, for the sugar)
        "_attr_dict",
        "_Geometry__local_shape_attr",
        "_Geometry__world_shape_attr",
        # round 3: the node's plug hash serial and a DAG node's taken path
        "_node_serial",
        "_taken_mdagpath",
    }
)


def _filtered(
    attr_obj: Attribute, category: str | None, data_type: str | None
) -> Attribute | None:
    """`attr_obj` if it is in `category` and holds `data_type` (each when given),
    else None: `DGNode.find_attr`'s filters."""
    if (not category or attr_obj.has_category(category)) and (
        not data_type or attr_obj.data_type == data_type
    ):
        return attr_obj
    return None


def get_short_name(name: Any) -> str:
    """Returns the short name of a given node."""
    return str(name).rsplit("|", 1)[-1]


def get_clean_name(name: Any) -> str:
    """Returns clean name of a given node (no namespace)"""
    return get_short_name(name).rsplit(":", 1)[-1]


@total_ordering
class DGNode(Node):
    """Base class for DG nodes.

    This is a pymel-style class that maintains API handles to objects (no need to worry
    about name and hierarchy changes), it also provides rich methods for interacting
    with node properites, attributes, and connections.

    It can also be extended to implement custom object types.

    Instances of this class can be used as dict keys and passed into functions that
    expect string names.

    A subclass of the root :class:`rig.nodetypes._base.Node` (its metaclass,
    NodeMeta, is inherited): ``Node("x")`` returns an instance of the class the
    node's type maps to, which carries this typed API and the DSL
    (``node.tx`` is a :class:`Plug`, ``<<`` / ``>>``, the ``=`` sugar).
    ``DGNode("x")`` and every subclass's call is the strict typed reference
    (see :class:`Node`): the node ``Node("x")`` gives, checked to be of the
    class; ``create`` makes a node."""

    # the maya native node type string
    NATIVE_NODE_TYPE = "entity"

    # custom node type string
    CUSTOM_NODE_TYPE = None

    # the OpenMaya function set for this type
    FN_SET = OpenMaya.MFnDependencyNode

    # a typed create inside `with container()` joins the scope (D13); a scene
    # registry, found again by name, sets False and stays out unless
    # ``create(container=True)``
    _CONTAINER_AWARE = True

    # The inputs, named, of a class whose typed create builds the node from
    # positional inputs (a skinCluster's geometry and influences, a blendShape's
    # shapes, a mesh's data, a nurbsCurve's points, a reference's file):
    # ``Node.create`` of its type with none raises TypeError naming them, before
    # anything is made (a bare ``cmds.blendShape`` deforms the selection). None
    # elsewhere.
    _CREATE_TAKES_INPUTS = None

    def __init__(self, node: str | OpenMaya.MObject | DGNode) -> None:
        """Initialize an instance from a node name or a MObject.

        Reached through the cast and ``Cls._wrap(x)`` (a class call is the
        reference, see :class:`Node`). A node of this class (or a subclass)
        shares its internals; any other node object is taken by its name. The
        state is written to ``__dict__``, in this order, so no ``__setattr__``
        runs.
        """
        d = self.__dict__
        if isinstance(node, type(self)):
            d["_mobject"]    = node._mobject
            d["_fn_set"]     = node._fn_set
            d["_fn_set1"]    = node._fn_set1  # NW6: API 1.0 handle store
            d["_objhandle1"] = node._objhandle1  # NW6: API 1.0 handle store
        else:
            if isinstance(node, OpenMaya.MObject):
                d["_mobject"] = node
            else:
                sel = OpenMaya.MSelectionList()
                try:
                    sel.add(str(node))
                except Exception:
                    # the lookup rule of Node(x): its error (not found,
                    # ambiguous, a pattern), or the node it finds (the
                    # current namespace's)
                    sel.add(_lookup(str(node), _type_label(type(self))))
                d["_mobject"] = sel.getDependNode(0)
            d["_fn_set"] = self.FN_SET(d["_mobject"])
            self._cache_api1_objects(d["_fn_set"].name())
        self.is_type(self.name, exact_type=False, failfast=True)
        d["_attr_dict"] = {}  # cache queried attributes

    def _cache_api1_objects(self, name):
        # cache a API 1.0 MFnDependencyNode for validation purpose
        # 2.0 mobjects crash maya after new scene...
        # TODO check if maya fixed this in 2023+
        sel = OpenMaya1.MSelectionList()
        sel.add(name)
        mobject1 = OpenMaya1.MObject()
        sel.getDependNode(0, mobject1)
        d                = self.__dict__
        d["_fn_set1"]    = OpenMaya1.MFnDependencyNode(mobject1)  # NW6: API 1.0 handle store
        d["_objhandle1"] = OpenMaya1.MObjectHandle(mobject1)  # NW6: API 1.0 handle store

    # --- dunders

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.name}")'

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.long_name)

    def __eq__(self, other: Any) -> bool:
        # another object answers for itself (a plug: ``plug == node`` is
        # False and builds nothing)
        if not isinstance(other, Node):
            return NotImplemented
        return isinstance(other, type(self)) and self.name == other.name

    def __gt__(self, other: Any) -> bool:
        return self.name > str(other)

    def __getattr__(self, attr_name):
        """Returns the Maya attribute ``attr_name`` as a DSL :class:`Plug` owned
        by this node (``node.tx.node is node``).

        Only runs once normal lookup failed, so Python members always win. A
        ``_`` name (Python probes dunders constantly: ``__deepcopy__``,
        ``__array__``, ``__apiobject__``) resolves only to a Maya attr that
        really exists on a live node; the API 1.0 handle is read first, it is
        safe on a node a new scene freed, and a half-built instance (``copy``)
        raises ``AttributeError``. A node a new scene, a file open or a
        reference unload freed raises ``already deleted!``, a cached attr too.

        The plug takes the handle of its attribute for a dynamic attr, and, on
        a node with more than one DAG path, is named through this node's path
        (``Node("|T2|S").v`` is ``T2|S.visibility``). An attr of another node (a shape's, read through
        its transform) keeps that node as its owner, or none.

        Real attributes always win (``curveShape.f`` is ``form``). Only once the
        lookup has raised do ``f`` / ``e`` become a ``Components`` on a mesh
        shape or a transform with exactly one mesh shape, and do the point
        aliases (``vtx`` / ``cv`` / ``pt`` / ``map`` / ``uv``) resolve through a
        transform with exactly one geometry shape. A NURBS-surface ``cv`` or a
        lattice ``pt`` is a ``ComponentPlug`` (``node.cv[u, v]``).

        The typed API reads Maya attrs with :meth:`find_attr`, which returns
        :class:`Attribute` instances.

        Example:
        ```
        node.my_attr << value
        ```
        """
        d = self.__dict__
        if attr_name[:1] == "_":
            fn = d.get("_fn_set")
            if fn is None or not _handle_valid(d) or not fn.hasAttribute(attr_name):
                raise AttributeError(attr_name)
        cache = d.get("_attr_dict")
        if cache is None:  # half-built (a failed __init__, object.__new__)
            raise AttributeError(attr_name)
        attr = cache.get(attr_name)
        if attr is None:
            try:
                attr = self.find_attr(attr_name)
            except AttributeError:
                if attr_name in _COMPONENT_TOKENS:
                    from rig._internal.node import _component_fallback

                    found = _component_fallback(self, attr_name)
                    if found is not None:
                        return found
                raise
        elif not d["_objhandle1"].isAlive():  # NW6: API 1.0 handle read (hot)
            # a node a new scene freed: its cached MPlug points at freed memory
            # (a miss raised through the fn set). `_attr_dict` is set last by
            # every constructor, so the handle is there.
            self.ensure_valid()
        plug_cls = _PLUG_CLASS
        if plug_cls is None:
            return attr
        attr_d = attr.__dict__
        owner  = attr_d["_node"]
        # with the handle of its attribute, for a dynamic attr
        attr1  = attr_d.get("_attr1")
        if attr_name in _MULTIDIM:
            # NURBS-surface ``cv`` / lattice ``pt``: ``node.cv[u, v]`` resolves
            # like the ``Plug("shape.cv[u][v]")`` string path
            plug = _COMPONENT_PLUG(attr_name, attr)
            if plug is not None:
                plug.__dict__["_node"] = owner
                return _named_through_owner(plug)
        if owner is self:
            plug = _new_attr(plug_cls, attr_d["_mplug"], None, attr1)
            plug.__dict__["_node"] = self
            # ``find_attr`` named the attr through this node's path if it has
            # more than one; one it did not is named as the MPlug
            if not str.__contains__(attr, "|"):  # _named_through_a_path
                return plug
            return _named_through_owner(plug)
        # an attr of another node: its owner, or none and the handle of that
        # node the attr took, which checks it until ``Plug.node`` casts it
        plug = _new_attr(plug_cls, attr_d["_mplug"], attr_d["_handle1"], attr1)
        plug.__dict__["_node"] = owner
        return _named_through_owner(plug)

    def __setattr__(self, name: str, value: Any) -> None:
        """``node.tx = 5`` is sugar for ``node.tx << 5``.

        The node's own state (``_mobject``, ``_attr_dict`` ...) and a Python
        attribute already stored on this instance (``vars(node)["tag"] = 1``)
        are stored. A class attribute of that name wins: a property's setter
        runs (``node.namespace = "ns"``), a read-only property raises, a data
        default declared on the class (``side = None`` in a subclass body) is
        shadowed on the instance, so a node class keeps Python state under the
        names it declares (``self.side = "L"``), and a method or other class
        member raises (``Plug(node.find_attr("rename")) << value`` reaches a
        Maya attr of such a name), unless a callable replaces a method on this
        instance (``node.get_parent = f``, ``mock.patch.object(node, ...)``). A
        ``_`` name is Python state unless the live node has a Maya attr of that
        name (``node.__parked__ = 4.0``). Any other name must be a Maya attr: a
        typo raises "Attribute not found" instead of adding a Python attribute,
        and a deleted node raises ``already deleted!``.
        """
        d = self.__dict__
        # the node state, and a Python attribute already stored on this instance
        # (``vars(node)["tag"] = ...``, a method mock.patch.object patched)
        if name in _NODE_STATE or name in d:
            d[name] = value
            return
        found = _class_attr(type(self), name)
        if found is not _MISSING:
            if name[:1] == "_" or hasattr(type(found), "__set__"):
                # a property setter runs, a read-only one raises, a private
                # class default is shadowed on the instance
                object.__setattr__(self, name, value)
                return
            # a callable shadows a method on this instance (monkeypatching,
            # ``mock.patch.object(node, "get_parent")``)
            if callable(value) and isinstance(
                found, (FunctionType, classmethod, staticmethod)
            ):
                d[name] = value
                return
            # a data default the class declares (not callable, not a
            # descriptor): Python state of the instance
            if not callable(found) and not hasattr(type(found), "__get__"):
                d[name] = value
                return
            raise AttributeError(
                f"{type(self).__name__}.{name} is a method or class attribute, "
                f"not a plug; use Plug(node.find_attr({name!r})) << value for a "
                f"Maya attr of that name"
            )
        if name[:1] == "_":
            fn = d.get("_fn_set")
            if fn is None or not _handle_valid(d) or not fn.hasAttribute(name):
                # private Python state (on a deleted or freed node too)
                d[name] = value
                return
        try:
            plug = type(self).__getattr__(self, name)
        except AttributeError as error:
            raise AttributeError(
                f"{error} (to keep Python state on a node object, declare a "
                f"default in its class body, {name} = None, or use a '_' name)"
            ) from error
        plug << value

    # --- DSL operators

    def __lshift__(self, other: Any) -> Any:
        """``node << X`` -- inject: a collection spec makes the node a member, an
        attribute spec adds an attribute, a matrix source on a transform drives
        its channels. See :func:`rig._internal.node._node_lshift`."""
        from rig._internal.node import _node_lshift

        return _node_lshift(self, other)

    def __rshift__(self, other: Any) -> Any:
        """``node >> None`` returns the node itself (a no-op: the node already is
        the typed node); ``node >> spec`` declares an output attribute and
        ``node >> Tag(...)`` queries membership. See
        :func:`rig._internal.node._node_rshift`."""
        if other is None:
            return self
        from rig._internal.node import _node_rshift

        return _node_rshift(self, other)

    def __fspath__(self) -> str:
        return self.name

    # --- properties & utils

    @property
    def mobject(self) -> OpenMaya.MObject:
        """Returns the mobject."""
        self.ensure_valid()
        return self._mobject

    @property
    def node_type(self) -> str:
        """Returns the node type."""
        if self.CUSTOM_NODE_TYPE:
            return self.CUSTOM_NODE_TYPE
        return cmds.nodeType(self.name)

    def has_base_type(self, base_type: str) -> bool:
        """Checks if this node inherits from the given base type."""
        return cmds.objectType(self.name, isAType=base_type)

    @property
    def fn_set(self) -> OpenMaya.MFnBase:
        """Returns the OpenMaya function set."""
        self.ensure_valid()
        return self._fn_set

    @property
    def is_valid(self) -> bool:
        """Check if this node is valid (not deleted, not freed by a new scene, a
        file open or a reference unload). False for a half-built node."""
        return _handle_valid(self.__dict__)

    def ensure_valid(self) -> None:
        """Raise erros if an object is deleted.

        A node deleted to the undo queue is still alive, and is named. A node a
        new scene, a file open or a reference unload freed is not: its fn sets
        point at freed memory, so it is named by its class only (reading its
        name would read another node's, or crash Maya).
        """
        handle = self._objhandle1  # NW6: API 1.0 handle read (hot)
        if not handle.isValid():
            if handle.isAlive():
                raise _deleted_error(self._fn_set1.name())  # NW6: API 1.0 handle read
            raise _deleted_error(type(self).__name__, freed=True)

    @property
    def name(self) -> str:
        """Returns the node name."""
        return self.fn_set.name()

    @property
    def uuid(self) -> str:
        """Returns the node's universally unique identifyer as a string."""
        return str(self.fn_set.uuid())

    @property
    def short_name(self) -> str:
        """Returns the short name."""
        return get_short_name(self.name)

    # for consistency with DAGNode
    long_name = name

    @property
    def clean_name(self) -> str:
        """Returns the clean node name (no namespace)."""
        return get_clean_name(self.name)

    def rename(self, new_name: Any) -> None:
        """Renames the node."""
        new_name = str(new_name)
        if self.short_name != new_name:
            cmds.rename(self.name, new_name)

    @classmethod
    def is_type(cls, node_name, failfast: bool = False, **kwargs) -> bool:
        """Checks if a node is a valid node of this type.

        Args:
            node_name: A node name to check.
            failfast: If True, raise error if this node is invalid.
        """
        # a node is dg unless it's a custom type
        if not cls.CUSTOM_NODE_TYPE:
            return True
        ctyp = get_custom_type(node_name)
        if ctyp and ctyp == cls.CUSTOM_NODE_TYPE:
            return True

        if failfast:
            raise ValueError(f"{node_name} is not a {cls.CUSTOM_NODE_TYPE}")
        return False

    @classmethod
    def exists(cls, name: Any) -> bool:
        """Whether ``cls(name)`` would return a node: True for a node of this
        class or a subclass (a joint is a transform: ``Transform.exists("j1")``
        is True), False for a name no node has, a node of another type
        (``DisplayLayer.exists("cube")``), a node object that was deleted, and
        ``None`` or ``""`` (no name). A name it cannot answer yes or no for
        raises as the reference does: AmbiguousNodeError (two ``a``: a False
        would let ``if not Transform.exists("a"): Transform.create(name="a")``
        add a third ``a``) and the pattern's NodeLookupError (``"red*"`` is a
        search: ``cmds.ls``)."""
        if name is None:
            return False
        try:
            node = cls(name)
        except (NodeNotFoundError, NodeTypeError):
            return False
        return node.is_valid

    @classmethod
    def _coerce(cls, node: Any) -> Any:
        """``cls(x)``'s hook for the node ``Node(x)`` gave when it is not an
        instance of ``cls``: the node object to return instead, or None (the
        reference then raises NodeTypeError).

        A class with no node type of its own (``NATIVE_NODE_TYPE`` /
        ``CUSTOM_NODE_TYPE`` in its body: a user's wrapper subclass, which is
        not registered for a type) wraps a node its ``is_type`` accepts, as
        its constructor does. A registered class takes nothing else."""
        own = cls.__dict__
        if own.get("CUSTOM_NODE_TYPE") or own.get("NATIVE_NODE_TYPE"):
            return None
        name = node.name
        if not cls.is_type(name, exact_type=False):
            return None
        return cls._wrap(name)

    @classmethod
    def _mismatch_hint(cls, node: Any) -> str:
        """Text the reference's NodeTypeError adds after ``Node('x') is ...``
        (empty by default)."""
        return ""

    @classmethod
    def find_all(cls, *args, exact_type: bool = True, **kwargs) -> list["DGNode"]:
        """Returns a list of objects of this type in the scene.

        Args:
            exact_type: If True, return nodes of this exact type. Otherwise return
                inherited types as well.
        """
        nodes    = []
        type_key = "exactType" if exact_type else "type"
        kwargs.setdefault(type_key, cls.NATIVE_NODE_TYPE)

        # a node listed by its type is built as this class (its exact type)
        for node in cmds.ls(*args, **kwargs):
            if cls.CUSTOM_NODE_TYPE:
                if get_custom_type(node) == cls.CUSTOM_NODE_TYPE:
                    nodes.append(cls._wrap(node))
            else:
                nodes.append(
                    cls._wrap(node)
                    if cmds.nodeType(node) == cls.NATIVE_NODE_TYPE
                    else _cast(node)
                )
        return nodes

    # --- creation & deletion

    @classmethod
    def _create(cls, *args, **kwargs) -> str:
        """[Internal] Creates a new node of this type. Can be overridden by subclasses.
        This class can only use Maya APIs and must return a node name string.

        Only ``name`` / ``n`` and ``skipSelect`` / ``ss`` reach ``cmds.createNode``.
        """
        name = kwargs.get("name", kwargs.get("n"))
        name = name or (cls.CUSTOM_NODE_TYPE or cls.NATIVE_NODE_TYPE)
        skip = kwargs.get("skipSelect", kwargs.get("ss"))
        if skip is None:
            return cmds.createNode(cls.NATIVE_NODE_TYPE, name=name)
        return cmds.createNode(cls.NATIVE_NODE_TYPE, name=name, skipSelect=skip)

    @classmethod
    def create(cls, *args, **kwargs) -> "DGNode":
        """Creates a new node of this type: ``_create``, then ``post_create``.

        A subclass that overrides ``create`` calls ``super().create()`` (as
        ``Mesh`` does), so the rules below hold for every class.

        Inside ``with container()`` the new node joins the scope with
        ``container.createNode``'s rules, in one undo step: an explicit
        ``name=`` / ``n=`` takes the flattened scope's prefix (after any
        namespace: ``ns:x`` is ``ns:inner_x``), ``skipSelect`` defaults to
        ``ContainerOptions.skip_selection`` (for the ``_create``s that forward
        it to ``cmds.createNode``: DGNode's and DAGNode's), the returned node is
        tagged for ``cleanup()`` when its type is a GC-eligible utility type,
        and every node the call made for itself is registered (a deformer's
        Orig shape under the user's mesh and a shared bind pose are not).
        ``container=False`` leaves the nodes unregistered, as it does for
        ``createNode`` (the prefix, ``skipSelect`` and the tag still apply). A
        scene registry (``_CONTAINER_AWARE = False``: display layers, sets and
        shading engines, references) stays out (no prefix, not registered)
        unless ``container=True`` (then registered, never prefixed). Only the
        outermost typed create does this. Outside a scope nothing changes
        (``container=`` is always consumed, never passed on).
        """
        hook = _TYPED_CREATE_HOOK
        if hook is None:
            kwargs.pop("container", None)
            return _create_template(cls, args, kwargs)
        return hook(cls, _create_template, args, kwargs)

    @classmethod
    def post_create(cls, new_node_name: str, *args, **kwargs) -> "DGNode":
        """Post creation operations. Can be overridden by subclasses."""
        if cls.CUSTOM_NODE_TYPE:
            set_custom_type(new_node_name, cls.CUSTOM_NODE_TYPE)
        # the node just made is of this class
        return cls._wrap(new_node_name)

    def duplicate(self, *args, **kwargs) -> list[DGNode]:
        """Thin wrapper around `cmds.duplicate()`."""
        old_nodes = set(cmds.ls(dagObjects=True, long=True))
        cmds.duplicate(self.name, *args, **kwargs)
        if kwargs.get("returnRootsOnly", kwargs.get("rr", False)):
            for node in cmds.ls(dagObjects=True, long=True):
                if node not in old_nodes:
                    return [_cast(node)]
            return []
        return [
            _cast(x) for x in cmds.ls(dagObjects=True, long=True) if x not in old_nodes
        ]

    def delete(
        self, nodes: str | DGNode | Sequence[str | DGNode] | None = None, **kwargs
    ) -> None:
        """Thin wrapper around `cmds.delete()`.
        If `nodes` is given, delete them. Otherwise delete this node."""
        cmds.delete(nodes or self.name, **kwargs)

    # --- namespace

    @property
    def namespace(self) -> str:
        """Returns the namespace of this node."""
        return self._fn_set.namespace

    @namespace.setter
    def namespace(self, namespace: str) -> None:
        """Sets the namespace of this node."""
        if not cmds.namespace(exists=namespace):
            cmds.namespace(add=namespace)
        self.rename(f"{namespace}:{self.clean_name}")

    # --- attribute methods

    def find_attr(
        self,
        attr:      Attribute | str | int,
        category:  str       | None      = None,
        data_type: str       | None      = None,
        quiet:     bool                  = False,
    ) -> Attribute | None:
        """Returns an attribute instance that matches the given criteria.

        Args:
            attr: The name or id of the attribute to search for.
            category: The attribute category.
            data_type: The attribute type string.
            quiet: If True, don't raise errors on failure.

        Returns:
            The attribute instance or None if no match was found. An attr of this
            node is owned by this node object (its ``node`` is ``self``); the
            normal attrs are cached, so the same instance is returned again.
        """
        # pass through; a DSL Plug (``mesh.skinMask``) gives the Attribute of its
        # MPlug, with its owner and handles, since the typed API reads
        # Attributes (a Plug's `get()` is numpy-shaped, its `==` / `<` build
        # nodes). Its MPlug is read once its node is known alive.
        if isinstance(attr, Attribute):
            _ensure_owner_alive(attr)
            if attr.plug.node() != self.mobject:
                if quiet:
                    return None
                raise AttributeError(f"{attr} doesn't belong to {self}.")
            if type(attr) is not Attribute:
                attr_d = attr.__dict__
                typed  = _new_attr(
                    Attribute, attr_d["_mplug"], attr_d["_handle1"], attr_d.get("_attr1")
                )
                typed.__dict__["_node"] = attr_d["_node"]
                # named through its owner's path, as the plug was
                if str.__contains__(attr, "|"):  # _named_through_a_path
                    typed = _named_through_owner(typed)
                return typed
            return attr

        # return cached attr, filtered like a new lookup
        attr_obj = self._attr_dict.get(attr)
        if attr_obj is not None:
            if not category and not data_type:
                return attr_obj
            return _filtered(attr_obj, category, data_type)

        # find mplug
        try:
            if re.match(r"^.*[\[\.].*$", str(attr)):
                sel = OpenMaya.MSelectionList()
                sel.add(f"{self.name}.{attr}")
                plug = sel.getPlug(0)
            else:
                plug = self.fn_set.findPlug(attr, False)
        except RuntimeError:
            # Fallback: component-alias resolution. Maya's
            # ``MFnDependencyNode.findPlug()`` doesn't recognise short
            # component aliases like ``vtx`` / ``cv`` / ``pt``, but
            # ``cmds.listAttr`` does -- and crucially it respects node
            # type, so e.g. ``mesh.cv`` (a nurbsCurve / nurbsSurface
            # alias) is rejected here rather than silently resolving to
            # ``controlPoints``. The static set acts as a fast filter so
            # we only invoke ``cmds.listAttr`` for known aliases, not
            # for arbitrary failed plug lookups (typos etc.).
            canonical = None
            if isinstance(attr, str) and attr in _COMPONENT_ALIASES:
                try:
                    resolved = cmds.listAttr(f"{self.name}.{attr}[0]")
                except (RuntimeError, ValueError):
                    resolved = None
                if resolved:
                    # cmds.listAttr returns the canonical name, possibly
                    # with an element index suffix (e.g. ``"pnts[0]"``).
                    canonical = resolved[0].split("[")[0]
                    # Normalize Maya 2026's renamed mesh-component plugs back
                    # to the DSL's stable canonical so Node attribute access
                    # matches Plug construction and stays on the live attr.
                    canonical = _CANONICAL_COMPONENT_PLUG.get(canonical, canonical)
            if canonical is not None:
                try:
                    plug = self.fn_set.findPlug(canonical, False)
                except RuntimeError:
                    # ``cmds.listAttr`` enumerated the alias but
                    # ``findPlug`` rejected the canonical name. Honour
                    # the ``quiet`` contract and surface the resolution
                    # mismatch in the error so it's debuggable.
                    if quiet:
                        return None
                    raise AttributeError(
                        f"Attribute not found: {self}.{attr} "
                        f"(alias resolved to {canonical!r}, but findPlug failed)"
                    )
            else:
                # ``findPlug`` doesn't resolve node-level aliases (e.g. a
                # blendShape target weight aliased to ``smile``); ``MSelectionList``
                # does -- and returns the concrete element (``weight[0]``) whose
                # ``name()`` reports the alias. Fall back to it before giving up.
                try:
                    sel = OpenMaya.MSelectionList()
                    sel.add(f"{self.name}.{attr}")
                    plug = sel.getPlug(0)
                except RuntimeError:
                    if quiet:
                        return None
                    # nicer error message
                    raise AttributeError(f"Attribute not found: {self}.{attr}")

        # Component aliases that name a *real* attribute (mesh ``pnts`` and
        # its short name ``pt``) resolve via the primary ``findPlug`` branch
        # above, skipping the ``cmds.listAttr`` normalization; re-canonicalize
        # so ``Node`` access matches the ``Plug(...)`` path on every geometry
        # type and Maya version.
        plug = self._canonicalize_component_alias_plug(attr, plug)

        # A bare child-of-multi name resolves to a plug whose enclosing array
        # element is unresolved (logicalIndex -1); it is unusable and aborts
        # Maya when sliced. Reject it like any other missing attribute.
        if _is_unresolved_multi_child(plug):
            if quiet:
                return None
            raise AttributeError(
                f"Attribute not found: {self}.{attr} "
                f"(resolves to {plug.name()!r}, a child of an unindexed multi; "
                f"index the parent multi element first)"
            )

        # An attr of this node is owned by this node object. One found on
        # another node (a transform's shape) finds its own node when asked (it
        # takes a handle of that node, see `_ensure_owner_alive`), and is not
        # cached: that shape can be deleted or replaced under the node.
        # Only normal attrs are cached: dynamic attrs can be renamed, and
        # dynamic and extension attrs can be deleted and re-added.
        if plug.node() != self._mobject:
            attr_obj = Attribute(plug)
        else:
            attr_obj = _new_attr(Attribute, plug)
            attr_obj.__dict__["_node"] = self
            # named through this node's path in the str buffer cmds reads, if
            # the node has more than one (see `_named_through_owner`); the fn
            # set was just read, so the node is valid
            fn = self._fn_set
            if isinstance(fn, OpenMaya.MFnDagNode) and fn.isInstanced(True):
                attr_obj = _full_name_buffer(attr_obj)
            attr_class = self._fn_set.attributeClass(_attr_mobject(attr_obj))
            if attr_class == _NORMAL_ATTR:
                # with the instanced indices: an element of a world space attr
                # (``worldMatrix[1]``) is otherwise named like its array
                ln                  = plug.partialName(False, False, True, False, False, True)
                sn                  = plug.partialName(False, False, True, False, False, False)
                self._attr_dict[ln] = attr_obj
                self._attr_dict[sn] = attr_obj
            else:
                # a dynamic attr keeps a handle of its attribute, which a delete
                # frees once it leaves the undo queue, and so does an extension
                # attr, which deleteExtension frees (see `_ensure_owner_alive`)
                name = attr if type(attr) is str and "." not in attr else None
                attr_obj.__dict__["_attr1"] = _attr_handle(
                    # NW6: API 1.0 handle read
                    plug, fn1=self._fn_set1, name=name,
                    extension=attr_class == _EXTENSION_ATTR,
                )

        return _filtered(attr_obj, category, data_type)

    def _canonicalize_component_alias_plug(
        self, attr: Attribute | str | int, plug: "OpenMaya.MPlug"
    ) -> "OpenMaya.MPlug":
        """Normalize a component alias that ``findPlug`` resolved to a
        version-specific plug name back to the DSL's stable canonical.

        ``pnts`` (and its short name ``pt``) are *real* mesh attributes, so
        ``MFnDependencyNode.findPlug`` resolves them directly in
        :meth:`find_attr`'s primary branch -- bypassing the ``cmds.listAttr``
        fallback that normalizes the other aliases. Left alone,
        ``node.pnts`` / ``node.pt`` would diverge from ``Plug("mesh.pnts[0]")``
        (which Maya parses as a vertex component and maps to
        ``controlPoints``). Re-resolve so both paths agree and stay on the
        stable, cross-version canonical that mirrors the live geometry.

        No-op for non-aliases or plugs whose name isn't a renamed component
        plug, so it never perturbs ordinary attribute access.
        """
        if not (isinstance(attr, str) and attr in _COMPONENT_ALIASES):
            return plug
        name      = plug.partialName(False, False, False, False, False, True)
        canonical = _CANONICAL_COMPONENT_PLUG.get(name)
        if canonical is None:
            return plug
        try:
            return self.fn_set.findPlug(canonical, False)
        except RuntimeError:
            return plug

    def find_alias(self, alias: str, quiet: bool = False) -> Attribute | None:
        """Returns an attribute that has the given alias.
        Not using MFnDependencyNode.findAlias() as it doesn't work for multi attrs.

        Args:
            alias: An alias to search.
            quiet: If True, don't raise errors on failure.

        Returns:
            The attribute instance or None if no match was found.
        """
        alias_list = cmds.aliasAttr(f"{self.name}", query=True) or []
        for i in range(0, len(alias_list), 2):
            if alias_list[i] == alias:
                sel       = OpenMaya.MSelectionList()
                attr_name = alias_list[i + 1]
                sel.add(f"{self.name}.{attr_name}")
                # a plug of this node: checked through this node's handle, and
                # its attribute's, if dynamic
                plug = sel.getPlug(0)
                return _new_attr(
                    # NW6: API 1.0 handle read
                    Attribute, plug, self._objhandle1, _attr_handle(plug, fn1=self._fn_set1)
                )
        if not quiet:
            raise RuntimeError(f"Alias not found: {self}.{alias}")

    def has_attr(self, attr: Attribute | str) -> bool:
        """Checks if an attribute exists."""
        if isinstance(attr, Attribute):
            return not attr.mobject.isNull()
        return self.fn_set.hasAttribute(attr)

    def set_attrs(self, skip_missing=False, **kwargs) -> None:
        """Gracefully sets attributes on a node from given kwargs."""

        # skip missing attributes (except for notes which are a special case)
        if skip_missing:
            kwargs = {
                k: v for k, v in kwargs.items() if (self.has_attr(k) or k == "notes")
            }

        for attr, value in kwargs.items():
            # special handling for notes which are visible in the attribute editor
            # but require an addAttr call if setting it for the first time.
            if attr == "notes" and not self.has_attr(attr):
                self.add_attr(attr, dt="string")

            # set the attribute
            a = self.find_attr(attr)
            if a:
                a.set(value)

    def add_attr(self, long_name: str, **kwargs) -> Attribute | None:
        """Adds an attribute of a given name. Thin wrapper of cmds.addAttr().

        Args:
            long_name: The attribute long name.
            kwargs: Keyword arguments passed to cmds.addAttr().

        Returns:
            The attribute object created or None if a compound attr is requested.
        """
        # enforce long name
        kwargs["longName"] = long_name
        if "ln" in kwargs:
            kwargs.pop("ln")

        # add attribute
        cmds.addAttr(self.name, **kwargs)

        # compound attr are not actually created until all child attrs are added
        # there fore quiet is set to True so that find_attr won't fail
        return self.find_attr(long_name, quiet=True)

    def delete_attr(self, attr: Attribute | str) -> bool:
        """Deletse an attribute of a given name. Thin wrapper of cmds.deleteAttr()."""
        if self.has_attr(attr):
            if isinstance(attr, Attribute):
                attr = attr.name
            cmds.deleteAttr(self.name, attribute=attr)
            return True
        return False

    def rename_attr(self, old_name: Attribute | str, new_name: str) -> Attribute:
        """Renames a given attribute. Thin wrapper of cmds.renameAttr()."""
        if not self.has_attr(old_name):
            raise RuntimeError(f"Attribute not found: {self.name}.{old_name}")
        elif self.has_attr(new_name):
            raise RuntimeError(f"Attribute already exists: {self.name}.{new_name}")

        # an attr is renamed whatever its name (a Plug's `!=` builds a node)
        if isinstance(old_name, Attribute) or old_name != new_name:
            if isinstance(old_name, Attribute):
                if old_name.node != self:
                    raise RuntimeError(
                        f"Attribute {old_name} does not belong to {self}"
                    )
                cmds.renameAttr(old_name.full_name, new_name)
            else:
                cmds.renameAttr(f"{self.name}.{old_name}", new_name)

        return self.find_attr(new_name)

    def list_attr(self, *args, **kwargs) -> list[Attribute]:
        """Lists attributes on this node. Thin wrapper of cmds.listAttr().

        Compound child names returned by ``cmds.listAttr`` for multi-of-
        compound attributes (e.g. ``publishedNodeInfo.publishedNode``)
        cannot be resolved into a plug without an element index -- the
        underlying ``MSelectionList.add(node.publishedNodeInfo.publishedNode)``
        call requires ``publishedNodeInfo[N].publishedNode``. These
        unresolvable child names are silently skipped (via
        ``find_attr(quiet=True)``) so that ``list_attr()`` returns the
        list of usable Attribute objects rather than raising.
        """
        attrs = [
            self.find_attr(x, quiet=True)
            for x in cmds.listAttr(self.name, *args, **kwargs) or []
        ]
        return [a for a in attrs if a is not None]

    def list_connections(self, **kwargs) -> list[DGNode | Attribute]:
        """Lists connected attrs on this node. Thin wrapper of
        cmds.listConnections()."""
        casted = {}
        result = []
        for each in cmds.listConnections(self.name, **kwargs) or []:
            obj = casted.get(each)
            if not obj:
                obj          = _cast(each)
                casted[each] = obj
            result.append(obj)
        return result

    # copy this implementation from Attribute
    find_connected_nodes = Attribute.find_connected_nodes

    def _attr_data_type_fallback(self, attr: Attribute) -> str:
        """A hook for node classes to resolve the data type of an given attribute on this node.

        This is called by Attribute.data_type, when it fails to resolve the data type
        using cmds.getAttr(attr_name, type=True).

        cmds.getAttr should always work but it's not the case in practice. This method
        gives node classes a chance to correct any undesired Maya behavior.
        """
        # Attribute.data_type's own query still holds if its call ran this node's
        # class hook first, and that hook changes neither the scene nor the attr
        # before it gets here
        typ = _queried_data_type(attr, self)
        if typ is None or not _keeps_query(type(self)):
            typ = cmds.getAttr(attr.full_name, type=True)
        # maintain consistent type string with cmds.addAttr()
        if typ == "TdataCompound":
            return "compound"
        return typ

    # --- misc

    def remove_from_all_sets(self) -> None:
        """Removes this node from all object sets."""
        result = cmds.listConnections(
            self.name,
            source      = False,
            destination = True,
            type        = "objectSet",
            plugs       = True,
            connections = True,
        )
        if result:
            for i in range(0, len(result), 2):
                cmds.disconnectAttr(result[i], result[i + 1])


# Node classes whose data type fallback hook only reads the scene before it calls
# DGNode's, mapped to that hook. With DGNode's own hook nothing runs in between.
_BASE_FALLBACK_HOOK  = DGNode._attr_data_type_fallback
_QUERY_KEEPING_HOOKS = {}


def _keeps_query(node_cls: type) -> bool:
    """True if the fallback hook of `node_cls` reaches DGNode's with the scene and
    the attr unchanged, so the type Attribute.data_type queried still holds. Any
    other class or hook (a subclass override, a class-level patch) queries again;
    an instance hook is ruled out by `Attribute.data_type` itself."""
    hook = node_cls._attr_data_type_fallback
    if hook is _BASE_FALLBACK_HOOK:
        return True
    # the class's own hook, whose super() call reaches DGNode's unpatched hook
    return (
        _QUERY_KEEPING_HOOKS.get(node_cls) is hook
        and DGNode._attr_data_type_fallback is _BASE_FALLBACK_HOOK
    )