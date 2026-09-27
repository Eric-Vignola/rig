"""
:class:`Node` -- composition wrapper around any ``DGNode``/``DAGNode`` /
typed subclass that returns :class:`Plug` instances from attribute access.

Construct from a name string, an existing ``DGNode``, or a string that
``PyNode`` can resolve. ``Node.create("transform", name="cube1")``
forwards to ``PyNode.create()`` and registers the new node with the active
container scope (so ``with container():`` works transparently).

The wrapper does NOT subclass ``DGNode`` -- that would require subclassing
every typed subclass (Mesh, Joint, Transform, ...). Instead, ``__getattr__``
delegates to the underlying ``DGNode``, whose own lookup returns Maya
attributes as :class:`Plug` instances (typed nodes speak the DSL too). A Plug
of the wrapped node is owned by this wrapper (``node.tx.node is node``) and,
on a node with more than one DAG path, named through the wrapper's path
(``Node("|T2|S").v`` is ``T2|S.visibility``). A wrapper never falls back to
its node's name: after a delete, a new scene, a file open or a reference
unload it raises ``already deleted!``.

``node << X`` and ``node >> X`` run the same module functions for a wrapper
and a typed node (:func:`_node_lshift` / :func:`_node_rshift`), and so does the
geometry-component fallback (:func:`_component_fallback`).

Container nodes are a :class:`Container` subclass of ``Node`` (defined in
:mod:`rig._internal.container`) -- they get all of ``Node``'s attribute
machinery for free and only add the (dormant in v1) publish API.
"""

from __future__ import annotations

import numbers
from typing import Any, Union

import numpy as np
from rig.nodetypes import _base
from rig.nodetypes._base import _MISSING, _class_attr, _handle_alive, Attribute, PyNode
from rig.nodetypes.dg_node import _COMPONENT_TOKENS, DGNode  # noqa: F401 (re-export)
from rig._internal.plug import Plug


class Node:
    """Wrapper that exposes a Maya node through the rig DSL.

    Attribute lookups (``node.translate``, ``node.weight``) return :class:`Plug`
    instances. Method lookups (``node.add_attr(...)``, ``node.list_connections(...)``)
    delegate to the underlying ``DGNode``.
    """

    __slots__ = ("_dg_node",)

    def __init__(self, node_or_name: Union[str, DGNode, "Node", Any]) -> None:
        if isinstance(node_or_name, Node):
            object.__setattr__(self, "_dg_node", node_or_name._dg_node)
        elif isinstance(node_or_name, DGNode):
            object.__setattr__(self, "_dg_node", node_or_name)
        elif isinstance(node_or_name, Attribute):
            # ``Node(plug)`` strips the attribute and returns the owning
            # node -- symmetric with the original DSL's ``node._`` idiom.
            # ``Attribute.node`` returns a PyNode (DGNode subclass), and
            # ``Plug.node`` the Node the plug was read from: wrap its typed node.
            owner = node_or_name.node
            if isinstance(owner, Node):
                owner = owner._dg_node
            object.__setattr__(self, "_dg_node", owner)
        else:
            # If a string carries an attribute suffix (``"pCube.tx"``,
            # ``"pCube.translate"``, ``"pCubeShape.vtx[0]"``), strip it so
            # users don't have to manually split node-from-attribute. Only
            # splits on the FIRST ``.`` so paths and namespaces are
            # preserved (Maya doesn't use ``.`` in node names).
            if isinstance(node_or_name, str) and "." in node_or_name:
                node_or_name = node_or_name.split(".", 1)[0]
            # Let PyNode resolve -- supports names, MObjects, MDagPaths.
            resolved = PyNode(node_or_name)
            if isinstance(resolved, Attribute):
                # Defensive -- shouldn't happen after the strip above, but
                # if PyNode resolves something exotic to an Attribute,
                # extract the owning node rather than raise.
                object.__setattr__(self, "_dg_node", resolved.node)
                return
            object.__setattr__(self, "_dg_node", resolved)

    # -- factory -- #

    @classmethod
    def create(cls, node_type: str, **kwargs: Any) -> "Node":
        """Create a new node and register it with the active container scope.

        Returns a :class:`Node` wrapping the new node.
        """
        # Use container.createNode so the new node enters the active scope
        # AND gets the right name prefix when nested in a flattened block.
        from rig._internal.container import container

        return container.createNode(node_type, **kwargs)

    @classmethod
    def wrap(cls, value: Any) -> Any:
        """Wrap a ``maya.cmds`` result (str / list-of-str) as :class:`Node` /
        :class:`PlugList`.

        Use this when calling ``maya.cmds`` directly (instead of going through
        :mod:`rig.bridges.commands`) and you want the result back in DSL form::

            from maya import cmds
            from rig import Node

            n   = Node.wrap(cmds.createNode("transform"))     # -> Node
            sel = Node.wrap(cmds.ls(sl=True))                  # -> list[Node]
            x   = Node.wrap(5.0)                                # -> 5.0 (passthrough)
            none = Node.wrap(None)                              # -> None

        Strings that aren't valid node names are passed through unchanged
        (so ``Node.wrap(cmds.getAttr("foo.attr", asString=True))`` won't try
        to coerce a value-string into a Node).
        """
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return cls(value)
            except Exception:
                return value
        if isinstance(value, (list, tuple)):
            from rig._internal.list import PlugList

            wrapped = [cls.wrap(v) for v in value]
            try:
                return PlugList(wrapped)
            except Exception:
                return wrapped
        return value

    # -- attribute lookup -- #

    def __getattr__(self, attr_name: str) -> Any:
        """Delegate to the underlying ``DGNode``, which returns Maya attributes
        as :class:`Plug` instances and resolves the geometry-component
        fallbacks (see :meth:`DGNode.__getattr__`): ``f`` / ``e`` on a mesh
        shape or a transform with one mesh shape, the point aliases through a
        transform with one geometry shape (``Node("pCube1").vtx``).

        A plug of the wrapped node is owned by this wrapper (``node.tx.node is
        node``); it is named through the wrapped node's path, which is this
        wrapper's. An attr of another node keeps its own owner, or none.
        """
        if attr_name.startswith("_"):
            # Python probes private and dunder names constantly
            # (``__deepcopy__``, ``_ipython_canary_method_should_not_exist_``),
            # so only a Maya attribute that really exists on the node gets
            # through; ``_dg_node`` itself is the slot this lookup runs on. A
            # node a new scene freed has none (its fn set points at freed memory).
            if (
                attr_name == "_dg_node"
                or not _handle_alive(self._dg_node.__dict__)
                or not self._dg_node.has_attr(attr_name)
            ):
                raise AttributeError(attr_name)
        dg_node = self._dg_node
        result  = getattr(dg_node, attr_name)
        if isinstance(result, Plug) and result.__dict__.get("_node") is dg_node:
            # the plug is owned by the node object it was read from: this wrapper
            result.__dict__["_node"] = self
        return result

    def _attr_data_type_fallback(self, attr: Any) -> str:
        """Forward the data-type resolution hook to the wrapped ``DGNode``.

        :attr:`rig.nodetypes.Attribute.data_type` calls this on the owning
        node to resolve a generic ("typed" / "Tdata") attribute -- e.g. a
        ``choice`` node's ``output`` -- whose concrete type depends on its
        connections. Because :meth:`__getattr__` deliberately rejects every
        ``_``-prefixed name, the wrapped ``_dg_node``'s type-aware
        (``Choice`` / ``DGNode``) implementation is unreachable through normal
        delegation, so ``Plug(...).data_type`` would raise instead of
        resolving. Forward it explicitly.
        """
        return self._dg_node._attr_data_type_fallback(attr)

    def __setattr__(self, name: str, value: Any) -> None:
        """``node.tx = 5`` is sugar for ``node.tx << 5``.

        Internal state (``_``-prefix) bypasses to normal ``__setattr__``
        unless the node really has an attribute of that name. A name the
        wrapped node's class defines is set on the wrapped node (a property
        setter runs, a method or read-only property raises), as
        :meth:`DGNode.__setattr__` does. Any other name is a plug this
        wrapper's lookup finds (a ``Container``'s published names too).
        """
        if name.startswith("_") and (
            name == "_dg_node" or not self._dg_node.has_attr(name)
        ):
            object.__setattr__(self, name, value)
            return
        dg_node = self._dg_node
        if _class_attr(type(dg_node), name) is not _MISSING:
            setattr(dg_node, name, value)
            return
        self.__getattr__(name) << value

    # -- inject (for `node << Float("foo")`, `node << matrix`, etc.) -- #

    def __lshift__(self, other: Any) -> Any:
        """``node << X`` -- see :func:`_node_lshift`: a collection spec makes
        the node a member, an attribute spec adds an attribute, a matrix
        source on a transform drives its channels; anything else raises
        ``TypeError`` (use ``node.<attr> << value`` to target a channel)."""
        return _node_lshift(self, other)

    # -- introspect + output-attr declaration -- #

    def __rshift__(self, other: Any) -> Any:
        """``node >> None`` returns the underlying typed
        :class:`rig.nodetypes.dg_node.DGNode` instance (e.g. a
        ``Transform`` or ``Mesh``), kicking the caller out of the rig
        DSL into the ``rig.nodetypes`` typed-node world.

        ``node >> spec`` adds the attribute described by ``spec`` to
        this node as an OUTPUT (``writable=False``) attribute. This is
        the mirror of ``node << spec`` (which adds an INPUT attribute,
        ``writable=True`` by default). The mnemonic is *the operator
        points the way the data flows*: ``<<`` flows IN, ``>>`` flows
        OUT.

        Output-only attrs on a plain Node are useful for locked
        metadata / constants (downstream code can READ them; nothing
        can clobber them).

        ``node >> Tag("x")`` (a collection spec) QUERIES membership and
        returns a plain value: the RHS family decides between declaring
        an output attribute and asking a question.

        Anything else raises :class:`TypeError` (use :class:`Plug`'s
        ``>>`` for value introspection or attr-spec cloning). See
        :func:`_node_rshift`.
        """
        if other is None:
            return self._dg_node
        return _node_rshift(self, other)

    # -- equality / hashing / str -- #

    def __str__(self) -> str:
        return str(self._dg_node)

    def __repr__(self) -> str:
        return f'Node("{self._dg_node}")'

    def __hash__(self) -> int:
        return hash(self._dg_node)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Node):
            return self._dg_node == other._dg_node
        return self._dg_node == other

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    # -- string compatibility (so cmds.* accepts a Node) -- #

    def __fspath__(self) -> str:
        return str(self._dg_node)


# ``Attribute.full_name`` reads a Node's name from the wrapped ``DGNode`` directly
# instead of through the ``__getattr__`` forwarding. ``Attribute.data_type`` lets
# the wrapped node's hook reuse its query only through this forwarding hook.
_base._NODE_WRAPPER_CLASS = Node
_base._NODE_WRAPPER_HOOK  = Node._attr_data_type_fallback


# --------------------------------------------------------------------- #
#  Module-level helpers
# --------------------------------------------------------------------- #


def _node_lshift(node: Any, other: Any) -> Any:
    """``node << other`` for a :class:`Node` or a typed ``DGNode``: dispatches by
    RHS type:

    - Collection spec (``Tag``, a material, ...) -- makes this node a
      member (``node << Tag("x")`` on a per-node kind means the
      collection itself) and returns the node.
    - ``_AttrSpec`` (``Float``, ``Vector``, ``lock``, ...) -- adds an
      attribute on this node.
    - **Matrix-shaped source on a transform** -- applies the matrix to
      the transform's t/r/s/shear channels:
        * Plug-typed matrix source => inserts a live ``decomposeMatrix``
          shorthand (via ``_shorthand._matrix_to_transform``).
        * Static numpy / nested list (3x3, 4x4, 9-flat, 16-flat) =>
          decomposes in Python via Maya's API and ``setAttr``s the
          channels (via ``_decompose._try_matrix_source_routing``).
          For 3x3 inputs the existing translation is preserved.
    - Anything else => ``TypeError`` (use ``node.<attr> << value`` to
      target a specific channel).
    """
    from rig._internal.types import (
        _is_attribute_spec,
        _is_matrix,
        _is_member_spec,
    )

    # 0. Collection-spec injection -- membership; returns the node.
    if _is_member_spec(other):
        return other.inject(node)

    # 1. Attribute-spec injection -- add an attribute on this node.
    if _is_attribute_spec(other):
        return other.apply(node)

    # 2. Live Plug-typed matrix source on a transform -> decomposeMatrix
    #    shorthand. _matrix_to_transform handles the bare-node case
    #    (attr=="") by wiring all four channels.
    if _is_matrix(other):
        from rig._internal.decompose import _node_is_transform
        from rig._internal.shorthand import _matrix_to_transform

        if _node_is_transform(str(node)) and _matrix_to_transform(other, node):
            return node

    # 3. Static numpy / nested-list matrix source on a transform
    #    -> Tier C decomposition via _try_matrix_source_routing on
    #    node.matrix (which honours rotateOrder and preserves
    #    translation for 3x3 inputs).
    if (
        other is not None
        and not isinstance(other, (str, bytes))
        and not isinstance(other, numbers.Real)
    ):
        try:
            arr = np.asarray(other)
        except (ValueError, TypeError):
            arr = None
        if arr is not None and arr.dtype != object:
            shape = arr.shape
            size  = arr.size
            is_matrix_shape = (
                shape == (3, 3)
                or shape == (4, 4)
                or (arr.ndim == 1 and size in (9, 16))
            )
            if is_matrix_shape:
                from rig._internal.decompose import (
                    _node_is_transform,
                    _try_matrix_source_routing,
                )

                if _node_is_transform(str(node)) and _try_matrix_source_routing(
                    node.matrix, arr
                ):
                    return node

    raise TypeError(
        f"Cannot inject {type(other).__name__} into a bare Node; "
        f"use node.<attr> << {other!r} or wrap in an _AttrSpec."
    )


def _node_rshift(node: Any, other: Any) -> Any:
    """``node >> other`` for a :class:`Node` or a typed ``DGNode``, once the
    caller handled ``>> None``: ``>> spec`` declares an output-only attribute
    (``writable=False``), ``>> Tag("x")`` queries membership. Anything else
    raises :class:`TypeError`."""
    # Lazy imports to avoid circulars.
    from rig._internal.types import _is_attribute_spec, _is_member_spec

    if _is_member_spec(other):
        return other.query(node)
    if _is_attribute_spec(other):
        # Stamp writable=False onto a fresh copy of the spec so the
        # caller's instance is untouched (specs may be reused).
        return _apply_spec_as_output(other, node)
    raise TypeError(
        "'>>' on a Node supports `>> None` (returns the typed "
        "DGNode), `>> spec` (declares an output-only attr, "
        "writable=False), or use Plug's `>> None` to read a value "
        "/ `Plug >> Node` to clone an attribute spec."
    )


def _component_fallback(node: Any, attr_name: str) -> Any:
    """What ``node.<attr_name>`` gives once the attribute lookup raised, for a
    component token (see ``DGNode.__getattr__``): ``f`` / ``e`` a
    :class:`Components` on a mesh shape or a transform with exactly one mesh
    shape, a point alias (``vtx`` / ``cv`` / ``pt`` / ``map`` / ``uv``) that
    alias on a transform's one geometry shape. None when there is no such
    shape (the caller re-raises its own error); two shapes raise an
    ``AttributeError`` naming them. ``node`` is a typed node, or a
    :class:`Node` (the members helpers read its ``_dg_node``)."""
    # Lazy: members.py imports Node at module top.
    from rig._internal.members import _maybe_components, _single_geometry_shape

    if attr_name in ("f", "e"):
        return _maybe_components(node, attr_name)
    shape = _single_geometry_shape(node)
    if shape is not None:
        return getattr(shape, attr_name)
    return None


def _apply_spec_as_output(spec: Any, target: "Node") -> Any:
    """Apply ``spec`` to ``target`` as a writable=False (output) attr.

    Used by :meth:`Node.__rshift__` to implement the
    ``node >> Float("x")`` idiom for declaring an output-only attribute.
    Stamps ``writable=False`` onto a SHALLOW COPY of ``spec`` so the
    caller's spec instance is left intact (specs may be reused across
    multiple nodes / multiple calls).

    Maya semantics:
        ``writable=False`` makes the attribute non-settable / non-
        connectable as a destination. ``readable`` and ``storable``
        keep their defaults (True) so downstream code can still read
        the value and the attr survives ``.ma`` save/reopen.

    For nodes that compute their outputs via a custom DG plugin, the
    writable=False semantic is functionally meaningful -- the plugin
    writes, nothing else can.

    For plain Node usage, ``writable=False`` attrs are useful for
    locked metadata / constants -- downstream code can read them
    (``readable=True``), but nothing can overwrite them.
    """
    # Shallow-copy the spec to avoid mutating the caller's instance.
    # _AttrSpec stores all addAttr kwargs in .kargs (a dict); copying
    # the dict + the few sibling fields is enough.
    import copy as _copy

    cloned                   = _copy.copy(spec)
    cloned.kargs             = dict(spec.kargs)
    cloned.kargs["writable"] = False
    return cloned.apply(target)


def lift(obj: Any) -> Union[Plug, Node]:
    """Cast ``obj`` into the rig DSL.

    - String containing ``.`` => :class:`Plug`.
    - String without ``.``     => :class:`Node`.
    - ``Attribute``           => :class:`Plug`.
    - ``DGNode``              => :class:`Node`.
    - ``Plug`` / ``Node``     => returned as-is.
    """
    if isinstance(obj, (Plug, Node)):
        return obj
    if isinstance(obj, Attribute):
        # read through the node object the attr holds, as ``Plug(attr)`` is
        return Plug(obj)
    if isinstance(obj, DGNode):
        return Node(obj)
    if isinstance(obj, str):
        if "." in obj:
            return Plug(obj)
        return Node(obj)
    raise TypeError(f"Cannot lift {type(obj).__name__} into the rig DSL")