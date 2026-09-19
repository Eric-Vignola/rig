"""
:class:`Node` -- composition wrapper around any ``DGNode``/``DAGNode`` /
typed subclass that returns :class:`Plug` instances from attribute access.

Construct from a name string, an existing ``DGNode``, or a string that
``PyNode`` can resolve. ``Node.create("transform", name="cube1")``
forwards to ``PyNode.create()`` and registers the new node with the active
container scope (so ``with container():`` works transparently).

The wrapper does NOT subclass ``DGNode`` -- that would require subclassing
every typed subclass (Mesh, Joint, Transform, ...). Instead, ``__getattr__``
delegates to the underlying ``DGNode`` and re-wraps any returned
``Attribute`` as a ``Plug``.

Container nodes are a :class:`Container` subclass of ``Node`` (defined in
:mod:`rig._internal.container`) -- they get all of ``Node``'s attribute
machinery for free and only add the (dormant in v1) publish API.
"""

from __future__ import annotations

import numbers
from typing import Any, Union

import numpy as np
from rig.maya.attribute import Attribute
from rig.maya.nodetypes._base import PyNode
from rig.maya.nodetypes.dg_node import _COMPONENT_ALIASES, DGNode
from rig._internal.plug import _maybe_component_plug, Plug


# Names that ``__getattr__`` resolves as geometry components AFTER the real
# attribute lookup fails: faces / edges become a ``Components`` (they have no
# plug), and the point aliases (``vtx`` / ``cv`` / ``pt`` / ...) reach through
# a transform to its single geometry shape.
_COMPONENT_TOKENS = frozenset({"f", "e"}) | _COMPONENT_ALIASES


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
            # ``Attribute.node`` returns a PyNode (DGNode subclass).
            object.__setattr__(self, "_dg_node", node_or_name.node)
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
        """Delegate to the underlying ``DGNode``; re-wrap ``Attribute`` returns
        as :class:`Plug`.

        Real attributes always win (``curveShape.f`` is ``form``,
        ``meshShape.face`` is a live plug). Only once the lookup has raised
        do ``f`` / ``e`` become a :class:`Components` on a mesh shape or a
        transform with exactly one mesh shape, and do the point aliases
        (``vtx`` / ``cv`` / ``pt`` / ``map`` / ``uv``) resolve through a
        transform with exactly one geometry shape (``Node("pCube1").vtx``).
        Two shapes raise an ``AttributeError`` naming them.
        """
        if attr_name.startswith("_"):
            # Python probes private and dunder names constantly
            # (``__deepcopy__``, ``_ipython_canary_method_should_not_exist_``),
            # so only a Maya attribute that really exists on the node gets
            # through; ``_dg_node`` itself is the slot this lookup runs on.
            if attr_name == "_dg_node" or not self._dg_node.has_attr(attr_name):
                raise AttributeError(attr_name)
        try:
            result = getattr(self._dg_node, attr_name)
        except AttributeError:
            if attr_name in _COMPONENT_TOKENS:
                # Lazy: members.py imports Node at module top.
                from rig._internal.members import (
                    _maybe_components,
                    _single_geometry_shape,
                )

                if attr_name in ("f", "e"):
                    components = _maybe_components(self, attr_name)
                    if components is not None:
                        return components
                else:
                    shape = _single_geometry_shape(self)
                    if shape is not None:
                        return getattr(shape, attr_name)
            raise
        if isinstance(result, Attribute) and not isinstance(result, Plug):
            # Upgrade multi-dimensional geometry components (NURBS-surface
            # ``cv``, lattice ``pt``) to a ComponentPlug so ``node.cv[u][v]`` /
            # ``node.pt[s][t][u]`` resolve like the ``Plug("shape.cv[u][v]")``
            # string path; everything else falls back to a plain Plug.
            component_plug = _maybe_component_plug(attr_name, result)
            if component_plug is not None:
                return component_plug
            return Plug(result.plug)
        return result

    def _attr_data_type_fallback(self, attr: Any) -> str:
        """Forward the data-type resolution hook to the wrapped ``DGNode``.

        :attr:`rig.maya.attribute.Attribute.data_type` calls this on the owning
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
        unless the node really has an attribute of that name.
        """
        if name.startswith("_") and (
            name == "_dg_node" or not self._dg_node.has_attr(name)
        ):
            object.__setattr__(self, name, value)
            return
        plug = self.__getattr__(name)
        plug << value

    # -- inject (for `node << Float("foo")`, `node << matrix`, etc.) -- #

    def __lshift__(self, other: Any) -> Any:
        """``node << X`` -- dispatches by RHS type:

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
            return other.inject(self)

        # 1. Attribute-spec injection -- add an attribute on this node.
        if _is_attribute_spec(other):
            return other.apply(self)

        # 2. Live Plug-typed matrix source on a transform -> decomposeMatrix
        #    shorthand. _matrix_to_transform handles the bare-node case
        #    (attr=="") by wiring all four channels.
        if _is_matrix(other):
            from rig._internal.decompose import _node_is_transform
            from rig._internal.shorthand import _matrix_to_transform

            if _node_is_transform(str(self)) and _matrix_to_transform(other, self):
                return self

        # 3. Static numpy / nested-list matrix source on a transform
        #    -> Tier C decomposition via _try_matrix_source_routing on
        #    self.matrix (which honours rotateOrder and preserves
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

                    if _node_is_transform(str(self)) and _try_matrix_source_routing(
                        self.matrix, arr
                    ):
                        return self

        raise TypeError(
            f"Cannot inject {type(other).__name__} into a bare Node; "
            f"use node.<attr> << {other!r} or wrap in an _AttrSpec."
        )

    # -- introspect + output-attr declaration -- #

    def __rshift__(self, other: Any) -> Any:
        """``node >> None`` returns the underlying typed
        :class:`rig.maya.nodetypes.dg_node.DGNode` instance (e.g. a
        ``Transform`` or ``Mesh``), kicking the caller out of the rig
        DSL into the ``rig.maya.nodetypes`` typed-node world.

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
        ``>>`` for value introspection or attr-spec cloning).
        """
        # Lazy imports to avoid circulars.
        from rig._internal.types import _is_attribute_spec, _is_member_spec

        if other is None:
            return self._dg_node
        if _is_member_spec(other):
            return other.query(self)
        if _is_attribute_spec(other):
            # Stamp writable=False onto a fresh copy of the spec so the
            # caller's instance is untouched (specs may be reused).
            return _apply_spec_as_output(other, self)
        raise TypeError(
            "'>>' on a Node supports `>> None` (returns the typed "
            "DGNode), `>> spec` (declares an output-only attr, "
            "writable=False), or use Plug's `>> None` to read a value "
            "/ `Plug >> Node` to clone an attribute spec."
        )

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


# --------------------------------------------------------------------- #
#  Module-level helpers
# --------------------------------------------------------------------- #


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
        return Plug(obj.plug)
    if isinstance(obj, DGNode):
        return Node(obj)
    if isinstance(obj, str):
        if "." in obj:
            return Plug(obj)
        return Node(obj)
    raise TypeError(f"Cannot lift {type(obj).__name__} into the rig DSL")