"""
:class:`Node` -- the DSL node factory and the root of every node class.

``Node`` lives in :mod:`rig.nodetypes._base` and is re-exported here:
``Node("pCube1")`` returns the typed node (``Transform``, ``Mesh``, ...), and
``Node(x) is x`` for a node object. Every node class (``DGNode`` and its
subclasses) carries the DSL: attribute access returns :class:`Plug` instances
owned by the node (``node.tx.node is node``; on a node with more than one DAG
path named through its path, ``Node("|T2|S").v`` is ``T2|S.visibility``),
``<<`` / ``>>`` inject and introspect, ``node.tx = 5`` is ``node.tx << 5``. A
node never falls back to its name: after a delete, a new scene, a file open or
a reference unload it raises ``already deleted!``.
``Node.create("transform", name="cube1")`` registers the new node with the
active container scope (so ``with container():`` works transparently).

This module holds the cold-path DSL helpers the node classes import lazily
(``node << X`` / ``node >> X``: :func:`_node_lshift` / :func:`_node_rshift`,
and the geometry-component fallback :func:`_component_fallback`) and
:func:`lift`. Container nodes are a :class:`Container` subclass of ``DGNode``
(defined in :mod:`rig._internal.container`).
"""

from __future__ import annotations

import numbers
from typing import Any, Union

import numpy as np
from rig.nodetypes._base import Attribute, Node
from rig._internal.plug import Plug


# --------------------------------------------------------------------- #
#  Module-level helpers
# --------------------------------------------------------------------- #


def _node_lshift(node: Any, other: Any) -> Any:
    """``node << other`` for a node (``DGNode.__lshift__``): dispatches by RHS
    type:

    - Membership (a collection spec: ``Tag``, a material; a layer node; a
      kind or removal token) -- makes this node a member (``node <<
      Tag("x")`` on a per-node kind means the collection itself) and
      returns the node.
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
        _is_membership,
    )

    # 0. Membership -- returns the node.
    if _is_membership(other):
        return other._member().inject(node)

    # 1. Attribute-spec injection -- add an attribute on this node.
    if _is_attribute_spec(other):
        return other.apply(node)

    # 2. Live Plug-typed matrix source on a transform -> decomposeMatrix
    #    shorthand. _matrix_to_transform handles the bare-node case
    #    (attr=="") by wiring all four channels.
    if _is_matrix(other):
        from rig._internal.decompose import _node_is_transform
        from rig._internal.shorthand import _matrix_to_transform

        if _node_is_transform(node) and _matrix_to_transform(other, node):
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

                if _node_is_transform(node) and _try_matrix_source_routing(
                    node.matrix, arr
                ):
                    return node

    # a node on the right is named "Node", whatever its class, as it always was
    kind = "Node" if isinstance(other, Node) else type(other).__name__
    raise TypeError(
        f"Cannot inject {kind} into a bare Node; "
        f"use node.<attr> << {other!r} or wrap in an _AttrSpec."
    )


def _node_rshift(node: Any, other: Any) -> Any:
    """``node >> other`` for a node (``DGNode.__rshift__``), once the caller
    handled ``>> None`` (the node itself): ``>> spec`` declares an output-only attribute
    (``writable=False``), ``>> Tag("x")`` answers membership ids and ``>>
    Layer()`` enumerates. Anything else raises :class:`TypeError`."""
    # Lazy imports to avoid circulars.
    from rig._internal.types import _is_attribute_spec, _is_membership

    if _is_membership(other):
        return other._member().query(node)
    if _is_attribute_spec(other):
        # Stamp writable=False onto a fresh copy of the spec so the
        # caller's instance is untouched (specs may be reused).
        return _apply_spec_as_output(other, node)
    raise TypeError(
        "'>>' on a Node supports `>> None` (returns the node "
        "itself), `>> spec` (declares an output-only attr, "
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
    ``AttributeError`` naming them. ``node`` is the node object the lookup
    ran on."""
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

    Used by ``DGNode.__rshift__`` (:func:`_node_rshift`) to implement the
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
    - String without ``.``     => the node (the :class:`Node` factory).
    - ``Attribute``           => :class:`Plug` of the same plug, owned by the
      node object the attribute holds (``Plug(attr)``).
    - ``Plug`` / a node        => returned as-is.
    """
    if isinstance(obj, (Plug, Node)):
        return obj
    if isinstance(obj, Attribute):
        # read through the node object the attr holds, as ``Plug(attr)`` is
        return Plug(obj)
    if isinstance(obj, str):
        if "." in obj:
            return Plug(obj)
        return Node(obj)
    raise TypeError(f"Cannot lift {type(obj).__name__} into the rig DSL")
