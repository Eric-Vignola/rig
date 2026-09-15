"""
Type-aware shorthand connection translator.

When the source and destination plugs have *different* but *related* types
(e.g. matrix -> transform, quaternion -> euler), this module auto-inserts the
correct conversion node (``decomposeMatrix``, ``composeMatrix``,
``quatToEuler``, ``eulerToQuat``) and forwards the connection through it.

Activated by default via ``rig.set_options(use_shorthand=True)`` (the default).

Examples that use shorthand::

    obj2.t << obj1.wm        # decomposeMatrix.outputTranslate -> t
    obj2.r << obj1.wm        # decomposeMatrix.outputRotate -> r
    obj2.r << some_quat      # quatToEuler.outputRotate -> r
    matrix_attr << obj1.t    # composeMatrix(translate=...)
"""

from __future__ import annotations

from typing import Any

from rig._internal.container import ContainerOptions
from rig._internal.math_nodes import (
    _compose_matrix,
    _constant,
    _decompose_matrix,
    _euler_to_quaternion,
    _quaternion_to_euler,
)
from rig._internal.node import Node
from rig._internal.plug import _inject_value
from rig._internal.types import (
    _get_compound,
    _is_compound,
    _is_control_point,
    _is_matrix,
    _is_quaternion,
    _is_transform,
    _is_vector,
)


def shorthand(src: Any, dst: Any) -> bool:
    """Try to wire ``src -> dst`` via type-aware translation.

    Returns ``True`` if a translation network was inserted and the connection
    was made; ``False`` if no shorthand pair matched (caller should fall back
    to the standard injection logic).
    """
    if not ContainerOptions.use_shorthand:
        return False

    if _is_matrix(src):
        if _is_transform(dst):
            return _matrix_to_transform(src, dst)
        if _is_quaternion(dst):
            return _matrix_to_quaternion(src, dst)
        if _is_control_point(dst):
            return _matrix_to_point(src, dst)
        if _is_vector(dst):
            return _matrix_to_vector(src, dst)

    if _is_quaternion(src):
        if _is_transform(dst):
            return _quaternion_to_transform(src, dst)
        if _is_vector(dst):
            return _quaternion_to_vector(src, dst)
        if _is_matrix(dst):
            return _quaternion_to_matrix(src, dst)

    if _is_transform(src):
        if _is_quaternion(dst):
            return _transform_to_quaternion(src, dst)
        if _is_vector(dst) or _is_control_point(dst):
            return _transform_to_vector(src, dst)
        if _is_matrix(dst):
            return _transform_to_matrix(src, dst)

    if _is_vector(src):
        if _is_transform(dst):
            return _vector_to_transform(src, dst)
        if _is_quaternion(dst):
            return _vector_to_quaternion(src, dst)
        if _is_matrix(dst):
            return _vector_to_matrix(src, dst)

    return False


# ---------- matrix -> ... ----------------------------------------------- #


def _matrix_to_transform(src: Any, dst: Any) -> bool:
    """matrix -> transform: insert decomposeMatrix and route the channels."""
    attr = _attr_path(dst)
    # Try `dst.ro` first; if dst is typed-atomic (e.g. transform.matrix) or
    # otherwise has no `.ro`, fall back to the owning node's rotateOrder.
    rotate_order    = _safe_attr(dst, "ro") or _safe_attr(_node_of(dst), "ro")
    decomposed      = _decompose_matrix(src, rotate_order=rotate_order)
    decomposed_node = decomposed.node

    # Bare transform (no `.attr`) OR `.matrix` plug on a transform --
    # wire all four channels via the decomposeMatrix outputs. The sibling
    # fallback in Plug.__getattr__ makes ``dst.s/.r/.t/.shear`` resolve
    # to the owning node's channels even when ``dst`` is ``transform.matrix``.
    if not attr or attr in ("matrix", "m"):
        _bare_inject(decomposed_node.outputScale,     dst.s)
        _bare_inject(decomposed_node.outputRotate,    dst.r)
        _bare_inject(decomposed_node.outputTranslate, dst.t)
        _bare_inject(decomposed_node.outputShear,     dst.shear)
        return True

    if attr in ("scale", "scaleX", "scaleY", "scaleZ"):
        _bare_inject(decomposed_node.outputScale, dst)
        return True
    if attr in ("rotate", "rotateX", "rotateY", "rotateZ"):
        _bare_inject(decomposed_node.outputRotate, dst)
        return True
    if attr in ("translate", "translateX", "translateY", "translateZ"):
        _bare_inject(decomposed_node.outputTranslate, dst)
        return True
    if attr in ("shear", "shearXY", "shearXZ", "shearYZ"):
        _bare_inject(decomposed_node.outputShear, dst)
        return True

    return False


def _matrix_to_quaternion(src: Any, dst: Any) -> bool:
    decomposed = _decompose_matrix(src)
    _bare_inject(decomposed.node.outputQuat, dst)
    return True


def _matrix_to_vector(src: Any, dst: Any) -> bool:
    decomposed = _decompose_matrix(src, rotate_order=_safe_attr(_node_of(dst), "ro"))
    _bare_inject(decomposed, dst)
    return True


def _matrix_to_point(src: Any, dst: Any) -> bool:
    """Matrix -> control-point (vtx, cv, ...).

    Uses :attr:`Attribute.node` to obtain the owning shape node and
    :meth:`DAGNode.get_parent` (canonical API) to walk to the transform
    -- no ``str.split('.')`` parsing, no ``cmds.listRelatives``.
    """
    shape_node = _node_of(dst)
    try:
        transform_node = shape_node.get_parent()
    except AttributeError:
        # ``shape_node`` is not a DAGNode (no get_parent); bail out.
        return False
    if transform_node is None:
        return False

    decomposed = _decompose_matrix(src, rotate_order=_safe_attr(transform_node, "ro"))
    _bare_inject(decomposed, dst)
    return True


# ---------- quaternion -> ... ------------------------------------------- #


def _quaternion_to_transform(src: Any, dst: Any) -> bool:
    attr = _attr_path(dst)
    if not attr:
        node = _quaternion_to_euler(src, rotate_order=_safe_attr(dst, "ro"))
        _bare_inject(node, dst.r)
        return True
    if attr in ("rotate", "rotateX", "rotateY", "rotateZ"):
        node = _quaternion_to_euler(src, rotate_order=_safe_attr(_node_of(dst), "ro"))
        _bare_inject(node, dst)
        return True
    return False


def _quaternion_to_vector(src: Any, dst: Any) -> bool:
    """Quaternion -> vector: drop the W and inject XYZ via a constant
    aggregator so the destination receives a single compound vec3
    connection (canonical Maya idiom)."""

    children = _get_compound(src)
    new_plug = _constant([0, 0, 0])
    for src_child, dst_child in zip(children[:3], _get_compound(new_plug)):
        _bare_inject(src_child, dst_child)
    _bare_inject(new_plug, dst)
    return True


def _quaternion_to_matrix(src: Any, dst: Any) -> bool:
    node = _compose_matrix(rotate=src)
    _bare_inject(node, dst)
    return True


# ---------- vector / euler -> ... --------------------------------------- #


def _vector_to_transform(src: Any, dst: Any) -> bool:
    attr = _attr_path(dst)
    if not attr:
        _bare_inject(src, dst.t)
    else:
        _bare_inject(src, dst)
    return True


def _vector_to_quaternion(src: Any, dst: Any) -> bool:
    node = _euler_to_quaternion(src)
    _bare_inject(node, dst)
    return True


def _vector_to_matrix(src: Any, dst: Any) -> bool:
    node = _compose_matrix(translate=src)
    _bare_inject(node, dst)
    return True


# ---------- transform -> ... -------------------------------------------- #


def _transform_to_quaternion(src: Any, dst: Any) -> bool:
    attr = _attr_path(src)
    if not attr:
        decomposed = _decompose_matrix(src.matrix)
        _bare_inject(decomposed.node.outputQuat, dst)
        return True
    if attr in ("rotate", "rotateX", "rotateY", "rotateZ"):
        node = _euler_to_quaternion(src)
        _bare_inject(node, dst)
        return True
    return False


def _transform_to_vector(src: Any, dst: Any) -> bool:
    attr = _attr_path(src)
    if not attr:
        _bare_inject(src.t, dst)
    else:
        _bare_inject(src, dst)
    return True


def _transform_to_matrix(src: Any, dst: Any) -> bool:
    attr = _attr_path(src)
    if not attr:
        _bare_inject(src.matrix, dst)
        return True
    if attr in ("scale", "scaleX", "scaleY", "scaleZ"):
        node = _compose_matrix(scale=src)
        _bare_inject(node, dst)
        return True
    if attr in ("rotate", "rotateX", "rotateY", "rotateZ"):
        node = _compose_matrix(rotate=src, rotate_order=_safe_attr(_node_of(src), "ro"))
        _bare_inject(node, dst)
        return True
    if attr in ("translate", "translateX", "translateY", "translateZ"):
        node = _compose_matrix(translate=src)
        _bare_inject(node, dst)
        return True
    if attr in ("shear", "shearXY", "shearXZ", "shearYZ"):
        node = _compose_matrix(shear=src)
        _bare_inject(node, dst)
        return True
    return False


# ---------- helpers ---------------------------------------------------- #


def _attr_path(plug: Any) -> str:
    """Return the attribute path part of ``plug`` (everything after the node).

    For ``"pCube1.translate"`` returns ``"translate"``. For a bare
    :class:`Node` returns ``""`` so callers can route the source matrix
    to the node's standard transform channels (``t``/``r``/``s``)
    directly.

    Uses :attr:`Attribute.name` (canonical API) when ``plug`` exposes it,
    falling back to string parsing only for raw strings.
    """
    # Bare Node: no attribute path. Pre-v3.R, this branch was missing and
    # ``plug.name`` returned the NODE name (e.g. ``"b_xform"``), causing
    # the downstream ``if attr in (\"matrix\", \"m\"):`` check to evaluate
    # False and the matrix-to-transform router to bail. The bare-Node case
    # means \"use the node's t/r/s channels directly.\"
    if isinstance(plug, Node):
        return ""

    # Prefer the canonical Attribute.name (rename-safe).
    if hasattr(plug, "name") and not isinstance(plug, str):
        try:
            return plug.name
        except (RuntimeError, AttributeError):
            pass
    s = str(plug)
    return ".".join(s.split(".")[1:])


def _node_of(plug: Any) -> Any:
    """Return the :class:`Node` of a plug-like input.

    Uses :attr:`Attribute.node` (canonical API) when available, falling
    back to ``str.split('.')`` parsing only for raw strings.
    """

    # Prefer the canonical Attribute.node (rename-safe).
    if hasattr(plug, "node") and not isinstance(plug, str):
        try:
            return plug.node
        except (RuntimeError, AttributeError):
            pass
    return Node(str(plug).split(".")[0])


def _safe_attr(obj: Any, *names: str) -> Any:
    """Return ``obj.<name>`` for the first attribute that exists, or ``None``.

    Catches ``AttributeError`` (no such attr) AND ``TypeError`` (MPlug
    introspection failures on typed-atomic plugs like ``.matrix``).
    """
    for name in names:
        try:
            return getattr(obj, name)
        except (AttributeError, TypeError):
            continue
    return None


def _bare_inject(src: Any, dst: Any) -> None:
    """Set/connect ``src`` into ``dst`` WITHOUT trying shorthand again
    (we're already mid-shorthand). Avoids infinite recursion.
    """

    _inject_value(dst, src)