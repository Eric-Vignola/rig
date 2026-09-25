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
    _COMPOUND_DATA_TYPES,
    _get_compound,
    _is_compound,
    _is_control_point,
    _is_transform,
)
from rig.nodetypes._base import (
    _NON_MATRIX_ATTR_API_TYPES,
    _SCALAR_ATTR_API_TYPES,
    Attribute,
)


def shorthand(src: Any, dst: Any) -> bool:
    """Try to wire ``src -> dst`` via type-aware translation.

    Returns ``True`` if a translation network was inserted and the connection
    was made; ``False`` if no shorthand pair matched (caller should fall back
    to the standard injection logic).
    """
    if not ContainerOptions.use_shorthand:
        return False

    # every source test below is False for a non-attribute
    if not isinstance(src, Attribute):
        return False

    # src / dst type facts, each queried at most once for this call
    src_facts = _Facts(src)
    dst_facts = _Facts(dst)

    if src_facts.is_matrix:
        if _is_transform(dst):
            return _matrix_to_transform(src, dst)
        if dst_facts.is_quaternion:
            return _matrix_to_quaternion(src, dst)
        if _is_control_point(dst):
            return _matrix_to_point(src, dst)
        if dst_facts.is_vector:
            return _matrix_to_vector(src, dst)

    if src_facts.is_quaternion:
        if _is_transform(dst):
            return _quaternion_to_transform(src, dst)
        if dst_facts.is_vector:
            return _quaternion_to_vector(src, dst)
        if dst_facts.is_matrix:
            return _quaternion_to_matrix(src, dst)

    if _is_transform(src):
        if dst_facts.is_quaternion:
            return _transform_to_quaternion(src, dst)
        if dst_facts.is_vector or _is_control_point(dst):
            return _transform_to_vector(src, dst)
        if dst_facts.is_matrix:
            return _transform_to_matrix(src, dst)

    if src_facts.is_vector:
        if _is_transform(dst):
            return _vector_to_transform(src, dst)
        if dst_facts.is_quaternion:
            return _vector_to_quaternion(src, dst)
        if dst_facts.is_matrix:
            return _vector_to_matrix(src, dst)

    return False


# marks a `_Facts` query that has not run yet
_UNSET = object()


class _Facts:
    """The type facts ``shorthand`` tests on one operand, each queried once.

    ``is_matrix``, ``is_compound``, ``is_quaternion`` and ``is_vector`` answer
    exactly like the ``rig._internal.types`` predicates of the same names, but
    share one ``data_type`` and one ``num_children`` query between them. Only
    valid for the duration of one ``shorthand`` call: the type of a generic
    plug follows its connections.
    """

    __slots__ = ("_obj", "_data_type", "_num_children", "_is_compound")

    def __init__(self, obj: Any) -> None:
        self._obj          = obj
        self._data_type    = _UNSET
        self._num_children = _UNSET
        self._is_compound  = _UNSET

    def _get_data_type(self) -> str | None:
        """Returns the operand's ``data_type``, or None if the query raised."""
        if self._data_type is _UNSET:
            try:
                self._data_type = self._obj.data_type
            except Exception:
                self._data_type = None
        return self._data_type

    def _get_num_children(self) -> int | None:
        """Returns the operand's ``num_children``, or None if the query raised."""
        if self._num_children is _UNSET:
            try:
                self._num_children = self._obj.num_children
            except Exception:
                self._num_children = None
        return self._num_children

    @property
    def is_matrix(self) -> bool:
        """True if the operand is an attribute holding a 4x4 matrix."""
        obj = self._obj
        if not isinstance(obj, Attribute):
            return False
        try:
            # a scalar or numeric compound kind never holds a matrix
            if obj.mobject.apiType() in _NON_MATRIX_ATTR_API_TYPES:
                return False
        except Exception:
            return False
        return self._get_data_type() == "matrix"

    @property
    def is_compound(self) -> bool:
        """True if the operand is an attribute with compound children."""
        if self._is_compound is _UNSET:
            self._is_compound = self._query_is_compound()
        return self._is_compound

    @property
    def is_quaternion(self) -> bool:
        """True if the operand is a compound attribute with 4 children."""
        return self.is_compound and self._get_num_children() == 4

    @property
    def is_vector(self) -> bool:
        """True if the operand is a compound attribute with 3 children."""
        return self.is_compound and self._get_num_children() == 3

    def _query_is_compound(self) -> bool:
        """The ``_is_compound`` rule, on the shared queries."""
        obj = self._obj
        if not isinstance(obj, Attribute):
            return False
        try:
            if obj.plug.isCompound:
                return True
            # a numeric / unit / enum / message kind is never a compound
            if obj.mobject.apiType() in _SCALAR_ATTR_API_TYPES:
                return False
        except Exception:
            return False
        num_children = self._get_num_children()
        if num_children is not None and num_children > 0:
            return True
        # generic-typed compound (e.g. choice.output typed by its input)
        return self._get_data_type() in _COMPOUND_DATA_TYPES


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