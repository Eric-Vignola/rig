"""
Static-matrix decomposition + transform-channel routing for the ``<<`` operator.

When a user assigns a numpy / nested-list matrix to a transform channel,
we decompose the matrix in Python (via Maya's API) and route the relevant
component to the right channel. This complements ``_shorthand`` (which
builds live ``decomposeMatrix`` networks for *plug* sources) -- this module
handles the *static* numpy-array case.

Routing rules (matrix-shaped sources only):

============================  ============================================
Destination                   Behaviour
============================  ============================================
``transform.matrix``          Decompose in local space. Set ``.t``, ``.r``
                              (honouring ``rotateOrder``), ``.s``,
                              ``.shear``. For 3x3 inputs, ``.t`` is left
                              untouched.

``transform.worldMatrix``     Convert to local via
``transform.worldMatrix[*]``  ``M * parentInverseMatrix[*]``, then
                              decompose. For 3x3 inputs the existing
                              world translation is preserved.

``transform.t``               Set ``.t`` to translation component only.
``transform.r``               Set ``.r`` to euler honouring ``rotateOrder``.
``transform.s``               Set ``.s`` to scale component only.
``transform.shear``           Set ``.shear`` to shear component only.

quaternion compound           Set ``[x, y, z, w]`` quaternion (works on
(``inputQuat``, custom        any node -- built-in or custom Quat attr).
``Quat("foo")``)

non-transform matrix attr     Raw setAttr (Tier B coercion: ravel to 16).

scalar dst with matrix src    Returns ``False`` so caller raises a clean
                              shape ValueError via Tier A.
============================  ============================================

Per ``<<`` semantics, every setAttr disconnects any incoming connection
on the leaf first -- "you assign, you own."
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from rig.nodetypes._base import Attribute, PyNode
from rig._internal.types import _is_quaternion


LOGGER = logging.getLogger(__name__)


# Matrix attribute names we route through the local-space decomposition path.
_MATRIX_LOCAL_ALIASES = frozenset({"matrix", "m"})

# Matrix attribute names we route through the world-space (parent-inverse)
# decomposition path.
_MATRIX_WORLD_ALIASES = frozenset({"worldMatrix", "wm"})


# --------------------------------------------------------------------- #
#  Decomposition (Maya API)
# --------------------------------------------------------------------- #


def _decompose_matrix_complete(m4x4: np.ndarray, rotate_order: int = 0) -> dict:
    """Decompose a 4x4 numpy matrix into translation / euler / quaternion /
    scale / shear components, honouring ``rotate_order`` for euler.

    Returns a dict with keys: ``translation`` (3-list), ``euler``
    (3-list, degrees), ``quaternion`` (4-list, x/y/z/w), ``scale``
    (3-list), ``shear`` (3-list), ``rotate_order`` (int).

    Uses ``maya.api.OpenMaya.MTransformationMatrix`` -- no external dep.
    """
    if not (0 <= rotate_order <= 5):
        raise ValueError(f"rotate_order must be 0..5 (XYZ..ZYX), got {rotate_order!r}")
    flat = m4x4.flatten().tolist()
    if len(flat) != 16:
        raise ValueError(f"Expected 16 elements, got {len(flat)}")

    m_matrix = om.MMatrix(flat)
    m_xform  = om.MTransformationMatrix(m_matrix)

    trans = m_xform.translation(om.MSpace.kTransform)

    euler = m_xform.rotation(asQuaternion=False)
    if euler.order != rotate_order:
        # `reorderIt` re-derives angles in the requested order; merely
        # setting `.order` only re-labels.
        euler.reorderIt(rotate_order)
    angles_deg = [
        np.degrees(euler.x),
        np.degrees(euler.y),
        np.degrees(euler.z),
    ]

    quat  = m_xform.rotation(asQuaternion=True)
    scale = m_xform.scale(om.MSpace.kTransform)
    shear = m_xform.shear(om.MSpace.kTransform)

    return {
        "translation":  [trans.x, trans.y, trans.z],
        "euler":        angles_deg,
        "quaternion":   [quat.x, quat.y, quat.z, quat.w],
        "scale":        list(scale),
        "shear":        list(shear),
        "rotate_order": rotate_order,
    }


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #


def _node_is_transform(node: Any) -> bool:
    """Return True if ``node`` (Node, DGNode, or string) wraps a kTransform.

    Uses :meth:`DGNode.has_base_type` (canonical API) instead of raw
    MSelectionList / MFn introspection.
    """
    try:
        wrapped = node if hasattr(node, "has_base_type") else PyNode(str(node))
        return wrapped.has_base_type("transform")
    except (RuntimeError, ValueError):
        return False


def _resolve_rotate_order(node: Any) -> int:
    """Return the integer rotate-order (0..5) for ``node``, or 0 (XYZ) if
    the attribute doesn't exist.

    Uses :meth:`DGNode.find_attr` + :meth:`Attribute.get` (rename-safe API)
    rather than ``cmds.getAttr`` with string-formatted names.
    """
    try:
        wrapped = node if hasattr(node, "find_attr") else PyNode(str(node))
        return int(wrapped.find_attr("rotateOrder").get())
    except (RuntimeError, AttributeError):
        return 0


def _coerce_to_4x4(arr: np.ndarray) -> Optional[tuple]:
    """Coerce a matrix-shaped array into a (4, 4) numpy array.

    Returns:
        ``(m4x4, had_translation)`` where ``had_translation`` is True if the
        input was full 4x4 / 16-flat (i.e. carried a translation row), or
        False if the input was 3x3 / 9-flat (translation must be preserved
        from the destination).

        Returns ``None`` if the array isn't matrix-shaped.
    """
    if arr.ndim == 2 and arr.shape == (4, 4):
        return arr.astype(float).copy(), True
    if arr.ndim == 2 and arr.shape == (3, 3):
        m         = np.eye(4)
        m[:3, :3] = arr
        return m, False
    if arr.ndim == 1 and arr.size == 16:
        return arr.astype(float).reshape(4, 4), True
    if arr.ndim == 1 and arr.size == 9:
        m         = np.eye(4)
        m[:3, :3] = arr.reshape(3, 3)
        return m, False
    return None


# --------------------------------------------------------------------- #
#  Connection-clearing setAttr
# --------------------------------------------------------------------- #


def _disconnect_compound(plug_str: str) -> None:
    """Disconnect any incoming connections on ``plug_str`` and on each of
    its compound children.

    Used before setAttr on transform channels so an incoming live driver
    doesn't immediately overwrite the value (or block the setAttr outright).
    Uses :meth:`Attribute.get_connected_attrs` + :meth:`Attribute.disconnect`
    + :meth:`Attribute.num_children`/:meth:`Attribute.child` for compound
    enumeration (no string-parsing, no ``cmds.attributeQuery``).
    """
    try:
        root_attr = Attribute(plug_str)
    except Exception:
        return

    targets = [root_attr]
    # Walk compound children via the canonical Attribute API.
    try:
        if not root_attr.is_multi:
            for i in range(root_attr.num_children):
                targets.append(root_attr.child(i))
    except (RuntimeError, AttributeError, TypeError):
        pass  # Non-compound attr (e.g. matrix data) -- no children to enumerate.

    for t_attr in targets:
        try:
            srcs = t_attr.get_connected_attrs(src=True, dst=False) or []
        except RuntimeError:
            srcs = []
        for src in srcs:
            try:
                src.disconnect(t_attr)
            except RuntimeError:
                pass


def _setattr_disconnect(plug_str: str, *values: float) -> None:
    """Disconnect any incoming connections on ``plug_str`` (and children),
    then set the value via :meth:`Attribute.set` (canonical API).

    "You assign, you own" -- matches ``<<`` semantics.
    """
    _disconnect_compound(plug_str)
    try:
        plug_attr = Attribute(plug_str)
        plug_attr.set(*values)
    except (RuntimeError, Exception):
        pass


# --------------------------------------------------------------------- #
#  Transform channel set helpers (skip-translation aware)
# --------------------------------------------------------------------- #


def _apply_full_decomposition(
    node:             Any,
    components:       dict,
    skip_translation: bool = False,
) -> None:
    """Set ``.t``, ``.r``, ``.s``, ``.shear`` on ``node`` from the given
    decomposed ``components`` dict. Skips ``.t`` when ``skip_translation``
    is True (3x3 input -- preserve existing translation).

    Uses :meth:`DGNode.find_attr` (canonical API) to obtain attribute
    plugs by name rather than building ``f"{node}.t"`` strings.
    """
    wrapped = node if hasattr(node, "find_attr") else PyNode(str(node))

    def _set_channel(attr_name: str, values: Any) -> None:
        try:
            channel = wrapped.find_attr(attr_name, quiet=True)
        except (RuntimeError, AttributeError):
            channel = None
        if channel is None:
            return
        _disconnect_compound(channel.full_name)
        try:
            channel.set(*values)
        except (RuntimeError, Exception):
            pass

    if not skip_translation:
        _set_channel("t", components["translation"])
    _set_channel("r",     components["euler"])
    _set_channel("s",     components["scale"])
    _set_channel("shear", components["shear"])


# --------------------------------------------------------------------- #
#  Public router
# --------------------------------------------------------------------- #


def _try_matrix_source_routing(dst: Any, arr: np.ndarray) -> bool:
    """Attempt to route a matrix-shaped numpy source to a transform
    channel / matrix attribute / quaternion attribute.

    Args:
        dst: The destination :class:`rig.Plug`.
        arr: A numpy array of shape (3,3), (4,4), or 1-D size 9 / 16.

    Returns:
        True if the source was matrix-shaped AND the destination was
        recognised AND the assignment was applied.
        False otherwise (caller falls through to standard validation /
        injection paths).
    """
    coerced = _coerce_to_4x4(arr)
    if coerced is None:
        return False
    m4x4, had_translation = coerced

    attr     = dst.alias
    plug_str = str(dst)
    try:
        node = dst.node
    except Exception:
        return False
    node_str = str(node)

    is_transform = _node_is_transform(node_str)
    try:
        dst_data_type = dst.data_type
    except Exception:
        dst_data_type = None

    # Lazy import -- circular dep with _types at module-load time.

    # 1. Quaternion compound (built-in inputQuat, custom Quat("foo")).
    if _is_quaternion(dst):
        rot_order = _resolve_rotate_order(node_str) if is_transform else 0
        d         = _decompose_matrix_complete(m4x4, rot_order)
        _setattr_disconnect(plug_str, *d["quaternion"])
        return True

    # 2. Non-transform -- only matrix-typed dst is meaningful. Use the
    #    canonical ``Attribute.set()`` directly (Plug IS an Attribute,
    #    so no re-wrap needed) -- same path power-users would reach for
    #    via ``node.input.set([...], type='matrix')``.
    if not is_transform:
        if dst_data_type == "matrix":
            _disconnect_compound(plug_str)
            try:
                dst.set(*m4x4.ravel().tolist(), type="matrix")
            except RuntimeError as e:
                LOGGER.debug("matrix set on %s failed: %s", dst, e)
            return True
        return False  # let caller raise via Tier A shape mismatch

    # 3. Transform-side per-channel routing.
    rot_order = _resolve_rotate_order(node_str)

    # 3a. transform.matrix -- local-space decomposition.
    if attr in _MATRIX_LOCAL_ALIASES:
        d = _decompose_matrix_complete(m4x4, rot_order)
        _apply_full_decomposition(node_str, d, skip_translation=not had_translation)
        return True

    # 3b. transform.worldMatrix / worldMatrix[*] -- convert to local first.
    if attr in _MATRIX_WORLD_ALIASES or ".worldMatrix[" in plug_str:
        # Detect index (default 0).
        index = 0
        if ".worldMatrix[" in plug_str:
            try:
                index = int(plug_str.rsplit("[", 1)[1].rstrip("]"))
            except (ValueError, IndexError):
                index = 0
        try:
            wrapped         = PyNode(node_str)
            parent_inv_flat = wrapped.find_attr(f"parentInverseMatrix[{index}]").get()
        except (RuntimeError, AttributeError):
            parent_inv_flat = list(np.eye(4).ravel())
        parent_inv = om.MMatrix(parent_inv_flat)

        if had_translation:
            local_mm  = om.MMatrix(m4x4.flatten().tolist()) * parent_inv
            local_4x4 = np.array(local_mm).reshape(4, 4)
        else:
            # Preserve world translation by composing the existing world
            # translation into row 3 before re-localising.
            try:
                wrapped          = PyNode(node_str)
                world_flat       = wrapped.find_attr(f"worldMatrix[{index}]").get()
                existing_world_t = world_flat[12:15]
            except (RuntimeError, AttributeError):
                existing_world_t = [0.0, 0.0, 0.0]
            m4x4[3, :3] = existing_world_t
            local_mm    = om.MMatrix(m4x4.flatten().tolist()) * parent_inv
            local_4x4   = np.array(local_mm).reshape(4, 4)

        d = _decompose_matrix_complete(local_4x4, rot_order)
        _apply_full_decomposition(node_str, d, skip_translation=False)
        return True

    # 3c. Single-channel injections from a matrix source.
    d = _decompose_matrix_complete(m4x4, rot_order)
    if attr in ("translate", "t"):
        _setattr_disconnect(plug_str, *d["translation"])
        return True
    if attr in ("rotate", "r"):
        _setattr_disconnect(plug_str, *d["euler"])
        return True
    if attr in ("scale", "s"):
        _setattr_disconnect(plug_str, *d["scale"])
        return True
    if attr == "shear":
        _setattr_disconnect(plug_str, *d["shear"])
        return True

    # Transform but unrecognised matrix-receiving channel -- fall through.
    return False