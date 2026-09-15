"""
Quaternion operations for the rig DSL.

Direct port of Eric Vignola's ``rig/quaternion/quaternion_functions.py``.
Public, quaternion-specific operations: ``add`` / ``multiply`` /
``subtract`` (Hamilton arithmetic) / ``negate`` / ``conjugate`` /
``to_vector`` / ``from_axis_angle`` / ``to_axis_angle``.

The cross-type operations ``normalize`` / ``inverse`` / ``angle`` /
``angle_degrees`` / ``to_euler`` / ``to_matrix`` / ``slerp`` live here as
public per-type functions; they are also reachable (operation-first) via
the top-level dispatch verbs exported from :mod:`rig`.

All public functions are :func:`vectorize` + :func:`memoize`.
Quaternion operations require the ``quatNodes`` Maya plugin -- loaded
lazily via the v1 NodeOp framework.
"""

from __future__ import annotations

from typing import Any, Optional

from maya import cmds
from rig._internal.container import container
from rig._internal.math_nodes import (
    _quaternion_add,
    _quaternion_multiply,
    _quaternion_subtract,
    _quaternion_to_euler,
    constant,
)
from rig._internal.memoize import memoize, vectorize
from rig._internal.types import _get_compound
from rig.matrix import compose
from rig.trigonometry import atan2, atan2d
from rig.vector import length


__all__ = [
    "add",
    "multiply",
    "subtract",
    "negate",
    "normalize",
    "inverse",
    "conjugate",
    "angle",
    "angle_degrees",
    "to_euler",
    "to_matrix",
    "to_vector",
    "slerp",
    "pow",
    "from_axis_angle",
    "to_axis_angle",
]


# --------------------------------------------------------------------- #
#  Arithmetic
# --------------------------------------------------------------------- #


@vectorize
@memoize
def add(quat1: Any, quat2: Any) -> Any:
    """``add(q1, q2)`` -- sum of two quaternions via ``quatAdd``."""
    return _quaternion_add(quat1, quat2)


@vectorize
@memoize
def multiply(quat1: Any, quat2: Any) -> Any:
    """``multiply(q1, q2)`` -- quaternion product via ``quatProd``."""
    return _quaternion_multiply(quat1, quat2)


@vectorize
@memoize
def subtract(quat1: Any, quat2: Any) -> Any:
    """``subtract(q1, q2)`` -- quaternion subtraction via ``quatSub``."""
    return _quaternion_subtract(quat1, quat2)


# --------------------------------------------------------------------- #
#  Unary
# --------------------------------------------------------------------- #


@vectorize
@memoize
def negate(quat: Any) -> Any:
    """``negate(q)`` -- ``quatNegate`` node."""
    container._load_plugin_for_quaternion()
    node = container.createNode("quatNegate", name="quat_negate1")
    node.inputQuat << quat
    return node.outputQuat


@vectorize
@memoize
def normalize(quat: Any) -> Any:
    """``normalize(q)`` -- ``quatNormalize`` node."""
    container._load_plugin_for_quaternion()
    node = container.createNode("quatNormalize", name="quat_normalize1")
    node.inputQuat << quat
    return node.outputQuat


@vectorize
@memoize
def inverse(quat: Any) -> Any:
    """``inverse(q)`` -- ``quatInvert`` node."""
    container._load_plugin_for_quaternion()
    node = container.createNode("quatInvert", name="quat_inverse1")
    node.inputQuat << quat
    return node.outputQuat


@vectorize
@memoize
def conjugate(quat: Any) -> Any:
    """``conjugate(q)`` -- ``quatConjugate`` node."""
    container._load_plugin_for_quaternion()
    node = container.createNode("quatConjugate", name="quat_conjugate1")
    node.inputQuat << quat
    return node.outputQuat


# --------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------- #


@vectorize
@memoize
def angle(quat1: Any, quat2: Any) -> Any:
    """``angle(q1, q2)`` -- angle (radians) of the shortest arc between
    two quaternions: ``2 * atan2(|vec(q1^-1*q2)|, w(q1^-1*q2))``."""

    with container("quat_angle1"):
        quat1 = container.publish_input(quat1, "input1")
        quat2 = container.publish_input(quat2, "input2")
        qd    = multiply(inverse(quat1), quat2)
        return container.publish_output(
            2 * atan2(length(to_vector(qd)), qd.outputQuatW), "output"
        )


@vectorize
@memoize
def angle_degrees(quat1: Any, quat2: Any) -> Any:
    """``angle_degrees(q1, q2)`` -- angle (degrees) of the shortest arc
    between two quaternions: ``2 * atan2d(|vec(q1^-1*q2)|, w(q1^-1*q2))``.

    Uses the degree-variant ``atan2d`` directly (no redundant
    radians->degrees round-trip); the leading ``2 *`` keeps the result a
    plain ``double``. Value-identical to the former ``degrees(angle(...))``
    under any scene angle unit."""

    with container("quat_angle_degrees1"):
        quat1 = container.publish_input(quat1, "input1")
        quat2 = container.publish_input(quat2, "input2")
        qd    = multiply(inverse(quat1), quat2)
        return container.publish_output(
            2 * atan2d(length(to_vector(qd)), qd.outputQuatW), "output"
        )


# --------------------------------------------------------------------- #
#  Conversions
# --------------------------------------------------------------------- #


@vectorize
@memoize
def to_euler(quat: Any, rotate_order: Optional[Any] = None) -> Any:
    """``to_euler(quat, rotate_order=)`` -- quat -> euler angles via
    ``quatToEuler`` honouring rotate order."""
    return _quaternion_to_euler(quat, rotate_order=rotate_order)


@vectorize
@memoize
def to_matrix(quat: Any) -> Any:
    """``to_matrix(quat)`` -- quat -> rotation matrix via ``composeMatrix``
    (with quaternion routed to ``inputQuat``)."""

    return compose(rotate=quat)


@vectorize
@memoize
def to_vector(quat: Any) -> Any:
    """``to_vector(quat)`` -- extract the 3-vector component (x, y, z) of
    a quaternion as a constant network."""
    with container("quat_to_vector1"):
        quat  = container.publish_input(quat, "input")
        attrs = _get_compound(quat)
        return container.publish_output(constant(attrs[:-1]), "output")


# --------------------------------------------------------------------- #
#  Interpolation
# --------------------------------------------------------------------- #


@vectorize
@memoize
def slerp(quat0: Any, quat1: Any, weight: Any = 0.5) -> Any:
    """``slerp(q0, q1, weight=0.5)`` -- spherical linear interpolation
    via ``quatSlerp``."""
    container._load_plugin_for_quaternion()
    with container("quat_slerp1"):
        quat0  = container.publish_input(quat0,  "input1")
        quat1  = container.publish_input(quat1,  "input2")
        weight = container.publish_input(weight, "weight")
        node   = container.createNode("quatSlerp", name="quat_slerp1")
        node.input1Quat << quat0
        node.input2Quat << quat1
        node.inputT     << weight
        return container.publish_output(node.outputQuat, "output")


@vectorize
@memoize
def pow(quat: Any, weight: Any = 0.5) -> Any:
    """``pow(q, t)`` -- fractional power of a unit quaternion.

    Equivalent to ``slerp(identity, q, t)``, built as a ``quatSlerp`` whose
    ``input1Quat`` is left at its default identity ``[0, 0, 0, 1]`` (verified),
    so no explicit identity node is needed. For a unit quaternion this is the
    exact quaternion power ``q**t`` (``q**0`` = identity, ``q**1`` = q,
    ``q**0.5`` = half rotation, ``q**-1`` = inverse). No clamp: ``t<0`` /
    ``t>1`` extrapolate.
    """
    container._load_plugin_for_quaternion()
    with container("quat_pow1"):
        quat   = container.publish_input(quat, "input")
        weight = container.publish_input(weight, "weight")
        node   = container.createNode("quatSlerp", name="quat_pow1")
        node.input2Quat << quat
        node.inputT     << weight
        return container.publish_output(node.outputQuat, "output")


# --------------------------------------------------------------------- #
#  Plugin-load helper attached to the container module for convenience.
#  (The v1 NodeOp framework already lazy-loads quatNodes for the *_op
#  helpers; raw createNode calls here need their own loadPlugin.)
# --------------------------------------------------------------------- #


def _load_plugin_for_quaternion_impl() -> None:
    """Idempotent lazy-load of the quatNodes plugin."""
    if not getattr(_load_plugin_for_quaternion_impl, "_loaded", False):
        cmds.loadPlugin("quatNodes", quiet=True)
        _load_plugin_for_quaternion_impl._loaded = True


# Attach to the container singleton so call-sites can call
# ``container._load_plugin_for_quaternion()`` symmetrically with
# ``container.createNode``.
container._load_plugin_for_quaternion = _load_plugin_for_quaternion_impl


# --------------------------------------------------------------------- #
#  v3.A -- axis-angle conversions (quatNodes plugin, all Maya)
# --------------------------------------------------------------------- #


@vectorize
@memoize
def from_axis_angle(axis_vec: Any, angle: Any) -> Any:
    """``from_axis_angle(axis, angle)`` -- build a quaternion from an
    axis (vec3) and an angle.

    ``angle`` is interpreted as Maya's standard ``doubleAngle`` (radians
    internally; pass ``math.radians(deg)`` if you have degrees).

    Uses the native ``axisAngleToQuat`` node from the ``quatNodes``
    plugin (auto-loaded if needed).
    """
    cmds.loadPlugin("quatNodes", quiet=True)
    with container("from_axis_angle1"):
        axis_vec = container.publish_input(axis_vec, "axis", at="double3")
        angle    = container.publish_input(angle, "angle")
        node     = container.createNode("axisAngleToQuat", name="from_axis_angle1")
        node.inputAxis  << axis_vec
        node.inputAngle << angle
        return container.publish_output(node.outputQuat, "output")


@vectorize
@memoize
def to_axis_angle(quat: Any) -> tuple:
    """``to_axis_angle(quat)`` -- decompose a quaternion into ``(axis, angle)``.

    Returns a 2-tuple ``(axis_plug, angle_plug)``. The angle uses Maya's
    standard ``doubleAngle`` (radians internally).

    Uses the native ``quatToAxisAngle`` node from the ``quatNodes``
    plugin (auto-loaded if needed).
    """
    cmds.loadPlugin("quatNodes", quiet=True)
    with container("to_axis_angle1"):
        quat = container.publish_input(quat, "input")
        node = container.createNode("quatToAxisAngle", name="to_axis_angle1")
        node.inputQuat << quat
        return (
            container.publish_output(node.outputAxis, "axis"),
            container.publish_output(node.outputAngle, "angle"),
        )