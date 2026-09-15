"""
Euler-angle operations for the rig DSL.

Direct port of Eric Vignola's ``rig/euler/euler_functions.py``.
The one public, euler-specific operation is ``reorder`` (change an
euler's rotate order).

The cross-type operations ``to_matrix`` / ``to_quaternion`` / ``slerp``
live here as public per-type functions; they are also reachable
(operation-first) via the top-level dispatch verbs exported from
:mod:`rig`.

All public functions are :func:`vectorize` + :func:`memoize`.
"""

from __future__ import annotations

from typing import Any, Optional

from rig._internal.container import container
from rig._internal.math_nodes import _euler_to_quaternion, _quaternion_to_euler
from rig._internal.memoize import memoize, vectorize
from rig.matrix import compose


__all__ = [
    "reorder",
    "to_matrix",
    "to_quaternion",
    "slerp",
]


@vectorize
@memoize
def to_matrix(token: Any, rotate_order: Optional[Any] = None) -> Any:
    """``to_matrix(euler, rotate_order=)`` -- euler -> rotation matrix
    via ``composeMatrix``."""

    return compose(rotate=token, rotate_order=rotate_order)


@vectorize
@memoize
def to_quaternion(token: Any, rotate_order: Optional[Any] = None) -> Any:
    """``to_quaternion(euler, rotate_order=)`` -- euler -> quaternion via
    ``eulerToQuat`` honouring rotate order."""
    return _euler_to_quaternion(token, rotate_order=rotate_order)


@vectorize
@memoize
def reorder(token: Any, rotate_order0: Any, rotate_order1: Any) -> Any:
    """``reorder(euler, rotate_order0, rotate_order1)`` -- convert an euler
    from one rotate order (``rotate_order0``) to another
    (``rotate_order1``) by routing through quaternion space.

    Both rotate orders are MANDATORY positional arguments -- omitting
    either raises a plain ``TypeError`` (there is no default / no
    hand-rolled guard).
    """
    quat = to_quaternion(token, rotate_order=rotate_order0)
    return _quaternion_to_euler(quat, rotate_order=rotate_order1)


@vectorize
@memoize
def slerp(input1: Any, input2: Any, weight: Any = 0.5) -> Any:
    """``slerp(e0, e1, weight=0.5)`` -- interpolate two euler angles along
    the shortest great-circle arc by routing through quaternion space:
    euler -> quaternion -> quaternion-slerp -> euler (XYZ rotate order).

    The euler counterpart of :func:`rig.quaternion.slerp` /
    :func:`rig.matrix.slerp`."""
    # Function-local import: keep ``euler``'s module-scope dependency graph
    # free of ``quaternion`` (which already imports matrix/vector/trig) so a
    # future quaternion->euler edge can't form a cycle. ``matrix.slerp`` /
    # ``matrix.dist`` set this precedent.
    from rig.quaternion import slerp as _q_slerp

    with container("euler_slerp1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        weight = container.publish_input(weight, "weight")
        quat   = _q_slerp(to_quaternion(input1), to_quaternion(input2), weight=weight)
        return container.publish_output(_quaternion_to_euler(quat), "output")