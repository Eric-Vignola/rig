"""
Operation-first cross-type dispatch verbs for the rig DSL (Model A).

These are the operation-first public form for cross-type math: each verb
classifies its (first) argument with :func:`rig._internal.types.math_type`
and delegates to the matching public per-type function in the relevant
datatype module (which callers may also call directly when they already
know the operand type). The twelve verbs are re-exported flat from
:mod:`rig` so callers write::

    from rig import dist, lerp, slerp, blend, normalize, to_euler, to_matrix

Rules:
  * Unsupported ``(verb, type)`` pairs raise ``TypeError`` (no invented math).
  * ``lerp`` / ``elerp`` reject quaternion / euler inputs -- rotations must
    interpolate via ``slerp``.
  * ``to_matrix`` / ``to_euler`` / ``to_quaternion`` convert among the three
    rotation representations only; to build an orientation matrix from an
    aim + up vector use :func:`rig.matrix.aim`.
  * ``rotate_order`` (where accepted) is forwarded only to the
    euler-involving branch; the quaternion<->matrix conversions ignore it.

This module sits ABOVE matrix / vector / quaternion / euler in the import
DAG -- it imports them, none import it -- so composing their impls forms no
cycle. Per the codebase's sibling-import idiom, the per-type functions are
imported via their direct submodule path (never ``from rig import
matrix``, which would re-enter the half-initialised package).
"""

from __future__ import annotations

from typing import Any

from rig._internal.types import _is_list, math_type
from rig.euler import (
    slerp as _euler_slerp,
    to_matrix as _euler_to_matrix,
    to_quaternion as _euler_to_quaternion,
)
from rig.matrix import (
    blend as _matrix_blend,
    dist as _matrix_dist,
    inverse as _matrix_inverse,
    lerp as _matrix_lerp,
    normalize as _matrix_normalize,
    slerp as _matrix_slerp,
    to_euler as _matrix_to_euler,
    to_quaternion as _matrix_to_quaternion,
)
from rig.quaternion import (
    angle as _quaternion_angle,
    angle_degrees as _quaternion_angle_degrees,
    inverse as _quaternion_inverse,
    normalize as _quaternion_normalize,
    slerp as _quaternion_slerp,
    to_euler as _quaternion_to_euler,
    to_matrix as _quaternion_to_matrix,
)
from rig.vector import (
    angle as _vector_angle,
    angle_degrees as _vector_angle_degrees,
    dist as _vector_dist,
    elerp as _vector_elerp,
    lerp as _vector_lerp,
    normalize as _vector_normalize,
    slerp as _vector_slerp,
)


__all__ = [
    "dist",
    "lerp",
    "slerp",
    "blend",
    "elerp",
    "normalize",
    "inverse",
    "angle",
    "angle_degrees",
    "to_euler",
    "to_quaternion",
    "to_matrix",
]


def _classify(obj: Any) -> str:
    """Resolve the math type of ``obj`` for dispatch.

    A :class:`~rig.PlugList` is a *vectorised* argument: classify it
    by its first element (the per-type impl, being ``@vectorize``, broadcasts
    the whole list). Everything else goes straight to :func:`math_type`,
    which also handles plain Python literals (``[1, 2, 3]`` => ``"vector"``).
    """
    if _is_list(obj):
        return math_type(obj[0]) if len(obj) else "unknown"
    return math_type(obj)


def _unsupported(verb: str, math_kind: str, hint: str = "") -> None:
    """Raise a uniform ``TypeError`` for an unsupported ``(verb, type)``."""
    message = f"{verb}() does not support {math_kind} operands"
    if hint:
        message += f"; {hint}"
    raise TypeError(message)


# --------------------------------------------------------------------- #
#  Binary metric / interpolation verbs
# --------------------------------------------------------------------- #


def dist(a: Any, b: Any) -> Any:
    """``dist(a, b)`` -- Euclidean distance between two vectors or two
    matrices (translation-to-translation). Cross-type ``dist(vec, matrix)``
    works in the single underlying node."""
    kind = _classify(a)
    if kind == "vector":
        return _vector_dist(a, b)
    if kind == "matrix":
        return _matrix_dist(a, b)
    _unsupported("dist", kind)


def lerp(a: Any, b: Any, weight: Any = 0.5) -> Any:
    """``lerp(a, b, weight=0.5)`` -- linear blend. Supports scalar, vector
    (component-wise) and matrix (raw element blend). Rejects rotations --
    use :func:`slerp` for quaternion / euler."""
    kind = _classify(a)
    if kind in ("scalar", "vector"):
        return _vector_lerp(a, b, weight=weight)
    if kind == "matrix":
        return _matrix_lerp(a, b, weight=weight)
    if kind in ("euler", "quaternion"):
        _unsupported("lerp", kind, "use slerp() for rotations")
    _unsupported("lerp", kind)


def slerp(a: Any, b: Any, weight: Any = 0.5) -> Any:
    """``slerp(a, b, weight=0.5)`` -- spherical / rotation-aware blend.
    Supports vector, quaternion, euler and matrix (matrix = ORIENTATION-ONLY,
    returns a pure rotation matrix; use :func:`blend` for full T/R/S/shear)."""
    kind = _classify(a)
    if kind == "vector":
        return _vector_slerp(a, b, weight=weight)
    if kind == "quaternion":
        return _quaternion_slerp(a, b, weight=weight)
    if kind == "euler":
        return _euler_slerp(a, b, weight=weight)
    if kind == "matrix":
        return _matrix_slerp(a, b, weight=weight)
    _unsupported("slerp", kind)


def blend(a: Any, b: Any, weight: Any = 0.5) -> Any:
    """``blend(a, b, weight=0.5)`` -- full-transform interpolation. For a
    matrix this blends T/R/S/shear via ``blendMatrix`` (linear translate &
    scale, quaternion-slerp rotation, native shear; extrapolates). Other
    types are unsupported -- use ``slerp`` (orientation/rotation) or ``lerp``
    (component-wise) explicitly."""
    kind = _classify(a)
    if kind == "matrix":
        return _matrix_blend(a, b, weight=weight)
    _unsupported("blend", kind)


def elerp(a: Any, b: Any, weight: Any = 0.5) -> Any:
    """``elerp(a, b, weight=0.5)`` -- exponential blend ``a^(1-w) * b^w``
    (scalar or vector). Rejects rotations -- use :func:`slerp`."""
    kind = _classify(a)
    if kind in ("scalar", "vector"):
        return _vector_elerp(a, b, weight=weight)
    if kind in ("euler", "quaternion"):
        _unsupported("elerp", kind, "use slerp() for rotations")
    _unsupported("elerp", kind)


def angle(a: Any, b: Any) -> Any:
    """``angle(a, b)`` -- unsigned angle (radians) between two vectors or
    the shortest-arc angle between two quaternions."""
    kind = _classify(a)
    if kind == "vector":
        return _vector_angle(a, b)
    if kind == "quaternion":
        return _quaternion_angle(a, b)
    _unsupported("angle", kind)


def angle_degrees(a: Any, b: Any) -> Any:
    """``angle_degrees(a, b)`` -- :func:`angle` expressed in degrees."""
    kind = _classify(a)
    if kind == "vector":
        return _vector_angle_degrees(a, b)
    if kind == "quaternion":
        return _quaternion_angle_degrees(a, b)
    _unsupported("angle_degrees", kind)


# --------------------------------------------------------------------- #
#  Unary verbs
# --------------------------------------------------------------------- #


def normalize(x: Any) -> Any:
    """``normalize(x)`` -- unit-length for a vector / quaternion, or
    orthonormalise (drop scale + shear) for a matrix."""
    kind = _classify(x)
    if kind == "vector":
        return _vector_normalize(x)
    if kind == "quaternion":
        return _quaternion_normalize(x)
    if kind == "matrix":
        return _matrix_normalize(x)
    _unsupported("normalize", kind)


def inverse(x: Any) -> Any:
    """``inverse(x)`` -- matrix inverse or quaternion inverse."""
    kind = _classify(x)
    if kind == "matrix":
        return _matrix_inverse(x)
    if kind == "quaternion":
        return _quaternion_inverse(x)
    _unsupported("inverse", kind)


# --------------------------------------------------------------------- #
#  Rotation-representation conversions (euler <-> quaternion <-> matrix)
# --------------------------------------------------------------------- #


def to_euler(token: Any, rotate_order: Any = None) -> Any:
    """``to_euler(token, rotate_order=None)`` -- convert a quaternion or a
    matrix to euler angles honouring ``rotate_order``."""
    kind = _classify(token)
    if kind == "quaternion":
        return _quaternion_to_euler(token, rotate_order=rotate_order)
    if kind == "matrix":
        return _matrix_to_euler(token, rotate_order=rotate_order)
    _unsupported("to_euler", kind)


def to_quaternion(token: Any, rotate_order: Any = None) -> Any:
    """``to_quaternion(token, rotate_order=None)`` -- convert an euler
    (honouring ``rotate_order``) or a matrix to a quaternion."""
    kind = _classify(token)
    if kind == "euler":
        return _euler_to_quaternion(token, rotate_order=rotate_order)
    if kind == "matrix":
        return _matrix_to_quaternion(token)
    _unsupported("to_quaternion", kind)


def to_matrix(token: Any, rotate_order: Any = None) -> Any:
    """``to_matrix(token, rotate_order=None)`` -- convert an euler
    (honouring ``rotate_order``) or a quaternion to a rotation matrix.

    A vector input is rejected: to build an orientation matrix from an
    aim + up vector use :func:`rig.matrix.aim`."""
    kind = _classify(token)
    if kind == "euler":
        return _euler_to_matrix(token, rotate_order=rotate_order)
    if kind == "quaternion":
        return _quaternion_to_matrix(token)
    if kind == "vector":
        _unsupported(
            "to_matrix",
            kind,
            "build an orientation from aim/up vectors with matrix.aim()",
        )
    _unsupported("to_matrix", kind)