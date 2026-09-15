"""
Matrix operations for the rig DSL.

Direct port of Eric Vignola's ``rig/matrix/matrix_functions.py``.
Public, matrix-specific operations: ``decompose`` / ``transpose`` /
``compose`` / ``fourbyfour`` / ``aim`` (orthonormal orientation from an
aim + up vector) / ``multiply`` / ``add`` / ``determinant`` plus the
extraction helpers (``translation`` / ``rotation`` / ``scale_of`` /
``axis`` / ``column`` / ``row`` / ``transform_point`` /
``transform_vector``).

The cross-type operations ``inverse`` / ``lerp`` / ``slerp`` /
``normalize`` / ``to_euler`` / ``to_quaternion`` / ``dist`` live here as
public per-type functions; they are also reachable (operation-first) via
the top-level dispatch verbs exported from :mod:`rig`.

All public functions are :func:`vectorize` + :func:`memoize`.
"""

from __future__ import annotations

from typing import Any, Optional

from rig._internal.container import container
from rig._internal.generators import sequences
from rig._internal.math_nodes import (
    _compose_matrix,
    _decompose_matrix,
    _matrix_add,
    _matrix_inverse,
    _matrix_multiply,
)
from rig._internal.maya_version import is_at_least
from rig._internal.memoize import memoize, vectorize
from rig._internal.types import _get_compound, X, Y
from rig.functions import rev


__all__ = [
    # Decompose / convert
    "decompose",
    "inverse",
    "transpose",
    "to_quaternion",
    "to_euler",
    # Compose / build
    "compose",
    "fourbyfour",
    "aim",
    # Arithmetic
    "multiply",
    "add",
    "lerp",
    "blend",
    # Interpolation (orientation slerp / full blend) / orthonormalize / power
    "slerp",
    "pow",
    "normalize",
    # Determinant
    "determinant",
    # Extraction helpers
    "translation",
    "rotation",
    "scale_of",
    "axis",
    "column",
    "row",
    "transform_point",
    "transform_vector",
    # Distance
    "dist",
]


# --------------------------------------------------------------------- #
#  Decompose / convert
# --------------------------------------------------------------------- #


@vectorize
@memoize
def decompose(token: Any, rotate_order: Optional[Any] = None) -> Any:
    """``decompose(matrix)`` -- returns the ``decomposeMatrix`` Node so the
    caller can read ``.outputTranslate`` (default), ``.outputRotate``,
    ``.outputScale``, ``.outputShear`` or ``.outputQuat`` from it."""
    return _decompose_matrix(token, rotate_order=rotate_order)


@vectorize
@memoize
def inverse(token: Any) -> Any:
    """``inverse(matrix)`` -- ``inverseMatrix`` node."""
    return _matrix_inverse(token)


@vectorize
@memoize
def transpose(token: Any) -> Any:
    """``transpose(matrix)`` -- ``transposeMatrix`` node."""
    node = container.createNode("transposeMatrix")
    node.inputMatrix << token
    return node.outputMatrix


@vectorize
@memoize
def to_quaternion(token: Any) -> Any:
    """``to_quaternion(matrix)`` -- extract quat via decomposeMatrix."""
    return decompose(token).outputQuat


@vectorize
@memoize
def to_euler(token: Any, rotate_order: Optional[Any] = None) -> Any:
    """``to_euler(matrix, rotate_order=...)`` -- extract euler via
    decomposeMatrix honouring rotate order."""
    return decompose(token, rotate_order=rotate_order).outputRotate


# --------------------------------------------------------------------- #
#  Compose / build
# --------------------------------------------------------------------- #


@vectorize
@memoize
def compose(
    scale:        Optional[Any] = None,
    rotate:       Optional[Any] = None,
    translate:    Optional[Any] = None,
    shear:        Optional[Any] = None,
    rotate_order: Optional[Any] = None,
) -> Any:
    """``compose(scale=, rotate=, translate=, shear=, rotate_order=)``
    -- ``composeMatrix`` node. ``rotate`` may be a 3-channel euler or
    a 4-channel quaternion (auto-detected by channel count)."""
    with container("compose1"):
        if scale is not None:
            scale = container.publish_input(scale, "scale")
        if rotate is not None:
            rotate = container.publish_input(rotate, "rotate")
        if translate is not None:
            translate = container.publish_input(translate, "translate")
        if shear is not None:
            shear = container.publish_input(shear, "shear")
        return container.publish_output(
            _compose_matrix(
                scale        = scale,
                rotate       = rotate,
                translate    = translate,
                shear        = shear,
                rotate_order = rotate_order,
            ),
            "output",
        )


@vectorize
@memoize
def fourbyfour(
    x:        Optional[Any] = None,
    y:        Optional[Any] = None,
    z:        Optional[Any] = None,
    position: Optional[Any] = None,
) -> Any:
    """``fourbyfour(x=, y=, z=, position=)`` -- build a 4x4 from up to 4
    vectors (the matrix rows) via a ``fourByFourMatrix`` node. ``x`` /
    ``y`` / ``z`` fill rows 0-2 (the basis vectors) and ``position``
    fills row 3 (translation). Identity for any omitted row."""
    with container("fourbyfour1"):
        node = container.createNode("fourByFourMatrix", name="fourbyfour1")
        # (published-input name, source vec3, the three row plugs to fill)
        rows = (
            ("x", x, (node.in00, node.in01, node.in02)),
            ("y", y, (node.in10, node.in11, node.in12)),
            ("z", z, (node.in20, node.in21, node.in22)),
            ("position", position, (node.in30, node.in31, node.in32)),
        )
        for name, vec, plugs in rows:
            if vec is None:
                continue
            vec = container.publish_input(vec, name, at="double3")
            for plug, comp in sequences(plugs, _get_compound(vec)):
                plug << comp
        return container.publish_output(node.output, "output")


@vectorize
@memoize
def aim(
    aim_vector: Any,
    up_vector:  Any = Y,
    aim_axis:   Any = X,
    up_axis:    Any = Y,
) -> Any:
    """``aim(aim, up=Y, aim_axis=X, up_axis=Y)`` -- build an orthonormal
    orientation matrix from an aim vector and an up vector via the Maya
    ``aimMatrix`` node (like an aimConstraint).

    This is the lone primitive for vector-driven orientation; to get the
    rotation as an euler or quaternion, compose with the conversion verbs::

        to_euler(aim(a, u), rotate_order=0)
        to_quaternion(aim(a, u))
    """
    with container("matrix_aim1"):
        aim_vector = container.publish_input(aim_vector, "aim_vector", at="double3")
        up_vector  = container.publish_input(up_vector,  "up_vector",  at="double3")
        aim_axis   = container.publish_input(aim_axis,   "aim_axis",   at="double3")
        up_axis    = container.publish_input(up_axis,    "up_axis",    at="double3")
        node       = container.createNode("aimMatrix", name="matrix_aim1")
        node.primaryMode           << 1  # aim
        node.secondaryMode         << 2  # align
        node.primaryTargetVector   << aim_vector
        node.secondaryTargetVector << up_vector
        node.primaryInputAxis      << aim_axis
        node.secondaryInputAxis    << up_axis
        return container.publish_output(node.outputMatrix, "output")


# --------------------------------------------------------------------- #
#  Arithmetic
# --------------------------------------------------------------------- #


@vectorize
@memoize
def multiply(*tokens: Any, **kwargs: Any) -> Any:
    """``multiply(m0, m1, ..., local=False)`` -- ``multMatrix`` chain.
    If exactly two args and one is a vector, builds a
    ``pointMatrixMult`` instead. ``local`` flag controls
    ``vectorMultiply`` for that case."""
    return _matrix_multiply(*tokens, **kwargs)


@vectorize
@memoize
def add(*tokens: Any, **kwargs: Any) -> Any:
    """``add(m0, m1, ..., weights=[w0, w1, ...])`` -- ``addMatrix`` or
    ``wtAddMatrix`` if weights are provided."""
    return _matrix_add(*tokens, **kwargs)


@vectorize
@memoize
def lerp(input1: Any, input2: Any, weight: Any = 0.5) -> Any:
    """``lerp(m0, m1, weight=0.5)`` -- element-wise linear blend between
    two matrices via a ``wtAddMatrix`` with weights ``[1-w, w]``.

    This blends the raw matrix elements; for transform-aware
    interpolation (lerp translation, slerp rotation, elerp scale) use
    :func:`slerp`."""

    with container("matrix_lerp1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        weight = container.publish_input(weight, "weight")
        node   = container.createNode("wtAddMatrix")
        node.wtMatrix[0].matrixIn << input1
        node.wtMatrix[0].weightIn << rev(weight)
        node.wtMatrix[1].matrixIn << input2
        node.wtMatrix[1].weightIn << weight
        return container.publish_output(node.matrixSum, "output")


# --------------------------------------------------------------------- #
#  Interpolation (orientation slerp / full blend) / orthonormalize / power
# --------------------------------------------------------------------- #


@vectorize
@memoize
def blend(input1: Any, input2: Any, weight: Any = 0.5) -> Any:
    """``blend(m0, m1, weight=0.5)`` -- full-transform interpolation
    (T/R/S/shear) via Maya's native ``blendMatrix``: linear translate &
    scale, quaternion-slerp rotation (shortest path), native shear. A
    single weight drives all components. No clamp -- values outside
    ``[0, 1]`` extrapolate.

    For orientation-only spherical interpolation use :func:`slerp`."""
    with container("matrix_blend1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        weight = container.publish_input(weight, "weight")
        node   = container.createNode("blendMatrix", name="matrix_blend1")
        node.inputMatrix            << input1
        node.target[0].targetMatrix << input2
        node.target[0].weight       << weight
        return container.publish_output(node.outputMatrix, "output")


@vectorize
@memoize
def slerp(input1: Any, input2: Any, weight: Any = 0.5) -> Any:
    """``slerp(m0, m1, weight=0.5)`` -- ORIENTATION-ONLY spherical
    interpolation between two matrices. Returns a pure rotation matrix
    (translate=0, scale=1, shear=0) built from a quaternion-slerp of
    each input's rotation. No clamp: ``t<0`` / ``t>1`` extrapolate.

    For full-transform (T/R/S/shear) interpolation use :func:`blend`."""
    # Function-local import: ``matrix`` must NEVER import ``quaternion``
    # at module scope -- ``quaternion`` imports ``matrix`` up top, so a
    # module-scope back-edge would form an import cycle. ``matrix.dist``
    # is the established precedent for this pattern.
    from rig.quaternion import slerp as _q_slerp

    with container("matrix_slerp1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        weight = container.publish_input(weight, "weight")
        m0     = decompose(input1)
        m1     = decompose(input2)
        rotate = _q_slerp(m0.outputQuat, m1.outputQuat, weight=weight)
        return container.publish_output(compose(rotate=rotate), "output")


@vectorize
@memoize
def pow(input1: Any, weight: Any = 0.5) -> Any:
    """``pow(m, t)`` -- fractional transform toward identity (TRS blend).

    Equivalent to ``blend(identity, m, t)``, built as a ``blendMatrix`` whose
    ``inputMatrix`` is left at its default identity (verified), so no explicit
    identity node is needed: linear translate & scale, quaternion-slerp
    rotation, native shear. NOT a true matrix power. No clamp: ``t<0`` /
    ``t>1`` extrapolate.

    Intentionally shadows the built-in :func:`pow` so callers reading
    ``matrix.pow`` get this fractional-transform op; nothing in this
    module uses the built-in.
    """
    with container("matrix_pow1"):
        input1 = container.publish_input(input1, "input")
        weight = container.publish_input(weight, "weight")
        node   = container.createNode("blendMatrix", name="matrix_pow1")
        node.target[0].targetMatrix << input1
        node.target[0].weight       << weight
        return container.publish_output(node.outputMatrix, "output")


@vectorize
@memoize
def normalize(token: Any) -> Any:
    """``normalize(m)`` -- orthonormalize a transform matrix: keep its
    rotation and translation, drop scale and shear so the basis vectors
    are unit-length and orthogonal. Built via ``decompose`` ->
    ``compose`` (rotate + translate only), yielding the rigid-body part
    of the matrix."""
    with container("matrix_normalize1"):
        token = container.publish_input(token, "input")
        node  = decompose(token)
        return container.publish_output(
            compose(rotate=node.outputRotate, translate=node.outputTranslate),
            "output",
        )


# --------------------------------------------------------------------- #
#  Determinant
# --------------------------------------------------------------------- #


@vectorize
@memoize
def determinant(token: Any) -> Any:
    """``determinant(matrix)`` -- Maya 2024+ uses native ``determinant``
    node; older computes manually from the 3x3 rotation block via
    ``pointMatrixMult`` (vector x matrix in local mode)."""
    if is_at_least(2024):
        node = container.createNode("determinant", name="matrix_determinant1")
        node.input << token
        return node.output

    with container("matrix_determinant1"):
        token  = container.publish_input(token, "input")
        x_axis = multiply([1, 0, 0], token, local=True)
        y_axis = multiply([0, 1, 0], token, local=True)
        z_axis = multiply([0, 0, 1], token, local=True)
        x      = _get_compound(x_axis)
        y      = _get_compound(y_axis)
        z      = _get_compound(z_axis)
        return container.publish_output(
            x[0] * (y[1] * z[2] - y[2] * z[1])
            - x[1] * (y[0] * z[2] - y[2] * z[0])
            + x[2] * (y[0] * z[1] - y[1] * z[0]),
            "output",
        )


# --------------------------------------------------------------------- #
#  v3.A -- matrix decomposition / extraction helpers (Maya 2024+)
#
#  ``translation`` / ``rotation`` / ``scale_of`` use the native
#  ``translationFromMatrix`` / ``rotationFromMatrix`` / ``scaleFromMatrix``
#  nodes on Maya 2024+, with a ``decomposeMatrix``-based fallback on
#  older Maya. ``axis`` / ``column`` / ``row`` are 2024-only -- they
#  raise ``RuntimeError`` on older Maya (no equivalent native node and
#  hand-composing them is awkward).
# --------------------------------------------------------------------- #


@vectorize
@memoize
def translation(token: Any) -> Any:
    """``translation(matrix)`` -- extract the translate vec3 from a matrix.

    Maya 2024+ uses the native ``translationFromMatrix`` node; older
    falls back to ``decompose(matrix).outputTranslate``.
    """

    if is_at_least(2024):
        node = container.createNode("translationFromMatrix", name="translation1")
        node.input << token
        return node.output
    return decompose(token).outputTranslate


@vectorize
@memoize
def rotation(token: Any) -> Any:
    """``rotation(matrix)`` -- extract the rotate vec3 (degrees, XYZ
    rotate-order) from a matrix.

    Maya 2024+ uses the native ``rotationFromMatrix`` node; older falls
    back to ``decompose(matrix).outputRotate``.
    """

    if is_at_least(2024):
        node = container.createNode("rotationFromMatrix", name="rotation1")
        node.input << token
        return node.output
    return decompose(token).outputRotate


@vectorize
@memoize
def scale_of(token: Any) -> Any:
    """``scale_of(matrix)`` -- extract the scale vec3 from a matrix.

    Named ``scale_of`` (not ``scale``) to avoid collision with
    :func:`matrix.compose`'s ``scale=`` kwarg in import-star contexts.

    Maya 2024+ uses the native ``scaleFromMatrix`` node; older falls
    back to ``decompose(matrix).outputScale``.
    """

    if is_at_least(2024):
        node = container.createNode("scaleFromMatrix", name="scale_of1")
        node.input << token
        return node.output
    return decompose(token).outputScale


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "X": 0, "Y": 1, "Z": 2, 0: 0, 1: 1, 2: 2}


@vectorize
@memoize
def axis(token: Any, axis: Any = 0) -> Any:
    """``axis(matrix, axis=0)`` -- extract the X/Y/Z basis vector from a
    matrix's rotation portion. ``axis`` may be ``0/1/2`` or
    ``"x"/"y"/"z"``.

    Maya 2024+ uses the native ``axisFromMatrix`` node. Pre-2024 falls
    back to ``vectorProduct(operation=3)`` (Vector Matrix Product)
    transforming the matching unit vector -- a single node that ignores
    translation and yields the basis vector.
    """
    if axis not in _AXIS_INDEX:
        raise ValueError(f"axis must be 0/1/2 or 'x'/'y'/'z'; got {axis!r}")
    idx = _AXIS_INDEX[axis]

    with container("axis1"):
        token = container.publish_input(token, "matrix")
        if not is_at_least(2024):
            unit = (1, 0, 0) if idx == 0 else (0, 1, 0) if idx == 1 else (0, 0, 1)
            node = container.createNode("vectorProduct", name="axis1")
            node.operation << 3  # Vector Matrix Product (ignores translation)
            node.input1    << unit
            node.matrix    << token
            return container.publish_output(node.output, "output")

        node = container.createNode("axisFromMatrix", name="axis1")
        node.input << token
        node.axis  << idx
        return container.publish_output(node.output, "output")


@vectorize
@memoize
def column(token: Any, index: Any = 0) -> Any:
    """``column(matrix, index)`` -- extract a column (vec4) from a matrix.

    ``index`` is ``0..3``. Maya 2024+ ``columnFromMatrix`` node only --
    raises on older Maya.
    """

    if not is_at_least(2024):
        raise RuntimeError("matrix.column() requires Maya 2024+")

    with container("column1"):
        token = container.publish_input(token, "matrix")
        index = container.publish_input(index, "index")
        node  = container.createNode("columnFromMatrix", name="column1")
        node.matrix << token
        node.input  << index
        return container.publish_output(node.output, "output")


@vectorize
@memoize
def row(token: Any, index: Any = 0) -> Any:
    """``row(matrix, index)`` -- extract a row (vec3 / vec4) from a matrix.

    Maya stores matrices in row-major order, so:
      * row 0/1/2 = X/Y/Z basis vector (same as :func:`axis`)
      * row 3 = translation

    Maya 2024+ uses the native ``rowFromMatrix`` node and returns the
    raw row including the homogeneous w component. Pre-2024 falls back
    to :func:`axis` (rows 0-2) or :func:`translation` (row 3) for
    literal int indices. A Plug-valued ``index`` requires Maya 2024+.
    """
    if not is_at_least(2024):
        # Pre-2024 fallback handles literal int indices only.
        if isinstance(index, int):
            if index in (0, 1, 2):
                return axis(token, axis=index)
            if index == 3:
                return translation(token)
            raise ValueError(f"matrix.row index must be 0-3; got {index!r}")
        raise RuntimeError("matrix.row() with a Plug-valued index requires Maya 2024+")

    with container("row1"):
        token = container.publish_input(token, "matrix")
        index = container.publish_input(index, "index")
        node  = container.createNode("rowFromMatrix", name="row1")
        node.matrix << token
        node.input  << index
        return container.publish_output(node.output, "output")


@vectorize
@memoize
def transform_point(point: Any, token: Any) -> Any:
    """``transform_point(p, m)`` -- multiply a point (vec3) by a matrix
    *with* translation. Maya 2024+ uses native
    ``multiplyPointByMatrix``; older falls back to ``pointMatrixMult``.
    """

    with container("transform_point1"):
        point = container.publish_input(point, "point")
        token = container.publish_input(token, "matrix")
        if is_at_least(2024):
            node = container.createNode(
                "multiplyPointByMatrix", name="transform_point1"
            )
            node.input  << point
            node.matrix << token
            return container.publish_output(node.output, "output")
        # Pre-2024 fallback: pointMatrixMult.
        node = container.createNode("pointMatrixMult", name="transform_point1")
        node.inMatrix << token
        node.inPoint  << point
        return container.publish_output(node.output, "output")


@vectorize
@memoize
def transform_vector(vector_in: Any, token: Any) -> Any:
    """``transform_vector(v, m)`` -- multiply a vector (vec3) by a matrix
    *without* translation (rotation + scale only).

    Maya 2024+ uses native ``multiplyVectorByMatrix``. Pre-2024 falls
    back to ``vectorProduct(operation=3)`` (Vector Matrix Product),
    which also ignores the translation portion of the matrix.
    """
    with container("transform_vector1"):
        vector_in = container.publish_input(vector_in, "vector")
        token     = container.publish_input(token, "matrix")
        if not is_at_least(2024):
            node = container.createNode("vectorProduct", name="transform_vector1")
            node.operation << 3  # Vector Matrix Product
            node.input1    << vector_in
            node.matrix    << token
            return container.publish_output(node.output, "output")

        node = container.createNode("multiplyVectorByMatrix", name="transform_vector1")
        node.input  << vector_in
        node.matrix << token
        return container.publish_output(node.output, "output")


# --------------------------------------------------------------------- #
#  Distance
# --------------------------------------------------------------------- #


def dist(matrix1: Any, matrix2: Any) -> Any:
    """``dist(m1, m2)`` -- distance between the translation components
    of two matrices (re-export from :mod:`rig.functions`).

    Builds a single ``distanceBetween`` node fed via its ``inMatrix1``
    / ``inMatrix2`` slots -- Maya pulls the translation column in C++,
    so no extra ``decomposeMatrix`` / ``translationFromMatrix`` nodes
    are needed.

    Cross-type calls work naturally: ``dist(some_vec, some_matrix)``
    routes the vector to ``point1`` and the matrix to ``inMatrix2`` in
    the same node. Listed under both :mod:`rig.vector`
    (point-to-point semantics) and :mod:`rig.matrix`
    (transform-to-transform semantics) for discoverability -- same
    underlying memoised network either way.
    """
    # Imported function-locally: ``vector`` imports ``matrix`` at module
    # scope, so importing ``vector`` up top would form an import cycle.
    from rig.vector import dist as _vector_dist

    return _vector_dist(matrix1, matrix2)