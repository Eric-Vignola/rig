"""
Vector operations for the rig DSL.

Direct port of Eric Vignola's ``rig/vector/vector_functions.py``.
Public, vector-specific operations: the core products ``dot`` / ``cross``
/ ``length``, the scalar ``triple_product`` of three vectors, and
``rotate`` (rotate a vector by an euler).

The cross-type operations ``dist`` / ``lerp`` / ``slerp`` / ``elerp`` /
``normalize`` / ``angle`` / ``angle_degrees`` live here as public per-type
functions; they are also reachable (operation-first) via the top-level
dispatch verbs exported from :mod:`rig`
(``from rig import dist, lerp, slerp, ...``).

All public functions are :func:`vectorize` + :func:`memoize`.

Convention constants ``X``, ``Y``, ``Z`` (re-exported from
:mod:`rig._internal.types`) are the canonical unit-axis vectors.
"""

from __future__ import annotations

import numbers
from typing import Any

from maya import cmds
from rig._internal.container import container, ContainerOptions
from rig._internal.generators import sequences
from rig._internal.math_nodes import _constant, condition
from rig._internal.maya_version import is_at_least
from rig._internal.memoize import memoize, vectorize
from rig._internal.types import (
    _get_compound,
    _is_compound,
    _is_matrix,
    _is_matrix_literal,
    _is_real,
    _is_scalar_value,
    _is_sequence,
    X,
    Y,
    Z,
)
from rig.functions import rev
from rig.trigonometry import atan2, atan2d, sind


__all__ = [
    "X",
    "Y",
    "Z",
    "triple_product",
    "angle",
    "angle_degrees",
    "dot",
    "cross",
    "length",
    "normalize",
    "dist",
    "rotate",
    "lerp",
    "slerp",
    "elerp",
]


# --------------------------------------------------------------------- #
#  Triple product (scalar determinant of three vectors)
# --------------------------------------------------------------------- #


@vectorize
@memoize
def triple_product(x: Any, y: Any, z: Any) -> Any:
    """``triple_product(x, y, z)`` -- scalar triple product ``dot(x, cross(y, z))``,
    i.e. the determinant of the 3x3 matrix whose rows are the three
    vectors. Maya 2024+ uses the native ``determinant`` node fed from a
    ``fourByFourMatrix``; older computes it manually.

    (Distinct from :func:`rig.matrix.determinant`, which takes a
    single 4x4 matrix.)"""
    with container("vector_triple_product1"):
        x       = container.publish_input(x, "input1", at="double3")
        y       = container.publish_input(y, "input2", at="double3")
        z       = container.publish_input(z, "input3", at="double3")
        x_chans = _get_compound(x)
        y_chans = _get_compound(y)
        z_chans = _get_compound(z)

        if is_at_least(2024):
            m = container.createNode("fourByFourMatrix")
            for plug, comp in sequences([m.in00, m.in01, m.in02], x_chans):
                plug << comp
            for plug, comp in sequences([m.in10, m.in11, m.in12], y_chans):
                plug << comp
            for plug, comp in sequences([m.in20, m.in21, m.in22], z_chans):
                plug << comp
            node = container.createNode("determinant")
            node.input << m.output
            return container.publish_output(node.output, "output")

        # Legacy: expanded 3x3 determinant formula.
        return container.publish_output(
            x_chans[0] * (y_chans[1] * z_chans[2] - y_chans[2] * z_chans[1])
            - x_chans[1] * (y_chans[0] * z_chans[2] - y_chans[2] * z_chans[0])
            + x_chans[2] * (y_chans[0] * z_chans[1] - y_chans[1] * z_chans[0]),
            "output",
        )


# --------------------------------------------------------------------- #
#  Angle
# --------------------------------------------------------------------- #


@vectorize
@memoize
def angle(vector1: Any, vector2: Any) -> Any:
    """``angle(v1, v2)`` -- unsigned angle (radians) between two vectors:
    ``atan2(|v1 x v2|, v1 * v2)``.

    Robust across the full 0-pi range (including parallel / antiparallel)
    and independent of the scene's angle unit, for parity with
    :func:`rig.quaternion.angle`."""
    with container("vector_angle1"):
        vector1 = container.publish_input(vector1, "input1", at="double3")
        vector2 = container.publish_input(vector2, "input2", at="double3")
        return container.publish_output(
            atan2(length(cross(vector1, vector2)), dot(vector1, vector2)),
            "output",
        )


@vectorize
@memoize
def angle_degrees(vector1: Any, vector2: Any) -> Any:
    """``angle_degrees(v1, v2)`` -- unsigned angle (degrees) between two
    vectors: ``atan2d(|v1 x v2|, v1 * v2)``.

    Mirrors :func:`angle` (radians) but uses the degree-variant ``atan2d``
    so the native ``atan2`` node's ``doubleAngle`` output is returned
    directly -- no redundant radians->degrees round-trip. The output is a
    ``doubleAngle`` plug (idiomatic for the degree-variant trig ops); its
    value is identical to the former ``degrees(angle(...))`` under any
    scene angle unit."""
    with container("vector_angle_degrees1"):
        vector1 = container.publish_input(vector1, "input1", at="double3")
        vector2 = container.publish_input(vector2, "input2", at="double3")
        return container.publish_output(
            atan2d(length(cross(vector1, vector2)), dot(vector1, vector2)),
            "output",
        )


@vectorize
@memoize
def dot(vector1: Any, vector2: Any, normalize: bool = False) -> Any:
    """``dot(v1, v2, normalize=False)`` -- vector dot product.

    Maya 2024+ uses the native ``dotProduct`` node when ``normalize=False``
    (the native node has no normalize flag). Falls back to ``vectorProduct``
    op=1 otherwise.
    """
    if is_at_least(2024) and not normalize:
        node = container.createNode("dotProduct", name="dot1")
        node.input1 << vector1
        node.input2 << vector2
        return node.output

    node = container.createNode("vectorProduct", name="dot1")
    node.operation       << 1
    node.normalizeOutput << normalize
    node.input1          << vector1
    node.input2          << vector2
    return node.outputX


@vectorize
@memoize
def cross(vector1: Any, vector2: Any, normalize: bool = False) -> Any:
    """``cross(v1, v2, normalize=False)`` -- vector cross product.

    Maya 2024+ uses the native ``crossProduct`` node when ``normalize=False``.
    Falls back to ``vectorProduct`` op=2 otherwise.
    """
    if is_at_least(2024) and not normalize:
        node = container.createNode("crossProduct", name="cross1")
        node.input1 << vector1
        node.input2 << vector2
        return node.output

    node = container.createNode("vectorProduct", name="cross1")
    node.operation       << 2
    node.normalizeOutput << normalize
    node.input1          << vector1
    node.input2          << vector2
    return node.output


@vectorize
@memoize(
    foldable=lambda a, k: (
        len(a) == 1
        and not (_is_matrix(a[0]) or _is_matrix_literal(a[0]))
        and _is_sequence(a[0])
        and all(_is_real(x) for x in a[0])
    )
)
def length(vector_in: Any) -> Any:
    """``length(v)`` -- vector magnitude (Euclidean norm).

    Maya 2024+ uses the native ``length`` node when given a vec3. Falls
    back to ``distanceBetween`` (with origin = 0) for older Maya or matrix
    inputs (Plug OR literal -- translation magnitude).
    """
    # A matrix input -- Plug OR literal (flat-16 / nested 4x4) -- routes
    # through ``distanceBetween.inMatrix1`` (translation magnitude). Checked
    # BEFORE the pure-numeric short-circuit so a flat-16 literal is not
    # mistaken for a 16-element vector and reduced to its Frobenius norm.
    is_matrix_in = _is_matrix(vector_in) or _is_matrix_literal(vector_in)

    if (
        ContainerOptions.constant_folding
        and not is_matrix_in
        and _is_sequence(vector_in)
        and all(isinstance(x, numbers.Real) for x in vector_in)
    ):
        return sum(x**2 for x in vector_in) ** 0.5

    if is_at_least(2024) and not is_matrix_in and _is_compound(vector_in):
        node = container.createNode("length", name="length1")
        node.input << vector_in
        return node.output

    node = container.createNode("distanceBetween", name="length1")
    if is_matrix_in:
        node.inMatrix1 << vector_in
    else:
        node.point1 << vector_in
    return node.distance


@vectorize
@memoize
def normalize(vector_in: Any) -> Any:
    """``normalize(v)`` -- scale a vector to unit length.

    Maya 2024+ uses the native ``normalize`` node; older Maya uses
    ``v / length(v)`` with div-by-zero quieting.
    """
    if is_at_least(2024):
        node = container.createNode("normalize", name="normalize1")
        node.input << vector_in
        return node.output

    with container("normalize1"):
        vector_in = container.publish_input(vector_in, "input")
        magnitude = length(vector_in)
        div       = vector_in / magnitude
        # Quiet div-by-zero: if magnitude == 0, switch op to "no operation".
        div.operation << condition(magnitude == 0, 0, 2)
        return container.publish_output(div, "output")


@vectorize
@memoize
def dist(vector1: Any, vector2: Any) -> Any:
    """``dist(a, b)`` -- point-to-point or matrix-to-matrix distance via
    a single ``distanceBetween`` node."""
    node = container.createNode("distanceBetween", name="dist1")

    # input1 -- a matrix (Plug OR literal) feeds inMatrix1; else point1. The
    # ``_is_matrix_literal`` companion catches raw matrix lists that the
    # attribute-only ``_is_matrix`` misses. Kept inside the try/except so a
    # raising ``_is_matrix`` still degrades gracefully to point1.
    try:
        if _is_matrix(vector1) or _is_matrix_literal(vector1):
            node.inMatrix1 << vector1
        else:
            node.point1 << vector1
    except Exception:
        node.point1 << vector1

    # input2
    try:
        if _is_matrix(vector2) or _is_matrix_literal(vector2):
            node.inMatrix2 << vector2
        else:
            node.point2 << vector2
    except Exception:
        node.point2 << vector2

    return node.distance


# --------------------------------------------------------------------- #
#  v3.A -- vector.rotate (Maya 2024+ native rotateVector)
# --------------------------------------------------------------------- #


@vectorize
@memoize
def rotate(vector_in: Any, rotate_vec: Any, rotate_order: Any = 0) -> Any:
    """``rotate(v, euler, rotate_order=0)`` -- rotate a vector by an
    euler triple. ``rotate_order`` is ``0..5`` (XYZ, YZX, ...).

    Maya 2024+ uses the native ``rotateVector`` node. Pre-2024 falls
    back to ``composeMatrix`` (build a rotation matrix from the euler)
    + ``vectorProduct(operation=3)`` (transform the vector by it,
    ignoring translation). Two nodes vs. one, but functionally
    equivalent.
    """
    with container("rotate1"):
        vector_in  = container.publish_input(vector_in, "vector", at="double3")
        rotate_vec = container.publish_input(rotate_vec, "rotate", at="double3")
        if not is_at_least(2024):
            cm = container.createNode("composeMatrix", name="rotate_compose1")
            cm.inputRotate      << rotate_vec
            cm.inputRotateOrder << rotate_order
            vp = container.createNode("vectorProduct", name="rotate_vp1")
            vp.operation << 3  # Vector Matrix Product
            vp.input1    << vector_in
            vp.matrix    << cm.outputMatrix
            return container.publish_output(vp.output, "output")

        node = container.createNode("rotateVector", name="rotate1")
        node.input       << vector_in
        node.rotate      << rotate_vec
        node.rotateOrder << rotate_order
        return container.publish_output(node.output, "output")


# --------------------------------------------------------------------- #
#  Interpolation -- linear / spherical / exponential
# --------------------------------------------------------------------- #


@vectorize
@memoize(foldable="scalar")
def lerp(input1: Any, input2: Any, weight: Any = 0.5) -> Any:
    """``lerp(a, b, weight=0.5)`` -- linear interpolation
    ``(b - a) * weight + a``. Maya 2024+ uses the native ``lerp`` node
    when all inputs are scalar; older builds the formula via
    multiply / subtract / add."""
    if ContainerOptions.constant_folding and all(
        isinstance(x, numbers.Real) for x in (input1, input2, weight)
    ):
        return (input2 - input1) * weight + input1

    # The native 2024 ``lerp`` node only handles genuinely SCALAR inputs. A
    # raw vector literal (``[x, y, z]``) is neither a compound Plug nor a
    # scalar, so it must NOT take this path -- otherwise
    # ``node.input1 << [x, y, z]`` injects a sequence into a scalar attr and
    # raises. ``_is_scalar_value`` (= not compound AND not sequence) catches
    # the literal-vector case that an ``_is_compound``-only check (attribute-
    # only) misses; the compound path below then publishes it as a
    # ``double3`` / ``double4`` and lerps per channel.
    inputs_are_scalar = all(_is_scalar_value(x) for x in (input1, input2, weight))
    if is_at_least(2024) and inputs_are_scalar:
        node = container.createNode("lerp", name="lerp1")
        node.input1 << input1
        node.input2 << input2
        node.weight << weight
        return node.output

    # Compound inputs (vec3 lerp) or pre-2024 scalar -- use ``blendWeighted``
    # for cleaner output than arithmetic compose (1 node per channel
    # vs. ~4 multiply+sum nodes per channel).
    weight_is_real = isinstance(weight, numbers.Real)

    with container("lerp1"):
        input1       = container.publish_input(input1, "input1")
        input2       = container.publish_input(input2, "input2")
        weight       = container.publish_input(weight, "weight")
        w_complement = (1.0 - weight) if weight_is_real else (1 - weight)

        if not _is_compound(input1) and not _is_compound(input2):
            # Pre-2024 scalar: single blendWeighted node.
            node = container.createNode("blendWeighted", name="lerp1")
            node.input[0]  << input1
            node.input[1]  << input2
            node.weight[0] << w_complement
            node.weight[1] << weight
            result = node.output
        else:
            # Compound: per-channel blendWeighted, gathered into a vec output.
            c1      = _get_compound(input1)
            c2      = _get_compound(input2)
            outputs = []
            for a, b in sequences(c1, c2):
                node = container.createNode("blendWeighted", name="lerp1", ss=True)
                node.input[0]  << a
                node.input[1]  << b
                node.weight[0] << w_complement
                node.weight[1] << weight
                outputs.append(node.output)
            # Gather per-channel scalars into a vec output. Could be a
            # `PlugList(outputs)` here (saving 1 node) but that returns
            # a PlugList from `lerp`, which @vectorize-ed downstream
            # consumers like `compose(translate=...)` then fan out per-
            # channel -- producing N composeMatrix nodes instead of 1.
            # Until @vectorize learns to treat fixed-arity scalar
            # PlugLists as compound singletons (deferred to v4.S+),
            # keep the `_constant` aggregator here so the result stays
            # a single Plug.
            result    = _constant([0] * len(outputs), name="lerp_out1")
            node_name = str(result).split(".")[0]
            suffixes  = ["X", "Y", "Z", "W"]
            for idx, scalar_out in enumerate(outputs):
                cmds.connectAttr(
                    str(scalar_out),
                    f"{node_name}.value{suffixes[idx]}",
                    force=True,
                )

        return container.publish_output(result, "output")


@vectorize
@memoize
def slerp(input1: Any, input2: Any, weight: Any = 0.5) -> Any:
    """``slerp(a, b, weight=0.5)`` -- spherical linear interpolation
    between two vectors.

    Traces the ``a -> b`` minor great-circle arc (the angle reported by
    ``angleBetween`` in ``[0deg, 180deg]``) and blends magnitude along the
    way, implementing Eric's original vector-slerp formula::

        (a*sin((1-w)*theta) + b*sin(w*theta)) / sin(theta)

    on the **raw, unnormalized** inputs, where ``theta`` is the angle between
    ``a`` and ``b``. Correct for every angle 0deg-180deg and for inputs of
    any (including unequal) magnitude:

    * The arc always goes ``a -> b`` directly -- never the antipodal
      "shortest path" a quaternion ``slerp`` takes past 90deg.
    * Magnitude is interpolated, so ``slerp((0,0,-5), (10,0,0), 0.5)``
      gives ``(7.071, 0, -3.536)``, and a 120deg-apart pair lands in the
      correct quadrant rather than the mirror one.

    Degenerate inputs -- parallel (theta -> 0deg) or anti-parallel (theta -> 180deg) --
    make ``sin(theta) -> 0`` and the spherical formula undefined. A robust
    guard detects ``sin(theta) <= 1e-6`` (``sin`` is >= 0 on ``[0deg, 180deg]``, so
    one threshold catches BOTH ends) and falls back to a plain linear
    interpolation ``a*(1-w) + b*w`` -- always finite, never divides by
    zero.

    History: an earlier 2-node ``quatSlerp`` trick (pure-imaginary
    quaternions with W=0) matched this only for unit inputs at 0deg-90deg. It
    normalized its inputs and took the shortest path, flipping results
    into the wrong quadrant past 90deg and dropping magnitude blending.
    This restores the always-correct legacy network.

    For quaternions, prefer :func:`rig.quaternion.slerp`.
    """

    with container("slerp1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        weight = container.publish_input(weight, "weight")

        # theta = the angle between the two vectors. ``angleBetween.angle`` is a
        # ``doubleAngle`` plug, always in [0deg, 180deg] (the minor arc), so the
        # interpolation can never take the antipodal shortest path. Feed theta
        # straight into the native ``sin`` node via the degree-variant
        # ``sind``: ``angleBetween.angle`` -> ``sin.input`` is an angle->angle
        # connection, so NO radians()/degrees() conversion nodes are needed.
        ab = container.createNode("angleBetween", name="slerp_ab1")
        ab.vector1 << input1
        ab.vector2 << input2
        theta = ab.angle
        sine  = sind(theta)

        # Spherical term on the RAW (unnormalized) vectors so magnitude is
        # blended too: (a*sin((1-w)*theta) + b*sin(w*theta)) / sin(theta). The ``/ 1``
        # is a handle onto the divide node so we can override its operand
        # and operation below.
        spherical = (
            input1 * sind(rev(weight) * theta) + input2 * sind(weight * theta)
        ) / 1

        # Degenerate when sin(theta) ~= 0 (theta ~= 0deg parallel OR theta ~= 180deg anti-
        # parallel). sin is >= 0 across [0deg, 180deg], so a single ``<=``
        # threshold catches both ends.
        degenerate = sine <= 1e-6

        # Quiet the divide-by-zero: switch the divide node to a no-op
        # (operation 0) when degenerate, else divide (operation 2) by
        # sin(theta). The degenerate result is discarded by the outer
        # ``condition`` anyway, but this keeps Maya from emitting a runtime
        # warning / inf during DG evaluation.
        spherical.operation << condition(degenerate, 0, 2)
        spherical.input2    << sine

        # At a degenerate angle the great-circle arc is undefined; fall
        # back to a finite linear interpolation a*(1-w) + b*w.
        linear = input1 * rev(weight) + input2 * weight

        output = condition(degenerate, linear, spherical)
        return container.publish_output(output, "output")


@vectorize
@memoize(foldable="scalar")
def elerp(input1: Any, input2: Any, weight: Any = 0.5) -> Any:
    """``elerp(a, b, weight=0.5)`` -- exponential lerp
    ``a^(1-w) * b^w``. Useful for blending scale values."""

    if ContainerOptions.constant_folding and all(
        isinstance(x, numbers.Real) for x in (input1, input2, weight)
    ):
        return input1 ** rev(weight) * input2**weight

    with container("elerp1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        weight = container.publish_input(weight, "weight")
        return container.publish_output(
            input1 ** rev(weight) * input2**weight, "output"
        )