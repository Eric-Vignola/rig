"""
Trigonometric functions for the rig DSL.

Direct port of Eric Vignola's ``rig/trigonometry/trig_functions.py`` --
16 functions covering sin / cos / tan / asin / acos / atan / atan2 in
both radians and degrees, plus ``radians`` / ``degrees`` conversion.

Maya 2024+ ships dedicated single-input ``sin`` / ``cos`` / ``tan`` /
``asin`` / ``acos`` / ``atan`` / ``atan2`` nodes. Pre-2024 the same
behaviour is built from ``eulerToQuat`` (sin/cos), ``angleBetween``
(asin/acos/atan), or composed via formulas (atan2).

The version dispatch uses :class:`rig.NodeOp` -- declarative
``@op.impl(since=2024, scope='scalar')`` registrations for the modern
node, ``@op.impl(since=0, scope='scalar')`` for the legacy fallback.
The framework handles compound (``double3`` / ``double4``) fan-out
automatically.

All functions are :func:`vectorize` + :func:`memoize`. Pure-Python
short-circuit when the input is a literal number.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

from rig._internal.container import container, ContainerOptions
from rig._internal.math_nodes import _multiply_divide_op, condition, constant
from rig._internal.maya_version import is_at_least
from rig._internal.memoize import memoize, vectorize
from rig._internal.node_ops import NodeOp, SCOPE_SCALAR
from rig._internal.types import _get_compound
from rig.functions import abs as _abs, inf, rev


__all__ = [
    # Conversions
    "degrees",
    "radians",
    # sin / cos / tan (degrees + radians)
    "sind",
    "cosd",
    "tand",
    "sin",
    "cos",
    "tan",
    # inverse trig (degrees + radians)
    "asind",
    "asin",
    "acosd",
    "acos",
    "atand",
    "atan",
    "atan2d",
    "atan2",
]


# --------------------------------------------------------------------- #
#  Conversions
# --------------------------------------------------------------------- #


@vectorize
@memoize(foldable="scalar")
def degrees(token: Any) -> Any:
    """``degrees(x)`` -- radians -> degrees (multiply by ``180/pi``)."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.degrees(token)
    return _multiply_divide_op(token, (180.0 / math.pi), operation=1, name="degrees1")


@vectorize
@memoize(foldable="scalar")
def radians(token: Any) -> Any:
    """``radians(x)`` -- degrees -> radians (multiply by ``pi/180``)."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.radians(token)
    return _multiply_divide_op(token, (math.pi / 180.0), operation=1, name="radians1")


# --------------------------------------------------------------------- #
#  NodeOps -- modern (Maya 2024+) + legacy impls
# --------------------------------------------------------------------- #

# Each of these underlies the public sind/cosd/tand/asind/acosd/atand
# functions below (the degree variants -- closest to the native node
# semantics). The radian variants compose via degrees() / radians().


_sind_op = NodeOp("sind", requires_plugin="quatNodes")


@_sind_op.impl(since=2024, scope=SCOPE_SCALAR)
def _sind_modern(token: Any) -> Any:
    """Maya 2024+ ``sin`` node -- single scalar input/output."""
    node = container.createNode("sin", name="sin1")
    node.input << token
    return node.output


@_sind_op.impl(since=0, scope=SCOPE_SCALAR)
def _sind_legacy(token: Any) -> Any:
    """Pre-2024 sin via ``eulerToQuat.outputQuatX``.

    ``outputQuatX = sin(inputRotateX/2)``, so feeding ``2*x`` gives
    ``sin(x)``. Token is in degrees here.
    """
    node = container.createNode("eulerToQuat", name="sind1")
    node.inputRotateX << token * 2
    return node.outputQuatX


_cosd_op = NodeOp("cosd", requires_plugin="quatNodes")


@_cosd_op.impl(since=2024, scope=SCOPE_SCALAR)
def _cosd_modern(token: Any) -> Any:
    node = container.createNode("cos", name="cos1")
    node.input << token
    return node.output


@_cosd_op.impl(since=0, scope=SCOPE_SCALAR)
def _cosd_legacy(token: Any) -> Any:
    """``cos`` via ``eulerToQuat.outputQuatW = cos(inputRotateX/2)``."""
    node = container.createNode("eulerToQuat", name="cosd1")
    node.inputRotateX << token * 2
    return node.outputQuatW


_tand_op = NodeOp("tand")


@_tand_op.impl(since=2024, scope=SCOPE_SCALAR)
def _tand_modern(token: Any) -> Any:
    node = container.createNode("tan", name="tan1")
    node.input << token
    return node.output


_asind_op = NodeOp("asind", requires_plugin="quatNodes")


@_asind_op.impl(since=2024, scope=SCOPE_SCALAR)
def _asind_modern(token: Any) -> Any:
    node = container.createNode("asin", name="asin1")
    node.input << token
    return node.output


_acosd_op = NodeOp("acosd", requires_plugin="quatNodes")


@_acosd_op.impl(since=2024, scope=SCOPE_SCALAR)
def _acosd_modern(token: Any) -> Any:
    node = container.createNode("acos", name="acos1")
    node.input << token
    return node.output


_atand_op = NodeOp("atand")


@_atand_op.impl(since=2024, scope=SCOPE_SCALAR)
def _atand_modern(token: Any) -> Any:
    """Maya 2024+ ``atan`` node -- single scalar input/output.

    Returns ``atan.output`` directly (a ``doubleAngle`` plug). Consumer
    conversion (e.g. ``radians()`` in :func:`atan`) handles the
    doubleAngle -> double translation correctly via Maya's working-unit
    auto-conversion.
    """
    node = container.createNode("atan", name="atan1")
    node.input << token
    return node.output


_atan2d_op = NodeOp("atan2d")


@_atan2d_op.impl(since=2024, scope=SCOPE_SCALAR)
def _atan2d_modern(y: Any, x: Any) -> Any:
    node = container.createNode("atan2", name="atan2_1")
    node.input1 << y
    node.input2 << x
    return node.output


# --------------------------------------------------------------------- #
#  Public API -- sin / cos / tan and degree variants
# --------------------------------------------------------------------- #


@vectorize
@memoize(foldable="scalar")
def sind(token: Any) -> Any:
    """``sind(x)`` -- sine of ``x`` (degrees)."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.sin(math.radians(token))
    return _sind_op(token)


@vectorize
@memoize(foldable="scalar")
def cosd(token: Any) -> Any:
    """``cosd(x)`` -- cosine of ``x`` (degrees)."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.cos(math.radians(token))
    return _cosd_op(token)


@vectorize
@memoize(foldable="scalar")
def tand(token: Any) -> Any:
    """``tand(x)`` -- tangent of ``x`` (degrees).

    Maya 2024+ uses native ``tan`` node. Older falls back to
    ``sind(x) / cosd(x)`` with div-by-zero quieting.
    """
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.tan(math.radians(token))
    try:
        return _tand_op(token)
    except RuntimeError:
        # Legacy fallback -- compose from sind/cosd.
        with container("tand1"):
            token         = container.publish_input(token, "input")
            s             = sind(token)
            c             = cosd(token)
            div           = s / c
            div.operation = condition(c == 0, 0, 2)
            return container.publish_output(condition(c == 0, inf(), div), "output")


@vectorize
@memoize(foldable="scalar")
def sin(token: Any) -> Any:
    """``sin(x)`` -- sine of ``x`` (radians)."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.sin(token)
    with container("sin1"):
        token = container.publish_input(token, "input")
        return container.publish_output(sind(degrees(token)), "output")


@vectorize
@memoize(foldable="scalar")
def cos(token: Any) -> Any:
    """``cos(x)`` -- cosine of ``x`` (radians)."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.cos(token)
    with container("cos1"):
        token = container.publish_input(token, "input")
        return container.publish_output(cosd(degrees(token)), "output")


@vectorize
@memoize(foldable="scalar")
def tan(token: Any) -> Any:
    """``tan(x)`` -- tangent of ``x`` (radians)."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.tan(token)
    with container("tan1"):
        token = container.publish_input(token, "input")
        return container.publish_output(tand(degrees(token)), "output")


# --------------------------------------------------------------------- #
#  Inverse trig
# --------------------------------------------------------------------- #


@vectorize
@memoize(foldable="scalar")
def asind(token: Any) -> Any:
    """``asind(x)`` -- arc sine of ``x``, result in degrees.

    Maya 2024+ uses native ``asin``. Older uses an ``angleBetween``
    network (Eric's hack -- see ``_asind_legacy``).
    """
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.degrees(math.asin(token))
    try:
        return _asind_op(token)
    except RuntimeError:
        return _asind_legacy(token)


def _asind_legacy(token: Any) -> Any:
    """Pre-2024 ``asin`` via an ``angleBetween`` triangle-trick.

    Lazy-imports ``rev`` and ``abs`` from functions to avoid circular
    deps at module-load time.
    """

    with container("asind1"):
        token   = container.publish_input(token, "input")
        results = []
        for target in _get_compound(token):
            node = container.createNode("angleBetween")
            node.vector1 << 0
            node.vector2 << 0
            adj = rev(target * target) ** 0.5
            node.vector1X << adj
            node.vector1Y << target
            node.vector2X << condition(_abs(target) == 1, 1, adj)
            results.append(condition(target < 0, -node.angle, node.angle))
        if len(results) > 1:
            return container.publish_output(constant(results), "output")
        return container.publish_output(results[0], "output")


@vectorize
@memoize(foldable="scalar")
def asin(token: Any) -> Any:
    """``asin(x)`` -- arc sine of ``x``, result in radians."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.asin(token)
    with container("asin1"):
        token = container.publish_input(token, "input")
        return container.publish_output(radians(asind(token)), "output")


@vectorize
@memoize(foldable="scalar")
def acosd(token: Any) -> Any:
    """``acosd(x)`` -- arc cosine of ``x``, result in degrees."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.degrees(math.acos(token))
    try:
        return _acosd_op(token)
    except RuntimeError:
        return _acosd_legacy(token)


def _acosd_legacy(token: Any) -> Any:
    """Pre-2024 ``acos`` via an ``angleBetween`` triangle-trick."""

    with container("acosd1"):
        token   = container.publish_input(token, "input")
        results = []
        for target in _get_compound(token):
            node = container.createNode("angleBetween")
            node.vector1 << 0
            node.vector2 << 0
            adj = rev(target * target) ** 0.5
            node.vector1X << target
            node.vector1Y << adj
            node.vector2X << condition(_abs(target) == 1, 1, adj)
            results.append(node.angle)
        if len(results) > 1:
            return container.publish_output(constant(results), "output")
        return container.publish_output(results[0], "output")


@vectorize
@memoize(foldable="scalar")
def acos(token: Any) -> Any:
    """``acos(x)`` -- arc cosine of ``x``, result in radians."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.acos(token)
    with container("acos1"):
        token = container.publish_input(token, "input")
        return container.publish_output(radians(acosd(token)), "output")


@vectorize
@memoize(foldable="scalar")
def atand(token: Any) -> Any:
    """``atand(x)`` -- arc tangent of ``x``, result in degrees.

    Maya 2024+ uses native ``atan`` node (v3.T). Pre-2024 uses an
    ``angleBetween`` y-vs-x triangle.
    """
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.degrees(math.atan(token))

    try:
        return _atand_op(token)
    except RuntimeError:
        pass

    with container("atand1"):
        token   = container.publish_input(token, "input")
        results = []
        for target in _get_compound(token):
            node = container.createNode("angleBetween")
            node.vector1  << [1, 0, 0]
            node.vector2  << [1, 0, 0]
            node.vector1Y << target
            results.append(condition(target < 0, -node.angle, node.angle))
        if len(results) > 1:
            return container.publish_output(constant(results), "output")
        return container.publish_output(results[0], "output")


@vectorize
@memoize(foldable="scalar")
def atan(token: Any) -> Any:
    """``atan(x)`` -- arc tangent of ``x``, result in radians.

    Composes via ``radians(atand(x))``. On Maya 2024+, ``atand`` uses
    the native ``atan`` node (v3.T); on older Maya it uses an
    ``angleBetween`` triangle. Either way ``radians()`` handles the
    final degree -> radian conversion correctly via Maya's working-unit
    auto-conversion at the multiplyDivide consumer.
    """
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.atan(token)
    with container("atan1"):
        token = container.publish_input(token, "input")
        return container.publish_output(radians(atand(token)), "output")


def _atan2_legacy(y: Any, x: Any) -> Any:
    """Pre-2024 ``atan2(y, x)`` -- ``atan(y/x)`` with quadrant fix-ups,
    result in radians.

    Both :func:`atan2` and :func:`atan2d` route their pre-2024 path here
    rather than delegating to each other. Mutual delegation recurses until
    the stack gives out, and since ``RecursionError`` subclasses
    ``RuntimeError`` the ``except RuntimeError`` dispatch used elsewhere in
    this module swallows it and hands back a garbage network rather than
    failing.
    """
    div = y / x
    div.operation << condition(x == 0, 0, 2)  # quiet div-by-zero
    div = atan(div)

    out: Any = condition((x > 0), div, 0)
    out = condition(((x < 0) & (y >= 0)), div + math.pi, out)
    out = condition(((x < 0) & (y < 0)),  div - math.pi, out)
    out = condition(((x == 0) & (y > 0)), math.pi / 2,   out)
    out = condition(((x == 0) & (y < 0)), -math.pi / 2,  out)
    return out


@vectorize
@memoize(foldable="scalar")
def atan2d(y: Any, x: Any) -> Any:
    """``atan2d(y, x)`` -- principal value of arc tangent of ``y/x``,
    result in degrees. Maya 2024+ uses native ``atan2`` node; older
    converts the legacy radian network with ``degrees()``.
    """
    if (
        ContainerOptions.constant_folding
        and isinstance(y, numbers.Real)
        and isinstance(x, numbers.Real)
    ):
        return math.degrees(math.atan2(y, x))

    if is_at_least(2024):
        return _atan2d_op(y, x)

    with container("atan2d_1"):
        y = container.publish_input(y, "y")
        x = container.publish_input(x, "x")
        return container.publish_output(degrees(_atan2_legacy(y, x)), "output")


@vectorize
@memoize(foldable="scalar")
def atan2(y: Any, x: Any) -> Any:
    """``atan2(y, x)`` -- principal value of arc tangent of ``y/x``,
    result in radians.

    Maya 2024+ uses native ``atan2`` (then ``radians`` to convert the
    degree-output to radians). Older composes from ``atan(y/x)`` with
    quadrant fix-ups via ``condition``.
    """
    if (
        ContainerOptions.constant_folding
        and isinstance(y, numbers.Real)
        and isinstance(x, numbers.Real)
    ):
        return math.atan2(y, x)

    with container("atan2_1"):
        y = container.publish_input(y, "y")
        x = container.publish_input(x, "x")
        if is_at_least(2024):
            return container.publish_output(radians(atan2d(y, x)), "output")
        return container.publish_output(_atan2_legacy(y, x), "output")