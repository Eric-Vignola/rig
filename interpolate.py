"""
Interpolation utilities for the rig DSL.

Direct port of Eric Vignola's ``rig/interpolate/interpolate_functions.py``.
Piecewise sampling (``sequence``, and ``interp``: ``numpy.interp``, also
``rig.interp``), the Perlin smoothstep / smootherstep curves, and
``inverse_lerp``.

The interpolation primitives ``lerp`` / ``slerp`` / ``blend`` /
``elerp`` are the top-level cross-type dispatch verbs (``from
rig import lerp, slerp, blend, elerp``); full transform-matrix
(T/R/S/shear) interpolation is the flat ``blend`` applied to a matrix
(matrix ``slerp`` is orientation only).

All public functions are :func:`vectorize` + :func:`memoize` (except
``sequence`` which only @memoize, since it operates on lists, and ``interp``,
which takes numpy's list ``x`` itself and builds on ``sequence``).
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from rig._internal.container import _publish_tokens, container, ContainerOptions
from rig._internal.fuse import Fused
from rig._internal.journal import user_call as _user_call
from rig._internal.list import List
from rig._internal.maya_version import is_at_least
from rig._internal.memoize import memoize, vectorize
from rig._internal.operands import operands
from rig._internal.types import _is_scalar_value, math_type
from rig.functions import choice, clamp, searchsorted
from rig.nodetypes._base import Attribute
from rig.vector import lerp as _v_lerp


__all__ = ["sequence", "interp", "smoothstep", "smootherstep", "inverse_lerp"]


# --------------------------------------------------------------------- #
#  Piecewise sampling
# --------------------------------------------------------------------- #


_ENDS = ("extrapolate", "hold")


def _pick(test: Any, when_true: Any, when_false: Any) -> Any:
    """``when_true`` where ``test`` holds, else ``when_false``: picked in Python for
    a known test, else through a ``choice``, which keeps doubles where a
    ``condition`` node would round its values to floats."""
    if isinstance(test, (bool, np.bool_)):
        return when_true if test else when_false
    return choice([when_false, when_true], selector=test)


@operands(config=("method", "ends"))
@memoize
def sequence(x: Any, xp: Any, yp: Any, method: Any = _v_lerp, ends: Any = "extrapolate") -> Any:
    """``sequence(x, xp, yp, method=lerp, ends="extrapolate")`` -- sample a
    piecewise function.

    Args:
        x: x-coordinate at which to sample.
        xp: monotonically increasing x-coordinates of the data points.
        yp: y-coordinates of the data points (same length as ``xp``).
        method: interpolation method called as ``method(y0, y1, weight=w)``.
            Defaults to the flat :func:`rig.lerp` verb. Pass another
            verb (:func:`rig.slerp` for rotations,
            :func:`rig.elerp` for scale) or any ``tween`` easing
            curve for other behaviour.
        ends: what happens beyond ``xp[0]`` and ``xp[-1]``.
            ``"extrapolate"`` (the default) carries on along the end segments;
            ``"hold"`` keeps ``yp[0]`` / ``yp[-1]``, as ``numpy.interp`` does
            (``x`` is clamped to the ends). A plug holds while it is non-zero,
            so the mode can switch live (``ends=rail.rotateProjection == 2``).
    """
    # checked before the container: calling a str (a config-looking name, which
    # the operand check passes through) failed only once the network was built
    if not callable(method):
        raise TypeError(
            f"rig.interpolate.sequence() argument 'method': {method!r} is not "
            f"callable. Pass a verb such as rig.lerp or rig.slerp, or a tween "
            f"easing curve."
        )
    # a plug, or a fused value (rig.fuse() traces this body: `rig._internal.fuse`)
    live_ends = isinstance(ends, (Attribute, Fused))
    if not live_ends and ends not in _ENDS:
        raise ValueError(
            f"rig.interpolate.sequence() argument 'ends': {ends!r} is not one of "
            f"'extrapolate', 'hold' or a plug (non-zero holds)."
        )

    with container("sequence1"):
        # Compute segment count BEFORE publishing -- ``xp[:-1]`` (negative
        # stop) doesn't work on a multi-Plug, so we use positive indices.
        n = len(xp) - 1
        x = container.publish_input(x, "x")
        # xp / yp are typed by their elements (scalars or vectors), never
        # inferred: a short list of scalars would otherwise read as one
        # compound (vec3 / double4).
        xp = _publish_tokens(xp, "xp")
        yp = _publish_tokens(yp, "yp")
        # Holding the ends is sampling at x clamped to them: at xp[0] the first
        # segment's weight is 0, at xp[n] the last one's is 1, whatever the method.
        # A live switch goes through a choice, which keeps doubles (a condition
        # node holds floats).
        if live_ends:
            x = _pick(ends != 0, clamp(x, xp[0], xp[n]), x)
        elif ends == "hold":
            x = clamp(x, xp[0], xp[n])
        # Locate the segment via searchsorted; clamp by slicing xp[:n].
        i  = searchsorted(xp[:n], x, return_index=True, side="left")
        x0 = choice(xp[:n],        selector=i)
        x1 = choice(xp[1 : n + 1], selector=i)
        y0 = choice(yp[:n],        selector=i)
        y1 = choice(yp[1 : n + 1], selector=i)

        weight = (x - x0) / (x1 - x0)
        return container.publish_output(_user_call(method, y0, y1, weight=weight), "output")


@operands
def interp(x: Any, xp: Any, fp: Any, left: Any = None, right: Any = None, period: Any = None) -> Any:
    """``interp(x, xp, fp, left=None, right=None, period=None)`` -- ``numpy.interp``.

    The linear interpolant of the points ``(xp, fp)`` at ``x``, with numpy's
    semantics: beyond the ends it holds ``fp[0]`` / ``fp[-1]``, or ``left`` /
    ``right`` when given; ``period`` wraps ``x`` and ``xp`` (``left`` / ``right``
    are then not used). ``xp`` must be increasing; numpy does not check either.
    The values are scalars, as numpy's are: for vectors or another method use
    `sequence`, whose ``ends="hold"`` holds the ends the same way. ``x`` may be a
    list, which gives one value per ``x``.

    All numbers fold to ``numpy.interp``'s own result: a float, or a list of
    floats for a list ``x``. Otherwise this builds `sequence` with
    ``ends="hold"``, plus a ``choice`` for each ``left`` / ``right`` given (it
    keeps doubles, where a ``condition`` node rounds to floats), equal to
    numpy's within double rounding. ``period`` needs numeric ``xp``:
    numpy sorts them, and rig does when it builds the network.

    Raises:
        TypeError: if an ``fp`` value is not a scalar, or ``period`` is not a number.
        ValueError: if ``xp`` and ``fp`` differ in length or are empty, if
            ``period`` is 0, or if ``period`` is given with an ``xp`` plug.
    """
    many = isinstance(x, (list, tuple, List))
    xs, xp, fp = (list(x) if many else [x]), list(xp), list(fp)
    if len(xp) != len(fp):
        raise ValueError("rig.interp(): fp and xp are not of the same length.")
    if not xp:
        raise ValueError("rig.interp(): array of sample points is empty")
    for value in fp:
        if math_type(value) != "scalar":
            raise TypeError(
                f"rig.interp(): fp holds {value!r}, which is not a scalar; numpy's interp "
                f"takes scalars only. For vectors use rig.interpolate.sequence(x, xp, "
                f"yp, method=..., ends='hold')."
            )
    if period is not None:
        if isinstance(period, Attribute) or not isinstance(period, numbers.Real):
            raise TypeError(f"rig.interp(): period must be a number, got {period!r}")
        if period == 0:
            raise ValueError("rig.interp(): period must be a non-zero value")
        if not all(isinstance(v, numbers.Real) for v in xp):
            raise ValueError(
                "rig.interp(): period needs numeric xp: numpy sorts xp % period, and "
                "rig does it when it builds the network."
            )

    literal = all(isinstance(v, numbers.Real) for v in (*xs, *xp, *fp)) and all(
        v is None or isinstance(v, numbers.Real) for v in (left, right))
    if ContainerOptions.constant_folding and literal:
        out = np.interp(xs, xp, fp, left=left, right=right, period=period)
        return [float(v) for v in out] if many else float(out[0])

    if period is not None:
        # numpy's normalisation: xp % period sorted, one wrapped point at each end
        period = abs(period)
        pairs  = sorted(((v % period, f) for v, f in zip(xp, fp)), key=lambda p: p[0])
        xp     = [pairs[-1][0] - period] + [p[0] for p in pairs] + [pairs[0][0] + period]
        fp     = [pairs[-1][1]] + [p[1] for p in pairs] + [pairs[0][1]]
        left = right = None

    def one(value):
        if period is not None:
            # numpy's x % period is never negative; rig's % keeps the sign of x
            value = (value % period + period) % period
        out = fp[0] if len(xp) == 1 else sequence(value, xp, fp, method=_v_lerp, ends="hold")
        if left is not None:
            out = _pick(value < xp[0], left, out)
        if right is not None:
            out = _pick(value > xp[-1], right, out)
        return out

    return List([one(v) for v in xs]) if many else one(xs[0])


# --------------------------------------------------------------------- #
#  Smoothstep / smootherstep
# --------------------------------------------------------------------- #


@operands(config=("normalize",))
@vectorize
@memoize(foldable="scalar")
def smoothstep(
    input1: Any, input2: Any, weight: Any = 0.5, normalize: bool = False
) -> Any:
    """``smoothstep(a, b, weight=0.5, normalize=False)`` -- the smoothstep
    ``3x^2 - 2x^3`` of ``x``, where ``weight`` sits between the edges ``a``
    and ``b`` (``x = clamp((weight - a) / (b - a), 0, 1)``), as GLSL's
    ``smoothstep(a, b, weight)`` and Maya's ``smoothStep`` node.

    Returns that 0-1 value; ``normalize=True`` remaps it into ``[a, b]``
    (``x * (b - a) + a``). Literals, scalar plugs and vector plugs give the
    same answer.

    Maya 2024+ uses the native ``smoothStep`` node when inputs are scalar
    and ``normalize=False``. Otherwise builds the formula manually.
    """

    if ContainerOptions.constant_folding and all(
        isinstance(x, numbers.Real) for x in (weight, input1, input2)
    ):
        x = clamp((weight - input1) / (input2 - input1), 0.0, 1.0)
        x = x * x * (3.0 - 2.0 * x)
        if not normalize:
            return x
        return x * (input2 - input1) + input1

    if (
        is_at_least(2024)
        and all(_is_scalar_value(v) for v in (weight, input1, input2))
        and not normalize
    ):
        node = container.createNode("smoothStep", name="smoothstep1")
        node.input     << weight
        node.leftEdge  << input1
        node.rightEdge << input2
        return node.output

    with container("smoothstep1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        weight = container.publish_input(weight, "weight")
        x      = clamp((weight - input1) / (input2 - input1), 0.0, 1.0)
        output = x * x * (3.0 - 2.0 * x)
        if normalize:
            output = output * (input2 - input1) + input1
        return container.publish_output(output, "output")


@operands(config=("normalize",))
@vectorize
@memoize(foldable="scalar")
def smootherstep(
    input1: Any, input2: Any, weight: Any = 0.5, normalize: bool = False
) -> Any:
    """``smootherstep(a, b, weight=0.5, normalize=False)`` -- Perlin
    smootherstep ``6x^5 - 15x^4 + 10x^3`` of ``x``, where ``weight`` sits
    between the edges ``a`` and ``b`` (``x = clamp((weight - a) / (b - a),
    0, 1)``); continuous 1st AND 2nd derivatives at the edges, unlike
    smoothstep which is only C1.

    Returns that 0-1 value; ``normalize=True`` remaps it into ``[a, b]``,
    as `smoothstep` does."""

    if ContainerOptions.constant_folding and all(
        isinstance(x, numbers.Real) for x in (weight, input1, input2)
    ):
        x = clamp((weight - input1) / (input2 - input1), 0.0, 1.0)
        x = x * x * x * (x * (x * 6 - 15) + 10)
        if not normalize:
            return x
        return x * (input2 - input1) + input1

    with container("smootherstep1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        weight = container.publish_input(weight, "weight")
        x      = clamp((weight - input1) / (input2 - input1), 0.0, 1.0)
        output = x * x * x * (x * (x * 6 - 15) + 10)
        if normalize:
            output = output * (input2 - input1) + input1
        return container.publish_output(output, "output")


# --------------------------------------------------------------------- #
#  v3.A -- inverse_lerp (Maya 2024+ native, with scalar + Plug fallback)
# --------------------------------------------------------------------- #


@operands
@vectorize
@memoize(foldable="scalar")
def inverse_lerp(input1: Any, input2: Any, weight: Any) -> Any:
    """``inverse_lerp(a, b, x)`` -- return ``t`` such that
    ``lerp(a, b, t) == x``.

    Equivalent to ``(x - a) / (b - a)``. Maya 2024+ uses the native
    ``inverseLerp`` node; older Maya falls back to the explicit
    arithmetic.

    Useful for re-mapping a known value range to ``[0, 1]``::

        # Map distance in [0, 100] to a 0..1 weight.
        weight = inverse_lerp(0.0, 100.0, sphere.distance)
    """
    if ContainerOptions.constant_folding and all(
        isinstance(v, numbers.Real) for v in (input1, input2, weight)
    ):
        return (weight - input1) / (input2 - input1)

    if is_at_least(2024) and all(_is_scalar_value(v) for v in (input1, input2, weight)):
        node = container.createNode("inverseLerp", name="inverse_lerp1")
        node.input1        << input1
        node.input2        << input2
        node.interpolation << weight
        return node.output

    with container("inverse_lerp1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        weight = container.publish_input(weight, "weight")
        return container.publish_output((weight - input1) / (input2 - input1), "output")