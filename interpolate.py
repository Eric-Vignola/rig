"""
Interpolation utilities for the rig DSL.

Direct port of Eric Vignola's ``rig/interpolate/interpolate_functions.py``.
Piecewise sampling (``sequence``), the Perlin smoothstep / smootherstep
curves, and ``inverse_lerp``.

The interpolation primitives ``lerp`` / ``slerp`` / ``blend`` /
``elerp`` are the top-level cross-type dispatch verbs (``from
rig import lerp, slerp, blend, elerp``); full transform-matrix
(T/R/S/shear) interpolation is the flat ``blend`` applied to a matrix
(matrix ``slerp`` is orientation only).

All public functions are :func:`vectorize` + :func:`memoize` (except
``sequence`` which only @memoize, since it operates on lists).
"""

from __future__ import annotations

import numbers
from typing import Any

from rig._internal.container import container, ContainerOptions
from rig._internal.maya_version import is_at_least
from rig._internal.memoize import memoize, vectorize
from rig._internal.types import _is_scalar_value
from rig.functions import choice, clamp, searchsorted
from rig.vector import lerp as _v_lerp


__all__ = ["sequence", "smoothstep", "smootherstep", "inverse_lerp"]


# --------------------------------------------------------------------- #
#  Piecewise sampling
# --------------------------------------------------------------------- #


@memoize
def sequence(x: Any, xp: Any, yp: Any, method: Any = _v_lerp) -> Any:
    """``sequence(x, xp, yp, method=lerp)`` -- sample a piecewise function.

    Args:
        x: x-coordinate at which to sample.
        xp: monotonically increasing x-coordinates of the data points.
        yp: y-coordinates of the data points (same length as ``xp``).
        method: interpolation method called as ``method(y0, y1, weight=w)``.
            Defaults to the flat :func:`rig.lerp` verb. Pass another
            verb (:func:`rig.slerp` for rotations,
            :func:`rig.elerp` for scale) or any ``tween`` easing
            curve for other behaviour.
    """

    with container("sequence1"):
        # Compute segment count BEFORE publishing -- ``xp[:-1]`` (negative
        # stop) doesn't work on a multi-Plug, so we use positive indices.
        n = len(xp) - 1
        x = container.publish_input(x, "x")
        # xp / yp are sequences of scalars -- explicitly type them as
        # scalar multi so _infer_attr_type doesn't ambiguously interpret
        # short lists as compound (vec3 / double4).
        xp = container.publish_input(xp, "xp", at="double", multi=True)
        yp = container.publish_input(yp, "yp", at="double", multi=True)
        # Locate the segment via searchsorted; clamp by slicing xp[:n].
        i  = searchsorted(xp[:n], x, return_index=True, side="left")
        x0 = choice(xp[:n],        selector=i)
        x1 = choice(xp[1 : n + 1], selector=i)
        y0 = choice(yp[:n],        selector=i)
        y1 = choice(yp[1 : n + 1], selector=i)

        weight = (x - x0) / (x1 - x0)
        return container.publish_output(method(y0, y1, weight=weight), "output")


# --------------------------------------------------------------------- #
#  Smoothstep / smootherstep
# --------------------------------------------------------------------- #


@vectorize
@memoize(foldable="scalar")
def smoothstep(
    input1: Any, input2: Any, weight: Any = 0.5, normalize: bool = False
) -> Any:
    """``smoothstep(a, b, weight=0.5, normalize=False)`` -- Perlin
    smoothstep ``3w^2 - 2w^3`` between ``a`` and ``b``.

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
        if not normalize:
            output = output * (input2 - input1) + input1
        return container.publish_output(output, "output")


@vectorize
@memoize(foldable="scalar")
def smootherstep(
    input1: Any, input2: Any, weight: Any = 0.5, normalize: bool = False
) -> Any:
    """``smootherstep(a, b, weight=0.5, normalize=False)`` -- Perlin
    smootherstep ``6w^5 - 15w^4 + 10w^3`` (continuous 1st AND 2nd
    derivatives at endpoints, unlike smoothstep which is only C1)."""

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
        if not normalize:
            output = output * (input2 - input1) + input1
        return container.publish_output(output, "output")


# --------------------------------------------------------------------- #
#  v3.A -- inverse_lerp (Maya 2024+ native, with scalar + Plug fallback)
# --------------------------------------------------------------------- #


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