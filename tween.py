"""
Easing curves for animation rigging.

Direct port of Eric Vignola's ``rig/tween/tween_functions.py`` -- 30+
easing equations from the standard Penner / Robert Tweener family.

Each function takes a normalised time value ``t`` and returns the
corresponding tweened value. Inputs outside ``[0, 1]`` are valid and
extrapolate per the underlying math (v3.S -- clamp removed):

* **Polynomial families** (linear/quad/cubic/quart/quint) extrapolate
  smoothly as ``t^N``. ``in_quad(2)`` returns ``4``; ``in_cubic(-0.5)``
  returns ``-0.125``.
* **Back / elastic / bounce** families have meaningful overshoot
  behavior outside ``[0, 1]`` -- useful for animation principles like
  anticipation (``t < 0``) and follow-through (``t > 1``).
* **Sine / expo / circ** extrapolate per their underlying functions.

If you need clamped behavior, pass clamped input explicitly::

    result = in_quad(clamp(weight, 0, 1))

Naming convention (snake_case adaptation of Eric's camelCase):

* ``in_*``    -- accelerate from zero velocity
* ``out_*``   -- decelerate to zero velocity
* ``in_out_*`` -- accelerate to halfway, then decelerate
* ``out_in_*`` -- decelerate to halfway, then accelerate

Variants:
``linear`` / ``quad`` / ``cubic`` / ``quart`` / ``quint`` /
``sine`` / ``expo`` / ``circ`` / ``elastic`` / ``back`` / ``bounce``.

All public functions are :func:`vectorize` + :func:`memoize`. Each
wraps its network in a ``container("...")`` with published ``input``
and ``output`` attributes (third_party.rig style) for clean grouping
in the Maya node editor and a uniform module-factory interface.

References:
    * https://kodi.wiki/view/Tweeners
    * https://easings.net/
"""

from __future__ import annotations

from typing import Any

from rig._internal.container import container
from rig._internal.math_nodes import condition
from rig._internal.memoize import memoize, vectorize
from rig.functions import pow, sqrt
from rig.trigonometry import cosd, sind


__all__ = [
    # Linear
    "in_linear",
    "out_linear",
    # Quadratic
    "in_quad",
    "out_quad",
    "in_out_quad",
    "out_in_quad",
    # Cubic
    "in_cubic",
    "out_cubic",
    "in_out_cubic",
    "out_in_cubic",
    # Quartic
    "in_quart",
    "out_quart",
    "in_out_quart",
    "out_in_quart",
    # Quintic
    "in_quint",
    "out_quint",
    "in_out_quint",
    "out_in_quint",
    # Sinusoidal
    "in_sine",
    "out_sine",
    "in_out_sine",
    "out_in_sine",
    # Exponential
    "in_expo",
    "out_expo",
    "in_out_expo",
    "out_in_expo",
    # Circular
    "in_circ",
    "out_circ",
    "in_out_circ",
    "out_in_circ",
    # Elastic
    "in_elastic",
    "out_elastic",
    "in_out_elastic",
    "out_in_elastic",
    # Back
    "in_back",
    "out_back",
    "in_out_back",
    "out_in_back",
    # Bounce
    "out_bounce",
    "in_bounce",
    "in_out_bounce",
    "out_in_bounce",
]


# --------------------------------------------------------------------- #
#  Linear
# --------------------------------------------------------------------- #


@vectorize
@memoize
def in_linear(t: Any) -> Any:
    """Linear ease-in (no easing) -- identity function (returns ``t``)."""
    with container("in_linear1"):
        t = container.publish_input(t, "input")
        return container.publish_output(t, "output")


@vectorize
@memoize
def out_linear(t: Any) -> Any:
    """Linear ease-out (no easing). Mirror of in_linear."""
    with container("out_linear1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_linear(1 - t), "output")


# --------------------------------------------------------------------- #
#  Quadratic (t^2)
# --------------------------------------------------------------------- #


@vectorize
@memoize
def in_quad(t: Any) -> Any:
    """Quadratic ease-in (t^2). Accelerating from zero velocity."""
    with container("in_quad1"):
        t = container.publish_input(t, "input")
        return container.publish_output(pow(t, 2), "output")


@vectorize
@memoize
def out_quad(t: Any) -> Any:
    """Quadratic ease-out (-t*(t-2)). Decelerating to zero velocity."""
    with container("out_quad1"):
        t = container.publish_input(t, "input")
        return container.publish_output(-t * (t - 2), "output")


@vectorize
@memoize
def in_out_quad(t: Any) -> Any:
    """Quadratic ease-in/out -- accel to halfway, then decel."""
    with container("in_out_quad1"):
        t       = container.publish_input(t, "input")
        lesser  = in_quad(t * 2) * 0.5
        greater = out_quad((t - 0.5) * 2) * 0.5 + 0.5
        return container.publish_output(condition(t < 0.5, lesser, greater), "output")


@vectorize
@memoize
def out_in_quad(t: Any) -> Any:
    """Quadratic ease-out/in -- decel to halfway, then accel."""
    with container("out_in_quad1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_quad(1 - t), "output")


# --------------------------------------------------------------------- #
#  Cubic (t^3)
# --------------------------------------------------------------------- #


@vectorize
@memoize
def in_cubic(t: Any) -> Any:
    """Cubic ease-in (t^3)."""
    with container("in_cubic1"):
        t = container.publish_input(t, "input")
        return container.publish_output(pow(t, 3), "output")


@vectorize
@memoize
def out_cubic(t: Any) -> Any:
    """Cubic ease-out ((t-1)^3 + 1)."""
    with container("out_cubic1"):
        t = container.publish_input(t, "input")
        return container.publish_output(pow((t - 1), 3) + 1, "output")


@vectorize
@memoize
def in_out_cubic(t: Any) -> Any:
    """Cubic ease-in/out.

    Note: the upstream Eric Vignola source uses ``in_quart`` /
    ``out_quart`` here -- assumed to be a copy-paste typo. This v2 port
    uses ``in_cubic`` / ``out_cubic`` for self-consistency.
    """
    with container("in_out_cubic1"):
        t       = container.publish_input(t, "input")
        lesser  = in_cubic(t * 2) * 0.5
        greater = out_cubic((t - 0.5) * 2) * 0.5 + 0.5
        return container.publish_output(condition(t < 0.5, lesser, greater), "output")


@vectorize
@memoize
def out_in_cubic(t: Any) -> Any:
    """Cubic ease-out/in."""
    with container("out_in_cubic1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_cubic(1 - t), "output")


# --------------------------------------------------------------------- #
#  Quartic (t^4)
# --------------------------------------------------------------------- #


@vectorize
@memoize
def in_quart(t: Any) -> Any:
    """Quartic ease-in (t^4)."""
    with container("in_quart1"):
        t = container.publish_input(t, "input")
        return container.publish_output(pow(t, 4), "output")


@vectorize
@memoize
def out_quart(t: Any) -> Any:
    """Quartic ease-out (1 - (t-1)^4)."""
    with container("out_quart1"):
        t = container.publish_input(t, "input")
        return container.publish_output(-1 * (pow((t - 1), 4) - 1), "output")


@vectorize
@memoize
def in_out_quart(t: Any) -> Any:
    """Quartic ease-in/out."""
    with container("in_out_quart1"):
        t       = container.publish_input(t, "input")
        lesser  = in_quart(t * 2) * 0.5
        greater = out_quart((t - 0.5) * 2) * 0.5 + 0.5
        return container.publish_output(condition(t < 0.5, lesser, greater), "output")


@vectorize
@memoize
def out_in_quart(t: Any) -> Any:
    """Quartic ease-out/in."""
    with container("out_in_quart1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_quart(1 - t), "output")


# --------------------------------------------------------------------- #
#  Quintic (t^5)
# --------------------------------------------------------------------- #


@vectorize
@memoize
def in_quint(t: Any) -> Any:
    """Quintic ease-in (t^5)."""
    with container("in_quint1"):
        t = container.publish_input(t, "input")
        return container.publish_output(pow(t, 5), "output")


@vectorize
@memoize
def out_quint(t: Any) -> Any:
    """Quintic ease-out ((t-1)^5 + 1)."""
    with container("out_quint1"):
        t = container.publish_input(t, "input")
        return container.publish_output(pow((t - 1), 5) + 1, "output")


@vectorize
@memoize
def in_out_quint(t: Any) -> Any:
    """Quintic ease-in/out."""
    with container("in_out_quint1"):
        t       = container.publish_input(t, "input")
        lesser  = in_quint(t * 2) * 0.5
        greater = out_quint((t - 0.5) * 2) * 0.5 + 0.5
        return container.publish_output(condition(t < 0.5, lesser, greater), "output")


@vectorize
@memoize
def out_in_quint(t: Any) -> Any:
    """Quintic ease-out/in."""
    with container("out_in_quint1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_quint(1 - t), "output")


# --------------------------------------------------------------------- #
#  Sinusoidal (sin(t))
# --------------------------------------------------------------------- #


@vectorize
@memoize
def in_sine(t: Any) -> Any:
    """Sine ease-in (1 - cos(t*90deg))."""
    with container("in_sine1"):
        t = container.publish_input(t, "input")
        return container.publish_output(-1 * cosd(t * 90) + 1, "output")


@vectorize
@memoize
def out_sine(t: Any) -> Any:
    """Sine ease-out (sin(t*90deg))."""
    with container("out_sine1"):
        t = container.publish_input(t, "input")
        return container.publish_output(sind(t * 90), "output")


@vectorize
@memoize
def in_out_sine(t: Any) -> Any:
    """Sine ease-in/out (-0.5 * (cos(180deg*t) - 1))."""
    with container("in_out_sine1"):
        t = container.publish_input(t, "input")
        return container.publish_output(-0.5 * (cosd(180 * t) - 1), "output")


@vectorize
@memoize
def out_in_sine(t: Any) -> Any:
    """Sine ease-out/in."""
    with container("out_in_sine1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_sine(1 - t), "output")


# --------------------------------------------------------------------- #
#  Exponential (2^t)
# --------------------------------------------------------------------- #


@vectorize
@memoize
def in_expo(t: Any) -> Any:
    """Exponential ease-in (2^(10*(t-1)))."""
    with container("in_expo1"):
        t = container.publish_input(t, "input")
        return container.publish_output(pow(2, 10 * (t - 1)), "output")


@vectorize
@memoize
def out_expo(t: Any) -> Any:
    """Exponential ease-out (1 - 2^(-10*t))."""
    with container("out_expo1"):
        t = container.publish_input(t, "input")
        return container.publish_output(-1 * pow(2, (-10 * t)) + 1, "output")


@vectorize
@memoize
def in_out_expo(t: Any) -> Any:
    """Exponential ease-in/out.

    Note: the upstream Eric Vignola source uses ``in_quint`` /
    ``out_quint`` here -- assumed to be a copy-paste typo. This v2 port
    uses ``in_expo`` / ``out_expo`` for self-consistency.
    """
    with container("in_out_expo1"):
        t       = container.publish_input(t, "input")
        lesser  = in_expo(t * 2) * 0.5
        greater = out_expo((t - 0.5) * 2) * 0.5 + 0.5
        return container.publish_output(condition(t < 0.5, lesser, greater), "output")


@vectorize
@memoize
def out_in_expo(t: Any) -> Any:
    """Exponential ease-out/in."""
    with container("out_in_expo1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_expo(1 - t), "output")


# --------------------------------------------------------------------- #
#  Circular (sqrt(1-t^2))
# --------------------------------------------------------------------- #


@vectorize
@memoize
def in_circ(t: Any) -> Any:
    """Circular ease-in (1 - sqrt(1 - t^2))."""
    with container("in_circ1"):
        t = container.publish_input(t, "input")
        return container.publish_output(-1 * sqrt(1 - (t * t)) + 1, "output")


@vectorize
@memoize
def out_circ(t: Any) -> Any:
    """Circular ease-out (sqrt(1 - (t-1)^2))."""
    with container("out_circ1"):
        t = container.publish_input(t, "input")
        return container.publish_output(sqrt(1 - pow((t - 1), 2)), "output")


@vectorize
@memoize
def in_out_circ(t: Any) -> Any:
    """Circular ease-in/out."""
    with container("in_out_circ1"):
        t       = container.publish_input(t, "input")
        lesser  = in_circ(t * 2) * 0.5
        greater = out_circ((t - 0.5) * 2) * 0.5 + 0.5
        return container.publish_output(condition(t < 0.5, lesser, greater), "output")


@vectorize
@memoize
def out_in_circ(t: Any) -> Any:
    """Circular ease-out/in."""
    with container("out_in_circ1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_circ(1 - t), "output")


# --------------------------------------------------------------------- #
#  Elastic (exponentially decaying sine wave)
# --------------------------------------------------------------------- #

# Standard elastic constants.
_ELASTIC_PERIOD = 0.3
_ELASTIC_S      = (_ELASTIC_PERIOD / 360) * 90


@vectorize
@memoize
def in_elastic(t: Any) -> Any:
    """Elastic ease-in (exponentially decaying sine wave)."""
    with container("in_elastic1"):
        t = container.publish_input(t, "input")
        u = t - 1
        return container.publish_output(
            -1 * pow(2, 10 * u) * sind((u - _ELASTIC_S) * 360 / _ELASTIC_PERIOD),
            "output",
        )


@vectorize
@memoize
def out_elastic(t: Any) -> Any:
    """Elastic ease-out (exponentially decaying sine wave)."""
    with container("out_elastic1"):
        t = container.publish_input(t, "input")
        return container.publish_output(
            pow(2, (-10 * t)) * sind((t - _ELASTIC_S) * 360 / _ELASTIC_PERIOD) + 1,
            "output",
        )


@vectorize
@memoize
def in_out_elastic(t: Any) -> Any:
    """Elastic ease-in/out."""
    with container("in_out_elastic1"):
        t       = container.publish_input(t, "input")
        lesser  = in_elastic(t * 2) * 0.5
        greater = out_elastic((t - 0.5) * 2) * 0.5 + 0.5
        return container.publish_output(condition(t < 0.5, lesser, greater), "output")


@vectorize
@memoize
def out_in_elastic(t: Any) -> Any:
    """Elastic ease-out/in."""
    with container("out_in_elastic1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_elastic(1 - t), "output")


# --------------------------------------------------------------------- #
#  Back (overshooting cubic)
# --------------------------------------------------------------------- #

# Standard back-easing overshoot constant.
_BACK_S = 1.70158


@vectorize
@memoize
def in_back(t: Any) -> Any:
    """Back ease-in (overshooting cubic: t^2*((s+1)*t - s)).

    Designed to overshoot -- the formula naturally extrapolates outside
    ``[0, 1]`` for anticipation / follow-through animation.
    """
    with container("in_back1"):
        t = container.publish_input(t, "input")
        return container.publish_output(t * t * ((_BACK_S + 1) * t - _BACK_S), "output")


@vectorize
@memoize
def out_back(t: Any) -> Any:
    """Back ease-out.

    Designed to overshoot -- the formula naturally extrapolates outside
    ``[0, 1]`` for anticipation / follow-through animation.
    """
    with container("out_back1"):
        t = container.publish_input(t, "input")
        u = t - 1
        return container.publish_output(
            u * u * ((_BACK_S + 1) * u + _BACK_S) + 1, "output"
        )


@vectorize
@memoize
def in_out_back(t: Any) -> Any:
    """Back ease-in/out."""
    with container("in_out_back1"):
        t       = container.publish_input(t, "input")
        lesser  = in_back(t * 2) * 0.5
        greater = out_back((t - 0.5) * 2) * 0.5 + 0.5
        return container.publish_output(condition(t < 0.5, lesser, greater), "output")


@vectorize
@memoize
def out_in_back(t: Any) -> Any:
    """Back ease-out/in."""
    with container("out_in_back1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_back(1 - t), "output")


# --------------------------------------------------------------------- #
#  Bounce (exponentially decaying parabolic)
# --------------------------------------------------------------------- #


@vectorize
@memoize
def out_bounce(t: Any) -> Any:
    """Bounce ease-out (exponentially decaying parabolic). Defined
    first because the in/in_out/out_in variants compose from this."""
    with container("out_bounce1"):
        t  = container.publish_input(t, "input")
        b1 = 7.5625 * t * t
        b2 = t - (1.5 / 2.75)
        b2 = 7.5625 * b2 * b2 + 0.75
        b3 = t - (2.25 / 2.75)
        b3 = 7.5625 * b3 * b3 + 0.9375
        b4 = t - (2.625 / 2.75)
        b4 = 7.5625 * b4 * b4 + 0.984375

        b3 = condition(t < (2.5 / 2.75), b3, b4)
        b2 = condition(t < (2 / 2.75), b2, b3)
        return container.publish_output(condition(t < (1 / 2.75), b1, b2), "output")


@vectorize
@memoize
def in_bounce(t: Any) -> Any:
    """Bounce ease-in (1 - out_bounce(1-t))."""
    with container("in_bounce1"):
        t = container.publish_input(t, "input")
        return container.publish_output(1 - out_bounce(1 - t), "output")


@vectorize
@memoize
def in_out_bounce(t: Any) -> Any:
    """Bounce ease-in/out."""
    with container("in_out_bounce1"):
        t       = container.publish_input(t, "input")
        lesser  = in_bounce(t * 2) * 0.5
        greater = out_bounce((t - 0.5) * 2) * 0.5 + 0.5
        return container.publish_output(condition(t < 0.5, lesser, greater), "output")


@vectorize
@memoize
def out_in_bounce(t: Any) -> Any:
    """Bounce ease-out/in."""
    with container("out_in_bounce1"):
        t = container.publish_input(t, "input")
        return container.publish_output(in_out_bounce(1 - t), "output")