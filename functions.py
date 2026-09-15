"""
Top-level math / aggregation utilities for the rig DSL.

Direct port of Eric Vignola's ``rig/functions.py`` adapted to the v1
language (NodeOps, ``container.createNode``, snake_case naming, the
``<<`` / ``>>`` operators, NumPy-style strict broadcast).

These functions all build small Maya node networks and return the
output :class:`Plug` for further chaining. Vectorised via
:func:`vectorize` and memoised via :func:`memoize` (where applicable).

Many function names shadow Python builtins (``int``, ``abs``, ``min``,
``max``, ``round``, ``floor``, ``ceil``, ``pow``, ``sum``, ``any``,
``all``). **Never** ``from rig.functions import *`` -- always
use a qualified import::

    from rig import functions as f
    f.abs(node.tx)
    f.clamp(node.tx, 0, 1)
"""

from __future__ import annotations

import builtins
import math
import numbers
from typing import Any, Sequence

from maya import cmds
from rig.maya.attribute import Attribute
from rig._internal.container import container, ContainerOptions
from rig._internal.list import PlugList
from rig._internal.math_nodes import (
    _constant,
    _multiply_divide_op,
    _plus_minus_average_op,
    condition,
)
from rig._internal.maya_version import is_at_least
from rig._internal.memoize import memoize, vectorize
from rig._internal.types import (
    _is_compound,
    _is_real,
    _is_scalar_value,
    _is_sequence,
)

# Public API surface. ``functions`` shadows ~11 Python builtins (``abs``,
# ``int``, ``round``, ``pow``, ``sum``, ``min``, ``max``, ``all``, ``any``,
# ``floor`` via ``math`` etc.), so it must NEVER be star-imported -- always
# ``from rig import functions as f``. ``__all__`` defines the surface
# and keeps the leaked stdlib / infra imports (``math``, ``cmds``,
# ``condition``, ``container``, ...) out of ``from functions import *``.
__all__ = [
    "frame",
    "clamp",
    "abs",
    "int",
    "round",
    "floor",
    "ceil",
    "trunc",
    "sum",
    "avg",
    "max",
    "min",
    "exp",
    "sign",
    "sqrt",
    "pow",
    "log",
    "choice",
    "rev",
    "searchsorted",
    "all",
    "any",
    "argmin",
    "argmax",
    "diff",
    "cumsum",
    "pi",
    "inf",
    "equal",
    "not_equal",
    "greater_than",
    "less_than",
    "greater_or_equal",
    "less_or_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
]


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #


def _sequence_to_node(obj: Any) -> Any:
    """Catch case where a sequence is passed where a Plug is expected.

    Wraps a literal sequence as a constant Plug; pass-through otherwise.
    """
    if _is_sequence(obj):
        return _constant(obj)
    return obj


# --------------------------------------------------------------------- #
#  Time / animation
# --------------------------------------------------------------------- #


@memoize
def frame() -> Any:
    """``frame()`` -- return a Plug that maps 1:1 with the timeline.

    Creates an ``animCurveTL`` node with linear keys at t=0 and t=1, with
    pre/post infinity set to linear. The output Plug ticks +1.0 per frame.
    """
    node = container.createNode("animCurveTL", name="frame1")
    cmds.setKeyframe(str(node), t=0, v=0.0)
    cmds.setKeyframe(str(node), t=1, v=1.0)
    cmds.keyTangent(str(node), e=True, itt="linear", ott="linear")
    node.preInfinity  << 4
    node.postInfinity << 4
    return node.o


# --------------------------------------------------------------------- #
#  Common scalar / compound math
# --------------------------------------------------------------------- #


@memoize(foldable="scalar")
def clamp(token: Any, min_value: Any, max_value: Any) -> Any:
    """``clamp(x, lo, hi)`` -- Maya 2024+ uses ``clampRange``; older falls
    back to nested ``condition`` nodes."""
    if ContainerOptions.constant_folding and builtins.all(
        isinstance(x, numbers.Real) for x in (token, min_value, max_value)
    ):
        return builtins.max(builtins.min(token, max_value), min_value)

    if is_at_least(2024) and builtins.all(
        _is_scalar_value(x) for x in (token, min_value, max_value)
    ):
        node = container.createNode("clampRange", name="clamp1")
        node.input   << token
        node.minimum << min_value
        node.maximum << max_value
        return node.output

    with container("clamp1"):
        token     = container.publish_input(token,     "input")
        min_value = container.publish_input(min_value, "min")
        max_value = container.publish_input(max_value, "max")
        less_than = condition(token < min_value, min_value, token)
        return container.publish_output(
            condition(token > max_value, max_value, less_than), "output"
        )


@vectorize
@memoize(foldable="scalar")
def abs(token: Any) -> Any:
    """``abs(x)`` -- Maya 2024+ uses ``absolute`` node; older uses
    ``condition(token < 0, -token, token)``."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return builtins.abs(token)

    if is_at_least(2024) and _is_scalar_value(token):
        node = container.createNode("absolute", name="abs1")
        node.input << token
        return node.output

    with container("abs1"):
        token = container.publish_input(token, "input")
        return container.publish_output(condition(token < 0, -token, token), "output")


@vectorize
@memoize(foldable="scalar")
def int(token: Any) -> Any:
    """``int(x)`` -- truncate-toward-zero via ``_constant`` long cast."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return builtins.int(token)

    with container("int1"):
        token = container.publish_input(token, "input")
        return container.publish_output(
            _constant(
                condition(token > 0, token - 0.4999999, token + 0.4999999),
                dtype="long",
            ),
            "output",
        )


@vectorize
@memoize(foldable="scalar")
def round(token: Any, digits: Any = 0) -> Any:
    """``round(x, digits=0)`` -- round to ``digits`` decimal places.

    Maya 2024+ uses the native ``round`` node (rounds-to-nearest-int).
    For ``digits != 0``, scales by ``10**digits`` first, rounds, then
    scales back. Older Maya falls back to a long-cast trick (multiply by
    scale, cast to long, divide back).

    The native path saves a node compared to the long-cast trick (1
    ``round`` node vs 1 ``network`` constant + 1 ``multiplyDivide``).
    """
    if ContainerOptions.constant_folding and builtins.all(
        isinstance(x, numbers.Real) for x in (token, digits)
    ):
        return builtins.round(token, digits)

    if is_at_least(2024) and _is_scalar_value(token):
        if isinstance(digits, numbers.Real) and digits == 0:
            # Cleanest case: native round, no scaling needed.
            node = container.createNode("round", name="round1")
            node.input << token
            return node.output
        if isinstance(digits, numbers.Real):
            # Scale, round, unscale -- still fewer nodes than the legacy
            # _constant + multiplyDivide pair.
            scale = 10**digits
            with container("round1"):
                token = container.publish_input(token, "input")
                node  = container.createNode("round", name="round1")
                node.input << token * scale
                return container.publish_output(node.output / scale, "output")

    # Legacy fallback (handles compound + non-numeric digits + pre-2024).
    with container("round1"):
        token = container.publish_input(token, "input")
        scale = 10**-digits
        return container.publish_output(
            _constant(token / scale, dtype="long") * scale, "output"
        )


@vectorize
@memoize(foldable="scalar")
def floor(token: Any) -> Any:
    """``floor(x)`` -- Maya 2024+ uses ``floor`` node; older uses long cast."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.floor(token)

    if is_at_least(2024) and _is_scalar_value(token):
        node = container.createNode("floor", name="floor1")
        node.input << token
        return node.output

    with container("floor1"):
        token = container.publish_input(token, "input")
        return container.publish_output(
            _constant(token - 0.4999999, dtype="long"), "output"
        )


@vectorize
@memoize(foldable="scalar")
def ceil(token: Any) -> Any:
    """``ceil(x)`` -- Maya 2024+ uses ``ceil`` node; older uses long cast."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.ceil(token)

    if is_at_least(2024) and _is_scalar_value(token):
        node = container.createNode("ceil", name="ceil1")
        node.input << token
        return node.output

    with container("ceil1"):
        token = container.publish_input(token, "input")
        return container.publish_output(
            _constant(token + 0.4999999, dtype="long"), "output"
        )


@vectorize
@memoize(foldable="scalar")
def trunc(token: Any) -> Any:
    """``trunc(x)`` -- toward-zero truncation. Maya 2024+ uses ``truncate``
    node; older falls back to ``condition(x<0, ceil(x), floor(x))``."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.trunc(token)

    if is_at_least(2024) and _is_scalar_value(token):
        node = container.createNode("truncate", name="trunc1")
        node.input << token
        return node.output

    with container("trunc1"):
        token = container.publish_input(token, "input")
        return container.publish_output(
            condition(token < 0, ceil(token), floor(token)), "output"
        )


# --------------------------------------------------------------------- #
#  Aggregators (operate on lists of tokens)
# --------------------------------------------------------------------- #


@memoize(foldable="reduce")
def sum(tokens: Sequence[Any]) -> Any:
    """``sum([t0, t1, ...])`` -- single ``plusMinusAverage`` node."""
    # Only short-circuit when EVERY input is a plain Python number.
    # ``_is_node`` returns False for Plug instances, so the previous
    # ``not any(_is_node(x))`` gate accidentally fed Plug strings into
    # ``builtins.sum`` (alphabetical mis-results).
    if ContainerOptions.constant_folding and builtins.all(_is_real(x) for x in tokens):
        return builtins.sum(tokens)
    return _plus_minus_average_op(*tokens, operation=1, name="sum1")


@memoize(foldable="reduce")
def avg(tokens: Sequence[Any]) -> Any:
    """``avg([t0, t1, ...])`` -- single ``plusMinusAverage`` node, op=3."""
    if ContainerOptions.constant_folding and builtins.all(
        isinstance(x, numbers.Real) for x in tokens
    ):
        return builtins.sum(tokens) / float(len(tokens))
    return _plus_minus_average_op(*tokens, operation=3, name="avg1")


@memoize(foldable="reduce")
def max(tokens: Sequence[Any]) -> Any:
    """``max([t0, t1, ...])`` -- Maya 2024+ uses ``max`` node; older folds
    via nested ``condition``."""
    # Only short-circuit when EVERY input is a plain Python number.
    if ContainerOptions.constant_folding and builtins.all(_is_real(x) for x in tokens):
        return builtins.max(tokens)
    if len(tokens) < 2:
        raise ValueError(f"max() requires >= 2 inputs; got {len(tokens)}")

    if is_at_least(2024) and not builtins.any(_is_compound(x) for x in tokens):
        node = container.createNode("max", name="max1")
        # ``input`` is a multi-attribute -- must index per element so each
        # connection lands in its own slot. Connecting to the parent
        # ``input`` plug repeatedly only fills slot 0 (last-write-wins).
        for i, item in enumerate(tokens):
            node.input[i] << item
        return node.output

    with container("max1"):
        tokens = container.publish_input(list(tokens), "input", at="double", multi=True)
        output = tokens[0]
        for obj in tokens[1:]:
            output = condition(output > obj, output, obj)
        return container.publish_output(output, "output")


@memoize(foldable="reduce")
def min(tokens: Sequence[Any]) -> Any:
    """``min([t0, t1, ...])`` -- Maya 2024+ uses ``min`` node; older folds
    via nested ``condition``."""
    # Only short-circuit when EVERY input is a plain Python number.
    if ContainerOptions.constant_folding and builtins.all(_is_real(x) for x in tokens):
        return builtins.min(tokens)
    if len(tokens) < 2:
        raise ValueError(f"min() requires >= 2 inputs; got {len(tokens)}")

    if is_at_least(2024) and not builtins.any(_is_compound(x) for x in tokens):
        node = container.createNode("min", name="min1")
        # See ``max`` for why per-element indexing is required.
        for i, item in enumerate(tokens):
            node.input[i] << item
        return node.output

    with container("min1"):
        tokens = container.publish_input(list(tokens), "input", at="double", multi=True)
        output = tokens[0]
        for obj in tokens[1:]:
            output = condition(output < obj, output, obj)
        return container.publish_output(output, "output")


# --------------------------------------------------------------------- #
#  Power / sign / sqrt
# --------------------------------------------------------------------- #


@vectorize
@memoize(foldable="scalar")
def exp(token: Any) -> Any:
    """``exp(x)`` -- e**x via ``multiplyDivide`` op=3."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return math.exp(token)
    return _multiply_divide_op(math.e, token, operation=3, name="exp1")


@vectorize
@memoize(foldable="scalar")
def sign(token: Any) -> Any:
    """``sign(x)`` -- ``-1`` for x<0, ``+1`` otherwise."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return builtins.bool(token >= 0) - builtins.bool(token < 0)
    with container("sign1"):
        token = container.publish_input(token, "input")
        return container.publish_output(condition(token < 0, -1, 1), "output")


@vectorize
@memoize(foldable="scalar")
def sqrt(token: Any) -> Any:
    """``sqrt(x)`` -- same as ``x ** 0.5``."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return token**0.5
    return _multiply_divide_op(token, 0.5, operation=3, name="sqrt1")


@vectorize
@memoize(foldable="scalar")
def pow(base: Any, exponent: Any) -> Any:
    """``pow(b, e)`` -- same as ``b ** e``."""
    if ContainerOptions.constant_folding and builtins.all(
        isinstance(x, numbers.Real) for x in (base, exponent)
    ):
        return base**exponent
    return _multiply_divide_op(base, exponent, operation=3, name="pow1")


# --------------------------------------------------------------------- #
#  Logic / selection
# --------------------------------------------------------------------- #
#
#  Vector / geometry helpers (``dot``, ``cross``, ``length``, ``normalize``,
#  ``dist``) now live in :mod:`rig.vector` -- they are vector datatype
#  operations, not generic math.


@memoize
def choice(tokens: Sequence[Any], selector: Any = None) -> Any:
    """``choice([t0, t1, ...], selector=plug)`` -- wraps a Maya ``choice``
    node, returning ``node.output``.

    Maya's ``choice`` node has polymorphic-typed input slots (any type
    is accepted; the actual type is resolved when the first connection
    lands). The DSL must inject all tokens via *connection* (not setAttr)
    so the slot type is established. Numeric / sequence literals are
    auto-wrapped in ``_constant()`` so they connect-as-output rather
    than failing on ``set()`` for the untyped slot. (v3.U fix.)
    """
    node = container.createNode("choice")
    if selector is not None:
        node.selector << selector
    for obj in tokens:
        obj = _sequence_to_node(obj)
        # Auto-wrap literals (int / float / list / tuple / numpy) in
        # _constant so the polymorphic ``choice.input[i]`` slot gets a
        # type via connection, not via setAttr (which would need the
        # slot's data_type -- undefined for untyped multi inputs).
        if not isinstance(obj, Attribute):
            obj = _constant(obj)
        node.input << obj
    return node.output


@vectorize
@memoize(foldable="scalar")
def rev(token: Any) -> Any:
    """``rev(x)`` -- ``1 - x``, via ``reverse`` node."""
    if ContainerOptions.constant_folding and isinstance(token, numbers.Real):
        return 1.0 - token

    node = container.createNode("reverse")
    # ``reverse`` has a compound ``input`` (double3); route compound Plugs
    # AND raw vector literals there. ``_is_scalar_value`` (not compound AND
    # not sequence) catches the literal ``[x, y, z]`` that an
    # ``_is_compound``-only check (attribute-only) would miss -- letting it
    # crash on scalar ``inputX``.
    if not _is_scalar_value(token):
        node.input << token
        return node.output
    node.inputX << token
    return node.outputX


# --------------------------------------------------------------------- #
#  Search / aggregate over lists
# --------------------------------------------------------------------- #


@memoize
def searchsorted(
    tokens:       Sequence[Any],
    query:        Any,
    return_index: bool          = True,
    side:         str           = "left",
) -> Any:
    """``searchsorted(sorted_tokens, query)`` -- like ``numpy.searchsorted``.

    Returns the insertion index (``return_index=True``) or the matching
    token (``return_index=False``) for ``query`` in a sorted list.
    """
    with container("searchsorted1"):
        n      = len(tokens)
        tokens = container.publish_input(list(tokens), "input", at="double", multi=True)
        if side == "left":
            if not return_index:
                result: Any = tokens[0]
                for t in tokens[1:]:
                    result = condition(t <= query, t, result)
            else:
                result = 0
                i      = 1
                for t in tokens[1:]:
                    result = condition(t <= query, i, result)
                    i += 1
        elif side == "right":
            if not return_index:
                result = tokens[n - 1]
                for t in list(tokens[: n - 1])[::-1]:
                    result = condition(t >= query, t, result)
            else:
                result = n - 1
                i      = n - 2
                for t in list(tokens[: n - 1])[::-1]:
                    result = condition(t >= query, i, result)
                    i -= 1
        else:
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        return container.publish_output(result, "output")


@memoize(foldable="reduce")
def all(tokens: Sequence[Any]) -> Any:
    """``all([t0, t1, ...])`` -- True iff every token is non-zero."""
    if ContainerOptions.constant_folding and builtins.all(
        isinstance(x, numbers.Real) for x in tokens
    ):
        return builtins.all(tokens)

    with container("all1"):
        n      = len(tokens)
        tokens = container.publish_input(list(tokens), "input", at="double", multi=True)
        # Slice with [:] so iteration works whether tokens is a list
        # (passthrough mode) or a published multi-attr Plug (active mode).
        # Plug[:] returns PlugList; list[:] returns a copy; both iterable.
        total = sum([(1 - (x == 0)) for x in tokens[:]])
        return container.publish_output(condition(total == n, True, False), "output")


@memoize(foldable="reduce")
def any(tokens: Sequence[Any]) -> Any:
    """``any([t0, t1, ...])`` -- True iff any token is non-zero."""
    if ContainerOptions.constant_folding and builtins.all(
        isinstance(x, numbers.Real) for x in tokens
    ):
        return builtins.any(tokens)

    with container("any1"):
        tokens = container.publish_input(list(tokens), "input", at="double", multi=True)
        # See ``all`` for why we slice with [:] before iterating.
        total = sum([(1 - (x == 0)) for x in tokens[:]])
        return container.publish_output(condition(total > 0, True, False), "output")


@memoize(foldable="reduce")
def argmin(tokens: Sequence[Any]) -> Any:
    """``argmin(tokens)`` -- index of the lowest value."""
    if ContainerOptions.constant_folding and builtins.all(
        isinstance(x, numbers.Real) for x in tokens
    ):
        return tokens.index(builtins.min(tokens))

    with container("argmin1"):
        tokens = container.publish_input(list(tokens), "input", at="double", multi=True)
        # Slice with [:] so ``min`` and per-element iteration work whether
        # tokens is a list (passthrough mode) or a published multi-attr
        # Plug (active mode).
        tokens = tokens[:]
        target = min(tokens)
        result: Any = 0
        for i, t in enumerate(tokens[1:]):
            result = condition(t == target, i + 1, result)
        return container.publish_output(result, "output")


@memoize(foldable="reduce")
def argmax(tokens: Sequence[Any]) -> Any:
    """``argmax(tokens)`` -- index of the highest value."""
    if ContainerOptions.constant_folding and builtins.all(
        isinstance(x, numbers.Real) for x in tokens
    ):
        return tokens.index(builtins.max(tokens))

    with container("argmax1"):
        tokens = container.publish_input(list(tokens), "input", at="double", multi=True)
        # See ``argmin`` for why we slice with [:] before calling ``max``
        # and iterating.
        tokens = tokens[:]
        target = max(tokens)
        result: Any = 0
        for i, t in enumerate(tokens[1:]):
            result = condition(t == target, i + 1, result)
        return container.publish_output(result, "output")


@memoize
def diff(tokens: Sequence[Any]) -> Any:
    """``diff(tokens)`` -- pairwise differences (``b - a``). Returns a
    :class:`PlugList`."""

    with container("diff1"):
        n       = len(tokens) - 1
        tokens  = container.publish_input(list(tokens), "input", at="double", multi=True)
        results = PlugList()
        for a, b in zip(tokens[:n], tokens[1 : n + 1]):
            results.append(b - a)
        return results


@memoize
def cumsum(tokens: Sequence[Any]) -> Any:
    """``cumsum(tokens)`` -- running cumulative sums. Returns a
    :class:`PlugList`."""

    with container("cumsum1"):
        tokens  = container.publish_input(list(tokens), "input", at="double", multi=True)
        results = PlugList([0 + tokens[0]])
        for obj in tokens[1:]:
            results.append(obj + results[-1])
        return results


# --------------------------------------------------------------------- #
#  v3.A -- new high-value math nodes (Maya 2024+)
# --------------------------------------------------------------------- #


@memoize
def pi() -> Any:
    """``pi()`` -- returns a Plug producing the constant pi.

    Maya 2024+ uses the native ``pi`` node. Older Maya falls back to
    a ``_constant`` (single ``network`` node holding ``math.pi``).

    Examples::

        node.tx << pi()                       # tx = 3.14159...
        node.tx << pi() * 0.5                 # tx = pi/2
    """
    if not is_at_least(2024):
        # Pre-2024 fallback: a network node holding math.pi as a constant.
        return _constant(math.pi, name="pi1")
    node = container.createNode("pi", name="pi1")
    return node.output


def inf() -> float:
    """``inf()`` -- positive floating-point infinity (``math.inf``).

    A companion constant to :func:`pi`. Unlike :func:`pi` (which builds a
    node), this returns a plain Python ``float`` -- it is used as a sentinel
    for div-by-zero quieting (e.g. ``trigonometry.tand`` returns ``inf()``
    when ``cos == 0``). Defined as a function so call sites don't reach for
    the ``math`` module each time.
    """
    return math.inf


@vectorize
@memoize(foldable="scalar")
def log(token: Any, base: Any = math.e) -> Any:
    """``log(x, base=e)`` -- natural log by default; pass ``base=10`` for
    base-10. Maya 2024+ native ``log`` node.

    Raises :class:`RuntimeError` on Maya < 2024 (no native node, no
    fallback equivalent of similar quality).
    """
    if (
        ContainerOptions.constant_folding
        and isinstance(token, numbers.Real)
        and isinstance(base, numbers.Real)
    ):
        return math.log(token, base)
    if not is_at_least(2024):
        raise RuntimeError("functions.log() requires Maya 2024+")
    node = container.createNode("log", name="log1")
    node.input << token
    node.base  << base
    return node.output


# --------------------------------------------------------------------- #
#  v3.C -- fuzzy float comparison
# --------------------------------------------------------------------- #


@vectorize
@memoize(foldable="scalar")
def equal(input1: Any, input2: Any, eps: Any = 1e-6) -> Any:
    """``equal(a, b, eps=1e-6)`` -- fuzzy float equality. Returns ``1.0``
    when ``|a - b| <= eps``, else ``0.0``.

    Maya 2024+ uses the native ``equal`` node which has built-in epsilon.
    Falls back to ``abs(a - b) <= eps`` on older Maya.
    """
    # Note: ``all`` is shadowed by ``functions.all`` (rig-DSL helper);
    # use ``builtins.all`` for the all-numeric short-circuit check.
    if ContainerOptions.constant_folding and builtins.all(
        isinstance(v, numbers.Real) for v in (input1, input2, eps)
    ):
        return 1.0 if builtins.abs(input1 - input2) <= eps else 0.0

    if not is_at_least(2024):
        with container("equal1"):
            input1 = container.publish_input(input1, "input1")
            input2 = container.publish_input(input2, "input2")
            return container.publish_output(abs(input1 - input2) <= eps, "output")

    if not (_is_scalar_value(input1) and _is_scalar_value(input2)):
        # Native equal is scalar-only; fall back to the compound expression
        # for compound Plugs AND raw vector literals (which _is_compound,
        # being attribute-only, would miss -- letting [x,y,z] crash the
        # scalar node).
        with container("equal1"):
            input1 = container.publish_input(input1, "input1")
            input2 = container.publish_input(input2, "input2")
            return container.publish_output(abs(input1 - input2) <= eps, "output")

    node = container.createNode("equal", name="equal1")
    node.input1  << input1
    node.input2  << input2
    node.epsilon << eps
    return node.output


# --------------------------------------------------------------------- #
#  Comparison wrappers (v3.P)
#
#  Named alternatives to the operator overloads on :class:`Plug`.
#  Both forms produce identical node networks -- these wrappers are for
#  artists who prefer functional naming or need greppable usage.
#
#  ``functions.equal(a, b)``         is equivalent to  ``a == b``
#  ``functions.not_equal(a, b)``     is equivalent to  ``a != b``
#  ``functions.greater_than(a, b)``  is equivalent to  ``a > b``
#  ``functions.less_than(a, b)``     is equivalent to  ``a < b``
#  ``functions.greater_or_equal``    is equivalent to  ``a >= b``
#  ``functions.less_or_equal``       is equivalent to  ``a <= b``
#
#  ``equal`` already exists above with a tolerance parameter (uses the
#  native Maya 2024+ ``equal`` node) -- those are intentionally distinct
#  from the strict-comparison ``==`` operator. The ``not_equal`` /
#  ``greater_*`` / ``less_*`` wrappers below are strict (no epsilon).
# --------------------------------------------------------------------- #


def not_equal(input1: Any, input2: Any) -> Any:
    """``not_equal(a, b)`` -- equivalent to ``a != b``.

    Strict comparison via the Maya ``condition`` node. For compound
    inputs, fans out per channel and returns a compound result.
    """
    return input1 != input2


def greater_than(input1: Any, input2: Any) -> Any:
    """``greater_than(a, b)`` -- equivalent to ``a > b``.

    Uses Maya 2024+ native ``greaterThan`` node when available, else
    falls back to ``condition``-node construction. For compound inputs,
    fans out per channel and returns a compound result.
    """
    return input1 > input2


def less_than(input1: Any, input2: Any) -> Any:
    """``less_than(a, b)`` -- equivalent to ``a < b``.

    Uses Maya 2024+ native ``lessThan`` node when available, else falls
    back to ``condition``-node construction. For compound inputs, fans
    out per channel and returns a compound result.
    """
    return input1 < input2


def greater_or_equal(input1: Any, input2: Any) -> Any:
    """``greater_or_equal(a, b)`` -- equivalent to ``a >= b``.

    Uses Maya's ``condition`` node (no native >= node exists). For
    compound inputs, fans out per channel and returns a compound result.
    """
    return input1 >= input2


def less_or_equal(input1: Any, input2: Any) -> Any:
    """``less_or_equal(a, b)`` -- equivalent to ``a <= b``.

    Uses Maya's ``condition`` node (no native <= node exists). For
    compound inputs, fans out per channel and returns a compound result.
    """
    return input1 <= input2


# --------------------------------------------------------------------- #
#  Logical wrappers (v3.P)
#
#  Numpy-style names -- ``and`` / ``or`` are Python keywords so the
#  wrappers use the ``logical_*`` prefix.
#
#  ``functions.logical_and(a, b)``  is equivalent to  ``a & b``
#  ``functions.logical_or(a, b)``   is equivalent to  ``a | b``
#  ``functions.logical_xor(a, b)``  is equivalent to  ``a ^ b``
# --------------------------------------------------------------------- #


def logical_and(input1: Any, input2: Any) -> Any:
    """``logical_and(a, b)`` -- equivalent to ``a & b``.

    Uses Maya 2024+ native ``and`` node when available, else falls back
    to ``((a != 0) + (b != 0)) == 2`` construction. For compound inputs,
    fans out per channel and returns a compound result.
    """
    return input1 & input2


def logical_or(input1: Any, input2: Any) -> Any:
    """``logical_or(a, b)`` -- equivalent to ``a | b``.

    Uses Maya 2024+ native ``or`` node when available, else falls back
    to ``((a != 0) + (b != 0)) > 0`` construction. For compound inputs,
    fans out per channel and returns a compound result.
    """
    return input1 | input2


def logical_xor(input1: Any, input2: Any) -> Any:
    """``logical_xor(a, b)`` -- equivalent to ``a ^ b``.

    Uses ``((a != 0) + (b != 0)) == 1`` construction (no native xor node
    in Maya). For compound inputs, fans out per channel and returns a
    compound result.
    """
    return input1 ^ input2


def logical_not(input1: Any) -> Any:
    """``logical_not(a)`` -- equivalent to ``~a``.

    Returns ``1`` for falsy input (zero), ``0`` for any truthy input.
    Uses Maya 2024+ native ``not`` node when available; falls back to
    ``condition(a != 0, 0, 1)`` on older Maya. For compound inputs,
    fans out per channel and returns a compound result.

    Note: pre-v3.T the ``~`` operator built ``1 - x`` (subtract node)
    which gave correct results for boolean inputs but wrong values for
    non-boolean inputs (``~3.5 = -2.5``). v3.T switched ``~`` to true
    logical NOT for consistency with ``&`` / ``|`` / ``^``.
    """
    return ~input1