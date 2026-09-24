"""
Procedural pseudo-random number networks for the rig DSL.

Direct port of Eric Vignola's ``rig/random/random_functions.py``.
Six functions that build self-feeding cycle networks implementing a
Linear Congruential Generator (LCG) inside Maya's DG.

Stdlib-shadow note
==================
This module is named ``random`` -- same as Python's stdlib. Both can
coexist because Python addresses them by their fully-qualified names:

    sys.modules['random']               # Python stdlib
    sys.modules['rig.random']   # this module

Inside this file, ``from random import randint`` resolves to the
stdlib via Python 3's absolute-import default. Outside this package
the stdlib ``random`` is unaffected -- there is no actual conflict.

WARNING: Do NOT add ``rig/`` itself to ``sys.path`` -- put its *parent*
there. Adding ``rig/`` would let a bare ``import random`` from anywhere
resolve to this module instead of stdlib.

WARNING -- Maya cycle warning
============================
The networks built by this module rely on a **self-feeding cycle**
(``init`` plug fed back from ``update``) -- this is exactly how the
LCG algorithm advances state. Maya will issue its standard "graph
cycle" warning when these scenes load. **The warning is benign** -- the
cycle is intentional and Maya evaluates it correctly. To silence it
during scene load::

    cmds.cycleCheck(e=False)

References:
    * https://en.wikipedia.org/wiki/Linear_congruential_generator
    * https://www.ams.org/publicoutreach/feature-column/fcarc-random

Memoization policy
==================
Each function in this module dedupes calls **only when ``seed`` is
explicit**. With an explicit seed the call is fully deterministic, so
two calls with the same args MUST return the same Maya network -- not
two parallel copies doing identical math.

Auto-seeded calls (``seed=None``) still produce **fresh** independent
streams every time, which matches the v1 design intent: ``value()``
without a seed is a stochastic stream generator.

The seed-keyed caches are registered with
:func:`rig.prune_memoize_caches` so deleted nodes are evicted
on the next sweep.

Vectorisation via :func:`vectorize` is preserved so
``value([trigger1, trigger2, ...])`` still produces a PlugList.
"""

from __future__ import annotations

from functools import wraps
from random import randint as _python_randint
from typing import Any, Callable, Dict, Optional

from rig._internal.container import container
from rig._internal.math_nodes import condition, constant
from rig._internal.memoize import (
    _ALL_NODEOP_CACHES,
    _CacheEntry,
    _collect_handles,
    _stable_key,
    vectorize,
)
from rig._internal.types import _get_compound, _is_compound
from rig.functions import frame, sum as _sum

# ZX81 / minimal-LCG constants -- pre-baked. See Wikipedia LCG article
# for the full table of "Parameters in common use".
_LCG_MODULUS    = 2**16
_LCG_MULTIPLIER = 75
_LCG_INCREMENT  = 74

# Upper bound for auto-generated seeds (matches Eric's choice).
_SEED_RANGE_MAX = 123456789


__all__ = ["value", "uniform", "randint", "value3D", "uniform3D", "randint3D"]


# --------------------------------------------------------------------- #
#  Memoize-on-explicit-seed decorator
# --------------------------------------------------------------------- #
#
# This is NOT the generic ``@memoize`` from ``_memoize.py`` because we
# need to skip the cache when ``seed=None`` (auto-seeded calls must
# produce independent streams). The dispatcher below caches only when
# the kwargs include an explicit ``seed`` argument.
#
# Each wrapped function gets its own cache dict, and the dict is
# appended to the same ``_ALL_NODEOP_CACHES`` list that
# :func:`prune_memoize_caches` walks -- so stale entries (whose nodes
# were deleted by ``cleanup()``) are pruned alongside everything else.
# --------------------------------------------------------------------- #


class _SeedCacheHolder:
    """Wraps a cache dict with the attributes ``prune_memoize_caches``
    expects from a NodeOp (specifically ``._cache``)."""

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: Dict[int, _CacheEntry] = {}


def _memoize_on_seed(func: Callable[..., Any]) -> Callable[..., Any]:
    """Cache the wrapped function's return value when ``seed`` is given.

    Cache key = ``hash((args, sorted_kwargs, scope))`` -- same shape as
    :func:`@memoize`. When ``seed`` is ``None`` the function runs every
    time without consulting the cache.
    """
    holder = _SeedCacheHolder()
    cache  = holder._cache
    # Register so prune_memoize_caches() walks it.
    _ALL_NODEOP_CACHES.append(holder)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        seed = kwargs.get("seed", None)
        if seed is None:
            # Auto-seeded -- never memoize; produce a fresh stream.
            return func(*args, **kwargs)

        # Build a stable cache key (mirrors @memoize).
        try:
            scope_key = tuple(c._scope_key for c in container._stack[:1])
        except Exception:
            scope_key = ()

        try:
            key_tuple = (
                tuple(_stable_key(a) for a in args),
                tuple(sorted((k, _stable_key(v)) for k, v in kwargs.items())),
                scope_key,
            )
            key = hash(key_tuple)
        except TypeError:
            # Un-hashable arg -- bypass cache.
            return func(*args, **kwargs)

        entry = cache.get(key)
        if entry is not None:
            if all(h.isAlive() and h.isValid() for h in entry.handles):
                return entry.value
            del cache[key]

        result  = func(*args, **kwargs)

        handles = []
        _collect_handles(result, handles)
        cache[key] = _CacheEntry(value=result, handles=handles)
        return result

    wrapper._cache = cache
    return wrapper


# --------------------------------------------------------------------- #
#  Scalar generators
# --------------------------------------------------------------------- #


@vectorize
@_memoize_on_seed
def value(trigger: Optional[Any] = None, seed: Optional[int] = None) -> Any:
    """``value(trigger=None, seed=None)`` -- pseudo-random scalar Plug
    in [0, 1). Each frame (or when ``trigger`` changes) the network
    advances one LCG step and produces a new value.

    Named ``value`` (not ``random``) to avoid colliding with the module
    name ``random`` (``random.random`` was the old, clashing spelling).

    Args:
        trigger: optional Plug whose value-change signals the next
            random step. Defaults to :func:`rig.functions.frame`
            (timeline-tied). Compound triggers are summed to a scalar.
        seed: optional integer seed. If ``None``, a random seed is
            chosen from Python's RNG. **When ``seed`` is explicit the
            call is memoised**: repeated calls with the same args
            return the same Maya network. ``seed=None`` always
            produces a fresh independent stream.

    Examples::

        node.ty << value()                    # frame-driven, fresh stream
        node.ty << value(other_node.tx)       # update on tx change
        node.ty << value(seed=42)             # memoised -- same net every time
    """

    with container("value1"):
        if seed is None:
            seed = _python_randint(0, _SEED_RANGE_MAX)

        if trigger is None:
            trigger = frame()
        else:
            trigger = container.publish_input(trigger, "trigger")
            if _is_compound(trigger):
                trigger = _sum(_get_compound(trigger))

        # `init` initiates the seed and receives the feedback loop.
        init = constant([seed] * 3, dtype="long")

        # Catches the scene-load reset condition: when the cycle is
        # re-evaluated from scratch, .valueX may briefly be 0 -- re-inject
        # the seed to keep the sequence deterministic across reloads.
        reset = condition(init.valueX == 0, seed, init.valueX)

        # LCG step: x_{n+1} = (a * x_n + c) mod m.
        iteration = (_LCG_MULTIPLIER * reset + _LCG_INCREMENT) % _LCG_MODULUS

        # Package the new state alongside the trigger so a trigger
        # change forces a recompute.
        update = constant(
            [iteration, 0, trigger],
            name="CYCLE_SAFE_RANDOM_GENERATOR1",
        )

        # Close the cycle.
        init << update.value

        # Normalise to [0, 1).
        return container.publish_output(update.valueX / _LCG_MODULUS, "output")


@vectorize
@_memoize_on_seed
def uniform(
    start:   Any,
    end:     Any,
    trigger: Optional[Any] = None,
    seed:    Optional[int] = None,
) -> Any:
    """``uniform(start, end, trigger=None, seed=None)`` -- pseudo-random
    scalar Plug in ``[start, end)``. Memoised on explicit seed."""
    with container("uniform1"):
        if trigger is not None:
            trigger = container.publish_input(trigger, "trigger")
        start = container.publish_input(start, "start")
        end   = container.publish_input(end, "end")
        return container.publish_output(
            (end - start) * value(trigger=trigger, seed=seed) + start, "output"
        )


@vectorize
@_memoize_on_seed
def randint(
    start:   Any,
    end:     Any,
    trigger: Optional[Any] = None,
    seed:    Optional[int] = None,
) -> Any:
    """``randint(start, end, trigger=None, seed=None)`` -- pseudo-random
    integer Plug in ``[start, end]`` (long-cast of :func:`uniform`).
    Memoised on explicit seed."""
    with container("randint1"):
        if trigger is not None:
            trigger = container.publish_input(trigger, "trigger")
        start = container.publish_input(start, "start")
        end   = container.publish_input(end, "end")
        return container.publish_output(
            constant(uniform(start, end, trigger=trigger, seed=seed), dtype="long"),
            "output",
        )


# --------------------------------------------------------------------- #
#  3D generators
# --------------------------------------------------------------------- #


@vectorize
@_memoize_on_seed
def value3D(trigger: Optional[Any] = None, seed: Optional[Any] = None) -> Any:
    """``value3D(trigger=None, seed=None)`` -- pseudo-random 3-vector
    Plug. Each component has its own seed and runs an independent LCG.
    Memoised on explicit seed.

    Named ``value3D`` (not ``random3D``) to mirror the scalar
    :func:`value` rename (kills the ``random.random`` module clash)."""

    with container("value3d1"):
        if seed is None:
            seed = [
                _python_randint(0, _SEED_RANGE_MAX),
            ]

        if trigger is not None:
            trigger = container.publish_input(trigger, "trigger")
            if _is_compound(trigger):
                trigger = _sum(_get_compound(trigger))

        return container.publish_output(
            constant(
                [
                    value(trigger=trigger, seed=seed[0]),
                    value(trigger=trigger, seed=seed[1]),
                    value(trigger=trigger, seed=seed[2]),
                ]
            ),
            "output",
        )


@vectorize
@_memoize_on_seed
def uniform3D(
    start:   Any,
    end:     Any,
    trigger: Optional[Any] = None,
    seed:    Optional[Any] = None,
) -> Any:
    """``uniform3D(start, end, trigger=None, seed=None)`` -- pseudo-random
    3-vector Plug in ``[start, end)`` per component. Memoised on
    explicit seed."""

    with container("uniform3d1"):
        if trigger is not None:
            trigger = container.publish_input(trigger, "trigger")
            if _is_compound(trigger):
                trigger = _sum(_get_compound(trigger))
        start = container.publish_input(start, "start", at="double3")
        end   = container.publish_input(end, "end", at="double3")
        return container.publish_output(
            (end - start) * value3D(trigger=trigger, seed=seed) + start, "output"
        )


@vectorize
@_memoize_on_seed
def randint3D(
    start:   Any,
    end:     Any,
    trigger: Optional[Any] = None,
    seed:    Optional[Any] = None,
) -> Any:
    """``randint3D(start, end, trigger=None, seed=None)`` -- pseudo-random
    integer 3-vector Plug in ``[start, end]`` (long-cast of
    :func:`uniform3D`). Memoised on explicit seed."""

    with container("randint3d1"):
        if trigger is not None:
            trigger = container.publish_input(trigger, "trigger")
            if _is_compound(trigger):
                trigger = _sum(_get_compound(trigger))
        start = container.publish_input(start, "start", at="double3")
        end   = container.publish_input(end, "end", at="double3")
        return container.publish_output(
            constant(uniform3D(start, end, trigger=trigger, seed=seed), dtype="long"),
            "output",
        )