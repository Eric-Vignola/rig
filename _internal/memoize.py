"""
Memoization and vectorization decorators for the rig DSL.

``@memoize`` caches function returns keyed by a stable handle of every
``Plug``/``Node``/``PlugList`` argument plus the literal value of every
scalar argument.
scalar argument. The cache is auto-invalidated when any cached return value's
underlying Maya nodes have been deleted (via API 1.0 ``MObjectHandle.isAlive``,
which is the same staleness pattern used in ``rig.maya.nodetypes.dg_node``).

``@vectorize`` broadcasts a function call across :class:`PlugList`
arguments using **NumPy-style strict broadcasting**: every list / list-like
argument must be the same length, OR be length 1, OR be scalar. Mismatched
lengths raise :class:`ValueError` rather than silently capping (which is
the behaviour of the original Eric Vignola ``rig`` library).

A call is broadcast only when at least one argument is a :class:`PlugList`.
With only plain Python lists / scalars, the function is called directly
with its raw arguments.
"""

from __future__ import annotations

import numbers
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

# API 1.0 used because API 2.0 MObjects can crash Maya after a new-scene
# load (see rig.maya.nodetypes.dg_node._cache_api1_objects).
from maya import cmds, OpenMaya as OpenMaya1
from rig.maya.nodetypes._base import Attribute
from rig._internal.container import container, ContainerOptions
from rig._internal.generators import arguments
from rig._internal.list import PlugList
from rig._internal.node import Node
from rig._internal.types import (
    _is_list,
    _is_node,
    _is_plug,
    _is_real,
    _is_sequence,
)


def _stable_key(obj: Any) -> Any:
    """Return a hashable, scene-stable key for ``obj`` suitable for memo lookup.

    - Numbers => ``float`` (collapses ``5`` and ``5.0`` to one cache entry).
    - Strings => themselves.
    - ``None`` => ``None``.
    - Plug-like => ``((uuid, hashCode), attr_long_name)`` -- survives rename + delete.
    - Node-like (Node) => ``("node", (uuid, hashCode))`` -- survives rename + delete.
    - PlugList => tuple of element keys.
    - Other sequences => tuple of element keys.
    - Anything else => ``id(obj)`` (best-effort; usually un-cacheable).
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        # Important: must come before _is_attribute (Attribute IS a str).
        if isinstance(obj, Attribute):
            return _attribute_key(obj)
        return obj
    if isinstance(obj, numbers.Real):
        return float(obj)
    if _is_plug(obj):
        return _attribute_key(obj)
    if _is_node(obj):
        node = obj
        return ("node", _node_identity(node._dg_node.name))
    if _is_list(obj):
        return tuple(_stable_key(x) for x in obj)
    if _is_sequence(obj):
        return tuple(_stable_key(x) for x in obj)
    # Fallback -- won't survive scene reload but at least won't crash.
    return ("opaque", id(obj))


def _attribute_key(attr: Attribute) -> Tuple[Any, str]:
    """Return a ``((uuid, hashCode), attr_long_name)`` cache key for an Attribute.

    Uses the composite node identity (see :func:`_node_identity`) so the key
    survives renames AND Maya's MObjectHandle hashCode recycling on delete.
    """
    node_str = attr.full_name.split(".", 1)[0]  # "node.attr" -> "node"
    return (_node_identity(node_str), attr.alias)


def _node_identity(node_name: str) -> Tuple[str, int]:
    """Return a ``(uuid, hashCode)`` composite identity for the named node.

    The Maya UUID is rename-stable AND is **not** recycled when a node is
    deleted, which fixes the ``MObjectHandle.hashCode()`` recycling collision
    (Maya reuses a freed MObject slot -- and its hashCode -- for the next node
    created, e.g. the proxy curve that ``create_rail`` deletes every build).
    The hashCode is kept as a second element so that two *simultaneously
    live* nodes which share a UUID -- possible only via ``file -import`` /
    ``createReference`` of files carrying duplicate UUIDs -- still get
    distinct keys, since their MObjectHandles (hence hashCodes) differ. Both
    values come from the single ``MObject`` resolved here, so this costs the
    same one ``MSelectionList`` roundtrip the old hashCode-only path did.

    Resolved via API 1.0 only (never ``dg_node.uuid``, which reads the
    API 2.0 function set -- API 2.0 MObjects can crash Maya after a new-scene
    load).

    Raises:
        TypeError: if the node cannot be resolved in the current scene, so
            the ``@memoize`` / ``NodeOp`` / seed wrappers fall through their
            existing un-hashable ``except TypeError`` bypass and recompute,
            rather than synthesising a name-based key that could collide with
            a same-named node in a different scene.
    """
    try:
        sel = OpenMaya1.MSelectionList()
        sel.add(node_name)
        mobject1 = OpenMaya1.MObject()
        sel.getDependNode(0, mobject1)
        hash_code = OpenMaya1.MObjectHandle(mobject1).hashCode()
        try:
            uuid_str = OpenMaya1.MFnDependencyNode(mobject1).uuid().asString()
        except (AttributeError, RuntimeError):
            # API 1.0 MUuid may be unavailable on older Maya; cmds is universal.
            uuid_str = cmds.ls(node_name, uuid=True)[0]
        return (uuid_str, hash_code)
    except Exception as exc:
        raise TypeError(
            f"memoize: cannot resolve node {node_name!r} for cache keying"
        ) from exc


def _collect_handles(obj: Any, out: List[OpenMaya1.MObjectHandle]) -> None:
    """Walk ``obj`` and append API 1.0 MObjectHandles for any plug/node found.

    Used when caching a return value so we can later check ``isAlive()`` to
    invalidate dead entries.
    """
    if (
        obj is None
        or isinstance(obj, (str, bytes, numbers.Real))
        and not _is_plug(obj)
        and not _is_node(obj)
    ):
        return
    if _is_plug(obj):
        full_name = obj.full_name
        node_str  = full_name.split(".", 1)[0]
        try:
            sel = OpenMaya1.MSelectionList()
            sel.add(node_str)
            mobject1 = OpenMaya1.MObject()
            sel.getDependNode(0, mobject1)
            out.append(OpenMaya1.MObjectHandle(mobject1))
        except Exception:
            pass
        return
    if _is_node(obj):
        try:
            sel = OpenMaya1.MSelectionList()
            sel.add(str(obj))
            mobject1 = OpenMaya1.MObject()
            sel.getDependNode(0, mobject1)
            out.append(OpenMaya1.MObjectHandle(mobject1))
        except Exception:
            pass
        return
    if _is_list(obj) or _is_sequence(obj):
        for elt in obj:
            _collect_handles(elt, out)


def _fold_eligible(foldable: Any, args: tuple, kwargs: dict) -> bool:
    """Return ``True`` when a ``@memoize(foldable=...)`` call's inputs are all
    literal numbers -- so the wrapped function will constant-fold to a plain
    Python value rather than build a Maya node.

    ``foldable`` is the declaration passed to :func:`memoize`:

      * ``"scalar"`` -- every positional arg AND keyword value is a plain
        number (``abs(5)``, ``clamp(1, 2, 3)``, ``pow(2, 3)``).
      * ``"reduce"`` -- exactly one positional arg, a sequence whose elements
        are all numbers (``sum([1, 2, 3])``, ``max([4, 5, 6])``); any keyword
        values must be numbers too.
      * a callable ``predicate(args, kwargs) -> bool`` -- used directly, for
        functions whose fold condition is more specific (e.g. ``length``,
        which folds a literal vector but NOT a 9/16-element matrix literal).
    """
    if foldable == "scalar":
        return all(_is_real(a) for a in args) and all(
            _is_real(v) for v in kwargs.values()
        )
    if foldable == "reduce":
        return (
            len(args) == 1
            and _is_sequence(args[0])
            and all(_is_real(x) for x in args[0])
            and all(_is_real(v) for v in kwargs.values())
        )
    if callable(foldable):
        return bool(foldable(args, kwargs))
    return False


def memoize(
    func:     Optional[Callable[..., Any]] = None,
    *,
    foldable: Any                          = False,
) -> Callable[..., Any]:
    """Cache ``func``'s return keyed on argument identity.

    The cache survives renames AND node delete/recreate (keys on a composite
    ``(uuid, hashCode)`` node identity -- see :func:`_node_identity`) and is
    auto-pruned when any node referenced by a cached entry is deleted from
    the scene.

    ``foldable`` declares that ``func`` constant-folds to a plain Python value
    when all of its numeric inputs are literals. Pass ``"scalar"`` (every
    argument is a scalar number), ``"reduce"`` (the single sequence argument's
    elements are all numbers), or a custom ``predicate(args, kwargs) -> bool``
    (see :func:`_fold_eligible`). A fold-eligible call is handled *before* the
    cache when :data:`ContainerOptions.constant_folding` is ``True``: the
    Python value is computed directly and **never cached** -- recompute is
    cheap, and keeping folds out of the cache guarantees a folded scalar and a
    :func:`rig.force_nodes` node-network can never collide under one
    key (nor leave the immortal empty-handle entries a cached scalar would).
    When ``constant_folding`` is ``False`` the fold is skipped entirely, so the
    call falls through to build -- and dedupe -- a real Maya node network.

    Contract: a ``foldable`` function MUST return a non-node Python value
    whenever its fold predicate matches; otherwise the un-cached fast path
    would build an un-deduped node. Functions with no fold path leave
    ``foldable=False`` (the default) and are cached unconditionally.
    """
    # Support both ``@memoize`` and ``@memoize(foldable=...)`` syntaxes.
    if func is None:

        def deco(f: Callable[..., Any]) -> Callable[..., Any]:
            return memoize(f, foldable=foldable)

        return deco

    cache: Dict[int, "_CacheEntry"] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Fold-before-cache: when folding is enabled AND every numeric input is
        # a literal, compute the Python value directly and DO NOT cache it.
        # Running this BEFORE the cache lookup is what makes a runtime
        # ``constant_folding`` flip correct: a fold-eligible call never reads a
        # cached node that a prior ``force_nodes()`` block built under the same
        # key, and a fold result never poisons the cache for a later
        # ``force_nodes()`` build.
        if (
            foldable
            and ContainerOptions.constant_folding
            and _fold_eligible(foldable, args, kwargs)
        ):
            return func(*args, **kwargs)

        # Build a stable cache key from args + kwargs + active container scope.
        # Lazy import to avoid circular dep with _container.
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

        # Check cache
        if key in cache:
            entry = cache[key]
            if all(h.isAlive() and h.isValid() for h in entry.handles):
                return entry.value
            # Stale -- drop and recompute.
            del cache[key]

        result = func(*args, **kwargs)

        # Collect MObjectHandles for the result so we can validate later.
        handles: List[OpenMaya1.MObjectHandle] = []
        _collect_handles(result, handles)
        cache[key] = _CacheEntry(value=result, handles=handles)
        return result

    wrapper._cache    = cache
    wrapper._foldable = foldable
    _ALL_MEMOIZED.append(wrapper)
    return wrapper


# --------------------------------------------------------------------- #
#  Cache registries (used by garbage collection, v2.E)
# --------------------------------------------------------------------- #
#
# Two parallel registries:
#
#  * ``_ALL_MEMOIZED`` -- every ``@memoize``-decorated wrapper appends
#    itself here. Each wrapper owns a ``_cache`` dict.
#  * ``_ALL_NODEOP_CACHES`` -- every :class:`NodeOp` registers itself
#    here at construction. Each NodeOp owns a ``_cache`` dict (same
#    shape as ``@memoize`` caches).
#
# :func:`prune_memoize_caches` walks BOTH and drops entries whose
# handles no longer point to live MObjects.
#
# The wrappers already self-prune on lookup, but ``cleanup()`` calls
# this proactively after deleting nodes so the in-memory cache doesn't
# grow unbounded across long sessions that build/teardown many networks.
# --------------------------------------------------------------------- #


_ALL_MEMOIZED:      List[Callable[..., Any]] = []
_ALL_NODEOP_CACHES: List[Any] = []


def prune_memoize_caches() -> int:
    """Walk every ``@memoize`` cache AND every NodeOp cache; drop entries
    whose handles no longer point to live MObjects. Returns the number
    of entries dropped.
    """
    dropped = 0
    for wrapper in _ALL_MEMOIZED:
        cache = getattr(wrapper, "_cache", None)
        if cache is None:
            continue
        for key in list(cache):
            entry = cache[key]
            if not all(h.isAlive() and h.isValid() for h in entry.handles):
                del cache[key]
                dropped += 1
    for op in _ALL_NODEOP_CACHES:
        cache = getattr(op, "_cache", None)
        if cache is None:
            continue
        for key in list(cache):
            entry = cache[key]
            if not all(h.isAlive() and h.isValid() for h in entry.handles):
                del cache[key]
                dropped += 1
    return dropped


def _clear_all_caches() -> int:
    """Clear every ``@memoize`` cache AND every NodeOp cache completely.
    Returns the total number of entries dropped.

    Used by :func:`rig.set_options(maya_version=...)` when the user
    flips the target Maya version -- cached results from the previous
    target are no longer valid for new dispatch.
    """
    dropped = 0
    for wrapper in _ALL_MEMOIZED:
        cache = getattr(wrapper, "_cache", None)
        if cache is None:
            continue
        dropped += len(cache)
        cache.clear()
    for op in _ALL_NODEOP_CACHES:
        cache = getattr(op, "_cache", None)
        if cache is None:
            continue
        dropped += len(cache)
        cache.clear()
    return dropped


class _CacheEntry:
    """Holds a memoized return value plus the API 1.0 handles used for
    staleness checking."""

    __slots__ = ("value", "handles")

    def __init__(self, value: Any, handles: List[OpenMaya1.MObjectHandle]) -> None:
        self.value   = value
        self.handles = handles


def _broadcast_len(obj: Any) -> int:
    """Return the broadcast length of ``obj`` for NumPy-style strict shape checking.

    Scalars (numbers, strings, ``None``, single :class:`Plug` /
    :class:`Node`, anything with no ``len()``) return ``1`` -- meaning
    they broadcast freely against any other length.

    Sequence-like inputs return ``len(obj)``.
    """
    if obj is None:
        return 1
    if isinstance(obj, (numbers.Real, bytes)):
        return 1
    # Strings include Attribute / Plug (which subclass str). Treat any plain
    # string / Plug / Node as scalar.
    if isinstance(obj, str):
        return 1
    if _is_plug(obj) or _is_node(obj):
        return 1
    try:
        return len(obj)
    except TypeError:
        return 1


def vectorize(
    func:        Optional[Callable[..., Any]] = None,
    *,
    favor_index: Optional[int]                = None,
) -> Callable[..., Any]:
    """Broadcast ``func`` across :class:`PlugList` arguments -- NumPy-style strict.

    Triggers when any positional or keyword argument is a ``PlugList``.
    Once triggered, every list / sequence argument must satisfy
    NumPy's broadcasting rule: have the same length as the longest, OR
    be length 1, OR be scalar. Mismatched non-1 lengths raise
    :class:`ValueError`.

    This differs from the original Eric Vignola ``rig`` library (which
    silently capped each shorter argument to its last element). The
    permissive behaviour was a frequent source of subtle bugs from typos --
    we trade it for explicit error messages on a clean-slate rebuild.

    Returns a single value if there is one row, a :class:`PlugList`
    otherwise.

    Args:
        func: Function to wrap (when used as a plain decorator).
        favor_index: Optional positional index. If given, vectorisation
            stops once ``func`` has been called ``len(args[favor_index])``
            times. Used to limit broadcasts driven by a particular argument.

    Raises:
        ValueError: When two or more sequence-typed arguments have
            different non-1 lengths (NumPy-style broadcast violation).

    Examples::

        @vectorize
        def f(a, b, c): ...

        f(PlugList([a, b, c, d, e]), 5, [x, y, z, w, q])    # OK -- both length 5
        f(PlugList([a, b, c, d, e]), 5, [x])                 # OK -- [x] broadcasts
        f(PlugList([a, b, c, d, e]), 5, x)                   # OK -- x is scalar
        f(PlugList([a, b, c, d, e]), 5, [x, y, z])           # ValueError: lengths {3, 5}
    """
    # Support both @vectorize and @vectorize(favor_index=N) syntaxes.
    if func is None:

        def deco(f: Callable[..., Any]) -> Callable[..., Any]:
            return vectorize(f, favor_index=favor_index)

        return deco

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not args and not kwargs:
            return func()

        valid_args   = any(_is_list(x) for x in args)
        valid_kwargs = any(_is_list(v) for v in kwargs.values())

        # No PlugList among the args -- call func directly with raw arguments.
        if not (valid_args or valid_kwargs):
            return func(*args, **kwargs)

        # ---- NumPy-style strict broadcast shape check ------------------- #
        arg_lens   = [_broadcast_len(x) for x in args]
        kwarg_lens = [(k, _broadcast_len(v)) for k, v in kwargs.items()]
        non_scalar_lens = [n for n in arg_lens if n != 1] + [
            n for _, n in kwarg_lens if n != 1
        ]
        unique_lens = set(non_scalar_lens)
        if len(unique_lens) > 1:
            # Build a helpful error message naming offending positional /
            # keyword args.
            offenders = []
            for i, n in enumerate(arg_lens):
                if n != 1:
                    offenders.append(f"arg{i}=len {n}")
            for k, n in kwarg_lens:
                if n != 1:
                    offenders.append(f"{k}=len {n}")
            raise ValueError(
                f"vectorize({func.__name__!r}): cannot broadcast lengths "
                f"{sorted(unique_lens)} \u2014 every list argument must be the "
                f"same length, or scalar / length-1. Got: {', '.join(offenders)}."
            )

        # If no list-like arg has a length > 1, broadcast count is 1.
        broadcast_count = max(unique_lens) if unique_lens else 1

        max_count: Optional[int] = (
            len(args[favor_index]) if favor_index is not None else broadcast_count
        )

        # ---- Iterate (arguments() handles the per-row index walking) ---- #
        results: List[Any] = []
        count = 0
        for row_args, row_kwargs in arguments(*args, **kwargs):
            res = func(*row_args, **row_kwargs)
            if res is not None:
                results.append(res)
            count += 1
            if max_count is not None and count >= max_count:
                break

        if not results:
            return None
        if len(results) == 1:
            return results[0]

        # Lazy import to avoid circular dep
        try:
            return PlugList(results)
        except Exception:
            return results

    return wrapper