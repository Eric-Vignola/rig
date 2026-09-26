"""
Memoization and vectorization decorators for the rig DSL.

``@memoize`` caches function returns keyed by a stable handle of every
``Plug``/``Node``/``PlugList`` argument plus the literal value of every
scalar argument. The cache is auto-invalidated when any cached return value's
underlying Maya nodes have been deleted (via API 1.0 ``MObjectHandle.isAlive``,
which is the same staleness pattern used in ``rig.nodetypes.dg_node``), or
when a dynamic attribute a plug argument reads was deleted or renamed since
(see ``_AttrCheck``). Every new scene and file open clears the caches, and a
reference unload, reload or remove prunes them (the scene callbacks at the end
of this module).

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
import sys
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

# API 1.0 used because API 2.0 MObjects can crash Maya after a new-scene
# load (see rig.nodetypes.dg_node._cache_api1_objects). API 2.0 is only used
# on a wrapper's MObject once its API 1.0 handle says the node is valid.
from maya import cmds, OpenMaya as OpenMaya1
from maya.api import OpenMaya
from rig._internal import callbacks as _callbacks
from rig.nodetypes._base import _plug_identity_name, _unwrapped, Attribute
from rig.nodetypes.dg_node import DGNode
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
    The attr part is the identity of the Maya plug (see
    :func:`rig.nodetypes._base._plug_identity_name`): the alias with the
    instanced indices, since the alias names every element of an instanced
    (world space) attr without its index. ``worldMatrix[0]`` and
    ``worldMatrix[1]`` get one key each, and a plug read through either
    instance path of a node gets the same key.
    """
    node_str = attr.full_name.split(".", 1)[0]  # "node.attr" -> "node"
    identity = _named_dg_identity(attr, node_str)
    if identity is None:
        identity = _node_identity(node_str)
    return (identity, _plug_identity_name(attr))


def _named_dg_identity(attr: Attribute, node_str: str) -> Optional[Tuple[str, int]]:
    """``_node_identity(node_str)`` read from the API 1.0 objects of the DG node
    wrapper ``attr.full_name`` just took ``node_str`` from, or None.

    A DG node's name names no other node, so when the live wrapper's own name
    is ``node_str``, resolving that name finds the wrapper's node. A DAG path
    can go stale (an instance removed), so a DAG node resolves by name.
    """
    node = attr._node
    if isinstance(node, Node):
        node = node._dg_node
    try:
        if (
            isinstance(node, DGNode)
            and node._objhandle1.isValid()
            and not node._mobject.hasFn(OpenMaya.MFn.kDagNode)
            and node._fn_set.name() == node_str
        ):
            return (node._fn_set1.uuid().asString(), node._objhandle1.hashCode())
    except Exception:
        pass
    return None


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


class _AttrCheck:
    """Stands with the node handles of a cache entry for a dynamic attribute
    that a plug argument of the call reads (see `_entry_handles`).

    The key names the attribute (`_attribute_key`), and a name outlives the
    attribute: deleted and added again, or renamed while a new attribute takes
    its name, the name keys a network built on the old attribute (disconnected,
    or reading the renamed one). The check is alive while the node and that
    attribute are, and valid while the name still finds that attribute on the
    node, so such an entry is rebuilt. Only API 1.0 objects are read, and the
    node's fn set only once its handle says the node is valid.
    """

    __slots__ = ("node", "fn", "name", "attr")

    def __init__(self, node: Any, fn: Any, name: str, attr: Any) -> None:
        self.node = node  # the node's API 1.0 MObjectHandle
        self.fn   = fn    # its API 1.0 MFnDependencyNode
        self.name = name  # the attribute's long name
        self.attr = attr  # the attribute's API 1.0 MObjectHandle

    def isAlive(self) -> bool:  # noqa: N802 -- the MObjectHandle protocol
        return self.node.isAlive() and self.attr.isAlive()

    def isValid(self) -> bool:  # noqa: N802
        if not (self.node.isValid() and self.attr.isAlive()):
            return False
        try:
            return self.fn.attribute(self.name) == self.attr.objectRef()
        except RuntimeError:
            return False


def _attr_check(attr: Attribute) -> Optional[_AttrCheck]:
    """An `_AttrCheck` for the attribute of `attr`'s plug if it is a dynamic
    attribute of a live DG node, else None. (An extension attribute, deleted and
    added again for its whole node type, is not checked.)"""
    try:
        mplug = attr.__dict__["_mplug"]
        if not mplug.isDynamic:
            return None
        owner = attr.__dict__["_node"]
        node  = _unwrapped(attr.node if owner is None else owner)
        if not isinstance(node, DGNode) or not node._objhandle1.isValid():
            return None
        mobject = mplug.attribute()
        name    = OpenMaya.MFnAttribute(mobject).name
        attr1 = node._fn_set1.attribute(name)
        return _AttrCheck(node._objhandle1, node._fn_set1, name, OpenMaya1.MObjectHandle(attr1))
    except Exception:
        return None


def _collect_attr_checks(values: Any, out: List[Any]) -> None:
    """Append an `_AttrCheck` for each dynamic attribute a plug in `values` (the
    call arguments; a list, tuple or PlugList among them is walked) reads."""
    for value in values:
        if isinstance(value, Attribute):
            check = _attr_check(value)
            if check is not None:
                out.append(check)
        elif isinstance(value, (list, tuple)):
            _collect_attr_checks(value, out)


def _entry_handles(result: Any, args: tuple, kwargs: dict) -> List[Any]:
    """The staleness checks of a cache entry for a call on `args` / `kwargs` that
    returned `result`: the API 1.0 handles of the nodes in `result` (see
    `_collect_handles`), then an `_AttrCheck` per dynamic attribute a plug
    argument reads. An entry is used only while every one is alive and
    valid."""
    handles: List[Any] = []
    _collect_handles(result, handles)
    _collect_attr_checks(args, handles)
    if kwargs:
        _collect_attr_checks(kwargs.values(), handles)
    return handles


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

    cache: Dict[Tuple[Any, ...], "_CacheEntry"] = {}

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
            # Key on the tuple itself, not on hash(key_tuple): calls whose keys
            # merely share a hash (hash(-1.0) == hash(-2.0)) must not share an
            # entry. hash() still raises TypeError for an un-hashable arg.
            hash(key_tuple)
            key = key_tuple
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

        # Collect MObjectHandles for the result (and checks of the dynamic
        # attributes the arguments read) so we can validate later.
        cache[key] = _CacheEntry(value=result, handles=_entry_handles(result, args, kwargs))
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
# The scene callbacks at the end of this module clear every cache before
# a new scene or a file open and prune them after a reference unload.
#
# An in-place ``importlib.reload`` keeps both lists: the wrappers and
# NodeOps made before it (and ``node_ops`` / ``random``, which imported
# the lists) are still in use.
# --------------------------------------------------------------------- #


_ALL_MEMOIZED:      List[Callable[..., Any]] = globals().get("_ALL_MEMOIZED", [])
_ALL_NODEOP_CACHES: List[Any]                = globals().get("_ALL_NODEOP_CACHES", [])


def prune_memoize_caches() -> int:
    """Walk every ``@memoize`` cache AND every NodeOp cache; drop entries
    whose handles no longer point to live MObjects. Returns the number
    of entries dropped.

    Runs by itself after a reference unload or remove, which frees the
    reference's nodes only (see :func:`_prune_after_scene_change`); a new
    scene or a file open clears every cache instead
    (:func:`_clear_before_new_scene`). ``cleanup()`` calls it too.
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
        # an entry is only made once the callbacks that clear it are there
        # (pending if rig was imported before Maya was initialised)
        if not _SCENE_CALLBACKS_READY:
            _ensure_scene_callbacks()


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


# --------------------------------------------------------------------- #
#  Scene callbacks (round 3, decision D-A)
# --------------------------------------------------------------------- #
#
# A memo entry holds the Plugs its call returned -- and, under the owner
# rule, their nodes and those nodes' attr caches. Once a scene's nodes are
# freed such an entry can never be valid again: the callbacks below drop it
# right away instead of leaving it to its next lookup, so a long session
# does not keep every build's graph.
#
#  * kBeforeNew / kBeforeOpen -> clear every cache. Every node of the
#    scene is about to be freed, and Maya sends these messages only once the
#    new scene or the open goes ahead (not for "Unsaved changes", a missing
#    file, or a kBefore*Check callback that aborts). Clearing also drops the
#    entries that have no handles to prune by (a user @memoize function that
#    returns a node name, a result whose node could not be resolved), or
#    whose only handles are default nodes that outlive a new scene (time1,
#    lambert1, ...).
#  * kAfterUnloadReference / kAfterRemoveReference -> prune. Only the
#    reference's nodes are freed (a reload or a replace unloads first), so
#    only the entries holding one of them are dropped; every other entry
#    keeps deduping.
#  * kAfterNew / kAfterOpen -> prune, for whatever another tool's kBefore*
#    callback cached in the old scene after the clear.
#
# Registered once per Maya session through rig._internal.callbacks: a
# re-import replaces the callbacks (and clears the purged copy's caches),
# and an import before Maya is initialised registers on the first entry.
# --------------------------------------------------------------------- #


def _clear_before_new_scene(*args: Any) -> None:
    """kBeforeNew / kBeforeOpen: clear every memo cache (see above). A callback
    never raises into Maya."""
    try:
        _clear_all_caches()
    except Exception:  # noqa: BLE001
        pass


def _prune_after_scene_change(*args: Any) -> None:
    """kAfterUnloadReference / kAfterRemoveReference / kAfterNew / kAfterOpen:
    drop the entries whose nodes were freed (see above). A callback never raises
    into Maya."""
    try:
        prune_memoize_caches()
    except Exception:  # noqa: BLE001
        pass


def _scene_callback_specs() -> List[Tuple[Callable[..., Any], Any, Callable[..., Any]]]:
    msg = OpenMaya.MSceneMessage
    return [
        (msg.addCallback, msg.kBeforeNew,            _clear_before_new_scene),
        (msg.addCallback, msg.kBeforeOpen,           _clear_before_new_scene),
        (msg.addCallback, msg.kAfterNew,             _prune_after_scene_change),
        (msg.addCallback, msg.kAfterOpen,            _prune_after_scene_change),
        (msg.addCallback, msg.kAfterUnloadReference, _prune_after_scene_change),
        (msg.addCallback, msg.kAfterRemoveReference, _prune_after_scene_change),
    ]


def _register_scene_callbacks() -> bool:
    """Register this module's scene callbacks in place of an earlier import's.
    False, with nothing registered, while Maya is not initialised."""
    global _SCENE_CALLBACKS_READY
    _SCENE_CALLBACKS_READY = _callbacks.register(
        __name__, _THIS_MODULE, _scene_callback_specs(), release=_clear_all_caches
    )
    return _SCENE_CALLBACKS_READY


def _ensure_scene_callbacks() -> bool:
    """Make the registration an import before Maya was initialised left pending."""
    global _SCENE_CALLBACKS_READY
    _SCENE_CALLBACKS_READY = _callbacks.ensure(
        __name__, _THIS_MODULE, _scene_callback_specs(), release=_clear_all_caches
    )
    return _SCENE_CALLBACKS_READY


_THIS_MODULE           = sys.modules.get(__name__)
_SCENE_CALLBACKS_READY = False
_register_scene_callbacks()
