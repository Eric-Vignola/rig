"""
Version-keyed node-operation dispatch framework.

A :class:`NodeOp` describes a single math/transform operation (``sin``,
``lerp``, ``modulo``, ``decomposeMatrix``, ...) plus one or more versioned
implementations of how to build the corresponding Maya node-network. At
call time the framework picks the highest-``since`` implementation that is
``<= cmds.about(version=True)``, and (if the impl is scalar-only and the
input is compound) wraps the call in standard publish-input -> fan-out ->
publish-output container boilerplate.

This replaces dozens of hand-written ``if MAYA_VERSION >= 2024:`` branches
in Eric Vignola's original library with declarative, version-keyed registration.

Example::

    sin_op = NodeOp("sin", scalar_fn=math.sin, requires_plugin="quatNodes")

    @sin_op.impl(since=2024, scope="scalar")
    def _sin_2024(token):
        node = container.createNode("sin")
        node.input << token
        return node.output

    @sin_op.impl(since=0, scope="scalar")
    def _sin_legacy(token):
        node = container.createNode("eulerToQuat")
        node.inputRotateX << token * (360. / math.pi)
        return node.outputQuatX

    @vectorize
    @memoize
    def sin(token):
        return sin_op(token)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from maya import cmds
from rig._internal.container import container, ContainerOptions
from rig._internal.generators import sequences
from rig._internal.maya_version import get_target_version
from rig._internal.memoize import (
    _ALL_NODEOP_CACHES,
    _CacheEntry,
    _collect_handles,
    _stable_key,
)
from rig._internal.types import _get_compound, _is_compound, _is_real


# Valid scope values for impl declarations.
SCOPE_SCALAR   = "scalar"
SCOPE_COMPOUND = "compound"
SCOPE_AUTO     = "auto"
_VALID_SCOPES  = frozenset({SCOPE_SCALAR, SCOPE_COMPOUND, SCOPE_AUTO})


# Type aliases (string-form so they're not evaluated at runtime under py3.7)
ImplFn    = "Callable[..., Any]"
ImplEntry = "Tuple[int, str, ImplFn]"


class NodeOp:
    """Declarative description of a Maya node operation with version-keyed impls.

    Args:
        name: Used for naming containers and (optionally) output nodes.
            E.g. ``"sin"`` will produce containers like ``sin1``, ``sin2``...
        scalar_fn: Optional Python callable used to short-circuit when all
            arguments are plain numbers. E.g. ``math.sin`` for a ``sin`` op.
            Returning here skips Maya entirely.
        requires_plugin: Optional plugin name that must be loaded before any
            impl can run. The plugin is loaded lazily (idempotent) on the
            first call. Replaces module-import-time ``loadPlugin`` calls.
    """

    def __init__(
        self,
        name:            str,
        scalar_fn:       Optional[Callable[..., Any]] = None,
        requires_plugin: Optional[str]                = None,
    ) -> None:
        self.name            = name
        self.scalar_fn       = scalar_fn
        self.requires_plugin = requires_plugin
        # impls sorted highest-since first
        self._impls: List[Tuple[int, str, Callable[..., Any]]] = []
        self._plugin_loaded = False
        # Per-NodeOp cache. Same shape/semantics as @memoize: keyed on
        # (args, kwargs, container scope), value is _CacheEntry with
        # MObjectHandle-based liveness validation. Registered globally so
        # ``prune_memoize_caches()`` can sweep stale entries scene-wide.
        self._cache: Dict[int, Any] = {}
        _ALL_NODEOP_CACHES.append(self)

    # -- registration -- #

    def impl(
        self, since: int = 0, scope: str = SCOPE_SCALAR
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register an implementation.

        Args:
            since: Minimum Maya version (e.g. ``2024``) where this impl
                applies. ``0`` means "any version" (i.e. fallback).
            scope: How the impl handles compound inputs.
                ``"scalar"`` (default): the impl handles one channel only;
                the framework fans out per-channel for compound input.
                ``"compound"``: the impl handles compound inputs natively;
                the framework calls it directly without fan-out.
                ``"auto"``: the framework treats compound inputs as
                ``"scalar"`` (fan out) and non-compound inputs as
                ``"compound"`` (call directly). Useful when the impl is
                naturally scalar but accepts numbers/single plugs too.
        """
        if scope not in _VALID_SCOPES:
            raise ValueError(
                f"NodeOp.impl scope must be one of {sorted(_VALID_SCOPES)}, "
                f"got {scope!r}"
            )

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._impls.append((since, scope, fn))
            # highest-since first
            self._impls.sort(key=lambda x: -x[0])
            return fn

        return decorator

    # -- dispatch -- #

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # 1. all-numeric short-circuit (no Maya nodes -- no caching needed).
        #    Gated on ``constant_folding`` so ``force_nodes()`` forces even a
        #    scalar-only op to materialize its Maya node network.
        if (
            self.scalar_fn is not None
            and ContainerOptions.constant_folding
            and all(_is_real(a) for a in args)
        ):
            return self.scalar_fn(*args, **kwargs)

        # 2. cache lookup. Mirrors the @memoize body so that v1 operators
        #    (Plug.__add__ -> _plus_minus_average_op(...) -> NodeOp.__call__)
        #    get the same dedupe behaviour the v2 helpers get from their
        #    explicit ``@memoize def f(...)`` wrappers.
        try:
            scope_key = tuple(c._scope_key for c in container._stack[:1])
        except Exception:
            scope_key = ()

        cache_key: Optional[int]
        try:
            # Include the target Maya version in the key so that runtime
            # changes (e.g. via ``set_options(maya_version=N)``) correctly
            # invalidate cached results -- different versions can dispatch
            # to different impls and therefore produce different nodes.
            key_tuple = (
                tuple(_stable_key(a) for a in args),
                tuple(sorted((k, _stable_key(v)) for k, v in kwargs.items())),
                scope_key,
                get_target_version(),
            )
            cache_key = hash(key_tuple)
        except TypeError:
            # Un-hashable arg -- bypass cache (matches @memoize behaviour).
            cache_key = None

        if cache_key is not None and cache_key in self._cache:
            entry = self._cache[cache_key]
            if all(h.isAlive() and h.isValid() for h in entry.handles):
                return entry.value
            # Stale -- drop and recompute.
            del self._cache[cache_key]

        # 3. lazy plugin load (idempotent)
        if self.requires_plugin and not self._plugin_loaded:
            cmds.loadPlugin(self.requires_plugin, quiet=True)
            self._plugin_loaded = True

        # 4. version dispatch with NotImplementedError fall-through
        maya_ver = get_target_version()
        last_error: Optional[Exception] = None
        result:     Any = None
        for since, scope, fn in self._impls:
            if maya_ver < since:
                continue
            try:
                result = self._invoke(fn, scope, args, kwargs)
                break
            except NotImplementedError as e:
                last_error = e
                continue
        else:
            # No impl returned (loop exhausted without break).
            if last_error is not None:
                raise RuntimeError(
                    f"All impls of NodeOp({self.name!r}) for Maya {maya_ver} "
                    f"raised NotImplementedError. Last: {last_error}"
                )
            raise RuntimeError(
                f"No impl of NodeOp({self.name!r}) satisfies Maya {maya_ver}. "
                f"Registered impls: {[e[0] for e in self._impls]}"
            )

        # 5. Store in cache (with handles for staleness validation).
        if cache_key is not None:
            handles: List[Any] = []
            _collect_handles(result, handles)
            self._cache[cache_key] = _CacheEntry(value=result, handles=handles)

        return result

    # -- internals -- #

    def _invoke(
        self,
        fn:     Callable[..., Any],
        scope:  str,
        args:   tuple,
        kwargs: dict,
    ) -> Any:
        """Call ``fn`` with framework-managed scoping and (optionally) fan-out."""
        is_compound_input = any(_is_compound(a) for a in args)

        # Compound impl, or scalar impl with non-compound input -> call directly.
        if (
            scope == SCOPE_COMPOUND
            or (scope == SCOPE_AUTO and not is_compound_input)
            or (scope == SCOPE_SCALAR and not is_compound_input)
        ):
            return fn(*args, **kwargs)

        # Scalar impl + compound input -> framework fan-out per channel.
        # ``_constant`` is a lazy import: ``_math_nodes`` imports ``NodeOp``
        # from this module at top, so a top-level import here would close
        # the cycle.
        from rig._internal.math_nodes import _constant

        with container(f"{self.name}1"):
            # Publish each positional DATA arg as a stable input on the
            # outer container so external callers get a named interface
            # (e.g. ``(node1.t % node2.t).input1``) rather than reaching
            # into the raw ``_constant`` aggregator. Names are generic
            # (``input``/``input1``/``input2``/...) because the framework
            # doesn't know the impl's specific arg names.
            #
            # String args are SKIPPED -- they're operator/config parameters
            # (e.g. the ``"<"`` in ``_condition_op(input0, op, input1)``)
            # that aren't data and can't sensibly be published as attrs.
            # Non-string args are numbered sequentially so e.g. a ternary
            # impl ``(input0, op_str, input1)`` publishes ``input1`` and
            # ``input2`` (not ``input1`` and ``input3``).
            #
            # Under ``flatten_containers=True`` (the default), publish_input
            # is passthrough -- so behavior is unchanged in default mode and
            # only activates when users opt in to publishing.
            data_indices = [i for i, a in enumerate(args) if type(a) is not str]
            if data_indices:
                new_args = list(args)
                if len(data_indices) == 1:
                    i           = data_indices[0]
                    new_args[i] = container.publish_input(args[i], "input")
                else:
                    for pub_idx, i in enumerate(data_indices):
                        new_args[i] = container.publish_input(
                            args[i], f"input{pub_idx + 1}"
                        )
                args = tuple(new_args)

            channels_per_arg = [_get_compound(a) for a in args]

            # Channel-count enforcement: NodeOp fan-out broadcasts scalar
            # (1-element) inputs across compound (N-element) inputs, but
            # all multi-channel inputs must agree on N. Mixing e.g. a
            # 3-vector and a 4-quaternion as both compound inputs would
            # silently truncate (zip stops at shortest); raise instead.
            multi_lengths = {len(c) for c in channels_per_arg if len(c) > 1}
            if len(multi_lengths) > 1:
                raise ValueError(
                    f"NodeOp({self.name!r}): cannot fan across mismatched "
                    f"compound shapes -- got channel counts "
                    f"{[len(c) for c in channels_per_arg]} for args "
                    f"{[type(a).__name__ for a in args]}. All multi-channel "
                    f"inputs must have the same length (scalars broadcast)."
                )

            output_plugs = []
            for channel_tuple in sequences(*channels_per_arg):
                output_plugs.append(fn(*channel_tuple, **kwargs))

            count  = len(output_plugs)
            output = _constant([0] * count, name="output_plug1")
            output << output_plugs

            # Publish the assembled compound output. Under
            # ``flatten_containers=True`` (the default), publish_output is
            # passthrough -- returns ``output`` unchanged so existing
            # behavior is preserved. Under ``flatten_containers=False``,
            # returns the published Plug for a clean black-box interface.
            return container.publish_output(output, "output")