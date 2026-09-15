"""
Container scope and flattening.

The :data:`container` singleton is a context manager that tracks an active
stack of :class:`Container` (Maya container-node wrappers). The first
``with container("name")`` block in a stack creates a real Maya container
node; nested blocks default to *flatten* (no inner container, nodes go into
the outermost), with naming-prefix breadcrumbs on created nodes.

:class:`Container` is a subclass of :class:`rig.Node`. It inherits
all the attribute access (``ctn.foo`` returns a :class:`Plug`), spec injection
(``ctn << Float("blend")`` adds an attribute to the container itself), and
operator behaviour.

Behaviour overview
==================
``with container("name"):``
    Default -- creates a real Maya container at the top level, flattens
    when nested.

``with container("name", preserve=True):``
    Always creates a real Maya container, even when nested.

``with container("name", enabled=False):``
    Local override -- this block creates no Maya container (scope-only).

``rig.set_options(create_containers=False)``
    Global kill switch -- every ``with container():`` becomes scope-only.
    Useful for debugging when you want to peek at every node in the editor.
"""

from __future__ import annotations

import itertools
import logging
import numbers
from typing import Any, Dict, List, Optional, Set, Tuple

from maya import cmds
from rig.maya.attribute import Attribute
from rig._internal.maya_version import get_target_version, set_target_version
from rig._internal.node import Node


# Sentinel for "argument not passed" so ``None`` can mean "revert to default".
_UNSET = object()


LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
#  Plugin auto-loading for node types
# --------------------------------------------------------------------- #
#
# Some Maya node types live in optional plugins (``matrixNodes.mll``,
# ``quatNodes.mll``) that aren't always pre-loaded in vanilla mayapy / batch
# sessions. Rather than scatter ``cmds.loadPlugin`` calls across every site
# that creates one, we centralise the lookup here and load lazily on first
# use, with results cached for the rest of the session.
#
# To extend: add an entry to ``_NODE_TYPE_PLUGINS``. ``cmds.loadPlugin`` is
# idempotent and ``quiet=True`` swallows "already loaded" / "unknown plugin"
# noise, so over-listing is harmless.

_NODE_TYPE_PLUGINS: Dict[str, str] = {
    # ---- matrixNodes plugin ----
    "decomposeMatrix": "matrixNodes",
    "composeMatrix": "matrixNodes",
    "inverseMatrix": "matrixNodes",
    "multMatrix": "matrixNodes",
    "addMatrix": "matrixNodes",
    "wtAddMatrix": "matrixNodes",
    "pointMatrixMult": "matrixNodes",
    "transposeMatrix": "matrixNodes",
    "fourByFourMatrix": "matrixNodes",
    "determinant": "matrixNodes",
    "translationFromMatrix": "matrixNodes",
    "rotationFromMatrix": "matrixNodes",
    "scaleFromMatrix": "matrixNodes",
    "axisFromMatrix": "matrixNodes",
    "columnFromMatrix": "matrixNodes",
    "rowFromMatrix": "matrixNodes",
    "multiplyPointByMatrix": "matrixNodes",
    "multiplyVectorByMatrix": "matrixNodes",
    "aimMatrix": "matrixNodes",
    # ---- quatNodes plugin ----
    "quatAdd":         "quatNodes",
    "quatProd":        "quatNodes",
    "quatSub":         "quatNodes",
    "quatToEuler":     "quatNodes",
    "eulerToQuat":     "quatNodes",
    "quatNegate":      "quatNodes",
    "quatNormalize":   "quatNodes",
    "quatInvert":      "quatNodes",
    "quatConjugate":   "quatNodes",
    "quatSlerp":       "quatNodes",
    "quatToAxisAngle": "quatNodes",
    "axisAngleToQuat": "quatNodes",
}


_loaded_plugins: Set[str] = set()


def _ensure_plugin_for_node_type(node_type: str) -> None:
    """Lazily load the Maya plugin required by ``node_type``, if any.

    Idempotent and silent -- repeated calls are fast (cache hit), and a
    failed load is logged at debug level rather than raised so callers
    remain robust if Maya is running in a stripped-down environment.
    """
    plugin = _NODE_TYPE_PLUGINS.get(node_type)
    if plugin is None or plugin in _loaded_plugins:
        return
    try:
        cmds.loadPlugin(plugin, quiet=True)
    except RuntimeError as e:
        LOGGER.debug("loadPlugin(%s) failed for node type %s: %s", plugin, node_type, e)
    # Cache regardless of success -- if it failed, retrying won't help and
    # subsequent ``cmds.createNode`` will surface a clearer error.
    _loaded_plugins.add(plugin)


# --------------------------------------------------------------------- #
#  Global options
# --------------------------------------------------------------------- #


class ContainerOptions:
    """Mutable global flags for container behaviour."""

    create_containers:       bool = True           # if False, all `with container():` are no-ops
    use_shorthand:           bool = True           # type-aware shorthand (matrix->transform etc.)
    skip_selection:          bool = True           # createNode(skipSelect=True)
    cleanup_on_exit:         bool = False          # auto-call ``cleanup()`` on container exit
    maya_version:            Optional[int] = None  # target Maya version; None = actual
    flatten_containers:      bool = True           # if False, nested `with container():` create real Maya sub-containers AND publish_input/publish_output activate
    publish_attributes:      bool = True           # if False, publish_input/publish_output passthrough -- return source unchanged (no addAttr / no wiring)
    absorb_unit_conversions: bool = False          # if True, auto-inserted unitConversion nodes are pulled into the active container. Default is False because Maya's Node Editor "Hide Unit Conversion Nodes" view option (recently added) makes any container-member node that connects to a unitConversion render OUTSIDE its container -- so absorbing the conversion makes the rest of the container appear to leak. Set True if you want the conversions inside (closer to v3.A behavior) at the cost of the visual leak.
    native_multi_publish:    bool = False          # if False (default), MULTI/array attrs are NOT natively published on the container -- they stay on the real functional / host node and ``ctn.<name>`` resolves via ``Container.__getattr__`` + the multi registry. Maya's Node Editor renders a published array alias WITHOUT indices and rejects interactive (bare-parent) fan-in connects, so publishing arrays degrades the surface. Set True once Autodesk fixes array publishing to flip multis to native publishName/bindAttr like single plugs.
    constant_folding:        bool = True           # if True (default), math ops given only literal numbers return a Python value instead of building a Maya node (``abs(-5)`` -> ``5``). Set False (or use ``with force_nodes():``) so EVERY rig command materializes its node network even for all-literal inputs -- a debug/demo aid for inspecting the graph a literal call would build. Folds are never cached, so a node built under ``constant_folding=False`` and a folded scalar never collide, and dedupe of node-building calls is preserved in both modes.


def set_options(
    create_containers:       Optional[bool] = None,
    use_shorthand:           Optional[bool] = None,
    skip_selection:          Optional[bool] = None,
    cleanup_on_exit:         Optional[bool] = None,
    maya_version:            Any            = _UNSET,
    flatten_containers:      Optional[bool] = None,
    publish_attributes:      Optional[bool] = None,
    absorb_unit_conversions: Optional[bool] = None,
    native_multi_publish:    Optional[bool] = None,
    constant_folding:        Optional[bool] = None,
) -> None:
    """Set one or more global options. Unspecified args leave the current value
    unchanged. Always returns a snapshot dict of the current settings.

    Mirrors :func:`numpy.set_printoptions` -- setter only, returns ``None``.
    Use :func:`get_options` to introspect current values.

    Examples::

        rig.set_options(create_containers=False)   # debug: never create containers
        rig.set_options(use_shorthand=False)       # disable matrix->transform sugar
        rig.set_options(maya_version=2022)         # target Maya 2022 (legacy paths)
        rig.set_options(maya_version=None)         # revert to actual Maya version
        rig.set_options(constant_folding=False)    # debug: literal math builds nodes

        vals = rig.get_options()                   # introspect current settings

    The ``maya_version`` option lets you build rigs compatible with older
    Maya releases by forcing the DSL to dispatch to legacy node networks
    instead of native Maya 2024+ nodes. A few operations have NO legacy
    fallback (e.g. ``functions.pi()``, ``matrix.axis()``, ``vector.rotate()``)
    -- those will raise ``RuntimeError`` when ``maya_version < 2024``.

    Switching ``maya_version`` clears all NodeOp + ``@memoize`` caches so
    subsequent dedupe lookups don't return entries built for the previous
    target.
    """
    if create_containers is not None:
        ContainerOptions.create_containers = create_containers
    if use_shorthand is not None:
        ContainerOptions.use_shorthand = use_shorthand
    if skip_selection is not None:
        ContainerOptions.skip_selection = skip_selection
    if cleanup_on_exit is not None:
        ContainerOptions.cleanup_on_exit = cleanup_on_exit
    if flatten_containers is not None:
        ContainerOptions.flatten_containers = flatten_containers
    if publish_attributes is not None:
        ContainerOptions.publish_attributes = publish_attributes
    if absorb_unit_conversions is not None:
        ContainerOptions.absorb_unit_conversions = absorb_unit_conversions
    if native_multi_publish is not None:
        ContainerOptions.native_multi_publish = native_multi_publish
    if constant_folding is not None:
        ContainerOptions.constant_folding = constant_folding
    if maya_version is not _UNSET:
        previous                      = get_target_version()
        ContainerOptions.maya_version = maya_version
        set_target_version(maya_version)
        new = get_target_version()
        if new != previous:
            # Caches were keyed under the previous target version's
            # dispatch outcomes; clear them so subsequent calls re-dispatch.
            from rig._internal.memoize import _clear_all_caches

            _clear_all_caches()


def get_options() -> dict:
    """Return a snapshot dict of all current global options.

    Mirrors :func:`numpy.get_printoptions` -- pure getter, no side effects.
    Use :func:`set_options` to change settings.

    Examples::

        vals = rig.get_options()
        rig.get_options()                          # REPL auto-prints the dict
    """
    return {
        "create_containers":       ContainerOptions.create_containers,
        "use_shorthand":           ContainerOptions.use_shorthand,
        "skip_selection":          ContainerOptions.skip_selection,
        "cleanup_on_exit":         ContainerOptions.cleanup_on_exit,
        "maya_version":            ContainerOptions.maya_version,
        "flatten_containers":      ContainerOptions.flatten_containers,
        "publish_attributes":      ContainerOptions.publish_attributes,
        "absorb_unit_conversions": ContainerOptions.absorb_unit_conversions,
        "native_multi_publish":    ContainerOptions.native_multi_publish,
        "constant_folding":        ContainerOptions.constant_folding,
    }


class _ForceNodes:
    """Context manager returned by :func:`force_nodes` -- disables
    constant-folding within the block (restoring the prior setting on exit)
    so every rig command materializes its Maya node network, even for
    all-literal inputs.

    Save/restore (not hard-set-then-True) so nested ``with`` blocks and an
    outer ``set_options(constant_folding=False)`` compose correctly. The
    prior value is snapshotted per-instance, so nesting works; ``__exit__``
    runs on exceptions too, so the flag never leaks out of the block.

    NOT thread-safe: ``constant_folding`` is a process-global flag (like the
    rest of :class:`ContainerOptions` and the container stack), so a parallel
    build on another thread would observe the flipped flag. Use single-thread
    only -- which matches how Maya DG construction runs in practice.
    """

    __slots__ = ("_previous",)

    def __enter__(self) -> "_ForceNodes":
        self._previous                    = ContainerOptions.constant_folding
        ContainerOptions.constant_folding = False
        return self

    def __exit__(self, *exc: Any) -> bool:
        ContainerOptions.constant_folding = self._previous
        return False  # never suppress exceptions


def force_nodes() -> _ForceNodes:
    """Return a context manager that forces node creation within a ``with``
    block by disabling constant-folding (see
    :data:`ContainerOptions.constant_folding`).

    Inside the block, math ops given only literal numbers build their Maya
    node network instead of returning a Python value -- useful for inspecting
    or demoing the graph a literal expression would produce. The prior
    setting is restored on exit (including on exceptions).

    Dedupe is preserved: repeated identical calls inside the block return the
    same node network (folds are computed before -- and kept out of -- the
    cache, so node-building calls still memoize normally).

    Examples::

        from rig import force_nodes
        with force_nodes():
            plug = abs(-5)        # builds an ``absolute`` node, not ``5``
            plug2 = abs(-5)       # same node (deduped), not a second one

    Equivalent sticky form (no auto-revert)::

        rig.set_options(constant_folding=False)   # ... then back to True
    """
    return _ForceNodes()


def _load_defaults_from_file() -> None:
    """Load default options from ``rig/_defaults.toml`` if present.

    Read once at module import time. Each ``[options] key = value`` pair
    overrides the matching :class:`ContainerOptions` class attribute, so
    teams can ship a tuned default set without touching the code.

    Silently no-op when:
      * the file is missing -- in-code defaults remain authoritative
      * the file is unparseable -- logs a warning, in-code defaults win
      * a key in ``[options]`` doesn't match a known ContainerOptions
        attribute -- silently ignored (safe for forward/back compat)

    Per-call overrides via :func:`set_options` always take precedence
    over the file-loaded defaults.
    """
    import os

    try:
        import tomllib  # Python 3.11+ (Maya 2026+)
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return  # no TOML lib available -- skip silently

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "_defaults.toml",
    )
    if not os.path.isfile(config_path):
        return  # file missing -- in-code defaults win

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        LOGGER.warning(
            "Failed to parse %s (%s) -- using in-code defaults.",
            config_path,
            e,
        )
        return

    options = data.get("options", {})
    for key, value in options.items():
        if hasattr(ContainerOptions, key):
            setattr(ContainerOptions, key, value)
            LOGGER.debug("Loaded default %s=%r from %s", key, value, config_path)


# Apply at module-import time so the file-loaded defaults are in effect
# the moment ``rig`` is first imported.
_load_defaults_from_file()


# --------------------------------------------------------------------- #
#  Stack frame
# --------------------------------------------------------------------- #


# Process-monotonic id for stack frames. Used in ``_scope_key`` so that
# memoize entries for one ``with container(...)`` build can never collide
# with another build that happens to share ``(name, depth)``. A counter is
# used instead of ``id(self)`` because ``id()`` returns a CPython address
# that is recycled once a popped frame is garbage-collected, which let two
# sequential builds (e.g. repeated ``create_rail`` calls) share a scope key
# and serve each other stale cached networks.
_SCOPE_UID_COUNTER = itertools.count()


class _StackFrame:
    """A single entry in the active container stack.

    ``container_node`` is ``None`` when the frame represents a flattened
    nested block (no real Maya container was created).

    ``members`` is the list of Maya UUIDs of every node added through
    :meth:`_ContainerStack.add` while this frame was active. Used for the
    ``cmds.container(addNode=...)`` call when a real container exists.

    ``subgroups`` records flattened sub-scopes that were promoted into this
    frame on exit. Logical-only -- kept for a future ``Container.tree()``
    debug helper (v3).
    """

    __slots__ = (
        "name",
        "container_node",
        "preserve",
        "enabled",
        "depth",
        "members",
        "subgroups",
        "_scope_key",
        "cleanup_on_exit",
    )

    def __init__(
        self,
        name:            str,
        container_node:  Optional[Any],
        preserve:        bool,
        enabled:         bool,
        depth:           int,
        cleanup_on_exit: Optional[bool] = None,
    ) -> None:
        self.name           = name
        self.container_node = container_node  # rig.Container or None
        self.preserve       = preserve
        self.enabled        = enabled
        self.depth          = depth
        self.members:   List[str] = []   # node UUIDs
        self.subgroups: List[dict] = []  # logical sub-group records (for tree())
        # ``None`` = inherit ContainerOptions.cleanup_on_exit at __exit__ time;
        # explicit True/False overrides per-block.
        self.cleanup_on_exit: Optional[bool] = cleanup_on_exit
        # Used for memoize keying (so memo entries for nested containers don't
        # collide with the outer scope). The third element is a
        # process-monotonic uid -- NOT ``id(self)``, whose address recycles
        # after a popped frame is GC'd and would let separate builds sharing
        # ``(name, depth)`` collide on the same scope key.
        self._scope_key = (name, depth, next(_SCOPE_UID_COUNTER))


# --------------------------------------------------------------------- #
#  Container singleton
# --------------------------------------------------------------------- #


class _ContainerStack:
    """Module-level singleton: the active ``with container(...)`` stack.

    Used as a context-manager factory (``with container("name") as ctn: ...``)
    AND as the global registry that math-node factories consult to add new
    nodes to the active scope.
    """

    def __init__(self) -> None:
        self._stack: List[_StackFrame] = []
        # Pending args for the next __enter__ -- set by __call__.
        self._pending: Optional[dict] = None

    # -- context-manager entry point -- #

    def __call__(
        self,
        name:            str,
        preserve:        bool           = False,
        enabled:         bool           = True,
        cleanup_on_exit: Optional[bool] = None,
    ) -> "_ContainerStack":
        """Configure the next ``with container("name", ...) as ctn:`` block.

        Args:
            name: The container's name (also used for prefixing nodes
                created inside a flattened sub-scope).
            preserve: If ``True``, force-create a real Maya container even
                when nested. Default ``False`` (flatten when nested).
            enabled: If ``False``, do not create a Maya container even at
                the top level (scope-only no-op). Useful for debugging.
            cleanup_on_exit: If ``True``, call :func:`cleanup` on this
                container when the ``with`` block exits. ``None``
                (default) inherits :attr:`ContainerOptions.cleanup_on_exit`.
        """
        self._pending = {
            "name":            name,
            "preserve":        preserve,
            "enabled":         enabled,
            "cleanup_on_exit": cleanup_on_exit,
        }
        return self

    def __enter__(self) -> Any:
        if self._pending is None:
            raise RuntimeError(
                "container.__enter__ called without prior `container('name')` call"
            )
        args          = self._pending
        self._pending = None

        depth         = len(self._stack)
        is_top_level  = depth == 0
        wants_real_container = (
            args["enabled"]
            and ContainerOptions.create_containers
            and (
                is_top_level
                or args["preserve"]
                or not ContainerOptions.flatten_containers
            )
        )

        container_node = None
        if wants_real_container:
            ctn_name       = cmds.createNode("container", name=args["name"], skipSelect=True)
            container_node = Container(ctn_name)
            # If we're nested under an outer real container, register this
            # new sub-container as a member of the outer scope BEFORE we
            # push the new frame. At this point ``self._stack`` still has
            # only the outer frames, so ``self.add`` routes the new sub-
            # container to the outer leaf via ``cmds.container(outer,
            # edit=True, addNode=...)`` -- giving Maya the correct parent
            # -> child container relationship. Without this, sub-containers
            # (and their entire subtrees) would orphan to the scene root
            # under ``flatten_containers=False``.
            if self._stack:
                self.add(container_node)

        frame = _StackFrame(
            name            = args["name"],
            container_node  = container_node,
            preserve        = args["preserve"],
            enabled         = args["enabled"],
            depth           = depth,
            cleanup_on_exit = args["cleanup_on_exit"],
        )
        self._stack.append(frame)

        # Caller binds via `as ctn:` -- return the Container if we made one,
        # else None (flattened or disabled).
        return container_node

    def __exit__(self, _exc_type, _exc_val, _trace) -> None:
        if not self._stack:
            return
        frame = self._stack.pop()

        # If we flattened, promote our subgroup record + members to the parent
        # so the logical hierarchy is preserved for v3 debug tooling.
        if frame.container_node is None and self._stack:
            parent = self._stack[-1]
            parent.subgroups.append(
                {
                    "name":      frame.name,
                    "depth":     frame.depth,
                    "members":   list(frame.members),
                    "subgroups": list(frame.subgroups),
                }
            )
            parent.members.extend(frame.members)

        # ---- Auto-cleanup on exit ----
        # Decide whether to garbage-collect this scope's tagged orphans.
        # Per-block ``cleanup_on_exit=True/False`` overrides the global
        # ``ContainerOptions.cleanup_on_exit`` flag; ``None`` (the default)
        # inherits the global.
        should_cleanup = (
            frame.cleanup_on_exit
            if frame.cleanup_on_exit is not None
            else ContainerOptions.cleanup_on_exit
        )
        if should_cleanup:
            if frame.container_node is not None:
                # Real Maya container -- scope cleanup to its contents.
                cleanup(container=str(frame.container_node))
            elif frame.members:
                # Flattened scope -- sweep the explicit member list collected
                # during the with-block.
                names: List[str] = []
                for uid in frame.members:
                    name = _name_from_uuid(uid)
                    if name and cmds.objExists(name):
                        names.append(name)
                if names:
                    cleanup(_explicit_candidates=set(names))
            else:
                # No container, no members -- fall back to the scene-wide
                # sweep (catches anything tagged outside a containerised
                # scope, e.g. when ``create_containers=False``).
                cleanup()

    # -- node management -- #

    def createNode(
        self,
        node_type:  str,
        name:       Optional[str]  = None,
        ss:         Optional[bool] = None,
        skipSelect: Optional[bool] = None,
        container:  Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a Maya node and register it with the active scope.

        Returns a :class:`Node`. If ``name`` is given and we're inside a
        flattened sub-scope, prefixes ``name`` with the flattened scope name.
        """
        # Apply name prefix if we're inside a flattened sub-scope.
        if name is not None and self._stack:
            flatten_prefix = self._compute_flatten_prefix()
            if flatten_prefix:
                name = f"{flatten_prefix}_{name}"

        # Default skipSelect from options.
        if ss is None and skipSelect is None:
            skipSelect = ContainerOptions.skip_selection

        create_kwargs = dict(kwargs)
        if name is not None:
            create_kwargs["name"] = name
        if ss is not None:
            create_kwargs["ss"] = ss
        elif skipSelect is not None:
            create_kwargs["skipSelect"] = skipSelect

        # Lazily load the Maya plugin (matrixNodes / quatNodes) that owns
        # this node type, if any. No-op if already loaded or if the type
        # is built-in.
        _ensure_plugin_for_node_type(node_type)

        node_name = cmds.createNode(node_type, **create_kwargs)
        node      = Node(node_name)

        # GC ownership tag -- only on types we'd ever consider deleting.
        # See ``_GC_ELIGIBLE_TYPES`` for the whitelist; transforms / joints /
        # shapes / lights / objectSets are deliberately excluded so they're
        # never tagged and never eligible for cleanup().
        if node_type in _GC_ELIGIBLE_TYPES:
            try:
                cmds.addAttr(
                    node_name,
                    longName      = _RIG_TAG,
                    attributeType = "bool",
                    hidden        = True,
                )
                cmds.setAttr(f"{node_name}.{_RIG_TAG}", True, lock=True)
            except RuntimeError:
                # Some node types reject addAttr; without the tag the node
                # just won't be GC-eligible. That's a safe failure mode.
                pass

        # Add to the leaf real container (and record on every frame) by default.
        if container is None or container:
            self.add(node)

        return node

    def add(self, node: Any) -> None:
        """Add ``node`` (or list of nodes) to the leaf-level real container,
        and record its UUID in every frame on the stack."""
        if not self._stack:
            return

        node_names: List[str]
        if isinstance(node, (list, tuple)):
            node_names = [str(n) for n in node]
        else:
            node_names = [str(node)]

        # Track UUIDs on every stack frame.
        for name in node_names:
            try:
                uuid = _node_uuid(name)
            except ValueError:
                continue
            for frame in self._stack:
                if uuid not in frame.members:
                    frame.members.append(uuid)

        # Add to the leaf real container, if one exists.
        leaf = self._leaf_real_container()
        if leaf is not None:
            try:
                cmds.container(str(leaf), edit=True, addNode=node_names, force=True)
            except RuntimeError as e:
                LOGGER.debug(
                    "Failed to add %s to container %s: %s", node_names, leaf, e
                )

    def absorb_unit_conversions(self, dst: Any) -> None:
        """After connecting to ``dst``, absorb any auto-inserted
        ``unitConversion`` nodes into the active container.

        Maya inserts a ``unitConversion`` node whenever you ``connectAttr``
        between attributes of mismatched units (e.g. ``doubleLinear`` ->
        ``double``, ``doubleAngle`` -> ``double``). Without this cleanup
        the conversion node sits at scene root, and its presence forces
        Maya's Node Editor to render every internal node visibly --
        breaking container collapse.

        Mirror of Eric Vignola's ``Container._cleanup_unit_conversion``.
        Implementation uses raw ``cmds.listConnections`` with a Maya-side
        ``type=`` filter for the per-connection hot path. (Going through
        :meth:`Attribute.find_connected_nodes` cost ~5x more per call due
        to PyNode wrapping for every connection result + Python-side
        type filtering + repeated default-excludes set construction.)

        Args:
            dst: The destination plug that was just connected. Accepts
                an :class:`Attribute`, a :class:`Plug`, or a raw string
                path. Component plugs (``cv[N]``, ``vtx[N]``, etc.)
                are handled gracefully -- components don't trigger unit
                conversions, so the absorption is a no-op for them.
        """
        if not self._stack:
            return

        # Experiment knob: caller can opt out of auto-absorption to LEAVE
        # auto-inserted unitConversion nodes at the scene root -- useful
        # for diagnosing where unit-mismatched connections live.
        if not ContainerOptions.absorb_unit_conversions:
            return

        # Coerce dst to a string path for cmds. Avoids wrapping in
        # Attribute (which costs ~50us per call AND fails on component
        # plugs via TypeError("item is not a plug")).
        try:
            if isinstance(dst, str):
                dst_name = dst
            elif hasattr(dst, "full_name"):
                dst_name = dst.full_name
            else:
                dst_name = str(dst)
        except Exception:
            return

        # Single Maya call WITH the type filter -- let Maya's C++ do the
        # filtering rather than fetching all connections and filtering
        # in Python. RuntimeError on invalid attrs / component plugs
        # -> no-op (components have no unit conversion to absorb).
        try:
            convs = (
                cmds.listConnections(
                    dst_name,
                    source      = True,
                    destination = False,
                    type        = "unitConversion",
                )
                or []
            )
        except (RuntimeError, ValueError):
            return

        if convs:
            # Batched add (1 cmds.container call instead of N).
            self.add(convs)

    # -- introspection -- #

    @property
    def stack(self) -> List[_StackFrame]:
        """Read-only view of the active stack."""
        return list(self._stack)

    @property
    def containers(self) -> List[Optional[Any]]:
        """Compatibility shim mirroring Eric's ``container.containers`` list.

        Returns a list of ``Container`` instances (one per real container in
        the stack); flattened frames contribute ``None``.
        """
        return [f.container_node for f in self._stack]

    @property
    def is_active(self) -> bool:
        return bool(self._stack)

    # -- internals -- #

    def _leaf_real_container(self) -> Optional["Container"]:
        for frame in reversed(self._stack):
            if frame.container_node is not None:
                return frame.container_node
        return None

    def _compute_flatten_prefix(self) -> str:
        """Return ``foo_bar`` style breadcrumb prefix for the current stack
        of flattened sub-scopes (excludes the root real container)."""
        seen_real = False
        prefix_parts: List[str] = []
        for frame in self._stack:
            if frame.container_node is not None:
                seen_real = True
                continue
            if seen_real:
                prefix_parts.append(frame.name)
        return "_".join(prefix_parts)

    def _owns_attribute(self, attr):
        """Return True iff ``attr``'s owning node is a member of the leaf container.

        Used by :meth:`publish_input` to dispatch between the v4.G
        internal-plug form and the v4.G external-source extension.
        """
        if not isinstance(attr, Attribute):
            return False
        leaf = self._leaf_real_container()
        if leaf is None:
            return False
        try:
            owning_node = attr.node.name
        except Exception:
            return False
        try:
            members = cmds.container(str(leaf), query=True, nodeList=True) or []
        except Exception:
            return False
        return owning_node in members

    # -- Publishing API (v4.G) -- #

    def publish_input(
        self,
        source,
        name,
        value=True,
        **add_attr_kwargs,
    ):
        """Publish ``source`` as an INPUT on the active container.

        Two forms, dispatched by ``source`` kind:

        1. **Internal-plug form** (v4.G original) -- ``source`` is a
           :class:`Plug` owned by a node already INSIDE the active
           container. The published ``container.<name>`` is created with
           the plug's spec (cloned), seeded with the plug's current value
           when ``value=True``, and wired to drive ``source``.

        2. **External-source form** (v4.G extension) -- ``source`` is an
           external :class:`Plug`, a numeric scalar, a sequence, or
           ``None``. The published ``container.<name>`` is created via
           ``cmds.addAttr`` using ``add_attr_kwargs`` (auto-typing if no
           ``at`` / ``dt`` is provided); the source is then wired IN
           (Plug -> ``connectAttr``) or set (scalar/sequence ->
           ``setAttr``). The returned Plug becomes the reading point for
           any internal compute that needs the input. Lets you write
           module-factory functions::

               def lerp(a, b, w):
                   with container("lerp"):
                       a = container.publish_input(a, "input1")
                       b = container.publish_input(b, "input2")
                       w = container.publish_input(w, "weight",
                                                   min=0, max=1, dv=0.5)
                       return container.publish_output(
                           (b - a) * w + a, "output")

        ``add_attr_kwargs`` are forwarded directly to ``cmds.addAttr``
        in the external-source form (``min``, ``max``, ``dv``, ``at``,
        ``dt``, ``keyable``, etc.). They default ``keyable=True`` so the
        published knob shows up in the channel box. Ignored in the
        internal-plug form where the spec is already determined by the
        source plug.

        ``name`` is REQUIRED and should follow Maya's camelCase
        convention (e.g. ``"translateX"``, ``"weight"``, ``"blend"``).

        ``value=True`` (default) seeds / copies the source's value to the
        new container attribute. For internal-plug form: copies before
        wiring so the published knob starts at a meaningful value. For
        external-source form: drives the source into the new attr (no-op
        if ``source`` is ``None``).

        No-op when ``flatten_containers=True`` (the default) or when
        ``create_containers=False`` -- returns ``None``.

        Raises
        ------
        AttributeError
            if ``container.<name>`` already exists.
        RuntimeError
            if there is no active container scope when not in flatten mode.
        ValueError
            if ``name`` is empty.
        """
        if not name:
            raise ValueError(
                "name is required (publish_input must be given a "
                "camelCase Maya attr name)"
            )
        # Publish target = LEAF scope's container_node (if real). This
        # unifies all four cases into one rule:
        #   * create_containers=False / publish_attributes=False -> passthrough
        #     (covered by the precondition check below)
        #   * flatten_containers=False at any depth -> leaf is always real -> publish
        #   * flatten_containers=True at top level -> leaf is real -> publish
        #   * flatten_containers=True nested (default) -> leaf is flattened
        #     (container_node=None) -> passthrough so module-factory functions
        #     (lerp, vector_average, ...) don't pollute the outer container
        #     with their internals
        #   * with container(..., preserve=True) nested -> leaf is real even
        #     under flatten=True -> publish (caller explicitly asked for it)
        # Returning ``source`` unchanged in passthrough cases lets downstream
        # math read from it exactly as it would from a published Plug.
        if (
            not ContainerOptions.create_containers
            or not ContainerOptions.publish_attributes
        ):
            return source
        if not self._stack:
            return source
        leaf_frame = self._stack[-1]
        if leaf_frame.container_node is None:
            return source
        target = leaf_frame.container_node
        # Auto-resolve a bare multi-parent plug (e.g. ``worldMatrix``) to its
        # ``[0]`` element so it publishes as a SINGLE attr that can connect
        # onward. No-op for scalars / sequences / single plugs / explicit
        # ``multi=True``. See ``_resolve_multi_parent_source``.
        source = _resolve_multi_parent_source(source, add_attr_kwargs)
        # Internal-plug form: source is a Plug already owned by a node
        # inside the active container -> v4.G original behavior.
        if (
            isinstance(source, Attribute)
            and not add_attr_kwargs
            and self._owns_attribute(source)
        ):
            return _publish_to_container(
                source, target, direction="input", name=name, value=value
            )
        # External-source form: source is external Plug / scalar /
        # sequence / None -> create attr fresh, wire/set source IN.
        return _create_external_input(
            source,
            target,
            name            = name,
            value           = value,
            add_attr_kwargs = add_attr_kwargs,
        )

    def publish_output(
        self,
        source,
        name,
        value=True,
        **add_attr_kwargs,
    ):
        """Publish ``source`` as an OUTPUT on the active container.

        Three forms (dispatched by ``source`` kind):

        1. **Internal-plug form** (v4.G original) -- ``source`` is a
           :class:`Plug` owned by a node already INSIDE the active
           container. Spec cloned, value snapshot copied (when
           ``value=True``), source drives ``container.<name>``.

        2. **External-source form** -- ``source`` is an external
           :class:`Plug` (or accepted with kwargs that override the
           clone). Creates ``container.<name>`` per ``add_attr_kwargs``
           (auto-typed from source if no ``at``/``dt`` is given), then
           wires source -> container.<name>.

        3. **Sequence / PlugList form** (v4.G+ extension) -- ``source``
           is a sequence of Plugs / values. Creates a multi attribute
           on the container per ``add_attr_kwargs`` (auto-typed as
           compound multi if elements are vec3, scalar multi if
           elements are scalars), then wires each ``source[i]`` ->
           ``container.<name>[i]``.

        ``add_attr_kwargs`` are forwarded directly to ``cmds.addAttr``
        in the external / sequence forms (e.g. ``min``, ``max``,
        ``dv``, ``at``, ``dt``, ``multi``). Compound types (``double3``
        etc.) automatically get matching X/Y/Z children. Range kwargs
        on a compound-multi route to the children, not the parent.
        Defaults to ``keyable=False`` (outputs are typically read-only
        results).

        Returns the new Plug, or ``None`` when publishing is disabled
        (``flatten_containers=True``, ``create_containers=False``, or
        ``publish_attributes=False`` -- in which case the source is
        returned unchanged for passthrough mode).

        Raises
        ------
        AttributeError
            if ``container.<name>`` already exists.
        RuntimeError
            if there is no active container scope when not in flatten mode.
        ValueError
            if ``name`` is empty.
        """
        if not name:
            raise ValueError(
                "name is required (publish_output must be given a "
                "camelCase Maya attr name)"
            )
        # Publish target = LEAF scope's container_node (if real). See
        # ``publish_input`` for the full case-table; same rule applies here.
        if (
            not ContainerOptions.create_containers
            or not ContainerOptions.publish_attributes
        ):
            return source
        if not self._stack:
            return source
        leaf_frame = self._stack[-1]
        if leaf_frame.container_node is None:
            return source
        target = leaf_frame.container_node
        # Auto-resolve a bare multi-parent plug (e.g. ``worldMatrix``) to its
        # ``[0]`` element so it publishes as a SINGLE attr. No-op for
        # sequences / PlugLists / single plugs / explicit ``multi=True`` (the
        # sequence-output form still creates real multis from list sources).
        source = _resolve_multi_parent_source(source, add_attr_kwargs)
        # Internal-plug form: source is a Plug already owned by a node
        # inside the active container -> v4.G original behavior.
        if (
            isinstance(source, Attribute)
            and not add_attr_kwargs
            and self._owns_attribute(source)
        ):
            return _publish_to_container(
                source, target, direction="output", name=name, value=value
            )
        # External-source / sequence form -> new behavior.
        return _create_external_output(
            source,
            target,
            name            = name,
            value           = value,
            add_attr_kwargs = add_attr_kwargs,
        )


# Module-level singleton -- the public ``container`` name.
container = _ContainerStack()


def _resolve_multi_parent_source(source: Any, add_attr_kwargs: Dict[str, Any]) -> Any:
    """Resolve a multi-PARENT plug ``source`` to its ``[0]`` element when it
    is being published as a SINGULAR attribute.

    A bare multi parent such as ``transform.worldMatrix`` (a per-instance
    multi) carries no single value of its own. Mirroring its multi shape onto
    the container (via ``_clone_attribute`` / ``_infer_attr_type``) produces a
    published *multi* attr that cannot connect onward to a single-value
    consumer -- Maya raises "Incompatible multi-attribute parent levels".
    Resolving to element ``[0]`` -- the conventional "this instance" element
    -- lets a bare ``obj.wm`` "just work" when fed to matrix-consuming
    factories like :func:`rig.matrix.blend` /
    :func:`rig.matrix.slerp`.

    Skipped when:
      * ``source`` is not an :class:`Attribute` (scalars / sequences /
        ``None`` -- the sequence forms shape their own multis), or
      * the caller EXPLICITLY requested ``multi=True`` (the compound-multi /
        sequence publishing forms depend on that), or
      * ``source`` is not a multi parent (already a single value or an
        already-indexed element -- ``is_multi`` is False for both), or
      * ``source`` is a non-matrix multi -- a genuine *collection* such as
        ``plusMinusAverage.input1D`` (``data_type == "compound"``) or
        ``input3D``. Only matrix multi-parents (``worldMatrix`` /
        ``parentMatrix`` family, ``data_type == "matrix"``) carry no single
        value of their own; truncating a scalar/vector multi to ``[0]`` would
        be silent data loss.

    Falls back to returning ``source`` unchanged if the element lookup raises,
    so a malformed source still reaches the existing code paths.
    """
    if not isinstance(source, Attribute):
        return source
    if add_attr_kwargs.get("multi"):
        return source
    try:
        if source.is_multi and source.data_type == "matrix":
            return source[0]
    except (AttributeError, RuntimeError, TypeError, IndexError):
        pass
    return source


# --------------------------------------------------------------------- #
#  Native publishName / bindAttr machinery (v5)
# --------------------------------------------------------------------- #
#
# The publishing surface is built with Maya's NATIVE container publishing
# (``cmds.container(edit=True, publishName=..., bindAttr=...)``) instead of
# cloning a dynamic attribute onto the container node. ``container.<name>``
# becomes an ALIAS for the real inner plug, so deleting the container with
# ``cmds.container(removeContainer=True)`` drops only the alias -- the
# external<->inner connections (and the live rig) survive.
#
# Two homes for the bound attr:
#   * inner-plug publishes (outputs, internal inputs) bind the REAL inner
#     member plug directly (no extra node).
#   * created knobs / external-source inputs have no inner home, so they
#     live on a per-container HOST ``network`` node (one per materialized
#     container, lazily created).
#
# MULTI / array attrs are NOT published (Maya renders a published array
# alias without indices and rejects interactive fan-in). They stay on the
# functional / host node and ``Container.__getattr__`` resolves
# ``ctn.<name>`` through the multi registry. Flip ``native_multi_publish``
# to publish them natively once Autodesk fixes array publishing.

# Hidden bool tag identifying a per-container host network node. Distinct
# from ``__rig__`` (the GC-ownership tag) -- the host must NOT carry
# ``__rig__`` or ``cleanup`` could sweep it; it is created via raw
# ``cmds.createNode`` (never ``container.createNode``) so it never gets one.
_HOST_MARKER = "__rl_host__"

# ``{container_uuid: host_node_name}`` -- build-time reuse of the single host
# per container. Self-healing: a miss falls back to scanning the container's
# members for a ``__rl_host__`` node.
_HOST_CACHE: Dict[str, str] = {}

# ``{container_uuid: {published_name: (owner_node_uuid, real_attr_name)}}`` --
# resolution table for non-published MULTI attrs. Single plugs resolve via the
# native ``bindAttr`` query (scene-persistent); multis need this
# (session-scoped). ``real_attr_name`` is stored explicitly because an internal
# member multi keeps its ORIGINAL attr name (e.g. ``input3D``) even when
# published under a different name (e.g. ``arr``), so the published name alone
# cannot reconstruct the real plug.
_MULTI_REGISTRY: Dict[str, Dict[str, Tuple[str, str]]] = {}


def _is_host(node: str) -> bool:
    """True iff ``node`` carries the ``__rl_host__`` host-node marker.

    ``attributeQuery`` raises an assortment of exception types
    (``RuntimeError`` / ``ValueError`` / ``TypeError``) for an invalid node, so
    swallow broadly and report "not a host" for anything unqueryable.
    """
    try:
        return bool(cmds.attributeQuery(_HOST_MARKER, node=node, exists=True))
    except Exception:
        return False


def _get_or_create_host(container_node) -> "Node":
    """Return the per-container host ``network`` node, creating it lazily.

    The host holds created-knob / external-source input attrs that have no
    real inner home. Created via raw ``cmds.createNode`` (NOT
    ``container.createNode``) so it never gets the ``__rig__`` GC tag, then
    added as a member so ``bindAttr`` can bind its attrs to the container.
    """
    ctn = str(container_node)
    try:
        uuid = _node_uuid(ctn)
    except (
        Exception
    ):
        uuid = None

    if uuid is not None:
        cached = _HOST_CACHE.get(uuid)
        if cached and cmds.objExists(cached) and _is_host(cached):
            return Node(cached)

    # Fall back to scanning the container's members for an existing host.
    for member in cmds.container(ctn, query=True, nodeList=True) or []:
        if _is_host(member):
            if uuid is not None:
                _HOST_CACHE[uuid] = member
            return Node(member)

    # Create a fresh host. Raw createNode => no ``__rig__`` tag => GC-safe.
    host_name = cmds.createNode("network", name=f"{ctn}_host", skipSelect=True)
    try:
        cmds.addAttr(
            host_name, longName=_HOST_MARKER, attributeType="bool", hidden=True
        )
        cmds.setAttr(f"{host_name}.{_HOST_MARKER}", True, lock=True)
    except RuntimeError as e:
        LOGGER.debug("Failed to tag host %s: %s", host_name, e)
    try:
        cmds.container(ctn, edit=True, addNode=[host_name], force=True)
    except RuntimeError as e:
        LOGGER.debug("Failed to add host %s to %s: %s", host_name, ctn, e)
    if uuid is not None:
        _HOST_CACHE[uuid] = host_name
    return Node(host_name)


def _publish_native(container_node, inner_plug, name: str) -> str:
    """Natively publish ``inner_plug`` on ``container_node`` under ``name``.

    Emits ``publishName`` + ``bindAttr`` so ``container.<name>`` becomes an
    alias for the real ``inner_plug``. Callers MUST collision-check via
    :func:`_published_name_exists` first -- ``publishName`` silently
    auto-renames on collision (``"weight"`` -> ``"weight1"``) rather than
    raising. Returns the actual published name.
    """
    ctn       = str(container_node)
    published = cmds.container(ctn, edit=True, publishName=name) or name
    cmds.container(ctn, edit=True, bindAttr=(str(inner_plug), published))
    return published


def _published_name_exists(container_node, name: str) -> bool:
    """True iff ``name`` is already taken on ``container_node`` -- either a
    native published name / real attr (``attributeQuery`` sees both) or a
    registered multi."""
    ctn = str(container_node)
    try:
        if cmds.attributeQuery(name, node=ctn, exists=True):
            return True
    except Exception:
        pass
    try:
        uuid = _node_uuid(ctn)
    except Exception:
        return False
    return name in _MULTI_REGISTRY.get(uuid, {})


def _register_multi(container_node, name: str, owner) -> None:
    """Record ``ctn.<name>`` -> the REAL multi plug for a non-published multi so
    :meth:`Container.__getattr__` can resolve it.

    ``owner`` may be the real inner Plug (``node.attr``, whose attr can differ
    from ``name`` for an internal member multi) or a host Node (no attr, in
    which case the attr defaults to ``name``). Stores
    ``(owner_node_uuid, real_attr_name)`` so :func:`_resolve_published` can
    rebuild the real plug rename-safely.
    """
    owner_node, _, attr = str(owner).partition(".")
    try:
        uuid       = _node_uuid(str(container_node))
        owner_uuid = _node_uuid(owner_node)
    except Exception:
        return
    _MULTI_REGISTRY.setdefault(uuid, {})[name] = (owner_uuid, attr or name)


def _resolve_published(container_node, name: str):
    """Resolve ``ctn.<name>`` to the REAL bound Plug, or ``None``.

    Tries the native ``bindAttr`` table first (single plugs; survives scene
    save/reopen), then the multi registry (session-scoped). Returns the real
    inner / host Plug -- NEVER the published alias, whose ``findPlug`` lookup
    the DSL relies on RAISES.
    """
    from rig._internal.plug import Plug

    ctn = str(container_node)
    try:
        bind = cmds.container(ctn, query=True, bindAttr=True) or []
    except Exception:
        bind = []
    # bind is flat: [innerPlug, publishedName, innerPlug, publishedName, ...]
    for i in range(0, len(bind) - 1, 2):
        if bind[i + 1] == name and cmds.objExists(bind[i]):
            return Plug(bind[i])

    try:
        uuid = _node_uuid(ctn)
    except Exception:
        return None
    entry = _MULTI_REGISTRY.get(uuid, {}).get(name)
    if entry is not None:
        owner_uuid, attr = entry
        owner = _name_from_uuid(owner_uuid)
        if owner and cmds.objExists(f"{owner}.{attr}"):
            return Plug(f"{owner}.{attr}")
    return None


def _is_container_member(container_node, plug) -> bool:
    """True iff ``plug``'s owning node is a member of ``container_node``.

    Native ``bindAttr`` only binds attrs that live on a container member, so
    a source whose node is NOT a member (e.g. the ``Plug >> container``
    shortcut handed an external plug) must be routed through the host path
    instead of bound directly.
    """
    try:
        owner = plug.node.name
    except Exception:
        return False
    try:
        members = cmds.container(str(container_node), query=True, nodeList=True) or []
    except (RuntimeError, ValueError):
        return False
    return owner in members


def _plug_is_multi(plug) -> bool:
    """True iff ``plug`` is a multi / array attribute (robust to bare plugs)."""
    try:
        return bool(plug.is_multi)
    except (AttributeError, RuntimeError):
        return False


def _plug_is_bound(container_node, plug) -> bool:
    """True iff ``plug`` is ALREADY natively published (bound) on the container.

    Maya's ``bindAttr`` binds a given plug only ONCE; a second ``bindAttr`` of
    the same plug under a new name is silently SKIPPED (``Warning: Skipping
    <plug>. It is already published.``), leaving a dangling published name. The
    publish path detects this so it can route the second publish through a
    DISTINCT host carrier attr instead (identity / passthrough -- e.g. an ease
    whose output IS its input).
    """
    ctn = str(container_node)
    try:
        bind = cmds.container(ctn, query=True, bindAttr=True) or []
    except (RuntimeError, ValueError):
        return False
    plug_str = str(plug)
    # bind is flat: [innerPlug, publishedName, ...]; even indices are plugs.
    return any(bind[i] == plug_str for i in range(0, len(bind) - 1, 2))


def _destroy_published_name(container_node, name: str) -> bool:
    """Tear down a natively published ``name`` on ``container_node``.

    Returns ``True`` if a published name (native single or registered multi)
    was found and removed, ``False`` otherwise (caller falls back to a plain
    ``cmds.deleteAttr``).

    With native publishing, ``container.<name>`` is a ``publishName`` /
    ``bindAttr`` ALIAS, not a real attr. A plain ``cmds.deleteAttr`` on the
    alias deletes the bound attr and clears the bind table but leaves the
    published NAME dangling (``attributeQuery`` still reports it). The correct
    teardown is ``unbindAttr`` -> ``unpublishName``. The underlying carrier
    attr is deleted ONLY when it lives on the per-container host node (a
    purpose-built knob); attrs on real inner nodes are left intact so
    destroying the public API never mutates internal compute.
    """
    ctn   = str(container_node)
    found = False

    # Native single: locate the plug bound to ``name`` and unbind it.
    try:
        bind = cmds.container(ctn, query=True, bindAttr=True) or []
    except Exception:
        bind = []
    bound_plug = None
    for i in range(0, len(bind) - 1, 2):
        if bind[i + 1] == name:
            bound_plug = bind[i]
            break
    if bound_plug is not None:
        try:
            cmds.container(ctn, edit=True, unbindAttr=(bound_plug, name))
            found = True
        except RuntimeError as e:
            LOGGER.debug("unbindAttr %s.%s failed: %s", ctn, name, e)

    # Unpublish the name IFF it is actually a published name (NOT a genuine
    # attr added via ``ctn << Float(...)`` -- those have no publish entry and
    # must fall through to a plain ``deleteAttr``). Catches the dangling case
    # too (publishName persists even after its bindAttr was auto-cleared).
    try:
        published = cmds.container(ctn, query=True, publishName=True) or []
    except Exception:
        published = []
    if name in published:
        try:
            cmds.container(ctn, edit=True, unpublishName=name)
            found = True
        except (
            RuntimeError,
            ValueError,
        ) as e:
            LOGGER.debug("unpublishName %s.%s failed: %s", ctn, name, e)

    # Registered (non-published) multi teardown.
    try:
        uuid = _node_uuid(ctn)
    except Exception:
        uuid = None
    if uuid is not None:
        reg = _MULTI_REGISTRY.get(uuid)
        if reg and name in reg:
            owner_uuid, attr = reg[name]
            owner = _name_from_uuid(owner_uuid)
            # Only delete the carrier when it lives on the host; an internal
            # member multi's real attr must NOT be deleted on unpublish.
            if owner and _is_host(owner) and cmds.objExists(f"{owner}.{attr}"):
                try:
                    cmds.deleteAttr(f"{owner}.{attr}")
                except RuntimeError as e:
                    LOGGER.debug("deleteAttr %s.%s failed: %s", owner, attr, e)
            del reg[name]
            found = True

    # Delete the underlying carrier ONLY when it lives on the host node.
    if bound_plug is not None and cmds.objExists(bound_plug):
        owner_node = bound_plug.split(".", 1)[0]
        if _is_host(owner_node):
            try:
                cmds.deleteAttr(bound_plug)
            except RuntimeError as e:
                LOGGER.debug("deleteAttr %s failed: %s", bound_plug, e)

    return found


# --------------------------------------------------------------------- #
#  Publishing helper (v4.G) -- used by ``container.publish_input``,
#  ``container.publish_output``, and the ``Plug.__rshift__`` shortcut.
# --------------------------------------------------------------------- #


def _publish_to_container(
    plug,
    container_node,
    *,
    direction,
    name,
    value=True,
):
    """Internal dispatcher used by ``publish_input`` and ``publish_output``.

    Adds a new attribute on ``container_node`` that mirrors ``plug``'s
    type / shape, optionally copies the current value, and wires a
    connection in the requested direction.

    Parameters
    ----------
    plug : Plug
        The source attribute to publish.
    container_node : Node
        The container Node to receive the published attribute.
    direction : {'input', 'output'}
        - ``'input'``  : ``container.<name>`` drives ``plug`` (knob).
        - ``'output'`` : ``plug`` drives ``container.<name>`` (result).
    name : str
        The published attribute name (REQUIRED).
    value : bool, default True
        Copy the source's current value to the new attr before wiring.

    Returns
    -------
    Plug
        The newly created Plug on the container.

    Raises
    ------
    AttributeError
        if ``<container_node>.<name>`` already exists.
    ValueError
        if ``direction`` is not ``'input'`` or ``'output'``, or if ``name`` is empty.
    """
    if direction not in ("input", "output"):
        raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")
    if not name:
        raise ValueError("name is required")

    from rig._internal.node import Node

    if isinstance(container_node, Node):
        container_target = container_node
    else:
        container_target = Node(str(container_node))

    # Collision check -- raise a clearer error than Maya would. (``publishName``
    # silently auto-renames on collision rather than raising, so we must
    # pre-check; ``_published_name_exists`` covers native names AND multis.)
    if _published_name_exists(container_target, name):
        raise AttributeError(
            f"{container_target}.{name} already exists; choose a different name."
        )

    # Resolve a bare matrix multi-parent (e.g. ``worldMatrix``) to its ``[0]``
    # element here -- the shared choke point -- so EVERY publish entry point
    # agrees. The ``>>`` shortcut (``Plug.__rshift__``) routes here directly,
    # bypassing ``publish_input`` / ``publish_output`` (which resolve external
    # sources up front), so without this it would publish a *multi* whose alias
    # could not connect onward. Idempotent for the already-resolved
    # internal-plug path (``[0]`` element ``is_multi`` is False).
    plug = _resolve_multi_parent_source(plug, {})

    # The ``>>`` shortcut (``Plug.__rshift__``) routes here for ANY source,
    # including EXTERNAL plugs whose node is not a member of this container.
    # ``bindAttr`` requires the bound plug to live on a container member, so
    # an external source goes through the host path (create the attr on the
    # per-container host node, wire the source in) -- exactly like the
    # external-source forms of ``publish_input`` / ``publish_output``.
    if not _is_container_member(container_target, plug):
        if direction == "input":
            return _create_external_input(
                plug, container_target, name=name, value=value, add_attr_kwargs={}
            )
        return _create_external_output(
            plug, container_target, name=name, value=value, add_attr_kwargs={}
        )

    # MULTI / array plugs are not natively published (Maya renders a published
    # array alias without indices and rejects interactive fan-in) unless the
    # ``native_multi_publish`` capability flag is set. They stay on the real
    # inner node and resolve through the multi registry via
    # ``Container.__getattr__``.
    if _plug_is_multi(plug) and not ContainerOptions.native_multi_publish:
        _register_multi(container_target, name, plug)
        return plug

    # A given inner plug can be natively bound only ONCE -- ``bindAttr``
    # silently skips a second bind of the same plug, leaving a dangling
    # published name. This happens for identity / passthrough publishes where
    # the same plug is exposed under two names (e.g. ``in_linear``, whose
    # output IS its input). Route the second publish through the host path so
    # it gets a DISTINCT carrier attr (``host.<name>``) wired to the
    # already-bound plug -- restoring the pre-native two-attr passthrough.
    if _plug_is_bound(container_target, plug):
        if direction == "input":
            return _create_external_input(
                plug, container_target, name=name, value=value, add_attr_kwargs={}
            )
        return _create_external_output(
            plug, container_target, name=name, value=value, add_attr_kwargs={}
        )

    # Native publish: ``container.<name>`` becomes an alias for the real inner
    # plug -- no clone, no value-seed, no connectAttr. The alias IS the plug,
    # so external drives land on it (input) / it is the source (output), and
    # the binding survives ``removeContainer``. ``direction`` and ``value`` are
    # implied by the bound plug's writability and retained only for API parity
    # (the alias shares the inner plug's live value; there is nothing to seed).
    _publish_native(container_target, plug, name)
    return plug


# Maps an Attribute.data_type string to addAttr kwargs that will
# create a matching attribute via ``cmds.addAttr``. Used by the
# external-source form of ``publish_input`` when cloning the source
# plug's spec is overridden by user kwargs.
_DATA_TYPE_TO_ADDATTR: Dict[str, Dict[str, str]] = {
    "double":       {"at": "double"},
    "doubleLinear": {"at": "doubleLinear"},
    "doubleAngle":  {"at": "doubleAngle"},
    "long":         {"at": "long"},
    "short":        {"at": "short"},
    "bool":         {"at": "bool"},
    "float":        {"at": "float"},
    "string":       {"dt": "string"},
    "message":      {"at": "message"},
    "matrix":       {"dt": "matrix"},
    "double3":      {"at": "double3"},
    "float3":       {"at": "float3"},
}


# Compound attribute types that need explicit X/Y/Z (/W) child attrs
# when added via ``cmds.addAttr``. Maps parent attributeType to a tuple
# of (child attributeType, axis-letter sequence).
_COMPOUND_CHILDREN: Dict[str, Tuple[str, str]] = {
    "double3": ("double", "XYZ"),
    "float3":  ("float", "XYZ"),
    "long3":   ("long", "XYZ"),
    "short3":  ("short", "XYZ"),
    "double4": ("double", "XYZW"),
    "float4":  ("float", "XYZW"),
}

# addAttr kwargs that Maya only accepts on SCALAR child attrs, not on
# the compound parent. When a caller passes e.g. ``min=0, max=1`` along
# with ``at="double3"``, we route those onto the X/Y/Z children.
_CHILD_ONLY_KWARGS = frozenset(
    {
        "min",
        "minValue",
        "smn",
        "max",
        "maxValue",
        "smx",
        "softMin",
        "softMinValue",
        "softMax",
        "softMaxValue",
        "dv",
        "defaultValue",
    }
)


def _add_attr_with_compound_children(node_name: str, kwargs: Dict[str, Any]) -> None:
    """``cmds.addAttr`` wrapper that creates compound parents + their
    X/Y/Z (/W) children when ``at`` is one of the compound types.

    Range / default kwargs (``min``, ``max``, ``dv``, ...) are routed
    to the children, since Maya rejects them on compound parents.

    For non-compound attrs this is just a passthrough to ``cmds.addAttr``.
    """
    name = kwargs["longName"]
    at   = kwargs.get("at")

    if at not in _COMPOUND_CHILDREN:
        cmds.addAttr(node_name, **kwargs)
        return

    child_at, axes = _COMPOUND_CHILDREN[at]

    parent_kwargs = {k: v for k, v in kwargs.items() if k not in _CHILD_ONLY_KWARGS}
    child_extra   = {k: v for k, v in kwargs.items() if k in _CHILD_ONLY_KWARGS}

    cmds.addAttr(node_name, **parent_kwargs)
    for axis in axes:
        cmds.addAttr(
            node_name,
            longName      = f"{name}{axis}",
            attributeType = child_at,
            parent        = name,
            **child_extra,
        )


def _infer_attr_type(source: Any) -> Dict[str, Any]:
    """Return ``addAttr`` kwargs that match ``source``'s type / shape.

    Handles plain numbers, strings, :class:`Attribute` (mirrors its
    ``data_type``), short sequences, and 2D arrays / PlugList of
    compounds (returned with ``multi=True``). Falls back to ``{"at":
    "double"}`` for anything unrecognised so the caller still gets a
    valid attr.
    """
    if isinstance(source, bool):
        return {"at": "bool"}
    if isinstance(source, int):
        return {"at": "long"}
    if isinstance(source, numbers.Real):
        return {"at": "double"}
    # NOTE: Attribute MUST be checked BEFORE str -- :class:`Plug` is a
    # ``str`` subclass (via Attribute's inheritance chain), so a string
    # check would otherwise match Plug instances and mis-route them to
    # ``{"dt": "string"}``.
    if isinstance(source, Attribute):
        try:
            dt = source.data_type
        except Exception:
            return {"at": "double"}
        spec = dict(_DATA_TYPE_TO_ADDATTR.get(dt, {"at": "double"}))
        # If the source is itself a multi, mirror that.
        try:
            if source.is_multi:
                spec["multi"] = True
        except (AttributeError, RuntimeError):
            pass
        return spec
    if isinstance(source, str):
        return {"dt": "string"}
    if source is None:
        return {"at": "double"}
    # PlugList / list of Attribute -> infer multi from first element's shape.
    try:
        first_attr_elem = next((x for x in source if isinstance(x, Attribute)), None)
    except (TypeError, AttributeError):
        first_attr_elem = None
    if first_attr_elem is not None:
        try:
            dt = first_attr_elem.data_type
        except Exception:
            dt = "double"
        # v4.R: 3- or 4-element scalar PlugList -> compound vec3/vec4
        # (preserves the v4.Q-style compound publish shape for math-op
        # outputs that previously went through a `_constant` aggregator).
        # Falls through to multi-attr default for variable-length lists
        # or non-uniform element types.
        try:
            n_elems = len(source)
        except (TypeError, AttributeError):
            n_elems = None
        _COMPOUND_FROM_SCALAR = {
            ("double", 3): "double3",
            ("double", 4): "double4",
            ("float", 3): "float3",
            ("float", 4): "float4",
            ("long", 3): "long3",
            ("short", 3): "short3",
        }
        compound_at = _COMPOUND_FROM_SCALAR.get((dt, n_elems))
        if compound_at and all(
            isinstance(x, Attribute) and getattr(x, "data_type", None) == dt
            for x in source
        ):
            return {"at": compound_at}
        return {
            **_DATA_TYPE_TO_ADDATTR.get(dt, {"at": "double"}),
            "multi": True,
        }
    # Numpy / nested-list shape inference (sequence of vectors / matrices).
    try:
        import numpy as np

        arr = np.asarray(source)
    except Exception:
        arr = None
    if arr is not None and arr.ndim == 2:
        n_cols = arr.shape[1]
        if n_cols == 3:
            return {"at": "double3", "multi": True}
        if n_cols == 4:
            return {"at": "double4", "multi": True}
        if n_cols == 16:
            return {"dt": "matrix", "multi": True}
    try:
        n = len(source)
    except (TypeError, AttributeError):
        return {"at": "double"}
    if n == 16:
        return {"dt": "matrix"}
    # Flat all-numeric vec3 / vec4 literal -> compound double3 / double4.
    # Mirrors the PlugList-of-3/4 and 2D-array (``ndim == 2``) compound
    # heuristics above so a raw ``[x, y, z]`` passed to ``publish_input``
    # types as a vector, not a scalar. Without this a flat vector literal
    # fell through to ``{"at": "double"}`` and ``publish_input`` crashed
    # injecting the sequence into a scalar attr -- which broke ``slerp`` /
    # ``lerp`` on literal vectors. ``bool`` is excluded (an ``int``
    # subclass, but not a vector channel).
    if n in (3, 4) and all(
        isinstance(x, numbers.Real) and not isinstance(x, bool) for x in source
    ):
        return {"at": "double3" if n == 3 else "double4"}
    return {"at": "double"}


def _create_external_input(
    source:          Any,
    container_node,
    *,
    name:            str,
    value:           bool,
    add_attr_kwargs: Dict[str, Any],
):
    """Create the input attr on the per-container HOST node, publish it, and
    wire/set ``source`` IN.

    Used by :meth:`_ContainerStack.publish_input` for the v4.G
    external-source form (source is external Plug / scalar / sequence /
    ``None``). The created knob has no real inner home, so it lives on the
    host ``network`` node; ``container.<name>`` is a native publishName /
    bindAttr alias for ``host.<name>`` (or, for a multi, the host attr resolved
    through the multi registry). Returns the real ``host.<name>`` Plug so the
    DSL can read it (the alias's ``findPlug`` would raise).
    """
    from rig._internal.plug import Plug
    from rig.spec._base import _clone_attribute

    if isinstance(container_node, Node):
        container_target = container_node
    else:
        container_target = Node(str(container_node))

    # Collision check -- raise a clearer error than Maya would.
    if _published_name_exists(container_target, name):
        raise AttributeError(
            f"{container_target}.{name} already exists; choose a different name."
        )

    host = _get_or_create_host(container_target)

    # 1. Create the attribute on the HOST node.
    if isinstance(source, Attribute) and not add_attr_kwargs:
        # External Plug, no kwargs override -- clone its spec.
        _clone_attribute(source, host, attr_name=name)
    else:
        # Scalar / sequence / None / external-Plug-with-kwargs -- addAttr.
        kwargs             = dict(add_attr_kwargs)
        kwargs["longName"] = name
        if "at" not in kwargs and "dt" not in kwargs:
            kwargs.update(_infer_attr_type(source))
        # Default to keyable=True so published knobs land in the channel
        # box (matches user expectations for an exposed input).
        kwargs.setdefault("keyable", True)
        _add_attr_with_compound_children(str(host), kwargs)

    new_plug = Plug(f"{host}.{name}")

    # 2. Publish the host attr natively (single) or register it (multi).
    if _plug_is_multi(new_plug) and not ContainerOptions.native_multi_publish:
        _register_multi(container_target, name, host)
    else:
        _publish_native(container_target, new_plug, name)

    # 3. Wire / set the source via the DSL ``<<`` operator. It dispatches
    #    automatically: Plug -> ``connectAttr``; numeric/sequence ->
    #    ``setAttr``; ``None`` -> disconnect (no-op for a fresh attr).
    if value and source is not None:
        new_plug << source

    return new_plug


def _create_external_output(
    source:          Any,
    container_node,
    *,
    name:            str,
    value:           bool,
    add_attr_kwargs: Dict[str, Any],
):
    """Create the output attr on the per-container HOST node, publish it, and
    wire ``source`` INTO it.

    Used by :meth:`_ContainerStack.publish_output` for the external-source
    and sequence-source forms (source is external Plug, sequence /
    PlugList of values or Plugs, scalar, or ``None``). The attr is created on
    the host ``network`` node; ``container.<name>`` is a native publishName /
    bindAttr alias for ``host.<name>`` (single / compound) or resolves through
    the multi registry (sequence / multi). Returns the real ``host.<name>``
    Plug.

    Sequence sources create a multi attribute and connect each
    ``source[i]`` -> ``host.<name>[i]``. Compound element types
    (``double3`` etc.) automatically get their X/Y/Z child attrs.
    """
    from rig._internal.list import PlugList
    from rig._internal.plug import Plug
    from rig._internal.types import _get_compound, _is_sequence
    from rig.spec._base import _clone_attribute

    if isinstance(container_node, Node):
        container_target = container_node
    else:
        container_target = Node(str(container_node))

    if _published_name_exists(container_target, name):
        raise AttributeError(
            f"{container_target}.{name} already exists; choose a different name."
        )

    host = _get_or_create_host(container_target)

    # Sequence / PlugList -> multi attribute on the host, per-index connect.
    is_sequence_source = isinstance(source, PlugList) or (
        not isinstance(source, Attribute)
        and not isinstance(source, str)
        and not isinstance(source, numbers.Real)
        and not isinstance(source, bool)
        and source is not None
        and _is_sequence(source)
    )
    if is_sequence_source:
        kwargs             = dict(add_attr_kwargs)
        kwargs["longName"] = name
        if "at" not in kwargs and "dt" not in kwargs:
            # v4.R: inspect the WHOLE source (not just first element)
            # so the 3-/4-element-scalar -> compound double3/double4
            # heuristic added in `_infer_attr_type` can fire. If the
            # inferred shape is a real compound (no `multi`), publish
            # as compound instead of forcing multi-attribute. Falls
            # back to the legacy behavior (force multi) for variable-
            # length sequences and other non-compound cases.
            inferred = _infer_attr_type(source)
            inferred_compound = "multi" not in inferred and inferred.get("at") in (
                "double3",
                "double4",
                "float3",
                "float4",
                "long3",
                "short3",
            )
            if inferred_compound:
                kwargs.update(inferred)
                kwargs.setdefault("keyable", False)
                _add_attr_with_compound_children(str(host), kwargs)
                new_plug = Plug(f"{host}.{name}")
                # Compound single (double3 etc.) -- native-publishable.
                _publish_native(container_target, new_plug, name)
                if value:
                    # Per-channel connect into the compound's children.
                    # Source must yield exactly N elements where N matches
                    # the compound axis count -- guaranteed by the
                    # `_infer_attr_type` heuristic above.
                    children = _get_compound(new_plug)
                    for elem, dst_child in zip(source, children):
                        dst_child << elem
                return new_plug
            # Otherwise fall through to multi-attribute path below.
            inferred.pop("multi", None)
            kwargs.update(inferred)
        # Force multi for sequence sources.
        kwargs["multi"] = True
        # Outputs are typically read-only results; default to non-keyable
        # but still channel-box visible would require explicit cb=True.
        kwargs.setdefault("keyable", False)

        _add_attr_with_compound_children(str(host), kwargs)
        new_plug = Plug(f"{host}.{name}")

        # MULTI attrs are not natively published (see module header) unless the
        # ``native_multi_publish`` flag is set -- they resolve through the
        # multi registry via ``Container.__getattr__``.
        if ContainerOptions.native_multi_publish:
            _publish_native(container_target, new_plug, name)
        else:
            _register_multi(container_target, name, host)

        if value:
            for i, elem in enumerate(source):
                # Each iteration: source[i] drives host.<name>[i].
                # ``<<`` on a single-index multi destination handles
                # both Plug (connect) and value (setAttr) sources via
                # the existing dispatch.
                new_plug[i] << elem

        return new_plug

    # External single Plug -> single-attribute clone + connect on the host.
    if isinstance(source, Attribute):
        if add_attr_kwargs:
            kwargs             = dict(add_attr_kwargs)
            kwargs["longName"] = name
            if "at" not in kwargs and "dt" not in kwargs:
                kwargs.update(_infer_attr_type(source))
            kwargs.setdefault("keyable", False)
            _add_attr_with_compound_children(str(host), kwargs)
        else:
            _clone_attribute(source, host, attr_name=name)
        new_plug = Plug(f"{host}.{name}")

        if _plug_is_multi(new_plug) and not ContainerOptions.native_multi_publish:
            _register_multi(container_target, name, host)
        else:
            _publish_native(container_target, new_plug, name)

        if value:
            # Output direction: source drives the new host attr.
            try:
                source.connect(new_plug, force=True)
            except Exception:
                pass
        return new_plug

    # Scalar / None / opaque value -> create attr on host, publish, seed.
    kwargs             = dict(add_attr_kwargs)
    kwargs["longName"] = name
    if "at" not in kwargs and "dt" not in kwargs:
        kwargs.update(_infer_attr_type(source))
    kwargs.setdefault("keyable", False)
    _add_attr_with_compound_children(str(host), kwargs)
    new_plug = Plug(f"{host}.{name}")
    _publish_native(container_target, new_plug, name)
    if value and source is not None:
        new_plug << source
    return new_plug


# --------------------------------------------------------------------- #
#  Container -- Maya container node wrapper, subclass of Node
# --------------------------------------------------------------------- #


class Container(Node):
    """Wraps a Maya ``container`` node.

    Subclass of :class:`rig.Node` -- inherits attribute access
    (``ctn.<plug_name>`` returns a :class:`Plug`), spec injection
    (``ctn << Float("blend")`` adds an attribute on the container itself),
    operator semantics, hashing, ``__str__`` / ``__fspath__``.

    Instances are produced by ``with container("name") as ctn:``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f'Container("{self._dg_node}")'

    # -- attribute lookup -- #

    def __getattr__(self, attr_name: str) -> Any:
        """Resolve ``ctn.<name>`` to the REAL bound plug for natively
        published names, falling back to :meth:`Node.__getattr__`.

        Native ``bindAttr`` makes ``container.<name>`` an alias whose
        ``MFnDependencyNode.findPlug`` lookup (which ``Node.__getattr__``
        relies on) RAISES. So published / registered names must be resolved
        via the bindAttr table + multi registry FIRST, returning the real
        inner / host Plug. Genuine container attrs (``ctn << Float("blend")``
        adds a real attr ON the container) are not in those tables and fall
        through to the inherited ``Node.__getattr__`` (``findPlug`` works on
        them). ``_``-prefixed names short-circuit (mirrors the base guard).
        """
        if attr_name.startswith("_"):
            raise AttributeError(attr_name)
        resolved = _resolve_published(self, attr_name)
        if resolved is not None:
            return resolved
        return super().__getattr__(attr_name)

    # -- garbage collection -- #

    def cleanup(
        self,
        *,
        extra_types:    frozenset = frozenset(),
        aggressive:     bool      = False,
        memoize_caches: bool      = True,
        dry_run:        bool      = False,
    ) -> dict:
        """Garbage-collect orphan rig-owned utility nodes inside this
        container.

        Convenience wrapper that delegates to the top-level
        :func:`cleanup` with ``container=self.name``. See that function
        for the argument and return-value documentation.
        """
        return cleanup(
            container      = self.name,
            extra_types    = extra_types,
            aggressive     = aggressive,
            memoize_caches = memoize_caches,
            dry_run        = dry_run,
        )


# --------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------- #


def _node_uuid(node_name: str) -> str:
    """Return the Maya UUID of ``node_name``.

    Uses :attr:`DGNode.uuid` (canonical API) which routes via
    ``MFnDependencyNode.uuid()`` -- rename-safe and faster than
    ``cmds.ls(... uid=True)``.
    """
    try:
        return Node(node_name).uuid
    except (ValueError, RuntimeError):
        raise ValueError(f"Cannot resolve UUID for {node_name!r}")


def _name_from_uuid(uid: str) -> Optional[str]:
    """Return the current Maya name for ``uid``, or None if not found."""
    if not uid:
        return None
    result = cmds.ls(uid) or []
    return result[0] if result else None


# --------------------------------------------------------------------- #
#  Garbage collection (v2.E)
# --------------------------------------------------------------------- #
#
# Design overview
# ===============
#
# The rig DSL creates utility nodes (multiplyDivide, decomposeMatrix,
# unitConversion, etc.) as side-effects of the math operators. When a
# user makes a mistake or temporarily queries a value, those nodes can
# get orphaned with no downstream consumer.
#
# :func:`cleanup` finds and deletes those orphans iteratively (deleting
# a leaf orphan exposes its inputs as new leaves) and then issues a
# single ``cmds.delete`` for the full set.
#
# Safety is enforced by THREE filters:
#
#  1. **Per-node ``__rig__`` tag** -- every node ``container.createNode``
#     creates carries a hidden bool attr. Without the tag, cleanup ignores
#     the node entirely (e.g. user-authored controllers).
#
#  2. **Type whitelist** (:data:`_GC_ELIGIBLE_TYPES`) -- only utility DG
#     types are eligible. ``transform``, ``joint``, shapes, lights,
#     ``objectSet``, etc. are NEVER tagged in :meth:`createNode` and so
#     never become candidates.
#
#  3. **Downstream-connection check** -- a candidate is only deleted when
#     every downstream consumer is *also* in the obsolete set. Iterates
#     to fixpoint.
#
# The scene IS the implicit top-level container -- :func:`cleanup` works
# whether or not ``rig.set_options(create_containers=True)`` is on.
#
# --------------------------------------------------------------------- #


# Hidden-attr name used as the rig-ownership marker.
_RIG_TAG = "__rig__"

# Node types we'll ever consider deleting. Strictly DG utility types --
# nothing DAG, nothing geometry, nothing scene-default. Created nodes of
# OTHER types are never tagged, so even if a user adds something exotic
# via ``Node.create("locator", ...)`` it's safe.
_GC_ELIGIBLE_TYPES: frozenset = frozenset(
    {
        # Arithmetic
        "multiplyDivide",
        "plusMinusAverage",
        "addDoubleLinear",
        "multDoubleLinear",
        "blendColors",
        "blendTwoAttr",
        # Branching / clamping
        "condition",
        "clamp",
        "setRange",
        "remapValue",
        "remapColor",
        # Matrix
        "decomposeMatrix",
        "composeMatrix",
        "multMatrix",
        "inverseMatrix",
        "transposeMatrix",
        "pointMatrixMult",
        "aimMatrix",
        "wtAddMatrix",
        "fourByFourMatrix",
        "pickMatrix",
        "blendMatrix",
        # Quaternion (quatNodes plugin)
        "quatAdd",
        "quatSub",
        "quatProd",
        "quatNegate",
        "quatNormalize",
        "quatInvert",
        "quatConjugate",
        "quatToEuler",
        "eulerToQuat",
        "quatSlerp",
        # Vector / geometry
        "angleBetween",
        "distanceBetween",
        "vectorProduct",
        # Maya 2024+ math nodes
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "absolute",
        "floor",
        "ceil",
        "truncate",
        "clampRange",
        "min",
        "max",
        "normalize",
        "determinant",
        "lerp",
        "smoothStep",
        "sum",
        "average",
        "power",
        "log",
        "exp",
        "negate",
        "modulo",
        "divide",
        "subtract",
        "multiply",
        "add",
        "and",
        "or",
        "not",
        "equal",
        "greaterThan",
        "lessThan",
        # Helpers
        "reverse",
        # Auto-inserted by Maya (unit mismatch)
        "unitConversion",
        # Sidecars used by ``_constant``
        #   - ``network``  : scalar / 1..4-element vector constants
        #   - ``holdMatrix``: matrix-shaped (3x3 / 4x4 / 9-flat / 16-flat) constants
        "network",
        "holdMatrix",
    }
)


def _is_rig_owned(node: str) -> bool:
    """True iff ``node`` carries the ``__rig__`` ownership tag."""
    try:
        return bool(cmds.attributeQuery(_RIG_TAG, node=node, exists=True))
    except (RuntimeError, ValueError):
        return False


def _is_gc_eligible(
    node:        str,
    extra_types: frozenset = frozenset(),
    aggressive:  bool      = False,
) -> bool:
    """Pre-flight filters before we even consider the connection check."""
    if not cmds.objExists(node):
        return False
    if not _is_rig_owned(node):
        return False
    if not aggressive:
        if cmds.nodeType(node) not in (_GC_ELIGIBLE_TYPES | extra_types):
            return False
    if cmds.lockNode(node, q=True)[0]:
        return False
    try:
        if cmds.referenceQuery(node, isNodeReferenced=True):
            return False
    except RuntimeError:
        # referenceQuery raises on non-DG things; safe default = not referenced.
        pass
    return True


def _all_rig_owned(
    scope: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return ``[(node_name, owner_label), ...]`` for every rig-owned node.

    ``owner_label`` is the containing container's name, or
    ``"(scene root)"`` if the node isn't inside any container.

    If ``scope`` is given, restrict to nodes inside that container
    (still filtered by the ``__rig__`` tag).
    """
    pairs: List[Tuple[str, str]] = []
    if scope is not None:
        for n in cmds.container(scope, q=True, nodeList=True) or []:
            if _is_rig_owned(n):
                pairs.append((n, scope))
        return pairs

    # Scene-wide: enumerate every node whose ``__rig__`` plug exists.
    # ``cmds.ls('*.__rig__', objectsOnly=True)`` is the fast path.
    tagged = cmds.ls(f"*.{_RIG_TAG}", objectsOnly=True, recursive=True) or []
    for n in tagged:
        try:
            owner = cmds.container(q=True, findContainer=n) or "(scene root)"
        except RuntimeError:
            owner = "(scene root)"
        pairs.append((n, owner))
    return pairs


_INFRASTRUCTURE_TYPES = frozenset(
    {
        "hyperLayout",  # Hypergraph layout -- auto-attached to container members
        "container",  # Container itself -- message connection to members
        "objectSet",  # Set membership -- message connection
    }
)


def _consumers(node: str) -> set:
    """Downstream consumers of ``node`` minus self (LCG-cycle protection)
    and minus container infrastructure nodes.

    Maya auto-creates ``hyperLayout`` (and connects ``container`` /
    ``objectSet``) message-style links when nodes join a Maya container.
    These are NOT real data consumers -- counting them would make every
    container member look "alive" and break the orphan check.
    """
    raw = set(cmds.listConnections(node, source=False, destination=True) or []) - {node}
    return {
        c
        for c in raw
        if cmds.objExists(c) and cmds.nodeType(c) not in _INFRASTRUCTURE_TYPES
    }


def _compute_obsolete_set(candidates: set, max_iterations: int = 20) -> set:
    """Iteratively (virtually) determine the full obsolete set.

    A candidate is obsolete when every downstream consumer is itself
    in the obsolete set. Each pass adds newly-orphaned candidates;
    stops at fixpoint or ``max_iterations`` (defensive bound).

    NO ``cmds.delete`` is called here.
    """
    obsolete: set = set()
    for _ in range(max_iterations):
        new = set()
        for n in candidates - obsolete:
            if not cmds.objExists(n):
                obsolete.add(n)
                continue
            if _consumers(n).issubset(obsolete):
                new.add(n)
        if not new:
            break
        obsolete |= new
    return obsolete


def cleanup(
    *,
    container:            Optional[str] = None,
    extra_types:          frozenset     = frozenset(),
    aggressive:           bool          = False,
    memoize_caches:       bool          = True,
    dry_run:              bool          = False,
    _explicit_candidates: Optional[set] = None,
) -> dict:
    """Garbage-collect rig-owned utility nodes that no longer have any
    downstream consumer.

    Args:
        container: If given, restrict cleanup to a single Maya container's
            tagged contents. ``None`` (default) means scene-wide.
        extra_types: Additional node types to add to the GC whitelist for
            this call (e.g. plugin types this codebase doesn't know about).
        aggressive: If ``True``, bypass the type whitelist (still requires
            the ``__rig__`` ownership tag and the connection check).
        memoize_caches: If ``True`` (default), prune stale entries from
            every ``@memoize`` cache after deletion.
        dry_run: If ``True``, compute the obsolete set and report it but
            do NOT call ``cmds.delete``.
        _explicit_candidates: PRIVATE -- used by ``Container.__exit__``
            for flattened scopes to pass an explicit member list.

    Returns:
        A report dict::

            {
                "by_owner": {
                    "arm_rig":      ["mul3", "decomposeMatrix2"],
                    "(scene root)": ["mul9"],
                },
                "deleted_containers": ["empty_temp1"],
                "memoize_entries_pruned": 12,
                "dry_run": False,
            }

    The single ``cmds.delete`` call (when not dry-run) covers BOTH the
    obsolete nodes and the empty containers -- one operation, one undo
    chunk.
    """
    # ---- 1. Build the candidate map (by owner) ----
    by_owner: Dict[str, set] = {}
    if _explicit_candidates is not None:
        for n in _explicit_candidates:
            if not _is_gc_eligible(n, extra_types, aggressive):
                continue
            try:
                owner = cmds.container(q=True, findContainer=n) or "(scene root)"
            except RuntimeError:
                owner = "(scene root)"
            by_owner.setdefault(owner, set()).add(n)
    else:
        for n, owner in _all_rig_owned(scope=container):
            if _is_gc_eligible(n, extra_types, aggressive):
                by_owner.setdefault(owner, set()).add(n)

    # ---- 2. Compute the full obsolete set, virtually ----
    flat     = set().union(*by_owner.values()) if by_owner else set()
    obsolete = _compute_obsolete_set(flat)

    # ---- 3. Promote empty containers ----
    # If a container's full member set is in obsolete, the container is
    # itself empty after the delete.
    empty_ctns: set = set()
    if container is not None:
        ctn_iter = [container]
    else:
        ctn_iter = cmds.ls(type="container") or []
    for ctn in ctn_iter:
        if not cmds.objExists(ctn):
            continue
        members = set(cmds.container(ctn, q=True, nodeList=True) or [])
        if members and members.issubset(obsolete):
            empty_ctns.add(ctn)
        elif not members:
            # Container is already empty (orphan from a previous sweep).
            # Only include when scope is explicitly the whole scene.
            if container is None or container == ctn:
                empty_ctns.add(ctn)

    # ---- 4. Build the report (BEFORE deletion) ----
    report = {
        "by_owner": {
            owner: sorted(nodes & obsolete)
            for owner, nodes in by_owner.items()
            if (nodes & obsolete)
        },
        "deleted_containers": sorted(empty_ctns),
        "memoize_entries_pruned": 0,
        "dry_run": dry_run,
    }

    # ---- 5. Single delete call ----
    all_to_delete = obsolete | empty_ctns
    if all_to_delete and not dry_run:
        try:
            cmds.delete(list(all_to_delete))
        except RuntimeError as e:
            LOGGER.debug("cmds.delete failed in cleanup: %s", e)

    # ---- 6. Memoize cache prune ----
    if memoize_caches and not dry_run:
        try:
            from rig._internal.memoize import prune_memoize_caches

            report["memoize_entries_pruned"] = prune_memoize_caches()
        except Exception as e:
            LOGGER.debug("prune_memoize_caches failed: %s", e)

    return report


# --------------------------------------------------------------------- #
#  Imports needed by GC at module-bottom (kept here to avoid forward refs)
# --------------------------------------------------------------------- #

from typing import (
    Dict,
    Tuple,
)