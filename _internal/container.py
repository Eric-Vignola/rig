"""
Container scope and flattening.

The :data:`container` singleton is a context manager that tracks an active
stack of :class:`Container` (Maya container nodes). The first
``with container("name")`` block in a stack creates a real Maya container
node; nested blocks default to *flatten* (no inner container, nodes go into
the outermost), with naming-prefix breadcrumbs on created nodes.

:class:`Container` is a node class (a subclass of
:class:`rig.nodetypes.DGNode`, so a :class:`rig.Node`). It inherits all the
attribute access (``ctn.foo`` returns a :class:`Plug` owned by the container),
spec injection (``ctn << Float("blend")`` adds an attribute to the container
itself), and operator behaviour.

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
from maya.api import OpenMaya
from rig.nodetypes import (
    _base as _nodetypes_base,
    dg_node as _dg_node_module,
    errors as _nodetypes_errors,
)
from rig.nodetypes._base import (
    _cast_node,
    _check_attrs,
    _class_attr,
    _MISSING,
    _PLAIN_NODE_NAME,
    _path_instance_number,
    _shared_refused,
    Attribute,
    is_valid_maya_uid,
)
from rig.nodetypes.dag_node import _parent_path, DAGNode
from rig.nodetypes.dg_node import _create_template, DGNode
from rig.nodetypes.transform import Transform
from rig._internal.maya_version import get_target_version, set_target_version
from rig._internal.node import Node
from rig._internal.undo import _undo_chunk


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
        "_member_set",
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
        self.members:     List[str]  = []     # node UUIDs
        self._member_set: Set[str]   = set()  # mirror of ``members`` for O(1) dedupe
        self.subgroups:   List[dict] = []     # logical sub-group records (for tree())
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
            container_node = Container._wrap(ctn_name)
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
                self._add(container_node)

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
            parent._member_set.update(frame.members)

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
        **kwargs:   Any,
    ) -> Any:
        """Create a Maya node and register it with the active scope.

        Returns the typed node (the name Maya returned, cast as Maya resolves
        it: ``_cast_node``). If ``name`` is given and we're inside a flattened
        sub-scope, prefixes ``name`` with the flattened scope name.

        A ``parent=`` / ``p=`` is read before anything is made, by the
        reference rule (``DAGNode(parent)``): a name no node has raises
        NodeNotFoundError (Maya would make the node at the world with a
        warning), an ambiguous one AmbiguousNodeError. ``shared=`` is refused
        (TypeError): a create always makes a new node.
        """
        # Apply name prefix if we're inside a flattened sub-scope, on the leaf
        # of a namespaced name (``ns:x`` is ``ns:inner_x``).
        if name is not None and self._stack:
            flatten_prefix = self._compute_flatten_prefix()
            if flatten_prefix:
                name = _flattened_name(flatten_prefix, name)

        # Default skipSelect from options.
        if ss is None and skipSelect is None:
            skipSelect = ContainerOptions.skip_selection

        create_kwargs = dict(kwargs)
        if kwargs:
            # (the math nodes pass no other keyword: nothing to read)
            if "shared" in kwargs:
                raise _shared_refused(f"container.createNode({node_type!r}, shared=...)", node_type, name)
            for key in ("parent", "p"):
                parent = kwargs.get(key)
                if parent is not None:
                    create_kwargs[key] = _parent_path(parent)
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
        node      = _cast_node(node_name)

        # GC ownership tag -- only on types we'd ever consider deleting.
        # See ``_GC_ELIGIBLE_TYPES`` for the whitelist; transforms / joints /
        # shapes / lights / objectSets are deliberately excluded so they're
        # never tagged and never eligible for cleanup().
        if node_type in _GC_ELIGIBLE_TYPES:
            _gc_tag(node_name)

        # Add to the leaf real container (and record on every frame) by default.
        if container is None or container:
            self._add(node)

        return node

    def add(self, node: Any) -> None:
        """Add ``node`` (or list of nodes) to the leaf-level real container,
        and record its UUID in every frame on the stack."""
        self._add(node)

    def _add(self, node: Any) -> None:
        """:meth:`add`, for a node just created or one the caller holds."""
        if not self._stack:
            return

        items      = list(node) if isinstance(node, (list, tuple)) else [node]
        node_names = [str(n) for n in items]

        # Track UUIDs on every stack frame. A live node object gives its uuid,
        # so read it there instead of re-resolving the name (a name that parses
        # as a uuid still resolves, and raises, as before).
        for item, name in zip(items, node_names):
            dg_node = item if isinstance(item, DGNode) else None
            try:
                if (
                    dg_node is not None
                    and dg_node.is_valid
                    and not (len(name) >= 32 and is_valid_maya_uid(name))
                ):
                    uuid = dg_node.uuid
                else:
                    uuid = _node_uuid(name)
            except ValueError:
                continue
            for frame in self._stack:
                if uuid not in frame._member_set:
                    frame.members.append(uuid)
                    frame._member_set.add(uuid)

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
        to the typed cast of every connection result + Python-side
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
            return _is_direct_member(str(leaf), owning_node)
        except Exception:
            return False

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
        _refuse_container_member_name(name)
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

        3. **Sequence / List form** (v4.G+ extension) -- ``source``
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
        _refuse_container_member_name(name)
        # Auto-resolve a bare multi-parent plug (e.g. ``worldMatrix``) to its
        # ``[0]`` element so it publishes as a SINGLE attr. No-op for
        # sequences / Lists / single plugs / explicit ``multi=True`` (the
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


# --------------------------------------------------------------------- #
#  Typed creators (D13): ``Transform.create()`` & co. inside a scope
# --------------------------------------------------------------------- #


def _call_tracking_creation(fn: Any, args: tuple, kwargs: dict, discard: bool = False) -> tuple:
    """Call ``fn`` and return ``(result, created)``: the full names of the
    nodes Maya created during the call for the call itself, the nodes a scope
    registers. With ``discard``, a call that raises has what it made for
    itself deleted (`_discard_made`) before the error propagates: a create or
    define whose attribute value is refused leaves no half-built node.

    A node-added callback is the only exact way to tell what a command
    made from what it merely returned: a query returns nodes it looked up,
    ``parent`` and ``rename`` return nodes that already existed, and
    ``polyCube`` makes a shape it never returns. A node the command created
    and deleted again within the call is dropped (its handle is no longer
    valid). So is a node it made for a node that already existed, which a
    delete of the scope's container must not take with it (see
    `_made_for_others`): a deformer's ``...ShapeOrig`` under the user's mesh,
    a history shape, a skin's shared ``bindPose``. Used by the ``rc.*``
    bridges and by the typed creators.
    """
    handles = []

    def on_added(obj, _client_data):
        handles.append(OpenMaya.MObjectHandle(obj))

    callback_id = OpenMaya.MDGMessage.addNodeAddedCallback(on_added, "dependNode")
    try:
        result = fn(*args, **kwargs)
    except BaseException:
        OpenMaya.MMessage.removeCallback(callback_id)
        if discard:
            _discard_made(handles)
        raise
    OpenMaya.MMessage.removeCallback(callback_id)

    made    = [handle.object() for handle in handles if handle.isValid()]
    created = []
    for obj in made:
        if _made_for_others(obj, made, result):
            continue
        if obj.hasFn(OpenMaya.MFn.kDagNode):
            created.append(OpenMaya.MFnDagNode(obj).fullPathName())
        else:
            created.append(OpenMaya.MFnDependencyNode(obj).name())
    return result, created


# what a failed call's cleanup never deletes: a scope's container (made with
# its first member) and its hyperLayout (deleting it deletes the container)
_KEPT_ON_DISCARD = frozenset({"container", "dagContainer", "hyperLayout"})


def _discard_made(handles: list) -> None:
    """Delete the nodes a call that raised made for itself (the live
    ``handles`` of `_call_tracking_creation`; not a node made for a node that
    existed before, `_made_for_others`, nor a scope's container), with
    ``cmds.delete`` (never ``cmds.undo()``): the failed call leaves the scene
    as it found it. A cleanup that fails is logged; the call's own error is
    the one raised."""
    made  = [handle.object() for handle in handles if handle.isValid()]
    names = []
    for obj in made:
        if _made_for_others(obj, made, None):
            continue
        fn = OpenMaya.MFnDependencyNode(obj)
        if fn.typeName in _KEPT_ON_DISCARD:
            continue
        if obj.hasFn(OpenMaya.MFn.kDagNode):
            names.append(OpenMaya.MFnDagNode(obj).fullPathName())
        else:
            names.append(fn.absoluteName())
    # a DAG child goes with its parent
    alive = [
        name for name in names
        if cmds.objExists(name) and not any(name.startswith(f"{other}|") for other in names)
    ]
    if alive:
        try:
            cmds.delete(alive)
        except RuntimeError as error:
            LOGGER.debug("could not delete %s after a failed create: %s", alive, error)


def _made_for_others(obj: Any, made: list, result: Any) -> bool:
    """True for a node a call made (`obj`, one of the MObjects `made`) that
    belongs to a node that existed before the call, so a scope must not
    register it (deleting the scope's container would delete it too):

    * an intermediate DAG object under a node the call did not make: the
      ``...ShapeOrig`` a skinCluster, blendShape or cluster puts under the
      deformed mesh's transform, the history shape a poly command adds to a
      mesh that had none. Deleted with the deformer, the mesh lost its rest
      shape and froze posed; left alone, Maya restores it when the deformer
      goes;
    * a ``dagPose`` the call does not return: the ``bindPose`` a skinCluster
      makes for its joints, which every later skin of those joints shares.

    A node the call returns (`result`: a node name or a list of them) is its
    own, whatever it is (``rc.dagPose(save=True, name="p")``)."""
    if obj.hasFn(OpenMaya.MFn.kDagNode):
        dag = OpenMaya.MFnDagNode(obj)
        if not dag.isIntermediateObject or not dag.parentCount():
            return False
        parent = dag.parent(0)
        if parent.hasFn(OpenMaya.MFn.kWorld) or any(parent == other for other in made):
            return False
        return not _returned(result, dag.fullPathName(), dag.partialPathName())
    if obj.apiType() != OpenMaya.MFn.kDagPose:
        return False
    name = OpenMaya.MFnDependencyNode(obj).name()
    return not _returned(result, name, name)


def _returned(result: Any, long_name: str, short_name: str) -> bool:
    """True if the command result `result` (a node name or a list of them)
    names the node of `long_name` / `short_name`."""
    names = result if isinstance(result, (list, tuple)) else (result,)
    return any(
        isinstance(name, str) and str(name) in (long_name, short_name) for name in names
    )


def _gc_tag(node_name: str) -> None:
    """Give ``node_name`` the hidden, locked ``__rig__`` bool that makes it a
    :func:`cleanup` candidate. The caller checks the type against
    ``_GC_ELIGIBLE_TYPES``. Some node types reject addAttr; without the tag
    the node just won't be GC-eligible, a safe failure mode."""
    try:
        cmds.addAttr(
            node_name,
            longName      = _RIG_TAG,
            attributeType = "bool",
            hidden        = True,
        )
        cmds.setAttr(f"{node_name}.{_RIG_TAG}", True, lock=True)
    except RuntimeError:
        pass


# The typed creates running now, acting or not: only the outermost one joins
# the scope (a create inside another, e.g. SkinCluster's or create_hierarchy's,
# runs plain and is tracked by the outer one).
_TYPED_DEPTH = 0

# The ``_create``s that forward ``skipSelect`` to ``cmds.createNode``; the
# others (the mesh command, skinCluster, blendShape, displayLayer, the shading
# engine's sets, the reference's file) take no such flag.
_SELECT_FORWARDING = (DGNode._create.__func__, DAGNode._create.__func__)

_DG_POST_CREATE = DGNode.post_create.__func__


def _makes_only_its_node(cls: type, run: Any) -> bool:
    """True for a typed create that makes exactly the node it returns, so it runs
    without the node-added tracking (about 30 us a create): `DGNode.create`'s
    template with DGNode's ``_create`` on a DG class, or DAGNode's on a transform
    class (a shape type's ``createNode`` also makes its transform), and DGNode's
    ``post_create`` (an override may make more nodes)."""
    if run is not _create_template:
        return False
    if getattr(cls.post_create, "__func__", None) is not _DG_POST_CREATE:
        return False
    make = getattr(cls._create, "__func__", None)
    if make is _SELECT_FORWARDING[0]:
        return not issubclass(cls, DAGNode)
    return make is _SELECT_FORWARDING[1] and issubclass(cls, Transform)


def _typed_create(
    cls: type, run: Any, args: tuple, kwargs: dict, name_index: Optional[int] = None
) -> Any:
    """The typed-create hook (``dg_node._TYPED_CREATE_HOOK``): run
    ``run(cls, args, kwargs)``, the body of `DGNode.create` or of an
    ``@_typed_creator``, joined to the active scope with
    :meth:`_ContainerStack.createNode`'s rules.

    ``container=`` is consumed, and says whether the new nodes are registered
    with the scope, as it does for ``createNode``: None means
    ``cls._CONTAINER_AWARE``, True registers, False does not. The call runs
    plain (``run`` alone) when no scope is open, when a typed create is already
    running (the depth counts every typed create, acting or not), or on a scene
    registry (``_CONTAINER_AWARE = False``) that is not registered. Otherwise,
    in one undo chunk (one ``cmds.undo()`` reverts the create, the tag and the
    registration):

    * on a ``_CONTAINER_AWARE`` class, an explicit ``name=`` / ``n=`` (or the
      positional name at ``name_index``) takes the flattened scope's prefix on
      its leaf, after any namespace (:func:`_flattened_name`), as in
      ``createNode``; a registry (registered with ``container=True``) is found
      again by name, so it is never prefixed;
    * ``skipSelect=True`` is added when ``ContainerOptions.skip_selection`` is
      on, neither ``ss`` nor ``skipSelect`` was given, and ``run`` is
      `DGNode.create`'s template on a class whose ``_create`` forwards it;
    * the returned node is tagged for :func:`cleanup` when
      ``cls.NATIVE_NODE_TYPE`` is GC-eligible and the class has no
      ``CUSTOM_NODE_TYPE`` (a user's metadata node is never collected);
    * when registered, ``run`` runs inside :func:`_call_tracking_creation`,
      unless it makes only the node it returns (:func:`_makes_only_its_node`:
      ``Transform.create()``, ``Joint.create()``, a DG class's create), and
      every node the call made for itself (not a deformer's Orig shape under
      the user's mesh, nor a shared bind pose), else the returned node, is
      registered with :meth:`_ContainerStack.add`.

    Every other keyword reaches ``run`` untouched.
    """
    global _TYPED_DEPTH
    joins     = kwargs.pop("container", None)
    aware     = cls._CONTAINER_AWARE
    registers = aware if joins is None else bool(joins)
    if not container._stack or _TYPED_DEPTH or not (aware or registers):
        _TYPED_DEPTH += 1
        try:
            return run(cls, args, kwargs)
        finally:
            _TYPED_DEPTH -= 1

    if aware:
        prefix = container._compute_flatten_prefix()
        if prefix:
            for key in ("name", "n"):
                name = kwargs.get(key)
                if name:
                    kwargs[key] = _flattened_name(prefix, name)
            if name_index is not None and len(args) > name_index and args[name_index]:
                args = (
                    *args[:name_index],
                    _flattened_name(prefix, args[name_index]),
                    *args[name_index + 1 :],
                )
    if (
        ContainerOptions.skip_selection
        and "ss" not in kwargs
        and "skipSelect" not in kwargs
        and run is _create_template
        and getattr(cls._create, "__func__", None) in _SELECT_FORWARDING
    ):
        kwargs["skipSelect"] = True

    with _undo_chunk("rig.create"):
        _TYPED_DEPTH += 1
        try:
            if not registers or _makes_only_its_node(cls, run):
                result, created = run(cls, args, kwargs), None
            else:
                result, created = _call_tracking_creation(run, (cls, args, kwargs), {})
        finally:
            _TYPED_DEPTH -= 1

        if (
            cls.NATIVE_NODE_TYPE in _GC_ELIGIBLE_TYPES
            and not cls.CUSTOM_NODE_TYPE
            and isinstance(result, DGNode)
        ):
            _gc_tag(result.name)
        if registers:
            if created and not _made_only(result, created):
                container.add(created)
            elif result is not None:
                container.add(result)
    return result


def _flattened_name(prefix: str, name: Any) -> str:
    """``name`` with the flattened scope's ``prefix`` on its leaf: ``x`` is
    ``inner_x``, ``ns:x`` is ``ns:inner_x`` and ``:x`` is ``:inner_x`` (a prefix
    in front of the namespace would name, and make, another namespace)."""
    namespace, colon, leaf = str(name).rpartition(":")
    return f"{namespace}{colon}{prefix}_{leaf}"


def _made_only(result: Any, created: list) -> bool:
    """True when the one node a tracked typed create made (`created`, full
    names) is the node object it returns: registering the object reads its uuid
    directly, where registering the name casts it again."""
    if len(created) != 1 or not isinstance(result, DGNode):
        return False
    try:
        name = result.long_name if isinstance(result, DAGNode) else result.name
    except (RuntimeError, ValueError):
        return False
    return name == created[0]


# `DGNode.create`, the typed create `Node.create` may add ``skipSelect`` to
_DG_CREATE = DGNode.create.__func__


def _makes_just_its_node(node_cls: type) -> bool:
    """True for a class whose typed create makes just the node of its type by
    ``cmds.createNode``: `DGNode.create` with DGNode's or DAGNode's ``_create``
    and DGNode's ``post_create``, no custom type (``transform``, ``joint``,
    ``choice`` ...). Its only positional argument would be a DAG parent, which
    ``Node.create`` takes as ``parent=``."""
    return (
        not node_cls.CUSTOM_NODE_TYPE
        and getattr(node_cls.create, "__func__", None) is _DG_CREATE
        and getattr(node_cls._create, "__func__", None) in _SELECT_FORWARDING
        and getattr(node_cls.post_create, "__func__", None) is _DG_POST_CREATE
    )


# The keywords of ``Node.create`` on a type no class is registered for that go
# to :meth:`_ContainerStack.createNode` (``cmds.createNode``'s flags and the
# scope's ``container=``); every other keyword is an attribute of the new node.
_NODE_CREATE_FLAGS = frozenset({"name", "n", "parent", "p", "skipSelect", "ss", "container"})


def _node_create(node_type: str, args: tuple, kwargs: dict) -> Any:
    """``Node.create(node_type, *args, **kwargs)`` (``_base._NODE_CREATE_HOOK``;
    `kwargs` is the call's own dict).

    A type a node class is registered for runs that class's ``create(*args,
    **kwargs)``: the typed create, which joins an open scope by
    :func:`_typed_create`'s rules (the registries opt out). ``skipSelect=True``
    is added first when ``ContainerOptions.skip_selection`` is on, neither
    ``ss`` nor ``skipSelect`` was given, and the class's create is
    `DGNode.create` with a ``_create`` that forwards the flag, as
    :meth:`_ContainerStack.createNode` defaults it (a registered shader
    class's own create makes the shader network). A type Maya classifies a
    surface shader, with no class of its own (``_base._CLASSIFY``:
    ``anisotropic``, a plug-in shader), is ``Material.create(type=...)``:
    the network, out of the scope unless ``container=True``. Any other type is
    :meth:`_ContainerStack.createNode`, given ``cmds.createNode``'s flags
    (``name`` / ``n``, ``parent`` / ``p``, ``skipSelect`` / ``ss``) and
    ``container=``; every other keyword is an attribute, as for a typed
    create: its name (and an enum field name given as its value) is checked
    on the type before the node is made, and the value is set with ``<<``
    once it exists (``Node.create("multiplyDivide", operation="divide",
    input1X=3)``). Both refuse a positional argument where the node takes
    none: after an unregistered type, or a type whose class makes just its
    node (:func:`_makes_just_its_node`; ``parent=`` places it). A class that
    builds its node from inputs (``_CREATE_TAKES_INPUTS``: a skinCluster, a
    blendShape, a mesh, a nurbsCurve, a reference) refuses a call with none,
    before anything is made (``cmds.blendShape`` alone deforms the selection).
    ``shared=`` is refused first, for every type: create always makes a new
    node."""
    if "shared" in kwargs:
        raise _shared_refused(
            f"Node.create({node_type!r}, shared=...)", node_type, kwargs.get("name", kwargs.get("n"))
        )
    node_cls = _nodetypes_base._NODE_CLASS_DICT.get(node_type)
    if node_cls is None:
        if args:
            raise TypeError(
                f"Node.create({node_type!r}, ...) takes keyword arguments only: no "
                f"node class is registered for {node_type!r}, so it is made by "
                f"createNode (got {len(args)} positional argument(s) after the type)"
            )
        classify = _nodetypes_base._CLASSIFY
        shader   = classify(node_type) if classify is not None else None
        if shader is not None:
            # a surface shader without an exact class: its network, as
            # Material.create makes it (out of the scope unless container=True)
            return shader.create(type=node_type, **kwargs)
        if kwargs and not _NODE_CREATE_FLAGS.issuperset(kwargs):
            return _node_create_with_attrs(node_type, kwargs)
        return container.createNode(node_type, **kwargs)
    if args and _makes_just_its_node(node_cls):
        raise TypeError(
            f"Node.create({node_type!r}, ...) takes keyword arguments only: "
            f"{node_cls.__name__}'s create makes just the node (got {len(args)} "
            f"positional argument(s) after the type; a parent is parent=...)"
        )
    inputs = node_cls._CREATE_TAKES_INPUTS
    if inputs and not args:
        raise TypeError(
            f"Node.create({node_type!r}, ...) needs the inputs a {node_cls.__name__} "
            f"is built from, after the type: Node.create({node_type!r}, {inputs}, "
            f"...), as {node_cls.__name__}.create takes them"
        )
    if (
        ContainerOptions.skip_selection
        and "ss" not in kwargs
        and "skipSelect" not in kwargs
        and getattr(node_cls.create, "__func__", None) is _DG_CREATE
        and getattr(node_cls._create, "__func__", None) in _SELECT_FORWARDING
    ):
        kwargs["skipSelect"] = True
    return node_cls.create(*args, **kwargs)


def _node_create_with_attrs(node_type: str, kwargs: dict) -> Any:
    """`_node_create` of an unregistered `node_type` given attribute keywords
    (the keywords of `kwargs` not in ``_NODE_CREATE_FLAGS``): each checked on
    the type before the node is made (a typo raises AttributeError, a wrong
    enum field name TypeError, nothing made), then set with ``<<``, in one
    ``rig.create`` undo step (a value the node refuses deletes it)."""
    attrs = {key: value for key, value in kwargs.items() if key not in _NODE_CREATE_FLAGS}
    for key in attrs:
        del kwargs[key]
    _ensure_plugin_for_node_type(node_type)
    try:
        _check_attrs(attrs, node_type=node_type)
    except AttributeError as error:
        raise AttributeError(
            f"Node.create({node_type!r}, ...): {error}, and no createNode flag is named "
            f"so ({', '.join(sorted(_NODE_CREATE_FLAGS))}); nothing was made"
        ) from None
    with _undo_chunk("rig.create"):
        node = container.createNode(node_type, **kwargs)
        # a value the node refuses deletes it: nothing half-built is left
        _dg_node_module._set_or_delete(node, attrs)
    return node


def _scope_name(name: str) -> Optional[str]:
    """The name a create inside the active flattened scope gives ``name`` (the
    scope prefix on its leaf, as :meth:`_ContainerStack.createNode` spells
    it), or None outside one: the scope hint of a NodeNotFoundError
    (``errors._SCOPE_HINT_HOOK``; "'inner_k' exists (the scope prefix)")."""
    if not container._stack:
        return None
    prefix = container._compute_flatten_prefix()
    return _flattened_name(prefix, name) if prefix else None


class _DefineScope:
    """What ``define`` reads from the container scope (``dg_node._DEFINE_HOOK``;
    the D31 pattern keeps nodetypes free of ``rig._internal`` imports)."""

    @staticmethod
    def prefix(aware: bool, typed: bool) -> str:
        """The flattened scope's prefix the create of a define puts on its
        name's leaf, or "": :func:`_typed_create` prefixes the outermost typed
        create (`typed`) of a container-aware class (`aware`),
        :meth:`_ContainerStack.createNode` every name."""
        if not container._stack or (typed and (_TYPED_DEPTH or not aware)):
            return ""
        return container._compute_flatten_prefix()

    @staticmethod
    def frames() -> List[tuple]:
        """The scope's frames, outermost first, as ``(requested name, the real
        Container or None)``."""
        return [(frame.name, frame.container_node) for frame in container._stack]

    @staticmethod
    def owner(node_name: str) -> Optional[str]:
        """The container that holds ``node_name``, or None."""
        return cmds.container(query=True, findContainer=[node_name])

    @staticmethod
    def create_node(node_type: str, **kwargs: Any) -> Any:
        return container.createNode(node_type, **kwargs)

    # the node-added tracking, the undo chunk and the plug-in loading
    track         = staticmethod(_call_tracking_creation)
    chunk         = staticmethod(_undo_chunk)
    ensure_plugin = staticmethod(_ensure_plugin_for_node_type)


_dg_node_module._TYPED_CREATE_HOOK = _typed_create
_dg_node_module._DEFINE_HOOK = _DefineScope
_nodetypes_base._NODE_CREATE_HOOK = _node_create
_nodetypes_errors._SCOPE_HINT_HOOK = _scope_name


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

    A per-instance multi read through an instance path resolves to that path's
    element, as its name does in cmds (``Node("|T2|S").worldMatrix`` is
    ``worldMatrix[1]``, the instance under ``T2``); any other matrix multi to
    ``[0]``.

    Falls back to returning ``source`` unchanged if the element lookup raises,
    so a malformed source still reaches the existing code paths.
    """
    if not isinstance(source, Attribute):
        return source
    if add_attr_kwargs.get("multi"):
        return source
    try:
        if source.is_multi and source.data_type == "matrix":
            index = _path_instance_number(source)
            return source[0 if index is None else index]
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


def _has_attr_or_alias(node: str, attr: str) -> bool:
    """``cmds.attributeQuery(attr, node=node, exists=True)``, via the API.

    A plain name that resolves to exactly one node is answered with
    ``hasAttribute`` (long or short name) plus ``findAlias``, since
    ``attributeQuery(exists)`` also matches aliases. Any other input takes
    ``attributeQuery`` itself, so its exceptions are unchanged.
    """
    if isinstance(node, str) and _PLAIN_NODE_NAME.match(node):
        try:
            sel = OpenMaya.MSelectionList()
            sel.add(node)
            if sel.length() == 1:
                fn = OpenMaya.MFnDependencyNode(sel.getDependNode(0))
                return fn.hasAttribute(attr) or not fn.findAlias(attr).isNull()
        except (RuntimeError, ValueError, TypeError):
            pass
    return bool(cmds.attributeQuery(attr, node=node, exists=True))


def _is_host(node: str) -> bool:
    """True iff ``node`` carries the ``__rl_host__`` host-node marker.

    ``attributeQuery`` raises an assortment of exception types
    (``RuntimeError`` / ``ValueError`` / ``TypeError``) for an invalid node, so
    swallow broadly and report "not a host" for anything unqueryable.
    """
    try:
        return _has_attr_or_alias(node, _HOST_MARKER)
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
            return _cast_node(cached)

    # Fall back to scanning the container's members for an existing host.
    for member in cmds.container(ctn, query=True, nodeList=True) or []:
        if _is_host(member):
            if uuid is not None:
                _HOST_CACHE[uuid] = member
            return _cast_node(member)

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
    return _cast_node(host_name)


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
    return _is_direct_member(str(container_node), owner)


def _is_direct_member(container_name: str, node_name: str) -> bool:
    """True iff ``node_name`` is listed in ``container_name``'s ``nodeList``.

    ``nodeList`` holds DIRECT members only and a node is a direct member of at
    most one container, so ``findContainer`` answers without scanning the
    member list, for a ``node_name`` in the format ``nodeList`` lists it in
    (the node's partial path name). Any other name, an underworld node (listed
    by its name under its shape), an instanced DAG node (listed under one of
    its paths only) and a query failure are inconclusive; they fall back to the
    ``nodeList`` scan, where an unqueryable container reports "not a member".
    """
    try:
        if "->" not in node_name:
            sel = OpenMaya.MSelectionList()
            sel.add(node_name)
            mobj = sel.getDependNode(0)
            if mobj.hasFn(OpenMaya.MFn.kDagNode):
                instanced = OpenMaya.MFnDagNode(mobj).isInstanced()
                listed    = sel.getDagPath(0).partialPathName()
            else:
                instanced = False
                listed    = OpenMaya.MFnDependencyNode(mobj).name()
            if sel.length() == 1 and not instanced and listed == node_name:
                found = cmds.container(query=True, findContainer=[node_name])
                if not found:
                    return False
                return found == container_name or _same_node(found, container_name)
    except (RuntimeError, ValueError):
        pass
    try:
        members = cmds.container(container_name, query=True, nodeList=True) or []
    except (RuntimeError, ValueError):
        return False
    return node_name in members


def _same_node(name_a: str, name_b: str) -> bool:
    """True iff the two names resolve to the same node. Raises ValueError if
    either does not resolve to exactly one node.

    Nodes of several references of one file share their UUIDs, so only the
    nodes themselves tell such copies apart.
    """
    nodes = []
    for name in (name_a, name_b):
        sel = OpenMaya.MSelectionList()
        sel.add(name)
        if sel.length() != 1:
            raise ValueError(f"{name!r} does not name exactly one node")
        nodes.append(sel.getDependNode(0))
    return nodes[0] == nodes[1]


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


def _refuse_container_member_name(name: str) -> None:
    """Raise ``ValueError`` if ``name`` is a Python member of :class:`Container`
    (``name``, ``uuid``, ``cleanup``, ...): ``ctn.<name>`` gives the member, so
    an attribute published under that name could never be read back. A member
    of the metaclass (``wrap``) is not one: a container reads it as its
    published attribute."""
    if _class_attr(Container, name) is not _MISSING:
        raise ValueError(
            f"{name!r} is a Container attribute "
            f"({type(getattr(Container, name)).__name__}); "
            f"choose another published name."
        )


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
        if ``direction`` is not ``'input'`` or ``'output'``, if ``name`` is empty,
        or if ``name`` is a :class:`Container` attribute (``name``, ``cleanup``,
        ...), which ``ctn.<name>`` would give instead of the published plug.
    """
    if direction not in ("input", "output"):
        raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")
    if not name:
        raise ValueError("name is required")
    _refuse_container_member_name(name)

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
    ``data_type``), short sequences, and 2D arrays / List of
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
    # List / list of Attribute -> infer multi from first element's shape.
    try:
        first_attr_elem = next((x for x in source if isinstance(x, Attribute)), None)
    except (TypeError, AttributeError):
        first_attr_elem = None
    if first_attr_elem is not None:
        try:
            dt = first_attr_elem.data_type
        except Exception:
            dt = "double"
        # v4.R: 3- or 4-element scalar List -> compound vec3/vec4
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
    # Mirrors the List-of-3/4 and 2D-array (``ndim == 2``) compound
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
    List of values or Plugs, scalar, or ``None``). The attr is created on
    the host ``network`` node; ``container.<name>`` is a native publishName /
    bindAttr alias for ``host.<name>`` (single / compound) or resolves through
    the multi registry (sequence / multi). Returns the real ``host.<name>``
    Plug.

    Sequence sources create a multi attribute and connect each
    ``source[i]`` -> ``host.<name>[i]``. Compound element types
    (``double3`` etc.) automatically get their X/Y/Z child attrs.
    """
    from rig._internal.plug import Plug
    from rig._internal.types import _get_compound, _is_list, _is_sequence
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

    # Sequence / List -> multi attribute on the host, per-index connect.
    is_sequence_source = _is_list(source) or (
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
#  Container -- Maya container node class, subclass of DGNode
# --------------------------------------------------------------------- #


class Container(DGNode):
    """A Maya ``container`` node.

    Subclass of :class:`rig.nodetypes.DGNode` (so a :class:`rig.Node`) --
    inherits attribute access (``ctn.<plug_name>`` returns a :class:`Plug`
    owned by the container: ``ctn.<plug_name>.node is ctn``), spec injection
    (``ctn << Float("blend")`` adds an attribute on the container itself),
    operator semantics, hashing, ``__str__`` / ``__fspath__``. Not registered
    for the ``container`` node type: a cast of a container (``Node("ctn")``)
    gives a plain node, which compares equal to the Container, both ways, with
    the same hash.

    Instances are produced by ``with container("name") as ctn:``.
    ``Container("box")`` refers to an existing container node (a node of
    another type raises NodeTypeError).
    """

    # how the reference's errors name this unregistered class's nodes
    _TYPE_LABEL = "container"

    _DEFINE_REFUSED = (
        "a container is made by its scope: with container('x'): makes one; "
        "Container('x') refers to one"
    )

    def __repr__(self) -> str:
        return f'Container("{self.name}")'

    @classmethod
    def _coerce(cls, node: Any) -> Any:
        """The reference's hook (see ``DGNode._coerce``): the unregistered
        Container wraps a node of type ``container`` (a cast of one is a plain
        node); any other node is refused."""
        if cmds.objectType(node.name, isAType="container"):
            return cls._wrap(node.name)
        return None

    def __eq__(self, other: Any) -> bool:
        # symmetric with a plain node of the same container (DGNode's own test
        # is ``isinstance(other, type(self))``, which a plain node fails)
        return isinstance(other, DGNode) and self.name == other.name

    __hash__ = DGNode.__hash__

    # the root Node's container-aware factory and finder: DGNode's typed
    # ``create`` / ``find_all`` would build / list nodes of this unregistered
    # class's inherited type ("entity")
    create   = classmethod(Node.create.__func__)
    find_all = classmethod(Node.find_all.__func__)

    # -- attribute lookup -- #

    def __getattr__(self, attr_name: str) -> Any:
        """Resolve ``ctn.<name>`` to the REAL bound plug for natively
        published names, falling back to :meth:`DGNode.__getattr__`.

        Native ``bindAttr`` makes ``container.<name>`` an alias whose
        ``MFnDependencyNode.findPlug`` lookup (which ``DGNode.__getattr__``
        relies on) RAISES. So published / registered names must be resolved
        via the bindAttr table + multi registry FIRST, returning the real
        inner / host Plug. Genuine container attrs (``ctn << Float("blend")``
        adds a real attr ON the container) are not in those tables and fall
        through to the inherited ``DGNode.__getattr__`` (``findPlug`` works on
        them), whose ``_`` rule applies: a ``_`` name resolves only to a Maya
        attr of the live container. Python members (``name``, ``cleanup``, ...)
        win over published names, which the publish guard refuses.
        """
        if attr_name[:1] != "_":
            resolved = _resolve_published(self, attr_name)
            if resolved is not None:
                return resolved
        return DGNode.__getattr__(self, attr_name)

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
        # a name Maya returned (or a node's own name): Maya's resolution
        return _cast_node(node_name).uuid
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
        return _has_attr_or_alias(node, _RIG_TAG)
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