"""Dynamic wrappers around ``maya.cmds`` that return :class:`Node` /
:class:`PlugList` instead of bare strings.

Usage::

    from rig.bridges import commands as rc

    nodes = rc.ls(sl=True)              # -> PlugList[Node, Node, ...]
    new   = rc.createNode("transform")  # -> Node
    rc.parent(child, parent)            # auto-coerces Node -> str
    rc.delete(some_node, container=False)  # opt out of container add

Wrappers are built lazily on first attribute access via PEP 562
``__getattr__`` (Python 3.7+). No import-time cost, no ``exec`` of
string-formatted code, no ~700 wrappers materialized eagerly.

Studio plugins that register commands on ``maya.cmds`` are picked up
automatically -- they appear after the plugin loads, on first call.

For occasional use without going through this module, use
:meth:`Node.wrap` to convert a ``maya.cmds`` result manually::

    from maya import cmds
    from rig import Node

    n = Node.wrap(cmds.createNode("transform"))
"""

from __future__ import annotations

from typing import Any, Callable

from maya import cmds as _mc
from maya.api import OpenMaya as _om
from rig._internal.node import Node


# Per-command wrapper cache. Built lazily by ``__getattr__``.
_WRAPPER_CACHE: dict = {}


def _call_tracking_creation(fn: Callable, args: tuple, kwargs: dict) -> tuple:
    """Call ``fn`` and return ``(result, created)``: the full names of the
    nodes Maya created during the call.

    A node-added callback is the only exact way to tell what a command
    made from what it merely returned: a query returns nodes it looked up,
    ``parent`` and ``rename`` return nodes that already existed, and
    ``polyCube`` makes a shape it never returns. A node the command created
    and deleted again within the call is dropped (its handle is no longer
    valid).
    """
    handles = []

    def on_added(obj, _client_data):
        handles.append(_om.MObjectHandle(obj))

    callback_id = _om.MDGMessage.addNodeAddedCallback(on_added, "dependNode")
    try:
        result = fn(*args, **kwargs)
    finally:
        _om.MMessage.removeCallback(callback_id)

    created = []
    for handle in handles:
        if not handle.isValid():
            continue
        obj = handle.object()
        if obj.hasFn(_om.MFn.kDagNode):
            created.append(_om.MFnDagNode(obj).fullPathName())
        else:
            created.append(_om.MFnDependencyNode(obj).name())
    return result, created


# Commands that should NOT have their args auto-coerced (they take callback
# strings, raw expressions, or evaluate code -- coercing args would break them).
_NO_COERCE = frozenset(
    {
        "evalDeferred",
        "scriptJob",
        "scriptNode",
        "expression",
        "undo",
        "redo",
        "undoInfo",
        "warning",
        "error",
    }
)


def _coerce(value: Any) -> Any:
    """Convert :class:`Node` / list-of-Nodes back to bare strings.

    Maya commands take string node names; the DSL holds Node objects. This
    bridge converts on the way IN to a cmds call. Plug objects are already
    str-subclasses so they pass through cmds directly without conversion.
    """
    if isinstance(value, Node):
        return str(value)
    if isinstance(value, (list, tuple)) and value:
        # Only convert if the list contains Nodes -- otherwise pass through
        # unchanged to avoid mutating arbitrary list inputs.
        if any(isinstance(x, Node) for x in value):
            return [str(x) if isinstance(x, Node) else x for x in value]
    return value


def _wrap_result(result: Any) -> Any:
    """Wrap a cmds output: str -> :class:`Node`, list[str] -> :class:`PlugList`,
    else passthrough (for booleans, numerics, dicts, None, etc.).

    Strings that don't resolve to valid nodes are passed through unchanged
    (so ``rc.getAttr("foo.attr")`` returning a string value won't be coerced).
    """
    if result is None:
        return None
    if isinstance(result, bool):  # bool is a subclass of int -- guard first
        return result
    if isinstance(result, str):
        try:
            return Node(result)
        except Exception:
            return result
    if isinstance(result, (list, tuple)):
        from rig._internal.list import PlugList

        wrapped = []
        for r in result:
            if isinstance(r, str):
                try:
                    wrapped.append(Node(r))
                except Exception:
                    wrapped.append(r)
            else:
                wrapped.append(r)
        try:
            return PlugList(wrapped)
        except Exception:
            return wrapped
    return result


def _make_wrapper(name: str) -> Callable:
    """Build a wrapper function for ``maya.cmds.<name>``."""
    fn          = getattr(_mc, name)
    coerce_args = name not in _NO_COERCE

    def wrapper(*args, **kwargs):
        # Opt-out of auto-container-add via ``container=False`` kwarg.
        # (Pop BEFORE the cmds call so it doesn't reach Maya.)
        add_to_container = kwargs.pop("container", True)

        if coerce_args:
            args   = tuple(_coerce(a) for a in args)
            kwargs = {k: _coerce(v) for k, v in kwargs.items()}

        # Only the nodes this call CREATES join the active container scope;
        # a query, a parent or a rename never moves a node into it. Track
        # creation only when a scope is open, so the callback costs nothing
        # otherwise.
        from rig._internal.container import container

        if add_to_container and container._stack:
            result, created = _call_tracking_creation(fn, args, kwargs)
            if created:
                try:
                    container.add(created)
                except Exception:
                    pass
        else:
            result = fn(*args, **kwargs)

        return _wrap_result(result)

    wrapper.__name__     = name
    wrapper.__qualname__ = f"rig.bridges.commands.{name}"
    wrapper.__doc__      = fn.__doc__
    wrapper.__wrapped__  = fn  # functools convention; help/inspect see original
    return wrapper


def __getattr__(name: str) -> Callable:
    """PEP 562: lazily wrap ``maya.cmds.<name>`` on first access.

    Picks up studio plugin commands automatically -- any time a plugin loads
    and registers a new ``maya.cmds.<name>``, calling ``rc.<name>`` for the
    first time after will build and cache a wrapper.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    cached = _WRAPPER_CACHE.get(name)
    if cached is not None:
        return cached
    fn = getattr(_mc, name, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"'rig.bridges.commands' has no command {name!r} "
            f"(no such attribute in maya.cmds)"
        )
    wrapped              = _make_wrapper(name)
    _WRAPPER_CACHE[name] = wrapped
    return wrapped


def __dir__() -> list:
    """Tab-completion: list all wrappable maya.cmds callables.

    Includes anything currently registered on ``maya.cmds`` (built-ins +
    loaded plugin commands).
    """
    return sorted(
        n for n in dir(_mc) if not n.startswith("_") and callable(getattr(_mc, n, None))
    )