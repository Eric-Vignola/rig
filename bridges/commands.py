"""Dynamic wrappers around ``maya.cmds`` that return :class:`Node` /
:class:`List` instead of bare strings.

Usage::

    from rig.bridges import commands as rc

    nodes = rc.ls(sl=True)              # -> List[Node, Node, ...]
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

# the node-added tracking the typed creators share (it lives with the scope)
from rig._internal.container import _call_tracking_creation
from rig._internal.node import Node
from rig.nodetypes._base import _cast_node, _ensure_node_valid, Attribute


# Per-command wrapper cache. Built lazily by ``__getattr__``.
_WRAPPER_CACHE: dict = {}


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
    str-subclasses so they pass through cmds directly without conversion, once
    their node is checked (see `_checked_plug`).
    """
    if isinstance(value, Node):
        return str(value)
    if isinstance(value, Attribute):
        return _checked_plug(value)
    if isinstance(value, (list, tuple)) and value:
        # Only convert if the list contains Nodes -- otherwise pass through
        # unchanged to avoid mutating arbitrary list inputs.
        if any(isinstance(x, Node) for x in value):
            return [
                str(x) if isinstance(x, Node)
                else _checked_plug(x) if isinstance(x, Attribute)
                else x
                for x in value
            ]
        for x in value:
            if isinstance(x, Attribute):
                _checked_plug(x)
    return value


def _checked_plug(plug: Attribute) -> Attribute:
    """`plug`, once its node is checked: a plug whose node was deleted or freed
    raises its ``"... already deleted!"`` (see `_ensure_node_valid`) instead of
    cmds reading its str buffer, which then names the node that took the name.
    cmds reads that buffer, as before (a component plug's names the component)."""
    _ensure_node_valid(plug)
    return plug


def _wrap_result(result: Any) -> Any:
    """Wrap a cmds output: str -> :class:`Node`, list[str] -> :class:`List`,
    else passthrough (for booleans, numerics, dicts, None, etc.).

    Strings that don't resolve to valid nodes are passed through unchanged
    (so ``rc.getAttr("foo.attr")`` returning a string value won't be coerced).
    A string is a name Maya returned, cast as Maya resolves it (see
    ``_cast_node``), never by ``Node(x)``'s lookup rule for written names.
    """
    if result is None:
        return None
    if isinstance(result, bool):  # bool is a subclass of int -- guard first
        return result
    if isinstance(result, str):
        try:
            return _cast_node(result)
        except Exception:
            return result
    if isinstance(result, (list, tuple)):
        from rig._internal.list import List

        wrapped = []
        for r in result:
            if isinstance(r, str):
                try:
                    wrapped.append(_cast_node(r))
                except Exception:
                    wrapped.append(r)
            else:
                wrapped.append(r)
        try:
            return List(wrapped)
        except Exception:
            return wrapped
    return result


def _make_wrapper(name: str) -> Callable:
    """Build a wrapper function for ``maya.cmds.<name>``. Each call runs the
    function maya.cmds has NOW: a plug-in unloaded and loaded again gets a new
    one, and calling the old one crashes Maya."""
    original    = getattr(_mc, name)
    coerce_args = name not in _NO_COERCE

    def wrapper(*args, **kwargs):
        fn = getattr(_mc, name, None)
        if fn is None:
            raise RuntimeError(f"maya.cmds has no command {name!r} (its plug-in is unloaded)")

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
    wrapper.__doc__      = original.__doc__
    wrapper.__wrapped__  = original  # functools convention; help/inspect see original
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