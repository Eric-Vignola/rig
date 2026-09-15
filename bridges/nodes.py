"""Dynamic factories for every Maya node type, registered as
``rig.bridges.nodes.<nodetype>``.

Usage::

    from rig.bridges import nodes

    # Create + initial attribute setup in one call
    n = nodes.plusMinusAverage(operation=2)
    t = nodes.transform(name="cube1", translate=[1, 2, 3])
    cm = nodes.composeMatrix(inputScale=[5, 0.1, 5])

    # createNode kwargs honored
    child = nodes.transform(parent=root, name="child")

    # Opt out of active container scope
    standalone = nodes.transform(name="floater", container=False)

    # Python keyword collisions: append a trailing underscore (PEP 8)
    a = nodes.and_(name="myAnd")    # creates an "and" node
    o = nodes.or_(name="myOr")
    n = nodes.not_(name="myNot")

Inspired by Eric Vignola's third_party.rig.nodes (same UX), implemented
via PEP 562 module-level ``__getattr__`` instead of ``exec``'d
string-formatted function bodies. Wrappers are built lazily on first
attribute access -- zero startup cost, real Python functions, working
IDE intellisense, real ``help()``, real tracebacks, smaller memory
footprint (only used wrappers cached). Studio plugin nodetypes are
picked up automatically.

Attribute kwargs are applied via the DSL ``<<`` operator (see
:meth:`rig.Plug.__lshift__`), so compound vectors, matrices,
spec objects, and type-shorthand all work naturally::

    nodes.transform(translate=[1, 2, 3])               # compound vector
    nodes.transform(matrix=np.eye(4))                  # routes through _decompose
    nodes.plusMinusAverage(input1D=[1, 2, 3])          # multi-attr fan-out

For occasional direct ``cmds.createNode`` use (without going through
this module), ``Node.create()`` and the explicit converter
:meth:`Node.wrap` remain available.
"""

from __future__ import annotations

import keyword
from typing import Any, Callable

from maya import cmds as _mc


# Per-name wrapper cache. Built lazily by ``__getattr__``.
_WRAPPER_CACHE: dict = {}

# Cached set of Maya node types. Loaded lazily on first lookup; refresh
# via ``_refresh_node_types()`` if a plugin loads after this module.
_NODE_TYPES: frozenset = frozenset()
_NODE_TYPES_LOADED = False


# Recognised createNode-time kwargs (vs. attribute kwargs). Values are
# the canonical long form passed to ``container.createNode``.
_CREATE_KWARGS = {
    "name":       "name",
    "n":          "name",
    "parent":     "parent",
    "p":          "parent",
    "shared":     "shared",
    "s":          "shared",
    "skipSelect": "skipSelect",
    "ss":         "skipSelect",
}


def _refresh_node_types() -> None:
    """Re-query Maya for the current set of registered node types.

    Useful after a plugin loads new nodetypes mid-session. Clears the
    per-name wrapper cache so subsequent ``nodes.<name>`` lookups pick
    up the refreshed type set.
    """
    global _NODE_TYPES, _NODE_TYPES_LOADED
    try:
        _NODE_TYPES = frozenset(_mc.ls(nt=True))
    except Exception:
        _NODE_TYPES = frozenset()
    _NODE_TYPES_LOADED = True
    _WRAPPER_CACHE.clear()


def _ensure_node_types_loaded() -> None:
    if not _NODE_TYPES_LOADED:
        _refresh_node_types()


def _resolve_alias(name: str) -> str:
    """Map ``foo_`` -> ``foo`` if ``foo`` is a Python keyword.

    Lets us expose nodetypes whose names collide with Python reserved
    words (e.g. ``and``, ``or``, ``not``, ``if``, ``else``, ``import``,
    ``class``) via the PEP 8 trailing-underscore convention.

    Otherwise returns ``name`` unchanged.
    """
    if name.endswith("_") and not name.endswith("__"):
        bare = name[:-1]
        if keyword.iskeyword(bare):
            return bare
    return name


def _make_factory(node_type: str) -> Callable:
    """Build a factory function for ``cmds.createNode(node_type, ...)``."""

    def factory(**kwargs: Any):
        # Lazy imports to avoid circular dependency at module load.
        from rig._internal.container import container

        # Split kwargs into createNode-time vs attribute-init.
        create_kwargs = {}
        for short, canonical in _CREATE_KWARGS.items():
            if short in kwargs:
                create_kwargs[canonical] = kwargs.pop(short)
        add_to_container = kwargs.pop("container", True)

        # Create the node (auto-joins active container scope unless
        # opted out via ``container=False``).
        node = container.createNode(
            node_type, container=add_to_container, **create_kwargs
        )

        # Remaining kwargs -> DSL ``<<`` inject. Handles scalars, compound
        # vectors, matrices (via _decompose routing), type-shorthand,
        # spec objects (Float / Vector / etc.), and multi-attr fan-out.
        for attr_name, value in kwargs.items():
            getattr(node, attr_name) << value

        return node

    factory.__name__     = node_type
    factory.__qualname__ = f"rig.bridges.nodes.{node_type}"
    factory.__doc__ = (
        f"Create a Maya ``{node_type}`` node and apply attribute kwargs.\n\n"
        f"Recognised createNode kwargs (consumed before attribute init):\n"
        f"  - ``name`` / ``n``: node name\n"
        f"  - ``parent`` / ``p``: parent transform\n"
        f"  - ``shared`` / ``s``: see ``cmds.createNode(shared=...)``\n"
        f"  - ``skipSelect`` / ``ss``: see ``cmds.createNode(skipSelect=...)``\n"
        f"  - ``container``: opt out of active container scope (default True)\n\n"
        f"All remaining kwargs are interpreted as initial attribute values\n"
        f"and applied via the DSL ``<<`` operator.\n\n"
        f"Returns the new ``Node``."
    )
    return factory


def __getattr__(name: str) -> Callable:
    """PEP 562: lazily build a factory for ``nodes.<name>`` on first access.

    Resolves Python-keyword aliases (``and_`` -> ``and`` nodetype) and
    raises ``AttributeError`` if the underlying nodetype isn't registered
    with Maya.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    cached = _WRAPPER_CACHE.get(name)
    if cached is not None:
        return cached

    actual = _resolve_alias(name)

    _ensure_node_types_loaded()
    if actual not in _NODE_TYPES:
        raise AttributeError(
            f"'rig.bridges.nodes' has no node type {name!r} "
            f"(no such Maya nodetype registered). "
            f"If you just loaded a plugin, call "
            f"``rig.bridges.nodes._refresh_node_types()``."
        )

    factory              = _make_factory(actual)
    _WRAPPER_CACHE[name] = factory
    return factory


def __dir__() -> list:
    """Tab-completion: list every registered Maya nodetype.

    For nodetypes whose names collide with Python keywords (``and``,
    ``or``, ``not``, etc.), returns the safe trailing-underscore alias
    form (``and_``, ``or_``, ``not_``).
    """
    _ensure_node_types_loaded()
    names = []
    for t in _NODE_TYPES:
        if keyword.iskeyword(t):
            names.append(t + "_")
        else:
            names.append(t)
    return sorted(names)