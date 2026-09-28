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

An enum attribute takes a field name as well as its int, by long or short
name, as ``<<`` reads one (the exact name, else the one field that matches
with case and outer spaces ignored, else once case, spaces, ``_`` and a ``-``
between letters are ignored; a leading ``-`` is a sign)::

    nodes.transform(rotateOrder="xzy")                 # 3, as ro="xzy"
    nodes.multiplyDivide(operation="power")            # 3 (Maya's "Power")
    nodes.decomposeMatrix(inputRotateOrder="zxy")      # 2

The names are read before the node is created, so a wrong one raises
``TypeError`` (naming the fields) and creates nothing.

For occasional direct node creation (without going through this module),
``Node.create()`` (a registered type's typed create, else the scope's
``createNode``) and the explicit converter :meth:`Node.wrap` remain
available.
"""

from __future__ import annotations

import keyword
from typing import Any, Callable

from maya import cmds as _mc
from maya.api import OpenMaya as _om


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


def _enum_kwargs(node_type: str, kwargs: dict) -> dict:
    """`kwargs` (a factory's attribute kwargs) with each plain str given to an
    enum attribute of `node_type` replaced by the value of that field, read
    before the node is created: a wrong name raises TypeError naming the fields
    (see :func:`rig.nodetypes._base._enum_value`) and nothing is created. A
    list or tuple for a multi or a compound (``displayLevel=["Show", "Hide"]``)
    has every name it holds read the same way, and is kept as it is.

    The attribute is the type's static one, found by its long or short name
    through ``OpenMaya.MNodeClass``. A name it cannot describe (a dynamic
    attribute, a type it does not know) keeps its str for the ``<<`` after
    creation, which reads an enum field name the same way.
    """
    from rig.nodetypes._base import _check_enum_names, _enum_value, _holds_text, _is_text

    node_class = None
    resolved   = dict(kwargs)
    for attr_name, value in kwargs.items():
        if not _holds_text(value):
            continue
        if node_class is None:
            node_class = _om.MNodeClass(node_type)
        try:
            attribute = node_class.attribute(attr_name)
        except (RuntimeError, TypeError, ValueError):
            continue
        if attribute.isNull():  # no such static attribute: MNodeClass does not raise
            continue
        where = f"{node_type}.{attr_name}"
        if _is_text(value) and attribute.hasFn(_om.MFn.kEnumAttribute):
            resolved[attr_name] = _enum_value(attribute, value, where)
        else:
            _check_enum_names(attribute, value, where)
    return resolved


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

        # Enum field names (``rotateOrder="xzy"``) are read before the node
        # exists, so a wrong one raises with the scene untouched.
        if kwargs:
            kwargs = _enum_kwargs(node_type, kwargs)

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
        f"and applied via the DSL ``<<`` operator. An enum attribute takes a\n"
        f"field name (``rotateOrder='xzy'``), read before the node is created:\n"
        f"a wrong name raises TypeError and creates nothing.\n\n"
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