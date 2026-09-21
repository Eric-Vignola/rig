"""Attribute modifiers -- singletons used in ``<<`` chains.

Examples::

    plug << lock                 # cmds.setAttr(plug, lock=True)
    plug << unlock               # cmds.setAttr(plug, lock=False)
    plug << hide                 # keyable=False, channelBox=False
    plug << unhide               # keyable=True, channelBox=False
    plug << skip                 # no-op -- leave this plug exactly as it is
    node << Note("Helper text")  # adds a hidden 'notes' string attr

    # v4.S: attribute removal -- two equivalent forms
    plug << destroy              # delete this specific plug
    node << destroy("foo")       # delete node.foo
    node << destroy("foo", "bar") # delete multiple
    node << destroy("foo", strict=True)  # raise if connections exist
    node << destroy("foo", silent=True)  # no-op if attr missing

These modifiers can appear anywhere in a ``<<`` chain because injection
returns the LHS, e.g. ``node << Float("blend") << 0.5 << lock``.
"""

from __future__ import annotations

from typing import Any, Optional

from rig.spec._base import _AttrSpec

# Singleton modifier instances. Lowercase per PEP 8 (these are not classes).
lock   = _AttrSpec(lock=True)
unlock = _AttrSpec(lock=False)
hide   = _AttrSpec(keyable=False, channelBox=False)
unhide = _AttrSpec(keyable=True, channelBox=False)


class _SkipMarker(_AttrSpec):
    """Singleton sentinel meaning "leave this target exactly as it is".

    Earns its keep inside a per-channel fan-out, where ``None`` means
    *disconnect*::

        ctrl.t << [skip, 4.0, skip]   # write Y; X and Z keep their drivers
        ctrl.t << [None, 4.0, None]   # write Y; X and Z are DISCONNECTED
    """

    def apply(self, target: Any) -> Any:
        return target

    def __repr__(self) -> str:
        return "<skip>"


skip = _SkipMarker()


# --------------------------------------------------------------------- #
#  destroy -- attribute removal DSL (v4.S)
# --------------------------------------------------------------------- #


class _DestroySpec(_AttrSpec):
    """Spec returned by ``destroy(...)`` -- deletes named attrs from a Node.

    Created via the ``destroy`` factory:

        node << destroy("foo")
        node << destroy("foo", "bar", "baz")
        node << destroy("foo", strict=True)   # raise if connected
        node << destroy("foo", silent=True)   # no-op if missing
        node << destroy("foo", verbose=True)  # log each broken connection

    Subclass of :class:`_AttrSpec` so the existing ``<<`` dispatch
    (``Node.__lshift__`` / ``Plug.__lshift__``) routes us through
    ``_is_attribute_spec`` -> ``self.apply(target)`` automatically.
    """

    def __init__(
        self,
        attr_names: tuple,
        strict:     bool  = False,
        silent:     bool  = False,
        verbose:    bool  = False,
    ) -> None:
        # Bypass _AttrSpec's longName-based init -- we don't add an attr.
        self.kargs:        dict = {}
        self.size:         Optional[int] = None
        self.overwrite:    bool = True
        self.compound:     Optional[list] = None
        self.compoundType: Optional[str] = None
        self.notes:        Optional[str] = None
        # Destroy-specific state.
        self.attr_names = tuple(attr_names)
        self.strict     = strict
        self.silent     = silent
        self.verbose    = verbose

    def apply(self, target: Any) -> Any:
        """Delete each named attr from ``target`` (a Node or Plug)."""
        # Lazy import to avoid circular dep with _internal.plug.
        from rig._internal.node import Node
        from rig._internal.plug import _do_destroy, Plug

        if isinstance(target, Plug):
            raise TypeError(
                f"destroy(...) is for Nodes, not Plugs. To destroy a "
                f"specific plug, use the bare-sentinel form: "
                f"'plug << destroy' (got plug={target!r})"
            )
        node_str = str(target) if not isinstance(target, Node) else str(target)
        for attr_name in self.attr_names:
            full = f"{node_str}.{attr_name}"
            _do_destroy(
                full, strict=self.strict, silent=self.silent, verbose=self.verbose
            )
        return target


class _DestroyMarker(_AttrSpec):
    """Singleton sentinel + callable factory for the ``destroy`` symbol.

    Two usage forms:

        plug << destroy              # SENTINEL -- delete this plug
        node << destroy("foo", ...)  # CALLABLE -- returns a _DestroySpec

    Subclass of :class:`_AttrSpec` so ``_is_attribute_spec(destroy)``
    returns True and the existing ``<<`` dispatch routes through
    ``destroy.apply(target)``.
    """

    def __init__(self) -> None:
        # Bypass _AttrSpec's longName-based init -- we're a sentinel.
        self.kargs:        dict = {}
        self.size:         Optional[int] = None
        self.overwrite:    bool = True
        self.compound:     Optional[list] = None
        self.compoundType: Optional[str] = None
        self.notes:        Optional[str] = None

    def __call__(
        self,
        *attr_names: str,
        strict:      bool = False,
        silent:      bool = False,
        verbose:     bool = False,
    ) -> _DestroySpec:
        if not attr_names:
            raise ValueError(
                "destroy() requires at least one attr name "
                "(e.g. destroy('foo') or destroy('foo', 'bar'))"
            )
        if strict and silent:
            raise ValueError(
                "destroy(...): strict=True and silent=True are incompatible "
                "(strict requires the attr to exist)"
            )
        return _DestroySpec(attr_names, strict=strict, silent=silent, verbose=verbose)

    def apply(self, target: Any) -> Any:
        """Bare-sentinel application: ``plug << destroy`` deletes the plug.

        Raises :class:`TypeError` when applied to a Node (no attr name
        was supplied -- point the user to the callable form).
        """
        # Lazy import to avoid circular dep with _internal.plug.
        from rig._internal.node import Node
        from rig._internal.plug import _do_destroy, Plug

        if isinstance(target, Plug):
            _do_destroy(str(target))
            return target
        if isinstance(target, Node):
            raise TypeError(
                f"'node << destroy' is ambiguous (no attr name supplied). "
                f"Use 'node << destroy(\"attr_name\")' to delete a named "
                f"attr, or 'node.attr_name << destroy' to target a plug."
            )
        # Anything else -- treat as a plug-string.
        _do_destroy(str(target))
        return target

    def __repr__(self) -> str:
        return "<destroy>"


# Module-level singleton -- this is the public ``destroy`` symbol.
destroy = _DestroyMarker()


class Note(_AttrSpec):
    """Adds a hidden, write-once ``notes`` string attribute.

    If ``text`` is given, the note is set immediately.
    """

    def __init__(self, text: Optional[str] = None) -> None:
        super().__init__("notes")
        self.kargs["dataType"] = "string"
        self.kargs.pop("dt",            None)
        self.kargs.pop("attributeType", None)
        self.kargs.pop("at",            None)
        self.kargs["keyable"]  = False
        self.kargs["writable"] = False
        self.kargs["hidden"]   = True
        self.kargs.pop("k", None)
        self.kargs.pop("w", None)
        self.kargs.pop("h", None)
        if text is not None:
            self.notes = text