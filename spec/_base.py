"""
Base class for the rig attribute-specification DSL.

An :class:`_AttrSpec` describes "an attribute to add" -- its long name,
type, default, range, and other ``cmds.addAttr`` kwargs. The spec is
applied to a node by injection::

    node << Float("weight", min=0, max=1) << 0.5 << lock

Direction:
    * ``Node << spec``  ->  ``cmds.addAttr(node, ...)`` then return new ``Plug``
    * ``Plug    << spec``  ->  ``cmds.setAttr(plug, ...)`` (modifier-only specs)

The spec returns the new (or modified) ``Plug`` so chaining works:
``... << 5 << lock`` first sets the value, then locks.

Ported from Eric Vignola's BSD-3 ``rig.attributes._Attribute``, slimmed to
use ``rig.maya``'s ``Attribute.data_type`` instead of regex-parsing
``getAddAttrCmd()`` output.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from maya import cmds


LOGGER = logging.getLogger(__name__)


class _AttrSpec:
    """Base class for all attribute-specification objects.

    Subclasses set ``self.kargs['attributeType']`` or ``['dataType']`` and
    optionally provide ``self.compound`` (list of child suffixes like
    ``['X', 'Y', 'Z']``) and ``self.compoundType`` (override child type).
    """

    def __init__(self, name: Optional[str] = None, **kargs: Any) -> None:
        self.kargs: Dict[str, Any] = dict(kargs)

        if name is not None:
            # Normalise short->long flag names; default keyable=True, longName=name.
            self.kargs["keyable"] = self._pop_alias(
                self.kargs, ("keyable", "k"), default=True
            )
            self.kargs["longName"] = self._pop_alias(
                self.kargs, ("longName", "ln"), default=name
            )
            # Custom (non-cmds) options handled by .apply():
            self.size: Optional[int] = self._pop_alias(
                self.kargs, ("size",), default=None
            )
            self.overwrite: bool = self._pop_alias(
                self.kargs, ("overwrite",), default=True
            )
        else:
            # Modifier-only spec (lock/hide/etc.) -- kargs are pure setAttr flags.
            self.size      = None
            self.overwrite = True

        self.compound:     Optional[List[str]] = None  # e.g. ['X', 'Y', 'Z']
        self.compoundType: Optional[str] = None        # override children's at=
        self.notes:        Optional[str] = None        # set by Note

    @staticmethod
    def _pop_alias(d: Dict[str, Any], keys: tuple, default: Any) -> Any:
        """Pop the first matching key in ``keys`` from ``d`` and return its value,
        or ``default`` if none match. Removes all matched keys."""
        result = default
        found  = False
        for k in keys:
            if k in d:
                if not found:
                    result = d.pop(k)
                    found  = True
                else:
                    d.pop(k)
        return result

    # -- application -- #

    def apply(self, target: Any) -> Any:
        """Apply this spec to ``target`` (a :class:`Node` or :class:`Plug`).

        Returns the resulting :class:`Plug` (the new attribute, or the
        modified one for modifier-only specs).
        """
        # Modifier-only spec (no longName) -- ``cmds.setAttr`` edit on whatever
        # the target currently points at.
        long_name = self.kargs.get("longName")
        if long_name is None:
            return self._apply_modifier(target)

        # Need a clean node string regardless of whether target is Node or Plug.
        from rig._internal.node import Node
        from rig._internal.plug import Plug

        if isinstance(target, Plug):
            node_string = str(target.node)
            wrap_node   = Node(node_string)
        elif isinstance(target, Node):
            node_string = str(target)
            wrap_node   = target
        else:
            node_string = str(target)
            wrap_node   = Node(node_string)

        return self._apply_addattr(node_string, wrap_node)

    # -- internals -- #

    def _apply_modifier(self, target: Any) -> Any:
        """Modifier-only spec: apply via :meth:`Attribute.set` (canonical API).

        ``target`` is a :class:`Plug` (which subclasses :class:`Attribute`),
        so ``target.set(**kargs)`` routes through the rename-safe API.
        """
        from rig._internal.types import _get_compound, _is_compound

        kargs = dict(self.kargs)

        # If only keyable/channelBox flags AND target is compound, fan out.
        if (
            len(kargs) == 2
            and "keyable" in kargs
            and "channelBox" in kargs
            and _is_compound(target)
        ):
            for child in _get_compound(target):
                try:
                    child.set(**kargs)
                except RuntimeError as e:
                    LOGGER.debug("set modifier on %s failed: %s", child, e)

        try:
            target.set(**kargs)
        except RuntimeError as e:
            LOGGER.debug("set modifier on %s failed: %s", target, e)
        return target

    def _apply_addattr(self, node_string: str, wrap_node: Any) -> Any:
        """Add the attribute to ``node_string`` and return its :class:`Plug`."""
        kargs         = dict(self.kargs)
        long_name     = kargs["longName"]
        multi         = self._pop_alias(kargs, ("multi", "m"), default=False)
        default_value = self._pop_alias(kargs, ("defaultValue", "dv"), default=0)

        # If attribute already exists, optionally delete-then-recreate.
        # Use DGNode.has_attr (canonical API) instead of cmds.attributeQuery.
        if wrap_node.has_attr(long_name):
            if self.overwrite:
                # Use DGNode-level API: find_attr + Attribute.set + delete_attr.
                try:
                    existing_attr = wrap_node.find_attr(long_name, quiet=True)
                    if existing_attr is not None:
                        existing_attr.set(lock=False)
                except (RuntimeError, AttributeError):
                    pass
                try:
                    wrap_node.delete_attr(long_name)
                except RuntimeError as e:
                    LOGGER.debug(
                        "Failed to delete %s.%s: %s", node_string, long_name, e
                    )
            else:
                # Don't overwrite -- return existing.
                return getattr(wrap_node, long_name)

        # ---- Compound (Vector / Quat / Color / Euler) ---- #
        if self.compound:
            self._add_compound(node_string, kargs, multi, default_value, wrap_node)
        else:
            # Single attribute. Restore default_value for non-compound.
            if "defaultValue" not in self.kargs and "dv" not in self.kargs:
                # No DV explicitly given -- drop entirely (cmds.addAttr can't
                # take dv=0 for some types like message).
                kargs.pop("defaultValue", None)
                kargs.pop("dv", None)
            else:
                kargs["defaultValue"] = default_value
            if multi:
                kargs["multi"] = True
            # Use DGNode.add_attr (rename-safe wrapper that uses self.name).
            kargs.pop("longName", None)
            wrap_node.add_attr(long_name, **kargs)

        # ---- Multi pre-sizing ---- #
        if multi and self.size is not None:
            self._presize_multi(node_string, long_name, default_value, wrap_node)

        # ---- Note string set ---- #
        new_plug = getattr(wrap_node, long_name)
        if self.notes is not None:
            new_plug << self.notes

        return new_plug

    def _add_compound(
        self,
        node_string:   str,
        kargs:         Dict[str, Any],
        multi:         Any,
        default_value: Any,
        wrap_node:     Any,
    ) -> None:
        """Build a compound (parent + N children) via :meth:`DGNode.add_attr`."""
        count = len(self.compound)
        kargs = dict(kargs)

        if "at" in kargs or "attributeType" in kargs:
            at_value    = self._pop_alias(kargs, ("attributeType", "at"), default="double")
            kargs["at"] = f"{at_value}{count}"
        elif "dt" in kargs or "dataType" in kargs:
            dt_value    = self._pop_alias(kargs, ("dataType", "dt"), default="double")
            kargs["dt"] = f"{dt_value}{count}"

        kargs.pop("defaultValue", None)
        kargs.pop("dv", None)
        if multi:
            kargs["multi"] = True
        # DGNode.add_attr is rename-safe and uses self.name internally.
        long_name = kargs.pop("longName")
        wrap_node.add_attr(long_name, **kargs)

        # Children
        for i in range(count):
            child_kargs           = dict(self.kargs)
            child_kargs["parent"] = child_kargs["longName"]
            child_kargs.pop("p",     None)
            child_kargs.pop("multi", None)
            child_kargs.pop("m",     None)
            child_kargs.pop("size",  None)

            if self.compoundType:
                child_kargs["attributeType"] = self.compoundType

            # Per-child default value (if user passed a sequence).
            base_dv = self.kargs.get("dv", self.kargs.get("defaultValue"))
            if base_dv is not None:
                if isinstance(base_dv, (list, tuple)) and len(base_dv) > i:
                    child_kargs["defaultValue"] = base_dv[i]
                else:
                    child_kargs.pop("defaultValue", None)
                    child_kargs.pop("dv", None)

            child_long_name = f"{self.kargs['longName']}{self.compound[i]}"
            child_kargs.pop("longName", None)
            wrap_node.add_attr(child_long_name, **child_kargs)

    def _presize_multi(
        self,
        node_string:   str,
        long_name:     str,
        default_value: Any,
        wrap_node:     Any,
    ) -> None:
        """Allocate ``self.size`` indices on a multi attribute.

        Uses :meth:`Attribute.set` (canonical API) on each indexed element
        plug rather than building element strings for ``cmds.setAttr``.
        """
        for i in range(self.size or 0):
            try:
                element_plug = wrap_node.find_attr(f"{long_name}[{i}]", quiet=True)
            except (RuntimeError, AttributeError):
                element_plug = None
            if element_plug is None:
                continue
            try:
                element_plug.set(default_value)
            except (RuntimeError, TypeError):
                try:
                    element_plug.set("", type="string")
                except RuntimeError:
                    try:
                        element_plug.set(
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            type="matrix",
                        )
                    except RuntimeError as e:
                        LOGGER.debug(
                            "presize failed for %s.%s[%d]: %s",
                            node_string,
                            long_name,
                            i,
                            e,
                        )


# ---------------------------------------------------------------------- #
#  Attribute cloning (used by `>>` clone-attr and Container.publish)
# ---------------------------------------------------------------------- #


def _clone_attribute(
    src_plug:  Any,
    dst_node:  Any,
    attr_name: Optional[str]  = None,
    multi:     Optional[bool] = None,
    connect:   bool           = False,
) -> Any:
    """Inspect ``src_plug``'s data type and add an equivalent attribute to
    ``dst_node`` named ``attr_name`` (defaults to the source's short name).

    Optionally connect ``src_plug`` into the new attribute.

    Returns the new :class:`Plug`. Slimmed from Eric's 160-line version by
    leaning on ``rig.maya``'s ``Attribute.data_type`` / ``attribute_type`` /
    ``is_multi`` / ``num_children`` properties.
    """
    from rig._internal.node import Node
    from rig._internal.plug import Plug
    from rig._internal.types import _is_attribute, _is_compound

    # Resolve dst_node to a Node wrapper.
    if not isinstance(dst_node, Node):
        dst_node = Node(dst_node)

    # Default attr name from source.
    if attr_name is None:
        attr_name = (
            src_plug.alias
            if isinstance(src_plug, Plug)
            else str(src_plug).rsplit(".", 1)[-1]
        )

    # Auto-detect multi from src.
    if multi is None and isinstance(src_plug, Plug):
        try:
            multi = src_plug.is_multi
        except Exception:
            multi = False

    spec     = _spec_from_attribute(src_plug, attr_name, bool(multi))
    new_plug = dst_node << spec

    # v4.F.b: if the source is a populated multi, mirror its existing
    # indices onto the new attribute so downstream slice / iteration /
    # batch-write idioms work without manual priming.
    #
    # Without this, ``node.multi >> dst`` would clone the multi root but
    # leave ``dst.multi[:]`` empty, forcing the user to ``dst.multi[N-1]
    # << 0`` to materialize indices first.
    #
    # As a side effect, the source's current values are copied into the
    # new indices (since materializing a multi index requires a setAttr).
    # Treat as a clone-with-snapshot semantic; downstream code can
    # disconnect / overwrite as needed.
    if multi and isinstance(src_plug, Plug):
        try:
            src_indices = src_plug.get_logical_indices() or []
        except Exception:
            src_indices = []
        for idx in src_indices:
            try:
                new_plug[idx] << src_plug[idx].get()
            except Exception:
                pass

    if connect:
        new_plug << src_plug

    return new_plug


def _safe_type_probe(src_plug: Any, prop: str) -> Optional[str]:
    """Read type property ``prop`` off ``src_plug``, ``None`` if unresolvable.

    Polymorphic plugs (``kGenericAttribute`` -- ``choice.output`` and
    friends) carry no type until an input is wired, and report that by
    raising rather than by returning a sentinel.
    """
    try:
        return getattr(src_plug, prop)
    except Exception:
        LOGGER.debug("could not resolve %s of %s", prop, src_plug, exc_info=True)
        return None


def _spec_from_attribute(src_plug: Any, attr_name: str, multi: bool) -> "_AttrSpec":
    """Return the matching :class:`_AttrSpec` subclass for ``src_plug``."""
    from rig._internal.plug import Plug

    # Lazy import to avoid circular dep on the spec leaf modules.
    from rig.spec.compound import Color, Euler, Quat, Vector
    from rig.spec.enum_attr import Enum
    from rig.spec.numeric import Angle, Bool, Float, Int, Time
    from rig.spec.typed import (
        Matrix,
        Mesh,
        Message,
        NurbsCurve,
        NurbsSurface,
        String,
    )

    if not isinstance(src_plug, Plug):
        # Bare plug-like -- wrap.
        src_plug = Plug(str(src_plug))

    data_type = _safe_type_probe(src_plug, "data_type")
    attr_type = _safe_type_probe(src_plug, "attribute_type")

    # Compound first (Vector / Color / Euler / Quat).
    try:
        nchildren = src_plug.num_children
    except Exception:
        nchildren = 0

    if nchildren == 3:
        # Distinguish doubleAngle (Euler) from double (Vector).
        try:
            child_type = src_plug.child(0).attribute_type
        except Exception:
            child_type = None
        if child_type == "doubleAngle":
            return Euler(attr_name, multi=multi)
        return Vector(attr_name, multi=multi)
    if nchildren == 4:
        return Quat(attr_name, multi=multi)

    # Enum: needs to copy the enum names. Use Attribute.enums (canonical
    # API) instead of cmds.attributeQuery(... listEnum=True).
    if attr_type == "enum":
        try:
            enum_names = src_plug.enums or ["off", "on"]
        except (RuntimeError, IndexError, AttributeError):
            enum_names = ["off", "on"]
        return Enum(attr_name, en=enum_names, multi=multi)

    # Single-attribute mapping.
    mapping = {
        "long":         Int,
        "int":          Int,
        "matrix":       Matrix,
        "doubleAngle":  Angle,
        "doubleLinear": Float,
        "double":       Float,
        "float":        Float,
        "double3":      Vector,
        "bool":         Bool,
        "string":       String,
        "time":         Time,
        "mesh":         Mesh,
        "nurbsCurve":   NurbsCurve,
        "nurbsSurface": NurbsSurface,
        "message":      Message,
    }
    spec_cls = mapping.get(data_type) or mapping.get(attr_type) or Float
    return spec_cls(attr_name, multi=multi)