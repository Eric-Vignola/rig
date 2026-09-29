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
``... << 5 << lock`` first sets the value, then locks. A new attribute's
``Plug`` is owned by the node object the spec went to
(``(node << Float("x")).node is node``; ``plug << Float("x")`` adds ``x`` to
``plug.node``).

Re-declaring an attribute the node has already keeps it, with its value and
its connections, and applies the settings the spec was given (the keywords
its caller passed: ``dv``, ``min`` / ``max``, the soft range, ``keyable``,
``hidden``, ``niceName``, an ``Enum``'s ``en`` ...; never rig's own
``keyable=True``): ``node << Float("w", dv=5); node.w << drv.tx; node <<
Float("w", max=10)`` keeps ``w`` driven and gives it a max. Another kind of
attribute raises TypeError; ``overwrite=True`` deletes it and adds it again
(see ``_AttrSpec._redeclare``).

Ported from Eric Vignola's BSD-3 ``rig.attributes._Attribute``, slimmed to
use ``rig.nodetypes.Attribute.data_type`` instead of regex-parsing
``getAddAttrCmd()`` output.
"""

from __future__ import annotations

import difflib
import logging
import numbers
from typing import Any, Dict, List, Optional

from maya import cmds
from maya.api import OpenMaya
# rig.nodetypes imports neither rig.spec nor rig._internal.plug
from rig._internal.undo import _undo_chunk  # a leaf module: maya.cmds only
from rig.nodetypes._base import (
    Attribute,
    Node,
    _attr_handle,
    _attr_state,
    _enum_fields,
    _handle_valid,
    _parse_enum_names,
)


LOGGER = logging.getLogger(__name__)

# addAttr's short flags -> long names (``cmds.help("addAttr")``, Maya 2025): the
# settings of a re-declaration are keyed by long name
_LONG_FLAGS = {
    "at": "attributeType", "bt": "binaryTag", "ci": "cachedInternally", "ct": "category",
    "dcb": "disconnectBehaviour", "dt": "dataType", "dv": "defaultValue", "en": "enumName",
    "h": "hidden", "hnv": "hasMinValue", "hsn": "hasSoftMinValue", "hsx": "hasSoftMaxValue",
    "hxv": "hasMaxValue", "im": "indexMatters", "is": "internalSet", "k": "keyable",
    "ln": "longName", "m": "multi", "max": "maxValue", "min": "minValue",
    "nc": "numberOfChildren", "nn": "niceName", "p": "parent", "pxy": "proxy",
    "r": "readable", "s": "storable", "smn": "softMinValue", "smx": "softMaxValue",
    "sn": "shortName", "uac": "usedAsColor", "uaf": "usedAsFilename", "uap": "usedAsProxy",
    "w": "writable",
}
# every keyword a named spec takes: addAttr's flags when it adds an attribute
# (long and short, ``cmds.help("addAttr")`` without edit / query / exists), and
# the spec's own ``size`` / ``overwrite``; any other keyword (a typo, define's
# ``update=``) raises TypeError when the spec is made, before any edit
_SPEC_FLAGS = (
    frozenset(_LONG_FLAGS) | frozenset(_LONG_FLAGS.values())
    | frozenset(("enforcingUniqueName", "eun", "fromPlugin", "fp", "worldSpace", "ws", "size", "overwrite"))
)
# the settings that are numbers (a default is a number too, or one per child)
_RANGE_FLAGS = frozenset(("minValue", "maxValue", "softMinValue", "softMaxValue"))
# what makes the attribute (its name and kind), never a setting
_STRUCTURE = frozenset((
    "longName", "attributeType", "dataType", "multi", "size", "overwrite", "parent",
    "numberOfChildren",
))
# the settings ``addAttr -edit`` changes (runs\r4b\NC8\probe_addattr*.txt); the
# default goes through the API in rig's undoable command where it can (Maya's
# undo of an ``addAttr -edit -defaultValue`` sets it to 0), keyable through
# ``setAttr -keyable``, hidden through the API; any other setting cannot change
# (equal to the attribute's, or a TypeError): hasMinValue / hasMaxValue among
# them, since an edit toggles them whatever the value and its undo / redo are
# not inverses (runs\r4b\FIX; a min / max given sets them)
_EDITABLE   = frozenset((
    "defaultValue", "minValue", "maxValue", "softMinValue", "softMaxValue", "niceName",
    "enumName", "category",
))
# a compound's settings that go to its children only (the others go to the
# parent and to each child, as on creation)
_PER_CHILD  = frozenset((
    "defaultValue", "minValue", "maxValue", "softMinValue", "softMaxValue", "hasMinValue",
    "hasMaxValue", "hasSoftMinValue", "hasSoftMaxValue",
))


def _plug_of(node_string: str, long_name: str, node: Any = None, added: Any = None) -> Any:
    """The :class:`Plug` of the attribute ``long_name`` of ``node``, the node
    object the spec was applied to, **owned by** it:
    ``(node << Float("x")).node is node``, as ``node.x.node`` is. Its MPlug is
    found by name on the node's fn set (``findPlug``), not through
    ``Node.__getattr__``, so a Python member's name (``rename``) and a leading
    underscore (``__parked__``) resolve too. A compound spec gives its parent
    plug, a multi spec its array root.

    Its str buffer, which maya.cmds reads, is ``node_string.long_name``, where
    ``node_string`` is the node's name (``str(node)``): the shortest unique path
    of a DAG node, through the path the node object holds, so a node whose short
    name is not unique (``|A|X``) is named ``A|X.x``, and an instanced one
    through its own path (``Node("|T2|S") << Float("k")`` is ``T2|S.k``, whose
    ``full_name`` is ``T2|S.k`` too).

    ``added``, the attr ``add_attr`` returned for it, hands it the handle of its
    attribute (``_attr1``, which a delete of a dynamic attr frees once it leaves
    the undo queue, see ``_ensure_owner_alive``) when it is alive and names the
    same attribute; otherwise that handle is looked up once (``_attr_handle``).

    A node that is not a live node object, or a name ``findPlug`` does not
    resolve (an alias, ``smile`` for ``weight[0]``, or a component name,
    ``vtx[1]``), takes the string path (``_named_plug``): a plug with no owner,
    checked through the node's API 1.0 handle, whose name resolves as
    ``Plug(name)`` resolves it."""
    from rig._internal.plug import Plug, _named_plug  # deferred: plug.py imports rig.spec

    if node is not None:
        d  = node.__dict__
        fn = d.get("_fn_set")
        if fn is not None and _handle_valid(d):
            try:
                mplug = fn.findPlug(long_name, False)
            except RuntimeError:
                mplug = None
            if mplug is not None:
                known = None if added is None else added.__dict__
                attr1 = None if known is None else known.get("_attr1")
                if (
                    attr1 is None
                    or not attr1.isAlive()
                    or known["_mplug"].attribute() != mplug.attribute()
                ):
                    # NW6: API 1.0 handle read (the node's API 1.0 fn set)
                    attr1 = _attr_handle(mplug, fn1=d.get("_fn_set1"), name=long_name)
                plug  = str.__new__(Plug, f"{node_string}.{long_name}")
                state = _attr_state(mplug, None, attr1)
                state["_node"] = node
                plug.__dict__.update(state)
                return plug
    return _named_plug(f"{node_string}.{long_name}", node, long_name, added)


class _AttrSpec:
    """Base class for all attribute-specification objects.

    Subclasses set ``self.kargs['attributeType']`` or ``['dataType']`` and
    optionally provide ``self.compound`` (list of child suffixes like
    ``['X', 'Y', 'Z']``) and ``self.compoundType`` (override child type).

    ``overwrite=True`` replaces an attribute the node has already (deleted,
    value and connections with it, and added again); by default, and with
    ``overwrite=False`` written out, the attribute is re-declared: kept, the
    settings passed applied (``_redeclare``).
    """

    def __init__(self, name: Optional[str] = None, **kargs: Any) -> None:
        self.kargs: Dict[str, Any] = dict(kargs)
        # the keywords the caller passed: on an existing attribute, the settings
        # applied (see `_redeclare`); rig's own defaults (keyable=True) are not
        self._given: tuple = tuple(kargs)

        if name is not None:
            unknown = sorted(set(kargs) - _SPEC_FLAGS)
            if unknown:
                raise TypeError(_unknown_flags(type(self).__name__, name, unknown))
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
            # True: delete an existing attribute and add it again; otherwise (the
            # default, False written out too) re-declare it (`_redeclare`)
            self.overwrite: bool = bool(self._pop_alias(
                self.kargs, ("overwrite",), default=False
            ))
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
        modified one for modifier-only specs). A named spec's plug is owned by
        the node object the spec went to (see ``_plug_of``): ``target`` itself
        for a node (``(node << Float("x")).node is node``), and, for a
        :class:`Plug` (or an :class:`Attribute`), the node object it holds
        (``plug.node``, never a cast of its node's name): ``plug << Float("x")``
        adds ``x`` to ``plug.node`` and returns a plug owned by it. Anything
        else is read as a name (``Node(str(target))``).
        """
        # Modifier-only spec (no longName) -- ``cmds.setAttr`` edit on whatever
        # the target currently points at.
        long_name = self.kargs.get("longName")
        if long_name is None:
            return self._apply_modifier(target)

        if isinstance(target, Attribute):
            # its owner (a plug built from a name casts its node once)
            wrap_node = target.node
        elif isinstance(target, Node):
            wrap_node = target
        else:
            wrap_node = Node(str(target))

        # the node's name (a DAG node's shortest unique path, through the path
        # it holds) names the new plug for cmds
        return self._apply_addattr(str(wrap_node), wrap_node)

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
        """Add the attribute to ``wrap_node`` (named ``node_string``) and return
        its :class:`Plug`, owned by ``wrap_node`` (see ``_plug_of``)."""
        kargs         = dict(self.kargs)
        long_name     = kargs["longName"]
        multi         = self._pop_alias(kargs, ("multi", "m"), default=False)
        default_value = self._pop_alias(kargs, ("defaultValue", "dv"), default=0)

        # An existing attribute is re-declared (its settings applied, its value
        # and connections kept), or with overwrite=True deleted and added again.
        # Use DGNode.has_attr (canonical API) instead of cmds.attributeQuery.
        if wrap_node.has_attr(long_name):
            _refuse_static(node_string, long_name, wrap_node)
            if not self.overwrite:
                return self._redeclare(node_string, long_name, wrap_node)
            # every value addAttr would refuse, before the attribute is deleted
            self._check_values(f"{node_string}.{long_name}")
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
            # a name the node already answers to another way (an alias, a
            # container's published name): a second attribute would be one
            # the name no longer reaches
            _refuse_other_name(node_string, long_name, wrap_node)

        # ---- Compound (Vector / Quat / Color / Euler) ---- #
        added = None
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
            added = wrap_node.add_attr(long_name, **kargs)

        # ---- Multi pre-sizing ---- #
        if multi and self.size is not None:
            self._presize_multi(node_string, long_name, default_value, wrap_node)

        # ---- Note string set ---- #
        new_plug = _plug_of(node_string, long_name, wrap_node, added)
        if self.notes is not None:
            new_plug << self.notes

        return new_plug

    # -- re-declaration -- #

    def _settings(self) -> Dict[str, Any]:
        """The settings of this spec by long flag name: the keywords its caller
        passed, with the values the spec keeps for them (an ``Enum``'s joined
        ``en``, a ``dv`` given by field name as its index), minus the structure
        (``_STRUCTURE``). rig's own defaults (``keyable=True``, a ``Note``'s
        flags) are not settings."""
        given    = {_LONG_FLAGS.get(k, k) for k in getattr(self, "_given", ())} - _STRUCTURE
        settings: Dict[str, Any] = {}
        for key, value in self.kargs.items():
            flag = _LONG_FLAGS.get(key, key)
            if flag in given and flag not in settings:
                settings[flag] = value
        return settings

    def _kind(self) -> tuple:
        """What this spec makes: ``(attributeType, dataType, multi, children)``,
        children as ``((name, attributeType), ...)`` for a compound; the
        attributeType / dataType a spec does not name are None."""
        k        = self.kargs
        at       = k.get("attributeType", k.get("at"))
        dt       = k.get("dataType", k.get("dt"))
        multi    = bool(k.get("multi", k.get("m", False)))
        children = ()
        if self.compound:
            name     = k["longName"]
            children = tuple((f"{name}{s}", self.compoundType or at) for s in self.compound)
            at       = f"{at}{len(self.compound)}"
        return at, dt, multi, children

    def _redeclare(self, node_string: str, long_name: str, wrap_node: Any) -> Any:
        """``node << spec`` when the node has the dynamic attribute already
        (a static attribute is refused before, `_refuse_static`) and the spec
        is not ``overwrite=True``: the attribute stays, with its value and its
        connections, and the settings the spec was given (``_settings``) apply
        in place.

        * An attribute of another kind (attributeType / dataType, multi, a
          compound's children) raises TypeError naming ``overwrite=True``.
        * Otherwise every setting is checked first, and one that cannot apply
          raises TypeError with nothing changed: a setting ``addAttr`` cannot
          edit that differs from the attribute's (hasMinValue / hasMaxValue
          among them), a min above the max, a soft range outside the range, a
          default outside the range (the one given, or the attribute's own
          when only a range is given), an enum default that names no field,
          a keyable or hidden change on a referenced node's attribute (Maya
          refuses the one, never saves the other). Then they apply, in one
          undo step (``rig.attr``): the default through the API in rig's
          undoable command (so an undo restores the previous default),
          ``addAttr -edit`` (min / max, soft min / max, niceName, enumName,
          category, which Maya adds to the attribute's), ``setAttr
          -keyable``, and hidden through the API. A compound's default (a
          list, one per child) and ranges go to its children, its niceName,
          hidden, category and keyable to the parent and to each child, as on
          creation.
        * The value and the connections are never touched. A default is the
          attribute's default, not its value (the value is read before the
          edit, so a plug never set keeps the value it had); ``size=`` pre-sizes
          only a new multi. A ``Note``'s text is set, as on a new attribute.

        Returns the plug, owned by ``wrap_node``."""
        plug_name, fn, targets, edits = self._redeclare_plan(node_string, long_name, wrap_node)
        if targets:
            at, multi = targets[0][2], targets[0][3]
            if not multi and at not in _NO_VALUE and any(e[0] or e[1] is not None for e in edits):
                # a plug never set reads its default: read it now, so that a
                # new default (or one moved into a new range) leaves it as it is
                try:
                    cmds.getAttr(plug_name)
                except (RuntimeError, ValueError) as e:
                    LOGGER.debug("could not read %s before its edit: %s", plug_name, e)
            # one undo step: a chunk when more than one command edits
            steps = sum(bool(e[0]) + sum(x is not None for x in e[1:]) for e in edits)
            if steps > 1:
                with _undo_chunk("rig.attr"):
                    for (target, name, _, _, _), edit in zip(targets, edits):
                        _apply_settings(target, fn.attribute(name), *edit)
            else:
                for (target, name, _, _, _), edit in zip(targets, edits):
                    _apply_settings(target, fn.attribute(name), *edit)

        plug = _plug_of(node_string, long_name, wrap_node)
        if self.notes is not None:
            plug << self.notes
        return plug

    def _redeclare_plan(self, node_string: str, long_name: str, wrap_node: Any) -> tuple:
        """`_redeclare`'s checks, which write nothing: ``(plug name, fn set,
        targets, edits)``, one target ``(plug, attribute name, attributeType,
        multi, settings)`` per attribute edited (a compound's parent, then its
        children) and its checked edit; no target without settings. Raises
        TypeError for another kind or a setting that cannot apply."""
        plug_name = f"{node_string}.{long_name}"
        fn        = wrap_node.fn_set
        attribute = fn.attribute(long_name)
        have      = _existing_kind(node_string, long_name, attribute)
        want      = self._kind()
        if not _same_kind(have, want):
            have_text, want_text = _describe(*have), _describe(*want)
            raise TypeError(
                f"'{plug_name}' exists as {have_text}, not {want_text}; "
                f"overwrite=True replaces it"
            )

        settings = self._existing_settings(plug_name, attribute)
        if not settings:
            return plug_name, fn, [], []
        at, _, multi, children = have
        referenced = fn.isFromReferencedFile and _from_the_file(node_string, long_name)
        if children:
            shared  = {f: v for f, v in settings.items() if f not in _PER_CHILD}
            targets = [(plug_name, long_name, at, multi, shared)]
            index   = "[0]" if multi else ""
            for i, (child, child_at) in enumerate(children):
                own = {f: v for f, v in settings.items() if f != "defaultValue"}
                dv  = settings.get("defaultValue")
                if isinstance(dv, (list, tuple)) and len(dv) > i:
                    own["defaultValue"] = dv[i]  # a scalar is ignored, as on creation
                # a multi's children are named through element 0, which only a
                # default edit would make: their default is fixed, as a multi's
                child_plug = f"{plug_name}{index}.{child}" if multi else f"{node_string}.{child}"
                targets.append((child_plug, child, child_at, multi, own))
        else:
            targets = [(plug_name, long_name, at, multi, settings)]
        # every check before any edit
        edits = [_check_settings(*target, referenced=referenced) for target in targets]
        return plug_name, fn, targets, edits

    def _precheck(self, target: Any) -> None:
        """The checks ``target << spec`` runs before its first edit, writing
        nothing: a ``List << spec`` runs them on every element first, so a
        refused element leaves every other one as it was."""
        long_name = self.kargs.get("longName")
        if long_name is None:
            return
        wrap_node   = target.node if isinstance(target, Attribute) else target
        node_string = str(wrap_node)
        if wrap_node.has_attr(long_name):
            _refuse_static(node_string, long_name, wrap_node)
            if self.overwrite:
                self._check_values(f"{node_string}.{long_name}")
            else:
                self._redeclare_plan(node_string, long_name, wrap_node)
        else:
            _refuse_other_name(node_string, long_name, wrap_node)

    def _existing_settings(self, plug_name: str, attribute: Any) -> Dict[str, Any]:
        """`_settings` for the existing attribute ``attribute``: the hook where
        a spec reads a setting against it (an ``Enum``'s default given by
        field name, see ``Enum``)."""
        return self._settings()

    def _check_values(self, plug_name: str) -> None:
        """The values ``addAttr`` would refuse, checked before an
        ``overwrite=True`` deletes the attribute (TypeError, nothing changed):
        a range bound that is no number, a default that is no number (a
        compound's: a number or a list of them)."""
        for key, value in self.kargs.items():
            flag = _LONG_FLAGS.get(key, key)
            if flag in _RANGE_FLAGS and not _is_number(value):
                raise TypeError(f"'{plug_name}': {key}={value!r} is not a number; nothing was changed")
            if flag == "defaultValue":
                values = value if self.compound and isinstance(value, (list, tuple)) else [value]
                if not all(_is_number(v) for v in values):
                    raise TypeError(
                        f"'{plug_name}': {key}={value!r} is not a number"
                        + (" or a list of numbers" if self.compound else "")
                        + "; nothing was changed"
                    )

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

            # Per-child default value, from a sequence given as dv= or
            # defaultValue= (never the sequence itself); a scalar is ignored.
            base_dv = self._pop_alias(child_kargs, ("defaultValue", "dv"), default=None)
            if isinstance(base_dv, (list, tuple)) and len(base_dv) > i:
                child_kargs["defaultValue"] = base_dv[i]

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
#  Re-declaration helpers (`_AttrSpec._redeclare`)
# ---------------------------------------------------------------------- #

# attributeTypes with no numeric value: no range check, no value read
_NO_VALUE = frozenset(("typed", "message", "matrix"))


def _existing_kind(node_string: str, long_name: str, attribute: Any) -> tuple:
    """``(attributeType, dataType, multi, children)`` of the dynamic attribute
    ``long_name`` of the node ``node_string`` (``attribute``, its MObject), as
    ``_AttrSpec._kind`` spells a spec's (dataType None unless the attributeType
    is ``typed``). ``addAttr -query`` and the API (``attributeQuery -node``
    costs about 155 us a call, ``addAttr -query`` 17 us); a multi compound's
    children are named through element 0, which a query does not make."""
    plug = f"{node_string}.{long_name}"
    at   = cmds.addAttr(plug, query=True, attributeType=True)
    dt   = None
    if at == "typed":
        dt = (cmds.addAttr(plug, query=True, dataType=True) or [None])[0]
    multi    = OpenMaya.MFnAttribute(attribute).array
    children = ()
    if attribute.hasFn(OpenMaya.MFn.kCompoundAttribute):
        compound = OpenMaya.MFnCompoundAttribute(attribute)
        names    = [
            OpenMaya.MFnAttribute(compound.child(i)).name for i in range(compound.numChildren())
        ]
        children = tuple(
            (name, cmds.addAttr(
                f"{plug}[0].{name}" if multi else f"{node_string}.{name}",
                query=True, attributeType=True,
            ))
            for name in names
        )
    return at, dt, multi, children


def _same_kind(have: tuple, want: tuple) -> bool:
    """Whether the existing attribute (``have``) is of the kind a spec makes
    (``want``); a spec that names no attributeType / dataType does not check it."""
    at, dt, multi, children = want
    if at is not None and at != have[0]:
        return False
    if dt is not None and (have[0] != "typed" or dt != have[1]):
        return False
    return multi == have[2] and children == have[3]


def _describe(at: Optional[str], dt: Optional[str], multi: bool, children: tuple) -> str:
    """``a double``, ``a multi double``, ``a string``, ``a double3 of double
    (wX, wY, wZ)``: a kind as ``_existing_kind`` / ``_AttrSpec._kind`` give it."""
    if at == "typed" and dt == "matrix":
        kind = "matrix data"  # not the attributeType matrix
    elif at in (None, "typed") and dt:
        kind = dt
    else:
        kind = at or "attribute"
    if children:
        types = {t for _, t in children}
        if len(types) == 1:
            kind += f" of {types.pop()}"
        kind += " (" + ", ".join(name for name, _ in children) + ")"
    if multi:
        return "a multi " + kind
    return ("an " if kind[0] in "aeiou" else "a ") + kind


def _same_setting(current: Any, value: Any) -> bool:
    """Whether an attribute's setting (an ``addAttr`` query) is ``value``."""
    if isinstance(value, bool) or isinstance(current, bool):
        return bool(current) == bool(value)
    if isinstance(current, (list, tuple)) and not isinstance(value, (list, tuple)):
        return list(current) == [value]
    if isinstance(current, (int, float)) and isinstance(value, (int, float)):
        return abs(current - value) <= 1e-9 * max(1.0, abs(value))
    return current == value


def _is_number(value: Any) -> bool:
    """A real number (a bool counts, as Maya reads it)."""
    return isinstance(value, numbers.Real)


def _unknown_flags(kind: str, name: str, unknown: List[str]) -> str:
    """The TypeError text of a spec given keywords that are no addAttr flag."""
    words = ", ".join(f"{key}=" for key in unknown)
    text  = f"{kind}({name!r}): {words} {'is not an addAttr flag' if len(unknown) == 1 else 'are not addAttr flags'}"
    if "update" in unknown:
        text += (
            "; re-declaring an attribute applies the settings you pass and keeps its "
            "value (update= is define's: it sets the values of a node define finds)"
        )
    close = [
        found for key in unknown if key != "update"
        for found in difflib.get_close_matches(key, sorted(_SPEC_FLAGS), n=1, cutoff=0.75)
    ]
    if close:
        text += f" (did you mean {', '.join(f'{c}=' for c in close)}?)"
    return text + "; nothing was changed"


def _refuse_static(node_string: str, long_name: str, wrap_node: Any) -> None:
    """A declaration names a static attribute of the node (its long or short
    name: ``Float("s")`` is ``scale``): TypeError before any edit. The node
    type owns it; a declaration adds or re-declares a dynamic attribute."""
    attribute = wrap_node.fn_set.attribute(long_name)
    fn        = OpenMaya.MFnAttribute(attribute)
    if fn.dynamic:
        return
    named = fn.name if fn.name == long_name else f"{fn.name}, whose short name is {long_name!r}"
    raise TypeError(
        f"'{node_string}.{long_name}' is a static attribute of the "
        f"{wrap_node.fn_set.typeName} ({named}); a declaration adds a dynamic "
        f"attribute: pick another name"
    )


def _refuse_other_name(node_string: str, long_name: str, wrap_node: Any) -> None:
    """A new attribute's name the node already answers to another way: an
    alias (``aliasAttr``, a blendShape target) or a container's published
    name. TypeError before any edit: a second attribute of that name would be
    one the name no longer reaches."""
    fn = wrap_node.fn_set
    if fn.findAlias(long_name).isNull():
        return
    # (a container's published name is an alias of its borderConnections)
    if fn.typeName in ("container", "dagContainer"):
        raise TypeError(
            f"'{node_string}.{long_name}' is a name the container publishes; "
            f"re-declare the published attribute on its node, or pick another name"
        )
    pairs  = cmds.aliasAttr(node_string, query=True) or []
    target = dict(zip(pairs[::2], pairs[1::2])).get(long_name, "another attribute")
    raise TypeError(
        f"'{node_string}.{long_name}' is an alias of {node_string}.{target}; "
        f"re-declare {target}, or pick another name"
    )


def _from_the_file(node_string: str, long_name: str) -> bool:
    """Whether the dynamic attribute ``long_name`` of a referenced node comes
    from the referenced file (not added in this scene, which a reference edit
    ``addAttr ... -longName <name>`` records)."""
    try:
        reference = cmds.referenceQuery(node_string, referenceNode=True)
        edits     = cmds.referenceQuery(reference, editStrings=True, editCommand="addAttr") or []
    except RuntimeError:
        return True
    added = f"-longName {long_name} "
    return not any(added in edit and edit.rsplit(" ", 1)[-1].lstrip("|").endswith(node_string.rsplit("|", 1)[-1]) for edit in edits)


def _check_settings(
    target: str, name: str, at: str, multi: bool, settings: dict, referenced: bool = False
) -> tuple:
    """Check the settings of one attribute (``target``, its plug name) before any
    edit: TypeError for one that cannot apply. Returns what `_apply_settings`
    does: ``(addAttr edit flags, default for the API or None, keyable,
    hidden)``, None for a keyable / hidden not given. ``referenced``: the
    attribute comes from a referenced file."""
    edit = {}
    for flag, value in settings.items():
        if flag in ("keyable", "hidden"):
            continue
        if (
            flag in _EDITABLE
            and not (flag == "defaultValue" and multi)  # Maya: a multi's default is fixed
            and not (flag == "enumName" and at != "enum")
        ):
            edit[flag] = value
        else:
            # addAttr cannot edit it: the attribute must have it already
            try:
                current = cmds.addAttr(target, query=True, **{flag: True})
            except (RuntimeError, TypeError, ValueError) as e:
                raise TypeError(
                    f"'{target}': {flag}={value!r} cannot be read on the existing "
                    f"attribute ({str(e).strip()}); overwrite=True replaces it"
                ) from None
            if not _same_setting(current, value):
                raise TypeError(
                    f"'{target}': {flag} cannot be changed on an existing attribute "
                    f"(it is {current!r}, the spec gives {value!r}); overwrite=True "
                    f"replaces it"
                )
    keyable, hidden = settings.get("keyable"), settings.get("hidden")
    if referenced:
        # Maya refuses a keyable change of an attribute from a referenced file,
        # and a hidden flag set here is not saved as a reference edit
        if keyable is not None and bool(cmds.getAttr(target, keyable=True)) != bool(keyable):
            raise TypeError(
                f"'{target}' comes from a referenced file: its keyable state is set in "
                f"that file (Maya refuses the change); nothing was changed"
            )
        if hidden is not None and bool(cmds.addAttr(target, query=True, hidden=True)) != bool(hidden):
            raise TypeError(
                f"'{target}' comes from a referenced file: its hidden flag is set in that "
                f"file (Maya would not save the change); nothing was changed"
            )
    if at not in _NO_VALUE and at != "enum" and (
        {"minValue", "maxValue", "defaultValue", "softMinValue", "softMaxValue"} & edit.keys()
    ):
        # Maya ignores a min above the max and a default outside the range,
        # silently, and moves the default into a new range: refuse them here,
        # before any edit
        def bound(flag: str, has_flag: str) -> Any:
            if flag in edit:
                return edit[flag]
            if not cmds.addAttr(target, query=True, **{has_flag: True}):
                return None
            return cmds.addAttr(target, query=True, **{flag: True})

        low, high = bound("minValue", "hasMinValue"), bound("maxValue", "hasMaxValue")
        if low is not None and high is not None and low > high:
            raise TypeError(f"'{target}': min={low} is above max={high}")
        if {"softMinValue", "softMaxValue"} & edit.keys():
            # a soft bound given: inside the soft pair and the range
            soft_low  = bound("softMinValue", "hasSoftMinValue")
            soft_high = bound("softMaxValue", "hasSoftMaxValue")
            if soft_low is not None and soft_high is not None and soft_low > soft_high:
                raise TypeError(f"'{target}': softMinValue={soft_low} is above softMaxValue={soft_high}")
            for soft, label in ((soft_low, "softMinValue"), (soft_high, "softMaxValue")):
                if soft is None or label not in edit:
                    continue
                if low is not None and soft < low:
                    raise TypeError(f"'{target}': {label}={soft} is below min={low}")
                if high is not None and soft > high:
                    raise TypeError(f"'{target}': {label}={soft} is above max={high}")
        dv = edit.get("defaultValue")
        if isinstance(dv, (int, float)):
            if low is not None and dv < low:
                raise TypeError(f"'{target}': dv={dv} is below min={low}")
            if high is not None and dv > high:
                raise TypeError(f"'{target}': dv={dv} is above max={high}")
        elif "defaultValue" not in edit and {"minValue", "maxValue"} & edit.keys():
            # a new range moves the attribute's own default into it, silently
            # and for good (an undo keeps it): refuse, naming dv=
            current = cmds.addAttr(target, query=True, defaultValue=True)
            if isinstance(current, (int, float)):
                if low is not None and current < low:
                    raise TypeError(f"'{target}': the default {current} is below min={low}; pass dv=")
                if high is not None and current > high:
                    raise TypeError(f"'{target}': the default {current} is above max={high}; pass dv=")
    if at == "enum" and isinstance(edit.get("defaultValue"), numbers.Integral):
        fields = (
            _parse_enum_names(edit["enumName"]) if "enumName" in edit
            else _enum_fields(OpenMaya.MFnEnumAttribute(_attribute_of(target)))
        )
        if edit["defaultValue"] not in {value for _, value in fields}:
            listed = ", ".join(f"{field}={value}" for field, value in fields) or "none"
            raise TypeError(
                f"'{target}': dv={edit['defaultValue']} is not one of its enum fields' values: {listed}"
            )
    default = edit.pop("defaultValue", None) if _API_DEFAULT.get(at) else None
    return edit, default, keyable, hidden


def _attribute_of(target: str) -> Any:
    """The attribute MObject of the plug name ``target``."""
    sel = OpenMaya.MSelectionList()
    sel.add(target)
    return sel.getPlug(0).attribute()


def _apply_settings(
    target:    str,
    attribute: Any,
    edit:      dict,
    default:   Any,
    keyable:   Any,
    hidden:    Any,
) -> None:
    """Apply one attribute's settings, checked by `_check_settings`; ``attribute``
    is its MObject (API 2.0). The default first, so a new range already holds
    it (Maya moves a default a range edit excludes, for good)."""
    from rig.nodetypes.plugins import _run_undoable

    if default is not None:
        _run_undoable(_DefaultEdit(attribute, default))
    if edit:
        cmds.addAttr(target, edit=True, **edit)
    if keyable is not None and bool(cmds.getAttr(target, keyable=True)) != bool(keyable):
        cmds.setAttr(target, keyable=bool(keyable))
    if hidden is not None and OpenMaya.MFnAttribute(attribute).hidden != bool(hidden):
        _run_undoable(_HiddenEdit(attribute, bool(hidden)))


class _HiddenEdit:
    """An attribute's ``hidden`` flag, set through rig's undoable command (one
    undo step): ``addAttr -edit`` cannot change it. API only, as the command
    requires."""

    def __init__(self, attribute: Any, state: bool) -> None:
        self.attribute = attribute
        self.state     = state
        self.previous  = state

    def doIt(self) -> None:
        fn            = OpenMaya.MFnAttribute(self.attribute)
        self.previous = fn.hidden
        fn.hidden     = self.state

    def redoIt(self) -> None:
        OpenMaya.MFnAttribute(self.attribute).hidden = self.state

    def undoIt(self) -> None:
        OpenMaya.MFnAttribute(self.attribute).hidden = self.previous


# the attributeTypes whose default `_DefaultEdit` sets through the API, with
# the unit ``addAttr -defaultValue`` reads the number in (the internal one: a
# doubleAngle's in radians, a doubleLinear's in centimeters); any other keeps
# ``addAttr -edit -defaultValue``
_API_DEFAULT = {
    "double": "numeric", "float": "numeric", "long": "numeric", "short": "numeric",
    "byte": "numeric", "char": "numeric", "bool": "numeric", "enum": "enum",
    "doubleAngle": "angle", "doubleLinear": "distance",
}


class _DefaultEdit:
    """An attribute's default, set through rig's undoable command (one undo
    step whose undo restores the previous default: Maya's undo of an
    ``addAttr -edit -defaultValue`` sets it to 0). API only."""

    def __init__(self, attribute: Any, value: Any) -> None:
        self.attribute = attribute
        self.value     = value
        self.previous  = None

    def _fn(self) -> Any:
        if self.attribute.hasFn(OpenMaya.MFn.kEnumAttribute):
            return OpenMaya.MFnEnumAttribute(self.attribute)
        if self.attribute.hasFn(OpenMaya.MFn.kUnitAttribute):
            return OpenMaya.MFnUnitAttribute(self.attribute)
        return OpenMaya.MFnNumericAttribute(self.attribute)

    def _set(self, fn: Any, value: Any) -> None:
        if isinstance(fn, OpenMaya.MFnEnumAttribute):
            fn.default = int(value)
        elif isinstance(fn, OpenMaya.MFnUnitAttribute):
            if self.attribute.apiType() == OpenMaya.MFn.kDoubleAngleAttribute:
                fn.default = OpenMaya.MAngle(float(value), OpenMaya.MAngle.kRadians)
            else:
                fn.default = OpenMaya.MDistance(float(value), OpenMaya.MDistance.kCentimeters)
        elif self.attribute.apiType() == OpenMaya.MFn.kNumericAttribute and (
            fn.numericType() == OpenMaya.MFnNumericData.kBoolean
        ):
            fn.default = bool(value)
        elif fn.numericType() in (OpenMaya.MFnNumericData.kFloat, OpenMaya.MFnNumericData.kDouble):
            fn.default = float(value)
        else:
            fn.default = int(value)

    def doIt(self) -> None:
        fn      = self._fn()
        default = fn.default
        if isinstance(default, OpenMaya.MAngle):
            default = default.asRadians()
        elif isinstance(default, OpenMaya.MDistance):
            default = default.asCentimeters()
        self.previous = default
        self._set(fn, self.value)

    def redoIt(self) -> None:
        self._set(self._fn(), self.value)

    def undoIt(self) -> None:
        self._set(self._fn(), self.previous)



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
    leaning on ``rig.nodetypes.Attribute``'s ``data_type`` / ``attribute_type`` /
    ``is_multi`` / ``num_children`` properties.
    """
    from rig._internal.node import Node
    from rig._internal.plug import Plug
    from rig._internal.types import _is_attribute, _is_compound

    # Resolve dst_node to its node object.
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
        MeshAttr,
        Message,
        NurbsCurveAttr,
        NurbsSurfaceAttr,
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
        "mesh":         MeshAttr,
        "nurbsCurve":   NurbsCurveAttr,
        "nurbsSurface": NurbsSurfaceAttr,
        "message":      Message,
    }
    spec_cls = mapping.get(data_type) or mapping.get(attr_type) or Float
    return spec_cls(attr_name, multi=multi)