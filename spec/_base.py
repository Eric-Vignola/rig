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

import logging
from typing import Any, Dict, List, Optional

from maya import cmds
from maya.api import OpenMaya
# rig.nodetypes imports neither rig.spec nor rig._internal.plug
from rig.nodetypes._base import Attribute, Node, _attr_handle, _attr_state, _handle_valid


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
# what makes the attribute (its name and kind), never a setting
_STRUCTURE = frozenset((
    "longName", "attributeType", "dataType", "multi", "size", "overwrite", "parent",
    "numberOfChildren",
))
# the settings ``addAttr -edit`` changes (runs\r4b\NC8\probe_addattr*.txt); keyable
# goes through ``setAttr -keyable``, hidden through the API, hasMinValue /
# hasMaxValue only when they differ (an edit toggles them, whatever the value);
# any other setting cannot change: equal to the attribute's, or a TypeError
_EDITABLE   = frozenset((
    "defaultValue", "minValue", "maxValue", "softMinValue", "softMaxValue", "niceName",
    "enumName", "category",
))
_HAS_RANGE  = ("hasMinValue", "hasMaxValue")
# a compound's settings that go to its children (the parent takes the others)
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
            if not self.overwrite:
                return self._redeclare(node_string, long_name, wrap_node)
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
        """``node << spec`` when the node has the attribute already and the spec
        is not ``overwrite=True``: the attribute stays, with its value and its
        connections, and the settings the spec was given (``_settings``) apply
        in place.

        * A static attribute: its plug when the spec has no settings; a setting
          raises TypeError (the node type owns its settings).
        * A dynamic attribute of another kind (attributeType / dataType, multi, a
          compound's children) raises TypeError naming ``overwrite=True``.
        * Otherwise every setting is checked first, and one that cannot apply
          raises TypeError with nothing changed: a setting ``addAttr`` cannot
          edit that differs from the attribute's, a min above the max, a default
          outside the range. Then they apply: ``addAttr -edit`` (default, min /
          max, soft min / max, niceName, enumName, category; hasMinValue /
          hasMaxValue when they differ), ``setAttr -keyable``, and hidden through
          the API (one undo step, rig's undoable command). A compound's default
          (a list, one per child) and ranges go to its children, its niceName,
          hidden and category to the parent, keyable to both.
        * The value and the connections are never touched. A default is the
          attribute's default, not its value (the value is read before the
          edit, so a plug never set keeps the value it had); ``size=`` pre-sizes
          only a new multi. A ``Note``'s text is set, as on a new attribute.

        Returns the plug, owned by ``wrap_node``."""
        plug_name = f"{node_string}.{long_name}"
        settings  = self._settings()
        fn        = wrap_node.fn_set
        if not OpenMaya.MFnAttribute(fn.attribute(long_name)).dynamic:
            if settings:
                raise TypeError(
                    f"'{plug_name}' is a static attribute; its settings cannot be changed"
                )
            return _plug_of(node_string, long_name, wrap_node)

        have = _existing_kind(node_string, long_name)
        want = self._kind()
        if not _same_kind(have, want):
            have_text, want_text = _describe(*have), _describe(*want)
            raise TypeError(
                f"'{plug_name}' exists as {have_text}, not {want_text}; "
                f"overwrite=True replaces it"
            )

        if settings:
            at, _, multi, children = have
            if children:
                targets = [(plug_name, long_name, at, multi, {
                    f: v for f, v in settings.items() if f not in _PER_CHILD
                })]
                index = "[0]" if multi else ""
                for i, (child, child_at) in enumerate(children):
                    own = {f: v for f, v in settings.items() if f in _PER_CHILD or f == "keyable"}
                    dv  = own.pop("defaultValue", None)
                    if isinstance(dv, (list, tuple)) and len(dv) > i:
                        own["defaultValue"] = dv[i]  # a scalar is ignored, as on creation
                    # a multi's children are named through element 0, which only a
                    # default edit would make: their default is fixed, as a multi's
                    child_plug = f"{plug_name}{index}.{child}" if multi else f"{node_string}.{child}"
                    targets.append((child_plug, child, child_at, multi, own))
            else:
                targets = [(plug_name, long_name, at, multi, settings)]

            # every check before any edit
            edits = [_check_settings(*target) for target in targets]
            if not multi and at not in _NO_VALUE and any(e[0] or e[1] for e in edits):
                # a plug never set reads its default: read it now, so that a
                # new default (or one moved into a new range) leaves it as it is
                try:
                    cmds.getAttr(plug_name)
                except (RuntimeError, ValueError) as e:
                    LOGGER.debug("could not read %s before its edit: %s", plug_name, e)
            for (target, name, _, _, _), edit in zip(targets, edits):
                _apply_settings(target, fn.attribute(name), *edit)

        plug = _plug_of(node_string, long_name, wrap_node)
        if self.notes is not None:
            plug << self.notes
        return plug

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


def _existing_kind(node_string: str, long_name: str) -> tuple:
    """``(attributeType, dataType, multi, children)`` of the dynamic attribute
    ``long_name`` of the node ``node_string``, as ``_AttrSpec._kind`` spells a
    spec's (dataType None unless the attributeType is ``typed``)."""
    at = cmds.attributeQuery(long_name, node=node_string, attributeType=True)
    dt = None
    if at == "typed":
        dt = (cmds.addAttr(f"{node_string}.{long_name}", query=True, dataType=True) or [None])[0]
    multi    = bool(cmds.attributeQuery(long_name, node=node_string, multi=True))
    kids     = cmds.attributeQuery(long_name, node=node_string, listChildren=True) or []
    children = tuple(
        (kid, cmds.attributeQuery(kid, node=node_string, attributeType=True)) for kid in kids
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


def _check_settings(target: str, name: str, at: str, multi: bool, settings: dict) -> tuple:
    """Check the settings of one attribute (``target``, its plug name) before any
    edit: TypeError for one that cannot apply. Returns what `_apply_settings`
    does: ``(addAttr edit flags, {hasMinValue / hasMaxValue: state}, keyable,
    hidden)``, None for a keyable / hidden not given."""
    edit, has = {}, {}
    for flag, value in settings.items():
        if flag in ("keyable", "hidden"):
            continue
        if flag in _HAS_RANGE:
            has[flag] = bool(value)
        elif (
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
    if at not in _NO_VALUE and at != "enum" and (
        has or {"minValue", "maxValue", "defaultValue"} & edit.keys()
    ):
        # Maya ignores a min above the max and a default outside the range,
        # silently: refuse them here, before any edit
        def bound(flag: str, has_flag: str) -> Any:
            if flag in edit:
                return edit[flag]
            if has.get(has_flag) is False:
                return None
            return cmds.addAttr(target, query=True, **{flag: True})

        low, high = bound("minValue", "hasMinValue"), bound("maxValue", "hasMaxValue")
        if low is not None and high is not None and low > high:
            raise TypeError(f"'{target}': min={low} is above max={high}")
        dv = edit.get("defaultValue")
        if isinstance(dv, (int, float)):
            if low is not None and dv < low:
                raise TypeError(f"'{target}': dv={dv} is below min={low}")
            if high is not None and dv > high:
                raise TypeError(f"'{target}': dv={dv} is above max={high}")
    return edit, has, settings.get("keyable"), settings.get("hidden")


def _apply_settings(
    target:    str,
    attribute: Any,
    edit:      dict,
    has:       dict,
    keyable:   Any,
    hidden:    Any,
) -> None:
    """Apply one attribute's settings, checked by `_check_settings`; ``attribute``
    is its MObject (API 2.0)."""
    if edit:
        cmds.addAttr(target, edit=True, **edit)
    for flag, state in has.items():
        # an addAttr edit toggles hasMinValue / hasMaxValue, whatever the value given
        if bool(cmds.addAttr(target, query=True, **{flag: True})) != state:
            cmds.addAttr(target, edit=True, **{flag: state})
    if keyable is not None and bool(cmds.getAttr(target, keyable=True)) != bool(keyable):
        cmds.setAttr(target, keyable=bool(keyable))
    if hidden is not None and OpenMaya.MFnAttribute(attribute).hidden != bool(hidden):
        from rig.nodetypes.plugins import _run_undoable

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