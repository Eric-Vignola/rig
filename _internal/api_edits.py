"""
The API fast paths of rig's edits (round 5, phase D): a connect (`api_connect`) and a
single numeric value set (`api_set`), each made through an ``MDGModifier`` and journaled
(`rig._internal.journal.add`), so that a statement's API edits are one undo step. Locks,
keyable and channelBox flags, disconnects and every other edit stay on ``cmds``.

Each fast path takes the API route only where its guards prove the edit is the one the
``cmds`` route would make, with the same result: otherwise, and whenever the API edit
fails, it returns False and the caller runs its ``cmds`` route unchanged, which raises
today's errors itself (at today's points, with today's texts and chains: no API
exception is in flight then, the ``except`` block ends in here).

rig's undo plug-in (``undoable_api_command``) must be loaded for a journal to be
registered. It is loaded at the first API edit of a session (by full path from rig's
folder, else by name from ``MAYA_PLUG_IN_PATH``, see `rig.nodetypes.plugins`). If it
cannot be loaded, the fast paths take the ``cmds`` route, and one warning per session
says that builds run slower; the load is tried again only after a plug-in is loaded or
unloaded (`rig.nodetypes.plugins._fast_path_ready`).

While a build's journals are on the undo queue, ``unloadPlugin('undoable_api_command')``
is refused (run ``cmds.flushUndo()`` first). A forced unload leaves those undo entries
inert: undoing them then does nothing.

``journal.API_EDITS = False`` sends every edit through ``cmds`` (for A/B runs, and the
tests that pin the ``cmds`` route).
"""

from __future__ import annotations

from typing import Any

from maya.api import OpenMaya
from rig._internal import journal

_MDGModifier  = OpenMaya.MDGModifier
_MFnAttribute = OpenMaya.MFnAttribute
_FREE         = OpenMaya.MPlug.kFreeToChange
_add          = journal.add

# bound at the first fast-path call (`_bind`), once rig's classes exist: the exact plug
# classes (Attribute, Plug; a subclass, a ComponentPlug included, takes cmds) and rig's
# own node naming (the `name` / `fn_set` / `ensure_valid` a node class must keep)
_API_CLASSES: tuple = ()
_NAMES:       tuple = ()
_FN_SETS:     tuple = ()
_ENSURE:      Any   = None
_BOUND              = False


def _bind() -> None:
    global _API_CLASSES, _NAMES, _FN_SETS, _ENSURE, _BOUND
    global _names_own_plug, _attr_mobject, _ready

    from rig._internal.plug import Plug
    from rig.nodetypes import _base
    from rig.nodetypes.dag_node import DAGNode
    from rig.nodetypes.dg_node import DGNode
    from rig.nodetypes.mesh import Mesh
    from rig.nodetypes.plugins import _fast_path_ready as _ready

    _names_own_plug = _base._names_own_plug
    _attr_mobject   = _base._attr_mobject
    _API_CLASSES    = (_base.Attribute, Plug)
    _NAMES          = (DGNode.__dict__["name"], DAGNode.__dict__["name"])
    _FN_SETS        = (DGNode.__dict__["fn_set"], Mesh.__dict__["fn_set"])
    _ENSURE         = DGNode.__dict__["ensure_valid"]
    _BOUND          = True


_OBJECT_GETATTRIBUTE = object.__dict__["__getattribute__"]


def _rig_named(owner: Any) -> bool:
    """True if ``owner`` (a plug's node object) is named by rig's own naming: its class
    keeps rig's ``name``, ``fn_set`` and ``ensure_valid`` (and Python's attribute lookup),
    none is set on the node itself, and a DAG node has no taken path to rebind (naming it
    would rebind it first, `DAGNode.name`). The name of its plug is then the name of its
    MPlug (`_names_own_plug` checks the rest), so the API edit is the cmds edit."""
    cls = type(owner)
    try:
        if (
            cls.__getattribute__ is not _OBJECT_GETATTRIBUTE
            or cls.name not in _NAMES
            or cls.fn_set not in _FN_SETS
            or cls.ensure_valid is not _ENSURE
        ):
            return False
    except AttributeError:
        return False
    d = owner.__dict__
    return "ensure_valid" not in d and "_taken_mdagpath" not in d


def api_connect(src: Any, dst: Any) -> bool:
    """``src.connect(dst)`` through an ``MDGModifier``, journaled. True if it was made;
    False, with nothing changed, when the caller must take its ``cmds`` route
    (`Attribute.connect`). The guards:

    * ``src`` and ``dst`` are exactly ``Attribute`` or ``Plug``, each with a node object
      named by rig's own naming (`_rig_named`);
    * both nodes valid (``_handle.isValid()``: a node deleted to the undo queue is alive,
      not valid), each dynamic attribute alive;
    * neither plug an array root, the two plugs on two nodes (``a.x << a.x``, a child
      into its parent);
    * ``dst`` free to change (``isFreeToChange(True, True)``: not locked, nor a parent or
      an array root of it; not a destination, nor a connected child or parent);
    * each name resolves to its own MPlug (`_names_own_plug`);
    * ``src`` readable and connectable, ``dst`` writable and connectable;
    * rig's undo command ready (`rig.nodetypes.plugins._fast_path_ready`).

    A ``doIt`` that raises (incompatible types: Maya prints its ``Error:`` line) or a
    registration that fails is undone here, and the ``cmds`` route runs after this
    returns, outside the ``except`` block."""
    if not _BOUND:
        _bind()
    classes = _API_CLASSES
    if type(src) not in classes or type(dst) not in classes or not journal.API_EDITS:
        return False
    sd = src.__dict__
    dd = dst.__dict__
    so = sd.get("_node")
    do = dd.get("_node")
    if so is None or do is None:
        return False
    handle = dd.get("_handle")
    if handle is None or not handle.isValid():
        return False
    handle = sd.get("_handle")
    if handle is None or not handle.isValid():
        return False
    handle = dd.get("_attrh")
    if handle is not None and not handle.isAlive():
        return False
    handle = sd.get("_attrh")
    if handle is not None and not handle.isAlive():
        return False
    sp = sd.get("_mplug")
    dp = dd.get("_mplug")
    if sp is None or dp is None:
        return False
    try:
        if (
            sp.isArray
            or dp.isArray
            or sp.node() == dp.node()
            or dp.isFreeToChange(True, True) != _FREE
            or not _rig_named(do)
            or not _rig_named(so)
            or not _names_own_plug(dst)
            or not _names_own_plug(src)
        ):
            return False
        fn = _MFnAttribute(sp.attribute())
        if not (fn.readable and fn.connectable):
            return False
        fn = _MFnAttribute(dp.attribute())
        if not (fn.writable and fn.connectable):
            return False
    except Exception:  # noqa: BLE001 (a plug the guards cannot read: the cmds route)
        return False
    if not _ready():
        return False
    modifier = _MDGModifier()
    try:
        modifier.connect(sp, dp)
    except Exception:  # noqa: BLE001 (nothing done: the cmds route)
        return False
    return _apply(modifier)


def _apply(modifier: Any) -> bool:
    """Applies ``modifier`` (one edit, queued) and journals it (`rig._internal.journal.add`).
    True if both were done; False, with the edit undone, when either fails: the caller then
    takes its ``cmds`` route, outside this ``except`` block, which raises today's error
    (CR-6: no API exception is chained onto it). The one exception: when the statement's
    earlier edits could not be registered (`journal.add` after an undo; `journal._flush`
    reverted them), the edit is undone and that error propagates, so the statement never
    goes on without them."""
    try:
        modifier.doIt()
    except Exception:  # noqa: BLE001 (the cmds route raises today's error)
        modifier.undoIt()
        return False
    try:
        _add(modifier)
    except Exception as error:  # noqa: BLE001 (registered nothing: the cmds route makes the edit)
        modifier.undoIt()
        if getattr(error, "_rig_reverted", False):
            raise
        return False
    return True


# --------------------------------------------------------------------------- value sets

_NT                  = OpenMaya.MFnNumericData
_MFnNumericAttribute = OpenMaya.MFnNumericAttribute
_MFnUnitAttribute    = OpenMaya.MFnUnitAttribute
_MFnEnumAttribute    = OpenMaya.MFnEnumAttribute
_MDistance           = OpenMaya.MDistance
_MAngle              = OpenMaya.MAngle
_K_NUMERIC           = OpenMaya.MFn.kNumericAttribute
_K_ENUM              = OpenMaya.MFn.kEnumAttribute
_K_LINEAR            = (OpenMaya.MFn.kDoubleLinearAttribute, OpenMaya.MFn.kFloatLinearAttribute)
_K_ANGLE             = (OpenMaya.MFn.kDoubleAngleAttribute, OpenMaya.MFn.kFloatAngleAttribute)

# the int values cmds.setAttr reads into a float kind (past 32 bits it refuses one: "Error
# reading data element")
_INT_LO, _INT_HI = -(2 ** 31), 2 ** 31 - 1
# {numeric type: the modifier's method} of the float kinds
_FLOAT_METHODS = {_NT.kDouble: "newPlugValueDouble", _NT.kFloat: "newPlugValueFloat"}
# {numeric type: (lowest, highest int value, the modifier's method)} of the integer kinds:
# the type's range (cmds wraps, zeroes or refuses past it); a char is held to 0..127
_INT_KINDS = {
    _NT.kInt:   (_INT_LO, _INT_HI, "newPlugValueInt"),
    _NT.kShort: (-32768, 32767, "newPlugValueShort"),
    _NT.kByte:  (0, 255, "newPlugValueChar"),
    _NT.kChar:  (0, 127, "newPlugValueChar"),
}
# the hard (min, max) a numeric kind has of its own: any other hard limit takes cmds
_OWN_LIMITS = {_NT.kByte: (0.0, 255.0), _NT.kChar: (0.0, 255.0), _NT.kBoolean: (0.0, 1.0)}


def _float_value(value: Any, tv: type) -> Any:
    """``value`` as the float a float or unit kind takes from cmds.setAttr, or None: a float,
    or an int cmds reads (32 bits); a bool, a numpy scalar, anything else: None."""
    if tv is float:
        return value
    if tv is int and _INT_LO <= value <= _INT_HI:
        return float(value)
    return None


def _set_plan(mobject: Any, value: Any, tv: type) -> Any:
    """``(method, value)``: the ``MDGModifier`` call that sets ``value`` on a plug of the
    attribute ``mobject`` as ``cmds.setAttr`` would (CR-4), or None: the cmds route.

    * double / float: a float, or an int within 32 bits;
    * doubleLinear / floatLinear, doubleAngle / floatAngle: the same, in the UI unit
      (``MDistance`` / ``MAngle``, as cmds reads it);
    * long, short, byte, char: an int within the type's range (a char: 0..127);
    * enum: an int within its fields' ``getMin()`` .. ``getMax()``;
    * bool: an int or a bool, 0 or 1;
    * an attribute with a hard min or max (other than a byte's, a char's or a bool's own),
      a non-writable one, a proxy, and every other kind (time, string, typed, compound,
      message, generic ...) or value (a float into an integer kind, a bool into a number,
      a numpy scalar ...): None. cmds rounds, wraps, zeroes or refuses those."""
    kind = mobject.apiType()
    if kind == _K_NUMERIC:
        fn = _MFnNumericAttribute(mobject)
        nt = fn.numericType()
        has_min, has_max = fn.hasMin(), fn.hasMax()
        if (has_min or has_max) and not (
            has_min and has_max and (fn.getMin(), fn.getMax()) == _OWN_LIMITS.get(nt)
        ):
            return None
        method = _FLOAT_METHODS.get(nt)
        if method is not None:
            value = _float_value(value, tv)
            if value is None:
                return None
        elif nt in _INT_KINDS:
            lo, hi, method = _INT_KINDS[nt]
            if tv is not int or not lo <= value <= hi:
                return None
        elif nt == _NT.kBoolean:
            if (tv is not int and tv is not bool) or (value != 0 and value != 1):
                return None
            method, value = "newPlugValueBool", bool(value)
        else:
            return None
    elif kind == _K_ENUM:
        fn = _MFnEnumAttribute(mobject)
        if tv is not int or not fn.getMin() <= value <= fn.getMax():
            return None
        method = "newPlugValueShort"
    elif kind in _K_LINEAR or kind in _K_ANGLE:
        fn = _MFnUnitAttribute(mobject)
        if fn.hasMin() or fn.hasMax():
            return None
        value = _float_value(value, tv)
        if value is None:
            return None
        if kind in _K_LINEAR:
            method, value = "newPlugValueMDistance", _MDistance(value, _MDistance.uiUnit())
        else:
            method, value = "newPlugValueMAngle", _MAngle(value, _MAngle.uiUnit())
    else:
        return None
    if not fn.writable or fn.isProxyAttribute:
        return None
    return method, value


def api_set(attr: Any, value: Any) -> bool:
    """``attr.set(value)`` (one value, no flag) through an ``MDGModifier``, journaled. True
    if it was made; False, with nothing changed, when the caller must take its ``cmds``
    route (`Attribute.set`, which calls this only for an attribute of a fixed kind outside
    any array, `Attribute._is_fixed_kind_outside_array`: today's route makes no type query
    there). The guards:

    * ``value`` an int, a float or a bool, and ``attr`` exactly ``Attribute`` or ``Plug``,
      with a node object named by rig's own naming (`_rig_named`);
    * its node valid (``_handle.isValid()``), its dynamic attribute alive;
    * the plug no array root, no destination, free to change (``isFreeToChange(True,
      True)``: not locked, nor a locked or connected parent); a destination takes the
      cmds route, whose disconnect-and-retry (``<<``, `rig._internal.plug._do_set`)
      retries here;
    * its name resolves to its own MPlug (`_names_own_plug`);
    * the kind and the value pass CR-4's table (`_set_plan`);
    * rig's undo command ready (`rig.nodetypes.plugins._fast_path_ready`).

    A set that would be its own undo step either way takes cmds, which is as fast without
    the journal's registration: one made outside any statement (``attr.set(1.0)`` called
    directly), and the one set of a statement that is just this set (``attr << 1.0`` at the
    top level: the outermost statement's operands are ``attr`` and ``value``, and nothing is
    journaled yet).

    Locks, keyable and channelBox flags stay on cmds (plan_D H22)."""
    tv = type(value)
    if tv is not float and tv is not int and tv is not bool:
        return False
    state = journal._S
    if not state.open:
        return False
    if not state.pending:
        entry = state.entry
        if entry is not None and len(entry) == 2 and entry[0] is attr and entry[1] is value:
            return False
    if not _BOUND:
        _bind()
    if type(attr) not in _API_CLASSES or not journal.API_EDITS:
        return False
    d     = attr.__dict__
    owner = d.get("_node")
    if owner is None:
        return False
    handle = d.get("_handle")
    if handle is None or not handle.isValid():
        return False
    handle = d.get("_attrh")
    if handle is not None and not handle.isAlive():
        return False
    mplug = d.get("_mplug")
    if mplug is None:
        return False
    try:
        if (
            mplug.isArray
            or mplug.isDestination
            or mplug.isFreeToChange(True, True) != _FREE
            or not _rig_named(owner)
            or not _names_own_plug(attr)
        ):
            return False
        plan = _set_plan(_attr_mobject(attr), value, tv)
    except Exception:  # noqa: BLE001 (a plug the guards cannot read: the cmds route)
        return False
    if plan is None or not _ready():
        return False
    modifier = _MDGModifier()
    try:
        getattr(modifier, plan[0])(mplug, plan[1])
    except Exception:  # noqa: BLE001 (nothing done: the cmds route)
        return False
    return _apply(modifier)
