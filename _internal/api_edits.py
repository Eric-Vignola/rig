"""
The API fast paths of rig's edits (round 5, phase D): a connect (`api_connect`) made
through an ``MDGModifier`` and journaled (`rig._internal.journal.add`), so that a
statement's API edits are one undo step.

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
    global _names_own_plug, _ready

    from rig._internal.plug import Plug
    from rig.nodetypes import _base
    from rig.nodetypes.dag_node import DAGNode
    from rig.nodetypes.dg_node import DGNode
    from rig.nodetypes.mesh import Mesh
    from rig.nodetypes.plugins import _fast_path_ready as _ready

    _names_own_plug = _base._names_own_plug
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
        modifier.doIt()
    except Exception:  # noqa: BLE001 (the cmds route raises today's error)
        modifier.undoIt()
        return False
    try:
        _add(modifier)
    except Exception:  # noqa: BLE001 (registered nothing: the cmds route makes the edit)
        modifier.undoIt()
        return False
    return True
