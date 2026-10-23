"""
The statement journal: rig's API edits of one statement become ONE undo step.

A *statement* is one call from user code into rig: ``a.tx << b.ty``, ``(a.tx + b.tx) * 0.5``,
``rf.clamp(x, 0, 1)``, ``rn.multiplyDivide(input1X=2)``, ``Transform.create(tx=1)`` ... Its
entry points are decorated with `statement` (or call `run_statement`): the outermost one opens
the scope, a nested one only tests a flag. While the scope is open, every API edit rig makes
(an ``MDGModifier`` applied already, see `add`) joins the *pending* list; when the outermost
entry returns or raises, the pending list goes on Maya's undo queue once, through rig's undo
command (``rigUndoableAPICommand``): undo walks it backwards, redo forwards.

Order with ``cmds``: Maya records a ``cmds`` edit at once, so rig registers the pending list
before it (a *barrier*). rig's own ``cmds`` edits go through `cmds_edit`
(``cmds_edit.setAttr(...)``) and `mel_eval`, which register first. ``createNode``,
``loadPlugin`` and the member add of a node a container scope just made commute with the
pending edits and stay raw (the lint test ``test_r5_journal.TestBarrierLint`` holds the list);
that member add is deferred to the statement's end and other flush points (`flush_members`).
Code a user wrote that rig calls inside a statement (a ``_create`` / ``post_create`` override,
a spec subclass's ``apply``, a fallback hook, an iterator operand ...) runs through
`user_call` / `outside`: the pending list is registered first and the scope is closed while it
runs, so its own ``cmds`` edits and rig statements are recorded in the order it makes them.

An API edit made outside any statement (``attr.set(1.0)`` called directly) is registered at
once, as its own step: a missed entry point costs registrations, never order.

The redo queue: Maya flushes it at the first command that goes on the undo queue, so a ``cmds``
edit flushes it at once, and a journaled edit only when its journal is registered. After an
undo or a redo (an MEventMessage), while Maya's redo queue is not empty, the next API edit is
registered at once, so the redo queue is flushed where the ``cmds`` edit would have flushed it
(a node made, then undone, is freed at the same point, and a statement that still names it
raises the same error).

Undo off: registering through the command with undo off keeps nothing (Maya runs the
command's ``doIt``, which does nothing, and drops it). Every change of the undo state is a
``cmds`` edit (a barrier) or happens outside a statement, so a pending list never holds edits
made under another undo state, and nothing reads ``undoInfo -q -state``.

A user's callback that deletes a node or unloads a reference inside a statement registers the
pending list first (`_before_removal`), and a journal's undo or redo passes over an
``MDGModifier`` the scene no longer lets it undo or redo, as Maya passes over a ``cmds``
edit's (`_walk`).

A leaf module: ``maya.cmds``, ``maya.mel``, ``OpenMaya`` and `rig._internal.callbacks`; rig's
plug-in is reached when a journal is registered.
"""

from __future__ import annotations

import functools
import inspect
import sys
from typing import Any, Callable

from maya import cmds as _cmds
from maya import mel as _mel
from maya.api.OpenMaya import MDGModifier as _MDGModifier
from maya.api.OpenMaya import MObjectHandle as _MObjectHandle

# The API fast paths (`rig._internal.api_edits`: the connects and the single numeric value
# sets) read it at every edit: False
# sends every edit through cmds (the scopes, barriers and call-outs stay, and change nothing
# then), for A/B runs and for the tests that pin the cmds route.
API_EDITS = True


class _State:
    """The scope flag, the pending list, the outermost statement's positional arguments while
    it runs (``entry``: `rig._internal.api_edits.api_set` reads it, None between statements),
    and ``redo``: True when Maya's redo queue may hold entries (an undo or a redo ran since,
    or since this import), see `add`. Maya runs rig on its main thread only."""

    __slots__ = ("open", "pending", "entry", "redo", "chunk", "members")

    def __init__(self) -> None:
        self.open    = False
        self.pending = []
        self.entry   = None
        self.redo    = True
        # True while the open statement holds an undo chunk (`one_step`)
        self.chunk   = False
        # the deferred member adds: ``[(leaf container, node), ...]`` of the nodes
        # container.createNode made in the open statement (`defer_member`)
        self.members = []


_S = _State()

# S7 prototype: {id(node object): (node object, identity)} for the open statement
IDS: dict = {}


def _walk(ops: Any, method: str) -> None:
    """Calls ``method`` on every op of ``ops`` in order, also after one raises; then
    re-raises the first error, so that a journal is never half undone or redone silently.

    An ``MDGModifier`` that fails is passed over silently, as Maya passes over the undo or
    redo of a ``cmds`` edit the scene no longer has: an edit made with undo recording off
    (``undoInfo -stateWithoutFlush off``) between the journal and its undo can replace the
    connection a modifier would remove (``TconnectAttributeCmd::undoIt`` says nothing then;
    ``MDGModifier.undoIt`` raises ``kFailure``)."""
    first = None
    for op in ops:
        try:
            getattr(op, method)()
        except Exception as error:  # noqa: BLE001 (re-raised below)
            if first is None and type(op) is not _MDGModifier:
                first = error
    if first is not None:
        raise first


class _Journal:
    """One undo step: applied edits (``MDGModifier`` or any object with ``doIt`` / ``undoIt``),
    in the order they were made. Its ``doIt`` does nothing: the edits are applied already."""

    __slots__ = ("ops",)

    # what a failed load of rig's plug-in names (`rig.nodetypes.plugins._run_undoable`)
    UNDO_EDIT = "the API edits of a statement"
    # its undoIt / redoIt make no cmds call (MDGModifier undoIt / doIt only): rig's command
    # calls them without turning undo recording off and on around them
    RECORDS_NO_CMDS = True

    def __init__(self, ops: list) -> None:
        self.ops = ops

    def doIt(self) -> None:  # noqa: N802 (the command's object protocol)
        pass

    def undoIt(self) -> None:  # noqa: N802
        _walk(reversed(self.ops), "undoIt")

    def redoIt(self) -> None:  # noqa: N802
        _walk(self.ops, "doIt")


# rig.nodetypes.plugins, bound on the first `_register` (it imports this module): the
# module, read at each call, so a rebinding or a mock.patch of its function holds, without
# the per-call ``from ... import`` (1.2 us under Maya's import hook, once per statement)
_PLUGINS = None


def _register(ops: list) -> None:
    """Puts ``ops`` on the undo queue as one step, through `rig.nodetypes.plugins._run_undoable`
    (its barrier finds nothing pending: `_flush` takes the list first). Raises (the edits are
    left applied; the caller reverts them) when rig's command cannot be had (a forced unload)."""
    global _PLUGINS
    plugins = _PLUGINS
    if plugins is None:
        from rig.nodetypes import plugins

        _PLUGINS = plugins
    plugins._run_undoable(_Journal(ops))


def _flush() -> None:
    """Registers the pending list (at a barrier, or when the outermost statement ends). When
    that fails, the edits are reverted (never left applied and unregistered) and a
    ``RuntimeError`` says so, without the registration's error chained."""
    ops, _S.pending = _S.pending, []
    try:
        _register(ops)
    except Exception as error:  # noqa: BLE001
        try:
            _walk(reversed(ops), "undoIt")
        except Exception:  # noqa: BLE001 (reverting is all that can be done)
            pass
        failure = RuntimeError(
            f"rig: {len(ops)} edit(s) of this statement could not be put on the undo queue "
            f"({error}); they were reverted"
        )
        failure._rig_reverted = True  # `add`'s callers never pass over it (`api_edits._apply`)
        raise failure from None


def barrier(error: BaseException | None = None) -> None:
    """Registers the pending list now, if there is one: before a ``cmds`` edit, an undo
    chunk's open or close, or a registration of rig's command for another object. With
    ``error``, an exception propagating, a failed registration does not replace it: its
    ``RuntimeError`` (the edits reverted, `_flush`) becomes a note of ``error``."""
    if _S.pending:
        try:
            _flush()
        except RuntimeError as failure:
            if error is None:
                raise
            error.add_note(str(failure))


def add(op: Any) -> None:
    """Journals ``op``, an edit applied just now: pending inside a statement, else registered
    at once as its own step (which raises when it cannot be registered: the caller reverts
    ``op`` and takes its ``cmds`` route). Also registered at once, after what is pending, when
    Maya's redo queue is not empty (after an undo or a redo): the ``cmds`` edit would have
    flushed it here. If what is pending cannot be registered then, `_flush` reverts it and
    its ``RuntimeError`` (marked ``_rig_reverted``) must propagate: the caller reverts
    ``op`` too and re-raises it, never taking its ``cmds`` route past the lost edits."""
    if _S.open:
        if not _SCENE_CALLBACKS_READY:
            # an import before Maya was initialised: the new / open discard, before a list
            # it would drop exists
            _ensure_scene_callbacks()
        if _S.redo:
            if _cmds.undoInfo(query=True, redoQueueEmpty=True):
                _S.redo = False
            else:
                if _S.pending:
                    _flush()
                _register([op])
                return
        _S.pending.append(op)
    else:
        _register([op])


STEP = "rig.replace"


def one_step() -> None:
    """Make the open statement ONE undo step from here on, with what it journaled so far:
    an undo chunk (``rig.replace``) opened now, without a barrier (the pending list is
    registered inside it), and closed when the outermost statement ends. Called before a
    ``cmds`` edit that a statement's other edits follow, so that one undo takes back the
    whole statement: a disconnect, then a set or a connect (`rig._internal.plug._do_set`,
    ``plug << value`` over a driven plug), or the data drop of ``slot << None``. Opening a
    chunk records nothing on the undo queue. Outside a statement, while Maya undoes or
    redoes, or once the statement holds its chunk, it does nothing."""
    if not _S.open or _S.chunk:
        return
    from maya.api.OpenMaya import MGlobal

    if MGlobal.isUndoing() or MGlobal.isRedoing():
        return
    _cmds.undoInfo(openChunk=True, chunkName=STEP)
    _S.chunk = True


def _close_chunk(outer: bool) -> None:
    """Closes the chunk `one_step` opened for the statement ending, and gives back the
    state of the statement it ran inside (`outside`), ``outer``. The deferred member adds
    go inside the chunk first (`flush_members`)."""
    try:
        if _S.chunk:
            try:
                if _S.members:
                    flush_members()
            finally:
                _cmds.undoInfo(closeChunk=True)
    finally:
        _S.chunk = outer


# --------------------------------------------------------------------- #
# The deferred member adds (round 6 S2, user decision 1A): the ``container -addNode`` of a
# node container.createNode made just now waits in ``_S.members`` and is sent once per run
# of consecutive nodes with the same leaf container (`rig._internal.container._flush_members`,
# through S0's `_add_members`), the names read then. FLUSH POINTS: the outermost
# statement's end (normal or raising: an error mid-statement leaves today's membership),
# an undo chunk's close (`undo._Chunk.__exit__`, `_close_chunk`: the membership stays in
# the chunk's one undo step), every rig ``container`` call (`container_cmd`), `outside`
# (before user code), and every `cmds_edit` / `mel_eval` of a command not in
# `NO_MEMBERSHIP`. So the pending nodes were made in this statement, and no user code ran
# since: nothing but rig's own code, which reads membership only through these points,
# runs while one is pending.
# --------------------------------------------------------------------- #

# the commands that never change or read container membership: no flush before them
NO_MEMBERSHIP = frozenset({
    "setAttr", "addAttr", "deleteAttr", "connectAttr", "disconnectAttr", "aliasAttr",
    "parent", "xform",
})

# rig._internal.container, bound on the first flush (it imports this module)
_CONTAINER = None


def defer_member(leaf: Any, node: Any) -> None:
    """Defers the member add of ``node`` (made just now) to the container ``leaf`` until
    the next flush point (see above). Outside a statement it is sent at once."""
    if _S.open:
        _S.members.append((leaf, node))
    else:
        _send_deferred([(leaf, node)])


def _send_deferred(pairs: list) -> None:
    global _CONTAINER
    module = _CONTAINER
    if module is None:
        from rig._internal import container as module

        _CONTAINER = module
    module._flush_members(pairs)


def flush_members(error: BaseException | None = None) -> None:
    """Sends the deferred member adds now (see above). With ``error``, an exception
    propagating, a failure is a note of it, never a replacement (as `barrier`)."""
    pairs, _S.members = _S.members, []
    if not pairs:
        return
    try:
        _send_deferred(pairs)
    except Exception as failure:  # noqa: BLE001 (re-raised or noted)
        if error is None:
            raise
        error.add_note(f"rig: the member add of {len(pairs)} new node(s) failed: {failure}")


def container_cmd(*args: Any, **kwargs: Any) -> Any:
    """rig's one route to ``maya.cmds.container`` (the lint
    ``test_r6_deferred_members.TestContainerRoute``): the deferred member adds are sent
    first, so a query sees them; an edit (no true ``q`` / ``query``) goes through
    `cmds_edit` (the journal's barrier), a query is raw."""
    if _S.members:
        flush_members()
    if kwargs.get("query") or kwargs.get("q"):
        return _cmds.container(*args, **kwargs)
    return cmds_edit.container(*args, **kwargs)


def run_statement(func: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """``func(*args, **kwargs)`` as a statement: the outermost call opens the scope, and
    registers what it journaled when ``func`` returns or raises (the exception propagates
    unchanged; a registration that fails then is a note of it, see `barrier`). A chunk
    `one_step` opened is closed after that registration."""
    if _S.open:
        return func(*args, **kwargs)
    if LIST_TYPE:
        # the List rule: a List argument holding a deleted or freed element raises
        # before anything is built (see `check_lists`)
        check_lists(args, kwargs)
    _S.open  = True
    _S.entry = args
    IDS.clear()
    outer, _S.chunk = _S.chunk, False
    try:
        result = func(*args, **kwargs)
    except BaseException as error:
        _S.open  = False
        _S.entry = None
        IDS.clear()
        try:
            try:
                # the members first: today's membership, never replacing ``error``
                if _S.members:
                    flush_members(error)
            finally:
                barrier(error)
        finally:
            _close_chunk(outer)
        raise
    _S.open  = False
    _S.entry = None
    IDS.clear()
    try:
        try:
            if _S.members:
                flush_members()
        finally:
            if _S.pending:
                _flush()
    finally:
        _close_chunk(outer)
    return result


def statement(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorates a statement entry point (`run_statement`). Refuses a generator or a
    coroutine function: its body runs after the call returns, outside the scope."""
    if inspect.isgeneratorfunction(func) or inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
        raise TypeError(f"journal.statement cannot wrap {func.__qualname__!r}: a generator or a coroutine")

    @functools.wraps(func)
    def entry(*args: Any, **kwargs: Any) -> Any:
        if _S.open:
            return func(*args, **kwargs)
        return run_statement(func, args, kwargs)

    entry._rig_statement = True
    return entry


class outside:
    """``with outside():`` runs its block outside the open statement, if one is open: the
    pending list is registered first and the scope is closed until the block ends, so code
    a user wrote records its ``cmds`` edits, and makes its rig statements, in its own order."""

    __slots__ = ("was_open",)

    def __enter__(self) -> None:
        self.was_open = _S.open
        if self.was_open:
            # the deferred member adds before user code (it may read membership)
            if _S.members:
                flush_members()
            if _S.pending:
                _flush()
            _S.open = False
            IDS.clear()

    def __exit__(self, *exc: Any) -> bool:
        if self.was_open:
            _S.open = True
        return False


class cmds_route:
    """``with cmds_route():`` sends rig's edits in its block through ``cmds`` (`API_EDITS`
    off, then back as it was): for a rig site that swallows a failed edit, where the API
    route's ``MDGModifier`` would print Maya's ``Error:`` line for a call that then
    succeeds (`rig._internal.container`'s external output connect)."""

    __slots__ = ("was",)

    def __enter__(self) -> None:
        global API_EDITS
        self.was, API_EDITS = API_EDITS, False

    def __exit__(self, *exc: Any) -> bool:
        global API_EDITS
        API_EDITS = self.was
        return False


# {module name: True if it is rig's own}: names only, so no user object is kept alive
_RIG_MODULES: dict = {}


def _in_rig(module: str) -> bool:
    return (module == "rig" or module.startswith("rig.")) and not module.startswith(_NOT_RIG_CODE)


# the packages of rig the barrier lint does not read: their code is a user's
_NOT_RIG_CODE = ("rig.examples", "rig._tests")


def rig_code(fn: Any) -> bool:
    """True if ``fn`` (a function, a method, a class) is defined in rig's own package, not in
    ``rig.examples`` or ``rig._tests``: code the barrier rule and its lint cover. Answered by
    its ``__module__`` (a method reads its function's), cached per module name."""
    module = getattr(fn, "__module__", None) or ""
    try:
        return _RIG_MODULES[module]
    except KeyError:
        known = _RIG_MODULES[module] = _in_rig(module)
        return known


def user_call(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)``, outside the open statement unless ``fn`` is rig's own."""
    if not _S.open or rig_code(fn):
        return fn(*args, **kwargs)
    with outside():
        return fn(*args, **kwargs)


# The List rule (`rig._internal.list`, user decision 6C 2026-10-17): every rig operation on a
# List that holds a deleted or freed element raises, while the List never refuses Python's
# ``len()``, indexing or iteration (Maya's argument readers go through them). So rig checks
# the Lists it is given where it is given them: the outermost statement's arguments
# (`run_statement`), every ``cmds_edit`` call, rig's command wrappers and queries. Filled
# by rig._internal.list when it is imported (it imports this module): ``[the List types,
# the check]``; kept across an ``importlib.reload`` of this module (the module dict is
# reused), so the Lists of the import that filled it stay checked.
LIST_TYPE: list = globals().get("LIST_TYPE") or []


def check_lists(args: tuple, kwargs: Any = None) -> None:
    """The List rule's check of every List among ``args`` and the values of ``kwargs``, and
    among the elements of a plain list or tuple given there (one level: ``[L1, L2]``): a List
    holding a deleted or freed element raises its RuntimeError. Reads nothing but the List
    types, unless a List is found (``LIST_TYPE`` filled)."""
    types, check = LIST_TYPE
    for a in args:
        t = type(a)
        if t in types:
            check(a)
        elif (t is list or t is tuple) and not types.isdisjoint(map(type, a)):
            for b in a:
                if type(b) in types:
                    check(b)
    if kwargs:
        check_lists(tuple(kwargs.values()))


class _Edits:
    """``cmds_edit.<command>(...)``: the ``maya.cmds`` command, called once the pending list
    is registered. The command is read from ``maya.cmds`` on every call (a plug-in's command
    after a reload, a test's ``mock.patch.object(cmds, ...)``)."""

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)

        # a command that can change or read membership sends the deferred member adds first
        flushes = name not in NO_MEMBERSHIP
        renames = name == "rename"

        def command(*args: Any, **kwargs: Any) -> Any:
            if LIST_TYPE:
                check_lists(args, kwargs)
            if flushes and _S.members:
                flush_members()
            if _S.pending:
                _flush()
            if renames:
                IDS.clear()
            if kwargs:
                return getattr(_cmds, name)(*args, **kwargs)
            # a maya.cmds function given an empty keyword dict costs about 1 us more
            return getattr(_cmds, name)(*args)

        command.__name__ = command.__qualname__ = name
        setattr(self, name, command)
        return command


cmds_edit = _Edits()


def mel_eval(script: str) -> Any:
    """``maya.mel.eval(script)`` once the deferred member adds are sent and the pending list
    is registered."""
    if _S.members:
        flush_members()
    if _S.pending:
        _flush()
    return _mel.eval(script)


# --------------------------------------------------------------------- #
#  Scene callbacks
# --------------------------------------------------------------------- #
#
# kBeforeNew / kBeforeOpen drop a pending list: it belongs to the scene going away. The list
# is empty whenever control is outside a statement, and rig itself opens no file inside one
# (`cmds_edit.file` registers first), so only a user's MMessage callback run inside a
# statement can get here. A node removal and a reference unload / remove / load / import
# register a pending list first (`_before_removal`, the same case). Registered once per
# Maya session through rig._internal.callbacks (a re-import replaces them; an import before
# Maya is initialised registers at the first journal).
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# The scene epoch (the List rule, `rig._internal.list`): a List that found every element it
# holds in the scene keeps the epoch it found them at, and is checked again only once the
# epoch moved. ``EPOCH[0]`` moves when a node is about to be removed (Maya calls the
# node-removed callback BEFORE the node goes: it is still valid there), when a scene is
# about to be replaced or a reference unloaded / removed, and once such a change is over
# (the Undo / Redo events, kAfterNew / kAfterOpen / kAfterUnloadReference /
# kAfterRemoveReference). ``DOOMED`` holds the handles of the nodes announced removed:
# while one of them is still valid the removal is in progress (a user's node-removed
# callback runs before rig's or after it), and a List found valid then keeps no epoch (it
# would be stale the moment the node goes). Settled (every doomed handle invalid, or one of
# the "over" events) it is emptied. Nothing else is kept: a List holds no registration, and
# a node is never told which Lists hold it.
# shared by every import of rig in this session (kept on ``sys``, as the callback registry
# is): a purge of ``sys.modules['rig*']`` or an ``importlib.reload`` removes the callbacks
# of the copy it replaces, and a List of that copy must keep seeing the epoch move
EPOCH: list = sys.__dict__.setdefault("_rig_scene_epoch", [0])
DOOMED: list = sys.__dict__.setdefault("_rig_scene_doomed", [])

# a doomed list this long is pruned of the handles already invalid (a long build that
# deletes many nodes, and no List checked meanwhile)
_DOOMED_PRUNE = 256


def _settle(*_args: Any) -> None:
    """A removal is over (the Undo / Redo events, a scene replaced, a reference unloaded or
    removed): the epoch moves and the doomed handles are dropped."""
    EPOCH[0] += 1
    del DOOMED[:]


def _discard(*_args: Any) -> None:
    """kBeforeNew / kBeforeOpen: the scene epoch moves (first, see `_node_removed`); drop the
    pending list and the deferred member adds (see above)."""
    EPOCH[0] += 1
    _S.pending = []
    _S.members = []


def _undone_or_redone(*_args: Any) -> None:
    """The Undo / Redo events: Maya's redo queue may hold entries now (see `add`); a removal
    an undo or a redo made is over (the scene epoch)."""
    EPOCH[0] += 1
    del DOOMED[:]
    _S.redo = True
    IDS.clear()


class _Untrusted:
    """A doomed entry for a removal whose node handle could not be recorded (Maya called the
    node-removed callback with a Python error left pending: a user's object whose ``str()``
    raised inside Maya's argument parser, ``cmds.delete([obj, node])``): it stays "valid",
    so no List keeps an epoch until the next undo, redo or scene event empties the doomed
    list (`_settle`). A slowdown only: every List use checks its elements meanwhile."""

    __slots__ = ()

    @staticmethod
    def isValid() -> bool:  # noqa: N802 (MObjectHandle's spelling)
        return True


_UNTRUSTED = _Untrusted()


def _node_removed(node: Any, *_args: Any) -> None:
    """A node is about to be removed: the scene epoch moves FIRST (nothing before it can
    fail: a List must hear of every removal), then the node's handle is doomed (see above;
    `_UNTRUSTED` when it cannot be recorded), then `_before_removal`'s registration."""
    EPOCH[0] += 1
    try:
        DOOMED.append(_MObjectHandle(node))
    except BaseException:  # noqa: BLE001 (a pending error: no List is trusted, see _Untrusted)
        DOOMED.append(_UNTRUSTED)
    if len(DOOMED) > _DOOMED_PRUNE:
        DOOMED[:] = [h for h in DOOMED if h.isValid()]
    if _S.pending:
        barrier()


def _before_removal(*_args: Any) -> None:
    """A node is about to be removed, or a reference unloaded, removed, loaded or imported,
    while a statement has edits pending: only a user's callback run inside the statement
    gets here (rig's own deletes and file edits are ``cmds_edit`` barriers). The pending
    list is registered first, as the ``cmds`` route would have recorded it: the removal
    goes on the undo queue after the edits made before it (an undo then never meets a
    journal whose node is deleted to the undo queue or freed by the unload)."""
    if _S.pending:
        barrier()


def _add_node_removed(node_type: str, callback: Callable[..., Any]) -> int:
    from maya.api import OpenMaya

    return OpenMaya.MDGMessage.addNodeRemovedCallback(callback, node_type)


def _scene_callback_specs() -> list:
    from maya.api import OpenMaya

    msg   = OpenMaya.MSceneMessage
    event = OpenMaya.MEventMessage.addEventCallback
    return [
        (msg.addCallback,   msg.kBeforeNew,               _discard),
        (msg.addCallback,   msg.kBeforeOpen,              _discard),
        (event,             "Undo",                       _undone_or_redone),
        (event,             "Redo",                       _undone_or_redone),
        (_add_node_removed, "dependNode",                 _node_removed),
        (msg.addCallback,   msg.kBeforeUnloadReference,   _before_removal),
        (msg.addCallback,   msg.kBeforeRemoveReference,   _before_removal),
        (msg.addCallback,   msg.kAfterNew,                _settle),
        (msg.addCallback,   msg.kAfterOpen,               _settle),
        (msg.addCallback,   msg.kAfterUnloadReference,    _settle),
        (msg.addCallback,   msg.kAfterRemoveReference,    _settle),
        (msg.addCallback,   msg.kBeforeLoadReference,     _before_removal),
        (msg.addCallback,   msg.kBeforeImportReference,   _before_removal),
    ]


def _register_scene_callbacks() -> bool:
    """Registers this module's scene callbacks in place of an earlier import's. False, with
    nothing registered, while Maya is not initialised."""
    global _SCENE_CALLBACKS_READY
    from rig._internal import callbacks

    _SCENE_CALLBACKS_READY = callbacks.register(__name__, _THIS_MODULE, _scene_callback_specs())
    return _SCENE_CALLBACKS_READY


def _ensure_scene_callbacks() -> bool:
    """Makes the registration an import before Maya was initialised left pending."""
    global _SCENE_CALLBACKS_READY
    from rig._internal import callbacks

    _SCENE_CALLBACKS_READY = callbacks.ensure(__name__, _THIS_MODULE, _scene_callback_specs())
    return _SCENE_CALLBACKS_READY


_THIS_MODULE           = sys.modules.get(__name__)
_SCENE_CALLBACKS_READY = False
_register_scene_callbacks()
