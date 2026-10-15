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
pending edits and stay raw (the lint test ``test_r5_journal.TestBarrierLint`` holds the list).
Code a user wrote that rig calls inside a statement (a ``_create`` / ``post_create`` override,
a spec subclass's ``apply``, a fallback hook, an iterator operand ...) runs through
`user_call` / `outside`: the pending list is registered first and the scope is closed while it
runs, so its own ``cmds`` edits and rig statements are recorded in the order it makes them.

An API edit made outside any statement (``attr.set(1.0)`` called directly) is registered at
once, as its own step: a missed entry point costs registrations, never order.

Undo off: registering through the command with undo off keeps nothing (Maya runs the
command's ``doIt``, which does nothing, and drops it). Every change of the undo state is a
``cmds`` edit (a barrier) or happens outside a statement, so a pending list never holds edits
made under another undo state, and nothing reads ``undoInfo -q -state``.

A leaf module: ``maya.cmds``, ``maya.mel`` and `rig._internal.callbacks`; rig's plug-in is
reached when a journal is registered.
"""

from __future__ import annotations

import functools
import inspect
import sys
from typing import Any, Callable

from maya import cmds as _cmds
from maya import mel as _mel

# The API fast paths (`rig._internal.api_edits`: the connects) read it at every edit: False
# sends every edit through cmds (the scopes, barriers and call-outs stay, and change nothing
# then), for A/B runs and for the tests that pin the cmds route.
API_EDITS = True


class _State:
    """The scope flag and the pending list (Maya runs rig on its main thread only)."""

    __slots__ = ("open", "pending")

    def __init__(self) -> None:
        self.open    = False
        self.pending = []


_S = _State()


def _walk(ops: Any, method: str) -> None:
    """Calls ``method`` on every op of ``ops`` in order, also after one raises; then
    re-raises the first error, so that a journal is never half undone or redone silently."""
    first = None
    for op in ops:
        try:
            getattr(op, method)()
        except Exception as error:  # noqa: BLE001 (re-raised below)
            if first is None:
                first = error
    if first is not None:
        raise first


class _Journal:
    """One undo step: applied edits (``MDGModifier`` or any object with ``doIt`` / ``undoIt``),
    in the order they were made. Its ``doIt`` does nothing: the edits are applied already."""

    __slots__ = ("ops",)

    # what a failed load of rig's plug-in names (`rig.nodetypes.plugins._run_undoable`)
    UNDO_EDIT = "the API edits of a statement"

    def __init__(self, ops: list) -> None:
        self.ops = ops

    def doIt(self) -> None:  # noqa: N802 (the command's object protocol)
        pass

    def undoIt(self) -> None:  # noqa: N802
        _walk(reversed(self.ops), "undoIt")

    def redoIt(self) -> None:  # noqa: N802
        _walk(self.ops, "doIt")


def _register(ops: list) -> None:
    """Puts ``ops`` on the undo queue as one step, through `rig.nodetypes.plugins._run_undoable`
    (its barrier finds nothing pending: `_flush` takes the list first). Raises (the edits are
    left applied; the caller reverts them) when rig's command cannot be had (a forced unload)."""
    from rig.nodetypes.plugins import _run_undoable

    _run_undoable(_Journal(ops))


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
        raise RuntimeError(
            f"rig: {len(ops)} edit(s) of this statement could not be put on the undo queue "
            f"({error}); they were reverted"
        ) from None


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
    ``op`` and takes its ``cmds`` route)."""
    if _S.open:
        if not _SCENE_CALLBACKS_READY:
            # an import before Maya was initialised: the new / open discard, before a list
            # it would drop exists
            _ensure_scene_callbacks()
        _S.pending.append(op)
    else:
        _register([op])


def run_statement(func: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """``func(*args, **kwargs)`` as a statement: the outermost call opens the scope, and
    registers what it journaled when ``func`` returns or raises (the exception propagates
    unchanged; a registration that fails then is a note of it, see `barrier`)."""
    if _S.open:
        return func(*args, **kwargs)
    _S.open = True
    try:
        result = func(*args, **kwargs)
    except BaseException as error:
        _S.open = False
        barrier(error)
        raise
    _S.open = False
    if _S.pending:
        _flush()
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
            if _S.pending:
                _flush()
            _S.open = False

    def __exit__(self, *exc: Any) -> bool:
        if self.was_open:
            _S.open = True
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


class _Edits:
    """``cmds_edit.<command>(...)``: the ``maya.cmds`` command, called once the pending list
    is registered. The command is read from ``maya.cmds`` on every call (a plug-in's command
    after a reload, a test's ``mock.patch.object(cmds, ...)``)."""

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)

        def command(*args: Any, **kwargs: Any) -> Any:
            if _S.pending:
                _flush()
            if kwargs:
                return getattr(_cmds, name)(*args, **kwargs)
            # a maya.cmds function given an empty keyword dict costs about 1 us more
            return getattr(_cmds, name)(*args)

        command.__name__ = command.__qualname__ = name
        setattr(self, name, command)
        return command


cmds_edit = _Edits()


def mel_eval(script: str) -> Any:
    """``maya.mel.eval(script)`` once the pending list is registered."""
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
# statement can get here. Registered once per Maya session through rig._internal.callbacks
# (a re-import replaces them; an import before Maya is initialised registers at the first
# journal).
# --------------------------------------------------------------------- #


def _discard(*_args: Any) -> None:
    """kBeforeNew / kBeforeOpen: drop the pending list (see above)."""
    _S.pending = []


def _scene_callback_specs() -> list:
    from maya.api import OpenMaya

    msg = OpenMaya.MSceneMessage
    return [
        (msg.addCallback, msg.kBeforeNew,  _discard),
        (msg.addCallback, msg.kBeforeOpen, _discard),
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
