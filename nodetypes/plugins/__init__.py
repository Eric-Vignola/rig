"""Plugin loading for ``rig``, plus the plugins it ships with.

This directory holds rig's plug-in ``undoable_api_command.py``. It registers
``rigUndoableAPICommand``, which puts a Python object's API edits on Maya's
undo queue (see that file). A plugin that ships here is always loaded by full
path, never searched for by name first, so it needs no environment setup to be
found. Loading it from a folder that is not on ``MAYA_PLUG_IN_PATH`` costs about
160-200 ms (Maya's plug-in path check), against about 7 ms from a folder on it;
rig pays that once per session, at its first API edit.

:func:`load_plugin` is a context manager that makes sure a plugin is loaded
before the block runs: a plugin bundled here by full path, any other name by
name, which searches ``MAYA_PLUG_IN_PATH``. A name that is neither bundled nor
on the path raises the original Maya error unchanged. Whether the undoable API
command is loaded is read from ``maya.cmds`` (about 0.1 us, where a
``pluginInfo`` query costs about 20 us), so it is loaded once per session, and
again only after an unload.

rig's own API edits go through the command: `_run_undoable` (an object with
``doIt`` / ``undoIt`` / ``redoIt``, one unnamed undo step), and the journals of
the API fast paths (`rig._internal.api_edits`: the connects and the single
numeric value sets of a build, each an ``MDGModifier`` applied by
``api_edits._apply`` and journaled by ``journal.add``: one undo step per
statement, see `rig._internal.journal`; `_fast_path_ready` says whether they may
run, else the caller takes its ``cmds`` route). rig loads its plug-in at the
first such edit of a session.

If rig's plug-in cannot be loaded (neither from this folder nor by name), the
edits that have no other undoable route (`_run_undoable`: the mesh point, UV and
colour set edits, skin weights, an attribute's default or hidden flag
re-declared) raise a clear ``RuntimeError``; the connects and value sets take
their ``cmds`` route, with one warning per session that builds run slower. When
Maya refuses to initialise the file, the fast paths do not try it again in that
session (each try prints Maya's error again).

While a build's journals are on the undo queue, Maya refuses
``unloadPlugin('undoable_api_command')``: ``cmds.flushUndo()`` first. An unload
forced anyway leaves those undo entries inert (undoing them does nothing).
"""

import os
from contextlib import contextmanager
from typing import List, Optional, Union

import maya.cmds as cmds
from maya.api import OpenMaya
from rig._internal.journal import barrier, cmds_edit

_BUNDLED_DIR       = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_EXTENSIONS = (".py", ".mll", ".so", ".bundle")

# the command undoable_api_command.py registers (its COMMAND_NAME)
UNDOABLE_API_COMMAND = "rigUndoableAPICommand"
# rig's bundled plug-ins that register a command: the command in maya.cmds is the
# loaded check (a same-named plug-in loaded from elsewhere does not pass it)
_BUNDLED_COMMANDS = {"undoable_api_command": UNDOABLE_API_COMMAND}


def bundled_plugin_path(name: str) -> Optional[str]:
    """Full path of the plugin ``name`` shipped in this package, or ``None``."""
    for ext in _PLUGIN_EXTENSIONS:
        path = os.path.join(_BUNDLED_DIR, name + ext)
        if os.path.isfile(path):
            return path
    return None


def _ensure_loaded(name: str, quiet: bool) -> List[str]:
    """Loads the plugin ``name`` unless it is loaded already; returns the names
    of the plugins this call loaded (what ``loadPlugin`` returned), empty when
    it was loaded already."""
    command = _BUNDLED_COMMANDS.get(name)
    if command is not None:
        if hasattr(cmds, command):
            return []
        if cmds.pluginInfo(name, query=True, loaded=True):
            # another plug-in of that name, or rig's after a forced unload (Maya makes
            # no maya.cmds function again): a loadPlugin would change nothing
            raise RuntimeError(
                f"a plug-in named {name!r} is loaded, but maya.cmds has no {command}: "
                + _RECOVERY.format(name=name)
            )
    elif cmds.pluginInfo(name, query=True, loaded=True):
        return []
    path = bundled_plugin_path(name)
    if command is None:
        return cmds.loadPlugin(path or name, quiet=quiet) or []  # by name: MAYA_PLUG_IN_PATH
    errors = []

    def heard(message, kind, *_args):
        if kind == OpenMaya.MCommandMessage.kError:
            errors.append(message)

    listening = OpenMaya.MCommandMessage.addCommandOutputCallback(heard)
    try:
        loaded = cmds.loadPlugin(path or name, quiet=quiet) or []  # by name: MAYA_PLUG_IN_PATH
    finally:
        OpenMaya.MMessage.removeCallback(listening)
    if not hasattr(cmds, command):
        # Maya prints an initializePlugin error and returns: loadPlugin does not raise
        error = RuntimeError(
            f"rig's plug-in {path} did not register {command} (see Maya's error above): "
            + _RECOVERY.format(name=name)
        )
        if not cmds.pluginInfo(name, query=True, loaded=True):
            # Maya refused to initialise the file (it is not loaded): Maya's first error line,
            # for the messages of the fast paths and of `_run_undoable` (`_refusal`)
            lines = [line.strip() for text in errors for line in text.splitlines() if line.strip()]
            error._rig_refused = lines[0] if lines else ""
        raise error
    return loaded


def _refusal(error: BaseException) -> Optional[str]:
    """What Maya said when it refused to initialise rig's plug-in, if ``error`` (from
    `_ensure_loaded`) is that refusal: the file exists, Maya printed its initializePlugin
    error and left the plug-in unloaded. None for any other load failure."""
    reason = getattr(error, "_rig_refused", None)
    if reason is None:
        return None
    return (
        f"Maya refused to initialise it ({reason or 'no error text'}; its error is printed "
        "above), so it is not loaded"
    )


# how to get a bundled plug-in's command back without a restart
_RECOVERY = "run cmds.flushUndo() and cmds.unloadPlugin({name!r}), then try again (or restart Maya)"


@contextmanager
def load_plugin(
    plugins:        Union[str, List[str]],
    quiet:          bool                  = True,
    unload_on_exit: bool                  = False,
):
    """
    A context manager that ensures a given plugin(s) is loaded.

    Args:
        plugins: A plugin name, or a list of them: a plugin bundled in this
            package by full path, any other name from ``MAYA_PLUG_IN_PATH``.
        quiet: Passed to ``loadPlugin``.
        unload_on_exit: When the block ends (returned or raised), unload the
            plugins THIS call loaded, the last one first; a plugin that was
            loaded already stays loaded. Maya refuses to unload a plugin that
            is still in use (its nodes in the scene, its commands on the undo
            queue): a refusal is one ``cmds.warning`` per plugin, never an
            exception out of the block.
    """
    if isinstance(plugins, str):
        plugins = [plugins]

    loaded = []
    try:
        # load plugins, keeping the names this call loaded (unloaded below also when
        # a later name fails to load)
        for each in plugins:
            loaded.extend(_ensure_loaded(each, quiet))
        yield
    finally:
        # unload what this call loaded, if requested (unloadPlugin has no quiet flag)
        if unload_on_exit:
            for name in reversed(loaded):
                try:
                    cmds_edit.unloadPlugin(name)
                except Exception as error:  # noqa: BLE001 (a refusal is reported, not raised)
                    cmds.warning(f"load_plugin: could not unload {name}: {str(error).strip()}")
                else:
                    if cmds.pluginInfo(name, query=True, loaded=True):
                        cmds.warning(f"load_plugin: could not unload {name}: it is still in use")


def _run_undoable(obj):
    """[Internal] Puts the edit of ``obj`` (an object with ``doIt`` / ``undoIt`` /
    ``redoIt``) on the undo queue through ``rigUndoableAPICommand``, without the
    wrapper's undo chunk: one unnamed undo step. The command alone costs about
    4.5 us in a wired scene (round U measured 3.3 us, against about 20 us for the
    chunked ``with load_plugin(...): cmds.rigUndoableAPICommand(obj)`` and about
    38 us for the call sites' former route). Undoing or redoing the step costs
    about 37 us, where a ``cmds`` entry costs about 18 us: the command's
    ``undoIt`` / ``redoIt`` turn undo recording off and on again around ``obj``'s.

    For an ``obj`` whose ``doIt`` calls no ``cmds`` (a ``cmds`` call in ``doIt``
    would be its own undo step); inside a user's or rig's undo chunk, the step
    joins that chunk. Loads rig's plug-in the first time (and again after an
    unload); the command is fetched from ``maya.cmds`` on every call, never
    kept across a reload. Returns what the command returns.

    A journal barrier: the statement's pending API edits are registered first
    (`rig._internal.journal`), so ``obj``'s step comes after them.

    These edits have no other undoable route: when rig's plug-in cannot be loaded,
    a ``RuntimeError`` says that rig needs it for the edit (``obj``'s class names it,
    its ``UNDO_EDIT``), where rig looked for it and Maya's reason, and nothing is
    changed."""
    barrier()
    command = getattr(cmds, UNDOABLE_API_COMMAND, None)
    if command is None:
        try:
            _ensure_loaded("undoable_api_command", True)
        except Exception as error:  # noqa: BLE001 (re-raised, clear)
            edit = getattr(type(obj), "UNDO_EDIT", None) or type(obj).__name__
            raise RuntimeError(
                f"rig needs its undo plug-in 'undoable_api_command' for {edit}, and it "
                f"could not be loaded ({_plugin_whereabouts()}): {_load_failure(error)}"
            ) from None
        command = getattr(cmds, UNDOABLE_API_COMMAND)
    return command.run(obj)


def _load_failure(error: BaseException) -> str:
    """Why rig's plug-in could not be loaded, for the messages of a failed load: Maya's
    refusal and its reason (`_refusal`), else the text of ``error``."""
    return _refusal(error) or str(error).strip()


def _plugin_whereabouts() -> str:
    """Where rig looks for its undo plug-in, for the messages of a failed load: by full
    path in rig's folder, else by name on ``MAYA_PLUG_IN_PATH`` (`_ensure_loaded`)."""
    path = bundled_plugin_path("undoable_api_command")
    if path is not None:
        return f"rig loads it by full path from its own folder: {path}"
    return (
        f"it is not in rig's folder {_BUNDLED_DIR}, so rig looked it up by name on "
        "MAYA_PLUG_IN_PATH"
    )




# the kind epoch (`rig.nodetypes._base._KIND_EPOCH`) in which a load for the fast paths
# failed, `_SESSION` when Maya refused the file (not retried), None when none is kept; and
# whether this session warned of it
_LOAD_FAILED = None
_SESSION     = "session"
_WARNED      = False


def _fast_path_ready() -> bool:
    """[Internal] Whether rig's API fast paths (`rig._internal.api_edits`) may journal an
    edit now; they take their ``cmds`` route on False. True when rig's command is in
    ``maya.cmds`` (rig's plug-in is loaded here the first time, through `_ensure_loaded`,
    and again after an unload), and Maya is neither undoing nor redoing. About 0.3 us once
    loaded.

    A load that fails warns once per session that builds run slower. It is not tried
    again until a plug-in is loaded or unloaded (a new kind epoch: a failed load by name
    can cost a search of ``MAYA_PLUG_IN_PATH``), and never again in the session when Maya
    refused to initialise rig's file (each try would print Maya's initializePlugin error
    again). A plug-in a user loads is found in ``maya.cmds`` either way.

    The undo state is not read (``undoInfo -q -state`` costs about 6 us): with undo off,
    the edit is applied and its registration keeps nothing, neither a step nor the
    modifier, which is what a ``cmds`` edit leaves too."""
    if getattr(cmds, UNDOABLE_API_COMMAND, None) is None and not _load_for_fast_paths():
        return False
    return not (OpenMaya.MGlobal.isUndoing() or OpenMaya.MGlobal.isRedoing())


def _load_for_fast_paths() -> bool:
    global _LOAD_FAILED, _WARNED
    from rig.nodetypes import _base

    failed = _LOAD_FAILED
    if failed is not None and (failed is _SESSION or failed is _base._KIND_EPOCH[0]):
        return False
    try:
        _ensure_loaded("undoable_api_command", True)
    except Exception as error:  # noqa: BLE001 (any load failure: the cmds route)
        if _refusal(error) is not None:
            _LOAD_FAILED = _SESSION
        # kept only while the plug-in callbacks start a new epoch at a load or an unload
        elif _base._PLUGIN_CALLBACKS_READY or _base._ensure_plugin_callbacks():
            _LOAD_FAILED = _base._KIND_EPOCH[0]
        if not _WARNED:
            _WARNED = True
            cmds.warning(
                "rig: its undo plug-in 'undoable_api_command' could not be loaded "
                f"({_plugin_whereabouts()}: {_load_failure(error)}); rig's connections and "
                "value sets take the cmds route while it is missing, so builds run slower"
            )
        return False
    _LOAD_FAILED = None
    return True
