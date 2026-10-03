"""Plugin loading for ``rig``, plus the plugins it ships with.

This directory holds rig's plug-in ``undoable_api_command.py``. It registers
``rigUndoableAPICommand``, which puts a Python object's API edits on Maya's
undo queue (see that file). A plugin that ships here is always loaded by full
path, never searched for by name first, so it needs no environment setup.

:func:`load_plugin` is a context manager that makes sure a plugin is loaded
before the block runs: a plugin bundled here by full path, any other name by
name, which searches ``MAYA_PLUG_IN_PATH``. A name that is neither bundled nor
on the path raises the original Maya error unchanged. Whether the undoable API
command is loaded is read from ``maya.cmds`` (about 0.1 us, where a
``pluginInfo`` query costs about 20 us), so it is loaded once per session, and
again only after an unload.

rig's own API edits go through the command: `_run_undoable` (an object with
``doIt`` / ``undoIt`` / ``redoIt``) and `_register_api_edit` (an ``MDGModifier``;
`_api_edit_ready` says whether it may run, else the caller takes its ``cmds``
route), each one unnamed undo step.
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
    path   = bundled_plugin_path(name)
    loaded = cmds.loadPlugin(path or name, quiet=quiet) or []  # by name: MAYA_PLUG_IN_PATH
    if command is not None and not hasattr(cmds, command):
        # Maya prints an initializePlugin error and returns: loadPlugin does not raise
        raise RuntimeError(
            f"rig's plug-in {path} did not register {command} (see Maya's error above): "
            + _RECOVERY.format(name=name)
        )
    return loaded


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
    (`rig._internal.journal`), so ``obj``'s step comes after them."""
    barrier()
    command = getattr(cmds, UNDOABLE_API_COMMAND, None)
    if command is None:
        _ensure_loaded("undoable_api_command", True)
        command = getattr(cmds, UNDOABLE_API_COMMAND)
    return command.run(obj)


class _APIEdit:
    """[Internal] An ``MDGModifier`` as the object `_run_undoable` takes: ``doIt`` and
    ``redoIt`` run the modifier's ``doIt``, ``undoIt`` its ``undoIt``."""

    __slots__ = ("m",)

    def __init__(self, modifier):
        self.m = modifier

    def doIt(self):
        self.m.doIt()

    def undoIt(self):
        self.m.undoIt()

    def redoIt(self):
        self.m.doIt()


def _api_edit_ready() -> bool:
    """[Internal] True when `_register_api_edit` may run now: ``rigUndoableAPICommand``
    is in ``maya.cmds`` (rig's plug-in is loaded the first time through `_ensure_loaded`,
    and again after an unload), and Maya is neither undoing nor redoing. False, and the
    caller takes its ``cmds`` route, when the plug-in does not load (a load that registers
    nothing, the state a forced unload leaves) or during an undo or a redo. About 0.3 us
    once loaded.

    The undo state is not read (``undoInfo -q -state`` costs about 6 us): with undo off,
    the command applies the edit and Maya keeps nothing, neither a step nor the modifier,
    which is what a ``cmds`` edit leaves too."""
    if getattr(cmds, UNDOABLE_API_COMMAND, None) is None:
        try:
            _ensure_loaded("undoable_api_command", True)
        except Exception:  # noqa: BLE001 (any load failure: the caller's cmds route)
            return False
    return not (OpenMaya.MGlobal.isUndoing() or OpenMaya.MGlobal.isRedoing())


def _register_api_edit(modifier) -> None:
    """[Internal] Applies ``modifier`` (an ``MDGModifier`` holding the edit, not done
    yet) inside ``rigUndoableAPICommand`` (`_run_undoable`): one unnamed undo step whose
    undo is the modifier's ``undoIt`` and whose redo its ``doIt``; inside an undo chunk,
    the step joins the chunk. The caller has checked `_api_edit_ready`. The
    registration adds about 8 us to the modifier's own ``doIt`` in a wired scene, and
    about 18 us inside a rail_spine build.

    An exception out of the modifier's ``doIt`` (or out of the command) registers
    nothing and propagates; the caller then runs the modifier's ``undoIt`` (what ``doIt``
    did before it failed) and takes its ``cmds`` route outside its ``except`` block, so
    no API traceback is chained onto an error of that route."""
    _run_undoable(_APIEdit(modifier))
