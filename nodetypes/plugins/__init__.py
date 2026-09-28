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
"""

import os
from contextlib import contextmanager
from typing import List, Optional, Union

import maya.cmds as cmds

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
    elif cmds.pluginInfo(name, query=True, loaded=True):
        return []
    path   = bundled_plugin_path(name)
    loaded = cmds.loadPlugin(path or name, quiet=quiet) or []  # by name: MAYA_PLUG_IN_PATH
    if command is not None and not hasattr(cmds, command):
        # Maya prints an initializePlugin error and returns: loadPlugin does not raise
        raise RuntimeError(
            f"rig's plug-in {path} did not register {command}: see Maya's error above, "
            f"or unload the other plug-in named {name!r} first"
        )
    return loaded


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

    # load plugins, keeping the names this call loaded
    loaded = []
    for each in plugins:
        loaded.extend(_ensure_loaded(each, quiet))

    try:
        yield
    finally:
        # unload what this call loaded, if requested (unloadPlugin has no quiet flag)
        if unload_on_exit:
            for name in reversed(loaded):
                try:
                    cmds.unloadPlugin(name)
                except Exception as error:  # noqa: BLE001 (a refusal is reported, not raised)
                    cmds.warning(f"load_plugin: could not unload {name}: {str(error).strip()}")
                else:
                    if cmds.pluginInfo(name, query=True, loaded=True):
                        cmds.warning(f"load_plugin: could not unload {name}: it is still in use")
