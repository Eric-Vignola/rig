"""Plugin loading for ``rig``, plus the plugins it ships with.

:func:`load_plugin` is a context manager that makes sure a plugin is loaded
before the block runs. It first asks Maya to load the plugin by name, which
searches ``MAYA_PLUG_IN_PATH``. If Maya cannot find it there, the plugin is
looked for in this directory and loaded by full path instead -- so a plugin
that ships inside ``rig`` (``undoable_api_command.py``) needs no environment
setup. A name that is neither on the path nor bundled raises the original
Maya error unchanged.
"""

import os
from contextlib import contextmanager
from typing import List, Optional, Union

import maya.cmds as cmds

_BUNDLED_DIR       = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_EXTENSIONS = (".py", ".mll", ".so", ".bundle")


def bundled_plugin_path(name: str) -> Optional[str]:
    """Full path of the plugin ``name`` shipped in this package, or ``None``."""
    for ext in _PLUGIN_EXTENSIONS:
        path = os.path.join(_BUNDLED_DIR, name + ext)
        if os.path.isfile(path):
            return path
    return None


def _ensure_loaded(name: str, quiet: bool) -> None:
    if cmds.pluginInfo(name, query=True, loaded=True):
        return
    try:
        cmds.loadPlugin(name, quiet=quiet)          # MAYA_PLUG_IN_PATH lookup
        return
    except RuntimeError:
        bundled = bundled_plugin_path(name)
        if bundled is None:
            raise                                   # not ours; keep Maya's message
    cmds.loadPlugin(bundled, quiet=quiet)           # Maya names it after the file stem


@contextmanager
def load_plugin(
    plugins:        Union[str, List[str]],
    quiet:          bool                  = True,
    unload_on_exit: bool                  = False,
):
    """
    A context manager that ensures a given plugin(s) is loaded.
    """
    if isinstance(plugins, str):
        plugins = [plugins]

    # load plugins
    for each in plugins:
        _ensure_loaded(each, quiet)

    try:
        yield
    finally:
        # unload plugins if requested
        if unload_on_exit:
            for each in plugins:
                if cmds.pluginInfo(each, query=True, loaded=True):
                    cmds.unloadPlugin(each, quiet=quiet)
