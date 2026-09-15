import inspect
from collections.abc import Iterable

import maya.cmds as cmds
from rig.maya.nodetypes._base import PyNode

# -------------------------- MAYA COMMAND WRAPPERS --------------------------- #
"""
Wraps every callable command into a function that will return the output as PyNodes
"""


def is_sequence(obj):
    """tests if given object is a 'classic' sequence"""
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict))


def _pynodify(obj):
    """recursively wraps any maya objects in pynodes"""
    if is_sequence(obj):
        return [_pynodify(x) for x in obj]

    if obj and cmds.objExists(obj):
        return PyNode(obj)
    return obj


# wrap anything if not a __builtins__
for name, _ in inspect.getmembers(cmds, callable):
    if name not in dir(__builtins__):
        code = """def {0}(*args, **kargs):

            # run the command and wrap the output with PyNode
            return _pynodify(cmds.{0}(*args, **kargs))

        """

        exec(code.format(name))