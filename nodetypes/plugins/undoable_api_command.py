"""rig's undoable API command: puts a Python object's API edits on Maya's undo queue.

Maya records an API 2.0 edit (``MFnMesh.setPoints``, ``MFnSkinCluster.setWeights``, ...)
only through a registered command. This plug-in registers one, ``rigUndoableAPICommand``
(a name of rig's own: another studio plug-in registers ``runUndoableAPICommand``),
and replaces it in ``maya.cmds`` with a wrapper that takes the object to run.

* ``cmds.rigUndoableAPICommand(obj)`` runs the command inside an undo chunk named
  ``rigUndoableAPICommand``: one undo step, which ``cmds`` calls made in ``obj.doIt``
  join (Maya undoes them after ``obj.undoIt`` and redoes them before ``obj.redoIt``).
* ``cmds.rigUndoableAPICommand.run(obj)`` is the same call without the chunk: one
  unnamed undo step (about 3 us where the chunk costs about 14 us more), for an
  ``obj`` whose ``doIt`` calls no ``cmds`` (without the chunk, such a call is an
  undo step of its own).
* ``cmds`` calls made in ``obj.undoIt`` / ``obj.redoIt`` are not recorded (recorded,
  they would flush the redo queue).
* The object is handed to the command for that call only: ``doIt`` takes it and clears
  the slot, so the command keeps nothing alive once its entry leaves the queue, and a
  bare call (``rigUndoableAPICommand`` in MEL, or the builtin without an object) raises
  instead of running an earlier object again.
* The command is wrapped once: loading this file again, or importing it as a module,
  never wraps it twice, and unloading the plug-in takes the wrapper out of
  ``maya.cmds``. A wrapper kept from before an unload hands over to the current one
  (its own builtin is gone), or raises when the plug-in is not loaded.
"""

import functools

from maya import cmds
from maya.api import OpenMaya

COMMAND_NAME = "rigUndoableAPICommand"


class UndoableAPICommand(OpenMaya.MPxCommand):
    """
    A Python API 2.0 (OpenMaya) command wrapper with undo & redo support.

    Custom API commands can be triggered using this wrapper command so that they
    don't have to be registered individually. This helps prevent the need to create
    a bunch of random plug-ins for various operations.

    Usage:
    ```python
    from rig.nodetypes.plugins import load_plugin

    # first make a command with doIt(), undoIt(), and redoIt() implemented.
    class MyCommand:

        def __init__(self) -> None:
            with load_plugin("undoable_api_command"):
                cmds.rigUndoableAPICommand(self)

        def doIt(self) -> None:
            print("I'm doing it!")

        def undoIt(self) -> None:
            print("I'm undoing it!")

        def redoIt(self) -> None:
            print("I'm redoing it!")

    # then run this command simply by
    MyCommand()
    ```
    """

    # the object handed over for the call in progress: the wrapper sets it, doIt takes it
    call_class = None

    def __init__(self):
        super(UndoableAPICommand, self).__init__()
        self.py_class = None

    def isUndoable(self):
        return True

    def doIt(self, args):
        cls = type(self)
        py_class, cls.call_class = cls.call_class, None
        if py_class is None:
            raise RuntimeError(
                f"{COMMAND_NAME}: no object to run; call cmds.{COMMAND_NAME}(obj) from Python"
            )
        self.py_class = py_class
        result        = py_class.doIt()
        if result is not None:
            self.setResult(result)

    def redoIt(self):
        self._unrecorded(self.py_class.redoIt)

    def undoIt(self):
        self._unrecorded(self.py_class.undoIt)

    def _unrecorded(self, method):
        """Calls ``method`` with undo recording off, so that the ``cmds`` calls it makes
        are not recorded (a recorded command would flush the redo queue)."""
        state = cmds.undoInfo(query=True, state=True)
        if state:
            cmds.undoInfo(stateWithoutFlush=False)
        try:
            result = method()
            if result is not None:
                self.setResult(result)
        finally:
            if state:
                cmds.undoInfo(stateWithoutFlush=True)

    @classmethod
    def wrap_command(cls):
        """In order to bypass the normal argument required for commands due to trying
        to be compatible with MEL, we have to wrap the created
        cmds.rigUndoableAPICommand so the class can be passed.

        A command wrapped already (a builtin with ``__wrapped__``) is left alone."""
        cmd_func = getattr(cmds, COMMAND_NAME, None)
        if cmd_func is None:
            # registered again after a forced unload: Maya makes no new maya.cmds function
            raise RuntimeError(f"maya.cmds has no {COMMAND_NAME} (after a forced unload, restart Maya)")
        if hasattr(cmd_func, "__wrapped__"):
            return

        def run(py_class):
            current = getattr(cmds, COMMAND_NAME, None)
            if current is not wrapped:
                # a wrapper kept from before an unload: its builtin is gone (calling it
                # crashes Maya), so hand over to the current command, if there is one
                if current is None:
                    raise RuntimeError(f"{COMMAND_NAME}: the plug-in is not loaded")
                return current.run(py_class)
            # hand the object over for this one call; doIt takes it and clears the slot
            cls.call_class = py_class
            try:
                return cmd_func()
            finally:
                cls.call_class = None

        @functools.wraps(cmd_func)
        def wrapped(py_class):
            cmds.undoInfo(openChunk=True, chunkName=COMMAND_NAME)
            try:
                return run(py_class)
            finally:
                cmds.undoInfo(closeChunk=True)

        # now overwrite the one in cmds with this new wrapped version
        wrapped.run = run
        setattr(cmds, COMMAND_NAME, wrapped)


def maya_useNewAPI():
    pass


def creator():
    return UndoableAPICommand()


def initializePlugin(m_object):
    m_plugin = OpenMaya.MFnPlugin(m_object)
    try:
        # let maya create the command, and then replace it with a wrapped version that
        # can accept a python class
        m_plugin.registerCommand(COMMAND_NAME, creator)
        UndoableAPICommand.wrap_command()
    except Exception as error:
        raise RuntimeError(f"Unable to initialize {COMMAND_NAME}: {error}") from error


def uninitializePlugin(m_object):
    m_plugin = OpenMaya.MFnPlugin(m_object)
    try:
        m_plugin.deregisterCommand(COMMAND_NAME)
    except Exception as error:
        raise RuntimeError(f"Unable to uninitialize {COMMAND_NAME}: {error}") from error
    # a forced unload (entries still on the undo queue) leaves the wrapper in maya.cmds,
    # where a reload would find it: take it out
    if hasattr(getattr(cmds, COMMAND_NAME, None), "__wrapped__"):
        delattr(cmds, COMMAND_NAME)
