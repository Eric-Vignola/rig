import functools

from maya import cmds
from maya.api import OpenMaya

COMMAND_NAME = "runUndoableAPICommand"


class UndoableAPICommand(OpenMaya.MPxCommand):
    """
    A Python API 2.0 (OpenMaya) command wrapper with undo & redo support.

    Custom API commands can be triggered using this wrapper command so that they
    don't have to be registered individually. This helps prevent the need to create
    a bunch of random plug-ins for various operations.

    Usage:
    ```python
    from rig.maya.plugins import load_plugin

    # first make a command with doIt(), undoIt(), and redoIt() implemented.
    class MyCommand:

        def __init__(self) -> None:
            with load_plugin("undoable_api_command"):
                cmds.runUndoableAPICommand(self)

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

    call_class = None

    def __init__(self):
        super(UndoableAPICommand, self).__init__()
        self.py_class = None

    def isUndoable(self):
        return True

    def doIt(self, args):
        self.py_class = self.call_class
        result        = self.py_class.doIt()
        if result is not None:
            self.setResult(result)

    def redoIt(self):
        result = self.py_class.redoIt()
        if result is not None:
            self.setResult(result)

    def undoIt(self):
        state = cmds.undoInfo(query=True, state=True)
        if state:
            cmds.undoInfo(stateWithoutFlush=False)
        try:
            result = self.py_class.undoIt()
            if result is not None:
                self.setResult(result)
        finally:
            if state:
                cmds.undoInfo(stateWithoutFlush=True)

    @classmethod
    def wrap_command(cls):
        """In order to bypass the normal argument required for commands due to trying
        to be compatible with MEL, we have to wrap the created
        cmds.runUndoableAPICommand so the class can be passed."""
        cmd_func = getattr(cmds, COMMAND_NAME)

        @functools.wraps(cmd_func)
        def wrapped(py_class):
            # make sure we store the class so it wont go out of scope and then call it
            cls.call_class = py_class
            cmds.undoInfo(openChunk=True, chunkName=COMMAND_NAME)
            try:
                return cmd_func()
            finally:
                cmds.undoInfo(closeChunk=True)

        # now overwrite the one in cmds with this new wrapped version
        wrapped.__wrapped__ = cmd_func
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
    except BaseException:
        raise RuntimeError("Unable to initialize: {}".format(COMMAND_NAME))


def uninitializePlugin(m_object):
    m_plugin = OpenMaya.MFnPlugin(m_object)
    try:
        m_plugin.deregisterCommand(COMMAND_NAME)
    except BaseException:
        raise RuntimeError("Unable to uninitialize: {}".format(COMMAND_NAME))