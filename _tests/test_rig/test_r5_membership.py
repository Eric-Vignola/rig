"""Round 5, B0: API edits registered through rig's undo command.

B0 (``rig.nodetypes.plugins``): `_register_api_edit` applies an ``MDGModifier`` inside
``rigUndoableAPICommand`` (one unnamed undo step: undo is the modifier's ``undoIt``, redo
its ``doIt``); `_api_edit_ready` says whether it may run (the command loads, Maya is not
undoing or redoing). With undo off the command applies the edit and Maya keeps nothing.

C1 (eager API membership, 12f6921) is reverted (round 5 BCFIX): every member a scope
makes joins its container through ``cmds.container -edit -addNode -force``, as in
v2.0.0a4.
"""

import weakref
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import container
from rig.nodetypes import plugins
from rig.nodetypes.plugins import UNDOABLE_API_COMMAND, _api_edit_ready, _register_api_edit
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

PLUGIN = "undoable_api_command"


def _plug(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


def _connect_modifier(src, dst):
    modifier = OpenMaya.MDGModifier()
    modifier.connect(_plug(src), _plug(dst))
    return modifier


class _MembershipCase(UndoWalk, MayaTestCase):
    """A new scene, the undo queue on, infinite and flushed; rig's plug-in loaded."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        plugins._ensure_loaded(PLUGIN, True)
        cmds.flushUndo()

    def tearDown(self):
        try:
            while container._stack:
                container.__exit__(None, None, None)
            cmds.flushUndo()
        finally:
            super().tearDown()


class TestRegisterApiEdit(_MembershipCase):
    """B0: one unnamed step; the readiness invariants (the command unloaded, a load
    failure, undo off, during undo and during redo); a failing doIt registers nothing."""

    def setUp(self):
        super().setUp()
        cmds.createNode("multiplyDivide", name="a")
        cmds.createNode("multiplyDivide", name="b")
        cmds.flushUndo()

    def connected(self):
        return cmds.isConnected("a.outputX", "b.input1X")

    def test_one_unnamed_step_undone_and_redone(self):
        self.assertTrue(_api_edit_ready())
        _register_api_edit(_connect_modifier("a.outputX", "b.input1X"))
        self.assertTrue(self.connected())
        self.assertEqual(self.undo_name(), "")
        self.undo_steps(1)
        self.assertFalse(self.connected())
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(1)
        self.assertTrue(self.connected())

    def test_inside_a_chunk_the_step_joins_it(self):
        cmds.undoInfo(openChunk=True)
        try:
            cmds.setAttr("a.input1X", 3.0)
            _register_api_edit(_connect_modifier("a.outputX", "b.input1X"))
        finally:
            cmds.undoInfo(closeChunk=True)
        self.undo_steps(1)
        self.assertEqual((self.connected(), cmds.getAttr("a.input1X")), (False, 0.0))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_undo_off_applies_and_keeps_nothing(self):
        kept = []

        class Tracked(plugins._APIEdit):
            __slots__ = ("__weakref__",)

            def __init__(self, modifier):
                super().__init__(modifier)
                kept.append(weakref.ref(self))

        cmds.undoInfo(state=False)
        try:
            self.assertTrue(_api_edit_ready())
            with mock.patch.object(plugins, "_APIEdit", Tracked):
                _register_api_edit(_connect_modifier("a.outputX", "b.input1X"))
        finally:
            cmds.undoInfo(state=True)
        self.assertTrue(self.connected())
        self.assertEqual(len(kept), 1)
        self.assertIsNone(kept[0]())  # Maya kept neither the step nor the modifier
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_the_command_unloaded_is_loaded_again(self):
        cmds.flushUndo()
        cmds.unloadPlugin(PLUGIN)
        self.assertFalse(hasattr(cmds, UNDOABLE_API_COMMAND))
        self.assertTrue(_api_edit_ready())
        self.assertTrue(hasattr(cmds, UNDOABLE_API_COMMAND))

    def test_a_load_failure_is_not_ready(self):
        cmds.flushUndo()
        cmds.unloadPlugin(PLUGIN)
        with mock.patch.object(cmds, "loadPlugin", return_value=None):
            self.assertFalse(_api_edit_ready())  # "did not register": no exception
        # the state a forced unload leaves: loaded, no maya.cmds function
        self.assertTrue(_api_edit_ready())
        wrapper = getattr(cmds, UNDOABLE_API_COMMAND)
        delattr(cmds, UNDOABLE_API_COMMAND)
        try:
            with mock.patch.object(cmds, "loadPlugin") as load:
                self.assertFalse(_api_edit_ready())
            load.assert_not_called()
        finally:
            setattr(cmds, UNDOABLE_API_COMMAND, wrapper)

    def test_not_ready_during_undo_and_redo(self):
        seen = []

        class Spy:
            def doIt(self):
                seen.append(("doIt", _api_edit_ready()))

            def undoIt(self):
                seen.append(("undoIt", _api_edit_ready()))

            def redoIt(self):
                seen.append(("redoIt", _api_edit_ready()))

        plugins._run_undoable(Spy())
        cmds.undo()
        cmds.redo()
        self.assertEqual(seen, [("doIt", True), ("undoIt", False), ("redoIt", False)])

    def test_a_failing_doit_registers_nothing(self):
        cmds.setAttr("b.input1Y", lock=True)
        cmds.flushUndo()
        modifier = _connect_modifier("a.outputX", "b.input1X")
        modifier.connect(_plug("a.outputY"), _plug("b.input1Y"))  # refused: locked
        with self.assertRaises(RuntimeError):
            _register_api_edit(modifier)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.assertTrue(self.connected())  # what doIt did before it failed ...
        modifier.undoIt()
        self.assertFalse(self.connected())  # ... the caller undoes
