"""Tests for ``rig._internal.undo`` -- the per-inject undo chunk."""

from maya import cmds
from rig._internal.undo import _undo_chunk
from rig._tests._base import MayaTestCase


class TestUndoChunk(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # mayapy starts with the undo queue off; the chunk is meaningless
        # without it.
        cmds.undoInfo(state=True, infinity=True)

    def test_two_commands_undo_as_one_step(self):
        with _undo_chunk("rig.test"):
            cmds.createNode("transform", name="first")
            cmds.createNode("transform", name="second")
        self.assertTrue(cmds.objExists("first"))
        self.assertTrue(cmds.objExists("second"))
        cmds.undo()
        self.assertFalse(cmds.objExists("first"))
        self.assertFalse(cmds.objExists("second"))
        cmds.redo()
        self.assertTrue(cmds.objExists("first"))
        self.assertTrue(cmds.objExists("second"))

    def test_chunk_closes_when_the_block_raises(self):
        with self.assertRaises(RuntimeError):
            with _undo_chunk("rig.test"):
                cmds.createNode("transform", name="inside")
                raise RuntimeError("boom")
        cmds.createNode("transform", name="after")
        # ``after`` is its own undo step because the chunk was closed by
        # the ``finally``; a still-open chunk would swallow it.
        cmds.undo()
        self.assertFalse(cmds.objExists("after"))
        self.assertTrue(cmds.objExists("inside"))
        cmds.undo()
        self.assertFalse(cmds.objExists("inside"))
