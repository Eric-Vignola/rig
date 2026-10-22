"""``mat.astype(Cls)`` and ``shade.convert(mat, Cls)`` convert the object they
are called on IN PLACE (user decision 2026-10-22, option A).

Maya cannot change a node's type: a new node is built and the old one deleted,
as before. The same Python object is then re-pointed to the new node (its node
state, and its class), inside the conversion's one undo step: an undo brings
the old node AND the old class back on that object, a redo the new ones. Lists
and dict / set keys holding the object keep working. Other objects of the old
node, looked up separately, and plugs taken before, raise "already deleted"
(the plugs come back with an undo).
"""

from unittest import mock

from maya import cmds
from rig import List, Node, memoize, shade
from rig.shade import Blinn, Lambert, Material, Phong
from rig._internal import shade_convert
from rig._tests._base import MayaTestCase


DELETED = r"^RED__rigold already deleted!$"


class TestAstypeInPlace(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.undo_on()
        self.red = Lambert.define("RED", color=[1, 0, 0])
        self.warn = mock.patch.object(cmds, "warning")
        self.warn.start()
        self.addCleanup(self.warn.stop)

    def test_same_object_class_switch(self):
        red = self.red
        self.assertIs(red.astype(Blinn), red)
        self.assertIs(type(red), Blinn)
        self.assertEqual(cmds.nodeType("RED"), "blinn")
        self.assertEqual(repr(red), 'Blinn("RED")')
        self.assertEqual(cmds.getAttr("RED.color")[0], (1.0, 0.0, 0.0))
        red.eccentricity << 0.5
        self.assertAlmostEqual(cmds.getAttr("RED.eccentricity"), 0.5)
        self.assertIs(red.astype(Lambert), red)
        self.assertIs(type(red), Lambert)
        self.assertEqual(cmds.nodeType("RED"), "lambert")
        self.assertEqual(Node("RED"), red)
        self.assertEqual(cmds.ls("*__rigold*"), [])

    def test_undo_redo_switch_class_back(self):
        red = self.red
        red.astype(Blinn)
        red.astype(Lambert)
        cmds.undo()
        self.assertEqual(cmds.nodeType("RED"), "blinn")
        self.assertIs(type(red), Blinn)
        self.assertEqual(red.get_name(), "RED")
        red.color.get()  # usable
        cmds.undo()
        self.assertEqual(cmds.nodeType("RED"), "lambert")
        self.assertIs(type(red), Lambert)
        cmds.redo()
        self.assertEqual(cmds.nodeType("RED"), "blinn")
        self.assertIs(type(red), Blinn)
        cmds.redo()
        self.assertEqual(cmds.nodeType("RED"), "lambert")
        self.assertIs(type(red), Lambert)
        red.color << (0, 1, 0)
        self.assertEqual(cmds.getAttr("RED.color")[0], (0.0, 1.0, 0.0))

    def test_one_undo_step(self):
        cmds.flushUndo()
        self.red.astype(Phong)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), shade_convert.CHUNK)
        cmds.undo()
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "")
        self.assertIs(type(self.red), Lambert)

    def test_list_and_keys(self):
        red   = self.red
        items = List([red])
        key   = hash(red)
        seen, table = {red}, {red: 1}
        red.astype(Blinn)
        self.assertIs(items[0], red)
        self.assertEqual([str(x) for x in items], ["RED"])
        self.assertEqual(hash(red), key)
        self.assertIn(red, seen)
        self.assertEqual(table[red], 1)
        cmds.undo()
        self.assertIn(red, seen)
        self.assertEqual(table[red], 1)

    def test_old_objects_and_plugs_die(self):
        red   = self.red
        other = Lambert("RED")
        plug  = red.color
        child = red.colorR
        red.astype(Blinn)
        for call in (other.get_name, plug.get, child.get, lambda: plug << (0, 0, 1)):
            with self.assertRaisesRegex(RuntimeError, DELETED):
                call()
        self.assertEqual(str(other), "RED__rigold (deleted)")
        taken = red.color          # the new node's
        self.assertIsNot(taken, plug)
        taken.get()
        cmds.undo()                # the old node again: its plugs are back on red
        self.assertEqual(list(plug.get()), [1.0, 0.0, 0.0])
        self.assertIs(type(plug)._raw_get_node(plug), red)
        self.assertEqual(other.get_name(), "RED")
        with self.assertRaisesRegex(RuntimeError, r"already deleted!$"):
            taken.get()
        cmds.redo()
        taken.get()
        with self.assertRaisesRegex(RuntimeError, DELETED):
            plug.get()

    def test_freed_old_node_plug(self):
        red  = self.red
        plug = red.color
        red.astype(Blinn)
        cmds.flushUndo()           # the old node is freed now
        with self.assertRaisesRegex(RuntimeError, r"already deleted!$"):
            plug.get()
        red.color.get()

    def test_shade_convert_in_place(self):
        red = self.red
        self.assertIs(shade.convert(red, "blinn"), red)
        self.assertIs(type(red), Blinn)
        got = shade.convert("RED", Phong)   # a name: its own object, converted
        self.assertIs(type(got), Phong)
        self.assertIs(type(red), Blinn)     # red was another object of the node
        with self.assertRaisesRegex(RuntimeError, r"already deleted!$"):
            red.get_name()

    def test_refusal_leaves_object(self):
        red = self.red
        red.astype(Blinn)
        red.eccentricity << 0.6
        with self.assertRaises(ValueError):
            red.astype(Lambert, strict=True)
        self.assertIs(type(red), Blinn)
        with mock.patch.object(shade_convert, "_restore", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                red.astype(Phong)
        self.assertIs(type(red), Blinn)
        self.assertEqual(cmds.nodeType("RED"), "blinn")
        red.color.get()
        cmds.undo()                # the eccentricity set (the failure left no step)
        self.assertIs(type(red), Blinn)
        cmds.undo()
        self.assertIs(type(red), Lambert)

    def test_dry_run_and_same_type(self):
        red = self.red
        report = red.astype(Blinn, dry_run=True)
        self.assertIsInstance(report, shade_convert.Conversion)
        self.assertIs(type(red), Lambert)
        self.assertIs(red.astype(Lambert), red)

    def test_memo_after_switch(self):
        @memoize
        def kind(node):
            return Material(node)

        red = self.red
        self.assertIs(type(kind(red)), Lambert)
        red.astype(Blinn)
        self.assertIs(type(kind(red)), Blinn)
        cmds.undo()
        self.assertIs(type(kind(red)), Lambert)
