"""rig's working units, cm and degrees: a scene set to others gets one warning, at rig's first
statement there, and another once a scene is made or opened or its units change
(``rig._internal.units``)."""
from unittest import mock

from maya import cmds

from rig import Node
from rig._internal import units
from rig._tests._base import MayaTestCase


class TestUnits(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        linear = cmds.currentUnit(query=True, linear=True)
        angle = cmds.currentUnit(query=True, angle=True)
        self.addCleanup(cmds.currentUnit, linear=linear, angle=angle)
        patcher = mock.patch.object(units.cmds, "warning")
        self.warning = patcher.start()
        self.addCleanup(patcher.stop)
        self.a = Node.create("transform", name="a")      # a statement: this scene is checked

    def warnings(self):
        return [call.args[0] for call in self.warning.call_args_list if call.args[0].startswith("rig: this scene")]

    def test_cm_and_degrees_say_nothing(self):
        self.a.tx << self.a.ty * 2
        self.assertTrue(units.CHECKED[0])
        self.assertEqual(self.warnings(), [])

    def test_other_units_warn_once(self):
        cmds.currentUnit(linear="m")
        self.assertFalse(units.CHECKED[0])               # the units changed: checked again
        self.a.tx << self.a.ty * 2
        self.a.tz << self.a.ty * 3
        self.assertEqual(self.warnings(), [
            "rig: this scene works in m and deg: rig's math is built for cm and degrees (Maya's default), "
            "and some of it gives other values in other units (rig's CHEATSHEET, Setup)"])

    def test_a_change_of_units_warns_again(self):
        cmds.currentUnit(angle="rad")
        self.a.tx << 1
        cmds.currentUnit(angle="deg", linear="mm")
        self.a.tx << 2
        cmds.currentUnit(linear="cm")
        self.a.tx << 3                                   # cm and degrees again: nothing
        found = self.warnings()
        self.assertEqual(len(found), 2)
        self.assertIn("works in cm and rad", found[0])
        self.assertIn("works in mm and deg", found[1])

    def test_a_new_scene_is_checked_again(self):
        self.a.tx << 1
        self.assertTrue(units.CHECKED[0])
        cmds.file(new=True, force=True)
        self.assertFalse(units.CHECKED[0])
