from maya import cmds
from rig.maya.nodetypes import DisplayLayer, PyNode
from rig._tests._base import MayaTestCase


class TestDisplayLayer(MayaTestCase):
    """
    display layer unit tests.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.default_layer = DisplayLayer("defaultLayer")
        self.xform         = PyNode(cmds.group(empty=True))
        self.layer         = DisplayLayer.create()
        self.layer.add_members(self.xform)

    def test_members(self):
        self.assertEqual(self.layer.get_members(), [self.xform])
        self.layer.clear()
        self.assertEqual(self.layer.get_members(), [])

    def test_find(self):
        self.assertEqual(DisplayLayer.find_all(), [self.default_layer, self.layer])

    def test_get_or_create_and_for_node(self):
        self.assertEqual(DisplayLayer.get_or_create(self.layer.name), self.layer)
        cmds.select(self.xform.name)
        fresh = DisplayLayer.get_or_create("fresh")
        self.assertEqual(fresh.name, "fresh")
        self.assertEqual(fresh.get_members(), [])
        self.assertEqual(DisplayLayer.get_or_create("fresh"), fresh)
        with self.assertRaisesRegex(TypeError, "not a displayLayer"):
            DisplayLayer.get_or_create(self.xform.name)
        self.assertEqual(DisplayLayer.for_node(self.xform), self.layer)
        other = PyNode(cmds.group(empty=True, name="other"))
        self.assertIsNone(DisplayLayer.for_node(other))
        self.assertFalse(fresh.is_default)
        self.assertTrue(self.default_layer.is_default)
        cmds.namespace(add="ns")
        cmds.namespace(set="ns")
        try:
            spaced = DisplayLayer.get_or_create("spaced")
            self.assertEqual(spaced.name, "ns:spaced")
            self.assertEqual(DisplayLayer.get_or_create("spaced"), spaced)
        finally:
            cmds.namespace(set=":")

    def test_remove_members_and_delete(self):
        other = PyNode(cmds.group(empty=True, name="other"))
        fresh = DisplayLayer.get_or_create("fresh")
        fresh.add_members(other)
        # a node held by another layer is left there
        self.layer.remove_members(other)
        self.assertEqual(DisplayLayer.for_node(other), fresh)
        self.layer.remove_members(self.xform)
        self.assertIsNone(DisplayLayer.for_node(self.xform))
        self.assertEqual(self.layer.get_members(), [])
        fresh.delete()
        self.assertFalse(cmds.objExists("fresh"))
        self.assertIsNone(DisplayLayer.for_node(other))
        with self.assertRaises(TypeError):
            self.default_layer.delete()
        self.assertTrue(cmds.objExists("defaultLayer"))