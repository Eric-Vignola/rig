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