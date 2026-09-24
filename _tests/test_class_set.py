from maya import cmds
from rig.nodetypes import PyNode
from rig._tests._base import MayaTestCase


class TestSet(MayaTestCase):
    """
    Set node types' unit tests.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.set_node = PyNode.create("objectSet")

    def test_membership(self):
        self.assertEqual(self.set_node.get_members(), [])

        n = PyNode.create("transform")
        self.set_node.add_members(n)
        self.assertEqual(self.set_node.get_members(), [n])

        n2 = PyNode.create("transform")
        self.set_node.add_members([n, n2])
        self.assertEqual(set(self.set_node.get_members()), {n, n2})

        n3 = PyNode(cmds.polyCube(ch=False)[0])
        self.set_node.add_members(f"{n3}.vtx[2]")
        self.assertEqual(set(self.set_node.get_members()), {n, n2})
        self.assertEqual(len(self.set_node.get_members(as_components=True)), 3)