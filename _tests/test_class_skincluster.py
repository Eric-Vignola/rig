from maya import cmds
from rig.maya.nodetypes import Joint, PyNode, SkinCluster
from rig._tests._base import MayaTestCase


class TestSkinCluster(MayaTestCase):
    """
    Unit test for SkinCluster node class.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        # make a joint chain
        self.joint1 = PyNode.create("joint", name="cube_1_joint")
        self.joint2 = PyNode.create("joint", name="cube_2_joint", parent=self.joint1)
        self.joint2.t.set(0, 1, 0)

        # make a cube
        self.cube = PyNode(cmds.polyCube(name="base", height=2, ch=False)[0])

        # create a skincluster
        self.skin = PyNode.create("skinCluster", self.cube, (self.joint1, self.joint2))
        cmds.skinPercent(
            self.skin, f"{self.cube}.vtx[0:3]", transformValue=[(self.joint1, 1)]
        )
        cmds.skinPercent(
            self.skin, f"{self.cube}.vtx[4:7]", transformValue=[(self.joint2, 1)]
        )

    def test_properties(self):
        self.assertEqual(self.skin.get_influence_objects(), [self.joint1, self.joint2])

    def test_add_influence_objects_keeps_the_selection(self):
        """Maya selects an influence it adds singly, discarding the pick."""
        cmds.selectPref(trackSelectionOrder=True)
        extra_joint = Joint.create()
        # One at a time, the way a user clicks: a bulk select coalesces the
        # components into ranges and sorts them, leaving no order to keep.
        cmds.select(clear=True)
        for index in (0, 3, 1):
            cmds.select(f"{self.cube}.vtx[{index}]", add=True)
        before = cmds.ls(orderedSelection=True, long=True)

        self.skin.add_influence_objects(extra_joint)

        self.assertEqual(cmds.ls(orderedSelection=True, long=True), before)
        # The displayed component of an order-tracking picker is the last one.
        self.assertTrue(
            cmds.ls(orderedSelection=True)[-1].endswith(".vtx[1]"),
            msg=str(cmds.ls(orderedSelection=True)),
        )
        self.assertIn(extra_joint, self.skin.get_influence_objects())

    def test_add_influence_objects_leaves_an_empty_selection_empty(self):
        extra_joint = Joint.create()
        cmds.select(clear=True)

        self.skin.add_influence_objects(extra_joint)

        self.assertEqual(cmds.ls(selection=True), [])

    def test_set_weights_keeps_the_selection(self):
        """set_weights routes joints it has to add through the same helper."""
        data = self.skin.serialize()
        self.skin.delete()
        skin = PyNode.create("skinCluster", self.cube, self.joint1)

        cmds.select([f"{self.cube}.vtx[0]", f"{self.cube}.vtx[3]"])
        before = cmds.ls(selection=True, long=True)

        # One joint short of the data, so exactly one influence gets added.
        skin.set_weights(data)

        self.assertEqual(cmds.ls(selection=True, long=True), before)

    def test_weights(self):
        org_data = self.skin.serialize()
        cmds.skinPercent(
            self.skin, f"{self.cube}.vtx[0]", transformValue=[(self.joint2, 1)]
        )
        data = self.skin.serialize()
        self.assertNotEqual(org_data, data)

        self.skin.set_weights(org_data)
        data = self.skin.serialize()
        self.assertEqual(org_data, data)

        # test apply skin data to a skincluster with less influences
        self.skin.delete()
        skin = PyNode.create("skinCluster", self.cube, self.joint1)
        skin.set_weights(org_data)
        data = skin.serialize()
        self.assertEqual(org_data, data)

        # test apply skin data to a skincluster with more influences (non-additive)
        skin.delete()
        extra_joint = Joint.create()
        skin = PyNode.create(
            "skinCluster", self.cube, [self.joint1, self.joint2, extra_joint]
        )
        skin.set_weights(org_data)
        data = skin.serialize()
        self.assertEqual(org_data, data)
        self.assertFalse(extra_joint.clean_name in data.influences)

        # test apply skin data to a skincluster with more influences (additive)
        skin.delete()
        skin = PyNode.create(
            "skinCluster", self.cube, [self.joint1, self.joint2, extra_joint]
        )
        skin.set_weights(org_data, additive=True)
        data = skin.serialize()
        self.assertNotEqual(org_data, data)
        self.assertTrue(extra_joint.clean_name in data.influences)

    def test_create_from_skin_data(self):
        # rebuild a skincluster directly from a SkinData object: influences are
        # taken from the data and weights are applied during creation
        data = self.skin.serialize()
        self.skin.delete()

        skin = SkinCluster.create(self.cube, data)
        self.assertEqual(skin.get_influence_objects(), [self.joint1, self.joint2])
        self.assertEqual(skin.serialize(), data)