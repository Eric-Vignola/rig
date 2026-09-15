import unittest

from maya import cmds
from rig._tests._base import initialize_standalone, MayaTestCase

# start standalone so that cmds can be imported
initialize_standalone()

try:
    cmds.loadPlugin("SkeletonDeltaBlend")
    registered = True
except Exception:
    registered = False

if registered:
    from rig.maya.nodetypes import Joint, SkeletonDeltaBlend


@unittest.skipIf(not registered, "Plugin not found.")
class TestSkelDeltaBlend(MayaTestCase):
    """
    Skeleton delta blend node unit tests.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        # make a driven skel
        self.root   = Joint.create(name="root_joint")
        self.joint2 = Joint.create(name="child_1_joint", parent=self.root)
        self.joint3 = Joint.create(name="child_2_joint", parent=self.joint2)

        # make a ref skel and a anim skel
        self.ref_root  = self.root.duplicate_skeleton(suffix="ref")
        self.anim_root = self.root.duplicate_skeleton(suffix="anim")

        # create blend
        self.blend = SkeletonDeltaBlend.create(
            self.root,
            ref_root  = self.ref_root,
            anim_root = self.anim_root,
        )

    def test_query(self):
        self.assertEqual(self.blend.num_targets,            0)
        self.assertEqual(self.blend.get_reference_root(),   self.ref_root)
        self.assertEqual(self.blend.get_output_skel_root(), self.root)

    def test_target(self):
        target = self.root.duplicate_skeleton(suffix="target1")
        c      = target.get_children(type="joint")[0]
        c.t.set(1, 2, 3)
        mat = c.get_matrix(world_space=True)
        self.blend.add_target(target, "test")

        self.assertEqual(self.blend.num_targets, 1)
        self.assertEqual(self.blend.get_targets(), ["test"])
        mat_dict = self.blend.get_target_matrices("test", world_space=True)
        self.assertEqual(mat_dict[self.joint2], mat)

        self.blend.empty_target_from_reference("test2")
        self.assertEqual(self.blend.num_targets, 2)
        self.assertEqual(self.blend.get_targets(), ["test", "test2"])
        self.assertEqual(self.blend.get_target_index("test2"), 1)
        self.assertEqual(self.blend.get_target_index("aaa"), -1)
        self.assertEqual(self.blend.get_target_name(1), "test2")
        self.blend.set_target_name(1, "abc")
        self.assertEqual(self.blend.get_target_name(1), "abc")
        self.blend.set_target_name("test", "ddd")
        self.assertEqual(self.blend.get_targets(), ["ddd", "abc"])

        self.blend.remove_target("ddd")
        self.assertEqual(self.blend.get_targets(), ["abc"])