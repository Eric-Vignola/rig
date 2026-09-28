"""Joint orient / rotation conversions and orient_joint (their euler math comes
from ``cgmath.transforms``; no test covered them before cgmath 1.0.4 absorbed the
standalone ``transforms`` package)."""

import unittest

import numpy as np
from maya import cmds
from rig.nodetypes import Joint, Node
from rig._tests._base import MayaTestCase


def _world(node):
    return np.array(cmds.xform(node, query=True, worldSpace=True, matrix=True)).reshape(4, 4)


def _joint(name, parent=None, t=(0, 0, 0), r=(0, 0, 0), jo=(0, 0, 0)):
    cmds.select(clear=True)
    j = cmds.joint(name=name)
    if parent:
        cmds.parent(j, parent)
    cmds.setAttr(f"{j}.t", *t)
    cmds.setAttr(f"{j}.r", *r)
    cmds.setAttr(f"{j}.jo", *jo)
    return j


class TestJointOrients(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.grp = cmds.createNode("transform", name="grp")
        cmds.setAttr("grp.r", 12, -30, 45)
        self.j1 = _joint("j1", "grp", t=(1, 2, 3), r=(5, 10, 15), jo=(20, -35, 50))

    def test_convert_orients_to_rotation_keeps_the_world_matrix(self):
        before = _world(self.j1)
        Joint(self.j1).convert_orients_to_rotation()
        self.assertTrue(np.allclose(cmds.getAttr(f"{self.j1}.jo")[0], (0, 0, 0)))
        self.assertTrue(np.allclose(_world(self.j1), before, atol=1e-5))

    def test_convert_rotation_to_orients_keeps_the_world_matrix(self):
        before = _world(self.j1)
        Joint(self.j1).convert_rotation_to_orients()
        self.assertTrue(np.allclose(cmds.getAttr(f"{self.j1}.r")[0], (0, 0, 0)))
        self.assertTrue(np.allclose(_world(self.j1), before, atol=1e-5))

    def test_hierarchy_conversions_keep_every_world_matrix(self):
        j2 = _joint("j2", self.j1, t=(3, 0, 0), r=(0, 25, 0), jo=(10, 0, 5))
        j3 = _joint("j3", j2, t=(2, 1, 0), jo=(0, 0, -40))
        chain = (self.j1, j2, j3)
        before = [_world(j) for j in chain]
        Joint(self.j1).hierarchy_to_rotations()
        for j, m in zip(chain, before):
            with self.subTest(joint=j, step="to rotations"):
                self.assertTrue(np.allclose(cmds.getAttr(f"{j}.jo")[0], (0, 0, 0)))
                self.assertTrue(np.allclose(_world(j), m, atol=1e-5))
        Joint(self.j1).hierarchy_to_orients()
        for j, m in zip(chain, before):
            with self.subTest(joint=j, step="to orients"):
                self.assertTrue(np.allclose(cmds.getAttr(f"{j}.r")[0], (0, 0, 0), atol=1e-6))
                self.assertTrue(np.allclose(_world(j), m, atol=1e-5))

    def test_orient_joint_aims_the_axis_at_the_child(self):
        # a fresh pair per case: a root joint takes its up vector from its own
        # frame, so reusing one joint could leave the up axis along the aim
        cases = (("x", "y"), ("y", "z"), ("z", "x"), ("-z", "y"), ("-x", "-y"))
        for i, (aim_axis, up_axis) in enumerate(cases):
            with self.subTest(aim_axis=aim_axis, up_axis=up_axis):
                a = _joint(f"a{i}", t=(0, 0, 0))
                b = _joint(f"b{i}", a, t=(2, 1, 0.5))
                child_pos = _world(b)[3, :3].copy()
                Joint(a).orient_joint(aim_axis=aim_axis, up_axis=up_axis)
                m = _world(a)
                row = "xyz".index(aim_axis[-1])
                aim = m[row, :3] / np.linalg.norm(m[row, :3])
                if aim_axis.startswith("-"):
                    aim = -aim
                to_child = (child_pos - m[3, :3]) / np.linalg.norm(child_pos - m[3, :3])
                self.assertAlmostEqual(float(aim @ to_child), 1.0, places=5)
                self.assertTrue(np.allclose(_world(b)[3, :3], child_pos, atol=1e-5))  # the child did not move
                self.assertEqual(cmds.listRelatives(b, parent=True), [a])             # and is reparented

    def test_orient_joint_on_an_end_joint_zeroes_it(self):
        end = _joint("end", self.j1, t=(1, 0, 0), r=(3, 4, 5), jo=(6, 7, 8))
        Joint(end).orient_joint()
        self.assertTrue(np.allclose(cmds.getAttr(f"{end}.jo")[0], (0, 0, 0)))
        self.assertTrue(np.allclose(cmds.getAttr(f"{end}.r")[0], (0, 0, 0)))

    def test_orient_math_comes_from_cgmath(self):
        # rig carries no dependency on the standalone ``transforms`` package
        import rig.nodetypes.joint as joint_module

        source = open(joint_module.__file__, encoding="utf-8").read()
        self.assertNotIn("from transforms import", source)
        self.assertIn("from cgmath.transforms import", source)
        self.assertIsInstance(Node(self.j1), Joint)


if __name__ == "__main__":
    unittest.main()
