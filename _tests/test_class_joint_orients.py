"""Joint orient / rotation conversions and orient (their euler math comes from
``cgmath.transforms``; no test covered them before cgmath 1.0.4 absorbed the
standalone ``transforms`` package). `TestOneMethodWithAKeyword` holds the names
review's merges (user decision 2026-10-30): each twin that one flag split is one
method, its results the twins' and a leaf joint no crash."""

import unittest

import numpy as np
from maya import cmds
from rig.nodetypes import AmbiguousNodeError, Joint, Node
from rig._tests._base import MayaTestCase

# the twins the merges retired, and the spelling each hint gives now
_OLD_METHODS = {
    "hierarchy_to_rotations": "node.orients_to_rotations(hierarchy=True)",
    "hierarchy_to_orients":   "node.rotations_to_orients(hierarchy=True)",
    "orient_chain":           "node.orient(aim_axis, up_axis, hierarchy=True)",
}


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

    def test_orients_to_rotations_keeps_the_world_matrix(self):
        before = _world(self.j1)
        Joint(self.j1).orients_to_rotations()
        self.assertTrue(np.allclose(cmds.getAttr(f"{self.j1}.jo")[0], (0, 0, 0)))
        self.assertTrue(np.allclose(_world(self.j1), before, atol=1e-5))

    def test_rotations_to_orients_keeps_the_world_matrix(self):
        before = _world(self.j1)
        Joint(self.j1).rotations_to_orients()
        self.assertTrue(np.allclose(cmds.getAttr(f"{self.j1}.r")[0], (0, 0, 0)))
        self.assertTrue(np.allclose(_world(self.j1), before, atol=1e-5))

    def test_hierarchy_conversions_keep_every_world_matrix(self):
        j2 = _joint("j2", self.j1, t=(3, 0, 0), r=(0, 25, 0), jo=(10, 0, 5))
        j3 = _joint("j3", j2, t=(2, 1, 0), jo=(0, 0, -40))
        chain = (self.j1, j2, j3)
        before = [_world(j) for j in chain]
        Joint(self.j1).orients_to_rotations(hierarchy=True)
        for j, m in zip(chain, before):
            with self.subTest(joint=j, step="to rotations"):
                self.assertTrue(np.allclose(cmds.getAttr(f"{j}.jo")[0], (0, 0, 0)))
                self.assertTrue(np.allclose(_world(j), m, atol=1e-5))
        Joint(self.j1).rotations_to_orients(hierarchy=True)
        for j, m in zip(chain, before):
            with self.subTest(joint=j, step="to orients"):
                self.assertTrue(np.allclose(cmds.getAttr(f"{j}.r")[0], (0, 0, 0), atol=1e-6))
                self.assertTrue(np.allclose(_world(j), m, atol=1e-5))

    def test_orient_aims_the_axis_at_the_child(self):
        # a fresh pair per case: a root joint takes its up vector from its own
        # frame, so reusing one joint could leave the up axis along the aim
        cases = (("x", "y"), ("y", "z"), ("z", "x"), ("-z", "y"), ("-x", "-y"))
        for i, (aim_axis, up_axis) in enumerate(cases):
            with self.subTest(aim_axis=aim_axis, up_axis=up_axis):
                a = _joint(f"a{i}", t=(0, 0, 0))
                b = _joint(f"b{i}", a, t=(2, 1, 0.5))
                child_pos = _world(b)[3, :3].copy()
                Joint(a).orient(aim_axis=aim_axis, up_axis=up_axis)
                m = _world(a)
                row = "xyz".index(aim_axis[-1])
                aim = m[row, :3] / np.linalg.norm(m[row, :3])
                if aim_axis.startswith("-"):
                    aim = -aim
                to_child = (child_pos - m[3, :3]) / np.linalg.norm(child_pos - m[3, :3])
                self.assertAlmostEqual(float(aim @ to_child), 1.0, places=5)
                self.assertTrue(np.allclose(_world(b)[3, :3], child_pos, atol=1e-5))  # the child did not move
                self.assertEqual(cmds.listRelatives(b, parent=True), [a])             # and is reparented

    def test_orient_on_an_end_joint_zeroes_it(self):
        end = _joint("end", self.j1, t=(1, 0, 0), r=(3, 4, 5), jo=(6, 7, 8))
        Joint(end).orient()
        self.assertTrue(np.allclose(cmds.getAttr(f"{end}.jo")[0], (0, 0, 0)))
        self.assertTrue(np.allclose(cmds.getAttr(f"{end}.r")[0], (0, 0, 0)))

    def test_orient_math_comes_from_cgmath(self):
        # rig carries no dependency on the standalone ``transforms`` package
        import rig.nodetypes.joint as joint_module

        source = open(joint_module.__file__, encoding="utf-8").read()
        self.assertNotIn("from transforms import", source)
        self.assertIn("from cgmath.transforms import", source)
        self.assertIsInstance(Node(self.j1), Joint)


class TestOneMethodWithAKeyword(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.grp = cmds.createNode("transform", name="grp")
        cmds.setAttr("grp.r", 12, -30, 45)

    def _chain(self, prefix):
        """A bent chain of 4 under the turned group, every channel off its zero."""
        a = _joint(f"{prefix}a", self.grp, t=(1, 2, 3), r=(5, 10, 15), jo=(20, -35, 50))
        b = _joint(f"{prefix}b", a, t=(3, 0, 1), r=(0, 25, 0), jo=(10, 0, 5))
        c = _joint(f"{prefix}c", b, t=(2, 1, 0), jo=(0, 0, -40))
        d = _joint(f"{prefix}d", c, t=(0, 2, 1), r=(8, 0, 0))
        return a, b, c, d

    def test_a_leaf_joint_converts_alone(self):
        # the twins crashed here: listRelatives answers None on a leaf, and they appended to it
        leaf   = _joint("leaf", self.grp, t=(1, 0, 0), r=(5, 10, 15), jo=(20, -35, 50))
        before = _world(leaf)
        Joint(leaf).orients_to_rotations(hierarchy=True)
        self.assertTrue(np.allclose(cmds.getAttr(f"{leaf}.jo")[0], (0, 0, 0)))
        self.assertTrue(np.allclose(_world(leaf), before, atol=1e-5))
        Joint(leaf).rotations_to_orients(hierarchy=True)
        self.assertTrue(np.allclose(cmds.getAttr(f"{leaf}.r")[0], (0, 0, 0), atol=1e-6))
        self.assertTrue(np.allclose(_world(leaf), before, atol=1e-5))

    def test_hierarchy_converts_the_joints_below_and_no_others(self):
        a, b, c, d = self._chain("x")
        Joint(b).orients_to_rotations(hierarchy=True)
        self.assertTrue(np.allclose(cmds.getAttr(f"{a}.jo")[0], (20, -35, 50)))  # above: not touched
        for j in (b, c, d):
            with self.subTest(joint=j):
                self.assertTrue(np.allclose(cmds.getAttr(f"{j}.jo")[0], (0, 0, 0)))
        Joint(c).rotations_to_orients(hierarchy=True)
        self.assertFalse(np.allclose(cmds.getAttr(f"{b}.r")[0], (0, 0, 0)))      # above: not touched
        for j in (c, d):
            with self.subTest(joint=j):
                self.assertTrue(np.allclose(cmds.getAttr(f"{j}.r")[0], (0, 0, 0), atol=1e-6))

    def test_hierarchy_refuses_a_short_name_two_joints_share(self):
        # listRelatives gives the short name bare: each joint is read through rig's
        # lookup, whose error names the paths, not handed to getAttr (two matrices)
        a, b, c, _ = self._chain("x")
        cmds.createNode("joint", name="xd", parent=b)
        for convert in (Joint(a).orients_to_rotations, Joint(a).rotations_to_orients):
            with self.subTest(convert=convert.__name__):
                with self.assertRaisesRegex(AmbiguousNodeError, "use a path"):
                    convert(hierarchy=True)
        self.assertTrue(np.allclose(cmds.getAttr(f"{c}.jo")[0], (0, 0, -40)))  # raised before any edit

    def test_by_default_a_conversion_is_the_one_joint(self):
        a, b, _, _ = self._chain("x")
        Joint(a).orients_to_rotations()
        self.assertTrue(np.allclose(cmds.getAttr(f"{a}.jo")[0], (0, 0, 0)))
        self.assertTrue(np.allclose(cmds.getAttr(f"{b}.jo")[0], (10, 0, 5)))
        Joint(a).rotations_to_orients()
        self.assertTrue(np.allclose(cmds.getAttr(f"{a}.r")[0], (0, 0, 0), atol=1e-6))
        self.assertTrue(np.allclose(cmds.getAttr(f"{b}.r")[0], (0, 25, 0)))

    def test_orient_hierarchy_is_the_root_then_each_joint_below(self):
        # the walk the chain twin made: the root as a root joint, then the others, deepest
        # first; a chain built the same way and walked by hand lands on the same channels
        one, two = self._chain("x"), self._chain("y")
        Joint(one[0]).orient("y", "-z", hierarchy=True)
        Joint(two[0]).orient("y", "-z", root_joint=True)
        for j in reversed(two[1:]):
            Joint(j).orient("y", "-z")
        for x, y in zip(one, two):
            for attr in ("jo", "r", "t"):
                with self.subTest(joint=x, attr=attr):
                    self.assertTrue(np.allclose(cmds.getAttr(f"{x}.{attr}")[0], cmds.getAttr(f"{y}.{attr}")[0]))

    def test_orient_hierarchy_takes_the_top_joint_as_a_root(self):
        # under a turned parent the two frames differ: hierarchy reads the joint's own,
        # as root_joint=True does, not the parent's as a plain orient does
        for flag, kwargs in (("h", {"hierarchy": True}), ("b", {"hierarchy": True, "root_joint": True})):
            with self.subTest(kwargs=kwargs):
                top   = self._chain(f"top{flag}")[0]
                plain = self._chain(f"plain{flag}")[0]
                alone = self._chain(f"alone{flag}")[0]
                Joint(top).orient(**kwargs)
                Joint(plain).orient()
                Joint(alone).orient(root_joint=True)
                self.assertFalse(np.allclose(cmds.getAttr(f"{top}.r")[0], cmds.getAttr(f"{plain}.r")[0], atol=1e-3))
                self.assertTrue(np.allclose(cmds.getAttr(f"{top}.r")[0], cmds.getAttr(f"{alone}.r")[0]))

    def test_orient_hierarchy_on_an_end_joint_zeroes_it(self):
        end = _joint("end", self.grp, t=(1, 0, 0), r=(3, 4, 5), jo=(6, 7, 8))
        Joint(end).orient(hierarchy=True)
        self.assertTrue(np.allclose(cmds.getAttr(f"{end}.jo")[0], (0, 0, 0)))
        self.assertTrue(np.allclose(cmds.getAttr(f"{end}.r")[0], (0, 0, 0)))

    def test_the_keywords_are_keyword_only(self):
        joint = Joint(_joint("j", self.grp))
        for call in (
            lambda: joint.orients_to_rotations(True),
            lambda: joint.rotations_to_orients(True),
            lambda: joint.orient("x", "y", True),
            lambda: joint.orient("x", "y", False, True),
        ):
            with self.assertRaises(TypeError):
                call()

    def test_a_retired_twin_names_its_keyword(self):
        joint = Joint(_joint("j", self.grp))
        for old, new in _OLD_METHODS.items():
            with self.subTest(old=old):
                self.assertFalse(hasattr(Joint, old))
                with self.assertRaises(AttributeError) as caught:
                    getattr(joint, old)()
                self.assertIn(f"; node.{old} is now {new}", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
