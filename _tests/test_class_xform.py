import unittest

from maya import cmds
from maya.api import OpenMaya
from rig.maya.node_name import get_suffix
from rig.maya.nodetypes import PyNode
from rig._tests._base import MayaTestCase


class TestXformNodes(MayaTestCase):
    """
    Transform node types' unit tests.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        # make a joint chain
        self.root   = PyNode.create("joint", name="root_joint")
        self.c_name = "child_joint"
        self.joint2 = PyNode.create("joint", name=self.c_name, parent=self.root)
        self.joint3 = PyNode.create("joint", name=self.c_name, parent=self.joint2)

    def test_attr(self):
        self.root.set_xfrom_attrs_locked(True)
        for attr in self.root.iter_xform_attrs():
            self.assertTrue(attr.is_locked)
        self.root.set_xfrom_attrs_locked(False)
        for attr in self.root.iter_xform_attrs():
            self.assertFalse(attr.is_locked)

    def test_matrix(self):
        t = OpenMaya.MVector(1, 1, 1)
        self.joint2.t.set(*t)
        mat = self.joint2.get_matrix()
        mat = OpenMaya.MTransformationMatrix(mat)
        self.assertEqual(mat.translation(OpenMaya.MSpace.kWorld), t)

        mat = self.joint2.get_matrix(world_space=False)
        mat = OpenMaya.MTransformationMatrix(mat)
        self.assertEqual(mat.translation(OpenMaya.MSpace.kWorld), t)

        self.joint3.set_matrix(mat, world_space=False)
        mat = self.joint3.get_matrix()
        mat = OpenMaya.MTransformationMatrix(mat)
        self.assertEqual(mat.translation(OpenMaya.MSpace.kWorld), t * 2)

        mat = self.joint3.get_matrix(world_space=False)
        mat = OpenMaya.MTransformationMatrix(mat)
        self.assertEqual(mat.translation(OpenMaya.MSpace.kWorld), t)

    def test_hier(self):
        # skeleton traverse
        self.joint3.rename("child_joint3")
        self.assertEqual(self.joint3.get_root_joint(), self.root)
        self.assertEqual(list(self.root.iter_joints()), [self.joint2, self.joint3])

        result = self.root.iter_joints(yield_self=True)
        self.assertEqual(list(result), [self.root, self.joint2, self.joint3])

        result = self.root.iter_joints(match_name="joint", yield_self=True)
        self.assertEqual(list(result), [self.root, self.joint2, self.joint3])

        result = self.root.iter_joints(match_name="jo", exact_match=True)
        self.assertEqual(list(result), [])

        result = self.root.iter_joints(match_name="child_joint", exact_match=True)
        self.assertEqual(list(result), [self.joint2])

        result = self.root.iter_joints(match_name=".*3")
        self.assertEqual(list(result), [self.joint3])

        self.assertEqual(self.root.find_joint("jo"), self.joint2)

        # shape search
        cube1 = PyNode(cmds.polyCube(name="cube1", ch=False)[0])
        cube1.set_parent(self.joint2)
        cube2 = PyNode(cmds.polyCube(name="cube2", ch=False)[0])
        cube2.set_parent(self.joint3)

        result = self.root.iter_shapes("mesh", match_name="cube")
        self.assertEqual(list(result), [cube1, cube2])

        result = self.root.iter_shapes("mesh", match_name="cube", exact_match=True)
        self.assertEqual(list(result), [])

        result = self.root.iter_shapes("mesh", match_name=".*1", exact_match=True)
        self.assertEqual(list(result), [cube1])

        result = self.root.find_shape("mesh", "cu")
        self.assertEqual(result, cube1)

    def test_duplicate(self):
        p        = PyNode.create("transform")
        dup_root = self.root.duplicate_skeleton(parent=p)
        self.assertEqual(dup_root.get_parent(), p)
        self.assertEqual(len(list(dup_root.iter_joints())), 2)

        dup_root = self.root.duplicate_skeleton(include_list=[self.joint2])
        self.assertIsNone(dup_root.get_parent())
        self.assertEqual(len(list(dup_root.iter_joints())), 2)
        self.assertEqual(dup_root.get_children()[0].short_name, self.joint2.short_name)

        dup_root.rename_skeleton("test")
        for each in dup_root.iter_joints(yield_self=True):
            self.assertEqual(get_suffix(each), "test")

        with self.assertRaises(RuntimeError):
            dup_root.rename_skeleton("a %_")

    def test_duplicate_geom(self):
        xform = PyNode(cmds.polyCube(ch=False)[0])
        s     = PyNode.create("objectSet")
        s.add_members([xform] + xform.get_children())
        self.assertEqual(set(s.get_members()), set([xform] + xform.get_children()))

        parent = PyNode.create("transform")
        xform.set_parent(parent)
        xform.t.set(1, 2, 3)
        self.root.set_parent(xform)

        dup = xform.duplicate_geometry()

        self.assertIsNone(dup.get_parent())
        self.assertEqual(dup.short_name, xform.short_name)
        self.assertEqual(len(dup.get_children()), 1)
        c = dup.get_children()[0]
        self.assertEqual(c.node_type, "mesh")
        self.assertFalse(dup in s.get_members())
        self.assertFalse(c in s.get_members())

        self.assertEqual(dup.t.get()[0],         (0, 0, 0))
        self.assertEqual(dup.get_scale_pivot(),  OpenMaya.MPoint())
        self.assertEqual(dup.get_rotate_pivot(), OpenMaya.MPoint())

    def test_shapes(self):
        xform = PyNode(cmds.polyCube(name="test", ch=False)[0])
        self.root.set_parent(xform)
        shape = xform.get_children(type="mesh")[0]
        self.assertEqual(xform.get_shape(), shape)
        self.assertEqual(xform.get_shapes(), [shape])