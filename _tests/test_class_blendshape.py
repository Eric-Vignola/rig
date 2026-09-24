import numpy as np
from maya import cmds
from rig.nodetypes import BlendShape, PyNode
from rig._tests._base import MayaTestCase


class TestBlendShape(MayaTestCase):
    """
    Test the BlendShape class
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        # make 2 cubes and a blend shape
        self.base        = PyNode(cmds.polyCube(name="base", ch=False)[0])
        self.target_name = "target"
        self.target      = PyNode(cmds.polyCube(name="target", ch=False)[0])
        self.bls         = PyNode.create("blendShape", self.target, self.base)

        # move a vert
        cmds.xform(f"{self.target}.vtx[3]", ws=True, t=(10, 2, 3))
        cmds.xform(f"{self.target}.vtx[4]", ws=True, t=(-1, 2, -3))
        self.indices      = np.array([3, 4])
        self.index_ranges = ["vtx[3:4]"]
        self.offsets      = np.array([(9.5, 1.5, 2.5), (-0.5, 1.5, -2.5)])

        # set weight
        self.weight = 0.5
        cmds.setAttr(f"{self.bls}.target", self.weight)

    def test_properties(self):
        self.assertEqual(self.bls.num_targets, 1)
        self.assertEqual(self.bls.get_targets(), [self.target_name])
        base_shape = self.base.get_children(type="mesh")[0]
        self.assertEqual(self.bls.get_geometries(), [base_shape])
        self.assertEqual(base_shape.get_deformers(node_type="blendShape"), [self.bls])
        self.assertEqual(self.bls.get_target_index(self.target_name), 0)
        self.assertEqual(self.bls.get_target_name(0), self.target_name)
        self.assertEqual(self.bls.get_target_weight(0), self.weight)

    def test_serialize(self):
        target_shape = self.target.get_children(type="mesh")[0]
        self.assertTrue(target_shape.world_shape_attr.is_connected)
        org_data = self.bls.get_target_data(self.target_name)
        self.assertTrue(np.allclose(org_data.indices, self.indices))
        self.assertTrue(np.allclose(org_data.offsets, self.offsets))
        self.assertTrue(target_shape.world_shape_attr.is_connected)

        new_target = "new_target"
        i          = self.bls.add_empty_target(new_target)
        self.assertEqual(self.bls.num_targets, 2)
        self.assertEqual(self.bls.get_targets(), [self.target_name, new_target])
        self.assertEqual(self.bls.get_target_index(new_target), 1)

        data = self.bls.get_target_data(new_target)
        self.assertEqual(data.indices.size, 0)
        self.assertEqual(data.offsets.size, 0)

        self.bls.set_target_data(i, org_data)
        data = self.bls.get_target_data(i)
        self.assertTrue(np.allclose(data.indices, self.indices))
        self.assertTrue(np.allclose(data.offsets, self.offsets))
        _, _, comp_attr = self.bls.target_data_attrs(i)
        self.assertEqual(comp_attr.get(), self.index_ranges)

        self.assertEqual(len(self.bls.serialize()), 2)
        self.assertEqual(len(self.bls.serialize(match_name="new_.*")), 1)

        # make sure we can serialize without an originalGeometry plug.
        new_bls = PyNode.create("blendShape", self.target, self.base)
        new_bls.originalGeometry[0].break_connections()
        self.assertEqual(len(new_bls.serialize()), 1)

    def test_create_from_morph_data(self):
        # rebuild a blendshape directly from a MorphList: targets are taken
        # from the data and offsets are applied during creation
        data = self.bls.serialize()
        self.bls.delete()

        bls = BlendShape.create(self.base, data)
        self.assertEqual(bls.get_targets(), [self.target_name])
        self.assertEqual(bls.serialize(), data)

        # a single MorphData is accepted as well
        bls.delete()
        bls = BlendShape.create(self.base, data[0])
        self.assertEqual(bls.get_targets(), [self.target_name])
        self.assertEqual(bls.serialize(), data)

    def test_rebuild(self):
        org_points   = self.target.get_children(type="mesh")[0].get_points()
        rebuilt_mesh = self.bls.rebuild_target(0)
        new_points   = rebuilt_mesh.get_children(type="mesh")[0].get_points()
        self.assertTrue(np.allclose(org_points, new_points, rtol=0.001))