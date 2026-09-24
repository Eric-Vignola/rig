import logging
import os

import maya.cmds as cmds
import numpy as np
import rig.maya.nodetypes.mesh as mesh_api
from rig.maya.nodetypes import Axis
from cgmath.geometry import MapData
from rig._tests._base import MayaTestCase

LOGGER = logging.getLogger(__name__)


class TestMeshMaps(MayaTestCase):
    def setUp(self):
        super().setUp()

        self._map_01_name            = "TESTMAP01"
        self._map_02_name            = "TESTMAP02"
        self._invalid_map_name       = "INVALIDMAPNAME"
        self._map_01_rename_name     = "RENAMEDTESTMAP01"
        self._map_test_category      = "TEST"
        self._scene_test_object_name = "TestCube"
        # cube only has 8 verts so values will always of size 8
        self._map_default_values     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._map_default_one_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self._map_01_default_values  = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        self._map_02_default_values  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        map_size                     = len(self._map_default_values)

        self._map_smaller_sized_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self._map_smaller_sized_compare_values = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.0,
            0.0,
        ]

        self._map_larger_sized_values = [
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
        ]
        self._map_larger_sized_compare_values = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

        self._map_01_data = MapData(
            name       = self._map_01_name,
            categories = [self._map_test_category],
            indices    = np.fromiter(range(map_size), int),
            values     = np.array(self._map_01_default_values),
        )

        self._map_02_data = MapData(
            name       = self._map_02_name,
            categories = [self._map_test_category],
            indices    = np.fromiter(range(map_size), int),
            values     = np.array(self._map_02_default_values),
        )

        self._mirror_positive_dir_x_map_values = [
            0.7,
            0.7,
            0.5,
            0.5,
            0.3,
            0.3,
            0.1,
            0.1,
        ]
        self._mirror_positive_dir_y_map_values = [
            0.6,
            0.5,
            0.6,
            0.5,
            0.4,
            0.3,
            0.4,
            0.3,
        ]
        self._mirror_positive_dir_z_map_values = [
            0.8,
            0.7,
            0.6,
            0.5,
            0.6,
            0.5,
            0.8,
            0.7,
        ]
        self._mirror_negative_dir_x_map_values = [
            0.8,
            0.8,
            0.6,
            0.6,
            0.4,
            0.4,
            0.2,
            0.2,
        ]
        self._mirror_negative_dir_y_map_values = [
            0.8,
            0.7,
            0.8,
            0.7,
            0.2,
            0.1,
            0.2,
            0.1,
        ]
        self._mirror_negative_dir_z_map_values = [
            0.2,
            0.1,
            0.4,
            0.3,
            0.4,
            0.3,
            0.2,
            0.1,
        ]

    def open_scene_and_get_test_mesh(self, scene_mesh_name: str) -> mesh_api.Mesh:
        test_maya_file = os.path.splitext(__file__)[0] + ".ma"
        cmds.file(test_maya_file, f=True, o=True)
        return mesh_api.Mesh(scene_mesh_name)

    def test_add_rename_remove_map(self):
        test_mesh = self.open_scene_and_get_test_mesh(self._scene_test_object_name)

        # check attr isn't already there, add attr/map, check map name is the same
        self.assertFalse(test_mesh.has_attr(self._map_01_name))
        test_mesh.add_map(self._map_01_name, category=self._map_test_category)
        self.assertTrue(test_mesh.has_attr(self._map_01_name))

        # rename map, check old name doesn't exist anymore by trying to rename it again, check new name exists
        self.assertTrue(
            test_mesh.rename_attr(
                self._map_01_name,
                self._map_01_rename_name,
            )
        )
        with self.assertRaises(RuntimeError):
            test_mesh.rename_attr(
                self._map_01_name,
                self._map_01_rename_name,
            )

        self.assertFalse(test_mesh.has_attr(self._map_01_name))
        self.assertTrue(test_mesh.has_attr(self._map_01_rename_name))

        self.assertTrue(len(test_mesh.get_map_attrs()) == 1)
        self.assertTrue(test_mesh.delete_attr(self._map_01_rename_name))
        self.assertTrue(len(test_mesh.get_map_attrs()) == 0)
        self.assertFalse(test_mesh.has_attr(self._map_01_rename_name))

        # now add by name then rename and remove by the create map mplug
        new_map_plug = test_mesh.add_map(
            self._map_01_name, category=self._map_test_category
        )
        self.assertTrue(new_map_plug is not None)
        self.assertTrue(test_mesh.has_attr(self._map_01_name))
        self.assertTrue(
            test_mesh.rename_attr(
                new_map_plug,
                self._map_01_rename_name,
            )
        )
        self.assertTrue(test_mesh.has_attr(self._map_01_rename_name))
        self.assertTrue(new_map_plug.name == self._map_01_rename_name)

        self.assertTrue(len(test_mesh.get_map_attrs()) == 1)
        self.assertTrue(test_mesh.delete_attr(new_map_plug))
        self.assertTrue(len(test_mesh.get_map_attrs()) == 0)
        self.assertFalse(test_mesh.has_attr(self._map_01_rename_name))

    def test_set_default_values_and_duplicate_map(self):
        test_mesh = self.open_scene_and_get_test_mesh(self._scene_test_object_name)

        # check attr isn't already there, add attr/map, check map name is the same
        self.assertFalse(test_mesh.has_attr(self._map_01_name))
        self.assertFalse(test_mesh.has_attr(self._map_02_name))
        self.assertTrue(len(test_mesh.get_map_attrs()) == 0)

        map_01_plug = test_mesh.add_map(
            self._map_01_name,
            category = self._map_test_category,
            values   = self._map_01_default_values,
        )
        self.assertTrue(test_mesh.has_attr(self._map_01_name))
        self.assertTrue(len(test_mesh.get_map_attrs()) == 1)
        self.assertListEqual(
            test_mesh.get_map_values(self._map_01_name), self._map_01_default_values
        )

        map_02_plug = test_mesh.duplicate_map(self._map_01_name, self._map_02_name)
        # self.assertTrue(test_mesh.has_attr(self._map_02_name))
        self.assertTrue(len(test_mesh.get_map_attrs()) == 2)

        self.assertListEqual(
            test_mesh.get_map_values(self._map_01_name),
            test_mesh.get_map_values(self._map_02_name),
        )

        with self.assertRaises(RuntimeError):
            test_mesh.duplicate_map(self._map_01_name, self._map_02_name)

        self.assertTrue(map_01_plug is not None)
        self.assertTrue(map_02_plug is not None)

        self.assertListEqual(
            test_mesh.get_map_values(map_01_plug),
            test_mesh.get_map_values(map_02_plug),
        )

        # cleanup
        self.assertTrue(test_mesh.delete_attr(map_01_plug))
        self.assertTrue(test_mesh.delete_attr(map_02_plug))
        self.assertTrue(len(test_mesh.get_map_attrs()) == 0)

    def test_set_map_values(self):
        test_mesh = self.open_scene_and_get_test_mesh(self._scene_test_object_name)

        self.assertFalse(test_mesh.has_attr(self._map_01_name))
        map_01_plug = test_mesh.add_map(
            self._map_01_name,
            category=self._map_test_category,
        )
        self.assertTrue(map_01_plug is not None)
        self.assertTrue(test_mesh.has_attr(self._map_01_name))

        self.assertListEqual(
            test_mesh.get_map_values(map_01_plug),
            self._map_default_values,
        )

        test_mesh.set_map_values(map_01_plug, self._map_01_default_values)
        self.assertListEqual(
            test_mesh.get_map_values(map_01_plug),
            self._map_01_default_values,
        )

        test_mesh.set_map_values(map_01_plug, 1.0)
        self.assertListEqual(
            test_mesh.get_map_values(map_01_plug),
            self._map_default_one_values,
        )

        self.assertListEqual(
            test_mesh.get_map_values(map_01_plug),
            self._map_default_one_values,
        )

        test_mesh.set_map_values(map_01_plug, self._map_smaller_sized_values)
        self.assertListEqual(
            test_mesh.get_map_values(map_01_plug),
            self._map_smaller_sized_compare_values,
        )

        test_mesh.set_map_values(map_01_plug, self._map_larger_sized_values)
        self.assertListEqual(
            test_mesh.get_map_values(map_01_plug),
            self._map_larger_sized_compare_values,
        )

        # cleanup
        self.assertTrue(test_mesh.delete_attr(map_01_plug))

    def test_gathering_map_data(self):
        test_mesh = self.open_scene_and_get_test_mesh(self._scene_test_object_name)

        map_01_plug = test_mesh.add_map(
            self._map_01_name,
            category = self._map_test_category,
            values   = self._map_01_default_values,
        )

        map_02_plug = test_mesh.add_map(
            self._map_02_name,
            category = self._map_test_category,
            values   = self._map_02_default_values,
        )

        map_01_data = test_mesh.get_map_data(map_01_plug)
        map_02_data = test_mesh.get_map_data(map_02_plug)

        self.assertEqual(map_01_data, self._map_01_data)
        self.assertEqual(map_02_data, self._map_02_data)

        raw_search_maps_data_dict = test_mesh.serialize_maps()
        raw_search_maps_data_dict = {x.name: x for x in raw_search_maps_data_dict}

        self.assertEqual(map_01_data, raw_search_maps_data_dict[self._map_01_name])
        self.assertEqual(map_02_data, raw_search_maps_data_dict[self._map_02_name])

        map_03_default_category_plug = test_mesh.add_map(
            self._map_01_rename_name,
            values=self._map_01_default_values,
        )

        maps_category_search_data_dict = test_mesh.serialize_maps(
            category=self._map_test_category
        )
        maps_category_search_data_dict = {
            x.name: x for x in maps_category_search_data_dict
        }
        self.assertTrue(len(maps_category_search_data_dict) == 2)

        map_02_data = test_mesh.get_map_data(self._map_02_name)
        self.assertEqual(map_02_data, maps_category_search_data_dict[self._map_02_name])

        second_raw_search_maps_data_dict = test_mesh.serialize_maps()
        second_raw_search_maps_data_dict = {
            x.name: x for x in second_raw_search_maps_data_dict
        }
        self.assertTrue(len(second_raw_search_maps_data_dict) == 3)
        self.assertEqual(
            second_raw_search_maps_data_dict[self._map_02_name],
            maps_category_search_data_dict[self._map_02_name],
        )

        # cleanup
        self.assertTrue(test_mesh.delete_attr(map_01_plug))
        self.assertTrue(test_mesh.delete_attr(map_02_plug))
        self.assertTrue(test_mesh.delete_attr(map_03_default_category_plug))

    def test_mirror_map(self):
        test_mesh = self.open_scene_and_get_test_mesh(self._scene_test_object_name)

        map_01_plug = test_mesh.add_map(
            self._map_01_name,
            category = self._map_test_category,
            values   = self._map_01_default_values,
        )
        map_02_plug = test_mesh.add_map(
            self._map_02_name,
            category = self._map_test_category,
            values   = self._map_01_default_values,
        )
        map_03_plug = test_mesh.add_map(
            self._map_01_rename_name,
            category = self._map_test_category,
            values   = self._map_01_default_values,
        )
        # mirror from positive direction
        test_mesh.mirror_map(map_01_plug, Axis.X)
        test_mesh.mirror_map(map_02_plug, Axis.Y)
        test_mesh.mirror_map(map_03_plug, Axis.Z)

        positive_x_mirror_values = test_mesh.get_map_values(map_01_plug)
        positive_y_mirror_values = test_mesh.get_map_values(map_02_plug)
        positive_z_mirror_values = test_mesh.get_map_values(map_03_plug)

        self.assertListEqual(
            positive_x_mirror_values, self._mirror_positive_dir_x_map_values
        )
        self.assertListEqual(
            positive_y_mirror_values, self._mirror_positive_dir_y_map_values
        )
        self.assertListEqual(
            positive_z_mirror_values, self._mirror_positive_dir_z_map_values
        )

        # reset values
        test_mesh.set_map_values(map_01_plug, self._map_01_default_values)
        test_mesh.set_map_values(map_02_plug, self._map_01_default_values)
        test_mesh.set_map_values(map_03_plug, self._map_01_default_values)

        # mirror from negative direction
        test_mesh.mirror_map(map_01_plug, Axis.X, True)
        test_mesh.mirror_map(map_02_plug, Axis.Y, True)
        test_mesh.mirror_map(map_03_plug, Axis.Z, True)

        negative_x_mirror_values = test_mesh.get_map_values(map_01_plug)
        negative_y_mirror_values = test_mesh.get_map_values(map_02_plug)
        negative_z_mirror_values = test_mesh.get_map_values(map_03_plug)

        self.assertListEqual(
            negative_x_mirror_values, self._mirror_negative_dir_x_map_values
        )
        self.assertListEqual(
            negative_y_mirror_values, self._mirror_negative_dir_y_map_values
        )
        self.assertListEqual(
            negative_z_mirror_values, self._mirror_negative_dir_z_map_values
        )