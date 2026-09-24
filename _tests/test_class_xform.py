import unittest

from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes import PyNode, Transform
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
            self.assertEqual(str(each).rsplit("_", 1)[-1], "test")

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

    def _add_user_attrs(self):
        """Adds one user defined attr of each serialized flavour on the root."""
        self.root.add_attr("ud_string", dataType="string").set("abc")
        self.root.add_attr("ud_string_array", dataType="stringArray").set(["a", "b"])
        self.root.add_attr("ud_double_array", dataType="doubleArray").set([1.5, 2.5])
        self.root.add_attr("ud_double", attributeType="double").set(3.5)
        self.root.add_attr("ud_bool", attributeType="bool").set(True)
        self.joint2.t.set(1, 2, 3)

    @staticmethod
    def _dag_paths():
        """Returns the long name of every transform in the scene, cameras aside."""
        cameras = ("persp", "top", "front", "side")
        paths   = cmds.ls(type="transform", long=True)
        return sorted(p for p in paths if p.split("|")[1] not in cameras)

    def test_hierarchy_duplicate_short_names(self):
        # the fixture nests two joints of the same short name,
        # add the same name again in another branch.
        other = PyNode.create("joint", name="other_joint", parent=self.root)
        PyNode.create("joint", name=self.c_name, parent=other)

        expected  = self._dag_paths()
        hierarchy = self.root.serialize_hierarchy()

        self.new_scene()
        Transform.create_hierarchy(hierarchy)
        self.assertEqual(self._dag_paths(), expected)

    def test_hierarchy_duplicate_short_names_shapes(self):
        # locators and space transforms are created by their own commands
        for node_type in ("locator", "space_transform"):
            self.new_scene()
            root  = PyNode.create("transform", name="root")
            child = PyNode.create("transform", name="child", parent=root)
            PyNode.create("transform", name="child", parent=child)

            expected  = self._dag_paths()
            hierarchy = root.serialize_hierarchy()

            hierarchy[2].node_type = node_type

            self.new_scene()
            Transform.create_hierarchy(hierarchy)
            self.assertEqual(self._dag_paths(), expected)

    def test_hierarchy_locator_round_trip(self):
        # a transform holding a locator shape serializes as a "locator"
        # and comes back with a locator shape
        loc = PyNode(cmds.spaceLocator(name="probe_loc")[0])
        loc.set_parent(self.joint2)
        loc.t.set(1, 2, 3)

        hierarchy = self.root.serialize_hierarchy()
        self.assertEqual([x.node_type for x in hierarchy], ["joint", "joint", "joint", "locator"])

        self.new_scene()
        created = Transform.create_hierarchy(hierarchy)
        shapes  = cmds.listRelatives(created[3], s=True)
        self.assertEqual(cmds.nodeType(created[3]),          "transform")
        self.assertEqual([cmds.nodeType(s) for s in shapes], ["locator"])
        self.assertEqual(cmds.getAttr(f"{created[3]}.t")[0], (1, 2, 3))

    def test_hierarchy_compound_user_attrs_round_trip(self):
        root = self.root.name
        cmds.addAttr(root, ln="vec", at="double3")
        for axis in "XYZ":
            cmds.addAttr(root, ln=f"vec{axis}", at="double", parent="vec")
        cmds.addAttr(root, ln="grp",  at="compound", nc=2)
        cmds.addAttr(root, ln="grpA", at="double",   parent="grp")
        cmds.addAttr(root, ln="grpB", at="bool",     parent="grp")
        cmds.addAttr(root, ln="flat", at="double")
        cmds.setAttr(f"{root}.vec", 1, 2, 3)
        cmds.setAttr(f"{root}.grpA", 4.5)
        cmds.setAttr(f"{root}.grpB", True)
        cmds.setAttr(f"{root}.flat", 6.5)

        hierarchy = self.root.serialize_hierarchy()
        specs     = hierarchy[0].user_defined_attributes
        self.assertEqual(specs["vec"]["numberOfChildren"], 3)
        self.assertEqual(specs["grp"]["numberOfChildren"], 2)
        self.assertEqual(specs["vecX"]["parent"],          "vec")
        self.assertEqual(specs["grpB"]["parent"],          "grp")
        self.assertNotIn("value", specs["vec"])
        self.assertNotIn("parent", specs["flat"])

        self.new_scene()
        with self.assertNoLogs("rig.nodetypes.transform", level="WARNING"):
            root, _, _ = Transform.create_hierarchy(hierarchy)

        self.assertEqual(cmds.attributeQuery("vec", node=root, listChildren=True), ["vecX", "vecY", "vecZ"])
        self.assertEqual(cmds.attributeQuery("grp", node=root, listChildren=True), ["grpA", "grpB"])
        self.assertEqual(cmds.getAttr(f"{root}.vec"), [(1.0, 2.0, 3.0)])
        self.assertEqual(cmds.getAttr(f"{root}.grpA"), 4.5)
        self.assertEqual(cmds.getAttr(f"{root}.grpB"), True)
        self.assertEqual(cmds.getAttr(f"{root}.flat"), 6.5)

    def test_hierarchy_enum_and_multi_user_attrs_round_trip(self):
        root = self.root.name
        cmds.addAttr(root, ln="en", at="enum",   enumName="a=1:b=5:c")
        cmds.addAttr(root, ln="md", at="double", multi=True)
        cmds.addAttr(root, ln="ms", dt="string", multi=True)
        cmds.addAttr(root, ln="me", at="double", multi=True)  # left empty
        cmds.setAttr(f"{root}.en",    5)
        cmds.setAttr(f"{root}.md[0]", 1.5)
        cmds.setAttr(f"{root}.md[2]", 3.5)
        cmds.setAttr(f"{root}.ms[1]", "abc", type="string")

        hierarchy = self.root.serialize_hierarchy()
        specs     = hierarchy[0].user_defined_attributes
        self.assertEqual(specs["en"]["enumName"],      "a=1:b=5:c")
        self.assertEqual(specs["en"]["value"],         5)
        self.assertEqual(specs["md"]["attributeType"], "double")
        self.assertEqual(specs["md"]["value"],         [[0, 1.5], [2, 3.5]])
        self.assertEqual(specs["ms"]["dataType"],      "string")
        self.assertEqual(specs["ms"]["value"],         [[1, "abc"]])
        self.assertEqual(specs["me"]["value"],         [])
        for name in ("md", "ms", "me"):
            self.assertTrue(specs[name]["multi"])

        self.new_scene()
        with self.assertNoLogs("rig.nodetypes.transform", level="WARNING"):
            root, _, _ = Transform.create_hierarchy(hierarchy)

        self.assertEqual(cmds.attributeQuery("en", node=root, listEnum=True), ["a=1:b=5:c"])
        self.assertEqual(cmds.getAttr(f"{root}.en"), 5)
        for name in ("md", "ms", "me"):
            self.assertTrue(cmds.attributeQuery(name, node=root, multi=True))
        self.assertEqual(cmds.getAttr(f"{root}.md", multiIndices=True), [0, 2])
        self.assertEqual(cmds.getAttr(f"{root}.md[2]"), 3.5)
        self.assertEqual(cmds.getAttr(f"{root}.ms[1]"), "abc")
        self.assertEqual(cmds.getAttr(f"{root}.me", multiIndices=True), None)

    def test_hierarchy_serialize_skips_unsupported_nodes(self):
        # a constraint is a transform by inheritance only, it is not serialized
        driver = PyNode.create("transform", name="driver")
        cmds.parentConstraint(driver.long_name, self.joint3.long_name)

        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            hierarchy = self.root.serialize_hierarchy()

        self.assertEqual([x.node_type for x in hierarchy], ["joint", "joint", "joint"])
        self.assertEqual(len(logs.output), 1)
        self.assertIn("1 parentConstraint", logs.output[0])

    def test_hierarchy_create_skips_unsupported_nodes(self):
        # older files hold constraints, they are skipped along with what is under them
        hierarchy = self.root.serialize_hierarchy()

        hierarchy[1].node_type = "parentConstraint"

        self.new_scene()
        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            created = Transform.create_hierarchy(hierarchy)

        self.assertEqual(len(created), 1)
        self.assertEqual(self._dag_paths(), ["|root_joint"])
        self.assertEqual(cmds.ls(type="constraint"), [])
        self.assertEqual(len(logs.output), 1)
        self.assertIn("1 joint", logs.output[0])
        self.assertIn("1 parentConstraint", logs.output[0])

    def test_create_with_bad_parent(self):
        # nothing is created when the parent is missing or ambiguous
        before = self._dag_paths()
        for parent in ("not_a_node", self.c_name):
            with self.assertRaises(ValueError):
                PyNode.create("joint", name="orphan", parent=parent)
            self.assertEqual(self._dag_paths(), before)

    def test_hierarchy_user_attrs_round_trip(self):
        self._add_user_attrs()
        hierarchy = self.root.serialize_hierarchy()

        # typed attrs are stored as "dataType", the others as "attributeType"
        specs = hierarchy[0].user_defined_attributes
        for name in ("ud_string", "ud_string_array", "ud_double_array"):
            self.assertIn("dataType", specs[name])
            self.assertNotIn("attributeType", specs[name])
        for name in ("ud_double", "ud_bool"):
            self.assertIn("attributeType", specs[name])
            self.assertNotIn("dataType", specs[name])

        self.new_scene()
        with self.assertNoLogs("rig.nodetypes.transform", level="WARNING"):
            root, joint2, _ = Transform.create_hierarchy(hierarchy)

        self.assertEqual(cmds.getAttr(f"{root}.ud_string"),       "abc")
        self.assertEqual(cmds.getAttr(f"{root}.ud_string_array"), ["a", "b"])
        self.assertEqual(cmds.getAttr(f"{root}.ud_double_array"), [1.5, 2.5])
        self.assertEqual(cmds.getAttr(f"{root}.ud_double"),       3.5)
        self.assertEqual(cmds.getAttr(f"{root}.ud_bool"),         True)
        self.assertEqual(cmds.getAttr(f"{joint2}.t")[0],          (1, 2, 3))

    def test_hierarchy_legacy_attribute_type(self):
        self._add_user_attrs()
        hierarchy = self.root.serialize_hierarchy()

        # older files stored array data types as "attributeType"
        specs = hierarchy[0].user_defined_attributes
        for name in ("ud_string_array", "ud_double_array"):
            specs[name]["attributeType"] = specs[name].pop("dataType")

        self.new_scene()
        with self.assertNoLogs("rig.nodetypes.transform", level="WARNING"):
            root, joint2, _ = Transform.create_hierarchy(hierarchy)

        self.assertEqual(cmds.getAttr(f"{root}.ud_string_array"), ["a", "b"])
        self.assertEqual(cmds.getAttr(f"{root}.ud_double_array"), [1.5, 2.5])
        self.assertEqual(cmds.getAttr(f"{joint2}.t")[0],          (1, 2, 3))

    def test_hierarchy_bad_user_attr(self):
        self._add_user_attrs()
        hierarchy = self.root.serialize_hierarchy()

        # a bad attr is skipped, it must not stop the transforms
        # nor the attrs coming after it
        specs                                = hierarchy[0].user_defined_attributes
        bad                                  = {"attributeType": "notAType", "value": 1, "keyable": False}
        hierarchy[0].user_defined_attributes = {"ud_bad": bad, **specs}

        self.new_scene()
        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            root, joint2, _ = Transform.create_hierarchy(hierarchy)

        self.assertEqual(len(logs.output), 1)
        self.assertIn("ud_bad", logs.output[0])
        self.assertFalse(cmds.objExists(f"{root}.ud_bad"))
        self.assertEqual(cmds.getAttr(f"{root}.ud_string"), "abc")
        self.assertEqual(cmds.getAttr(f"{joint2}.t")[0], (1, 2, 3))

    def test_shapes(self):
        xform = PyNode(cmds.polyCube(name="test", ch=False)[0])
        self.root.set_parent(xform)
        shape = xform.get_children(type="mesh")[0]
        self.assertEqual(xform.get_shape(), shape)
        self.assertEqual(xform.get_shapes(), [shape])