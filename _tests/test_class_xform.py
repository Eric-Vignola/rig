import inspect
import unittest
import warnings
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes import Joint, Node, Transform
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.joint import replace_suffix
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


class TestXformNodes(MayaTestCase):
    """
    Transform node types' unit tests.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        # make a joint chain
        self.root   = Node.create("joint", name="root_joint")
        self.c_name = "child_joint"
        self.joint2 = Node.create("joint", name=self.c_name, parent=self.root)
        self.joint3 = Node.create("joint", name=self.c_name, parent=self.joint2)

    def test_attr(self):
        self.root.set_xform_attrs_locked(True)
        for attr in self.root.iter_xform_attrs():
            self.assertTrue(attr.is_locked())
        self.root.set_xform_attrs_locked(False)
        for attr in self.root.iter_xform_attrs():
            self.assertFalse(attr.is_locked())

    def test_matrix(self):
        # re-pinned (user decision 2026-10-29: local by default, world by keyword): the
        # world reads ask for it
        t = OpenMaya.MVector(1, 1, 1)
        self.joint2.t.set(*t)
        mat = self.joint2.get_matrix(world_space=True)
        mat = OpenMaya.MTransformationMatrix(mat)
        self.assertEqual(mat.translation(OpenMaya.MSpace.kWorld), t)

        mat = self.joint2.get_matrix(world_space=False)
        mat = OpenMaya.MTransformationMatrix(mat)
        self.assertEqual(mat.translation(OpenMaya.MSpace.kWorld), t)

        self.joint3.set_matrix(mat, world_space=False)
        mat = self.joint3.get_matrix(world_space=True)
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
        cube1 = Node(cmds.polyCube(name="cube1", ch=False)[0])
        cube1.set_parent(self.joint2)
        cube2 = Node(cmds.polyCube(name="cube2", ch=False)[0])
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
        p        = Node.create("transform")
        dup_root = self.root.duplicate_skeleton(parent=p)
        self.assertEqual(dup_root.get_parent(), p)
        self.assertEqual(len(list(dup_root.iter_joints())), 2)

        dup_root = self.root.duplicate_skeleton(include_list=[self.joint2])
        self.assertIsNone(dup_root.get_parent())
        self.assertEqual(len(list(dup_root.iter_joints())), 2)
        self.assertEqual(dup_root.get_children()[0].get_short_name(), self.joint2.get_short_name())

        dup_root.rename_skeleton("test")
        for each in dup_root.iter_joints(yield_self=True):
            self.assertEqual(str(each).rsplit("_", 1)[-1], "test")

        with self.assertRaises(RuntimeError):
            dup_root.rename_skeleton("a %_")

    def _namespaced_chain(self):
        """VLR_RIG:SK:root -> spine_01 -> head, in nested namespaces."""
        cmds.namespace(add="VLR_RIG")
        cmds.namespace(add="SK", parent="VLR_RIG")
        root  = Node.create("joint", name="VLR_RIG:SK:root")
        spine = Node.create("joint", name="VLR_RIG:SK:spine_01", parent=root)
        head  = Node.create("joint", name="VLR_RIG:SK:head", parent=spine)
        return root, spine, head

    @staticmethod
    def _joint_names(top):
        return [x.get_short_name() for x in top.iter_joints(yield_self=True)]

    def test_replace_suffix_keeps_namespace(self):
        self.assertEqual(replace_suffix("VLR_RIG:SK:root", "test"), "VLR_RIG:SK:root_test")
        self.assertEqual(replace_suffix("VLR_RIG:SK:spine_01", "test"), "VLR_RIG:SK:spine_test")
        self.assertEqual(replace_suffix("|grp|ns:arm_L_jnt", "ctl"), "ns:arm_L_ctl")
        self.assertEqual(replace_suffix("arm_L_jnt", "ctl"), "arm_L_ctl")
        self.assertEqual(replace_suffix("arm", "ctl"), "arm_ctl")

    def test_rename_skeleton_keeps_namespace(self):
        root, _, _ = self._namespaced_chain()
        root.rename_skeleton("test")
        self.assertEqual(
            self._joint_names(root),
            ["VLR_RIG:SK:root_test", "VLR_RIG:SK:spine_test", "VLR_RIG:SK:head_test"],
        )

    def test_duplicate_skeleton_namespace(self):
        root, _, _ = self._namespaced_chain()
        originals = self._joint_names(root)

        dup = root.duplicate_skeleton()
        self.assertEqual(self._joint_names(dup), ["root", "spine_01", "head"])
        self.assertEqual(self._joint_names(root), originals)

        dup = root.duplicate_skeleton(name="VLR_RIG:copy")
        self.assertEqual(dup.get_short_name(), "VLR_RIG:copy")

    def test_duplicate_skeleton_namespace_prefix(self):
        root, _, _ = self._namespaced_chain()
        dup = root.duplicate_skeleton(prefix="L")
        self.assertEqual(self._joint_names(dup), ["L_root", "L_spine_01", "L_head"])

    def test_duplicate_skeleton_namespace_suffix(self):
        root, _, _ = self._namespaced_chain()
        dup = root.duplicate_skeleton(suffix="DUP")
        self.assertEqual(self._joint_names(dup), ["root_DUP", "spine_DUP", "head_DUP"])

    def test_duplicate_skeleton_namespace_include_list(self):
        root, spine, _ = self._namespaced_chain()
        dup = root.duplicate_skeleton(include_list=[spine])
        self.assertEqual(self._joint_names(dup), ["root", "spine_01"])

        dup = root.duplicate_skeleton(include_list=["VLR_RIG:SK:spine_01"], prefix="L")
        self.assertEqual(self._joint_names(dup), ["L_root", "L_spine_01"])

    def test_duplicate_skeleton_include_list_with_prefix_or_suffix(self):
        root  = Node.create("joint", name="root_jnt")
        spine = Node.create("joint", name="spine_jnt", parent=root)
        Node.create("joint", name="head_jnt", parent=spine)

        dup = root.duplicate_skeleton(include_list=[spine], prefix="L")
        self.assertEqual(self._joint_names(dup), ["L_root_jnt", "L_spine_jnt"])

        dup = root.duplicate_skeleton(include_list=[spine], suffix="DUP")
        self.assertEqual(self._joint_names(dup), ["root_DUP", "spine_DUP"])

    def test_duplicate_geom(self):
        xform = Node(cmds.polyCube(ch=False)[0])
        s     = Node.create("objectSet")
        s.add_members([xform] + xform.get_children())
        self.assertEqual(set(s.get_members()), set([xform] + xform.get_children()))

        parent = Node.create("transform")
        xform.set_parent(parent)
        xform.find_attr("t").set(1, 2, 3)
        self.root.set_parent(xform)

        dup = xform.duplicate_geometry()

        self.assertIsNone(dup.get_parent())
        self.assertEqual(dup.get_short_name(), xform.get_short_name())
        self.assertEqual(len(dup.get_children()), 1)
        c = dup.get_children()[0]
        self.assertEqual(c.get_node_type(), "mesh")
        self.assertFalse(dup in s.get_members())
        self.assertFalse(c in s.get_members())

        self.assertEqual(dup.find_attr("t").get()[0],         (0, 0, 0))
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
        other = Node.create("joint", name="other_joint", parent=self.root)
        Node.create("joint", name=self.c_name, parent=other)

        expected  = self._dag_paths()
        hierarchy = self.root.serialize_hierarchy()

        self.new_scene()
        Transform.create_hierarchy(hierarchy)
        self.assertEqual(self._dag_paths(), expected)

    def test_hierarchy_duplicate_short_names_shapes(self):
        # a locator's transform is created by its own command
        self.new_scene()
        root  = Node.create("transform", name="root")
        child = Node.create("transform", name="child", parent=root)
        Node.create("transform", name="child", parent=child)

        expected  = self._dag_paths()
        hierarchy = root.serialize_hierarchy()

        hierarchy[2].node_type = "locator"

        self.new_scene()
        Transform.create_hierarchy(hierarchy)
        self.assertEqual(self._dag_paths(), expected)

    def test_hierarchy_locator_round_trip(self):
        # a transform holding a locator shape serializes as a "locator"
        # and comes back with a locator shape
        loc = Node(cmds.spaceLocator(name="probe_loc")[0])
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

    def test_hierarchy_under_a_parent_outside_the_data(self):
        # the data's roots go under a node the data does not hold and keep their world
        # matrices (on cgmath 2.0.0 it raised: rig wrote that node's name into the data,
        # where a parent is a uuid now; runs\r5\CG2)
        self.root.t.set(1, 2, 3)
        self.root.r.set(10, 20, 30)
        self.joint2.t.set(0, 2, 0)
        self.joint3.t.set(0, 0, 3)
        world     = [cmds.xform(x.get_long_name(), q=True, matrix=True, worldSpace=True)
                     for x in (self.root, self.joint2, self.joint3)]
        hierarchy = self.root.serialize_hierarchy()

        self.new_scene()
        outside = cmds.createNode("transform", name="outside")
        cmds.xform(outside, translation=(-4, 0, 2), rotation=(0, 45, 0))
        # re-pinned (user decision 2026-10-29): the world matrices are kept by keyword
        created = Transform.create_hierarchy(hierarchy, parent="outside", world_space=True)
        self.assertEqual(cmds.listRelatives(created[0], parent=True), ["outside"])
        self.assertEqual(len(hierarchy), 3)  # the caller's data is left as it was
        for name, matrix in zip(created, world):
            got = cmds.xform(name, q=True, matrix=True, worldSpace=True)
            for a, b in zip(got, matrix):
                self.assertAlmostEqual(a, b, places=6)

    def test_hierarchy_under_a_parent_outside_the_data_below_a_moved_group(self):
        # the outside node sits under a moved and rotated group: the roots are solved
        # against its world matrix (they were solved against its local one, and landed
        # off by the group's move: runs\r5\CG2\verify)
        self.root.t.set(1, 2, 3)
        self.root.r.set(10, 20, 30)
        self.joint2.t.set(0, 2, 0)
        world     = [cmds.xform(x.get_long_name(), q=True, matrix=True, worldSpace=True)
                     for x in (self.root, self.joint2, self.joint3)]
        hierarchy = self.root.serialize_hierarchy()

        self.new_scene()
        group   = cmds.createNode("transform", name="group")
        cmds.xform(group, translation=(0, 3, -1), rotation=(0, 45, 15))
        outside = cmds.createNode("transform", name="outside", parent=group)
        cmds.xform(outside, translation=(1, 0, 0), rotation=(30, 0, 0))
        created = Transform.create_hierarchy(hierarchy, parent="outside", world_space=True)
        self.assertEqual(cmds.listRelatives(created[0], parent=True), ["outside"])
        for name, matrix in zip(created, world):
            got = cmds.xform(name, q=True, matrix=True, worldSpace=True)
            for a, b in zip(got, matrix):
                self.assertAlmostEqual(a, b, places=6)

    def test_serialize_hierarchy_writes_parents_without_deprecation_warnings(self):
        # cgmath 2.0.0 names a parent parent_uuid; parent_node, the 1.0.4 name, is its deprecated
        # alias there and gone on 3.0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            hierarchy = self.root.serialize_hierarchy()
            self.assertIsNone(hierarchy[0].get_parent())
            self.assertIs(hierarchy[1].get_parent(), hierarchy[0])
            self.assertIs(hierarchy[2].get_parent(), hierarchy[1])

    def test_hierarchy_compound_user_attrs_round_trip(self):
        root = self.root.get_name()
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
        root = self.root.get_name()
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
        driver = Node.create("transform", name="driver")
        cmds.parentConstraint(driver.get_long_name(), self.joint3.get_long_name())

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
                Node.create("joint", name="orphan", parent=parent)
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
        xform = Node(cmds.polyCube(name="test", ch=False)[0])
        self.root.set_parent(xform)
        shape = xform.get_children(type="mesh")[0]
        self.assertEqual(xform.get_shape(), shape)
        self.assertEqual(xform.get_shapes(), [shape])


# --- the space convention (user decision 2026-10-29): local / object space by default, as
# Maya's API; world space with the keyword-only world_space=True. The setups and numbers are
# the probes' (tspace/probe: P1 matrices, P2 / P3.3 pivots, P3.1 / P3.2 matches, P4
# hierarchies, P6 bounding boxes). Each test failed against 65eb6f4's sources (negative
# control), except the guards.


def _make(node_type, name, parent=None, **values):
    """``cmds.createNode`` of `node_type`, named `name`, under `parent`, then `_set`; the name."""
    if parent is None:
        cmds.createNode(node_type, name=name)
    else:
        cmds.createNode(node_type, name=name, parent=parent)
    _set(name, **values)
    return name


def _set(name, **values):
    """Sets each ``attr=value`` of the node `name` (a tuple: a compound's children)."""
    for attr, value in values.items():
        if isinstance(value, tuple):
            cmds.setAttr(f"{name}.{attr}", *value)
        else:
            cmds.setAttr(f"{name}.{attr}", value)


def _matrix(name, attr="matrix"):
    """The matrix attribute `attr` of the node `name` (``matrix``, ``worldMatrix[0]`` ...), 4x4."""
    return np.array(cmds.getAttr(f"{name}.{attr}")).reshape(4, 4)


def _array(matrix):
    """An MMatrix, an MTransformationMatrix or 16 numbers, 4x4."""
    if isinstance(matrix, OpenMaya.MTransformationMatrix):
        matrix = matrix.asMatrix()
    return np.array(matrix).reshape(4, 4)


def _trs(t=(0, 0, 0), r=(0, 0, 0), s=(1, 1, 1)):
    """The matrix of a translate, a rotate (degrees, xyz) and a scale, 4x4."""
    matrix = OpenMaya.MTransformationMatrix()
    matrix.setScale(s, OpenMaya.MSpace.kTransform)
    matrix.setRotation(OpenMaya.MEulerRotation([np.radians(v) for v in r]))
    matrix.setTranslation(OpenMaya.MVector(*t), OpenMaya.MSpace.kTransform)
    return _array(matrix)


def _point(point):
    """An MPoint's x, y, z."""
    return np.array([point.x, point.y, point.z])


def _box(box):
    """An MBoundingBox as xform's six numbers: min x, y, z, max x, y, z."""
    return [box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z]


class TestTheSpaceKeyword(MayaTestCase):
    """Every world_space= of the transforms is keyword-only and False by default: a bool given
    by position is a TypeError (``get_matrix(True)`` was the world matrix)."""

    TEST_START_NEW_SCENE = True

    def test_every_world_space_is_a_keyword_defaulting_to_false(self):
        for function in (
            DAGNode.get_bounding_box, Transform.get_matrix, Transform.set_matrix,
            Transform.get_rotate_pivot, Transform.get_scale_pivot, Transform.set_pivots,
            Transform.match_matrix, Joint.match_hierarchy, Transform.create_hierarchy,
            Transform.load_hierarchy,
        ):
            with self.subTest(function.__qualname__):
                parameter = inspect.signature(function).parameters["world_space"]
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameter.default, False)
        parameter = inspect.signature(Transform.get_matrix).parameters["as_transform_matrix"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_a_bool_by_position_is_a_type_error(self):
        grp  = Transform.create(name="grp")
        cube = Node(cmds.polyCube(name="cube", ch=False)[0])
        root = Joint.create(name="root")
        data = root.serialize_hierarchy()
        cube.t.set(1, 2, 3)
        nodes, world = cmds.ls(long=True), _matrix("cube", "worldMatrix[0]")
        for label, call in (
            ("get_bounding_box", lambda: cube.get_bounding_box(True)),
            ("get_matrix", lambda: cube.get_matrix(True)),
            ("get_matrix as_transform_matrix", lambda: cube.get_matrix(False, True)),
            ("set_matrix", lambda: cube.set_matrix(np.eye(4).ravel().tolist(), True)),
            ("get_rotate_pivot", lambda: cube.get_rotate_pivot(True)),
            ("get_scale_pivot", lambda: cube.get_scale_pivot(True)),
            ("set_pivots", lambda: cube.set_pivots((5, 5, 5), True)),
            ("match_matrix", lambda: cube.match_matrix(grp, True)),
            ("match_hierarchy", lambda: root.match_hierarchy(root, True)),
            ("create_hierarchy", lambda: Transform.create_hierarchy(data, "grp", True)),
        ):
            with self.subTest(label):
                with self.assertRaises(TypeError):
                    call()
        self.assertEqual(cmds.ls(long=True), nodes)
        np.testing.assert_allclose(_matrix("cube", "worldMatrix[0]"), world, atol=1e-12)
        self.assertEqual(cmds.xform("cube", query=True, objectSpace=True, rotatePivot=True), [0, 0, 0])


class TestMatrixSpaces(UndoWalk, MayaTestCase):
    """get_matrix / set_matrix: the local matrix (``.matrix``, the offset parent matrix left
    out) by default, the world matrix with world_space=True (probes P1.1-P1.4)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # under a moved, rotated, non-uniformly scaled group: its world matrix holds a shear
        _make("transform", "grp", t=(3, -2, 5), r=(30, 45, 60), s=(2, 0.5, 1.5))
        _make("transform", "xf", "grp", t=(1, 2, 3), r=(10, 20, 30), s=(1.2, 0.8, 1.1))
        self.xf = Transform("xf")

    def test_get_matrix_reads_the_local_matrix_by_default(self):
        local, world = _matrix("xf"), _matrix("xf", "worldMatrix[0]")
        self.assertGreater(np.abs(local - world).max(), 1.0)
        np.testing.assert_allclose(_array(self.xf.get_matrix()), local, atol=1e-12)
        np.testing.assert_allclose(_array(self.xf.get_matrix(world_space=True)), world, atol=1e-12)
        transform = self.xf.get_matrix(as_transform_matrix=True)
        self.assertIsInstance(transform, OpenMaya.MTransformationMatrix)
        np.testing.assert_allclose(_array(transform), local, atol=1e-12)
        transform = self.xf.get_matrix(world_space=True, as_transform_matrix=True)
        np.testing.assert_allclose(_array(transform), world, atol=1e-12)

    def test_the_local_matrix_leaves_the_offset_parent_matrix_out(self):
        offset = _trs(t=(0, 0, 3), r=(0, 90, 0))
        cmds.setAttr("xf.offsetParentMatrix", offset.ravel().tolist(), type="matrix")
        local = _array(self.xf.get_matrix())
        np.testing.assert_allclose(local, _matrix("xf"), atol=1e-12)
        np.testing.assert_allclose(local @ offset, _matrix("xf", "dagLocalMatrix"), atol=1e-12)
        self.assertGreater(np.abs(local - _matrix("xf", "dagLocalMatrix")).max(), 1.0)
        np.testing.assert_allclose(_array(self.xf.get_matrix(world_space=True)), _matrix("xf", "worldMatrix[0]"),
                                   atol=1e-12)

    def test_set_matrix_sets_the_local_matrix_by_default(self):
        target = _trs(t=(-1, 4, 2), r=(15, -30, 45), s=(1, 2, 0.5))
        self.xf.set_matrix(target.ravel().tolist())
        np.testing.assert_allclose(_matrix("xf"), target, atol=1e-9)
        self.xf.set_matrix(OpenMaya.MMatrix(target.ravel().tolist()), world_space=True)
        np.testing.assert_allclose(_matrix("xf", "worldMatrix[0]"), target, atol=1e-9)

    def test_a_local_set_keeps_the_offset_parent_matrix(self):
        offset = _trs(t=(0, 0, 3))
        cmds.setAttr("xf.offsetParentMatrix", offset.ravel().tolist(), type="matrix")
        target = _trs(t=(1, 1, 1), r=(0, 45, 0))
        self.xf.set_matrix(OpenMaya.MTransformationMatrix(OpenMaya.MMatrix(target.ravel().tolist())))
        np.testing.assert_allclose(_matrix("xf"), target, atol=1e-9)
        np.testing.assert_allclose(_matrix("xf", "offsetParentMatrix"), offset, atol=1e-12)

    def test_each_space_round_trips(self):
        """guard: either space, by keyword, sets back what it reads."""
        world = _matrix("xf", "worldMatrix[0]")
        for world_space in (False, True):
            with self.subTest(world_space=world_space):
                self.xf.set_matrix(self.xf.get_matrix(world_space=world_space), world_space=world_space)
                np.testing.assert_allclose(_matrix("xf", "worldMatrix[0]"), world, atol=1e-9)

    def test_a_world_set_redoes_where_it_was_set(self):
        # (Maya's redo of xform -ws -m takes the parent's matrix out twice: 5.6 to 12.4 off in the
        # review's probes; the world matrix is set as the local one that puts the node there)
        offset = _trs(t=(0, 0, 3), r=(0, 90, 0))
        cmds.setAttr("xf.offsetParentMatrix", offset.ravel().tolist(), type="matrix")
        target = _trs(t=(-1, 4, 2), r=(15, -30, 45), s=(1, 2, 0.5))
        self.xf.set_matrix(target.ravel().tolist(), world_space=True)
        np.testing.assert_allclose(_matrix("xf", "worldMatrix[0]"), target, atol=1e-9)
        self.undo_steps(1)
        self.redo_steps(1)
        np.testing.assert_allclose(_matrix("xf", "worldMatrix[0]"), target, atol=1e-9)
        np.testing.assert_allclose(_matrix("xf", "offsetParentMatrix"), offset, atol=1e-12)


class TestSetMatrixOnACompensatedJoint(UndoWalk, MayaTestCase):
    """set_matrix on a joint whose segment scale compensate is on, its inverse scale wired to a
    non-uniformly scaled parent joint (as cmds.parent and the joint tools leave it): exact in both
    spaces (``xform -m`` alone moved it 2.24 world units for its own matrix, 3.43 / 2.38 off for a
    new one; probes P1.2, P1.7, P1.11, P1.12), one undo step, the compensation back on even when
    the set fails; under a negatively scaled one its channels kept; every other node keeps the
    plain ``xform -m``."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _make("transform", "grp", t=(3, -2, 5), r=(30, 45, 60), s=(2, 0.5, 1.5))
        _make("joint", "pj", "grp", t=(0.5, 1, 0), r=(5, 10, 15), jo=(20, 0, -10), s=(1, 2, 0.5))
        _make("joint", "j_on", "pj")
        cmds.connectAttr("pj.scale", "j_on.inverseScale")
        _set("j_on", ssc=1, t=(2, 0.5, -1), r=(15, -30, 45), jo=(30, 20, -10), ra=(10, 5, -20), ro=1,
             s=(1.5, 0.7, 1.2))
        self.joint = Node("j_on")

    @staticmethod
    def _target(name="j_on"):
        """A new local matrix the joint can hold (S Rx(20) S^-1 L, moved): its rotate changes."""
        scale  = np.diag(list(cmds.getAttr(f"{name}.s")[0]) + [1.0])
        target = scale @ _trs(r=(20, 0, 0)) @ np.linalg.inv(scale) @ _matrix(name)
        target[3, :3] += (0.3, -0.2, 0.1)
        return target

    def assert_joint_kept(self):
        """The orient, rotate axis, compensation and inverse scale wiring as setUp left them."""
        np.testing.assert_allclose(cmds.getAttr("j_on.jo")[0], (30, 20, -10), atol=1e-9)
        np.testing.assert_allclose(cmds.getAttr("j_on.ra")[0], (10, 5, -20), atol=1e-9)
        self.assertIs(cmds.getAttr("j_on.ssc"), True)
        self.assertEqual(cmds.listConnections("j_on.inverseScale", plugs=True), ["pj.scale"])

    def test_its_own_matrix_round_trips_in_each_space(self):
        world = _matrix("j_on", "worldMatrix[0]")
        for world_space in (False, True):
            with self.subTest(world_space=world_space):
                self.joint.set_matrix(self.joint.get_matrix(world_space=world_space), world_space=world_space)
                np.testing.assert_allclose(_matrix("j_on", "worldMatrix[0]"), world, atol=1e-9)
                self.assert_joint_kept()

    def test_a_new_matrix_lands_in_each_space(self):
        for world_space in (False, True):
            with self.subTest(world_space=world_space):
                target = self._target()
                if world_space:
                    target = target @ _matrix("j_on", "parentMatrix[0]")
                self.joint.set_matrix(target.ravel().tolist(), world_space=world_space)
                np.testing.assert_allclose(_matrix("j_on", "worldMatrix[0]" if world_space else "matrix"), target,
                                           atol=1e-9)
                self.assert_joint_kept()

    def test_one_undo_step(self):
        channels = ("t", "r", "s", "ssc")
        before   = {at: cmds.getAttr(f"j_on.{at}") for at in channels}
        target   = self._target()
        self.joint.set_matrix(target.ravel().tolist())
        after = {at: cmds.getAttr(f"j_on.{at}") for at in channels}
        self.assertEqual(self.undo_name(), "rig.Joint.set_matrix")
        self.undo_steps(1)
        self.assertEqual({at: cmds.getAttr(f"j_on.{at}") for at in channels}, before)
        self.redo_steps(1)
        self.assertEqual({at: cmds.getAttr(f"j_on.{at}") for at in channels}, after)
        np.testing.assert_allclose(_matrix("j_on"), target, atol=1e-9)

    def test_the_compensation_comes_back_on_when_the_set_fails(self):
        """guard: nothing toggled the compensation before, so nothing was left off."""
        before = {at: cmds.getAttr(f"j_on.{at}") for at in ("t", "r", "s")}
        target = self._target()
        with mock.patch.object(cmds, "xform", side_effect=RuntimeError("xform failed")):
            with self.assertRaisesRegex(RuntimeError, "^xform failed$"):
                self.joint.set_matrix(target.ravel().tolist())
        self.assertEqual({at: cmds.getAttr(f"j_on.{at}") for at in before}, before)
        self.assert_joint_kept()

    def test_a_negative_inverse_scale_keeps_the_channels(self):
        # under a parent joint scaled (-1, -1, -1), xform -m alone placed the joint but flipped
        # its scale and rotate, which moved the joint its scale compensates (1.61, the review's
        # probe p4b)
        _make("joint", "nj", "grp", t=(1, 0, 0), r=(0, 30, 0), s=(-1, -1, -1))
        _make("joint", "jn", "nj", t=(0, 2, 0), r=(15, -30, 45), s=(1.5, 0.7, 1.2), jo=(30, 20, -10))
        cmds.connectAttr("nj.scale", "jn.inverseScale")
        _make("joint", "jc", "jn", t=(0, 1.5, 0), r=(-20, 35, 10), s=(1.1, 0.9, 1.4), jo=(0, 0, 30))
        cmds.connectAttr("jn.scale", "jc.inverseScale")
        channels = {at: cmds.getAttr(f"jn.{at}")[0] for at in ("t", "r", "s")}
        worlds   = [_matrix(name, "worldMatrix[0]") for name in ("jn", "jc")]
        for world_space in (False, True):
            with self.subTest(world_space=world_space):
                Node("jn").set_matrix(Node("jn").get_matrix(world_space=world_space), world_space=world_space)
                for at, value in channels.items():
                    np.testing.assert_allclose(cmds.getAttr(f"jn.{at}")[0], value, atol=1e-9, err_msg=at)
                for name, world in zip(("jn", "jc"), worlds):
                    np.testing.assert_allclose(_matrix(name, "worldMatrix[0]"), world, atol=1e-9, err_msg=name)
                self.assertIs(cmds.getAttr("jn.ssc"), True)

    def test_a_locked_compensation_is_an_error_and_nothing_is_set(self):
        # (xform -m alone set it, misplaced)
        before = _matrix("j_on")
        cmds.setAttr("j_on.ssc", lock=True)
        with self.assertRaisesRegex(RuntimeError, "segmentScaleCompensate' is locked or connected"):
            self.joint.set_matrix(self._target().ravel().tolist())
        np.testing.assert_array_equal(_matrix("j_on"), before)
        self.assert_joint_kept()

    def test_every_other_node_keeps_the_plain_set(self):
        # compensation off; on, under a uniformly scaled parent joint; on, under a transform (its
        # inverse scale unwired); a transform: xform -m alone sets them exactly, untouched
        _make("joint", "j_off", "pj", ssc=0, t=(1, 0, 0), jo=(10, 0, 0), s=(1.5, 0.7, 1.2))
        cmds.connectAttr("pj.scale", "j_off.inverseScale")
        _make("joint", "uj", "grp", s=(2, 2, 2))
        _make("joint", "j_uniform", "uj", t=(1, 0, 0), jo=(0, 30, 0), s=(1.5, 0.7, 1.2))
        cmds.connectAttr("uj.scale", "j_uniform.inverseScale")
        _make("joint", "j_under_xf", "grp", t=(0, 1, 0), jo=(0, 0, 15), s=(1.5, 0.7, 1.2))
        for name in ("j_off", "j_uniform", "j_under_xf", "grp"):
            for world_space in (False, True):
                with self.subTest(name=name, world_space=world_space):
                    target = self._target(name)
                    if world_space:
                        target = target @ _matrix(name, "parentMatrix[0]")
                    with mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as set_attr:
                        Node(name).set_matrix(target.ravel().tolist(), world_space=world_space)
                    got = _matrix(name, "worldMatrix[0]" if world_space else "matrix")
                    np.testing.assert_allclose(got, target, atol=1e-9)
                    self.assertEqual(set_attr.call_count, 0)
        # (the compensated joint's set is seen: its compensation off, then on)
        with mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as set_attr:
            self.joint.set_matrix(self._target().ravel().tolist())
        self.assertEqual([call.args for call in set_attr.call_args_list],
                         [("j_on.segmentScaleCompensate", False), ("j_on.segmentScaleCompensate", True)])


class TestPivotSpaces(MayaTestCase):
    """The pivots: in the node's object space by default (``xform -os``), in world space with
    world_space=True; set_pivots sets in the space the getters read and takes their MPoint, so a
    pivot read and set back stays where it is (probes P2.1-P2.7, P3.3: the local pair was
    asymmetric, 0.078 / 0.134 off, and an MPoint a TypeError)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _make("transform", "grp", t=(3, -2, 5), r=(30, 45, 60), s=(2, 0.5, 1.5))
        _make("transform", "xf", "grp", t=(1, 2, 3), r=(10, 20, 30), s=(1.2, 0.8, 1.1), ro=2, ra=(15, -25, 35),
              rp=(0.5, 1, -0.5), sp=(-1, 0.25, 0.75), rpt=(0.1, 0.2, 0.3), spt=(-0.2, 0.1, 0.05))
        self.xf = Transform("xf")

    def test_the_getters_read_object_space_by_default(self):
        world = _matrix("xf", "worldMatrix[0]")
        for flag, getter in (("rotatePivot", self.xf.get_rotate_pivot), ("scalePivot", self.xf.get_scale_pivot)):
            with self.subTest(flag):
                local, in_world = getter(), getter(world_space=True)
                self.assertIsInstance(local, OpenMaya.MPoint)
                np.testing.assert_allclose(_point(local), cmds.xform("xf", query=True, os=True, **{flag: True}),
                                           atol=1e-9)
                np.testing.assert_allclose(_point(in_world), cmds.xform("xf", query=True, ws=True, **{flag: True}),
                                           atol=1e-9)
                # a point in the node's own frame: times its world matrix, the world one
                np.testing.assert_allclose((np.append(_point(local), 1.0) @ world)[:3], _point(in_world), atol=1e-9)

    def test_set_pivots_sets_object_space_by_default(self):
        world = _matrix("xf", "worldMatrix[0]")
        self.xf.set_pivots((1, 1, 1))
        np.testing.assert_allclose(_point(self.xf.get_rotate_pivot()), (1, 1, 1), atol=1e-9)
        np.testing.assert_allclose(_point(self.xf.get_scale_pivot()), (1, 1, 1), atol=1e-9)
        self.xf.set_pivots(OpenMaya.MPoint(2, 4, 6, 2))
        np.testing.assert_allclose(_point(self.xf.get_rotate_pivot()), (1, 2, 3), atol=1e-9)
        self.xf.set_pivots(OpenMaya.MVector(2, 0, -1), world_space=True)
        np.testing.assert_allclose(_point(self.xf.get_rotate_pivot(world_space=True)), (2, 0, -1), atol=1e-9)
        np.testing.assert_allclose(_point(self.xf.get_scale_pivot(world_space=True)), (2, 0, -1), atol=1e-9)
        # the node never moved
        np.testing.assert_allclose(_matrix("xf", "worldMatrix[0]"), world, atol=1e-9)

    def test_a_pivot_read_and_set_back_stays_in_each_space(self):
        for world_space in (False, True):
            with self.subTest(world_space=world_space):
                local, world = self.xf.get_rotate_pivot(), self.xf.get_rotate_pivot(world_space=True)
                self.xf.set_pivots(self.xf.get_rotate_pivot(world_space=world_space), world_space=world_space)
                np.testing.assert_allclose(_point(self.xf.get_rotate_pivot()), _point(local), atol=1e-9)
                np.testing.assert_allclose(_point(self.xf.get_rotate_pivot(world_space=True)), _point(world), atol=1e-9)
                np.testing.assert_allclose(_point(self.xf.get_scale_pivot()), _point(local), atol=1e-9)

    def test_duplicate_geometry_puts_the_pivots_at_its_own_origin(self):
        # (its set_pivots((0, 0, 0)) is the object origin now; it was the world one)
        moved = Transform.create(name="moved")
        moved.t.set(1, 2, 3)
        moved.r.set(0, 45, 0)
        cube = Node(cmds.polyCube(name="cube", ch=False)[0])
        cube.t.set(4, 0, 0)
        cube.r.set(10, 0, 0)
        cmds.xform("cube", pivots=(5, 5, 5), worldSpace=True)
        dup = cube.duplicate_geometry(parent=moved)
        for label, getter in (("rotate", dup.get_rotate_pivot), ("scale", dup.get_scale_pivot)):
            with self.subTest(label):
                self.assertEqual(getter(), OpenMaya.MPoint())
                np.testing.assert_allclose(_point(getter(world_space=True)), (1, 2, 3), atol=1e-9)


class TestMatchMatrix(MayaTestCase):
    """match_matrix: the local matrix by default, a pose copy (the target's place only when both
    share a parent space); the world matrix with world_space=True (probe P3.1: 16.0 off under
    another parent in local)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _make("transform", "grpA", t=(10, 0, 0), r=(0, 90, 0))
        _make("transform", "grpB", t=(-5, 3, 0), r=(45, 0, 0), s=(2, 2, 2))
        _make("transform", "src", "grpA", t=(1, 2, 3), r=(10, 20, 30))
        _make("transform", "sib", "grpA")
        _make("transform", "dst", "grpB")
        self.src = Transform("src")

    def test_the_local_match_is_a_pose_copy(self):
        """guard: local was already match_matrix's default."""
        world = _matrix("src", "worldMatrix[0]")
        Transform("sib").match_matrix(self.src)
        np.testing.assert_allclose(_matrix("sib", "worldMatrix[0]"), world, atol=1e-9)
        Transform("dst").match_matrix(self.src)
        np.testing.assert_allclose(_matrix("dst"), _matrix("src"), atol=1e-9)
        self.assertGreater(np.abs(_matrix("dst", "worldMatrix[0]") - world).max(), 10)

    def test_the_world_match_snaps(self):
        """guard: world_space=True, by keyword, as before."""
        Transform("dst").match_matrix(self.src, world_space=True)
        np.testing.assert_allclose(_matrix("dst", "worldMatrix[0]"), _matrix("src", "worldMatrix[0]"), atol=1e-9)


def _chain(prefix, parent, posed):
    """Four joints under `parent`, each inverse scale wired to its parent joint's scale (as
    cmds.parent wires it); the names."""
    names, up = [], parent
    for i in range(4):
        name = _make("joint", f"{prefix}{i}", up, t=(2, 0, 0) if i else (0, 0, 0),
                     jo=((0, 0, 10), (0, 0, -20), (0, 15, 0), (0, 0, 0))[i])
        if up != parent:
            cmds.connectAttr(f"{up}.scale", f"{name}.inverseScale")
        if posed:
            _set(name, r=((0, 0, 30), (0, 40, 0), (20, 0, 0), (0, 0, 0))[i])
        names.append(name)
        up = name
    return names


class TestMatchHierarchy(UndoWalk, MayaTestCase):
    """Joint.match_hierarchy from any joint of either chain, a root included; the roots are
    matched too, parents before children (probe P3.2: an AttributeError on a root, the root never
    matched, 1.91 / 1.87 off in world space: listRelatives -ad lists the deepest first); redone
    where it matched."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _make("transform", "grpA", t=(10, 0, 0), r=(0, 90, 0))
        _make("transform", "grpB", t=(-5, 3, 0), r=(45, 0, 0), s=(2, 2, 2))
        self.posed = _chain("A", "grpA", posed=True)
        self.rest  = _chain("B", "grpB", posed=False)

    def assert_matched(self, attr):
        for posed, rest in zip(self.posed, self.rest):
            np.testing.assert_allclose(_matrix(rest, attr), _matrix(posed, attr), atol=1e-9, err_msg=rest)

    def test_from_a_root_each_local_matrix(self):
        Joint("B0").match_hierarchy("A0")
        self.assert_matched("matrix")

    def test_from_a_child_the_root_too(self):
        Joint("B1").match_hierarchy(Joint("A2"))
        self.assert_matched("matrix")

    def test_world_space_from_a_root(self):
        Joint("B0").match_hierarchy("A0", world_space=True)
        self.assert_matched("worldMatrix[0]")

    def test_world_space_from_a_child(self):
        Joint("B2").match_hierarchy("A1", world_space=True)
        self.assert_matched("worldMatrix[0]")

    def test_another_joint_count_raises(self):
        _make("joint", "B4", "B3")
        for start in ("B0", "B3"):
            with self.subTest(start=start):
                with self.assertRaisesRegex(RuntimeError, "^mismatch in number of joints$"):
                    Joint(start).match_hierarchy("A0")

    def test_a_world_match_redoes_where_it_matched(self):
        # one step per joint, the scene evaluated after each as the viewport does (the redo of
        # xform -ws -m took each parent's matrix out twice; one step around the joints redoes them
        # with their inverse scales from before the redo: 0.94 / 1.44 off in the fix's probe q4)
        Joint("B1").match_hierarchy("A2", world_space=True)
        matched = [_matrix(name, "worldMatrix[0]") for name in self.rest]
        for step in (self.undo_steps, self.redo_steps):
            for _ in self.rest:
                step(1)
                for name in self.rest:
                    _matrix(name, "worldMatrix[0]")
        for name, world in zip(self.rest, matched):
            np.testing.assert_allclose(_matrix(name, "worldMatrix[0]"), world, atol=1e-9, err_msg=name)


class TestBoundingBoxSpaces(MayaTestCase):
    """DAGNode.get_bounding_box: Maya's box in the parent's space by default (``xform -q -bb
    -os``); with world_space=True the world box of the geometry, every offset parent matrix
    counted (probe P6: it was (0.5, 99.5, -0.5) for (9.5, 101, -0.5), the parent's box carried by
    the grandparent's world matrix)."""

    TEST_START_NEW_SCENE = True

    WORLD = (9.5, 101, -0.5, 10.5, 103, 0.5)

    def setUp(self):
        super().setUp()
        _make("transform", "G", t=(0, 100, 0))
        _make("transform", "P", "G", t=(10, 0, 0), r=(0, 0, 90), s=(2, 1, 1))
        cmds.parent(cmds.polyCube(name="cube", ch=False)[0], "P", relative=True)
        _set("cube", t=(1, 0, 0))

    def test_the_box_in_the_parent_s_space_by_default(self):
        box = _box(Node("cube").get_bounding_box())
        np.testing.assert_allclose(box, (0.5, -0.5, -0.5, 1.5, 0.5, 0.5), atol=1e-9)
        np.testing.assert_allclose(box, cmds.xform("cube", query=True, boundingBox=True, objectSpace=True), atol=1e-9)
        np.testing.assert_allclose(_box(Node("cubeShape").get_bounding_box()), (-0.5, -0.5, -0.5, 0.5, 0.5, 0.5),
                                   atol=1e-9)

    def test_the_world_box(self):
        for name in ("cube", "cubeShape", "P", "G"):
            with self.subTest(name):
                np.testing.assert_allclose(_box(Node(name).get_bounding_box(world_space=True)), self.WORLD, atol=1e-9)

    def test_the_world_box_counts_the_offset_parent_matrices(self):
        # rig places nodes by their offsets: Maya's own boxes of the groups above leave them out
        cmds.setAttr("cube.offsetParentMatrix", _trs(t=(0, 0, 5)).ravel().tolist(), type="matrix")
        cmds.setAttr("P.offsetParentMatrix", _trs(t=(0, 3, 0)).ravel().tolist(), type="matrix")
        for name in ("cube", "cubeShape", "P", "G"):
            with self.subTest(name):
                np.testing.assert_allclose(_box(Node(name).get_bounding_box(world_space=True)),
                                           (9.5, 104, 4.5, 10.5, 106, 5.5), atol=1e-9)

    def test_a_node_without_geometry_gives_its_own_box_in_world_space(self):
        # a joint at (0, 3, 0) under P: P's x axis is the world y, doubled; its y axis the world -x
        _make("joint", "jnt", "P", t=(0, 3, 0))
        local = _box(Node("jnt").get_bounding_box())
        half  = (local[3] - local[0]) / 2
        np.testing.assert_allclose(local, (-half, 3 - half, -half, half, 3 + half, half), atol=1e-9)
        np.testing.assert_allclose(_box(Node("jnt").get_bounding_box(world_space=True)),
                                   (7 - half, 100 - 2 * half, -half, 7 + half, 100 + 2 * half, half), atol=1e-9)

    def test_the_world_box_leaves_out_the_intermediate_and_hidden_shapes(self):
        # a cube skinned to a joint moved up 10 (its Orig shape stays at the bind pose, y -0.5 to
        # 0.5) and a hidden cube beside it, under a moved group
        _make("transform", "S", t=(5, 0, 0))
        cmds.parent(cmds.polyCube(name="skinned", ch=False)[0], "S", relative=True)
        cmds.skinCluster(cmds.createNode("joint", name="j1"), "skinned", toSelectedBones=True)
        cmds.setAttr("j1.ty", 10)
        cmds.parent(cmds.polyCube(name="hidden", ch=False)[0], "S", relative=True)
        _set("hidden", tx=20, v=False)
        for name in ("skinned", "S"):
            with self.subTest(name):
                np.testing.assert_allclose(_box(Node(name).get_bounding_box(world_space=True)),
                                           (4.5, 9.5, -0.5, 5.5, 10.5, 0.5), atol=1e-9)


class TestHierarchySpaces(MayaTestCase):
    """create_hierarchy / load_hierarchy under a parent: by default the data's values as local
    values (the data is local: made again under the node it was serialized under, it lands where
    it was); with world_space=True each root keeps its world matrix in the data, its values read
    as world values (probes P4.2-P4.6)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _make("transform", "pgrp", t=(5, 1, -2), r=(0, 30, 0), s=(1, 2, 1))
        _make("joint", "root", "pgrp", t=(1, 0, 0), jo=(0, 0, 30))
        _make("joint", "mid", "root", t=(2, 0, 0), r=(0, 0, 20), jo=(10, 0, 0), s=(1, 1.5, 1), ro=1)
        cmds.connectAttr("root.scale", "mid.inverseScale")
        _make("transform", "tgt", t=(-3, 4, 1), r=(20, 0, 0))
        self.local = [_matrix(name) for name in ("root", "mid")]
        self.world = [_matrix(name, "worldMatrix[0]") for name in ("root", "mid")]
        self.data  = Transform("root").serialize_hierarchy()

    def makers(self, **kwargs):
        return (
            ("create_hierarchy", lambda parent: Transform.create_hierarchy(self.data, parent=parent, uuids=False, **kwargs)),
            ("load_hierarchy", lambda parent: Transform.load_hierarchy(self.data, parent=parent, uuids=False, **kwargs)),
        )

    def test_by_default_the_values_are_local_under_the_parent(self):
        for label, make in self.makers():
            with self.subTest(label):
                made = make("pgrp")
                for node, world in zip(made, self.world):
                    np.testing.assert_allclose(_matrix(node.get_long_name(), "worldMatrix[0]"), world, atol=1e-9)
                made = make("tgt")
                for node, local in zip(made, self.local):
                    np.testing.assert_allclose(_matrix(node.get_long_name()), local, atol=1e-9)

    def test_world_space_keeps_each_root_s_world_matrix_in_the_data(self):
        """guard: world_space=True, by keyword, as the old default did."""
        for label, make in self.makers(world_space=True):
            with self.subTest(label):
                made = make("tgt")
                np.testing.assert_allclose(_matrix(made[0].get_long_name(), "worldMatrix[0]"), self.local[0], atol=1e-9)
                np.testing.assert_allclose(_matrix(made[1].get_long_name()), self.local[1], atol=1e-9)