"""``create_hierarchy`` returns typed nodes; a ``List`` casts what a user gives it (HL, 2026-10-16).

The user's two reports:

* ``Transform.create(name)`` returns a ``Transform`` node, but ``Transform.create_hierarchy(data)``
  returned a list of str. It returns a ``List`` of the typed nodes it made (``Joint``,
  ``Transform``, the locator's transform, the space_transform box), parents first, as
  ``load_hierarchy`` does.
* ``List(Node("root").serialize_hierarchy())`` was a List of ``TransformData`` (``lst[1].tx``
  an AttributeError). A List holds numbers and rig objects: a str is cast as before, an
  OpenMaya object through ``Node()`` / ``Plug()``, any other object through its ``str``
  (``str(TransformData)`` is its node's name), and an object that names nothing is a TypeError
  naming its index, its type and its str, with nothing added. ``List(items)``, ``append``,
  ``extend``, ``insert``, ``+=`` and item / slice assignment cast alike.

* `TestUserExamples`: the two reports, as the user wrote them.
* `TestCreateHierarchyReturn`: the element types, parents first, renamed duplicates, under a
  parent, inside a container.
* `TestUserCasting`: a HierarchyData, a list of TransformData, the API objects, every way to add
  an element, an object that names nothing, a mix, numeric values, nested sequences.
* `TestValueLists`: the Lists rig builds from values keep them (broadcasts of vectors, matrices,
  strings and enums, Python values, bound methods, operators, slices, ``multi[:]``, a
  vectorized call's and a command's results).
* `TestCost`: ``List(nodes)`` and ``List(range(1000))`` cost no more than the loop they ran
  before.

Negative control: on fa2afe3 (runs\\features\\hierlist\\negctl.json) the tests of the two
reports fail; the value-List tests pass there too (they pin what is kept).
"""

import numbers
import time
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya

from rig import List, Node, container
from rig._internal import list as list_module
from rig._internal.memoize import vectorize
from rig._internal.plug import Plug
from rig._tests._base import MayaTestCase
from rig.bridges import commands as rc
from rig.nodetypes import Joint, Transform
from rig.nodetypes.errors import NodeNotFoundError


def _skeleton():
    """root (a transform) > pelvis (a joint, tx 3.5) > probe (a locator) and box (a
    space_transform, from the data); the HierarchyData of it."""
    root   = Transform.create(name="root")
    pelvis = Joint.create(name="pelvis", parent=root)
    pelvis.tx << 3.5
    probe = cmds.spaceLocator(name="probe")[0]
    cmds.parent(probe, pelvis.get_long_name())
    Transform.create(name="box", parent=pelvis)
    data = root.serialize_hierarchy()
    data[3].node_type = "space_transform"
    return data


class _Case(MayaTestCase):
    def setUp(self):
        super().setUp()
        self.new_scene()


class TestUserExamples(_Case):
    def test_create_hierarchy_returns_a_list_of_typed_nodes(self):
        dst = _skeleton()
        self.new_scene()
        one = Transform.create()
        self.assertEqual(repr(one), 'Transform("transform1")')
        made = Transform.create_hierarchy(dst)
        self.assertIs(type(made), List)
        self.assertEqual([type(x) for x in made], [Transform, Joint, Transform, Transform])
        self.assertEqual(made[1].tx.get(), 3.5)

    def test_a_list_of_a_hierarchy_data_is_its_nodes(self):
        hier     = _skeleton()
        rig_list = List(hier)
        self.assertEqual([type(x) for x in rig_list], [Transform, Joint, Transform, Transform])
        self.assertEqual(rig_list[1].tx.get(), 3.5)
        self.assertEqual(List([str(x) for x in hier])[1].tx.get(), 3.5)
        self.assertEqual([x.get_long_name() for x in rig_list],
                         ["|root", "|root|pelvis", "|root|pelvis|probe", "|root|pelvis|box"])


class TestCreateHierarchyReturn(_Case):
    def test_each_node_is_typed_as_its_create_types_it(self):
        made = Transform.create_hierarchy(_skeleton_after_new_scene(self))
        root, pelvis, probe, box = made
        self.assertIs(type(root), Transform)
        self.assertIs(type(pelvis), Joint)
        self.assertIs(type(probe), Transform)
        self.assertEqual(cmds.nodeType(probe.get_shape().get_name()), "locator")
        self.assertIs(type(box), Transform)
        self.assertEqual(cmds.nodeType(box.get_shape().get_name()), "mesh")
        self.assertEqual(cmds.ls(selection=True, long=True), [x.get_long_name() for x in made])

    def test_parents_come_first(self):
        made = Transform.create_hierarchy(_skeleton_after_new_scene(self))
        for i, node in enumerate(made):
            parent = node.get_parent()
            if i == 0:
                self.assertIsNone(parent)
            else:
                self.assertIn(parent.get_long_name(), [x.get_long_name() for x in made[:i]])

    def test_a_renamed_duplicate_is_the_node_maya_made(self):
        data  = _skeleton()
        again = Transform.create_hierarchy(data)
        self.assertEqual([x.get_short_name() for x in again], ["root1", "pelvis", "probe", "box"])
        self.assertEqual([x.get_long_name() for x in again],
                         ["|root1", "|root1|pelvis", "|root1|pelvis|probe", "|root1|pelvis|box"])
        self.assertEqual(again[1].tx.get(), 3.5)
        self.assertEqual([str(x) for x in data], ["root", "pelvis", "probe", "box"])  # data kept

    def test_under_a_parent(self):
        data  = _skeleton()
        group = Transform.create(name="grp")
        made  = Transform.create_hierarchy(data, parent="grp")
        self.assertEqual([x.get_long_name() for x in made],
                         ["|grp|root", "|grp|root|pelvis", "|grp|root|pelvis|probe", "|grp|root|pelvis|box"])
        self.assertEqual(made[0].get_parent(), group)
        self.assertEqual(made.tx.get().tolist(), [0.0, 3.5, -3.5, 0.0])

    def test_one_undo_step(self):
        # (HL_FIX F9) one undo step; undone and redone, the List's nodes live
        data = _skeleton_after_new_scene(self)
        cmds.undoInfo(state=True, infinity=True)
        cmds.flushUndo()
        before = set(cmds.ls(long=True))
        made   = Transform.create_hierarchy(data)
        self.assertEqual(cmds.undoInfo(q=True, undoName=True), "rig.Transform.create_hierarchy")
        cmds.undo()
        self.assertEqual(set(cmds.ls(long=True)), before)
        cmds.redo()
        self.assertEqual([x.get_long_name() for x in made],
                         ["|root", "|root|pelvis", "|root|pelvis|probe", "|root|pelvis|box"])
        self.assertEqual(made[1].tx.get(), 3.5)

    def test_an_error_mid_create_undoes_as_one_step(self):
        data = _skeleton_after_new_scene(self)
        cmds.undoInfo(state=True, infinity=True)
        cmds.flushUndo()
        before = set(cmds.ls(long=True))
        with mock.patch.object(Transform, "set_attrs", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                Transform.create_hierarchy(data)
        self.assertNotEqual(set(cmds.ls(long=True)), before)  # the nodes were made
        cmds.undo()
        self.assertEqual(set(cmds.ls(long=True)), before)
        cmds.redo()
        cmds.undo()
        self.assertEqual(set(cmds.ls(long=True)), before)

    def test_inside_a_container(self):
        data = _skeleton_after_new_scene(self)
        with container("box_ctn") as ctn:
            made = Transform.create_hierarchy(data)
        self.assertEqual([type(x) for x in made], [Transform, Joint, Transform, Transform])
        members = set(cmds.container(str(ctn), query=True, nodeList=True) or [])
        for node in made:
            self.assertIn(node.get_short_name(), members)


def _skeleton_after_new_scene(case):
    data = _skeleton()
    case.new_scene()
    return data


class TestUserCasting(_Case):
    def setUp(self):
        super().setUp()
        self.data = _skeleton()

    def test_a_list_of_transform_data(self):
        items = List(list(self.data))
        self.assertEqual([type(x) for x in items], [Transform, Joint, Transform, Transform])

    def test_every_way_to_add_an_element_casts(self):
        root, pelvis, probe, box = self.data
        items = List()
        items.append(root)
        items.extend([pelvis])
        items.insert(0, probe)
        items += [box]
        self.assertEqual([x.get_short_name() for x in items], ["probe", "root", "pelvis", "box"])
        self.assertTrue(all(isinstance(x, Node) for x in items))
        items[0] = box
        items[1:2] = [pelvis, probe]
        self.assertEqual([x.get_short_name() for x in items], ["box", "pelvis", "probe", "pelvis", "box"])
        self.assertTrue(all(isinstance(x, Node) for x in items))
        self.assertIs(type(items), List)

    def test_the_api_objects(self):
        sel = OpenMaya.MSelectionList()
        sel.add("pelvis")
        sel.add("pelvis.tx")
        items = List([sel.getDependNode(0), sel.getDagPath(0), sel.getPlug(1)])
        self.assertEqual([type(x) for x in items], [Joint, Joint, Plug])
        self.assertEqual(str(items[2]), "pelvis.translateX")
        self.assertEqual(items[2].get(), 3.5)

    def test_an_object_that_names_nothing_is_a_type_error_and_nothing_is_added(self):
        class Thing:
            def __str__(self):
                return "no_such_thing"

        items = List([Node("root")])
        with self.assertRaisesRegex(
            TypeError,
            r"^List element \[1\] \(Thing 'no_such_thing'\) is no node or plug \(its str names no "
            r"node or plug\): a List holds numbers and rig nodes / plugs",
        ):
            items.append(Thing())
        with self.assertRaisesRegex(TypeError, r"^List element \[2\] \(Thing"):
            items.extend([self.data[1], Thing()])
        with self.assertRaisesRegex(TypeError, r"^List element \[0\] \(Thing"):
            items.insert(0, Thing())
        with self.assertRaisesRegex(TypeError, r"^List element \[0\] \(Thing"):
            items[0] = Thing()
        with self.assertRaisesRegex(TypeError, r"^List element \[1\] \(Thing"):
            List([1.0, Thing()])
        with self.assertRaisesRegex(TypeError, r"^List element \[1\] \(object "):
            items += [object()]
        self.assertEqual([str(x) for x in items], ["root"])

    def test_what_is_no_element(self):
        for i, (value, why) in enumerate((
            ({"a": 1}, "it has no node or plug"),
            ({1, 2}, "it has no node or plug"),
            (("root", "pelvis"), "a sequence holds numbers and plugs only"),
            ([Node("root")], "a sequence holds numbers and plugs only"),
            (np.array(["a"]), "a sequence holds numbers and plugs only"),
            ((Node("root").tx, {"a": 1}), "a sequence holds numbers and plugs only"),
            (3 + 4j, "its str names no node or plug"),
            (np.complex128(1), "its str names no node or plug"),
        )):
            with self.subTest(i=i, kind=type(value).__name__):
                with self.assertRaisesRegex(TypeError, rf"^List element \[0\] \({type(value).__name__} .*{why}"):
                    List([value])

    def test_a_name_more_than_one_node_has(self):
        # (HL_FIX F1) each TransformData is found by its uuid: the node it was serialized from
        Transform.create_hierarchy(self.data)  # a second pelvis
        self.assertEqual([x.get_long_name() for x in List(self.data)],
                         ["|root", "|root|pelvis", "|root|pelvis|probe", "|root|pelvis|box"])
        # in another scene (no node has the uuid) the name decides, an ambiguous one raises
        self.new_scene()
        data = _skeleton_after_new_scene(self)
        self.assertFalse(any(cmds.ls(x.uuid) for x in data))
        Transform.create_hierarchy(data)
        self.assertEqual([x.get_long_name() for x in List(data[:1])], ["|root"])
        Transform.create_hierarchy(data)
        with self.assertRaisesRegex(TypeError, r"^List element \[1\] \(TransformData 'pelvis'\) .*ambiguous"):
            List(data)

    def test_left_and_right_joints_of_the_same_name(self):
        self.new_scene()
        root = Joint.create(name="root")
        for side in ("L_leg", "R_leg"):
            Joint.create(name="end", parent=Joint.create(name=side, parent=root))
        cmds.setAttr("root|R_leg|end.tx", 7)
        hier = root.serialize_hierarchy()
        self.assertEqual([str(x) for x in hier], ["root", "L_leg", "end", "R_leg", "end"])
        items = List(hier)
        self.assertEqual([x.get_long_name() for x in items],
                         ["|root", "|root|L_leg", "|root|L_leg|end", "|root|R_leg", "|root|R_leg|end"])
        self.assertEqual(items[4].tx.get(), 7.0)

    def test_a_decoy_name_and_a_renamed_node(self):
        self.new_scene()
        root = Joint.create(name="root")
        Joint.create(name="pelvis", parent=root)
        hier = root.serialize_hierarchy()
        cmds.createNode("transform", name="pelvis")  # a world node named like it
        self.assertEqual([x.get_long_name() for x in List(hier)], ["|root", "|root|pelvis"])
        cmds.rename("root|pelvis", "hip")             # the original renamed: still that node
        self.assertEqual([x.get_long_name() for x in List(hier)], ["|root", "|root|hip"])
        self.assertIs(type(List(hier)[1]), Joint)

    def test_a_plain_name_raises_as_before(self):
        with self.assertRaises(NodeNotFoundError):
            List(["no_such_node"])
        cube = cmds.polyCube(name="cube")[0]
        self.assertEqual(len(List([f"{cube}.vtx[0:3]"])), 4)
        items = List()
        items.append(f"{cube}.vtx[0:3]")
        self.assertEqual(len(items), 4)
        # (HL_FIX F5) a range that names nothing raises, wherever it is added
        why = r"\(str 'nope.vtx\[0:2\]'\) is no node or plug \(it names nothing\)"
        for add in (lambda: List(["nope.vtx[0:2]"]), lambda: items.append("nope.vtx[0:2]"),
                    lambda: items.insert(0, "nope.vtx[0:2]"), lambda: items.extend(["nope.vtx[0:2]"])):
            with self.assertRaisesRegex(TypeError, why):
                add()
        with self.assertRaisesRegex(TypeError, why):
            items[1:2] = ["nope.vtx[0:2]"]
        self.assertEqual(len(items), 4)

    def test_the_slice_hint_of_a_negative_index(self):
        cube  = cmds.polyCube(name="cube")[0]
        items = List(["root", "pelvis"])
        with self.assertRaisesRegex(TypeError, r"names 2 elements; assign them to a slice \(lst\[1:2\] = \.\.\.\)"):
            items[-1] = f"{cube}.vtx[0:1]"
        with self.assertRaisesRegex(TypeError, r"\(lst\[0:1\] = \.\.\.\)"):
            items[0] = f"{cube}.vtx[0:1]"
        items[1:2] = [f"{cube}.vtx[0:1]"]
        self.assertEqual(len(items), 3)

    def test_the_reasons_name_no_api_text(self):
        # (HL_FIX F6) a class, a function, a data object, a null MPlug / MObject
        for value, why in (
            (Transform, "it has no node or plug"),
            (Node("root").get_parent, "it has no node or plug"),
            (len, "it has no node or plug"),
            (self.data, "its str names no node or plug"),
            (OpenMaya.MPlug(), "a null MPlug"),
            (OpenMaya.MObject(), "MObject is null"),
        ):
            with self.subTest(type(value).__name__):
                with self.assertRaises(TypeError) as caught:
                    List([value])
                message = str(caught.exception)
                self.assertIn(why, message)
                self.assertNotIn("kInvalidParameter", message)
                self.assertNotIn("is a pattern", message)

    def test_a_mix(self):
        sel = OpenMaya.MSelectionList()
        sel.add("root.ty")
        inner = List([Node("probe")])
        items = List([
            Node("root"), "pelvis.tx", 2.5, 3, None, True, np.float32(1), np.int64(2),
            np.bool_(True), self.data[2], sel.getPlug(0), (1, 2, 3), [[1, 0], [0, 1]],
            np.eye(4), inner, Node(cmds.polyCube(name="cube")[0]).f[:2],
        ])
        kinds = [type(x).__name__ for x in items]
        self.assertEqual(kinds[:11], ["Transform", "Plug", "float", "int", "NoneType", "bool",
                                      "float32", "int64", "bool_", "Transform", "Plug"])
        self.assertEqual(items[11], (1, 2, 3))
        self.assertEqual(items[12], [[1, 0], [0, 1]])
        self.assertIs(type(items[13]), np.ndarray)
        self.assertIs(items[14], inner)
        self.assertEqual(type(items[15]).__name__, "Components")

    def test_a_numeric_value_is_kept(self):
        nodes = List(["root", "pelvis"])
        nodes.t << List([(1, 2, 3), (4, 5, 6)])
        self.assertEqual(nodes.t.get().tolist(), [[1, 2, 3], [4, 5, 6]])

    def test_an_array(self):
        # (HL_FIX F8) an array has no truth value: List(array) was a ValueError
        self.assertEqual(list(List(np.arange(3))), [0, 1, 2])
        rows = List(np.zeros((3, 3)))
        self.assertEqual([type(x) for x in rows], [np.ndarray] * 3)
        nodes = List(["root", "pelvis"])
        self.assertEqual(list(List(nodes.tx.get())), [0.0, 3.5])
        self.assertEqual(len(List(np.zeros(0))), 0)
        self.assertEqual([type(x) for x in List([np.int32(1), np.uint8(2)])], [np.int32, np.uint8])

    def test_openmaya_values_are_values(self):
        # (HL_FIX F2) a vector, a point, a matrix ... as it is, as a tuple is
        values = [
            OpenMaya.MVector(1, 2, 3), OpenMaya.MFloatVector(1, 2, 3), OpenMaya.MPoint(1, 2, 3),
            OpenMaya.MFloatPoint(1, 2, 3), OpenMaya.MMatrix(), OpenMaya.MFloatMatrix(),
            OpenMaya.MTransformationMatrix(), OpenMaya.MEulerRotation(1, 2, 3), OpenMaya.MQuaternion(),
            OpenMaya.MColor((1, 0, 0)), OpenMaya.MDoubleArray([1, 2]), OpenMaya.MPointArray(),
            OpenMaya.MVectorArray(), OpenMaya.MMatrixArray(),
        ]
        items = List(values)
        self.assertEqual([type(x) for x in items], [type(x) for x in values])
        nodes = List(["root", "pelvis"])
        nodes.t << List([OpenMaya.MVector(1, 2, 3)] * 2)
        self.assertEqual(nodes.t.get().tolist(), [[1, 2, 3], [1, 2, 3]])
        matrix = OpenMaya.MMatrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1])
        nodes.offsetParentMatrix << List([matrix] * 2)
        self.assertEqual(cmds.getAttr("pelvis.offsetParentMatrix")[12:15], [5.0, 6.0, 7.0])
        items = List()
        items.append(OpenMaya.MVector(1, 2, 3))
        items += [(OpenMaya.MPoint(), 1.0)]
        self.assertEqual([type(x) for x in items], [OpenMaya.MVector, tuple])

    def test_an_operand_row_of_plugs_and_numbers(self):
        # (HL_FIX F3) a row mixing plugs and numbers is a value, as in a plain list
        a, b, c = (Transform.create(name=n) for n in "abc")
        c.tx << 4
        nodes = List([a, b])
        nodes.t << List([(c.tx, 1, 2), (3, c.tx, 4)])
        self.assertEqual(nodes.t.get().tolist(), [[4, 1, 2], [3, 4, 4]])
        self.assertEqual(cmds.listConnections("a.tx", s=True, p=True), ["c.translateX"])
        self.assertEqual(cmds.listConnections("b.ty", s=True, p=True), ["c.translateX"])
        nodes.t << List([[c.tx, c.ty, c.tz]] * 2)
        self.assertEqual(cmds.listConnections("b.tz", s=True, p=True), ["c.translateZ"])
        row = (c.tx, 1, 2)
        self.assertIs(List([row])[0], row)

    def test_a_list_gives_its_elements_as_they_are(self):
        # (HL_FIX F4) List(other), extend(other), += other: a List's elements are kept
        nodes   = List(["root", "pelvis"])
        # re-pinned (user decision 2026-10-16, the property ban): node.mobject
        # is a method now, which does not broadcast; the List of values is built
        # as rig builds one (a broadcast's)
        objects = list_module._values_list(x.get_mobject() for x in nodes)
        self.assertEqual([type(x) for x in List(objects)], [OpenMaya.MObject] * 2)
        methods = nodes.get_parent
        self.assertEqual(len(List(methods)), 2)
        items = List()
        items.extend(objects)
        items += methods
        self.assertEqual([type(x).__name__ for x in items], ["MObject", "MObject", "method", "method"])
        with self.assertRaisesRegex(TypeError, r"^List element \[0\] \(method .*it has no node or plug"):
            List(list(methods))  # a plain list of them is a user's: refused

    def test_a_copy_keeps_the_elements_and_the_state(self):
        # (HL_FIX F4) copy / deepcopy / pickle rebuilt the List through the casting append
        import copy
        import pickle

        nodes   = List(["root", "pelvis"])
        # re-pinned (the property ban): the values List built as rig builds one
        objects = list_module._values_list(x.get_mobject() for x in nodes)
        for copied in (copy.copy(objects), copy.copy(List(objects))):
            self.assertIs(type(copied), List)
            self.assertEqual([type(x) for x in copied], [OpenMaya.MObject] * 2)
        methods = copy.copy(nodes.get_parent)
        self.assertIs(type(methods), list_module._Methods)
        self.assertEqual(methods._method, "get_parent")
        values = List([(1, 2), 3.0, None, [[1, 0], [0, 1]]])
        for copied in (copy.deepcopy(values), pickle.loads(pickle.dumps(values))):
            self.assertIs(type(copied), List)
            self.assertEqual(list(copied), list(values))
        deep = copy.deepcopy(values)
        self.assertIsNot(deep[3], values[3])
        cmds.addAttr("root", ln="arr", at="double", multi=True)
        node = Node("root")
        node.arr[:] << [1.0, 2.0]
        self.assertEqual(copy.copy(node.arr[:])._parent_multi, node.arr)
        self.assertEqual([str(x) for x in copy.copy(nodes)], ["root", "pelvis"])


class TestValueLists(_Case):
    """What rig builds from values keeps them (each of these was so before)."""

    def setUp(self):
        super().setUp()
        for name in ("a", "b"):
            Transform.create(name=name)
            cmds.addAttr(name, ln="label", dt="string")
            cmds.setAttr(f"{name}.label", f"{name}_label", type="string")
        self.nodes = List(["a", "b"])

    def test_vectors_matrices_strings_and_enums(self):
        self.nodes.t << [(1, 2, 3), (4, 5, 6)]
        self.assertEqual(self.nodes.t.get().tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(self.nodes.matrix.get().shape, (2, 4, 4))
        self.assertEqual(list(self.nodes.label.get()), ["a_label", "b_label"])
        self.nodes.ro << ["zxy", "yxz"]
        self.assertEqual(self.nodes.ro.get().tolist(), [2, 4])

    def test_python_values_of_each_element(self):
        # re-pinned (user decision 2026-10-16, the property ban): is_locked and
        # get_mobject are methods, which do not broadcast (a bare predicate has
        # no truth value); a broadcast of values is a plug's (``nodes.t``)
        with self.assertRaisesRegex(TypeError, r"call it on each one"):
            bool(self.nodes.is_locked)
        self.assertEqual([x.is_locked() for x in self.nodes], [False, False])
        objects = list_module._values_list(x.get_mobject() for x in self.nodes)
        self.assertIs(type(objects), List)
        self.assertEqual([type(x) for x in objects], [OpenMaya.MObject] * 2)
        self.assertEqual([type(x) for x in objects[0:2]], [OpenMaya.MObject] * 2)
        self.assertEqual([type(x) for x in objects[[1, 0]]], [OpenMaya.MObject] * 2)
        methods = self.nodes.get_children
        self.assertIs(type(methods), list_module._Methods)
        self.assertEqual(len(methods), 2)

    def test_operators_and_queries(self):
        self.assertEqual(list(List([1, 2]) + 1), [2, 3])
        self.assertEqual(list(-List([1.5, 2])), [-1.5, -2])
        self.nodes.tx << [1, 2]
        sums = self.nodes.tx + self.nodes.ty
        self.assertTrue(all(isinstance(x, Plug) for x in sums))
        Node("b").ty << Node("a").tx
        inputs = self.nodes.ty.get_inputs()
        self.assertEqual([type(x) for x in inputs], [List, List])
        self.assertEqual([len(x) for x in inputs], [0, 1])

    def test_multi_slices(self):
        cmds.addAttr("a", ln="arr", at="double", multi=True)
        cmds.addAttr("a", ln="vecs", at="double3", multi=True)
        for axis in "XYZ":
            cmds.addAttr("a", ln=f"vecs{axis}", at="double", parent="vecs")
        node = Node("a")
        node.arr[:] << [1.0, 2.0, 3.0]
        self.assertEqual(node.arr[:].get().tolist(), [1.0, 2.0, 3.0])
        node.vecs[:] << [(1, 2, 3), (4, 5, 6)]
        self.assertEqual(node.vecs[:].get().tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(node.vecs[:]._parent_multi, node.vecs)

    def test_a_vectorized_call_and_a_command_s_results(self):
        @vectorize
        def pair(x):
            return (x, x * 2)

        self.assertEqual(list(pair(List([1, 2]))), [(1, 2), (2, 4)])
        self.assertEqual(rc.getAttr("a.t"), [(0.0, 0.0, 0.0)])
        self.assertEqual(list(rc.getAttr("a.t")), cmds.getAttr("a.t"))
        self.assertEqual(rc.ls("a", "b"), cmds.ls("a", "b"))

    def test_the_value_list_keeps_its_elements_as_before(self):
        kept = list_module._values_list([(1, 2), {"x": 1}, OpenMaya.MObject(), "a"])
        self.assertEqual(kept[0], (1, 2))
        self.assertEqual(kept[1], {"x": 1})
        self.assertIs(type(kept[2]), OpenMaya.MObject)
        self.assertIs(type(kept[3]), Transform)


def _before(items):
    """The List build of fa2afe3, for the cost comparison."""
    out = list.__new__(List)
    object.__setattr__(out, "_parent_multi", None)
    if items:
        for x in items:
            if isinstance(x, numbers.Real) or x is None:
                list.append(out, x)
            elif isinstance(x, (Plug, Node)):
                list.append(out, x)
            else:
                list.append(out, list_module._lift_or_pass(x))
    return out


class TestCost(_Case):
    def _best(self, fn, arg, reps=7, loops=20):
        best = None
        for _ in range(reps):
            t = time.perf_counter()
            for _ in range(loops):
                fn(arg)
            dt = time.perf_counter() - t
            best = dt if best is None else min(best, dt)
        return best

    def test_a_list_of_nodes_and_of_numbers_costs_no_more(self):
        nodes   = [Transform.create(name=f"n{i}") for i in range(200)]
        numbers = range(1000)
        for items in (nodes, numbers):
            with self.subTest(items=type(items[0]).__name__):
                now    = self._best(List, items)
                before = self._best(_before, items)
                self.assertLess(now, before * 1.3, (now, before))

    def test_numpy_numbers_and_vector_tuples_cost_no_more(self):
        # (HL_FIX F7) np.int32 is no int on Windows; a tuple of floats is checked flat
        for items in ([np.int32(i) for i in range(1000)], [(1.0, 2.0, 3.0)] * 300):
            with self.subTest(items=type(items[0]).__name__):
                now    = self._best(List, items)
                before = self._best(_before, items)
                self.assertLess(now, before * 1.5, (now, before))

    def test_append_costs_a_python_call(self):
        # (HL_FIX F7) append is a Python method now (it casts): about 3x list.append,
        # the cost of a Python call. A loose bound against the bare Python override
        # (its one type check costs little more)
        class Override(list):
            def append(self, value):
                list.append(self, value)

        plugs = [Plug("persp.tx")] * 300

        def appended(items):
            out = List()
            for x in items:
                out.append(x)

        def overridden(items):
            out = Override()
            for x in items:
                out.append(x)

        for items in (plugs, [float(i) for i in range(1000)]):
            with self.subTest(items=type(items[0]).__name__):
                now    = self._best(appended, items)
                before = self._best(overridden, items)
                # re-pinned (user decisions 1A / 6C 2026-10-17, the List rule): an
                # appended node or plug is checked (its handle), so the bound is 5x
                self.assertLess(now, before * 5, (now, before))
