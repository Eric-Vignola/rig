"""Round U, fixes independent of the undo work.

Step U5 (user decision, round-4a deferred items): a lattice ``pt`` int key that
leaves two or more axes unspecified raises :class:`IndexError` with a hint,
before any scene query. ``lat.pt[0]`` would be a flattened slab, and a chained
``lat.pt[0][1]`` silently read ``pt[0][0][1]`` from it (round-4a repro
``case_f11_lattice``). The slab is spelled ``pt[0, :, :]`` and one point
``pt[s, t, u]``. A key leaving one axis (``pt[s, t]``, a line) is unchanged, as
is any key with a slice, a NURBS-surface ``cv[u]`` / ``cv[u][v]``, a
``Plug("shape.pt[...]")`` string, ``Components(...)`` and the bool
``TypeError``.

Step U6 (user decision, round-4a deferred items): an enum attribute takes a
field name as well as its int, through ``<<``, ``=``, ``Attribute.set``, a List
broadcast and the ``rig.bridges.nodes`` factories: the exact name first, then
the one field that matches once case, spaces, ``_`` and ``-`` are ignored. A
wrong name is a ``TypeError`` naming the fields, raised before any edit (a List
sets nothing, a factory creates no node). A str into a string attribute, a str
into a numeric attribute and the operators' plain-str refusal are unchanged.

The review of U6 (:class:`TestEnumNamesReview`): an exact hit counts only when
Maya names that value back (``fieldValue("")`` is the first value, so ``""``
set the first field); the loose match is two tiers, case and outer spaces
first, then spaces, ``_`` and a ``-`` between letters (a leading ``-`` is a
sign: ``"-x"`` is never ``"X"``); a list into a compound or a multi, a List
row of a compound and a factory's list kwarg have every name read before the
first set; an ``Enum`` spec's ``dv`` takes a field name; a ``Layer`` /
material spec's enum kwargs are read before its node is made.
"""

import re
from unittest import mock

import numpy as np
from maya import cmds

from rig import Components, List, Node
from rig._internal import plug as plug_module
from rig._internal.plug import ComponentPlug, InjectionError, Plug
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk
from rig.bridges import nodes as rn
from rig.nodetypes import _base as base_module
from rig.spec import Enum, lock


def _scene():
    """Every node in the scene, by long name."""
    return set(cmds.ls(long=True))


def _hint(shape, given, left_names="t, u", colons=":, :"):
    return (
        f"{shape}.pt[{given}] leaves 2 of 3 axes unspecified, and a chained index "
        "would pick from the flattened selection: write "
        f"pt[{given}, {left_names}] for one point or pt[{given}, {colons}] for the slab"
    )


class TestLatticePartialIndex(MayaTestCase):
    """``pt[s]`` (and chains on it) raises; every other indexing form holds."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cube = cmds.polyCube(name="ru_cube", ch=False)[0]
        lattice = cmds.lattice(cube, divisions=(3, 4, 5))[1]
        self.shape = cmds.listRelatives(lattice, shapes=True)[0]
        self.node = Node(self.shape)

    # -- the refusal -- #

    def test_bare_int_raises_with_hint_and_builds_nothing(self):
        for key, given in ((0, "0"), (-1, "-1"), (2, "2"), ((0,), "0")):
            with self.subTest(key=key):
                before = _scene()
                with self.assertRaises(IndexError) as caught:
                    self.node.pt[key]
                self.assertEqual(str(caught.exception), _hint(self.shape, given))
                self.assertEqual(_scene(), before)

    def test_chained_partial_raises_with_hint(self):
        # the round-4a repro: today pt[0][1] read pt[0][0][1]
        before = _scene()
        with self.assertRaises(IndexError) as caught:
            self.node.pt[0][1]
        self.assertEqual(str(caught.exception), _hint(self.shape, "0"))
        with self.assertRaises(IndexError):
            self.node.pt[1][2][3]
        self.assertEqual(_scene(), before)

    def test_held_handle_raises_and_names_the_live_node(self):
        handle  = self.node.pt
        renamed = cmds.rename(self.shape, "ru_lattice_renamed")
        with self.assertRaises(IndexError) as caught:
            handle[0]
        self.assertEqual(str(caught.exception), _hint(renamed, "0"))

    def test_raises_before_any_scene_query(self):
        # no axis size is read: the refusal needs only the key and the rank,
        # so an out-of-range partial key gets the same hint
        with mock.patch.object(
            ComponentPlug, "_axis_sizes", side_effect=AssertionError("queried")
        ):
            for key in (0, 99, -7):
                with self.subTest(key=key):
                    with self.assertRaisesRegex(IndexError, r"leaves 2 of 3 axes"):
                        self.node.pt[key]

    # -- unchanged forms -- #

    def test_line_is_unchanged(self):
        line = self.node.pt[0, 1]
        self.assertIsInstance(line, List)
        self.assertEqual(
            [str(p) for p in line], [f"{self.shape}.pt[0][1][{u}]" for u in range(5)]
        )
        self.assertEqual(str(self.node.pt[0, 1][2]), str(self.node.pt[0, 1, 2]))
        self.assertTrue(self.node.pt[0, 1][2].equals(self.node.pt[0, 1, 2]))
        self.assertEqual(str(self.node.pt[-1, -1][-1]), f"{self.shape}.pt[2][3][4]")

    def test_element_is_unchanged(self):
        elem = self.node.pt[0, 1, 2]
        self.assertIsInstance(elem, ComponentPlug)
        self.assertNotIsInstance(elem, List)
        self.assertEqual(str(elem), f"{self.shape}.pt[0][1][2]")
        self.assertEqual(elem.name, Plug(f"{self.shape}.pt[0][1][2]").name)

    def test_slices_are_unchanged(self):
        slab = self.node.pt[0, :, :]
        self.assertEqual(len(slab), 20)
        self.assertEqual(str(slab[0]), f"{self.shape}.pt[0][0][0]")
        self.assertEqual(str(slab[-1]), f"{self.shape}.pt[0][3][4]")
        # a slice pads the missing trailing axis, as numpy does
        self.assertEqual([str(p) for p in self.node.pt[0, :]], [str(p) for p in slab])
        plane = self.node.pt[:, :, 0]
        self.assertEqual(len(plane), 12)
        self.assertEqual(str(plane[-1]), f"{self.shape}.pt[2][3][0]")
        self.assertEqual(len(self.node.pt[:]), 60)
        self.assertEqual(len(self.node.pt[0, 1:3]), 10)
        self.assertEqual(len(self.node.pt[()]), 60)

    def test_other_refusals_are_unchanged(self):
        for key in (True, (0, True), (True, 0, 0), "a"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(TypeError, r"must be int or slice"):
                    self.node.pt[key]
        with self.assertRaisesRegex(IndexError, r"is 3-D but 4 indices given"):
            self.node.pt[0, 0, 0, 0]
        with self.assertRaisesRegex(IndexError, r"axis 1 index 9 out of range"):
            self.node.pt[0, 9]

    def test_surface_cv_is_unchanged(self):
        surface = cmds.listRelatives(cmds.nurbsPlane(name="ru_plane")[0], shapes=True)[0]
        cv = Node(surface).cv
        row = cv[1]
        self.assertIsInstance(row, List)
        self.assertEqual([str(p) for p in row], [f"{surface}.cv[1][{v}]" for v in range(4)])
        self.assertEqual(str(cv[1][2]), f"{surface}.cv[1][2]")
        self.assertTrue(cv[1][2].equals(cv[1, 2]))
        self.assertEqual(str(cv[-1][0]), f"{surface}.cv[3][0]")
        with self.assertRaisesRegex(TypeError, r"must be int or slice"):
            cv[True]

    def test_plug_strings_and_components_are_unchanged(self):
        self.assertEqual(
            Plug(f"{self.shape}.pt[0][1][2]").name, self.node.pt[0, 1, 2].name
        )
        points = Components(self.shape, "pt")
        self.assertEqual(points.count, 60)
        long_name = cmds.ls(self.shape, long=True)[0]
        self.assertEqual(points[0].names, [f"{long_name}.pt[0][0][0]"])
        self.assertEqual(
            Components(self.shape, "pt", [[0, 1, 2]]).names, [f"{long_name}.pt[0][1][2]"]
        )


# rotateOrder's fields, as a wrong name's TypeError lists them
_RO_FIELDS = "xyz=0, yzx=1, zxy=2, xzy=3, yxz=4, zyx=5"


class TestEnumFieldNames(UndoWalk, MayaTestCase):
    """An enum attribute set by field name (U6)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.t = Node(cmds.createNode("transform", name="ru_t"))

    def ro(self, name="ru_t"):
        return cmds.getAttr(f"{name}.rotateOrder")

    # -- the set forms -- #

    def test_lshift_takes_a_field_name(self):
        self.t.ro << "zxy"
        self.assertEqual(self.ro(), 2)
        self.t.rotateOrder << "zyx"
        self.assertEqual(self.ro(), 5)

    def test_assignment_takes_a_field_name(self):
        self.t.ro = "yzx"
        self.assertEqual(self.ro(), 1)

    def test_plug_of_find_attr_takes_a_field_name(self):
        # find_attr returns the typed Attribute, which has no ``<<`` for any
        # value; the DSL Plug of it is ``Plug(node.find_attr(name))``
        Plug(self.t.find_attr("ro")) << "xzy"
        self.assertEqual(self.ro(), 3)

    def test_attribute_set_takes_a_field_name(self):
        self.t.find_attr("ro").set("zyx")
        self.assertEqual(self.ro(), 5)
        self.t.find_attr("ro").set(2)
        self.assertEqual(self.ro(), 2)
        self.t.ro.set("yxz")
        self.assertEqual(self.ro(), 4)

    def test_maya_field_names_match_loosely(self):
        md = Node(cmds.createNode("multiplyDivide", name="ru_md"))
        for name in ("Divide", "divide", "DIVIDE"):
            with self.subTest(name=name):
                md.operation << 0
                md.operation << name
                self.assertEqual(cmds.getAttr("ru_md.operation"), 2)
        md.operation << "no_operation"
        self.assertEqual(cmds.getAttr("ru_md.operation"), 0)
        cond = Node(cmds.createNode("condition", name="ru_cond"))
        for name, value in (
            ("greater than", 2),
            ("greater_than", 2),
            ("Greater-Than", 2),
            ("Greater Than", 2),
            ("lessOrEqual", 5),
        ):
            with self.subTest(name=name):
                cond.operation << 0
                cond.operation << name
                self.assertEqual(cmds.getAttr("ru_cond.operation"), value)

    def test_sparse_custom_enum(self):
        self.t << Enum("mode", en="a=5:b=10")
        self.t.mode << "b"
        self.assertEqual(cmds.getAttr("ru_t.mode"), 10)
        self.t.mode << "a"
        self.assertEqual(cmds.getAttr("ru_t.mode"), 5)
        with self.assertRaisesRegex(TypeError, r"ru_t\.mode: 'c' .*: a=5, b=10$"):
            self.t.mode << "c"
        self.assertEqual(cmds.getAttr("ru_t.mode"), 5)

    def test_node_state(self):
        self.t.nodeState << "HasNoEffect"
        self.assertEqual(cmds.getAttr("ru_t.nodeState"), 1)
        self.t.nodeState << "waiting_normal"
        self.assertEqual(cmds.getAttr("ru_t.nodeState"), 8)
        self.t.nodeState << "Normal"
        self.assertEqual(cmds.getAttr("ru_t.nodeState"), 0)

    def test_wide_sparse_enum_is_read_from_its_declaration(self):
        cmds.addAttr(
            "ru_t", longName="wide", attributeType="enum", enumName="a=-32000:b=32000"
        )
        declared = mock.Mock(wraps=base_module._declared_enum_fields)
        with mock.patch.object(base_module, "_declared_enum_fields", declared):
            self.t.wide << "B"
            self.assertEqual(cmds.getAttr("ru_t.wide"), 32000)
            with self.assertRaisesRegex(TypeError, r"'c' .*: a=-32000, b=32000$"):
                self.t.wide << "c"
        self.assertEqual(declared.call_count, 2)
        self.assertEqual(cmds.getAttr("ru_t.wide"), 32000)

    def test_multi_and_compound_enums(self):
        cmds.addAttr(
            "ru_t", longName="menu", attributeType="enum", enumName="p:q:r", multi=True
        )
        self.t.menu[2] << "r"
        self.assertEqual(cmds.getAttr("ru_t.menu[2]"), 2)
        cmds.addAttr("ru_t", longName="pair", attributeType="compound", numberOfChildren=2)
        cmds.addAttr("ru_t", longName="pa", attributeType="enum", enumName="x:y", parent="pair")
        cmds.addAttr("ru_t", longName="pb", attributeType="enum", enumName="x:y", parent="pair")
        self.t.pair << ["x", "y"]
        self.assertEqual((cmds.getAttr("ru_t.pa"), cmds.getAttr("ru_t.pb")), (0, 1))
        self.t.pair << "y"
        self.assertEqual((cmds.getAttr("ru_t.pa"), cmds.getAttr("ru_t.pb")), (1, 1))

    # -- wrong names -- #

    def test_wrong_name_raises_naming_the_fields(self):
        self.t.ro << 4
        forms = {
            "<<": lambda: self.t.ro << "bad",
            "=": lambda: setattr(self.t, "ro", "bad"),
            "set": lambda: self.t.find_attr("ro").set("bad"),
            "number str": lambda: self.t.find_attr("ro").set("2"),
        }
        for form, call in forms.items():
            with self.subTest(form=form):
                with self.assertRaises(TypeError) as caught:
                    call()
                message = str(caught.exception)
                self.assertIn("ru_t.rotateOrder: ", message)
                self.assertIn("is not one of its enum fields", message)
                self.assertTrue(message.endswith(_RO_FIELDS), message)
                self.assertEqual(self.ro(), 4)

    def test_a_plug_name_gets_the_connect_hint(self):
        driver = Node(cmds.createNode("transform", name="ru_driver"))
        with self.assertRaises(TypeError) as caught:
            self.t.ro << "ru_driver.ro"
        self.assertTrue(
            str(caught.exception).endswith(
                f"{_RO_FIELDS}; to connect, write Plug('ru_driver.ro')"
            ),
            str(caught.exception),
        )
        self.assertEqual(len(self.t.ro.get_inputs()), 0)
        with self.assertRaises(TypeError) as caught:
            self.t.ro << "4.0"
        self.assertNotIn("Plug(", str(caught.exception))
        self.t.ro << Plug("ru_driver.ro")
        self.assertTrue(cmds.isConnected("ru_driver.rotateOrder", "ru_t.rotateOrder"))
        self.assertTrue(driver.ro.equals(self.t.ro.get_inputs()[0]))

    def test_an_ambiguous_loose_name_raises(self):
        cmds.addAttr("ru_t", longName="dup", attributeType="enum", enumName="a b:a_b")
        with self.assertRaisesRegex(
            TypeError, r"'AB' matches 2 of its enum fields .*: a b=0, a_b=1$"
        ):
            self.t.dup << "AB"
        self.t.dup << "a_b"
        self.assertEqual(cmds.getAttr("ru_t.dup"), 1)
        self.t.dup << "a b"
        self.assertEqual(cmds.getAttr("ru_t.dup"), 0)

    # -- connections and locks -- #

    def test_a_connected_enum_is_disconnected_and_set(self):
        driver = cmds.createNode("transform", name="ru_driver")
        cmds.connectAttr(f"{driver}.rotateOrder", "ru_t.rotateOrder")
        with self.assertRaises(TypeError):
            self.t.ro << "bad"
        self.assertTrue(cmds.isConnected(f"{driver}.rotateOrder", "ru_t.rotateOrder"))
        self.t.ro << "zxy"
        self.assertFalse(cmds.isConnected(f"{driver}.rotateOrder", "ru_t.rotateOrder"))
        self.assertEqual(self.ro(), 2)

    def test_a_locked_enum_raises_injection_error_first(self):
        self.t.ro << lock
        for name in ("zxy", "bad"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(InjectionError, r"locked attribute"):
                    self.t.ro << name
                self.assertEqual(self.ro(), 0)

    # -- List broadcast -- #

    def _list(self):
        return List(
            [Node(cmds.createNode("transform", name=f"ru_l{i}")) for i in range(3)]
        )

    def _list_ro(self):
        return [self.ro(f"ru_l{i}") for i in range(3)]

    def test_list_broadcast_of_one_name(self):
        nodes = self._list()
        nodes.ro << "zxy"
        self.assertEqual(self._list_ro(), [2, 2, 2])
        nodes.ro = "yzx"
        self.assertEqual(self._list_ro(), [1, 1, 1])

    def test_list_broadcast_of_names(self):
        nodes = self._list()
        nodes.ro << ["xzy", "yxz", "zyx"]
        self.assertEqual(self._list_ro(), [3, 4, 5])
        nodes.ro << ("xyz", 1, "zxy")
        self.assertEqual(self._list_ro(), [0, 1, 2])
        nodes.ro << np.array(["zyx", "yxz", "xzy"])
        self.assertEqual(self._list_ro(), [5, 4, 3])

    def test_list_with_one_wrong_name_sets_nothing(self):
        nodes = self._list()
        for row, other in ((1, ["xzy", "bad", "zyx"]), (2, ["xzy", "yxz", "bad"])):
            with self.subTest(other=other):
                with self.assertRaisesRegex(
                    TypeError, rf"^List row {row}, ru_l{row}\.rotateOrder: 'bad'"
                ):
                    nodes.ro << other
                self.assertEqual(self._list_ro(), [0, 0, 0])

    # -- rig.bridges.nodes factories -- #

    def test_factory_flags_take_field_names(self):
        cases = (
            ("transform", {"rotateOrder": "xzy"}, "rotateOrder", 3),
            ("transform", {"ro": "xzy"}, "rotateOrder", 3),
            ("decomposeMatrix", {"inputRotateOrder": "zxy"}, "inputRotateOrder", 2),
            ("multiplyDivide", {"operation": "power"}, "operation", 3),
            ("multiplyDivide", {"op": "Divide"}, "operation", 2),
        )
        for node_type, kwargs, attr, value in cases:
            with self.subTest(node_type=node_type, kwargs=kwargs):
                node = getattr(rn, node_type)(**kwargs)
                self.assertEqual(cmds.getAttr(f"{node}.{attr}"), value)

    def test_factory_wrong_name_creates_no_node(self):
        before = set(cmds.ls(long=True))
        with self.assertRaises(TypeError) as caught:
            rn.transform(name="ru_bad", translate=[1, 2, 3], rotateOrder="bad")
        self.assertEqual(
            str(caught.exception),
            f"transform.rotateOrder: 'bad' is not one of its enum fields; write a "
            f"field name or its int: {_RO_FIELDS}",
        )
        self.assertEqual(set(cmds.ls(long=True)), before)

    def test_factory_leaves_an_undescribed_attribute_to_lshift(self):
        class Blind:
            def __init__(self, node_type):
                pass

            def attribute(self, name):
                raise RuntimeError("(kFailure): Object does not exist")

        with mock.patch.object(rn._om, "MNodeClass", Blind):
            node = rn.transform(rotateOrder="xzy")
            self.assertEqual(cmds.getAttr(f"{node}.rotateOrder"), 3)
            with self.assertRaises(TypeError):
                rn.transform(name="ru_late", rotateOrder="bad")
        # the << after creation raises, as for any other failing value
        self.assertTrue(cmds.objExists("ru_late"))

    def test_factory_non_enum_values_are_unchanged(self):
        node = rn.transform(translate=[1, 2, 3], rotateOrder=4)
        self.assertEqual(cmds.getAttr(f"{node}.rotateOrder"), 4)
        with self.assertRaises(InjectionError):
            rn.transform(name="ru_tx", translateX="abc")

    # -- unchanged -- #

    def test_string_and_numeric_attributes_are_unchanged(self):
        cmds.addAttr("ru_t", longName="label", dataType="string")
        self.t.label << "zxy"
        self.assertEqual(cmds.getAttr("ru_t.label"), "zxy")
        self.t.find_attr("label").set("xyz")
        self.assertEqual(cmds.getAttr("ru_t.label"), "xyz")
        with self.assertRaisesRegex(
            InjectionError, r"does not accept data of type 'string'"
        ):
            self.t.tx << "zxy"

    def test_numeric_sets_do_not_read_field_names(self):
        refuse = mock.Mock(side_effect=AssertionError("a field name was read"))
        with mock.patch.object(plug_module, "_enum_value", refuse), mock.patch.object(
            base_module, "_enum_value", refuse
        ):
            self.t.ro << 3
            self.t.ro = 1
            self.t.find_attr("ro").set(2)
            List([self.t]).ro << 5
        refuse.assert_not_called()
        self.assertEqual(self.ro(), 5)

    def test_operator_refusals_still_hold(self):
        before = set(cmds.ls(long=True))
        operations = {
            "+": lambda: self.t.ro + "zxy",
            "==": lambda: self.t.ro == "zxy",
            "<": lambda: self.t.ro < "zxy",
            "*": lambda: "zxy" * self.t.ro,
            "List +": lambda: List([self.t]).ro + "zxy",
        }
        for label, call in operations.items():
            with self.subTest(operator=label):
                with self.assertRaisesRegex(TypeError, r"'zxy' is a plain str"):
                    call()
        self.assertEqual(set(cmds.ls(long=True)), before)
        self.assertEqual(self.ro(), 0)

    # -- undo -- #

    def test_one_undo_reverts_a_name_set(self):
        cmds.flushUndo()
        self.t.ro << "zxy"
        self.assertEqual(self.ro(), 2)
        self.undo_steps(1)
        self.assertEqual(self.ro(), 0)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(1)
        self.assertEqual(self.ro(), 2)


class TestEnumNamesReview(MayaTestCase):
    """The round-U review of U6: an empty name is no field, a leading ``-`` is a
    sign, every name of a set that reaches several leaves is read before the
    first set, an ``Enum`` spec default takes a field name, and a collection
    spec's enum kwargs are read before its node is made."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.t = Node(cmds.createNode("transform", name="ru_t"))

    def ro(self, name="ru_t"):
        return cmds.getAttr(f"{name}.rotateOrder")

    # -- an empty name -- #

    def test_an_empty_or_blank_name_is_no_field(self):
        self.t.ro << 3
        md = Node(cmds.createNode("multiplyDivide", name="ru_md"))
        md.operation << 2
        self.t << Enum("sparse", en="a=5:b=10")
        self.t.sparse << "b"
        nodes = List([Node(cmds.createNode("transform", name=f"ru_l{i}")) for i in range(2)])
        nodes.ro << 4
        forms = {
            "<<": lambda: self.t.ro << "",
            "=": lambda: setattr(self.t, "ro", ""),
            "set": lambda: self.t.find_attr("ro").set(""),
            "blank": lambda: self.t.ro << "   ",
            "sparse": lambda: self.t.sparse << "",
            "operation": lambda: md.operation << "",
            "List row": lambda: nodes.ro << ["", "zyx"],
        }
        for form, call in forms.items():
            with self.subTest(form=form):
                with self.assertRaisesRegex(TypeError, r"is not one of its enum fields"):
                    call()
        self.assertEqual(self.ro(), 3)
        self.assertEqual(cmds.getAttr("ru_md.operation"), 2)
        self.assertEqual(cmds.getAttr("ru_t.sparse"), 10)
        self.assertEqual([self.ro("ru_l0"), self.ro("ru_l1")], [4, 4])
        before = _scene()
        with self.assertRaisesRegex(TypeError, r"^transform\.ro: '' is not one of its enum fields"):
            rn.transform(name="ru_empty", ro="")
        self.assertEqual(_scene(), before)

    # -- the sign of an axis -- #

    def test_a_leading_dash_is_a_sign(self):
        mp = Node(cmds.createNode("motionPath", name="ru_mp"))
        mp.frontAxis << 2
        for name in ("-x", "-X"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    TypeError, rf"'{name}' is not one of its enum fields.*X=0, Y=1, Z=2$"
                ):
                    mp.frontAxis << name
                self.assertEqual(cmds.getAttr("ru_mp.frontAxis"), 2)
        mp.frontAxis << "x"
        self.assertEqual(cmds.getAttr("ru_mp.frontAxis"), 0)
        before = _scene()
        with self.assertRaises(TypeError):
            rn.motionPath(name="ru_mp2", upAxis="-z")
        self.assertEqual(_scene(), before)
        self.t << Enum("signed", en="X:Y:Z:-X:-Y:-Z")
        for name, value in (("x", 0), ("-x", 3), ("-X", 3), ("X", 0), ("-z", 5), (" -y ", 4)):
            with self.subTest(name=name):
                self.t.signed << name
                self.assertEqual(cmds.getAttr("ru_t.signed"), value)
        self.t << Enum("aim", en="x:y:z")
        self.t.aim << 2
        with self.assertRaises(TypeError):
            self.t.aim << "-x"
        self.assertEqual(cmds.getAttr("ru_t.aim"), 2)

    def test_a_field_with_outer_spaces(self):
        fog = Node(cmds.createNode("envFog", name="ru_fog"))
        fields = dict(
            base_module._enum_fields(
                base_module.OpenMaya.MFnEnumAttribute(fog.find_attr("fogAxis").mobject)
            )
        )
        self.assertEqual(fields[" -X"], 1)
        fog.fogAxis << "-X"
        self.assertEqual(cmds.getAttr("ru_fog.fogAxis"), 1)
        fog.fogAxis << "-z"
        self.assertEqual(cmds.getAttr("ru_fog.fogAxis"), 5)
        fog.fogAxis << "X"
        self.assertEqual(cmds.getAttr("ru_fog.fogAxis"), 0)

    def test_a_dash_between_letters_still_matches(self):
        cond = Node(cmds.createNode("condition", name="ru_cond"))
        cond.operation << "Greater-Than"
        self.assertEqual(cmds.getAttr("ru_cond.operation"), 2)
        self.t.nodeState << "waiting-normal"
        self.assertEqual(cmds.getAttr("ru_t.nodeState"), 8)
        self.t << Enum("sgn", en="positive:negative")
        self.t.sgn << "nega-tive"
        self.assertEqual(cmds.getAttr("ru_t.sgn"), 1)

    # -- every name read before the first set -- #

    def _pair(self, node="ru_t"):
        cmds.addAttr(node, longName="pair", attributeType="compound", numberOfChildren=2)
        cmds.addAttr(node, longName="pa", attributeType="enum", enumName="x:y", parent="pair")
        cmds.addAttr(node, longName="pb", attributeType="enum", enumName="x:y", parent="pair")

    def _pairs(self, node="ru_t"):
        return cmds.getAttr(f"{node}.pa"), cmds.getAttr(f"{node}.pb")

    def test_a_list_into_a_compound_sets_nothing_on_a_wrong_name(self):
        self._pair()
        forms = {
            "<<": lambda: self.t.pair << ["y", "bad"],
            "=": lambda: setattr(self.t, "pair", ["y", "bad"]),
            "ndarray": lambda: self.t.pair << np.array(["y", "bad"]),
        }
        for form, call in forms.items():
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    TypeError, r"^ru_t\.pair\.pb: 'bad' is not one of its enum fields"
                ):
                    call()
                self.assertEqual(self._pairs(), (0, 0))
        self.t.pair << ["y", "x"]
        self.assertEqual(self._pairs(), (1, 0))
        self.t.pair << ["x", 1]
        self.assertEqual(self._pairs(), (0, 1))

    def test_a_str_into_a_compound_of_different_enums(self):
        cmds.addAttr("ru_t", longName="mix", attributeType="compound", numberOfChildren=2)
        cmds.addAttr("ru_t", longName="ma", attributeType="enum", enumName="p:q", parent="mix")
        cmds.addAttr("ru_t", longName="mb", attributeType="enum", enumName="q:r", parent="mix")
        with self.assertRaisesRegex(TypeError, r"^ru_t\.mix\.mb: 'p' is not one of its enum fields"):
            self.t.mix << "p"
        self.assertEqual((cmds.getAttr("ru_t.ma"), cmds.getAttr("ru_t.mb")), (0, 0))
        self.t.mix << "q"
        self.assertEqual((cmds.getAttr("ru_t.ma"), cmds.getAttr("ru_t.mb")), (1, 0))

    def test_a_str_for_a_numeric_leaf_sets_nothing(self):
        # the leaves ahead of it were set first (round U2 FIX review:
        # runs\rU2\review_semantics_completeness\p_partial.py, p_enum3.py)
        cmds.addAttr("ru_t", longName="cpd", attributeType="compound", numberOfChildren=2)
        cmds.addAttr("ru_t", longName="ce", attributeType="enum", enumName="p:q", parent="cpd")
        cmds.addAttr("ru_t", longName="cf", attributeType="double", parent="cpd")
        cmds.setAttr("ru_t.ce", 1)

        def values():
            return (cmds.getAttr("ru_t.t")[0], cmds.getAttr("ru_t.r")[0], cmds.getAttr("ru_t.ce"),
                    cmds.getAttr("ru_t.cf"), self.ro())

        before = values()
        forms = {
            "t << [5, 'abc', 7]": (lambda: self.t.t << [5, "abc", 7], "ru_t.translate.translateY"),
            "t = [5, 'abc', 7]": (lambda: setattr(self.t, "t", [5, "abc", 7]), "ru_t.translate.translateY"),
            "r << (1, 2, 'x')": (lambda: self.t.r << (1, 2, "x"), "ru_t.rotate.rotateZ"),
            "cpd << [0, 'abc']": (lambda: self.t.cpd << [0, "abc"], "ru_t.cpd.cf"),
            "List([ro, tx]) << 'zxy'": (lambda: List([self.t.ro, self.t.tx]) << "zxy", "List row 1, ru_t.translateX"),
        }
        for form, (call, where) in forms.items():
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    InjectionError, rf"^Cannot set '{re.escape(where)}': a numeric attribute does not accept "
                    r"data of type 'string' \('\w+'\); nothing was set$"
                ):
                    call()
                self.assertEqual(values(), before)
        # a str for a string leaf, an enum name and a number together still set
        cmds.addAttr("ru_t", longName="scp", attributeType="compound", numberOfChildren=2)
        cmds.addAttr("ru_t", longName="sn", dataType="string", parent="scp")
        cmds.addAttr("ru_t", longName="sd", attributeType="double", parent="scp")
        self.t.scp << ["hello", 2.0]
        self.t.cpd << ["p", 3]
        self.assertEqual((cmds.getAttr("ru_t.sn"), cmds.getAttr("ru_t.sd")), ("hello", 2.0))
        self.assertEqual((cmds.getAttr("ru_t.ce"), cmds.getAttr("ru_t.cf")), (0, 3.0))

    def test_a_list_into_a_multi_root_creates_nothing_on_a_wrong_name(self):
        cmds.addAttr("ru_t", longName="menu", attributeType="enum", enumName="p:q:r", multi=True)
        with self.assertRaisesRegex(TypeError, r"^ru_t\.menu\[2\]: 'bad' is not one of its enum fields"):
            self.t.menu << ["q", "r", "bad"]
        self.assertIsNone(cmds.getAttr("ru_t.menu", multiIndices=True))
        self.t.menu << ["q", "r"]
        self.assertEqual([cmds.getAttr(f"ru_t.menu[{i}]") for i in range(2)], [1, 2])
        with self.assertRaises(TypeError):
            self.t.menu << "bad"  # the auto-appended element
        self.assertEqual(cmds.getAttr("ru_t.menu", multiIndices=True), [0, 1])

    def test_a_list_of_compound_rows_sets_nothing_on_a_wrong_name(self):
        other = Node(cmds.createNode("transform", name="ru_o"))
        self._pair()
        self._pair("ru_o")
        rows = List([self.t.pair, other.pair])
        with self.assertRaisesRegex(TypeError, r"^List row 1, ru_o\.pair\.pa: 'bad'"):
            rows << ["y", "bad"]
        with self.assertRaisesRegex(TypeError, r"^List row 1, ru_o\.pair\.pb: 'bad'"):
            rows << [["y", "y"], ["x", "bad"]]
        self.assertEqual((self._pairs(), self._pairs("ru_o")), ((0, 0), (0, 0)))
        rows << [["y", "x"], ["x", "y"]]
        self.assertEqual((self._pairs(), self._pairs("ru_o")), ((1, 0), (0, 1)))

    def test_a_factory_list_kwarg_creates_nothing_on_a_wrong_name(self):
        before = _scene()
        with self.assertRaisesRegex(
            TypeError, r"^lodGroup\.displayLevel\[1\]: 'bad' is not one of its enum fields"
        ):
            rn.lodGroup(name="ru_lod", displayLevel=["show", "bad"])
        with self.assertRaisesRegex(TypeError, r"^transform\.rotateOrder: 'bad'"):
            rn.transform(name="ru_ro", rotateOrder=["bad"])
        self.assertEqual(_scene(), before)
        lod = rn.lodGroup(name="ru_lod", displayLevel=["show", "hide"])
        self.assertEqual([cmds.getAttr(f"{lod}.displayLevel[{i}]") for i in range(2)], [1, 2])
        node = rn.transform(rotateOrder=["xzy"])
        self.assertEqual(cmds.getAttr(f"{node}.rotateOrder"), 3)

    # -- an Enum spec's default -- #

    def test_an_enum_spec_default_takes_a_field_name(self):
        cases = (
            ("m0", {"en": "a:b:c", "dv": "b"}, 1),
            ("m1", {"en": "a:b:c", "defaultValue": "C"}, 2),
            ("m2", {"en": "a=5:b=10", "dv": "b"}, 10),
            ("m3", {"en": ["low", "high"], "dv": "HIGH"}, 1),
            ("m4", {"en": "a:b:c", "dv": 2}, 2),
        )
        for name, kwargs, value in cases:
            with self.subTest(kwargs=kwargs):
                self.t << Enum(name, **kwargs)
                self.assertEqual(cmds.getAttr(f"ru_t.{name}"), value)
                self.assertEqual(
                    cmds.attributeQuery(name, node="ru_t", listDefault=True), [float(value)]
                )
        self.t << Enum("many", en="a:b:c", multi=True, dv="b")
        self.assertEqual(cmds.attributeQuery("many", node="ru_t", listDefault=True), [1.0])
        with self.assertRaisesRegex(
            TypeError, r"^Enum 'bad' dv: 'd' is not one of its enum fields; .*: a=0, b=1, c=2$"
        ):
            Enum("bad", en="a:b:c", dv="d")

    # -- collection-spec kwargs -- #

    def test_collection_spec_enum_kwargs_are_read_before_the_node_is_made(self):
        from rig import Layer
        from rig.shade import Lambert

        cube = Node(cmds.polyCube(name="ru_cube", ch=False)[0])
        layers = set(cmds.ls(type="displayLayer"))
        with self.assertRaisesRegex(
            TypeError, r"^displayLayer\.displayType: 'refrence' is not one of its enum fields"
        ):
            cube << Layer("ru_bad", displayType="refrence")
        self.assertEqual(set(cmds.ls(type="displayLayer")), layers)
        cube << Layer("ru_ok", displayType="reference")
        self.assertEqual(cmds.getAttr("ru_ok.displayType"), 2)
        with self.assertRaisesRegex(TypeError, r"^ru_ok\.displayType: 'x' is not one of its enum fields"):
            cube << Layer("ru_ok", update=True, displayType="x")
        self.assertEqual(cmds.getAttr("ru_ok.displayType"), 2)
        before = _scene()
        with self.assertRaisesRegex(TypeError, r"^lambert\.matteOpacityMode: 'solid mate'"):
            cube << Lambert("ru_m2", matteOpacityMode="solid mate")
        self.assertEqual(_scene(), before)
        cube << Lambert("ru_m1", matteOpacityMode="solid matte")
        self.assertEqual(cmds.getAttr("ru_m1.matteOpacityMode"), 1)
