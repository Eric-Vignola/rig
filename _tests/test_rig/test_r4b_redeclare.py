"""Round 4b, step NC8: re-declaring an attribute (user decision, "apply settings,
keep value").

``node << Float("w", ...)`` on a node that has ``w`` already keeps the attribute,
its value and its connections, and applies the settings the spec was given: the
keywords its caller passed (``dv``, ``min`` / ``max``, the soft range,
``keyable``, ``hidden``, ``niceName``, an ``Enum``'s ``en`` ...), never rig's own
defaults (``keyable=True``). A default is the attribute's default, not its value.
``overwrite=True`` is the explicit delete + re-add; ``overwrite=False`` written
out means the default. Refused with a TypeError before any edit: another kind of
attribute (attributeType / dataType, multi, a compound's children), a setting
``addAttr`` cannot edit that differs, a min above the max, a default outside the
range, and any setting on a static attribute (with none, its plug).

Maya's own limit, not rig's: an undone ``addAttr -edit -defaultValue`` leaves the
default at 0 (``runs\\r4b\\NC8\\probe_addattr.txt``), so the undo test edits no
default.
"""

import os
import shutil
import tempfile

from maya import cmds

from rig import List, Node
from rig.nodetypes import Transform
from rig.spec import Enum, Euler, Float, Note, String, Vector
from rig._tests._base import MayaTestCase


def _q(plug, flag):
    return cmds.addAttr(plug, query=True, **{flag: True})


def _inputs(plug):
    return cmds.listConnections(plug, source=True, destination=False, plugs=True) or []


def _state(plug):
    """What a refused re-declaration must leave as it was."""
    return {
        "value":   cmds.getAttr(plug),
        "inputs":  _inputs(plug),
        "type":    cmds.getAttr(plug, type=True),
        "min":     _q(plug, "minValue"),
        "max":     _q(plug, "maxValue"),
        "dv":      _q(plug, "defaultValue"),
        "nn":      cmds.attributeQuery(plug.split(".", 1)[1], node=plug.split(".", 1)[0], niceName=True),
        "keyable": cmds.getAttr(plug, keyable=True),
        "hidden":  _q(plug, "hidden"),
        "ls":      set(cmds.ls()),
    }


class TestRedeclare(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.node = Node.create("transform", name="rd")
        self.drv  = Node.create("transform", name="drv")
        cmds.setAttr("drv.tx", 7)

    # -- the user's example, verbatim -- #

    def test_user_decision_example(self):
        node, drv = self.node, self.drv
        node << Float("test", dv=5)
        node.test << drv.tx
        plug = node << Float("test")
        self.assertIs(plug.node, node)
        self.assertEqual(str(plug), "rd.test")
        self.assertEqual(cmds.getAttr("rd.test"), 7)
        self.assertEqual(_inputs("rd.test"), ["drv.translateX"])
        self.assertEqual(_q("rd.test", "defaultValue"), 5)
        node << Float("test", max=10)
        self.assertEqual(_q("rd.test", "maxValue"), 10)
        self.assertEqual(cmds.getAttr("rd.test"), 7)
        self.assertEqual(_inputs("rd.test"), ["drv.translateX"])

    # -- no settings: nothing changes -- #

    def test_no_settings_changes_nothing(self):
        self.node << Float("w", min=-1, max=4, nn="Weight", k=False, hidden=True, dv=2)
        cmds.setAttr("rd.w", 3)
        before = _state("rd.w")
        plug = self.node << Float("w")
        self.assertEqual(_state("rd.w"), before)
        self.assertFalse(cmds.getAttr(str(plug), keyable=True))

    def test_keyable_default_is_not_a_setting(self):
        # rig's keyable=True is not re-applied: a hidden or non-keyable attr stays so
        self.node << Float("quiet", k=False)
        self.node << Float("secret", hidden=True)
        self.node << Float("quiet", min=0)
        self.node << Float("secret", max=3)
        self.assertFalse(cmds.getAttr("rd.quiet", keyable=True))
        self.assertTrue(_q("rd.secret", "hidden"))
        self.assertEqual(_q("rd.quiet", "minValue"), 0)
        self.assertEqual(_q("rd.secret", "maxValue"), 3)
        # passed explicitly, it applies
        self.node << Float("quiet", keyable=True)
        self.assertTrue(cmds.getAttr("rd.quiet", keyable=True))

    # -- each setting -- #

    def test_each_editable_setting_applies_and_keeps_value_and_connection(self):
        cases = (
            ({"dv": 3},                  "defaultValue",  3),
            ({"defaultValue": 4},        "defaultValue",  4),
            ({"min": -2},                "minValue",      -2),
            ({"minValue": -3},           "minValue",      -3),
            ({"max": 20},                "maxValue",      20),
            ({"softMinValue": 1},        "softMinValue",  1),
            ({"smx": 6},                 "softMaxValue",  6),
            ({"nn": "Blend Amount"},     "niceName",      "Blend Amount"),
            ({"niceName": "Other"},      "niceName",      "Other"),
            ({"hidden": True},           "hidden",        True),
            ({"h": True},                "hidden",        True),
            ({"category": "rigging"},    "category",      ["rigging"]),
        )
        for kwargs, flag, expected in cases:
            with self.subTest(kwargs=kwargs):
                cmds.file(new=True, force=True)
                node = Node.create("transform", name="rd")
                drv  = Node.create("transform", name="drv")
                cmds.setAttr("drv.tx", 7)
                node << Float("w", min=0, max=10) << 2.5
                node << Float("d", min=0, max=10)
                node.d << drv.tx
                for name, value in (("w", 2.5), ("d", 7)):
                    plug = node << Float(name, **kwargs)
                    self.assertIs(plug.node, node)
                    self.assertEqual(_q(f"rd.{name}", flag), expected)
                    self.assertEqual(cmds.getAttr(f"rd.{name}"), value)
                self.assertEqual(_inputs("rd.d"), ["drv.translateX"])

    def test_keyable_and_hidden_both_ways(self):
        self.node << Float("w") << 1.5
        self.node << Float("w", k=False)
        self.assertFalse(cmds.getAttr("rd.w", keyable=True))
        self.node << Float("w", keyable=True)
        self.assertTrue(cmds.getAttr("rd.w", keyable=True))
        self.node << Float("w", hidden=True)
        self.assertTrue(cmds.attributeQuery("w", node="rd", hidden=True))
        self.node << Float("w", hidden=False)
        self.assertFalse(cmds.attributeQuery("w", node="rd", hidden=True))
        self.assertEqual(cmds.getAttr("rd.w"), 1.5)

    def test_has_min_and_max_apply_only_when_they_differ(self):
        # Maya's addAttr -edit toggles them whatever the value: given twice, a
        # removal stays a removal
        self.node << Float("w", min=0, max=10) << 4
        for _ in range(2):
            self.node << Float("w", hasMaxValue=False)
            self.assertFalse(_q("rd.w", "hasMaxValue"))
            self.assertTrue(_q("rd.w", "hasMinValue"))
        self.node << Float("w", hnv=False)
        self.assertFalse(_q("rd.w", "hasMinValue"))
        self.node << Float("w", max=8)
        self.assertEqual((_q("rd.w", "hasMaxValue"), _q("rd.w", "maxValue")), (True, 8))
        self.assertEqual(cmds.getAttr("rd.w"), 4)

    def test_a_default_is_not_the_value(self):
        # a plug never set reads its default: re-declaring a default (or a range
        # the default moves into) leaves the value it had, saved and reopened too
        self.node << Float("w", dv=5)
        self.node << Float("rr", dv=5, min=0, max=10)
        self.node << Float("w", dv=3)
        self.node << Float("rr", min=20, max=30)
        self.assertEqual((_q("rd.w", "defaultValue"), cmds.getAttr("rd.w")), (3, 5))
        self.assertEqual((_q("rd.rr", "minValue"), cmds.getAttr("rd.rr")), (20, 5))
        folder = tempfile.mkdtemp(prefix="r4b_nc8_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.addCleanup(cmds.file, new=True, force=True)
        path = os.path.join(folder, "redeclare.ma")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(path, open=True, force=True)
        self.assertEqual(cmds.getAttr("rd.w"), 5)
        self.assertEqual(cmds.getAttr("rd.rr"), 5)
        self.assertEqual(_q("rd.w", "defaultValue"), 3)

    def test_a_range_that_cannot_hold_is_refused_before_any_edit(self):
        self.node << Float("w", min=0, max=10, dv=5) << 2
        before = _state("rd.w")
        for kwargs, text in (
            ({"min": 20},                   r"min=20 is above max=10\.0"),
            ({"min": 3, "max": 1},          r"min=3 is above max=1"),
            ({"dv": 50},                    r"dv=50 is outside its range \[0\.0, 10\.0\]"),
            ({"dv": 50, "nn": "Never"},     r"dv=50 is outside"),
            ({"dv": -1, "max": 4},          r"dv=-1 is outside its range \[0\.0, 4\]"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(TypeError, text):
                    self.node << Float("w", **kwargs)
                self.assertEqual(_state("rd.w"), before)
        # both moved together, it holds
        self.node << Float("w", min=20, max=30)
        self.assertEqual((_q("rd.w", "minValue"), _q("rd.w", "maxValue")), (20, 30))
        self.assertEqual(cmds.getAttr("rd.w"), 2)

    def test_a_setting_addattr_cannot_edit_must_match(self):
        self.node << Float("w", sn="ww") << 3
        self.node.w << self.drv.tx
        before = _state("rd.w")
        # the same: nothing to do
        self.node << Float("w", sn="ww", writable=True, storable=True)
        self.assertEqual(_state("rd.w"), before)
        for kwargs, flag in (
            ({"sn": "wx", "nn": "Never"}, "shortName"),
            ({"writable": False},         "writable"),
            ({"storable": False},         "storable"),
            ({"usedAsColor": True},       "usedAsColor"),
            ({"hasSoftMaxValue": True},   "hasSoftMaxValue"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    TypeError, rf"'rd\.w': {flag} cannot be changed on an existing attribute.*overwrite=True replaces it"
                ):
                    self.node << Float("w", **kwargs)
                self.assertEqual(_state("rd.w"), before)

    # -- Enum -- #

    def test_enum_fields_edit_keeps_the_value_index(self):
        self.node << Enum("mode", en="a:b:c") << 2
        self.node << Enum("mode", en=["x", "y", "z", "w"])
        self.assertEqual(cmds.attributeQuery("mode", node="rd", listEnum=True), ["x:y:z:w"])
        self.assertEqual(cmds.getAttr("rd.mode"), 2)
        self.assertEqual(cmds.getAttr("rd.mode", asString=True), "z")
        self.node << Enum("mode", en={"x": 0, "y": 1, "z": 2}, dv="y")
        self.assertEqual(_q("rd.mode", "defaultValue"), 1)
        self.assertEqual(cmds.getAttr("rd.mode"), 2)
        # no en= given: the fields stay (the spec's own "False:True:" is not a setting)
        self.node << Enum("mode")
        self.assertEqual(cmds.attributeQuery("mode", node="rd", listEnum=True), ["x:y:z"])

    # -- compound, multi -- #

    def test_compound_settings_go_to_children_and_parent(self):
        self.node << Vector("vec", defaultValue=[1, 2, 3], min=0, max=10)
        cmds.setAttr("rd.vecX", 4)
        self.node.vecY << self.drv.tx
        self.node << Vector("vec", defaultValue=[7, 8, 9], max=20, niceName="Vee", k=False)
        self.assertEqual([_q(f"rd.vec{c}", "defaultValue") for c in "XYZ"], [7, 8, 9])
        self.assertEqual([_q(f"rd.vec{c}", "maxValue") for c in "XYZ"], [20, 20, 20])
        self.assertEqual([_q(f"rd.vec{c}", "minValue") for c in "XYZ"], [0, 0, 0])
        self.assertEqual(_q("rd.vec", "niceName"), "Vee")
        self.assertEqual(
            [cmds.getAttr(f"rd.{a}", keyable=True) for a in ("vec", "vecX", "vecY", "vecZ")],
            [False, False, False, False],
        )
        self.assertEqual(cmds.getAttr("rd.vec"), [(4.0, 7.0, 3.0)])
        self.assertEqual(_inputs("rd.vecY"), ["drv.translateX"])
        # a scalar default is ignored, as on creation
        self.node << Vector("vec", dv=5)
        self.assertEqual([_q(f"rd.vec{c}", "defaultValue") for c in "XYZ"], [7, 8, 9])

    def test_multi_keeps_its_elements(self):
        self.node << Float("arr", multi=True, size=3, dv=2.0)
        self.node.arr[1] << 6
        self.node.arr[4] << self.drv.tx
        plug = self.node << Float("arr", multi=True, size=8, max=5, nn="Array")
        self.assertEqual(str(plug), "rd.arr")
        self.assertEqual(_q("rd.arr", "maxValue"), 5)
        self.assertEqual(_q("rd.arr", "niceName"), "Array")
        self.assertEqual(cmds.getAttr("rd.arr", multiIndices=True), [0, 1, 2, 4])
        self.assertEqual([cmds.getAttr(f"rd.arr[{i}]") for i in (0, 1, 2, 4)], [2, 6, 2, 7])
        self.assertEqual(_inputs("rd.arr[4]"), ["drv.translateX"])
        # Maya cannot change a multi's default: the same default passes, another raises
        self.node << Float("arr", multi=True, dv=2)
        with self.assertRaisesRegex(TypeError, r"defaultValue cannot be changed.*overwrite=True"):
            self.node << Float("arr", multi=True, dv=3)

    def test_multi_compound_children_are_edited(self):
        # its children are named through element 0 (never made): ranges edit,
        # the default is fixed, as a multi's
        self.node << Vector("vm", multi=True)
        self.node.vm[2] << [1, 2, 3]
        self.node << Vector("vm", multi=True, min=-1, nn="Vees")
        self.assertEqual(cmds.attributeQuery("vmX", node="rd", minimum=True), [-1])
        self.assertEqual(_q("rd.vm", "niceName"), "Vees")
        self.node << Vector("vm", multi=True, defaultValue=[0, 0, 0])
        with self.assertRaisesRegex(TypeError, r"'rd\.vm\[0\]\.vmX': defaultValue cannot be changed"):
            self.node << Vector("vm", multi=True, defaultValue=[4, 5, 6])
        self.assertEqual(cmds.getAttr("rd.vm", multiIndices=True), [2])
        self.assertEqual(cmds.getAttr("rd.vm[2]"), [(1, 2, 3)])

    # -- static, another kind, overwrite -- #

    def test_static_attribute(self):
        plug = self.node << Float("tx")
        self.assertIs(plug.node, self.node)
        self.assertEqual(str(plug), "rd.translateX")
        before = set(cmds.ls())
        for spec in (Float("tx", min=0), Float("translateX", k=False), Float("visibility", nn="V")):
            with self.subTest(spec=spec.kargs["longName"]):
                with self.assertRaisesRegex(TypeError, r"'rd\.\w+' is a static attribute; its settings cannot be changed"):
                    self.node << spec
        self.assertEqual(set(cmds.ls()), before)
        self.assertTrue(cmds.getAttr("rd.tx", keyable=True))
        self.assertFalse(cmds.attributeQuery("tx", node="rd", minExists=True))

    def test_another_kind_is_refused_before_any_edit(self):
        self.node << Float("w", max=3) << 2
        self.node.w << self.drv.tx
        self.node << Euler("rot")
        before = _state("rd.w")
        for spec, text in (
            (Enum("w", en="a:b"),               "exists as a double, not an enum"),
            (String("w"),                       "exists as a double, not a string"),
            (Float("w", multi=True, max=9),     "exists as a double, not a multi double"),
            (Vector("w"),                       r"exists as a double, not a double3 of double \(wX, wY, wZ\)"),
            (Vector("rot"),                     r"'rd\.rot' exists as a double3 of doubleAngle \(rotX, rotY, rotZ\), "
                                                r"not a double3 of double"),
        ):
            with self.subTest(spec=type(spec).__name__):
                with self.assertRaisesRegex(TypeError, text + ".*; overwrite=True replaces it"):
                    self.node << spec
        self.assertEqual(_state("rd.w"), before)

    def test_overwrite_true_replaces_and_false_is_the_default(self):
        self.node << Float("w", min=0, max=10) << 4
        self.node.w << self.drv.tx
        # overwrite=False written out: the settings apply, the value and wire stay
        self.node << Float("w", max=20, overwrite=False)
        self.assertEqual((_q("rd.w", "maxValue"), cmds.getAttr("rd.w")), (20, 7))
        self.assertEqual(_inputs("rd.w"), ["drv.translateX"])
        # overwrite=True: deleted and added again, value and connection gone
        plug = self.node << Float("w", max=5, overwrite=True)
        self.assertEqual(str(plug), "rd.w")
        self.assertEqual(_inputs("rd.w"), [])
        self.assertEqual(cmds.getAttr("rd.w"), 0)
        self.assertEqual((_q("rd.w", "hasMinValue"), _q("rd.w", "maxValue")), (False, 5))
        # and it replaces another kind too
        self.node << Enum("w", en="a:b", overwrite=True)
        self.assertEqual(cmds.getAttr("rd.w", type=True), "enum")

    # -- the other left-hand sides -- #

    def test_a_plug_or_a_list_on_the_left_follows_the_rule(self):
        other = Node.create("transform", name="rd2")
        for node in (self.node, other):
            node << Float("w", max=3)
            node.w << self.drv.tx
        self.node.tx << Float("w", max=10)
        self.assertEqual(_q("rd.w", "maxValue"), 10)
        List([self.node, other]) << Float("w", nn="Both")
        for name in ("rd", "rd2"):
            self.assertEqual(_q(f"{name}.w", "niceName"), "Both")
            self.assertEqual(_inputs(f"{name}.w"), ["drv.translateX"])

    def test_output_spec_rerun_and_a_writable_attr(self):
        out = self.node >> Float("out")
        again = self.node >> Float("out", nn="Result")
        self.assertEqual(str(again), str(out))
        self.assertFalse(_q("rd.out", "writable"))
        self.assertEqual(_q("rd.out", "niceName"), "Result")
        self.node << Float("w")
        with self.assertRaisesRegex(TypeError, r"'rd\.w': writable cannot be changed.*overwrite=True"):
            self.node >> Float("w")

    def test_plug_clone_onto_a_taken_name_still_never_overwrites(self):
        self.node << Float("w") << 3
        self.drv << Float("w")
        with self.assertRaisesRegex(TypeError, "never overwrites"):
            self.drv.tx >> "rd.w"
        with self.assertRaisesRegex(TypeError, "never overwrites"):
            self.node.tx >> "w"
        self.assertEqual(cmds.getAttr("rd.w"), 3)

    def test_note_text_applies(self):
        self.node << Note("first")
        self.node << Note("second")
        self.assertEqual(cmds.getAttr("rd.notes"), "second")
        self.node << Note()
        self.assertEqual(cmds.getAttr("rd.notes"), "second")
        self.assertTrue(_q("rd.notes", "hidden"))

    def test_rec_w8_a_rerun_keeps_fk_ik_and_its_connection(self):
        def build():
            root = Transform.define("rig")
            arm  = Transform.define("L_arm", parent=root)
            ctl  = Transform.define("ctl", parent=arm)
            return ctl, ctl << Float("fk_ik", min=0, max=1)

        ctl, fk_ik = build()
        fk_ik << self.drv.ty
        cmds.setAttr("drv.ty", 0.25)
        ctl2, fk_ik2 = build()
        self.assertEqual(ctl2, ctl)
        self.assertEqual(str(fk_ik2), str(fk_ik))
        self.assertEqual(cmds.getAttr(str(fk_ik2)), 0.25)
        self.assertEqual(_inputs(str(fk_ik2)), ["drv.translateY"])
        self.assertEqual(len(cmds.ls("ctl", type="transform")), 1)

    # -- edges: undo, locked, namespaces, a referenced node -- #

    def test_undo_restores_the_settings_and_keeps_value_and_wire(self):
        self.node << Float("w", min=0, max=10, nn="Before")
        self.node.w << self.drv.tx
        cmds.undoInfo(state=True, infinity=True)
        cmds.undoInfo(openChunk=True)
        try:
            self.node << Float("w", max=20, nn="After", hidden=True, k=False)
        finally:
            cmds.undoInfo(closeChunk=True)
        self.assertEqual((_q("rd.w", "maxValue"), _q("rd.w", "niceName"), _q("rd.w", "hidden")), (20, "After", True))
        cmds.undo()
        self.assertEqual((_q("rd.w", "maxValue"), _q("rd.w", "niceName"), _q("rd.w", "hidden")), (10, "Before", False))
        self.assertTrue(cmds.getAttr("rd.w", keyable=True))
        self.assertEqual((cmds.getAttr("rd.w"), _inputs("rd.w")), (7, ["drv.translateX"]))
        cmds.redo()
        self.assertEqual((_q("rd.w", "maxValue"), _q("rd.w", "niceName"), _q("rd.w", "hidden")), (20, "After", True))
        self.assertFalse(cmds.getAttr("rd.w", keyable=True))

    def test_locked_attr_stays_locked(self):
        self.node << Float("w") << 4
        cmds.setAttr("rd.w", lock=True)
        self.node << Float("w", max=10, nn="Locked")
        self.assertEqual((cmds.getAttr("rd.w"), cmds.getAttr("rd.w", lock=True)), (4, True))
        self.assertEqual(_q("rd.w", "maxValue"), 10)

    def test_namespaced_and_referenced_nodes(self):
        folder = tempfile.mkdtemp(prefix="r4b_nc8_ref_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.addCleanup(cmds.file, new=True, force=True)
        self.node << Float("w", max=3) << 2
        path = os.path.join(folder, "src.ma")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        cmds.file(path, reference=True, namespace="ref")
        ref = Node("ref:rd")
        plug = ref << Float("w", max=9)
        self.assertEqual(str(plug), "ref:rd.w")
        self.assertEqual((_q("ref:rd.w", "maxValue"), cmds.getAttr("ref:rd.w")), (9, 2))
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:c")
        Node("ns:c") << Float("k") << 1
        Node("ns:c") << Float("k", nn="Kay")
        self.assertEqual((_q("ns:c.k", "niceName"), cmds.getAttr("ns:c.k")), ("Kay", 1))
