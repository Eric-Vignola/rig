"""Tests for ``rig.spec`` -- attribute-specification DSL."""

from maya import cmds
from rig import Node
from rig.spec import (
    Bool,
    Color,
    Enum,
    Euler,
    Float,
    hide,
    Int,
    lock,
    Matrix,
    Note,
    Quat,
    String,
    unhide,
    unlock,
    Vector,
)
from rig._tests._base import MayaTestCase


class TestNumericSpecs(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_float(self):
        node = Node.create("transform", name="ctrl")
        plug = node << Float("blend")
        self.assertTrue(cmds.attributeQuery("blend", node="ctrl", exists=True))
        self.assertEqual(
            cmds.attributeQuery("blend", node="ctrl", attributeType=True), "double"
        )

    def test_float_with_min_max(self):
        node = Node.create("transform", name="ctrl")
        node << Float("weight", min=0, max=1)
        self.assertEqual(cmds.attributeQuery("weight", node="ctrl", min=True), [0.0])
        self.assertEqual(cmds.attributeQuery("weight", node="ctrl", max=True), [1.0])

    def test_int(self):
        node = Node.create("transform", name="ctrl")
        node << Int("count")
        self.assertEqual(
            cmds.attributeQuery("count", node="ctrl", attributeType=True), "long"
        )

    def test_bool(self):
        node = Node.create("transform", name="ctrl")
        node << Bool("flag")
        self.assertEqual(
            cmds.attributeQuery("flag", node="ctrl", attributeType=True), "bool"
        )

    def test_chained_set(self):
        node = Node.create("transform", name="ctrl")
        node << Float("weight") << 0.5
        self.assertAlmostEqual(cmds.getAttr("ctrl.weight"), 0.5)

    def test_chained_lock(self):
        node = Node.create("transform", name="ctrl")
        node << Float("weight") << 0.5 << lock
        self.assertTrue(cmds.getAttr("ctrl.weight", lock=True))


class TestCompoundSpecs(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_vector_creates_three_children(self):
        node = Node.create("transform", name="ctrl")
        node << Vector("aim")
        for child in ("aimX", "aimY", "aimZ"):
            self.assertTrue(cmds.attributeQuery(child, node="ctrl", exists=True), child)

    def test_quat_creates_four_children(self):
        node = Node.create("transform", name="ctrl")
        node << Quat("rot")
        for child in ("rotX", "rotY", "rotZ", "rotW"):
            self.assertTrue(cmds.attributeQuery(child, node="ctrl", exists=True), child)

    def test_color_creates_rgb_children(self):
        node = Node.create("transform", name="ctrl")
        node << Color("tint")
        for child in ("tintR", "tintG", "tintB"):
            self.assertTrue(cmds.attributeQuery(child, node="ctrl", exists=True), child)

    def test_euler_uses_doubleAngle(self):
        node = Node.create("transform", name="ctrl")
        node << Euler("rot")
        self.assertEqual(
            cmds.attributeQuery("rotX", node="ctrl", attributeType=True),
            "doubleAngle",
        )


class TestTypedSpecs(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_string(self):
        node = Node.create("transform", name="ctrl")
        node << String("label")
        self.assertEqual(cmds.getAttr("ctrl.label", type=True), "string")

    def test_matrix(self):
        node = Node.create("transform", name="ctrl")
        node << Matrix("xform")
        self.assertEqual(
            cmds.attributeQuery("xform", node="ctrl", attributeType=True), "matrix"
        )

    def test_string_set_value(self):
        node = Node.create("transform", name="ctrl")
        node << String("label") << "hello"
        self.assertEqual(cmds.getAttr("ctrl.label"), "hello")


class TestEnumSpec(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_enum_with_list(self):
        node = Node.create("transform", name="ctrl")
        node << Enum("mode", en=["off", "on", "auto"])
        names = cmds.attributeQuery("mode", node="ctrl", listEnum=True)[0].split(":")
        self.assertIn("off",  names)
        self.assertIn("on",   names)
        self.assertIn("auto", names)

    def test_enum_with_string(self):
        node = Node.create("transform", name="ctrl")
        node << Enum("mode", en="red:green:blue:")
        names = cmds.attributeQuery("mode", node="ctrl", listEnum=True)[0].split(":")
        self.assertIn("red",   names)
        self.assertIn("green", names)
        self.assertIn("blue",  names)


class TestModifierSpecs(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_lock_singleton(self):
        node = Node.create("transform", name="ctrl")
        node.tx << 5 << lock
        self.assertTrue(cmds.getAttr("ctrl.tx", lock=True))

    def test_unlock_singleton(self):
        node = Node.create("transform", name="ctrl")
        node.tx << 5 << lock
        node.tx << unlock
        self.assertFalse(cmds.getAttr("ctrl.tx", lock=True))

    def test_hide_singleton(self):
        node = Node.create("transform", name="ctrl")
        node.tx << hide
        self.assertFalse(cmds.getAttr("ctrl.tx", keyable=True))

    def test_unhide_singleton(self):
        node = Node.create("transform", name="ctrl")
        node.tx << hide
        node.tx << unhide
        self.assertTrue(cmds.getAttr("ctrl.tx", keyable=True))


class TestNote(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_note_creates_string_attr(self):
        node = Node.create("transform", name="ctrl")
        node << Note("This is a description.")
        self.assertTrue(cmds.attributeQuery("notes", node="ctrl", exists=True))
        self.assertEqual(cmds.getAttr("ctrl.notes"), "This is a description.")


class TestMultiAttribute(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_multi_float(self):
        node = Node.create("transform", name="ctrl")
        node << Float("blend", multi=True)
        self.assertTrue(cmds.attributeQuery("blend", node="ctrl", multi=True))

    def test_multi_vector(self):
        node = Node.create("transform", name="ctrl")
        node << Vector("aim", multi=True)
        self.assertTrue(cmds.attributeQuery("aim", node="ctrl", multi=True))


class TestOverwrite(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_overwrite_replaces_existing(self):
        node = Node.create("transform", name="ctrl")
        node << Float("blend", min=0, max=1)
        # Re-add with different range -- should replace.
        node << Float("blend", min=-5, max=5)
        self.assertEqual(cmds.attributeQuery("blend", node="ctrl", min=True), [-5.0])

    def test_no_overwrite_keeps_existing(self):
        node = Node.create("transform", name="ctrl")
        node << Float("blend", min=0, max=1)
        node << Float("blend", min=-5, max=5, overwrite=False)
        # Original kept.
        self.assertEqual(cmds.attributeQuery("blend", node="ctrl", min=True), [0.0])

class TestUnderscoreNames(MayaTestCase):
    """Attributes whose name starts with ``_`` (``__parked__``) go through the
    same paths as any other name; only names the node does NOT have keep
    raising, so Python's private/dunder probes stay cheap and harmless."""

    TEST_START_NEW_SCENE = True

    def test_leading_underscore_spec_returns_plug(self):
        node = Node.create("transform", name="ctrl")
        plug = node << Float("__parked__")
        self.assertEqual(str(plug), "ctrl.__parked__")
        plug << 3.5
        self.assertEqual(node.__parked__ >> None, 3.5)
        node.__parked__ = 4.0
        self.assertEqual(cmds.getAttr("ctrl.__parked__"), 4.0)

    def test_leading_underscore_no_overwrite_returns_existing(self):
        node = Node.create("transform", name="ctrl")
        node << Float("_x") << 2
        plug = node << Float("_x", overwrite=False)
        self.assertEqual(str(plug), "ctrl._x")
        self.assertEqual(plug >> None, 2.0)

    def test_missing_underscore_name_still_raises(self):
        node = Node.create("transform", name="ctrl")
        with self.assertRaises(AttributeError):
            node._nope
        with self.assertRaises(AttributeError):
            node.__deepcopy__
