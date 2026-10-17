from maya import cmds
from rig.nodetypes import DisplayLayer, Node
from rig._tests._base import MayaTestCase


class TestDisplayLayer(MayaTestCase):
    """
    display layer unit tests.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.default_layer = DisplayLayer("defaultLayer")
        self.xform         = Node(cmds.group(empty=True))
        self.layer         = DisplayLayer.create()
        # re-pinned (user decision 2026-10-16, ONE WAY ONLY): the raw
        # add_members / remove_members raise; << and layer.remove are the forms
        self.xform << self.layer

    def test_members(self):
        self.assertEqual(self.layer.get_members(), [self.xform])
        self.layer.clear()
        self.assertEqual(self.layer.get_members(), [])

    def test_find(self):
        self.assertEqual(DisplayLayer.find_all(), [self.default_layer, self.layer])

    def test_get_or_create_and_for_node(self):
        """Historical id (v2.0.0a2): pinned DisplayLayer.get_or_create and for_node; it
        now pins DisplayLayer.define (get_or_create's replacement, round 4b NC4) and
        for_node."""
        self.assertEqual(DisplayLayer.define(self.layer.get_name()), self.layer)
        cmds.select(self.xform.get_name())
        fresh = DisplayLayer.define("fresh")
        self.assertEqual(fresh.get_name(), "fresh")
        self.assertEqual(fresh.get_members(), [])
        self.assertEqual(DisplayLayer.define("fresh"), fresh)
        with self.assertRaisesRegex(TypeError, "not a displayLayer"):
            DisplayLayer.define(self.xform.get_name())
        self.assertEqual(DisplayLayer.for_node(self.xform), self.layer)
        other = Node(cmds.group(empty=True, name="other"))
        self.assertIsNone(DisplayLayer.for_node(other))
        self.assertFalse(fresh.is_default())
        self.assertTrue(self.default_layer.is_default())
        cmds.namespace(add="ns")
        cmds.namespace(set="ns")
        try:
            spaced = DisplayLayer.define("spaced")
            self.assertEqual(spaced.get_name(), "ns:spaced")
            self.assertEqual(DisplayLayer.define("spaced"), spaced)
        finally:
            cmds.namespace(set=":")

    def test_remove_members_and_delete(self):
        other = Node(cmds.group(empty=True, name="other"))
        fresh = DisplayLayer.define("fresh")
        # re-pinned (user decision 2026-10-16, ONE WAY ONLY): the raw verbs
        # raise, naming the rig form, with nothing changed; << and
        # layer.remove(x) (the shorthand x << -layer) are the forms
        with self.assertRaisesRegex(TypeError, r"layer\.add_members is not rig's form: write cube << "):
            fresh.add_members(other)
        with self.assertRaisesRegex(TypeError, r"layer\.remove_members is not rig's form: write .*\.remove\(cube\)"):
            self.layer.remove_members(self.xform)
        self.assertEqual(DisplayLayer.for_node(self.xform), self.layer)
        other << fresh
        # a node held by another layer is left there
        self.layer.remove(other)
        self.assertEqual(DisplayLayer.for_node(other), fresh)
        self.layer.remove(self.xform)
        self.assertIsNone(DisplayLayer.for_node(self.xform))
        self.assertEqual(self.layer.get_members(), [])
        fresh.delete()
        self.assertFalse(cmds.objExists("fresh"))
        self.assertIsNone(DisplayLayer.for_node(other))
        with self.assertRaises(TypeError):
            self.default_layer.delete()
        self.assertTrue(cmds.objExists("defaultLayer"))